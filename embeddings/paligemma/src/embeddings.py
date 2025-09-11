from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from huggingface_hub import HfApi, HfFolder
import os
import logging
from datasets import load_dataset, get_dataset_config_names, Image
import argparse
from donut import JSONParseEvaluator
import numpy as np
from tqdm import tqdm
import re
import json
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from peft import PeftModel, prepare_model_for_kbit_training
from PIL import Image as PILImage
import debugpy
import csv
from os.path import join, dirname, abspath


WANDB_PROJECT = "Paligemma"
MAX_LENGTH = 512
PROMPT = "extract JSON."
LIMIT = 218
RESULTS_DIR = "/app/results"


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def init_hf_hub():
    HfFolder.save_token(os.environ["HUGGINGFACE_HUB_TOKEN"])


def get_paligemma(paligemma_model_version: str):
    log_info("Loading Model and Processor")

    # Configuración de cuantización 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 1) Carga del modelo base cuantizado
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224",
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model = prepare_model_for_kbit_training(base_model)

    model = PeftModel.from_pretrained(
        base_model,
        "de-Rodrigo/paligemma-merit",
        subfolder=paligemma_model_version,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
        "de-Rodrigo/paligemma-merit",
        subfolder=paligemma_model_version,
    )

    return model, processor


def get_dataset_iterator(dataset_name: str, subset_name: str, decode=None):
    log_info("Loading Dataset")

    dataset = load_dataset(
        dataset_name, subset_name, split=split, streaming=True
    )

    if decode:
        dataset = dataset.cast_column("image", Image(decode=False))
    dataset_iterator = iter(dataset)

    return dataset_iterator



def token2json(tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        if added_vocab is None:
            added_vocab = processor.tokenizer.get_added_vocab()

        output = {}

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            key_escaped = re.escape(key)

            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}


def get_sample_data(sample):

    img = sample["image"]

    # If the image is not a PIL Image, try converting it (e.g., from a NumPy array)
    if not isinstance(img, PILImage.Image):
        img = PILImage.fromarray(img)

    # Convert the image to RGB if it's not already (this ensures 3 color channels)
    if img.mode != "RGB":
        img = img.convert("RGB")

    if dataset_name in ("de-Rodrigo/merit",
        "naver-clova-ix/cord-v2", 
        "de-Rodrigo/merit-secret"):
        
        gt = sample["ground_truth"]
        if dataset_name in ("de-Rodrigo/merit", "de-Rodrigo/merit-secret"):
            gt = gt.replace("'", '"')
        gt = json.loads(gt)

        if dataset_name == "naver-clova-ix/cord-v2":
            gt = gt["gt_parse"]
        
    else:
        ocr_words = sample["ocr_words"]
        words_list = [{"word": word} for word in ocr_words]
        page = {"page_0": words_list}
        gt = {"gt_parse": page}



    return img, gt


def get_sample_img_name(non_decoded_sample):
    return non_decoded_sample["image"]["path"]

def compose_url(owner: str, repo: str, branch: str, image_name: str):
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/data/{image_name}"




def get_visual_embeddings(dataset_iter, non_decoded_dataset_iter, subset_name):

    all_embeddings = []
    info_urls = []
    imgs_subset = []

    for i, (sample, non_decoded_sample) in tqdm(enumerate(zip(dataset_iter, non_decoded_dataset_iter))):

        # Get image and ground truth
        image, _ = get_sample_data(sample)
        image_name = get_sample_img_name(non_decoded_sample)

        if image_name == None:
        
            image_name = str(i).zfill(6)
            image_name = f"{subset_name}_{image_name}.png"
            img_url = compose_url(owner, repo, branch, image_name)
            imgs_subset.append(subset_name)
        
        # Subsets in in Merit Dataset are not ordered by school name
        else:
            img_url = compose_url(owner, repo, branch, image_name)
            imgs_subset.append(image_name.split("_")[1])

        inputs = processor(text=PROMPT, images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            vision_model = model.vision_tower
            vision_model = vision_model.to(device)
            vision_out = vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            )

            image_hidden_states = vision_out.last_hidden_state
            image_embedding = image_hidden_states.mean(dim=1)
            all_embeddings.append(image_embedding.squeeze(0).detach().cpu().numpy())
            info_urls.append(img_url)

            if max_samples is not None and i >= max_samples:
                break

    return all_embeddings, info_urls, imgs_subset


def get_dataset_embeddings():

    all_embeddings_global = []
    all_labels = []
    all_img_urls = []

    for subset in subsets:
        log_info(f"Processing {subset}")

        dataset_iter = get_dataset_iterator(dataset_name, subset)
        non_decoded_dataset_iter = get_dataset_iterator(dataset_name, subset, True)

        subset_embeddings, info_urls, imgs_subsets = get_visual_embeddings(
            dataset_iter, non_decoded_dataset_iter, subset
        )
        subset_embeddings = np.stack(subset_embeddings, axis=0)  # [n_imágenes, hidden_dim]
        all_embeddings_global.append(subset_embeddings)
        all_labels.extend(imgs_subsets)
        all_img_urls.extend(info_urls)


    all_embeddings_global = np.concatenate(all_embeddings_global, axis=0)

    return all_embeddings_global, all_labels, all_img_urls


def load_secrets(file_path: str):

    with open(file_path, encoding="utf-8") as config_file:
        secrets = json.load(config_file)

    return secrets


def get_repo_config():

    secrets_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    secrets = load_secrets(secrets_path)

    owner = secrets["owner"]
    repo = secrets["repo"]
    branch = "main"

    return owner, repo, branch


def save_csv():
    
    n_dim = all_embeddings_global.shape[1]
    header = [f"dim_{j}" for j in range(n_dim)] + ["label", "img"]
    
    file_name = f"paligemma_{re.sub(r'[/-]', '_', dataset_name)}_{subset_name}_embeddings.csv"
    csv_path = os.path.join(RESULTS_DIR, file_name)
    
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
    
        for i in range(all_embeddings_global.shape[0]):
            embedding_values = list(all_embeddings_global[i])
            label = all_labels[i]
            img_info = all_img_urls[i]
            writer.writerow(embedding_values + [label, img_info])

    return csv_path, file_name


def push_csv_to_hf_space(csv_path, file_name):

    repo_id = "de-Rodrigo/Embeddings"
    path_in_repo = f"data/{file_name}"

    api = HfApi()
    api.upload_file(
        path_or_fileobj=csv_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Upload Paligemma {dataset_name} results CSV"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--max_samples", type=str)
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    dataset_name = args.dataset
    subset_name = args.subset
    split = args.split
    paligemma_model_version = args.model
    max_samples = args.max_samples
    try:
        max_samples = int(max_samples)
    except:
        max_samples = None
    loop = args.loop
    debug_script = args.debug

    if debug_script:
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger to connect...")
        debugpy.wait_for_client()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_hf_hub()
    owner, repo, branch = get_repo_config()

    model, processor = get_paligemma(paligemma_model_version)

    if loop:
        subsets_to_process = []
        # subsets_to_process = [
        #     ("de-Rodrigo/merit-aux", "IIT-CDIP", "train"),
        #     ("de-Rodrigo/merit-secret", "all", "test"),
        #     ("de-Rodrigo/merit", "es-digital-paragraph-degradation-seq", "train"),
        #     ("de-Rodrigo/merit", "es-digital-line-degradation-seq", "train"),
        #     ("de-Rodrigo/merit", "es-digital-seq", "train"),
        #     ("de-Rodrigo/merit", "es-digital-rotation-degradation-seq", "train"),
        #     ("de-Rodrigo/merit", "es-digital-zoom-degradation-seq", "train"),
        #     ("de-Rodrigo/merit", "es-render-seq", "train"),
        # ]
        # subsets_to_process = [
        #     ("de-Rodrigo/merit-aux", "britanico-asc-synth", "train"),
        #     ("de-Rodrigo/merit-aux", "fomento-asc-synth", "train"),
        #     ("de-Rodrigo/merit-aux", "maravillas-asc-synth", "train"),
        #     ("de-Rodrigo/merit-aux", "mater-asc-synth", "train"),
        #     ("de-Rodrigo/merit-aux", "montealto-asc-synth", "train"),
        #     ("de-Rodrigo/merit-aux", "pilar-asc-synth", "train"),
        #     ("de-Rodrigo/merit-aux", "recuerdo-asc-synth", "train"),
        #     ("de-Rodrigo/merit-aux", "retamar-asc-synth", "train"),
        #     ("de-Rodrigo/merit-aux", "sanpablo-asc-synth", "train"),
        #     ("de-Rodrigo/merit-aux", "sanpatricio-asc-synth", "train")
        # ]
    
    else:
        subsets_to_process = [(dataset_name, subset_name, split)]

    for dataset_name, subset_name, split in subsets_to_process:

        if subset_name == "all":
            subsets = get_dataset_config_names(dataset_name)
        else:
            subsets = [subset_name]

        all_embeddings_global, all_labels, all_img_urls = get_dataset_embeddings()
        csv_path, file_name = save_csv()
        push_csv_to_hf_space(csv_path, file_name)
