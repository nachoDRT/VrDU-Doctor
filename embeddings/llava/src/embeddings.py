import argparse
import debugpy
import torch
import os
import json
import csv
import re
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage
from huggingface_hub import HfApi, HfFolder
from os.path import join, dirname, abspath
from datasets import get_dataset_config_names, Image, load_dataset
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration



RESULTS_DIR = "/app/results"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
PROMPT = f"USER: <image>\nExtract JSON.\nASSISTANT:"


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def init_hf_hub():
    HfFolder.save_token(os.environ["HUGGINGFACE_HUB_TOKEN"])


def get_repo_config():

    secrets_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    secrets = load_secrets(secrets_path)

    owner = secrets["owner"]
    repo = secrets["repo"]
    branch = "main"

    return owner, repo, branch


def load_secrets(file_path: str):

    with open(file_path, encoding="utf-8") as config_file:
        secrets = json.load(config_file)

    return secrets


def save_csv():
    
    n_dim = all_embeddings_global.shape[1]
    header = [f"dim_{j}" for j in range(n_dim)] + ["label", "img"]
    
    file_name = f"llava_{re.sub(r'[/-]', '_', dataset_name)}_{subset_name}_embeddings.csv"
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
        commit_message=f"Upload Llava {dataset_name} results CSV"
    )


def get_llava():
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision='a272c74')

    # Define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    # Load the base model with adapters on top
    llava = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        revision='a272c74',
    )

    return llava, processor


def get_dataset_iterator(dataset_name: str, subset_name: str, decode=None):
    log_info("Loading Dataset")

    dataset = load_dataset(
        dataset_name, subset_name, split=split, streaming=True
    )

    if decode:
        dataset = dataset.cast_column("image", Image(decode=False))
    dataset_iterator = iter(dataset)

    return dataset_iterator


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
    llava_model_version = args.model
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

    model, processor = get_llava()

    if loop:
        subsets_to_process = []
    
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
