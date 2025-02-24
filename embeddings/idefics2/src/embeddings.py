import json
import utils
import torch
import logging
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Idefics2ForConditionalGeneration,
)
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import hf_hub_download
import argparse
from tqdm import tqdm
import os
from huggingface_hub import HfApi, HfFolder
from donut import JSONParseEvaluator
import numpy as np
from peft import PeftModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
from os.path import join, dirname, abspath


"""
Restructured script extracted from Niels Rogge GitHub with minor modifications
You can find Niels project here: 
https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Idefics2
"""

RESULTS_DIR = "/app/results"
PEFT_MODEL_ID = "de-Rodrigo/idefics2-merit"
EMBEDDINGS_REPO = ""


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def log_error(msg: str):
    print("")
    logging.error(msg)
    print("")


def resize_embeddings(model, processor, repo_name):

    log_info(f"MODEL before resizing embeddings: {model.get_input_embeddings().weight.shape}")

    # TODO: Re-train and save input/output embeddings!!
    try:
        # Input embeddings
        filepath = hf_hub_download(
            repo_id=repo_name,
            filename="input_embeddings.pt",
            repo_type="dataset",
        )
        input_embeddings = torch.load(filepath, map_location="cuda:0")
        input_embeddings_module = torch.nn.Embedding(
            input_embeddings.shape[0],
            input_embeddings.shape[1],
            _weight=input_embeddings,
        )

        # Output embeddings
        filepath = hf_hub_download(
            repo_id=repo_name,
            filename="output_embeddings.pt",
            repo_type="dataset",
        )
        output_embeddings = torch.load(filepath, map_location="cuda:0")
        output_embeddings_module = torch.nn.Linear(
            output_embeddings.shape[0], output_embeddings.shape[1], bias=False
        )
        output_embeddings_module.weight = output_embeddings

        # Set them accordingly
        model.resize_token_embeddings(len(processor.tokenizer))
        model.set_input_embeddings(input_embeddings_module)
        model.set_output_embeddings(output_embeddings_module)

    except Exception as e:
        log_error(f"Error when resizing embeddigns: {e}")
        log_info("Unable to load saved embeddings. Resizing embeddigns from scratch")
        # dummy_dataset = utils.Idefics2Dataset(
        #     processor, model, dataset_name, subset_name, "test"
        # )
        dummy_dataset = utils.Idefics2Dataset(
            processor, model, "de-Rodrigo/merit", "es-render-seq", "test"
        )

        model_with_embeddings = dummy_dataset.get_model()
        log_info(f"MODEL (WITH EMBEDDINGS): {model.get_input_embeddings().weight.shape}")
        
        processor = dummy_dataset.get_processor()
        
        model.resize_token_embeddings(len(processor.tokenizer))
        model.set_input_embeddings(model_with_embeddings.get_input_embeddings())
        model.set_output_embeddings(model_with_embeddings.get_output_embeddings())

    return model, processor


def get_idefics2():

    processor = AutoProcessor.from_pretrained(PEFT_MODEL_ID, subfolder=idefics_model_version)
    # processor = AutoProcessor.from_pretrained(PEFT_MODEL_ID)
    # Define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # Load the base model with adapters on top
    base_model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(
        base_model,
        PEFT_MODEL_ID,
        subfolder=idefics_model_version,
        # torch_dtype=torch.float16,
        # quantization_config=quantization_config,
    )
    # model = Idefics2ForConditionalGeneration.from_pretrained(
    #     PEFT_MODEL_ID,
    #     torch_dtype=torch.float16,
    #     quantization_config=quantization_config,
    # )

    model, processor = resize_embeddings(model, processor, EMBEDDINGS_REPO)

    return model, processor


def get_dataset_iterator(dataset_name: str, subset_name: str):
    log_info("Loading Dataset")

    dataset = load_dataset(
        dataset_name, subset_name, split="test", streaming=True
    )
    dataset_iterator = iter(dataset)

    return dataset_iterator


def config_prompt(processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract JSON."},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    return prompt, processor


def resize_image(image, new_width):

    original_width, original_height = image.size
    new_height = int((new_width / original_width) * original_height)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image


def get_sample_data(sample):

    # log_info("Getting Image")

    img = sample["image"]
    # gt = json.loads(sample["ground_truth"])
    # gt = gt["gt_parse"]
    gt = sample["ground_truth"]
    gt = gt.replace("'", '"')
    gt = json.loads(gt)

    return img, gt


def process_dataset(dataset_iterator, model, processor, prompt):
    evaluator = JSONParseEvaluator()
    accs = []
    output_list = []

    for i, sample in tqdm(enumerate(dataset_iterator)):

        # Get image and ground truth
        image, gt = get_sample_data(sample)

        # Get inputs
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

        # Generate token IDs
        generated_ids = model.generate(**inputs, max_new_tokens=768)

        # Decode back into text
        generated_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Convert to dict
        generated_json = utils.token2json(generated_texts[0], processor)
        score = evaluator.cal_acc(generated_json, gt)

        accs.append(score)
        output_list.append(generated_json)

    return np.mean(accs)


def get_visual_embeddings(dataset_iter, subset_name, model, processor, prompt):
    
    all_embeddings = []
    info_urls = []

    for i, sample in tqdm(enumerate(dataset_iter)):
        image, _ = get_sample_data(sample)
        image_name = str(i).zfill(6)
        image_name = f"{subset_name}_{image_name}.png"
        img_url = compose_url(owner, repo, branch, image_name)

        # Get inputs
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

        encoder_outputs = model.get_encoder()(inputs)
        # Los embeddings se encuentran en last_hidden_state
        image_embeddings = encoder_outputs.last_hidden_state

        # Average the embeddings across patches
        image_embedding = image_embeddings.mean(dim=1)
        all_embeddings.append(image_embedding.squeeze(0).detach().cpu().numpy())

        info_urls.append(img_url)

    return all_embeddings, info_urls 


def init_hf_hub():
    HfFolder.save_token(os.environ["HUGGINGFACE_HUB_TOKEN"])


def plot():
    unique_subsets = np.unique(all_labels)
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subsets)))

    for idx, subset in enumerate(unique_subsets):
        indices = [i for i, label in enumerate(all_labels) if label == subset]
        subset_points = reduced_embeddings[indices, :]
        plt.scatter(subset_points[:, 0], subset_points[:, 1], 
                    color=colors[idx], label=subset, alpha=0.6)

    plt.title("Clusters de Embeddings de Imagen (PCA) - Todos los Subsets")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()

    save_path = os.path.join(RESULTS_DIR, "clusters_all_subsets.png")
    plt.savefig(save_path)
    plt.close()


def save_csv():
    csv_path = os.path.join(RESULTS_DIR, "pca_results_idefics2.csv")
    
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["x", "y", "label", "img"])
    
        for i in range(reduced_embeddings.shape[0]):
            x, y = reduced_embeddings[i]
            label = all_labels[i]
            img_info = all_img_urls[i]
            writer.writerow([x, y, label, img_info])

    return csv_path


def push_csv_to_hf_space():

    repo_id = "de-Rodrigo/Embeddings"
    path_in_repo = "data/data_idefics2.csv"

    api = HfApi()
    api.upload_file(
        path_or_fileobj=csv_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="space",
        commit_message="Upload PCA results CSV"
    )


def compose_url(owner: str, repo: str, branch: str, image_name: str):
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/data/{image_name}"



def get_dataset_embeddings():

    all_embeddings_global = []
    all_labels = []

    for subset_name in subsets:
        print(f"Processing {subset_name}")

        # Get test dataset
        dataset_iter = get_dataset_iterator(dataset_name, subset_name)

        subset_embeddings, info_urls = get_visual_embeddings(dataset_iter, subset_name, i2_model, i2_processor, i2_prompt)
        subset_embeddings = np.stack(subset_embeddings, axis=0)  # [n_imágenes, hidden_dim]
        all_embeddings_global.append(subset_embeddings)
        all_labels.extend([subset_name] * subset_embeddings.shape[0])
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    dataset_name = args.dataset
    subset_name = args.subset
    idefics_model_version = args.model

    init_hf_hub()
    owner, repo, branch = get_repo_config()

    if subset_name == "all":
        subsets = get_dataset_config_names(dataset_name)
        
        # Just to avoid an error resizing embeddings
        subset_name = subsets[0]

    else:
        subsets = [subset_name]

    # Project config
    logging.basicConfig(level=logging.INFO)

    # Load model and processor
    i2_model, i2_processor = get_idefics2()
    # Config prompt
    i2_prompt, i2_processor = config_prompt(i2_processor)

    all_embeddings_global, all_labels, all_img_urls = get_dataset_embeddings()

    # Reduce dimensionality by using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings_global)

    plot()
    csv_path = save_csv()
    push_csv_to_hf_space()
