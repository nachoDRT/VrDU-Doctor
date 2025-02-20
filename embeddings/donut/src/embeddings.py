import re
import os
import cv2
import json
import torch
import utils
import logging
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from datasets import load_dataset, get_dataset_config_names
from donut import JSONParseEvaluator
from transformers import DonutProcessor, VisionEncoderDecoderModel
import argparse
from huggingface_hub import HfApi, HfFolder
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


RESULTS_DIR = "/app/results"


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def get_donut(subfolder: str):
    log_info("Loading Model and Processor")

    model = VisionEncoderDecoderModel.from_pretrained("de-Rodrigo/donut-merit", subfolder=subfolder)

    processor = DonutProcessor.from_pretrained("de-Rodrigo/donut-merit", subfolder=subfolder)

    return model, processor


def get_dataset_iterator(dataset_name: str, subset_name: str):
    log_info("Loading Dataset")

    dataset = load_dataset(
        dataset_name, subset_name, split="test", streaming=True
    )
    dataset_iterator = iter(dataset)

    return dataset_iterator


def get_sample_data(sample):

    img = sample["image"]
    gt = sample["ground_truth"]
    gt = gt.replace("'", '"')
    gt = json.loads(gt)
    # gt = gt["gt_parse"]

    return img, gt



def save_img(img, path):

    if img.dtype != np.uint8:
        img = (255 * img / np.max(img)).astype(np.uint8)
    cv2.imwrite(path, img)



def get_visual_embeddings(dataset_iterator):

    all_embeddings = []

    for i, sample in tqdm(enumerate(dataset_iterator)):
        # Get image and ground truth
        image, gt = get_sample_data(sample)

        # Prepare image
        pixel_values = processor(image, return_tensors="pt").pixel_values

        encoder_outputs = model.get_encoder()(pixel_values)
        # Los embeddings se encuentran en last_hidden_state
        image_embeddings = encoder_outputs.last_hidden_state

        # Average the embeddings across patches
        image_embedding = image_embeddings.mean(dim=1)
        all_embeddings.append(image_embedding.squeeze(0).detach().cpu().numpy())


    return all_embeddings


def init_hf_hub():
    HfFolder.save_token(os.environ["HUGGINGFACE_HUB_TOKEN"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    init_hf_hub()

    dataset_name = args.dataset
    subset_name = args.subset
    donut_model_version = args.model

    if subset_name == "all":
        subsets = get_dataset_config_names(dataset_name)
    else:
        subsets = [subset_name]

    # Project config
    logging.basicConfig(level=logging.INFO)

    # Load model and processor
    model, processor = get_donut(donut_model_version)

    all_embeddings_global = []
    all_labels = []

    for subset in subsets:
        print(f"Processing {subset}")
        dataset_iter = get_dataset_iterator(dataset_name, subset)
        subset_embeddings = get_visual_embeddings(dataset_iter)
        subset_embeddings = np.stack(subset_embeddings, axis=0)  # [n_imágenes, hidden_dim]
        all_embeddings_global.append(subset_embeddings)
        # Asignamos la etiqueta del subset a cada embedding de esa parte
        all_labels.extend([subset] * subset_embeddings.shape[0])

    # Concatenamos embeddings de todos los subsets en un único array
    all_embeddings_global = np.concatenate(all_embeddings_global, axis=0)

    # Reducir la dimensionalidad a 2D usando PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings_global)

    # Graficar: asignamos un color distinto a cada subset
    unique_subsets = np.unique(all_labels)
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subsets)))  # Paleta de colores

    for idx, subset in enumerate(unique_subsets):
        # Obtenemos los índices de los embeddings correspondientes a este subset
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