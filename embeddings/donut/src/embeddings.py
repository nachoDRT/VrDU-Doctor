import os
import cv2
import json
import logging
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, get_dataset_config_names, Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
import argparse
from huggingface_hub import HfApi, HfFolder
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
from os.path import join, dirname, abspath
from typing import Dict
import requests
from io import BytesIO
import base64
import PIL
from sklearn.manifold import TSNE
import debugpy
import re


RESULTS_DIR = "/app/results"
UPLOAD_IMAGES_TO_REPO = True


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def get_donut(version: str):
    log_info("Loading Model and Processor")

    if version == "vanilla":
        
        config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    
    else:
        model = VisionEncoderDecoderModel.from_pretrained("de-Rodrigo/donut-merit", subfolder=version)
        processor = DonutProcessor.from_pretrained("de-Rodrigo/donut-merit", subfolder=version)

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


# def get_img_info(sample):
#     return "https://docs.bokeh.org/static/snakebite.jpg"


def get_sample_img_name(non_decoded_sample):
    return non_decoded_sample["image"]["path"]


def compose_url(owner: str, repo: str, branch: str, image_name: str):

    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/data/{image_name}"


def encode_image(img):

    img.thumbnail((128, 100000), PIL.Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


def get_visual_embeddings(dataset_iterator, non_decoded_dataset_iterator, subset):

    all_embeddings = []
    info_list = []
    info_urls = []
    imgs_b64 = []
    imgs_subset = []

    for i, (sample, non_decoded_sample) in tqdm(enumerate(zip(dataset_iterator, non_decoded_dataset_iterator))):
        # Get image and ground truth
        image, gt = get_sample_data(sample)
        # img_info = get_img_info(sample)
        image_name = get_sample_img_name(non_decoded_sample)
        
        if image_name == None:
        
            image_name = str(i).zfill(6)
            image_name = f"{subset}_{image_name}.png"
            img_url = compose_url(owner, repo, branch, image_name)
            imgs_subset.append(subset)
        
        # Subsets in in Merit Dataset are not ordered by school name
        else:
            img_url = compose_url(owner, repo, branch, image_name)
            imgs_subset.append(image_name.split("_")[1])
            image_name = f"{subset_name}_{image_name}"

        # Prepare image
        pixel_values = processor(image, return_tensors="pt").pixel_values

        encoder_outputs = model.get_encoder()(pixel_values)
        # Los embeddings se encuentran en last_hidden_state
        image_embeddings = encoder_outputs.last_hidden_state

        # Average the embeddings across patches
        image_embedding = image_embeddings.mean(dim=1)
        all_embeddings.append(image_embedding.squeeze(0).detach().cpu().numpy())

        info_list.append(f"data/{image_name}")
        info_urls.append(img_url)
        imgs_b64.append(encode_image(image))

        if max_samples is not None and i >= max_samples:
            break


    return all_embeddings, info_list, info_urls, imgs_b64, imgs_subset


def init_hf_hub():
    HfFolder.save_token(os.environ["HUGGINGFACE_HUB_TOKEN"])


def get_dataset_embeddings():

    all_embeddings_global = []
    all_labels = []
    all_img_infos = []
    all_img_urls = []
    all_imgs_b64 = []

    for subset in subsets:
        log_info(f"Processing {subset}")
        
        dataset_iter = get_dataset_iterator(dataset_name, subset)
        non_decoded_dataset_iter = get_dataset_iterator(dataset_name, 'es-digital-seq', True)

        subset_embeddings, subset_img_infos, info_urls, imgs_b64, imgs_subsets = get_visual_embeddings(dataset_iter, non_decoded_dataset_iter, subset)
        subset_embeddings = np.stack(subset_embeddings, axis=0)  # [n_imágenes, hidden_dim]
        all_embeddings_global.append(subset_embeddings)
        # all_labels.extend([subset] * subset_embeddings.shape[0])
        all_img_infos.extend(subset_img_infos)
        all_img_urls.extend(info_urls)
        all_imgs_b64.extend(imgs_b64)
        all_labels.extend(imgs_subsets)

    all_embeddings_global = np.concatenate(all_embeddings_global, axis=0)

    return all_embeddings_global, all_labels, all_img_infos, all_img_urls, all_imgs_b64


def plot(reduced_embeddings):
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


def save_csv(embeddings):

    n_dim = embeddings.shape[1]
    header = [f"dim_{j}" for j in range(n_dim)] + ["label", "img"]
    
    file_name = f"donut_{re.sub(r'[/-]', '_', dataset_name)}_{subset_name}_embeddings.csv"
    csv_path = os.path.join(RESULTS_DIR, file_name)
    
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
    
        for i in range(embeddings.shape[0]):
            embedding_values = list(embeddings[i])
            label = all_labels[i]
            img_info = all_img_urls[i]
            writer.writerow(embedding_values + [label, img_info])
    
    return csv_path, file_name


def get_repo_config():

    secrets_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    secrets = load_secrets(secrets_path)

    owner = secrets["owner"]
    repo = secrets["repo"]
    token = secrets["token"]
    branch = "main"

    return owner, token, repo, branch


def get_last_commit_sha(api_url, token, branch):

    response = requests.get(f"{api_url}/git/ref/heads/{branch}", headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        print("Error fetching the branch reference:", response.text)
        exit()

    last_commit_sha = response.json()["object"]["sha"]

    return last_commit_sha


def get_last_commit_tree(api_url, token, last_commit_sha):

    response = requests.get(f"{api_url}/git/commits/{last_commit_sha}", headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        print("Error fetching the latest commit:", response.text)
        exit()

    base_tree_sha = response.json()["tree"]["sha"]

    return base_tree_sha


def create_imgs_blobs(api_url, token, file_names, base64_imgs):

    blobs = []

    for base64_img, file_name in zip(base64_imgs, file_names):

        blob_response = requests.post(
            f"{api_url}/git/blobs",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"content": base64_img, "encoding": "base64"},
        )

        if blob_response.status_code != 201:
            print(f"Error creating blob for {file_name}:", blob_response.text)
            exit()

        blob_sha = blob_response.json()["sha"]
        blobs.append({"path": file_name, "mode": "100644", "type": "blob", "sha": blob_sha})

    return blobs


def create_new_tree(api_url, token, base_tree_sha, blobs):

    tree_response = requests.post(
        f"{api_url}/git/trees",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"base_tree": base_tree_sha, "tree": blobs},
    )
    if tree_response.status_code != 201:
        print("Error creating the tree:", tree_response.text)
        exit()

    new_tree_sha = tree_response.json()["sha"]

    return new_tree_sha


def create_new_commit(api_url, token, last_commit_sha, new_tree_sha):

    commit_message = "Uploading multiple images in a single commit"
    commit_response = requests.post(
        f"{api_url}/git/commits",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"message": commit_message, "tree": new_tree_sha, "parents": [last_commit_sha]},
    )
    if commit_response.status_code != 201:
        print("Error creating the commit:", commit_response.text)
        exit()

    new_commit_sha = commit_response.json()["sha"]

    return new_commit_sha


def update_branch(api_url, token, branch, new_commit_sha):

    update_ref_response = requests.patch(
        f"{api_url}/git/refs/heads/{branch}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"sha": new_commit_sha},
    )
    if update_ref_response.status_code != 200:
        print("Error updating the branch reference:", update_ref_response.text)
        exit()
    else:
        print("Images Uploaded")


def upload_multiple_files_to_github(file_names, base64_imgs):

    # Base URL for the GitHub API
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    last_commit_sha = get_last_commit_sha(api_url, token, branch)
    base_tree_sha = get_last_commit_tree(api_url, token, last_commit_sha)
    blobs = create_imgs_blobs(api_url, token, file_names, base64_imgs)
    new_tree_sha = create_new_tree(api_url, token, base_tree_sha, blobs)
    new_commit_sha = create_new_commit(api_url, token, last_commit_sha, new_tree_sha)
    update_branch(api_url, token, branch, new_commit_sha)


def load_secrets(file_path: str) -> Dict:

    with open(file_path, encoding="utf-8") as config_file:
        secrets = json.load(config_file)

    return secrets


def push_csv_to_hf_space(csv_path, file_name):

    repo_id = "de-Rodrigo/Embeddings"
    path_in_repo = f"data/{file_name}"

    api = HfApi()
    api.upload_file(
        path_or_fileobj=csv_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Upload Donut {dataset_name} embeddings"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--max_samples", type=str)
    args = parser.parse_args()

    # Debug
    if eval(args.debug):
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger to connect...")
        debugpy.wait_for_client()

    init_hf_hub()
    owner, token, repo, branch = get_repo_config()

    dataset_name = args.dataset
    subset_name = args.subset
    split = args.split
    donut_model_version = args.model
    max_samples = args.max_samples

    try:
        max_samples = int(max_samples)
    except:
        max_samples = None

    if subset_name == "all":
        subsets = get_dataset_config_names(dataset_name)
    else:
        subsets = [subset_name]

    # Project config
    logging.basicConfig(level=logging.INFO)

    # Load model and processor
    model, processor = get_donut(donut_model_version)


    all_embeddings_global, all_labels, all_img_infos, all_img_urls,  all_imgs_b64 = get_dataset_embeddings()

    # Reduce dimensionality by using PCA or TSNE
    pca = PCA(n_components=2)
    reduced_embeddings_pca = pca.fit_transform(all_embeddings_global)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    reduced_embeddings_tsne = tsne.fit_transform(all_embeddings_global)


    if UPLOAD_IMAGES_TO_REPO:
        upload_multiple_files_to_github(all_img_infos, all_imgs_b64)

    # WARNING: Comparing different PCA or TSNE plots is not a good idea, but it's useful to see individaul results
    plot(reduced_embeddings_pca)
    plot(reduced_embeddings_tsne)
    
    csv_path, file_name = save_csv(all_embeddings_global)

    push_csv_to_hf_space(csv_path, file_name)