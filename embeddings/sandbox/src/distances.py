""" Compute distances between embeddings in raw embedding space
    Every img is linked to a unique CSV with multiple embeddings
"""


import argparse
from os.path import join, abspath, dirname
from os import listdir
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import matplotlib.pyplot as plt


TARGET_DATASETS = ["es-digital-seq", "es-render-seq"]


def load_csv(csv_path: str):
    data = pd.read_csv(csv_path)
    filtered_data = data[[col for col in data.columns if col.startswith("dim_")]]
    return filtered_data


def load_real_dataset_data(model: str):

    """ Return a list with N elements (as many as imags in the dataset) where each element 
    is a df with the embeddings of the image"""

    print(f"Loading dataset real dataset:")
    root = join(dirname(dirname(abspath(__file__))), model, "merit-secret")
    csv_paths = [join(root, csv) for csv in listdir(root) if csv.endswith(".csv")]

    real_dataset = [load_csv(csv_path) for csv_path in tqdm(csv_paths)]

    return real_dataset


def load_target_dataset(target_dataset: str):

    print(f"Loading dataset {target_dataset}:")
    root = join(dirname(dirname(abspath(__file__))), model, target_dataset)

    try:
        csv_paths = [join(root, csv) for csv in listdir(root) if csv.endswith(".csv")]
        target_dataset = [load_csv(csv_path) for csv_path in tqdm(csv_paths)]
        return target_dataset
    
    except FileNotFoundError:
        print(f"Dataset {target_dataset} not found at {root}")
        return None
    

def get_target_dataset_paths(target_dataset: str):
    
    root = join(dirname(dirname(abspath(__file__))), model, target_dataset)

    try:
        csv_paths = [join(root, csv) for csv in listdir(root) if csv.endswith(".csv")]
        return csv_paths
    
    except FileNotFoundError:
        print(f"Dataset {target_dataset} not found at {root}")
        return None


def main():

    real_dataset_data  = load_real_dataset_data(model)

    all_boxplot_data = []
    labels = []

    for target_dataset in TARGET_DATASETS:
        
        # target_dataset_data = load_target_dataset(target_dataset)
        target_dataset_paths = get_target_dataset_paths(target_dataset)

        if target_dataset_paths:
            distances_list = []

            print(f"Computing distances between {target_dataset} and real datasets")
            for target_dataset_sample_path in tqdm(target_dataset_paths):
                
                img_embeddings = load_csv(target_dataset_sample_path)
                synthetic_embeddings = img_embeddings.to_numpy().reshape(1, -1)
                real_embeddings = np.vstack([df.to_numpy() for df in real_dataset_data]).reshape(152, -1)
                distances = cosine_distances(synthetic_embeddings, real_embeddings)
                distances_list.extend(distances.flatten())

            all_boxplot_data.append(distances_list)
            labels.append(target_dataset)
    
    # Generar el boxplot
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(all_boxplot_data, showmeans=True)
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
    plt.xlabel("Target Dataset")
    plt.ylabel("Cosine Distance")
    plt.title("Distances between synthetic and real embeddings")
    plt.ylim(0, 1)
    plt.show()


def test():

    synth_emb = np.random.rand(1, 4800, 1024) # Data from 1 image
    real_emb = np.random.rand(152, 4800, 1024) # Data from 152 (real) images

    synth_emb_flat = synth_emb.reshape(1, -1)
    real_emb_flat = real_emb.reshape(152, -1)

    d = cosine_distances(synth_emb_flat, real_emb_flat)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    model = args.model

    main()
    # test()