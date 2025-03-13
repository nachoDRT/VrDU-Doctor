""" Compute distances between embeddings in raw embedding space
    Every img is linked to a unique CSV with multiple embeddings
"""

from os.path import join, abspath, dirname
from os import listdir
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
import math


TARGET_DATASETS = ["es-digital-paragraph-degradation-seq", "es-digital-line-degradation-seq", "es-digital-seq", "es-digital-rotation-degradation-seq", "es-digital-zoom-degradation-seq","es-render-seq"]
MODEL = "donut"
DISTANCE = "L2"


@cuda.jit
def l2_distance_kernel(synthetic, real, result):
    i, j = cuda.grid(2)
    if i < synthetic.shape[0] and j < real.shape[0]:
        sum_sq = 0.0
        for k in range(synthetic.shape[1]):
            diff = synthetic[i, k] - real[j, k]
            sum_sq += diff * diff
        result[i, j] = math.sqrt(sum_sq)

def l2_distances_gpu(synthetic_embeddings, real_embeddings):
    m, d = synthetic_embeddings.shape
    n, _ = real_embeddings.shape
    result = np.empty((m, n), dtype=np.float32)
    
    # Copiamos los datos a la GPU
    d_synth = cuda.to_device(synthetic_embeddings.astype(np.float32))
    d_real = cuda.to_device(real_embeddings.astype(np.float32))
    d_result = cuda.to_device(result)
    
    threadsperblock = (10, 10)
    blockspergrid_x = int(np.ceil(m / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(n / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    l2_distance_kernel[blockspergrid, threadsperblock](d_synth, d_real, d_result)
    
    # Devolvemos el resultado a la CPU
    d_result.copy_to_host(result)
    return result


@cuda.jit
def cosine_distance_kernel(synthetic, real, result):
    i, j = cuda.grid(2)
    if i < synthetic.shape[0] and j < real.shape[0]:
        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for k in range(synthetic.shape[1]):
            a = synthetic[i, k]
            b = real[j, k]
            dot += a * b
            norm1 += a * a
            norm2 += b * b
        # Evitar división por cero
        if norm1 > 0 and norm2 > 0:
            norm1 = math.sqrt(norm1)
            norm2 = math.sqrt(norm2)
            cosine_sim = dot / (norm1 * norm2)
        else:
            cosine_sim = 0.0
        result[i, j] = 1.0 - cosine_sim


def cosine_distances_gpu(synthetic_embeddings, real_embeddings):
    # synthetic_embeddings shape (m, d)
    m, d = synthetic_embeddings.shape
    # real_embeddings shape (n, d)
    n, _ = real_embeddings.shape
    result = np.empty((m, n), dtype=np.float32)
    
    # Data to GPU
    d_synth = cuda.to_device(synthetic_embeddings.astype(np.float32))
    d_real = cuda.to_device(real_embeddings.astype(np.float32))
    d_result = cuda.to_device(result)
    
    # Grid y block size
    threadsperblock = (10, 10)
    blockspergrid_x = int(np.ceil(m / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(n / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cosine_distance_kernel[blockspergrid, threadsperblock](d_synth, d_real, d_result)
    
    # Copy data back to CPU
    d_result.copy_to_host(result)
    return result



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

    # List of 152 dataframes (4800, 1024) to (152, 4800, 1024) dataframe
    real_embeddings = np.vstack([df.to_numpy() for df in real_dataset])
    # (152, 4800, 1024) to (152, 4800 * 1024) dataframe
    real_embeddings = real_embeddings.reshape(152, -1)

    return real_embeddings


def get_target_dataset_paths(target_dataset: str):
    
    root = join(dirname(dirname(abspath(__file__))), model, target_dataset)

    try:
        csv_paths = [join(root, csv) for csv in listdir(root) if csv.endswith(".csv")]
        return csv_paths
    
    except FileNotFoundError:
        print(f"Dataset {target_dataset} not found at {root}")
        return None


def compute_distances():
    real_embeddings  = load_real_dataset_data(model)
    
    all_boxplot_data = []
    labels = []
    csv_rows = []  # Aquí almacenamos cada fila para el CSV

    for target_dataset in TARGET_DATASETS:
        target_dataset_paths = get_target_dataset_paths(target_dataset)

        if target_dataset_paths:
            distances_list = []

            print(f"Computing distances between {target_dataset} and real datasets")
            
            # Procesamos cada muestra sintética del target_dataset
            for target_dataset_sample_path in tqdm(target_dataset_paths):
                img_embeddings = load_csv(target_dataset_sample_path)
                synthetic_embeddings = img_embeddings.to_numpy().reshape(1, -1)

                if DISTANCE == "L2":
                    distances = l2_distances_gpu(synthetic_embeddings, real_embeddings)
                elif DISTANCE == "cosine":
                    distances = cosine_distances_gpu(synthetic_embeddings, real_embeddings)
                # Acumulamos las distancias para el boxplot (se usa la versión flatten)
                distances_list.extend(distances.flatten())
                
                # Guardamos una fila para el CSV: [tag del dataset, dist_0, dist_1, ..., dist_151]
                row = [target_dataset] + distances.flatten().tolist()
                csv_rows.append(row)

            all_boxplot_data.append(distances_list)
            labels.append(target_dataset)
    
    # Generamos el boxplot
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(all_boxplot_data, showmeans=True)
    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
    plt.xlabel("Target Dataset")
    ylabel = "L2 Distance" if DISTANCE == "L2" else "Cosine Distance"
    plt.ylabel(ylabel)
    plt.title("Distances between synthetic and real embeddings")
    plt.show()
    
    # Escribimos los resultados en un CSV
    # Se crea un header: la primera columna es 'target_dataset' y luego dist_0, dist_1, ... dist_N
    n_real = real_embeddings.shape[0]  # Por ejemplo, 152
    columns = ["target_dataset"] + [f"dist_{i}" for i in range(n_real)]

    df = pd.DataFrame(csv_rows, columns=columns)
    
    # Guardamos el CSV en el mismo directorio base de los datos (donde se encuentran los CSV de cada target_dataset)
    base_path = join(dirname(dirname(abspath(__file__))), model)
    csv_file = join(base_path, f"results_{DISTANCE}.csv")
    df.to_csv(csv_file, index=False)
    print(f"CSV results written to {csv_file}")


def test_compute_distances():

    synth_emb = np.random.rand(1, 4800, 1024) # Data from 1 image
    real_emb = np.random.rand(152, 4800, 1024) # Data from 152 (real) images

    synth_emb_flat = synth_emb.reshape(1, -1)
    real_emb_flat = real_emb.reshape(152, -1)

    d = cosine_distances(synth_emb_flat, real_emb_flat)


def cluster_embeddings():
    real_embeddings  = load_real_dataset_data(model)
    for target_dataset in TARGET_DATASETS:
        target_dataset_paths = get_target_dataset_paths(target_dataset)
    



if __name__ == "__main__":

    # model = args.model
    model = MODEL

    # compute_distances()
    # test_compute_distances()
    cluster_embeddings()