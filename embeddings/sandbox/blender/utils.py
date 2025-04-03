import pandas as pd
from os.path import dirname, abspath, join
from typing import Dict
import numpy as np

SHEETS = [
    "real",
    "es-digital-line",
    "es-digital-paragraph",
    "es-digital-seq",
    "es-digital-rotation",
    "es-digital-zoom",
    "es-render-seq",
]


def load_data() -> Dict:

    path = join(dirname(abspath(__file__)), "plots", "df_all_pca_donut.xlsx")
    data = {}

    for sheet_name in SHEETS:
        df = pd.read_excel(path, sheet_name=sheet_name)
        data[sheet_name] = df

    return data


def write_data(distance_results: Dict):

    path = join(dirname(abspath(__file__)), "plots", "df_real_to_synth_distance_pca_donut.xlsx")

    with pd.ExcelWriter(path) as writer:

        for sheet, df in distance_results.items():
            df.to_excel(writer, sheet_name=sheet)


def create_distance_database():

    data = load_data()

    real_samples = data["real"]
    real_names = real_samples["name"]
    real_coords = real_samples[["PC1", "PC2", "PC3"]].to_numpy()

    distance_results = {}

    for sheet in SHEETS[1:]:
        synth_samples = data[sheet]
        synth_names = synth_samples["name"]
        synth_coords = synth_samples[["PC1", "PC2", "PC3"]].to_numpy()

        diff = real_coords[:, None, :] - synth_coords[None, :, :]
        distances = np.linalg.norm(diff, axis=2)

        distance_df = pd.DataFrame(distances, index=real_names, columns=synth_names)
        distance_df["mean_distance"] = distance_df.mean(axis=1)
        distance_results[sheet] = distance_df

    write_data(distance_results)


if __name__ == "__main__":

    create_distance_database()
