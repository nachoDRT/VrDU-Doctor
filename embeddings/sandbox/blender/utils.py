import pandas as pd
from os.path import dirname, abspath, join
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from typing import List
from sklearn.svm import LinearSVC
import pickle

SHEETS = [
    "real",
    "es-digital-line",
    "es-digital-paragraph",
    "es-digital-seq",
    "es-digital-rotation",
    "es-digital-zoom",
    "es-render-seq",
]


def load_data(file: str, sheets: List) -> Dict:

    path = join(dirname(abspath(__file__)), "plots", file)
    data = {}

    for sheet_name in sheets:
        df = pd.read_excel(path, sheet_name=sheet_name)
        data[sheet_name] = df

    return data


def write_data(results: Dict, file_name: str):

    path = join(dirname(abspath(__file__)), "plots", file_name)

    with pd.ExcelWriter(path) as writer:

        for sheet, df in results.items():
            df.to_excel(writer, sheet_name=sheet)


def create_distance_database():

    data = load_data("df_all_pca_donut.xlsx", SHEETS)

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

    write_data(distance_results, "df_real_to_synth_distance_pca_donut.xlsx")


def compute_hiperplanes(feature: str, visualize: bool = False, decision: str = "ensure-ovr"):

    data_real = load_data("df_all_pca_donut.xlsx", ["real"])["real"]
    data_categories = load_data("df_heatmap_real.xlsx", ["heatmaps"])["heatmaps"]

    X = data_real[["PC1", "PC2", "PC3"]]
    y = data_categories[feature]

    print(f"Different clusters: {y.unique()}")

    # One plane per class
    if decision == "ensure-ovr":
        clf = LinearSVC()
        clf.fit(X, y)

    # More than one plane per class
    else:
        clf = SVC(kernel="linear", decision_function_shape=decision)
        clf.fit(X, y)

    if visualize:
        plot_plane(clf, X, y)

    df = pd.DataFrame(columns=["w1", "w2", "w3", "b"])
    for w, b in zip(clf.coef_, clf.intercept_):
        df = df._append(pd.Series([*w, b], index=df.columns), ignore_index=True)

    df.index = [f"hiperplane_{i}" for i in range(len(df))]

    model_path = join(dirname(abspath(__file__)), "plots", f"model_{feature}.npz")
    model_data = {"coef": clf.coef_, "intercept": clf.intercept_, "classes": clf.classes_}

    print("Saving Model")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f, protocol=4)

    # Just checking we can open the model
    # print("Opening model")
    # with open(model_path, "rb") as f:
    #     model_data = pickle.load(f)

    # clf_ = LinearSVC()
    # clf_.classes_ = model_data["classes"]
    # clf_.coef_ = model_data["coef"]
    # clf_.intercept_ = model_data["intercept"]

    return df


def plot_plane(clf, X, y):

    xx, yy = np.meshgrid(
        np.linspace(X["PC1"].min(), X["PC1"].max(), 10), np.linspace(X["PC2"].min(), X["PC2"].max(), 10)
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if y.dtype == bool or not pd.api.types.is_numeric_dtype(y):
        y_cat = y.astype("category")
        y_numeric = y_cat.cat.codes
    else:
        y_numeric = y

    ax.scatter(X["PC1"], X["PC2"], X["PC3"], c=y_numeric, cmap="viridis", s=50)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    # w[0]*x + w[1]*y + w[2]*z + b = 0  => z = -(w[0]*x + w[1]*y + b)/w[2]
    for w, b in zip(clf.coef_, clf.intercept_):
        zz = -(w[0] * xx + w[1] * yy + b) / w[2]
        ax.plot_surface(xx, yy, zz, alpha=0.3, color="red")

    plt.show()

    # print(clf.predict([[-0.8, -1, 3]]))


def create_hiperplanes_database():

    # columns = [
    #     "v_info_blocks",
    #     "v_density",
    #     "layout",
    #     "columns",
    #     "grades",
    #     "source",
    #     "shadows",
    #     "header_badge",
    #     "signed",
    #     "stamped",
    #     "table_pos",
    #     "orthogonal_distotion",
    #     "white_border",
    #     "annonimization_marks",
    #     "wrinkles",
    #     "height",
    #     "width",
    #     "rotation",
    # ]

    columns = [
        "v_info_blocks",
        "v_density",
        "layout",
        "columns",
        "grades",
        "source",
        "shadows",
        "header_badge",
        "signed",
        "stamped",
        "table_pos",
    ]

    hiperplanes = {}
    for column in columns:
        hiperplanes[column] = compute_hiperplanes(column, visualize=False)

    write_data(hiperplanes, "hiperplanes.xlsx")


if __name__ == "__main__":

    # create_distance_database()
    create_hiperplanes_database()
