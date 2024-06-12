import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from pathlib import Path

COLORS = ["b", "g", "r", "c", "m", "y", "k"]


def flatten_weights(split: dict) -> np.array:
    flat_weigths = []

    split_data = copy.deepcopy(split)
    split_data.pop("name")

    for chunk_weights in split_data.values():
        flat_weigths.extend(chunk_weights)

    return flat_weigths


def draw_scatter_plot(f_splits: dict, file_name: str):
    splits = []
    weights = []
    colors = []

    for split_name, split_info in f_splits.items():
        splits.append(split_name)
        weights.append(split_info["weights"])
        colors.append(split_info["color"])

    _, ax = plt.subplots(figsize=(10, 6))

    box = ax.boxplot(weights, patch_artist=True)

    # Set colors
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(splits)
    ax.set_xlabel("Splits")
    ax.set_ylabel("Weights")
    ax.set_title("Weights Previsualization")

    file_path = os.path.join(Path(__file__).resolve().parents[1], "output", file_name)
    plt.savefig(file_path)


def previsualize(splits: List):

    flat_splits = {}

    for i, split in enumerate(splits):
        flat_w = flatten_weights(split)
        flat_splits[split["name"]] = {}
        flat_splits[split["name"]]["weights"] = flat_w
        flat_splits[split["name"]]["color"] = COLORS[i]

    draw_scatter_plot(flat_splits, "test.pdf")
