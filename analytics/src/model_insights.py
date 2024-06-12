import os
import argparse
import debugpy
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from transformers import LayoutLMv2ForTokenClassification


TRANSPOSE = True


def access_weights(model):
    weights_dict = {}
    for name, param in model.named_parameters():
        # msg = f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n"
        weights_tensor = param[:2]
        weights_tensor_list = weights_tensor.tolist()
        weights_dict[name] = weights_tensor_list

    return weights_dict


def write_csv(weights_data: list, headers: list) -> None:
    """
    Write the weights dictionary to a CSV file.

    Args:
        weights (dict): The weights per layer.
    """
    if TRANSPOSE:
        weights_data = np.transpose(np.array(weights_data))
    df = pd.DataFrame(weights_data, columns=headers)
    csv_path = os.path.join(
        Path(__file__).resolve().parents[1], "output", "model_weights.csv"
    )
    df.to_csv(csv_path, index=False)


def analyze_layer_names(layers_weights: dict) -> Tuple[int, dict]:
    layer_names_dict = {}
    max_layer_names_components = 0

    for key in layers_weights.keys():
        components = key.split(".")
        layer_names_dict[key] = components
        layer_names_components = len(components)
        if layer_names_components > max_layer_names_components:
            max_layer_names_components = layer_names_components

    return max_layer_names_components, layer_names_dict


def analyze_weights(layers_weights: dict) -> int:
    max_layer_weights = 0

    for value in layers_weights.values():
        flat_list = []
        if isinstance(value, list):
            flat_list.extend(flatten_weights(value))
        else:
            flat_list.append(value)

        n_layer_weights = len(flat_list)
        if n_layer_weights > max_layer_weights:
            max_layer_weights = n_layer_weights

    return max_layer_weights


def compose_headers(n_names: int, n_weights: int, layers_names: dict) -> list:
    if TRANSPOSE:
        headers = [f"chunk_{i}" for i, _ in enumerate(layers_names)]
    else:
        name_headers = [f"name_{n}" for n in range(n_names)]
        weights_headers = [f"weight_{n}" for n in range(n_weights)]
        headers = name_headers.extend(weights_headers)

    return headers


def flatten_weights(weights):
    flat_list = []
    for item in weights:
        if isinstance(item, list):
            flat_list.extend(flatten_weights(item))
        else:
            flat_list.append(item)
    return flat_list


def compose_data(
    layers_weights: dict, layer_names_dict: dict, n_names: int, n_weights: int
):
    data = []

    for layer_name, layer_weights in layers_weights.items():
        row_data = layer_names_dict[layer_name]
        for _ in range(len(row_data), n_names):
            row_data.append("")
        for weight in layer_weights:
            row_data.extend(flatten_weights([weight]))
        for _ in range(len(row_data), n_weights + n_names):
            row_data.append("")
        data.append(row_data)

    return data


def format_weights_data(layers_weights: dict):
    max_nested_names, layer_names_dict = analyze_layer_names(layers_weights)
    max_weights = analyze_weights(layers_weights)

    headers = compose_headers(max_nested_names, max_weights, layer_names_dict)
    data = compose_data(layers_weights, layer_names_dict, max_nested_names, max_weights)

    return data, headers


def save_weights_data(weights_data: list, headers: list):
    write_csv(weights_data, headers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--debug", type=str)
    args = parser.parse_args()

    # Debug
    if eval(args.debug):
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger to connect...")
        debugpy.wait_for_client()

    # Load model
    model_path = os.path.join("/app/models", args.model_folder, args.model_name)
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_path)

    # Access model weights
    weights_dict = access_weights(model)
    data, headers = format_weights_data(weights_dict)
    save_weights_data(data, headers)
