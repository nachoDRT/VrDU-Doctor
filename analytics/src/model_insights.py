import os
import argparse
import debugpy
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from transformers import LayoutLMv2ForTokenClassification


TRANSPOSE = True


def get_weights(model):
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


def get_chunk_multimodal(components: list) -> str:
    """Given a chunk, determine if the weights are textual, visual or layout

    Args:
        components (list): the name of the chunk divided into tags

    Returns:
        str: "textual", "visual" or "layout"
    """

    # TODO refine the split condition. We could have "visual_segment" and won't be able to detect it right now
    if "visual" in components:
        multimodal = "visual"
    elif "encoder" in components:
        multimodal = "textual"
    else:
        multimodal = "N/A"

    return multimodal


def analyze_layer_names(layers_weights: dict) -> Tuple[int, dict]:
    chunks_names = {}
    max_layer_names_components = 0

    for key in layers_weights.keys():
        components = key.split(".")
        chunks_names[key] = {}
        chunks_names[key]["components"] = components
        chunks_names[key]["multimodal"] = get_chunk_multimodal(components)
        layer_names_components = len(components)
        if layer_names_components > max_layer_names_components:
            max_layer_names_components = layer_names_components

    return max_layer_names_components, chunks_names


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


def compose_headers(n_names: int, n_weights: int, chunks_names: dict) -> list:
    if TRANSPOSE:
        headers = [f"chunk_{i}" for i, _ in enumerate(chunks_names)]
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
    layers_weights: dict, chunks_names: dict, n_names: int, n_weights: int
) -> list:
    data = []

    for layer_name, layer_weights in layers_weights.items():
        row_data = chunks_names[layer_name]["components"]
        for _ in range(len(row_data), n_names):
            row_data.append("")
        for weight in layer_weights:
            row_data.extend(flatten_weights([weight]))
        for _ in range(len(row_data), n_weights + n_names):
            row_data.append("")
        data.append(row_data)

    return data


def format_weights_data(layers_weights: dict) -> Tuple[list, list, dict, dict]:
    max_names_tags, chunks_names = analyze_layer_names(layers_weights)
    max_weights = analyze_weights(layers_weights)

    # Gather data dimensions
    data_dims = {}
    data_dims["chunks_max_names_tags"] = max_names_tags
    data_dims["chunks_max_weights"] = max_weights

    headers = compose_headers(max_names_tags, max_weights, chunks_names)
    data = compose_data(layers_weights, chunks_names, max_names_tags, max_weights)

    return data, headers, chunks_names, data_dims


def split_weights_data(
    data: list, chunks_names: dict, data_dims: dict
) -> Tuple[dict, dict, dict]:
    """_summary_

    Args:
        data (list): a list of 'n' lists, being 'n' the number of chunks
        chunks_names (dict): a dict with lists as values. Every list contains the tags
            of the chunk
        data_dims (dict): Data with the max dims of the data (max chunk-name tags, and
            max weighst)

    Returns:
        Tuple[dict, dict, dict]: One dictionary per partition (visual, textual) + one to
            gather those cases with no detected modal type (n_a)
    """

    t_w = {}
    v_w = {}
    n_a_w = {}

    max_names_tags = data_dims["chunks_max_names_tags"]
    # max_weights = data_dims["chunks_max_weights"]

    for i, (chunk_key, chunk_value) in enumerate(chunks_names.items()):
        weights = [
            weight for weight in data[i][max_names_tags:] if type(weight) == float
        ]
        if chunk_value["multimodal"] == "visual":
            v_w[chunk_key] = weights
        elif chunk_value["multimodal"] == "textual":
            t_w[chunk_key] = weights
        else:
            n_a_w[chunk_key] = weights

    return t_w, v_w, n_a_w


def save_data(weights_data: list, headers: list):
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
    weights_dict = get_weights(model)
    data, headers, chunks_names, data_dims = format_weights_data(weights_dict)

    # Split data
    t_w, v_w, n_a_w = split_weights_data(data, chunks_names, data_dims)

    # Pre visualize data
    # TODO

    # Process data
    # TODO

    # Save data
    save_data(data, headers)
