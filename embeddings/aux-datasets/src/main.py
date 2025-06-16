from push_to_hub import *
import argparse
import os
from os.path import abspath, dirname, join
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datasets import load_dataset

SAMPLES_LIMIT = 150


def get_IIT_CDIP_subset():
    images_bytes = []
    ground_truths = []

    # Construct the base path: <current_script_path>/data/dataset_name/imagesa/a/a
    root_path = dirname(dirname(abspath(__file__)))
    base_path = join(root_path, "data", "IIT-CDIP", "imagesa", "a", "a")

    i = 0
    limit_reached = False
    for current_root, dirs, files in tqdm(os.walk(base_path)):
        # Process each TIFF file in the current directory
        for file in sorted(files):
            if file.lower().endswith(".tif"):
                tif_path = os.path.join(current_root, file)

                try:
                    # Open the TIFF image and convert it to PNG bytes
                    with Image.open(tif_path) as img:
                        buffer = BytesIO()
                        img.save(buffer, format="PNG")
                        png_bytes = buffer.getvalue()
                        images_bytes.append(png_bytes)
                        # Append a placeholder ground truth ("_")
                        ground_truths.append({"-": "-"})
                        i += 1

                        if i > SAMPLES_LIMIT:
                            limit_reached = True
                            break

                except Exception as e:
                    print(f"Warning: Could not process image {tif_path}. Error: {e}")
                    continue

            if limit_reached:
                break
        if limit_reached:
            break

    # Return the subset in the required format
    return {"image": images_bytes, "ground_truth": ground_truths}


def get_subset_from_hub(dataset_name: str):
    images_bytes = []
    ground_truths = []

    if dataset_name == "PDFA":
        dataset_name = "pixparse/pdfa-eng-wds"
    elif dataset_name == "IDL":
        dataset_name = "pixparse/idl-wds"

    ds = load_dataset(dataset_name, split="train", streaming=True)

    i = 0
    for sample in tqdm(ds):
        processed = False
        if "tif" in sample and sample["tif"] is not None:
            try:
                with Image.open(BytesIO(sample["tif"])) as img:
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    png_bytes = buffer.getvalue()
                    images_bytes.append(png_bytes)
                    ground_truths.append({"-": "-"})
                    i += 1
                    processed = True
            except Exception as e:
                print(
                    f"Warning: Could not process 'tif' image for sample {sample.get('__key__', 'unknown')}. Error: {e}"
                )
        if not processed and "pdf" in sample and sample["pdf"] is not None:
            try:
                from pdf2image import convert_from_bytes

                pdf_bytes = sample["pdf"]
                pages = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=1)
                if pages:
                    buffer = BytesIO()
                    pages[0].save(buffer, format="PNG")
                    png_bytes = buffer.getvalue()
                    images_bytes.append(png_bytes)
                    ground_truths.append({"-": "-"})
                    i += 1
                    processed = True
            except Exception as e:
                print(f"Warning: Could not process 'pdf' for sample {sample.get('__key__', 'unknown')}. Error: {e}")

        if i >= SAMPLES_LIMIT:
            break

    return {"image": images_bytes, "ground_truth": ground_truths}


def get_aux_synth_dataset(dataset_name: str):

    images_bytes = []
    ground_truths = []

    root_path = dirname(dirname(abspath(__file__)))
    base_path = join(root_path, "data", dataset_name, "dataset_output", "images")

    i = 0

    for current_root, dirs, files in os.walk(base_path):
        for file in sorted(files):
            if file.lower().endswith(".png"):
                png_path = os.path.join(current_root, file)
                try:
                    with open(png_path, "rb") as f:
                        png_bytes = f.read()
                    images_bytes.append(png_bytes)
                    ground_truths.append({"-": "-"})
                    i += 1
                    if i >= SAMPLES_LIMIT:
                        break
                except Exception as e:
                    print(f"Warning: Unable to process image {png_path}. Error: {e}")
                    continue
        if i >= SAMPLES_LIMIT:
            break

    return {"image": images_bytes, "ground_truth": ground_truths}


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    dataset_name = args.dataset

    dataset = []

    if dataset_name == "IIT-CDIP":
        subset = get_IIT_CDIP_subset()

    elif dataset_name == "PDFA" or dataset_name == "IDL":
        subset = get_subset_from_hub(dataset_name)

    elif "-".join(dataset_name.split("-")[1:]) == "asc-synth":
        subset = get_aux_synth_dataset(dataset_name)

    else:
        print(f"{dataset_name} implementation is not available")

    dataset.append(("train", subset))
    dataset = format_data(dict(dataset))
    push_dataset_to_hf(dataset, dataset_name, "merit-aux")
