from push_to_hub import *
import argparse
import os
from os.path import abspath, dirname, join
from tqdm import tqdm
from PIL import Image
from io import BytesIO


SAMPLES_LIMIT = 150


def get_subset(dataset_name):
    images_bytes = []
    ground_truths = []

    # Construct the base path: <current_script_path>/data/dataset_name/imagesa/a/a
    root_path = dirname(dirname(abspath(__file__)))
    base_path = join(root_path, "data", dataset_name, "imagesa", "a", "a")

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
                        ground_truths.append("_")
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


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    dataset_name = args.dataset

    dataset = []

    subset = get_subset(dataset_name)
    dataset.append(("train", subset))
    dataset = format_data(dict(dataset))
    push_dataset_to_hf(dataset, dataset_name, "merit-aux")
