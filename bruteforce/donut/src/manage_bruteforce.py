import subprocess
import itertools
import argparse
import shutil
from pathlib import Path


SUBSETS = [
    "deus",
    "liceo",
    "lusitano",
    "monterraso",
    "patria"
]

def get_subsets_combinations(subsets: list, n: int = 2):
    combinations = list(itertools.combinations(subsets, n))

    return combinations


def get_combination_dataset_name(comb: tuple) -> str:
    name = ""

    for element in comb:
        if name != "":
            name += "-"
        name += element

    return name


def delete_hf_cache_memory():

    cache_dir = Path.home() / ".cache/huggingface/datasets"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("HuggingFace cache was deleted")
    else:
        raise Warning("HuggingFace cache was not found")


def manage_training_session(combs: list, n: int):
    
    for i, combination in enumerate(combs):

        prompt_datasets = []
        for component in combination:
            prompt_datasets.append("--school_name_subset")
            prompt_datasets.append(component)

        prompt = ["python", "src/train.py", "--debug",  "False", "--dataset_name", "de-Rodrigo/merit", "--dataset_subset", "es-digital-seq"]
        prompt.extend(prompt_datasets)
        prompt.extend(["--test_real"])
        subprocess.run(prompt)

        # Delete cache memory
        # delete_hf_cache_memory()


if __name__ == "__main__":

    # Define parsing values
    parser = argparse.ArgumentParser()
    parser.add_argument("--combinations_length", type=int)
    args = parser.parse_args()

    # Get parsed values
    n = args.combinations_length

    # Get combinations
    combinations = get_subsets_combinations(SUBSETS, n)
    print(len(combinations))

    manage_training_session(combinations, n)
