import subprocess
import itertools
import argparse
import shutil
from pathlib import Path


# SUBSETS = [
#     "deus",
#     "liceo",
#     "lusitano",
#     "monterraso",
#     "patria"
# ]

SUBSETS = [
    "deus",
    "monterraso",
    "patria"
]

# SUBSETS = [
#     "retamar_train"
# ]

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


def manage_training_session(combs: list, n: int, freeze_encoder):
    
    for i, combination in enumerate(combs):

        prompt_datasets = []
        for component in combination:
            prompt_datasets.append("--school_name_subset")
            prompt_datasets.append(component)

        # prompt = ["python", "src/train.py", "--debug",  "False", "--dataset_name", "de-Rodrigo/merit-aux", "--dataset_subset", "retamar_train-asc-synth"]
        prompt = ["python", "src/train.py", "--debug",  "False", "--dataset_name", "combination", "--dataset_subset", "-"]
        prompt.extend(prompt_datasets)
        prompt.extend(["--test_real"])
        if freeze_encoder:
            prompt.extend(["--freeze_encoder"])
        subprocess.run(prompt)

        # Delete cache memory
        # delete_hf_cache_memory()


if __name__ == "__main__":

    # Define parsing values
    parser = argparse.ArgumentParser()
    parser.add_argument("--combinations_length", type=int)
    parser.add_argument("--freeze_encoder", action="store_true", default=False)
    args = parser.parse_args()

    # Get parsed values
    n = args.combinations_length
    freeze_encoder = args.freeze_encoder

    # Get combinations
    combinations = get_subsets_combinations(SUBSETS, n)
    print(len(combinations))

    manage_training_session(combinations, n, freeze_encoder)
