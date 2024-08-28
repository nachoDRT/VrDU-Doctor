import os
import torch
import logging
import requests
from PIL import Image
from datasets import load_dataset
from huggingface_hub import login
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration


CHAMELEON_DIR = os.path.join(os.getcwd(), "chameleon_files")
DATA_DIR = os.path.join(os.getcwd(), "data", "cached_data")


def load_chameleon():

    chameleon_processor = ChameleonProcessor.from_pretrained(
        os.path.join(CHAMELEON_DIR, "processor")
    )
    chameleon_model = ChameleonForConditionalGeneration.from_pretrained(
        os.path.join(CHAMELEON_DIR, "model")
    )

    return chameleon_model, chameleon_processor


def download_chameleon():

    if not os.path.exists(CHAMELEON_DIR):
        os.makedirs(CHAMELEON_DIR)
        logging.info(f"New directory created: {CHAMELEON_DIR}")
    else:
        logging.info(f"{CHAMELEON_DIR} already exists")

    logging.info("Downloading Chameleon Processor")
    chameleon_processor = ChameleonProcessor.from_pretrained(
        "facebook/chameleon-7b", cache_dir=os.path.join(CHAMELEON_DIR, "processor")
    )

    logging.info("Downloading Chameleon Model")
    chameleon_model = ChameleonForConditionalGeneration.from_pretrained(
        "facebook/chameleon-7b",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        cache_dir=os.path.join(CHAMELEON_DIR, "model"),
    )

    return chameleon_model, chameleon_processor


def get_chameleon():

    try:
        logging.info("Loading downloaded files")
        chameleon_model, chameleon_processor = load_chameleon()
    except OSError:
        logging.info("No downloaded files found: downloading files")
        chameleon_model, chameleon_processor = download_chameleon()

    return chameleon_model, chameleon_processor


def get_test_dataset(dataset_name, subset, split="test"):
    dataset = load_dataset(dataset_name, name=subset, split=split, num_proc=8, cache_dir=DATA_DIR)

    return dataset


def extract_grades(chameleon_model, chameleon_processor, dataset):

    images = []
    grades = []
    g_truth = []

    prompt = """
    Extract all subjects and their corresponding grades from the document in the image below. 
    Each subject and grade pair should be identified and listed. This means you have to read the text in the image and extract the information.
    For clarity, the output should be formatted similarly to this example:
    {
    "year_9": [
        {"subject": "Greek", "grade": "50"},
        {"subject": "Religious Studies", "grade": "89"},
        {"subject": "Chinese", "grade": "51"},
        {"subject": "Calculus II", "grade": "90"},
        {"subject": "Biology and Geology", "grade": "59"},
        {"subject": "World Sciences", "grade": "69"},
        {"subject": "Business Economics", "grade": "50"},
        {"subject": "Artistic Drawing", "grade": "55"}
    ]
    }
    Note that the subjects and grades in the example are not real and are only meant to illustrate the expected output format.
    Now is your turn to extract the subjects and grades from the following image.
    <image>
    """

    for i, sample in enumerate(dataset):
        image = sample["image"]
        gt = sample["ground_truth"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        inputs = processor(prompt, image, return_tensors="pt").to(model.device, torch.bfloat16)

        output = chameleon_model.generate(**inputs, max_new_tokens=1024)
        output = processor.decode(output[0], skip_special_tokens=True)
        
        images.append(image)
        grades.append(output)
        g_truth.append(gt)

        if i > 2:
            break

    return images, grades, g_truth


def compare_results(extracted_grades, gt_grades):

    for i, (ext, gt) in enumerate(zip(extracted_grades, gt_grades)):
        logging.info(f"Sample {i}")
        logging.info(f"Extracted: {ext}")
        logging.info(f"Ground Truth: {gt}")
        logging.info("")

if __name__ == "__main__":

    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    logging.basicConfig(level=logging.INFO)

    # Get model and samples
    model, processor = get_chameleon()
    test_set = get_test_dataset("de-Rodrigo/merit", "en-digital-seq")

    # Process samples
    imgs, extracted_grades, gt_grades = extract_grades(model, processor, test_set)
    
    # Compare results
    compare_results(extracted_grades, gt_grades)
