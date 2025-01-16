import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch
from datasets import load_dataset
import logging
import json
from donut import JSONParseEvaluator
from typing import Dict
import numpy as np


SAMPLES_LIMIT = 2


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def get_dataset_iterator():
    log_info("Loading Dataset")

    dataset = load_dataset("de-Rodrigo/merit", "en-digital-seq", split="test", streaming=True)
    dataset_iterator = iter(dataset)

    return dataset_iterator


def get_model_and_processor():
    
    model = Idefics2ForConditionalGeneration.from_pretrained("files/", device_map="auto")
    model.to(device)

    processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")

    return model, processor


def detect_json(response: str) -> str:

    start = response.find("{")
    response = response[start:]
    end = response.rfind("}")
    response = response[: end + 1]

    return response


def clean_json(grades: str) -> Dict:

    raw_json = grades.encode("utf-8").decode("unicode_escape")
    corrected_json = raw_json.encode("latin-1").decode("utf-8")

    try:
        grades_dict = json.loads(corrected_json)
    except:
        print(corrected_json)
        grades_dict = {}

    return grades_dict


def get_inputs(image):

    images = [image]

    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Look at the image and extract:\n"
                        "- The subjects and their grades.\n"
                        "- The level (9, 10, 11, or 12) they correspond to.\n\n"
                        "You must return a SINGLE JSON object in the exact following format:\n\n"
                        "{\n"
                        '  "year_9": [  # Or year_10, year_11, or year_12 as appropriate\n'
                        '    {"subject": "...", "grade": "..."},\n'
                        '    {"subject": "...", "grade": "..."}\n'
                        "  ]\n"
                        "}\n\n"
                        "DO NOT include any additional text, explanations, or comments. "
                        "Use the key 'year_9', 'year_10', 'year_11', or 'year_12' based on what can be inferred from the image."
                    ),
                },
                {
                    "type": "image",
                },
            ],
        },
    ],
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    print(text)

    inputs = processor(images=images, text=text, return_tensors="pt").to(device)

    return inputs


def get_answer(inputs):

    generated_text = model.generate(**inputs, max_new_tokens=500)
    generated_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]

    return generated_text


def get_sample_data(sample):

    log_info("Getting Image")

    img = sample["image"]

    if img.mode != "RGB":
        img = img.convert("RGB")

    gt = json.loads(sample["ground_truth"])
    gt = gt["gt_parse"]

    return img, gt


def process_dataset(dataset_iterator):
    evaluator = JSONParseEvaluator()
    accs = []
    output_list = []

    for i, sample in enumerate(dataset_iterator):
        image, gt = get_sample_data(sample)

        inputs = get_inputs(image)
        generated_text = get_answer(inputs)

        grades = detect_json(generated_text)
        grades = clean_json(grades)

        score = evaluator.cal_acc(grades, gt)
        
        accs.append(score)
        output_list.append(grades)

        print("Generated text:")
        print(grades)
        print(gt)

        if i + 1 >= SAMPLES_LIMIT:
            break

    print(f"Mean accuracy: {np.mean(accs)}")

    


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model, processor = get_model_and_processor()
    dataset_iterator = get_dataset_iterator()
    process_dataset(dataset_iterator)
