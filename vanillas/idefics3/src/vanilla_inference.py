import requests
import torch
from PIL import Image
from io import BytesIO
import logging
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from donut import JSONParseEvaluator
import json
import numpy as np
from typing import Dict


SAMPLES_LIMIT = 2


def get_answer(inputs):
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    print(generated_texts[0])
    print(generated_texts[1])

    return generated_texts


def get_inputs(image):
    
    images = [image]

    # Create inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
                {"type": "image"},
                {"type": "text", "text": "What can we see in this image?"},
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "In which city is that bridge located?"},
            ]
        }
    ]

    prompts = [processor.apply_chat_template([message], add_generation_prompt=True) for message in messages]
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)

    return inputs


def get_sample_data(sample):

    log_info("Getting Image")

    img = sample["image"]

    if img.mode != "RGB":
        img = img.convert("RGB")

    gt = json.loads(sample["ground_truth"])
    gt = gt["gt_parse"]

    return img, gt


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def get_model_and_processor():
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
    model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3", torch_dtype=torch.bfloat16, device_map="auto")

    return model, processor


def get_dataset_iterator():
    log_info("Loading Dataset")

    dataset = load_dataset("de-Rodrigo/merit", "en-digital-seq", split="test", streaming=True)
    dataset_iterator = iter(dataset)

    return dataset_iterator


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

    model, processor = get_model_and_processor()
    dataset_iterator = get_dataset_iterator()
    process_dataset(dataset_iterator)