import base64
import json
import os
from os.path import join, abspath, dirname
from openai import OpenAI
from typing import Dict
from donut import JSONParseEvaluator
import numpy as np
from datasets import load_dataset
from io import BytesIO

SAMPLES_LIMIT = 100


def encode_image(img):

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


def load_secrets(file_path: str) -> Dict:

    with open(file_path, encoding="utf-8") as config_file:
        secrets = json.load(config_file)

    return secrets


def init_apis():

    secrets_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    secrets = load_secrets(secrets_path)
    os.environ["OPENAI_API_KEY"] = secrets["openai"]


def detect_json(response: str) -> str:

    start = response.find("{")
    response = response[start:]
    end = response.rfind("}")
    response = response[: end + 1]

    return response


def clean_json(grades: str) -> Dict:

    raw_json = grades.encode("utf-8").decode("unicode_escape")
    corrected_json = raw_json.encode("latin-1").decode("utf-8")
    grades_dict = json.loads(corrected_json)

    return grades_dict


def get_ouput_seq(base64_image, client):

    response = client.chat.completions.create(
        model="gpt-4o",
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
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    grades = detect_json(str(response.choices[0]))
    grades = clean_json(grades)

    return grades


def get_sample_data(sample):

    print("Getting Image")

    img = sample["image"]
    gt = json.loads(sample["ground_truth"])
    gt = gt["gt_parse"]

    return img, gt


def get_dataset_iterator():

    print("Loading Dataset")

    dataset = load_dataset("de-Rodrigo/merit", "en-digital-seq", split="train", streaming=True)
    dataset_iterator = iter(dataset)

    return dataset_iterator


def process_dataset(dataset_iterator):

    client = OpenAI()

    evaluator = JSONParseEvaluator()
    accs = []
    output_list = []

    for i, sample in enumerate(dataset_iterator):

        image, gt = get_sample_data(sample)

        base64_image = encode_image(image)

        seq = get_ouput_seq(base64_image, client)
        score = evaluator.cal_acc(seq, gt)

        accs.append(score)
        output_list.append(seq)
        print(gt)
        print(seq, score)
        print("")

        if i + 1 >= SAMPLES_LIMIT:
            break

    print(f"Mean accuracy: {np.mean(accs)}")


def main():

    init_apis()
    dataset_iterator = get_dataset_iterator()
    process_dataset(dataset_iterator)


if __name__ == "__main__":

    main()
