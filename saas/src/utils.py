import json
import base64
from typing import Dict
from io import BytesIO
from os.path import join, abspath, dirname
import os
from datasets import load_dataset


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


def get_sample_data(sample):

    print("Getting Image")

    img = sample["image"]
    gt = json.loads(sample["ground_truth"])
    gt = gt["gt_parse"]

    return img, gt


def get_dataset_iterator():

    print("Loading Dataset")

    dataset = load_dataset("de-Rodrigo/merit", "en-render-seq", split="train", streaming=True)
    dataset_iterator = iter(dataset)

    return dataset_iterator
