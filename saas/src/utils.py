import json
import base64
from typing import Dict
from io import BytesIO
from os.path import join, abspath, dirname
import os
from datasets import load_dataset, Image, get_dataset_config_names
import re


def encode_image(img):

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


def load_secrets(file_path: str) -> Dict:
    return load_json(file_path)


def load_config(file_path: str) -> Dict:
    return load_json(file_path)


def load_json(file_path: str) -> Dict:

    with open(file_path, encoding="utf-8") as config_file:
        file_content = json.load(config_file)

    return file_content


def init_apis():

    secrets_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    secrets = load_secrets(secrets_path)
    os.environ["OPENAI_API_KEY"] = secrets["openai"]
    os.environ["MISTRAL_API_KEY"] = secrets["mistral"]


def detect_json(response: str) -> str:

    start = response.find("{")
    response = response[start:]
    end = response.rfind("}")
    response = response[: end + 1]

    return response


def clean_json(grades: str) -> Dict:

    raw_json = grades.encode("utf-8").decode("unicode_escape")
    corrected_json = raw_json.encode("latin-1").decode("utf-8")
    corrected_json = re.sub(r"'", '"', corrected_json)

    try:
        grades_dict = json.loads(corrected_json)
    except:
        # print(corrected_json)
        grades_dict = {}

    return grades_dict


def get_sample_data(sample):

    img = sample["image"]
    gt = sample["ground_truth"]

    gt = gt.replace("'", '"')
    gt = json.loads(gt)

    # print(f"Ground Truth: {type(gt)} {gt}")
    # gt = gt["gt_parse"]

    return img, gt


def get_sample_img_name(sample):

    print("Getting Image Name")
    img_name = sample["image"]["path"]

    return img_name


def get_dataset_iterator(dataset_name: str, subset_name: str, decode=None):

    dataset = load_dataset(dataset_name, subset_name, split="test", streaming=True)

    if decode:
        dataset = dataset.cast_column("image", Image(decode=False))

    dataset_iterator = iter(dataset)

    return dataset_iterator


def save_dataset_jsonl(file_name, dataset_jsonl):

    path = join(dirname(dirname(abspath(__file__))), "output", file_name)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for item in dataset_jsonl:
            f.write(json.dumps(item) + "\n")


def list_fine_tunes(client):
    try:
        response = client.fine_tuning.jobs.list()

        jobs = list(response)
        if not jobs:
            print("No fine-tuning jobs found.")
            return

        print("Fine-tuning jobs found:")
        for job in jobs:
            model_name = getattr(job, "fine_tuned_model", "N/A")
            job_status = job.status
            print(f"Job ID: {job.id}, Model: {model_name}, Status: {job_status}")

    except Exception as e:
        print(f"Error listing fine-tuning jobs: {e}")
