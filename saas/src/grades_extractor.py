import base64
import json
import os
from os.path import join, abspath, dirname
from openai import OpenAI
from typing import Dict


def encode_image():
    image_path = join(dirname(dirname(abspath(__file__))), "data", "test.jpg")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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


def main():

    init_apis()

    # Getting the base64 string
    base64_image = encode_image()

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "What is in this image? Please extract the academic subjects and their "
                            "associated grades, and return the results in JSON format."
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
    print(grades)


if __name__ == "__main__":

    main()
