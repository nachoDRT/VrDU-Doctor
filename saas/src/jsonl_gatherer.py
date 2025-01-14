from utils import *
import requests
import argparse


TRAINING_SAMPLES = 10


def upload_image_to_github(base64_image, image_name, upload_img: bool = False):

    secrets_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    secrets = load_secrets(secrets_path)

    # Config
    owner = secrets["owner"]
    repo = secrets["repo"]
    token = secrets["token"]
    branch = "main"
    remote_file_name = "openai-vfinetune/data"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{remote_file_name}/{image_name}"

    if upload_img:
        payload = {"message": "Uploading image", "content": base64_image, "branch": branch}
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        response = requests.put(api_url, json=payload, headers=headers)

        if response.status_code == 201:
            print("Image successfully uploaded")
            print("Public URL:", response.json()["content"]["download_url"])
        else:
            print("Error uploading image:", response.status_code)
            print("Details:", response.json())

    return api_url


def format_sample(image, image_name, gt, args: dict):
    base64_image = encode_image(image)
    url = upload_image_to_github(base64_image, image_name, args["save_imgs"])

    sample = {
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that extracts grades from strudents' transcripts of records.",
            },
            {
                "role": "user",
                "content": (
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
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{url}"},
                    }
                ],
            },
            {"role": "assistant", "content": gt},
        ]
    }

    return sample


def get_dataset_jsonl(decoded_ds_iterator, non_decoded_ds_iterator, args: dict):

    dataset_jsonl = []

    for i, (decoded_sample, non_decoded_sample) in enumerate(zip(decoded_ds_iterator, non_decoded_ds_iterator)):
        image, d_gt = get_sample_data(decoded_sample)
        image_name = get_sample_img_name(non_decoded_sample)
        sample = format_sample(image, image_name, d_gt, args)
        dataset_jsonl.append(sample)

        if i + 1 >= TRAINING_SAMPLES:
            break

    return dataset_jsonl


def main(args: dict):

    decoded_ds_iterator = get_dataset_iterator()
    non_decoded_ds_iterator = get_dataset_iterator(True)
    dataset_jsonl = get_dataset_jsonl(decoded_ds_iterator, non_decoded_ds_iterator, args)
    save_dataset_jsonl("openai_finetuning_dataset.jsonl", dataset_jsonl)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_imgs", required=True, type=str)
    args = parser.parse_args()

    save_imgs = args.save_imgs.lower() in ("true", "1", "yes", "y")
    args = {}
    args["save_imgs"] = save_imgs

    main(args)
