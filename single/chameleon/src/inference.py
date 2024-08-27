import torch
import requests
import os
import logging
from huggingface_hub import login
from PIL import Image
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

CHAMELEON_DIR = os.path.join(os.getcwd(), "chameleon_files")


def load_chameleon():
    chameleon_processor = ChameleonProcessor.from_pretrained(
        os.path.join(CHAMELEON_DIR, "processor")
    )
    model = ChameleonForConditionalGeneration.from_pretrained(
        os.path.join(CHAMELEON_DIR, "model")
    )

    return model, chameleon_processor


def download_chameleon():
    if not os.path.exists(CHAMELEON_DIR):
        os.makedirs(CHAMELEON_DIR)
        logging.info(f"New directory created: {CHAMELEON_DIR}")
    else:
        logging.info(f"{CHAMELEON_DIR} already exists")

    logging.info("Downloading Chameleon Processor")
    processor = ChameleonProcessor.from_pretrained(
        "facebook/chameleon-7b", cache_dir=os.path.join(CHAMELEON_DIR, "processor")
    )

    logging.info("Downloading Chameleon Model")
    model = ChameleonForConditionalGeneration.from_pretrained(
        "facebook/chameleon-7b",
        torch_dtype=torch.float16,
        device_map="cuda",
        cache_dir=os.path.join(CHAMELEON_DIR, "model"),
    )

    return model, processor


if __name__ == "__main__":

    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    logging.basicConfig(level=logging.INFO)

    try:
        logging.info("Loading downloaded files")
        model, processor = load_chameleon()
    except OSError:
        logging.info("No downloaded files found: downloading files")
        model, processor = download_chameleon()

    # prepare image and text prompt
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = "What do you see in this image?<image>"

    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    inputs = {k: v.to(torch.float16) for k, v in inputs.items()}

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=50)
    print(processor.decode(output[0], skip_special_tokens=True))
