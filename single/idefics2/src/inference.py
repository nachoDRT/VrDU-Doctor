import json
import utils
import torch
import logging
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Idefics2ForConditionalGeneration,
)
from datasets import load_dataset
from huggingface_hub import hf_hub_download


"""
Restructured cript extracted from Niels Rogge GitHub with minor modifications
You can find Niels project here: 
https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Idefics2
"""

EMBEDDINGS_REPO = "nielsr/idefics2-embeddings"
DATASET = {"name": "naver-clova-ix/cord-v2", "subset": "", "split": "train"}
SAMPLES_LIMIT = 2
SALIENCY = False


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def log_error(msg: str):
    print("")
    logging.error(msg)
    print("")


def resize_embeddings(model, processor, repo_name):

    try:
        # Input embeddings
        filepath = hf_hub_download(
            repo_id=repo_name,
            filename="input_embeddings.pt",
            repo_type="dataset",
        )
        input_embeddings = torch.load(filepath, map_location="cuda:0")
        input_embeddings_module = torch.nn.Embedding(
            input_embeddings.shape[0],
            input_embeddings.shape[1],
            _weight=input_embeddings,
        )

        # Output embeddings
        filepath = hf_hub_download(
            repo_id=repo_name,
            filename="output_embeddings.pt",
            repo_type="dataset",
        )
        output_embeddings = torch.load(filepath, map_location="cuda:0")
        output_embeddings_module = torch.nn.Linear(
            output_embeddings.shape[0], output_embeddings.shape[1], bias=False
        )
        output_embeddings_module.weight = output_embeddings

        # Set them accordingly
        model.resize_token_embeddings(len(processor.tokenizer))
        model.set_input_embeddings(input_embeddings_module)
        model.set_output_embeddings(output_embeddings_module)

    except Exception as e:
        log_error(f"Error when resizing embeddigns: {e}")
        log_info("Unable to load saved embeddings. Resizing embeddigns from scratch")
        dummy_dataset = utils.Idefics2Dataset(
            processor, model, DATASET["name"], DATASET["subset"], DATASET["train"]
        )

    return model


def get_idefics2():
    peft_model_id = "nielsr/idefics2-cord-demo"

    processor = AutoProcessor.from_pretrained(peft_model_id)

    # Define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # Load the base model with adapters on top
    model = Idefics2ForConditionalGeneration.from_pretrained(
        peft_model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    model = resize_embeddings(model, processor, EMBEDDINGS_REPO)

    return model, processor


def get_dataset_iterator():
    log_info("Loading Dataset")

    dataset = load_dataset(
        DATASET["name"], DATASET["subset"], split="test", streaming=True
    )
    dataset_iterator = iter(dataset)

    return dataset_iterator


def config_prompt(processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract JSON."},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    return prompt


def resize_image(image, new_width):

    original_width, original_height = image.size
    new_height = int((new_width / original_width) * original_height)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image


def get_sample_data(sample):

    log_info("Getting Image")

    img = sample["image"]
    gt = json.loads(sample["ground_truth"])
    gt = gt["gt_parse"]

    if SALIENCY:
        img = resize_image(img, 512)

    return img, gt


def process_dataset(dataset_iterator, model, processor, prompt):

    for i, sample in enumerate(dataset_iterator):

        # Get image and ground truth
        image, gt = get_sample_data(sample)

        # Get inputs
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

        # Generate token IDs
        generated_ids = model.generate(**inputs, max_new_tokens=768)

        # Decode back into text
        generated_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Convert to dict
        generated_json = utils.token2json(generated_texts[0], processor)

        log_info(f"Grades detected: {generated_json}")

        if i + 1 >= SAMPLES_LIMIT:
            break


if __name__ == "__main__":

    # Project config
    logging.basicConfig(level=logging.INFO)

    # Load model and processor
    i2_model, i2_processor = get_idefics2()

    # Get test dataset
    dataset_iter = get_dataset_iterator()

    # Config prompt
    i2_prompt = config_prompt(i2_processor)

    # Inference
    process_dataset(dataset_iter, i2_model, i2_processor, i2_prompt)
