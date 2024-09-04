import re
import os
import cv2
import torch
import utils
import logging
import numpy as np
from PIL import Image
from datetime import datetime
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel


SALIENCY = False
SALIENCIES_ROOT = "/app/saliencies/files/"
SAMPLES_LIMIT = 3


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def get_donut():
    log_info("Loading Model and Processor")

    model = VisionEncoderDecoderModel.from_pretrained("de-Rodrigo/donut-merit")

    processor = DonutProcessor.from_pretrained("de-Rodrigo/donut-merit")

    return model, processor


def resize_image(image, new_width):

    original_width, original_height = image.size
    new_height = int((new_width / original_width) * original_height)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image


def get_dataset_iterator():
    log_info("Loading Dataset")

    dataset = load_dataset(
        "de-Rodrigo/merit", "en-digital-seq", split="train", streaming=True
    )
    dataset_iterator = iter(dataset)

    return dataset_iterator


def get_image(sample):
    log_info("Getting Image")

    # sample = next(iterator)
    img = sample["image"]

    if SALIENCY:
        img = resize_image(img, 512)

    return img


def compute_saliency(outputs, pixels, donut_p, image):

    token_logits = torch.stack(outputs.scores, dim=1)
    token_probs = torch.softmax(token_logits, dim=-1)
    token_texts = []

    for token_index in range(len(token_probs[0])):

        target_token_prob = token_probs[
            0, token_index, outputs.sequences[0, token_index]
        ]

        if pixels.grad is not None:
            pixels.grad.zero_()

        target_token_prob.backward(retain_graph=True)

        saliency = pixels.grad.data.abs().squeeze().mean(dim=0)

        token_id = outputs.sequences[0][token_index].item()
        token_text = donut_p.tokenizer.decode([token_id])
        log_info(f"Considered sequence token: {token_text}")

        safe_token_text = re.sub(r'[<>:"/\\|?*]', "_", token_text)
        current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")

        unique_safe_token_text = f"{safe_token_text}_{current_datetime}"
        file_name = f"saliency_{unique_safe_token_text}.png"

        saliency = utils.convert_tensor_to_rgba_image(saliency)

        """Merge saliency image twice 1st: remove black background and fuse, 
        2nd fuse again to still see original document"""
        saliency = utils.add_transparent_image(np.array(image), saliency)
        saliency = utils.convert_rgb_to_rgba_image(saliency)
        saliency = utils.add_transparent_image(np.array(image), saliency, 0.7)

        saliency = utils.label_frame(saliency, token_text)

        save_img(saliency, os.path.join(SALIENCIES_ROOT, file_name))
        token_texts.append(token_text)

    utils.saliency_video(SALIENCIES_ROOT, token_texts)

    return token_index


def save_img(img, path):

    if img.dtype != np.uint8:
        img = (255 * img / np.max(img)).astype(np.uint8)
    cv2.imwrite(path, img)


def compute_output(donut_m, donut_p, pixels, image):
    log_info("Computing Output")

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = donut_p.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    donut_m.to(device)

    pixels = pixels.to(device)
    pixels.requires_grad = True

    outputs = donut_m.generate.__wrapped__(
        model,
        pixels,
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=donut_m.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=donut_p.tokenizer.pad_token_id,
        eos_token_id=donut_p.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[donut_p.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
    )

    if SALIENCY:
        compute_saliency(outputs, pixels, donut_p, image)

    sequence = donut_p.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(donut_p.tokenizer.eos_token, "").replace(
        donut_p.tokenizer.pad_token, ""
    )
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    return sequence


def process_dataset(dataset_iterator):

    for i, sample in enumerate(dataset_iterator):

        # Get image
        image = get_image(sample)

        # Prepare image
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Get output
        output = compute_output(model, processor, pixel_values, image)
        print(output)

        if i + 1 >= SAMPLES_LIMIT:
            break


if __name__ == "__main__":

    # Project config
    logging.basicConfig(level=logging.INFO)

    # Load model and processor
    model, processor = get_donut()

    # Get dataset
    dataset_iter = get_dataset_iterator()

    # Process datset
    process_dataset(dataset_iter)
