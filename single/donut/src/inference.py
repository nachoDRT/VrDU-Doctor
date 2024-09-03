import re
import os
import cv2
import torch
import utils
import logging
import numpy as np
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel


SALIENCY = True
SALIENCIES_ROOT = "/app/saliencies/files/"


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def get_donut():
    log_info("Loading Model and Processor")

    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )

    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )

    return model, processor


def get_image():
    log_info("Loading Image")

    dataset = load_dataset("hf-internal-testing/example-documents")
    img = dataset["test"][2]["image"]

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
        file_name = f"saliency_{safe_token_text}.png"

        saliency = utils.convert_tensor_to_rgba_image(saliency)

        """Merge saliency image twice 1st: remove black background and fuse, 
        2nd fuse again to still see original document"""
        saliency = utils.add_transparent_image(np.array(image), saliency)
        saliency = utils.convert_rgb_to_rgba_image(saliency)
        saliency = utils.add_transparent_image(np.array(image), saliency, 0.7)

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


if __name__ == "__main__":

    # Project config
    logging.basicConfig(level=logging.INFO)

    # Load model and processor
    model, processor = get_donut()

    # Get image
    image = get_image()

    # Prepare image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Get output
    output = compute_output(model, processor, pixel_values, image)
    print(output)
