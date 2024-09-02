import re
import torch
import logging
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel


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


def compute_output(donut_m, donut_p):
    log_info("Computing Output")

    task_prompt = ""
    decoder_input_ids = donut_p.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    donut_m.to(device)

    outputs = donut_m.generate(
        pixel_values.to(device),
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
    output = compute_output(model, processor)
    print(output)
