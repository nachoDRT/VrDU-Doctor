import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import torch
import logging
import json
from donut import JSONParseEvaluator
import numpy as np


SAMPLES_LIMIT = 100


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def get_model_and_processor():
    model = VisionEncoderDecoderModel.from_pretrained("de-Rodrigo/donut-merit", subfolder="en-digital-paragraph-degradation-seq")
    model.to(device)
    
    processor = DonutProcessor.from_pretrained("de-Rodrigo/donut-merit")

    return model, processor


def get_sample_data(sample):

    log_info("Getting Image")

    img = sample["image"]

    if img.mode != "RGB":
        img = img.convert("RGB")

    gt = json.loads(sample["ground_truth"])
    gt = gt["gt_parse"]

    return img, gt


def get_dataset_iterator():
    log_info("Loading Dataset")

    dataset = load_dataset("de-Rodrigo/merit", "en-digital-seq", split="test", streaming=True)
    dataset_iterator = iter(dataset)

    return dataset_iterator


def process_dataset():
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    evaluator = JSONParseEvaluator()
    accs = []
    output_list = []

    for i, sample in enumerate(dataset_iterator):
        image, gt = get_sample_data(sample)

        pixel_values = processor(image, return_tensors="pt").pixel_values

        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        
        grades = processor.token2json(sequence)

        score = evaluator.cal_acc(grades, gt)
        
        accs.append(score)
        output_list.append(grades)
        
        print("Ouput", grades)
        print("GT", gt)

        if i + 1 >= SAMPLES_LIMIT:
            break
    
    print(f"Mean accuracy: {np.mean(accs)}")



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model, processor = get_model_and_processor()
    dataset_iterator = get_dataset_iterator()
    process_dataset()