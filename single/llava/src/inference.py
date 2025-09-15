from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from lightning.pytorch.loggers import WandbLogger
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import HfFolder
from donut import JSONParseEvaluator
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
import re
import argparse
import torch
import os
import wandb
import numpy as np
import logging
from tqdm import tqdm
from PIL import Image
import json


WANDB_PROJECT = "Llava"
PROMPT = f"USER: <image>\nExtract JSON.\nASSISTANT:"
LIMIT = 218
MAX_LENGTH = 384


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def init_hf_hub():
    HfFolder.save_token(os.environ["HUGGINGFACE_HUB_TOKEN"])


def init_wandb():
    session_name = f"llava_test_{subfolder}"
    # Inicia un run y un logger de Lightning
    run = wandb.init(
        project=WANDB_PROJECT,
        name=session_name,
        entity="ciclab-comillas",
        reinit=True
    )
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=session_name,
        entity="ciclab-comillas",
        log_model=False
    )
    return run, wandb_logger


def get_dataset_iterator(dataset_name: str, subset_name: str):
    log_info("Loading Dataset")

    if dataset_name in ("de-Rodrigo/merit", "de-Rodrigo/merit-secret"):
        dataset = load_dataset(
            dataset_name, subset_name, split="test", streaming=True
        )
    else:
        dataset = load_dataset(
            dataset_name, split="test", streaming=True
        )
    dataset_iterator = iter(dataset)

    return dataset_iterator


def get_dataset_iterator(dataset_name: str, subset_name: str):
    log_info("Loading Dataset")

    if dataset_name in ("de-Rodrigo/merit", "de-Rodrigo/merit-secret"):
        dataset = load_dataset(
            dataset_name, subset_name, split="test", streaming=True
        )
    else:
        dataset = load_dataset(
            dataset_name, split="test", streaming=True
        )
    dataset_iterator = iter(dataset)

    return dataset_iterator


def get_sample_data(sample):

    img = sample["image"]

    # If the image is not a PIL Image, try converting it (e.g., from a NumPy array)
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    # Convert the image to RGB if it's not already (this ensures 3 color channels)
    if img.mode != "RGB":
        img = img.convert("RGB")

    if dataset_name in ("de-Rodrigo/merit",
        "naver-clova-ix/cord-v2", 
        "de-Rodrigo/merit-secret"):
        
        gt = sample["ground_truth"]
        if dataset_name in ("de-Rodrigo/merit", "de-Rodrigo/merit-secret"):
            gt = gt.replace("'", '"')
        gt = json.loads(gt)

        if dataset_name == "naver-clova-ix/cord-v2":
            gt = gt["gt_parse"]
        
    else:
        ocr_words = sample["ocr_words"]
        words_list = [{"word": word} for word in ocr_words]
        page = {"page_0": words_list}
        gt = {"gt_parse": page}

    # print(gt)


    return img, gt

def token2json(tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        if added_vocab is None:
            added_vocab = processor.tokenizer.get_added_vocab()

        output = {}

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            key_escaped = re.escape(key)

            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def process_dataset(dataset_iterator):
    evaluator = JSONParseEvaluator()
    accs = []
    output_list = []

    for i, sample in tqdm(enumerate(dataset_iterator)):

        # Get image and ground truth
        image, gt = get_sample_data(sample)

        inputs = processor(text=PROMPT, images=image, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Autoregressively generate
        # We use greedy decoding here, for more fancy methods see https://huggingface.co/blog/how-to-generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

            # Next we turn each predicted token ID back into a string using the decode method
            # We chop of the prompt, which consists of image tokens and our text prompt
            image_token_index = model.config.image_token_index
            num_image_tokens = len(generated_ids[generated_ids==image_token_index])
            num_text_tokens = len(processor.tokenizer.encode(PROMPT))
            num_prompt_tokens = num_image_tokens + num_text_tokens + 2
            generated_text = processor.batch_decode(generated_ids[:, num_prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            generated_json = token2json(generated_text)
            # print("PREDICTION", generated_json)

            score = evaluator.cal_acc(generated_json, gt)

            accs.append(score)
            output_list.append(generated_json)

            if i >= LIMIT:
                break

    return accs, output_list


def get_llava(llava_model_version: str, subfolder: str):

    # Define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )

    # Load the base model with adapters on top
    model = LlavaForConditionalGeneration.from_pretrained(
        llava_model_version,
        subfolder=subfolder,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    
    processor = AutoProcessor.from_pretrained(llava_model_version, subfolder=subfolder)

    return model, processor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--subset_name", type=str)
    parser.add_argument("--llava_model_version", type=str)
    parser.add_argument("--subfolder", type=str)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    subset_name = args.subset_name
    llava_model_version = args.llava_model_version
    subfolder = args.subfolder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_hf_hub()
    run, wandb_logger = init_wandb()

    model, processor = get_llava(llava_model_version, subfolder)


    if subset_name == "all":
        subsets = get_dataset_config_names(dataset_name)
    else:
        subsets = [subset_name]

    model_acc = []
    
    for subset_name in subsets:
        print(f"Processing {subset_name}")
        dataset_iter = get_dataset_iterator(dataset_name, subset_name)

        # Process dataset
        accs, outputs_list = process_dataset(dataset_iter)

        f1 = np.mean(accs)
        print(f"Mean accuracy {subset_name}: {f1}")
        print(accs)
        model_acc.extend(accs)

    print("\n", model_acc)
    wandb.log({"test_f1": np.mean(model_acc)})
    run.finish()