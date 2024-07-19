import argparse
import debugpy
import torch
import random
import json
import re
import os
import wandb
import lightning as L
import numpy as np
from huggingface_hub import login
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Dict
from nltk import edit_distance
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


"""
Script extracted from Niels Rogge GitHub with minor modifications
You can find Niels project here: 
https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Idefics2
"""


USE_LORA = False
USE_QLORA = True
USE_ADD_ADAPTER = True
MAX_LENGTH = 768
MODEL_REPO_ID = "de-Rodrigo/idefics2-merit"


class Idefics2Dataset(Dataset):
    """
    PyTorch Dataset for Idefics2. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt).
    """

    def __init__(
        self,
        processor,
        model,
        dataset_name_or_path: str,
        subset: str,
        split: str = "train",
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        self.processor = processor
        self.model = model

        self.dataset = load_dataset(dataset_name_or_path, name=subset, split=self.split)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        self.added_tokens = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    for gt_json in gt_jsons
                ]
            )

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.added_tokens:
                obj = f"<{obj}/>"
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = self.rocessor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
            self.added_tokens.extend(list_of_tokens)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        sample = self.dataset[idx]

        # Inputs
        image = sample["image"]
        target_sequence = random.choice(self.gt_token_sequences[idx])

        return image, target_sequence


class Idefics2ModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, pixel_attention_mask, labels = batch

        outputs = self.model(input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, pixel_attention_mask, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask,
            max_new_tokens=768)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn(self.processor, self.model), batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn(self.processor, self.model), batch_size=self.batch_size, shuffle=False, num_workers=4)


class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(MODEL_REPO_ID,
            commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub(MODEL_REPO_ID,
            commit_message=f"Training done")
        pl_module.model.push_to_hub(MODEL_REPO_ID,
            commit_message=f"Training done")


def load_model() -> Idefics2ForConditionalGeneration:
    """ Three options for training, from the lowest precision training to the highest 
    precision training:
        - QLora
        - Standard Lora
        - Full fine-tuning
    """
    
    if USE_QLORA or USE_LORA:
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        loadel_model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            quantization_config=bnb_config if USE_QLORA else None,
        )
        if USE_ADD_ADAPTER:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
                use_dora=False if USE_QLORA else True,
                init_lora_weights="gaussian",
            )
            loadel_model.add_adapter(lora_config)
            loadel_model.enable_adapters()
    else:
        """For full fine-tuning, we can speed up the model using Flash Attention, only available
        on certain devices, see: https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features"""
        loadel_model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
        )

    return loadel_model


def apply_peft():
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian",
    )

    peft_model = prepare_model_for_kbit_training(peft_model)
    peft_model = get_peft_model(peft_model, lora_config)

    return peft_model


def train_collate_fn(examples, processor, model):
    texts = []
    images = []
    for example in examples:
        image, ground_truth = example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract JSON."},
                    {"type": "image"},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ground_truth}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == model.config.image_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    pixel_attention_mask = batch["pixel_attention_mask"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, pixel_attention_mask, labels


def eval_collate_fn(examples, processor, model):
    images = []
    texts = []
    answers = []
    for example in examples:
        image, ground_truth = example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract JSON."},
                    {"type": "image"},
                ]
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images.append([image])
        texts.append(text.strip())
        answers.append(ground_truth)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    pixel_attention_mask = batch["pixel_attention_mask"]

    return input_ids, attention_mask, pixel_values, pixel_attention_mask, answers


def init_pl_module(processor, model):
    configuration = {"max_epochs": 10,
        "val_check_interval": 1.0,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 8,
        "lr": 1e-4,
        "batch_size": 2,
        "precision": "16-mixed",
        # "seed":2022,
        "warmup_steps": 50,
        "result_path": "./result",
        "verbose": True,
    }

    model_module = Idefics2ModelPLModule(configuration, processor, model)

    return model_module, configuration


def train_idefics2(idefics2, configuration):
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_logger = WandbLogger(project="Idefics2", name="merit")

    trainer = L.Trainer(
            accelerator="gpu",
            devices="auto", # Use available GPUs
            max_epochs=configuration.get("max_epochs"),
            check_val_every_n_epoch=configuration.get("check_val_every_n_epoch"),
            gradient_clip_val=configuration.get("gradient_clip_val"),
            accumulate_grad_batches=configuration.get("accumulate_grad_batches"),
            precision=configuration.get("precision"),
            num_sanity_val_steps=0,
            logger=wandb_logger,
            callbacks=[PushToHubCallback(), early_stop_callback],
    )

    trainer.fit(idefics2)


if __name__ == "__main__":

    # Define parsing values
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    args = parser.parse_args()

    # Debug
    if eval(args.debug):
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger to connect...")
        debugpy.wait_for_client()

    # Load Processor
    idefics2_processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)

    # Load Model
    idefics2 = load_model()

    # Load dataset partitions
    train_dataset = Idefics2Dataset(args.dataset, subset=args.subset, split="train", sort_json_key=False)
    val_dataset = Idefics2Dataset(args.dataset, subset=args.subset, split="validation", sort_json_key=False)
    
    # Apply Parameter-Efficient Fine-Tuning (PEFT)
    if not USE_ADD_ADAPTER:
        idefics2 = apply_peft()

    # Collate Function
    image_token_id = idefics2_processor.tokenizer.additional_special_tokens_ids[idefics2_processor.tokenizer.additional_special_tokens.index("<image>")]

    # Callback
    early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=2, verbose=False, mode="min")

    # Pytorch Lightning module
    idefics2_module, config = init_pl_module(idefics2_processor, idefics2)
    
    # Train
    train_idefics2(idefics2_module, config)
    