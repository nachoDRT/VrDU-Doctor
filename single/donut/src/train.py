import argparse
import debugpy
import json
import random
import torch
import re
import os
import wandb
from huggingface_hub import login, HfApi
import pytorch_lightning as pl
import numpy as np
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderConfig,
    DonutProcessor,
    VisionEncoderDecoderModel,
)
from typing import Any, List, Tuple
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from nltk import edit_distance
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
from PIL import Image
import ast

HF_CARD_FILES = ["/app/src/card/README.md", "/app/src/card/.huggingface.yaml", "/app/src/card/assets/dragon_huggingface.png"]


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch

        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, labels, answers = batch
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            device=self.device,
        )

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, ""
            )
            seq = re.sub(
                r"<.*?>", "", seq, count=1
            ).strip()  # remove first task start token
            predictions.append(seq)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            # NOT NEEDED ANYMORE
            # answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


class DonutDataset(Dataset):
    """
    PyTorch Dataset for Donut. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into pixel_values (vectorized image) and labels (input_ids of the tokenized string).

    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
        prompt_end_token: the special token at the end of the sequences
        sort_json_key: whether or not to sort the JSON keys
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        subset: str,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = (
            prompt_end_token if prompt_end_token else task_start_token
        )
        self.sort_json_key = sort_json_key

        self.dataset = load_dataset(dataset_name_or_path, name=subset, split=self.split, num_proc=8)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        self.added_tokens = []
        for sample in self.dataset:
            try:
                ground_truth = json.loads(sample["ground_truth"])
            except json.decoder.JSONDecodeError:
                ground_truth = ast.literal_eval(sample["ground_truth"])

            if "gt_parses" in ground_truth:
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            elif "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict):
                gt_jsons = [ground_truth["gt_parse"]]
            elif isinstance(ground_truth, dict):
                # Si el diccionario no está envuelto en una clave, lo usamos directamente
                gt_jsons = [ground_truth]
            else:
                raise ValueError("El formato de ground_truth no es el esperado.")

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + processor.tokenizer.eos_token
                    for gt_json in gt_jsons
                ]
            )

        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(
            self.prompt_end_token
        )

    def json2token(
        self,
        obj: Any,
        update_special_tokens_for_json_key: bool = True,
        sort_json_key: bool = True,
    ):
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
                        self.add_tokens([rf"<s_{k}>", rf"</s_{k}>"])
                    output += (
                        rf"<s_{k}>"
                        + self.json2token(
                            obj[k], update_special_tokens_for_json_key, sort_json_key
                        )
                        + rf"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [
                    self.json2token(
                        item, update_special_tokens_for_json_key, sort_json_key
                    )
                    for item in obj
                ]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
            self.added_tokens.extend(list_of_tokens)

    def __len__(self) -> int:
        return self.dataset_length



    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads the image from the dataset at the given index, processes it into a tensor,
        and tokenizes the corresponding ground truth sequence.

        Returns:
            pixel_values: Preprocessed image tensor.
            labels: Tokenized ground truth sequence with masked tokens (model doesn't need to predict prompt or pad tokens).
            target_sequence: The original ground truth string.
        """
        # Retrieve the sample at index 'idx' from the dataset
        sample = self.dataset[idx]

        # Load the image from the sample
        image = sample["image"]

        # If the image is not a PIL Image, try converting it (e.g., from a NumPy array)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Convert the image to RGB if it's not already (this ensures 3 color channels)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process the image using the processor
        # 'random_padding' is enabled for training to introduce variability
        pixel_values = processor(
            image,
            random_padding=self.split == "train",
            return_tensors="pt"
        ).pixel_values
        # Remove any extra dimensions added by the processor
        pixel_values = pixel_values.squeeze()

        # Randomly select one target sequence from the available ground truth sequences for this sample
        target_sequence = random.choice(self.gt_token_sequences[idx])
        
        # Tokenize the target sequence without adding special tokens
        # The sequence is padded or truncated to 'max_length' and converted into a tensor
        input_ids = processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        # Clone the tokenized input_ids to create labels, and mask out the pad tokens
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id

        # Return the processed image tensor, the labels, and the original target sequence
        return pixel_values, labels, target_sequence


class PushToHubCallback(Callback):
    def __init__(self, model_output_name, dataset_subset, save_dir="checkpoints"):
        self.api = HfApi()
        self.model_output_name = model_output_name
        self.dataset_subset = dataset_subset
        self.save_dir = save_dir

    def on_train_epoch_end(self, trainer, pl_module):
        """Sube el modelo al final de cada epoch."""
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")

        # Save model locally
        epoch_subfolder = os.path.join(self.save_dir, f"{self.model_output_name}_{self.dataset_subset}_epoch{trainer.current_epoch}")
        pl_module.model.save_pretrained(epoch_subfolder)
        pl_module.processor.save_pretrained(epoch_subfolder)

        # Upload model to the hub
        repo_id = f"de-Rodrigo/{self.model_output_name}"
        self.api.upload_folder(
            folder_path=epoch_subfolder,
            path_in_repo=self.dataset_subset,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Training in progress, epoch {trainer.current_epoch}"
        )

        # Upload extra files
        self._upload_card_files(repo_id)

    def on_train_end(self, trainer, pl_module):
        """Sube la versión final del modelo después del entrenamiento."""
        print(f"Pushing model to the hub after training")

        # Save model locally
        final_model_dir = os.path.join(self.save_dir, f"{self.model_output_name}_final")
        pl_module.model.save_pretrained(final_model_dir)
        pl_module.processor.save_pretrained(final_model_dir)

        repo_id = f"de-Rodrigo/{self.model_output_name}"
        
        # Upload model to the hub
        self.api.upload_folder(
            folder_path=final_model_dir,
            path_in_repo=self.dataset_subset,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Training done, final model uploaded"
        )

        # Upload extra files
        self._upload_card_files(repo_id)

    def _upload_card_files(self, repo_id):
        """Sube archivos adicionales como README, config.json, etc."""
        for file in HF_CARD_FILES:
            print(f"Uploading {file} to {repo_id}")
            self.api.upload_file(
                path_or_fileobj=file,
                path_in_repo='/'.join(file.split('/')[4:]),
                repo_id=repo_id,
                repo_type="model",
                commit_message="Uploading additional files"
            )

        
def load_session_datasets(dataset_name: str, subset: str = ""):

    train_dataset = DonutDataset(
        dataset_name,
        subset=subset,
        max_length=max_length,
        split="train",
        task_start_token="<s_cord-v2>",
        prompt_end_token="<s_cord-v2>",
        sort_json_key=False,
    )

    val_dataset = DonutDataset(
        dataset_name,
        subset=subset,
        max_length=max_length,
        split="validation",
        task_start_token="<s_cord-v2>",
        prompt_end_token="<s_cord-v2>",
        sort_json_key=False,
    )

    return train_dataset, val_dataset


def load_secret_dataset(dataset_name: str, subsets: list):
    
    train_datasets = []
    val_datasets = []
    
    for subset in subsets:
        train_ds = DonutDataset(
            dataset_name,
            subset=subset,
            max_length=max_length,
            split="test",
            task_start_token="<s_cord-v2>",
            prompt_end_token="<s_cord-v2>",
            sort_json_key=False,
        )
        val_ds = DonutDataset(
            dataset_name,
            subset=subset,
            max_length=max_length,
            split="test",
            task_start_token="<s_cord-v2>",
            prompt_end_token="<s_cord-v2>",
            sort_json_key=False,
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
    return train_dataset, val_dataset


if __name__ == "__main__":

    # Define parsing values
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_subset", type=str)
    args = parser.parse_args()

    # Debug
    if eval(args.debug):
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger to connect...")
        debugpy.wait_for_client()

    # Define constants
    dataset = args.dataset_name
    dataset_subset = args.dataset_subset
    model_output_name = "".join(["donut-", dataset.split('/')[-1]])

    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

    # Load model and processor
    image_size = [1280, 960]
    max_length = 768

    config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
    config.encoder.image_size = image_size
    config.decoder.max_length = max_length

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base", config=config
    )

    # Dataset instances
    processor.image_processor.size = image_size[::-1]
    processor.image_processor.do_align_long_axis = False

    if dataset == "de-Rodrigo/merit-secret":
        subsets_disponibles = ['britanico', 'fomento', 'maravillas', 'mater', 'montealto', 'pilar', 'recuerdo', 'retamar', 'sanpablo', 'sanpatricio']
        train_dataset, val_dataset = load_secret_dataset(dataset, subsets_disponibles)
    
    else:
        train_dataset, val_dataset = load_session_datasets(dataset_name=dataset, subset=dataset_subset)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        ["<s_cord-v2>"]
    )[0]

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    percentage = 0.1
    num_samples = int(len(val_dataset) * percentage)
    indices = torch.randperm(len(val_dataset))[:num_samples].tolist()
    sampler = SubsetRandomSampler(indices)
    val_dataloader = DataLoader(val_dataset, batch_size=1, sampler=sampler, num_workers=4)



    # Train
    config = {
        "max_steps": 20000,
        "val_check_interval": 0.5,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "num_training_samples_per_epoch": 800,
        "lr": 3e-5,
        "train_batch_sizes": [8],
        "val_batch_sizes": [1],
        "num_nodes": 1,
        "warmup_steps": 300,
        "result_path": "./result",
        "verbose": True,
    }

    model_module = DonutModelPLModule(config, processor, model)

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_logger = WandbLogger(project="Donut", name=dataset_subset)

    early_stop_callback = EarlyStopping(
        monitor="val_edit_distance", patience=4, verbose=False, mode="min"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.get("max_epochs"),
        max_steps=config.get("max_steps"),
        val_check_interval=config.get("val_check_interval"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision=16,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[PushToHubCallback("donut-merit", dataset_subset)],
    )

    trainer.fit(model_module)
