import torch
import numpy as np
import json
import wandb
import os
import shutil
import argparse
from typing import Optional, Union
from datasets import load_metric, load_dataset

os.environ["WANDB_SILENT"] = "true"

from transformers import (
    LayoutLMv2ForTokenClassification,
    LayoutLMv2FeatureExtractor,
    LayoutXLMTokenizer,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from transformers.file_utils import PaddingStrategy
from torch.utils.data import DataLoader
from dataclasses import dataclass

LOAD_DATASET_FROM_PY = "/app/src/load_dataset.py"
WANDB_LOGGING_PATH = "/app/config/wandb_logging.json"
HUGGINGFACE_LOGGING_PATH = "/app/config/huggingface_logging.json"
DATASET_FOLDER = "/app/data/train-val/spanish/"

MAX_TRAIN_STEPS = 5000
EVAL_FRECUENCY = 500
LOGGING_STEPS = 1
CONSISTENCY = 20


def get_dataset_name() -> str:
    dataset_name = ""

    for subset in os.listdir(DATASET_FOLDER):
        if dataset_name != "":
            dataset_name += "-"
        dataset_name += subset

    return dataset_name


def get_training_session_name(args: argparse.Namespace, wandb_config: dict) -> str:

    if args.dataset_name:
        name = "".join([wandb_config["name"], "_", args.dataset_name])
    else:
        dataset_name = get_dataset_name()
        name = "".join([wandb_config["name"], "_", dataset_name])

    return name


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


def delete_gpu_data(model: LayoutLMv2ForTokenClassification, trainer: Trainer):
    del model
    del trainer
    torch.cuda.empty_cache()


@dataclass
class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    feature_extractor: LayoutLMv2FeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # prepare image input
        image = self.feature_extractor(
            [feature["original_image"] for feature in features], return_tensors="pt"
        ).pixel_values

        # prepare text input
        for feature in features:
            del feature["image"]
            del feature["id"]
            del feature["original_image"]
            del feature["entities"]
            del feature["relations"]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["image"] = image

        return batch


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Define parsing values
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=False)
    args = parser.parse_args()

    # Logging in wandb
    with open(WANDB_LOGGING_PATH) as f:
        wandb_config = json.load(f)

    training_session_name = get_training_session_name(args, wandb_config)
    wandb.login()

    # Load dataset using a '.py' file
    dataset = load_dataset(
        LOAD_DATASET_FROM_PY, "xfun_es", download_mode="force_redownload"
    )
    dataset.shuffle()

    labels = dataset["train"].features["labels"].feature.names

    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in enumerate(labels)}

    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")

    data_collator = DataCollatorForTokenClassification(
        feature_extractor,
        tokenizer,
        pad_to_multiple_of=None,
        padding="max_length",
        max_length=512,
    )

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator)

    output_dir = "/app/models_output"
    if not os.path.exists(output_dir):
        print(f"{output_dir} does not exist")
        os.makedirs(output_dir)

    for i in range(CONSISTENCY):

        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config["entity"],
            name="".join([training_session_name, "_", str(i)]),
        )

        model = LayoutLMv2ForTokenClassification.from_pretrained(
            "microsoft/layoutxlm-base", id2label=id2label, label2id=label2id
        )

        # Metrics
        metric = load_metric("seqeval")
        return_entity_level_metrics = False

        args = TrainingArguments(
            output_dir="/app/models_output",
            max_steps=MAX_TRAIN_STEPS,
            learning_rate=2.5e-5,
            fp16=True,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            push_to_hub=False,
            remove_unused_columns=False,
            logging_strategy="steps",
            logging_steps=LOGGING_STEPS,
            evaluation_strategy="steps",
            eval_steps=EVAL_FRECUENCY,
            report_to="wandb",
            load_best_model_at_end=True,
            save_total_limit=1,
        )

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train
        trainer.train()
        # Save best iteration model
        best_checkpoint_dir = (
            f"/app/models_output/{training_session_name}/iteration_{i}"
        )
        trainer.save_model(best_checkpoint_dir)

        # Test
        test_results = trainer.predict(test_dataset)

        wandb.log(
            {
                "test_loss": test_results.metrics["test_loss"],
                "test_accuracy": test_results.metrics["test_accuracy"],
                "test_precision": test_results.metrics["test_precision"],
                "test_recall": test_results.metrics["test_recall"],
                "test_f1": test_results.metrics["test_f1"],
            }
        )
        wandb.finish()

        delete_gpu_data(model, trainer)
