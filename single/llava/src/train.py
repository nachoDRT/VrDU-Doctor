import argparse
import os
import random
import torch
import json
import re
import wandb
import numpy as np
import ast
from huggingface_hub import login
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration, AutoModel, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from torch.utils.data import Dataset, ConcatDataset
from typing import Any, Dict
from datasets import load_dataset
import lightning as L
from torch.utils.data import DataLoader, SubsetRandomSampler
from nltk import edit_distance
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from huggingface_hub import HfApi
import copy


MAX_LENGTH = 384
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
WANDB_PROJECT = "Llava"
HF_CARD_FILES = [
    "/app/src/card/README.md",
    "/app/src/card/.huggingface.yaml",
    "/app/src/card/assets/dragon_huggingface.png",
]


class LlavaDataset(Dataset):
    """
    PyTorch Dataset for LLaVa. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and ground truth data (json/jsonl/txt).
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        subset: str,
        split: str = "train",
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        if dataset_name_or_path == "de-Rodrigo/merit" or dataset_name_or_path == "de-Rodrigo/merit-secret":
            self.dataset = load_dataset(dataset_name_or_path, name=subset, split=self.split, num_proc=8)
        else:
            self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        if dataset_name_or_path in (
            "de-Rodrigo/merit", 
            "naver-clova-ix/cord-v2", 
            "de-Rodrigo/merit-secret"
        ):
            for sample in self.dataset:
                if dataset_name_or_path == "de-Rodrigo/merit-secret":
                    ground_truth = ast.literal_eval(sample["ground_truth"])
                else:
                    ground_truth = json.loads(sample["ground_truth"])
                
                if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                    assert isinstance(ground_truth["gt_parses"], list)
                    gt_jsons = ground_truth["gt_parses"]
                elif isinstance(ground_truth, dict):
                    gt_jsons = [ground_truth]
                else:
                    assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                    gt_jsons = [ground_truth["gt_parse"]]

                self.gt_token_sequences.append(
                    [
                        self.json2token(
                            gt_json,
                            sort_json_key=self.sort_json_key,
                        )
                        for gt_json in gt_jsons  # load json from list of json
                    ]
                )
        else:
            for sample in self.dataset:
                ocr_words = sample["ocr_words"]
                words_list = [{"word": word} for word in ocr_words]
                page = {"page_0": words_list}
                ground_truth = {"gt_parse": page}

                if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                    assert isinstance(ground_truth["gt_parses"], list)
                    gt_jsons = ground_truth["gt_parses"]
                else:
                    assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                    gt_jsons = [ground_truth["gt_parse"]]

                self.gt_token_sequences.append(
                    [
                        self.json2token(
                            gt_json,
                            sort_json_key=self.sort_json_key,
                        )
                        for gt_json in gt_jsons  # load json from list of json
                    ]
                )

    def json2token(self, obj: Any, sort_json_key: bool = True):
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
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

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

        # inputs
        image = sample["image"]
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1

        return image, target_sequence


class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model, freeze_encoder):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

        if freeze_encoder:
            if hasattr(self.model, "vision_tower"):
                for p in self.model.vision_tower.parameters():
                    p.requires_grad = False
                self.model.vision_tower.eval()
            elif hasattr(self.model, "vision_model"):
                for p in self.model.vision_model.parameters():
                    p.requires_grad = False
                self.model.vision_model.eval()
            else:
                print("⚠️ No se encontró vision encoder en el modelo PaliGemma")


    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, labels = batch

        outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
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
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        percentage = 0.1
        num_samples = int(len(val_dataset) * percentage)
        # índices aleatorios sin reemplazo
        indices = torch.randperm(len(val_dataset))[:num_samples].tolist()
        sampler = SubsetRandomSampler(indices)

        return DataLoader(val_dataset, sampler=sampler, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)


class PushToHubCallback(Callback):
    """
    Callback para LLaVA + LoRA + 4bit:
    - Sube al Hub (solo si mejora 'monitor').
    - Fusiona LoRA en un modelo completo (fp16) para publicación.
    - Sube también el adapter LoRA en subcarpeta 'adapter/'.
    - Sube el processor.
    - Actualiza ficheros de la card al final del entrenamiento.

    Parámetros:
      model_output_name: nombre base del repo en el Hub (de-Rodrigo/<model_output_name>)
      dataset_subset: subcarpeta en el repo para tus checkpoints por subset
      monitor: métrica a vigilar en trainer.callback_metrics (p.ej. "val_loss", "val_cider", etc.)
      mode: "min" o "max" según si la métrica se minimiza o maximiza
      save_dir: carpeta local donde volcar temporalmente los artefactos antes de subir
    """
    def __init__(
        self,
        model_output_name: str,
        dataset_subset: str,
        monitor: str = "val_edit_distance",
        mode: str = "min",
        save_dir: str = "checkpoints",
    ):
        super().__init__()
        self.api = HfApi()
        self.model_output_name = model_output_name
        self.dataset_subset = dataset_subset
        self.save_dir = save_dir
        self.monitor = monitor

        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else -float('inf')

        # Repo destino
        self.repo_id = f"de-Rodrigo/{self.model_output_name}"

        # Asegura carpeta local
        os.makedirs(self.save_dir, exist_ok=True)

    # ---------- Lightning hooks ----------

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        improved = (
            current_score < self.best_score if self.mode == "min"
            else current_score > self.best_score
        )
        if improved:
            print(f"[PushToHubCallback] {self.monitor} mejora: {self.best_score} -> {current_score}")
            self.best_score = current_score
            self._push_model(trainer, pl_module, epoch=trainer.current_epoch)

    def on_train_end(self, trainer, pl_module):
        print("[PushToHubCallback] Entrenamiento finalizado. Subiendo ficheros de la card…")
        self._upload_card_files(self.repo_id)

    # ---------- Helpers ----------

    def _is_4bit(self, model: torch.nn.Module) -> bool:
        """
        Detecta carga en 4-bit aunque el modelo esté envuelto por PEFT.
        """
        # Intento obtener el modelo base real
        base = getattr(model, "get_base_model", lambda: None)()
        if base is None:
            base = getattr(model, "base_model", model)
            base = getattr(base, "model", base)

        for attr in ("is_loaded_in_4bit", "is_quantized"):
            flag = getattr(base, attr, False)
            if isinstance(flag, bool) and flag:
                return True
        return False

    def _push_model(self, trainer, pl_module, epoch: int):
        """
        Publica TODO en una sola carpeta del Hub:
        - adapter LoRA (adapter_config.json + adapter_model.safetensors)
        - modelo completo fusionado (fp16, shardeado si hace falta)
        - processor/tokenizer

        Compatible con LLaVA + QLoRA/Accelerate (evita .to('cpu') en 4-bit).
        """


        save_path = os.path.join(
            self.save_dir,
            f"{self.model_output_name}_{self.dataset_subset}_epoch{epoch}",
        )
        os.makedirs(save_path, exist_ok=True)

        def _infer_base_id(peft_or_base_model):
            base = getattr(peft_or_base_model, "get_base_model", lambda: None)()
            if base is None:
                base = getattr(peft_or_base_model, "base_model", peft_or_base_model)
                base = getattr(base, "model", base)
            cand = getattr(base, "name_or_path", None)
            if isinstance(cand, str) and cand:
                return cand
            cfg = getattr(base, "config", None)
            for attr in ("_name_or_path", "name_or_path"):
                cand = getattr(cfg, attr, None) if cfg is not None else None
                if isinstance(cand, str) and cand:
                    return cand
            return getattr(self, "base_model_id", None)

        def _load_base_cpu_fp16(base_id: str):
            cfg = AutoConfig.from_pretrained(base_id, trust_remote_code=True)
            # Intento específico para LLaVA si está disponible
            try:
                from transformers import LlavaForConditionalGeneration
                if getattr(cfg, "model_type", None) in {"llava", "llava_next", "llava-onevision"}:
                    return LlavaForConditionalGeneration.from_pretrained(
                        base_id,
                        torch_dtype=torch.float16,
                        device_map=None,
                        low_cpu_mem_usage=False,
                        trust_remote_code=True,
                    )
            except Exception:
                pass
            # Fallback estable
            try:
                return AutoModelForCausalLM.from_pretrained(
                    base_id,
                    torch_dtype=torch.float16,
                    device_map=None,
                    low_cpu_mem_usage=False,
                    trust_remote_code=True,
                )
            except Exception:
                return AutoModel.from_pretrained(
                    base_id,
                    torch_dtype=torch.float16,
                    device_map=None,
                    trust_remote_code=True,
                )

        # 1) Guardar adapter en la MISMA carpeta (esto NO crea config.json)
        is_peft = isinstance(pl_module.model, PeftModel)
        if is_peft:
            pl_module.model.save_pretrained(save_path)

        try:
            # 2) Cargar base limpio en CPU y fusionar adapter (o usar base si no hay PEFT)
            base_id = _infer_base_id(pl_module.model)
            if not base_id:
                raise RuntimeError(
                    "No pude inferir 'base_id'. Pásalo en el callback como self.base_model_id."
                )

            print(f"[PushToHubCallback] Cargando base limpio en CPU: {base_id}")
            base_cpu = _load_base_cpu_fp16(base_id)

            if is_peft:
                print("[PushToHubCallback] Aplicando adapter y fusionando…")

                merged = PeftModel.from_pretrained(base_cpu, save_path).merge_and_unload()
            else:
                merged = base_cpu

            # 3) Guardar modelo completo + generation_config (si existe)
            print("[PushToHubCallback] Guardando modelo completo + processor/tokenizer…")
            merged.save_pretrained(save_path, safe_serialization=True, max_shard_size="1GB")
            if getattr(merged, "generation_config", None) is not None:
                merged.generation_config.save_pretrained(save_path)

            # 4) Guardar processor/tokenizer
            if hasattr(pl_module, "processor") and pl_module.processor is not None:
                pl_module.processor.save_pretrained(save_path)
            elif hasattr(pl_module, "tokenizer") and pl_module.tokenizer is not None:
                pl_module.tokenizer.save_pretrained(save_path)

            print("[PushToHubCallback] Contenido a subir:", sorted(os.listdir(save_path)))

            # 5) Subir TODO al Hub bajo la subcarpeta del dataset
            commit_msg = f"Best model up to epoch {epoch} ({self.monitor}={self.best_score})"
            print(f"[PushToHubCallback] Subiendo a {self.repo_id} en '{self.dataset_subset}' ({commit_msg})…")
            self.api.upload_folder(
                folder_path=save_path,
                path_in_repo=self.dataset_subset,  # pon "" si quieres raíz del repo
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=commit_msg,
            )

            # 6) Card
            self._upload_card_files(self.repo_id)

        except Exception as e:
            # Fallback: al menos sube adapter + processor/tokenizer
            print(f"[PushToHubCallback] Fusión full-fp16 falló: {e}\nSubo fallback (adapter + processor/tokenizer).")
            self.api.upload_folder(
                folder_path=save_path,
                path_in_repo=self.dataset_subset,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Fallback adapter upload at epoch {epoch} ({self.monitor}={self.best_score})",
            )
            self._upload_card_files(self.repo_id)


    def _upload_card_files(self, repo_id: str):
        for file_path in HF_CARD_FILES:

            print(f"[PushToHubCallback] Subiendo card file: {file_path} -> {repo_id}")
            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo="/".join(file_path.split(os.sep)[4:]),
                repo_id=repo_id,
                repo_type="model",
                commit_message="Update card files",
            )


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


def train_collate_fn(examples):
    images = []
    texts = []
    for example in examples:
        image, ground_truth = example
        images.append(image)
        # TODO: in the future we can replace this by processor.apply_chat_template
        # prompt = f"USER: <image>\nExtract JSON.\nASSISTANT: {ground_truth}"
        prompt = f"USER: <image>\nExtract JSON.\nASSISTANT:"

        texts.append(prompt)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, labels


def eval_collate_fn(examples):
    # we only feed the prompt to the model
    images = []
    texts = []
    answers = []
    for example in examples:
        image, ground_truth = example
        images.append(image)
        # TODO: in the future we can replace this by processor.apply_chat_template
        prompt = f"USER: <image>\nExtract JSON.\nASSISTANT:"
        texts.append(prompt)
        answers.append(ground_truth)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]

    return input_ids, attention_mask, pixel_values, answers


def load_secret_dataset(repo_id: str):

    val_datasets = []
    
    dataset_subsets = [
        "britanico",
        "fomento",
        "maravillas",
        "mater",
        "montealto",
        "pilar",
        "recuerdo",
        "retamar",
        "sanpablo",
        "sanpatricio",
    ]

    for subset in dataset_subsets:
        val_ds = LlavaDataset(repo_id, subset, split="test", sort_json_key=False)
        val_datasets.append(val_ds)

    val_dataset = ConcatDataset(val_datasets)

    return val_dataset


def save_and_push_model(processor, model, repo_id, dataset_subset, save_dir, commit_message, base_id=None):

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "initial_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    def _load_base_cpu_fp16(base_id: str):
        cfg = AutoConfig.from_pretrained(base_id, trust_remote_code=True)
        # Intento específico para LLaVA si está disponible en tu instalación
        try:
            # transformers >= 4.38 suele exponerse así
            from transformers import LlavaForConditionalGeneration
            if getattr(cfg, "model_type", None) in {"llava", "llava_next", "llava-onevision"}:
                return LlavaForConditionalGeneration.from_pretrained(
                    base_id,
                    torch_dtype=torch.float16,
                    device_map=None,
                    low_cpu_mem_usage=False,
                    trust_remote_code=True,
                )
        except Exception:
            pass
        # Fallback más estable: AutoModelForCausalLM (muchos repos LLaVA lo registran)
        try:
            return AutoModelForCausalLM.from_pretrained(
                base_id,
                torch_dtype=torch.float16,
                device_map=None,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )
        except Exception:
            # Último intento: AutoModel (por si el repo solo registró esto)
            return AutoModel.from_pretrained(
                base_id,
                torch_dtype=torch.float16,
                device_map=None,
                trust_remote_code=True,
            )

    is_peft = isinstance(model, PeftModel)
    if is_peft:
        # Guarda adapter en la MISMA carpeta (esto NO crea config.json)
        model.save_pretrained(checkpoint_dir)  # adapter_config.json + adapter_model.safetensors

        # Inferir id del base si no viene
        if base_id is None:
            base = getattr(model, "get_base_model", lambda: None)() or getattr(model, "base_model", model)
            base = getattr(base, "model", base)
            base_id = getattr(base, "name_or_path", None) or getattr(getattr(base, "config", None), "_name_or_path", None)
            if base_id is None:
                raise RuntimeError("No pude inferir base_id; pásalo como parámetro.")

        # Cargar base limpio y fusionar adapter -> ahora SÍ habrá config.json al guardar
        base_cpu = _load_base_cpu_fp16(base_id)
        merged = PeftModel.from_pretrained(base_cpu, checkpoint_dir).merge_and_unload()
        merged.save_pretrained(checkpoint_dir, safe_serialization=True, max_shard_size="1GB")
        if getattr(merged, "generation_config", None) is not None:
            merged.generation_config.save_pretrained(checkpoint_dir)

    else:
        # No-PEFT: guardar modelo completo (esto escribe config.json)
        model.save_pretrained(checkpoint_dir, safe_serialization=True, max_shard_size="1GB")
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(checkpoint_dir)

    # Processor / tokenizer
    if processor is not None:
        processor.save_pretrained(checkpoint_dir)

    print("Contenido a subir:", sorted(os.listdir(checkpoint_dir)))  # deberías ver 'config.json' aquí

    # Subida al Hub
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    api = HfApi()
    api.upload_folder(
        folder_path=checkpoint_dir,
        path_in_repo=dataset_subset,   # "" si quieres raíz del repo
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    # Archivos extra (card)
    for file in HF_CARD_FILES:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo="/".join(file.split(os.sep)[4:]),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Uploading additional files",
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_subsets", type=str)
    parser.add_argument("--freeze_encoder", action="store_true", default=False)
    parser.add_argument("--save_initial", action="store_true", default=False)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_subsets = args.dataset_subsets
    freeze_encoder = args.freeze_encoder
    save_initial = args.save_initial

    model_output_name = "".join(["llava-vrdu-", dataset_name.split('/')[-1]])

    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        revision='a272c74',
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision='a272c74')
    processor.tokenizer.padding_side = "right"

    if save_initial:
        save_and_push_model(
            processor=processor,
            model=model,
            repo_id="de-Rodrigo/llava-merit",
            dataset_subset="vanilla",
            save_dir="./initial_checkpoint",
            commit_message="Initial model upload (vanilla, but LoRA configured)"
        )
        print("Initial model has been uploaded.")
        exit()

    train_dataset = LlavaDataset(dataset_name, dataset_subsets, split="train", sort_json_key=False)
    val_dataset = load_secret_dataset("de-Rodrigo/merit-secret")

    session_name = "_".join([dataset_subsets])
    session_name = f"llava_{session_name}"

    config = {"max_epochs": 1,
        "val_check_interval": 0.01,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 8,
        "lr": 1e-4,
        "batch_size": 2,
        # "seed":2022,
        "num_nodes": 1,
        "warmup_steps": 50,
        "result_path": "./result",
        "verbose": True,
    }

    model_module = LlavaModelPLModule(config, processor, model, freeze_encoder)

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=session_name, entity="ciclab-comillas")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[PushToHubCallback("llava-merit", session_name)],
        log_every_n_steps=1,
        val_check_interval=config["val_check_interval"],
    )

    trainer.fit(model_module)