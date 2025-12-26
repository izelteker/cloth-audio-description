import os
import functools
import traceback
import gc

import numpy as np
import torch
import mlflow
import evaluate
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipConfig,
    AutoTokenizer,
    DataCollator,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image


## Code for Nvidia
if torch.cuda.is_available():
    DEVICE = "cuda"

## Code for Apple
elif torch.mps.is_available():
    DEVICE = "mps"

else:
    DEVICE = "cpu"


## TODO: Seperate data dir for colab & others
CKPT = "Salesforce/blip-image-captioning-base"
CFG_PATH = "conf/config.yaml"
DATA_DIR = "cloth-ds"


def load_data(
    dataset_name: str = DATASET_NAME,
    data_root: Path = DATA_ROOT,
    conf_path: Path = CFG_PATH,
    save_to_dir = True,
    **load_args
    ):
    """
    Loads dataset from disk if available, otherwise loads from HF and saves it optionally.

    Dataset path = data_root / dataset_name
    """
    # Load config
    cfg = OmegaConf.load(conf_path)

    if data_root.exists():
        print(f"Loading dataset from disk: {data_root}")
        ds = load_from_disk(data_root)
    else:
        print(f"Dataset not found at {data_root}. Loading dataset from HF...")

        ds = load_dataset(dataset_name, **load_args)

        if save_to_dir:
            dataset_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(data_root)
            print(f"Dataset saved to: {data_root}")

    return ds, cfg


class BlipDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, batch):
        images = [
            Image.fromarray(x["image"]) if not isinstance(x["image"], Image.Image) else x["image"]
            for x in batch
        ]
        texts = [x["text"] for x in batch]

        encoding = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=48,
            return_tensors="pt"
        )

        labels = encoding.input_ids.clone()
        labels[labels == self.pad_token_id] = -100  # ignore padding in loss

        return {
            "pixel_values": encoding.pixel_values,
            "input_ids": encoding.input_ids,     
            "attention_mask": encoding.attention_mask,
            "labels": labels,
        }


def prepare_data(dataset: Dataset):
    # original dataset (88400 samples)
    dataset = ds.rename_column("description", "text").select_columns(["image", "text"])

    # first, split off test set (e.g., 10%)
    split1 = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_val_dataset = split1['train']  # 90%
    test_dataset = split1['test']        # 10%

    # then, split train_val into train & validation (e.g., 80/20 of 90%)
    split2 = train_val_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split2['train']       # 72%
    val_dataset = split2['test']          # 18%

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Replace -100 so we can decode labels
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

    decoded_preds = processor.tokenizer.batch_decode(
        preds, skip_special_tokens=True
    )
    decoded_labels = processor.tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )

    return {
        "bleu": bleu.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )["bleu"],
        "meteor": meteor.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )["meteor"],
        "rougeL": rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )["rougeL"],
    }


def run_mlflow(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mlflow.set_experiment("cloth-finetune-first-experiment")

        with mlflow.start_run():
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log exception to MLflow
                mlflow.log_param("run_status", "failed")
                mlflow.log_text(traceback.format_exc(), "error_traceback.txt")
                raise e
    return wrapper


@run_mlflow
def main():
    processor = BlipProcessor.from_pretrained(CKPT)
    processor.image_processor.size = {"height": 192, "width": 192}

    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    training_args = cfg["training_args"]

    print("\nTraining model and logging with MLflow...")

    model = BlipForConditionalGeneration.from_pretrained(CKPT)

    mlflow.transformers.log_model(
        model,
        "BlipForConditionalGeneration",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=cloth_dataset["train"],
        eval_dataset=cloth_dataset["validation"],
        data_collator=BlipDataCollator(processor),
        compute_metrics=compute_metrics,
    )

    mlflow.log_params({
        "learning_rate": training_args.learning_rate,
        "train_batch_size": training_args.per_device_train_batch_size,
        "eval_batch_size": training_args.per_device_eval_batch_size,
        "num_train_epochs": training_args.num_train_epochs,
        "eval_steps": training_args.eval_steps,
        "save_total_limit": training_args.save_total_limit,
    })

    train_results = trainer.train()

    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    for k in ["bleu", "meteor", "rougeL"]:
        if k in train_results.metrics:
            mlflow.log_metric(k, train_results.metrics[k])

        

if __name__ == "__main__":
    main()