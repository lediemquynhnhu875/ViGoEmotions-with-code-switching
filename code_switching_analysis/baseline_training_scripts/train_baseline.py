import argparse
import ast
import inspect
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


MODEL_ZOO = {
    "xlm-r": "FacebookAI/xlm-roberta-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "phobert": "vinai/phobert-base-v2",
    "visobert": "uitnlp/visobert",
    "vibert": "FPTAI/vibert-base-cased",
    "cafebert": "uitnlp/CafeBERT",
}

LABEL_NAMES = [
    "amusement",
    "excitement",
    "joy",
    "love",
    "desire",
    "optimism",
    "caring",
    "pride",
    "admiration",
    "gratitude",
    "relief",
    "approval",
    "realization",
    "surprise",
    "curiosity",
    "confusion",
    "fear",
    "nervousness",
    "remorse",
    "embarrassment",
    "disappointment",
    "sadness",
    "grief",
    "disgust",
    "anger",
    "annoyance",
    "disapproval",
    "neutral",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_split_name(split_name):
    split_name = str(split_name).lower()
    if split_name in ["validation", "dev"]:
        return "val"
    return split_name


def read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(path)
        if {"train", "val", "test"}.issubset(set(xls.sheet_names)):
            frames = []
            for sheet in ["train", "val", "test"]:
                df = pd.read_excel(path, sheet_name=sheet)
                df["split"] = sheet
                frames.append(df)
            return pd.concat(frames, ignore_index=True)
        return pd.read_excel(path, sheet_name=xls.sheet_names[0])
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in [".json", ".jsonl"]:
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    raise ValueError(f"Unsupported file type: {path}")


def find_data_files(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")

    exts = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}
    files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No supported data files found under: {data_dir}")
    return files


def standardize_columns(df):
    df = df.copy()
    lower_map = {c.lower(): c for c in df.columns}

    if "text" not in df.columns:
        for candidate in ["comment", "sentence", "content", "clean_text"]:
            if candidate in lower_map:
                df = df.rename(columns={lower_map[candidate]: "text"})
                break

    if "labels" not in df.columns:
        for candidate in ["label", "emotion", "emotions", "target", "targets"]:
            if candidate in lower_map:
                df = df.rename(columns={lower_map[candidate]: "labels"})
                break

    if "split" not in df.columns:
        for candidate in ["set", "data_split"]:
            if candidate in lower_map:
                df = df.rename(columns={lower_map[candidate]: "split"})
                break

    missing = [c for c in ["text", "labels"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available columns: {list(df.columns)}")

    return df


def parse_labels(x, num_labels):
    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            x = [int(i) for i in re.findall(r"\d+", x)]

    if isinstance(x, (list, tuple)):
        x = list(x)
        unique_values = set(np.unique(x)) if len(x) else set()
        if len(x) == num_labels and unique_values.issubset({0, 1, 0.0, 1.0}):
            return [float(v) for v in x]

        y = np.zeros(num_labels, dtype=np.float32)
        for idx in x:
            idx = int(idx)
            if 0 <= idx < num_labels:
                y[idx] = 1.0
        return y.tolist()

    y = np.zeros(num_labels, dtype=np.float32)
    try:
        y[int(x)] = 1.0
    except Exception:
        pass
    return y.tolist()


def load_vigo_from_kaggle_dir(data_dir, num_labels):
    files = find_data_files(data_dir)
    print("Found data files:")
    for file_path in files:
        print(f"- {file_path}")

    named_split_frames = []
    single_frames = []

    for file_path in files:
        df = standardize_columns(read_table(file_path))
        name = file_path.stem.lower()

        if "split" in df.columns:
            single_frames.append(df)
        elif "train" in name:
            df["split"] = "train"
            named_split_frames.append(df)
        elif any(k in name for k in ["val", "dev", "validation"]):
            df["split"] = "val"
            named_split_frames.append(df)
        elif "test" in name:
            df["split"] = "test"
            named_split_frames.append(df)

    if single_frames:
        data = pd.concat(single_frames, ignore_index=True)
    elif named_split_frames:
        data = pd.concat(named_split_frames, ignore_index=True)
    else:
        raise ValueError(
            "Cannot infer splits. Use files named train/val/test, "
            "or provide a single file with column split/set."
        )

    data = standardize_columns(data)
    data["split"] = data["split"].map(normalize_split_name)
    data["labels"] = data["labels"].apply(lambda x: parse_labels(x, num_labels))
    data = data[["text", "labels", "split"]].copy()

    required_splits = {"train", "val", "test"}
    found_splits = set(data["split"].dropna().unique())
    missing_splits = required_splits - found_splits
    if missing_splits:
        raise ValueError(f"Missing splits: {missing_splits}. Found splits: {found_splits}")

    print("Data shape:", data.shape)
    print(data["split"].value_counts())
    return data


class WeightedMultiLabelTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_compute_metrics(threshold):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = sigmoid(logits)
        preds = (probs >= threshold).astype(int)
        labels = labels.astype(int)

        return {
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
            "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
            "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
            "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
            "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
            "hamming_loss": hamming_loss(labels, preds),
            "subset_accuracy": accuracy_score(labels, preds),
        }

    return compute_metrics


def build_training_args(args):
    kwargs = {
        "output_dir": str(args.output_dir),
        "run_name": args.run_name,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": torch.cuda.is_available() and not args.no_fp16,
    }

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def save_predictions(trainer, tokenized_test, test_df, output_dir, threshold):
    pred_out = trainer.predict(tokenized_test)
    test_logits = pred_out.predictions
    test_probs = sigmoid(test_logits)
    test_preds = (test_probs >= threshold).astype(int)
    test_labels = np.array(test_df["labels"].tolist(), dtype=int)

    pred_df = test_df[["text"]].copy()
    for i, label in enumerate(LABEL_NAMES):
        pred_df[f"true_{label}"] = test_labels[:, i]
        pred_df[f"prob_{label}"] = test_probs[:, i]
        pred_df[f"pred_{label}"] = test_preds[:, i]

    pred_path = Path(output_dir) / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print("Saved predictions to:", pred_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Kaggle input data folder, e.g. /kaggle/input/vigoemotions")
    parser.add_argument("--model_key", default="xlm-r", choices=sorted(MODEL_ZOO.keys()))
    parser.add_argument("--model_name", default=None, help="Override Hugging Face model name/path")
    parser.add_argument("--output_root", default="/kaggle/working/vigo_baseline_outputs")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=float, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    num_labels = len(LABEL_NAMES)
    id2label = {i: label for i, label in enumerate(LABEL_NAMES)}
    label2id = {label: i for i, label in enumerate(LABEL_NAMES)}

    model_name = args.model_name or MODEL_ZOO[args.model_key]
    args.run_name = f"{args.model_key}-vigoemotions"
    args.output_dir = Path(args.output_root) / args.run_name
    best_model_dir = args.output_dir / "best_model"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Model:", model_name)
    print("Output:", args.output_dir)

    data = load_vigo_from_kaggle_dir(args.data_dir, num_labels)
    train_df = data[data["split"] == "train"].reset_index(drop=True)
    val_df = data[data["split"] == "val"].reset_index(drop=True)
    test_df = data[data["split"] == "test"].reset_index(drop=True)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["text", "labels"]]),
            "validation": Dataset.from_pandas(val_df[["text", "labels"]]),
            "test": Dataset.from_pandas(test_df[["text", "labels"]]),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized_ds = dataset.map(tokenize_batch, batched=True)
    tokenized_ds = tokenized_ds.remove_columns(["text"])
    tokenized_ds.set_format("torch")

    train_label_matrix = np.array(train_df["labels"].tolist(), dtype=np.float32)
    positive_counts = train_label_matrix.sum(axis=0)
    negative_counts = len(train_label_matrix) - positive_counts
    pos_weight = negative_counts / np.clip(positive_counts, 1, None)
    pos_weight = np.clip(pos_weight, 1.0, 50.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float)

    pd.DataFrame(
        {
            "label": LABEL_NAMES,
            "positive_count": positive_counts.astype(int),
            "pos_weight": pos_weight,
        }
    ).to_csv(args.output_dir / "label_pos_weight.csv", index=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    trainer = WeightedMultiLabelTrainer(
        model=model,
        args=build_training_args(args),
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=make_compute_metrics(args.threshold),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        pos_weight=pos_weight_tensor,
    )

    trainer.train()

    val_metrics = trainer.evaluate(tokenized_ds["validation"])
    test_metrics = trainer.evaluate(tokenized_ds["test"], metric_key_prefix="test")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    with open(best_model_dir / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_NAMES, f, ensure_ascii=False, indent=2)

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, ensure_ascii=False, indent=2)

    save_predictions(trainer, tokenized_ds["test"], test_df, args.output_dir, args.threshold)

    print("Saved best model to:", best_model_dir)


if __name__ == "__main__":
    main()
