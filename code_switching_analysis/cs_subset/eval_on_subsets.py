#!/usr/bin/env python
"""
eval_on_subsets.py
==================
Re-evaluate một checkpoint đã train (baseline hoặc XLM-R + xLSTM) trên các subset
code-switching do `extract_cs_subset.py` sinh ra.

Khác với notebook eval hiện tại trong repo:
  * Dùng chung `cs_detector` qua file annotations -> không còn 2 định nghĩa CS lệch nhau.
  * Bootstrap 95% CI cho macro/micro F1 -> tránh kết luận sai trên subset nhỏ.
  * Paired permutation test giữa `cs_strict` và `control_pure_vi` (nhóm đối chứng
    đã cân bằng kích thước + phân bố nhãn) -> trả lời được "model có thực sự kém
    hơn trên code-switching không, hay chỉ do nhiễu / subset khác phân bố nhãn?".
  * Tuỳ chọn tune threshold trên val rồi áp dụng cho test (fair hơn 0.5 cứng).

Ví dụ
-----
python eval_on_subsets.py \
    --annotations ./cs_subsets/annotations/vigo_cs_annotations.csv \
    --model /kaggle/input/xlm-r-vigoemotions/best_model \
    --model-tag xlm-r-baseline \
    --tune-threshold \
    --out-dir ./subset_eval
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             precision_score, recall_score)

LABEL_NAMES = [
    "amusement", "excitement", "joy", "love", "desire", "optimism", "caring",
    "pride", "admiration", "gratitude", "relief", "approval", "realization",
    "surprise", "curiosity", "confusion", "fear", "nervousness", "remorse",
    "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger",
    "annoyance", "disapproval", "neutral",
]
NUM_LABELS = len(LABEL_NAMES)

SUBSET_DEFS = {
    "all":             lambda d: np.ones(len(d), dtype=bool),
    "pure_vi":         lambda d: d["cs_group"].eq("pure_vi").to_numpy(),
    "control_pure_vi": lambda d: d.get("control_pure_vi", pd.Series(False, index=d.index)).astype(bool).to_numpy(),
    "cs_strict":       lambda d: d["is_cs_strict"].astype(bool).to_numpy(),
    "cs_broad":        lambda d: d["is_cs_broad"].astype(bool).to_numpy(),
    "english_mixed":   lambda d: d["cs_group"].eq("english_mixed").to_numpy(),
    "teencode_slang":  lambda d: d["cs_group"].eq("teencode_slang").to_numpy(),
    "emoji_only":      lambda d: d["cs_group"].eq("emoji_only").to_numpy(),
    "other_noise":     lambda d: d["cs_group"].eq("other_noise").to_numpy(),
    "cs_heavy":        lambda d: d["cs_level"].eq("heavy").to_numpy(),
    "cs_light":        lambda d: d["cs_level"].eq("light").to_numpy(),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_annotations(path: str, split: str, text_col: str = "text"):
    df = pd.read_csv(path)
    df = df[df["split"] == split].reset_index(drop=True)
    if text_col != "text":
        if text_col not in df.columns:
            raise KeyError(f"Thiếu cột '{text_col}'. Có: "
                           f"{[c for c in df.columns if c.startswith('text')]}")
        df["text"] = df[text_col]
    label_cols = [f"label_{n}" for n in LABEL_NAMES]
    if all(c in df.columns for c in label_cols):
        y = df[label_cols].to_numpy(dtype=np.int8)
    else:
        y = np.zeros((len(df), NUM_LABELS), dtype=np.int8)
        for i, v in enumerate(df["label_ids"]):
            for j in ast.literal_eval(str(v)):
                y[i, int(j)] = 1
    return df, y


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

from checkpoint_loader import (load_model_and_tokenizer, resolve_checkpoint,
                               inspect_checkpoint, detect_structure,
                               sanity_check_predictions)


def predict_proba(texts, model_path: str, max_length: int, batch_size: int,
                  use_fast: bool = True, base_model: str = None,
                  tokenizer_path: str = None) -> np.ndarray:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model_and_tokenizer(model_path, base_model, tokenizer_path, use_fast)
    model = model.to(device).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            out.append(torch.sigmoid(logits).cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"    {i}/{len(texts)}", end="\r")
    return np.vstack(out)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n_samples": int(len(y_true)),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "avg_pred_per_sample": float(y_pred.sum(axis=1).mean()),
    }


def bootstrap_ci(y_true, y_pred, metric="macro_f1", n_boot=1000, seed=42, alpha=0.05):
    """95% CI bằng bootstrap trên chỉ số mẫu (percentile method)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 10:
        return (np.nan, np.nan)
    avg = "macro" if metric.startswith("macro") else "micro"
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = f1_score(y_true[idx], y_pred[idx], average=avg, zero_division=0)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def permutation_test(y_true_a, y_pred_a, y_true_b, y_pred_b, metric="macro_f1",
                     n_perm=2000, seed=42):
    """Two-sample permutation test cho hiệu F1 giữa 2 subset rời nhau.

    H0: hai subset đến từ cùng một phân phối hiệu năng.
    """
    rng = np.random.default_rng(seed)
    avg = "macro" if metric.startswith("macro") else "micro"

    def score(t, p):
        return f1_score(t, p, average=avg, zero_division=0)

    obs = score(y_true_a, y_pred_a) - score(y_true_b, y_pred_b)
    na = len(y_true_a)
    all_t = np.vstack([y_true_a, y_true_b])
    all_p = np.vstack([y_pred_a, y_pred_b])
    n = len(all_t)

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        ia, ib = perm[:na], perm[na:]
        diff = score(all_t[ia], all_p[ia]) - score(all_t[ib], all_p[ib])
        if abs(diff) >= abs(obs):
            count += 1
    return float(obs), float((count + 1) / (n_perm + 1))


def tune_threshold(y_true, y_prob, grid=None, metric="macro_f1"):
    grid = grid if grid is not None else np.arange(0.10, 0.71, 0.02)
    avg = "macro" if metric.startswith("macro") else "micro"
    best_t, best_v = 0.5, -1.0
    for t in grid:
        v = f1_score(y_true, (y_prob >= t).astype(int), average=avg, zero_division=0)
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True,
                    help="cs_subsets/annotations/vigo_cs_annotations.csv")
    ap.add_argument("--model", required=True,
                    help="Thư mục save_pretrained, thư mục Kaggle Model, hoặc file .pt state_dict")
    ap.add_argument("--base-model", default=None,
                    help="Tên/đường dẫn model gốc trên HF. BẮT BUỘC nếu --model là state_dict rời, "
                         "vd: vinai/phobert-base, xlm-roberta-base, FPTAI/vibert-base-cased")
    ap.add_argument("--tokenizer", default=None,
                    help="Nguồn tokenizer nếu checkpoint không kèm tokenizer")
    ap.add_argument("--model-tag", default="model")
    ap.add_argument("--split", default="test")
    ap.add_argument("--text-col", default="text",
                    help="Cột văn bản dùng làm đầu vào, vd text_s1 / text_s2 / text_s3")
    ap.add_argument("--out-dir", default="./subset_eval")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tune-threshold", action="store_true",
                    help="Tìm threshold tốt nhất trên split val rồi áp dụng cho test")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--slow-tokenizer", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probs-npy", default=None,
                    help="Dùng lại file xác suất đã lưu thay vì chạy inference")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. threshold
    threshold = args.threshold
    if args.tune_threshold:
        print("[*] Tune threshold trên val ...")
        val_df, val_y = load_annotations(args.annotations, "val", args.text_col)
        val_prob = predict_proba(val_df["text"].astype(str).tolist(), args.model,
                                 args.max_length, args.batch_size, not args.slow_tokenizer,
                                 args.base_model, args.tokenizer)
        threshold, best_val = tune_threshold(val_y, val_prob)
        print(f"    threshold = {threshold:.2f} (macro F1 val = {best_val:.4f})")

    # 2. inference trên split đích
    print(f"[*] Inference trên split={args.split} ...")
    df, y_true = load_annotations(args.annotations, args.split, args.text_col)
    if args.probs_npy and Path(args.probs_npy).exists():
        y_prob = np.load(args.probs_npy)
    else:
        y_prob = predict_proba(df["text"].astype(str).tolist(), args.model,
                               args.max_length, args.batch_size, not args.slow_tokenizer,
                               args.base_model, args.tokenizer)
        np.save(out_dir / f"probs_{args.split}.npy", y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    sanity_check_predictions(y_prob, threshold=threshold)

    # 3. metric theo subset
    print("[*] Tính metric theo subset ...")
    rows = []
    masks = {}
    for name, fn in SUBSET_DEFS.items():
        try:
            m = fn(df)
        except KeyError:
            continue
        if m.sum() < 5:
            continue
        masks[name] = m
        r = metrics(y_true[m], y_pred[m])
        lo_ma, hi_ma = bootstrap_ci(y_true[m], y_pred[m], "macro_f1", args.n_boot, args.seed)
        lo_mi, hi_mi = bootstrap_ci(y_true[m], y_pred[m], "micro_f1", args.n_boot, args.seed)
        r.update({"subset": name, "ratio": float(m.mean()), "threshold": threshold,
                  "model": args.model_tag,
                  "macro_f1_lo": lo_ma, "macro_f1_hi": hi_ma,
                  "micro_f1_lo": lo_mi, "micro_f1_hi": hi_mi})
        rows.append(r)

    cols = ["model", "subset", "n_samples", "ratio", "threshold",
            "micro_f1", "micro_f1_lo", "micro_f1_hi",
            "macro_f1", "macro_f1_lo", "macro_f1_hi", "weighted_f1",
            "micro_precision", "macro_precision", "micro_recall", "macro_recall",
            "hamming_loss", "subset_accuracy", "avg_pred_per_sample"]
    subset_df = pd.DataFrame(rows)[cols]
    subset_df.to_csv(out_dir / "subset_metrics.csv", index=False)

    # 4. per-label
    per_label = []
    for name, m in masks.items():
        yt, yp = y_true[m], y_pred[m]
        f1s = f1_score(yt, yp, average=None, zero_division=0)
        ps = precision_score(yt, yp, average=None, zero_division=0)
        rs = recall_score(yt, yp, average=None, zero_division=0)
        for i, lab in enumerate(LABEL_NAMES):
            per_label.append({"model": args.model_tag, "subset": name, "label": lab,
                              "support": int(yt[:, i].sum()), "precision": ps[i],
                              "recall": rs[i], "f1": f1s[i]})
    pd.DataFrame(per_label).to_csv(out_dir / "per_label_metrics.csv", index=False)

    # 5. significance
    tests = []
    for a, b in [("cs_strict", "control_pure_vi"), ("cs_strict", "pure_vi"),
                 ("cs_broad", "pure_vi"), ("teencode_slang", "pure_vi"),
                 ("cs_heavy", "cs_light")]:
        if a not in masks or b not in masks:
            continue
        ma, mb = masks[a], masks[b]
        if (ma & mb).any():          # phải rời nhau
            continue
        for metric in ["macro_f1", "micro_f1"]:
            obs, p = permutation_test(y_true[ma], y_pred[ma], y_true[mb], y_pred[mb],
                                      metric, args.n_perm, args.seed)
            tests.append({"model": args.model_tag, "subset_a": a, "subset_b": b,
                          "metric": metric, "n_a": int(ma.sum()), "n_b": int(mb.sum()),
                          "diff_a_minus_b": obs, "p_value": p,
                          "significant_005": p < 0.05})
    sig_df = pd.DataFrame(tests)
    sig_df.to_csv(out_dir / "significance_tests.csv", index=False)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "threshold_used": threshold}, f, ensure_ascii=False, indent=2)

    pd.set_option("display.width", 220)
    print("\n=== SUBSET METRICS ===")
    print(subset_df[["subset", "n_samples", "micro_f1", "macro_f1",
                     "macro_f1_lo", "macro_f1_hi"]].to_string(index=False))
    if len(sig_df):
        print("\n=== SIGNIFICANCE ===")
        print(sig_df.to_string(index=False))
    print(f"\nXong. Kết quả ở: {out_dir.resolve()}")


if __name__ == "__main__":
    main()