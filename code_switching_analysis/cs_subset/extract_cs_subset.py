#!/usr/bin/env python
"""
extract_cs_subset.py
====================
Trích xuất subset code-switching từ ViGoEmotions (hoặc bất kỳ dataset nào có
cột `text` / `labels` / `split`) để phục vụ:

  * Giai đoạn 1: re-evaluate baseline + train XLM-R + xLSTM trên subset CS.
  * Giai đoạn 2: chọn dữ liệu code-mixed (ViCM) để pretrain/adapt.

Script export ĐỒNG THỜI nhiều tier để không phải chạy lại khi đổi định nghĩa:

  all             toàn bộ split
  pure_vi         không English, không teencode, không emoji, không noise
  cs_strict       có >= min_english_count token tiếng Anh  (định nghĩa hẹp, Vi-En thật)
  cs_broad        có tiếng Anh HOẶC teencode               (định nghĩa rộng)
  english_mixed   cs_group == english_mixed  (== cs_strict)
  teencode_slang  chỉ teencode, không tiếng Anh
  emoji_only      chỉ emoji
  other_noise     url/mention/hashtag/laughter
  cs_heavy        english_ratio >= heavy_ratio hoặc >= 3 token tiếng Anh
  cs_light        có tiếng Anh nhưng chưa tới heavy
  control_pure_vi mẫu pure_vi cân bằng theo nhãn & kích thước với cs_strict
                  (dùng làm nhóm đối chứng công bằng khi so sánh macro/micro F1)

Ví dụ
-----
# Đọc trực tiếp từ Hugging Face (cần huggingface-cli login vì dataset gated)
python extract_cs_subset.py --source hf --hf-name uitnlp/vigoemotions --out-dir ./cs_subsets

# Đọc từ file local
python extract_cs_subset.py --source file --data-path ./corpus/dataset_V1.xlsx --out-dir ./cs_subsets

# Đọc lại file annotation đã có trong repo (nhanh, không cần tải dataset)
python extract_cs_subset.py --source annotations \
    --data-path code_switching_analysis/outputs_code_switching/vigo_code_switching_annotations.csv \
    --reannotate --out-dir ./cs_subsets

# Trên Kaggle
python extract_cs_subset.py --source file --data-path /kaggle/input/vigoemotions \
    --out-dir /kaggle/working/cs_subsets --save-hf
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_detector import CodeSwitchDetector, annotate_dataframe  # noqa: E402

LABEL_NAMES = [
    "amusement", "excitement", "joy", "love", "desire", "optimism", "caring",
    "pride", "admiration", "gratitude", "relief", "approval", "realization",
    "surprise", "curiosity", "confusion", "fear", "nervousness", "remorse",
    "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger",
    "annoyance", "disapproval", "neutral",
]
NUM_LABELS = len(LABEL_NAMES)

TABLE_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl", ".pkl", ".pickle"}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _normalize_split(name) -> str:
    n = str(name).strip().lower()
    return {"validation": "val", "valid": "val", "dev": "val", "eval": "val"}.get(n, n)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower = {c.lower(): c for c in df.columns}

    if "text" not in df.columns:
        for cand in ["comment", "sentence", "content", "clean_text", "raw_text", "cmt"]:
            if cand in lower:
                df = df.rename(columns={lower[cand]: "text"})
                break
    if "labels" not in df.columns:
        for cand in ["label", "emotion", "emotions", "target", "targets", "y"]:
            if cand in lower:
                df = df.rename(columns={lower[cand]: "labels"})
                break
    if "split" not in df.columns:
        for cand in ["set", "data_split", "partition"]:
            if cand in lower:
                df = df.rename(columns={lower[cand]: "split"})
                break

    if "text" not in df.columns:
        raise ValueError(f"Không tìm thấy cột text. Cột hiện có: {list(df.columns)}")
    return df


def _read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(path)
        sheets = {s.lower(): s for s in xls.sheet_names}
        if any(k in sheets for k in ("train", "val", "dev", "test")):
            frames = []
            for key, split in [("train", "train"), ("val", "val"), ("dev", "val"),
                               ("validation", "val"), ("test", "test")]:
                if key in sheets:
                    part = pd.read_excel(path, sheet_name=sheets[key])
                    part["split"] = split
                    frames.append(part)
            return pd.concat(frames, ignore_index=True)
        return pd.read_excel(path, sheet_name=xls.sheet_names[0])
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in {".json", ".jsonl"}:
        return pd.read_json(path, lines=(ext == ".jsonl"))
    if ext in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, tuple) and len(obj) == 3:
            frames = []
            for part, split in zip(obj, ["train", "val", "test"]):
                part = part.copy()
                part["split"] = split
                frames.append(part)
            return pd.concat(frames, ignore_index=True)
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        raise ValueError(f"Định dạng pickle không hỗ trợ: {type(obj)}")
    raise ValueError(f"Không hỗ trợ đuôi file: {ext}")


def load_from_file(data_path: str) -> pd.DataFrame:
    p = Path(data_path)
    if p.is_file():
        return _standardize_columns(_read_table(p))

    if not p.exists():
        raise FileNotFoundError(f"Không tồn tại: {p}")

    files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in TABLE_EXTS]
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu trong {p}")

    with_split, named = [], []
    for f in files:
        try:
            df = _standardize_columns(_read_table(f))
        except Exception as exc:                      # bỏ qua file phụ
            print(f"  [skip] {f.name}: {exc}")
            continue
        name = f.stem.lower()
        if "split" in df.columns:
            with_split.append(df)
        elif "train" in name:
            df["split"] = "train"; named.append(df)
        elif any(k in name for k in ("val", "dev")):
            df["split"] = "val"; named.append(df)
        elif "test" in name:
            df["split"] = "test"; named.append(df)

    frames = with_split or named
    if not frames:
        raise ValueError("Không suy ra được split. Đặt tên file train/val/test hoặc thêm cột split.")
    return _standardize_columns(pd.concat(frames, ignore_index=True))


def load_from_hf(name: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Cần `pip install datasets`.") from exc
    ds = load_dataset(name)
    frames = []
    for split_name, split_data in ds.items():
        part = split_data.to_pandas()
        part["split"] = _normalize_split(split_name)
        frames.append(part)
    return _standardize_columns(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def parse_label_list(value, num_labels: int = NUM_LABELS):
    """Chuẩn hoá về list index nhãn (multi-label)."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            value = [int(x) for x in re.findall(r"\d+", value)]
    if isinstance(value, (list, tuple)):
        value = list(value)
        # one-hot -> index
        if len(value) == num_labels and set(np.unique(value)).issubset({0, 1, 0.0, 1.0}):
            return [i for i, v in enumerate(value) if int(v) == 1]
        return [int(v) for v in value if str(v).strip().lstrip("-").isdigit()
                and 0 <= int(v) < num_labels]
    try:
        return [int(value)]
    except Exception:
        return []


def to_multihot(label_lists, num_labels: int = NUM_LABELS) -> np.ndarray:
    y = np.zeros((len(label_lists), num_labels), dtype=np.int8)
    for i, idxs in enumerate(label_lists):
        for j in idxs:
            y[i, j] = 1
    return y


# ---------------------------------------------------------------------------
# Subsets
# ---------------------------------------------------------------------------

def build_masks(df: pd.DataFrame) -> dict:
    return {
        "all": pd.Series(True, index=df.index),
        "pure_vi": df["cs_group"].eq("pure_vi"),
        "cs_strict": df["is_cs_strict"].fillna(False).astype(bool),
        "cs_broad": df["is_cs_broad"].fillna(False).astype(bool),
        "english_mixed": df["cs_group"].eq("english_mixed"),
        "teencode_slang": df["cs_group"].eq("teencode_slang"),
        "emoji_only": df["cs_group"].eq("emoji_only"),
        "other_noise": df["cs_group"].eq("other_noise"),
        "cs_heavy": df["cs_level"].eq("heavy"),
        "cs_light": df["cs_level"].eq("light"),
    }


def make_matched_control(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Chọn mẫu pure_vi có cùng kích thước & phân bố nhãn với cs_strict, theo từng split.

    Dùng làm nhóm đối chứng: nếu model tệ hơn trên cs_strict thì không phải do
    subset nhỏ hơn hay phân bố nhãn khác.
    """
    rng = np.random.default_rng(seed)
    keep = pd.Series(False, index=df.index)

    for split, part in df.groupby("split"):
        cs = part[part["is_cs_strict"]]
        pure = part[part["cs_group"] == "pure_vi"]
        if len(cs) == 0 or len(pure) == 0:
            continue

        # stratify theo nhãn "chính" (nhãn đầu tiên) để giữ phân bố tương tự
        cs_dist = cs["primary_label"].value_counts(normalize=True)
        n_target = min(len(cs), len(pure))

        chosen = []
        for label, frac in cs_dist.items():
            pool = pure.index[pure["primary_label"] == label].to_numpy()
            n_take = int(round(frac * n_target))
            if n_take == 0 or len(pool) == 0:
                continue
            n_take = min(n_take, len(pool))
            chosen.extend(rng.choice(pool, size=n_take, replace=False).tolist())

        # bù cho đủ n_target nếu thiếu do làm tròn / thiếu pool
        if len(chosen) < n_target:
            rest = np.setdiff1d(pure.index.to_numpy(), np.array(chosen, dtype=pure.index.dtype))
            if len(rest):
                extra = rng.choice(rest, size=min(n_target - len(chosen), len(rest)), replace=False)
                chosen.extend(extra.tolist())

        keep.loc[chosen] = True

    return keep


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame, out_dir: Path) -> None:
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("split")
        .agg(
            n=("text", "size"),
            cs_strict=("is_cs_strict", "sum"),
            cs_broad=("is_cs_broad", "sum"),
            has_english=("has_english", "sum"),
            has_teencode=("has_teencode", "sum"),
            has_emoji=("has_emoji", "sum"),
            n_laugh=("n_laugh", lambda s: int((s > 0).sum())),
            avg_english_ratio=("english_ratio", "mean"),
            avg_teencode_ratio=("teencode_ratio", "mean"),
            avg_tokens=("n_word_tokens", "mean"),
        )
        .reset_index()
    )
    for col in ["cs_strict", "cs_broad", "has_english", "has_teencode", "has_emoji"]:
        summary[f"{col}_pct"] = 100 * summary[col] / summary["n"]
    summary.to_csv(stats_dir / "summary_by_split.csv", index=False)

    pd.crosstab(df["split"], df["cs_group"], margins=True).to_csv(
        stats_dir / "cs_group_by_split.csv"
    )
    pd.crosstab(df["split"], df["cs_level"], margins=True).to_csv(
        stats_dir / "cs_level_by_split.csv"
    )

    # phân bố nhãn theo tier
    rows = []
    for i, name in enumerate(LABEL_NAMES):
        col = f"label_{name}"
        if col not in df.columns:
            continue
        for tier, mask in [("all", pd.Series(True, index=df.index)),
                           ("cs_strict", df["is_cs_strict"]),
                           ("cs_broad", df["is_cs_broad"]),
                           ("pure_vi", df["cs_group"].eq("pure_vi"))]:
            sub = df[mask]
            rows.append({
                "label": name, "label_id": i, "tier": tier,
                "support": int(sub[col].sum()),
                "ratio_in_tier": float(sub[col].mean()) if len(sub) else 0.0,
                "n_tier": int(len(sub)),
            })
    if rows:
        label_df = pd.DataFrame(rows)
        label_df.to_csv(stats_dir / "label_distribution_by_tier.csv", index=False)
        pivot = label_df.pivot(index="label", columns="tier", values="ratio_in_tier")
        if {"cs_strict", "pure_vi"}.issubset(pivot.columns):
            pivot["cs_minus_pure"] = pivot["cs_strict"] - pivot["pure_vi"]
        pivot.sort_values(pivot.columns[-1], ascending=False).to_csv(
            stats_dir / "label_shift_cs_vs_pure.csv"
        )

    # top token
    def _top(col: str, n: int = 200) -> pd.DataFrame:
        from collections import Counter
        c = Counter()
        for s in df[col].dropna():
            c.update(str(s).split())
        return pd.DataFrame(c.most_common(n), columns=["token", "count"])

    _top("english_tokens").to_csv(stats_dir / "top_english_tokens.csv", index=False)
    _top("teencode_tokens").to_csv(stats_dir / "top_teencode_tokens.csv", index=False)
    _top("english_guess_tokens").to_csv(stats_dir / "top_english_guess_tokens.csv", index=False)
    _top("foreign_other_tokens").to_csv(stats_dir / "top_foreign_other_tokens.csv", index=False)


def make_manual_check_sample(df: pd.DataFrame, out_dir: Path, n_per_group: int = 40,
                             seed: int = 42) -> None:
    """Mẫu phân tầng để bạn chấm tay precision/recall của detector.

    Rất nên làm trước khi báo cáo: rule-based detector không phải gold annotation,
    reviewer sẽ hỏi độ chính xác của nó.
    """
    cols = ["split", "text", "cs_group", "cs_level", "english_tokens",
            "english_guess_tokens", "teencode_tokens", "foreign_other_tokens",
            "n_english", "english_ratio"]
    cols = [c for c in cols if c in df.columns]
    parts = []
    for group, part in df.groupby("cs_group"):
        parts.append(part.sample(min(n_per_group, len(part)), random_state=seed)[cols])
    sample = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    sample["gold_is_code_switching"] = ""     # người chấm điền 1/0
    sample["note"] = ""
    sample.to_csv(out_dir / "manual_check_sample.csv", index=False)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_subsets(df: pd.DataFrame, out_dir: Path, formats, save_hf: bool,
                   keep_cols) -> pd.DataFrame:
    subsets_dir = out_dir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)

    masks = build_masks(df)
    if "control_pure_vi" in df.columns:
        masks["control_pure_vi"] = df["control_pure_vi"].astype(bool)

    index_rows = []
    for name, mask in masks.items():
        sub_all = df[mask]
        for split in ["train", "val", "test"]:
            part = sub_all[sub_all["split"] == split]
            if len(part) == 0:
                continue
            d = subsets_dir / name
            d.mkdir(parents=True, exist_ok=True)
            part_out = part[[c for c in keep_cols if c in part.columns]]
            if "csv" in formats:
                part_out.to_csv(d / f"{split}.csv", index=False)
            if "jsonl" in formats:
                part_out.to_json(d / f"{split}.jsonl", orient="records",
                                 lines=True, force_ascii=False)
            index_rows.append({
                "subset": name, "split": split, "n": len(part),
                "pct_of_split": 100 * len(part) / max((df["split"] == split).sum(), 1),
                "avg_tokens": float(part["n_word_tokens"].mean()),
                "avg_english_ratio": float(part["english_ratio"].mean()),
                "avg_labels_per_sample": float(part["n_labels"].mean())
                if "n_labels" in part.columns else np.nan,
            })

        if save_hf:
            _save_hf(sub_all, out_dir / "hf_datasets" / name, keep_cols)

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(out_dir / "subset_index.csv", index=False)
    return index_df


def _save_hf(sub: pd.DataFrame, path: Path, keep_cols) -> None:
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        print("  [warn] chưa cài `datasets`, bỏ qua --save-hf")
        return
    dd = {}
    for split in ["train", "val", "test"]:
        part = sub[sub["split"] == split]
        if len(part):
            dd[split] = Dataset.from_pandas(
                part[[c for c in keep_cols if c in part.columns]].reset_index(drop=True),
                preserve_index=False,
            )
    if dd:
        path.mkdir(parents=True, exist_ok=True)
        DatasetDict(dd).save_to_disk(str(path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "file", "annotations"], default="hf")
    ap.add_argument("--hf-name", default="uitnlp/vigoemotions")
    ap.add_argument("--data-path", default=None,
                    help="File hoặc thư mục dữ liệu (dùng với --source file/annotations)")
    ap.add_argument("--out-dir", default="./cs_subsets")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="labels")
    ap.add_argument("--split-col", default="split")

    ap.add_argument("--min-english-count", type=int, default=1)
    ap.add_argument("--min-english-ratio", type=float, default=0.0)
    ap.add_argument("--heavy-ratio", type=float, default=0.25)
    ap.add_argument("--count-guess-as-english", action="store_true", default=True,
                    help="Tính cả token không phải âm tiết tiếng Việt (ngoài lexicon) là bằng chứng CS")
    ap.add_argument("--lexicon-only", dest="count_guess_as_english", action="store_false",
                    help="Chỉ tính token có trong lexicon tiếng Anh (định nghĩa chặt nhất)")
    ap.add_argument("--no-wordfreq", action="store_true",
                    help="Không dùng package wordfreq để mở rộng lexicon")
    ap.add_argument("--reannotate", action="store_true",
                    help="Với --source annotations: bỏ cột CS cũ và chạy lại detector mới")

    ap.add_argument("--formats", default="csv,jsonl")
    ap.add_argument("--save-hf", action="store_true", help="Lưu thêm DatasetDict (save_to_disk)")
    ap.add_argument("--matched-control", action="store_true", default=True)
    ap.add_argument("--no-matched-control", dest="matched_control", action="store_false")
    ap.add_argument("--manual-check-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = {f.strip() for f in args.formats.split(",") if f.strip()}

    # ---- 1. Load ----------------------------------------------------------
    print(f"[1/6] Load dữ liệu (source={args.source}) ...")
    if args.source == "hf":
        df = load_from_hf(args.hf_name)
    else:
        if not args.data_path:
            ap.error("--data-path bắt buộc khi --source file/annotations")
        df = load_from_file(args.data_path)

    if args.text_col != "text" and args.text_col in df.columns:
        df = df.rename(columns={args.text_col: "text"})
    if args.label_col != "labels" and args.label_col in df.columns:
        df = df.rename(columns={args.label_col: "labels"})
    if args.split_col != "split" and args.split_col in df.columns:
        df = df.rename(columns={args.split_col: "split"})

    if "split" not in df.columns:
        df["split"] = "unknown"
    df["split"] = df["split"].map(_normalize_split)
    df = df[df["text"].notna()].reset_index(drop=True)
    if "id" not in df.columns:
        df.insert(0, "id", np.arange(len(df)))
    print(f"      {len(df)} dòng | split: {df['split'].value_counts().to_dict()}")

    if args.source == "annotations" and args.reannotate:
        drop = [c for c in df.columns if c.startswith(("cs_", "has_", "n_", "english_",
                                                       "teencode_", "is_cs", "emoji_",
                                                       "url_", "mention_", "hashtag_"))]
        df = df.drop(columns=drop, errors="ignore")

    # ---- 2. Nhãn ----------------------------------------------------------
    print("[2/6] Chuẩn hoá nhãn ...")
    if "labels" in df.columns:
        df["label_ids"] = df["labels"].apply(parse_label_list)
        y = to_multihot(df["label_ids"].tolist())
        for i, name in enumerate(LABEL_NAMES):
            df[f"label_{name}"] = y[:, i]
        df["n_labels"] = y.sum(axis=1)
        df["primary_label"] = df["label_ids"].apply(lambda x: x[0] if len(x) else -1)
    else:
        print("      [warn] không có cột labels -> bỏ qua thống kê theo nhãn")
        df["n_labels"] = 0
        df["primary_label"] = -1

    # ---- 3. Detector ------------------------------------------------------
    print("[3/6] Chạy code-switching detector ...")
    det = CodeSwitchDetector(
        min_english_count=args.min_english_count,
        min_english_ratio=args.min_english_ratio,
        count_guess_as_english=args.count_guess_as_english,
        use_wordfreq=not args.no_wordfreq,
        heavy_ratio=args.heavy_ratio,
    )
    df = annotate_dataframe(df, text_col="text", detector=det)
    print("      cs_group:", df["cs_group"].value_counts().to_dict())
    print(f"      cs_strict = {df['is_cs_strict'].sum()} ({100*df['is_cs_strict'].mean():.2f}%)"
          f" | cs_broad = {df['is_cs_broad'].sum()} ({100*df['is_cs_broad'].mean():.2f}%)")

    # ---- 4. Matched control ----------------------------------------------
    if args.matched_control:
        print("[4/6] Tạo nhóm đối chứng pure_vi cân bằng nhãn ...")
        df["control_pure_vi"] = make_matched_control(df, seed=args.seed)
        print(f"      control_pure_vi = {int(df['control_pure_vi'].sum())} mẫu")
    else:
        print("[4/6] Bỏ qua matched control")

    # ---- 5. Export --------------------------------------------------------
    print("[5/6] Export subset ...")
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(ann_dir / "vigo_cs_annotations.csv", index=False)

    keep_cols = (
        ["id", "text", "labels", "label_ids", "n_labels", "primary_label", "split",
         "cs_group", "cs_level", "is_cs_strict", "is_cs_broad",
         "has_english", "has_teencode", "has_emoji", "has_noise",
         "n_english", "english_ratio", "english_tokens",
         "n_teencode", "teencode_ratio", "teencode_tokens",
         "n_english_guess", "english_guess_tokens",
         "n_foreign_other", "foreign_other_tokens",
         "n_laugh", "n_emoji", "n_url", "n_mention", "n_hashtag",
         "n_word_tokens", "has_diacritics"]
        + [f"label_{n}" for n in LABEL_NAMES]
    )
    index_df = export_subsets(df, out_dir, formats, args.save_hf, keep_cols)

    # ---- 6. Stats ---------------------------------------------------------
    print("[6/6] Thống kê + mẫu kiểm tra tay ...")
    compute_stats(df, out_dir)
    make_manual_check_sample(df, out_dir, n_per_group=args.manual_check_n, seed=args.seed)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["n_rows"] = int(len(df))
    config["cs_group_counts"] = df["cs_group"].value_counts().to_dict()
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n=== SUBSET INDEX ===")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(index_df.to_string(index=False))
    print(f"\nXong. Kết quả ở: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
