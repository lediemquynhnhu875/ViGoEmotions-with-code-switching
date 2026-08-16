"""
run_subsets.py
==============
Điều phối toàn bộ Giai đoạn 1: dữ liệu thô -> ba phiên bản tiền xử lý ->
tập con code-switching -> đánh giá mọi checkpoint -> bảng gap.

Dùng trong notebook Kaggle:

    !git clone https://github.com/<user>/ViGoEmotions-with-code-switching.git repo
    %cd /kaggle/working/repo/code_switching_analysis/cs_subset
    !pip -q install wordfreq

    import run_subsets as R
    R.RAW_DATA = "/kaggle/input/<dataset>"
    R.CHECKPOINTS = [...]
    R.build_annotations()
    R.preview()
    R.run_all()
    R.check_vs_paper()
    R.report()
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW_DATA = "/kaggle/input/YOUR_VIGOEMOTIONS_DATASET"
OUT_DIR = Path("/kaggle/working/cs_multi")
ANNOT = OUT_DIR / "vigo_cs_multi.csv"
EVAL_DIR = "/kaggle/working/subset_eval_multi"

CHECKPOINTS = [
    dict(scenario="s3", model_tag="cafebert-s3", base_model="uitnlp/CafeBERT",
         model_path="/kaggle/input/.../cafe_s3_best"),
    dict(scenario="s3", model_tag="mbert-s3", base_model="google-bert/bert-base-multilingual-cased",
         model_path="/kaggle/input/.../mbert_s3_best"),
    dict(scenario="s3", model_tag="phobert-s3", base_model="vinai/phobert-base-v2",
         model_path="/kaggle/input/.../pho_s3_best"),
    dict(scenario="s3", model_tag="vibert-s3", base_model="FPTAI/vibert-base-cased",
         model_path="/kaggle/input/.../vibert_s3_best"),
    dict(scenario="s3", model_tag="visobert-s3", base_model="uitnlp/visobert",
         model_path="/kaggle/input/.../visobert_s3_best"),
    dict(scenario="s3", model_tag="xlmr-s3", base_model="FacebookAI/xlm-roberta-base",
         model_path="/kaggle/input/.../xlm-r_s3_best"),
]

# Macro F1 (%) trên tập dev, Bảng 3 bài báo — dùng để tự kiểm tra tiền xử lý
PAPER_MF1 = {
    "s1": dict(mbert=50.36, xlmr=56.71, phobert=57.03, vibert=49.91,
               visobert=62.33, cafebert=61.45),
    "s2": dict(mbert=53.08, xlmr=56.22, phobert=60.08, vibert=53.34,
               visobert=62.01, cafebert=60.73),
    "s3": dict(mbert=49.98, xlmr=56.01, phobert=56.00, vibert=49.75,
               visobert=61.18, cafebert=59.89),
}


# ---------------------------------------------------------------- bước 1
def build_annotations(raw_data=None, out=None, canonical="raw", scenarios=("s1", "s2", "s3")):
    """Thô -> text_s1/s2/s3 -> nhãn -> tập con code-switching -> CSV.

    Tập con được dò trên `canonical` (mặc định văn bản thô) và dùng chung cho
    mọi scenario, để các scenario so trên đúng cùng một tập câu.
    """
    import vigo_preprocess as V
    from cs_detector import CodeSwitchDetector, annotate_dataframe
    from extract_cs_subset import (LABEL_NAMES, _normalize_split, load_from_file,
                                   make_matched_control, parse_label_list, to_multihot)

    raw_data = raw_data or RAW_DATA
    out = Path(out or ANNOT)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[1/4] đọc dữ liệu thô ...")
    df = load_from_file(raw_data)
    df["split"] = df["split"].map(_normalize_split)
    df = df[df["text"].notna()].reset_index(drop=True)
    if "id" not in df.columns:
        df.insert(0, "id", np.arange(len(df)))
    df["text_raw"] = df["text"].astype(str)
    print(f"      {len(df)} dòng | {df['split'].value_counts().to_dict()}")

    print("[2/4] sinh các phiên bản tiền xử lý ...")
    df = V.build_all_scenarios(df, text_col="text_raw", scenarios=scenarios)

    print("[3/4] nhãn + code-switching ...")
    df["label_ids"] = df["labels"].apply(parse_label_list)
    y = to_multihot(df["label_ids"].tolist())
    for i, nm in enumerate(LABEL_NAMES):
        df[f"label_{nm}"] = y[:, i]
    df["n_labels"] = y.sum(axis=1)
    df["primary_label"] = df["label_ids"].apply(lambda x: x[0] if len(x) else -1)

    df["text"] = df["text_raw"] if canonical == "raw" else df[f"text_{canonical}"]
    df = annotate_dataframe(df, text_col="text", detector=CodeSwitchDetector())
    df["control_pure_vi"] = make_matched_control(df, seed=42)
    print(f"      cs_group: {df['cs_group'].value_counts().to_dict()}")

    print("[4/4] lưu ...")
    df.to_csv(out, index=False)
    print(f"      {out}")

    t = df[df.split == "test"]
    print("\n      cỡ mẫu tập con (test):")
    for k, v in [("all", len(t)), ("pure_vi", (t.cs_group == "pure_vi").sum()),
                 ("cs_strict", t.is_cs_strict.sum()),
                 ("control_pure_vi", t.control_pure_vi.sum()),
                 ("teencode_slang", (t.cs_group == "teencode_slang").sum()),
                 ("emoji_only", (t.cs_group == "emoji_only").sum())]:
        print(f"        {k:16s} {int(v)}")
    return df


def preview(path=None, n=4):
    """Kiểm tra các phiên bản văn bản có thật sự khác nhau."""
    df = pd.read_csv(path or ANNOT)
    cols = [c for c in ["text_raw", "text_s1", "text_s2", "text_s3"] if c in df.columns]
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            same = (df[cols[a]].astype(str) == df[cols[b]].astype(str)).mean()
            print(f"  {cols[a]:9s} vs {cols[b]:9s}: giống {100*same:5.1f}%")
    sel = df[df[cols[0]].astype(str) != df[cols[-1]].astype(str)].head(n)
    for _, r in sel.iterrows():
        print("\n  ---")
        for c in cols:
            print(f"  {c:9s}: {str(r[c])[:100]}")
    return df


# ---------------------------------------------------------------- bước 2
def run_all(ckpts=None, annotations=None, out_dir=None, skip_done=True,
            tune_threshold=True, slow_tokenizer=True, batch_size=32,
            max_length=128, threshold=0.5):
    """Chạy eval; mỗi checkpoint dùng đúng cột văn bản của scenario nó."""
    import eval_on_subsets as E

    ckpts = ckpts if ckpts is not None else CHECKPOINTS
    annotations = str(annotations or ANNOT)
    out_dir = str(out_dir or EVAL_DIR)
    ok, fail = [], []

    for d in ckpts:
        tag, sc = d["model_tag"], d.get("scenario", "s1")
        if skip_done and (Path(out_dir) / tag / "subset_metrics.csv").exists():
            print(f"[skip] {tag}")
            ok.append(tag)
            continue
        argv = [
            "eval_on_subsets.py",
            "--annotations", annotations,
            "--model", d["model_path"],
            "--model-tag", tag,
            "--base-model", d["base_model"],
            "--text-col", f"text_{sc}",
            "--out-dir", out_dir,
            "--batch-size", str(batch_size),
            "--max-length", str(max_length),
            "--threshold", str(threshold),
        ]
        if d.get("tokenizer"):
            argv += ["--tokenizer", d["tokenizer"]]
        if tune_threshold:
            argv += ["--tune-threshold"]
        if slow_tokenizer:
            argv += ["--slow-tokenizer"]

        print(f"\n{'='*62}\n{tag} | scenario={sc} | cột=text_{sc}\n{'='*62}")
        old = sys.argv
        sys.argv = argv
        try:
            E.main()
            ok.append(tag)
        except Exception as e:
            print(f"[FAIL] {tag}: {type(e).__name__}: {e}")
            fail.append(tag)
        finally:
            sys.argv = old

    print(f"\n=== {len(ok)} OK, {len(fail)} lỗi ===")
    if fail:
        print(fail)
    return ok, fail


def check_paths(ckpts=None):
    ckpts = ckpts if ckpts is not None else CHECKPOINTS
    rows = [{"model_tag": d["model_tag"], "scenario": d.get("scenario"),
             "exists": Path(d["model_path"]).exists(), "path": d["model_path"]}
            for d in ckpts]
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 95, "display.width", 240):
        print(df.to_string(index=False))
    if (~df.exists).any():
        print(f"\n[!] {int((~df.exists).sum())} đường dẫn không tồn tại — chưa Add Input?")
    return df


# ---------------------------------------------------------------- bước 3
def _load(out_dir=None):
    files = glob.glob(f"{out_dir or EVAL_DIR}/*/subset_metrics.csv")
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["arch"] = df.model.str.rsplit("-", n=1).str[0]
    df["scenario"] = df.model.str.rsplit("-", n=1).str[1]
    return df


def check_vs_paper(out_dir=None, tol=3.0):
    """Đối chiếu macro F1 với bài báo — cổng kiểm tra tiền xử lý."""
    df = _load(out_dir)
    if df is None:
        print("[!] chưa có kết quả")
        return None
    a = df[df.subset == "all"][["arch", "scenario", "threshold", "macro_f1"]].copy()
    a["đo_được"] = (a.macro_f1 * 100).round(2)
    a["bài_báo"] = [PAPER_MF1.get(s, {}).get(m, np.nan) for m, s in zip(a.arch, a.scenario)]
    a["lệch"] = (a["đo_được"] - a["bài_báo"]).round(2)
    a = a[["arch", "scenario", "threshold", "đo_được", "bài_báo", "lệch"]] \
        .sort_values(["scenario", "arch"]).reset_index(drop=True)
    print(a.to_string(index=False))
    ok = a["lệch"].abs() <= tol
    print(f"\n{int(ok.sum())}/{len(a)} model lệch trong ±{tol} điểm "
          f"(lệch dương 1–3 là bình thường do tune ngưỡng trên val).")
    if (~ok).any():
        print("[!] lệch lớn -> nghi tiền xử lý chưa khớp:")
        print(a[~ok].to_string(index=False))
    return a


def report(out_dir=None, metric="micro_f1"):
    """Bảng gap theo kiến trúc × scenario, cho cả code-switching và teencode."""
    df = _load(out_dir)
    if df is None:
        print("[!] chưa có kết quả")
        return None
    piv = df.pivot_table(index=["arch", "scenario"], columns="subset", values=metric)
    piv["gap_cs"] = piv["control_pure_vi"] - piv["cs_strict"]
    if {"pure_vi", "teencode_slang"}.issubset(piv.columns):
        piv["gap_teencode"] = piv["pure_vi"] - piv["teencode_slang"]

    print(f"=== {metric} theo tập con ===")
    print(piv.round(4).to_string())

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    for name, col in [("code-switching", "gap_cs"), ("teencode", "gap_teencode")]:
        if col not in piv.columns:
            continue
        tab = piv[col].unstack("scenario")
        print(f"\n=== gap {name} ===")
        print(tab.round(4).to_string())
        print(tab.mean().round(4).to_frame("trung bình").T.to_string())
        if wilcoxon is None:
            continue
        for c in tab.columns:
            v = tab[c].dropna()
            if len(v) >= 5:
                print(f"  {c}: mean={v.mean():+.4f} sd={v.std():.4f} "
                      f"dương {int((v>0).sum())}/{len(v)} Wilcoxon p={wilcoxon(v)[1]:.4f}")
        cols = [c for c in tab.columns if tab[c].notna().sum() >= 4]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                sub = tab[[cols[i], cols[j]]].dropna()
                if len(sub) >= 4:
                    p = wilcoxon(sub[cols[i]], sub[cols[j]])[1]
                    print(f"  {cols[i]} vs {cols[j]}: "
                          f"Δ={sub[cols[i]].mean()-sub[cols[j]].mean():+.4f} p={p:.4f}")
    return piv