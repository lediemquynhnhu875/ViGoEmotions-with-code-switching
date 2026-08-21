"""
cm_evaluate.py — ĐÁNH GIÁ MÔ HÌNH THEO TẬP CON
===============================================
Nạp checkpoint đã fine-tune, dò ngưỡng trên VAL, đánh giá trên TEST, tách kết
quả theo bất kỳ tập con nào bạn chọn.

KHÔNG phát hiện code-mixed — phần đó ở `cm_detect.py`.
File này chỉ đọc cột nhãn có sẵn trong annotations.

Quy trình
---------
    import cm_evaluate as E

    E.ANNOT = "/kaggle/working/cm/annotations/annotations.csv"
    E.list_subsets()                       # xem có tập con nào, cỡ mẫu bao nhiêu

    E.evaluate(
        model_path="/kaggle/input/.../cafe_s3_best",
        base_model="uitnlp/CafeBERT",
        model_tag="cafebert-s3",
        subsets=["all", "english_mixed", "chinese_mixed", "control_pure_vi"],
        tune_on="all",                     # dò ngưỡng trên subset này của val
    )

    E.evaluate_many(CHECKPOINTS, subsets=[...])
    E.report()
    E.export()

Chọn tập con
------------
`subsets=` nhận:
  * tên có sẵn:  "all", "english_mixed", "chinese_mixed", "cs_strict", ...
  * tên cột bool trong annotations:  "control_pure_vi", "has_emoji"
  * biểu thức pandas query:  "n_chinese > 0 and llm_confidence >= 0.8"
  * hàm:  lambda d: d.n_english.gt(2)
Xem `SUBSETS` để biết các tên dựng sẵn, hoặc thêm bằng `register()`.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_loader import load_model_and_tokenizer  # noqa: E402

# ---------------------------------------------------------------- cấu hình
ANNOT = "/kaggle/working/cm/annotations/annotations.csv"
OUT_DIR = "/kaggle/working/cm_eval"
SPLIT_TUNE = "val"
SPLIT_EVAL = "test"
MAX_LENGTH = 128
BATCH_SIZE = 32
USE_FAST = False                     # train_baseline.py dùng use_fast=False
THRESHOLD_GRID = np.arange(0.10, 0.81, 0.02)
TUNE_METRIC = "macro_f1"
N_BOOT = 1000
SEED = 42

LABEL_NAMES = [
    "amusement", "excitement", "joy", "love", "desire", "optimism", "caring",
    "pride", "admiration", "gratitude", "relief", "approval", "realization",
    "surprise", "curiosity", "confusion", "fear", "nervousness", "remorse",
    "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger",
    "annoyance", "disapproval", "neutral",
]
NUM_LABELS = len(LABEL_NAMES)

# ---------------------------------------------------------------- tập con
SUBSETS = {
    "all":             lambda d: pd.Series(True, index=d.index),
    "pure_vi":         lambda d: d["cs_group"].eq("pure_vi"),
    "control_pure_vi": lambda d: d.get("control_pure_vi", pd.Series(False, index=d.index)).astype(bool),
    "cs_strict":       lambda d: d["is_cs_strict"].astype(bool),
    "cs_broad":        lambda d: d.get("is_cs_broad", d["is_cs_strict"]).astype(bool),
    "english_mixed":   lambda d: d["cs_group"].eq("english_mixed"),
    "chinese_mixed":   lambda d: d["cs_group"].eq("chinese_mixed"),
    "translit_only":   lambda d: d.get("n_translit", pd.Series(0, index=d.index)).gt(0),
    "cs_heavy":        lambda d: d["cs_level"].eq("heavy"),
    "cs_light":        lambda d: d["cs_level"].eq("light"),
    "teencode_slang":  lambda d: d["cs_group"].eq("teencode_slang"),
    "emoji_only":      lambda d: d["cs_group"].eq("emoji_only"),
    "other_noise":     lambda d: d["cs_group"].eq("other_noise"),
    "cs_rule":         lambda d: d.get("is_cs_rule", pd.Series(False, index=d.index)).astype(bool),
}

DEFAULT_SUBSETS = ["all", "pure_vi", "control_pure_vi", "cs_strict",
                   "english_mixed", "chinese_mixed", "cs_heavy", "cs_light",
                   "teencode_slang", "emoji_only"]


def register(name, fn):
    """Thêm tập con tự định nghĩa.

        E.register("anh_nhieu", lambda d: d.n_english >= 3)
        E.register("tin_cay_cao", lambda d: d.llm_confidence >= 0.9)
    """
    SUBSETS[name] = fn
    print(f"[i] đã thêm tập con '{name}'")


def _mask(df, spec):
    """spec: tên dựng sẵn | tên cột bool | chuỗi query | hàm."""
    if callable(spec):
        return pd.Series(spec(df), index=df.index).fillna(False).astype(bool)
    if spec in SUBSETS:
        return pd.Series(SUBSETS[spec](df), index=df.index).fillna(False).astype(bool)
    if spec in df.columns:
        return df[spec].fillna(False).astype(bool)
    try:
        return df.index.isin(df.query(spec).index)
    except Exception as e:
        raise ValueError(
            f"Không hiểu tập con {spec!r}. Dùng tên trong E.SUBSETS "
            f"({list(SUBSETS)}), tên cột bool, chuỗi query, hoặc hàm. Lỗi: {e}")


# ---------------------------------------------------------------- dữ liệu
def load_annotations(path=None, split=None, text_col="text"):
    df = pd.read_csv(path or ANNOT)
    if split:
        df = df[df["split"] == split].reset_index(drop=True)
    if text_col != "text":
        if text_col not in df.columns:
            raise KeyError(f"Thiếu cột {text_col!r}. Có: "
                           f"{[c for c in df.columns if c.startswith('text')]}")
        df = df.copy()
        df["text"] = df[text_col]
    cols = [f"label_{n}" for n in LABEL_NAMES]
    if all(c in df.columns for c in cols):
        y = df[cols].to_numpy(dtype=np.int8)
    else:
        import ast
        y = np.zeros((len(df), NUM_LABELS), dtype=np.int8)
        for i, v in enumerate(df["label_ids"]):
            for j in ast.literal_eval(str(v)):
                y[i, int(j)] = 1
    return df, y


def list_subsets(path=None, splits=(SPLIT_TUNE, SPLIT_EVAL), text_col="text"):
    """Bảng cỡ mẫu của mọi tập con, theo từng split."""
    rows = []
    for sp in splits:
        d, _ = load_annotations(path, sp, text_col)
        for name in SUBSETS:
            try:
                rows.append({"subset": name, "split": sp, "n": int(_mask(d, name).sum())})
            except Exception:
                pass
    t = pd.DataFrame(rows).pivot(index="subset", columns="split", values="n")
    t = t.reindex([s for s in DEFAULT_SUBSETS if s in t.index] +
                  [s for s in t.index if s not in DEFAULT_SUBSETS])
    print(t.fillna(0).astype(int).to_string())
    return t


# ---------------------------------------------------------------- suy luận
def predict_proba(texts, model_path, base_model=None, tokenizer=None,
                  max_length=None, batch_size=None, use_fast=None, quiet=False):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model_and_tokenizer(
        model_path, base_model, tokenizer,
        USE_FAST if use_fast is None else use_fast)
    model = model.to(dev).eval()
    bs = batch_size or BATCH_SIZE
    ml = max_length or MAX_LENGTH
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True,
                      max_length=ml, return_tensors="pt").to(dev)
            out.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
            if not quiet and (i // bs) % 20 == 0:
                print(f"    {i}/{len(texts)}", end="\r")
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(out)


# ---------------------------------------------------------------- chỉ số
def metrics(yt, yp):
    return {
        "n_samples": int(len(yt)),
        "micro_f1": f1_score(yt, yp, average="micro", zero_division=0),
        "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
        "weighted_f1": f1_score(yt, yp, average="weighted", zero_division=0),
        "micro_precision": precision_score(yt, yp, average="micro", zero_division=0),
        "macro_precision": precision_score(yt, yp, average="macro", zero_division=0),
        "micro_recall": recall_score(yt, yp, average="micro", zero_division=0),
        "macro_recall": recall_score(yt, yp, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(yt, yp),
        "subset_accuracy": accuracy_score(yt, yp),
        "avg_pred_per_sample": float(yp.sum(axis=1).mean()),
        "avg_gold_per_sample": float(yt.sum(axis=1).mean()),
    }


def macro_f1_shared_labels(yt_a, yp_a, yt_b, yp_b, min_support=5):
    """Macro F1 tính trên CÙNG một tập nhãn cho hai nhóm.

    Macro F1 thường bị lệch khi hai tập con có tập nhãn xuất hiện khác nhau —
    nhãn support = 0 bị tính là F1 = 0 và kéo trung bình xuống. Hàm này chỉ
    lấy các nhãn có đủ support ở CẢ HAI nhóm.
    """
    keep = [i for i in range(NUM_LABELS)
            if yt_a[:, i].sum() >= min_support and yt_b[:, i].sum() >= min_support]
    if not keep:
        return np.nan, np.nan, 0
    a = f1_score(yt_a[:, keep], yp_a[:, keep], average="macro", zero_division=0)
    b = f1_score(yt_b[:, keep], yp_b[:, keep], average="macro", zero_division=0)
    return a, b, len(keep)


def bootstrap_ci(yt, yp, metric="micro_f1", n_boot=None, seed=None, alpha=0.05):
    n = len(yt)
    if n < 10:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed if seed is not None else SEED)
    avg = "macro" if metric.startswith("macro") else "micro"
    v = np.empty(n_boot or N_BOOT)
    for b in range(len(v)):
        idx = rng.integers(0, n, n)
        v[b] = f1_score(yt[idx], yp[idx], average=avg, zero_division=0)
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


def tune_threshold(yt, prob, grid=None, metric=None):
    grid = THRESHOLD_GRID if grid is None else grid
    avg = "macro" if (metric or TUNE_METRIC).startswith("macro") else "micro"
    best_t, best_v = 0.5, -1.0
    for t in grid:
        v = f1_score(yt, (prob >= t).astype(int), average=avg, zero_division=0)
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v


# ---------------------------------------------------------------- đánh giá
def evaluate(model_path, base_model=None, model_tag=None, tokenizer=None,
             annotations=None, subsets=None, tune_on="all", threshold=None,
             text_col="text", out_dir=None, save=True, ci=True, quiet=False):
    """Dò ngưỡng trên val rồi đánh giá trên test, tách theo từng tập con.

    tune_on : tên tập con của VAL dùng để dò ngưỡng. Đặt None để dùng
              `threshold` cố định. Ngưỡng tìm được áp dụng cho MỌI tập con của
              test — không dò riêng từng tập con, tránh rò rỉ thông tin.
    """
    tag = model_tag or Path(str(model_path)).name
    subs = list(subsets or DEFAULT_SUBSETS)
    ann = annotations or ANNOT
    out = Path(out_dir or OUT_DIR) / tag
    out.mkdir(parents=True, exist_ok=True)

    # --- ngưỡng ---
    if threshold is None and tune_on is not None:
        print(f"[{tag}] dò ngưỡng trên {SPLIT_TUNE}/{tune_on} ...")
        dv, yv = load_annotations(ann, SPLIT_TUNE, text_col)
        mv = _mask(dv, tune_on)
        if mv.sum() < 20:
            print(f"    [!] {tune_on} chỉ có {int(mv.sum())} mẫu — dùng 'all' thay thế")
            mv = pd.Series(True, index=dv.index)
        pv = predict_proba(dv.loc[mv, "text"].astype(str).tolist(), model_path,
                           base_model, tokenizer, quiet=quiet)
        threshold, best = tune_threshold(yv[mv.to_numpy()], pv)
        print(f"    ngưỡng = {threshold:.2f}  ({TUNE_METRIC} val = {best:.4f})")
    elif threshold is None:
        threshold = 0.5

    # --- test ---
    print(f"[{tag}] suy luận trên {SPLIT_EVAL} ...")
    dt, yt = load_annotations(ann, SPLIT_EVAL, text_col)
    prob = predict_proba(dt["text"].astype(str).tolist(), model_path,
                         base_model, tokenizer, quiet=quiet)
    np.save(out / f"probs_{SPLIT_EVAL}.npy", prob)
    pred = (prob >= threshold).astype(int)

    npred = pred.sum(axis=1).mean()
    flag = "  <<< NGHI NGỜ: head chưa nạp đúng" if npred > 10 else ""
    print(f"    nhãn/câu = {npred:.2f} (gold {yt.sum(axis=1).mean():.2f}){flag}")

    rows = []
    for name in subs:
        try:
            m = _mask(dt, name).to_numpy()
        except ValueError as e:
            print(f"    [bỏ qua] {e}")
            continue
        if m.sum() < 5:
            continue
        r = metrics(yt[m], pred[m])
        r.update(model=tag, subset=name, threshold=threshold,
                 ratio=float(m.mean()), text_col=text_col)
        if ci:
            for mt in ("micro_f1", "macro_f1"):
                lo, hi = bootstrap_ci(yt[m], pred[m], mt)
                r[f"{mt}_lo"], r[f"{mt}_hi"] = lo, hi
        rows.append(r)

    res = pd.DataFrame(rows)
    front = ["model", "subset", "n_samples", "ratio", "threshold",
             "micro_f1", "micro_f1_lo", "micro_f1_hi",
             "macro_f1", "macro_f1_lo", "macro_f1_hi"]
    res = res[[c for c in front if c in res.columns] +
              [c for c in res.columns if c not in front]]

    if save:
        res.to_csv(out / "metrics.csv", index=False)
        pd.DataFrame([{"model": tag, "model_path": str(model_path),
                       "base_model": base_model, "threshold": threshold,
                       "tune_on": tune_on, "text_col": text_col,
                       "split_tune": SPLIT_TUNE, "split_eval": SPLIT_EVAL,
                       "max_length": MAX_LENGTH, "use_fast": USE_FAST}]) \
            .to_json(out / "config.json", orient="records", indent=2)
        _per_label(dt, yt, pred, subs, out, tag)

    print(res[["subset", "n_samples", "micro_f1", "macro_f1"]].round(4).to_string(index=False))
    return res


def _per_label(dt, yt, pred, subs, out, tag):
    rows = []
    for name in subs:
        try:
            m = _mask(dt, name).to_numpy()
        except Exception:
            continue
        if m.sum() < 5:
            continue
        f1 = f1_score(yt[m], pred[m], average=None, zero_division=0)
        p = precision_score(yt[m], pred[m], average=None, zero_division=0)
        r = recall_score(yt[m], pred[m], average=None, zero_division=0)
        for i, lab in enumerate(LABEL_NAMES):
            rows.append({"model": tag, "subset": name, "label": lab,
                         "support": int(yt[m][:, i].sum()),
                         "precision": p[i], "recall": r[i], "f1": f1[i]})
    pd.DataFrame(rows).to_csv(out / "per_label.csv", index=False)


def evaluate_many(checkpoints, subsets=None, annotations=None, out_dir=None,
                  skip_done=True, **kw):
    """checkpoints: list dict(model_path, base_model, model_tag[, text_col, tokenizer])"""
    out = Path(out_dir or OUT_DIR)
    ok, fail = [], []
    for c in checkpoints:
        tag = c.get("model_tag") or Path(c["model_path"]).name
        if skip_done and (out / tag / "metrics.csv").exists():
            print(f"[skip] {tag}")
            ok.append(tag)
            continue
        print(f"\n{'='*64}\n{tag}\n{'='*64}")
        try:
            evaluate(subsets=subsets, annotations=annotations, out_dir=out_dir,
                     **{**c, "model_tag": tag}, **kw)
            ok.append(tag)
        except Exception as e:
            print(f"[FAIL] {tag}: {type(e).__name__}: {e}")
            fail.append(tag)
    print(f"\n=== {len(ok)} OK, {len(fail)} lỗi ===")
    if fail:
        print(fail)
    return ok, fail


# ---------------------------------------------------------------- báo cáo
def _load(out_dir=None):
    files = glob.glob(f"{out_dir or OUT_DIR}/*/metrics.csv")
    if not files:
        print("[!] chưa có kết quả")
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def compare(metric="micro_f1", models=None, subsets=None, out_dir=None, ci=True):
    """Bảng tập con × model."""
    df = _load(out_dir)
    if df is None:
        return None
    if models:
        df = df[df.model.isin(models)]
    if subsets:
        df = df[df.subset.isin(subsets)]
    t = df.pivot_table(index="subset", columns="model", values=metric).round(4)
    order = [s for s in DEFAULT_SUBSETS if s in t.index]
    t = t.reindex(order + [s for s in t.index if s not in order])
    t.insert(0, "n", df.groupby("subset")["n_samples"].first().reindex(t.index).astype("Int64"))
    print(f"=== {metric} ===")
    print(t.to_string())
    if ci and f"{metric}_lo" in df.columns:
        w = (df[f"{metric}_hi"] - df[f"{metric}_lo"]).groupby(df.subset).mean()
        print("\nđộ rộng CI 95% trung bình — chênh lệch nhỏ hơn con số này là nhiễu:")
        print(w.reindex(t.index).round(4).to_string())
    return t


def gaps(pairs=None, metric="micro_f1", out_dir=None):
    """Chênh lệch giữa các cặp tập con, cho từng model."""
    df = _load(out_dir)
    if df is None:
        return None
    pairs = pairs or [("control_pure_vi", "cs_strict"),
                      ("control_pure_vi", "english_mixed"),
                      ("control_pure_vi", "chinese_mixed"),
                      ("pure_vi", "cs_strict"),
                      ("pure_vi", "teencode_slang"),
                      ("cs_heavy", "cs_light")]
    piv = df.pivot_table(index="model", columns="subset", values=metric)
    out = pd.DataFrame(index=piv.index)
    for a, b in pairs:
        if a in piv.columns and b in piv.columns:
            out[f"{a} − {b}"] = piv[a] - piv[b]
    print(f"=== chênh lệch ({metric}) ===")
    print(out.round(4).to_string())
    print("\ntrung bình:")
    print(out.mean().round(4).to_string())
    if len(out) >= 5:
        try:
            from scipy.stats import wilcoxon
            print("\nWilcoxon (H0: chênh lệch = 0):")
            for c in out.columns:
                v = out[c].dropna()
                if len(v) >= 5:
                    print(f"  {c:32s} p = {wilcoxon(v)[1]:.4f}  "
                          f"dương {int((v>0).sum())}/{len(v)}")
        except ImportError:
            pass
    return out


def export(path="/kaggle/working/ket_qua.xlsx", out_dir=None):
    df = _load(out_dir)
    if df is None:
        return None
    sheets = {"chi_tiet": df}
    for m in ("micro_f1", "macro_f1", "n_samples", "avg_pred_per_sample"):
        if m in df.columns:
            sheets[m] = df.pivot_table(index="subset", columns="model", values=m)
    g = gaps(out_dir=out_dir)
    if g is not None:
        sheets["chenh_lech"] = g
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            for n, t in sheets.items():
                t.round(4).to_excel(w, sheet_name=n[:31])
        print(f"\n[OK] {p}")
    except Exception as e:
        print(f"[i] không ghi được xlsx ({type(e).__name__}) -> CSV")
        for n, t in sheets.items():
            t.round(4).to_csv(p.with_name(f"{p.stem}_{n}.csv"))
    return sheets


print("[cm_evaluate] list_subsets() -> evaluate() / evaluate_many() -> "
      "compare() / gaps() -> export()")