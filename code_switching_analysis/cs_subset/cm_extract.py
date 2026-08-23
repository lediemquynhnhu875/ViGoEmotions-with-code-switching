#!/usr/bin/env python
"""
cm_extract.py — TRÍCH XUẤT TẬP CON CODE-MIXED BẰNG LUẬT
========================================================
Dữ liệu thô -> nhãn code-mixed Việt–Anh–Trung bằng `cm_rules.py` -> file
annotations + file riêng cho từng tập con + thống kê + mẫu chấm tay.

KHÔNG nạp model, KHÔNG cần GPU, KHÔNG cần torch. Chạy trên kernel CPU vài giây.
Phần đánh giá mô hình là việc sau, đọc `annotations.csv` mà file này sinh ra.

Thay cho `extract_cs_subset.py` (dùng `cs_detector` cũ, chỉ thấy tiếng Anh) và
cho `cm_detect.py` (dùng LLM).

Chạy dòng lệnh
--------------
    python cm_extract.py --data-path /kaggle/input/vigoemotions \\
                         --out-dir /kaggle/working/cm_subsets

    # thêm ba phiên bản tiền xử lý của bài báo (cần cho bước train/eval sau)
    python cm_extract.py --data-path ... --out-dir ... --scenarios s1,s2,s3

    # định nghĩa chặt hơn: phải có >= 2 token ngoại lai mới tính là code-mixed
    python cm_extract.py --data-path ... --out-dir ... --min-cs-tokens 2

Dùng như thư viện
-----------------
    import cm_extract as X
    df = X.extract("/kaggle/input/vigoemotions", "/kaggle/working/cm_subsets")
    X.sample_sizes(df)

Đầu ra
------
    <out-dir>/
    ├── annotations/annotations.csv          <- file chính, dùng cho mọi bước sau
    ├── subsets/<tập con>/{train,val,test}.csv
    ├── subset_index.csv                     <- cỡ mẫu từng (tập con × split)
    ├── stats/
    │   ├── summary_by_split.csv
    │   ├── group_by_split.csv, level_by_split.csv
    │   ├── top_{cs,english,chinese,unknown,ambiguous}_tokens.csv
    │   └── label_shift_cs_vs_pure.csv       <- cảm xúc nào lệch giữa CS và thuần Việt
    ├── review_sample.csv                    <- mẫu phân tầng để chấm tay
    └── run_config.json                      <- tham số đã dùng, để tái lập
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Tập con
# ---------------------------------------------------------------------------
# Giữ đúng tên mà `cm_evaluate.py` dùng, để bước đánh giá sau không phải sửa gì.
MASKS = {
    "all":            lambda d: pd.Series(True, index=d.index),
    "pure_vi":        lambda d: d["cs_group"].eq("pure_vi"),
    "control_pure_vi": lambda d: d.get("control_pure_vi",
                                       pd.Series(False, index=d.index)).astype(bool),
    "cs_strict":      lambda d: d["is_cs_strict"].astype(bool),
    "cs_broad":       lambda d: d["is_cs_broad"].astype(bool),
    "english_mixed":  lambda d: d["cs_group"].eq("english_mixed"),
    "chinese_mixed":  lambda d: d["cs_group"].eq("chinese_mixed"),
    "other_foreign_mixed": lambda d: d["cs_group"].eq("other_foreign_mixed"),
    "translit_only":  lambda d: d["n_translit"].gt(0),
    "han_script":     lambda d: d["n_han"].gt(0),
    "cs_heavy":       lambda d: d["cs_level"].eq("heavy"),
    "cs_light":       lambda d: d["cs_level"].eq("light"),
    "cs_high_conf":   lambda d: d["cs_confidence"].ge(0.85),
    "cs_loose":       lambda d: d["is_cs_strict"].astype(bool) | d["n_unknown"].gt(0),
    "teencode_slang": lambda d: d["cs_group"].eq("teencode_slang"),
    "emoji_only":     lambda d: d["cs_group"].eq("emoji_only"),
    "other_noise":    lambda d: d["cs_group"].eq("other_noise"),
}


def _p(*a):
    print(*a, flush=True)


def mask(df, name):
    return pd.Series(MASKS[name](df), index=df.index).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Soi dữ liệu trước khi chạy
# ---------------------------------------------------------------------------
TABLE_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl",
              ".pkl", ".pickle"}


def list_input(root="/kaggle/input", max_depth=3, max_files=60):
    """Liệt kê dataset đang Add Input, để lấy đúng đường dẫn mà không phải đoán."""
    root = Path(root)
    if not root.exists():
        _p(f"[!] không có {root} — bạn đang chạy ngoài Kaggle?")
        return []
    rows = []
    for p in sorted(root.rglob("*")):
        depth = len(p.relative_to(root).parts)
        if depth > max_depth:
            continue
        if p.is_file():
            rows.append({"đường dẫn": str(p),
                         "MB": round(p.stat().st_size / 1e6, 2),
                         "đọc được": p.suffix.lower() in TABLE_EXTS})
        if len(rows) >= max_files:
            break
    if not rows:
        _p(f"[!] {root} rỗng — chưa Add Input dataset nào.")
        return []
    t = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 100, "display.width", 200):
        _p(t.to_string(index=False))
    dirs = sorted({str(Path(r).parent) for r in t[t["đọc được"]]["đường dẫn"]})
    if dirs:
        _p("\nThư mục có file dữ liệu đọc được — truyền một trong số này làm RAW_DATA:")
        for d in dirs:
            _p(f"    {d}")
    return t


def peek(data_path, n=5, text_col="text", label_col="labels", split_col="split"):
    """Đọc thử dữ liệu thô và in cấu trúc, TRƯỚC khi chạy extract().

    Cho biết ngay: có nhận ra cột text/labels/split không, nhãn đang ở dạng gì,
    split đặt tên thế nào. Sai ở đây thì mọi thứ phía sau sai theo.
    """
    from extract_cs_subset import _normalize_split, load_from_file, parse_label_list

    _p(f"đọc: {data_path}\n")
    p = Path(data_path)
    if p.is_dir():
        files = [f for f in sorted(p.rglob("*"))
                 if f.is_file() and f.suffix.lower() in TABLE_EXTS]
        _p(f"{len(files)} file dữ liệu trong thư mục:")
        for f in files[:20]:
            _p(f"    {f.name:44s} {f.stat().st_size/1e6:8.2f} MB")
        _p("")

    df = load_from_file(str(data_path))
    for src, dst in ((text_col, "text"), (label_col, "labels"), (split_col, "split")):
        if src != dst and src in df.columns:
            df = df.rename(columns={src: dst})

    _p(f"kích thước : {df.shape[0]} dòng × {df.shape[1]} cột")
    _p(f"cột        : {list(df.columns)}\n")

    for need, why in [("text", "văn bản đầu vào"),
                      ("labels", "nhãn cảm xúc"),
                      ("split", "chia train/val/test")]:
        if need in df.columns:
            _p(f"  [OK]  '{need}' ({why})")
        else:
            _p(f"  [!]   THIẾU '{need}' ({why}) -> truyền {need}_col='<tên cột thật>'")

    if "split" in df.columns:
        raw_counts = df["split"].value_counts().to_dict()
        norm = df["split"].map(_normalize_split).value_counts().to_dict()
        _p(f"\nsplit gốc      : {raw_counts}")
        _p(f"sau chuẩn hoá  : {norm}   (validation/valid/dev -> val)")
        if not {"val", "test"} & set(norm):
            _p("  [!] không có val/test sau chuẩn hoá — kiểm tra lại cột split")

    if "labels" in df.columns:
        _p(f"\nnhãn thô (3 dòng đầu): {df['labels'].head(3).tolist()}")
        ids = df["labels"].head(200).apply(parse_label_list)
        _p(f"sau khi bóc tách     : {ids.head(3).tolist()}")
        empty = int((ids.apply(len) == 0).sum())
        if empty > len(ids) * 0.1:
            _p(f"  [!] {empty}/{len(ids)} dòng đầu bóc ra rỗng — dạng nhãn có thể "
               f"chưa được hỗ trợ, xem parse_label_list trong extract_cs_subset.py")
        else:
            _p(f"  [OK] trung bình {ids.apply(len).mean():.2f} nhãn/câu")

    if "text" in df.columns:
        _p(f"\n{n} câu đầu:")
        for t in df["text"].head(n):
            _p(f"    {str(t)[:110]}")
        na = int(df["text"].isna().sum())
        if na:
            _p(f"  [!] {na} dòng text rỗng, sẽ bị loại khi chạy extract()")
    return df


# ---------------------------------------------------------------------------
# Bước chính
# ---------------------------------------------------------------------------
def extract(data_path, out_dir="./cm_subsets", scenarios=(), canonical="raw",
            detector=None, matched_control=True, formats=("csv",),
            review_n=200, seed=42, run_tests=True, text_col="text",
            label_col="labels", split_col="split", save=True, **detector_kw):
    """Thô -> nhãn code-mixed -> mọi file đầu ra. Trả về DataFrame đầy đủ.

    scenarios : ví dụ ("s1","s2","s3") để sinh thêm cột text_s1/s2/s3 theo đúng
        ba phiên bản tiền xử lý của bài báo. Để rỗng nếu bây giờ chỉ cần tập con
        (nhanh hơn, và S3 vốn cần ViSoLex nên đòi GPU).
    canonical : văn bản dùng để DÒ code-mixed. Mặc định 'raw' — dò trên văn bản
        thô rồi dùng chung cho mọi scenario, để các scenario sau này so sánh trên
        đúng cùng một tập câu. Dò trên text_s1 rồi so với text_s3 là so lệch tập.
    detector_kw : chuyển thẳng cho cm_rules.CodeMixDetector, ví dụ
        min_cs_tokens=2, min_confidence=0.8, count_loanword=True.
    """
    import cm_rules as R
    from extract_cs_subset import (LABEL_NAMES, _normalize_split, load_from_file,
                                   parse_label_list, to_multihot)

    out_dir = Path(out_dir)
    scenarios = tuple(scenarios or ())

    # --- 0. môi trường -----------------------------------------------------
    _p("[0/5] kiểm tra môi trường ...")
    info = R.env_report()
    if not info["wordfreq"]:
        _p("\n    [!] THIẾU wordfreq. Detector vẫn chạy nhưng tầng khử nhập nhằng\n"
           "        theo tần suất bị tắt, số liệu sẽ khác. Chạy:\n"
           "            pip install wordfreq\n"
           "        rồi RESTART kernel trước khi chốt số liệu.\n")
    if run_tests:
        bad = R.run_tests()
        if bad is not None and len(bad):
            raise RuntimeError(
                f"{len(bad)} ca đối chứng SAI — dừng lại, đừng sinh số liệu từ một "
                "detector đang hỏng. Xem bảng ở trên.")

    # --- 1. dữ liệu --------------------------------------------------------
    _p("\n[1/5] đọc dữ liệu thô ...")
    df = load_from_file(data_path)
    for src, dst in ((text_col, "text"), (label_col, "labels"), (split_col, "split")):
        if src != dst and src in df.columns:
            df = df.rename(columns={src: dst})
    if "split" not in df.columns:
        df["split"] = "unknown"
    df["split"] = df["split"].map(_normalize_split)
    df = df[df["text"].notna()].reset_index(drop=True)
    if "id" not in df.columns:
        df.insert(0, "id", np.arange(len(df)))
    df["id"] = df["id"].astype(str)
    df["text_raw"] = df["text"].astype(str)
    _p(f"      {len(df)} dòng | {df['split'].value_counts().to_dict()}")

    # --- 2. tiền xử lý (tuỳ chọn) -----------------------------------------
    if scenarios:
        _p(f"\n[2/5] sinh phiên bản tiền xử lý {list(scenarios)} ...")
        import vigo_preprocess as V
        df = V.build_all_scenarios(df, text_col="text_raw", scenarios=scenarios)
    else:
        _p("\n[2/5] bỏ qua tiền xử lý (không truyền --scenarios)")

    # --- 3. nhãn cảm xúc ---------------------------------------------------
    _p("\n[3/5] chuẩn hoá nhãn cảm xúc ...")
    if "labels" in df.columns:
        df["label_ids"] = df["labels"].apply(parse_label_list)
        y = to_multihot(df["label_ids"].tolist())
        for i, nm in enumerate(LABEL_NAMES):
            df[f"label_{nm}"] = y[:, i]
        df["n_labels"] = y.sum(axis=1)
        df["primary_label"] = df["label_ids"].apply(lambda x: x[0] if len(x) else -1)
        _p(f"      {y.sum()} nhãn / {len(df)} câu "
           f"(trung bình {y.sum(axis=1).mean():.2f} nhãn/câu)")
    else:
        _p("      [!] không có cột labels -> bỏ thống kê theo nhãn")
        df["n_labels"], df["primary_label"] = 0, -1

    # --- 4. detector -------------------------------------------------------
    src = "text_raw" if canonical == "raw" else f"text_{canonical}"
    if src not in df.columns:
        raise KeyError(f"Không có cột {src!r}. Thêm --scenarios {canonical} "
                       f"hoặc dùng --canonical raw.")
    _p(f"\n[4/5] dò code-mixed bằng luật trên '{src}' ...")
    df["text"] = df[src]
    det = detector or R.CodeMixDetector(**detector_kw)
    df = R.build_subsets(df, text_col="text", detector=det)
    if matched_control:
        df = R.make_control(df)

    # nhãn của bộ luật cũ, để đối chiếu trong báo cáo
    try:
        from cs_detector import CodeSwitchDetector, annotate_dataframe as _ann_old
        old = _ann_old(df[["text"]].copy(), text_col="text",
                       detector=CodeSwitchDetector())
        df["is_cs_old"] = old["is_cs_strict"].astype(bool).values
        _p(f"      bộ luật cũ bắt {int(df['is_cs_old'].sum())} | "
           f"bộ luật mới bắt {int(df['is_cs_strict'].sum())} | "
           f"đồng thuận {100*(df['is_cs_old']==df['is_cs_strict']).mean():.1f}%")
    except Exception as e:
        _p(f"      [i] bỏ qua đối chiếu bộ luật cũ ({type(e).__name__})")

    if not save:
        return df

    # --- 5. xuất -----------------------------------------------------------
    _p("\n[5/5] xuất file ...")
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    ann = out_dir / "annotations" / "annotations.csv"
    df.to_csv(ann, index=False)
    _p(f"      [annotations] {ann}")

    idx = export_subsets(df, out_dir, formats=formats)
    write_stats(df, out_dir)
    try:
        R.export_for_review(df, path=str(out_dir / "review_sample.csv"), n=review_n)
    except Exception as e:
        _p(f"      [i] không tạo được mẫu chấm tay ({type(e).__name__}: {e})")

    cfg = {"data_path": str(data_path), "out_dir": str(out_dir),
           "scenarios": list(scenarios), "canonical": canonical,
           "matched_control": matched_control, "seed": seed,
           "n_rows": int(len(df)),
           "wordfreq": info["wordfreq"], "cs_detector": info["cs_detector"],
           "detector": {k: getattr(det, k) for k in
                        ("min_cs_tokens", "min_confidence", "count_loanword",
                         "zipf_gap", "zipf_en_min", "context_promote",
                         "undiacritized_as_teencode", "allow_zh_collide",
                         "heavy_min", "heavy_ratio")},
           "cs_group_counts": df["cs_group"].value_counts().to_dict()}
    (out_dir / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    _p(f"\nXong. Kết quả ở: {out_dir.resolve()}")
    _p(f">>> File dùng cho bước đánh giá sau:\n    {ann}")
    return df


# ---------------------------------------------------------------------------
# Xuất từng tập con
# ---------------------------------------------------------------------------
KEEP_COLS_HEAD = [
    "id", "split", "text", "text_raw", "text_s1", "text_s2", "text_s3",
    "labels", "label_ids", "n_labels", "primary_label",
    "cs_group", "cs_level", "is_cs_strict", "is_cs_broad", "is_cs_old",
    "cs_tokens", "cs_langs", "cs_evidence", "cs_confidence", "cs_ratio",
    "n_cs_tokens", "n_english", "english_tokens", "english_ratio",
    "n_chinese", "n_han", "n_zh_translit", "chinese_tokens",
    "n_other_foreign", "other_foreign_tokens", "n_translit",
    "n_loanword", "loanword_tokens", "n_proper_noun", "proper_noun_tokens",
    "n_teencode", "teencode_tokens", "n_laugh", "laugh_tokens",
    "n_unknown", "unknown_tokens", "n_ambiguous", "ambiguous_tokens",
    "n_emoji", "n_url", "n_mention", "n_hashtag", "n_word_tokens",
    "has_diacritics", "control_pure_vi",
]


def export_subsets(df, out_dir, formats=("csv",), splits=("train", "val", "test")):
    """Mỗi tập con một thư mục, mỗi split một file."""
    out_dir = Path(out_dir)
    cols = [c for c in KEEP_COLS_HEAD if c in df.columns]
    cols += [c for c in df.columns if c.startswith("label_") and c not in cols
             and c != "label_ids"]

    rows = []
    for name in MASKS:
        try:
            m = mask(df, name)
        except Exception as e:
            _p(f"      [bỏ qua] {name}: {type(e).__name__}")
            continue
        for sp in splits:
            part = df[m & df["split"].eq(sp)]
            if len(part) == 0:
                continue
            d = out_dir / "subsets" / name
            d.mkdir(parents=True, exist_ok=True)
            if "csv" in formats:
                part[cols].to_csv(d / f"{sp}.csv", index=False)
            if "jsonl" in formats:
                part[cols].to_json(d / f"{sp}.jsonl", orient="records",
                                   lines=True, force_ascii=False)
            rows.append({
                "subset": name, "split": sp, "n": len(part),
                "pct_split": round(100 * len(part) / max(df["split"].eq(sp).sum(), 1), 2),
                "n_nhãn_tb": round(float(part["n_labels"].mean()), 3)
                if "n_labels" in part.columns else np.nan,
                "conf_tb": round(float(part["cs_confidence"].mean()), 3),
            })

    idx = pd.DataFrame(rows)
    idx.to_csv(out_dir / "subset_index.csv", index=False)
    t = idx.pivot(index="subset", columns="split", values="n")
    t = t.reindex([s for s in MASKS if s in t.index]).fillna(0).astype(int)
    _p("\n=== CỠ MẪU TỪNG TẬP CON ===")
    _p(t.to_string())

    small = [s for s in t.index
             if 0 < t.loc[s, [c for c in ("val", "test") if c in t.columns]].min() < 100]
    if small:
        _p(f"\n[!] tập con dưới 100 mẫu ở val/test — khoảng tin cậy sẽ rất rộng,\n"
           f"    đừng tuyên bố chênh lệch F1 trên chúng: {small}")
    _p(f"\n      [subsets] {out_dir / 'subsets'}")
    return idx


def write_stats(df, out_dir):
    s = Path(out_dir) / "stats"
    s.mkdir(parents=True, exist_ok=True)

    agg = df.groupby("split").agg(
        n=("text", "size"),
        cs_strict=("is_cs_strict", "sum"),
        cs_broad=("is_cs_broad", "sum"),
        english=("n_english", lambda x: int((x > 0).sum())),
        chinese=("n_chinese", lambda x: int((x > 0).sum())),
        han=("n_han", lambda x: int((x > 0).sum())),
        translit=("n_translit", lambda x: int((x > 0).sum())),
        teencode=("has_teencode", "sum"),
        emoji=("has_emoji", "sum"),
        unknown=("n_unknown", lambda x: int((x > 0).sum())),
        conf_tb=("cs_confidence", "mean"),
    )
    for c in ("cs_strict", "cs_broad", "english", "chinese"):
        agg[f"{c}_%"] = (100 * agg[c] / agg["n"]).round(2)
    agg.to_csv(s / "summary_by_split.csv")
    _p("\n=== THỐNG KÊ THEO SPLIT ===")
    _p(agg.to_string())

    pd.crosstab(df["split"], df["cs_group"], margins=True).to_csv(s / "group_by_split.csv")
    pd.crosstab(df["split"], df["cs_level"], margins=True).to_csv(s / "level_by_split.csv")

    from collections import Counter
    for col, name in [("cs_tokens", "top_cs_tokens"),
                      ("english_tokens", "top_english_tokens"),
                      ("chinese_tokens", "top_chinese_tokens"),
                      ("unknown_tokens", "top_unknown_tokens"),
                      ("ambiguous_tokens", "top_ambiguous_tokens"),
                      ("proper_noun_tokens", "top_proper_noun_tokens")]:
        if col not in df.columns:
            continue
        c = Counter()
        for v in df[col].dropna():
            c.update(t.strip().lower() for t in str(v).split("|") if t.strip())
        pd.DataFrame(c.most_common(300), columns=["token", "count"]) \
            .to_csv(s / f"{name}.csv", index=False)

    from extract_cs_subset import LABEL_NAMES
    labs = [f"label_{n}" for n in LABEL_NAMES if f"label_{n}" in df.columns]
    if labs:
        cs, pure = df[df.is_cs_strict], df[df.cs_group.eq("pure_vi")]
        rows = [{"label": l[6:], "support_cs": int(df.loc[df.is_cs_strict, l].sum()),
                 "ratio_cs": float(cs[l].mean()) if len(cs) else 0.0,
                 "ratio_pure_vi": float(pure[l].mean()) if len(pure) else 0.0}
                for l in labs]
        t = pd.DataFrame(rows)
        t["chênh"] = (t.ratio_cs - t.ratio_pure_vi).round(4)
        t.sort_values("chênh", ascending=False).to_csv(
            s / "label_shift_cs_vs_pure.csv", index=False)

    _p(f"\n      [stats] {s}")


def zip_output(out_dir="/kaggle/working/cm_subsets", what="all",
               zip_path=None, link=True):
    """Nén kết quả thành .zip và hiện link tải ngay trong notebook.

    what : "all"          — toàn bộ (annotations + subsets + stats + review)
           "annotations"  — chỉ annotations.csv, file duy nhất cần cho bước sau
           "subsets"      — chỉ thư mục subsets/
           "review"       — chỉ review_sample.csv để chấm tay
           "light"        — annotations + stats + review, bỏ subsets/ (nhẹ nhất)

    Zip luôn ghi vào /kaggle/working để vừa bấm link tải được, vừa hiện trong
    tab Output khi bạn Save Version.
    """
    import os
    import zipfile

    out_dir = Path(out_dir)
    if not out_dir.exists():
        _p(f"[!] không có {out_dir} — chạy extract() trước.")
        return None

    picks = {
        "all":         ["annotations", "subsets", "stats", "review_sample.csv",
                        "subset_index.csv", "run_config.json"],
        "light":       ["annotations", "stats", "review_sample.csv",
                        "subset_index.csv", "run_config.json"],
        "annotations": ["annotations"],
        "subsets":     ["subsets", "subset_index.csv"],
        "review":      ["review_sample.csv"],
    }
    if what not in picks:
        raise ValueError(f"what phải là một trong {list(picks)}")

    root = Path("/kaggle/working")
    if not root.exists():                       # chạy ngoài Kaggle
        root = out_dir.parent
    name = f"{out_dir.name}_{what}.zip" if what != "all" else f"{out_dir.name}.zip"
    zip_path = Path(zip_path) if zip_path else root / name

    files = []
    for item in picks[what]:
        p = out_dir / item
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file()]
    if not files:
        _p(f"[!] không có file nào khớp what={what!r} trong {out_dir}")
        return None

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=str(f.relative_to(out_dir)))
    mb = zip_path.stat().st_size / 1e6
    _p(f"[OK] {len(files)} file -> {zip_path}  ({mb:.2f} MB)")

    if link:
        try:
            from IPython.display import FileLink, display
            # FileLink kiểm tra sự tồn tại theo thư mục làm việc hiện tại, mà
            # notebook thường đã %cd sang thư mục repo -> tạm nhảy về rồi quay lại.
            cwd = os.getcwd()
            try:
                os.chdir(zip_path.parent)
                _p("Bấm vào link dưới để tải:")
                display(FileLink(zip_path.name))
            finally:
                os.chdir(cwd)
        except ImportError:
            pass
        _p(f"\nKhông thấy link? Bấm Save Version rồi lấy ở tab Output, "
           f"hoặc mở panel Data > /kaggle/working/{zip_path.name}")
    return zip_path


def sample_sizes(df, splits=("train", "val", "test")):
    """In lại bảng cỡ mẫu từ một DataFrame đã có nhãn."""
    rows = []
    for name in MASKS:
        r = {"tập con": name}
        for sp in splits:
            d = df[df["split"] == sp]
            r[sp] = int(mask(d, name).sum()) if len(d) else 0
        rows.append(r)
    t = pd.DataFrame(rows).set_index("tập con")
    _p(t.to_string())
    return t


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", required=True,
                    help="File hoặc thư mục dữ liệu thô (csv/xlsx/parquet/jsonl/pkl)")
    ap.add_argument("--out-dir", default="./cm_subsets")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="labels")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--scenarios", default="",
                    help="vd 's1,s2,s3' để sinh thêm cột tiền xử lý. Mặc định: không")
    ap.add_argument("--canonical", default="raw",
                    help="văn bản dùng để DÒ code-mixed: raw | s1 | s2 | s3")

    ap.add_argument("--min-cs-tokens", type=int, default=1,
                    help="số token ngoại lai tối thiểu để tính là code-mixed")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="bỏ nhãn có độ tin cậy dưới ngưỡng (0.85 = chỉ bằng chứng mạnh)")
    ap.add_argument("--count-loanword", action="store_true",
                    help="tính cả từ mượn đã Việt hoá (ship, like, video) là code-mixed")
    ap.add_argument("--zipf-gap", type=float, default=1.5,
                    help="chênh Zipf en-vi tối thiểu để một âm tiết Việt hợp lệ thành tiếng Anh")
    ap.add_argument("--no-context-promote", action="store_true",
                    help="tắt khử nhập nhằng theo ngữ cảnh (chặt nhất)")
    ap.add_argument("--undiacritized-as-teencode", action="store_true",
                    help="đếm 'khong', 'duoc' là teencode như bản cũ")

    ap.add_argument("--no-control", action="store_true",
                    help="bỏ nhóm đối chứng pure_vi cân bằng nhãn")
    ap.add_argument("--formats", default="csv", help="csv hoặc csv,jsonl")
    ap.add_argument("--review-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-tests", action="store_true",
                    help="bỏ qua bộ ca đối chứng của detector (không khuyến khích)")
    a = ap.parse_args()

    extract(
        a.data_path, a.out_dir,
        scenarios=[s.strip() for s in a.scenarios.split(",") if s.strip()],
        canonical=a.canonical,
        matched_control=not a.no_control,
        formats={f.strip() for f in a.formats.split(",") if f.strip()},
        review_n=a.review_n, seed=a.seed, run_tests=not a.no_tests,
        text_col=a.text_col, label_col=a.label_col, split_col=a.split_col,
        # -> CodeMixDetector
        min_cs_tokens=a.min_cs_tokens,
        min_confidence=a.min_confidence,
        count_loanword=a.count_loanword,
        zipf_gap=a.zipf_gap,
        context_promote=not a.no_context_promote,
        undiacritized_as_teencode=a.undiacritized_as_teencode,
    )


if __name__ == "__main__":
    main()