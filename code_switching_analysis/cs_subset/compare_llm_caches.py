"""So sanh hai cache LLM tren cung tap du lieu va tao file cham tay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no", "", "none", "nan"}:
        return False
    raise ValueError(f"Gia tri boolean khong hop le: {value!r}")


def available_tags(path):
    tags = set()
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            try:
                tag = json.loads(line).get("_tag")
                if tag:
                    tags.add(str(tag))
            except (json.JSONDecodeError, AttributeError):
                pass
    return sorted(tags)


def load_cache(path, tag, suffix):
    rows = {}
    bad_lines = 0
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if tag and row.get("_tag") != tag:
                    continue
                rows[str(row["id"])] = row
            except (json.JSONDecodeError, KeyError, TypeError):
                bad_lines += 1
    if not rows:
        raise ValueError(
            f"Khong doc duoc dong nao tu {path} voi tag={tag!r}. "
            f"Tag hien co: {available_tags(path)}")
    output = []
    for row in rows.values():
        output.append({
            "id": str(row["id"]), "text": str(row.get("text", "")),
            f"has_cs_{suffix}": to_bool(row.get("has_cs", False)),
            f"confidence_{suffix}": float(row.get("confidence", 0) or 0),
            f"tokens_{suffix}": json.dumps(row.get("tokens") or [], ensure_ascii=False),
        })
    return pd.DataFrame(output), bad_lines


def classification_metrics(gold, pred):
    gold = np.asarray(gold, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(gold, pred, labels=[False, True]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": int(len(gold)), "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "accuracy": float((gold == pred).mean()),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
    }


def paired_bootstrap(gold, pred_a, pred_b, rounds=2000, seed=42):
    """95% CI cho F1(A)-F1(B), lay mau cung index de giu tinh paired."""
    gold = np.asarray(gold, dtype=bool)
    pred_a = np.asarray(pred_a, dtype=bool)
    pred_b = np.asarray(pred_b, dtype=bool)
    if len(gold) < 2:
        return {"f1_delta_a_minus_b": 0.0, "ci95": [0.0, 0.0], "rounds": 0}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(rounds):
        idx = rng.integers(0, len(gold), len(gold))
        fa = classification_metrics(gold[idx], pred_a[idx])["f1"]
        fb = classification_metrics(gold[idx], pred_b[idx])["f1"]
        deltas.append(fa - fb)
    observed = (classification_metrics(gold, pred_a)["f1"]
                - classification_metrics(gold, pred_b)["f1"])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"f1_delta_a_minus_b": float(observed), "ci95": [float(lo), float(hi)],
            "rounds": rounds}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-a", required=True, help="Cache JSONL/TXT cua Qwen")
    ap.add_argument("--tag-a", default="local:Qwen3-8B")
    ap.add_argument("--name-a", default="qwen")
    ap.add_argument("--cache-b", required=True, help="Cache JSONL/TXT cua Aya")
    ap.add_argument("--tag-b", default="local:aya-expanse-8b")
    ap.add_argument("--name-b", default="aya")
    ap.add_argument("--out-dir", default="./qwen_vs_aya")
    ap.add_argument("--review-n", type=int, default=500)
    ap.add_argument("--disagreement-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score-review", default=None,
                    help="evaluation_sample.csv da dien gold_has_cs=0/1")
    ap.add_argument("--bootstrap-rounds", type=int, default=2000)
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a, bad_a = load_cache(args.cache_a, args.tag_a, args.name_a)
    b, bad_b = load_cache(args.cache_b, args.tag_b, args.name_b)
    ids_a, ids_b = set(a["id"]), set(b["id"])
    paired = a.merge(b, on="id", how="inner", suffixes=("_a", "_b"))
    if paired.empty:
        raise ValueError("Hai cache khong co ID chung.")
    paired["text"] = paired["text_a"].where(
        paired["text_a"].astype(bool), paired["text_b"])
    paired = paired.drop(columns=["text_a", "text_b"])
    ca, cb = f"has_cs_{args.name_a}", f"has_cs_{args.name_b}"
    paired["agree"] = paired[ca] == paired[cb]
    paired["decision_group"] = np.select(
        [paired[ca] & paired[cb], paired[ca] & ~paired[cb], ~paired[ca] & paired[cb]],
        ["both_positive", f"{args.name_a}_only", f"{args.name_b}_only"],
        default="both_negative")
    kappa = cohen_kappa_score(paired[ca], paired[cb])
    summary = {
        "cache_a_unique_ids": len(ids_a), "cache_b_unique_ids": len(ids_b),
        "n_common": len(paired), "only_in_a": len(ids_a - ids_b),
        "only_in_b": len(ids_b - ids_a), "bad_lines_a": bad_a,
        "bad_lines_b": bad_b,
        f"positive_{args.name_a}": int(paired[ca].sum()),
        f"positive_{args.name_b}": int(paired[cb].sum()),
        "agreement": float(paired["agree"].mean()),
        "cohen_kappa": None if np.isnan(kappa) else float(kappa),
        "disagreements": int((~paired["agree"]).sum()),
        "decision_groups": {k: int(v) for k, v in
                            paired["decision_group"].value_counts().items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paired.to_csv(out / "paired_predictions.csv", index=False)
    disagreements = paired[~paired["agree"]]
    disagreements.to_csv(out / "all_disagreements.csv", index=False)

    if args.score_review:
        review = pd.read_csv(args.score_review, dtype={"id": str})
        required = {"id", "gold_has_cs"}
        if not required.issubset(review.columns):
            raise ValueError(f"File review thieu cot: {sorted(required - set(review.columns))}")
        review = review.drop(columns=[ca, cb], errors="ignore").merge(
            paired[["id", ca, cb]], on="id", how="inner")
        valid = review["gold_has_cs"].astype(str).str.strip().isin(["0", "1"])
        review = review[valid].copy()
        if review.empty:
            raise ValueError("Chua co gold_has_cs hop le (chi nhan 0 hoac 1).")
        gold = review["gold_has_cs"].astype(int).astype(bool)
        scores = {
            "reviewed_rows": len(review),
            args.name_a: classification_metrics(gold, review[ca]),
            args.name_b: classification_metrics(gold, review[cb]),
            "paired_bootstrap": paired_bootstrap(
                gold, review[ca], review[cb], args.bootstrap_rounds, args.seed),
        }
        print(json.dumps(scores, ensure_ascii=False, indent=2))
        (out / "gold_scores.json").write_text(
            json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    evaluation = paired.sample(min(args.review_n, len(paired)), random_state=args.seed)
    evaluation = evaluation.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    evaluation["gold_has_cs"] = ""
    evaluation["gold_tokens"] = ""
    evaluation["review_note"] = ""
    evaluation.to_csv(out / "evaluation_sample.csv", index=False)
    error_review = disagreements.sample(
        min(args.disagreement_n, len(disagreements)), random_state=args.seed)
    error_review = error_review.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    error_review["gold_has_cs"] = ""
    error_review["error_type"] = ""
    error_review["review_note"] = ""
    error_review.to_csv(out / "disagreement_review.csv", index=False)
    print(f"Mau tinh metric: {len(evaluation)} -> {out / 'evaluation_sample.csv'}")
    print(f"Mau phan tich loi: {len(error_review)} -> {out / 'disagreement_review.csv'}")


if __name__ == "__main__":
    main()
