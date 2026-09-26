"""So sánh hai cache detector LLM và tạo một tập review dùng chung."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support


def load_cache(path, tag, suffix):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if tag and row.get("_tag") != tag:
            continue
        rows[str(row["id"])] = row

    output = []
    for row in rows.values():
        output.append({
            "id": str(row["id"]),
            "text": row.get("text", ""),
            f"has_cs_{suffix}": bool(row.get("has_cs", False)),
            f"confidence_{suffix}": float(row.get("confidence", 0) or 0),
            f"tokens_{suffix}": json.dumps(row.get("tokens") or [], ensure_ascii=False),
        })
    return pd.DataFrame(output)


def metrics(gold, pred):
    p, r, f1, _ = precision_recall_fscore_support(
        gold, pred, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f1}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-a", required=True)
    ap.add_argument("--tag-a", default="local:Qwen3-8B")
    ap.add_argument("--name-a", default="qwen")
    ap.add_argument("--cache-b", required=True)
    ap.add_argument("--tag-b", default="local:SeaLLM-7B-v2.5")
    ap.add_argument("--name-b", default="seallm")
    ap.add_argument("--out-dir", default="./llm_comparison")
    ap.add_argument("--review-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score-review", default=None,
                    help="CSV review đã điền gold_has_cs=0/1")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    a = load_cache(args.cache_a, args.tag_a, args.name_a)
    b = load_cache(args.cache_b, args.tag_b, args.name_b)
    paired = a.merge(b, on="id", how="inner", suffixes=("_a", "_b"))
    if paired.empty:
        raise ValueError("Hai cache không có ID chung hoặc tag không đúng.")
    if "text_a" in paired:
        paired["text"] = paired["text_a"].where(
            paired["text_a"].astype(bool), paired["text_b"])
        paired = paired.drop(columns=["text_a", "text_b"])

    ca = f"has_cs_{args.name_a}"
    cb = f"has_cs_{args.name_b}"
    paired["agree"] = paired[ca] == paired[cb]
    kappa = float(cohen_kappa_score(paired[ca], paired[cb]))
    summary = {
        "n_common": len(paired),
        f"positive_{args.name_a}": int(paired[ca].sum()),
        f"positive_{args.name_b}": int(paired[cb].sum()),
        "agreement": float(paired["agree"].mean()),
        "cohen_kappa": kappa,
        "disagreements": int((~paired["agree"]).sum()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paired.to_csv(out / "paired_predictions.csv", index=False)
    paired[~paired["agree"]].to_csv(out / "disagreements.csv", index=False)

    if args.score_review:
        review = pd.read_csv(args.score_review)
        review = review[review["gold_has_cs"].astype(str).str.strip().isin(["0", "1"])]
        gold = review["gold_has_cs"].astype(int)
        scores = {
            args.name_a: metrics(gold, review[ca].astype(bool)),
            args.name_b: metrics(gold, review[cb].astype(bool)),
        }
        print(json.dumps(scores, ensure_ascii=False, indent=2))
        (out / "gold_scores.json").write_text(
            json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    disagree = paired[~paired["agree"]]
    agree_pos = paired[paired["agree"] & paired[ca]]
    agree_neg = paired[paired["agree"] & ~paired[ca]]
    quota = max(args.review_n // 3, 1)
    parts = [
        disagree.sample(min(quota, len(disagree)), random_state=args.seed),
        agree_pos.sample(min(quota, len(agree_pos)), random_state=args.seed),
        agree_neg.sample(min(quota, len(agree_neg)), random_state=args.seed),
    ]
    review = pd.concat(parts).drop_duplicates("id")
    review = review.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    review["gold_has_cs"] = ""
    review["gold_tokens"] = ""
    review["review_note"] = ""
    review.to_csv(out / "review_sample.csv", index=False)
    print(f"Review sample: {len(review)} dòng -> {out / 'review_sample.csv'}")


if __name__ == "__main__":
    main()
