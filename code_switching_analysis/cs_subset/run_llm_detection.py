"""CLI phát hiện code-switching bằng Qwen chạy local, không cần API key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cm_detect as detector


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True,
                    help="File hoặc thư mục dữ liệu ViGoEmotions")
    ap.add_argument("--out-dir", default="./cm_llm_output")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--splits", default="train,val,test",
                    help="Các split cần gán nhãn; dùng val,test để chạy nhanh")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-confidence", type=float, default=0.70)
    ap.add_argument("--limit", type=int, default=None,
                    help="Chạy thử N câu trước khi xử lý toàn bộ")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--test-only", action="store_true",
                    help="Chỉ nạp model và kiểm tra hai câu mẫu rồi dừng")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Tổng số tiến trình/tài khoản chạy song song")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="Shard của tiến trình hiện tại, tính từ 0")
    ap.add_argument("--cache-path", default=None,
                    help="Đường dẫn cache; mặc định <out-dir>/llm_cache.jsonl")
    ap.add_argument("--merge-caches", nargs="+", default=None,
                    help="Gộp cache shard và dựng output mà không nạp LLM")
    return ap.parse_args()


def merge_caches(paths, destination, tag):
    """Gộp cache theo ID; bản xuất hiện sau được ưu tiên."""
    records = {}
    bad = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Không thấy cache: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("_tag") == tag:
                    records[str(row["id"])] = row
            except (json.JSONDecodeError, KeyError, TypeError):
                bad += 1
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        for row in records.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[merge] {len(records)} ID duy nhất -> {destination} | dòng lỗi={bad}")
    return set(records)


def main():
    args = parse_args()
    out = Path(args.out_dir)
    splits = tuple(x.strip() for x in args.splits.split(",") if x.strip())
    tag = "local:" + args.model.split("/")[-1]

    detector.RAW_DATA = args.data_path
    detector.OUT_DIR = out
    cache_path = Path(args.cache_path or out / "llm_cache.jsonl")
    detector.CACHE = str(cache_path)

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Cần num_shards >= 1 và 0 <= shard_index < num_shards")

    # Detector dùng text thô; S1 đủ để giữ tương thích với pipeline đánh giá.
    df = detector.prepare(raw_data=args.data_path, scenarios=("s1",))
    backend_kw = {"model": args.model, "load_in_4bit": not args.no_4bit}

    if args.merge_caches:
        done = merge_caches(args.merge_caches, cache_path, tag)
        expected = set(df.loc[df["split"].isin(splits), "id"].astype(str))
        missing = expected - done
        extra = done - expected
        print(f"[coverage] cần={len(expected)} | có={len(done & expected)} | "
              f"thiếu={len(missing)} | ngoài phạm vi={len(extra)}")
        if missing:
            raise RuntimeError(
                f"Cache chưa đủ, còn thiếu {len(missing)} ID. "
                f"Ví dụ: {sorted(missing)[:10]}")
        built = detector.build(df=df, tag=tag,
                               min_confidence=args.min_confidence, clean=True)
        detector.validate(built, review_n=200)
        annotation = detector.export(built, splits=splits)
        print(f"\nHoàn tất: {annotation}")
        return

    if args.test_only:
        if detector.test_llm(backend="local", **backend_kw) is None:
            raise RuntimeError("LLM local không vượt qua kiểm tra đầu vào/JSON.")
        return

    shard_df = df.iloc[args.shard_index::args.num_shards].copy()
    shard_need = shard_df[shard_df["split"].isin(splits)]
    print(f"[shard {args.shard_index}/{args.num_shards}] {len(shard_need)} câu thuộc shard")
    detector.detect(df=shard_df, backend="local", splits=splits, tag=tag,
                    batch_size=args.batch_size, limit=args.limit, **backend_kw)
    if args.limit is not None or args.num_shards > 1:
        print("\nĐã ghi cache shard. Chỉ xuất annotations sau khi gộp đủ cache.")
        return

    built = detector.build(df=df, tag=tag,
                           min_confidence=args.min_confidence, clean=True)
    detector.validate(built, review_n=200)
    annotation = detector.export(built, splits=splits)
    print(f"\nHoàn tất: {annotation}")


if __name__ == "__main__":
    main()
