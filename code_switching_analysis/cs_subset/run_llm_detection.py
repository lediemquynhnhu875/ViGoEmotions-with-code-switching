"""CLI phát hiện code-switching bằng Qwen chạy local, không cần API key."""

from __future__ import annotations

import argparse
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
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    splits = tuple(x.strip() for x in args.splits.split(",") if x.strip())
    tag = "local:" + args.model.split("/")[-1]

    detector.RAW_DATA = args.data_path
    detector.OUT_DIR = out
    detector.CACHE = str(out / "llm_cache.jsonl")

    # Detector dùng text thô; S1 đủ để giữ tương thích với pipeline đánh giá.
    df = detector.prepare(raw_data=args.data_path, scenarios=("s1",))
    backend_kw = {"model": args.model, "load_in_4bit": not args.no_4bit}

    if args.test_only:
        if detector.test_llm(backend="local", **backend_kw) is None:
            raise RuntimeError("LLM local không vượt qua kiểm tra đầu vào/JSON.")
        return

    detector.detect(df=df, backend="local", splits=splits, tag=tag,
                    batch_size=args.batch_size, limit=args.limit, **backend_kw)
    if args.limit is not None:
        print("\nĐã chạy thử và ghi cache. Bỏ --limit để tiếp tục phần còn lại; "
              "annotations chỉ được xuất khi hoàn tất.")
        return

    built = detector.build(df=df, tag=tag,
                           min_confidence=args.min_confidence, clean=True)
    detector.validate(built, review_n=200)
    annotation = detector.export(built, splits=splits)
    print(f"\nHoàn tất: {annotation}")


if __name__ == "__main__":
    main()
