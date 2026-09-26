# Code-Switching Subset Extraction — ViGoEmotions

Bộ script trích xuất subset code-switching cho pipeline:

- **Giai đoạn 1**: ViGoEmotions → detect CS subset → re-evaluate baseline → XLM-R + xLSTM → so sánh macro/micro F1.
- **Giai đoạn 2**: ViCM → pretrain/adapt → transfer sang ViGoEmotions → so sánh có/không ViCM adaptation.

## Định nghĩa ba scenario tiền xử lý

Ba scenario tạo ba phiên bản **đầu vào cho mô hình phân loại cảm xúc** từ cùng
một câu gốc. Chúng không phải ba cách định nghĩa code-switching.

| Scenario | Các bước chính | Xử lý emoji | Mục đích |
|---|---|---|---|
| **S1 — chuẩn hóa văn bản, giữ emoji** | Lowercase; chuẩn hóa HTML/pattern và dấu câu; rút gọn ký tự, emoji lặp; thay teencode bằng tiếng Việt chuẩn | Giữ emoji/emoticon dưới dạng ký hiệu | Bảo toàn tín hiệu cảm xúc trực tiếp từ emoji để tokenizer/model tự học |
| **S2 — chuẩn hóa văn bản, diễn giải emoji** | Thực hiện toàn bộ bước của S1, sau đó thay emoji/emoticon bằng mô tả tiếng Việt | Ví dụ `😂` → `cười ra nước mắt`, `❤️` → `yêu` | Biến tín hiệu hình ảnh thành token chữ, hữu ích với model xử lý emoji kém |
| **S3 — chuẩn hóa từ vựng bằng ViSoLex** | Dùng ViSoLex để chuẩn hóa từ phi chuẩn, teencode và biến thể mạng xã hội theo ngữ cảnh; sau đó lowercase/làm sạch khoảng trắng | Theo output của ViSoLex, không dùng bảng thay emoji thủ công của S2 | Đánh giá lợi ích của mô hình lexical normalization thay cho từ điển thủ công |

Các bước chung của S1/S2 phải giống nhau; khác biệt thực nghiệm chính là
**giữ emoji** ở S1 và **chuyển emoji thành chữ** ở S2. S3 chỉ hợp lệ khi dùng
output ViSoLex thật (ví dụ `s3_visolex_train-val-test.pkl`) hoặc chạy trực tiếp
model `uitnlp/visolex`. Nếu không nạp được ViSoLex và chỉ lowercase/áp dụng từ
điển thủ công thì đó là **fallback**, không được báo cáo như S3 chuẩn.

Việc phát hiện code-switching luôn chạy trên `text_raw`. Nếu dò trên `text_s1`,
`text_s2` hoặc `text_s3`, bước chuẩn hóa có thể dịch/thay thế token ngoại ngữ
(`thanks` → `cảm ơn`) và làm sai thành phần của code-switching subset. Sau khi
xác định cùng một tập ID trên văn bản thô, checkpoint S1/S2/S3 lần lượt được
đánh giá bằng `text_s1`/`text_s2`/`text_s3` của chính các ID đó.

## Files

| File | Vai trò |
|---|---|
| `cs_detector.py` | Detector Vi-En code-switching dùng chung. Thay cho 2 phiên bản đang lệch nhau trong repo. |
| `extract_cs_subset.py` | Đọc dataset → gắn nhãn CS → export subset + thống kê + mẫu kiểm tra tay. |
| `eval_on_subsets.py` | Chạy checkpoint đã train trên từng subset, kèm bootstrap CI + kiểm định ý nghĩa thống kê. |
| `requirements.txt` | Dependency. `wordfreq` là optional nhưng nên cài. |

```bash
pip install -r requirements.txt
```

## Quickstart

### Dùng LLM local (không cần API key)

Khuyến nghị `Qwen/Qwen3-8B` ở chế độ 4-bit trên GPU Kaggle. Kiểm tra model,
chạy thử 100 câu, rồi bỏ `--limit` để xử lý toàn bộ:

```bash
pip install -r requirements.txt
python run_llm_detection.py \
  --data-path /kaggle/input/vigoemotions \
  --out-dir /kaggle/working/cm_qwen3 \
  --test-only

python run_llm_detection.py \
  --data-path /kaggle/input/vigoemotions \
  --out-dir /kaggle/working/cm_qwen3 \
  --limit 100

python run_llm_detection.py \
  --data-path /kaggle/input/vigoemotions \
  --out-dir /kaggle/working/cm_qwen3
```

Kết quả chính nằm ở `cm_qwen3/annotations/annotations.csv`. Cache JSONL được
ghi sau mỗi batch, nên chạy lại cùng lệnh sẽ tiếp tục phần còn thiếu. Nếu GPU
không đủ bộ nhớ, dùng `--model Qwen/Qwen3-4B`.

### Chạy song song hai tài khoản Kaggle

Hai tài khoản phải dùng cùng dataset và model. Đặt `--shard-index` lần lượt
là `0` và `1`:

```bash
# tài khoản A
python run_llm_detection.py --data-path /kaggle/input/vigoemotions \
  --out-dir /kaggle/working/cm_qwen3 --num-shards 2 --shard-index 0

# tài khoản B
python run_llm_detection.py --data-path /kaggle/input/vigoemotions \
  --out-dir /kaggle/working/cm_qwen3 --num-shards 2 --shard-index 1
```

Sau khi đưa hai cache về cùng một notebook, gộp và xuất kết quả:

```bash
python run_llm_detection.py --data-path /kaggle/input/vigoemotions \
  --out-dir /kaggle/working/cm_qwen3 \
  --merge-caches /kaggle/input/cache-a/llm_cache.jsonl \
                 /kaggle/input/cache-b/llm_cache.jsonl
```

Lệnh merge kiểm tra đủ ID trước khi xuất. Cache cũ có thể được sao chép vào
cả hai tài khoản; mỗi shard sẽ tự bỏ qua các ID đã hoàn thành.

### So sánh Qwen3-8B với SeaLLM-7B-v2.5

SeaLLM là model public 7B tập trung vào tiếng Việt, tiếng Trung và các ngôn ngữ
Đông Nam Á. Chạy cùng dữ liệu bằng cách đổi `--model`; cache sẽ có tag riêng:

```bash
python run_llm_detection.py --data-path /kaggle/input/vigoemotions \
  --out-dir /kaggle/working/cm_seallm \
  --model SeaLLMs/SeaLLM-7B-v2.5 --batch-size 8
```

Sau khi hậu kiểm hai cache bằng cùng bộ luật, tạo bảng bất đồng và mẫu chấm tay:

```bash
python compare_llm_caches.py \
  --cache-a /kaggle/input/qwen-cache/llm_cache_clean.jsonl \
  --tag-a local:Qwen3-8B --name-a qwen \
  --cache-b /kaggle/input/seallm-cache/llm_cache_clean.jsonl \
  --tag-b local:SeaLLM-7B-v2.5 --name-b seallm \
  --out-dir /kaggle/working/qwen_vs_seallm --review-n 300
```

Điền `gold_has_cs` trong `review_sample.csv`, rồi chạy lại với
`--score-review <đường-dẫn-review.csv>`. Chọn model theo F1/precision/recall
trên gold review, không chọn theo số câu model tự gán code-switching. Aya vẫn
có preset `local_aya_8b`, nhưng cần chấp nhận quyền truy cập và truyền `hf_token`.

```bash
# 1. Trích xuất subset (từ Hugging Face — cần huggingface-cli login vì dataset gated)
python extract_cs_subset.py --source hf --hf-name uitnlp/vigoemotions --out-dir ./cs_subsets

# hoặc từ file local / thư mục Kaggle
python extract_cs_subset.py --source file --data-path ./corpus/dataset_V1.xlsx --out-dir ./cs_subsets
python extract_cs_subset.py --source file --data-path /kaggle/input/vigoemotions \
       --out-dir /kaggle/working/cs_subsets --save-hf

# 2. Re-evaluate baseline đã train theo subset
python eval_on_subsets.py --annotations ./cs_subsets/annotations/vigo_cs_annotations.csv \
       --model ./outputs/xlm-r/best_model --model-tag xlm-r-baseline --tune-threshold \
       --out-dir ./subset_eval
```

Trong notebook Kaggle:

```python
!python extract_cs_subset.py --source file --data-path /kaggle/input/vigoemotions --out-dir /kaggle/working/cs_subsets
import pandas as pd
train = pd.read_csv("/kaggle/working/cs_subsets/subsets/cs_strict/train.csv")
```

Hoặc dùng như thư viện:

```python
from cs_detector import CodeSwitchDetector, annotate_dataframe
df = annotate_dataframe(df, text_col="text")
cs_train = df[(df.split == "train") & df.is_cs_strict]
```

## Các tier được export

| Subset | Định nghĩa |
|---|---|
| `all` | Toàn bộ split |
| `pure_vi` | Không tiếng Anh, không teencode, không emoji, không noise |
| `cs_strict` | Có ≥ `min_english_count` token tiếng Anh — **định nghĩa hẹp, dùng làm kết quả chính** |
| `cs_broad` | Tiếng Anh **hoặc** teencode (tương đương `is_code_switched` cũ) |
| `english_mixed` | = `cs_strict` |
| `teencode_slang` | Chỉ teencode, không tiếng Anh |
| `emoji_only` | Chỉ emoji |
| `other_noise` | URL / mention / hashtag / laughter |
| `cs_heavy` / `cs_light` | `english_ratio ≥ 0.25` hoặc ≥3 token EN / còn lại |
| `control_pure_vi` | Mẫu `pure_vi` **cân bằng kích thước và phân bố nhãn** với `cs_strict` |

`control_pure_vi` là điểm quan trọng: so `cs_strict` với toàn bộ `pure_vi` là so sánh không công bằng
(khác cỡ mẫu, khác phân bố nhãn). Reviewer sẽ hỏi. Dùng nhóm đối chứng này khi báo cáo chênh lệch F1.

## Cấu trúc output

```
cs_subsets/
├── annotations/vigo_cs_annotations.csv     # toàn bộ dataset + mọi cột CS
├── subsets/<tier>/{train,val,test}.{csv,jsonl}
├── hf_datasets/<tier>/                     # nếu --save-hf
├── stats/
│   ├── summary_by_split.csv
│   ├── cs_group_by_split.csv
│   ├── cs_level_by_split.csv
│   ├── label_distribution_by_tier.csv
│   ├── label_shift_cs_vs_pure.csv          # cảm xúc nào lệch nhất giữa CS và pure_vi
│   └── top_{english,teencode,english_guess,foreign_other}_tokens.csv
├── manual_check_sample.csv                 # mẫu phân tầng để chấm tay
├── subset_index.csv
└── run_config.json
```

## Detector mới khác gì rule cũ

1. **Vietnamese phonotactics thay cho stopword list.** Tiếng Việt viết là đơn âm tiết với cấu trúc
   onset–nucleus–coda chặt. Token ASCII không khớp cấu trúc này gần như chắc chắn không phải tiếng
   Việt. Nhờ vậy `working`, `good`, `team`, `beautiful`, `handsome` bị bắt đúng, còn `cam`, `hàng`,
   `sang`, `con`, `nguyen` không bị nhận nhầm là tiếng Anh — điều mà danh sách `VI_STOPWORDS_ASCII`
   thủ công không làm được.
2. **Xử lý từ nhập nhằng theo ngữ cảnh.** `hot`, `man`, `tin`, `sang`... vừa là âm tiết Việt hợp lệ
   vừa là từ tiếng Anh; chỉ tính là tiếng Anh khi câu đã có ít nhất một token tiếng Anh chắc chắn,
   hoặc khi nằm trong danh sách loanword mạnh (`EN_LOAN_STRONG`).
3. **Tách laughter khỏi teencode.** Rule cũ đếm `haha`, `kkk`, `hihi` là teencode → subset teencode
   phồng lên tới ~43%. Giờ chúng nằm ở `other_noise`.
4. **Chuẩn hoá kéo dài ký tự** trước khi tra từ điển: `goooood` → `good`, `đẹppppp` → `đẹp`.
5. **Tách tên riêng nước ngoài.** `Dima Egiazarov` vào `foreign_other_tokens`, không tính là
   code-switching.
6. **Nhiều tier + `cs_level`**, đổi định nghĩa khi viết báo cáo mà không phải chạy lại.

**Lưu ý về con số**: `cs_broad` sẽ **thấp hơn** con số `code_switched ≈ 48%` trong
`outputs_code_switching/summary_by_split.csv` hiện tại, chủ yếu vì laughter đã bị loại khỏi teencode.
Đây là thay đổi có chủ đích. Khi viết luận văn nên báo cáo cả hai và giải thích lý do, hoặc chạy
`--lexicon-only` để có thêm một cột chặt nhất.

## Tham số hay dùng

```bash
--min-english-count 2      # cần ≥2 token EN mới coi là CS (subset sạch hơn, nhỏ hơn)
--min-english-ratio 0.15   # thêm ràng buộc tỉ lệ
--lexicon-only             # chỉ tính token có trong lexicon EN (chặt nhất, recall thấp hơn)
--heavy-ratio 0.3          # ngưỡng cs_heavy
--no-matched-control       # bỏ nhóm đối chứng
--save-hf                  # xuất thêm DatasetDict
```

Cài `wordfreq` để mở rộng lexicon tiếng Anh từ ~1.7k lên ~30k từ — nên có trước khi chốt số liệu:

```bash
pip install wordfreq
```

## Việc nên làm trước khi báo cáo

1. **Chấm tay `manual_check_sample.csv`** (~200 câu, cột `gold_is_code_switching` điền 1/0), rồi báo
   cáo precision/recall của detector. Đây là rule-based heuristic, không phải gold annotation —
   phần luận văn cần một câu định lượng về độ tin cậy của nó.
2. **Kiểm tra `stats/top_english_guess_tokens.csv` và `top_foreign_other_tokens.csv`.** Đây là hai
   nguồn false positive chính (typo tiếng Việt, tên riêng). Từ nào nhiễu thì thêm vào
   `EN_BLACKLIST` trong `cs_detector.py`.
3. **Dùng `--tune-threshold`** khi eval multi-label: threshold 0.5 cứng thường thiệt cho các nhãn
   hiếm và làm macro F1 giữa các subset khó so sánh.
4. **Đọc `significance_tests.csv`.** Nếu `cs_strict` vs `control_pure_vi` không có ý nghĩa thống kê
   thì chưa nên tuyên bố "model kém trên code-switching" — `english_mixed` ở test chỉ vài trăm mẫu.

## Dùng cho Giai đoạn 2 (ViCM)

Cùng detector chạy được trên ViCM để chọn dữ liệu adapt:

```bash
python extract_cs_subset.py --source file --data-path ./vicm --out-dir ./vicm_cs \
       --no-matched-control --formats jsonl
# dùng ./vicm_cs/subsets/cs_broad/train.jsonl cho MLM/DAPT
```

Giữ nguyên một detector cho cả hai giai đoạn để câu "adapt trên code-mixed text" có cùng định nghĩa
với subset dùng để đo — nếu không, phần so sánh có/không ViCM sẽ khó bảo vệ.
