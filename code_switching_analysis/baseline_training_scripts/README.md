# Train ViGoEmotions Baselines

Các script này dùng để train lại baseline trên ViGoEmotions từ dataset local/Kaggle input, không load dữ liệu qua Hugging Face.

## Cài thư viện

```bash
pip install -r requirements.txt
```

## Chạy trên Kaggle

Ví dụ dataset nằm ở `/kaggle/input/vigoemotions`:

```bash
python train_baseline.py --model_key xlm-r --data_dir /kaggle/input/vigoemotions
```

Hoặc dùng wrapper:

```bash
python train_xlm_r_baseline.py --data_dir /kaggle/input/vigoemotions
python train_phobert_baseline.py --data_dir /kaggle/input/vigoemotions
python train_mbert_baseline.py --data_dir /kaggle/input/vigoemotions
python train_visobert_baseline.py --data_dir /kaggle/input/vigoemotions
```

## Output

Checkpoint tốt nhất được lưu tại:

```text
/kaggle/working/vigo_baseline_outputs/<model>-vigoemotions/best_model
```

Các file kết quả:

```text
metrics.json
test_predictions.csv
label_pos_weight.csv
```

## Dataset format

Script hỗ trợ:

- `train.csv`, `val.csv`, `test.csv`
- `train.xlsx`, `val.xlsx`, `test.xlsx`
- một file chung có cột `split` hoặc `set`
- Excel nhiều sheet `train`, `val`, `test`
- `.parquet`, `.json`, `.jsonl`

Các cột cần có:

```text
text
labels
```

Nếu tên cột khác, script có hỗ trợ một số alias như `comment`, `sentence`, `content`, `label`, `target`, `set`.
