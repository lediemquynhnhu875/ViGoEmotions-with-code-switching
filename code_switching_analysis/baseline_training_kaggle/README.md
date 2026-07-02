# Baseline Training Notebooks for Kaggle

Chạy khuyến nghị:

1. `01_train_xlm_r_baseline_kaggle.ipynb`
2. `02_train_phobert_baseline_kaggle.ipynb`
3. `03_train_mbert_baseline_kaggle.ipynb`
4. `04_train_visobert_baseline_kaggle.ipynb`

Các notebook đều:

- Đọc ViGoEmotions từ Kaggle Dataset trong `/kaggle/input`.
- Fine-tune multi-label classifier với 28 labels.
- Dùng `BCEWithLogitsLoss(pos_weight=...)` giống hướng các notebook thực nghiệm cũ.
- Chọn best checkpoint theo validation `macro_f1`.
- Lưu best model vào `/kaggle/working/vigo_baseline_outputs/<model>-vigoemotions/best_model`.

Sau khi train xong, dùng path `best_model` đó trong notebook `vigo_baseline_subset_eval_kaggle.ipynb`.
