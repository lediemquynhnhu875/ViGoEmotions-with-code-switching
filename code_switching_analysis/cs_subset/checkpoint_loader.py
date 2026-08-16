"""
checkpoint_loader.py
====================
Nạp checkpoint ViGoEmotions ở mọi định dạng gặp phải. Gộp toàn bộ các bản vá
từng làm bằng monkey patch trên Kaggle thành một module dùng chung.

Xử lý bốn tình huống:

1. Thư mục `save_pretrained()` chuẩn (có config.json + trọng số), kể cả khi nằm
   lồng sâu trong thư mục con.
2. **Archive torch bị Kaggle Models giải nén.** Tệp .pt do torch.save tạo ra là
   một ZIP; Kaggle bung nó thành thư mục data.pkl + data/0,1,2... Module tự nén
   ngược lại rồi torch.load.
3. `torch.save(model, ...)` chứa nguyên nn.Module -> dùng thẳng.
4. **`torch.save(model.state_dict(), ...)` của kiến trúc tùy biến** (encoder +
   dropout + 1 Linear, đúng như bài báo ViGoEmotions mô tả). Module tự dò tên
   thuộc tính encoder và head từ chính state_dict rồi dựng lại kiến trúc đó.

Điểm quan trọng nhất là (4). Nếu nạp state_dict kiểu này vào
`AutoModelForSequenceClassification`:
  - họ BERT  -> head 1 lớp `classifier`, khớp 1-1, chạy đúng
  - họ RoBERTa -> head 2 lớp `classifier.dense` + `classifier.out_proj`,
    lớp `dense` bị khởi tạo NGẪU NHIÊN. Chỉ 2/200 tensor thiếu nên không
    ngưỡng cảnh báo thông thường nào bắt được, model vẫn chạy nhưng đầu ra là
    nhiễu (gán ~28 nhãn/câu).
Vì vậy module này yêu cầu tuyệt đối missing = 0 và unexpected = 0.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer)
from transformers.modeling_outputs import SequenceClassifierOutput

NUM_LABELS = 28

HF_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin",
                   "model.safetensors.index.json", "pytorch_model.bin.index.json")
STATE_DICT_EXTS = (".pt", ".pth", ".bin", ".ckpt", ".safetensors")
EMB_MARKERS = ("embeddings.word_embeddings.weight",     # BERT / RoBERTa / XLM-R
               "encoder.embed_tokens.weight",           # mBART / T5 (BARTpho, ViT5)
               "shared.weight",                         # mBART / T5 embedding dùng chung
               "embed_tokens.weight",
               "word_embedding.weight")


# ---------------------------------------------------------------------------
# Tìm và chuẩn hoá checkpoint
# ---------------------------------------------------------------------------
def _is_hf_dir(d: Path) -> bool:
    return (d / "config.json").exists() and any((d / w).exists() for w in HF_WEIGHT_NAMES)


def is_extracted_torch_archive(d: Path) -> bool:
    """Tệp .pt là ZIP; Kaggle Models bung nó thành thư mục."""
    return (d / "data.pkl").is_file() and (d / "data").is_dir()


def repack_extracted_torch_archive(d: Path, out_dir=None) -> Path:
    """Nén ngược thư mục đã bị bung về lại tệp .pt hợp lệ."""
    if out_dir is None:
        out_dir = Path("/kaggle/working") if Path("/kaggle/working").is_dir() \
            else Path(tempfile.gettempdir())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{d.name}_repacked.pt"
    if out.exists() and out.stat().st_size > 1024:
        print(f"    [i] dùng lại archive đã nén: {out}")
        return out
    files = sorted(f for f in d.rglob("*") if f.is_file())
    files.sort(key=lambda f: f.name != "data.pkl")      # data.pkl lên đầu
    print(f"    [i] nén ngược {len(files)} record -> {out}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in files:
            z.write(f, arcname=f"{d.name}/{f.relative_to(d).as_posix()}")
    return out


def resolve_checkpoint(model_path):
    """Trả về ('hf_dir', path) hoặc ('state_dict', path)."""
    p = Path(model_path)
    if p.is_file():
        return ("state_dict", p)
    if not p.exists():
        raise FileNotFoundError(f"Không tồn tại: {p}")

    if _is_hf_dir(p):
        return ("hf_dir", p)
    for d in sorted(x for x in p.rglob("*") if x.is_dir()):
        if _is_hf_dir(d):
            print(f"    [i] checkpoint HF ở thư mục con: {d}")
            return ("hf_dir", d)

    if is_extracted_torch_archive(p):
        print("    [i] archive torch.save đã bị giải nén thành thư mục")
        return ("state_dict", repack_extracted_torch_archive(p))
    for d in sorted(x for x in p.rglob("*") if x.is_dir()):
        if is_extracted_torch_archive(d):
            print(f"    [i] archive torch.save đã bị giải nén: {d}")
            return ("state_dict", repack_extracted_torch_archive(d))

    cands = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in STATE_DICT_EXTS]
    if cands:
        cands.sort(key=lambda f: f.stat().st_size, reverse=True)
        print(f"    [i] dùng state_dict: {cands[0]}")
        return ("state_dict", cands[0])

    listing = "\n      ".join(str(f) for f in list(p.rglob("*"))[:30])
    raise FileNotFoundError(f"Không tìm thấy trọng số trong {p}.\nNội dung:\n      {listing}")


def clean_state_dict(sd):
    """Bóc lớp bọc: DataParallel, Lightning, dict lồng."""
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint không chứa state_dict dạng dict.")
    out = {}
    for k, v in sd.items():
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Dò cấu trúc + dựng lại kiến trúc
# ---------------------------------------------------------------------------
def detect_structure(sd, num_labels=NUM_LABELS):
    """Trả về (enc_attr, head_attr, head_is_two_layer)."""
    keys = list(sd)
    enc_attr = None
    for marker in EMB_MARKERS:
        for k in keys:
            if k.endswith(marker):
                pre = k[: -len(marker)].rstrip(".")
                enc_attr = pre.split(".")[0] if pre else None
                break
        if enc_attr is not None:
            break

    head_cands = [k for k, v in sd.items()
                  if getattr(v, "ndim", 0) == 2 and v.shape[0] == num_labels]
    head_attr = head_cands[-1].split(".")[0] if head_cands else None

    two_layer = False
    if head_attr:
        sub = [k for k in keys if k.startswith(head_attr + ".")]
        two_layer = any(".dense." in k for k in sub) and any("out_proj" in k for k in sub)
    return enc_attr, head_attr, two_layer


class GenericClassifier(nn.Module):
    """backbone + dropout + 1 Linear, tên thuộc tính khớp checkpoint.

    Tương ứng mô tả trong bài báo: một lớp Dropout (p = 0.2) và một lớp fully
    connected với số nút đầu ra bằng số nhãn.

    Hỗ trợ cả kiến trúc encoder-decoder (BARTpho/mBART, ViT5/T5):
    backbone vẫn được dựng đầy đủ để state_dict khớp 100%, nhưng khi forward
    chỉ chạy phần encoder — T5Model sẽ báo lỗi nếu thiếu decoder_input_ids, còn
    MBartModel thì âm thầm dịch phải input_ids để tạo decoder input, cả hai đều
    không phải điều ta muốn ở bài phân loại.

    `pooling`: 'cls' lấy vị trí đầu, 'mean' lấy trung bình có mask.
    Nếu `check_vs_paper()` lệch nhiều với BARTpho/ViT5, thử đổi sang 'mean'.
    """

    def __init__(self, base_model, enc_attr, head_attr, num_labels=NUM_LABELS,
                 dropout=0.2, use_pooler=True, pooling="cls"):
        super().__init__()
        cfg = AutoConfig.from_pretrained(base_model)
        setattr(self, enc_attr, AutoModel.from_config(cfg))
        self.dropout = nn.Dropout(dropout)
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model")
        setattr(self, head_attr, nn.Linear(hidden, num_labels))
        self._enc_attr, self._head_attr = enc_attr, head_attr
        self.use_pooler = use_pooler
        self.pooling = pooling
        self.is_enc_dec = bool(getattr(cfg, "is_encoder_decoder", False))

    def forward(self, **kw):
        kw.pop("labels", None)
        backbone = getattr(self, self._enc_attr)

        if self.is_enc_dec:
            enc = getattr(backbone, "encoder", None)
            if enc is None:
                raise RuntimeError("Backbone encoder-decoder nhưng không có .encoder")
            out = enc(**kw)
        else:
            out = backbone(**kw)

        pooled = None if self.is_enc_dec else getattr(out, "pooler_output", None)
        if pooled is None or not self.use_pooler:
            h = out.last_hidden_state
            if self.pooling == "mean":
                mask = kw.get("attention_mask")
                if mask is None:
                    pooled = h.mean(dim=1)
                else:
                    mask = mask.unsqueeze(-1).to(h.dtype)
                    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = h[:, 0]

        return SequenceClassifierOutput(
            logits=getattr(self, self._head_attr)(self.dropout(pooled)))


# ---------------------------------------------------------------------------
# API chính
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_path, base_model=None, tokenizer_path=None,
                             use_fast=True, num_labels=NUM_LABELS, strict=True,
                             pooling="cls"):
    kind, path = resolve_checkpoint(model_path)

    if kind == "hf_dir":
        tok_src = tokenizer_path or (str(path) if (path / "tokenizer_config.json").exists()
                                     else base_model)
        if tok_src is None:
            raise ValueError("Checkpoint không kèm tokenizer. Cần base_model hoặc tokenizer.")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path), num_labels=num_labels,
            problem_type="multi_label_classification")
        return model, AutoTokenizer.from_pretrained(tok_src, use_fast=use_fast)

    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        obj = load_file(str(path))
    else:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)

    if isinstance(obj, nn.Module):
        print("    [i] checkpoint chứa nguyên nn.Module")
        tok_src = tokenizer_path or base_model
        if tok_src is None:
            raise ValueError("Checkpoint không kèm tokenizer. Cần base_model hoặc tokenizer.")
        return obj, AutoTokenizer.from_pretrained(tok_src, use_fast=use_fast)

    if not base_model:
        raise ValueError(f"{path} là state_dict rời -> cần base_model.")

    sd = clean_state_dict(obj)
    enc_attr, head_attr, two_layer = detect_structure(sd, num_labels)
    _cfg = AutoConfig.from_pretrained(base_model)
    if getattr(_cfg, "is_encoder_decoder", False):
        print(f"    [i] kiến trúc encoder-decoder ({_cfg.model_type}) "
              f"-> chỉ chạy phần encoder, pooling='{pooling}'")
    print(f"    [i] cấu trúc: backbone='{enc_attr}' head='{head_attr}' "
          f"{'(2 lớp)' if two_layer else '(1 lớp)'}")

    tok = AutoTokenizer.from_pretrained(tokenizer_path or base_model, use_fast=use_fast)

    if enc_attr is None:
        raise RuntimeError(
            f"Không dò được encoder. Key đầu: {list(sd)[:5]}. "
            "Kiến trúc seq2seq (BARTpho/ViT5) cần loader riêng.")

    if two_layer:
        print("    [i] head 2 lớp -> AutoModelForSequenceClassification")
        cfg = AutoConfig.from_pretrained(base_model, num_labels=num_labels,
                                         problem_type="multi_label_classification")
        model = AutoModelForSequenceClassification.from_config(cfg)
        tgt = model.base_model_prefix
        sd = {(f"{tgt}." + k[len(enc_attr) + 1:] if k.startswith(enc_attr + ".") else k): v
              for k, v in sd.items()}
    else:
        model = GenericClassifier(base_model, enc_attr, head_attr, num_labels,
                                  pooling=pooling)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"    [i] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"        missing   : {list(missing)[:8]}")
    if unexpected:
        print(f"        unexpected: {list(unexpected)[:8]}")
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Còn {len(missing)} tensor thiếu, {len(unexpected)} thừa. "
            "Kiến trúc dựng lại chưa khớp checkpoint — kiểm tra base_model.")
    return model, tok


def inspect_checkpoint(model_path, base_model=None, num_labels=NUM_LABELS):
    """In cấu trúc state_dict để chẩn đoán khi load thất bại."""
    kind, path = resolve_checkpoint(model_path)
    print(f"kind = {kind}\npath = {path}\n")
    if kind == "hf_dir":
        print("Thư mục HF hợp lệ.")
        return None

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    print(f"type(obj) = {type(obj)}")
    if isinstance(obj, nn.Module):
        print("=> nn.Module hoàn chỉnh.")
        return obj

    sd = clean_state_dict(obj)
    keys = list(sd)
    print(f"số tensor = {len(keys)}\n--- 12 key đầu ---")
    for k in keys[:12]:
        print(f"  {k:58s} {tuple(sd[k].shape)}")
    print("--- 6 key cuối ---")
    for k in keys[-6:]:
        print(f"  {k:58s} {tuple(sd[k].shape)}")
    enc, head, two = detect_structure(sd, num_labels)
    print(f"\nprefix cấp 1 : {sorted({k.split('.')[0] for k in keys})}")
    print(f"encoder='{enc}'  head='{head}'  số lớp head={2 if two else 1}")
    if base_model:
        cfg = AutoConfig.from_pretrained(base_model)
        print(f"\nbase_model: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}")
    return sd


def sanity_check_predictions(prob, gold_density=2.0, threshold=0.5):
    """Số nhãn/câu phải xấp xỉ mật độ nhãn của gold. >10 là head chưa nạp đúng."""
    n_pred = float((prob >= threshold).sum(axis=1).mean())
    ok = n_pred < 10
    print(f"    n_pred@{threshold} = {n_pred:.2f} (gold ≈ {gold_density}) "
          f"{'OK' if ok else '>>> NGHI NGỜ: head chưa nạp đúng'}")
    return ok, n_pred