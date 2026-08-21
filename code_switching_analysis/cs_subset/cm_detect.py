"""
cm_detect.py — PHÁT HIỆN CODE-MIXED
====================================
File này chỉ làm một việc: từ dữ liệu thô, tạo ra file annotations có nhãn
code-mixed Việt–Anh–Trung, kèm số liệu kiểm chứng và các file xuất.

KHÔNG chạy model, KHÔNG tính F1 của mô hình phân loại cảm xúc.
Phần đó nằm ở `cm_evaluate.py`.

Quy trình
---------
    import cm_detect as D

    D.RAW_DATA = "/kaggle/input/<dataset>"
    D.prepare()                       # 1. thô -> nhãn luật + text_s1/s2/s3
    D.detect(backend="local")         # 2. LLM gán nhãn code-mixed (có cache)
    D.build()                         # 3. dựng cột subset + nhóm đối chứng
    D.validate()                      # 4. đối chiếu LLM với luật, mẫu chấm tay
    D.export()                        # 5. xuất annotations + subset + thống kê

Đầu ra chính: `<OUT_DIR>/annotations/annotations.csv`
-> đưa file này sang cm_evaluate.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------- cấu hình
RAW_DATA = "/kaggle/input/YOUR_DATASET"
OUT_DIR = Path("/kaggle/working/cm")
CACHE = "/kaggle/working/cm_llm_cache.jsonl"
SPLITS = ("val", "test")            # chỉ gán nhãn LLM cho hai split này
SCENARIOS = ("s1", "s2", "s3")      # ba phiên bản tiền xử lý
CANONICAL = "raw"                   # văn bản dùng để phát hiện code-mixed

# Khai báo LLM một lần, dùng cho cả test_llm() và detect().
#     D.BACKEND = "openrouter"
#     D.BACKEND_KW = dict(api_key=OR_KEY, model="qwen/qwen-2.5-72b-instruct")
BACKEND = "openrouter"
BACKEND_KW = {}

# Các cấu hình LLM dựng sẵn. Đổi bằng một dòng:  D.use("local_7b", ...)
PRESETS = {
    # --- chạy trên GPU Kaggle, không cần API key ---
    "local_7b": dict(
        backend="local", model="Qwen/Qwen2.5-7B-Instruct", batch_size=8,
        _note="Miễn phí, tái lập được. Cần GPU + bitsandbytes. ~2–3 giờ."),
    "local_3b": dict(
        backend="local", model="Qwen/Qwen2.5-3B-Instruct", batch_size=12,
        _note="Nhẹ hơn, chạy được không cần 4-bit. Yếu hơn ở phần phiên âm. ~1 giờ."),
    "local_14b": dict(
        backend="local", model="Qwen/Qwen2.5-14B-Instruct", batch_size=6,
        _note="Chất lượng cao nhất trong nhóm local. Cần 4-bit + GPU ≥16GB. ~4 giờ."),

    # --- qua API ---
    "openrouter_paid": dict(
        backend="openrouter", model="qwen/qwen-2.5-72b-instruct", batch_size=15,
        _note="Nhanh nhất, chất lượng cao nhất. ~0.3 USD cho 4133 câu. ~20–40 phút."),
    "openrouter_free": dict(
        backend="openrouter", model="auto", free_only=True, batch_size=20,
        sleep=2, _note="Miễn phí nhưng bị giới hạn tốc độ, tự xoay vòng model."),
    "openrouter_deepseek": dict(
        backend="openrouter", model="deepseek/deepseek-chat", batch_size=15,
        _note="Rẻ hơn Qwen 72B, tiếng Trung vẫn tốt."),
    "gemini": dict(
        backend="gemini", model="auto", batch_size=15,
        _note="Cùng họ mô hình mà nhóm tác giả dùng để gán nhãn dataset."),
}


def use(preset, api_key=None, **override):
    """Chọn cấu hình LLM.

        D.use("local_7b")
        D.use("openrouter_paid", api_key=OR_KEY)
        D.use("openrouter_free", api_key=OR_KEY, batch_size=25)
    """
    global BACKEND, BACKEND_KW
    if preset not in PRESETS:
        raise ValueError(f"Không có preset {preset!r}. Có: {list(PRESETS)}")
    cfg = {k: v for k, v in PRESETS[preset].items() if not k.startswith("_")}
    BACKEND = cfg.pop("backend")
    if api_key:
        cfg["api_key"] = api_key
    BACKEND_KW = {**cfg, **override}
    shown = {k: (str(v)[:12] + "..." if k == "api_key" else v)
             for k, v in BACKEND_KW.items()}
    _p(f"[i] preset '{preset}' -> backend={BACKEND}")
    _p(f"    {PRESETS[preset]['_note']}")
    _p(f"    tham số: {shown}")
    return BACKEND, BACKEND_KW


def presets():
    """Bảng so sánh các lựa chọn."""
    rows = []
    for name, c in PRESETS.items():
        rows.append({"preset": name, "backend": c["backend"],
                     "model": c.get("model", ""),
                     "batch": c.get("batch_size", ""),
                     "ghi_chú": c["_note"]})
    t = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        _p(t.to_string(index=False))
    return t


def batch_for(preset=None):
    """batch_size khuyến nghị của preset đang dùng."""
    for name, c in PRESETS.items():
        if preset == name or (preset is None and c["backend"] == BACKEND
                              and c.get("model") == BACKEND_KW.get("model")):
            return c.get("batch_size", 15)
    return 15

_prepared = None                    # df sau bước prepare()
_built = None                       # df sau bước build()


def _p(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 1. chuẩn bị
def prepare(raw_data=None, scenarios=None, save=True):
    """Đọc dữ liệu thô -> nhãn bộ luật + ba phiên bản văn bản.

    Bước này phải chạy trước detect(): nó tạo has_teencode / has_emoji /
    has_noise / primary_label mà bước dựng subset cần.
    """
    global _prepared
    import vigo_preprocess as V
    from cs_detector import CodeSwitchDetector, annotate_dataframe
    from extract_cs_subset import (LABEL_NAMES, _normalize_split, load_from_file,
                                   parse_label_list, to_multihot)

    raw = raw_data or RAW_DATA
    scenarios = scenarios or SCENARIOS

    _p("[1/3] đọc dữ liệu thô ...")
    df = load_from_file(raw)
    df["split"] = df["split"].map(_normalize_split)
    df = df[df["text"].notna()].reset_index(drop=True)
    if "id" not in df.columns:
        df.insert(0, "id", np.arange(len(df)))
    df["id"] = df["id"].astype(str)
    df["text_raw"] = df["text"].astype(str)
    _p(f"      {len(df)} dòng | {df['split'].value_counts().to_dict()}")

    _p("[2/3] sinh các phiên bản tiền xử lý ...")
    df = V.build_all_scenarios(df, text_col="text_raw", scenarios=scenarios)

    _p("[3/3] nhãn cảm xúc + nhãn bộ luật ...")
    df["label_ids"] = df["labels"].apply(parse_label_list)
    y = to_multihot(df["label_ids"].tolist())
    for i, nm in enumerate(LABEL_NAMES):
        df[f"label_{nm}"] = y[:, i]
    df["n_labels"] = y.sum(axis=1)
    df["primary_label"] = df["label_ids"].apply(lambda x: x[0] if len(x) else -1)

    df["text"] = df["text_raw"]
    df = annotate_dataframe(df, text_col="text", detector=CodeSwitchDetector())
    df["is_cs_rule"] = df["is_cs_strict"]        # giữ nhãn luật để đối chiếu
    _p(f"      nhãn luật: {int(df['is_cs_rule'].sum())} câu code-switching")

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        f = OUT_DIR / "prepared.csv"
        df.to_csv(f, index=False)
        _p(f"      -> {f}")
    _prepared = df
    return df


def load_prepared(path=None):
    global _prepared
    f = Path(path or OUT_DIR / "prepared.csv")
    _prepared = pd.read_csv(f)
    _prepared["id"] = _prepared["id"].astype(str)
    _p(f"[i] nạp {len(_prepared)} dòng từ {f}")
    return _prepared


# ---------------------------------------------------------------- 2. LLM
_NOT_BACKEND = ("batch_size", "sleep")     # tham số của annotate(), không phải backend


def test_llm(backend=None, **kw):
    """Gọi thử 2 câu, in output thô + lỗi gốc. Chạy trước detect()."""
    import cs_llm_detector as E
    merged = {k: v for k, v in {**BACKEND_KW, **kw}.items() if k not in _NOT_BACKEND}
    return E.test_backend(backend or BACKEND, **merged)


def list_models(backend=None, api_key=None, **kw):
    import cs_llm_detector as E
    backend = backend or BACKEND
    api_key = api_key or BACKEND_KW.get("api_key")
    if backend == "openrouter":
        return E.list_openrouter_models(api_key, **kw)
    if backend == "gemini":
        return E.list_gemini_models(api_key, **kw)
    _p("backend 'local' không có danh sách — chọn model trên Hugging Face")


def detect(df=None, backend=None, cache=None, splits=None, tag=None,
           batch_size=15, sleep=0.0, limit=None, **backend_kw):
    """Gán nhãn code-mixed bằng LLM. Ghi cache theo dòng, chạy lại không mất công."""
    import cs_llm_detector as E
    d = df if df is not None else _prepared
    if d is None:
        raise RuntimeError("Chưa có dữ liệu — chạy prepare() hoặc load_prepared() trước.")
    bk = backend or BACKEND
    kw = {**BACKEND_KW, **backend_kw}
    # các tham số này của annotate(), không phải của backend
    batch_size = kw.pop("batch_size", batch_size)
    sleep = kw.pop("sleep", sleep)

    # tag mặc định gồm cả tên model -> chạy nhiều model không trộn cache lẫn nhau
    if tag is None:
        m = str(kw.get("model", "")).split("/")[-1].replace(":free", "")
        tag = f"{bk}:{m}" if m and m != "auto" else bk
    _p(f"[i] backend={bk} tag={tag} batch={batch_size}")
    return E.annotate(d, backend=bk, cache=cache or CACHE,
                      splits=splits or SPLITS, batch_size=batch_size,
                      sleep=sleep, limit=limit, tag=tag, **kw)


def default_tag(backend=None, model=None):
    bk = backend or BACKEND
    m = str(model or BACKEND_KW.get("model", "")).split("/")[-1].replace(":free", "")
    return f"{bk}:{m}" if m and m != "auto" else bk


def cache_status(cache=None, splits=None):
    """Đã gán được bao nhiêu, còn thiếu bao nhiêu."""
    import cs_llm_detector as E
    c = E.load_cache(cache or CACHE)
    d = _prepared
    if d is None or c.empty:
        return c
    need = d[d["split"].isin(splits or SPLITS)]
    done = set(c["id"].astype(str))
    miss = [i for i in need["id"].astype(str) if i not in done]
    _p(f"đã gán {len(done)} | cần {len(need)} | còn thiếu {len(miss)}")
    if c.get("has_cs") is not None:
        _p(f"trong đó code-mixed: {int(c['has_cs'].fillna(False).sum())}")
    return c


# ---------------------------------------------------------------- 2b. LỌC HẬU KIỂM
# LLM hay mắc ba lỗi precision. Tầng này bắt chúng bằng quy tắc, không cần
# gọi lại LLM:
#   (A) từ tiếng Việt không dấu (thi, tan, con, ban) bị gán là english
#   (B) tên riêng tiếng Việt (Việt Nam, Hà Nội) bị gán là english
#   (C) từ Hán-Việt (tạ ơn, quốc gia, gia đình) bị gán là chinese_translit
import re as _re

_HAN = _re.compile(r"[一-鿿㐀-䶿]")
_VN_DIA = _re.compile(
    r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    _re.IGNORECASE)
_ASCII_W = _re.compile(r"^[A-Za-z]+$")

# Từ tiếng Anh vẫn giữ dù tần suất tiếng Việt cao (do xuất hiện nhiều trong
# văn bản Việt). Danh sách nhỏ, chỉ những từ chắc chắn là chèn tiếng Anh.
_EN_KEEP = {
    "feel", "good", "check", "team", "deadline", "handsome", "hoodie", "sorry",
    "please", "thanks", "thank", "love", "hate", "beautiful", "amazing",
    "perfect", "crush", "flex", "vibe", "mood", "trend", "toxic", "fake",
    "real", "best", "worst", "happy", "sad", "angry", "boring", "cute",
}
# Âm tiết/từ tiếng Việt thường gặp, gồm nhiều từ Hán-Việt đã là từ tiếng Việt.
# Dùng để bác các token LLM gán nhầm. So khớp cả dạng có dấu lẫn bỏ dấu.
VI_COMMON = set("""
tôi tao mày bạn mình chúng ta các anh chị em con cái cha mẹ bố má ông bà cháu
là thì mà và hay nhưng vì với cho của trong trên dưới này kia đó đây rồi chưa
không được đi đến về ăn uống làm học chơi xem nghe nói nghĩ biết thấy thích yêu
ghét quá rất hơi lắm luôn nữa người nhà nước cửa bàn ghế sách vở
một hai ba bốn năm sáu bảy tám chín mười trăm nghìn triệu
tạ ơn cảm ơn xin lỗi chào hỏi thăm cho nhận trả lại
quốc gia dân tộc nhân dân chính phủ xã hội cộng đồng gia đình dòng họ
hạnh phúc đau khổ tình yêu tình cảm tâm hồn tinh thần vật chất
học sinh sinh viên giáo viên bác sĩ kỹ sư công nhân nông dân
công ty xí nghiệp đại học trung học tiểu học bệnh viện trường học
phụ nữ đàn ông thanh niên thiếu nhi trẻ em người lớn
thiên hạ giang hồ anh hùng hảo hán tiểu thư đại ca sư phụ tỷ muội huynh đệ
thời gian không gian tương lai quá khứ hiện tại
tự do bình đẳng công lý hoà bình chiến tranh
sức khỏe bệnh tật cuộc sống cuộc đời số phận
thi tan con ban cam tin sang hang long man mai bay tay hai chan chin
de vi co la ma may toi khong duoc nguoi nhieu it moi cu lai
viet nam ha noi sai gon hue da nang can tho hai phong
tot xau dep hay dở nhanh cham lon nho cao thap dai ngan
""".split())


def _strip_dia(s):
    import unicodedata
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


VI_COMMON |= {_strip_dia(w) for w in VI_COMMON}
_vn_syl_cache = {}


def _is_vn_syllable(w):
    """Âm tiết tiếng Việt hợp lệ? Dựa vào âm vị học, không cần từ điển."""
    if w not in _vn_syl_cache:
        try:
            from cs_detector import is_vietnamese_syllable
            _vn_syl_cache[w] = is_vietnamese_syllable(w)
        except Exception:
            _vn_syl_cache[w] = False
    return _vn_syl_cache[w]


def _freq(w, lang):
    """Tần suất từ, chỉ dùng làm tín hiệu phụ khi có wordfreq."""
    try:
        from wordfreq import word_frequency
        return word_frequency(w, lang)
    except Exception:
        return 0.0


def _is_vietnamese_word(w):
    """Từ này là tiếng Việt hay chất liệu ngoại lai?

    Ba tín hiệu, xếp theo độ tin cậy:
      1. Có dấu tiếng Việt -> chắc chắn không phải tiếng Anh
      2. Nằm trong danh sách từ Việt thường gặp
      3. Là âm tiết tiếng Việt hợp lệ về mặt âm vị học
    """
    w = w.lower().strip()
    if not w:
        return False
    if w in _EN_KEEP:
        return False
    if _VN_DIA.search(w):
        return True
    if w in VI_COMMON:
        return True
    if _is_vn_syllable(w):
        fv, fe = _freq(w, "vi"), _freq(w, "en")
        return not (fe > 0 and fe > fv * 3)     # tiếng Anh áp đảo thì mới loại
    return False


def _check_token(tok, sentence=""):
    """Trả về type đã sửa, hoặc None nếu phải loại bỏ token."""
    text = str(tok.get("text", "")).strip()
    typ = tok.get("type", "")
    gloss = str(tok.get("gloss", ""))
    if not text:
        return None
    parts = [p for p in _re.split(r"[\s\-_/]+", text) if p]

    if typ == "english":
        # (B) có dấu tiếng Việt -> không bao giờ là tiếng Anh
        if _VN_DIA.search(text):
            return "proper_noun" if text[:1].isupper() else None
        if not all(_ASCII_W.match(p) for p in parts):
            return None
        # (A) mọi thành phần đều là từ tiếng Việt -> loại
        if all(_is_vietnamese_word(p) for p in parts):
            return "proper_noun" if text[:1].isupper() and len(parts) >= 2 else None
        if len(text) <= 2:
            return None
        return "english"

    if typ == "chinese_script":
        return "chinese_script" if _HAN.search(text) else None

    if typ in ("chinese_translit", "other_foreign"):
        if _HAN.search(text):
            return "chinese_script"
        # (C) Hán-Việt: mọi thành phần đều là từ tiếng Việt thường gặp -> loại.
        # "tạ ơn", "quốc gia", "gia đình" là TIẾNG VIỆT, không phải tiếng Trung.
        low = [p.lower() for p in parts]
        if all(p in VI_COMMON or _strip_dia(p) in VI_COMMON for p in low):
            return None
        if text.lower() in VI_COMMON or _strip_dia(text.lower()) in VI_COMMON:
            return None
        # phải chứng minh được bằng chữ Hán trong gloss
        if typ == "chinese_translit" and not _HAN.search(gloss):
            return None
        return typ

    return typ                           # loanword_naturalized, proper_noun


def clean_cache(cache_in=None, cache_out=None, tag=None, verbose=True):
    """Lọc lại toàn bộ cache LLM bằng quy tắc. Ghi ra cache mới, giữ bản gốc."""
    import cs_llm_detector as E
    cin = Path(cache_in or CACHE)
    cout = Path(cache_out or cin.with_name(cin.stem + "_clean.jsonl"))

    import json
    kept, dropped, changed = 0, [], []
    with cout.open("w", encoding="utf-8") as f:
        for line in cin.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
            except Exception:
                continue
            if tag and o.get("_tag") != tag:
                continue
            new_toks = []
            for t in o.get("tokens") or []:
                nt = _check_token(t, o.get("text", ""))
                if nt is None:
                    dropped.append((o.get("text", "")[:60], t.get("text"), t.get("type")))
                    continue
                if nt != t.get("type"):
                    changed.append((t.get("text"), t.get("type"), nt))
                    t = {**t, "type": nt}
                new_toks.append(t)
            cs = [t for t in new_toks
                  if t["type"] in E.CS_TYPES_STRICT]
            o["tokens"] = new_toks
            o["has_cs"] = len(cs) > 0
            o["langs"] = sorted({"zh" if "chinese" in t["type"] else
                                 "en" if t["type"] == "english" else "other"
                                 for t in cs})
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            kept += 1

    if verbose:
        _p(f"[lọc] {kept} câu | bỏ {len(dropped)} token | đổi loại {len(changed)} token")
        if dropped:
            _p("\n  token bị loại (20 đầu):")
            for s, t, ty in dropped[:20]:
                _p(f"    {str(t)[:22]:24s} [{ty:20s}]  câu: {s}")
        if changed:
            _p("\n  token đổi loại (10 đầu):")
            for t, a, b in changed[:10]:
                _p(f"    {str(t)[:22]:24s} {a} -> {b}")
        _p(f"\n  -> {cout}")
    return cout


# ---------------------------------------------------------------- 3. dựng subset
def build(df=None, cache=None, tag=None, count_loanword=False,
          min_confidence=0.0, heavy_min=3, heavy_ratio=0.25, seed=42,
          clean=True):
    """Ghép nhãn LLM vào dữ liệu, sinh cột subset và nhóm đối chứng.

    clean=True: chạy bộ lọc hậu kiểm trước, loại các token LLM gán sai.
    """
    global _built
    import cs_llm_detector as E
    d = df if df is not None else _prepared
    if d is None:
        raise RuntimeError("Chưa có dữ liệu — chạy prepare() trước.")

    c = Path(cache or CACHE)
    if clean:
        c = clean_cache(c, tag=tag)

    out = E.build_subsets(d, cache=str(c), tag=tag,
                          count_loanword=count_loanword,
                          min_confidence=min_confidence,
                          heavy_min=heavy_min, heavy_ratio=heavy_ratio)
    out = E.make_control(out, seed=seed)
    _built = out
    return out


# ---------------------------------------------------------------- 4. kiểm chứng
def validate(df=None, cache=None, review_n=200, tag_a=None, tag_b=None):
    """Ba con số cần có trước khi báo cáo: đồng thuận với luật, LLM bắt thêm
    được gì, và mẫu để chấm tay."""
    import cs_llm_detector as E
    d = df if df is not None else _built
    if d is None:
        raise RuntimeError("Chưa build() — chưa có nhãn để kiểm chứng.")

    _p("=" * 60)
    _p("LLM so với bộ luật")
    _p("=" * 60)
    E.agreement(d, rule_col="is_cs_rule", llm_col="is_cs_strict")

    rule = d["is_cs_rule"].astype(bool)
    llm = d["is_cs_strict"].astype(bool)
    extra = d[llm & ~rule]
    missed = d[rule & ~llm]

    _p(f"\n{'='*60}\nLLM bắt thêm {len(extra)} câu mà luật bỏ sót\n{'='*60}")
    cols = [c for c in ["text", "llm_langs", "cs_tokens", "llm_confidence"] if c in d.columns]
    _p(extra[cols].head(15).to_string(index=False, max_colwidth=50))

    _p(f"\n{'='*60}\nLuật bắt {len(missed)} câu mà LLM bỏ qua\n{'='*60}")
    _p(missed[[c for c in ["text", "english_tokens"] if c in d.columns]]
       .head(10).to_string(index=False, max_colwidth=60))

    if "n_chinese" in d.columns:
        zh = d[d["n_chinese"] > 0]
        _p(f"\n{'='*60}\nCâu có tiếng Trung: {len(zh)}\n{'='*60}")
        _p(zh[cols].head(15).to_string(index=False, max_colwidth=50))
    if "n_translit" in d.columns:
        tl = d[d["n_translit"] > 0]
        _p(f"\ncâu có từ phiên âm (luật KHÔNG THỂ bắt): {len(tl)}")
        _p(tl[cols].head(10).to_string(index=False, max_colwidth=50))

    if tag_a and tag_b:
        _p(f"\n{'='*60}\nĐồng thuận giữa hai LLM\n{'='*60}")
        E.agreement_two_llms(cache or CACHE, tag_a, tag_b)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    E.export_for_review(d, path=str(OUT_DIR / "review_sample.csv"), n=review_n)
    return d


def score_review(path=None):
    """Precision/recall của LLM sau khi bạn điền cột gold_has_cs."""
    import cs_llm_detector as E
    return E.score_review(str(path or OUT_DIR / "review_sample.csv"))


# ---------------------------------------------------------------- 5. xuất
def export(df=None, out_dir=None, splits=None, formats=("csv",)):
    """Xuất annotations + từng subset + thống kê."""
    import cs_llm_detector as E
    d = df if df is not None else _built
    if d is None:
        raise RuntimeError("Chưa build().")
    out = Path(out_dir or OUT_DIR)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    ann = out / "annotations" / "annotations.csv"
    d.to_csv(ann, index=False)
    _p(f"[annotations] {ann}")

    idx = E.export_subsets(d, out_dir=str(out), splits=splits or SPLITS,
                           formats=formats)
    _stats(d, out)
    _p(f"\n>>> Đưa file này sang cm_evaluate.py:\n    {ann}")
    return ann


def _stats(d, out):
    s = out / "stats"
    s.mkdir(parents=True, exist_ok=True)
    d.groupby("split").agg(
        n=("text", "size"),
        cs=("is_cs_strict", "sum"),
        cs_rule=("is_cs_rule", "sum"),
        chinese=("n_chinese", lambda x: int((x > 0).sum())),
        translit=("n_translit", lambda x: int((x > 0).sum())),
        conf_tb=("llm_confidence", "mean"),
    ).to_csv(s / "summary_by_split.csv")
    pd.crosstab(d["split"], d["cs_group"], margins=True).to_csv(s / "group_by_split.csv")

    from collections import Counter
    c = Counter()
    for v in d.get("cs_tokens", pd.Series(dtype=str)).dropna():
        c.update(t.strip() for t in str(v).split("|") if t.strip())
    pd.DataFrame(c.most_common(300), columns=["token", "count"]) \
        .to_csv(s / "top_cs_tokens.csv", index=False)
    _p(f"[stats] {s}")


def summary(df=None):
    d = df if df is not None else _built
    if d is None:
        _p("chưa build()")
        return None
    t = pd.crosstab(d["split"], d["cs_group"], margins=True)
    _p(t.to_string())
    if "is_cs_rule" in d.columns:
        _p(f"\nluật: {int(d['is_cs_rule'].sum())} | LLM: {int(d['is_cs_strict'].sum())}"
           f" | chênh: {int(d['is_cs_strict'].sum()) - int(d['is_cs_rule'].sum()):+d}")
    return t


_p("[cm_detect] prepare() -> test_llm() -> detect() -> build() -> validate() -> export()")