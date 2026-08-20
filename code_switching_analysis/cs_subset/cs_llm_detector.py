"""
cs_llm_detector.py
==================
Nhận diện code-switching Việt–Anh–Trung bằng LLM, cho tập val và test của
ViGoEmotions.

Vì sao cần LLM
--------------
Bộ luật `cs_detector.py` chỉ bắt được chất liệu ngoại lai viết bằng chữ Latin
theo đúng chính tả gốc (feel good, check in). Ba loại sau nó **không** bắt được:

  1. Chữ Hán trực tiếp — 谢谢, 加油  (thực ra dò script là ra, đã xử lý ở đây)
  2. **Từ nước ngoài phiên âm sang chữ Việt** — "xia xìa" (谢谢), "ci gua",
     "pà pà" (爸爸). Đây là loại khó nhất: hình thức là chữ Việt, chỉ có ngữ
     nghĩa mới lộ ra nguồn gốc. Không quy tắc nào bắt được, chỉ LLM.
  3. Cụm tiếng Anh viết sai chính tả hoặc Việt hoá — "sờ tai" (style),
     "ô kê", "sến sẩm".

Thiết kế
--------
* Ba tầng: dò script (rẻ, chắc chắn) -> bộ luật -> LLM. LLM chỉ trả lời phần
  mà hai tầng trước không quyết được, nhưng vẫn xem toàn câu để có ngữ cảnh.
* **Có cache theo dòng** (JSONL, ghi nối tiếp). Kaggle chết giữa chừng thì chạy
  lại sẽ bỏ qua phần đã xong.
* **Phân loại token thành 6 nhóm** để bạn tự quyết ranh giới sau, không phải
  chạy lại: english / chinese_script / chinese_translit / other_foreign /
  loanword_naturalized / proper_noun.
* Xuất ra đúng lược đồ cột mà `eval_on_subsets.py` đang dùng, nên toàn bộ
  pipeline phía sau chạy không cần sửa.

Chọn LLM
--------
* `gemini`  — Gemini 2.5 Flash. **Khuyến nghị.** Chính nhóm tác giả ViGoEmotions
  dùng Gemini Flash để gán nhãn cảm xúc, nên dùng lại cùng họ mô hình là một
  lập luận nhất quán khi viết luận văn. Có gói miễn phí. Cần API key.
* `local`   — Qwen2.5-7B-Instruct chạy trên GPU Kaggle. Không cần key, không
  phụ thuộc mạng, tái lập được hoàn toàn. Qwen mạnh tiếng Trung nên hợp với
  yêu cầu phát hiện chữ Hán và phiên âm Hán-Việt.
* `openai`  — bất kỳ endpoint tương thích OpenAI.

Dùng hai LLM rồi đo mức đồng thuận là cách rẻ nhất để có một con số về độ tin
cậy của nhãn — xem `agreement()`.

Cách dùng
---------
    import cs_llm_detector as L

    L.annotate(df, backend="gemini", api_key="...", splits=("val", "test"))
    df2 = L.build_subsets(df, cache="cs_llm_cache.jsonl")
    L.agreement(df2)                      # LLM so với bộ luật
    L.export_for_review(df2, n=200)       # mẫu để chấm tay
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Tầng dò script — chắc chắn, không cần LLM
# ---------------------------------------------------------------------------
HAN_RE = re.compile(r"[一-鿿㐀-䶿]")
HIRAGANA_KATAKANA_RE = re.compile(r"[぀-ヿ]")
HANGUL_RE = re.compile(r"[가-힯ᄀ-ᇿ]")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
THAI_RE = re.compile(r"[฀-๿]")

SCRIPT_RES = {
    "chinese_script": HAN_RE,
    "japanese": HIRAGANA_KATAKANA_RE,
    "korean": HANGUL_RE,
    "cyrillic": CYRILLIC_RE,
    "thai": THAI_RE,
}


def scan_scripts(text: str) -> dict:
    """Đếm ký tự theo hệ chữ. Chữ Hán xuất hiện là chắc chắn có tiếng Trung."""
    s = str(text)
    return {k: len(r.findall(s)) for k, r in SCRIPT_RES.items()}


# ---------------------------------------------------------------------------
# 2. Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Bạn là chuyên gia ngôn ngữ học phân tích hiện tượng chuyển mã (code-switching) trong bình luận mạng xã hội tiếng Việt.

NHIỆM VỤ: với mỗi câu, xác định có chất liệu ngôn ngữ NGOẠI LAI xen vào tiếng Việt hay không, và liệt kê chính xác các token đó.

PHÂN LOẠI TOKEN (bắt buộc dùng đúng các nhãn này):

1. "english" — từ/cụm tiếng Anh viết đúng chính tả gốc.
   Ví dụ: feel good, check in, deadline, oversize, handsome

2. "chinese_script" — viết bằng chữ Hán.
   Ví dụ: 谢谢, 加油, 好

3. "chinese_translit" — tiếng Trung PHIÊN ÂM sang chữ Việt. ĐÂY LÀ LOẠI QUAN TRỌNG NHẤT.
   Ví dụ: "xia xìa" hoặc "xiê xiê" = 谢谢 (cảm ơn)
          "chia du" / "cha du" = 加油 (cố lên)
          "pà pà" = 爸爸 (bố), "ma ma" = 妈妈 (mẹ)
          "nỉ hảo" = 你好 (xin chào)
          "ai ya" = 哎呀
   Dấu hiệu: chuỗi âm tiết không có nghĩa tiếng Việt thông thường trong ngữ cảnh, nhưng khớp âm tiếng Trung phổ thông.

4. "other_foreign" — ngôn ngữ khác (Hàn, Nhật, Pháp...) kể cả dạng phiên âm.
   Ví dụ: "an nhon" = 안녕, "sa rang hê" = 사랑해, "ka wa i" = かわいい

5. "loanword_naturalized" — từ gốc ngoại đã Việt hoá hoàn toàn, người Việt dùng như từ Việt.
   Ví dụ: ship, like, share, video, internet, ok, cà phê, xe buýt, ga, sếp
   (Vẫn liệt kê ra, nhưng đánh dấu riêng để phía sau quyết định có tính hay không.)

6. "proper_noun" — tên riêng người, địa danh, thương hiệu.
   Ví dụ: Facebook, TikTok, Messi, Dima Egiazarov, Shopee
   (Liệt kê nhưng KHÔNG tính là chuyển mã.)

KHÔNG PHẢI CHUYỂN MÃ — tuyệt đối không liệt kê:
- Teencode / viết tắt tiếng Việt: ko, k, j, z, dc, mik, bth, vs, cx, mn, ny, cmt
- Tiếng cười, thán từ: haha, kkk, hihi, huhu, hehe
- Emoji, ký hiệu, dấu câu
- Tiếng Việt viết sai chính tả hoặc thiếu dấu: "khong", "duoc", "thich"
- Từ tiếng Việt thuần dù nghe lạ

QUY TẮC:
- Chỉ liệt kê token thực sự xuất hiện trong câu, giữ nguyên dạng gốc.
- Nếu phân vân giữa tiếng Việt và phiên âm ngoại, hãy nghiêng về tiếng Việt và hạ confidence.
- confidence là số thực 0.0–1.0 cho toàn câu.

ĐẦU RA: chỉ JSON hợp lệ, không giải thích, không markdown. Mảng có đúng số phần tử bằng số câu đầu vào, theo đúng thứ tự:

[
  {"id": <id câu>, "has_cs": true/false, "langs": ["en"|"zh"|"ko"|"ja"|"other"],
   "tokens": [{"text": "...", "type": "english|chinese_script|chinese_translit|other_foreign|loanword_naturalized|proper_noun", "gloss": "nghĩa tiếng Việt"}],
   "confidence": 0.0-1.0}
]

has_cs = true khi có ít nhất một token thuộc english, chinese_script, chinese_translit, hoặc other_foreign.
loanword_naturalized và proper_noun KHÔNG làm has_cs thành true."""

FEWSHOT = [
    ("Cái clip này hay vc, xem xong feel good luôn",
     {"has_cs": True, "langs": ["en"],
      "tokens": [{"text": "clip", "type": "loanword_naturalized", "gloss": "đoạn phim"},
                 {"text": "feel good", "type": "english", "gloss": "thấy dễ chịu"}],
      "confidence": 0.95}),
    ("xia xìa nhé bạn hiền, mai gặp",
     {"has_cs": True, "langs": ["zh"],
      "tokens": [{"text": "xia xìa", "type": "chinese_translit", "gloss": "谢谢 - cảm ơn"}],
      "confidence": 0.9}),
    ("加油 nha em, sắp thi rồi",
     {"has_cs": True, "langs": ["zh"],
      "tokens": [{"text": "加油", "type": "chinese_script", "gloss": "cố lên"}],
      "confidence": 1.0}),
    ("ko hiểu j luôn z mn ơi, bth mà",
     {"has_cs": False, "langs": [], "tokens": [], "confidence": 0.95}),
    ("Dima Egiazarov bởi vì chúng tôi là người Việt Nam",
     {"has_cs": False, "langs": [],
      "tokens": [{"text": "Dima Egiazarov", "type": "proper_noun", "gloss": "tên người"}],
      "confidence": 0.9}),
    ("bức ảnh xuất sắc ❤️ haha",
     {"has_cs": False, "langs": [], "tokens": [], "confidence": 0.98}),
]


def build_user_prompt(batch):
    """batch: list[(id, text)]"""
    ex = "\n".join(
        f"Câu: {t}\nJSON: {json.dumps(o, ensure_ascii=False)}" for t, o in FEWSHOT)
    lines = "\n".join(f'{{"id": {json.dumps(str(i), ensure_ascii=False)}, '
                      f'"text": {json.dumps(str(t), ensure_ascii=False)}}}'
                      for i, t in batch)
    return (f"VÍ DỤ THAM KHẢO\n{ex}\n\n"
            f"===\nPHÂN TÍCH {len(batch)} CÂU SAU. "
            f"Trả về JSON array đúng {len(batch)} phần tử, đúng thứ tự, "
            f"trường id giữ nguyên:\n{lines}")


# ---------------------------------------------------------------------------
# 3. Backend
# ---------------------------------------------------------------------------
def _extract_json(s: str):
    """Bóc JSON ra khỏi markdown, lời dẫn, hoặc phần suy luận của model."""
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    i, j = s.find("["), s.rfind("]")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        o = json.loads(s[i:j + 1])
        if isinstance(o, dict):
            for k in ("results", "data", "items", "output", "sentences"):
                if isinstance(o.get(k), list):
                    return o[k]
            return [o]
        return o
    return json.loads(s)


def list_gemini_models(api_key=None, only_generate=True):
    """Liệt kê model mà API key của bạn thực sự dùng được.

    Tên model Gemini thay đổi theo thời gian và theo khu vực, nên đừng đoán —
    chạy hàm này trước.
    """
    from google import genai
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    names = []
    for m in client.models.list():
        n = getattr(m, "name", "")
        acts = getattr(m, "supported_actions", None) or []
        if only_generate and acts and "generateContent" not in acts:
            continue
        names.append(n.replace("models/", ""))
    for n in names:
        print("  ", n)
    return names


def _rank_flash(names):
    """Xếp hạng model theo thứ tự ưu tiên dùng thử.

    Model mới nhất chưa chắc dùng được — gói miễn phí thường bị chặn (403).
    Trả về cả danh sách để backend tự lùi xuống bản cũ hơn khi bị từ chối.
    """
    bad = ("tts", "image", "embedding", "vision", "aqa", "live", "native-audio",
           "thinking", "exp-", "learnlm", "gemma")
    cand = [n for n in names if "flash" in n and not any(b in n for b in bad)]
    if not cand:
        cand = [n for n in names if "gemini" in n and not any(b in n for b in bad)]
    def score(n):
        m = re.search(r"(\d+)\.(\d+)", n)
        v = int(m.group(1)) * 10 + int(m.group(2)) if m else 0
        return (v, "lite" not in n, "preview" not in n, "latest" in n, -len(n))
    return sorted(cand, key=score, reverse=True)


def _pick_flash(names):
    r = _rank_flash(names)
    return r[0] if r else None


_DENIED = ("PERMISSION_DENIED", "NOT_FOUND", "403", "404",
           "denied access", "is not found", "not supported")


class GeminiBackend:
    def __init__(self, api_key=None, model="auto"):
        from google import genai
        from google.genai import types
        self._types = types
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Thiếu API key. Truyền api_key=... hoặc đặt biến GEMINI_API_KEY.")
        self.client = genai.Client(api_key=key)

        if model == "auto":
            names = []
            try:
                for m in self.client.models.list():
                    n = getattr(m, "name", "").replace("models/", "")
                    acts = getattr(m, "supported_actions", None) or []
                    if not acts or "generateContent" in acts:
                        names.append(n)
            except Exception as e:
                print(f"[!] không liệt kê được model ({type(e).__name__})")
            self.candidates = _rank_flash(names) or [
                "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest",
                "gemini-1.5-flash"]
            print(f"[i] thứ tự thử: {self.candidates[:6]}")
        else:
            self.candidates = [model]
        self.model = self.candidates[0]
        self._verified = False

    def __call__(self, user_prompt):
        cfg = self._types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, temperature=0.0,
            response_mime_type="application/json")
        last = None
        while self.candidates:
            try:
                r = self.client.models.generate_content(
                    model=self.model, contents=user_prompt, config=cfg)
                if not self._verified:
                    print(f"    [i] dùng model: {self.model}")
                    self._verified = True
                return r.text
            except Exception as e:
                msg = str(e)
                if any(k in msg for k in _DENIED):
                    print(f"    [bỏ] {self.model}: bị từ chối, thử model kế tiếp")
                    self.candidates.pop(0)
                    if not self.candidates:
                        raise RuntimeError(
                            "Không model nào dùng được với API key này. "
                            "Chạy L.list_gemini_models(key) để xem danh sách "
                            "rồi truyền model='...' thủ công.") from e
                    self.model = self.candidates[0]
                    last = e
                    continue
                raise            # lỗi khác (quota, mạng) -> để annotate() retry
        raise last


class OpenAIBackend:
    def __init__(self, api_key=None, model="gpt-4.1-mini", base_url=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                             base_url=base_url)
        self.model = model

    def __call__(self, user_prompt):
        r = self.client.chat.completions.create(
            model=self.model, temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_prompt}])
        return r.choices[0].message.content


def list_openrouter_models(api_key=None, free_only=True, query="", limit=40):
    """Liệt kê model trên OpenRouter. Đặt free_only=True để chỉ lấy model miễn phí."""
    import requests
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    h = {"Authorization": f"Bearer {key}"} if key else {}
    r = requests.get("https://openrouter.ai/api/v1/models", headers=h, timeout=30)
    r.raise_for_status()
    rows = []
    for m in r.json().get("data", []):
        mid = m.get("id", "")
        pr = m.get("pricing", {}) or {}
        free = str(pr.get("prompt", "1")) in ("0", "0.0") and \
               str(pr.get("completion", "1")) in ("0", "0.0")
        if free_only and not free:
            continue
        if query and query.lower() not in mid.lower():
            continue
        rows.append({"id": mid, "free": free,
                     "ctx": m.get("context_length"),
                     "gia_1M_input": pr.get("prompt")})
    rows.sort(key=lambda x: x["id"])
    for x in rows[:limit]:
        print(f"  {x['id']:56s} ctx={x['ctx']}")
    print(f"\n{len(rows)} model{' miễn phí' if free_only else ''}")
    return [x["id"] for x in rows]


# Ưu tiên theo họ model: Qwen và DeepSeek mạnh tiếng Trung nhất, hợp với việc
# phát hiện phiên âm Hán-Việt.
OR_FAMILY_RANK = ["qwen", "deepseek", "glm", "minimax", "llama", "gemma",
                  "mistral", "gemini", "gpt"]
OR_BAD = ("vision", "-vl-", "-vl:", "image", "embed", "tts", "audio", "coder",
          "math", "guard", "rerank", "distill", "-1b", "-2b", "-3b", "-4b",
          "thinking", "-r1", "reasoner")


def _fetch_openrouter(api_key=None):
    """Lấy danh sách model đang sống từ OpenRouter."""
    import requests
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    h = {"Authorization": f"Bearer {key}"} if key else {}
    r = requests.get("https://openrouter.ai/api/v1/models", headers=h, timeout=30)
    r.raise_for_status()
    out = []
    for m in r.json().get("data", []):
        pr = m.get("pricing", {}) or {}
        free = str(pr.get("prompt", "1")) in ("0", "0.0") and \
               str(pr.get("completion", "1")) in ("0", "0.0")
        out.append({"id": m.get("id", ""), "free": free,
                    "ctx": m.get("context_length") or 0,
                    "price": float(pr.get("prompt") or 0)})
    return out


def _rank_openrouter(api_key=None, prefer_free=True, min_ctx=8000):
    """Xếp hạng model OpenRouter theo mức phù hợp với tác vụ này.

    Lấy danh sách trực tiếp từ API thay vì dùng slug cứng — slug :free bị gỡ
    hoặc đổi tên khá thường xuyên.
    """
    try:
        models = _fetch_openrouter(api_key)
    except Exception as e:
        print(f"[!] không lấy được danh sách model ({type(e).__name__}) — dùng mặc định")
        return ["deepseek/deepseek-chat", "qwen/qwen-2.5-72b-instruct",
                "google/gemini-2.0-flash-001"]

    def fam(mid):
        for i, f in enumerate(OR_FAMILY_RANK):
            if f in mid.lower():
                return len(OR_FAMILY_RANK) - i
        return 0

    cand = [m for m in models
            if not any(b in m["id"].lower() for b in OR_BAD)
            and m["ctx"] >= min_ctx and fam(m["id"]) > 0]

    free = sorted([m for m in cand if m["free"]],
                  key=lambda m: (fam(m["id"]), m["ctx"]), reverse=True)
    paid = sorted([m for m in cand if not m["free"]],
                  key=lambda m: (fam(m["id"]), -m["price"], m["ctx"]), reverse=True)

    ids = ([m["id"] for m in free] + [m["id"] for m in paid]) if prefer_free \
        else ([m["id"] for m in paid] + [m["id"] for m in free])
    print(f"[i] {len(free)} model miễn phí, {len(paid)} model trả phí phù hợp")
    return ids[:12]


class OpenRouterBackend:
    """OpenRouter — một API key dùng được hàng trăm model, có nhiều model miễn phí.

    Lấy key ở https://openrouter.ai/keys
    """

    def __init__(self, api_key=None, model="auto", base_url="https://openrouter.ai/api/v1",
                 site="https://kaggle.com", title="ViGoEmotions-CS", json_mode=True):
        from openai import OpenAI
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("Thiếu OpenRouter key. Lấy ở https://openrouter.ai/keys")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.headers = {"HTTP-Referer": site, "X-Title": title}
        self.json_mode = json_mode

        if model == "auto":
            self.candidates = _rank_openrouter(key)
            print(f"[i] thứ tự thử: {self.candidates[:5]}")
        else:
            self.candidates = [model]
        self.model = self.candidates[0]
        self._verified = False

    def __call__(self, user_prompt):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
        while self.candidates:
            kw = dict(model=self.model, temperature=0.0, messages=msgs,
                      extra_headers=self.headers)
            if self.json_mode:
                kw["response_format"] = {"type": "json_object"}
            try:
                r = self.client.chat.completions.create(**kw)
                if not self._verified:
                    print(f"    [i] dùng model: {self.model}")
                    self._verified = True
                return r.choices[0].message.content
            except Exception as e:
                msg = str(e)
                # model không hỗ trợ json mode -> thử lại ở chế độ text
                if self.json_mode and ("response_format" in msg or "json" in msg.lower()):
                    print(f"    [i] {self.model} không hỗ trợ json mode -> chuyển sang text")
                    self.json_mode = False
                    continue
                # OpenRouter thường chỉ luôn slug thay thế trong thông báo lỗi
                alt = re.search(r"use this slug instead:\s*([\w\-./:]+)", msg)
                if alt and alt.group(1) not in self.candidates:
                    new = alt.group(1)
                    print(f"    [i] {self.model} đã bị gỡ -> dùng slug thay thế: {new}")
                    self.candidates[0] = new
                    self.model = new
                    self.json_mode = True
                    continue

                if any(k in msg for k in _DENIED) or "No endpoints" in msg \
                        or "not a valid model" in msg:
                    print(f"    [bỏ] {self.model}\n         lý do: {msg[:300]}")
                    if "data policy" in msg or "No endpoints" in msg:
                        print("         >>> Nhiều khả năng do CÀI ĐẶT QUYỀN RIÊNG TƯ.\n"
                              "             Vào https://openrouter.ai/settings/privacy và bật\n"
                              "             'Enable training and logging' — model :free bắt buộc\n"
                              "             phải bật mục này mới có endpoint.")
                    self.candidates.pop(0)
                    if not self.candidates:
                        raise RuntimeError(
                            f"Không model nào dùng được.\nLỗi cuối: {msg[:400]}\n"
                            "Kiểm tra theo thứ tự:\n"
                            "  1. https://openrouter.ai/settings/privacy -> bật training/logging\n"
                            "  2. L.list_openrouter_models(key) -> lấy đúng id model\n"
                            "  3. L.test_backend('openrouter', api_key=key, model='...')") from e
                    self.model = self.candidates[0]
                    self.json_mode = True
                    continue
                raise
        raise RuntimeError("hết model để thử")


class LocalBackend:
    """Qwen2.5-7B-Instruct trên GPU Kaggle. Không cần API key."""

    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=2048,
                 load_in_4bit=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            except Exception:
                pass
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, user_prompt):
        import torch
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None)
        return self.tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)


def make_backend(backend="gemini", **kw):
    return {"gemini": GeminiBackend, "openrouter": OpenRouterBackend,
            "openai": OpenAIBackend, "local": LocalBackend}[backend](**kw)


def test_backend(backend="openrouter", verbose=True, **kw):
    """Gọi thử MỘT lần với 2 câu, in ra lỗi gốc đầy đủ nếu hỏng.

    Chạy hàm này trước khi annotate — nó cho biết chính xác vấn đề nằm ở đâu
    thay vì chỉ thấy 'không dùng được'.
    """
    batch = [("t1", "Cái clip này hay vc, xem xong feel good luôn"),
             ("t2", "xia xìa nhé bạn hiền")]
    try:
        bk = make_backend(backend, **kw)
    except Exception as e:
        print(f"[LỖI khi khởi tạo] {type(e).__name__}: {e}")
        return None
    try:
        raw = bk(build_user_prompt(batch))
    except Exception as e:
        print(f"[LỖI khi gọi] {type(e).__name__}")
        print(f"{e}")
        return None
    if verbose:
        print("--- output thô ---")
        print(raw[:1200])
    try:
        parsed = _extract_json(raw)
        print(f"\n[OK] parse được {len(parsed)} phần tử")
        for o in parsed:
            print(f"  has_cs={o.get('has_cs')} langs={o.get('langs')} "
                  f"tokens={o.get('tokens')}")
        return parsed
    except Exception as e:
        print(f"\n[LỖI parse] {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# 4. Vòng gán nhãn có cache
# ---------------------------------------------------------------------------
def annotate(df, backend="gemini", cache="cs_llm_cache.jsonl", splits=("val", "test"),
             text_col="text", id_col="id", batch_size=15, sleep=0.0,
             max_retry=3, tag=None, limit=None, **backend_kw):
    """Gán nhãn code-switching bằng LLM, ghi từng dòng vào cache JSONL.

    Chạy lại sẽ bỏ qua các id đã có trong cache -> an toàn khi Kaggle ngắt.
    """
    cache = Path(cache)
    tag = tag or backend
    done = set()
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
                if o.get("_tag") == tag:
                    done.add(str(o["id"]))
            except Exception:
                pass
        print(f"[cache] đã có {len(done)} câu cho tag='{tag}'")

    sub = df[df["split"].isin(splits)] if splits else df
    todo = [(str(r[id_col]), str(r[text_col])) for _, r in sub.iterrows()
            if str(r[id_col]) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[i] cần gán {len(todo)}/{len(sub)} câu | batch={batch_size} "
          f"-> {(len(todo)+batch_size-1)//batch_size} lượt gọi")
    if not todo:
        return cache

    bk = make_backend(backend, **backend_kw)
    n_ok = n_fail = 0
    t0 = time.time()

    with cache.open("a", encoding="utf-8") as f:
        for bi in range(0, len(todo), batch_size):
            batch = todo[bi:bi + batch_size]
            prompt = build_user_prompt(batch)
            parsed = None
            for attempt in range(max_retry):
                try:
                    parsed = _extract_json(bk(prompt))
                    if isinstance(parsed, dict):
                        parsed = parsed.get("results") or parsed.get("data") or [parsed]
                    if len(parsed) != len(batch):
                        raise ValueError(f"trả về {len(parsed)} phần tử, cần {len(batch)}")
                    break
                except Exception as e:
                    if attempt == max_retry - 1:
                        print(f"    [fail] batch {bi//batch_size}: {type(e).__name__}: {str(e)[:90]}")
                        parsed = None
                    else:
                        time.sleep(2 ** attempt)

            if parsed is None:
                n_fail += len(batch)
                continue

            by_id = {str(o.get("id")): o for o in parsed}
            for sid, stext in batch:
                o = by_id.get(sid)
                if o is None:                      # LLM đổi id -> khớp theo thứ tự
                    idx = [i for i, (s, _) in enumerate(batch) if s == sid]
                    o = parsed[idx[0]] if idx and idx[0] < len(parsed) else {}
                rec = {"id": sid, "text": stext, "_tag": tag,
                       "has_cs": bool(o.get("has_cs", False)),
                       "langs": o.get("langs", []),
                       "tokens": o.get("tokens", []),
                       "confidence": float(o.get("confidence", 0.0) or 0.0)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
            f.flush()

            if (bi // batch_size) % 10 == 0:
                el = time.time() - t0
                print(f"    {n_ok}/{len(todo)} câu | {el:.0f}s "
                      f"| còn ~{el/max(n_ok,1)*(len(todo)-n_ok):.0f}s")
            if sleep:
                time.sleep(sleep)

    print(f"[xong] {n_ok} câu OK, {n_fail} lỗi -> {cache}")
    return cache


# ---------------------------------------------------------------------------
# 5. Ghép kết quả LLM vào DataFrame theo lược đồ cũ
# ---------------------------------------------------------------------------
CS_TYPES_STRICT = {"english", "chinese_script", "chinese_translit", "other_foreign"}


CACHE_COLS = ["id", "text", "_tag", "has_cs", "langs", "tokens", "confidence"]


def load_cache(cache="cs_llm_cache.jsonl", tag=None):
    p = Path(cache)
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
            except Exception:
                continue
            if tag and o.get("_tag") != tag:
                continue
            rows.append(o)
    if not rows:
        print(f"[!] cache rỗng: {p}  — chưa có câu nào được gán nhãn thành công")
        return pd.DataFrame(columns=CACHE_COLS)
    d = pd.DataFrame(rows)
    for c in CACHE_COLS:
        if c not in d.columns:
            d[c] = None
    return d.drop_duplicates(subset=["id", "_tag"], keep="last")


def build_subsets(df, cache="cs_llm_cache.jsonl", tag=None, id_col="id",
                  count_loanword=False, heavy_min=3, heavy_ratio=0.25,
                  min_confidence=0.0, keep_rule_cols=True):
    """Ghép nhãn LLM vào df, sinh đúng các cột mà pipeline đánh giá đang dùng.

    count_loanword: có tính từ mượn đã Việt hoá (ship, like, video) là chuyển mã không.
    min_confidence: bỏ qua nhãn has_cs có độ tin cậy dưới ngưỡng.
    """
    llm = load_cache(cache, tag)
    if llm.empty:
        raise ValueError("Cache rỗng.")
    llm["id"] = llm["id"].astype(str)

    def summarise(row):
        toks = row.get("tokens") or []
        by = {}
        for t in toks:
            by.setdefault(t.get("type", "?"), []).append(str(t.get("text", "")))
        keep = set(CS_TYPES_STRICT) | ({"loanword_naturalized"} if count_loanword else set())
        cs_tokens = [t for k, v in by.items() if k in keep for t in v]
        conf = float(row.get("confidence") or 0)
        has = bool(row.get("has_cs")) and conf >= min_confidence
        if count_loanword:
            has = has or bool(by.get("loanword_naturalized"))
        has = has and len(cs_tokens) > 0
        return pd.Series({
            "llm_has_cs": has,
            "llm_confidence": conf,
            "llm_langs": ",".join(row.get("langs") or []),
            "n_english": len(by.get("english", [])),
            "n_chinese": len(by.get("chinese_script", [])) + len(by.get("chinese_translit", [])),
            "n_translit": len(by.get("chinese_translit", [])) + len(by.get("other_foreign", [])),
            "n_loanword": len(by.get("loanword_naturalized", [])),
            "n_proper_noun": len(by.get("proper_noun", [])),
            "n_cs_tokens": len(cs_tokens),
            "cs_tokens": " | ".join(cs_tokens[:20]),
            "english_tokens": " ".join(by.get("english", [])[:20]),
            "chinese_tokens": " ".join((by.get("chinese_script", []) +
                                        by.get("chinese_translit", []))[:20]),
        })

    feats = llm.apply(summarise, axis=1)
    llm = pd.concat([llm[["id"]], feats], axis=1)

    out = df.copy()
    out[id_col] = out[id_col].astype(str)
    drop = [c for c in llm.columns if c != "id" and c in out.columns]
    out = out.drop(columns=drop).merge(llm, left_on=id_col, right_on="id", how="left")
    if "id" in out.columns and id_col != "id":
        out = out.drop(columns=["id"])

    miss = out["llm_has_cs"].isna().sum()
    if miss:
        print(f"[!] {miss} dòng chưa có nhãn LLM -> coi như không chuyển mã")
    out["llm_has_cs"] = out["llm_has_cs"].fillna(False).astype(bool)
    for c in ["n_english", "n_chinese", "n_translit", "n_loanword", "n_cs_tokens"]:
        out[c] = out[c].fillna(0).astype(int)

    # --- cột tương thích với eval_on_subsets.py ---
    n_words = out.get("n_word_tokens")
    if n_words is None:
        n_words = out["text"].astype(str).str.split().str.len().clip(lower=1)
    ratio = out["n_cs_tokens"] / n_words.clip(lower=1)
    out["english_ratio"] = ratio

    out["is_cs_strict"] = out["llm_has_cs"]
    has_teen = out["has_teencode"] if "has_teencode" in out.columns else False
    out["is_cs_broad"] = out["is_cs_strict"] | (has_teen if isinstance(has_teen, pd.Series) else False)

    has_emoji = out["has_emoji"] if "has_emoji" in out.columns else pd.Series(False, index=out.index)
    has_noise = out["has_noise"] if "has_noise" in out.columns else pd.Series(False, index=out.index)
    teen = has_teen if isinstance(has_teen, pd.Series) else pd.Series(False, index=out.index)

    grp = pd.Series("pure_vi", index=out.index)
    grp[has_noise.fillna(False)] = "other_noise"
    grp[has_emoji.fillna(False)] = "emoji_only"
    grp[teen.fillna(False)] = "teencode_slang"
    grp[out["is_cs_strict"]] = "english_mixed"
    grp[out["is_cs_strict"] & (out["n_chinese"] > 0)] = "chinese_mixed"
    out["cs_group"] = grp

    lvl = pd.Series("none", index=out.index)
    lvl[out["is_cs_strict"]] = "light"
    lvl[out["is_cs_strict"] & ((out["n_cs_tokens"] >= heavy_min) | (ratio >= heavy_ratio))] = "heavy"
    out["cs_level"] = lvl

    print("\ncs_group:", out["cs_group"].value_counts().to_dict())
    print("cs_level:", out["cs_level"].value_counts().to_dict())
    for sp, g in out.groupby("split"):
        print(f"  {sp:5s} n={len(g):5d}  cs_strict={int(g.is_cs_strict.sum()):5d} "
              f"({100*g.is_cs_strict.mean():.1f}%)  "
              f"trong đó có tiếng Trung: {int((g.n_chinese>0).sum())}")
    return out


def make_control(df, seed=42):
    """Nhóm đối chứng pure_vi cân bằng cỡ mẫu và nhãn với cs_strict, theo split."""
    import numpy as np
    rng = np.random.default_rng(seed)
    keep = pd.Series(False, index=df.index)
    key = "primary_label" if "primary_label" in df.columns else None
    for _, part in df.groupby("split"):
        cs = part[part.is_cs_strict]
        pure = part[part.cs_group == "pure_vi"]
        if len(cs) == 0 or len(pure) == 0:
            continue
        n = min(len(cs), len(pure))
        chosen = []
        if key:
            for lab, frac in cs[key].value_counts(normalize=True).items():
                pool = pure.index[pure[key] == lab].to_numpy()
                k = min(int(round(frac * n)), len(pool))
                if k:
                    chosen += rng.choice(pool, k, replace=False).tolist()
        rest = np.setdiff1d(pure.index.to_numpy(), np.array(chosen, dtype=pure.index.dtype))
        if len(chosen) < n and len(rest):
            chosen += rng.choice(rest, min(n - len(chosen), len(rest)), replace=False).tolist()
        keep.loc[chosen] = True
    df = df.copy()
    df["control_pure_vi"] = keep
    print(f"control_pure_vi: {int(keep.sum())} mẫu")
    return df


# ---------------------------------------------------------------------------
# 6. Xuất file
# ---------------------------------------------------------------------------
def export_subsets(df, out_dir="/kaggle/working/cs_llm", splits=("val", "test"),
                   formats=("csv",), keep_cols=None):
    """Tách ra file riêng cho từng (subset × split).

    Cấu trúc:
        out_dir/annotations/vigo_cs_llm.csv     <- file dùng cho eval_on_subsets.py
        out_dir/subsets/<subset>/<split>.csv    <- file rời để xem/nộp
        out_dir/subset_index.csv
    """
    out_dir = Path(out_dir)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    ann = out_dir / "annotations" / "vigo_cs_llm.csv"
    df.to_csv(ann, index=False)
    print(f"[annotations] {ann}")

    masks = {
        "all": pd.Series(True, index=df.index),
        "pure_vi": df["cs_group"].eq("pure_vi"),
        "cs_strict": df["is_cs_strict"].astype(bool),
        "english_mixed": df["cs_group"].eq("english_mixed"),
        "chinese_mixed": df["cs_group"].eq("chinese_mixed"),
        "teencode_slang": df["cs_group"].eq("teencode_slang"),
        "emoji_only": df["cs_group"].eq("emoji_only"),
        "other_noise": df["cs_group"].eq("other_noise"),
        "cs_heavy": df["cs_level"].eq("heavy"),
        "cs_light": df["cs_level"].eq("light"),
    }
    if "is_cs_broad" in df.columns:
        masks["cs_broad"] = df["is_cs_broad"].astype(bool)
    if "control_pure_vi" in df.columns:
        masks["control_pure_vi"] = df["control_pure_vi"].astype(bool)
    if "n_translit" in df.columns:
        masks["translit_only"] = df["is_cs_strict"] & df["n_translit"].gt(0)

    cols = keep_cols or [c for c in df.columns if not c.startswith("label_")]
    rows = []
    for name, m in masks.items():
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
            rows.append({"subset": name, "split": sp, "n": len(part),
                         "pct_split": round(100 * len(part) / max(df["split"].eq(sp).sum(), 1), 2)})

    idx = pd.DataFrame(rows)
    idx.to_csv(out_dir / "subset_index.csv", index=False)
    print()
    print(idx.pivot(index="subset", columns="split", values="n").fillna(0).astype(int).to_string())
    print(f"\n-> {out_dir}")
    return idx


# ---------------------------------------------------------------------------
# 7. Kiểm chứng
# ---------------------------------------------------------------------------
def agreement(df, rule_col="is_cs_strict_rule", llm_col="is_cs_strict"):
    """So nhãn LLM với nhãn bộ luật. Cần cột nhãn bộ luật đã đổi tên."""
    if rule_col not in df.columns:
        print(f"[!] không có cột '{rule_col}'. Trước khi build_subsets hãy chạy:\n"
              "    df['is_cs_strict_rule'] = df['is_cs_strict']")
        return None
    a, b = df[rule_col].astype(bool), df[llm_col].astype(bool)
    ct = pd.crosstab(a, b, rownames=["luật"], colnames=["LLM"])
    print(ct.to_string())
    agree = (a == b).mean()
    po, pe = agree, (a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    print(f"\nđồng thuận {100*agree:.1f}% | Cohen kappa {(po-pe)/(1-pe):.3f}")
    print(f"luật bắt {a.sum()} | LLM bắt {b.sum()} | chỉ LLM bắt {int((~a & b).sum())} "
          f"| chỉ luật bắt {int((a & ~b).sum())}")
    return ct


def agreement_two_llms(cache="cs_llm_cache.jsonl", tag_a="gemini", tag_b="local"):
    """Đo mức đồng thuận giữa hai LLM — con số về độ tin cậy của nhãn."""
    A = load_cache(cache, tag_a)[["id", "has_cs"]].rename(columns={"has_cs": "a"})
    B = load_cache(cache, tag_b)[["id", "has_cs"]].rename(columns={"has_cs": "b"})
    m = A.merge(B, on="id")
    if m.empty:
        print("[!] không có id chung")
        return None
    a, b = m.a.astype(bool), m.b.astype(bool)
    print(pd.crosstab(a, b, rownames=[tag_a], colnames=[tag_b]).to_string())
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    print(f"\nn={len(m)} | đồng thuận {100*po:.1f}% | kappa {(po-pe)/(1-pe):.3f}")
    return m


def export_for_review(df, path="llm_review_sample.csv", n=200, seed=42):
    """Mẫu phân tầng để chấm tay — bắt buộc có trước khi báo cáo."""
    cols = [c for c in ["id", "split", "text", "cs_group", "cs_level", "llm_langs",
                        "cs_tokens", "llm_confidence", "n_english", "n_chinese",
                        "n_translit", "n_loanword"] if c in df.columns]
    parts = []
    for _, g in df.groupby("cs_group"):
        parts.append(g.sample(min(n // max(df.cs_group.nunique(), 1), len(g)),
                              random_state=seed)[cols])
    low = df[df.llm_confidence.between(0.01, 0.7)]
    if len(low):
        parts.append(low.sample(min(40, len(low)), random_state=seed)[cols])
    s = pd.concat(parts).drop_duplicates(subset=["id"]).sample(frac=1, random_state=seed)
    s["gold_has_cs"] = ""
    s["gold_lang"] = ""
    s["note"] = ""
    s.to_csv(path, index=False)
    print(f"[OK] {len(s)} dòng -> {path}  (điền cột gold_has_cs bằng 1/0)")
    return s


def score_review(path="llm_review_sample.csv"):
    """Tính precision/recall của LLM sau khi bạn chấm tay."""
    d = pd.read_csv(path)
    d = d[d.gold_has_cs.astype(str).str.strip().isin(["0", "1"])]
    if d.empty:
        print("[!] chưa có dòng nào được chấm")
        return None
    gold = d.gold_has_cs.astype(int).astype(bool)
    pred = d.cs_group.isin(["english_mixed", "chinese_mixed"])
    tp = int((gold & pred).sum()); fp = int((~gold & pred).sum()); fn = int((gold & ~pred).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    print(f"n={len(d)}  TP={tp} FP={fp} FN={fn}")
    print(f"precision {P:.3f} | recall {R:.3f} | F1 {2*P*R/max(P+R,1e-9):.3f}")
    return dict(n=len(d), precision=P, recall=R)
