"""
cm_rules.py — PHÁT HIỆN CODE-MIXED VIỆT–ANH–TRUNG BẰNG LUẬT
============================================================
Thay thế hoàn toàn tầng LLM (`cs_llm_detector.py`) bằng luật + thư viện.
Đầu ra giữ nguyên lược đồ cột mà `cm_evaluate.py` đang dùng, nên phần đánh giá
mô hình không phải sửa một dòng nào.

Vì sao bỏ LLM
-------------
LLM gán nhãn ở mức token nhưng bịa ra token không có trong câu (~23,5%), và mắc
đúng ba lỗi precision đã được cảnh báo ngay trong prompt: từ Việt không dấu bị
gán là tiếng Anh, tên riêng tiếng Việt bị gán là tiếng Anh, và từ Hán-Việt bị
gán là tiếng Trung. Một nhãn không tái lập được và không kiểm chứng được từng
bước thì không dùng để chia tập con báo cáo được.

Bộ luật này đổi lại: mọi quyết định đều truy nguyên được về một bằng chứng cụ
thể (`Span.evidence`), chạy lại cho ra kết quả y hệt, và `explain()` in ra bảng
lý do cho từng token.

Khác gì `cs_detector.py` (bản cũ)
---------------------------------
1.  **Âm tiết tiếng Việt kiểm bằng bảng phụ âm đầu × vần thật**, không phải regex
    `[aeiouy]{1,3}`. Bản cũ nhận `get`, `care`, `keep`, `seen` là "có thể là âm
    tiết Việt" vì chỉ soi nguyên âm sau khi bỏ dấu; bảng vần thật loại chúng ngay
    (`g` không đứng trước `e`, `-are/-eep/-een` không phải vần tiếng Việt).
    Có thêm luật thanh điệu: âm tiết đóng bằng p/t/c/ch chỉ nhận sắc và nặng.
2.  **Quyết định bằng bằng chứng dương, không đoán.** Token không phải âm tiết
    Việt mà cũng không có trong từ điển Anh (`bruhh`, `sksksk`, tên riêng lạ)
    KHÔNG còn được tính là chuyển mã như `count_guess_as_english=True` của bản cũ
    — đây là nguồn false positive lớn nhất. Chúng vào cột `unknown_tokens` riêng.
3.  **Tần suất từ thay cho danh sách thủ công.** Với token vừa hợp lệ tiếng Việt
    vừa có trong từ điển Anh (`hot`, `man`, `tin`, `sang`, `ban`), quyết định dựa
    trên chênh lệch Zipf giữa tiếng Anh và tiếng Việt, cộng ngữ cảnh câu — thay
    cho `EN_BLACKLIST` / `EN_LOAN_STRONG` viết tay.
4.  **Khớp cụm nhiều từ.** `feel good`, `check in`, `red flag`, `long time no see`
    được bắt trọn cụm; bản cũ chỉ xét từng token rời.
5.  **Tiếng Trung thật sự được xử lý.** Chữ Hán bắt bằng script. Phiên âm sang chữ
    Việt (`xia xìa`, `nỉ hảo`, `pà pà`, `chia du`) khớp qua khung âm vị sinh tự
    động từ một từ điển khai báo bằng CHỮ HÁN, chặn cứng bằng danh sách Hán-Việt.
6.  **Mỗi nhãn có độ tin cậy và loại bằng chứng**, lọc được bằng `min_confidence`
    mà không phải chạy lại.

Phụ thuộc
---------
Lõi chỉ cần Python chuẩn + pandas. `wordfreq` là tuỳ chọn nhưng NÊN CÀI — thiếu
nó thì tầng khử nhập nhằng theo tần suất tắt và detector lùi về từ điển thủ công
(kém hơn hẳn). `pypinyin` tuỳ chọn, chỉ dùng để tự sinh pinyin khi bạn thêm mục
mới vào từ điển chữ Hán. `cs_detector.py` nếu nằm cùng thư mục sẽ được mượn lại
từ điển tiếng Anh và teencode để hai module không lệch nhau.

    pip install wordfreq

Cách dùng
---------
    import cm_rules as R

    R.env_report()          # xem đang chạy ở chế độ đầy đủ hay rút gọn
    R.demo()                # chạy thử toàn bộ, in bảng lý do vài câu mẫu
    R.run_tests()           # 34 ca đối chứng, gồm 3 lớp lỗi LLM từng mắc
    R.explain("Cái clip này hay vc, xem xong feel good luôn")   # soi 1 câu

    df = R.build_subsets(df)     # detector + mọi cột cm_evaluate.py cần
    df = R.make_control(df)      # nhóm đối chứng pure_vi cân bằng nhãn
    df.to_csv("annotations.csv", index=False)

Thay thẳng bước LLM trong `cm_detect.py` (prepare() vẫn dùng như cũ):

    import cm_detect as D, cm_rules as R
    D.RAW_DATA = "/kaggle/input/<dataset>"
    D.prepare()                                   # thô -> text_s1/s2/s3 + nhãn
    df = R.make_control(R.build_subsets(D._prepared))
    df.to_csv(D.OUT_DIR / "annotations/annotations.csv", index=False)

Trước khi báo cáo
-----------------
1.  `R.compare_with_cs_detector(df)` — bảng đối chiếu với bộ luật cũ, để nói
    được "chặt hơn ở chỗ nào, bao nhiêu câu đổi nhãn".
2.  `R.export_for_review(df, n=200)` -> chấm tay cột `gold_has_cs` ->
    `R.score_review()`. Bộ luật vẫn là heuristic, luận văn cần một con số
    precision/recall thật.
3.  `R.top_tokens(df)` và cột `unknown_tokens` — hai nơi soi false positive
    nhanh nhất. Từ nào nhiễu thì thêm vào `extra_stopwords` khi khởi tạo
    detector, đừng sửa thẳng từ điển trong file.

Các nhóm `cs_group`
-------------------
    pure_vi | english_mixed | chinese_mixed | other_foreign_mixed
    teencode_slang | emoji_only | other_noise

`other_foreign_mixed` (Hàn / Nhật) là nhóm riêng, không gộp vào english_mixed
như pipeline LLM cũ. Nó vẫn nằm trong `cs_strict`.

Muốn thêm tier "rộng" gồm cả token chưa xác định, không phải chạy lại:

    E.register("cs_loose", lambda d: d.is_cs_strict | d.n_unknown.gt(0))
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import pandas as pd
except ImportError:                                    # pandas chỉ cần cho API DataFrame
    pd = None


# ===========================================================================
# 0. Phụ thuộc tuỳ chọn
# ===========================================================================

try:
    from wordfreq import zipf_frequency as _zipf
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False

    def _zipf(word, lang):                             # type: ignore
        return 0.0

try:
    from pypinyin import lazy_pinyin as _lazy_pinyin
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    _lazy_pinyin = None


def zipf(word: str, lang: str) -> float:
    """Zipf 0–8. Thiếu wordfreq thì trả 0.0 và mọi luật dựa trên nó tự tắt."""
    if not HAS_WORDFREQ:
        return 0.0
    try:
        return float(_zipf(word, lang))
    except Exception:
        return 0.0


def env_report() -> dict:
    """In tình trạng phụ thuộc. Chạy đầu notebook để biết detector đang ở chế độ nào."""
    info = {
        "wordfreq": HAS_WORDFREQ,
        "pypinyin": HAS_PYPINYIN,
        "cs_detector": False,
    }
    try:
        import cs_detector          # noqa: F401
        info["cs_detector"] = True
    except Exception:
        pass
    print("[cm_rules] phụ thuộc:")
    for k, v in info.items():
        print(f"    {k:12s} {'có' if v else 'KHÔNG'}")
    if not HAS_WORDFREQ:
        print("    [!] thiếu wordfreq -> tầng khử nhập nhằng theo tần suất TẮT.\n"
              "        Cài bằng: pip install wordfreq  (nên có trước khi chốt số liệu)")
    return info


# ===========================================================================
# 1. Unicode, tokenizer
# ===========================================================================

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[\w.]+")
HASHTAG_RE = re.compile(r"(?<!\w)#[\w_]+")
HTML_TAG_RE = re.compile(r"<[^>]{1,15}>")

HAN_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
KANA_RE = re.compile(r"[぀-ヿ]")
HANGUL_RE = re.compile(r"[가-힯ᄀ-ᇿ]")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
THAI_RE = re.compile(r"[฀-๿]")

SCRIPT_RES = {
    "chinese_script": HAN_RE,
    "japanese": KANA_RE,
    "korean": HANGUL_RE,
    "cyrillic": CYRILLIC_RE,
    "thai": THAI_RE,
}

EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002190-\U000021FF"
    "\U00002300-\U000023FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)

VI_LETTERS = "A-Za-zÀ-ỹĐđ"
TOKEN_RE = re.compile(
    rf"[{VI_LETTERS}]+(?:['’\-_][{VI_LETTERS}]+)*"      # từ (có dấu tiếng Việt)
    r"|[一-鿿㐀-䶿]+"                    # cụm chữ Hán
    r"|[぀-ヿ]+|[가-힯]+"                # kana / hangul
    r"|\d+(?:[.,]\d+)*"                                  # số
    r"|[^\w\s]",                                         # dấu câu
    re.UNICODE,
)

VIET_DIACRITIC_RE = re.compile(
    r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộ"
    r"ớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.IGNORECASE)

ASCII_ALPHA_RE = re.compile(r"^[a-z]+$")
ELONGATION_RE = re.compile(r"(.)\1{2,}")

LAUGH_RE = re.compile(
    r"^(?:"
    r"(?:h[aeiouyăâêôơư]{1,2}){2,}"
    r"|(?:hj){2,}|(?:jz){2,}|(?:ck){2,}"
    r"|k{2,}|z{3,}|w+k+w*|hoho|huhu"
    r"|lol|lolz|lmao|lmfao|rofl|jaja|khkh|khjkhj|xd+"
    r")$",
    re.IGNORECASE,
)

TONE_CODEPOINTS = {0x0301: "sac", 0x0300: "huyen", 0x0309: "hoi",
                   0x0303: "nga", 0x0323: "nang"}
QUALITY_CODEPOINTS = {0x0306, 0x0302, 0x031B}          # breve, circumflex, horn


def strip_diacritics(text: str) -> str:
    """Bỏ toàn bộ dấu tiếng Việt, đ -> d."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def split_tone(word: str) -> Tuple[str, Optional[str], bool]:
    """Tách thanh điệu ra khỏi âm tiết.

    Trả về (âm tiết không thanh nhưng GIỮ dấu chất lượng ă â ê ô ơ ư,
            tên thanh hoặc None nếu thanh ngang,
            có dùng dấu tiếng Việt hay không).
    """
    nfd = unicodedata.normalize("NFD", word)
    tones, base, has_dia = set(), [], "đ" in word.lower()
    for ch in nfd:
        cp = ord(ch)
        if cp in TONE_CODEPOINTS:
            tones.add(TONE_CODEPOINTS[cp])
            has_dia = True
        else:
            if cp in QUALITY_CODEPOINTS:
                has_dia = True
            base.append(ch)
    if len(tones) > 1:
        return unicodedata.normalize("NFC", "".join(base)), "__nhiều__", has_dia
    return (unicodedata.normalize("NFC", "".join(base)),
            next(iter(tones)) if tones else None, has_dia)


def normalize_elongation(token: str) -> List[str]:
    """đẹppppp -> [đẹppppp, đẹpp, đẹp]; goooood -> [..., good]. Ứng viên tra từ điển.

    Chỉ rút gọn khi có chuỗi lặp TỪ 3 KÝ TỰ trở lên. Rút gọn cả cặp đôi bình
    thường sẽ hỏng: "mood" -> "mod" (trúng teencode), "good" -> "god",
    "seen" -> "sen" — đúng kiểu lỗi âm thầm rất khó tìm.
    """
    out = [token]
    if ELONGATION_RE.search(token):
        two = ELONGATION_RE.sub(r"\1\1", token)
        if two not in out:
            out.append(two)
        one = re.sub(r"(.)\1+", r"\1", token)
        if one not in out:
            out.append(one)
    return out


# ===========================================================================
# 2. Âm vị học tiếng Việt — bảng phụ âm đầu × vần
# ===========================================================================
# Đây là điểm khác biệt lớn nhất so với `cs_detector.is_vietnamese_syllable`.
# Bản cũ chỉ kiểm tra "có nguyên âm nào đó ở giữa" sau khi bỏ dấu, nên nhận nhầm
# rất nhiều từ tiếng Anh. Bảng dưới đây là vần tiếng Việt thật; một chuỗi chỉ hợp
# lệ khi tách được thành phụ âm đầu hợp lệ + vần có trong bảng, thoả ràng buộc
# chính tả và luật thanh điệu.

VN_ONSETS = [
    "ngh", "ng", "nh", "ch", "gh", "gi", "kh", "ph", "th", "tr", "qu",
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x",
]
VN_ONSETS_SORTED = sorted(VN_ONSETS, key=len, reverse=True)

_RIME_TEXT = """
a ac ach ai am an ang anh ao ap at au ay
ăc ăm ăn ăng ăp ăt
âc âm ân âng âp ât âu ây
e ec em en eng eo ep et
ê êch êm ên ênh êp êt êu
i ich im in inh ip it iu
ia iêc iêm iên iêng iêp iêt iêu
o oc oi om on ong op ot oong ooc
ô ôc ôi ôm ôn ông ôp ôt
ơ ơi ơm ơn ơp ơt
u uc ui um un ung up ut
ư ưc ưi ưm ưn ưng ưt ưu
ưa ươc ươi ươm ươn ương ươp ươt ươu
ua uôc uôi uôm uôn uông uôt
y ya yêm yên yêng yêu
oa oac oach oai oam oan oang oanh oao oap oat oay
oăc oăm oăn oăng oăp oăt
oe oem oen oeo oet
uân uâng uât uây
uê uêch uên uênh uêt
uy uya uych uyn uynh uyp uyt uyu uyên uyêt
uơ
"""
VN_RIMES = set(_RIME_TEXT.split())
# Thêm dạng bỏ dấu để nhận được tiếng Việt gõ không dấu ("khong", "duoc", "nguoi").
VN_RIMES_ALL = VN_RIMES | {strip_diacritics(r) for r in VN_RIMES}

_FRONT = tuple("eêiy")                 # vần bắt đầu bằng nguyên âm hàng trước
_STOP_CODAS = ("ch", "p", "t", "c")    # âm cuối tắc -> chỉ sắc / nặng
# Phụ âm đầu "p" trong tiếng Việt chỉ có ở từ mượn; không mở rộng tự do, nếu
# không "pop", "pin", "pat" đều thành "âm tiết Việt hợp lệ".
_P_ONSET_OK = {"pin", "pa", "pô", "pê", "pi", "pó", "pò", "pập", "phở"}

_vi_syllable_cache: Dict[str, bool] = {}


def _onset_rime_ok(onset: str, rime: str) -> bool:
    """Ràng buộc chính tả quốc ngữ giữa phụ âm đầu và vần."""
    front = rime.startswith(_FRONT)
    if onset in ("k", "gh", "ngh"):
        return front
    if onset in ("c", "g", "ng"):
        return not front
    if onset == "qu":
        return not rime.startswith(("u", "o"))
    return True


def is_vi_syllable(token: str) -> bool:
    """Token có phải MỘT âm tiết tiếng Việt viết đúng chính tả không?

    Chấp nhận cả dạng gõ không dấu. Không cần từ điển: dựa hoàn toàn vào cấu trúc
    âm tiết + ràng buộc chính tả + luật thanh điệu.
    """
    key = token.lower()
    cached = _vi_syllable_cache.get(key)
    if cached is not None:
        return cached

    result = False
    base, tone, has_dia = split_tone(key)
    if tone != "__nhiều__" and base and re.fullmatch(r"[a-zà-ỹđ]+", base):
        for onset in [""] + VN_ONSETS_SORTED:
            if onset and not base.startswith(onset):
                continue
            rime = base[len(onset):]
            # "gì", "gỉ": phụ âm đầu gi + vần i viết dính thành một chữ i
            if onset == "gi" and rime == "":
                rime = "i"
            if rime not in VN_RIMES_ALL or not _onset_rime_ok(onset, rime):
                continue
            if onset == "p" and key not in _P_ONSET_OK:
                continue
            # luật thanh: âm tiết đóng bằng p/t/c/ch chỉ đi với sắc hoặc nặng.
            # Chỉ áp dụng khi người viết CÓ dùng dấu — văn bản gõ không dấu thì
            # thanh ngang không nói lên điều gì.
            if has_dia and rime.endswith(_STOP_CODAS) and tone not in ("sac", "nang"):
                continue
            result = True
            break

    _vi_syllable_cache[key] = result
    return result


def vi_syllable_variants(token: str) -> bool:
    """is_vi_syllable, có thử cả dạng đã rút gọn ký tự kéo dài."""
    return any(is_vi_syllable(c) for c in normalize_elongation(token.lower()))


# ===========================================================================
# 3. Từ điển tiếng Việt / tiếng Anh / teencode
# ===========================================================================

# Mượn lại từ điển của cs_detector.py để hai module không định nghĩa lệch nhau.
_EN_CORE_FALLBACK = set("""
about after again all already also always amazing and another any anyone anything
are baby back bad beautiful because best better big boring boy bro business busy
call can care check cheap clean clear click close club come comment community
company complete confuse control cool copy could crazy create crush cute daily
dark data date deadline deal dear deep delete design different difficult digital
dislike done down download dream drop each early easy edit enjoy enough episode
even ever every everyone everything exactly excited expensive experience explain
face fact fail fake family famous fan fashion fast feel feeling few fight file
final finally find fine finish first fix flex focus follow food for free friend
from full fun funny future game gaming get gift girl give glad global goal good
great group grow guy handsome happen happy hard hate have heart help here high
history hit hobby hold home hope horror hot hour how however huge human idea idol
ignore image imagine important impossible improve include increase indeed info
inside inspire install instead interest interesting internet interview into
invite issue item job join joke joy just keep key kid kind know knowledge lack
language laptop large last late later laugh launch law lazy lead leader learn
leave left legend less lesson let level life light like limit line link list
listen little live local lock lonely long look lose loss lost lot loud love
lovely low loyal luck lucky make manage many market master match material matter
maybe mean meaning media medium meet meeting member memory mention menu message
method might mind mine minute miss mistake mix mobile model modern moment money
mood more morning most move movie much music must mystery name native natural
nature near need negative network never new news next nice night noise none
normal note nothing notice now number offer office official often okay old online
only open opinion option order other outside over own page pain pair part party
pass passion past pattern pause pay peace people perfect perform performance
perhaps period person personal perspective phone photo pick picture piece place
plan play player please pleasure plus point poor pop popular position positive
possible post power practice prefer premium prepare present press pressure pretty
prevent previous price pride print prior priority private prize probably problem
process produce product professional profile profit program progress project
promise promote proof proper property propose protect proud prove provide public
publish pull pure purpose push put quality question quick quiet quit quite quote
random range rank rap rapid rare rate rather reach react read reader ready real
reality realize really reason receive recent recipe recognize recommend record
recover reduce refer reflect refuse regard regret regular reject relate
relationship relax release relevant relief rely remain remember remind remote
remove repair repeat replace reply report request require research reserve
resource respect respond response responsible rest restaurant result retail
return reveal review rich ride right ring rise risk road robot role roll romance
romantic room root rough round route routine rule run running rush sad safe
safety salary sale same sample satisfy save say scale scary scene schedule school
science score screen script search season seat second secret section secure
security seed seek seem select self sell send senior sense sensitive sentence
separate series serious serve service session set setting settle several severe
shadow shake shame shape share sharp sheet shift shine ship shirt shock shoe
shoot shop shopping short shot should shoulder shout show shower shut shy sick
side sight sign signal significant silent silver similar simple simply since sing
singer single sink sir sister sit site situation size skill skin sky sleep slide
slow small smart smell smile smoke snap social society soft software solid
solution solve some someone something sometimes somewhere song soon sorry sort
soul sound source space speak special species specific speech speed spend spirit
split sponsor sport spot spread spring square stable staff stage stand standard
star start state statement station status stay step stick still stock stop store
storm story straight strange strategy stream street strength stress stretch
strike strong structure struggle student studio study stuff stupid style subject
submit subscribe success successful such sudden suffer suggest suit summer sun
super supply support suppose sure surface surprise survey survive sweet swim
switch symbol system table take talent talk tall target task taste teach teacher
team tech technology teen teenager tell term terrible test text thank thanks that
their them theme then theory there these they thick thin thing think third this
those though thought threat three through throw thus ticket tie tight time tiny
tip tired title today together tomorrow tone tonight too tool top topic total
touch tough tour tourist toward town toy track trade tradition traffic train
training transfer transform translate transport travel treat treatment tree trend
trending trial trick trip trophy trouble true trust truth try tune turn tutorial
twice twin type typical ugly ultimate unable under understand unfair unfollow
unhappy union unique unit universe university unknown unless until update upload
upon upset urban urge usage use used useful user usual vacation valid value
variety various version very victim victory video view viewer village violence
virtual virus visible vision visit visual vital vlog vocal voice volume volunteer
vote wait wake walk wall want war warm warn warning wash waste watch water wave
way weak wealth wear weather web website wedding week weekend weight weird
welcome welfare well what whatever when where whether which while white who whole
why wide wife wild will win wind window wine wing winner winter wisdom wise wish
with within without witness woman wonder wonderful word work worker working world
worry worse worst worth would wound wrap write writer writing wrong yeah year
yes yesterday yet you young your yourself youth youtube zero zone
""".split())

_TEENCODE_FALLBACK = set("""
ko k kh kg khg khum khong hok hong hem hp hnay dc đc dk đk dx đx dta j z dz zj
zi zì zậy zay vạy vay v vs vk ck r ùi ui rùi roi ròi lun mik mk mjk mìn mềnh mng
mn ae ny bff ib rep cmt cmts stt avt acc ad adm mod bt bth bthg qá qa wa wá cx cg
sml vcl vkl vl vc clm cmn cmnr đm dm đcm dcm vch tks thks plz pls nc ns nt nch đt
dt sdt cmnd cccd gato trl trlai hqua bgio bh bjo ntn nthe nthế nhma đky dky xl kb
kbiet tt ttinh qq qc xh xhoi ncl nchung ntnay ntnày bme cta mih cmnl mqh sem pải
đel nhữg thươg lqan lquan ngta nyc
""".split())


def _load_shared_lexicons() -> Tuple[Set[str], Set[str]]:
    """Ưu tiên dùng chung từ điển với cs_detector.py nếu import được."""
    try:
        import cs_detector as _cs
        en = set(_cs.EN_CORE) | set(_cs.EN_LOAN_STRONG)
        teen = set(_cs.TEENCODE)
        return en, teen
    except Exception:
        return set(_EN_CORE_FALLBACK), set(_TEENCODE_FALLBACK)


EN_LEXICON, TEENCODE = _load_shared_lexicons()

# Mở rộng từ điển tiếng Anh bằng wordfreq (top N từ). Chỉ dùng làm bằng chứng phụ:
# nhiều từ trong đó trùng âm tiết tiếng Việt nên vẫn phải qua tầng nhập nhằng.
def _load_wordfreq_lexicon(top_n: int = 30000) -> Set[str]:
    if not HAS_WORDFREQ:
        return set()
    try:
        from wordfreq import top_n_list
        return {w.lower() for w in top_n_list("en", top_n)
                if w.isalpha() and len(w) >= 3}
    except Exception:
        return set()


EN_LEXICON_WIDE = EN_LEXICON | _load_wordfreq_lexicon()

# Từ tiếng Anh ngắn / tiếng lóng mạng: luôn tính là tiếng Anh dù trùng dạng âm
# tiết Việt. Danh sách này CỐ TÌNH nhỏ — phần còn lại để tần suất quyết định.
EN_STRONG = set("""
bro sis omg wtf idk btw fyi tbh nvm thx gg af asap diy faq ceo pov bff goat
npc otp irl dm pm app bug fix dev meme vlog blog clip combo cover demo edit fail
feed flop hype menu mix noob quiz reel remix scam slay swag troll wifi crush flex
vibe vibes mood trend toxic fake deal seen team check feel drop hot top ship sale
share shop show stream style test view live like post reply review level list log
look note party pass peak plan play power pro rank rate real record save scan
score search sell sense sexy shot sign size skill smart soft solo sorry spam speed
sport staff star start stop story strong super sure sweet tag talk target taste
teen text thanks time tip tour track train trip trust try update upload user video
vip voice vote wall war warm watch web weekend welcome win wish work world young
deadline feedback outfit offline online comeback teamwork homework workshop
""".split())

# Cụm tiếng Anh nhiều từ. Khớp trọn cụm mạnh hơn hẳn xét từng token rời.
EN_PHRASES = [
    "feel good", "check in", "check out", "oh my god", "my god", "thank you",
    "thanks god", "so sad", "so cute", "so hot", "good job", "good night",
    "good morning", "good luck", "love you", "miss you", "no problem", "come on",
    "take care", "best friend", "long time no see", "next level", "no way",
    "one love", "out fit", "over think", "self love", "small talk", "too much",
    "true love", "win win", "you know", "i love you", "i miss you", "what the hell",
    "what the fuck", "oh no", "red flag", "green flag", "glow up", "plot twist",
    "deep talk", "hard core", "full time", "part time", "high light", "key word",
    "life style", "big deal", "big fan", "all in", "all star", "break up",
    "come back", "cut off", "make up", "work hard", "play hard", "on top",
    "let it go", "never mind", "of course", "by the way", "as soon as possible",
]

# Từ mượn đã Việt hoá hoàn toàn: liệt kê nhưng KHÔNG tính là chuyển mã (mặc định).
EN_LOANWORD_NATURALIZED = set("""
ok oke okay video internet email file laptop micro camera radio tivi
ga sếp phanh xăng lốp bơm săm bánh mì cà phê xà phòng
ship shipper like share comment inbox seen block acc nick web link app
pin sim card bill tour tip menu shopping mall taxi bus
""".split())

# Từ mượn đã Việt hoá thì không đồng thời là "bằng chứng tiếng Anh mạnh".
# Giao nhau giữa hai danh sách chỉ gây khó hiểu khi đọc lại luật.
EN_STRONG -= EN_LOANWORD_NATURALIZED

# Phiên âm tiếng Anh sang chữ Việt (bề mặt đã bỏ dấu).
EN_TRANSLIT = {
    "o ke": "ok", "oke": "ok", "o kei": "okay",
    "o mai got": "oh my god", "o mai gau": "oh my god",
    "gut chop": "good job", "sori": "sorry", "hen ry": "hungry",
    "so ry": "sorry", "sen kiu": "thank you", "ten kiu": "thank you",
    "bai bai": "bye bye", "hap pi": "happy",
}

# Ngôn ngữ khác (Hàn / Nhật) ở dạng phiên âm chữ Việt.
OTHER_FOREIGN_TRANSLIT = {
    "an nhon": "안녕 - xin chào", "an nhong": "안녕",
    "sa rang he": "사랑해 - anh yêu em", "sa rang hê": "사랑해",
    "o pa": "오빠 - anh", "op pa": "오빠", "on ni": "언니 - chị",
    "chin cha": "진짜 - thật à", "te bak": "대박 - đỉnh",
    "kam sa ham ni da": "감사합니다 - cảm ơn",
    "ka wa i": "かわいい - dễ thương", "a ri ga to": "ありがとう - cảm ơn",
    "sugoi": "すごい - tuyệt", "kawaii": "かわいい", "arigato": "ありがとう",
    "sayonara": "さようなら", "konichiwa": "こんにちは", "oppa": "오빠",
    "annyeong": "안녕", "saranghae": "사랑해", "daebak": "대박",
}

# Tên riêng / thương hiệu: không tính là chuyển mã.
PROPER_NOUNS = set("""
facebook fb tiktok tik youtube yt instagram ig twitter x google zalo messenger
shopee lazada tiki sendo grab gojek momo vnpay viettel vinaphone mobifone
vinfast vingroup vinhomes fpt vtv vtc htv thvl netflix spotify discord telegram
zoom android ios iphone samsung xiaomi oppo apple windows chrome
blackpink bts exo twice newjeans aespa seventeen
messi ronaldo neymar mbappe park hang seo
việt vietnam vn hanoi haiphong danang hue saigon cantho
""".split())

# Họ và tên đệm tiếng Việt: chặn "Nguyen Van A" bị đọc thành token lạ.
VI_NAME_PARTS = set("""
nguyễn trần lê phạm hoàng huỳnh phan vũ võ đặng bùi đỗ hồ ngô dương lý
văn thị đình xuân minh anh tuấn hùng dũng nam việt hà linh trang thảo phương
nguyen tran le pham hoang huynh phan vu vo dang bui do ho ngo duong ly
van thi dinh xuan minh anh tuan hung dung nam viet ha linh trang thao phuong
""".split())

# --- Tiếng Việt gõ KHÔNG DẤU -----------------------------------------------
# Đây là lỗi (A) mà LLM mắc nhiều nhất và bản luật cũ cũng không xử lý được:
# "that", "nay", "hay", "vay", "moi", "nguoi" đều là từ tiếng Việt bị bỏ dấu,
# đồng thời là từ tiếng Anh có thật. Tần suất thô không phân biệt được vì tiếng
# Anh dùng "that/hay/may" nhiều hơn hẳn.
#
# Cách dựng: lấy danh sách từ tiếng Việt thường gặp, GIỮ LẠI những từ CÓ DẤU
# tiếng Việt, rồi bỏ dấu. Ràng buộc "phải có dấu ở dạng gốc" loại được đúng
# những từ mượn tiếng Anh đã đi vào tiếng Việt (video, team, like, share) — vì
# chúng vốn không mang dấu — nên danh sách này không nuốt mất tiếng Anh thật.
def _build_vi_unaccented(top_n: int = 50000, min_zipf: float = 3.0) -> Set[str]:
    if not HAS_WORDFREQ:
        return set()
    try:
        from wordfreq import top_n_list
        out = set()
        for w in top_n_list("vi", top_n):
            wl = w.lower()
            if not VIET_DIACRITIC_RE.search(wl):
                continue                       # không có dấu -> không phải bằng chứng
            if zipf(wl, "vi") < min_zipf:
                continue
            bare = strip_diacritics(wl)
            if bare.isalpha() and len(bare) >= 2:
                out.add(bare)
        return out
    except Exception:
        return set()


_VI_UNACCENTED_FALLBACK = set("""
toi tao may ban minh chung ta cac anh chi em con cai cha me bo ma ong ba chau
la thi ma va hay nhung vi voi cho cua trong tren duoi nay kia do day roi chua
khong duoc di den ve an uong lam hoc choi xem nghe noi nghi biet thay thich yeu
ghet qua rat hoi lam luon nua nguoi nha nuoc cua ban ghe sach vo hang cam tin
mot hai ba bon nam sau bay tam chin muoi tram nghin trieu that vay moi cu lai
sang song long man tan chan chin de vi co la ma nhieu it dep xau nhanh cham lon
nho cao thap dai ngan tot hon kem gio ngay thang nam tuan sang trua chieu toi
""".split())

VI_UNACCENTED = _build_vi_unaccented() or _VI_UNACCENTED_FALLBACK

# Địa danh / danh từ riêng tiếng Việt hay bị nhận nhầm là tiếng Anh.
VI_PLACES = set("""
việt nam hà nội sài gòn hồ chí minh đà nẵng hải phòng cần thơ huế nha trang
đà lạt vũng tàu quảng ninh nghệ an thanh hóa bình dương đồng nai long an
viet nam ha noi sai gon ho chi minh da nang hai phong can tho hue nha trang
da lat vung tau quang ninh nghe an thanh hoa binh duong dong nai long an
""".split())


# ===========================================================================
# 4. Tiếng Trung — từ điển chữ Hán + khung âm vị
# ===========================================================================
# Từ điển khai báo bằng CHỮ HÁN kèm pinyin, không khai báo bằng chữ Việt. Các
# cách người Việt viết lại ("xia xìa", "xiê xiê", "xịa xịa") được sinh ra từ khung
# âm vị chứ không phải liệt kê tay — nhờ vậy recall khá mà danh sách gốc vẫn đóng,
# tức precision còn kiểm soát được.
#
# risk = "collide": chuỗi phiên âm trùng với từ/cụm tiếng Việt bình thường
# ("ma ma", "ba ba", "lao sư"). Những mục này chỉ được nhận khi câu có bằng chứng
# tiếng Trung khác (chữ Hán, hoặc một cụm phiên âm "safe" khác).
#
# LƯU Ý QUAN TRỌNG: mọi mục có âm Hán-Việt thông dụng (师父 sư phụ, 缘分 duyên
# phận, 厉害 lợi hại, 可以 khả dĩ, 美女 mỹ nữ, 大哥 đại ca, 姑娘 cô nương...) đã
# bị loại khỏi từ điển này. Đó là TIẾNG VIỆT. Chúng nằm trong HAN_VIET_STOP.

ZH_LEXICON: List[Tuple[str, str, str, str]] = [
    # (chữ Hán, pinyin, nghĩa, risk)
    ("谢谢", "xie xie", "cảm ơn", "safe"),
    ("你好", "ni hao", "xin chào", "safe"),
    ("加油", "jia you", "cố lên", "safe"),
    ("再见", "zai jian", "tạm biệt", "safe"),
    ("对不起", "dui bu qi", "xin lỗi", "safe"),
    ("没关系", "mei guan xi", "không sao", "safe"),
    ("不好意思", "bu hao yi si", "ngại quá", "safe"),
    ("我爱你", "wo ai ni", "anh yêu em", "safe"),
    ("爱你", "ai ni", "yêu em", "safe"),
    ("想你", "xiang ni", "nhớ em", "safe"),
    ("我喜欢你", "wo xi huan ni", "tôi thích bạn", "safe"),
    ("喜欢", "xi huan", "thích", "safe"),
    ("好久不见", "hao jiu bu jian", "lâu rồi không gặp", "safe"),
    ("早上好", "zao shang hao", "chào buổi sáng", "safe"),
    ("晚上好", "wan shang hao", "chào buổi tối", "safe"),
    ("生日快乐", "sheng ri kuai le", "sinh nhật vui vẻ", "safe"),
    ("新年快乐", "xin nian kuai le", "năm mới vui vẻ", "safe"),
    ("恭喜发财", "gong xi fa cai", "cung hỉ phát tài", "safe"),
    ("辛苦了", "xin ku le", "vất vả rồi", "safe"),
    ("太好了", "tai hao le", "tốt quá", "safe"),
    ("很好", "hen hao", "rất tốt", "safe"),
    ("好的", "hao de", "được thôi", "safe"),
    ("是的", "shi de", "đúng vậy", "safe"),
    ("真的", "zhen de", "thật à", "safe"),
    ("什么", "shen me", "cái gì", "safe"),
    ("怎么", "zen me", "thế nào", "safe"),
    ("为什么", "wei shen me", "tại sao", "safe"),
    ("不知道", "bu zhi dao", "không biết", "safe"),
    ("知道了", "zhi dao le", "biết rồi", "safe"),
    ("没事", "mei shi", "không có gì", "safe"),
    ("没有", "mei you", "không có", "safe"),
    ("不行", "bu xing", "không được", "safe"),
    ("不要", "bu yao", "đừng", "safe"),
    ("不是", "bu shi", "không phải", "safe"),
    ("等一下", "deng yi xia", "đợi một chút", "safe"),
    ("慢慢来", "man man lai", "từ từ thôi", "safe"),
    ("快点", "kuai dian", "nhanh lên", "safe"),
    ("漂亮", "piao liang", "xinh đẹp", "safe"),
    ("帅哥", "shuai ge", "anh đẹp trai", "safe"),
    ("小姐姐", "xiao jie jie", "chị xinh", "safe"),
    ("姐姐", "jie jie", "chị", "safe"),
    ("弟弟", "di di", "em trai", "safe"),
    ("妹妹", "mei mei", "em gái", "safe"),
    ("奶奶", "nai nai", "bà", "safe"),
    ("爷爷", "ye ye", "ông", "safe"),
    ("阿姨", "a yi", "dì", "safe"),
    ("叔叔", "shu shu", "chú", "safe"),
    ("亲爱的", "qin ai de", "thân mến", "safe"),
    ("我的天", "wo de tian", "trời ơi", "safe"),
    ("天哪", "tian na", "trời ơi", "safe"),
    ("哎呀", "ai ya", "ối chao", "collide"),
    ("哎哟", "ai yo", "ối", "collide"),
    ("牛逼", "niu bi", "đỉnh", "safe"),
    ("卧槽", "wo cao", "trời đất", "safe"),
    ("给力", "gei li", "đỉnh", "safe"),
    ("奥利给", "ao li gei", "cố lên", "safe"),
    ("绝绝子", "jue jue zi", "tuyệt đỉnh", "safe"),
    ("真香", "zhen xiang", "thơm thật", "safe"),
    ("好吃", "hao chi", "ngon", "safe"),
    ("好看", "hao kan", "đẹp", "safe"),
    ("太棒了", "tai bang le", "tuyệt quá", "safe"),
    ("乖乖", "guai guai", "ngoan", "safe"),
    ("爸爸", "ba ba", "bố", "collide"),
    ("妈妈", "ma ma", "mẹ", "collide"),
    ("哥哥", "ge ge", "anh trai", "collide"),
    ("宝宝", "bao bao", "cục cưng", "collide"),
    ("老板", "lao ban", "ông chủ", "collide"),
    ("老师", "lao shi", "thầy cô", "collide"),
    ("老公", "lao gong", "chồng", "collide"),
    ("老婆", "lao po", "vợ", "collide"),
    ("干杯", "gan bei", "cạn ly", "collide"),
    ("小心", "xiao xin", "cẩn thận", "collide"),
    ("晚安", "wan an", "chúc ngủ ngon", "collide"),
    ("一起", "yi qi", "cùng nhau", "collide"),
    ("红包", "hong bao", "bao lì xì", "collide"),
    ("麻烦", "ma fan", "phiền phức", "collide"),
    ("恭喜", "gong xi", "chúc mừng", "collide"),
]

# Bề mặt khai báo TƯỜNG MINH (đã bỏ dấu). Dùng cho các cách viết phổ biến mà
# tầng tần suất sẽ chặn nhầm vì mọi âm tiết đều là từ tiếng Việt thông dụng —
# điển hình là 加油 viết thành "chia du" ("chia" và "du" đều là từ Việt rất
# thường gặp, nhưng cụm "chia du" thì không phải tiếng Việt).
ZH_SURFACES: Dict[str, List[str]] = {
    "加油": ["chia du", "cha du", "gia du", "chia dau", "jia you", "chia dzu"],
    "谢谢": ["xia xia", "xie xie", "sia sia", "xe xe"],
    "你好": ["ni hao", "nhi hao"],
    "我爱你": ["wo ai ni", "ua ai ni"],
    "爸爸": ["pa pa", "ba ba"],
    "妈妈": ["ma ma"],
    "哥哥": ["cua cua", "ke ke", "ge ge"],
    "姐姐": ["chie chie", "jie jie", "che che"],
    "再见": ["chai chien", "zai jian"],
    "对不起": ["tui bu chi", "dui bu qi"],
    "哎呀": ["ai da", "ai ya"],
    "好久不见": ["hao chiu bu chien", "hao jiu bu jian"],
}

# Chặn cứng: dù khung âm vị có khớp thì các chuỗi này VẪN là tiếng Việt.
# Gồm từ Hán-Việt thông dụng và các cụm hay bị nhận nhầm. So khớp ở dạng bỏ dấu.
_HAN_VIET_TEXT = """
sư phụ sư huynh sư đệ sư tỷ sư muội đại ca đại hiệp tiểu thư công tử thiếu gia
hoàng thượng hoàng hậu thái tử vương gia nương nương công chúa quận chúa
duyên phận nhân duyên tiền duyên hữu duyên vô duyên
lợi hại khả dĩ khả ái mỹ nữ mỹ nam giai nhân
quốc gia gia đình gia tộc dòng họ tổ tiên
hạnh phúc đau khổ tình yêu tình cảm tâm hồn tinh thần vật chất
nhân dân chính phủ xã hội cộng đồng dân tộc quốc dân
học sinh sinh viên giáo viên bác sĩ kỹ sư công nhân nông dân
công ty xí nghiệp đại học trung học tiểu học bệnh viện trường học
phụ nữ đàn ông thanh niên thiếu nhi trẻ em người lớn
thiên hạ giang hồ anh hùng hảo hán tỷ muội huynh đệ bằng hữu tri kỷ
thời gian không gian tương lai quá khứ hiện tại vĩnh viễn
tự do bình đẳng công lý hoà bình hòa bình chiến tranh
sức khỏe bệnh tật cuộc sống cuộc đời số phận định mệnh
cảm ơn tạ ơn xin lỗi cáo lỗi chúc mừng chào hỏi
đồng học đồng chí đồng hương đồng nghiệp
cô nương công phu võ lâm minh chủ chưởng môn
""".strip()
# Ghi chú: "ba ba", "ma ma", "ca ca" CỐ Ý không nằm ở đây. Chúng đã được đánh
# risk="collide" trong ZH_LEXICON nên chỉ được nhận khi câu có bằng chứng tiếng
# Trung khác; chặn cứng thêm ở đây thì không bao giờ bắt được 爸爸妈妈 viết
# phiên âm ngay cạnh chữ Hán.
HAN_VIET_STOP = {strip_diacritics(line.strip()).lower()
                 for line in _HAN_VIET_TEXT.split("\n") if line.strip()}
HAN_VIET_STOP |= {strip_diacritics(w).lower() for w in _HAN_VIET_TEXT.split()}
# thêm mọi cụm 2 âm tiết liên tiếp trong danh sách trên
for _line in _HAN_VIET_TEXT.split("\n"):
    _ws = _line.split()
    for _i in range(len(_ws) - 1):
        HAN_VIET_STOP.add(strip_diacritics(f"{_ws[_i]} {_ws[_i+1]}").lower())


# --- khung âm vị: pinyin <-> chữ quốc ngữ --------------------------------
# Ánh xạ sang một bộ ký hiệu trung gian. Hai âm tiết được coi là "khớp âm" khi
# tập ký hiệu phụ âm đầu giao nhau VÀ tập ký hiệu vần giao nhau.

PY_INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
               "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]

PY_INIT_SKEL = {
    "": {"0"}, "b": {"b", "p"}, "p": {"p", "b"}, "m": {"m"}, "f": {"f", "v"},
    "d": {"d", "t"}, "t": {"t", "d"}, "n": {"n", "l"}, "l": {"l", "n"},
    "g": {"g", "k"}, "k": {"k", "g"}, "h": {"h", "kh"},
    "j": {"c", "z", "j", "g"}, "q": {"c", "kh", "s", "x"}, "x": {"s", "x"},
    "zh": {"c", "z", "tr"}, "ch": {"c", "x", "tr", "s"}, "sh": {"s", "x"},
    "r": {"z", "j", "r", "nh"}, "z": {"c", "z", "t"}, "c": {"s", "x", "t"},
    "s": {"s", "x"}, "y": {"j", "z", "i", "0"}, "w": {"w", "v", "u", "0"},
}

VI_INIT_SKEL = {
    "": {"0"}, "b": {"b", "p"}, "p": {"p", "b"}, "m": {"m"},
    "ph": {"f", "v"}, "v": {"v", "w", "u", "f"},
    "đ": {"d", "t"}, "d": {"z", "j", "d"}, "t": {"t", "d", "z"}, "th": {"t"},
    "n": {"n", "l"}, "l": {"l", "n"}, "g": {"g", "k"}, "gh": {"g", "k"},
    "c": {"k", "g"}, "k": {"k", "g"}, "q": {"k", "g"}, "qu": {"w", "k"},
    "kh": {"kh", "k", "h"}, "h": {"h", "kh"},
    "ch": {"c", "x", "tr", "s"}, "tr": {"c", "z", "tr"}, "gi": {"z", "j", "c"},
    "x": {"s", "x"}, "s": {"s", "x"}, "r": {"z", "j", "r"},
    "ng": {"ng"}, "ngh": {"ng"}, "nh": {"nh", "n", "r"},
    "w": {"w", "v", "u", "0"}, "y": {"j", "i", "0"},
}

PY_FINAL_SKEL = {
    "a": {"a"}, "o": {"o", "uo"}, "e": {"o", "e"}, "i": {"i"}, "u": {"u"},
    "v": {"u"}, "er": {"o", "oi"},
    "ai": {"ai"}, "ei": {"ei", "ai"}, "ao": {"ao", "au"},
    "ou": {"ou", "u", "au", "o"},
    "an": {"an", "en"}, "en": {"en", "an", "on"}, "ang": {"ang"},
    "eng": {"eng", "ong", "ang"}, "ong": {"ong", "ung"},
    "ia": {"ia", "ie"}, "ie": {"ie", "ia", "e"}, "iao": {"iao", "ieu", "iu"},
    "iu": {"iu", "iou"}, "iou": {"iu", "iou"},
    "ian": {"ien", "ian", "en"}, "in": {"in"}, "iang": {"iang", "uang"},
    "ing": {"ing", "in"}, "iong": {"iong", "ung"},
    "ua": {"ua", "oa"}, "uo": {"uo", "o", "ua"}, "uai": {"uai", "oai"},
    "ui": {"ui"}, "uei": {"ui"}, "uan": {"uan", "oan"}, "un": {"un"},
    "uen": {"un"}, "uang": {"uang", "oang"}, "ueng": {"ung"},
    "ve": {"ue", "uye"}, "van": {"uan", "uyen"}, "vn": {"un"},
    "ue": {"ue", "uye"},
}

VI_FINAL_SKEL = {
    "a": {"a"}, "ă": {"a"}, "â": {"a", "o", "e"}, "e": {"e", "o"}, "ê": {"e"},
    "i": {"i"}, "y": {"i"}, "o": {"o"}, "ô": {"o"}, "ơ": {"o", "e"},
    "u": {"u"}, "ư": {"u"},
    "ia": {"ie", "ia"}, "iê": {"ie"}, "ua": {"ua", "uo"}, "uô": {"uo"},
    "ưa": {"uo"}, "ươ": {"uo"}, "uơ": {"uo"},
    "ai": {"ai"}, "ay": {"ai", "ei"}, "ây": {"ei", "ai"}, "ao": {"ao"},
    "au": {"au", "ao", "ou"}, "âu": {"ou", "au"}, "eo": {"eo"},
    "êu": {"eu", "iu"}, "iu": {"iu"}, "ưu": {"uu", "iu"},
    "oi": {"oi"}, "ôi": {"oi", "ui"}, "ơi": {"oi"}, "ui": {"ui"}, "ưi": {"ui"},
    "an": {"an"}, "ăn": {"an"}, "ân": {"en", "an"}, "en": {"en"}, "ên": {"en"},
    "in": {"in"}, "on": {"on", "en"}, "ôn": {"on", "un"}, "ơn": {"on", "en"},
    "un": {"un"}, "ưn": {"un"},
    "ang": {"ang"}, "ăng": {"ang"}, "âng": {"ang", "eng"}, "eng": {"eng"},
    "ong": {"ong"}, "ông": {"ong", "ung"}, "ung": {"ung"}, "ưng": {"ung", "eng"},
    "anh": {"ang", "an", "ing"}, "inh": {"ing", "in"}, "ênh": {"eng"},
    "iên": {"ien"}, "iêu": {"iao", "ieu"}, "yêu": {"iao", "ieu"},
    "uôn": {"uan", "uon"}, "uông": {"uang", "uong"}, "ương": {"uang", "iang"},
    "ươi": {"uoi"}, "uôi": {"uoi", "ui"},
    "oa": {"ua", "oa"}, "oai": {"uai", "oai"}, "oan": {"uan", "oan"},
    "oang": {"uang"}, "oe": {"ue"}, "uy": {"ui"}, "uê": {"ue"},
    "uân": {"un", "uen", "uan"}, "uyên": {"uan", "uyen"},
}


def _split_py(syllable: str) -> Tuple[str, str]:
    s = syllable.strip().lower()
    s = re.sub(r"\d$", "", s).replace("ü", "v")
    for ini in PY_INITIALS:
        if s.startswith(ini):
            return ini, s[len(ini):]
    return "", s


def _split_vi(syllable: str) -> Tuple[str, str]:
    base, _tone, _ = split_tone(syllable.lower())
    for ons in VN_ONSETS_SORTED + ["w"]:
        if base.startswith(ons) and len(base) > len(ons):
            return ons, base[len(ons):]
    if base in ("w", "y"):
        return base, ""
    return "", base


def _final_skel(final: str, table: Dict[str, Set[str]]) -> Set[str]:
    """Tra bảng vần; không có thì gỡ dần âm cuối (tiếng phổ thông không có -p/-t/-k/-m)."""
    f = final.lower()
    if f in table:
        return table[f]
    for coda in ("ch", "nh", "ng", "p", "t", "c", "m", "n"):
        if f.endswith(coda) and len(f) > len(coda):
            head = f[: -len(coda)]
            if head in table:
                base = set(table[head])
                if coda in ("n", "m", "ng", "nh"):
                    base |= {v + "n" for v in base} | {v + "ng" for v in base}
                return base
    stripped = strip_diacritics(f)
    return {stripped} if stripped else {"0"}


def py_skeleton(syllable: str) -> Tuple[Set[str], Set[str]]:
    ini, fin = _split_py(syllable)
    return PY_INIT_SKEL.get(ini, {ini or "0"}), _final_skel(fin, PY_FINAL_SKEL)


def vi_skeleton(syllable: str) -> Tuple[Set[str], Set[str]]:
    ini, fin = _split_vi(syllable)
    return VI_INIT_SKEL.get(ini, {ini or "0"}), _final_skel(fin, VI_FINAL_SKEL)


def _build_zh_index() -> Tuple[Dict[int, List[dict]], Dict[str, dict]]:
    idx: Dict[int, List[dict]] = defaultdict(list)
    surf: Dict[str, dict] = {}
    for han, pinyin, gloss, risk in ZH_LEXICON:
        syls = pinyin.split()
        entry = {"han": han, "pinyin": pinyin, "gloss": gloss, "risk": risk,
                 "skel": [py_skeleton(s) for s in syls], "n": len(syls)}
        idx[len(syls)].append(entry)
        for s in ZH_SURFACES.get(han, []) + [pinyin]:
            surf[strip_diacritics(s).lower()] = entry
    return idx, surf


ZH_INDEX, ZH_SURFACE_INDEX = _build_zh_index()
ZH_MAX_LEN = max(ZH_INDEX) if ZH_INDEX else 0
ZH_SURFACE_MAX = max((len(k.split()) for k in ZH_SURFACE_INDEX), default=0)


def han_to_pinyin(text: str) -> str:
    """Chuyển chữ Hán sang pinyin nếu có pypinyin — dùng khi bạn thêm mục mới."""
    if not HAS_PYPINYIN:
        return ""
    try:
        return " ".join(_lazy_pinyin(text))
    except Exception:
        return ""


# ===========================================================================
# 5. Kết quả
# ===========================================================================

# Loại token. Bốn loại đầu tính là chuyển mã (khớp CS_TYPES_STRICT của pipeline cũ).
CS_TYPES_STRICT = {"english", "chinese_script", "chinese_translit", "other_foreign"}
ALL_TYPES = CS_TYPES_STRICT | {"loanword_naturalized", "proper_noun", "teencode",
                               "laugh", "vietnamese", "unknown"}

# Độ tin cậy gắn với từng loại bằng chứng.
EVIDENCE_CONF = {
    "han_script": 1.00,          # chữ Hán trong câu
    "zh_translit_safe": 0.85,    # khớp từ điển Hán, chuỗi không phải tiếng Việt
    "zh_translit_support": 0.70, # khớp mục dễ trùng, có bằng chứng Hoa khác trong câu
    "foreign_translit": 0.85,
    "en_phrase": 0.90,           # khớp cụm tiếng Anh nhiều từ
    "en_lexicon_nonvi": 0.95,    # trong từ điển Anh và KHÔNG phải âm tiết Việt
    "en_morphology": 0.85,       # gốc từ trong từ điển + hậu tố tiếng Anh
    "en_freq": 0.75,             # chỉ dựa vào tần suất
    "en_strong": 0.85,           # từ lóng mạng ngắn, danh sách đóng
    "en_ambiguous_context": 0.60,# âm tiết Việt hợp lệ, được ngữ cảnh tiếng Anh kéo theo
    "en_translit": 0.70,
}


@dataclass
class Span:
    """Một đoạn văn bản đã được gán loại, kèm lý do."""
    text: str
    start: int
    end: int
    type: str
    evidence: str = ""
    gloss: str = ""
    confidence: float = 0.0

    def __repr__(self) -> str:
        return (f"Span({self.text!r}, {self.type}, {self.evidence}, "
                f"conf={self.confidence:.2f})")


@dataclass
class CMResult:
    text: str = ""
    spans: List[Span] = field(default_factory=list)

    n_chars: int = 0
    n_tokens: int = 0
    n_word_tokens: int = 0
    has_diacritics: bool = False
    diacritic_ratio: float = 0.0

    n_emoji: int = 0
    n_url: int = 0
    n_mention: int = 0
    n_hashtag: int = 0
    n_han_chars: int = 0
    n_non_latin: int = 0

    def _of(self, *types) -> List[Span]:
        return [s for s in self.spans if s.type in types]

    # -- các con số dùng để dựng tập con -----------------------------------
    @property
    def cs_spans(self) -> List[Span]:
        return [s for s in self.spans if s.type in CS_TYPES_STRICT]

    @property
    def confidence(self) -> float:
        cs = self.cs_spans
        return max((s.confidence for s in cs), default=0.0)

    def to_row(self, max_tok: int = 20) -> dict:
        def join(spans):
            return " | ".join(s.text for s in spans[:max_tok])

        en = self._of("english")
        zh_script = self._of("chinese_script")
        zh_tl = self._of("chinese_translit")
        other = self._of("other_foreign")
        loan = self._of("loanword_naturalized")
        prop = self._of("proper_noun")
        teen = self._of("teencode")
        laugh = self._of("laugh")
        unk = self._of("unknown")
        amb = [s for s in self.spans
               if s.type == "vietnamese" and s.evidence == "ambiguous_vi"]
        cs = self.cs_spans

        denom = max(self.n_word_tokens, 1)
        langs = sorted({"zh" if s.type.startswith("chinese") else
                        "en" if s.type == "english" else "other" for s in cs})

        return {
            "n_chars": self.n_chars,
            "n_tokens": self.n_tokens,
            "n_word_tokens": self.n_word_tokens,
            "has_diacritics": self.has_diacritics,
            "diacritic_ratio": round(self.diacritic_ratio, 4),

            "n_english": len(en),
            "english_ratio": len(en) / denom,
            "english_tokens": join(en),
            "n_han": len(zh_script),
            "n_zh_translit": len(zh_tl),
            "n_chinese": len(zh_script) + len(zh_tl),
            "chinese_tokens": join(zh_script + zh_tl),
            "n_other_foreign": len(other),
            "other_foreign_tokens": join(other),
            # giữ đúng nghĩa cũ: n_translit = phiên âm Hoa + ngoại ngữ khác
            "n_translit": len(zh_tl) + len(other),

            "n_loanword": len(loan),
            "loanword_tokens": join(loan),
            "n_proper_noun": len(prop),
            "proper_noun_tokens": join(prop),
            "n_teencode": len(teen),
            "teencode_ratio": len(teen) / denom,
            "teencode_tokens": join(teen),
            "n_laugh": len(laugh),
            "laugh_tokens": join(laugh),
            "n_unknown": len(unk),
            "unknown_tokens": join(unk),
            "n_ambiguous": len(amb),
            "ambiguous_tokens": join(amb),

            "n_emoji": self.n_emoji,
            "n_url": self.n_url,
            "n_mention": self.n_mention,
            "n_hashtag": self.n_hashtag,
            "n_non_latin": self.n_non_latin,

            "n_cs_tokens": len(cs),
            "cs_tokens": join(cs),
            "cs_langs": ",".join(langs),
            "cs_evidence": ",".join(sorted({s.evidence for s in cs})),
            "cs_confidence": round(self.confidence, 3),

            "has_english": len(en) > 0,
            "has_chinese": len(zh_script) + len(zh_tl) > 0,
            "has_teencode": len(teen) > 0,
            "has_emoji": self.n_emoji > 0,
            "has_noise": (self.n_url + self.n_mention + self.n_hashtag) > 0,
        }


# ===========================================================================
# 6. Detector
# ===========================================================================

_EN_SUFFIXES = [("ing", 3), ("ed", 2), ("er", 2), ("est", 3), ("ly", 2),
                ("s", 1), ("es", 2), ("ness", 4), ("ful", 3), ("less", 4),
                ("tion", 4), ("ment", 4), ("able", 4), ("ous", 3), ("ive", 3)]


class CodeMixDetector:
    """Phát hiện code-mixed Việt–Anh–Trung hoàn toàn bằng luật.

    Tham số
    -------
    min_cs_tokens : số token ngoại lai tối thiểu để `is_cs_strict` bật.
    min_confidence : bỏ qua span có độ tin cậy dưới ngưỡng.
    count_loanword : có tính từ mượn đã Việt hoá (ship, like, video) là chuyển mã.
    zipf_gap : chênh lệch Zipf tối thiểu (en − vi) để một âm tiết Việt hợp lệ
        được coi là tiếng Anh. Càng lớn càng chặt. 1.5 ≈ tiếng Anh dùng nhiều
        hơn tiếng Việt khoảng 30 lần.
    zipf_en_min : ngưỡng Zipf tiếng Anh tối thiểu, tránh nhận từ hiếm.
    context_promote : cho phép ngữ cảnh kéo token nhập nhằng thành tiếng Anh khi
        câu đã có ít nhất một token tiếng Anh chắc chắn ở ngay bên cạnh.
    undiacritized_as_teencode : True thì "khong", "duoc", "biet" tính là teencode
        như bản cũ; False (mặc định) thì coi là tiếng Việt gõ thiếu dấu. Bật lên
        nếu cần so số liệu với bảng teencode cũ.
    allow_zh_collide : cho phép nhận các mục phiên âm dễ trùng tiếng Việt
        ("ma ma", "ba ba") khi câu có bằng chứng tiếng Trung khác.
    heavy_min / heavy_ratio : ngưỡng phân biệt cs_level heavy / light.
    """

    def __init__(
        self,
        min_cs_tokens: int = 1,
        min_confidence: float = 0.0,
        count_loanword: bool = False,
        zipf_gap: float = 1.5,
        zipf_en_min: float = 3.5,
        zipf_unknown_min: float = 2.5,
        context_promote: bool = True,
        undiacritized_as_teencode: bool = False,
        allow_zh_collide: bool = True,
        zh_min_syllables: int = 2,
        heavy_min: int = 3,
        heavy_ratio: float = 0.25,
        extra_english: Optional[Iterable[str]] = None,
        extra_teencode: Optional[Iterable[str]] = None,
        extra_stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        self.min_cs_tokens = min_cs_tokens
        self.min_confidence = min_confidence
        self.count_loanword = count_loanword
        self.zipf_gap = zipf_gap
        self.zipf_en_min = zipf_en_min
        self.zipf_unknown_min = zipf_unknown_min
        self.context_promote = context_promote
        self.undiacritized_as_teencode = undiacritized_as_teencode
        self.allow_zh_collide = allow_zh_collide
        self.zh_min_syllables = zh_min_syllables
        self.heavy_min = heavy_min
        self.heavy_ratio = heavy_ratio

        self.en_lexicon = set(EN_LEXICON_WIDE)
        if extra_english:
            self.en_lexicon |= {w.lower() for w in extra_english}
        self.teencode = set(TEENCODE)
        if extra_teencode:
            self.teencode |= {w.lower() for w in extra_teencode}
        self.stopwords = set(HAN_VIET_STOP)
        if extra_stopwords:
            self.stopwords |= {strip_diacritics(w).lower() for w in extra_stopwords}

        self.en_phrases = defaultdict(set)
        for p in EN_PHRASES:
            self.en_phrases[len(p.split())].add(p)
        self.en_phrase_max = max(self.en_phrases) if self.en_phrases else 0

    # -- tiện ích -----------------------------------------------------------

    @staticmethod
    def _is_word(tok: str) -> bool:
        return bool(re.search(rf"[{VI_LETTERS}]", tok)) or bool(HAN_RE.search(tok))

    def _en_morphology(self, word: str) -> bool:
        """Dạng biến hình của một từ tiếng Anh có trong từ điển."""
        for suf, n in _EN_SUFFIXES:
            if len(word) > n + 2 and word.endswith(suf):
                stem = word[:-n]
                for cand in (stem, stem + "e", stem[:-1] if len(stem) > 3 else stem):
                    if cand in self.en_lexicon and not is_vi_syllable(cand):
                        return True
        return False

    # -- các tầng -----------------------------------------------------------

    def _layer_script(self, tokens, claimed, spans, text) -> None:
        """Tầng 1: chữ viết. Chắc chắn nhất, không cần bàn."""
        for i, (tok, s, e) in enumerate(tokens):
            if HAN_RE.search(tok):
                gloss = han_to_pinyin(tok)
                spans.append(Span(tok, s, e, "chinese_script", "han_script",
                                  gloss, EVIDENCE_CONF["han_script"]))
                claimed.add(i)
            elif KANA_RE.search(tok) or HANGUL_RE.search(tok):
                spans.append(Span(tok, s, e, "other_foreign", "script_non_latin",
                                  "", 1.0))
                claimed.add(i)

    def _match_zh(self, words, claimed, spans, has_han: bool) -> bool:
        """Tầng 2: phiên âm tiếng Trung viết bằng chữ Việt.

        Khớp theo khung âm vị với từ điển chữ Hán. Ba lớp chặn, theo thứ tự:
          1. Chuỗi nằm trong HAN_VIET_STOP  -> loại thẳng (đây là tiếng Việt).
          2. Mọi âm tiết đều là từ tiếng Việt thông dụng -> cần bằng chứng thêm.
          3. Mục 'collide' -> chỉ nhận khi câu có bằng chứng tiếng Trung khác.
        """
        found_safe = False
        n = len(words)
        max_size = max(ZH_MAX_LEN, ZH_SURFACE_MAX)
        for size in range(min(max_size, n), self.zh_min_syllables - 1, -1):
            for start in range(0, n - size + 1):
                idxs = list(range(start, start + size))
                if any(words[i][0] in claimed for i in idxs):
                    continue
                toks = [words[i][1] for i in idxs]
                if not all(re.fullmatch(rf"[{VI_LETTERS}]+", t) for t in toks):
                    continue
                surface = strip_diacritics(" ".join(toks).lower())
                if surface in self.stopwords:
                    continue          # là từ Hán-Việt / cụm tiếng Việt -> bỏ

                entry, explicit = ZH_SURFACE_INDEX.get(surface), True
                if entry is not None and entry["n"] != size:
                    entry = None
                if entry is None:                        # thử khớp khung âm vị
                    explicit = False
                    vi_sk = [vi_skeleton(t) for t in toks]
                    for cand in ZH_INDEX.get(size, []):
                        if all(vi_sk[k][0] & cand["skel"][k][0] and
                               vi_sk[k][1] & cand["skel"][k][1] for k in range(size)):
                            entry = cand
                            break
                if entry is None:
                    continue

                # Cụm mà MỌI âm tiết đều là từ tiếng Việt rất thông dụng thì rất
                # dễ là tiếng Việt thật, không phải phiên âm. Chỉ nhận khi bề mặt
                # đã được khai báo tường minh, hoặc câu có bằng chứng Hoa khác.
                if explicit:
                    all_common = False
                elif HAS_WORDFREQ:
                    all_common = all(zipf(t.lower(), "vi") >= 4.3 and is_vi_syllable(t)
                                     for t in toks)
                else:
                    # Không có wordfreq thì không đo được "thông dụng". Lùi về luật
                    # chặt hơn: phải có ít nhất một âm tiết KHÔNG hợp lệ trong tiếng
                    # Việt (xia, wo, xie) thì mới nhận qua khung âm vị.
                    all_common = all(is_vi_syllable(t) for t in toks)
                if entry["risk"] == "collide" or all_common:
                    if not (self.allow_zh_collide and (has_han or found_safe)):
                        continue
                    ev, conf = ("zh_translit_support",
                                EVIDENCE_CONF["zh_translit_support"])
                else:
                    ev, conf = "zh_translit_safe", EVIDENCE_CONF["zh_translit_safe"]
                    found_safe = True
                s0, e0 = words[idxs[0]][2], words[idxs[-1]][3]
                spans.append(Span(" ".join(toks), s0, e0, "chinese_translit",
                                  ev, f"{entry['han']} - {entry['gloss']}", conf))
                claimed.update(words[i][0] for i in idxs)
        return found_safe

    def _match_surface_dict(self, words, claimed, spans, table, typ, ev, conf) -> None:
        """Khớp cụm theo bề mặt đã bỏ dấu (phiên âm Hàn/Nhật, phiên âm tiếng Anh)."""
        n = len(words)
        max_len = max((len(k.split()) for k in table), default=0)
        for size in range(min(max_len, n), 0, -1):
            for start in range(0, n - size + 1):
                idxs = list(range(start, start + size))
                if any(words[i][0] in claimed for i in idxs):
                    continue
                toks = [words[i][1] for i in idxs]
                key = strip_diacritics(" ".join(toks)).lower()
                if key in table:
                    s0, e0 = words[idxs[0]][2], words[idxs[-1]][3]
                    spans.append(Span(" ".join(toks), s0, e0, typ, ev,
                                      table[key], conf))
                    claimed.update(words[i][0] for i in idxs)

    def _match_en_phrase(self, words, claimed, spans) -> None:
        """Tầng 3: cụm tiếng Anh nhiều từ."""
        n = len(words)
        for size in range(min(self.en_phrase_max, n), 1, -1):
            table = self.en_phrases.get(size)
            if not table:
                continue
            for start in range(0, n - size + 1):
                idxs = list(range(start, start + size))
                if any(words[i][0] in claimed for i in idxs):
                    continue
                toks = [words[i][1].lower() for i in idxs]
                if any(VIET_DIACRITIC_RE.search(t) for t in toks):
                    continue
                if " ".join(toks) in table:
                    s0, e0 = words[idxs[0]][2], words[idxs[-1]][3]
                    spans.append(Span(" ".join(words[i][1] for i in idxs), s0, e0,
                                      "english", "en_phrase", "",
                                      EVIDENCE_CONF["en_phrase"]))
                    claimed.update(words[i][0] for i in idxs)

    def _classify_token(self, tok: str, is_first: bool) -> Tuple[str, str, float]:
        """Tầng 4: phân loại một token đơn. Trả về (loại, bằng chứng, độ tin cậy)."""
        low = tok.lower()
        cands = normalize_elongation(low)

        if LAUGH_RE.match(low):
            return "laugh", "laughter", 0.0
        if any(c in self.teencode for c in cands):
            # "khong", "duoc", "biet" nằm trong từ điển teencode cũ nhưng thực ra
            # chỉ là tiếng Việt gõ thiếu dấu — hiện tượng khác hẳn teencode
            # ("ko", "j", "z", "dc"). Gộp chung làm tập con teencode phồng lên,
            # đúng kiểu lỗi mà README đã ghi nhận với tiếng cười.
            plain_vi = low in VI_UNACCENTED and vi_syllable_variants(low)
            if self.undiacritized_as_teencode or not plain_vi:
                return "teencode", "teencode_dict", 0.0
            return "vietnamese", "vi_undiacritized", 0.0
        if low in PROPER_NOUNS or strip_diacritics(low) in PROPER_NOUNS:
            return "proper_noun", "brand_dict", 0.0

        # Có dấu tiếng Việt -> không bao giờ là tiếng Anh. Đây là ràng buộc cứng,
        # chính là lỗi (A)/(B) mà LLM liên tục mắc.
        if VIET_DIACRITIC_RE.search(tok):
            if vi_syllable_variants(low):
                return "vietnamese", "vi_syllable", 0.0
            return "vietnamese", "vi_diacritic", 0.0

        if not ASCII_ALPHA_RE.match(low):
            return "vietnamese", "non_ascii", 0.0
        if low in EN_LOANWORD_NATURALIZED:
            return "loanword_naturalized", "loanword_dict", 0.0
        if low in VI_NAME_PARTS or low in VI_PLACES:
            return "vietnamese", "vi_name", 0.0

        vi_ok = any(is_vi_syllable(c) for c in cands)
        in_lex = any(c in self.en_lexicon for c in cands)
        f_en, f_vi = zipf(low, "en"), zipf(low, "vi")

        # Danh sách đóng các từ lóng / viết tắt tiếng Anh: ưu tiên cao nhất.
        if low in EN_STRONG:
            return "english", "en_strong", EVIDENCE_CONF["en_strong"]

        # Tiếng Việt gõ không dấu ("that", "nay", "vay", "nguoi"). Tần suất tiếng
        # Anh của chúng luôn cao hơn nên không thể để tầng tần suất quyết định.
        # Những token này KHÔNG bao giờ được ngữ cảnh nâng thành tiếng Anh.
        if low in VI_UNACCENTED and vi_ok:
            return "vietnamese", "vi_undiacritized", 0.0

        if not vi_ok:
            if in_lex:
                return "english", "en_lexicon_nonvi", EVIDENCE_CONF["en_lexicon_nonvi"]
            if self._en_morphology(low):
                return "english", "en_morphology", EVIDENCE_CONF["en_morphology"]
            if f_en >= self.zipf_unknown_min and f_en > f_vi:
                return "english", "en_freq", EVIDENCE_CONF["en_freq"]
            # Không phải âm tiết Việt, không có bằng chứng tiếng Anh.
            # KHÔNG đoán. Viết hoa giữa câu -> nhiều khả năng tên riêng.
            if tok[:1].isupper() and not is_first and len(low) >= 3:
                return "proper_noun", "capitalized", 0.0
            return "unknown", "no_evidence", 0.0

        # --- token vừa là âm tiết Việt hợp lệ vừa giống tiếng Anh ---
        if in_lex and f_en >= self.zipf_en_min and (f_en - f_vi) >= self.zipf_gap:
            return "english", "en_freq", EVIDENCE_CONF["en_freq"]
        if in_lex:
            return "vietnamese", "ambiguous_vi", 0.0
        return "vietnamese", "vi_syllable", 0.0

    def _promote_ambiguous(self, spans: List[Span]) -> None:
        """Khử nhập nhằng theo ngữ cảnh.

        Bản cũ nâng TOÀN BỘ token nhập nhằng thành tiếng Anh khi câu có một token
        tiếng Anh bất kỳ — quá rộng ("con" trong "check con hàng" thành tiếng Anh).
        Ở đây chỉ nâng khi token nhập nhằng NẰM SÁT một span tiếng Anh chắc chắn.
        """
        if not self.context_promote:
            return
        order = sorted(range(len(spans)), key=lambda i: spans[i].start)
        strong = {i for i in order if spans[i].type == "english"
                  and spans[i].evidence != "en_ambiguous_context"}
        if not strong:
            return
        pos = {v: k for k, v in enumerate(order)}
        for i, sp in enumerate(spans):
            if sp.type != "vietnamese" or sp.evidence != "ambiguous_vi":
                continue
            k = pos[i]
            neighbours = {order[k - 1] if k > 0 else None,
                          order[k + 1] if k + 1 < len(order) else None}
            if neighbours & strong:
                sp.type = "english"
                sp.evidence = "en_ambiguous_context"
                sp.confidence = EVIDENCE_CONF["en_ambiguous_context"]

    @staticmethod
    def _merge_proper_nouns(spans: List[Span]) -> None:
        """Token viết hoa chưa xác định, đứng cạnh một tên riêng -> cũng là tên riêng.

        Bắt trọn "Dima Egiazarov", "Kim Ji Won": chữ đầu câu không được coi là
        bằng chứng viết hoa, nên nếu chỉ xét từng token thì token đầu luôn rơi
        vào nhóm chưa xác định.
        """
        order = sorted(spans, key=lambda s: s.start)
        for k, sp in enumerate(order):
            if sp.type != "unknown" or not sp.text[:1].isupper():
                continue
            for nb in (order[k - 1] if k > 0 else None,
                       order[k + 1] if k + 1 < len(order) else None):
                if nb is not None and nb.type == "proper_noun" and nb.text[:1].isupper():
                    sp.type, sp.evidence = "proper_noun", "name_sequence"
                    break

    # -- API ----------------------------------------------------------------

    def analyze(self, text) -> CMResult:
        res = CMResult()
        if text is None or (pd is not None and pd.isna(text)):
            return res
        raw = str(text)
        res.text = raw
        res.n_chars = len(raw)

        res.n_url = len(URL_RE.findall(raw))
        res.n_mention = len(MENTION_RE.findall(raw))
        res.n_hashtag = len(HASHTAG_RE.findall(raw))
        res.n_emoji = sum(len(m) for m in EMOJI_RE.findall(raw))
        res.n_han_chars = len(HAN_RE.findall(raw))
        res.n_non_latin = sum(len(r.findall(raw)) for r in SCRIPT_RES.values())

        # bỏ url/mention/hashtag/html nhưng GIỮ nguyên độ dài để offset còn đúng
        clean = HTML_TAG_RE.sub(lambda m: " " * len(m.group(0)), raw)
        for rgx in (URL_RE, MENTION_RE, HASHTAG_RE):
            clean = rgx.sub(lambda m: " " * len(m.group(0)), clean)

        tokens = [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(clean)]
        res.n_tokens = len(tokens)
        word_idx = [i for i, (t, _, _) in enumerate(tokens) if self._is_word(t)]
        res.n_word_tokens = len(word_idx)

        n_dia = len(VIET_DIACRITIC_RE.findall(raw))
        res.has_diacritics = n_dia > 0
        res.diacritic_ratio = n_dia / max(len(raw), 1)

        spans: List[Span] = []
        claimed: Set[int] = set()

        # tầng 1: chữ viết
        self._layer_script(tokens, claimed, spans, raw)
        has_han = res.n_han_chars > 0

        # danh sách token chữ, kèm chỉ số gốc và offset: (idx, text, start, end)
        words = [(i, tokens[i][0], tokens[i][1], tokens[i][2])
                 for i in word_idx if i not in claimed]

        # tầng 2: phiên âm ngoại ngữ (chạy trước vì có thể gồm token có dấu)
        self._match_zh(words, claimed, spans, has_han)
        words = [w for w in words if w[0] not in claimed]
        self._match_surface_dict(words, claimed, spans, OTHER_FOREIGN_TRANSLIT,
                                 "other_foreign", "foreign_translit",
                                 EVIDENCE_CONF["foreign_translit"])
        words = [w for w in words if w[0] not in claimed]
        self._match_surface_dict(words, claimed, spans, EN_TRANSLIT,
                                 "english", "en_translit",
                                 EVIDENCE_CONF["en_translit"])
        words = [w for w in words if w[0] not in claimed]

        # tầng 3: cụm tiếng Anh
        self._match_en_phrase(words, claimed, spans)
        words = [w for w in words if w[0] not in claimed]

        # tầng 4: từng token
        first_word_idx = word_idx[0] if word_idx else -1
        for i, tok, s, e in words:
            typ, ev, conf = self._classify_token(tok, i == first_word_idx)
            spans.append(Span(tok, s, e, typ, ev, "", conf))
            claimed.add(i)

        # tầng 5: ngữ cảnh
        self._promote_ambiguous(spans)
        self._merge_proper_nouns(spans)

        # lọc theo ngưỡng tin cậy
        if self.min_confidence > 0:
            for sp in spans:
                if sp.type in CS_TYPES_STRICT and sp.confidence < self.min_confidence:
                    sp.type = "unknown"
                    sp.evidence = f"dưới ngưỡng ({sp.evidence})"

        spans.sort(key=lambda s: s.start)
        res.spans = spans
        return res

    def analyze_many(self, texts: Sequence) -> List[CMResult]:
        return [self.analyze(t) for t in texts]

    # -- nhãn mức câu -------------------------------------------------------

    def label(self, res: CMResult) -> dict:
        """Từ CMResult ra nhãn mức câu: is_cs_strict / cs_group / cs_level."""
        row = res.to_row()
        n_cs = row["n_cs_tokens"] + (row["n_loanword"] if self.count_loanword else 0)
        is_strict = n_cs >= self.min_cs_tokens
        ratio = n_cs / max(row["n_word_tokens"], 1)

        # Thứ tự ưu tiên: chuyển mã > teencode > emoji > nhiễu > thuần Việt.
        # `other_foreign_mixed` là nhóm riêng (Hàn/Nhật), không gộp vào
        # english_mixed như pipeline LLM cũ — gộp vào sẽ làm bảng kết quả
        # english_mixed lẫn thứ không phải tiếng Anh.
        if is_strict:
            if row["n_chinese"] > 0:
                group = "chinese_mixed"
            elif row["n_other_foreign"] > 0 and row["n_english"] == 0:
                group = "other_foreign_mixed"
            else:
                group = "english_mixed"
        elif row["has_teencode"]:
            group = "teencode_slang"
        elif row["has_emoji"]:
            group = "emoji_only"
        elif row["has_noise"] or row["n_laugh"] > 0:
            group = "other_noise"
        else:
            group = "pure_vi"

        if not is_strict:
            level = "none"
        elif n_cs >= self.heavy_min or ratio >= self.heavy_ratio:
            level = "heavy"
        else:
            level = "light"

        row.update({
            "is_cs_strict": is_strict,
            "is_cs_broad": is_strict or row["has_teencode"],
            "cs_group": group,
            "cs_level": level,
            "cs_ratio": round(ratio, 4),
        })
        return row


_DEFAULT = CodeMixDetector()


def analyze(text, detector: Optional[CodeMixDetector] = None) -> CMResult:
    return (detector or _DEFAULT).analyze(text)


# ===========================================================================
# 7. Chẩn đoán
# ===========================================================================

def explain(text, detector: Optional[CodeMixDetector] = None) -> CMResult:
    """In bảng lý do cho từng token. Dùng khi soi false positive / false negative."""
    det = detector or _DEFAULT
    res = det.analyze(text)
    row = det.label(res)
    print(f'"{text}"')
    print(f"{'token':22s} {'loại':20s} {'bằng chứng':22s} {'conf':>5s}  chú")
    print("-" * 92)
    for sp in res.spans:
        mark = ">>" if sp.type in CS_TYPES_STRICT else "  "
        print(f"{mark}{sp.text[:20]:20s} {sp.type:20s} {sp.evidence:22s} "
              f"{sp.confidence:5.2f}  {sp.gloss[:24]}")
    print("-" * 92)
    print(f"is_cs_strict={row['is_cs_strict']}  cs_group={row['cs_group']}  "
          f"cs_level={row['cs_level']}  langs={row['cs_langs'] or '-'}  "
          f"conf={row['cs_confidence']}")
    if row["n_unknown"]:
        print(f"token chưa xác định (KHÔNG tính là chuyển mã): {row['unknown_tokens']}")
    if row["n_ambiguous"]:
        print(f"token nhập nhằng, xử là tiếng Việt: {row['ambiguous_tokens']}")
    return res


def why_not_vietnamese(word: str) -> None:
    """Giải thích vì sao một token không được coi là âm tiết tiếng Việt."""
    base, tone, has_dia = split_tone(word.lower())
    print(f"token      : {word}")
    print(f"bỏ thanh   : {base}   thanh={tone or 'ngang'}  có dấu={has_dia}")
    hit = False
    for ons in [""] + VN_ONSETS_SORTED:
        if ons and not base.startswith(ons):
            continue
        rime = base[len(ons):]
        if ons == "gi" and rime == "":
            rime = "i"
        ok_rime = rime in VN_RIMES_ALL
        ok_ons = _onset_rime_ok(ons, rime)
        if ok_rime:
            hit = True
            note = "OK" if ok_ons else "vi phạm chính tả (c/k/q, g/gh, ng/ngh, qu)"
            if ok_ons and has_dia and rime.endswith(_STOP_CODAS) \
                    and tone not in ("sac", "nang"):
                note = "âm cuối tắc nhưng thanh không phải sắc/nặng"
            print(f"  tách: âm đầu='{ons}' + vần='{rime}'  -> {note}")
    if not hit:
        print("  không tách được thành vần tiếng Việt nào -> chắc chắn không phải tiếng Việt")
    print(f"kết luận   : {'là' if is_vi_syllable(word) else 'KHÔNG phải'} âm tiết tiếng Việt")
    if HAS_WORDFREQ:
        print(f"zipf       : en={zipf(word.lower(),'en'):.2f}  vi={zipf(word.lower(),'vi'):.2f}")


# ===========================================================================
# 8. API DataFrame — tương thích cm_evaluate.py
# ===========================================================================

def annotate_dataframe(df, text_col: str = "text",
                       detector: Optional[CodeMixDetector] = None,
                       verbose: bool = True):
    """Thêm toàn bộ cột code-mixed vào DataFrame (không sửa df gốc)."""
    if pd is None:
        raise ImportError("Cần pandas.")
    det = detector or _DEFAULT
    rows = [det.label(det.analyze(t)) for t in df[text_col].tolist()]
    feats = pd.DataFrame(rows, index=df.index)
    overlap = [c for c in feats.columns if c in df.columns]
    out = df.drop(columns=overlap).join(feats)
    if verbose:
        print(f"[cm_rules] {len(out)} câu | "
              f"cs_strict={int(out.is_cs_strict.sum())} "
              f"({100*out.is_cs_strict.mean():.2f}%)")
        print(f"           cs_group: {out.cs_group.value_counts().to_dict()}")
        print(f"           tiếng Trung: {int((out.n_chinese>0).sum())} câu | "
              f"phiên âm: {int((out.n_translit>0).sum())} câu | "
              f"token chưa xác định: {int(out.n_unknown.sum())}")
    return out


def build_subsets(df, text_col: str = "text",
                  detector: Optional[CodeMixDetector] = None,
                  compat: bool = True, verbose: bool = True):
    """Chạy detector rồi sinh đúng lược đồ cột mà `cm_evaluate.py` mong đợi.

    compat=True: thêm bí danh `llm_has_cs` / `llm_confidence` / `llm_langs` /
    `is_cs_rule` để các đoạn code viết cho pipeline LLM cũ chạy được không sửa.
    """
    out = annotate_dataframe(df, text_col=text_col, detector=detector,
                             verbose=verbose)
    if compat:
        out["llm_has_cs"] = out["is_cs_strict"]
        out["llm_confidence"] = out["cs_confidence"]
        out["llm_langs"] = out["cs_langs"]
        if "is_cs_rule" not in out.columns:
            out["is_cs_rule"] = out["is_cs_strict"]
    if verbose and "split" in out.columns:
        for sp, g in out.groupby("split"):
            print(f"  {str(sp):6s} n={len(g):5d}  cs_strict={int(g.is_cs_strict.sum()):5d} "
                  f"({100*g.is_cs_strict.mean():5.2f}%)  "
                  f"Trung={int((g.n_chinese>0).sum()):4d}  "
                  f"phiên âm={int((g.n_translit>0).sum()):4d}")
    return out


def make_control(df, seed: int = 42, verbose: bool = True):
    """Nhóm đối chứng pure_vi cân bằng cỡ mẫu và phân bố nhãn với cs_strict."""
    import numpy as np
    rng = np.random.default_rng(seed)
    keep = pd.Series(False, index=df.index)
    key = "primary_label" if "primary_label" in df.columns else None
    grouper = df.groupby("split") if "split" in df.columns else [("all", df)]
    for _, part in grouper:
        cs = part[part.is_cs_strict]
        pure = part[part.cs_group == "pure_vi"]
        if len(cs) == 0 or len(pure) == 0:
            continue
        n = min(len(cs), len(pure))
        chosen: List = []
        if key:
            for lab, frac in cs[key].value_counts(normalize=True).items():
                pool = pure.index[pure[key] == lab].to_numpy()
                k = min(int(round(frac * n)), len(pool))
                if k:
                    chosen += rng.choice(pool, k, replace=False).tolist()
        rest = np.setdiff1d(pure.index.to_numpy(),
                            np.array(chosen, dtype=pure.index.dtype))
        if len(chosen) < n and len(rest):
            chosen += rng.choice(rest, min(n - len(chosen), len(rest)),
                                 replace=False).tolist()
        keep.loc[chosen] = True
    out = df.copy()
    out["control_pure_vi"] = keep
    if verbose:
        print(f"control_pure_vi: {int(keep.sum())} mẫu")
    return out


def export_for_review(df, path: str = "cm_rules_review.csv", n: int = 200,
                      seed: int = 42):
    """Mẫu phân tầng để chấm tay — bắt buộc có trước khi báo cáo precision/recall."""
    cols = [c for c in ["id", "split", "text", "cs_group", "cs_level", "cs_langs",
                        "cs_tokens", "cs_evidence", "cs_confidence", "n_english",
                        "n_chinese", "n_translit", "n_loanword", "unknown_tokens",
                        "ambiguous_tokens"] if c in df.columns]
    parts = []
    per = max(n // max(df.cs_group.nunique(), 1), 1)
    for _, g in df.groupby("cs_group"):
        parts.append(g.sample(min(per, len(g)), random_state=seed)[cols])
    low = df[df.cs_confidence.between(0.01, 0.75)]
    if len(low):
        parts.append(low.sample(min(40, len(low)), random_state=seed)[cols])
    unk = df[df.n_unknown > 0]
    if len(unk):
        parts.append(unk.sample(min(30, len(unk)), random_state=seed)[cols])
    s = pd.concat(parts)
    if "id" in s.columns:
        s = s.drop_duplicates(subset=["id"])
    s = s.sample(frac=1, random_state=seed)
    s["gold_has_cs"] = ""
    s["gold_lang"] = ""
    s["note"] = ""
    s.to_csv(path, index=False)
    print(f"[OK] {len(s)} dòng -> {path}  (điền cột gold_has_cs bằng 1/0)")
    return s


def score_review(path: str = "cm_rules_review.csv"):
    """Precision / recall của bộ luật sau khi bạn chấm tay."""
    d = pd.read_csv(path)
    d = d[d.gold_has_cs.astype(str).str.strip().isin(["0", "1"])]
    if d.empty:
        print("[!] chưa có dòng nào được chấm")
        return None
    gold = d.gold_has_cs.astype(int).astype(bool)
    pred = d.cs_group.isin(["english_mixed", "chinese_mixed"])
    tp = int((gold & pred).sum())
    fp = int((~gold & pred).sum())
    fn = int((gold & ~pred).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    print(f"n={len(d)}  TP={tp} FP={fp} FN={fn}")
    print(f"precision {P:.3f} | recall {R:.3f} | F1 {2*P*R/max(P+R,1e-9):.3f}")
    return {"n": len(d), "precision": P, "recall": R}


def agreement(df, col_a: str = "is_cs_rule", col_b: str = "is_cs_strict"):
    """Đồng thuận giữa hai cột nhãn (vd bộ luật cũ so với bộ luật mới)."""
    if col_a not in df.columns or col_b not in df.columns:
        print(f"[!] thiếu cột {col_a!r} hoặc {col_b!r}")
        return None
    a, b = df[col_a].astype(bool), df[col_b].astype(bool)
    ct = pd.crosstab(a, b, rownames=[col_a], colnames=[col_b])
    print(ct.to_string())
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"\nđồng thuận {100*po:.1f}% | Cohen kappa {kappa:.3f}")
    print(f"{col_a} bắt {int(a.sum())} | {col_b} bắt {int(b.sum())} | "
          f"chỉ {col_b}: {int((~a & b).sum())} | chỉ {col_a}: {int((a & ~b).sum())}")
    return ct


def top_tokens(df, col: str = "cs_tokens", n: int = 50):
    """Token bị gán nhiều nhất — nơi soi false positive nhanh nhất."""
    from collections import Counter
    c = Counter()
    for v in df[col].dropna():
        c.update(t.strip().lower() for t in str(v).split("|") if t.strip())
    t = pd.DataFrame(c.most_common(n), columns=["token", "count"])
    print(t.to_string(index=False))
    return t


# ===========================================================================
# 9. Bộ ca đối chứng
# ===========================================================================
# Mỗi ca là (câu, có chuyển mã?, các ngôn ngữ mong đợi). Ba nhóm cuối chính là ba
# lỗi LLM liên tục mắc — bộ luật phải trả lời đúng thì mới dùng được.

TEST_CASES: List[Tuple[str, bool, List[str]]] = [
    # --- dương tính: tiếng Anh ---
    ("Cái clip này hay vc, xem xong feel good luôn", True, ["en"]),
    ("deadline dí sát nút mà team chưa xong gì cả", True, ["en"]),
    ("anh ấy handsome thật sự, outfit hôm nay quá đỉnh", True, ["en"]),
    ("thôi mình break up đi, mệt rồi", True, ["en"]),
    ("red flag ngay từ đầu mà không ai nhận ra", True, ["en"]),
    ("check in ở đây xong rồi mọi người ơi", True, ["en"]),
    ("bài này đúng là plot twist, không đoán được", True, ["en"]),
    ("working từ sáng tới giờ chưa nghỉ", True, ["en"]),

    # --- dương tính: tiếng Trung ---
    ("加油 nha em, sắp thi rồi", True, ["zh"]),
    ("谢谢 mọi người đã ủng hộ", True, ["zh"]),
    ("xia xìa nhé bạn hiền, mai gặp", True, ["zh"]),
    ("nỉ hảo, mình mới học tiếng Trung", True, ["zh"]),
    ("chia du nha, cố lên nào", True, ["zh"]),
    ("wo ai ni, thật lòng đấy", True, ["zh"]),

    # --- dương tính: ngôn ngữ khác ---
    ("sa rang he oppa của em", True, ["other"]),

    # --- âm tính: từ tiếng Việt viết không dấu (lỗi A của LLM) ---
    ("con nay hon lao qua", False, []),
    ("chung toi la nguoi Viet Nam, con hieu hay khong thi ke", False, []),
    ("ban nay thi tot ma sang nay lai tre", False, []),
    ("tin nay that hay gia vay moi nguoi", False, []),
    ("cam on ban nhieu lam nhe", False, []),
    ("hang nay ban chay lam do", False, []),

    # --- âm tính: tên riêng tiếng Việt (lỗi B của LLM) ---
    ("Nguyễn Văn A ở Hà Nội mới chuyển vào Sài Gòn", False, []),
    ("Thanh Hóa quê tôi đẹp lắm", False, []),

    # --- âm tính: từ Hán-Việt (lỗi C của LLM) ---
    ("Tạ ơn vì có cái cớ, bằng chứng để ly dị cái loại rác rưởi nầy.", False, []),
    ("gia đình hạnh phúc, quốc gia thịnh vượng, thiên hạ thái bình", False, []),
    ("sư phụ dạy đệ tử phải giữ lễ nghĩa", False, []),
    ("đại ca giang hồ ngày xưa giờ đã hoàn lương", False, []),
    ("duyên phận đưa đẩy hai người gặp nhau", False, []),
    ("cô nương này lợi hại thật", False, []),

    # --- âm tính: teencode, tiếng cười, emoji ---
    ("ko hiểu j luôn z mn ơi, bth mà", False, []),
    ("bức ảnh xuất sắc ❤️ haha", False, []),
    ("kkkk cười rụng rốn luôn á", False, []),
    ("mik thấy bth thoi, cx dc mà mn 🙂", False, []),

    # --- âm tính: tên riêng nước ngoài không phải chuyển mã ---
    ("Dima Egiazarov bởi vì chúng tôi là người Việt Nam", False, []),
]


def run_tests(detector: Optional[CodeMixDetector] = None, verbose: bool = True):
    """Chạy bộ ca đối chứng. Trả về DataFrame các ca sai (rỗng là đạt)."""
    det = detector or _DEFAULT
    rows = []
    for text, want_cs, want_langs in TEST_CASES:
        res = det.analyze(text)
        row = det.label(res)
        got_langs = [l for l in row["cs_langs"].split(",") if l]
        ok_cs = bool(row["is_cs_strict"]) == want_cs
        ok_lang = (not want_langs) or set(want_langs) <= set(got_langs)
        rows.append({
            "text": text[:52], "mong_đợi": want_cs, "nhận_được": bool(row["is_cs_strict"]),
            "lang_mong_đợi": ",".join(want_langs), "lang_nhận": row["cs_langs"],
            "token": row["cs_tokens"][:38], "đạt": ok_cs and ok_lang,
        })
    if pd is None:
        for r in rows:
            print(r)
        return rows
    t = pd.DataFrame(rows)
    n_ok = int(t["đạt"].sum())
    if verbose:
        print(f"=== {n_ok}/{len(t)} ca đạt ===")
        bad = t[~t["đạt"]]
        if len(bad):
            with pd.option_context("display.width", 200, "display.max_colwidth", 54):
                print(bad.to_string(index=False))
        else:
            print("Không có ca sai.")
    return t[~t["đạt"]]


def compare_with_cs_detector(df, text_col: str = "text", n_show: int = 12):
    """Đối chiếu bộ luật này với `cs_detector.py` cũ, in các câu lệch nhau.

    Đây là bảng cần có khi viết luận văn: nói "bộ luật mới chặt hơn" thì phải chỉ
    ra chặt hơn ở chỗ nào và bao nhiêu câu bị đổi nhãn.
    """
    try:
        from cs_detector import CodeSwitchDetector, annotate_dataframe as ann_old
    except Exception as e:
        print(f"[!] không import được cs_detector: {e}")
        return None
    old = ann_old(df[[text_col]].copy(), text_col=text_col,
                  detector=CodeSwitchDetector())
    new = annotate_dataframe(df[[text_col]].copy(), text_col=text_col, verbose=False)
    m = pd.DataFrame({
        "text": df[text_col].astype(str).values,
        "cu": old["is_cs_strict"].astype(bool).values,
        "moi": new["is_cs_strict"].astype(bool).values,
        "token_cu": old["english_tokens"].fillna("").values,
        "token_moi": new["cs_tokens"].fillna("").values,
        "bang_chung": new["cs_evidence"].fillna("").values,
    })
    print(pd.crosstab(m.cu, m.moi, rownames=["luật cũ"], colnames=["luật mới"]).to_string())
    po = (m.cu == m.moi).mean()
    print(f"\nđồng thuận {100*po:.1f}% | cũ bắt {int(m.cu.sum())} | mới bắt {int(m.moi.sum())}")
    only_old = m[m.cu & ~m.moi]
    only_new = m[~m.cu & m.moi]
    with pd.option_context("display.max_colwidth", 46, "display.width", 200):
        print(f"\n--- {len(only_old)} câu luật CŨ bắt mà luật MỚI bỏ "
              f"(phần lớn nên là false positive của bản cũ) ---")
        print(only_old[["text", "token_cu"]].head(n_show).to_string(index=False))
        print(f"\n--- {len(only_new)} câu chỉ luật MỚI bắt "
              f"(chữ Hán, phiên âm, cụm nhiều từ) ---")
        print(only_new[["text", "token_moi", "bang_chung"]].head(n_show).to_string(index=False))
    return m


def demo() -> None:
    """Chạy thử toàn bộ: kiểm tra môi trường, bộ ca đối chứng, vài câu mẫu."""
    env_report()
    print()
    run_tests()
    print()
    for s in ("Cái clip này hay vc, xem xong feel good luôn",
              "xia xìa nhé bạn hiền, mai gặp",
              "Tạ ơn vì có cái cớ để ly dị cái loại rác rưởi nầy.",
              "con nay hon lao qua"):
        explain(s)
        print()


if HAS_WORDFREQ:
    print("[cm_rules] sẵn sàng. explain(text) để soi 1 câu | run_tests() để đối chứng "
          "| annotate_dataframe(df) -> build_subsets(df) -> make_control(df)")
else:
    print("[cm_rules] sẵn sàng NHƯNG THIẾU wordfreq -> tầng tần suất tắt. "
          "pip install wordfreq")


if __name__ == "__main__":
    demo()