"""
cs_detector.py
==============
Unified rule-based Vietnamese-English code-switching detector for ViGoEmotions.

Thay thế 2 phiên bản detector đang lệch nhau trong repo
(`vigo_code_switching_analysis.ipynb` và `vigo_baseline_subset_eval_kaggle.ipynb`)
bằng một module dùng chung cho cả Giai đoạn 1 (ViGoEmotions) và Giai đoạn 2 (ViCM).

Cải tiến so với rule cũ
-----------------------
1. **Vietnamese phonotactics filter**: tiếng Việt là đơn âm tiết và có cấu trúc
   âm tiết chặt (onset + nucleus + coda). Một token ASCII KHÔNG khớp cấu trúc
   này gần như chắc chắn không phải tiếng Việt -> tín hiệu code-switching mạnh
   hơn nhiều so với danh sách stopword ASCII thủ công.
   (vd: "working", "good", "team", "beautiful" đều bị loại khỏi tiếng Việt;
    "cam", "hang", "con", "sang" được nhận là âm tiết Việt hợp lệ.)
2. **Xử lý từ nhập nhằng (ambiguous)**: token vừa là âm tiết Việt hợp lệ vừa là
   từ tiếng Anh ("can", "man", "tin", "hot", "sang"...) chỉ được tính là tiếng
   Anh khi trong câu có ít nhất một token tiếng Anh *không nhập nhằng*
   (contextual disambiguation), hoặc khi nằm trong danh sách loanword mạnh.
3. **Tách laughter/onomatopoeia** (haha, kkk, hihi, hehe, huhu, lol) ra khỏi
   TEENCODE. Rule cũ đếm "haha" là teencode làm subset teencode phồng lên.
4. **Chuẩn hoá kéo dài ký tự** (đẹppppp -> đẹp, goooood -> good) trước khi tra từ điển.
5. **Tách proper noun / foreign-other** (tên riêng nước ngoài, romanization Hàn/Trung)
   khỏi `english_tokens`, tránh đếm "Dima Egiazarov" là code-switching.
6. **Nhiều tier**: `is_cs_strict` (chỉ English-mixed) và `is_cs_broad`
   (English + teencode) để chọn định nghĩa khi viết báo cáo mà không phải chạy lại.

Cách dùng
---------
    from cs_detector import CodeSwitchDetector, annotate_dataframe

    det = CodeSwitchDetector()
    det.analyze("Cái clip này hay vc, xem xong feel good luôn")

    df = annotate_dataframe(df, text_col="text")
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Optional, Sequence

try:
    import pandas as pd
except ImportError:  # pandas chỉ cần cho annotate_dataframe
    pd = None


# ---------------------------------------------------------------------------
# 1. Regex cơ bản
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[\w.]+")
HASHTAG_RE = re.compile(r"(?<!\w)#[\w_]+")
HTML_TAG_RE = re.compile(r"<[^>]{1,15}>")          # ViGoEmotions có <br>
NUMBER_RE = re.compile(r"^\d+([.,]\d+)?$")

# token = chuỗi chữ (có dấu tiếng Việt) hoặc số hoặc dấu câu
TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+(?:['-][A-Za-zÀ-ỹĐđ]+)*|\d+|[^\w\s]", re.UNICODE)

VIET_DIACRITIC_RE = re.compile(
    r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)

EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoticons, symbols, pictographs, supplemental
    "\U00002190-\U000021FF"   # arrows
    "\U00002300-\U000023FF"   # misc technical
    "\U00002600-\U000026FF"   # misc symbols
    "\U00002700-\U000027BF"   # dingbats
    "\U0000FE0F"              # variation selector
    "\U00002B00-\U00002BFF"
    "]+",
    flags=re.UNICODE,
)

# Non-Latin script (Hàn, Nhật, Trung, Thái, Cyrillic, Ả Rập)
NON_LATIN_RE = re.compile(
    "["
    "Ѐ-ӿ"           # Cyrillic
    "؀-ۿ"           # Arabic
    "฀-๿"           # Thai
    "぀-ヿ"           # Kana
    "㐀-䶿一-鿿"  # CJK
    "가-힯"           # Hangul
    "]"
)

ASCII_ALPHA_RE = re.compile(r"^[a-z]+$")
ELONGATION_RE = re.compile(r"(.)\1{2,}")

# Laughter / onomatopoeia -> KHÔNG tính là code-switching
LAUGH_RE = re.compile(
    r"^(?:"
    r"(?:h[aeiouyăâêôơư]{1,2}){2,}"      # haha, hehe, hihi, huhu, hoho, hơhơ
    r"|(?:hj){2,}|(?:jz){2,}|(?:ck){2,}"
    r"|k{2,}|z{3,}|w+k+w*"
    r"|lol|lmao|lmfao|rofl|jaja|khkh|khjkhj"
    r")$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 2. Vietnamese phonotactics (kiểm tra âm tiết tiếng Việt hợp lệ)
# ---------------------------------------------------------------------------

VN_ONSETS = (
    "ngh", "ng", "nh", "ch", "gh", "gi", "kh", "ph", "th", "tr", "qu",
    "b", "c", "d", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x",
)

VN_NUCLEI = {
    "a", "ai", "ao", "au", "ay",
    "e", "eo", "eu",
    "i", "ia", "ie", "ieu", "iu", "iaa",
    "o", "oa", "oai", "oao", "oay", "oe", "oeo", "oi", "oo",
    "u", "ua", "uai", "uao", "uay", "ue", "ui", "uo", "uoi", "uou", "uu",
    "uy", "uya", "uye", "uyu", "uau",
    "y", "ya", "ye", "yeu", "yea",
}

VN_CODAS = ("ngh", "ng", "nh", "ch", "c", "m", "n", "p", "t", "")

_VN_SYLLABLE_RE = re.compile(
    r"^(?P<onset>ngh|ng|nh|ch|gh|gi|kh|ph|th|tr|qu|[bcdghklmnprstvx])?"
    r"(?P<nucleus>[aeiouy]{1,3})"
    r"(?P<coda>ng|nh|ch|c|m|n|p|t)?$"
)


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt, đ -> d."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def normalize_elongation(token: str) -> str:
    """đẹppppp -> đẹpp -> (thử) đẹp ; goooood -> good."""
    return ELONGATION_RE.sub(r"\1\1", token)


def is_vietnamese_syllable(token: str) -> bool:
    """True nếu token (đã bỏ dấu, lowercase) là âm tiết tiếng Việt hợp lệ."""
    t = strip_diacritics(token).lower()
    if not ASCII_ALPHA_RE.match(t):
        return False
    m = _VN_SYLLABLE_RE.match(t)
    if not m:
        return False
    return m.group("nucleus") in VN_NUCLEI


# ---------------------------------------------------------------------------
# 3. Từ điển
# ---------------------------------------------------------------------------

# 3.1 Teencode / viết tắt tiếng Việt trên mạng (KHÔNG gồm laughter)
TEENCODE = set(
    """
ko k kh kg khg khum khong hok hong hem hp hnay
dc đc dk đk dx đx dta
j z dz zj zi zì zậy zay vạy vay v vs vk ck
r ùi ui rùi roi ròi lun
mik mk mjk mìn mềnh mng mn ae ny bff
ib rep cmt cmts stt avt acc ad add adm mod
bt bth bthg qá qa wa wá cx cg cchứ
sml vcl vkl vl vc clm cmn cmnr đm dm đcm dcm vch
tks thks thanks3 plz pls ok8
bn b4v t2 m2
nc ns nt nch tks
đt dt sdt cmnd cccd
gato ghen tuc
trl trlai
hqua bgio bh bjo
ntn nthe nthế nhma nhma
đky dky
xl xin loi
kb kbiet
tt ttinh
mng mngười
qq qc
xh xhoi
ncl nchung
    """.split()
)

# 3.2 Từ tiếng Anh phổ biến trong tiếng Việt mạng xã hội (lexicon lõi)
EN_CORE = set(
    """
about above accept access account action active actually add admin admire advice
after again against age agree ahead album alive all almost alone already also always
amazing american among amount analysis and angel angry animal announce another answer
any anymore anyone anything apple apply april area argue army around arrive art article
artist ask assume attack attention audience august author available avoid awesome away
baby back background bad bag balance ball band bank bar base basic battle beach beat
beautiful beauty because become bed beer before begin behavior behind believe below
benefit best better between beyond big bill billion bird birthday bit black blame
block blog blood blue board body bomb book boom boring born boss both bottle bottom
box boy brain brand break breakfast bring brother brown budget build building burn
business busy but butter button buy call camera camp campaign can cancel cancer
candidate cap capital captain car card care career carry case cash cast cat catch
cause celebrate cell center central century certain chain chair challenge champion
chance change channel chapter character charge charity chart chase cheap cheat check
cheer chef chemical chicken chief child children china chinese choice choose church
citizen city civil claim class classic classroom clean clear click client climate clip
clock close cloud club coach coast coffee cold collect college color come comfort comic
comment commit common community company compare compete complete complex computer
concept concern concert condition conference confidence confirm conflict confuse
congrats congratulation connect consider constant contact contain content contest
context continue contract control conversation cook cool copy corner correct cost
could count country couple course court cover crazy cream create creative credit crew
crime crisis critical criticism cross crowd cry culture cup current custom customer
cute cutting cycle daddy daily damage dance danger dark data date daughter day dead
deal dear death debate debt decade decide decision deep defense degree delete deliver
demand democracy department depend describe design desire desk despite destroy detail
develop device diet difference different difficult digital dinner direct direction
director dirty disagree discover discuss disease dislike display distance divide doctor
document dog dollar domestic door double doubt down download dozen draft drag drama draw
dream dress drink drive driver drop drug dry during duty each early earn earth east easy
eat economy edge edit editor education effect effort egg either election electric element
else email embarrass emotion employ empty end enemy energy engine english enjoy enough
enter entire environment episode equal equipment error escape especially essay establish
even evening event ever every everyone everything evidence exact example excellent except
exchange excited exciting excuse execute exercise exist exit expand expect expensive
experience expert explain explore express extra extreme eye face fact factor fail fair
faith fake fall family famous fan fantastic far farm fashion fast father fault favor
favorite fear feature feed feel feeling female few field fight figure file fill film
final finally finance find fine finger finish fire firm first fish fit five fix flag
flat flight floor flow flower fly focus follow follower food foot football force foreign
forest forever forget forgive form format former forward found four frame free freedom
fresh friend friendship from front fruit full fun function fund funny future gain game
gaming gap garden gas gate gay general generation gentle get gift girl girlfriend give
glad glass global goal god gold golden good google government grab grade graduate grand
grant grass grateful great green greet grey ground group grow growth guard guess guest
guide guilty guitar gun guy habit hair half hall hand handle hang happen happiness happy
hard hate have head health healthy hear heart heat heavy hell hello help hero herself
hide high hill himself hire history hit hobby hold hole holiday home homework honest
honey hope horror horse hospital host hot hotel hour house housing however huge human
humor hundred hungry hurt husband ice idea ideal identify idol ignore image imagine
impact important impossible improve include income increase indeed independent index
indian individual industry influence info information initial inside insist inspire
instagram install instead institution insurance intelligence interest interesting
internet interview into introduce invest invite involve iron island issue item its
japan jealous job join joke journey joy judge juice jump junior just justice keep key
kick kid kill kind king kiss kitchen knee knife know knowledge lab label labor lack lady
lake land landing language laptop large last late later laugh launch law lawyer lay lazy
lead leader league learn least leave lecture left legal legend lemon length less lesson
let letter level library license lie life lift light like likely limit line link lion
lip liquid list listen literature little live living load loan local location lock logic
lonely long look loop lose loss lost lot loud love lovely low loyal luck lucky lunch
machine mad magazine magic mail main maintain major make male mall man manage manager
many map march mark market marriage marry mass master match material math matter maybe
mayor meal mean meaning measure meat media medical medicine medium meet meeting member
memory mention menu mercy message metal method middle might military milk million mind
mine minute mirror miss mission mistake mix mobile model modern moment money monitor
month mood moon moral more morning most mother motion motor mount mountain mouse mouth
move movie much multi music must myself mystery nail name narrow nation national native
natural nature near nearly neat necessary neck need negative neighbor nervous net network
never new news next nice night nine noise none noon normal north nose note nothing notice
novel now nuclear number nurse object observe obvious occur ocean off offer office officer
official often oil okay old online only open operate opinion opportunity oppose option
orange order organic organization origin original other otherwise ought outdoor outside
over overall overcome owe own owner pack package page pain paint pair panel paper parent
park part participate particular partner party pass passion past path patient pattern
pause pay peace peak pen people percent perfect perform performance perhaps period person
personal perspective phone photo physical piano pick picture piece pilot pink pizza place
plan plane planet plant plastic plate play player please pleasure plenty plus pocket poem
poet point police policy political politics poll pool poor pop popular population port
position positive possible post pot potato potential pound power practice pray prefer
premium prepare present president press pressure pretty prevent previous price pride
primary prime print prior priority prison private prize probably problem process produce
product profession professional professor profile profit program progress project promise
promote proof proper property propose protect proud prove provide public publish pull
punch punish purchase pure purple purpose push put quality quarter queen question quick
quiet quit quite quote race radio rain raise random range rank rap rapid rare rate rather
rating reach react read reader ready real reality realize really reason receive recent
recipe recognize recommend record recover red reduce refer reflect reform refuse regard
region regret regular reject relate relation relationship relax release relevant relief
religion rely remain remember remind remote remove repair repeat replace reply report
represent republic request require research reserve resource respect respond response
responsible rest restaurant result retail retire return reveal review revolution rice rich
ride right ring rise risk river road robot rock role roll romance romantic roof room root
rose rough round route routine row royal rule run running rural rush sad safe safety
salad salary sale salt same sample sand satisfy save saving say scale scandal scary scene
schedule scheme school science score screen script sea search season seat second secret
section sector secure security seed seek seem select self sell senate send senior sense
sensitive sentence separate series serious serve service session set setting settle seven
several severe sex shadow shake shall shame shape share sharp she sheet shelf shell shift
shine ship shirt shock shoe shoot shop shopping short shot should shoulder shout show
shower shut shy sick side sight sign signal significant silent silver similar simple
simply since sing singer single sink sir sister sit site situation six size skill skin
skirt sky sleep slice slide slight slip slow small smart smell smile smoke snap snow soccer
social society sock soft software soil solar soldier solid solution solve some someone
something sometimes somewhere son song soon sorry sort soul sound soup source south space
speak special species specific speech speed spell spend spirit split spoke sponsor sport
spot spread spring square stable staff stage stair stand standard star start state
statement station status stay steal steel step stick still stock stomach stone stop store
storm story straight strange strategy stream street strength stress stretch strike strong
structure struggle student studio study stuff stupid style subject submit subscribe
success successful such sudden suffer sugar suggest suit summer sun super supply support
suppose sure surface surprise survey survive sweet swim switch symbol system table tail
take talent talk tall tank tape target task taste tax teach teacher team tear tech
technology teen teenager television tell temperature temple tend tent term terrible test
text thank thanks that theater their them theme themselves then theory therapy there
these they thick thin thing think third thirty this those though thought thousand threat
three throat through throw thus ticket tie tight time tiny tip tired title today together
toilet tomorrow tone tonight too tool tooth top topic total touch tough tour tourist
toward tower town toy track trade tradition traffic train training transfer transform
translate transport travel treat treatment tree trend trending trial trick trip trophy
trouble truck true trust truth try tube tune turn tutorial twenty twice twin twitter two
type typical ugly ultimate unable uncle under understand unfair unfollow unhappy union
unique unit universe university unknown unless until update upload upon upset urban urge
usage use used useful user usual vacation valid value variety various vast vegetable
vehicle version very veteran victim victory video view viewer village violence virtual
virus visible vision visit visual vital vlog vocal voice volume volunteer vote voter wage
wait wake walk wall want war warm warn warning wash waste watch water wave way weak wealth
weapon wear weather web website wedding week weekend weight weird welcome welfare well
west western wet whatever wheel when where whether which while white who whole whom whose
why wide wife wild will win wind window wine wing winner winter wire wisdom wise wish with
within without witness woman wonder wonderful wood word work worker working world worry
worse worst worth would wound wrap write writer writing wrong yard yeah year yellow yes
yesterday yet you young your yourself youth youtube zero zone
    """.split()
)

# 3.3 Loanword tiếng Anh rất phổ biến trong tiếng Việt DÙ có dạng âm tiết Việt hợp lệ.
#     Những từ này luôn được tính là tiếng Anh, không cần ngữ cảnh.
EN_LOAN_STRONG = {
    "hot", "top", "ship", "chat", "sad", "cam", "sub", "seen", "team", "check",
    "deal", "fan", "fake", "feel", "flex", "hit", "like", "live", "post", "reply",
    "review", "sale", "share", "shop", "show", "stream", "style", "test", "trend",
    "vibe", "view", "wow", "cool", "crush", "date", "drop", "game", "gym", "job",
    "kit", "lag", "level", "list", "log", "look", "mood", "note", "party", "pass",
    "peak", "plan", "play", "power", "pro", "profile", "rank", "rap", "rate",
    "real", "record", "repeat", "reset", "run", "save", "scan", "score", "search",
    "sell", "sense", "set", "sexy", "shot", "sign", "size", "skill", "smart",
    "soft", "solo", "sorry", "spam", "speed", "spirit", "sport", "staff", "star",
    "start", "stop", "story", "strong", "super", "sure", "sweet", "tag", "talk",
    "target", "taste", "teen", "text", "thanks", "time", "tip", "tour", "track",
    "train", "trip", "trust", "try", "tutorial", "unbox", "update", "upload",
    "user", "video", "vip", "voice", "vote", "wall", "war", "warm", "watch",
    "web", "weekend", "welcome", "win", "wish", "work", "world", "young", "zip",
    # slang / viết tắt tiếng Anh ngắn (<=3 ký tự vẫn tính)
    "bro", "sis", "omg", "wtf", "idk", "btw", "fyi", "tbh", "nvm", "thx", "ty",
    "gg", "gj", "af", "asap", "diy", "faq", "ceo", "mv", "ost", "pov", "cmv",
    "bff", "goat", "npc", "otp", "irl", "dm", "pm", "ads", "app", "bug", "fix",
    "dev", "meme", "vlog", "blog", "cast", "clip", "combo", "cover", "demo",
    "edit", "fail", "feed", "flop", "hype", "menu", "mix", "moment", "next",
    "nice", "noob", "pass", "pin", "poll", "pop", "quiz", "reel", "remix",
    "rich", "sad", "scam", "season", "seed", "shape", "sharp", "sim", "skin",
    "slay", "slow", "snack", "solid", "sound", "spoil", "spot", "swag", "tank",
    "tea", "tone", "tool", "trailer", "trip", "troll", "ugly", "vibes", "wifi",
}

# 3.4 Từ KHÔNG bao giờ tính là tiếng Anh dù trông giống (tên riêng VN, brand, âm tiết Việt phổ biến)
EN_BLACKLIST = {
    "vn", "vnd", "hn", "hcm", "sg", "tphcm", "vtv", "vinfast", "viettel",
    "shopee", "lazada", "tiki", "grab", "momo", "zalo",
    "anh", "em", "ba", "má", "chi", "con", "cam", "sang", "hang", "long", "song",
    "tin", "than", "them", "then", "the", "to", "at", "am", "an", "in", "on", "no",
    "so", "do", "go", "he", "it", "is", "as", "be", "by", "of", "or", "up", "us",
    "we", "me", "my", "hi", "ha", "la", "ma", "na", "pa", "ta", "va", "xa",
}


# ---------------------------------------------------------------------------
# 4. Detector
# ---------------------------------------------------------------------------


@dataclass
class CSResult:
    n_chars: int = 0
    n_tokens: int = 0
    n_word_tokens: int = 0

    has_diacritics: bool = False
    diacritic_ratio: float = 0.0

    n_english: int = 0
    english_ratio: float = 0.0
    english_tokens: str = ""

    n_english_guess: int = 0
    english_guess_tokens: str = ""

    n_teencode: int = 0
    teencode_ratio: float = 0.0
    teencode_tokens: str = ""

    n_laugh: int = 0
    laugh_tokens: str = ""

    n_foreign_other: int = 0
    foreign_other_tokens: str = ""

    n_emoji: int = 0
    n_url: int = 0
    n_mention: int = 0
    n_hashtag: int = 0
    n_non_latin: int = 0

    has_english: bool = False
    has_teencode: bool = False
    has_emoji: bool = False
    has_noise: bool = False

    is_cs_strict: bool = False       # chỉ English-mixed
    is_cs_broad: bool = False        # English hoặc teencode
    cs_group: str = "pure_vi"
    cs_level: str = "none"           # none | light | heavy


class CodeSwitchDetector:
    """Rule-based Vi-En code-switching detector.

    Parameters
    ----------
    min_english_count : số token tiếng Anh tối thiểu để coi là English-mixed.
    min_english_ratio : tỉ lệ token tiếng Anh tối thiểu (0 = tắt).
    count_guess_as_english : có tính token "đoán là tiếng Anh" (không có trong
        lexicon nhưng không phải âm tiết Việt) vào english_tokens hay không.
    use_wordfreq : thử mở rộng lexicon bằng package `wordfreq` nếu có.
    heavy_ratio : ngưỡng english_ratio để gán cs_level = 'heavy'.
    """

    def __init__(
        self,
        min_english_count: int = 1,
        min_english_ratio: float = 0.0,
        count_guess_as_english: bool = True,
        use_wordfreq: bool = True,
        wordfreq_top_n: int = 30000,
        heavy_ratio: float = 0.25,
        extra_english: Optional[Iterable[str]] = None,
        extra_teencode: Optional[Iterable[str]] = None,
    ) -> None:
        self.min_english_count = min_english_count
        self.min_english_ratio = min_english_ratio
        self.count_guess_as_english = count_guess_as_english
        self.heavy_ratio = heavy_ratio

        self.en_lexicon = set(EN_CORE)
        if extra_english:
            self.en_lexicon |= {w.lower() for w in extra_english}
        if use_wordfreq:
            self.en_lexicon |= self._load_wordfreq(wordfreq_top_n)
        self.en_lexicon -= EN_BLACKLIST

        self.teencode = set(TEENCODE)
        if extra_teencode:
            self.teencode |= {w.lower() for w in extra_teencode}

        # cache kết quả kiểm tra âm tiết
        self._vn_cache: dict = {}

    # -- lexicon helpers ---------------------------------------------------

    @staticmethod
    def _load_wordfreq(top_n: int) -> set:
        try:
            from wordfreq import top_n_list  # type: ignore

            words = {
                w.lower()
                for w in top_n_list("en", top_n)
                if w.isalpha() and len(w) >= 3
            }
            return words
        except Exception:
            return set()

    def _is_vn(self, token: str) -> bool:
        cached = self._vn_cache.get(token)
        if cached is None:
            cached = is_vietnamese_syllable(token)
            self._vn_cache[token] = cached
        return cached

    # -- token classification ---------------------------------------------

    def classify_token(self, token: str) -> str:
        """Trả về: 'vi' | 'en' | 'en_ambiguous' | 'en_guess' | 'teencode'
        | 'laugh' | 'foreign' | 'other'."""
        raw = token
        low = raw.lower()

        if VIET_DIACRITIC_RE.search(raw):
            # có dấu tiếng Việt -> chắc chắn không phải tiếng Anh
            return "teencode" if strip_diacritics(low) in self.teencode or low in self.teencode else "vi"

        if not ASCII_ALPHA_RE.match(low):
            return "other"

        if LAUGH_RE.match(low):
            return "laugh"

        # chuẩn hoá kéo dài ký tự: loooove -> loove -> love
        candidates = [low]
        norm = normalize_elongation(low)
        if norm != low:
            candidates.append(norm)
            candidates.append(re.sub(r"(.)\1+", r"\1", low))

        for cand in candidates:
            if cand in self.teencode:
                return "teencode"
        if low in EN_BLACKLIST:
            return "vi"

        for cand in candidates:
            if cand in EN_LOAN_STRONG:
                return "en"

        is_vn_shape = any(self._is_vn(c) for c in candidates)
        in_en_lex = any(c in self.en_lexicon for c in candidates)

        if in_en_lex and not is_vn_shape:
            return "en"
        if in_en_lex and is_vn_shape:
            return "en_ambiguous"
        if is_vn_shape:
            return "vi"

        # Không phải âm tiết Việt hợp lệ và không có trong lexicon tiếng Anh.
        # Tiếng Việt viết là ngôn ngữ đơn âm tiết với phonotactics chặt, nên
        # token kiểu này gần như chắc chắn là chất liệu ngoại lai.
        if len(low) < 3:
            return "other"
        if raw[:1].isupper() and len(low) >= 3:
            return "foreign"          # nhiều khả năng là tên riêng
        return "en_guess"

    # -- main --------------------------------------------------------------

    def analyze(self, text) -> CSResult:
        res = CSResult()
        if text is None or (pd is not None and pd.isna(text)):
            return res
        raw = str(text)
        res.n_chars = len(raw)

        # đếm noise trước khi tokenize
        res.n_url = len(URL_RE.findall(raw))
        res.n_mention = len(MENTION_RE.findall(raw))
        res.n_hashtag = len(HASHTAG_RE.findall(raw))
        res.n_emoji = sum(len(m) for m in EMOJI_RE.findall(raw))
        res.n_non_latin = len(NON_LATIN_RE.findall(raw))

        clean = HTML_TAG_RE.sub(" ", raw)
        clean = URL_RE.sub(" ", clean)
        clean = MENTION_RE.sub(" ", clean)
        clean = HASHTAG_RE.sub(" ", clean)

        tokens = TOKEN_RE.findall(clean)
        res.n_tokens = len(tokens)
        word_tokens = [t for t in tokens if re.search(r"[A-Za-zÀ-ỹĐđ]", t)]
        res.n_word_tokens = len(word_tokens)

        n_diac = len(VIET_DIACRITIC_RE.findall(raw))
        res.has_diacritics = n_diac > 0
        res.diacritic_ratio = n_diac / max(len(raw), 1)

        en, en_amb, en_guess, teen, laugh, foreign = [], [], [], [], [], []
        for tok in word_tokens:
            kind = self.classify_token(tok)
            if kind == "en":
                en.append(tok)
            elif kind == "en_ambiguous":
                en_amb.append(tok)
            elif kind == "en_guess":
                en_guess.append(tok)
            elif kind == "teencode":
                teen.append(tok.lower())
            elif kind == "laugh":
                laugh.append(tok.lower())
            elif kind == "foreign":
                foreign.append(tok)

        # Contextual disambiguation: token nhập nhằng chỉ tính là tiếng Anh
        # khi câu đã có ít nhất 1 token tiếng Anh chắc chắn.
        if en:
            en = en + en_amb          # đã có bằng chứng tiếng Anh -> nhận luôn từ nhập nhằng
        # nếu không, en_amb được coi là tiếng Việt

        # token không phải âm tiết Việt (en_guess) mặc định VẪN tính là bằng chứng
        # code-switching, nhưng luôn được lưu riêng để bạn báo cáo cả 2 con số.
        effective_en = en + en_guess if self.count_guess_as_english else en

        denom = max(res.n_word_tokens, 1)
        res.n_english = len(effective_en)
        res.english_ratio = len(effective_en) / denom
        res.english_tokens = " ".join(en[:30])
        res.n_english_guess = len(en_guess)
        res.english_guess_tokens = " ".join(en_guess[:30])
        res.n_teencode = len(teen)
        res.teencode_ratio = len(teen) / denom
        res.teencode_tokens = " ".join(teen[:30])
        res.n_laugh = len(laugh)
        res.laugh_tokens = " ".join(laugh[:20])
        res.n_foreign_other = len(foreign)
        res.foreign_other_tokens = " ".join(foreign[:20])

        res.has_english = (
            res.n_english >= self.min_english_count
            and res.english_ratio >= self.min_english_ratio
        )
        res.has_teencode = res.n_teencode > 0
        res.has_emoji = res.n_emoji > 0
        res.has_noise = (res.n_url + res.n_mention + res.n_hashtag) > 0

        res.is_cs_strict = res.has_english
        res.is_cs_broad = res.has_english or res.has_teencode

        if res.has_english:
            res.cs_group = "english_mixed"
        elif res.has_teencode:
            res.cs_group = "teencode_slang"
        elif res.has_emoji:
            res.cs_group = "emoji_only"
        elif res.has_noise or res.n_laugh > 0:
            res.cs_group = "other_noise"
        else:
            res.cs_group = "pure_vi"

        if not res.has_english:
            res.cs_level = "none"
        elif res.english_ratio >= self.heavy_ratio or res.n_english >= 3:
            res.cs_level = "heavy"
        else:
            res.cs_level = "light"

        return res

    def analyze_many(self, texts: Sequence) -> List[CSResult]:
        return [self.analyze(t) for t in texts]


# ---------------------------------------------------------------------------
# 5. pandas helper
# ---------------------------------------------------------------------------


def annotate_dataframe(df, text_col: str = "text", detector: Optional[CodeSwitchDetector] = None):
    """Thêm toàn bộ cột code-switching vào DataFrame (không sửa df gốc)."""
    if pd is None:
        raise ImportError("Cần pandas để dùng annotate_dataframe.")
    det = detector or CodeSwitchDetector()
    feats = pd.DataFrame([asdict(det.analyze(t)) for t in df[text_col].tolist()], index=df.index)
    overlap = [c for c in feats.columns if c in df.columns]
    out = df.drop(columns=overlap).join(feats)
    return out


__all__ = [
    "CodeSwitchDetector",
    "CSResult",
    "annotate_dataframe",
    "is_vietnamese_syllable",
    "strip_diacritics",
    "TEENCODE",
    "EN_CORE",
    "EN_LOAN_STRONG",
]
