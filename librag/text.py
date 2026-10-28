"""Arabic/English text processing shared by BM25, dataset cleaning and citation verification.

Arabic normalisation and light stemming follow the light10 procedure of
Larkey, Ballesteros & Connell (2007), the standard baseline for Arabic IR.
"""
from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

# ------------------------------------------------------------------ patterns
_AR_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_ARABIC_CHAR = re.compile("[؀-ۿݐ-ݿࢠ-ࣿ]")
_LATIN_CHAR = re.compile("[A-Za-z]")

# Persian code points that older journal platforms (e.g. Sinaweb) wrote into Arabic
# text: Farsi yeh U+06CC, keheh U+06A9, heh variants, yeh barree. Mapped at cleaning
# time so stored titles are proper Arabic and match queries typed on Arabic keyboards.
_PERSIAN_TO_ARABIC = str.maketrans({
    "ی": "ي", "ک": "ك", "ڪ": "ك", "ۀ": "ه", "ە": "ه", "ھ": "ه", "ہ": "ه", "ے": "ي",
})

_AR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي",
    "ی": "ي", "ک": "ك", "ڪ": "ك", "ۀ": "ه", "ە": "ه", "ھ": "ه", "ہ": "ه", "ے": "ي",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})

EN_STOPWORDS = frozenset("""
a an and are as at be been but by for from has have in into is it its of on or that the
their this to was were which with within without about using use based study studies
i am im looking look find want need some any books book articles article resources
please me my can you give recommend recommendations on about
""".split())

AR_STOPWORDS = frozenset("""
في من على الى إلى عن مع او أو و ثم ان أن إن كان كانت هذا هذه ذلك تلك التي الذي الذين
هو هي هم ما لا لم لن قد كل بعض بين عند حول خلال ضمن عبر لدى منذ حتى اي أي اريد أريد
ابحث أبحث كتب كتاب مصادر بحوث بحث دراسة دراسات عن حول موضوع
""".split())

_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_AR_SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "يه", "ه", "ي")  # light10 list after normalisation (ية→يه, ة→ه)


# ------------------------------------------------------------------ cleaning
def clean_text(s: str | None) -> str:
    """Strip HTML, unescape entities, NFKC-normalise, map Persian letters to Arabic, collapse whitespace."""
    if not s:
        return ""
    s = html.unescape(_TAG.sub(" ", str(s)))
    s = unicodedata.normalize("NFKC", s).translate(_PERSIAN_TO_ARABIC)
    s = s.replace("‏", " ").replace("‎", " ").replace("\xa0", " ")
    return _WS.sub(" ", s).strip()


def normalize(s: str) -> str:
    """Case-fold Latin, remove Arabic diacritics/tatweel and unify letter variants."""
    s = unicodedata.normalize("NFKC", s or "")
    s = _AR_DIACRITICS.sub("", s).replace(_TATWEEL, "")
    s = s.translate(_AR_MAP)
    return s.casefold()


AR_STOPWORDS = frozenset(normalize(w) for w in AR_STOPWORDS)


def detect_lang(s: str) -> str | None:
    """Script-based language guess: 'ar', 'en' or None (too little evidence)."""
    if not s:
        return None
    ar = len(_ARABIC_CHAR.findall(s))
    la = len(_LATIN_CHAR.findall(s))
    if ar + la < 3:
        return None
    return "ar" if ar >= la else "en"


def normalize_lang_code(code: str | None) -> str | None:
    if not code:
        return None
    c = code.strip().lower().replace("_", "-")
    c = c.split("-")[0]
    return {"ara": "ar", "arb": "ar", "eng": "en", "en": "en", "ar": "ar",
            "fre": "fr", "fra": "fr", "ger": "de", "deu": "de", "tur": "tr",
            "kur": "ku", "per": "fa", "fas": "fa"}.get(c, c if len(c) == 2 else None)


# ------------------------------------------------------------------ stemming
def ar_light_stem(tok: str) -> str:
    """light10 (Larkey et al., 2007) applied to an already normalised token."""
    if tok.startswith("و") and len(tok) - 1 >= 3:
        tok = tok[1:]
    for p in _AR_PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 2:
            tok = tok[len(p):]
            break
    for suf in _AR_SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 2:
            tok = tok[: -len(suf)]
    return tok


@lru_cache(maxsize=1)
def _english_stemmer():
    try:
        from nltk.stem import PorterStemmer  # type: ignore

        return PorterStemmer().stem
    except Exception:  # nltk missing → deterministic fallback
        return _simple_en_stem


def _simple_en_stem(tok: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ers", "er", "es", "ed", "ly", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def tokenize(s: str) -> list[str]:
    return _TOKEN.findall(normalize(s))


def analyze(s: str, stem: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Tokenise + (optionally) stop-word removal and language-specific stemming."""
    out = []
    en_stem = _english_stemmer() if stem else None
    for tok in tokenize(s):
        is_ar = bool(_ARABIC_CHAR.search(tok))
        if remove_stopwords and (tok in EN_STOPWORDS or tok in AR_STOPWORDS):
            continue
        if stem:
            tok = ar_light_stem(tok) if is_ar else en_stem(tok)
        if tok:
            out.append(tok)
    return out


# ------------------------------------------------------------------ similarity
def _token_set_ratio(a_toks: list[str], b_toks: list[str]) -> float:
    sa, sb = set(a_toks), set(b_toks)
    if not sa or not sb:
        return 0.0
    inter = " ".join(sorted(sa & sb))
    da = " ".join(sorted(sa - sb))
    db = " ".join(sorted(sb - sa))
    t1 = inter
    t2 = (inter + " " + da).strip()
    t3 = (inter + " " + db).strip()
    r = max(SequenceMatcher(None, t1, t2).ratio() if t1 else 0.0,
            SequenceMatcher(None, t1, t3).ratio() if t1 else 0.0,
            SequenceMatcher(None, t2, t3).ratio())
    return r


def title_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two titles, robust to word order and diacritics.

    Uses rapidfuzz when installed (faster), otherwise an equivalent pure-Python
    token-set ratio. Short candidates are penalised so that a two-word fragment
    cannot match a long title.
    """
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    na, nb = " ".join(ta), " ".join(tb)
    try:
        from rapidfuzz import fuzz  # type: ignore

        r = max(fuzz.ratio(na, nb), fuzz.token_set_ratio(na, nb)) / 100.0
    except Exception:
        r = max(SequenceMatcher(None, na, nb).ratio(), _token_set_ratio(ta, tb))
    coverage = min(len(ta), len(tb)) / max(len(ta), len(tb))
    if coverage < 0.5:
        r *= 0.5 + coverage
    return min(r, 1.0)


def surname_tokens(name: str) -> set[str]:
    """Normalised name tokens of length >= 3 (used for author agreement)."""
    return {t for t in tokenize(name) if len(t) >= 3}
