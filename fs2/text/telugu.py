"""Telugu front-end: Unicode normalisation -> numbers -> rule-based grapheme-to-phoneme.

Telugu orthography is (almost) phonemic, so a deterministic akshara parser gives reliable
phonemes without a lexicon:
  * consonant letters carry an inherent /a/ unless followed by a vowel sign (matra) or virama (్)
  * matras replace the inherent vowel
  * anusvara (ం) becomes the nasal homorganic to the following consonant (/m/ word-finally)
  * visarga (ః) -> /h/, candrabindu (ఁ) -> dropped (nasalisation), ZWJ/ZWNJ removed
  * Latin letters inside Telugu text are read with Telugu letter names
Punctuation becomes the short-pause token "sp".

The same G2P is used to write the pronunciation dictionary for Montreal Forced Aligner,
so phone sets always agree between alignment, training and inference.
"""
import re
import unicodedata

from .numbers_te import normalize_numbers_te

VOWELS_IND = {"అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
              "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఌ": ["l", "u"], "ౡ": ["l", "uu"], "ఎ": ["e"],
              "ఏ": ["ee"], "ఐ": ["ai"], "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"]}
MATRAS = {"ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"], "ృ": ["r", "u"],
          "ౄ": ["r", "uu"], "ె": ["e"], "ే": ["ee"], "ై": ["ai"], "ొ": ["o"], "ో": ["oo"],
          "ౌ": ["au"], "ౢ": ["l", "u"], "ౣ": ["l", "uu"]}
CONSONANTS = {"క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
              "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
              "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
              "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
              "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
              "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "ll", "ఴ": "zh", "వ": "v",
              "శ": "sh", "ష": "ss", "స": "s", "హ": "h", "ౘ": "ts", "ౙ": "dz", "ౚ": "r"}
VIRAMA, ANUSVARA, VISARGA, CANDRABINDU = "్", "ం", "ః", "ఁ"
IGNORED = {"ౕ", "ౖ", "఼", "ఽ", "‌", "‍"}  # length marks, nukta, avagraha, ZWNJ/ZWJ

_NASAL_FOR = {}
for _c in "కఖగఘఙ":
    _NASAL_FOR[_c] = "ng"
for _c in "చఛజఝఞౘౙ":
    _NASAL_FOR[_c] = "ny"
for _c in "టఠడఢణ":
    _NASAL_FOR[_c] = "nn"
for _c in "తథదధన":
    _NASAL_FOR[_c] = "n"
for _c in "పఫబభమ":
    _NASAL_FOR[_c] = "m"

PHONES = sorted(set(sum(VOWELS_IND.values(), []) + sum(MATRAS.values(), []) +
                    list(CONSONANTS.values()) + ["ng", "ny", "nn", "n", "m", "h"]))

_LATIN = {"a": "ఏ", "b": "బీ", "c": "సీ", "d": "డీ", "e": "ఈ", "f": "ఎఫ్", "g": "జీ", "h": "హెచ్",
          "i": "ఐ", "j": "జే", "k": "కే", "l": "ఎల్", "m": "ఎమ్", "n": "ఎన్", "o": "ఓ", "p": "పీ",
          "q": "క్యూ", "r": "ఆర్", "s": "ఎస్", "t": "టీ", "u": "యూ", "v": "వీ", "w": "డబ్ల్యూ",
          "x": "ఎక్స్", "y": "వై", "z": "జెడ్"}
_PUNCT_RE = re.compile(r"[,.;:!?।॥\-—–…\"'()\[\]{}]")


def normalize_telugu(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = normalize_numbers_te(text)
    text = re.sub(r"[A-Za-z]+", lambda m: " ".join(_LATIN[c.lower()] for c in m.group(0)), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_to_phones(word: str):
    phones = []
    chars = [c for c in word if c not in IGNORED]
    i = 0
    while i < len(chars):
        c = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else None
        if c in CONSONANTS:
            phones.append(CONSONANTS[c])
            if nxt == VIRAMA:
                i += 2
                continue
            if nxt in MATRAS:
                phones += MATRAS[nxt]
                i += 2
                continue
            phones.append("a")  # inherent vowel
        elif c in VOWELS_IND:
            phones += VOWELS_IND[c]
        elif c in MATRAS:  # stray matra (e.g. after ZWJ removal) -> just the vowel
            phones += MATRAS[c]
        elif c == ANUSVARA:
            # homorganic nasal; skip the following virama-less consonant lookup chars
            j = i + 1
            phones.append(_NASAL_FOR.get(chars[j], "m") if j < len(chars) else "m")
        elif c == VISARGA:
            phones.append("h")
        # CANDRABINDU, VIRAMA (orphan), and unknown symbols are dropped
        i += 1
    return phones


def text_to_phones(text):
    """Return (normalised_text, phone_list)."""
    norm = normalize_telugu(text)
    phones = []
    for chunk in re.split(r"(\s+|[,.;:!?।॥\-—–…])", norm):
        if not chunk or chunk.isspace():
            continue
        if _PUNCT_RE.fullmatch(chunk):
            if phones and phones[-1] != "sp":
                phones.append("sp")
            continue
        phones += word_to_phones(_PUNCT_RE.sub("", chunk))
    while phones and phones[-1] == "sp":
        phones.pop()
    return norm, phones


def lexicon_entry(word):
    return word_to_phones(word)
