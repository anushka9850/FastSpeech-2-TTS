"""English front-end: normalisation -> ARPAbet phonemes (with stress), via g2p_en.

g2p_en = CMUdict lookup + POS-tag-based homograph disambiguation (e.g. "conduct" noun vs verb)
+ a neural seq2seq model for out-of-vocabulary words (e.g. "andaroo", "rhyth").
Punctuation becomes the short-pause token "sp". Upper-case acronyms (AM, PM, USA) are spelled.
"""
import re

from .numbers_en import normalize_english

ARPABET = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G",
           "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T",
           "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}

# full phone inventory (stress-marked vowels), identical to MFA's english_us_arpa phone set
PHONES = sorted(set([p for p in ARPABET if p not in VOWELS] +
                    [v + s for v in VOWELS for s in "012"]))

LETTERS = {
    "A": ["EY1"], "B": ["B", "IY1"], "C": ["S", "IY1"], "D": ["D", "IY1"], "E": ["IY1"],
    "F": ["EH1", "F"], "G": ["JH", "IY1"], "H": ["EY1", "CH"], "I": ["AY1"], "J": ["JH", "EY1"],
    "K": ["K", "EY1"], "L": ["EH1", "L"], "M": ["EH1", "M"], "N": ["EH1", "N"], "O": ["OW1"],
    "P": ["P", "IY1"], "Q": ["K", "Y", "UW1"], "R": ["AA1", "R"], "S": ["EH1", "S"], "T": ["T", "IY1"],
    "U": ["Y", "UW1"], "V": ["V", "IY1"], "W": ["D", "AH1", "B", "AH0", "L", "Y", "UW0"],
    "X": ["EH1", "K", "S"], "Y": ["W", "AY1"], "Z": ["Z", "IY1"],
}
_PUNCT = set(",.;:!?-—–…")
_ACRONYM = re.compile(r"\b([A-Z]{2,5})\b")

_g2p = None


def _get_g2p():
    global _g2p
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


def _g2p_segment(text):
    out = []
    for tok in _get_g2p()(text):
        tok = tok.strip()
        if not tok:
            continue
        if tok in PHONES:
            out.append(tok)
        elif all(c in _PUNCT for c in tok):
            out.append("sp")
        # anything else (quotes, apostrophes) is dropped
    return out


def text_to_phones(text):
    """Return (normalised_text, phone_list)."""
    norm = normalize_english(text)
    phones = []
    pos = 0
    for m in _ACRONYM.finditer(norm):
        seg = norm[pos:m.start()]
        if seg.strip():
            phones += _g2p_segment(seg)
        for ch in m.group(1):
            phones += LETTERS[ch]
        pos = m.end()
    if norm[pos:].strip():
        phones += _g2p_segment(norm[pos:])
    # collapse repeated pauses, strip leading pause, end with a single pause
    clean = []
    for p in phones:
        if p == "sp" and (not clean or clean[-1] == "sp"):
            continue
        clean.append(p)
    while clean and clean[-1] == "sp":
        clean.pop()
    return norm, clean


def lexicon_entry(word):
    """Pronunciation of one word (used to write an MFA dictionary)."""
    return [p for p in _g2p_segment(word) if p != "sp"]
