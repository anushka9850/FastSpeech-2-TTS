"""Telugu number normalisation (cardinals, ordinals with the -వ suffix, clock times).

Written for this project: the reference FastSpeech 2 repo has no Telugu support at all.
Examples
    2026      -> రెండు వేల ఇరవై ఆరు
    12వ       -> పన్నెండవ
    10:45     -> పది నలభై ఐదు
    2026లో    -> రెండు వేల ఇరవై ఆరులో
"""
import re

_UNITS = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
          "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
          "పద్దెనిమిది", "పందొమ్మిది"]
_TENS = ["", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్బై", "ఎనభై", "తొంభై"]

TE_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")


def cardinal(n: int) -> str:
    if n < 0:
        return "మైనస్ " + cardinal(-n)
    if n < 20:
        return _UNITS[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else " " + _UNITS[o])
    if n < 1000:
        h, r = divmod(n, 100)
        if h == 1:
            return "వంద" if r == 0 else "నూట " + cardinal(r)
        return cardinal(h) + (" వందలు" if r == 0 else " వందల " + cardinal(r))
    if n < 100000:
        k, r = divmod(n, 1000)
        if k == 1:
            return "వెయ్యి" if r == 0 else "వెయ్యి " + cardinal(r)
        return cardinal(k) + (" వేలు" if r == 0 else " వేల " + cardinal(r))
    if n < 10000000:
        l, r = divmod(n, 100000)
        if l == 1:
            return "లక్ష" if r == 0 else "లక్ష " + cardinal(r)
        return cardinal(l) + (" లక్షలు" if r == 0 else " లక్షల " + cardinal(r))
    c, r = divmod(n, 10000000)
    if c == 1:
        return "కోటి" if r == 0 else "కోటి " + cardinal(r)
    return cardinal(c) + (" కోట్లు" if r == 0 else " కోట్ల " + cardinal(r))


def attach_suffix(words: str, suffix: str) -> str:
    """Attach a Telugu suffix glued to digits (12వ, 2026లో) to the spoken number."""
    if not suffix:
        return words
    if suffix.startswith("వ"):  # ordinal: drop the final -u / -i vowel sign: పన్నెండు -> పన్నెండవ
        if words.endswith("ు") or words.endswith("ి"):
            return words[:-1] + suffix
        return words + suffix
    return words + suffix


def _time(m):
    h, mi = int(m.group(1)), int(m.group(2))
    return " " + cardinal(h) + ("" if mi == 0 else " " + cardinal(mi)) + " "


def _num(m):
    n = int(m.group(1).replace(",", ""))
    return attach_suffix(cardinal(n), m.group(2) or "")


def normalize_numbers_te(text: str) -> str:
    text = text.translate(TE_DIGITS)
    text = re.sub(r"(\d{1,2}):(\d{2})", _time, text)
    text = re.sub(r"(\d+)\.(\d+)", lambda m: cardinal(int(m.group(1))) + " పాయింట్ " +
                  " ".join(_UNITS[int(d)] for d in m.group(2)), text)
    text = re.sub(r"(\d+)\s?%", lambda m: m.group(1) + " శాతం", text)
    text = re.sub(r"(\d[\d,]*)([ఀ-౿]*)", _num, text)
    return text
