"""English text normalisation: numbers, times, dates, ordinals, currency, abbreviations.

Written for this project (the reference repo relies on `inflect` + Keith Ito's cleaners).
Kept dependency-free so behaviour is fully deterministic and documented.
"""
import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]

_ORD_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                  "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
_MONTH_RE = "(?:" + "|".join(m.capitalize() for m in MONTHS) + "|" + "|".join(MONTHS) + \
            "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"

_ABBREV = [(re.compile(r"\b%s\." % a, re.IGNORECASE), b) for a, b in [
    ("mrs", "missus"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"),
    ("jr", "junior"), ("maj", "major"), ("gen", "general"), ("drs", "doctors"), ("rev", "reverend"),
    ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"),
    ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"), ("vs", "versus"),
    ("etc", "et cetera"), ("prof", "professor")]]


def cardinal(n: int) -> str:
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else " " + _ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + ("" if r == 0 else " " + cardinal(r))
    for value, name in _SCALES:
        if n >= value:
            q, r = divmod(n, value)
            return cardinal(q) + " " + name + ("" if r == 0 else " " + cardinal(r))
    return str(n)


def ordinal(n: int) -> str:
    words = cardinal(n).split(" ")
    last = words[-1]
    if last in _ORD_IRREGULAR:
        words[-1] = _ORD_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def year(n: int) -> str:
    """Read a 4-digit year the way it is normally spoken (2026 -> twenty twenty six)."""
    if 2000 <= n <= 2009:
        return cardinal(n)
    if 1100 <= n <= 2099 and n % 100 != 0:
        hi, lo = divmod(n, 100)
        lo_w = ("oh " + cardinal(lo)) if lo < 10 else cardinal(lo)
        return cardinal(hi) + " " + lo_w
    if 1100 <= n <= 1999 and n % 100 == 0:
        return cardinal(n // 100) + " hundred"
    return cardinal(n)


def _time(m):
    h, mi = int(m.group(1)), int(m.group(2))
    ampm = (m.group(3) or "").replace(".", "").strip().lower()
    out = cardinal(h)
    if mi == 0:
        out += "" if ampm else " o'clock"
    elif mi < 10:
        out += " oh " + cardinal(mi)
    else:
        out += " " + cardinal(mi)
    if ampm:
        out += " " + ampm.upper()  # "AM" is spelled letter-by-letter by the G2P front-end
    return " " + out + " "


def _day_month(m):  # "12 November" -> "twelfth November"
    return ordinal(int(m.group(1))) + " " + m.group(2)


def _month_day(m):  # "November 12" -> "November twelfth"
    return m.group(1) + " " + ordinal(int(m.group(2)))


def _ord_suffix(m):  # 1st, 22nd
    return ordinal(int(m.group(1)))


def _money(m):
    sym, amt = m.group(1), m.group(2).replace(",", "")
    unit = {"$": "dollars", "₹": "rupees", "£": "pounds", "€": "euros"}[sym]
    if "." in amt:
        a, b = amt.split(".", 1)
        s = cardinal(int(a or 0)) + " " + unit
        if int(b or 0):
            s += " " + cardinal(int(b)) + (" cents" if sym == "$" else " paise" if sym == "₹" else "")
        return s
    return cardinal(int(amt)) + " " + unit


def _decimal(m):
    a, b = m.group(1), m.group(2)
    return cardinal(int(a)) + " point " + " ".join(_ONES[int(d)] for d in b)


def _number(m):
    s = m.group(0).replace(",", "")
    n = int(s)
    if len(s) == 4 and 1100 <= n <= 2099:
        return year(n)
    return cardinal(n)


_MONTH_ABBR = {"jan": "January", "feb": "February", "mar": "March", "apr": "April", "jun": "June",
               "jul": "July", "aug": "August", "sep": "September", "sept": "September",
               "oct": "October", "nov": "November", "dec": "December"}


def normalize_numbers(text: str) -> str:
    text = re.sub(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?(?=\s+\d)",
                  lambda m: _MONTH_ABBR[m.group(1).lower()], text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?\s?[Mm]\.?)?(?=\W|$)", _time, text)
    text = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MONTH_RE + r")\b", _day_month, text)
    text = re.sub(r"\b(" + _MONTH_RE + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b(?!\d)", _month_day, text)
    text = re.sub(r"([$₹£€])\s?([\d,]*\d(?:\.\d+)?)", _money, text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: m.group(1) + " percent", text)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", _ord_suffix, text)
    text = re.sub(r"\b(\d+)\.(\d+)\b", _decimal, text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", _number, text)
    return text


def expand_abbreviations(text: str) -> str:
    for regex, rep in _ABBREV:
        text = regex.sub(rep, text)
    return text


def normalize_english(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("&", " and ")
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    text = re.sub(r"[\"()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
