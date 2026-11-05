"""Language-specific text front-ends and symbol tables.

Index 0 is padding, 1 is "sp" (short pause / punctuation), 2 is "spn" (unknown).
The symbol list is written to <preprocessed_path>/symbols.json during preprocessing and
read back at train / inference time, so a checkpoint always travels with its own vocabulary.
"""
import json

SPECIALS = ["<pad>", "sp", "spn"]


def get_frontend(language):
    language = language.lower()
    if language in ("en", "english"):
        from . import english as fe
    elif language in ("te", "telugu"):
        from . import telugu as fe
    else:
        raise ValueError(f"Unsupported language: {language}")
    return fe


def build_symbols(language):
    return SPECIALS + list(get_frontend(language).PHONES)


class SymbolTable:
    def __init__(self, symbols):
        self.symbols = list(symbols)
        self.index = {s: i for i, s in enumerate(self.symbols)}

    def __len__(self):
        return len(self.symbols)

    def encode(self, phones):
        unk = self.index["spn"]
        return [self.index.get(p, unk) for p in phones]

    def decode(self, ids):
        return [self.symbols[i] for i in ids]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.symbols, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
