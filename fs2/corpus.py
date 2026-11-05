"""Corpus readers -> a flat list of Utterance records.

Supported sources
  * Hugging Face datasets (ai4bharat/Svarah, SPRINGLab/IndicTTS_Telugu, ...), audio read
    without the `datasets` audio decoder (decode=False) so no torchcodec/ffmpeg is needed
  * manifests: .jsonl / .json / .csv / .tsv with an audio path + text column
  * Festival-style IndicTTS release: <root>/**/txt.done.data + <root>/**/wav/*.wav
"""
import csv
import glob
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import soundfile as sf


@dataclass
class Utterance:
    uid: str
    speaker: str
    text: str
    duration: float
    load: Callable  # -> (np.ndarray float32, sr)
    meta: dict = field(default_factory=dict)


def _file_loader(path):
    def load():
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        return wav.mean(axis=1), sr
    return load


def _bytes_loader(data):
    def load():
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return wav.mean(axis=1), sr
    return load


def _duration_of(path=None, data=None):
    try:
        info = sf.info(path if path is not None else io.BytesIO(data))
        return info.frames / float(info.samplerate)
    except Exception:
        return -1.0


def _safe_id(s):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")


# ------------------------------------------------------------------ manifests
def from_manifest(manifest, audio_root=None, audio_col="audio_filepath", text_col="text",
                  speaker_col=None, speaker_fn: Optional[Callable] = None, duration_col="duration"):
    ext = os.path.splitext(manifest)[1].lower()
    rows = []
    with open(manifest, encoding="utf-8") as f:
        if ext in (".jsonl", ".json"):
            first = f.read(1)
            f.seek(0)
            if ext == ".json" and first == "[":
                rows = json.load(f)
            else:
                rows = [json.loads(l) for l in f if l.strip()]
        else:
            rows = list(csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ","))
    root = audio_root or os.path.dirname(os.path.abspath(manifest))
    utts = []
    for i, r in enumerate(rows):
        a = r[audio_col]
        if isinstance(a, dict):
            a = a.get("path")
        path = a if os.path.isabs(a) else os.path.join(root, a)
        if not os.path.exists(path):
            continue
        spk = str(r[speaker_col]) if speaker_col else (speaker_fn(r) if speaker_fn else "spk0")
        dur = float(r[duration_col]) if r.get(duration_col) not in (None, "") else _duration_of(path=path)
        uid = _safe_id(os.path.splitext(os.path.basename(path))[0]) or f"utt{i:06d}"
        utts.append(Utterance(uid, spk, str(r[text_col]), dur, _file_loader(path), dict(r)))
    return utts


# ------------------------------------------------------------------ Festival (IndicTTS)
_FEST = re.compile(r'^\(\s*(\S+)\s+"(.*)"\s*\)\s*$')


def from_festival(root):
    utts = []
    for meta in glob.glob(os.path.join(root, "**", "txt.done.data"), recursive=True):
        base = os.path.dirname(meta)
        wav_dir = os.path.join(base, "wav") if os.path.isdir(os.path.join(base, "wav")) else base
        rel = os.path.relpath(base, root)
        speaker = _safe_id(rel) or "spk0"
        with open(meta, encoding="utf-8") as f:
            for line in f:
                m = _FEST.match(line.strip())
                if not m:
                    continue
                uid, text = m.group(1), m.group(2)
                path = os.path.join(wav_dir, uid + ".wav")
                if os.path.exists(path):
                    utts.append(Utterance(_safe_id(f"{speaker}_{uid}"), speaker, text.strip(),
                                          _duration_of(path=path), _file_loader(path), {"path": path}))
    return utts


# ------------------------------------------------------------------ Hugging Face
def from_hf(name, split=None, audio_col=None, text_col=None, speaker_fn=None, token=None,
            cache_dir=None, config_name=None):
    from datasets import load_dataset, Audio
    ds = load_dataset(name, config_name, split=split, token=token, cache_dir=cache_dir)
    cols = ds.column_names
    audio_col = audio_col or next(c for c in cols if c in ("audio", "audio_filepath", "speech"))
    text_col = text_col or next(c for c in cols if c in ("text", "sentence", "transcription", "transcript"))
    ds = ds.cast_column(audio_col, Audio(decode=False))
    def _row_bytes(i):
        a = ds[i][audio_col]
        data, path = a.get("bytes"), a.get("path")
        if data is None and path and os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
        return data, path

    def _lazy(i):
        def load():
            return _bytes_loader(_row_bytes(i)[0])()
        return load

    utts = []
    meta_ds = ds.remove_columns([audio_col])  # metadata only: cheap to iterate
    for i, meta in enumerate(meta_ds):
        meta = dict(meta)
        if isinstance(meta.get("duration"), (int, float)):
            dur, path = float(meta["duration"]), None
        else:
            data, path = _row_bytes(i)
            if data is None:
                continue
            dur = _duration_of(data=data)
        spk = speaker_fn(meta) if speaker_fn else "spk0"
        stem = _safe_id(os.path.splitext(os.path.basename(path or ""))[0]) or "utt"
        utts.append(Utterance(f"{stem}_{i:05d}", spk, str(meta[text_col]), dur, _lazy(i), meta))
    return utts
