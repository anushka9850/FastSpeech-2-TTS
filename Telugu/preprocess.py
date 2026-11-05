"""Preprocess IndicTTS Telugu for the Telugu model.

    python Telugu/preprocess.py                          # HF: SPRINGLab/IndicTTS_Telugu
    python Telugu/preprocess.py --source festival --corpus_path /data/IndicTTS/Telugu
                                                         # original IIT-M release (txt.done.data + wav/)
    python Telugu/preprocess.py --stage features

Speakers: the corpus has one male and one female speaker (4.47 h + 4.27 h). 6 h cannot be drawn
from a single speaker, so both are used with a speaker embedding; the split is stratified per
speaker and text-disjoint (both speakers read overlapping prompts).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fs2.preprocess_cli import main  # noqa: E402


def indictts_speaker(meta):
    for k in ("speaker", "speaker_id", "gender"):
        if meta.get(k) not in (None, ""):
            return str(meta[k]).strip().lower()
    return "spk0"


if __name__ == "__main__":
    main(os.path.join(HERE, "config.yaml"), indictts_speaker)
