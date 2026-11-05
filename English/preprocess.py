"""Preprocess Svarah for the English model.

    python English/preprocess.py                        # select 5 h/1 h + extract features (HF download)
    python English/preprocess.py --source manifest --manifest /path/svarah_manifest.json
    python English/preprocess.py --stage features       # re-run feature extraction only

Speaker identity: Svarah publishes per-utterance speaker metadata (gender, age group, native
language, state, district, qualification, job, domain) but no explicit speaker id. The tuple of
these fields is unique per speaker, so it is hashed into a stable pseudo speaker id used for the
speaker embedding and for speaker-stratified train/validation sampling.
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fs2.preprocess_cli import main  # noqa: E402

SPEAKER_FIELDS = ["gender", "age-group", "primary_language", "native_place_state",
                  "native_place_district", "highest_qualification", "job_category", "occupation_domain"]


def svarah_speaker(meta):
    if meta.get("speaker_id") not in (None, ""):
        return f"spk_{meta['speaker_id']}"
    key = "|".join(str(meta.get(k, "")) for k in SPEAKER_FIELDS)
    gender = str(meta.get("gender", "x"))[:1].lower() or "x"
    return f"{gender}_{hashlib.md5(key.encode()).hexdigest()[:8]}"


if __name__ == "__main__":
    main(os.path.join(HERE, "config.yaml"), svarah_speaker)
