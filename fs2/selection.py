"""Data selection / subsampling: ~5 h train + ~1 h validation per language, no overlap.

Algorithm (deterministic given the seed)
 1. Quality filter
      - duration in [min_dur, max_dur] seconds (max_dur keeps mels under max_seq_len frames)
      - non-empty transcript, no '[...]' / '<...>' annotation tags
      - speaking-rate filter: characters / second inside the [q_low, q_high] quantiles of the
        corpus -> removes truncated audio and transcripts that do not match the audio
 2. Text-disjoint grouping: utterances with the same normalised transcript form one group and
    are always assigned to the same split (IndicTTS speakers read overlapping prompts; this
    guarantees no sentence appears in both train and validation).
 3. Speaker-stratified allocation: every speaker's share of the val/train budget is proportional
    to the speaker's share of the filtered audio, so validation covers the same speakers as
    training (required for a speaker-embedding model) while the utterances never overlap.
 4. Groups are visited in seeded random order; each is added to validation while that speaker's
    validation quota is open, otherwise to training while its training quota is open.
"""
import random
import re
from collections import defaultdict


def _norm_text(t):
    return re.sub(r"[\W_]+", " ", t.lower(), flags=re.UNICODE).strip()


def _quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[k]


def select(utts, train_hours=5.0, val_hours=1.0, min_dur=1.0, max_dur=11.0, seed=1234,
           cps_quantiles=(0.01, 0.99), speakers=None):
    report = {"input_utterances": len(utts), "input_hours": sum(max(u.duration, 0) for u in utts) / 3600}
    if speakers:
        utts = [u for u in utts if u.speaker in set(speakers)]
    tag = re.compile(r"\[[^\]]*\]|<[^>]*>")
    ok = [u for u in utts if min_dur <= u.duration <= max_dur and u.text.strip() and not tag.search(u.text)]
    report["after_duration_filter"] = len(ok)
    cps = [len(_norm_text(u.text)) / u.duration for u in ok]
    lo, hi = _quantile(cps, cps_quantiles[0]), _quantile(cps, cps_quantiles[1])
    ok = [u for u, c in zip(ok, cps) if lo <= c <= hi]
    report["chars_per_sec_range"] = [round(lo, 2), round(hi, 2)]
    report["after_rate_filter"] = len(ok)
    report["filtered_hours"] = sum(u.duration for u in ok) / 3600

    groups = defaultdict(list)
    for u in ok:
        groups[_norm_text(u.text)].append(u)
    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)

    spk_dur = defaultdict(float)
    for u in ok:
        spk_dur[u.speaker] += u.duration
    total = sum(spk_dur.values())
    scale = min(1.0, total / ((train_hours + val_hours) * 3600)) if total else 1.0
    val_quota = {s: d / total * val_hours * 3600 * scale for s, d in spk_dur.items()}
    train_quota = {s: d / total * train_hours * 3600 * scale for s, d in spk_dur.items()}

    train, val = [], []
    used_val, used_train = defaultdict(float), defaultdict(float)
    for g in group_list:
        g_spk = defaultdict(float)
        for u in g:
            g_spk[u.speaker] += u.duration
        if all(used_val[s] + d <= val_quota[s] * 1.02 for s, d in g_spk.items()):
            val += g
            for s, d in g_spk.items():
                used_val[s] += d
        elif all(used_train[s] + d <= train_quota[s] * 1.02 for s, d in g_spk.items()):
            train += g
            for s, d in g_spk.items():
                used_train[s] += d

    assert not ({u.uid for u in train} & {u.uid for u in val}), "train/val utterance overlap"
    assert not ({_norm_text(u.text) for u in train} & {_norm_text(u.text) for u in val}), "train/val text overlap"
    report.update({
        "train_utterances": len(train), "train_hours": round(sum(u.duration for u in train) / 3600, 3),
        "val_utterances": len(val), "val_hours": round(sum(u.duration for u in val) / 3600, 3),
        "n_speakers": len(spk_dur),
        "per_speaker_hours": {s: {"train": round(used_train[s] / 3600, 3), "val": round(used_val[s] / 3600, 3)}
                              for s in sorted(spk_dur)},
        "seed": seed, "min_dur": min_dur, "max_dur": max_dur,
        "text_overlap_between_splits": 0, "utterance_overlap_between_splits": 0,
    })
    return train, val, report
