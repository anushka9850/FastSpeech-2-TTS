"""Shared preprocessing pipeline (called by English/preprocess.py and Telugu/preprocess.py).

Stages
  select    corpus -> quality filter -> 5 h / 1 h speaker-stratified, text-disjoint split
            -> resample to 22.05 kHz, trim silence, peak-normalise
            -> raw_path/wavs/<speaker>/<id>.wav + .lab, raw_path/{train,val}.tsv
  lexicon   pronunciation dictionary for Montreal Forced Aligner (only for alignment.method=mfa)
  features  text -> phonemes; log-mel, F0 (WORLD), energy; [MFA durations]; filtering;
            normalisation statistics; symbols.json / speakers.json / train.txt / val.txt
"""
import json
import os
import re
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from .audio import AudioProcessor
from .selection import select
from .text import build_symbols, get_frontend, SymbolTable


# ====================================================================== stage: select
def stage_select(cfg, utts, results_dir=None):
    d = cfg["data"]
    raw = cfg["path"]["raw_path"]
    os.makedirs(os.path.join(raw, "wavs"), exist_ok=True)
    train, val, report = select(utts, d["train_hours"], d["val_hours"], d["min_duration"],
                                d["max_duration"], d["seed"], tuple(d.get("cps_quantiles", (0.01, 0.99))),
                                d.get("speakers"))
    print(json.dumps({k: v for k, v in report.items() if k != "per_speaker_hours"}, indent=1))
    ap = AudioProcessor(cfg)
    fe = get_frontend(cfg["language"])
    for split, items in (("train", train), ("val", val)):
        rows = []
        for u in tqdm(items, desc=f"export {split}"):
            try:
                wav, sr = u.load()
            except Exception as e:  # unreadable file
                print("skip", u.uid, e)
                continue
            wav = ap.prepare(wav, sr, trim=True)
            spk_dir = os.path.join(raw, "wavs", u.speaker)
            os.makedirs(spk_dir, exist_ok=True)
            ap.save(os.path.join(spk_dir, u.uid + ".wav"), wav)
            norm, _ = fe.text_to_phones(u.text)
            with open(os.path.join(spk_dir, u.uid + ".lab"), "w", encoding="utf-8") as f:
                f.write(_lab_text(norm))
            rows.append(f"{u.uid}\t{u.speaker}\t{len(wav) / ap.sr:.3f}\t{u.text.strip()}")
        with open(os.path.join(raw, f"{split}.tsv"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
    report["trimmed_hours"] = {s: _tsv_hours(os.path.join(raw, f"{s}.tsv")) for s in ("train", "val")}
    _dump(report, os.path.join(raw, "selection_report.json"))
    if results_dir:
        _dump(report, os.path.join(results_dir, "data_selection.json"))
    return report


def _lab_text(norm):
    return re.sub(r"\s+", " ", re.sub(r"[,.;:!?।॥\"()\[\]{}—–…]", " ", norm)).strip()


def _tsv_hours(path):
    with open(path, encoding="utf-8") as f:
        return round(sum(float(l.split("\t")[2]) for l in f if l.strip()) / 3600, 3)


# ====================================================================== stage: lexicon (MFA)
def stage_lexicon(cfg):
    raw = cfg["path"]["raw_path"]
    fe = get_frontend(cfg["language"])
    words = set()
    for root, _, files in os.walk(os.path.join(raw, "wavs")):
        for fn in files:
            if fn.endswith(".lab"):
                with open(os.path.join(root, fn), encoding="utf-8") as f:
                    words.update(f.read().split())
    out = os.path.join(raw, "lexicon.txt")
    with open(out, "w", encoding="utf-8") as f:
        for w in sorted(words):
            pron = fe.lexicon_entry(w)
            if pron:
                f.write(f"{w}\t{' '.join(pron)}\n")
    print(f"wrote {len(words)} words -> {out}")


# ====================================================================== stage: features
def _read_tsv(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n").split("\t") for l in f if l.strip()]


def _textgrid_alignment(tg_path, sr, hop):
    import tgt
    tier = tgt.io.read_textgrid(tg_path, include_empty_intervals=True).get_tier_by_name("phones")
    sil = {"sil", "sp", "spn", "", "<eps>"}
    phones, durs = [], []
    start = end = 0.0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text
        if not phones:
            if p in sil:
                continue
            start = s
        if p not in sil:
            phones.append(p)
            end, end_idx = e, len(phones)
        else:
            phones.append("sp")
        durs.append(int(np.round(e * sr / hop) - np.round(s * sr / hop)))
    return phones[:end_idx], durs[:end_idx], start, end


def _feature_worker(args):
    uid, spk, text, wav_path, cfg = args
    try:
        ap = AudioProcessor(cfg)
        fe = get_frontend(cfg["language"])
        wav = ap.load(wav_path, trim=False)  # already trimmed at export
        norm, phones = fe.text_to_phones(text)
        durations = None
        if cfg["alignment"]["method"] == "mfa":
            tg = os.path.join(cfg["alignment"]["textgrid_path"], spk, uid + ".TextGrid")
            if not os.path.exists(tg):
                return uid, None, "no_textgrid"
            phones, durations, s, e = _textgrid_alignment(tg, ap.sr, ap.hop)
            if not phones:
                return uid, None, "empty_alignment"
            wav = wav[int(ap.sr * s):int(ap.sr * e)]
        if not phones:
            return uid, None, "no_phones"
        mel, energy = ap.mel_and_energy(wav)
        if durations is not None:
            T = int(sum(durations))
            mel, energy = mel[:T], energy[:T]
            if mel.shape[0] < T:
                return uid, None, "short_audio"
        T = mel.shape[0]
        if T > cfg["model"]["max_seq_len"]:
            return uid, None, "too_long"
        if T < len(phones):
            return uid, None, "fewer_frames_than_phones"
        f0 = ap.pitch(wav, T)
        voiced = f0[f0 > 0]
        f0i = ap.interpolate_f0(f0)
        if f0i is None:
            return uid, None, "unvoiced"
        pre = cfg["path"]["preprocessed_path"]
        np.save(os.path.join(pre, "mel", uid + ".npy"), mel)
        np.save(os.path.join(pre, "pitch", uid + ".npy"), f0i)
        np.save(os.path.join(pre, "energy", uid + ".npy"), energy)
        if durations is not None:
            np.save(os.path.join(pre, "duration", uid + ".npy"), np.array(durations, dtype=np.int64))
        rng = np.random.default_rng(zlib.crc32(uid.encode()))
        take = lambda x, k: x if len(x) <= k else rng.choice(x, k, replace=False)
        return uid, {"speaker": spk, "phones": phones, "norm": norm, "frames": T,
                     "voiced_f0": take(voiced, 200).astype(np.float64),
                     "f0_sum": float(voiced.sum()), "f0_sq": float((voiced.astype(np.float64) ** 2).sum()),
                     "f0_n": int(len(voiced)), "e_sum": float(energy.sum()),
                     "e_sq": float((energy.astype(np.float64) ** 2).sum()), "e_n": int(len(energy)),
                     "f0_sample": take(f0i, 200).astype(np.float64), "e_sample": take(energy, 200).astype(np.float64)}, "ok"
    except Exception as e:  # keep going on bad files, but report them
        return uid, None, f"error:{type(e).__name__}:{e}"


def stage_features(cfg, workers=4, results_dir=None):
    raw, pre = cfg["path"]["raw_path"], cfg["path"]["preprocessed_path"]
    for sub in ("mel", "pitch", "energy", "duration"):
        os.makedirs(os.path.join(pre, sub), exist_ok=True)
    splits = {s: _read_tsv(os.path.join(raw, f"{s}.tsv")) for s in ("train", "val")}
    jobs = []
    for s, rows in splits.items():
        for uid, spk, _, text in rows:
            jobs.append((uid, spk, text, os.path.join(raw, "wavs", spk, uid + ".wav"), cfg))
    results, reasons = {}, {}
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for uid, r, why in tqdm(ex.map(_feature_worker, jobs, chunksize=8), total=len(jobs), desc="features"):
                reasons[why] = reasons.get(why, 0) + 1
                if r is not None:
                    results[uid] = r
    else:
        for j in tqdm(jobs, desc="features"):
            uid, r, why = _feature_worker(j)
            reasons[why] = reasons.get(why, 0) + 1
            if r is not None:
                results[uid] = r
    if not results:
        raise RuntimeError(f"No utterance survived feature extraction: {reasons}")

    # statistics (computed on the TRAINING split only)
    train_ids = [r[0] for r in splits["train"] if r[0] in results]
    R = [results[u] for u in train_ids]
    f0_n = sum(r["f0_n"] for r in R)
    f0_mean = sum(r["f0_sum"] for r in R) / f0_n
    f0_std = float(np.sqrt(max(sum(r["f0_sq"] for r in R) / f0_n - f0_mean ** 2, 1e-6)))
    e_n = sum(r["e_n"] for r in R)
    e_mean = sum(r["e_sum"] for r in R) / e_n
    e_std = float(np.sqrt(max(sum(r["e_sq"] for r in R) / e_n - e_mean ** 2, 1e-6)))
    f0_s = np.concatenate([r["f0_sample"] for r in R])
    e_s = np.concatenate([r["e_sample"] for r in R])
    q = cfg["preprocessing"]["pitch"].get("bin_quantiles", [0.005, 0.995])
    p_lo, p_hi = np.quantile((f0_s - f0_mean) / f0_std, q)
    e_lo, e_hi = np.quantile((e_s - e_mean) / e_std, q)
    stats = {"pitch": [float(p_lo), float(p_hi), float(f0_mean), f0_std],
             "energy": [float(e_lo), float(e_hi), float(e_mean), e_std],
             "pitch_hz_range_voiced": [float(np.quantile(np.concatenate([r["voiced_f0"] for r in R]), 0.01)),
                                       float(np.quantile(np.concatenate([r["voiced_f0"] for r in R]), 0.99))]}
    _dump(stats, os.path.join(pre, "stats.json"))

    speakers = sorted({r["speaker"] for r in results.values()})
    _dump({s: i for i, s in enumerate(speakers)}, os.path.join(pre, "speakers.json"))
    symbols = SymbolTable(build_symbols(cfg["language"]))
    if cfg["alignment"]["method"] == "mfa":  # keep any phone MFA produced that we did not list
        extra = sorted({p for r in results.values() for p in r["phones"]} - set(symbols.symbols))
        symbols = SymbolTable(symbols.symbols + extra)
    symbols.save(os.path.join(pre, "symbols.json"))

    summary = {"reasons": reasons, "stats": stats, "speakers": speakers}
    for s, rows in splits.items():
        lines = [f"{uid}|{results[uid]['speaker']}|{' '.join(results[uid]['phones'])}|{results[uid]['norm']}"
                 for uid, *_ in rows if uid in results]
        with open(os.path.join(pre, f"{s}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        frames = sum(results[uid]["frames"] for uid, *_ in rows if uid in results)
        summary[s] = {"utterances": len(lines),
                      "hours": round(frames * cfg["preprocessing"]["stft"]["hop_length"]
                                     / cfg["preprocessing"]["audio"]["sampling_rate"] / 3600, 3)}
    _dump(summary, os.path.join(pre, "preprocess_summary.json"))
    if results_dir:
        _dump(summary, os.path.join(results_dir, "preprocess_summary.json"))
    print(json.dumps({k: summary[k] for k in ("train", "val", "reasons")}, indent=1))
    return summary


def _dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
