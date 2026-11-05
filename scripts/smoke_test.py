"""End-to-end pipeline test on a tiny synthetic corpus (no dataset download, CPU friendly).

    python scripts/smoke_test.py                 # both languages
    python scripts/smoke_test.py --lang Telugu --steps 60

For each language it builds ~60 synthetic "speech" utterances (2 speakers; every phoneme is a
harmonic or noise segment with its own spectral shape), then runs the REAL pipeline:
selection -> feature extraction -> training with the learned aligner (incl. validation, loss
plots, checkpointing) -> inference on the official test sentences -> variance-adaptor plots.
Outputs go to a scratch folder, never into Results/. The audio is not speech - this checks
that the code runs end to end, not model quality.
"""
import argparse
import copy
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
import zlib

import numpy as np
import soundfile as sf
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fs2.text import get_frontend  # noqa: E402

WORDS = {
    "English": "the a meeting orchestra audience research method time morning quickly praised "
               "will start evaluate proposed changed gained during on of to by and so should arrive "
               "experiment ceremony movement momentum considerable rhythm conduct twelve november".split(),
    "Telugu": "ఈ సమావేశం పులి అడవిలో నడిచింది ఆమె చీరను ధరించి వేడుకకు వచ్చింది తాడుపై అతని "
              "పట్టు చాలా బలంగా ఉంది నూతన ఔషధాల ప్రభావాన్ని పరిశోధనలో సమయానికి సిద్ధంగా "
              "ఉంచండి ఉదయం గంటలకు తేదీ నవంబర్".split(),
}
SR = 22050


def phone_sound(ph, dur, f0, rng):
    t = np.arange(int(dur * SR)) / SR
    h = zlib.crc32(ph.encode()) % 997
    if ph == "sp":
        return 0.003 * rng.standard_normal(len(t))
    voiced = ph[0].lower() in "aeiouylmnrvwjgbdz"
    if voiced:
        f = f0 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
        phase = 2 * np.pi * np.cumsum(f) / SR
        formants = [300 + (h % 7) * 90, 900 + (h % 11) * 120, 2200 + (h % 5) * 150]
        x = sum((1.0 / k) * np.sin(k * phase) * sum(np.exp(-((k * f0 - F) / 180.0) ** 2) for F in formants)
                for k in range(1, 30))
    else:
        x = rng.standard_normal(len(t))
        x = np.convolve(x, np.hanning(3 + h % 9), mode="same") * 0.3
    env = np.minimum(1, np.minimum(t, t[::-1]) / 0.01)
    return 0.4 * x * env


def make_corpus(lang, out, n=60, seed=0):
    rng = np.random.default_rng(seed)
    pyr = random.Random(seed)
    fe = get_frontend(lang)
    os.makedirs(os.path.join(out, "wavs"), exist_ok=True)
    rows = []
    for i in range(n):
        spk = "female" if i % 2 else "male"
        f0 = 210 if spk == "female" else 115
        text = " ".join(pyr.choice(WORDS[lang]) for _ in range(pyr.randint(4, 9))) + "."
        _, phones = fe.text_to_phones(text)
        segs = [np.zeros(int(0.15 * SR))]
        for p in phones:
            segs.append(phone_sound(p, rng.uniform(0.05, 0.12) if p != "sp" else 0.15, f0 * rng.uniform(0.9, 1.1), rng))
        segs.append(np.zeros(int(0.15 * SR)))
        wav = np.concatenate(segs).astype(np.float32)
        path = os.path.join(out, "wavs", f"{lang[:2].lower()}_{i:03d}.wav")
        sf.write(path, wav, SR)
        rows.append({"audio_filepath": path, "text": text, "speaker": spk, "duration": len(wav) / SR})
    man = os.path.join(out, "manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return man, sum(r["duration"] for r in rows) / 3600


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="both", choices=["both", "English", "Telugu"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--work", default=None, help="scratch folder (default: temp dir)")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    work = args.work or tempfile.mkdtemp(prefix="fs2_smoke_")
    langs = ["English", "Telugu"] if args.lang == "both" else [args.lang]
    for lang in langs:
        d = os.path.join(work, lang)
        man, hours = make_corpus(lang, os.path.join(d, "corpus"))
        with open(os.path.join(ROOT, lang, "config.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        c = copy.deepcopy(cfg)
        c["path"] = {"corpus_path": os.path.join(d, "corpus"), "raw_path": os.path.join(d, "raw"),
                     "preprocessed_path": os.path.join(d, "pre"), "ckpt_path": os.path.join(d, "ckpt"),
                     "log_path": os.path.join(d, "logs"), "result_path": os.path.join(d, "Results")}
        c["dataset"].update({"source": "manifest", "manifest": man, "audio_col": "audio_filepath",
                             "text_col": "text", "speaker_col": "speaker"})
        c["data"].update({"train_hours": hours * 0.75, "val_hours": hours * 0.2, "min_duration": 0.3,
                          "cps_quantiles": [0.0, 1.0]})
        c["train"].update({"batch_size": 4, "total_step": args.steps, "log_step": 5,
                           "val_step": max(10, args.steps // 2), "save_step": max(10, args.steps // 2),
                           "num_workers": 0})
        c["optimizer"]["warm_up_step"] = 20
        c["optimizer"]["lr_scale"] = 1e-3 / (256 ** -0.5 * 20 ** -0.5)  # same 1e-3 peak LR as the real run
        c["alignment"]["bin_loss_start_step"] = args.steps // 2
        c["alignment"]["bin_loss_warmup_steps"] = 5
        c["inference"]["test_sentences"] = os.path.join(ROOT, "test_sentences", f"{lang.lower()}.tsv")
        cfg_path = os.path.join(d, "config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
        py = sys.executable
        run([py, f"{lang}/preprocess.py", "--config", cfg_path, "--workers", "2"])
        run([py, f"{lang}/train.py", "--config", cfg_path, "--num_workers", "0"])
        run([py, f"{lang}/inference.py", "--config", cfg_path])
        res = c["path"]["result_path"]
        ids = [l.split("\t")[0] for l in open(c["inference"]["test_sentences"], encoding="utf-8") if l.strip()]
        expected = ["training_loss.png", "validation_loss.png"] + \
                   [f"test_generated_audios/{i}.wav" for i in ids] + \
                   [f"variance_adaptor_plots/{k}/{i}.png" for k in ("duration", "pitch", "energy") for i in ids]
        missing = [e for e in expected if not os.path.exists(os.path.join(res, e))]
        assert not missing, f"{lang}: missing outputs {missing}"
        print(f"\n[OK] {lang}: {len(expected)} expected outputs present in {res}\n")
    if not args.keep and not args.work:
        shutil.rmtree(work)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
