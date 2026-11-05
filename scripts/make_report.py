"""Build Report/Technical_Report.pdf (and .html) from the project's results.

    python scripts/make_report.py                 # after preprocessing, training and inference
    python scripts/make_report.py --html_only     # skip the PDF step

Everything measurable is computed from the files in Results/<Lang>/ (data selection, loss logs,
inference summaries, variance-adaptor values). Listening observations, which only a human can
make, are read from Report/observations_<Lang>.md (one "## <ID>" section per test sentence).
Sections whose inputs do not exist yet are marked "pending".
PDF rendering uses headless Chromium via Playwright (`pip install playwright && playwright install chromium`).
"""
import argparse
import csv
import html
import json
import os
import re
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
REPORT = os.path.join(ROOT, "Report")
LANGS = ["English", "Telugu"]
FRAME_MS = 256 / 22050 * 1000

SENTENCE_PURPOSE = {
    "E1": "Very long rare word (28 letters) - tests G2P on OOV-like morphology and duration stability over a long phoneme run.",
    "E2a": "Heteronym 'conduct' used as a NOUN - stress should fall on the first syllable (CON-duct).",
    "E2b": "Heteronym 'conduct' used as a VERB - stress should fall on the second syllable (con-DUCT).",
    "E3": "Word broken by a stray space ('rhyth m') - tests robustness to malformed input text.",
    "E4": "Time, date and year normalisation (10:45 AM, 12 November 2026) plus a code-mixed Telugu word ('andaroo').",
    "T1": "Long transliterated scientific compound (న్యూరోఫార్మకోలాజికల్) with conjunct consonants.",
    "T2a": "Homograph పట్టు = 'grip / hold' (on a rope).",
    "T2b": "Homograph పట్టు = 'silk' (silk saree) - same spelling, meaning from context.",
    "T3": "Word broken by a stray space ('అడ విలో' for అడవిలో) - Telugu counterpart of E3.",
    "T4": "Digits inside Telugu text: year (2026), ordinal with suffix (12వ) and clock time (10:45).",
}
FOCUS_WORDS = {"E2a": "conduct", "E2b": "conduct", "E3": None, "T2a": "పట్టు", "T2b": "పట్టు"}


# ------------------------------------------------------------------ helpers
def jload(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def read_csv_log(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out = {"step": np.array([int(r["step"]) for r in rows])}
    for k in rows[0]:
        if k != "step":
            out[k] = np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    return out


def esc(s):
    return html.escape(str(s))


def img(rel, caption=None, cls="fig"):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return f'<p class="pending">Figure pending: <code>{esc(rel)}</code> not generated yet.</p>'
    cap = f"<figcaption>{esc(caption)}</figcaption>" if caption else ""
    return f'<figure class="{cls}"><img src="../{esc(rel)}">{cap}</figure>'


def table(headers, rows, cls=""):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def read_observations(lang):
    p = os.path.join(REPORT, f"observations_{lang}.md")
    if not os.path.exists(p):
        return {}
    text = open(p, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"^##\s+(\S+)\s*\n(.*?)(?=^##\s|\Z)", text, re.S | re.M):
        body = m.group(2).strip()
        body = "\n".join(l for l in body.splitlines() if not l.strip().startswith(">"))  # drop prompts
        if body.strip() and "TODO" not in body:
            out[m.group(1)] = body.strip()
    return out


def md_inline(s):
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    paras = [p.strip() for p in s.split("\n\n") if p.strip()]
    out = []
    for p in paras:
        lines = p.splitlines()
        if all(l.strip().startswith(("- ", "* ")) for l in lines):
            out.append("<ul>" + "".join(f"<li>{l.strip()[2:]}</li>" for l in lines) + "</ul>")
        else:
            out.append(f"<p>{' '.join(lines)}</p>")
    return "".join(out)


# ------------------------------------------------------------------ automatic analyses
def analyse_losses(lang):
    tr = read_csv_log(os.path.join(ROOT, "Results", lang, "train_log.csv"))
    va = read_csv_log(os.path.join(ROOT, "Results", lang, "val_log.csv"))
    if tr is None or va is None or len(va["step"]) < 1:
        return None, []
    notes = []
    vm = va["mel"] + va["postnet_mel"]
    best_i = int(np.nanargmin(vm))
    best_step, best = int(va["step"][best_i]), float(vm[best_i])
    conv_i = int(np.argmax(vm <= best * 1.05))
    final_train = float(np.nanmean(tr["total"][-max(1, len(tr["total"]) // 20):]))
    first_train = float(tr["total"][0])
    notes.append(f"Training loss falls from {first_train:.2f} to {final_train:.2f} (mean of the last 5% of logged steps) "
                 f"over {int(tr['step'][-1])} steps.")
    notes.append(f"Validation mel + PostNet L1 reaches its minimum of {best:.3f} at step {best_step}; "
                 f"it is within 5% of that minimum from step {int(va['step'][conv_i])}, which we take as the convergence point.")
    if len(vm) >= 3:
        last = float(vm[-1])
        tr_mel = tr["mel"] + tr["postnet_mel"]
        late = tr["step"] >= best_step
        tr_drop = float(np.nanmean(tr_mel[late][:5]) - np.nanmean(tr_mel[late][-5:])) if late.sum() >= 10 else 0.0
        if last > best * 1.03 and tr_drop > 0:
            notes.append(f"After step {best_step} the validation mel loss rises by {100 * (last / best - 1):.1f}% while the "
                         f"training mel loss keeps decreasing - a sign of over-fitting on the ~5 h training set; "
                         f"the best-validation checkpoint (step {best_step}) is used for inference.")
        else:
            notes.append("Validation loss does not rise noticeably after its minimum: no significant over-fitting "
                         "is visible within the trained steps.")
        gap = float(np.nanmean(va["mel"][-3:]) / max(np.nanmean(tr["mel"][-max(3, len(tr['mel']) // 20):]), 1e-6))
        notes.append(f"Final validation/training mel-loss ratio is {gap:.2f} (1.0 = no generalisation gap).")
    t = tr["total"]
    if len(t) > 20:
        med = np.array([np.nanmedian(t[max(0, i - 10):i + 1]) for i in range(len(t))])
        spikes = int(np.sum(t[10:] > 1.5 * med[10:]))
        notes.append(f"{spikes} logged training steps exceed 1.5x the running median" +
                     (" - occasional instability, typically from long or badly aligned batches." if spikes
                      else " - training was numerically stable."))
    for k, name in (("duration", "duration"), ("pitch", "pitch"), ("energy", "energy")):
        if k in va:
            notes.append(f"Final validation {name} loss: {float(va[k][-1]):.3f} (best {float(np.nanmin(va[k])):.3f}).")
    if "ctc" in va:
        notes.append(f"Aligner forward-sum (CTC) loss on validation ends at {float(va['ctc'][-1]):.3f}; "
                     "a steadily decreasing value indicates the learned alignment became monotonic and sharp.")
    summary = {"best_step": best_step, "best_val_mel": best, "final_train_total": final_train,
               "final_val": {k: float(va[k][-1]) for k in va if k not in ("step", "lr") and not np.isnan(va[k][-1])}}
    return summary, notes


def find_sub(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


def analyse_sentence(lang, sid, vals, summ):
    notes = []
    ph, d = vals["phones"], np.array(vals["durations_frames"])
    f0, en = np.array(vals["pitch_hz"]), np.array(vals["energy"])
    real = [i for i, p in enumerate(ph) if p != "sp"]
    if not real:
        return notes
    rate = len(real) / (d.sum() * FRAME_MS / 1000)
    notes.append(f"{len(ph)} phonemes, {d.sum() * FRAME_MS / 1000:.2f} s of speech, {rate:.1f} phonemes/s; "
                 f"mean phoneme duration {np.mean(d[real]) * FRAME_MS:.0f} ms.")
    top = sorted(real, key=lambda i: -d[i])[:3]
    notes.append("Longest phonemes: " + ", ".join(f"<b>{esc(ph[i])}</b> #{i} ({d[i] * FRAME_MS:.0f} ms)" for i in top) +
                 ". Long phonemes cluster on stressed / phrase-final vowels when the duration predictor works as intended.")
    sp = [i for i, p in enumerate(ph) if p == "sp"]
    if sp:
        notes.append(f"Pause token(s) at position(s) {', '.join(map(str, sp))} predicted as "
                     f"{', '.join(f'{d[i] * FRAME_MS:.0f} ms' for i in sp)} (from punctuation).")
    if lang == "English":
        vow = [i for i in real if re.search(r"\d$", ph[i])]
    else:
        vow = [i for i in real if ph[i] in ("a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au")]
    if len(vow) > 3:
        half = len(vow) // 2
        a, b = np.mean(f0[vow[:half]]), np.mean(f0[vow[half:]])
        notes.append(f"F0 over vowels: mean {np.mean(f0[vow]):.0f} Hz, range {np.min(f0[vow]):.0f}-{np.max(f0[vow]):.0f} Hz; "
                     f"first-half vs second-half mean {a:.0f} -> {b:.0f} Hz "
                     f"({'declination - pitch falls towards the end, typical of a declarative' if b < a - 2 else 'no clear declination'}).")
        notes.append(f"Energy: highest on <b>{esc(ph[int(max(real, key=lambda i: en[i]))])}</b>, lowest on "
                     f"<b>{esc(ph[int(min(real, key=lambda i: en[i]))])}</b>; vowels average "
                     f"{np.mean(en[vow]):.1f} vs consonants {np.mean(en[[i for i in real if i not in vow]]):.1f} "
                     "(vowels should carry more energy than obstruents).")
    # sentence-specific checks
    from fs2.text import get_frontend
    fe = get_frontend("en" if lang == "English" else "te")
    w = FOCUS_WORDS.get(sid)
    if w:
        sub = fe.lexicon_entry(w)
        if lang == "English":
            # locate the POS-disambiguated pronunciation of 'conduct' inside the sentence
            cands = [["K", "AA1", "N", "D", "AH0", "K", "T"], ["K", "AA0", "N", "D", "AH1", "K", "T"],
                     ["K", "AH0", "N", "D", "AH1", "K", "T"]]
            for c in cands:
                j = find_sub(ph, c)
                if j >= 0:
                    v1, v2 = j + 1, j + 4
                    stress = "first syllable (noun)" if c[1].endswith("1") else "second syllable (verb)"
                    notes.append(f"'conduct' was transcribed as <code>{' '.join(c)}</code> (G2P POS tagging -> stress on {stress}). "
                                 f"Predicted vowel durations {d[v1] * FRAME_MS:.0f} ms vs {d[v2] * FRAME_MS:.0f} ms, "
                                 f"F0 {f0[v1]:.0f} vs {f0[v2]:.0f} Hz, energy {en[v1]:.1f} vs {en[v2]:.1f}: the stressed "
                                 f"vowel {'is' if (d[v1] > d[v2]) == c[1].endswith('1') else 'is NOT'} the longer one.")
                    break
        else:
            j = find_sub(ph, sub)
            if j >= 0:
                seg = range(j, j + len(sub))
                notes.append(f"'{esc(w)}' = <code>{' '.join(sub)}</code>: {sum(d[i] for i in seg) * FRAME_MS:.0f} ms total, "
                             f"geminate tt+tt {(d[j + 2] + d[j + 3]) * FRAME_MS:.0f} ms, mean F0 {np.mean(f0[list(seg)]):.0f} Hz. "
                             "Both readings share one phoneme string, so any difference between T2a and T2b comes only from "
                             "context through the encoder self-attention - the model cannot know the meaning.")
    if sid == "E3":
        j = find_sub(ph, ["EH1", "M"])
        if j >= 0:
            notes.append(f"The stray 'm' is spelled as the letter name <code>EH1 M</code> ({(d[j] + d[j + 1]) * FRAME_MS:.0f} ms) "
                         "and 'rhyth' is guessed by the neural G2P: the input error is faithfully reproduced as an extra "
                         "syllable - text pre-processing (spell-checking / joining fragments) is needed to repair it.")
    if sid == "T3":
        j = find_sub(ph, ["a", "dd", "a", "v", "i", "l", "oo"])
        if j >= 0:
            notes.append("'అడ విలో' gives the same phoneme string as అడవిలో because spaces are dropped by the "
                         "front-end, so the split is harmless for Telugu (unlike E3). Remaining risk: a slight "
                         f"lengthening of the word-final 'a' ({d[j + 2] * FRAME_MS:.0f} ms) if the model learned word-boundary cues.")
    if sid in ("E4", "T4"):
        notes.append(f"Normalised text: <i>{esc(vals['normalized_text'])}</i>.")
    return notes


# ------------------------------------------------------------------ sections
ARCH_SVG = """<svg viewBox="0 0 720 300" class="arch" xmlns="http://www.w3.org/2000/svg" font-family="Inter, Arial" font-size="12">
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#52514e"/></marker></defs>
<g fill="none" stroke="#52514e" stroke-width="1.3">
<rect x="10" y="125" width="95" height="44" rx="6" fill="#eef4fc"/><rect x="130" y="125" width="95" height="44" rx="6" fill="#eef4fc"/>
<rect x="250" y="40" width="210" height="215" rx="8" fill="#fbfaf7" stroke-dasharray="4 3"/>
<rect x="265" y="62" width="180" height="34" rx="6" fill="#fff4ee"/><rect x="265" y="106" width="180" height="34" rx="6" fill="#fff4ee"/>
<rect x="265" y="150" width="180" height="34" rx="6" fill="#fff4ee"/><rect x="265" y="200" width="180" height="34" rx="6" fill="#eaf7f1"/>
<rect x="485" y="125" width="95" height="44" rx="6" fill="#eef4fc"/><rect x="605" y="125" width="105" height="44" rx="6" fill="#eef4fc"/>
<path d="M105 147 H128" marker-end="url(#a)"/><path d="M225 147 H263" marker-end="url(#a)"/><path d="M445 147 H483" marker-end="url(#a)"/>
<path d="M580 147 H603" marker-end="url(#a)"/><path d="M355 96 V104" marker-end="url(#a)"/><path d="M355 140 V148" marker-end="url(#a)"/><path d="M355 184 V198" marker-end="url(#a)"/>
<rect x="130" y="230" width="95" height="40" rx="6" fill="#f2f1ee"/><path d="M225 250 H263" marker-end="url(#a)"/>
</g>
<g fill="#0b0b0b" text-anchor="middle">
<text x="57" y="144">Phonemes</text><text x="57" y="160" fill="#52514e">(G2P front-end)</text>
<text x="177" y="144">Encoder</text><text x="177" y="160" fill="#52514e">4 FFT blocks</text>
<text x="355" y="56" fill="#52514e">Variance adaptor</text>
<text x="355" y="84">Duration predictor</text><text x="355" y="128">Pitch predictor + embed</text>
<text x="355" y="172">Energy predictor + embed</text><text x="355" y="222">Length regulator</text>
<text x="532" y="144">Decoder</text><text x="532" y="160" fill="#52514e">6 FFT blocks</text>
<text x="657" y="144">Mel + PostNet</text><text x="657" y="160" fill="#52514e">-> HiFi-GAN</text>
<text x="177" y="248">Learned aligner</text><text x="177" y="262" fill="#52514e">(train only)</text>
<text x="70" y="195" fill="#52514e">+ speaker emb.</text>
</g></svg>"""


def section_implementation():
    mods = [
        ["Text front-end", "English cleaners + g2p_en (CMUdict)", "English: number/time/date normaliser written for the project, acronym spelling, g2p_en with POS-based heteronym disambiguation. Telugu: new rule-based akshara G2P (inherent vowel, virama, matras, homorganic anusvara), Telugu number words incl. ordinals (12వ) and times."],
        ["Symbols", "one global ARPAbet + pinyin table", "per-language symbol table saved with every checkpoint (72 English / 52 Telugu symbols incl. pad, sp, spn)."],
        ["Durations", "Montreal Forced Aligner TextGrids", "default: learned aligner trained jointly (beta-binomial prior + forward-sum CTC loss + MAS + bin loss), because MFA has no pretrained Telugu model. MFA remains available (scripts/run_mfa.sh, alignment.method: mfa)."],
        ["Pitch / energy", "phoneme-level averages written to disk", "frame-level F0 / energy stored; phoneme averages computed on the fly from whichever durations are used. Bins from robust quantiles of training statistics."],
        ["Variance adaptor", "-", "energy predictor now uses energy control (reference passes the pitch control - bug); vectorised length regulator; minimum one frame per phoneme at inference."],
        ["Data", "LJSpeech / AISHELL3 / LibriTTS loaders", "Svarah and IndicTTS-Telugu loaders (Hugging Face, manifests, Festival txt.done.data), quality filtering and speaker-stratified, text-disjoint 5 h / 1 h split."],
        ["Training", "separate preprocess/model/train YAMLs", "single config.yaml per language, best-validation checkpointing, CSV logs, automatic loss plots, resume, optional fp16 and early stopping."],
        ["Inference", "synthesize.py", "batch synthesis of the test sentences with duration/pitch/energy plots and numeric JSON per sentence; checkpoint is self-contained (symbols, speakers, statistics)."],
    ]
    return f"""
<h2>1. Implementation approach</h2>
<p>The implementation starts from <b>ming024/FastSpeech2</b> (PyTorch), the reference implementation named in the assignment, and follows the
FastSpeech 2 paper (Ren et al., ICLR 2021): a non-autoregressive Transformer that maps phonemes to mel-spectrograms with explicit
duration, pitch and energy predictors (the <i>variance adaptor</i>), followed by a pretrained HiFi-GAN vocoder.
One shared package, <code>fs2/</code>, holds the model and pipeline; <code>English/</code> and <code>Telugu/</code> contain the
language-specific <code>preprocess.py</code>, <code>train.py</code>, <code>inference.py</code> and <code>config.yaml</code>, so the two
models are trained completely independently.</p>
{ARCH_SVG}
<p class="caption">Figure 1. FastSpeech 2 as implemented. Encoder/decoder FFT blocks, PostNet and HiFi-GAN are taken unchanged from the reference repository.</p>
<h3>Components reused from the reference implementation</h3>
<ul><li><b>Transformer</b> (<code>fs2/transformer/</code>): FFT blocks (multi-head self-attention + 1-D conv feed-forward, kernel [9, 1]), sinusoidal positions, 4 encoder / 6 decoder layers, hidden 256, 2 heads; the only edit is that vocabulary size comes from the language's symbol table.</li>
<li><b>PostNet</b>: 5 conv layers, 512 channels, residual refinement of the mel.</li>
<li><b>Variance predictors</b>: 2 x (Conv1D-ReLU-LayerNorm-Dropout) + Linear, 256 quantisation bins for pitch and energy embeddings.</li>
<li><b>Loss</b>: L1 on mel before and after PostNet, MSE on log-duration, pitch and energy.</li>
<li><b>Optimiser</b>: Adam (0.9, 0.98, eps 1e-9) with Noam warm-up (4000 steps) and step annealing; gradient clipping 1.0.</li>
<li><b>Vocoder</b>: the reference repo's pretrained HiFi-GAN <i>universal</i> generator (22.05 kHz, 80 mels, fmax 8 kHz); features are computed with the same STFT settings so no vocoder fine-tuning is needed.</li></ul>
<h3>Modifications for English and Telugu</h3>
{table(["Area", "Reference repo", "This project"], mods)}
"""


def section_data():
    out = ["<h2>2. Datasets, subsampling and preprocessing</h2>",
           """<p><b>English - Svarah</b> (AI4Bharat): 9.6 h of Indian-accented English from 117 speakers across 65 districts, read and
extempore speech, released as a single split. No speaker id is published, so a stable pseudo-speaker id is derived from each
utterance's speaker metadata (gender, age group, native language, state, district, qualification, job, domain).
<b>Telugu - IndicTTS</b> (IIT Madras): studio recordings of one male (4.47 h) and one female (4.27 h) speaker at 48 kHz.
Because 6 h cannot come from a single Telugu speaker, both speakers are used with a speaker embedding.</p>
<p><b>Selection strategy (both languages, <code>fs2/selection.py</code>, seed 1234).</b> (1) keep 1-11 s utterances with a non-empty transcript
and no annotation tags; (2) drop the 1% slowest and 1% fastest utterances by characters per second - this removes truncated audio and
transcripts that do not match; (3) group utterances with identical normalised text and always place a group in one split, so no sentence
occurs in both train and validation (IndicTTS speakers share prompts); (4) give every speaker a share of the 1 h validation and 5 h
training budgets proportional to their share of the filtered audio, and fill validation first, then training, visiting groups in seeded
random order. Assertions in the code guarantee zero utterance and zero text overlap.</p>
<p><b>Preprocessing.</b> Resample to 22.05 kHz mono, trim leading/trailing silence (30 dB English, 35 dB Telugu), peak-normalise;
text normalisation and G2P (Section 1); 80-bin log-mel (n_fft 1024, hop 256, fmax 8 kHz); F0 with WORLD DIO + StoneMask, unvoiced
frames linearly interpolated; frame energy = L2 norm of the STFT magnitude; utterances longer than 1000 frames or with fewer frames
than phonemes are discarded; pitch/energy are z-normalised with training-set statistics.</p>"""]
    rows = []
    for lang in LANGS:
        sel = jload(os.path.join(ROOT, "Results", lang, "data_selection.json"))
        pre = jload(os.path.join(ROOT, "Results", lang, "preprocess_summary.json"))
        if not sel:
            rows.append([lang, "pending", "", "", "", "", ""])
            continue
        th = sel.get("trimmed_hours", {})
        rows.append([lang, f"{sel['input_utterances']} / {sel['input_hours']:.2f} h",
                     f"{sel['after_rate_filter']} / {sel['filtered_hours']:.2f} h", sel["n_speakers"],
                     f"{sel['train_utterances']} / {sel['train_hours']:.2f} h ({th.get('train', '-')} h trimmed)",
                     f"{sel['val_utterances']} / {sel['val_hours']:.2f} h ({th.get('val', '-')} h trimmed)",
                     (f"train {pre['train']['utterances']}, val {pre['val']['utterances']}" if pre else "pending")])
    out.append(table(["Language", "Corpus", "After filters", "Speakers", "Train", "Validation", "After feature checks"], rows))
    for lang in LANGS:
        pre = jload(os.path.join(ROOT, "Results", lang, "preprocess_summary.json"))
        if pre:
            st = pre["stats"]
            out.append(f"<p class='small'>{lang}: F0 mean {st['pitch'][2]:.1f} Hz (sd {st['pitch'][3]:.1f}), voiced F0 1-99% range "
                       f"{st['pitch_hz_range_voiced'][0]:.0f}-{st['pitch_hz_range_voiced'][1]:.0f} Hz; energy mean {st['energy'][2]:.2f} "
                       f"(sd {st['energy'][3]:.2f}); discarded: "
                       f"{esc(', '.join(f'{k}={v}' for k, v in pre['reasons'].items() if k != 'ok') or 'none')}.</p>")
    # training config
    rows = []
    for lang in LANGS:
        c = yaml.safe_load(open(os.path.join(ROOT, lang, "config.yaml"), encoding="utf-8"))
        rows.append([lang, c["dataset"]["name"], c["alignment"]["method"], c["train"]["batch_size"], c["train"]["total_step"],
                     c["optimizer"]["warm_up_step"], ", ".join(map(str, c["optimizer"]["anneal_steps"])),
                     f"{c['preprocessing']['pitch']['f0_min']}-{c['preprocessing']['pitch']['f0_max']} Hz",
                     "yes" if c["model"]["multi_speaker"] else "no"])
    out.append("<h3>Training configuration</h3>")
    out.append(table(["Model", "Data", "Durations", "Batch", "Steps", "Warm-up", "LR anneal at", "F0 search", "Speaker emb."], rows))
    out.append("<p class='small'>Model: 35.6 M parameters (incl. 0.5 M aligner). Peak learning rate 9.9e-4 at step 4000, x0.3 at each anneal step. "
               "The checkpoint with the lowest validation mel + PostNet L1 is used for inference.</p>")
    return "\n".join(out)


def section_losses():
    out = ["<h2>3. Training and validation loss</h2>"]
    for lang in LANGS:
        out.append(f"<h3>{lang}</h3>")
        out.append('<div class="row">' + img(f"Results/{lang}/training_loss.png", f"{lang}: training loss terms (raw and smoothed)") + "</div>")
        out.append('<div class="row">' + img(f"Results/{lang}/validation_loss.png", f"{lang}: validation loss terms") + "</div>")
        if os.path.exists(os.path.join(ROOT, "Results", lang, "train_vs_validation_loss.png")):
            out.append(img(f"Results/{lang}/train_vs_validation_loss.png", f"{lang}: train vs validation"))
        summ, notes = analyse_losses(lang)
        out.append("<ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>" if notes else
                   "<p class='pending'>Pending: run training to generate the loss logs.</p>")
    return "\n".join(out)


def section_sentences():
    out = ["<h2>4. Generated speech - observations per test sentence</h2>",
           "<p>Audio files are in <code>Results/&lt;Language&gt;/test_generated_audios/</code>. Measured quantities come from the "
           "model's own predictions; listening notes (pronunciation, naturalness, artefacts) are the author's, recorded in "
           "<code>Report/observations_&lt;Language&gt;.md</code>.</p>"]
    for lang in LANGS:
        inf = jload(os.path.join(ROOT, "Results", lang, "inference_summary.json"))
        obs = read_observations(lang)
        out.append(f"<h3>{lang}</h3>")
        if not inf:
            out.append("<p class='pending'>Pending: run inference.</p>")
            continue
        out.append(f"<p class='small'>Checkpoint step {inf['step']}, speaker <code>{esc(inf['speaker'])}</code>, vocoder {esc(inf['vocoder'])}.</p>")
        rows = []
        for sid, s in inf["sentences"].items():
            rows.append([f"<b>{sid}</b>", f"<span class='src'>{esc(s['text'])}</span>", esc(SENTENCE_PURPOSE.get(sid, "")),
                         f"{s['audio_seconds']} s", md_inline(obs[sid]) if sid in obs else "<span class='pending'>listening notes pending</span>"])
        out.append(table(["ID", "Text", "What it tests", "Length", "Listening observations"], rows, "sent"))
    return "\n".join(out)


def section_variance():
    out = ["<h2>5. Variance-adaptor analysis (duration, F0, energy)</h2>",
           "<p>For each test sentence the plots show the <i>predicted</i> values: duration in frames per phoneme (1 frame = 11.6 ms; "
           "grey bars are pauses), F0 in Hz after de-normalisation, and frame energy. The top panel of the pitch and energy figures is the "
           "frame-level contour the decoder actually receives after length regulation; the lower panel is the per-phoneme predictor output.</p>"]
    for lang in LANGS:
        out.append(f"<h3>{lang}</h3>")
        inf = jload(os.path.join(ROOT, "Results", lang, "inference_summary.json"))
        if not inf:
            out.append("<p class='pending'>Pending: run inference.</p>")
            continue
        for sid in inf["sentences"]:
            vals = jload(os.path.join(ROOT, "Results", lang, "variance_adaptor_plots", f"{sid}_values.json"))
            out.append(f'<div class="sent-block"><h4>{sid} <span class="src">{esc(inf["sentences"][sid]["text"])}</span></h4>')
            out.append(img(f"Results/{lang}/variance_adaptor_plots/duration/{sid}.png"))
            out.append('<div class="row2">' + img(f"Results/{lang}/variance_adaptor_plots/pitch/{sid}.png") +
                       img(f"Results/{lang}/variance_adaptor_plots/energy/{sid}.png") + "</div>")
            if vals:
                out.append("<ul>" + "".join(f"<li>{n}</li>" for n in analyse_sentence(lang, sid, vals, inf["sentences"][sid])) + "</ul>")
            out.append("</div>")
    return "\n".join(out)


def section_compare():
    rows = []
    for lang in LANGS:
        summ, _ = analyse_losses(lang)
        inf = jload(os.path.join(ROOT, "Results", lang, "inference_summary.json"))
        sel = jload(os.path.join(ROOT, "Results", lang, "data_selection.json"))
        if inf:
            ss = list(inf["sentences"].values())
            rate = np.mean([s["n_phones"] / max(s["audio_seconds"], 1e-6) for s in ss])
            prange = np.mean([s["pitch_hz_max"] - s["pitch_hz_min"] for s in ss])
        rows.append([lang, sel["n_speakers"] if sel else "-",
                     f"{summ['best_val_mel']:.3f} @ {summ['best_step']}" if summ else "pending",
                     f"{summ['final_val'].get('duration', float('nan')):.3f}" if summ else "-",
                     f"{summ['final_val'].get('pitch', float('nan')):.3f}" if summ else "-",
                     f"{rate:.1f}" if inf else "-", f"{prange:.0f} Hz" if inf else "-"])
    obs = {l: read_observations(l).get("comparison", "") for l in LANGS}
    extra = "".join(md_inline(v) for v in obs.values() if v)
    return f"""
<h2>6. English vs Telugu - comparison, limitations, improvements</h2>
{table(["Model", "Speakers", "Best val mel+PostNet L1", "Val duration", "Val pitch", "Phonemes/s (tests)", "Mean F0 range / sentence"], rows)}
<p><b>Data.</b> Telugu has two studio speakers reading prepared text; Svarah spreads 6 h over ~117 speakers with varied microphones,
accents and partly extempore speech, so each English speaker contributes only a few minutes. Expect the Telugu model to reach a lower
mel loss and sound cleaner and more consistent, while the English model averages over speakers (flatter prosody, occasional
speaker drift) - a property of the data rather than the architecture.</p>
<p><b>Front-end.</b> Telugu spelling is close to phonemic, so the rule-based G2P is reliable for native words; its weak points are loan
words written in Telugu script (T1: schwa deletion and English vowel qualities are not modelled) and the homograph పట్టు (T2), which no
grapheme-level model can resolve. English depends on CMUdict plus a neural G2P: it handles the heteronym pair (E2) through POS tagging,
but must guess OOV words (E1 is in CMUdict; 'rhyth', 'andaroo' are not) and is defeated by the malformed token in E3, while the same
error in Telugu (T3) is harmless because spaces are not pronounced.</p>
<p><b>Numbers.</b> Both front-ends expand the date/time sentences (E4, T4) deterministically; Telugu needs inflected forms
(వేల vs వేలు, the ordinal పన్నెండవ), which a naive digit-to-word mapping gets wrong.</p>
<p><b>Limitations.</b> ~5 h per language is about a fifth of LJSpeech, so expect some mumbling on long or rare words, limited prosodic
variety and occasional duration errors; the universal HiFi-GAN is not fine-tuned to these voices (slight buzz/metallic artefacts);
learned alignments can be imprecise on very short phonemes; global pitch normalisation is shared between male and female voices.</p>
<p><b>Improvements.</b> (1) fine-tune HiFi-GAN on ground-truth-aligned mels of each corpus; (2) per-speaker pitch normalisation;
(3) a Telugu lexicon for English loan words and a context-aware homograph model; (4) input text validation (merge stray fragments,
E3); (5) pretrain on a larger multilingual Indic/English set and fine-tune on the 5 h subsets; (6) MFA durations for English to compare
with the learned aligner; (7) for Svarah, restrict training to the cleanest read-speech speakers or add a reference encoder for speaker/style.</p>
{extra}"""


def section_repro():
    req = open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8").read()
    return f"""
<h2>7. Reproducibility</h2>
<p>Python 3.9-3.13, PyTorch 2.x (CUDA recommended; CPU works for inference). Dependencies (<code>requirements.txt</code>):</p>
<pre>{esc(req.strip())}</pre>
<pre>pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('averaged_perceptron_tagger_eng','cmudict')]"
huggingface-cli login                     # Svarah requires accepting its terms on Hugging Face

python scripts/smoke_test.py              # 5-minute end-to-end check on synthetic data

python English/preprocess.py              # select 5 h / 1 h, extract features
python English/train.py                   # -> English/checkpoints/best.pth.tar, Results/English/*loss.png
python English/inference.py               # -> Results/English/test_generated_audios, variance_adaptor_plots

python Telugu/preprocess.py               # or: --source festival --corpus_path /path/IndicTTS/Telugu
python Telugu/train.py
python Telugu/inference.py

python scripts/make_report.py             # rebuild this report</pre>
<p>All hyper-parameters live in <code>English/config.yaml</code> and <code>Telugu/config.yaml</code>; seeds are fixed (1234) for data selection,
shuffling and initialisation. Checkpoints embed the symbol table, speaker table and normalisation statistics, so
<code>inference.py</code> needs only the checkpoint and the config.</p>"""


CSS = """
@font-face { font-family: 'Noto Sans Telugu'; src: url('fonts/NotoSansTelugu-Regular.ttf'); }
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Inter','Segoe UI',Arial,'Noto Sans Telugu',sans-serif; color:#0b0b0b; font-size:10.5pt; line-height:1.45; }
h1 { font-size:20pt; margin:0 0 4px; } h2 { font-size:14pt; margin:22px 0 8px; border-bottom:1px solid #d9d8d3; padding-bottom:3px; break-after:avoid; }
h3 { font-size:11.5pt; margin:14px 0 6px; break-after:avoid; } h4 { font-size:10.5pt; margin:10px 0 4px; }
.sub { color:#52514e; margin-bottom:14px; } .small, .caption { font-size:9pt; color:#52514e; } .pending { color:#b45309; font-style:italic; }
table { border-collapse:collapse; width:100%; font-size:8.8pt; margin:6px 0 10px; break-inside:auto; }
th, td { border:1px solid #e0dfda; padding:4px 6px; vertical-align:top; text-align:left; } th { background:#f4f3ef; }
tr { break-inside:avoid; } .src { font-family:'Noto Sans Telugu','Inter',sans-serif; color:#52514e; font-weight:normal; }
figure { margin:6px 0; break-inside:avoid; } figure img { width:100%; } figcaption { font-size:8.5pt; color:#52514e; }
.row2 { display:grid; grid-template-columns:1fr 1fr; gap:8px; } .sent-block { break-inside:avoid-page; margin-bottom:10px; }
pre { background:#f6f5f1; padding:8px; font-size:8.3pt; white-space:pre-wrap; } code { font-size:9pt; background:#f4f3ef; padding:0 3px; }
svg.arch { width:100%; height:auto; margin:6px 0; } .status { background:#fff7e6; border:1px solid #f3d19c; padding:8px 10px; font-size:9.5pt; }
"""


def build(html_only=False):
    trained = {l: os.path.exists(os.path.join(ROOT, "Results", l, "inference_summary.json")) for l in LANGS}
    status = ("" if all(trained.values()) else
              "<div class='status'><b>Status:</b> results for " + ", ".join(l for l, t in trained.items() if not t) +
              " are not generated yet; sections marked <i>pending</i> fill in automatically when "
              "<code>scripts/make_report.py</code> is re-run after preprocessing, training and inference.</div>")
    body = f"""<h1>FastSpeech 2 Text-to-Speech for English and Telugu</h1>
<div class="sub">Technical report - implementation, training and variance-adaptor analysis. Datasets: Svarah (English), IndicTTS (Telugu).</div>
{status}
{section_implementation()}{section_data()}{section_losses()}{section_sentences()}{section_variance()}{section_compare()}{section_repro()}
<h2>References</h2><p class="small">[1] Y. Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech", ICLR 2021.
[2] C.-M. Chien, ming024/FastSpeech2, GitHub. [3] J. Kong et al., "HiFi-GAN", NeurIPS 2020. [4] R. Badlani et al., "One TTS Alignment
to Rule Them All", ICASSP 2022. [5] AI4Bharat, Svarah: Indian-accented English benchmark, 2023. [6] IIT Madras, Indic TTS database.</p>"""
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>FastSpeech 2 Technical Report</title><style>{CSS}</style></head><body>{body}</body></html>"
    html_path = os.path.join(REPORT, "Technical_Report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(page)
    print("wrote", html_path)
    if html_only:
        return
    from playwright.sync_api import sync_playwright
    pdf_path = os.path.join(REPORT, "Technical_Report.pdf")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto("file://" + html_path, wait_until="networkidle")
        pg.pdf(path=pdf_path, format="A4", print_background=True,
               margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
               display_header_footer=True, header_template="<span></span>",
               footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'>"
                               "<span class='pageNumber'></span> / <span class='totalPages'></span></div>")
        b.close()
    print("wrote", pdf_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html_only", action="store_true")
    build(ap.parse_args().html_only)
