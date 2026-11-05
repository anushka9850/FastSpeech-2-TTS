# Architecture

## 1. Pipeline
```
corpus (HF / manifest / Festival)
   │  fs2/corpus.py            → Utterance(uid, speaker, text, duration, load())
   ▼
selection  fs2/selection.py    → filter → text-disjoint groups → speaker-stratified 5 h / 1 h
   ▼
export     fs2/preprocess.py   → data/raw/<Lang>/wavs/<spk>/<uid>.wav|.lab, train.tsv, val.tsv
   ▼  (optional) scripts/run_mfa.sh → data/mfa/<Lang>/TextGrid
features   fs2/preprocess.py   → data/preprocessed/<Lang>/{mel,pitch,energy,[duration]}/<uid>.npy
                                 train.txt, val.txt, stats.json, speakers.json, symbols.json
   ▼
training   fs2/train.py        → <Lang>/checkpoints/{best,<step>}.pth.tar, logs/<Lang>/*.csv,
                                 Results/<Lang>/{training,validation}_loss.png
   ▼
inference  fs2/inference.py    → Results/<Lang>/test_generated_audios/<ID>.wav,
                                 variance_adaptor_plots/{duration,pitch,energy}/<ID>.png, <ID>_values.json
   ▼
report     scripts/make_report.py → Report/Technical_Report.{html,pdf}
```
`English/*.py` and `Telugu/*.py` are thin entry points that pass their own `config.yaml`
(and, for preprocessing, a corpus-specific speaker-id function) to the shared package.

## 2. Model (fs2/model)
| Block | Detail | Source |
|---|---|---|
| Phoneme embedding | vocab = language symbol table (72 EN / 52 TE), d=256 | ref, vocab made configurable |
| Encoder | 4 × FFT block (2-head self-attn, conv FFN 1024, kernels 9/1), dropout 0.2 | ref |
| Speaker embedding | `nn.Embedding(n_speakers, 256)` added to encoder output | ref |
| Duration predictor | 2×(Conv1D k3 → ReLU → LN → Dropout 0.5) → Linear, predicts log(d+1) | ref |
| Pitch predictor + embedding | same predictor; 256 linear bins over normalised F0 | ref |
| Energy predictor + embedding | same; uses **energy** control (bug fixed) | ref + fix |
| Length regulator | `searchsorted` over cumulative durations, batched | rewritten |
| Decoder | 6 × FFT block | ref |
| Mel linear + PostNet | 256→80, 5 conv layers 512 ch, residual | ref |
| Learned aligner | text-emb/mel conv projections, L2-distance attention × beta-binomial prior, MAS (numba) | new (Badlani et al. 2022) |
| Vocoder | HiFi-GAN V1 "universal", 22.05 kHz | ref checkpoint |

Parameter count: 35.6 M (aligner 0.5 M).

### Forward pass (training)
1. `src_mask` from phoneme lengths.
2. If learned alignment: soft attention between phoneme embeddings and target mel; hard alignment by
   MAS → integer durations (sum = number of mel frames).
3. Phoneme-level pitch/energy targets = mean of frame-level values inside each phoneme
   (`average_by_duration`).
4. Encoder → (+speaker) → variance adaptor with teacher-forced pitch/energy/duration → decoder → mel → PostNet.

### Inference
Durations = round(exp(pred) − 1) × d_control, minimum 1 frame per real phoneme; pitch/energy from predictors × controls.

## 3. Losses
`L = L1(mel) + L1(postnet) + MSE(log d) + MSE(pitch) + MSE(energy) + 1.0·ForwardSum + w(t)·Bin`,
w(t) ramps 0→1 between steps 6 k and 10 k (learned alignment only).

## 4. Data contracts
- `train.txt` / `val.txt`: `uid|speaker|space-separated phones|normalised text`
- `stats.json`: `pitch: [min_z, max_z, mean_hz, std_hz]`, `energy: [min_z, max_z, mean, std]` (training split only; bins use 0.5/99.5 % quantiles)
- Checkpoint: `{model, optimizer|None, step, config, symbols, speakers, stats, default_speaker}`

## 5. Modifications vs ming024/FastSpeech2
| File in reference | Here | Change |
|---|---|---|
| `text/*` (English cleaners, pinyin) | `fs2/text/` | new EN normaliser, acronym spelling, g2p_en; new Telugu G2P + numbers; per-language symbols |
| `preprocessor/*` | `fs2/corpus.py`, `selection.py`, `preprocess.py` | Svarah/IndicTTS loaders, subsampling, learned-alignment path, frame-level storage, robust stats |
| `transformer/Models.py` | `fs2/transformer/Models.py` | `n_src_vocab = config["n_symbols"]`, relative import |
| `model/modules.py` | `fs2/model/modules.py` | e_control bug fix, vectorised LR, on-the-fly averaging, min 1 frame |
| `model/fastspeech2.py` | same | aligner integration, dict outputs |
| `model/loss.py` | same | CTC + bin losses |
| `train.py`, `evaluate.py` | `fs2/train.py` | single config, best checkpoint, CSV logs, plots, resume, fp16, early stop |
| `synthesize.py` | `fs2/inference.py` | test-sentence batch, variance plots, JSON, self-contained checkpoint |
| `config/*/{preprocess,model,train}.yaml` | `<Lang>/config.yaml` | merged |
| `hifigan/` | `fs2/hifigan/` | unchanged (universal generator only) |

## 6. Directory ownership
- `fs2/` — library, language-agnostic except `fs2/text/`.
- `English/`, `Telugu/` — configs + entry points + checkpoints.
- `Results/`, `Report/` — generated artefacts (report also reads `Report/observations_*.md`).
- `data/`, `logs/` — generated, git-ignored.
