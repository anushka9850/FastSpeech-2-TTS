# PRD — FastSpeech 2 TTS for English and Telugu

## 1. Problem
Implement, train and evaluate FastSpeech 2 separately for English and Telugu, starting from the
reference implementation (ming024/FastSpeech2) and the paper, and deliver code, checkpoints,
generated audio, loss curves, variance-adaptor plots and a technical report in a public GitHub
repository. Deadline: **07-11-2026**.

## 2. Goals
| # | Goal | Measure of success |
|---|------|--------------------|
| G1 | Two independent models (EN, TE) | two configs, two checkpoint folders, no shared weights |
| G2 | ~5 h train / ~1 h validation per language, no overlap | `Results/<Lang>/data_selection.json` shows hours; code asserts 0 utterance and 0 text overlap |
| G3 | Reproducible pipeline | `preprocess.py → train.py → inference.py` per language; fixed seeds; one config file each |
| G4 | Telugu support | Telugu G2P + numbers, symbol table, IndicTTS loader; documented changes |
| G5 | Test-sentence synthesis | 5 wavs per language with exact assignment text (E1, E2a, E2b, E3, E4 / T1, T2a, T2b, T3, T4) |
| G6 | Variance-adaptor analysis | duration, pitch (F0), energy plot per sentence per language + numeric JSON |
| G7 | Loss plots | `training_loss.png`, `validation_loss.png` per language |
| G8 | Technical report | `Report/Technical_Report.pdf` with all seven required sections |

## 3. Non-goals
- State-of-the-art quality (5 h of data is small); training a new vocoder; real-time serving; a UI.

## 4. Users
- **Evaluator**: clones repo, reads report, listens to audio, may run inference from checkpoints.
- **Author (student)**: runs preprocessing/training on a GPU, writes listening notes, submits URL.

## 5. Functional requirements
1. **Data**: load Svarah (HF `ai4bharat/Svarah`, split `test`) and IndicTTS Telugu (HF `SPRINGLab/IndicTTS_Telugu` or the IIT-M Festival layout); filter, subsample, split; export 22.05 kHz audio.
2. **Front-ends**: English normalisation (numbers, times, dates, currency, abbreviations, acronyms) + ARPAbet G2P with heteronym handling; Telugu normalisation (digits incl. Telugu digits, ordinals with -వ, times) + rule-based G2P.
3. **Features**: 80-bin log-mel matching HiFi-GAN, WORLD F0, STFT energy, (optional) MFA durations.
4. **Model**: FastSpeech 2 encoder / variance adaptor / decoder / PostNet; speaker embedding; learned aligner.
5. **Training**: Noam schedule, grad clipping, validation every 1 k steps, best-checkpoint selection, resume, CSV logs, automatic loss plots.
6. **Inference**: test-sentence file or single text; pitch/energy/duration controls; HiFi-GAN with Griffin-Lim fallback; plots + JSON.
7. **Report**: generated from results + listening notes, PDF via headless Chromium.
8. **Smoke test**: full pipeline on synthetic data in minutes, no downloads.

## 6. Constraints & assumptions
- GPU strongly recommended for training (CPU ≈ 5 s/step → impractical for 60 k steps).
- Svarah has no speaker-id column → pseudo-speaker from metadata tuple.
- IndicTTS has 2 speakers with < 5 h each → multi-speaker model is mandatory for 6 h.
- PDF text of the Telugu test sentences was verified against the rendered page; extraction artefacts were not copied. Deliberate errors (`rhyth m`, `అడ విలో`) are preserved.
- Checkpoints > 100 MB → Git LFS or external link.

## 7. Acceptance checklist
- [ ] `python scripts/smoke_test.py` passes
- [ ] `Results/English|Telugu/data_selection.json` ≈ 5 h / 1 h
- [ ] Both models trained; `best.pth.tar` present (LFS or link)
- [ ] 2 × 5 wavs + 2 × 15 variance plots + 2 × 2 loss plots
- [ ] `Report/observations_*.md` filled; `Technical_Report.pdf` rebuilt
- [ ] Repo public; URL submitted
