# Tasks

Legend: [x] done in this package · [ ] to do by the author (needs GPU / datasets / ears)

## Phase 0 — Setup (≈30 min)
- [x] Project skeleton in the required folder structure
- [x] Vendor reference transformer + HiFi-GAN (MIT) into `fs2/`
- [ ] `pip install -r requirements.txt`, NLTK data, `huggingface-cli login`
- [ ] `python scripts/smoke_test.py` → "SMOKE TEST PASSED"

## Phase 1 — Front-ends
- [x] English normaliser: times, dates, years, ordinals, currency, decimals, %, abbreviations, acronyms
- [x] English G2P via g2p_en (heteronyms by POS; neural OOV)
- [x] Telugu G2P: inherent vowel, virama, matras, anusvara assimilation, visarga, Latin letters
- [x] Telugu numbers: cardinals to crores, oblique forms (వేల/వందల/నూట), ordinal -వ, clock time
- [x] Test sentences transcribed from the rendered PDF (not the broken text layer)

## Phase 2 — Data (≈1–2 h per language, mostly download)
- [x] Loaders: HF (lazy, no audio decoder), manifest, Festival
- [x] Selection: filters, text-disjoint groups, speaker-stratified 5 h/1 h
- [ ] `python English/preprocess.py` → check `Results/English/data_selection.json`
- [ ] `python Telugu/preprocess.py` → check `Results/Telugu/data_selection.json`
- [ ] Spot-listen 5 exported wavs per language in `data/raw/<Lang>/wavs/`

## Phase 3 — Model & training (≈6–8 h GPU per language)
- [x] Variance adaptor (fixed e_control), vectorised length regulator
- [x] Learned aligner + forward-sum + bin loss + MAS
- [x] Training loop: Noam LR, clipping, validation, best ckpt, resume, CSV logs, plots
- [ ] `python English/train.py` (watch `logs/English/samples/*_alignment.png` become diagonal by ~5 k steps)
- [ ] `python Telugu/train.py`
- [ ] If val loss rises early: enable `early_stop_patience: 5` or lower `total_step`

## Phase 4 — Inference & analysis
- [x] Batch inference with plots + JSON + summary
- [ ] `python English/inference.py`, `python Telugu/inference.py`
- [ ] Listen; fill `Report/observations_English.md`, `Report/observations_Telugu.md`

## Phase 5 — Report & submission
- [x] Report generator with auto analysis (losses, per-sentence variance)
- [ ] `python scripts/make_report.py` → `Report/Technical_Report.pdf`; read it end to end
- [ ] Git LFS for `best.pth.tar` (or upload + link in README)
- [ ] Push to a **public** GitHub repo; open the repo in a private window to confirm access
- [ ] Submit the URL before 07-11-2026

## Optional extensions
- [ ] MFA durations (`scripts/run_mfa.sh`) and compare with learned alignment
- [ ] Fine-tune HiFi-GAN per language
- [ ] Per-speaker pitch normalisation
