# Rules (engineering & assignment constraints)

## Assignment rules — never break
1. **Two separate models.** English and Telugu never share a checkpoint, config, symbol table or statistics.
2. **Data budget.** ~5 h training, ~1 h validation per language, selected by `fs2/selection.py` with the seed in the config. Do not hand-pick utterances.
3. **No overlap.** Train and validation must share no utterance and no sentence text (asserted in code). Normalisation statistics come from the training split only.
4. **Test text is sacred.** `test_sentences/*.tsv` holds the exact assignment sentences, including the deliberate errors `rhyth m` (E3) and `అడ విలో` (T3). Never "fix" them in the file; robustness is what is being tested.
5. **Folder structure** in the README is required by the assignment; do not rename `English/`, `Telugu/`, `Results/`, `Report/` or the files inside them.
6. **Document every change** from ming024/FastSpeech2 in `docs/architecture.md` §5 and Report §1.
7. **Report honesty.** Listening observations come from actually listening; numbers come from result files. Smoke-test outputs (synthetic data) must never be copied into `Results/`.

## Code rules
- Shared logic lives in `fs2/`; `English/*.py` and `Telugu/*.py` stay thin wrappers.
- All hyper-parameters in `<Lang>/config.yaml`; no magic numbers in code paths that differ per language.
- Relative paths in configs are resolved against the config's folder (`fs2.utils.load_config`).
- Phone sets must be identical across MFA lexicon, training and inference — always generate them from `fs2/text/<lang>.py`.
- Mel settings (22.05 kHz, 1024/256/1024, 80 mels, fmax 8000, natural log, 1e-5 floor) are fixed by the pretrained HiFi-GAN; changing them requires a new vocoder.
- Checkpoints must remain self-contained (symbols, speakers, stats inside).
- Every pipeline change must keep `python scripts/smoke_test.py` passing.
- Deterministic: seeds for selection, sampler, torch; stats sampling uses `crc32(uid)`.

## Git rules
- `data/`, `logs/` are ignored. Checkpoints via Git LFS (`.gitattributes`) or an external link in README.
- Commit Results (wav/png/json) and `Report/Technical_Report.pdf`.

## Style
- Python ≥ 3.9, PEP 8, 120-char lines, docstrings at module top explaining *why*.
- Plots: one y-axis per panel, recessive grid, blue = train/predicted, orange = validation/reference, phoneme labels in romanised phones (no complex-script shaping needed in matplotlib).
