# Memory — decisions, facts and gotchas

Persistent context for anyone (human or AI assistant) continuing this project.

## Key decisions (and why)
| Date | Decision | Reason |
|---|---|---|
| 2026-10-31 | Base on ming024/FastSpeech2, vendor `transformer/` and `hifigan/` unchanged | assignment names it; HiFi-GAN universal checkpoint ships with the repo |
| 2026-10-31 | Learned aligner is the default duration source | MFA has no pretrained Telugu model; jointly learned alignment works without external tools; same path for both languages keeps the comparison fair |
| 2026-10-31 | Keep MFA as an option (`alignment.method: mfa`) | paper-faithful baseline; lexicons are generated from our own G2P so phone sets match |
| 2026-10-31 | Store frame-level F0/energy, average per phoneme on the fly | one code path for MFA and learned durations |
| 2026-10-31 | Multi-speaker for both languages | Telugu: no single speaker has 6 h; Svarah: 117 speakers ~5 min each |
| 2026-10-31 | Svarah pseudo-speaker = hash of 8 metadata fields | dataset has no speaker-id column |
| 2026-10-31 | Text-disjoint split | IndicTTS male/female read the same prompts → same sentence could leak across splits |
| 2026-10-31 | Phoneme labels in plots are romanised phones | matplotlib cannot shape Telugu script |
| 2026-10-31 | Report built from HTML via headless Chromium with bundled Noto Sans Telugu | correct Telugu shaping in the PDF |

## Facts
- Svarah: HF `ai4bharat/Svarah`, split `test`, 6,656 rows, 9.6 h, 117 speakers; columns `audio_filepath, duration, text, gender, age-group, primary_language, native_place_state, native_place_district, highest_qualification, job_category, occupation_domain`. Needs HF login.
- IndicTTS Telugu: HF `SPRINGLab/IndicTTS_Telugu`, split `train`, 48 kHz; male 4.47 h, female 4.27 h. Original IIT-M release: `txt.done.data` Festival format.
- HiFi-GAN universal: 22.05 kHz, 80 mel, n_fft 1024, hop 256, fmax 8000.
- 1 frame = 256 / 22050 s = 11.61 ms. `max_seq_len` 1000 frames = 11.6 s → max utterance 11 s.
- Model 35.6 M params; best ckpt ≈ 143 MB; step ckpt with Adam ≈ 430 MB.
- Symbols: English 72 (69 ARPAbet-with-stress + pad/sp/spn), Telugu 52.
- CPU speed: ~5 s/step (batch 4, 2 cores) — train on GPU.

## Test sentence notes
- PDF text layer garbles Telugu (vowel signs reordered, e.g. "విశ్లషే ించారు"); the TSVs were transcribed from the rendered page: విశ్లేషించారు, వచ్చింది, నిశ్శబ్దంగా, సిద్ధంగా, న్యూరోఫార్మకోలాజికల్ (one word), పరిశోధనలో నూతన.
- Deliberate: E3 "rhyth m", T3 "అడ విలో" (double space in PDF, normalised to one — spaces are not pronounced).
- E2 heteronym: g2p_en gives noun `K AA1 N D AH0 K T`, verb `K AA0 N D AH1 K T`.
- E4 normalises to "ten forty five AM … twelfth November twenty twenty six"; AM spelled `EY1 EH1 M`.
- T4 normalises to "రెండు వేల ఇరవై ఆరు … పన్నెండవ … పది నలభై ఐదు గంటలకు".

## Gotchas
- NLTK behind a proxy may need `NLTK_ALLOW_PROXIED_URLOPEN=1` for `nltk.download`.
- `datasets` ≥ 4 decodes audio with torchcodec; we cast to `Audio(decode=False)` and read bytes with soundfile.
- The decoder truncates sequences longer than `max_seq_len` during training — keep `max_duration` ≤ 11 s.
- MAS needs frames ≥ phonemes; such utterances are filtered (`fewer_frames_than_phones`).
- `torch.load(..., weights_only=False)` is required for our checkpoints (they contain dicts/lists).
- Reference repo bug: energy predictor used `p_control`; fixed here.

## Status (at packaging time)
- Code complete; smoke test passes for both languages (CPU, synthetic data) incl. HiFi-GAN loading.
- Learning check: overfitting one batch of 4 synthetic utterances for 250 steps drives mel L1 2.1 → 0.41,
  duration 5.7 → 0.04, and the predicted mel reproduces the target's phoneme structure; the learned
  alignment is near-diagonal after ~200 steps. Expect a "blurry average spectrum" plateau (mel L1 ≈ 2)
  for the first few hundred steps of real training — it is normal and breaks once alignment sharpens.
- Not done here: real-data preprocessing, GPU training, listening notes, final PDF with results.
