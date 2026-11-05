# Design

## 1. Design principles
1. **One engine, two languages.** Everything language-specific is either in `fs2/text/<lang>.py` or in `<Lang>/config.yaml`; the rest is shared and tested once.
2. **Faithful where it matters.** Architecture, losses, optimiser and mel settings follow the paper and the reference repo so results are comparable; changes are limited to what Telugu and the datasets require, plus bug fixes.
3. **Self-describing artefacts.** Checkpoints carry symbols/speakers/stats; every plot has a JSON twin; every data choice is written to `data_selection.json`.
4. **Fail loudly, skip gracefully.** Bad audio files are skipped with a counted reason (`preprocess_summary.json → reasons`), non-finite losses skip the batch, missing outputs mark report sections "pending".

## 2. Text front-end design
### English
`raw → unicode quotes → abbreviations → numbers (time ▸ date ▸ money ▸ % ▸ ordinals ▸ decimals ▸ years/cardinals) → acronym split → g2p_en → punctuation→sp`
- Order matters: times before plain numbers (10:45), day-month before cardinals (12 November → twelfth).
- Years 1100–2099 read in pairs (twenty twenty six); 2000–2009 as cardinals.
- Acronyms (2–5 capitals) spelled with a fixed ARPAbet letter table, so "AM" ≠ the word "am".

### Telugu
`NFC → Telugu digits→ASCII → times → decimals → numbers+suffix → Latin letters→Telugu letter names → akshara parser`
- Akshara parser: consonant + (virama → bare | matra → vowel | none → /a/); independent vowels; anusvara → nasal of next consonant's place (velar ng, palatal ny, retroflex nn, dental n, labial m; else m); visarga → h.
- Ordinal suffix -వ attaches by replacing the final -ు/-ి of the number word (పన్నెండు → పన్నెండవ).
- Oblique numeral forms before a following number (రెండు వేల ఇరవై ఆరు, నూట ఒకటి).
- Phones are ASCII (aa, tt, dd, nn, sh, ss…) so plots and MFA lexicons are script-independent.

## 3. Alignment design
Soft attention A = softmax(−τ‖q_t − k_j‖² + log prior) between mel frames and phoneme embeddings;
forward-sum (CTC) loss maximises the likelihood of all monotonic paths; MAS extracts the single best
monotonic path whose column sums are the integer durations; a bin loss (from step 6 k) sharpens A toward
that path. Beta-binomial prior (scaling 1.0) makes training start near the diagonal.

## 4. Data selection design
Quality filter → group by normalised text → per-speaker quotas proportional to available audio → seeded
greedy fill (validation first). Rationale: validation covers the same speakers (needed for speaker
embeddings), contains no sentence seen in training, and stays reproducible.

## 5. Output / UX design
- CLI only. Each language: three commands, all defaulting to its `config.yaml`.
- Progress: tqdm for preprocessing; one log line per 100 steps; validation line per 1 k steps.
- Results folder exactly as the assignment specifies; extras (`*_values.json`, `*_mel.png`, `train_vs_validation_loss.png`, `inference_summary.json`, `data_selection.json`) never replace required files.

## 6. Visual design of figures
| Figure | Encoding |
|---|---|
| Loss | small multiples, one panel per loss term, shared x = step, log-y only when range > 30×; raw (faint) + smoothed (solid) |
| Duration | bar per phoneme (blue), pauses grey, y = frames with ms conversion in the label, total seconds in title |
| Pitch | top: frame-level F0 after length regulation with phoneme boundaries; bottom: per-phoneme predictor output; dashed orange = training mean |
| Energy | same layout as pitch in aqua |
Palette: blue #2a78d6, orange #eb6834, aqua #1baf7a, ink #0b0b0b, muted #52514e, grid #e4e3df; no dual axes.

## 7. Report design
A4, Inter/Noto Sans Telugu, sections 1–7 mirror the assignment's bullet list. Each auto-generated
analysis states the measured numbers it is based on, so the reader can check it against the plots.
