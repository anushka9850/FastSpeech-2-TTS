#!/usr/bin/env bash
# Optional: paper-faithful durations with Montreal Forced Aligner (MFA >= 2.2).
# Default pipeline uses the learned aligner (alignment.method: learned) and does NOT need this.
#
#   conda install -c conda-forge montreal-forced-aligner
#   bash scripts/run_mfa.sh English            # or Telugu
#   # then set `alignment.method: mfa` in <Lang>/config.yaml and run:
#   python <Lang>/preprocess.py --stage features
#
# English: our lexicon is ARPAbet-with-stress (g2p_en) = the english_us_arpa phone set, so the
#          pretrained acoustic model is used.  Set MFA_TRAIN=1 to train on the corpus instead.
# Telugu : there is no pretrained Telugu MFA model, so an acoustic model is trained on the
#          ~6 h corpus with the rule-based Telugu G2P lexicon (fs2/text/telugu.py).
set -euo pipefail
LANG_DIR=${1:?usage: run_mfa.sh English|Telugu}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw/$LANG_DIR"
OUT="$ROOT/data/mfa/$LANG_DIR"
JOBS=${JOBS:-$(nproc)}
mkdir -p "$OUT"

python "$ROOT/$LANG_DIR/preprocess.py" --stage lexicon

if [[ "$LANG_DIR" == "English" && "${MFA_TRAIN:-0}" != "1" ]]; then
  mfa model download acoustic english_us_arpa
  mfa align "$RAW/wavs" "$RAW/lexicon.txt" english_us_arpa "$OUT/TextGrid" -j "$JOBS" --clean --beam 100 --retry_beam 400
else
  mfa train "$RAW/wavs" "$RAW/lexicon.txt" "$OUT/acoustic_model.zip" --output_directory "$OUT/TextGrid" -j "$JOBS" --clean
fi
echo "TextGrids written to $OUT/TextGrid"
