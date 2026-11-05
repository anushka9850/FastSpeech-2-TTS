"""Synthesise the English test sentences (or any text) and draw variance-adaptor plots.

    python English/inference.py                                   # all test sentences -> Results/English/
    python English/inference.py --text "..." --id demo            # one sentence
    python English/inference.py --duration_control 1.1 --pitch_control 1.2
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fs2.inference import main  # noqa: E402

if __name__ == "__main__":
    main(os.path.join(HERE, "config.yaml"))
