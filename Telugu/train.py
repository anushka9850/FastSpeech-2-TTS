"""Train the Telugu FastSpeech 2 model.

    python Telugu/train.py                      # uses Telugu/config.yaml
    python Telugu/train.py --resume             # continue from the latest step checkpoint
    python Telugu/train.py --total_step 100000 --device cuda
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fs2.train import main  # noqa: E402

if __name__ == "__main__":
    main(os.path.join(HERE, "config.yaml"))
