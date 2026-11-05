"""Command-line driver for preprocessing (used by English/ and Telugu/ preprocess.py)."""
import argparse
import os

from . import corpus
from .preprocess import stage_select, stage_lexicon, stage_features
from .utils import load_config


def load_corpus(cfg, args, speaker_fn):
    ds = cfg["dataset"]
    source = args.source or ds["source"]
    if source == "hf":
        return corpus.from_hf(ds["hf_name"], ds.get("hf_split"), ds.get("audio_col"), ds.get("text_col"),
                              speaker_fn=speaker_fn, token=args.hf_token or os.environ.get("HF_TOKEN"),
                              cache_dir=ds.get("hf_cache_dir"), config_name=ds.get("hf_config"))
    if source == "manifest":
        manifest = args.manifest or ds.get("manifest")
        if not manifest:
            raise SystemExit("--manifest (or dataset.manifest in config.yaml) is required for source=manifest")
        return corpus.from_manifest(manifest, ds.get("audio_root"), ds.get("audio_col", "audio_filepath"),
                                    ds.get("text_col", "text"), ds.get("speaker_col"),
                                    speaker_fn=None if ds.get("speaker_col") else speaker_fn)
    if source == "festival":
        root = args.corpus_path or cfg["path"]["corpus_path"]
        return corpus.from_festival(root)
    raise SystemExit(f"unknown source {source}")


def main(default_config, speaker_fn, argv=None):
    ap = argparse.ArgumentParser(description="FastSpeech 2 preprocessing")
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--stage", default="all", choices=["select", "lexicon", "features", "all"])
    ap.add_argument("--source", default=None, choices=[None, "hf", "manifest", "festival"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--corpus_path", default=None)
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    res = cfg["path"]["result_path"]
    if args.stage in ("select", "all"):
        utts = load_corpus(cfg, args, speaker_fn)
        print(f"loaded {len(utts)} utterances from {cfg['dataset']['name']}")
        stage_select(cfg, utts, res)
    if args.stage == "lexicon" or (args.stage == "all" and cfg["alignment"]["method"] == "mfa"):
        stage_lexicon(cfg)
        if args.stage == "all":
            print("MFA alignment required: run scripts/run_mfa.sh, then `--stage features`.")
            return
    if args.stage in ("features", "all"):
        stage_features(cfg, args.workers, res)
