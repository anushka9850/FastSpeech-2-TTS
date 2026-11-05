"""Config loading, device selection, checkpoints, model construction, vocoder."""
import json
import os
import zipfile

import torch
import yaml

from .data import load_speakers, load_stats
from .model.fastspeech2 import FastSpeech2
from .text import SymbolTable

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PKG_DIR)


def load_config(path):
    """Load config.yaml; relative paths inside `path:` / `alignment:` are resolved
    against the folder that contains the config (English/ or Telugu/)."""
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(path)
    for k, v in cfg["path"].items():
        if isinstance(v, str) and v and not os.path.isabs(v):
            cfg["path"][k] = os.path.normpath(os.path.join(base, v))
    tg = cfg["alignment"].get("textgrid_path")
    if tg and not os.path.isabs(tg):
        cfg["alignment"]["textgrid_path"] = os.path.normpath(os.path.join(base, tg))
    cfg["_config_path"] = path
    return cfg


def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg, device):
    pre = cfg["path"]["preprocessed_path"]
    symbols = SymbolTable.load(os.path.join(pre, "symbols.json"))
    speakers = load_speakers(pre)
    stats = load_stats(pre)
    model = FastSpeech2(cfg, len(symbols), len(speakers), stats).to(device)
    return model, symbols, speakers, stats


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path, model, optimizer, step, cfg, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pre = cfg["path"]["preprocessed_path"]
    blob = {"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
            "step": step, "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "symbols": SymbolTable.load(os.path.join(pre, "symbols.json")).symbols,
            "speakers": load_speakers(pre), "stats": load_stats(pre)}
    if extra:
        blob.update(extra)
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_model_for_inference(ckpt_path, cfg, device):
    """Rebuild the model from what is stored *inside* the checkpoint (symbols, speakers, stats),
    so inference works even without the preprocessed data folder."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    symbols = SymbolTable(blob["symbols"])
    model = FastSpeech2(cfg, len(symbols), len(blob["speakers"]), blob["stats"])
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    meta = {"step": blob.get("step", 0), "default_speaker": blob.get("default_speaker")}
    return model, symbols, blob["speakers"], blob["stats"], meta


def latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    best = os.path.join(ckpt_dir, "best.pth.tar")
    if os.path.exists(best):
        return best
    steps = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth.tar") and f.split(".")[0].isdigit()]
    if not steps:
        return None
    return os.path.join(ckpt_dir, max(steps, key=lambda f: int(f.split(".")[0])))


# ---------------------------------------------------------------- vocoder
class Vocoder:
    """Pretrained HiFi-GAN "universal" generator shipped with the reference repo
    (22.05 kHz, 80-bin mel, fmax 8 kHz). Falls back to Griffin-Lim if unavailable."""

    def __init__(self, cfg, device, ap, kind=None):
        self.ap, self.device = ap, device
        self.kind = kind or cfg["vocoder"]["model"]
        self.gen = None
        if self.kind.lower() == "hifi-gan":
            try:
                self.gen = self._load_hifigan(cfg["vocoder"].get("checkpoint"))
            except Exception as e:
                print(f"[vocoder] HiFi-GAN unavailable ({e}); using Griffin-Lim")
                self.kind = "griffin-lim"

    def _load_hifigan(self, ckpt):
        from .hifigan.models import Generator
        hdir = os.path.join(PKG_DIR, "hifigan")
        with open(os.path.join(hdir, "config.json")) as f:
            h = AttrDict(json.load(f))
        ckpt = ckpt or os.path.join(hdir, "generator_universal.pth.tar")
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(PROJECT_DIR, ckpt)
        if not os.path.exists(ckpt) and os.path.exists(ckpt + ".zip"):
            with zipfile.ZipFile(ckpt + ".zip") as z:
                z.extractall(os.path.dirname(ckpt))
        gen = Generator(h)
        gen.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["generator"])
        gen.eval()
        gen.remove_weight_norm()
        return gen.to(self.device)

    @torch.no_grad()
    def __call__(self, mel):
        """mel: (T, n_mels) tensor/ndarray of log-mel -> float32 waveform in [-1, 1]."""
        if self.gen is None:
            m = mel.detach().cpu().numpy() if torch.is_tensor(mel) else mel
            return self.ap.griffin_lim(m)
        m = mel if torch.is_tensor(mel) else torch.from_numpy(mel)
        wav = self.gen(m.float().T.unsqueeze(0).to(self.device)).squeeze().cpu().numpy()
        return wav.astype("float32")


class AttrDict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.__dict__ = self
