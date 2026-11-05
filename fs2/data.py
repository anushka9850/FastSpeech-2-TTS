"""Dataset / batching for preprocessed features.

<preprocessed_path>/
    train.txt, val.txt      basename|speaker|phones (space separated)|normalised text
    mel/<basename>.npy      (T, 80) log-mel
    pitch/<basename>.npy    (T,) F0 in Hz, unvoiced frames linearly interpolated
    energy/<basename>.npy   (T,) frame energy
    duration/<basename>.npy (N_phones,) frames per phoneme   (only with MFA alignment)
    stats.json, speakers.json, symbols.json
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .text import SymbolTable
from .model.aligner import beta_binomial_prior


def read_meta(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            base, spk, phones, text = line.split("|", 3)
            items.append({"basename": base, "speaker": spk, "phones": phones.split(), "text": text})
    return items


def load_stats(pre_dir):
    with open(os.path.join(pre_dir, "stats.json")) as f:
        return json.load(f)


def load_speakers(pre_dir):
    with open(os.path.join(pre_dir, "speakers.json"), encoding="utf-8") as f:
        return json.load(f)


class TTSDataset(Dataset):
    def __init__(self, cfg, split):
        self.pre = cfg["path"]["preprocessed_path"]
        self.items = read_meta(os.path.join(self.pre, f"{split}.txt"))
        self.symbols = SymbolTable.load(os.path.join(self.pre, "symbols.json"))
        self.speakers = load_speakers(self.pre)
        st = load_stats(self.pre)
        self.p_mean, self.p_std = st["pitch"][2], st["pitch"][3]
        self.e_mean, self.e_std = st["energy"][2], st["energy"][3]
        self.use_mfa = cfg["alignment"]["method"] == "mfa"
        self.prior_scaling = cfg["alignment"].get("prior_scaling", 1.0)
        self.lengths = [self._mel_len(it) for it in self.items]

    def _mel_len(self, it):
        return int(np.load(os.path.join(self.pre, "mel", it["basename"] + ".npy"), mmap_mode="r").shape[0])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        b = it["basename"]
        mel = np.load(os.path.join(self.pre, "mel", b + ".npy"))
        pitch = (np.load(os.path.join(self.pre, "pitch", b + ".npy")) - self.p_mean) / self.p_std
        energy = (np.load(os.path.join(self.pre, "energy", b + ".npy")) - self.e_mean) / self.e_std
        text = np.array(self.symbols.encode(it["phones"]), dtype=np.int64)
        sample = {"id": b, "speaker": self.speakers[it["speaker"]], "text": text, "mel": mel,
                  "pitch": pitch.astype(np.float32), "energy": energy.astype(np.float32),
                  "raw_text": it["text"]}
        if self.use_mfa:
            sample["duration"] = np.load(os.path.join(self.pre, "duration", b + ".npy")).astype(np.int64)
        else:
            sample["prior"] = beta_binomial_prior(len(text), mel.shape[0], self.prior_scaling)
        return sample


def _pad1(seqs, value=0):
    L = max(len(s) for s in seqs)
    out = np.full((len(seqs), L), value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def collate(samples):
    samples = sorted(samples, key=lambda s: -len(s["text"]))
    src_lens = np.array([len(s["text"]) for s in samples])
    mel_lens = np.array([s["mel"].shape[0] for s in samples])
    T, n_mel = mel_lens.max(), samples[0]["mel"].shape[1]
    mels = np.zeros((len(samples), T, n_mel), dtype=np.float32)
    for i, s in enumerate(samples):
        mels[i, :s["mel"].shape[0]] = s["mel"]
    batch = {
        "ids": [s["id"] for s in samples],
        "raw_texts": [s["raw_text"] for s in samples],
        "speakers": torch.tensor([s["speaker"] for s in samples], dtype=torch.long),
        "texts": torch.from_numpy(_pad1([s["text"] for s in samples])),
        "src_lens": torch.from_numpy(src_lens),
        "max_src_len": int(src_lens.max()),
        "mels": torch.from_numpy(mels),
        "mel_lens": torch.from_numpy(mel_lens),
        "max_mel_len": int(T),
        "pitch": torch.from_numpy(_pad1([s["pitch"] for s in samples])),
        "energy": torch.from_numpy(_pad1([s["energy"] for s in samples])),
        "durations": None, "attn_prior": None,
    }
    if "duration" in samples[0]:
        batch["durations"] = torch.from_numpy(_pad1([s["duration"] for s in samples]))
    if "prior" in samples[0]:
        prior = np.zeros((len(samples), T, src_lens.max()), dtype=np.float32)
        for i, s in enumerate(samples):
            p = s["prior"]
            prior[i, :p.shape[0], :p.shape[1]] = p
        batch["attn_prior"] = torch.from_numpy(prior)
    return batch


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class BucketBatchSampler(Sampler):
    """Shuffle, then sort inside groups of `batch_size * bucket` by length -> less padding."""

    def __init__(self, lengths, batch_size, bucket=8, shuffle=True, seed=1234, drop_last=False):
        self.lengths, self.bs, self.bucket = lengths, batch_size, bucket
        self.shuffle, self.drop_last = shuffle, drop_last
        self.rng = random.Random(seed)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            self.rng.shuffle(idx)
        batches = []
        chunk = self.bs * self.bucket
        for s in range(0, len(idx), chunk):
            group = sorted(idx[s:s + chunk], key=lambda i: self.lengths[i])
            for b in range(0, len(group), self.bs):
                batch = group[b:b + self.bs]
                if len(batch) < self.bs and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        n = len(self.lengths) // self.bs
        return n if self.drop_last else n + (len(self.lengths) % self.bs > 0)
