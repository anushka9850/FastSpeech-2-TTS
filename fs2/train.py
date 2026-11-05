"""Training loop shared by English/train.py and Telugu/train.py.

Optimiser and schedule follow the reference repo / paper: Adam(0.9, 0.98, eps 1e-9), Noam warm-up
(lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)) with step-wise annealing, grad-clip 1.0.
Logs go to <log_path>/train_log.csv and val_log.csv; loss figures are regenerated in
Results/<Language>/ at every validation.
"""
import argparse
import csv
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from collections import Counter

from .data import TTSDataset, BucketBatchSampler, collate, to_device
from .model.loss import FastSpeech2Loss
from .plotting import plot_losses, plot_mel, plot_alignment
from .utils import load_config, get_device, build_model, save_checkpoint, count_params


class ScheduledOptim:
    def __init__(self, model, cfg, step=0):
        o = cfg["optimizer"]
        self.opt = torch.optim.Adam(model.parameters(), betas=o["betas"], eps=o["eps"],
                                    weight_decay=o["weight_decay"])
        self.warmup = o["warm_up_step"]
        self.anneal_steps, self.anneal_rate = o["anneal_steps"], o["anneal_rate"]
        self.init_lr = o.get("lr_scale", 1.0) * np.power(cfg["model"]["transformer"]["encoder_hidden"], -0.5)
        self.step_num = step

    def lr(self):
        s = max(self.step_num, 1)
        lr = np.min([np.power(s, -0.5), np.power(self.warmup, -1.5) * s])
        for a in self.anneal_steps:
            if s > a:
                lr *= self.anneal_rate
        return float(self.init_lr * lr)

    def step(self):
        self.step_num += 1
        for g in self.opt.param_groups:
            g["lr"] = self.lr()
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, sd):
        self.opt.load_state_dict(sd)


class CsvLog:
    def __init__(self, path, keys, resume):
        self.path, self.keys = path, keys
        if not resume or not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(["step"] + keys + ["lr"])

    def write(self, step, vals, lr):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([step] + [f"{vals[k]:.6f}" if k in vals else "" for k in self.keys] + [f"{lr:.3e}"])


KEYS = ["total", "mel", "postnet_mel", "duration", "pitch", "energy", "ctc", "bin"]


def _forward(model, batch):
    return model(batch["speakers"], batch["texts"], batch["src_lens"], batch["max_src_len"],
                 batch["mels"], batch["mel_lens"], batch["max_mel_len"], batch["pitch"], batch["energy"],
                 batch["durations"], batch["attn_prior"])


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, step):
    model.eval()
    sums, n = {}, 0
    last = None
    for batch in loader:
        batch = to_device(batch, device)
        out = _forward(model, batch)
        losses = loss_fn(batch, out, step)
        bs = batch["texts"].size(0)
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v) * bs
        n += bs
        last = (batch, out)
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}, last


def main(default_config, argv=None):
    ap = argparse.ArgumentParser(description="Train FastSpeech 2")
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--restore_step", type=int, default=0, help="resume from checkpoints/<step>.pth.tar")
    ap.add_argument("--resume", action="store_true", help="resume from the latest step checkpoint")
    ap.add_argument("--total_step", type=int, default=None, help="override optimizer total steps")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    if args.total_step:
        tcfg["total_step"] = args.total_step
    seed = tcfg.get("seed", 1234)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = get_device(args.device)

    ckpt_dir, log_dir = cfg["path"]["ckpt_path"], cfg["path"]["log_path"]
    res_dir = cfg["path"]["result_path"]
    for d in (ckpt_dir, log_dir, res_dir):
        os.makedirs(d, exist_ok=True)

    train_set, val_set = TTSDataset(cfg, "train"), TTSDataset(cfg, "val")
    bs = tcfg["batch_size"]
    nw = args.num_workers if args.num_workers is not None else tcfg.get("num_workers", 4)
    train_loader = DataLoader(train_set, batch_sampler=BucketBatchSampler(train_set.lengths, bs, seed=seed, drop_last=True)
                              if len(train_set) >= bs else BucketBatchSampler(train_set.lengths, bs, seed=seed),
                              collate_fn=collate, num_workers=nw, pin_memory=device.type == "cuda",
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)

    model, symbols, speakers, stats = build_model(cfg, device)
    optim = ScheduledOptim(model, cfg)
    step = 0
    restore = args.restore_step
    if args.resume:
        steps = [int(f.split(".")[0]) for f in os.listdir(ckpt_dir) if f.split(".")[0].isdigit()]
        restore = max(steps) if steps else 0
    if restore:
        blob = torch.load(os.path.join(ckpt_dir, f"{restore}.pth.tar"), map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        if blob.get("optimizer"):
            optim.load_state_dict(blob["optimizer"])
        step = optim.step_num = blob["step"]
        print(f"resumed from step {step}")
    loss_fn = FastSpeech2Loss(cfg)
    spk_secs = Counter()
    for it, n in zip(train_set.items, train_set.lengths):
        spk_secs[it["speaker"]] += n
    extra = {"default_speaker": spk_secs.most_common(1)[0][0]}
    train_log = CsvLog(os.path.join(log_dir, "train_log.csv"), KEYS, resume=step > 0)
    val_log = CsvLog(os.path.join(log_dir, "val_log.csv"), KEYS, resume=step > 0)
    best_path = os.path.join(log_dir, "best.json")
    best = json.load(open(best_path)) if (step > 0 and os.path.exists(best_path)) else {"val_mel_total": math.inf, "step": 0}

    print(f"[{cfg['language']}] device={device} params={count_params(model) / 1e6:.2f}M "
          f"train={len(train_set)} val={len(val_set)} symbols={len(symbols)} speakers={len(speakers)}")
    with open(os.path.join(log_dir, "config_used.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, indent=1)

    grad_acc, clip = tcfg.get("grad_acc_step", 1), tcfg["grad_clip_thresh"]
    total_step = tcfg["total_step"]
    use_amp = bool(tcfg.get("fp16", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    running, n_run, t0 = {}, 0, time.time()
    micro = 0
    patience = tcfg.get("early_stop_patience", 0)
    while step < total_step:
        for batch in train_loader:
            batch = to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = _forward(model, batch)
            if use_amp:  # losses in fp32
                out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}
            losses = loss_fn(batch, out, step)
            if not torch.isfinite(losses["total"]):
                print(f"step {step}: non-finite loss, batch skipped")
                optim.zero_grad()
                continue
            scaler.scale(losses["total"] / grad_acc).backward()
            micro += 1
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            n_run += 1
            if micro % grad_acc:
                continue
            scaler.unscale_(optim.opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step_num += 1
            for g in optim.opt.param_groups:
                g["lr"] = optim.lr()
            scaler.step(optim.opt)
            scaler.update()
            optim.zero_grad()
            step = optim.step_num

            if step % tcfg["log_step"] == 0 or step == 1:
                avg = {k: v / n_run for k, v in running.items()}
                train_log.write(step, avg, optim.lr())
                msg = " ".join(f"{k}={avg[k]:.4f}" for k in KEYS if k in avg)
                print(f"step {step}/{total_step} lr={optim.lr():.2e} {msg} ({time.time() - t0:.1f}s)")
                running, n_run, t0 = {}, 0, time.time()

            if step % tcfg["val_step"] == 0 or step == total_step:
                vl, last = evaluate(model, val_loader, loss_fn, device, step)
                val_log.write(step, vl, optim.lr())
                print("  [val] " + " ".join(f"{k}={vl[k]:.4f}" for k in KEYS if k in vl))
                score = vl["mel"] + vl["postnet_mel"]
                if score < best["val_mel_total"]:
                    best = {"val_mel_total": score, "step": step, **{k: vl[k] for k in vl}}
                    save_checkpoint(os.path.join(ckpt_dir, "best.pth.tar"), model, None, step, cfg, extra)
                    json.dump(best, open(best_path, "w"), indent=1)
                plot_losses(log_dir, res_dir, cfg["language_name"])
                _synth_sample(cfg, last, step, log_dir, device)
                if patience and step - best["step"] >= patience * tcfg["val_step"]:
                    print(f"early stopping: no val improvement for {patience} validations")
                    total_step = step
            if step % tcfg["save_step"] == 0 or step == total_step:
                save_checkpoint(os.path.join(ckpt_dir, f"{step}.pth.tar"), model, optim, step, cfg, extra)
                _prune(ckpt_dir, tcfg.get("keep_checkpoints", 3))
            if step >= total_step:
                break
    plot_losses(log_dir, res_dir, cfg["language_name"])
    for f in ("train_log.csv", "val_log.csv", "best.json"):
        if os.path.exists(os.path.join(log_dir, f)):
            shutil.copy(os.path.join(log_dir, f), os.path.join(res_dir, f))
    print(f"done. best val (mel+postnet) {best['val_mel_total']:.4f} at step {best['step']}")


def _prune(ckpt_dir, keep):
    steps = sorted(int(f.split(".")[0]) for f in os.listdir(ckpt_dir) if f.split(".")[0].isdigit())
    for s in steps[:-keep] if keep > 0 else []:
        os.remove(os.path.join(ckpt_dir, f"{s}.pth.tar"))


@torch.no_grad()
def _synth_sample(cfg, last, step, log_dir, device):
    """Save target vs predicted mel and alignment for the last validation utterance."""
    if last is None:
        return
    batch, out = last
    sd = os.path.join(log_dir, "samples")
    i = 0
    T = int(batch["mel_lens"][i])
    plot_mel(out["mel_postnet"][i, :T].float().cpu().numpy(), f"step {step} predicted (teacher-forced) {batch['ids'][i]}",
             os.path.join(sd, f"{step}_mel_pred.png"))
    plot_mel(batch["mels"][i, :T].cpu().numpy(), f"target {batch['ids'][i]}", os.path.join(sd, "target_mel.png"))
    if out.get("attn_soft") is not None:
        L = int(batch["src_lens"][i])
        plot_alignment(out["attn_soft"][i, 0, :T, :L].cpu().numpy(), f"step {step} learned alignment",
                       os.path.join(sd, f"{step}_alignment.png"))
