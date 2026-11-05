"""Static figures: loss curves and variance-adaptor (duration / F0 / energy) plots.

Style: one y-axis per panel (no dual axes), recessive grid, thin marks, fixed categorical
order (blue = train / predicted, orange = validation / reference).
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight",
    "axes.titlesize": 10, "axes.axisbelow": True, "axes.titleweight": "bold", "legend.frameon": False,
})

LOSS_KEYS = ["total", "mel", "postnet_mel", "duration", "pitch", "energy", "ctc", "bin"]


def read_log(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {"step": np.array([int(r["step"]) for r in rows])}
    for k in rows[0]:
        if k != "step":
            vals = [r[k] for r in rows]
            out[k] = np.array([float(v) if v not in ("", None) else np.nan for v in vals])
    return out


def _smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    ys = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), k, mode="valid")
    return np.concatenate([y[:w - 1], ys])


def plot_loss_panels(log, title, out_path, color, smooth=1, compare=None):
    keys = [k for k in LOSS_KEYS if k in log and not np.all(np.isnan(log[k]))]
    if not keys:
        return
    n = len(keys)
    cols = 4 if n > 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.4 * rows), squeeze=False)
    for ax, k in zip(axes.flat, keys):
        y = log[k]
        if smooth > 1:
            ax.plot(log["step"], y, color=color, alpha=0.25, lw=0.8)
        ax.plot(log["step"], _smooth(y, smooth), color=color, lw=1.6, label=title.split(" ")[0])
        if compare is not None and k in compare:
            ax.plot(compare["step"], compare[k], color=ORANGE, lw=1.6, marker="o", ms=3, label="validation")
        ax.set_title(k.replace("_", " "))
        ax.set_xlabel("step")
        if k in ("total", "mel", "postnet_mel", "duration", "pitch", "energy"):
            pos = y[np.isfinite(y) & (y > 0)]
            if len(pos) > 3 and pos.max() / max(pos.min(), 1e-9) > 30:
                ax.set_yscale("log")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    if compare is not None:
        axes.flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_losses(log_dir, results_dir, language):
    train = read_log(os.path.join(log_dir, "train_log.csv"))
    val = read_log(os.path.join(log_dir, "val_log.csv"))
    if train:
        w = max(1, min(25, len(train["step"]) // 20))
        plot_loss_panels(train, f"Training loss - {language} FastSpeech 2", os.path.join(results_dir, "training_loss.png"),
                         BLUE, smooth=w)
    if val:
        plot_loss_panels(val, f"Validation loss - {language} FastSpeech 2 (train curve for reference)",
                         os.path.join(results_dir, "validation_loss.png"), ORANGE, smooth=1, compare=None)
        # overlay figure: train vs validation for the main terms
        if train:
            _plot_overlay(train, val, os.path.join(results_dir, "train_vs_validation_loss.png"), language)


def _plot_overlay(train, val, out_path, language):
    keys = [k for k in ("total", "mel", "postnet_mel", "duration", "pitch", "energy") if k in train and k in val]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 2.5), squeeze=False)
    w = max(1, min(25, len(train["step"]) // 20))
    for ax, k in zip(axes.flat, keys):
        ax.plot(train["step"], _smooth(train[k], w), color=BLUE, lw=1.5, label="train")
        ax.plot(val["step"], val[k], color=ORANGE, lw=1.5, marker="o", ms=3, label="validation")
        ax.set_title(k.replace("_", " "))
        ax.set_xlabel("step")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Train vs validation - {language}", fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------ variance adaptor
def _phone_axis(ax, phones, step=1):
    ax.set_xticks(np.arange(len(phones)))
    ax.set_xticklabels(phones, rotation=90, fontsize=7 if len(phones) < 70 else 6)
    ax.set_xlim(-0.6, len(phones) - 0.4)


def plot_duration(phones, durations, frame_ms, title, out_path):
    d = np.asarray(durations)
    fig, ax = plt.subplots(figsize=(max(7, 0.16 * len(phones) + 2), 3.2))
    x = np.arange(len(phones))
    colors = [MUTED if p == "sp" else BLUE for p in phones]
    ax.bar(x, d, color=colors, width=0.8)
    _phone_axis(ax, phones)
    ax.set_ylabel(f"frames  (1 frame = {frame_ms:.1f} ms)")
    ax.set_title(f"{title} - predicted duration per phoneme (total {d.sum() * frame_ms / 1000:.2f} s)", loc="left")
    _save(fig, out_path)


def plot_contour(phones, durations, phone_values, frame_values, ylabel, what, title, out_path, color, voiced_ref=None):
    """Two panels sharing x=time: frame-level contour fed to the decoder, phoneme-level prediction."""
    d = np.asarray(durations)
    edges = np.concatenate([[0], np.cumsum(d)])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(7, 0.16 * len(phones) + 2), 5.2),
                                   gridspec_kw={"height_ratios": [1.2, 1]})
    t = np.arange(len(frame_values))
    ax1.plot(t, frame_values, color=color, lw=1.6)
    for e in edges[1:-1]:
        ax1.axvline(e, color=GRID, lw=0.6, zorder=0)
    centers = (edges[:-1] + edges[1:]) / 2
    ax1.set_xticks(centers)
    ax1.set_xticklabels(phones, rotation=90, fontsize=6 if len(phones) > 70 else 7)
    ax1.set_xlim(0, max(1, edges[-1]))
    ax1.grid(axis="x", visible=False)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"{title} - predicted {what} (frame level, after length regulation)", loc="left")
    if voiced_ref is not None:
        ax1.axhline(voiced_ref, color=ORANGE, lw=1, ls="--")
        ax1.text(edges[-1], voiced_ref, " training mean", color=MUTED, fontsize=7, va="center")
    x = np.arange(len(phones))
    ax2.plot(x, phone_values, color=color, lw=1.2, marker="o", ms=3.5)
    _phone_axis(ax2, phones)
    ax2.set_ylabel(ylabel)
    ax2.set_title(f"{what} per phoneme (variance-predictor output)", loc="left", fontsize=9)
    _save(fig, out_path)


def plot_mel(mel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.imshow(np.asarray(mel).T, origin="lower", aspect="auto", cmap="magma")
    ax.grid(False)
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    ax.set_title(title, loc="left")
    _save(fig, out_path)


def plot_alignment(attn, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.imshow(np.asarray(attn).T, origin="lower", aspect="auto", cmap="Blues")
    ax.grid(False)
    ax.set_xlabel("mel frame")
    ax.set_ylabel("phoneme index")
    ax.set_title(title, loc="left")
    _save(fig, out_path)


def _save(fig, out_path):
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
