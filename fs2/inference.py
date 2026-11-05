"""Inference + variance-adaptor analysis shared by English/inference.py and Telugu/inference.py.

For every sentence it writes
  Results/<Lang>/test_generated_audios/<ID>.wav
  Results/<Lang>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png
  Results/<Lang>/variance_adaptor_plots/<ID>_values.json   (phonemes + numeric predictions)
  Results/<Lang>/test_generated_audios/<ID>_mel.png
and a summary table Results/<Lang>/inference_summary.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .audio import AudioProcessor
from .plotting import plot_duration, plot_contour, plot_mel, BLUE, AQUA
from .text import get_frontend
from .utils import load_config, get_device, load_model_for_inference, latest_checkpoint, Vocoder


def read_sentences(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                sid, text = line.split("\t", 1)
                out.append((sid.strip(), text))
    return out


@torch.no_grad()
def synthesize(model, symbols, phones, speaker_id, device, p=1.0, e=1.0, d=1.0):
    ids = torch.tensor([symbols.encode(phones)], dtype=torch.long, device=device)
    lens = torch.tensor([ids.size(1)], device=device)
    spk = torch.tensor([speaker_id], dtype=torch.long, device=device)
    out = model(spk, ids, lens, ids.size(1), p_control=p, e_control=e, d_control=d)
    T = int(out["mel_lens"][0])
    return {
        "mel": out["mel_postnet"][0, :T].cpu(),
        "durations": out["durations"][0].cpu().numpy(),
        "log_duration": out["log_duration_pred"][0].cpu().numpy(),
        "pitch": out["pitch_pred"][0].cpu().numpy(),
        "energy": out["energy_pred"][0].cpu().numpy(),
    }


def main(default_config, argv=None):
    ap = argparse.ArgumentParser(description="FastSpeech 2 inference")
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--checkpoint", default=None, help="default: checkpoints/best.pth.tar or latest step")
    ap.add_argument("--sentences", default=None, help="TSV: ID<TAB>text (default: test_sentences/<lang>.tsv)")
    ap.add_argument("--text", default=None, help="synthesise a single sentence instead")
    ap.add_argument("--id", default="custom")
    ap.add_argument("--speaker", default=None, help="speaker name (see speakers.json); default from config")
    ap.add_argument("--pitch_control", type=float, default=1.0)
    ap.add_argument("--energy_control", type=float, default=1.0)
    ap.add_argument("--duration_control", type=float, default=1.0)
    ap.add_argument("--vocoder", default=None, choices=[None, "HiFi-GAN", "griffin-lim"])
    ap.add_argument("--out_dir", default=None, help="default: Results/<Language>")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    device = get_device(args.device)
    ckpt = args.checkpoint or latest_checkpoint(cfg["path"]["ckpt_path"])
    if not ckpt or not os.path.exists(ckpt):
        raise SystemExit(f"No checkpoint found in {cfg['path']['ckpt_path']} - train first or pass --checkpoint")
    model, symbols, speakers, stats, meta = load_model_for_inference(ckpt, cfg, device)
    step = meta["step"]
    spk_name = args.speaker or cfg["inference"].get("speaker") or meta["default_speaker"] or ""
    if spk_name not in speakers:
        spk_name = next(iter(speakers))
    spk_id = speakers[spk_name]

    audio = AudioProcessor(cfg)
    vocoder = Vocoder(cfg, device, audio, kind=args.vocoder)
    fe = get_frontend(cfg["language"])
    out_dir = args.out_dir or cfg["path"]["result_path"]
    wav_dir = os.path.join(out_dir, "test_generated_audios")
    var_dir = os.path.join(out_dir, "variance_adaptor_plots")
    for d in (wav_dir, *(os.path.join(var_dir, k) for k in ("duration", "pitch", "energy"))):
        os.makedirs(d, exist_ok=True)

    if args.text:
        sentences = [(args.id, args.text)]
    else:
        path = args.sentences or cfg["inference"]["test_sentences"]
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(cfg["_config_path"]), path)
        sentences = read_sentences(path)

    p_mean, p_std = stats["pitch"][2], stats["pitch"][3]
    e_mean, e_std = stats["energy"][2], stats["energy"][3]
    frame_ms = cfg["preprocessing"]["stft"]["hop_length"] / cfg["preprocessing"]["audio"]["sampling_rate"] * 1000
    summary = {"checkpoint": os.path.relpath(ckpt, os.path.dirname(cfg["_config_path"])), "step": step,
               "speaker": spk_name, "vocoder": vocoder.kind, "sentences": {}}
    print(f"checkpoint {ckpt} (step {step}) speaker={spk_name} vocoder={vocoder.kind} device={device}")
    for sid, text in sentences:
        norm, phones = fe.text_to_phones(text)
        unk = [p for p in phones if p not in symbols.index]
        t0 = time.time()
        r = synthesize(model, symbols, phones, spk_id, device, args.pitch_control, args.energy_control,
                       args.duration_control)
        wav = vocoder(r["mel"])
        rtf = (time.time() - t0) / max(len(wav) / audio.sr, 1e-6)
        audio.save(os.path.join(wav_dir, f"{sid}.wav"), wav)

        f0_hz = r["pitch"] * p_std + p_mean
        energy = r["energy"] * e_std + e_mean
        dur = r["durations"]
        f0_frames, e_frames = np.repeat(f0_hz, dur), np.repeat(energy, dur)
        plot_duration(phones, dur, frame_ms, sid, os.path.join(var_dir, "duration", f"{sid}.png"))
        plot_contour(phones, dur, f0_hz, f0_frames, "F0 (Hz)", "pitch (F0)", sid,
                     os.path.join(var_dir, "pitch", f"{sid}.png"), BLUE, voiced_ref=p_mean)
        plot_contour(phones, dur, energy, e_frames, "energy (|STFT| L2 norm)", "energy", sid,
                     os.path.join(var_dir, "energy", f"{sid}.png"), AQUA, voiced_ref=e_mean)
        plot_mel(r["mel"].numpy(), f"{sid} - generated mel-spectrogram", os.path.join(wav_dir, f"{sid}_mel.png"))

        values = {"id": sid, "text": text, "normalized_text": norm, "phones": phones,
                  "unknown_phones": unk, "durations_frames": dur.tolist(),
                  "durations_ms": [round(float(x) * frame_ms, 1) for x in dur],
                  "pitch_hz": [round(float(x), 2) for x in f0_hz],
                  "energy": [round(float(x), 3) for x in energy],
                  "pitch_normalized": [round(float(x), 4) for x in r["pitch"]],
                  "energy_normalized": [round(float(x), 4) for x in r["energy"]]}
        with open(os.path.join(var_dir, f"{sid}_values.json"), "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=1)
        nsp = [i for i, p in enumerate(phones) if p != "sp"]
        summary["sentences"][sid] = {
            "text": text, "normalized_text": norm, "n_phones": len(phones),
            "audio_seconds": round(len(wav) / audio.sr, 2), "rtf": round(rtf, 3),
            "mean_phone_ms": round(float(np.mean(dur[nsp])) * frame_ms, 1) if nsp else 0,
            "max_phone": phones[int(np.argmax(dur))] if len(dur) else None,
            "max_phone_ms": round(float(dur.max()) * frame_ms, 1) if len(dur) else 0,
            "pitch_hz_mean": round(float(np.mean(f0_hz)), 1), "pitch_hz_min": round(float(np.min(f0_hz)), 1),
            "pitch_hz_max": round(float(np.max(f0_hz)), 1),
            "pitch_final_slope_hz": round(float(f0_hz[-1] - f0_hz[max(0, len(f0_hz) - 6)]), 1),
            "energy_mean": round(float(np.mean(energy)), 2), "unknown_phones": unk,
        }
        print(f"{sid}: {len(phones)} phones -> {len(wav) / audio.sr:.2f}s  RTF {rtf:.3f}")
    with open(os.path.join(out_dir, "inference_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"wrote audio + plots to {out_dir}")
