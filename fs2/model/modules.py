"""Variance adaptor (duration / pitch / energy predictors + length regulator).

Adapted from ming024/FastSpeech2 `model/modules.py`. Modifications:
  * energy uses `e_control` (the reference passes `p_control` to the energy predictor - a bug)
  * length regulator is vectorised (searchsorted over cumulative durations) instead of a
    python loop over phonemes - same output, much faster
  * phoneme-level pitch/energy targets are computed on the fly from frame-level features and
    the (MFA or learned) durations, so both alignment methods share one code path
  * bins come from stats passed in, not from a file path baked into the config
"""
from collections import OrderedDict

import torch
import torch.nn as nn


def get_mask_from_lengths(lengths, max_len=None):
    max_len = int(max_len if max_len is not None else lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ids >= lengths.unsqueeze(1)  # True = padding


def length_regulate(x, durations, max_len=None):
    """x: B x T_text x C, durations: B x T_text (int) -> B x T_mel x C, mel_lens."""
    durations = durations.long().clamp(min=0)
    cum = torch.cumsum(durations, dim=1)                       # B, T_text
    mel_lens = cum[:, -1]
    T = int(max_len if max_len is not None else max(int(mel_lens.max().item()), 1))
    pos = torch.arange(T, device=x.device).unsqueeze(0).expand(x.size(0), -1).contiguous()
    idx = torch.searchsorted(cum, pos, right=True).clamp(max=x.size(1) - 1)  # B, T_mel
    out = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(2)))
    out = out.masked_fill((pos >= mel_lens.unsqueeze(1)).unsqueeze(-1), 0.0)
    return out, mel_lens


def average_by_duration(frame_values, durations):
    """frame_values: B x T_mel, durations: B x T_text -> phoneme-level mean B x T_text."""
    durations = durations.long()
    cs = torch.cat([torch.zeros_like(frame_values[:, :1]), torch.cumsum(frame_values, dim=1)], dim=1)
    end = torch.cumsum(durations, dim=1).clamp(max=frame_values.size(1))
    start = (end - durations).clamp(min=0)
    total = torch.gather(cs, 1, end) - torch.gather(cs, 1, start)
    return torch.where(durations > 0, total / durations.clamp(min=1).float(), torch.zeros_like(total))


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class VariancePredictor(nn.Module):
    """2 x (Conv1D -> ReLU -> LayerNorm -> Dropout) -> Linear  (paper Fig. 1c)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg["transformer"]["encoder_hidden"]
        f = cfg["variance_predictor"]["filter_size"]
        k = cfg["variance_predictor"]["kernel_size"]
        p = cfg["variance_predictor"]["dropout"]
        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(d, f, k, (k - 1) // 2)), ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(f)), ("dropout_1", nn.Dropout(p)),
            ("conv1d_2", Conv(f, f, k, (k - 1) // 2)), ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(f)), ("dropout_2", nn.Dropout(p)),
        ]))
        self.linear_layer = nn.Linear(f, 1)

    def forward(self, x, mask):
        out = self.linear_layer(self.conv_layer(x)).squeeze(-1)
        return out.masked_fill(mask, 0.0) if mask is not None else out


class VarianceAdaptor(nn.Module):
    def __init__(self, cfg, stats):
        super().__init__()
        self.duration_predictor = VariancePredictor(cfg)
        self.pitch_predictor = VariancePredictor(cfg)
        self.energy_predictor = VariancePredictor(cfg)
        n_bins = cfg["variance_embedding"]["n_bins"]
        d = cfg["transformer"]["encoder_hidden"]
        pmin, pmax = stats["pitch"][:2]
        emin, emax = stats["energy"][:2]
        self.register_buffer("pitch_bins", torch.linspace(pmin, pmax, n_bins - 1))
        self.register_buffer("energy_bins", torch.linspace(emin, emax, n_bins - 1))
        self.pitch_embedding = nn.Embedding(n_bins, d)
        self.energy_embedding = nn.Embedding(n_bins, d)

    def forward(self, x, src_mask, max_mel_len=None, pitch_target=None, energy_target=None,
                duration_target=None, p_control=1.0, e_control=1.0, d_control=1.0):
        log_d_pred = self.duration_predictor(x, src_mask)

        p_pred = self.pitch_predictor(x, src_mask)
        p_used = pitch_target if pitch_target is not None else _apply_control(p_pred, p_control)
        x = x + self.pitch_embedding(torch.bucketize(p_used, self.pitch_bins))

        e_pred = self.energy_predictor(x, src_mask)
        e_used = energy_target if energy_target is not None else _apply_control(e_pred, e_control)
        x = x + self.energy_embedding(torch.bucketize(e_used, self.energy_bins))

        if duration_target is not None:
            d_used = duration_target
        else:
            d_used = torch.clamp(torch.round((torch.exp(log_d_pred) - 1) * d_control), min=0)
            d_used = d_used.masked_fill(src_mask, 0)
            # every real phoneme gets at least one frame so nothing is silently skipped
            d_used = torch.where(src_mask, d_used, d_used.clamp(min=1))
        x, mel_lens = length_regulate(x, d_used, max_mel_len)
        mel_mask = get_mask_from_lengths(mel_lens, x.size(1))
        return x, p_pred, e_pred, log_d_pred, d_used.long(), mel_lens, mel_mask


def _apply_control(pred, control):
    # control may be a float or a per-phoneme tensor (B x T_text)
    return pred * control if not torch.is_tensor(control) else pred * control.to(pred)
