"""Unsupervised duration learning ("One TTS Alignment To Rule Them All", Badlani et al. 2022).

Used when `alignment.method: learned` (default). It removes the dependency on Montreal Forced
Aligner, which has no pretrained Telugu acoustic model:
  * AlignmentEncoder: soft text<->mel attention from L2 distances of projected features,
    multiplied by a beta-binomial prior that favours near-diagonal alignments
  * ForwardSumLoss: CTC-style likelihood of all monotonic alignments
  * Monotonic Alignment Search (MAS): hard alignment -> integer durations per phoneme
  * BinLoss: pulls the soft alignment towards the hard one
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import betabinom

try:
    from numba import njit
except ImportError:  # pragma: no cover - pure python fallback (slow)
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not (a and callable(a[0])) else a[0]


class AlignmentEncoder(nn.Module):
    def __init__(self, n_mel_channels=80, n_text_channels=256, n_att_channels=80, temperature=0.0005):
        super().__init__()
        self.temperature = temperature
        self.key_proj = nn.Sequential(
            nn.Conv1d(n_text_channels, n_text_channels * 2, 3, padding=1), nn.ReLU(),
            nn.Conv1d(n_text_channels * 2, n_att_channels, 1))
        self.query_proj = nn.Sequential(
            nn.Conv1d(n_mel_channels, n_mel_channels * 2, 3, padding=1), nn.ReLU(),
            nn.Conv1d(n_mel_channels * 2, n_mel_channels, 1), nn.ReLU(),
            nn.Conv1d(n_mel_channels, n_att_channels, 1))

    def forward(self, queries, keys, src_mask, attn_prior):
        """queries: B x n_mel x T_mel, keys: B x n_text x T_text, src_mask: B x T_text (True = pad),
        attn_prior: B x T_mel x T_text. Returns soft attention and its log-probabilities (B,1,T_mel,T_text)."""
        k = self.key_proj(keys)        # B, C, T1
        q = self.query_proj(queries)   # B, C, T2
        # squared L2 distance without materialising B x C x T2 x T1
        dist = (q.pow(2).sum(1).unsqueeze(2) + k.pow(2).sum(1).unsqueeze(1)
                - 2 * torch.bmm(q.transpose(1, 2), k))  # B, T2, T1
        attn = -self.temperature * dist
        attn = F.log_softmax(attn, dim=2) + torch.log(attn_prior + 1e-8)
        attn = attn.unsqueeze(1)
        attn_logprob = attn.clone()
        attn = attn.masked_fill(src_mask[:, None, None, :], -float("inf"))
        attn = torch.softmax(attn, dim=3)
        return attn, attn_logprob


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        padded = F.pad(attn_logprob, (1, 0), value=self.blank_logprob)  # blank = class 0
        total = 0.0
        for b in range(attn_logprob.shape[0]):
            tl, ml = int(in_lens[b]), int(out_lens[b])
            target = torch.arange(1, tl + 1, device=attn_logprob.device).unsqueeze(0)
            lp = padded[b].permute(1, 0, 2)[:ml, :, :tl + 1]  # T_mel, 1, T_text+1
            lp = F.log_softmax(lp, dim=-1)
            total = total + F.ctc_loss(lp, target, input_lengths=torch.tensor([ml]),
                                       target_lengths=torch.tensor([tl]), zero_infinity=True)
        return total / attn_logprob.shape[0]


class BinLoss(nn.Module):
    def forward(self, hard, soft):
        log_sum = torch.log(torch.clamp(soft[hard == 1], min=1e-12)).sum()
        return -log_sum / hard.sum()


@njit(cache=True)
def _mas_width1(log_attn):
    T2, T1 = log_attn.shape
    opt = np.zeros((T2, T1), dtype=np.float32)
    log_p = np.full((T2, T1), -np.inf)
    prev = np.zeros((T2, T1), dtype=np.int64)
    log_p[0, 0] = log_attn[0, 0]
    for i in range(1, T2):
        for j in range(T1):
            best = log_p[i - 1, j]
            bj = j
            if j > 0 and log_p[i - 1, j - 1] >= best:
                best = log_p[i - 1, j - 1]
                bj = j - 1
            log_p[i, j] = log_attn[i, j] + best
            prev[i, j] = bj
    j = T1 - 1
    for i in range(T2 - 1, -1, -1):
        opt[i, j] = 1
        j = prev[i, j]
    return opt


def mas(attn, in_lens, out_lens):
    """attn: B x 1 x T_mel x T_text soft attention -> hard 0/1 alignment (same shape)."""
    a = attn.detach().float().cpu().numpy()
    hard = np.zeros_like(a)
    for b in range(a.shape[0]):
        tl, ml = int(in_lens[b]), int(out_lens[b])
        hard[b, 0, :ml, :tl] = _mas_width1(np.log(np.clip(a[b, 0, :ml, :tl], 1e-12, None)))
    return torch.from_numpy(hard).to(attn.device)


_PRIOR_CACHE = {}


def beta_binomial_prior(n_text, n_mel, scaling=1.0):
    key = (n_text, n_mel)
    if key not in _PRIOR_CACHE:
        x = np.arange(n_text)[None, :]
        i = np.arange(1, n_mel + 1)[:, None]
        prior = betabinom(n_text - 1 if n_text > 1 else 1, scaling * i, scaling * (n_mel + 1 - i)).pmf(x)
        if n_text == 1:
            prior = np.ones((n_mel, 1))
        _PRIOR_CACHE[key] = prior.astype(np.float32)
        if len(_PRIOR_CACHE) > 20000:
            _PRIOR_CACHE.clear()
    return _PRIOR_CACHE[key]
