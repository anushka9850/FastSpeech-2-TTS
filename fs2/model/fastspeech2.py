"""FastSpeech 2 (Ren et al., 2021) - encoder / variance adaptor / mel decoder / PostNet.

Structure follows ming024/FastSpeech2 `model/fastspeech2.py`; additions:
  * optional learned aligner (durations without MFA)
  * speaker embedding sized from the speaker table produced by preprocessing
  * phoneme-level pitch/energy targets averaged on the fly from frame-level features
"""
import torch
import torch.nn as nn

from ..transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor, get_mask_from_lengths, average_by_duration
from .aligner import AlignmentEncoder, mas


class FastSpeech2(nn.Module):
    def __init__(self, cfg, n_symbols, n_speakers, stats):
        super().__init__()
        mcfg = dict(cfg["model"])
        mcfg["n_symbols"] = n_symbols
        self.cfg = mcfg
        n_mels = cfg["preprocessing"]["mel"]["n_mel_channels"]
        self.encoder = Encoder(mcfg)
        self.variance_adaptor = VarianceAdaptor(mcfg, stats)
        self.decoder = Decoder(mcfg)
        self.mel_linear = nn.Linear(mcfg["transformer"]["decoder_hidden"], n_mels)
        self.postnet = PostNet(n_mel_channels=n_mels)
        self.speaker_emb = nn.Embedding(n_speakers, mcfg["transformer"]["encoder_hidden"]) \
            if (mcfg.get("multi_speaker", False) and n_speakers > 1) else None
        self.learn_alignment = cfg["alignment"]["method"] == "learned"
        if self.learn_alignment:
            self.aligner = AlignmentEncoder(n_mels, mcfg["transformer"]["encoder_hidden"])

    def forward(self, speakers, texts, src_lens, max_src_len, mels=None, mel_lens=None,
                max_mel_len=None, pitch=None, energy=None, durations=None, attn_prior=None,
                p_control=1.0, e_control=1.0, d_control=1.0):
        src_mask = get_mask_from_lengths(src_lens, max_src_len)
        attn_soft = attn_hard = attn_logprob = None

        if self.learn_alignment and mels is not None:
            emb = self.encoder.src_word_emb(texts)
            attn_soft, attn_logprob = self.aligner(mels.transpose(1, 2), emb.transpose(1, 2),
                                                   src_mask, attn_prior)
            attn_hard = mas(attn_soft, src_lens, mel_lens)
            durations = attn_hard.sum(2)[:, 0, :].long()  # B, T_text

        # frame-level -> phoneme-level targets
        p_target = average_by_duration(pitch, durations) if (pitch is not None and durations is not None) else None
        e_target = average_by_duration(energy, durations) if (energy is not None and durations is not None) else None

        x = self.encoder(texts, src_mask)
        if self.speaker_emb is not None:
            x = x + self.speaker_emb(speakers).unsqueeze(1)

        x, p_pred, e_pred, log_d_pred, d_used, out_lens, mel_mask = self.variance_adaptor(
            x, src_mask, max_mel_len, p_target, e_target, durations, p_control, e_control, d_control)

        x, mel_mask = self.decoder(x, mel_mask)
        mel = self.mel_linear(x)
        mel_post = self.postnet(mel) + mel

        return {
            "mel": mel, "mel_postnet": mel_post, "pitch_pred": p_pred, "energy_pred": e_pred,
            "log_duration_pred": log_d_pred, "durations": d_used, "src_mask": src_mask,
            "mel_mask": mel_mask, "mel_lens": out_lens, "pitch_target": p_target,
            "energy_target": e_target, "attn_soft": attn_soft, "attn_hard": attn_hard,
            "attn_logprob": attn_logprob,
        }
