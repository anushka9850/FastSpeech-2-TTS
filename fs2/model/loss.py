"""FastSpeech 2 loss (reference repo) + alignment losses when durations are learned.

total = L1(mel) + L1(postnet mel) + MSE(pitch) + MSE(energy) + MSE(log(d+1))
        [+ w_ctc * ForwardSum + w_bin(step) * BinLoss]
"""
import torch
import torch.nn as nn

from .aligner import ForwardSumLoss, BinLoss


class FastSpeech2Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        a = cfg["alignment"]
        self.learned = a["method"] == "learned"
        self.ctc_w = a.get("ctc_loss_weight", 1.0)
        self.bin_w = a.get("bin_loss_weight", 1.0)
        self.bin_start = a.get("bin_loss_start_step", 5000)
        self.bin_warmup = a.get("bin_loss_warmup_steps", 5000)
        self.l1, self.mse = nn.L1Loss(), nn.MSELoss()
        self.forward_sum, self.bin = ForwardSumLoss(), BinLoss()

    def forward(self, batch, out, step):
        src_keep = ~out["src_mask"]
        mel_keep = ~out["mel_mask"]
        T = mel_keep.size(1)
        mel_t = batch["mels"][:, :T]
        mel_keep = mel_keep[:, :mel_t.size(1)]
        m = mel_keep.unsqueeze(-1)

        mel_loss = self.l1(out["mel"][:, :mel_t.size(1)].masked_select(m), mel_t.masked_select(m))
        post_loss = self.l1(out["mel_postnet"][:, :mel_t.size(1)].masked_select(m), mel_t.masked_select(m))
        pitch_loss = self.mse(out["pitch_pred"].masked_select(src_keep),
                              out["pitch_target"].detach().masked_select(src_keep))
        energy_loss = self.mse(out["energy_pred"].masked_select(src_keep),
                               out["energy_target"].detach().masked_select(src_keep))
        log_d_target = torch.log(out["durations"].float() + 1).detach()
        dur_loss = self.mse(out["log_duration_pred"].masked_select(src_keep), log_d_target.masked_select(src_keep))

        total = mel_loss + post_loss + pitch_loss + energy_loss + dur_loss
        losses = {"mel": mel_loss, "postnet_mel": post_loss, "pitch": pitch_loss,
                  "energy": energy_loss, "duration": dur_loss}
        if self.learned and out["attn_logprob"] is not None:
            ctc = self.forward_sum(out["attn_logprob"], batch["src_lens"], batch["mel_lens"])
            total = total + self.ctc_w * ctc
            losses["ctc"] = ctc
            if step >= self.bin_start:
                w = self.bin_w * min(1.0, (step - self.bin_start) / max(1, self.bin_warmup))
                b = self.bin(out["attn_hard"], out["attn_soft"])
                total = total + w * b
                losses["bin"] = b
        losses["total"] = total
        return losses
