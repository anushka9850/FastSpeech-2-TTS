"""Audio feature extraction (mel, F0, energy), I/O and Griffin-Lim fallback.

Mel definition matches the reference repo (and therefore the pretrained HiFi-GAN vocoder):
  22.05 kHz, n_fft = win = 1024, hop = 256, 80 mel bins, fmin 0, fmax 8000,
  centred reflect-padded STFT magnitude, natural log with 1e-5 floor.
Energy = L2 norm of the STFT magnitude per frame (as in the reference repo).
F0 = WORLD DIO + StoneMask (as in the reference repo), with a librosa pYIN fallback.
"""
import numpy as np
import librosa
import soundfile as sf


class AudioProcessor:
    def __init__(self, cfg):
        a = cfg["preprocessing"]["audio"]
        s = cfg["preprocessing"]["stft"]
        m = cfg["preprocessing"]["mel"]
        p = cfg["preprocessing"]["pitch"]
        self.sr = a["sampling_rate"]
        self.trim_top_db = a.get("trim_top_db", 35)
        self.n_fft, self.hop, self.win = s["filter_length"], s["hop_length"], s["win_length"]
        self.n_mels, self.fmin, self.fmax = m["n_mel_channels"], m["mel_fmin"], m["mel_fmax"]
        self.f0_min, self.f0_max = p.get("f0_min", 60.0), p.get("f0_max", 600.0)
        self.mel_basis = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                                             fmin=self.fmin, fmax=self.fmax)

    # ---------------------------------------------------------------- I/O
    def load(self, path_or_file, trim=True):
        wav, sr = sf.read(path_or_file, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        return self.prepare(wav, sr, trim)

    def prepare(self, wav, sr, trim=True):
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != self.sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        if trim and self.trim_top_db:
            wav, _ = librosa.effects.trim(wav, top_db=self.trim_top_db, frame_length=self.n_fft,
                                          hop_length=self.hop)
        peak = np.abs(wav).max() if len(wav) else 0.0
        if peak > 0:
            wav = wav / peak * 0.95
        return wav.astype(np.float32)

    def save(self, path, wav):
        sf.write(path, np.clip(wav, -1, 1), self.sr, subtype="PCM_16")

    # ---------------------------------------------------------------- features
    def mel_and_energy(self, wav):
        spec = np.abs(librosa.stft(wav, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
                                   window="hann", center=True, pad_mode="reflect"))
        mel = np.log(np.clip(self.mel_basis @ spec, 1e-5, None))
        energy = np.linalg.norm(spec, axis=0)
        return mel.T.astype(np.float32), energy.astype(np.float32)  # (T, n_mels), (T,)

    def pitch(self, wav, n_frames):
        f0 = None
        try:
            import pyworld as pw
            x = wav.astype(np.float64)
            fp = self.hop / self.sr * 1000
            f0, t = pw.dio(x, self.sr, f0_floor=self.f0_min, f0_ceil=self.f0_max, frame_period=fp)
            f0 = pw.stonemask(x, f0, t, self.sr)
        except ImportError:
            f0, _, _ = librosa.pyin(wav, fmin=self.f0_min, fmax=self.f0_max, sr=self.sr,
                                    frame_length=self.n_fft, hop_length=self.hop, center=True)
            f0 = np.nan_to_num(f0)
        f0 = np.asarray(f0, dtype=np.float32)
        if len(f0) >= n_frames:
            f0 = f0[:n_frames]
        else:
            f0 = np.pad(f0, (0, n_frames - len(f0)))
        return f0

    @staticmethod
    def interpolate_f0(f0):
        """Linear interpolation over unvoiced frames (reference-repo behaviour)."""
        voiced = f0 > 0
        if voiced.sum() < 2:
            return None
        idx = np.arange(len(f0))
        return np.interp(idx, idx[voiced], f0[voiced]).astype(np.float32)

    # ---------------------------------------------------------------- vocoder fallback
    def griffin_lim(self, mel, n_iter=60):
        """mel: (T, n_mels) natural-log mel -> waveform."""
        mag = np.exp(mel.T)
        return librosa.feature.inverse.mel_to_audio(
            mag, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            power=1.0, fmin=self.fmin, fmax=self.fmax, n_iter=n_iter)
