"""
291-D audio vector per clip:
  - 128: MFCC means over time (n_mfcc=128)
  - 128: log-mel means over time (n_mels=128)
  - 35:  OpenSMILE eGeMAPSv02 low-level descriptors — mean over 25 bands +
        std over first 10 bands (25 + 10 = 35)

``.mp4`` / ``.mkv`` / … are decoded with **ffmpeg**: uses ``ffmpeg`` on PATH if
present, otherwise the binary shipped by **imageio-ffmpeg** (no admin install).
``.wav`` uses librosa/soundfile when possible.
"""

from __future__ import annotations

import shutil
import subprocess
from io import BytesIO

import numpy as np
import soundfile as sf

TARGET_SR = 16000
N_MFCC = 128
N_MELS = 128
N_FFT = 2048
HOP = 512
PROSODY_DIM = 35  # 25 LLD means + 10 LLD std (first 10 coeffs)

_VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".webm", ".mov")


def _resolve_ffmpeg_exe() -> str:
    system = shutil.which("ffmpeg")
    if system:
        return system
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _load_audio_ffmpeg(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    ffmpeg = _resolve_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")[:500]
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err}")
    y, sr = sf.read(BytesIO(proc.stdout), always_2d=False, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


class Audio291Extractor:
    """Reuses one OpenSMILE extractor (cheap to keep per process)."""

    def __init__(self) -> None:
        import opensmile

        self._lld = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

    def from_waveform(self, y: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return np.zeros(N_MFCC + N_MELS + PROSODY_DIM, dtype=np.float32)

        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR).astype(
                np.float32
            )
            sr = TARGET_SR

        min_len = int(0.5 * sr)
        if y.size < min_len:
            y = np.pad(y, (0, min_len - y.size))

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP,
            n_mels=N_MELS,
            fmax=sr // 2,
        )
        mfcc_mean = mfcc.mean(axis=1).astype(np.float32)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP,
            fmax=sr // 2,
        )
        mel_db = librosa.power_to_db(mel + 1e-10)
        mel_mean = mel_db.mean(axis=1).astype(np.float32)

        lld = self._lld.process_signal(y, int(sr)).values
        mean25 = lld.mean(axis=0).astype(np.float32)
        std10 = lld[:, :10].std(axis=0).astype(np.float32)
        prosody = np.concatenate([mean25, std10])

        out = np.concatenate([mfcc_mean, mel_mean, prosody]).astype(np.float32)
        assert out.shape == (291,), out.shape
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def from_media_path(self, path: str) -> np.ndarray:
        lower = path.lower()
        if lower.endswith(_VIDEO_SUFFIXES):
            y, sr = _load_audio_ffmpeg(path, TARGET_SR)
        else:
            import librosa

            y, sr = librosa.load(path, sr=None, mono=True)
        return self.from_waveform(y, int(sr))
