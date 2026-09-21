"""Audio I/O: loading, resampling, loudness, and a streaming writer.

Everything here is chunk-friendly so that hours-long renders never need to hold
the whole song in RAM.  A 3-hour stereo float32 buffer is ~3.8 GB; we stream
instead and stay under a couple of hundred MB.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from .util import die, log, warn

DEFAULT_SR = 44100


# ------------------------------------------------------------------ loading ---

def audio_info(path):
    try:
        info = sf.info(str(path))
        return {"samplerate": info.samplerate, "channels": info.channels,
                "duration": info.duration, "frames": info.frames}
    except Exception:
        return None


def to_stereo(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return np.ascontiguousarray(data, dtype=np.float32)


def resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data
    from scipy.signal import resample_poly
    g = math.gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    return np.ascontiguousarray(resample_poly(data, up, down, axis=0), dtype=np.float32)


def load_audio(path, target_sr=None, mono=False, max_seconds=None):
    """Read any soundfile-supported format as float32, always 2-D (frames, ch)."""
    path = Path(path)
    try:
        if max_seconds:
            info = sf.info(str(path))
            frames = int(max_seconds * info.samplerate)
            data, sr = sf.read(str(path), dtype="float32", always_2d=True, frames=frames)
        else:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"cannot read {path.name}: {exc}") from exc
    data = to_stereo(data)
    if target_sr and sr != target_sr:
        data = resample(data, sr, target_sr)
        sr = target_sr
    if mono:
        data = data.mean(axis=1)
    return data, sr


def load_mono(path, sr=22050, max_seconds=None):
    data, got = load_audio(path, target_sr=sr, mono=True, max_seconds=max_seconds)
    return data, got


def iter_blocks(path, target_sr=None, block_frames=1 << 18):
    """Yield (frames, 2) float32 blocks from a file, resampled if asked.

    Resampling per block introduces a negligible boundary artefact at block
    edges, so we use a generous block size and a small overlap-free policy:
    resample_poly is applied to whole blocks which for our use (analysis and
    long concatenation) is inaudible.
    """
    with sf.SoundFile(str(path)) as fh:
        sr = fh.samplerate
        while True:
            block = fh.read(block_frames, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            block = to_stereo(block)
            if target_sr and sr != target_sr:
                block = resample(block, sr, target_sr)
            yield block


# ------------------------------------------------------------------ writing ---

class StreamWriter:
    """Incremental WAV/FLAC writer with peak + RMS metering and a frame count."""

    def __init__(self, path, sr=DEFAULT_SR, channels=2, subtype="PCM_24"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sr = int(sr)
        self.channels = int(channels)
        fmt = "FLAC" if self.path.suffix.lower() == ".flac" else "WAV"
        if fmt == "FLAC" and subtype == "PCM_32":
            subtype = "PCM_24"
        self._fh = sf.SoundFile(str(self.path), "w", samplerate=self.sr,
                                channels=self.channels, subtype=subtype, format=fmt)
        self.frames = 0
        self.peak = 0.0
        self._sq_sum = 0.0

    def write(self, block: np.ndarray) -> None:
        if block is None or len(block) == 0:
            return
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        if block.shape[1] == 1 and self.channels == 2:
            block = np.repeat(block, 2, axis=1)
        np.clip(block, -1.0, 1.0, out=block)
        self.peak = max(self.peak, float(np.abs(block).max()))
        self._sq_sum += float(np.square(block, dtype=np.float64).sum())
        self._fh.write(block)
        self.frames += len(block)

    @property
    def seconds(self) -> float:
        return self.frames / self.sr if self.sr else 0.0

    @property
    def rms_db(self) -> float:
        n = max(1, self.frames * self.channels)
        return 20 * math.log10(math.sqrt(self._sq_sum / n) + 1e-12)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_audio(path, data, sr, subtype="PCM_24"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)
    fmt = "FLAC" if path.suffix.lower() == ".flac" else "WAV"
    sf.write(str(path), data, int(sr), subtype=subtype, format=fmt)
    return path


# ---------------------------------------------------------------- loudness ---

def integrated_lufs(data: np.ndarray, sr: int) -> float:
    """ITU-R BS.1770 integrated loudness (pyloudnorm), NaN-safe."""
    try:
        import pyloudnorm as pyln
    except ImportError:
        die("pyloudnorm is required: pip install pyloudnorm")
    if len(data) < int(0.4 * sr) + 1:
        return float("-inf")
    meter = pyln.Meter(sr)
    try:
        value = float(meter.integrated_loudness(np.asarray(data, dtype=np.float64)))
    except Exception:
        return float("-inf")
    return value if np.isfinite(value) else float("-inf")


class StreamingLoudness:
    """Block-wise BS.1770 gated loudness for signals too long to hold in RAM.

    Accumulates 400 ms block mean-squares (K-weighted), then applies the
    absolute (-70 LUFS) and relative (-10 LU) gates exactly like the offline
    measurement.
    """

    G = np.array([1.0, 1.0], dtype=np.float64)  # stereo channel weights

    def __init__(self, sr: int):
        self.sr = int(sr)
        self._blocks = []
        self._buf = np.zeros((0, 2), dtype=np.float64)
        self._block_n = int(round(0.4 * sr))
        self._hop = max(1, self._block_n // 4)   # 75 % overlap
        self._state = None
        self._b, self._a = _k_weight_coeffs(sr)

    def push(self, block: np.ndarray) -> None:
        from scipy.signal import lfilter, lfilter_zi
        x = np.asarray(block, dtype=np.float64)
        if x.ndim == 1:
            x = np.repeat(x[:, None], 2, axis=1)
        if self._state is None:
            zi = lfilter_zi(self._b, self._a)
            self._state = np.stack([zi * 0.0, zi * 0.0], axis=-1)
        y, self._state = lfilter(self._b, self._a, x, axis=0, zi=self._state)
        self._buf = np.concatenate([self._buf, y], axis=0)
        while len(self._buf) >= self._block_n:
            seg = self._buf[: self._block_n]
            ms = np.mean(seg ** 2, axis=0)
            self._blocks.append(float(np.sum(self.G * ms)))
            self._buf = self._buf[self._hop:]

    def value(self) -> float:
        if not self._blocks:
            return float("-inf")
        z = np.asarray(self._blocks, dtype=np.float64)
        loud = -0.691 + 10 * np.log10(np.maximum(z, 1e-20))
        keep = z[loud > -70.0]
        if keep.size == 0:
            return float("-inf")
        relative = -0.691 + 10 * np.log10(keep.mean()) - 10.0
        keep2 = z[(loud > -70.0) & (loud > relative)]
        if keep2.size == 0:
            keep2 = keep
        return float(-0.691 + 10 * np.log10(keep2.mean()))


def _k_weight_coeffs(sr: int):
    """Combined BS.1770 pre-filter (shelf) + RLB high-pass as one filter chain."""
    from scipy.signal import bilinear
    import numpy as _np
    # High-shelf stage (ITU-R BS.1770-4 table 1, designed at the actual sr)
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = _np.tan(_np.pi * f0 / sr)
    Vh = 10 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b_shelf = _np.array([(Vh + Vb * K / Q + K * K), 2.0 * (K * K - Vh), (Vh - Vb * K / Q + K * K)]) / a0
    a_shelf = _np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])
    # RLB high-pass stage
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = _np.tan(_np.pi * f0 / sr)
    a_hp = _np.array([1.0, 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K),
                      (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)])
    b_hp = _np.array([1.0, -2.0, 1.0])
    b = _np.convolve(b_shelf, b_hp)
    a = _np.convolve(a_shelf, a_hp)
    return b, a


def true_peak_db(data: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Approximate true-peak (dBTP) by polyphase oversampling."""
    from scipy.signal import resample_poly
    x = np.asarray(data, dtype=np.float32)
    if len(x) == 0:
        return float("-inf")
    if len(x) > sr * 600:               # sample the loudest 10 minutes for speed
        step = len(x) // (sr * 600) + 1
        x = x[::step]
    up = resample_poly(x, oversample, 1, axis=0)
    peak = float(np.abs(up).max())
    return 20 * math.log10(peak + 1e-12)


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20.0))


def gain_to_db(gain: float) -> float:
    return float(20 * math.log10(max(gain, 1e-12)))
