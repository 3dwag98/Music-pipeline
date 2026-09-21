"""Streaming DSP building blocks.

Every processor keeps its own state and accepts blocks of any length, so the
exact same chain works for a 30-second preview and for a 6-hour render.
"""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------- biquads ---

def _biquad(kind, sr, freq, q=0.7071, gain_db=0.0):
    freq = float(np.clip(freq, 10.0, sr * 0.49))
    w0 = TWO_PI * freq / sr
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    A = 10 ** (gain_db / 40.0)
    if kind == "lowpass":
        b = [(1 - cos_w0) / 2, 1 - cos_w0, (1 - cos_w0) / 2]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
    elif kind == "notch":
        b = [1.0, -2 * cos_w0, 1.0]
        a = [1 + alpha, -2 * cos_w0, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cos_w0, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cos_w0, 1 - alpha / A]
    elif kind == "lowshelf":
        sq = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cos_w0 + sq),
             2 * A * ((A - 1) - (A + 1) * cos_w0),
             A * ((A + 1) - (A - 1) * cos_w0 - sq)]
        a = [(A + 1) + (A - 1) * cos_w0 + sq,
             -2 * ((A - 1) + (A + 1) * cos_w0),
             (A + 1) + (A - 1) * cos_w0 - sq]
    elif kind == "highshelf":
        sq = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cos_w0 + sq),
             -2 * A * ((A - 1) + (A + 1) * cos_w0),
             A * ((A + 1) + (A - 1) * cos_w0 - sq)]
        a = [(A + 1) - (A - 1) * cos_w0 + sq,
             2 * ((A - 1) - (A + 1) * cos_w0),
             (A + 1) - (A - 1) * cos_w0 - sq]
    else:
        raise ValueError(f"unknown filter kind {kind!r}")
    b = np.asarray(b, dtype=np.float64) / a[0]
    a = np.asarray(a, dtype=np.float64) / a[0]
    return b, a


class Biquad:
    """One stateful biquad section (stereo-aware)."""

    def __init__(self, kind, sr, freq, q=0.7071, gain_db=0.0, channels=2):
        self.b, self.a = _biquad(kind, sr, freq, q, gain_db)
        self.channels = channels
        self._zi = np.zeros((2, channels), dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter
        x = np.asarray(x, dtype=np.float64)
        squeeze = x.ndim == 1
        if squeeze:
            x = x[:, None]
        if x.shape[1] != self._zi.shape[1]:
            self._zi = np.zeros((2, x.shape[1]), dtype=np.float64)
        y, self._zi = lfilter(self.b, self.a, x, axis=0, zi=self._zi)
        y = y.astype(np.float32)
        return y[:, 0] if squeeze else y


class FilterChain:
    def __init__(self, *filters):
        self.filters = list(filters)

    def process(self, x):
        for f in self.filters:
            x = f.process(x)
        return x


def static_filter(kind, sr, freq, q=0.7071, gain_db=0.0):
    """One-shot (non-streaming) filter for whole buffers."""
    from scipy.signal import lfilter
    b, a = _biquad(kind, sr, freq, q, gain_db)
    return lambda x: lfilter(b, a, np.asarray(x, dtype=np.float64), axis=0).astype(np.float32)


# ------------------------------------------------------------- saturation ---

def soft_clip(x, drive=1.0):
    return np.tanh(np.asarray(x, dtype=np.float32) * drive) / math.tanh(max(drive, 1e-6))


def tape_saturate(x, drive=1.4, asym=0.06):
    """Asymmetric soft saturation: adds mostly 2nd/3rd harmonics like tape."""
    x = np.asarray(x, dtype=np.float32)
    y = np.tanh(drive * (x + asym * x * x))
    return (y / math.tanh(drive)).astype(np.float32)


def bit_crush(x, bits=12, sr_ratio=1):
    """Quantise to `bits` and optionally sample-and-hold by `sr_ratio`."""
    x = np.asarray(x, dtype=np.float32)
    if sr_ratio > 1:
        n = len(x)
        keep = x[::sr_ratio]
        x = np.repeat(keep, sr_ratio, axis=0)[:n]
        if len(x) < n:
            pad = np.repeat(x[-1:], n - len(x), axis=0)
            x = np.concatenate([x, pad], axis=0)
    levels = float(2 ** (bits - 1))
    return (np.round(x * levels) / levels).astype(np.float32)


# -------------------------------------------------------------- dynamics ----

def _env_follow(key, state, a_att, a_rel):
    """Switched one-pole attack/release follower (runs on the control-rate signal)."""
    key = np.asarray(key, dtype=np.float64)
    out = np.empty_like(key)
    for i in range(key.shape[0]):
        v = key[i]
        a = a_att if v > state else a_rel
        state = a * state + (1.0 - a) * v
        out[i] = state
    return out


def _limiter_gain(target, state, a_rel):
    """Instant-attack, smoothed-release gain envelope."""
    out = np.empty_like(target)
    for i in range(target.shape[0]):
        t = target[i]
        state = t if t < state else a_rel * state + (1.0 - a_rel) * t
        out[i] = state
    return out


try:  # optional: numba compiles the two loops above (~40x, exact per-sample)
    from numba import njit as _njit  # type: ignore
    _env_follow = _njit(cache=True, fastmath=True)(_env_follow)
    _limiter_gain = _njit(cache=True, fastmath=True)(_limiter_gain)
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is entirely optional
    HAVE_NUMBA = False


#: Control-rate decimation.  Envelopes are computed every N samples and the
#: gain is held back up to audio rate.  At 44.1 kHz, N=16 gives a 2.7 kHz
#: control rate - transparent for attacks >= 2 ms and ~16x faster than a
#: per-sample Python loop, which matters on multi-hour renders.
CONTROL_DECIM = 1 if HAVE_NUMBA else 16


def _decimate(key, decim, mode="max"):
    n = len(key)
    if decim <= 1:
        return np.asarray(key, dtype=np.float64), n
    pad = (-n) % decim
    if pad:
        key = np.concatenate([key, np.repeat(key[-1:], pad)])
    frames = np.asarray(key, dtype=np.float64).reshape(-1, decim)
    return (frames.max(axis=1) if mode == "max" else frames.min(axis=1)), n


def _upsample_hold(ctrl, n, decim):
    if decim <= 1:
        return np.asarray(ctrl)[:n]
    return np.repeat(ctrl, decim)[:n]


class Compressor:
    """Feed-forward peak compressor with a soft knee, stateful across blocks."""

    def __init__(self, sr, threshold_db=-18.0, ratio=3.0, attack_ms=12.0,
                 release_ms=180.0, knee_db=6.0, makeup_db=0.0, decim=None):
        self.sr = sr
        self.threshold = float(threshold_db)
        self.ratio = max(1.0, float(ratio))
        self.knee = max(0.0, float(knee_db))
        self.makeup = 10 ** (makeup_db / 20.0)
        self.decim = int(decim or CONTROL_DECIM)
        ctrl_sr = sr / self.decim
        self.a_att = math.exp(-1.0 / max(1e-6, max(0.1, attack_ms) * 0.001 * ctrl_sr))
        self.a_rel = math.exp(-1.0 / max(1e-6, max(1.0, release_ms) * 0.001 * ctrl_sr))
        self._env = 0.0
        self.last_reduction_db = 0.0

    def _curve(self, level_db):
        over = level_db - self.threshold
        out = np.zeros_like(over)
        if self.knee > 0:
            half = self.knee / 2.0
            upper = over >= half
            mid = (over > -half) & (~upper)
            out[upper] = over[upper] - over[upper] / self.ratio
            t = over[mid] + half
            out[mid] = (1.0 - 1.0 / self.ratio) * (t * t) / (2.0 * self.knee)
        else:
            mask = over > 0
            out[mask] = over[mask] - over[mask] / self.ratio
        return out

    def process(self, x, sidechain=None):
        x = np.asarray(x, dtype=np.float32)
        squeeze = x.ndim == 1
        if squeeze:
            x = x[:, None]
        if len(x) == 0:
            return x[:, 0] if squeeze else x
        src = x if sidechain is None else np.asarray(sidechain, dtype=np.float32).reshape(len(x), -1)
        key = np.abs(src).max(axis=1)
        ctrl, n = _decimate(key, self.decim, "max")
        env = _env_follow(ctrl, self._env, self.a_att, self.a_rel)
        self._env = float(env[-1])
        reduction = self._curve(20 * np.log10(np.maximum(env, 1e-9)))
        self.last_reduction_db = float(reduction.max()) if len(reduction) else 0.0
        gain = _upsample_hold(10 ** (-reduction / 20.0), n, self.decim)
        y = x * gain.astype(np.float32)[:, None] * self.makeup
        return y[:, 0] if squeeze else y


class Limiter:
    """Look-ahead brickwall limiter that keeps peaks under `ceiling_db`."""

    def __init__(self, sr, ceiling_db=-1.0, lookahead_ms=5.0, release_ms=120.0,
                 decim=None, true_peak=True, oversample=4):
        self.sr = sr
        self.ceiling = 10 ** (ceiling_db / 20.0)
        self.true_peak = bool(true_peak)
        self.oversample = int(oversample) if true_peak else 1
        self.look = max(1, int(lookahead_ms * 0.001 * sr))
        self.decim = int(decim or CONTROL_DECIM)
        ctrl_sr = sr / self.decim
        self.a_rel = math.exp(-1.0 / max(1e-6, max(1.0, release_ms) * 0.001 * ctrl_sr))
        self._delay = np.zeros((self.look, 2), dtype=np.float32)
        self._gain = 1.0

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        squeeze = x.ndim == 1
        if squeeze:
            x = np.repeat(x[:, None], 2, axis=1)
        elif x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        if len(x) == 0:
            return x[:, 0] if squeeze else x
        buf = np.concatenate([self._delay, x], axis=0)
        peak = self._detect(buf)
        # a peak `look` samples ahead must already start pulling the gain down
        ahead = np.concatenate([peak[self.look:], np.zeros(self.look)])
        peak = np.maximum(peak, ahead)
        target = np.minimum(1.0, self.ceiling / np.maximum(peak, 1e-9))
        ctrl, n = _decimate(target, self.decim, "min")
        gain = _limiter_gain(ctrl, self._gain, self.a_rel)
        self._gain = float(gain[-1])
        out = buf * _upsample_hold(gain, n, self.decim).astype(np.float32)[:, None]
        self._delay = buf[len(buf) - self.look:]
        y = np.ascontiguousarray(out[: len(x)])
        np.clip(y, -self.ceiling, self.ceiling, out=y)
        return y[:, 0] if squeeze else y

    def _detect(self, buf):
        """Per-sample peak used to compute gain.

        In true-peak mode the detector runs on a 4x-oversampled copy, so the
        gain accounts for what the waveform does BETWEEN samples.  A plain
        sample-domain limiter set to -1.0 dBFS can still reconstruct at over
        +2 dBTP on dense material, which then clips in any lossy encode - this
        is what stops that.  The gain is still applied at base rate, which is
        standard practice and keeps the cost to one polyphase resample.
        """
        if not self.true_peak or self.oversample <= 1:
            return np.abs(buf).max(axis=1).astype(np.float64)
        from scipy.signal import resample_poly
        up = resample_poly(buf, self.oversample, 1, axis=0)
        mono = np.abs(up).max(axis=1)
        need = len(buf) * self.oversample
        if len(mono) < need:
            mono = np.concatenate([mono, np.repeat(mono[-1:], need - len(mono))])
        return mono[:need].reshape(len(buf), self.oversample).max(axis=1).astype(np.float64)

    def flush(self):
        tail = np.clip(self._delay * self._gain, -self.ceiling, self.ceiling)
        self._delay = np.zeros_like(self._delay)
        return tail


class Ducker:
    """Sidechain ducking - the gentle lofi 'pump' under every kick."""

    def __init__(self, sr, depth_db=-3.0, attack_ms=5.0, release_ms=160.0, decim=None):
        self.depth = 10 ** (depth_db / 20.0)
        self.decim = int(decim or CONTROL_DECIM)
        ctrl_sr = sr / self.decim
        self.a_att = math.exp(-1.0 / max(1e-6, max(0.1, attack_ms) * 0.001 * ctrl_sr))
        self.a_rel = math.exp(-1.0 / max(1e-6, max(1.0, release_ms) * 0.001 * ctrl_sr))
        self._env = 0.0
        self._ref = 1e-6

    def process(self, x, trigger):
        x = np.asarray(x, dtype=np.float32)
        if len(x) == 0:
            return x
        trig = np.abs(np.asarray(trigger, dtype=np.float32))
        if trig.ndim > 1:
            trig = trig.max(axis=1)
        ctrl, n = _decimate(trig, self.decim, "max")
        env = _env_follow(ctrl, self._env, self.a_att, self.a_rel)
        self._env = float(env[-1])
        self._ref = max(self._ref * 0.999, float(env.max()) if len(env) else 0.0, 1e-6)
        shaped = np.clip(env / self._ref, 0.0, 1.0)
        gain = _upsample_hold(1.0 - (1.0 - self.depth) * shaped, n, self.decim).astype(np.float32)
        return x * (gain[:, None] if x.ndim == 2 else gain)


# ----------------------------------------------------------------- reverb ----

def _ring_read(line, idx, count):
    n = len(line)
    end = idx + count
    if end <= n:
        return line[idx:end]
    return np.concatenate([line[idx:], line[: end - n]])


def _ring_write(line, idx, data):
    n = len(line)
    end = idx + len(data)
    if end <= n:
        line[idx:end] = data
    else:
        first = n - idx
        line[idx:] = data[:first]
        line[: end - n] = data[first:]


class FDNReverb:
    """8-line feedback-delay-network reverb: smooth, cheap and streamable.

    All state lives in the delay lines, so a tail carries correctly across
    block boundaries - essential when an hours-long song is rendered section by
    section.  Processing is chunked at the shortest delay length, which makes
    every read depend only on already-written history and lets the inner loop
    run as vectorised numpy instead of per-sample Python.
    """

    PRIMES = [1153, 1523, 1789, 2063, 2377, 2699, 3011, 3323]

    def __init__(self, sr, room=0.72, damping=0.32, width=1.0, predelay_ms=18.0):
        self.sr = sr
        scale = sr / 44100.0
        self.lengths = [max(64, int(p * scale)) for p in self.PRIMES]
        self.m = len(self.lengths)
        self.lines = [np.zeros(n, dtype=np.float32) for n in self.lengths]
        self.idx = [0] * self.m
        self.chunk = min(self.lengths)
        self.feedback = float(np.clip(0.55 + 0.42 * float(room), 0.0, 0.985))
        self.damping = float(np.clip(damping, 0.0, 0.95))
        self.width = float(np.clip(width, 0.0, 1.0))
        self.pre = max(0, int(predelay_ms * 0.001 * sr))
        self._pre_buf = np.zeros((self.pre, 2), dtype=np.float32) if self.pre else None
        self._lp = np.zeros((1, self.m), dtype=np.float64)
        h = np.array([[1.0]])
        while h.shape[0] < self.m:
            h = np.block([[h, h], [h, -h]])
        self.mix = (h[: self.m, : self.m] / math.sqrt(self.m)).astype(np.float32)
        self.inject = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(self.m)], dtype=np.float32)
        self.left_taps = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(self.m)], dtype=np.float32)
        self.right_taps = 1.0 - self.left_taps

    def process(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = np.repeat(x[:, None], 2, axis=1)
        if len(x) == 0:
            return x
        if self.pre:
            buf = np.concatenate([self._pre_buf, x], axis=0)
            self._pre_buf = buf[len(buf) - self.pre:]
            x = buf[: len(x)]
        mono = x.mean(axis=1).astype(np.float32)
        n = len(mono)
        out_l = np.empty(n, dtype=np.float32)
        out_r = np.empty(n, dtype=np.float32)
        b = np.array([1.0 - self.damping])
        a = np.array([1.0, -self.damping])
        mix_t = self.mix.T * self.feedback
        inject = (0.25 * self.inject)[None, :]
        pos = 0
        while pos < n:
            count = min(self.chunk, n - pos)
            taps = np.empty((count, self.m), dtype=np.float32)
            for j in range(self.m):
                taps[:, j] = _ring_read(self.lines[j], self.idx[j], count)
            out_l[pos:pos + count] = (taps @ self.left_taps) * 0.35
            out_r[pos:pos + count] = (taps @ self.right_taps) * 0.35
            fed = taps @ mix_t + mono[pos:pos + count, None] * inject
            damped, self._lp = lfilter(b, a, fed.astype(np.float64), axis=0, zi=self._lp)
            damped = damped.astype(np.float32)
            for j in range(self.m):
                _ring_write(self.lines[j], self.idx[j], damped[:, j])
                self.idx[j] = (self.idx[j] + count) % self.lengths[j]
            pos += count
        w = self.width
        left = out_l * (0.5 + 0.5 * w) + out_r * (0.5 - 0.5 * w)
        right = out_r * (0.5 + 0.5 * w) + out_l * (0.5 - 0.5 * w)
        return np.stack([left, right], axis=1).astype(np.float32)


# ------------------------------------------------------- tape wow & flutter ---

class TapeWow:
    """Time-varying fractional delay = pitch drift.  The signature lofi wobble."""

    def __init__(self, sr, wow_hz=0.7, wow_ms=2.6, flutter_hz=7.3, flutter_ms=0.35,
                 drift=0.35, seed=0):
        self.sr = sr
        self.rng = np.random.default_rng(seed)
        self.wow_hz, self.wow_ms = wow_hz, wow_ms
        self.flutter_hz, self.flutter_ms = flutter_hz, flutter_ms
        self.drift = drift
        self.max_delay = int((wow_ms + flutter_ms + 6.0) * 0.001 * sr) + 4
        self._buf = np.zeros((self.max_delay, 2), dtype=np.float32)
        self._phase_w = self.rng.random() * TWO_PI
        self._phase_f = self.rng.random() * TWO_PI
        self._phase_d = self.rng.random() * TWO_PI
        self._t = 0

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = np.repeat(x[:, None], 2, axis=1)
        n = len(x)
        t = (np.arange(n) + self._t) / self.sr
        wow = np.sin(TWO_PI * self.wow_hz * t + self._phase_w) * self.wow_ms
        flut = np.sin(TWO_PI * self.flutter_hz * t + self._phase_f) * self.flutter_ms
        slow = np.sin(TWO_PI * 0.09 * t + self._phase_d) * self.wow_ms * self.drift
        delay_samp = np.clip((wow + flut + slow + self.wow_ms + 2.0) * 0.001 * self.sr,
                             1.0, self.max_delay - 2.0)
        buf = np.concatenate([self._buf, x], axis=0)
        base = np.arange(n, dtype=np.float64) + self.max_delay
        read = base - delay_samp
        i0 = np.floor(read).astype(np.int64)
        frac = (read - i0).astype(np.float32)[:, None]
        i0 = np.clip(i0, 0, len(buf) - 2)
        out = buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac
        self._buf = buf[len(buf) - self.max_delay:]
        self._t += n
        return out.astype(np.float32)


class VinylNoise:
    """Continuous vinyl bed: pink-ish noise floor, sparse crackle, 33 rpm ticks."""

    def __init__(self, sr, level_db=-38.0, crackle=1.0, rpm=33.3333, seed=0, hiss_db=-52.0):
        self.sr = sr
        self.level = 10 ** (level_db / 20.0)
        self.hiss = 10 ** (hiss_db / 20.0)
        self.crackle = float(crackle)
        self.period = int(sr * 60.0 / rpm) if rpm else 0
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._pink_state = [np.zeros(1) for _ in range(3)]
        self._lp = Biquad("lowpass", sr, 9000.0, 0.7, channels=2)
        self._hp = Biquad("highpass", sr, 38.0, 0.7, channels=2)

    def _pink_noise(self, n):
        """Paul Kellett pink filter as three one-poles - vectorised via lfilter."""
        from scipy.signal import lfilter
        white = self.rng.standard_normal(n)
        out = np.zeros(n)
        poles = ((0.99765, 0.0990460), (0.96300, 0.2965164), (0.57000, 1.0526913))
        for k, (pole, gain) in enumerate(poles):
            y, self._pink_state[k] = lfilter([gain], [1.0, -pole], white, zi=self._pink_state[k])
            out += y
        return (out + white * 0.1848) * 0.11

    def block(self, n):
        bed = self._pink_noise(n)
        noise = np.stack([bed, np.roll(bed, 7)], axis=1).astype(np.float32) * self.level
        noise += self.rng.standard_normal((n, 2)).astype(np.float32) * self.hiss
        if self.crackle > 0:
            rate = 24.0 * self.crackle
            count = self.rng.poisson(rate * n / self.sr)
            if count:
                pos = self.rng.integers(0, max(1, n - 3), size=count)
                amp = (self.rng.random(count) ** 3.2).astype(np.float32) * 0.42 * self.crackle
                sign = self.rng.choice([-1.0, 1.0], size=count).astype(np.float32)
                for k in range(count):
                    p = int(pos[k])
                    a = amp[k] * sign[k]
                    noise[p, 0] += a
                    noise[p, 1] += a * float(self.rng.random() * 0.8 + 0.2)
                    noise[p + 1] -= a * 0.5
        if self.period:
            first = (-self._t) % self.period
            for p in range(int(first), n, self.period):
                if p + 2 < n:
                    noise[p] += 0.05 * self.crackle
                    noise[p + 1] -= 0.03 * self.crackle
        self._t += n
        return self._hp.process(self._lp.process(noise))


class StereoWiden:
    """Mid/side widener with a mono-safe bass split."""

    def __init__(self, sr, width=1.25, bass_mono_hz=140.0):
        self.width = float(width)
        self._lp = Biquad("lowpass", sr, bass_mono_hz, 0.7, channels=2)

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            return x
        low = self._lp.process(x)
        high = x - low
        mid = high.mean(axis=1, keepdims=True)
        side = (high[:, :1] - high[:, 1:2]) * 0.5 * self.width
        wide = np.concatenate([mid + side, mid - side], axis=1)
        mono_low = low.mean(axis=1, keepdims=True)
        return (wide + np.repeat(mono_low, 2, axis=1)).astype(np.float32)


def haas(x, sr, ms=12.0, mix=0.35):
    """Tiny delay on one side - instant width for mono sources."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = np.repeat(x[:, None], 2, axis=1)
    d = int(ms * 0.001 * sr)
    if d <= 0:
        return x
    out = x.copy()
    out[d:, 1] = x[d:, 1] * (1 - mix) + x[:-d, 1] * mix
    return out


# ---------------------------------------------------------------- utilities ---

def equal_power_fade(n, fade_in=True):
    if n <= 1:
        return np.ones((max(n, 0), 1), dtype=np.float32)
    t = np.linspace(0.0, math.pi / 2, n, dtype=np.float32)
    curve = np.sin(t) if fade_in else np.cos(t)
    return curve[:, None]


def cosine_fade(n, fade_in=True):
    if n <= 1:
        return np.ones((max(n, 0), 1), dtype=np.float32)
    curve = (0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, n))).astype(np.float32)
    if not fade_in:
        curve = curve[::-1]
    return curve[:, None]


def dc_block(x, sr):
    return Biquad("highpass", sr, 22.0, 0.7, channels=2 if np.ndim(x) > 1 else 1).process(x)


def normalise_peak(x, peak_db=-1.0):
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.abs(x).max())
    if peak <= 0:
        return x
    return (x * (10 ** (peak_db / 20.0) / peak)).astype(np.float32)
