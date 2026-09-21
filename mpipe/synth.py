"""Instrument synthesis for the local lofi engine.

Every sound is generated from scratch - no samples, no loops, no third-party
audio of any kind.  That is deliberate: it is what makes the output original
and keeps it clear of anyone else's recording rights.

Each voice returns a mono float32 array.  Rendered notes are cached by
(instrument, midi, duration, velocity) because a song repeats the same notes
hundreds of times; the cache is what makes hours-long renders practical.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .theory import midi_to_hz

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------- envelopes ---

def adsr(n, sr, attack=0.005, decay=0.25, sustain=0.6, release=0.3, curve=2.0):
    """Exponential-ish ADSR of exactly `n` samples."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    a = min(int(attack * sr), n)
    d = min(int(decay * sr), max(0, n - a))
    r = min(int(release * sr), max(0, n - a - d))
    s = max(0, n - a - d - r)
    parts = []
    if a:
        parts.append(np.linspace(0.0, 1.0, a, dtype=np.float32) ** (1.0 / curve))
    if d:
        parts.append(sustain + (1.0 - sustain) * np.linspace(1.0, 0.0, d, dtype=np.float32) ** curve)
    if s:
        parts.append(np.full(s, sustain, dtype=np.float32))
    if r:
        parts.append(sustain * np.linspace(1.0, 0.0, r, dtype=np.float32) ** curve)
    env = np.concatenate(parts) if parts else np.zeros(n, dtype=np.float32)
    if len(env) < n:
        env = np.concatenate([env, np.zeros(n - len(env), dtype=np.float32)])
    return env[:n].astype(np.float32)


def perc_env(n, sr, attack=0.002, decay=0.4, curve=3.5):
    """Percussive: instant attack, exponential decay (piano, mallets, plucks)."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t / max(1e-4, decay / curve))
    a = max(1, int(attack * sr))
    env[:a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)
    return env.astype(np.float32)


def _lp(x, sr, freq, order=1):
    """One-shot lowpass used to shape a single rendered note."""
    from scipy.signal import butter, lfilter
    b, a = butter(order, min(0.99, freq / (sr * 0.5)), btype="low")
    return lfilter(b, a, np.asarray(x, dtype=np.float64)).astype(np.float32)


def _bp(x, sr, low, high, order=2):
    from scipy.signal import butter, lfilter
    lo = max(1e-4, low / (sr * 0.5))
    hi = min(0.99, high / (sr * 0.5))
    b, a = butter(order, [lo, hi], btype="band")
    return lfilter(b, a, np.asarray(x, dtype=np.float64)).astype(np.float32)


def _fade_tail(x, sr, ms=8.0):
    """Always end on zero so concatenated notes never click."""
    n = min(len(x), int(ms * 0.001 * sr))
    if n > 1:
        x[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return x


# ------------------------------------------------------------- oscillators ---

def _phase(freq, n, sr, detune_cents=0.0, drift=0.0, rng=None):
    f = freq * (2.0 ** (detune_cents / 1200.0))
    t = np.arange(n, dtype=np.float64) / sr
    if drift and rng is not None:
        wobble = np.sin(TWO_PI * (0.6 + rng.random() * 0.8) * t + rng.random() * TWO_PI)
        f = f * (1.0 + drift * 0.004 * wobble)
        return TWO_PI * np.cumsum(f) / sr
    return TWO_PI * f * t


@lru_cache(maxsize=64)
def _wave_table(kind: str, harmonics: int, size: int = 4096):
    """One cycle of a band-limited waveform, built once and reused."""
    k = np.arange(1, harmonics + 1, dtype=np.float64)
    if kind == "square":
        k = k[k % 2 == 1]
    ph = TWO_PI * np.arange(size, dtype=np.float64) / size
    table = (np.sin(ph[:, None] * k[None, :]) / k).sum(axis=1)
    scale = (2.0 / math.pi) if kind == "saw" else (4.0 / math.pi) * 0.5
    return (table * scale).astype(np.float32)


def wavetable_osc(freq, n, sr, kind="saw", harmonics=None, phase0=0.0):
    """Read a band-limited table with linear interpolation.

    Far cheaper than summing sines per note, which matters because pads hold
    long notes and an hours-long render asks for a lot of them.
    """
    nyq = sr * 0.45
    k_max = harmonics or max(1, int(nyq / max(freq, 1.0)))
    k_max = int(min(max(k_max, 1), 64))
    table = _wave_table(kind, k_max)
    size = len(table)
    inc = float(freq) * size / sr
    idx = (phase0 + inc * np.arange(n, dtype=np.float64)) % size
    i0 = idx.astype(np.int64)
    frac = (idx - i0).astype(np.float32)
    i1 = (i0 + 1) % size
    return (table[i0] * (1.0 - frac) + table[i1] * frac).astype(np.float32)


def bandlimited_saw(freq, n, sr, harmonics=None):
    return wavetable_osc(freq, n, sr, "saw", harmonics)


def bandlimited_square(freq, n, sr, harmonics=None):
    return wavetable_osc(freq, n, sr, "square", harmonics)


# -------------------------------------------------------------- instruments ---

def rhodes(midi, dur, sr, velocity=1.0, rng=None):
    """FM electric piano - the Rhodes-ish bell that defines lofi chords."""
    rng = rng or np.random.default_rng(0)
    n = max(1, int(dur * sr))
    f = midi_to_hz(midi)
    t = np.arange(n, dtype=np.float64) / sr
    # modulator index falls fast -> the classic attack 'bark', then pure-ish tone
    idx_env = np.exp(-t / 0.09) * (2.6 + 2.4 * velocity)
    mod = np.sin(TWO_PI * f * t) * idx_env
    body = np.sin(TWO_PI * f * t + mod)
    # tine: a high, quickly-decaying partial
    tine = np.sin(TWO_PI * f * 9.0 * t) * np.exp(-t / 0.045) * 0.16 * velocity
    # a second, slightly detuned voice gives the chorus-y Rhodes shimmer
    det = np.sin(TWO_PI * f * 1.0015 * t + mod * 0.9) * 0.45
    env = perc_env(n, sr, attack=0.004, decay=min(dur, 0.35 + 2.2 * max(0.0, 1.0 - midi / 100.0)), curve=2.2)
    out = (body + det + tine) * env * (0.28 * (0.45 + 0.55 * velocity))
    return _fade_tail(out.astype(np.float32), sr)


def felt_piano(midi, dur, sr, velocity=1.0, rng=None):
    """Soft upright/felt piano: inharmonic partials plus hammer noise."""
    rng = rng or np.random.default_rng(0)
    n = max(1, int(dur * sr))
    f = midi_to_hz(midi)
    t = np.arange(n, dtype=np.float64) / sr
    B = 0.0006                      # string inharmonicity
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, 13):
        fk = f * k * math.sqrt(1.0 + B * k * k)
        if fk > sr * 0.45:
            break
        amp = (1.0 / (k ** 1.45)) * (0.6 + 0.4 * velocity)
        decay = 2.4 / (1.0 + 0.55 * k)
        out += amp * np.sin(TWO_PI * fk * t + rng.random() * TWO_PI) * np.exp(-t / decay)
    hammer = _bp(rng.standard_normal(n), sr, 400.0, 6000.0) * np.exp(-t / 0.012) * 0.09 * velocity
    env = perc_env(n, sr, attack=0.003, decay=min(dur, 3.0), curve=1.6)
    return _fade_tail(((out * 0.32 + hammer) * env).astype(np.float32), sr)


def _karplus(freq, n, sr, t60=1.8, brightness=0.55, seed=0):
    """Karplus-Strong pluck, chunked so the inner loop stays vectorised numpy."""
    rng = np.random.default_rng(seed)
    N = max(4, int(round(sr / max(freq, 20.0))))
    buf = rng.standard_normal(N).astype(np.float32)
    # a lowpass on the excitation sets how bright the pluck starts out
    k = max(1, int((1.0 - brightness) * 12))
    if k > 1:
        kernel = np.ones(k, dtype=np.float32) / k
        buf = np.convolve(buf, kernel, mode="same").astype(np.float32)
    buf *= 1.0 / (np.abs(buf).max() + 1e-9)
    out = np.empty(n, dtype=np.float32)
    # loop gain per round trip, set from the wanted -60 dB time rather than a
    # magic constant: one period lasts 1/freq seconds, so aim for 10^(-3/(f*t60))
    decay = float(np.clip(10.0 ** (-3.0 / max(1e-3, freq * t60)), 0.90, 0.99995))
    pos = 0
    prev = 0.0
    while pos < n:
        count = min(N, n - pos)
        chunk = buf[:count].copy()
        out[pos:pos + count] = chunk
        shifted = np.concatenate([[prev], chunk[:-1]]) if count > 1 else np.array([prev], dtype=np.float32)
        filt = ((chunk + shifted) * 0.5 * decay).astype(np.float32)
        prev = float(chunk[-1])
        buf = np.concatenate([filt, buf[count:]]) if count < N else filt
        pos += count
    return out


def nylon_guitar(midi, dur, sr, velocity=1.0, rng=None):
    """Plucked nylon-string guitar (Karplus-Strong) - the jazzy lofi comp."""
    n = max(1, int(dur * sr))
    f = midi_to_hz(midi)
    seed = int(midi * 977 + int(velocity * 100)) & 0xFFFF
    body = _karplus(f, n, sr, t60=2.4, brightness=0.16 + 0.12 * velocity, seed=seed)
    # nylon strings are dark: roll the pluck off well before the 3rd harmonic
    # dominates, then add back a touch of the finger-noise transient on top.
    body = _lp(body, sr, max(900.0, f * 4.5), order=2)
    t = np.arange(n, dtype=np.float64) / sr
    click = _bp(np.random.default_rng(seed).standard_normal(n), sr, 800.0, 4500.0)
    click = click * np.exp(-t / 0.007) * 0.09 * velocity
    env = perc_env(n, sr, attack=0.002, decay=min(dur, 2.2), curve=1.2)
    out = (body * 0.85 + click) * env * (0.5 + 0.5 * velocity)
    return _fade_tail(out.astype(np.float32), sr)


def vibraphone(midi, dur, sr, velocity=1.0, rng=None):
    """Struck metal bar with the motor tremolo."""
    n = max(1, int(dur * sr))
    f = midi_to_hz(midi)
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n, dtype=np.float64)
    for ratio, amp, dec in ((1.0, 1.0, 2.6), (4.0, 0.30, 1.1), (9.2, 0.12, 0.6)):
        if f * ratio < sr * 0.45:
            out += amp * np.sin(TWO_PI * f * ratio * t) * np.exp(-t / dec)
    trem = 1.0 - 0.28 * (0.5 + 0.5 * np.sin(TWO_PI * 4.6 * t))
    env = perc_env(n, sr, attack=0.002, decay=min(dur, 2.4), curve=1.8)
    return _fade_tail((out * trem * env * 0.26 * (0.5 + 0.5 * velocity)).astype(np.float32), sr)


def kalimba(midi, dur, sr, velocity=1.0, rng=None):
    n = max(1, int(dur * sr))
    f = midi_to_hz(midi)
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n, dtype=np.float64)
    for ratio, amp, dec in ((1.0, 1.0, 1.1), (2.76, 0.28, 0.35), (5.4, 0.10, 0.16)):
        if f * ratio < sr * 0.45:
            out += amp * np.sin(TWO_PI * f * ratio * t) * np.exp(-t / dec)
    thumb = _bp((rng or np.random.default_rng(0)).standard_normal(n), sr, 900.0, 7000.0)
    thumb = thumb * np.exp(-t / 0.004) * 0.12
    env = perc_env(n, sr, attack=0.001, decay=min(dur, 1.2), curve=2.2)
    return _fade_tail(((out * 0.3 + thumb) * env * (0.5 + 0.5 * velocity)).astype(np.float32), sr)


def muted_trumpet(midi, dur, sr, velocity=1.0, rng=None):
    """Harmon-mute-ish lead: buzzy but narrow, with breath and slow vibrato."""
    rng = rng or np.random.default_rng(0)
    n = max(1, int(dur * sr))
    f = midi_to_hz(midi)
    t = np.arange(n, dtype=np.float64) / sr
    vib = 1.0 + 0.004 * np.sin(TWO_PI * 5.2 * t) * np.clip(t / 0.35, 0, 1)
    ph = TWO_PI * np.cumsum(f * vib) / sr
    out = np.zeros(n, dtype=np.float64)
    for k, amp in ((1, 1.0), (2, 0.55), (3, 0.42), (4, 0.24), (5, 0.16), (6, 0.09)):
        if f * k < sr * 0.45:
            out += amp * np.sin(ph * k)
    breath = _bp(rng.standard_normal(n), sr, 1500.0, 6000.0) * 0.05 * np.exp(-t / 0.08)
    env = adsr(n, sr, attack=0.045, decay=0.18, sustain=0.72, release=min(0.4, dur * 0.4), curve=1.5)
    return _fade_tail(((out * 0.16 + breath) * env * (0.45 + 0.55 * velocity)).astype(np.float32), sr)


def warm_pad(midi, dur, sr, velocity=1.0, rng=None):
    """Detuned analogue-ish pad - the glue layer under everything."""
    rng = rng or np.random.default_rng(0)
    n = max(1, int(dur * sr))
    f = midi_to_hz(midi)
    out = np.zeros(n, dtype=np.float64)
    for cents in (-9.0, -3.5, 0.0, 4.5, 8.0):
        out += bandlimited_saw(f * (2.0 ** (cents / 1200.0)), n, sr, harmonics=14)
    out += bandlimited_saw(f * 0.5, n, sr, harmonics=10) * 0.4
    t = np.arange(n, dtype=np.float64) / sr
    shimmer = 1.0 + 0.05 * np.sin(TWO_PI * 0.23 * t + rng.random() * TWO_PI)
    env = adsr(n, sr, attack=min(0.9, dur * 0.3), decay=0.5, sustain=0.85,
               release=min(1.4, dur * 0.45), curve=1.3)
    return _fade_tail((out * shimmer * env * 0.055 * (0.5 + 0.5 * velocity)).astype(np.float32), sr)


def sub_bass(midi, dur, sr, velocity=1.0, rng=None):
    """Round sine sub with a short pitch drop - sits under the kick, never fights it."""
    n = max(1, int(dur * sr))
    f = midi_to_hz(midi)
    t = np.arange(n, dtype=np.float64) / sr
    glide = f * (1.0 + 0.06 * np.exp(-t / 0.035))
    ph = TWO_PI * np.cumsum(glide) / sr
    out = np.sin(ph) + 0.16 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph)
    env = adsr(n, sr, attack=0.008, decay=0.22, sustain=0.72,
               release=min(0.22, dur * 0.35), curve=1.8)
    return _fade_tail((out * env * 0.34 * (0.55 + 0.45 * velocity)).astype(np.float32), sr)


def upright_bass(midi, dur, sr, velocity=1.0, rng=None):
    """Acoustic double bass: damped string plus finger thump."""
    rng = rng or np.random.default_rng(0)
    n = max(1, int(dur * sr))
    f = midi_to_hz(midi)
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n, dtype=np.float64)
    for k, amp, dec in ((1, 1.0, 0.85), (2, 0.34, 0.45), (3, 0.16, 0.28), (4, 0.07, 0.18)):
        if f * k < sr * 0.45:
            out += amp * np.sin(TWO_PI * f * k * t) * np.exp(-t / dec)
    thump = _lp(rng.standard_normal(n), sr, 700.0, order=2) * np.exp(-t / 0.010) * 0.22 * velocity
    env = perc_env(n, sr, attack=0.004, decay=min(dur, 1.1), curve=1.7)
    return _fade_tail(((out * 0.30 + thump) * env * (0.5 + 0.5 * velocity)).astype(np.float32), sr)


def bansuri(midi, dur, sr, velocity=1.0, rng=None):
    """Bamboo flute: breathy sine-dominant tone with a slow vibrato."""
    rng = rng or np.random.default_rng(0)
    n = max(1, int(dur * sr))
    f = midi_to_hz(midi)
    t = np.arange(n, dtype=np.float64) / sr
    vib = 1.0 + 0.007 * np.sin(TWO_PI * 4.8 * t) * np.clip(t / 0.4, 0, 1)
    ph = TWO_PI * np.cumsum(f * vib) / sr
    out = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.07 * np.sin(3 * ph)
    # breath noise is band-limited around the embouchure, not full-band white
    air = _bp(rng.standard_normal(n), sr, 1200.0, 5000.0)
    air = air * 0.07 * np.clip(np.exp(-t / 0.5) + 0.22, 0, 1)
    env = adsr(n, sr, attack=0.07, decay=0.2, sustain=0.78, release=min(0.5, dur * 0.4), curve=1.4)
    return _fade_tail(((out * 0.22 + air) * env * (0.45 + 0.55 * velocity)).astype(np.float32), sr)


INSTRUMENTS = {
    "rhodes": rhodes,
    "felt_piano": felt_piano,
    "nylon_guitar": nylon_guitar,
    "vibraphone": vibraphone,
    "kalimba": kalimba,
    "muted_trumpet": muted_trumpet,
    "pad": warm_pad,
    "sub_bass": sub_bass,
    "upright_bass": upright_bass,
    "bansuri": bansuri,
}


# ------------------------------------------------------------- note caching ---

class NoteCache:
    """Memoise rendered notes.  A 3-hour song reuses a few hundred distinct
    (instrument, pitch, length, velocity) combinations thousands of times, so
    this turns synthesis from the bottleneck into a rounding error."""

    def __init__(self, sr, max_entries=4096, dur_step=0.05, vel_step=0.125):
        self.sr = sr
        self.max_entries = max_entries
        self.dur_step = dur_step
        self.vel_step = vel_step
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def render(self, name, midi, dur, velocity=1.0, rng=None):
        fn = INSTRUMENTS.get(name)
        if fn is None:
            raise KeyError(f"unknown instrument {name!r}")
        midi = int(round(midi))
        dur_q = max(self.dur_step, round(float(dur) / self.dur_step) * self.dur_step)
        vel_q = min(1.0, max(0.125, round(float(velocity) / self.vel_step) * self.vel_step))
        key = (name, midi, round(dur_q, 3), round(vel_q, 3))
        hit = self._cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        local = np.random.default_rng((abs(hash(key)) % (2 ** 32)))
        out = fn(midi, dur_q, self.sr, vel_q, local)
        if len(self._cache) >= self.max_entries:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = out
        return out

    def stats(self):
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "entries": len(self._cache)}
