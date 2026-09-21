"""Synthesised boom-bap / lofi drum kit.

Same rule as the instruments: nothing sampled, everything generated, so the
output carries no one else's recording rights.  Each piece returns mono
float32.  A `DrumKit` renders and caches the pieces once per song.
"""

from __future__ import annotations

import math

import numpy as np

from .synth import _bp

TWO_PI = 2.0 * math.pi


def kick(sr, tune=52.0, decay=0.42, click=0.5, dirt=0.3, seed=0):
    """Pitch-swept sine with a click transient - the boom of boom-bap."""
    rng = np.random.default_rng(seed)
    n = int(0.9 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    # frequency sweeps from ~4x tune down to tune in the first few ms
    freq = tune * (1.0 + 3.2 * np.exp(-t / 0.020))
    body = np.sin(TWO_PI * np.cumsum(freq) / sr) * np.exp(-t / decay)
    tick = _bp(rng.standard_normal(n), sr, 1200.0, 5200.0) * np.exp(-t / 0.004) * click * 0.55
    sub = np.sin(TWO_PI * tune * 0.5 * t) * np.exp(-t / (decay * 0.7)) * 0.18
    out = body * 0.95 + tick + sub
    # Saturate first, then re-apply the decay.  Doing it the other way round
    # lets the drive flatten the envelope and the kick turns into a bass note.
    out = np.tanh(out * (1.0 + dirt * 1.6)) / (1.0 + dirt * 0.5)
    out *= np.exp(-t / (decay * 1.25))
    return _tail(out.astype(np.float32) * 1.05, sr)


def snare(sr, tune=190.0, decay=0.20, snap=0.85, seed=1):
    """Tonal shell plus band-limited noise - soft, dusty, never harsh."""
    rng = np.random.default_rng(seed)
    n = int(0.7 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    shell = (np.sin(TWO_PI * tune * t) + 0.6 * np.sin(TWO_PI * tune * 1.47 * t))
    shell *= np.exp(-t / (decay * 0.55)) * 0.35
    wires = _bp(rng.standard_normal(n), sr, 1400.0, 7000.0) * np.exp(-t / decay) * snap * 0.5
    body = _bp(rng.standard_normal(n), sr, 220.0, 900.0) * np.exp(-t / (decay * 0.7)) * 0.18
    out = shell + wires + body
    return _tail(np.tanh(out * 1.25).astype(np.float32) * 0.62, sr)


def rimshot(sr, tune=430.0, decay=0.075, seed=2):
    """Cross-stick / rim click - the lazy backbeat that keeps lofi soft."""
    rng = np.random.default_rng(seed)
    n = int(0.3 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    tone = (np.sin(TWO_PI * tune * t) + 0.5 * np.sin(TWO_PI * tune * 2.31 * t))
    tone *= np.exp(-t / decay)
    noise = _bp(rng.standard_normal(n), sr, 1800.0, 6500.0) * np.exp(-t / 0.018) * 0.42
    return _tail(((tone * 0.4 + noise) * 0.68).astype(np.float32), sr)


def hihat(sr, decay=0.055, open_hat=False, tone=8200.0, seed=3):
    """Metallic hat from inharmonic square partials + filtered noise."""
    rng = np.random.default_rng(seed)
    dur = 0.55 if open_hat else 0.22
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float64) / sr
    ratios = (2.0, 3.0, 4.16, 5.43, 6.79, 8.21)
    metal = np.zeros(n, dtype=np.float64)
    base = tone / 6.0
    for r in ratios:
        f = base * r
        if f < sr * 0.47:
            metal += np.sign(np.sin(TWO_PI * f * t))
    metal = _bp(metal, sr, 6000.0, min(16000.0, sr * 0.46))
    noise = _bp(rng.standard_normal(n), sr, 5000.0, min(15000.0, sr * 0.45))
    d = decay * (5.5 if open_hat else 1.0)
    env = np.exp(-t / d)
    out = (metal * 0.16 + noise * 0.5) * env
    return _tail(out.astype(np.float32) * (0.30 if open_hat else 0.34), sr)


def shaker(sr, decay=0.045, seed=4):
    rng = np.random.default_rng(seed)
    n = int(0.22 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    noise = _bp(rng.standard_normal(n), sr, 4000.0, 11000.0)
    env = np.exp(-t / decay) * np.clip(t / 0.004, 0, 1)
    return _tail((noise * env * 0.26).astype(np.float32), sr)


def ride(sr, decay=1.1, seed=5):
    rng = np.random.default_rng(seed)
    n = int(1.6 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    metal = np.zeros(n, dtype=np.float64)
    for r in (1.0, 1.41, 1.93, 2.71, 3.55, 4.61, 6.02):
        f = 520.0 * r
        if f < sr * 0.47:
            metal += np.sin(TWO_PI * f * t + rng.random() * TWO_PI) / r
    ping = metal * np.exp(-t / 0.35) * 0.5
    wash = _bp(rng.standard_normal(n), sr, 3000.0, 12000.0) * np.exp(-t / decay) * 0.30
    return _tail(((ping + wash) * 0.22).astype(np.float32), sr)


def tom(sr, tune=120.0, decay=0.35, seed=6):
    n = int(0.8 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    freq = tune * (1.0 + 0.9 * np.exp(-t / 0.05))
    body = np.sin(TWO_PI * np.cumsum(freq) / sr) * np.exp(-t / decay)
    skin = _bp(np.random.default_rng(seed).standard_normal(n), sr, 300.0, 2500.0)
    skin = skin * np.exp(-t / 0.02) * 0.16
    return _tail(((body * 0.7 + skin) * 0.55).astype(np.float32), sr)


def tabla(sr, tune=290.0, decay=0.32, seed=7):
    """Tabla-ish 'na' stroke for the devotional preset."""
    rng = np.random.default_rng(seed)
    n = int(0.6 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n, dtype=np.float64)
    for r, a, d in ((1.0, 1.0, decay), (2.0, 0.42, decay * 0.5), (3.02, 0.22, decay * 0.3)):
        out += a * np.sin(TWO_PI * tune * r * t) * np.exp(-t / d)
    slap = _bp(rng.standard_normal(n), sr, 1500.0, 6000.0) * np.exp(-t / 0.008) * 0.3
    return _tail(((out * 0.4 + slap) * 0.6).astype(np.float32), sr)


def _tail(x, sr, ms=6.0):
    n = min(len(x), int(ms * 0.001 * sr))
    if n > 1:
        x[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return x


PIECES = {
    "kick": kick, "snare": snare, "rim": rimshot, "hat": hihat,
    "shaker": shaker, "ride": ride, "tom": tom, "tabla": tabla,
}


class DrumKit:
    """Renders one kit up front, then hands out cached hits by name+velocity."""

    STYLES = {
        "dusty": dict(kick=dict(tune=50.0, decay=0.45, dirt=0.45),
                      snare=dict(tune=185.0, decay=0.18, snap=0.7),
                      hat=dict(decay=0.045, tone=7600.0)),
        "soft": dict(kick=dict(tune=54.0, decay=0.36, click=0.3, dirt=0.15),
                     snare=dict(tune=200.0, decay=0.15, snap=0.55),
                     hat=dict(decay=0.038, tone=8600.0)),
        "punchy": dict(kick=dict(tune=56.0, decay=0.32, click=0.75, dirt=0.35),
                       snare=dict(tune=205.0, decay=0.22, snap=1.0),
                       hat=dict(decay=0.055, tone=9200.0)),
        "brush": dict(kick=dict(tune=48.0, decay=0.50, click=0.2, dirt=0.1),
                      snare=dict(tune=175.0, decay=0.26, snap=0.45),
                      hat=dict(decay=0.07, tone=7000.0)),
    }

    def __init__(self, sr, style="dusty", seed=0, tuning=0.0):
        self.sr = sr
        self.style = style if style in self.STYLES else "dusty"
        self.seed = int(seed)
        self.shift = 2.0 ** (float(tuning) / 12.0)
        self._cache = {}

    def _params(self, name):
        p = dict(self.STYLES[self.style].get(name, {}))
        if "tune" in p:
            p["tune"] = p["tune"] * self.shift
        return p

    def hit(self, name, velocity=1.0):
        name = "hat_open" if name == "openhat" else name
        key = (name, round(float(velocity) / 0.05) * 0.05)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        base = name.replace("_open", "")
        fn = PIECES.get(base)
        if fn is None:
            return np.zeros(1, dtype=np.float32)
        kwargs = self._params(base)
        if name.endswith("_open"):
            kwargs["open_hat"] = True
        sig = fn(self.sr, seed=self.seed + hash(base) % 1000, **kwargs)
        out = (sig * float(np.clip(velocity, 0.0, 1.5))).astype(np.float32)
        self._cache[key] = out
        return out


# ---------------------------------------------------------------- patterns ---
#: 16th-note grids over one bar.  1.0 = accent, lower = ghost note.
PATTERNS = {
    "boom_bap": {
        "kick":  [1.0, 0, 0, 0, 0, 0, .85, 0, 0, 0, 1.0, 0, 0, 0, 0, 0],
        "snare": [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, .35],
        "hat":   [.8, 0, .45, 0, .8, 0, .45, 0, .8, 0, .45, 0, .8, 0, .55, .3],
    },
    "lazy": {
        "kick":  [1.0, 0, 0, 0, 0, 0, 0, 0, .8, 0, 0, 0, 0, 0, .5, 0],
        "rim":   [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
        "hat":   [.7, 0, 0, .4, .7, 0, 0, .4, .7, 0, 0, .4, .7, 0, .5, 0],
    },
    "halftime": {
        "kick":  [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .7, 0, 0, 0, 0, 0],
        "snare": [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, .3],
        "hat":   [.7, 0, .4, 0, .7, 0, .4, 0, .7, 0, .4, 0, .7, 0, .4, .35],
    },
    "swing": {
        "kick":  [1.0, 0, 0, 0, 0, 0, .7, 0, .5, 0, 0, 0, 0, 0, .6, 0],
        "rim":   [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
        "ride":  [.7, 0, .35, 0, .7, 0, .35, 0, .7, 0, .35, 0, .7, 0, .35, .3],
        "shaker": [0, 0, .3, 0, 0, 0, .3, 0, 0, 0, .3, 0, 0, 0, .3, 0],
    },
    "shuffle": {
        "kick":  [1.0, 0, 0, .4, 0, 0, .8, 0, 0, 0, 1.0, 0, 0, 0, 0, .35],
        "snare": [0, 0, .2, 0, 1.0, 0, 0, .25, 0, 0, .2, 0, 1.0, 0, .3, 0],
        "hat":   [.75, 0, .4, .3, .75, 0, .4, .3, .75, 0, .4, .3, .75, 0, .5, .4],
    },
    "brushed": {
        "kick":  [1.0, 0, 0, 0, 0, 0, 0, 0, .7, 0, 0, 0, 0, 0, 0, 0],
        "snare": [.25, .2, .3, .2, .9, .2, .3, .2, .25, .2, .3, .2, .9, .2, .35, .25],
        "ride":  [.6, 0, .3, 0, .6, 0, .3, 0, .6, 0, .3, 0, .6, 0, .3, 0],
    },
    "tabla": {
        "tabla": [1.0, 0, .4, 0, .7, 0, .45, 0, 1.0, 0, .4, 0, .7, .35, .5, 0],
        "shaker": [0, .25, 0, .25, 0, .25, 0, .25, 0, .25, 0, .25, 0, .25, 0, .25],
    },
    "none": {},
}
