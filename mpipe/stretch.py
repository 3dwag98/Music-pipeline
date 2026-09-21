"""Tempo/key analysis and WSOLA time-stretching - pure numpy + scipy.

librosa is nice but a heavy install on Windows, so everything the pipeline
actually needs for beat-matching is implemented here.  `analyze_audio` falls
back to these when librosa is absent and cross-checks against it when present.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .audio import load_audio

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ----------------------------------------------------------------- analysis ---

def onset_envelope(mono, sr, hop=512, win=1024):
    """Spectral-flux onset strength, one value per hop."""
    n = (len(mono) - win) // hop
    if n <= 2:
        return np.zeros(1), hop / sr
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    frames = mono[idx] * np.hanning(win)[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1))
    mag = np.log1p(mag * 8.0)
    flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
    flux -= flux.mean()
    if flux.std() > 0:
        flux /= flux.std()
    return flux, hop / sr


def detect_bpm(mono, sr, lo=55.0, hi=180.0, prior_bpm=85.0, prior_octaves=0.75):
    """Tempo from the onset-envelope autocorrelation, with a tempo prior.

    A bare autocorrelation peak happily locks onto 2/3 or 3/2 of the real pulse
    (a dotted-half looks just as periodic as the beat).  A log-normal prior
    around typical lofi tempos, plus an explicit check of the common metrical
    relatives, is what keeps a 90 BPM track from being read as 60.
    """
    flux, step = onset_envelope(mono, sr)
    if len(flux) < 8:
        return 80.0, 0.0
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lags = np.arange(len(ac)) * step

    def comb(lag_idx):
        """Sum the autocorrelation at 1x..4x a lag - a real pulse lines up at all."""
        total, used = 0.0, 0
        for mult in (1, 2, 3, 4):
            j = lag_idx * mult
            if j < len(ac):
                total += ac[j] / mult
                used += 1
        return total / max(1, used)

    def prior(bpm):
        return math.exp(-0.5 * (math.log2(bpm / prior_bpm) / prior_octaves) ** 2)

    cand = np.where((lags >= 60.0 / hi) & (lags <= 60.0 / lo))[0]
    if len(cand) == 0:
        return 80.0, 0.0
    scored = [(comb(i) * prior(60.0 / (lags[i] + 1e-9)), i) for i in cand]
    raw_score, best = max(scored)
    bpm = 60.0 / (lags[best] + 1e-9)

    # re-test the metrical relatives; a 2/3 or 3/2 error is the common failure
    best_bpm, best_score = bpm, raw_score
    for mult in (0.5, 2.0 / 3.0, 0.75, 1.5, 2.0, 3.0):
        alt = bpm * mult
        if not (lo <= alt <= hi):
            continue
        lag = 60.0 / alt
        idx = int(round(lag / step))
        if 0 < idx < len(ac):
            score = comb(idx) * prior(alt)
            if score > best_score:
                best_bpm, best_score = alt, score
    bpm = best_bpm

    confidence = float(np.clip(best_score, 0.0, 1.0))
    while bpm > hi:
        bpm /= 2.0
    while bpm < lo:
        bpm *= 2.0
    return round(float(bpm), 2), confidence


def beat_phase(mono, sr, bpm, hop=512):
    """Seconds from the file start to the first downbeat of a 4-beat bar."""
    flux, step = onset_envelope(mono, sr, hop=hop)
    if len(flux) < 8 or not bpm:
        return 0.0
    bar = 4 * 60.0 / bpm
    bar_frames = bar / step
    best_off, best_score = 0.0, -1e9
    for frac in np.linspace(0.0, 1.0, 64, endpoint=False):
        pos = frac * bar_frames
        score = 0.0
        while pos < len(flux):
            score += flux[int(pos)]
            pos += bar_frames
        if score > best_score:
            best_score, best_off = score, frac * bar
    return float(best_off)


def chroma(mono, sr, n_fft=8192, hop=4096):
    """Pitch-class energy profile via an FFT folded onto 12 semitones."""
    if len(mono) < n_fft:
        mono = np.pad(mono, (0, n_fft - len(mono)))
    n = max(1, (len(mono) - n_fft) // hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    valid = (freqs > 55.0) & (freqs < 2200.0)
    pcs = np.zeros(freqs.shape, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(np.where(freqs > 0, freqs, 1e-9) / 440.0)
    pcs[valid] = np.round(midi[valid]).astype(np.int64) % 12
    out = np.zeros(12)
    win = np.hanning(n_fft)
    for i in range(n):
        seg = mono[i * hop: i * hop + n_fft]
        if len(seg) < n_fft:
            break
        mag = np.abs(np.fft.rfft(seg * win)) ** 2
        for pc in range(12):
            out[pc] += mag[valid & (pcs == pc)].sum()
    total = out.sum()
    return out / total if total > 0 else out


def detect_key(mono, sr):
    """Krumhansl-Schmuckler key estimate.  Returns ('A Minor', confidence)."""
    c = chroma(mono, sr)
    if c.sum() <= 0:
        return "C Major", 0.0
    best, best_score, runner = "C Major", -2.0, -2.0
    for shift in range(12):
        for name, profile in (("Major", MAJOR_PROFILE), ("Minor", MINOR_PROFILE)):
            score = float(np.corrcoef(c, np.roll(profile, shift))[0, 1])
            if not np.isfinite(score):
                continue
            if score > best_score:
                runner = best_score
                best, best_score = f"{KEY_NAMES[shift]} {name}", score
            elif score > runner:
                runner = score
    return best, round(float(max(0.0, best_score - max(runner, 0.0))), 3)


def analyze_file(path, max_seconds=180.0):
    mono, sr = load_audio(path, target_sr=22050, mono=True, max_seconds=max_seconds)
    if len(mono) == 0:
        return {"bpm": 0.0, "key": "C Major", "duration": 0.0}
    bpm, conf = detect_bpm(mono, sr)
    key, key_conf = detect_key(mono, sr)
    return {"bpm": bpm, "bpm_confidence": round(conf, 3), "key": key,
            "key_confidence": key_conf, "duration": round(len(mono) / sr, 2),
            "downbeat": round(beat_phase(mono, sr, bpm), 3)}


# ------------------------------------------------------------- time / pitch ---

def time_stretch(x, rate, frame=2048, search=384):
    """WSOLA time-stretch.  rate > 1 makes it shorter/faster, < 1 longer/slower.

    WSOLA (not a phase vocoder) because drums are the point here: it keeps
    transients crisp instead of smearing them into a flam.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if abs(rate - 1.0) < 1e-4 or len(x) < frame * 3:
        return x
    hop_s = frame // 2
    hop_a = max(1, int(round(hop_s * rate)))
    win = np.hanning(frame).astype(np.float32)[:, None]
    ref = x.mean(axis=1)
    out_len = int(len(x) / rate) + frame * 2
    out = np.zeros((out_len, x.shape[1]), dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    ideal, write = 0.0, 0
    tmpl = None
    tmpl_n = frame - hop_s
    limit = len(x) - frame - 1
    while write + frame < out_len:
        # The ideal read position advances by exactly hop_a every frame.  The
        # search only moves where we *copy* from; letting it move `ideal` too
        # makes the read head drift and silently truncates the output.
        read = int(round(ideal))
        if read > limit:
            break
        start = read
        if tmpl is not None:
            lo = max(0, read - search)
            hi = min(limit, read + search)
            if hi > lo:
                seg = ref[lo: hi + tmpl_n]
                if len(seg) >= tmpl_n:
                    corr = np.correlate(seg, tmpl, mode="valid")
                    start = lo + int(np.argmax(corr[: hi - lo + 1]))
        start = int(np.clip(start, 0, limit))
        chunk = x[start:start + frame] * win
        out[write:write + frame] += chunk
        norm[write:write + frame] += win[:, 0]
        nxt = ref[start + hop_s: start + hop_s + tmpl_n]
        tmpl = nxt if len(nxt) == tmpl_n else None
        ideal += hop_a
        write += hop_s
    end = write + frame
    out = out[:end] / np.maximum(norm[:end], 1e-4)[:, None]
    return np.ascontiguousarray(out, dtype=np.float32)


def pitch_shift(x, semitones, sr, frame=2048):
    """Shift pitch without changing length: stretch, then resample."""
    if abs(semitones) < 1e-3:
        return np.asarray(x, dtype=np.float32)
    from scipy.signal import resample_poly
    ratio = 2.0 ** (float(semitones) / 12.0)
    stretched = time_stretch(x, 1.0 / ratio, frame=frame)
    frac = Fraction(ratio).limit_denominator(400)
    up, down = frac.denominator, frac.numerator   # resample slower/faster
    out = resample_poly(stretched, up, down, axis=0)
    # stretch + resample rounding can drift a few ms; pin the length exactly so
    # a pitch-shifted track still lines up with the grid it was cut to.
    want = len(np.asarray(x))
    if len(out) > want:
        out = out[:want]
    elif len(out) < want:
        pad = np.zeros((want - len(out),) + out.shape[1:], dtype=out.dtype)
        out = np.concatenate([out, pad], axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def fit_to_tempo(x, sr, src_bpm, dst_bpm, max_percent=18.0):
    """Stretch a track to a target tempo, refusing edits that would sound bad."""
    if not src_bpm or not dst_bpm:
        return np.asarray(x, dtype=np.float32), 1.0
    rate = float(src_bpm) / float(dst_bpm)
    for mult in (1.0, 2.0, 0.5):                 # allow half/double-time matches
        test = rate * mult
        if abs(test - 1.0) * 100 <= max_percent:
            rate = test
            break
    else:
        return np.asarray(x, dtype=np.float32), 1.0
    if abs(rate - 1.0) < 0.002:
        return np.asarray(x, dtype=np.float32), 1.0
    return time_stretch(x, rate), rate


#: Camelot-style neighbours: which keys mix without a clash.
def key_distance(key_a, key_b):
    """How far apart two keys are for mixing purposes (0 = interchangeable).

    Relative major/minor (C Major / A Minor) share every note, so they count as
    the same key - treating them as 3 semitones apart makes the track ordering
    shuffle needlessly and sends the pitch shifter chasing a move it should not
    make.
    """
    def parse(k):
        parts = str(k).split()
        name = parts[0].replace("Db", "C#").replace("Eb", "D#").replace("Gb", "F#") \
                       .replace("Ab", "G#").replace("Bb", "A#")
        pc = KEY_NAMES.index(name) if name in KEY_NAMES else 0
        mode = "Minor" if len(parts) > 1 and parts[1].lower().startswith("min") else "Major"
        return pc, mode
    pa, ma = parse(key_a)
    pb, mb = parse(key_b)
    if ma != mb:
        # fold the minor onto its relative major before comparing
        rel_a = (pa + 3) % 12 if ma == "Minor" else pa
        rel_b = (pb + 3) % 12 if mb == "Minor" else pb
        d = min((rel_a - rel_b) % 12, (rel_b - rel_a) % 12)
        return d + (0.0 if d == 0 else 0.75)
    return float(min((pa - pb) % 12, (pb - pa) % 12))


def semitones_to_key(from_key, to_key, max_shift=4):
    """Smallest shift that moves `from_key` onto `to_key`, or 0 if too far.

    Relative keys already share their notes, so they need no shift at all.
    """
    def parse(k):
        parts = str(k).split()
        name = parts[0].replace("Db", "C#").replace("Eb", "D#") \
            .replace("Gb", "F#").replace("Ab", "G#").replace("Bb", "A#")
        pc = KEY_NAMES.index(name) if name in KEY_NAMES else 0
        minor = len(parts) > 1 and parts[1].lower().startswith("min")
        return (pc + 3) % 12 if minor else pc        # fold to the relative major
    delta = (parse(to_key) - parse(from_key)) % 12
    if delta > 6:
        delta -= 12
    return delta if abs(delta) <= max_shift else 0
