"""Turn a list of tracks into ONE continuous piece - or a classic crossfaded mix.

`build_mix`  : tracks back to back with crossfades (a compilation).
`build_song` : tracks beat-matched to one tempo, pitch-matched to one key, joined
               with bar-aligned DJ transitions and glued by a continuous vinyl
               bed, an optional never-stopping drum spine and one shared master
               chain - so hours of material read as a single long lofi song.

Both stream to disk, so length is limited by free space, not by RAM.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

from . import dsp
from .audio import StreamWriter, StreamingLoudness, integrated_lufs, load_audio
from .drums import PATTERNS, DrumKit
from .stretch import (analyze_file, fit_to_tempo, key_distance, pitch_shift,
                      semitones_to_key, time_stretch)
from .theory import parse_key, key_name
from .util import check_disk, fmt_time, log, warn

DEFAULT_SR = 44100


# ---------------------------------------------------------------- analysis ---

def analyze_tracks(paths, cache_path=None, quiet=False):
    """Analyse every input once and memoise it (analysis is the slow part)."""
    cache = {}
    if cache_path and Path(cache_path).exists():
        try:
            cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    out = []
    for path in paths:
        path = Path(path)
        try:
            key = f"{path.resolve()}|{path.stat().st_mtime_ns}|{path.stat().st_size}"
        except OSError:
            key = str(path)
        info = cache.get(key)
        if info is None:
            info = analyze_file(path)
            cache[key] = info
            if not quiet:
                log(f"  {path.name:<40} {info['bpm']:>6.1f} BPM  {info['key']:<9} "
                    f"{fmt_time(info['duration'])}")
        info = dict(info)
        info["path"] = str(path)
        info["name"] = path.stem
        out.append(info)
    if cache_path:
        try:
            Path(cache_path).write_text(json.dumps(cache, indent=1), encoding="utf-8")
        except OSError:
            pass
    return out


def choose_target(tracks, bpm=None, key=None):
    """Pick the tempo and key the whole song will live in."""
    bpms = [t["bpm"] for t in tracks if t.get("bpm")]
    if bpm:
        target_bpm = float(bpm)
    elif bpms:
        target_bpm = float(np.median(bpms))
    else:
        target_bpm = 80.0
    target_bpm = float(np.clip(target_bpm, 55.0, 110.0))

    if key:
        target_key = key
    else:
        # the key that needs the least total shifting across the whole list
        keys = [t.get("key", "C Major") for t in tracks]
        if keys:
            target_key = min(keys, key=lambda k: sum(key_distance(k, o) for o in keys))
        else:
            target_key = "A Minor"
    return round(target_bpm, 2), target_key


def order_tracks(tracks, mode="harmonic", seed=0):
    """Sequence the tracks: harmonic journey, shuffled, or as given."""
    if mode == "asis":
        return list(tracks)
    if mode == "shuffle":
        out = list(tracks)
        random.Random(seed).shuffle(out)
        return out
    # greedy nearest-key walk - each transition moves as little as possible
    remaining = list(tracks)
    start = min(remaining, key=lambda t: t.get("bpm", 0))
    order = [start]
    remaining.remove(start)
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda t: (key_distance(last.get("key", "C Major"),
                                                         t.get("key", "C Major")),
                                            abs(t.get("bpm", 80) - last.get("bpm", 80))))
        order.append(nxt)
        remaining.remove(nxt)
    return order


# ------------------------------------------------------------- tone shaping ---

BANDS_HZ = np.array([40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500,
                     4000, 6300, 10000, 14000], dtype=float)


def band_profile(x, sr):
    """Average energy per band, in dB - a cheap spectral fingerprint of 'tone'."""
    mono = x.mean(axis=1) if x.ndim > 1 else x
    n = min(len(mono), sr * 60)
    if n < 4096:
        return np.zeros(len(BANDS_HZ))
    seg = mono[:n]
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1.0 / sr)
    out = np.zeros(len(BANDS_HZ))
    for i, centre in enumerate(BANDS_HZ):
        lo, hi = centre / 1.26, centre * 1.26
        sel = (f >= lo) & (f < hi)
        out[i] = 10 * math.log10(S[sel].sum() + 1e-20) if sel.any() else -200.0
    return out


def match_tone(x, sr, target, source=None, strength=0.7, max_db=6.0):
    """Nudge a track's spectral balance toward `target` so a mix sounds cohesive."""
    from scipy.signal import firwin2, lfilter
    source = band_profile(x, sr) if source is None else source
    if not np.isfinite(target).all() or not np.isfinite(source).all():
        return x
    delta = np.clip((target - source) * strength, -max_db, max_db)
    delta -= delta.mean()                    # correct balance, not level
    gains = 10 ** (delta / 20.0)
    freqs = np.concatenate([[0.0], BANDS_HZ, [sr / 2.0]])
    amps = np.concatenate([[gains[0]], gains, [gains[-1]]])
    freqs = np.clip(freqs / (sr / 2.0), 0.0, 1.0)
    freqs[0], freqs[-1] = 0.0, 1.0
    keep = np.concatenate([[True], np.diff(freqs) > 1e-6])
    taps = firwin2(257, freqs[keep], amps[keep])
    y = lfilter(taps, [1.0], np.asarray(x, dtype=np.float64), axis=0)
    return np.roll(y, -128, axis=0).astype(np.float32)   # undo the FIR delay


# --------------------------------------------------------------- glue layer ---

class GlueBed:
    """Everything that runs continuously under the whole song.

    The vinyl bed and the drum spine never reset between tracks - that
    continuity is most of what makes a long mix feel like one piece rather
    than a playlist.
    """

    def __init__(self, sr, bpm, vinyl=1.0, spine=0.0, spine_style="dusty",
                 spine_pattern="lazy", seed=0, swing=0.16):
        self.sr = sr
        self.bpm = bpm
        self.vinyl = dsp.VinylNoise(sr, level_db=-41.0 + 5.0 * min(vinyl, 1.5),
                                    crackle=vinyl, seed=seed) if vinyl > 0 else None
        self.spine_level = float(spine)
        self.swing = swing
        self.kit = DrumKit(sr, spine_style, seed=seed) if spine > 0 else None
        self.pattern = PATTERNS.get(spine_pattern, PATTERNS["lazy"])
        self.rng = random.Random(seed)
        self.bar_samples = int(round(60.0 / bpm * 4 * sr))
        self._spine_buf = np.zeros((0, 2), dtype=np.float32)
        self._complete = 0        # samples of _spine_buf that no future bar can change
        self._pos = 0

    def _render_bar(self):
        """One bar of the spine, plus room for hits that ring past the barline."""
        n = self.bar_samples + int(1.6 * self.sr)
        buf = np.zeros((n, 2), dtype=np.float32)
        step_s = 60.0 / self.bpm / 4.0
        for piece, grid in self.pattern.items():
            for step, vel in enumerate(grid):
                if vel <= 0 or self.rng.random() < 0.08:
                    continue
                swing = self.swing * step_s if step % 2 == 1 else 0.0
                start = int((step * step_s + swing) * self.sr)
                sig = self.kit.hit(piece, vel * (0.85 + 0.25 * self.rng.random()))
                end = min(n, start + len(sig))
                if end <= start:
                    continue
                chunk = sig[: end - start]
                pan = 0.0 if piece in ("kick", "snare") else (-0.25 if "hat" in piece else 0.25)
                buf[start:end, 0] += chunk * math.sqrt(0.5 * (1 - pan)) * 1.414
                buf[start:end, 1] += chunk * math.sqrt(0.5 * (1 + pan)) * 1.414
        return buf

    def _ensure_spine(self, n):
        """Render bars until `n` samples are settled, overlapping each bar's tail."""
        while self._complete < n:
            bar = self._render_bar()
            end = self._complete + len(bar)
            if len(self._spine_buf) < end:
                pad = np.zeros((end - len(self._spine_buf), 2), dtype=np.float32)
                self._spine_buf = np.concatenate([self._spine_buf, pad], axis=0)
            self._spine_buf[self._complete:end] += bar
            self._complete += self.bar_samples

    def block(self, n, spine_gain=1.0):
        out = np.zeros((n, 2), dtype=np.float32)
        if self.vinyl is not None:
            out += self.vinyl.block(n)
        if self.kit is not None and self.spine_level > 0:
            self._ensure_spine(n)
            out += self._spine_buf[:n] * (self.spine_level * float(spine_gain))
            self._spine_buf = self._spine_buf[n:]
            self._complete = max(0, self._complete - n)
        self._pos += n
        return out


class MasterBus:
    """One shared master chain for the whole song - the other half of the glue."""

    def __init__(self, sr, tape=0.7, width=1.25, warmth=1.0, ceiling_db=-1.0):
        self.rumble = dsp.Biquad("highpass", sr, 28.0, 0.7)
        self.mud = dsp.Biquad("peak", sr, 270.0, 1.0, gain_db=-1.8 * warmth)
        self.low = dsp.Biquad("lowshelf", sr, 110.0, 0.7, gain_db=1.2 * warmth)
        self.presence = dsp.Biquad("peak", sr, 2600.0, 0.8, gain_db=1.6)
        self.air = dsp.Biquad("highshelf", sr, 11000.0, 0.7, gain_db=-3.0 * tape)
        self.wow = dsp.TapeWow(sr, wow_ms=1.6 * tape, flutter_ms=0.22 * tape) if tape > 0 else None
        self.widen = dsp.StereoWiden(sr, width=width, bass_mono_hz=140.0)
        self.glue = dsp.Compressor(sr, threshold_db=-14.0, ratio=1.8, attack_ms=30.0,
                                   release_ms=280.0, knee_db=9.0)
        self.limiter = dsp.Limiter(sr, ceiling_db=ceiling_db, lookahead_ms=6.0,
                                   release_ms=150.0)
        self.tape = tape

    def process(self, x):
        y = self.presence.process(self.mud.process(self.low.process(self.rumble.process(x))))
        if self.wow is not None:
            y = self.wow.process(y)
            y = dsp.tape_saturate(y, drive=1.0 + 0.35 * self.tape)
            y = self.air.process(y)
        y = self.widen.process(y)
        return self.limiter.process(self.glue.process(y))

    def flush(self):
        return self.limiter.flush()


# -------------------------------------------------------------- preparation ---

def prepare_track(info, sr, target_bpm, target_key, lufs=-15.0, tone_target=None,
                  max_stretch=18.0, max_shift=4, tone_strength=0.7, lofi=0.0):
    """Load one track and fit it to the song's tempo, key, level and tone."""
    x, _ = load_audio(info["path"], target_sr=sr)
    if len(x) < sr:
        return None, {}
    notes = {}

    rate = 1.0
    if target_bpm and info.get("bpm"):
        x, rate = fit_to_tempo(x, sr, info["bpm"], target_bpm, max_percent=max_stretch)
        notes["stretch"] = round(rate, 4)
        if abs(rate - 1.0) < 1e-6 and abs(info["bpm"] - target_bpm) > 2:
            notes["stretch_skipped"] = "would need more than the allowed change"

    shift = 0
    if target_key and info.get("key"):
        shift = semitones_to_key(info["key"], target_key, max_shift=max_shift)
        if shift:
            x = pitch_shift(x, shift, sr)
    notes["pitch_semitones"] = shift

    measured = integrated_lufs(x, sr)
    if np.isfinite(measured):
        x = x * (10 ** ((lufs - measured) / 20.0))
        notes["lufs_in"] = round(float(measured), 1)

    if tone_target is not None and tone_strength > 0:
        x = match_tone(x, sr, tone_target, strength=tone_strength)

    if lofi > 0:
        x = dsp.Biquad("lowpass", sr, 16000.0 - 8000.0 * lofi, 0.7).process(x)
        x = dsp.tape_saturate(x, drive=1.0 + 0.5 * lofi)
        if lofi > 0.5:
            x = dsp.bit_crush(x, bits=int(16 - 4 * lofi))
        notes["lofi"] = round(lofi, 2)

    np.clip(x, -1.0, 1.0, out=x)
    return x, notes


def snap_to_bars(x, sr, bpm, beats_per_bar=4, downbeat=0.0):
    """Trim a track to a whole number of bars, starting on its downbeat."""
    bar = 60.0 / bpm * beats_per_bar
    start = int(max(0.0, downbeat) * sr)
    if start >= len(x):
        start = 0
    body = x[start:]
    bars = int(len(body) / (bar * sr))
    if bars < 1:
        return body
    return body[: int(round(bars * bar * sr))]


# ------------------------------------------------------------------- output ---

ROMAN = ["", " (II)", " (III)", " (IV)", " (V)", " (VI)", " (VII)", " (VIII)"]


def _title_for(info, manifest_titles, seen=None):
    """Chapter title, with a numeral when a track comes round again.

    YouTube chapters have to be distinguishable; three identical names in one
    description read as a mistake.
    """
    base = manifest_titles.get(Path(info["path"]).stem) or \
        Path(info["path"]).stem.lstrip("0123456789_- ").replace("-", " ").replace("_", " ").title()
    base = base.strip() or "Untitled"
    if seen is None:
        return base
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base + (ROMAN[count] if count < len(ROMAN) else f" ({count + 1})")


def build_song(paths, out_path, minutes=0.0, sr=DEFAULT_SR, bpm=None, key=None,
               order="harmonic", crossfade_bars=4, seed=0, vinyl=0.8, spine=0.0,
               tape=0.7, width=1.25, lufs=-14.0, peak_db=-1.0, tone_strength=0.7,
               lofi=0.0, max_stretch=18.0, max_shift=4, fade_out=12.0,
               cache_path=None, titles=None, progress=True, spine_style="dusty",
               spine_pattern="lazy"):
    """Build ONE continuous song from many tracks.  Streams; returns a report."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("no input tracks")
    titles = titles or {}

    log(f"Analysing {len(paths)} tracks...")
    tracks = analyze_tracks(paths, cache_path=cache_path, quiet=not progress)
    target_bpm, target_key = choose_target(tracks, bpm, key)
    ordered = order_tracks(tracks, order, seed)
    log(f"\nTarget: {target_bpm:.1f} BPM, {target_key}   "
        f"(order: {order}, crossfade: {crossfade_bars} bars)")

    bar_s = 60.0 / target_bpm * 4
    cf_n = max(int(0.5 * sr), int(crossfade_bars * bar_s * sr))
    target_n = int(minutes * 60 * sr) if minutes else 0
    if target_n:
        check_disk(Path(out_path).parent, int(target_n * 2 * 3 * 1.1), "the song")

    # the tone target is the average colour of the first few tracks
    tone_target = None
    if tone_strength > 0:
        profiles = []
        for info in ordered[: min(4, len(ordered))]:
            sample, _ = load_audio(info["path"], target_sr=sr, max_seconds=60)
            if len(sample):
                profiles.append(band_profile(sample, sr))
        if profiles:
            tone_target = np.mean(profiles, axis=0)

    glue = GlueBed(sr, target_bpm, vinyl=vinyl, spine=spine, seed=seed,
                   spine_style=spine_style, spine_pattern=spine_pattern)
    bus = MasterBus(sr, tape=tape, width=width, ceiling_db=peak_db)
    meter = StreamingLoudness(sr)
    writer = StreamWriter(out_path, sr=sr, channels=2, subtype="PCM_24")

    chapters, used, prev_tail, idx, repeats = [], [], None, 0, 0
    seen_titles = {}
    skipped = []
    rng = random.Random(seed)

    def emit(block, spine_gain=1.0):
        if len(block) == 0:
            return
        out = bus.process(block + glue.block(len(block), spine_gain))
        meter.push(out)
        writer.write(out)

    try:
        while True:
            info = ordered[idx % len(ordered)]
            pass_no = idx // len(ordered)
            if pass_no and idx % len(ordered) == 0:
                repeats += 1
            # later passes drift tempo/pitch slightly so a long song never
            # repeats itself note-for-note
            drift_bpm = target_bpm * (1.0 + (rng.random() - 0.5) * 0.02 * min(pass_no, 3))
            x, notes = prepare_track(info, sr, drift_bpm, target_key, lufs=lufs - 1.0,
                                     tone_target=tone_target, max_stretch=max_stretch,
                                     max_shift=max_shift, tone_strength=tone_strength,
                                     lofi=lofi)
            if x is None or len(x) < cf_n + sr:
                idx += 1
                if idx > len(ordered) * 40:
                    break
                continue
            x = snap_to_bars(x, sr, target_bpm, downbeat=info.get("downbeat", 0.0))
            if len(x) < cf_n * 2:
                idx += 1
                continue

            title = _title_for(info, titles, seen_titles)
            if notes.get("stretch_skipped"):
                skipped.append(f"{info['name']} ({info.get('bpm', 0):.0f} BPM)")
            if prev_tail is None:
                chapters.append((0.0, title))
                emit(x[:-cf_n])
                prev_tail = x[-cf_n:].copy()
            else:
                c = min(cf_n, len(prev_tail), len(x))
                chapters.append((writer.seconds, title))
                fade_out_c = dsp.equal_power_fade(c, fade_in=False)
                fade_in_c = dsp.equal_power_fade(c, fade_in=True)
                # bass swap: the outgoing low end steps aside before the new one
                # arrives, so two basses never stack into mud mid-transition
                lp = dsp.Biquad("highpass", sr, 180.0, 0.7)
                head = x[:c].copy()
                half = c // 2
                head[:half] = lp.process(head[:half])
                blend = prev_tail[:c] * fade_out_c + head * fade_in_c
                emit(blend)
                body = x[c:]
                if len(body) > cf_n:
                    emit(body[:-cf_n])
                    prev_tail = body[-cf_n:].copy()
                else:
                    prev_tail = body
            used.append({"track": info["name"], "bpm_in": info.get("bpm"),
                         "key_in": info.get("key"), **notes})
            idx += 1
            total = writer.frames + len(prev_tail)
            if progress:
                log(f"  [{len(chapters):>3}] {title:<34} -> {fmt_time(writer.seconds)}")
            if target_n and total >= target_n:
                break
            if not target_n and idx >= len(ordered):
                break
            if idx > len(ordered) * 200:
                warn("stopping: inputs are too short to reach the requested length")
                break

        if prev_tail is not None and len(prev_tail):
            n = min(len(prev_tail), int(fade_out * sr))
            if n > 1:
                prev_tail = prev_tail.copy()
                prev_tail[-n:] *= dsp.cosine_fade(n, fade_in=False)
            emit(prev_tail, spine_gain=0.0)
        tail = bus.flush()
        if len(tail):
            meter.push(tail)
            writer.write(tail)
    finally:
        writer.close()

    report = {
        "path": str(out_path), "seconds": writer.seconds, "samplerate": sr,
        "target_bpm": target_bpm, "target_key": target_key, "order": order,
        "crossfade_bars": crossfade_bars, "segments": len(chapters),
        "unique_tracks": len({u["track"] for u in used}), "passes": repeats + 1,
        "peak_dbfs": round(20 * math.log10(writer.peak + 1e-12), 2),
        "lufs": round(meter.value(), 2), "chapters": chapters, "tracks": used,
        "spine": spine, "vinyl": vinyl,
        "not_beat_matched": sorted(set(skipped)),
    }
    if skipped:
        warn(f"left at their own tempo (too far from {target_bpm:.0f} BPM to stretch "
             f"cleanly): {', '.join(sorted(set(skipped)))}")
        log("      raise --max-stretch, or set --bpm closer to these tracks.")
    return report


def build_mix(paths, out_path, minutes=0.0, sr=DEFAULT_SR, crossfade=5.0,
              final_fade=8.0, shuffle=False, seed=0, titles=None, progress=True,
              peak_db=-1.0):
    """The classic compilation: tracks back to back with equal-power crossfades."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("no input tracks")
    titles = titles or {}
    order = list(paths)
    if shuffle:
        random.Random(seed).shuffle(order)

    cf = int(crossfade * sr)
    fade_n = int(final_fade * sr)
    hold = max(cf, fade_n)
    target_n = int(minutes * 60 * sr) if minutes else 0
    if target_n:
        check_disk(Path(out_path).parent, int(target_n * 2 * 3 * 1.1), "the mix")

    limiter = dsp.Limiter(sr, ceiling_db=peak_db, lookahead_ms=5.0, release_ms=130.0)
    meter = StreamingLoudness(sr)
    writer = StreamWriter(out_path, sr=sr, channels=2, subtype="PCM_24")
    chapters, prev_tail, i, repeated = [], None, 0, False

    def emit(block):
        if len(block) == 0:
            return
        out = limiter.process(block)
        meter.push(out)
        writer.write(out)

    try:
        while True:
            path = order[i % len(order)]
            if i >= len(order):
                repeated = True
            x, _ = load_audio(path, target_sr=sr)
            if len(x) < sr:
                i += 1
                if i > len(order) * 40:
                    break
                continue
            title = titles.get(path.stem) or path.stem.lstrip("0123456789_- ") \
                .replace("-", " ").replace("_", " ").title()
            if prev_tail is None:
                chapters.append((0.0, title))
                body = x
            else:
                c = min(cf, len(prev_tail), len(x))
                if len(prev_tail) > c:
                    emit(prev_tail[:-c])
                chapters.append((writer.seconds, title))
                if c > 0:
                    blend = (prev_tail[-c:] * dsp.equal_power_fade(c, False)
                             + x[:c] * dsp.equal_power_fade(c, True))
                    emit(blend)
                body = x[c:]
            if len(body) > hold:
                emit(body[:-hold])
                prev_tail = body[-hold:].copy()
            else:
                prev_tail = body.copy()
            i += 1
            if progress:
                log(f"  [{len(chapters):>3}] {title:<34} -> {fmt_time(writer.seconds)}")
            total = writer.frames + len(prev_tail)
            if target_n and total >= target_n:
                break
            if not target_n and i >= len(order):
                break
            if i > len(order) * 200:
                warn("stopping: inputs are too short to reach the requested length")
                break

        if prev_tail is not None and len(prev_tail):
            n = min(len(prev_tail), fade_n)
            if n > 1:
                prev_tail = prev_tail.copy()
                prev_tail[-n:] *= dsp.cosine_fade(n, fade_in=False)
            emit(prev_tail)
        tail = limiter.flush()
        if len(tail):
            meter.push(tail)
            writer.write(tail)
    finally:
        writer.close()

    return {"path": str(out_path), "seconds": writer.seconds, "segments": len(chapters),
            "chapters": chapters, "repeated": repeated,
            "peak_dbfs": round(20 * math.log10(writer.peak + 1e-12), 2),
            "lufs": round(meter.value(), 2), "samplerate": sr}


def write_tracklist(path, chapters, total_seconds):
    """YouTube chapter list.  Needs 3+ entries and must start at 00:00."""
    long_form = total_seconds >= 3600
    lines = [f"{fmt_time(t, long_form)} {name}" for t, name in chapters]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines
