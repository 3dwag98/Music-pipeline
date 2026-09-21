"""The local lofi engine: composes and renders music with no model and no GPU.

    plan_song()  -> a structure of sections
    build_events() -> concrete notes and drum hits per section
    render_song() -> streams audio to disk, section by section

Because it is pure DSP it runs on the CPU while ACE-Step has the GPU, it never
runs out of VRAM, and it can render for hours at many times realtime.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict

import numpy as np

from . import dsp
from .audio import StreamWriter
from .drums import PATTERNS, DrumKit
from .synth import NoteCache
from .theory import (key_name, make_progression, melody_pool, midi_to_hz,
                     parse_key, scale_degrees)
from .util import fmt_time, log

BEAT_UNITS = 16          # sixteenth-note grid


# ------------------------------------------------------------------- specs ---

@dataclass
class Section:
    name: str
    bars: int
    drums: bool = True
    bass: bool = True
    chords: bool = True
    lead: bool = False
    pad: bool = True
    texture: bool = True
    intensity: float = 1.0
    cutoff: float = 16000.0       # lowpass on the musical bus, for filter sweeps
    chord_shift: int = 0          # rotate the progression for variation


@dataclass
class SongSpec:
    title: str = "Untitled"
    seed: int = 0
    bpm: float = 78.0
    key: str = "A Minor"
    swing: float = 0.16
    beats_per_bar: int = 4
    bars_per_loop: int = 4
    chord_instrument: str = "rhodes"
    lead_instrument: str = "vibraphone"
    bass_instrument: str = "sub_bass"
    pad_instrument: str = "pad"
    drum_style: str = "dusty"
    drum_pattern: str = "boom_bap"
    richness: float = 0.85
    humanize: float = 1.0
    vinyl: float = 1.0
    tape: float = 1.0
    reverb: float = 0.35
    width: float = 1.45
    lead_density: float = 0.55
    progression: str = ""
    sections: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["sections"] = [asdict(s) if hasattr(s, "name") else s for s in self.sections]
        return d


# -------------------------------------------------------------- arrangement ---

ARC_SHORT = [
    ("intro", 4, dict(drums=False, lead=False, cutoff=2600.0, intensity=0.55)),
    ("verse", 8, dict(lead=False, intensity=0.85)),
    ("main", 8, dict(lead=True, intensity=1.0)),
    ("break", 4, dict(drums=False, lead=True, intensity=0.6, cutoff=5200.0)),
    ("main", 8, dict(lead=True, intensity=1.0, chord_shift=1)),
    ("outro", 4, dict(lead=False, intensity=0.6, cutoff=3200.0)),
]

ARC_LOOP = [
    ("verse", 8, dict(lead=False, intensity=0.85)),
    ("main", 8, dict(lead=True, intensity=1.0)),
    ("lift", 8, dict(lead=True, intensity=1.05, chord_shift=1)),
    ("break", 4, dict(drums=False, lead=True, intensity=0.6, cutoff=5200.0)),
    ("main", 8, dict(lead=True, intensity=0.95, chord_shift=2)),
    ("rest", 8, dict(lead=False, intensity=0.8, chord_shift=3)),
]


def plan_song(spec: SongSpec, minutes: float, rng: random.Random):
    """Lay out sections until the wanted length is covered."""
    bar_s = 60.0 / spec.bpm * spec.beats_per_bar
    target_bars = max(8, int(round(minutes * 60.0 / bar_s)))
    sections, bars = [], 0

    intro = Section("intro", 4, **ARC_SHORT[0][2])
    sections.append(intro)
    bars += intro.bars

    arc = ARC_SHORT[1:-1] if target_bars < 60 else ARC_LOOP
    i = 0
    while bars < target_bars - 4:
        name, length, kwargs = arc[i % len(arc)]
        kwargs = dict(kwargs)
        if i >= len(arc):                     # after the first pass, keep varying
            kwargs["chord_shift"] = (kwargs.get("chord_shift", 0) + i // len(arc)) % 4
            if rng.random() < 0.22:
                kwargs["cutoff"] = rng.choice([4200.0, 6500.0, 16000.0])
            if rng.random() < 0.15:
                kwargs["drums"] = False
            if rng.random() < 0.3:
                kwargs["lead"] = not kwargs.get("lead", False)
        length = min(length, max(4, target_bars - bars - 4))
        sections.append(Section(name, length, **kwargs))
        bars += length
        i += 1

    outro = Section("outro", max(4, target_bars - bars), **ARC_SHORT[-1][2])
    sections.append(outro)
    return sections


# ------------------------------------------------------------------ melody ---

def make_motif(rng: random.Random, length=6):
    """A short rhythm+contour idea that the whole song refers back to."""
    steps = []
    pos = 0
    durations = [2, 2, 3, 4, 4, 6, 8]
    while pos < BEAT_UNITS and len(steps) < length:
        d = rng.choice(durations)
        if pos + d > BEAT_UNITS:
            d = BEAT_UNITS - pos
        steps.append((pos, d))
        pos += d
        if rng.random() < 0.28:               # a rest keeps it from feeling mechanical
            pos += rng.choice([1, 2])
    contour = [rng.choice([-2, -1, 0, 0, 1, 1, 2, 3]) for _ in steps]
    return list(zip([s for s, _ in steps], [d for _, d in steps], contour))


def transform_motif(motif, rng: random.Random, mode=None):
    mode = mode or rng.choice(["same", "same", "transpose", "invert", "retro", "shift", "sparse"])
    out = list(motif)
    if mode == "transpose":
        step = rng.choice([-2, -1, 1, 2])
        out = [(s, d, c + step) for s, d, c in out]
    elif mode == "invert":
        out = [(s, d, -c) for s, d, c in out]
    elif mode == "retro":
        contours = [c for _, _, c in out][::-1]
        out = [(s, d, contours[i]) for i, (s, d, _) in enumerate(out)]
    elif mode == "shift":
        delta = rng.choice([1, 2, -1])
        out = [((s + delta) % BEAT_UNITS, d, c) for s, d, c in out]
        out.sort()
    elif mode == "sparse":
        out = [e for i, e in enumerate(out) if i % 2 == 0] or out[:1]
    return out, mode


def melody_for_bar(chord, motif, root_pc, scale, rng, register=72, density=1.0):
    """Turn the motif into real pitches over this bar's chord."""
    strong, weak = melody_pool(chord, root_pc, scale, low=register - 7, high=register + 14)
    if not strong:
        return []
    notes = []
    anchor = min(strong, key=lambda m: abs(m - register))
    idx = strong.index(anchor)
    for start, dur, contour in motif:
        if rng.random() > density:
            continue
        target = idx + contour
        if 0 <= target < len(strong):
            midi = strong[target]
        else:
            midi = strong[max(0, min(len(strong) - 1, target))]
        # an off-beat note may lean on a neighbouring scale tone (a passing note)
        if start % 4 != 0 and weak and rng.random() < 0.35:
            near = min(weak, key=lambda m: abs(m - midi))
            if abs(near - midi) <= 2:
                midi = near
        notes.append((start, dur, midi))
    return notes


# ------------------------------------------------------------------ events ---

def build_events(spec: SongSpec, section: Section, chords, motif, rng, root_pc, scale,
                 bar_index=0):
    """Concrete note/hit events for one section, in (bar, step) grid time."""
    events = []           # (bar, step_float, stem, kind, midi, dur_steps, velocity, pan)
    pattern = PATTERNS.get(spec.drum_pattern, PATTERNS["boom_bap"])
    n_chords = len(chords)
    lead_motif = motif

    for bar in range(section.bars):
        chord = chords[(bar + section.chord_shift) % n_chords]
        intensity = section.intensity

        # ---- chord comping -------------------------------------------------
        if section.chords:
            comp = _comp_pattern(rng, spec.drum_pattern)
            for step, hold, vel in comp:
                for j, midi in enumerate(chord["notes"]):
                    pan = -0.55 + 1.10 * (j / max(1, len(chord["notes"]) - 1))
                    events.append((bar, step + j * 0.08, "chords", "note", midi,
                                   hold, vel * intensity * (0.85 + 0.15 * rng.random()),
                                   pan))

        # ---- bass ----------------------------------------------------------
        if section.bass:
            root = chord["bass_midi"]
            fifth = root + 7
            plan = [(0, 6, 1.0, root)]
            r = rng.random()
            if r < 0.45:
                plan.append((8, 5, 0.75, root))
            elif r < 0.75:
                plan.append((8, 3, 0.7, fifth))
                plan.append((12, 3, 0.6, root))
            else:
                plan.append((10, 5, 0.7, root - 0 if rng.random() < 0.5 else fifth))
            for step, hold, vel, midi in plan:
                events.append((bar, step, "bass", "note", midi, hold,
                               vel * intensity, 0.0))

        # ---- pad -----------------------------------------------------------
        if section.pad:
            for j, midi in enumerate(chord["notes"][:4]):
                events.append((bar, 0.0, "pad", "note", midi - 12, BEAT_UNITS,
                               0.5 * intensity, -0.7 + 1.4 * (j % 2)))

        # ---- lead ----------------------------------------------------------
        if section.lead:
            if bar % 2 == 0:
                lead_motif, _ = transform_motif(motif, rng)
            density = spec.lead_density * (0.6 if bar % 4 == 3 else 1.0)
            for step, dur, midi in melody_for_bar(chord, lead_motif, root_pc, scale,
                                                  rng, register=74, density=density):
                events.append((bar, step, "lead", "note", midi, dur,
                               (0.62 + 0.3 * rng.random()) * intensity, 0.12))

        # ---- drums ---------------------------------------------------------
        if section.drums and pattern:
            for piece, grid in pattern.items():
                for step, vel in enumerate(grid):
                    if vel <= 0:
                        continue
                    v = vel * intensity
                    if rng.random() < 0.06:            # occasional dropped hit
                        continue
                    if piece == "hat" and rng.random() < 0.05:
                        piece_name = "hat_open"
                    else:
                        piece_name = piece
                    events.append((bar, float(step), "drums", piece_name, 0, 1,
                                   v * (0.88 + 0.24 * rng.random()),
                                   0.0 if piece in ("kick", "snare") else
                                   (-0.30 if piece in ("hat", "hat_open") else 0.34)))
            # a fill at the end of every 8th bar
            if (bar_index + bar + 1) % 8 == 0 and rng.random() < 0.6:
                for k in range(3):
                    events.append((bar, 13.0 + k, "drums",
                                   rng.choice(["snare", "tom", "rim"]), 0, 1,
                                   0.5 + 0.18 * k, -0.3 + 0.3 * k))
    return events


def _comp_pattern(rng, drum_pattern):
    """(step, hold_in_steps, velocity) for the chord instrument."""
    base = {
        "boom_bap": [(0, 7, 0.85), (6, 4, 0.55), (10, 6, 0.7)],
        "lazy":     [(0, 8, 0.8), (8, 8, 0.65)],
        "halftime": [(0, 12, 0.8), (12, 4, 0.5)],
        "swing":    [(0, 5, 0.8), (4, 3, 0.5), (7, 4, 0.65), (12, 4, 0.6)],
        "shuffle":  [(0, 6, 0.85), (6, 3, 0.5), (10, 5, 0.7)],
        "brushed":  [(0, 8, 0.75), (8, 8, 0.6)],
        "tabla":    [(0, 8, 0.7), (8, 8, 0.6)],
        "none":     [(0, 16, 0.7)],
    }.get(drum_pattern, [(0, 8, 0.8), (8, 8, 0.65)])
    out = list(base)
    if rng.random() < 0.3 and len(out) > 1:
        out = out[:-1]
    return out


# ------------------------------------------------------------------ render ---

def _step_time(step, spec, sr):
    """Grid step -> sample offset inside the bar, with swing applied."""
    beat = 60.0 / spec.bpm
    step_s = beat / 4.0
    swing = 0.0
    if spec.swing and int(step) % 2 == 1:
        swing = spec.swing * step_s
    return (step * step_s + swing) * sr


class LofiChain:
    """The signature processing, kept stateful so hours stream seamlessly."""

    def __init__(self, sr, spec: SongSpec, seed=0):
        self.sr = sr
        self.spec = spec
        self.drum_comp = dsp.Compressor(sr, threshold_db=-16.0, ratio=3.2,
                                        attack_ms=6.0, release_ms=140.0)
        # Gentle glue only - a hard bus ratio flattens the arrangement dynamics
        # that make an hours-long listen bearable.
        self.bus_comp = dsp.Compressor(sr, threshold_db=-13.0, ratio=1.7,
                                       attack_ms=28.0, release_ms=260.0, knee_db=9.0)
        self.ducker = dsp.Ducker(sr, depth_db=-2.6 * spec.tape - 0.8,
                                 attack_ms=4.0, release_ms=180.0)
        self.reverb = dsp.FDNReverb(sr, room=0.55 + 0.35 * spec.reverb,
                                    damping=0.42, width=0.95, predelay_ms=22.0)
        self.wow = dsp.TapeWow(sr, wow_hz=0.62, wow_ms=2.2 * spec.tape,
                               flutter_ms=0.30 * spec.tape, seed=seed)
        self.vinyl = dsp.VinylNoise(sr, level_db=-40.0 + 6.0 * min(spec.vinyl, 1.5),
                                    crackle=spec.vinyl, seed=seed + 11)
        self.widen = dsp.StereoWiden(sr, width=spec.width, bass_mono_hz=140.0)
        # Tone stack.  Lofi is dark, but not muddy: cut the 250 Hz build-up that
        # stacked chords + bass + kick create, keep a little presence so the
        # melody reads on a phone speaker, and only then roll the top off.
        self.air = dsp.Biquad("highshelf", sr, 10500.0, 0.7, gain_db=-4.0 * spec.tape)
        self.rumble = dsp.Biquad("highpass", sr, 30.0, 0.7)
        self.warmth = dsp.Biquad("lowshelf", sr, 120.0, 0.7, gain_db=1.4)
        self.mud = dsp.Biquad("peak", sr, 265.0, 1.0, gain_db=-2.8)
        self.presence = dsp.Biquad("peak", sr, 2600.0, 0.8, gain_db=2.4)
        self.sweep = dsp.Biquad("lowpass", sr, 16000.0, 0.7)
        self._sweep_hz = 16000.0

    def set_cutoff(self, hz):
        hz = float(np.clip(hz, 300.0, 18000.0))
        if abs(hz - self._sweep_hz) > 1.0:
            self.sweep = dsp.Biquad("lowpass", self.sr, hz, 0.7)
            self._sweep_hz = hz

    def process(self, stems: dict, n: int) -> np.ndarray:
        drums = stems.get("drums")
        if drums is not None:
            drums = self.drum_comp.process(drums)
        music = np.zeros((n, 2), dtype=np.float32)
        for name in ("chords", "bass", "pad", "lead"):
            part = stems.get(name)
            if part is not None:
                music += part
        if drums is not None:
            music = self.ducker.process(music, np.abs(drums).max(axis=1))
        mix = music + (drums if drums is not None else 0.0)
        mix = self.sweep.process(mix)
        send = self.reverb.process(mix * (0.18 + 0.30 * self.spec.reverb))
        mix = mix + send
        mix = self.presence.process(self.mud.process(self.warmth.process(self.rumble.process(mix))))
        if self.spec.tape > 0:
            mix = self.wow.process(mix)
            mix = dsp.tape_saturate(mix, drive=1.0 + 0.45 * self.spec.tape)
            mix = self.air.process(mix)
        mix = self.widen.process(mix)
        if self.spec.vinyl > 0:
            mix = mix + self.vinyl.block(n)
        return self.bus_comp.process(mix)


STEM_GAINS = {"drums": 0.95, "chords": 0.62, "bass": 0.80, "pad": 0.45, "lead": 0.52}


def render_section(spec, section, events, chords, sr, cache, kit, tail_samples,
                   carry=None):
    """Render one section into per-stem buffers.  Returns (stems, carry_tail)."""
    beat = 60.0 / spec.bpm
    bar_samples = int(round(beat * spec.beats_per_bar * sr))
    n = bar_samples * section.bars
    total = n + tail_samples
    stems = {name: np.zeros((total, 2), dtype=np.float32)
             for name in ("drums", "chords", "bass", "pad", "lead")}
    if carry:
        for name, buf in carry.items():
            if name in stems and len(buf):
                k = min(len(buf), total)
                stems[name][:k] += buf[:k]

    step_s = beat / 4.0
    for bar, step, stem, kind, midi, hold, vel, pan in events:
        start = bar * bar_samples + int(_step_time(step, spec, sr))
        if start >= total:
            continue
        vel = float(np.clip(vel, 0.02, 1.4))
        if stem == "drums":
            sig = kit.hit(kind, vel)
        else:
            instrument = {"chords": spec.chord_instrument, "bass": spec.bass_instrument,
                          "pad": spec.pad_instrument, "lead": spec.lead_instrument}[stem]
            dur = max(0.08, hold * step_s * (1.35 if stem == "pad" else 1.05))
            sig = cache.render(instrument, midi, dur, vel)
        end = min(total, start + len(sig))
        if end <= start:
            continue
        chunk = sig[: end - start]
        left = math.sqrt(max(0.0, 0.5 * (1.0 - pan)))
        right = math.sqrt(max(0.0, 0.5 * (1.0 + pan)))
        stems[stem][start:end, 0] += chunk * (left * 1.414)
        stems[stem][start:end, 1] += chunk * (right * 1.414)

    body = {k: v[:n] * STEM_GAINS[k] for k, v in stems.items()}
    carry_out = {k: v[n:].copy() * STEM_GAINS[k] for k, v in stems.items()}
    return body, carry_out, n


def render_song(spec: SongSpec, out_path, minutes=3.0, sr=44100, progress=True,
                stems_dir=None, target_lufs=-14.0, peak_db=-1.0):
    """Compose and stream a complete song to `out_path`.  Returns a report dict."""
    rng = random.Random(spec.seed)
    root_pc, scale = parse_key(spec.key)
    label, chords = make_progression(root_pc, scale, rng, spec.bars_per_loop, spec.richness)
    spec.progression = label
    spec.sections = plan_song(spec, minutes, rng)

    cache = NoteCache(sr)
    kit = DrumKit(sr, spec.drum_style, seed=spec.seed)
    chain = LofiChain(sr, spec, seed=spec.seed)
    limiter = dsp.Limiter(sr, ceiling_db=peak_db, lookahead_ms=6.0, release_ms=140.0)
    motif = make_motif(rng, length=rng.randint(4, 7))

    beat = 60.0 / spec.bpm
    bar_samples = int(round(beat * spec.beats_per_bar * sr))
    tail = int(3.0 * sr)
    total_bars = sum(s.bars for s in spec.sections)

    from .audio import StreamingLoudness
    meter = StreamingLoudness(sr)
    writer = StreamWriter(out_path, sr=sr, channels=2, subtype="PCM_24")
    carry = None
    bar_cursor = 0
    markers = []

    try:
        for idx, section in enumerate(spec.sections):
            events = build_events(spec, section, chords, motif, rng, root_pc, scale,
                                  bar_index=bar_cursor)
            chain.set_cutoff(section.cutoff)
            body, carry, n = render_section(spec, section, events, chords, sr, cache,
                                            kit, tail, carry)
            out = chain.process(body, n)
            if idx == 0:
                out[: min(len(out), int(1.5 * sr))] *= dsp.cosine_fade(
                    min(len(out), int(1.5 * sr)), True)
            out = limiter.process(out)
            meter.push(out)
            writer.write(out)
            markers.append({"section": section.name, "bar": bar_cursor,
                            "time": round(bar_cursor * bar_samples / sr, 2)})
            bar_cursor += section.bars
            if progress:
                log(f"    [{idx + 1}/{len(spec.sections)}] {section.name:<6} "
                    f"{section.bars:>3} bars  -> {fmt_time(writer.seconds)}")

        # let the reverb and the last notes ring out, then fade
        for _ in range(3):
            body = {k: np.zeros((bar_samples, 2), dtype=np.float32) for k in STEM_GAINS}
            if carry:
                for k, buf in carry.items():
                    if len(buf):
                        m = min(len(buf), bar_samples)
                        body[k][:m] += buf[:m]
                        carry[k] = buf[m:]
            out = limiter.process(chain.process(body, bar_samples))
            meter.push(out)
            writer.write(out)
        flush = limiter.flush()
        if len(flush):
            writer.write(flush)
    finally:
        writer.close()

    return {
        "path": str(out_path),
        "seconds": writer.seconds,
        "bars": total_bars,
        "sections": [s.name for s in spec.sections],
        "markers": markers,
        "progression": label,
        "key": key_name(root_pc, scale),
        "bpm": spec.bpm,
        "peak_dbfs": round(20 * math.log10(writer.peak + 1e-12), 2),
        "lufs": round(meter.value(), 2),
        "note_cache": cache.stats(),
    }
