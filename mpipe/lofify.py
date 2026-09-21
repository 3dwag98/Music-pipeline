"""Turn an existing song into a lofi version of itself.

The classic moves, in the order they actually belong:

  1. work out the tempo, key and downbeat
  2. deal with the vocal (proper separation if demucs is installed, the
     centre-channel trick if not)
  3. slow it down - pitch dropping with it, like a tape running slow
  4. run the lofi colour chain (lowpass, wobble, bit reduction, room)
  5. optionally lay a boom-bap kit under it, beat-locked to the new tempo
  6. sit it on a continuous vinyl bed
  7. master to the target loudness with a true-peak ceiling

Rights note: this rebuilds a recording you supply.  A lofi remix of someone
else's record is still their record - the CLI makes you say the audio is yours
before it will run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import dsp, effects
from .audio import integrated_lufs, load_audio, true_peak_db, write_audio
from .stretch import analyze_file
from .util import fmt_time, log, warn


@dataclass
class LofiSettings:
    """Everything that shapes the result.  Defaults are the familiar sound."""
    speed: float = 0.88             # 0.88 = the usual "slowed" amount
    keep_pitch: bool = False        # False = pitch falls with speed, like tape
    semitones: float = 0.0          # extra transposition on top
    vocals: str = "reduce"          # keep | reduce | remove
    vocal_amount: float = 0.8
    drums: str = "off"              # off | add
    drum_level: float = 0.5
    drum_style: str = "dusty"
    drum_pattern: str = "lazy"
    swing: float = 0.16
    amount: float = 0.6             # overall lofi character
    lowpass_hz: float = 0.0         # 0 = derive from `amount`
    bitcrush_bits: int = 0          # 0 = derive from `amount`
    vinyl: float = 0.8
    reverb: float = 0.3
    telephone: float = 0.0
    mp3_artifacts: float = 0.0
    wobble: bool = True
    lufs: float = -14.0
    peak_db: float = -1.0
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def _handle_vocals(audio, sr, settings, report):
    """Separate or attenuate the lead vocal.  Returns the reshaped audio."""
    mode = (settings.vocals or "keep").lower()
    if mode == "keep":
        report["vocals"] = "kept"
        return audio
    amount = float(np.clip(settings.vocal_amount, 0.0, 1.0))
    if mode == "remove":
        amount = 1.0

    stems = None
    if effects.have_demucs():
        log("    separating stems with demucs (this is the slow part)...")
        stems = effects.separate_stems(audio, sr)
    if stems and "vocals" in stems:
        rest = sum(v for k, v in stems.items() if k != "vocals")
        out = rest + stems["vocals"] * (1.0 - amount)
        report["vocals"] = f"{mode} via demucs ({amount:.0%})"
        report["stems"] = sorted(stems)
        return np.ascontiguousarray(out[: len(audio)], dtype=np.float32)

    out = effects.reduce_centre(audio, amount=amount, sr=sr)
    report["vocals"] = (f"{mode} via centre-channel reduction ({amount:.0%})"
                        + ("" if effects.have_demucs() else "; install demucs for real separation"))
    return out


def _drum_bed(length, sr, bpm, settings, downbeat=0.0):
    """A beat-locked boom-bap kit the same length as the track."""
    from .song import GlueBed
    bed = GlueBed(sr, bpm, vinyl=0.0, spine=1.0, seed=settings.seed,
                  spine_style=settings.drum_style,
                  spine_pattern=settings.drum_pattern, swing=settings.swing)
    offset = max(0, int(downbeat * sr))
    out = np.zeros((length, 2), dtype=np.float32)
    pos = offset
    block = 1 << 17
    while pos < length:
        take = min(block, length - pos)
        out[pos:pos + take] = bed.block(take)
        pos += take
    return out


def lofify_audio(audio, sr, settings: LofiSettings, info=None, progress=True):
    """Apply the whole treatment to an in-memory track.  Returns (audio, report)."""
    report = {"settings": settings.to_dict()}
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.repeat(audio[:, None], 2, axis=1)

    src_bpm = float((info or {}).get("bpm") or 0.0)
    downbeat = float((info or {}).get("downbeat") or 0.0)
    report["source_bpm"] = src_bpm or None
    report["source_key"] = (info or {}).get("key")

    audio = _handle_vocals(audio, sr, settings, report)

    speed = float(np.clip(settings.speed, 0.5, 1.5))
    semitones = float(settings.semitones)
    if not settings.keep_pitch:
        # a tape running slow drops pitch with speed; that is the sound people
        # mean by "slowed", so it is the default
        semitones += 12.0 * math.log2(speed)
    if abs(speed - 1.0) > 1e-3 or abs(semitones) > 1e-3:
        if progress:
            log(f"    {speed:.2f}x speed, {semitones:+.2f} semitones "
                f"({effects.backend_name()})")
        audio = effects.time_pitch(audio, sr, stretch=speed, semitones=semitones)
    report["speed"] = round(speed, 3)
    report["semitones"] = round(semitones, 2)
    new_bpm = (src_bpm * speed) if src_bpm else 0.0
    report["output_bpm"] = round(new_bpm, 2) if new_bpm else None

    chain = effects.lofi_chain(
        sr, amount=settings.amount,
        lowpass_hz=settings.lowpass_hz or None,
        bitcrush_bits=settings.bitcrush_bits or None,
        wobble=settings.wobble, room=settings.reverb,
        telephone=settings.telephone, mp3_artifacts=settings.mp3_artifacts)
    audio = chain(audio, reset=True)

    if settings.drums == "add" and new_bpm:
        if progress:
            log(f"    laying a {settings.drum_pattern} kit at {new_bpm:.1f} BPM")
        bed = _drum_bed(len(audio), sr, new_bpm, settings, downbeat / max(speed, 1e-6))
        ducked = dsp.Ducker(sr, depth_db=-2.5, attack_ms=5.0, release_ms=180.0)
        audio = ducked.process(audio, np.abs(bed).max(axis=1))
        audio = audio + bed * float(np.clip(settings.drum_level, 0.0, 2.0))
        report["drums"] = f"{settings.drum_pattern}/{settings.drum_style} at {new_bpm:.1f} BPM"
    elif settings.drums == "add":
        warn("cannot add drums: the tempo could not be detected")
        report["drums"] = "skipped (no tempo)"
    else:
        report["drums"] = "off"

    if settings.vinyl > 0:
        noise = dsp.VinylNoise(sr, level_db=-41.0 + 5.0 * min(settings.vinyl, 1.5),
                               crackle=settings.vinyl, seed=settings.seed)
        pos, block = 0, 1 << 17
        while pos < len(audio):
            take = min(block, len(audio) - pos)
            audio[pos:pos + take] += noise.block(take)
            pos += take

    measured = integrated_lufs(audio, sr)
    if np.isfinite(measured):
        audio = audio * (10 ** ((settings.lufs - measured) / 20.0))
        report["lufs_in"] = round(float(measured), 2)
    audio = effects.brickwall(audio, sr, ceiling_db=settings.peak_db, true_peak=True)
    after = integrated_lufs(audio, sr)
    drift = settings.lufs - after
    if np.isfinite(after) and abs(drift) > 0.4:
        audio = effects.brickwall(audio * (10 ** (drift / 20.0)), sr,
                                  ceiling_db=settings.peak_db, true_peak=True)
        after = integrated_lufs(audio, sr)

    report["lufs_out"] = round(float(after), 2) if np.isfinite(after) else None
    report["true_peak_db"] = round(true_peak_db(audio, sr), 2)
    report["seconds"] = round(len(audio) / sr, 2)
    report["backend"] = effects.backend_name()
    return audio, report


def lofify_file(src, dest, settings: LofiSettings, sr=44100, analysis=None,
                progress=True, mp3_quality=320):
    """Read a song, lofi it, write the result.  Returns a report dict."""
    src, dest = Path(src), Path(dest)
    info = analysis or analyze_file(src)
    if progress:
        log(f"    source: {info['bpm']:.1f} BPM, {info['key']}, "
            f"{fmt_time(info['duration'])}")
    audio, _ = load_audio(src, target_sr=sr)
    if len(audio) < sr:
        raise RuntimeError(f"{src.name} is too short to work with")
    audio, report = lofify_audio(audio, sr, settings, info=info, progress=progress)
    write_audio(dest, audio, sr, mp3_quality=mp3_quality)
    report["source"] = str(src)
    report["path"] = str(dest)
    report["source_duration"] = info["duration"]
    return report


#: Ready-made looks, so the common cases are one flag.
PRESETS = {
    "classic":  dict(speed=0.88, amount=0.6, vinyl=0.8, reverb=0.3, vocals="reduce"),
    "slowed":   dict(speed=0.82, amount=0.5, vinyl=0.6, reverb=0.55, vocals="keep"),
    "study":    dict(speed=0.90, amount=0.55, vinyl=0.7, reverb=0.25, vocals="remove",
                     drums="add", drum_pattern="lazy", drum_level=0.45),
    "sleep":    dict(speed=0.78, amount=0.75, vinyl=0.5, reverb=0.7, vocals="remove",
                     lowpass_hz=4200.0),
    "tape":     dict(speed=0.92, amount=0.85, vinyl=1.2, reverb=0.25, vocals="reduce",
                     bitcrush_bits=10, telephone=0.25),
    "instrumental": dict(speed=1.0, amount=0.45, vinyl=0.5, reverb=0.2, vocals="remove"),
}


def settings_from_preset(name, **overrides):
    base = PRESETS.get(str(name).lower())
    if base is None:
        raise KeyError(name)
    merged = dict(base)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return LofiSettings(**merged)
