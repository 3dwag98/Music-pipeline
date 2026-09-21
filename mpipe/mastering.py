"""Mastering: trim, loudness-normalise, limit - in memory or streamed.

`master_file` handles single tracks.  `normalise_stream` handles files that are
too long to hold in RAM (a 3-hour mix is ~3.8 GB as float32) by measuring in
one streaming pass and applying the gain in a second.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from . import dsp
from .audio import (StreamWriter, StreamingLoudness, integrated_lufs, load_audio,
                    true_peak_db, write_audio)
from .util import fmt_time, log, warn

#: YouTube plays back at roughly -14 LUFS and asks for true peaks under -1 dBTP.
YOUTUBE_LUFS = -14.0
YOUTUBE_PEAK_DB = -1.0


def trim_silence(data, sr, threshold_db=-50.0, pad_s=0.1):
    frame = 2048
    n = len(data) // frame
    if n == 0:
        return data
    rms = np.sqrt((data[: n * frame].reshape(n, frame, -1) ** 2).mean(axis=(1, 2)) + 1e-12)
    loud = np.where(20 * np.log10(rms) > threshold_db)[0]
    if len(loud) == 0:
        return data[:0]
    pad = int(pad_s * sr)
    start = max(0, loud[0] * frame - pad)
    end = min(len(data), (loud[-1] + 1) * frame + pad)
    return data[start:end]


def apply_fades(data, sr, fade_in, fade_out):
    for secs, at_start in ((fade_in, True), (fade_out, False)):
        n = min(len(data), int(secs * sr))
        if n <= 1:
            continue
        curve = dsp.cosine_fade(n, at_start)
        if at_start:
            data[:n] *= curve
        else:
            data[-n:] *= curve
    return data


def master_file(src, dest, lufs=YOUTUBE_LUFS, peak_db=YOUTUBE_PEAK_DB,
                fade_in=0.05, fade_out=0.05, trim=True, min_seconds=10.0,
                silence_floor=-45.0, mp3_quality=320):
    """Master one track.  Returns a report dict, or None if the track is rejected."""
    data, sr = load_audio(src)
    if trim:
        data = trim_silence(data, sr)
    if len(data) < min_seconds * sr:
        return {"skipped": f"under {min_seconds:.0f}s after trimming silence"}
    before = integrated_lufs(data, sr)
    if not np.isfinite(before) or before < silence_floor:
        return {"skipped": f"almost silent ({before:.1f} LUFS) - likely a failed generation"}

    data = data * (10 ** ((lufs - before) / 20.0))
    data = apply_fades(data, sr, fade_in, fade_out)
    # whole track in memory, so use the one-shot true-peak brickwall
    from . import effects
    data = effects.brickwall(data, sr, ceiling_db=peak_db, true_peak=True)
    after = integrated_lufs(data, sr)
    # one correction pass: limiting a loud master pulls the level down a little
    drift = lufs - after
    if np.isfinite(after) and abs(drift) > 0.4:
        data = data * (10 ** (drift / 20.0))
        np.clip(data, -(10 ** (peak_db / 20.0)), 10 ** (peak_db / 20.0), out=data)
        after = integrated_lufs(data, sr)

    write_audio(dest, data, sr, subtype="PCM_24", mp3_quality=mp3_quality)
    return {"skipped": None, "lufs_in": round(float(before), 1),
            "lufs_out": round(float(after), 1), "seconds": round(len(data) / sr, 2),
            "true_peak_db": round(true_peak_db(data, sr), 2), "samplerate": sr}


def measure_stream(path, block=1 << 18):
    """Streaming LUFS + true peak + duration for a file of any length."""
    with sf.SoundFile(str(path)) as fh:
        sr = fh.samplerate
        meter = StreamingLoudness(sr)
        peak = 0.0
        tp = -120.0
        frames = 0
        while True:
            data = fh.read(block, dtype="float32", always_2d=True)
            if len(data) == 0:
                break
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            meter.push(data)
            peak = max(peak, float(np.abs(data).max()))
            if frames < sr * 900:            # oversample the first 15 min for dBTP
                tp = max(tp, true_peak_db(data, sr))
            frames += len(data)
        return {"lufs": meter.value(), "peak_db": 20 * math.log10(peak + 1e-12),
                "true_peak_db": tp, "seconds": frames / sr, "samplerate": sr,
                "frames": frames}


def normalise_stream(src, dest, lufs=YOUTUBE_LUFS, peak_db=YOUTUBE_PEAK_DB,
                     fade_out=0.0, block=1 << 18, subtype="PCM_24", report=None,
                     mp3_quality=320):
    """Two-pass loudness normalisation that never loads the whole file.

    Pass 1 measures, pass 2 applies a single static gain through a look-ahead
    limiter.  Works the same for a 2-minute track and a 6-hour mix.
    """
    src, dest = Path(src), Path(dest)
    stats = report or measure_stream(src)
    sr = stats["samplerate"]
    if not np.isfinite(stats["lufs"]):
        warn(f"{src.name}: cannot measure loudness (silent?), copying unchanged")
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        return stats

    gain = 10 ** ((lufs - stats["lufs"]) / 20.0)
    limiter = dsp.Limiter(sr, ceiling_db=peak_db, lookahead_ms=6.0, release_ms=140.0,
                          true_peak=True)
    meter = StreamingLoudness(sr)
    total = stats["frames"]
    fade_n = int(max(0.0, fade_out) * sr)
    fade_start = max(0, total - fade_n)

    writer = StreamWriter(dest, sr=sr, channels=2, subtype=subtype,
                          mp3_quality=mp3_quality)
    pos = 0
    try:
        with sf.SoundFile(str(src)) as fh:
            while True:
                data = fh.read(block, dtype="float32", always_2d=True)
                if len(data) == 0:
                    break
                if data.shape[1] == 1:
                    data = np.repeat(data, 2, axis=1)
                data = data * gain
                if fade_n and pos + len(data) > fade_start:
                    idx = np.arange(pos, pos + len(data), dtype=np.float64)
                    curve = np.clip((total - idx) / max(1, fade_n), 0.0, 1.0)
                    data = data * (curve ** 1.5).astype(np.float32)[:, None]
                out = limiter.process(data)
                meter.push(out)
                writer.write(out)
                pos += len(data)
            tail = limiter.flush()
            if len(tail):
                meter.push(tail)
                writer.write(tail)
    finally:
        writer.close()

    return {"lufs_in": round(stats["lufs"], 2), "lufs_out": round(meter.value(), 2),
            "gain_db": round(20 * math.log10(gain + 1e-12), 2),
            "peak_dbfs": round(20 * math.log10(writer.peak + 1e-12), 2),
            "seconds": writer.seconds, "samplerate": sr}
