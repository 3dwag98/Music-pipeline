#!/usr/bin/env python3
"""Smoke + correctness tests.  Run:  python tests/test_pipeline.py

Fast enough to run before every commit (about a minute).  Covers the things
that are easy to break silently: music theory staying in key, time-stretch and
pitch-shift accuracy, limiter ceilings, loudness targeting, fingerprint
discrimination, and a full generate -> song -> check round trip.
"""

import argparse
import itertools
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

FAILED = []
PASSED = 0


def check(name, condition, detail=""):
    global PASSED
    if condition:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED.append(f"{name}: {detail}")
        print(f"  FAIL {name}  {detail}")


def near(a, b, tol):
    return abs(a - b) <= tol


# ------------------------------------------------------------------ theory ---

def test_theory():
    print("\ntheory")
    from mpipe.theory import (make_progression, parse_key, scale_degrees,
                              midi_to_hz, diatonic_triad)
    violations = 0
    for key in ("A Minor", "C Major", "F Dorian", "G Mixolydian", "C Lydian",
                "E Phrygian", "Bb Major", "F# Minor"):
        pc, scale = parse_key(key)
        allowed = set(scale_degrees(pc, scale)) | {(pc + 11) % 12}
        for seed in range(25):
            _, chords = make_progression(pc, scale, random.Random(seed), 4)
            for chord in chords:
                if chord.get("borrowed"):
                    continue
                for note in chord["notes"]:
                    if note % 12 not in allowed:
                        violations += 1
    check("chords stay in key (8 keys x 25 seeds)", violations == 0,
          f"{violations} out-of-key notes")
    check("Dorian keeps its major IV", diatonic_triad(3, "dorian") == "maj")
    check("Mixolydian keeps its minor v", diatonic_triad(4, "mixolydian") == "min")
    check("Lydian keeps its diminished IV", diatonic_triad(3, "lydian") == "dim")
    check("A4 = 440 Hz", near(midi_to_hz(69), 440.0, 1e-6))


# ------------------------------------------------------------------- synth ---

def test_synth():
    print("\nsynthesis")
    from mpipe.synth import INSTRUMENTS, NoteCache
    sr = 44100
    for name, fn in INSTRUMENTS.items():
        sig = fn(60, 1.0, sr, 0.8, np.random.default_rng(0))
        finite = np.isfinite(sig).all()
        check(f"{name} renders clean", finite and 0.0 < float(np.abs(sig).max()) <= 1.0,
              f"peak={float(np.abs(sig).max()):.3f} finite={finite}")
    cache = NoteCache(sr)
    for i in range(500):
        cache.render("rhodes", 60 + (i % 12), 1.0, 0.8)
    check("note cache hits", cache.stats()["hit_rate"] > 0.9, str(cache.stats()))


def test_drums():
    print("\ndrums")
    from mpipe.drums import PATTERNS, DrumKit
    kit = DrumKit(44100, "dusty", 0)
    for piece in ("kick", "snare", "rim", "hat", "hat_open", "shaker", "ride", "tom", "tabla"):
        sig = kit.hit(piece, 1.0)
        check(f"{piece} renders", len(sig) > 100 and np.isfinite(sig).all()
              and float(np.abs(sig).max()) > 0.01)
    kick = kit.hit("kick", 1.0)
    w = int(0.05 * 44100)
    rms = np.array([np.sqrt((kick[i * w:(i + 1) * w] ** 2).mean()) for i in range(8)])
    drop = 20 * math.log10(rms[-1] / (rms[0] + 1e-12) + 1e-12)
    check("kick actually decays", drop < -8.0, f"only {drop:.1f} dB over 400 ms")
    for name, pattern in PATTERNS.items():
        ok = all(len(grid) == 16 for grid in pattern.values())
        check(f"pattern '{name}' is a 16-step bar", ok)


# --------------------------------------------------------------------- dsp ---

def test_dsp():
    print("\ndsp")
    from mpipe import dsp
    sr = 44100
    loud = (np.random.default_rng(1).standard_normal((sr * 3, 2)) * 0.9).astype("float32")
    for ceiling in (-1.0, -0.3, -3.0):
        out = dsp.Limiter(sr, ceiling_db=ceiling).process(loud)
        peak_db = 20 * math.log10(float(np.abs(out).max()) + 1e-12)
        check(f"limiter holds {ceiling} dBFS", peak_db <= ceiling + 0.01,
              f"got {peak_db:.3f}")
    # Below the threshold the limiter must pass the signal untouched apart from
    # its look-ahead delay, which a look-ahead design necessarily introduces.
    quiet = (np.random.default_rng(2).standard_normal((sr, 2)) * 0.02).astype("float32")
    lim = dsp.Limiter(sr, -1.0)
    out = lim.process(quiet)
    look = lim.look
    check("limiter is transparent below threshold",
          float(np.abs(out[look:] - quiet[:len(quiet) - look]).max()) < 1e-6,
          f"max diff {float(np.abs(out[look:] - quiet[:len(quiet) - look]).max()):.2e}")
    check("limiter loses no samples (process + flush)",
          len(out) + len(lim.flush()) == len(quiet) + look)
    # a stateful chain must give the same answer block-by-block as in one go
    sig = (np.random.default_rng(3).standard_normal((sr, 2)) * 0.2).astype("float32")
    whole = dsp.Biquad("lowpass", sr, 1200.0).process(sig)
    stream = dsp.Biquad("lowpass", sr, 1200.0)
    chunks = np.concatenate([stream.process(sig[i:i + 997]) for i in range(0, len(sig), 997)])
    check("biquad state survives block boundaries",
          float(np.abs(whole - chunks).max()) < 1e-5,
          f"max diff {float(np.abs(whole - chunks).max()):.2e}")
    rev = dsp.FDNReverb(sr)
    tail = rev.process(np.zeros((sr, 2), dtype="float32"))
    rev2 = dsp.FDNReverb(sr)
    rev2.process(sig)
    tail2 = rev2.process(np.zeros((sr, 2), dtype="float32"))
    check("reverb tail carries across blocks",
          float(np.abs(tail2).max()) > float(np.abs(tail).max()))


# ------------------------------------------------------------------ stretch ---

def test_stretch():
    print("\ntime-stretch and pitch-shift")
    from mpipe.stretch import pitch_shift, time_stretch
    sr = 44100
    t = np.arange(sr * 4) / sr
    tone = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype("float32")

    def f0(x):
        mono = x.mean(axis=1)
        spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        return float(np.fft.rfftfreq(len(mono), 1 / sr)[int(np.argmax(spec))])

    for semis in (-5, -2, 3, 7, 12):
        out = pitch_shift(tone, semis, sr)
        cents = 1200 * math.log2(f0(out) / (440 * 2 ** (semis / 12)))
        check(f"pitch shift {semis:+d} st accurate", abs(cents) < 15,
              f"{cents:+.1f} cents off")
        check(f"pitch shift {semis:+d} st keeps length", len(out) == len(tone),
              f"{len(out)} vs {len(tone)}")
    for rate in (0.85, 1.15, 1.4):
        out = time_stretch(tone, rate)
        expected = len(tone) / rate
        check(f"time stretch rate {rate} length", abs(len(out) - expected) / expected < 0.02,
              f"{len(out)} vs {expected:.0f}")


# ---------------------------------------------------------------- mastering ---

def test_loudness():
    print("\nloudness")
    from mpipe.audio import StreamingLoudness, integrated_lufs
    sr = 44100
    t = np.arange(sr * 6) / sr
    sig = np.stack([0.4 * np.sin(2 * np.pi * 997 * t)] * 2, axis=1).astype("float32")
    offline = integrated_lufs(sig, sr)
    meter = StreamingLoudness(sr)
    for i in range(0, len(sig), 4096):
        meter.push(sig[i:i + 4096])
    check("streaming LUFS matches pyloudnorm", abs(meter.value() - offline) < 0.3,
          f"{meter.value():.2f} vs {offline:.2f}")


# -------------------------------------------------------------- fingerprint ---

def test_fingerprint(tmp):
    print("\nfingerprinting")
    from mpipe.engine import SongSpec, render_song
    from mpipe.fingerprint import (Ledger, feature_vector, fingerprint_file, similarity)
    files = []
    for seed, key, bpm in ((11, "A Minor", 74), (12, "C Major", 82), (13, "D Minor", 78)):
        path = tmp / f"fp{seed}.wav"
        render_song(SongSpec(seed=seed, bpm=bpm, key=key), path, minutes=0.6, progress=False)
        files.append(path)
    prints = [fingerprint_file(f)[0] for f in files]
    check("identical file scores ~1.0", similarity(prints[0], prints[0]) > 0.85,
          f"{similarity(prints[0], prints[0])}")
    worst = max(similarity(a, b) for a, b in itertools.combinations(prints, 2))
    check("unrelated tracks score low", worst < 0.35, f"worst pair {worst}")
    ledger = Ledger(tmp / "ledger.json")
    ledger.add(files[0], prints[0], 40.0, features=feature_vector(files[0]))
    found = ledger.check(prints[0], features=feature_vector(files[0]))
    check("ledger catches a re-export", len(found) > 0)
    clean = ledger.check(prints[1], features=feature_vector(files[1]))
    check("ledger ignores a different track", len(clean) == 0, str(clean))


# --------------------------------------------------------- end-to-end round ---

def test_end_to_end(tmp):
    print("\nend to end")
    from mpipe.engine import SongSpec, render_song
    from mpipe.mastering import measure_stream
    from mpipe.song import build_song, write_tracklist
    from mpipe.stretch import detect_bpm
    import soundfile as sf

    lib = tmp / "lib"
    lib.mkdir(exist_ok=True)
    for seed, key, bpm in ((1, "A Minor", 72), (2, "C Major", 84), (3, "F Dorian", 76)):
        render_song(SongSpec(seed=seed, bpm=bpm, key=key), lib / f"{seed:02d}.wav",
                    minutes=0.7, progress=False)
    files = sorted(lib.glob("*.wav"))
    check("engine produced tracks", len(files) == 3)

    out = tmp / "song.wav"
    report = build_song(files, out, minutes=4.0, spine=0.5, vinyl=0.8, seed=1,
                        progress=False, cache_path=tmp / "analysis.json")
    check("song reached the target length", report["seconds"] >= 4 * 60 * 0.9,
          f"{report['seconds']:.0f}s")
    check("song peak under ceiling", report["peak_dbfs"] <= -0.99,
          f"{report['peak_dbfs']}")
    stats = measure_stream(out)
    check("song loudness near -14 LUFS", abs(stats["lufs"] + 14.0) < 1.5,
          f"{stats['lufs']:.2f}")

    data, sr = sf.read(str(out), dtype="float32")
    mono = data.mean(axis=1)
    tempos = []
    for i in range(0, len(mono) // sr - 45, 45):
        bpm, _ = detect_bpm(mono[i * sr:(i + 45) * sr], sr)
        tempos.append(bpm)
    spread = max(tempos) - min(tempos) if tempos else 0
    check("tempo stays locked across transitions", spread < 3.0,
          f"windows: {tempos}")

    lines = write_tracklist(tmp / "tl.txt", report["chapters"], report["seconds"])
    check("tracklist starts at 00:00", lines[0].startswith("00:00"), lines[0])
    check("tracklist has 3+ chapters", len(lines) >= 3, f"{len(lines)}")


def test_comfy_workflows():
    print("\ncomfyui workflows")
    import json as _json
    from mpipe.comfy import (apply_settings, collect_outputs, describe_workflow,
                             find_nodes, load_workflow, set_input, text_field_of)

    for name in ("art", "acestep"):
        graph = load_workflow(name)
        check(f"{name}.json loads", len(graph) > 3)
        check(f"{name}.json is API format",
              all("class_type" in node for node in graph.values()))
        # every wired input must point at a node that exists
        dangling = []
        for nid, node in graph.items():
            for field, value in (node.get("inputs") or {}).items():
                if isinstance(value, list) and len(value) == 2 and value[0] not in graph:
                    dangling.append(f"{nid}.{field}->{value[0]}")
        check(f"{name}.json has no dangling links", not dangling, str(dangling))
        check(f"{name}.json describes cleanly", len(describe_workflow(graph)) == len(graph))

    art = load_workflow("art")
    apply_settings(art, prompt="P", negative="N", seed=5, steps=9,
                   width=640, height=360)
    check("art prompt patched", art["6"]["inputs"]["text"] == "P")
    check("art negative patched", art["7"]["inputs"]["text"] == "N")
    check("art seed/steps patched",
          art["3"]["inputs"]["seed"] == 5 and art["3"]["inputs"]["steps"] == 9)
    check("art size patched",
          art["5"]["inputs"]["width"] == 640 and art["5"]["inputs"]["height"] == 360)

    ace = load_workflow("acestep")
    apply_settings(ace, prompt="TAGS", negative="NEG", lyrics="LYR", seconds=42)
    pos = next(n for n in ace.values() if (n.get("_meta") or {}).get("title") == "POSITIVE")
    neg = next(n for n in ace.values() if (n.get("_meta") or {}).get("title") == "NEGATIVE")
    # the audio encoder's prompt lives in `tags`, not `text` - patching the wrong
    # field is silent, so assert the right one moved and no bogus key appeared
    check("ace prompt goes to `tags`", pos["inputs"]["tags"] == "TAGS")
    check("ace invents no `text` field", "text" not in pos["inputs"])
    check("ace lyrics only on positive",
          pos["inputs"]["lyrics"] == "LYR" and neg["inputs"]["lyrics"] != "LYR")
    check("ace seconds patched", ace["44"]["inputs"]["seconds"] == 42.0)

    graph = load_workflow("art")
    before = _json.dumps(graph["3"]["inputs"]["model"])
    set_input(graph, "SAMPLER", "model", "clobbered")
    check("wired inputs are never overwritten",
          _json.dumps(graph["3"]["inputs"]["model"]) == before)
    check("unknown field is not created",
          set_input(graph, "SAMPLER", "not_a_real_field", 1) == 0
          and "not_a_real_field" not in graph["3"]["inputs"])
    check("selector matches by title", find_nodes(graph, "SAMPLER") == ["3"])
    check("selector matches by class", find_nodes(graph, "KSampler") == ["3"])
    check("selector matches by id", find_nodes(graph, "3") == ["3"])
    check("selector is case-insensitive", find_nodes(graph, "sampler") == ["3"])
    check("text field of audio node is tags",
          text_field_of({"class_type": "TextEncodeAceStepAudio",
                         "inputs": {"tags": "", "lyrics": ""}}) == "tags")

    outputs = {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
               "59": {"audio": [{"filename": "b.flac", "subfolder": "", "type": "output"}]}}
    check("collects images", [i["filename"] for i in collect_outputs(outputs, "image")] == ["a.png"])
    check("collects audio", [i["filename"] for i in collect_outputs(outputs, "audio")] == ["b.flac"])
    check("collects everything", len(collect_outputs(outputs)) == 2)


def test_comfy_roundtrip(tmp):
    """Drive the mock ComfyUI server exactly as the real one is driven."""
    print("\ncomfyui round trip")
    import socket
    import subprocess
    import time as _time
    from mpipe.comfy import ComfyClient, apply_settings, load_workflow

    port = 8199
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
    env_py = sys.executable
    proc = subprocess.Popen(
        [env_py, "-c",
         f"import runpy,sys;sys.argv=['mock'];"
         f"import importlib.util;"
         f"spec=importlib.util.spec_from_file_location('m', r'{ROOT}/mock_comfy.py');"
         f"m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
         f"from http.server import ThreadingHTTPServer;"
         f"ThreadingHTTPServer(('127.0.0.1', {port}), m.Handler).serve_forever()"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        client = ComfyClient(f"http://127.0.0.1:{port}")
        for _ in range(60):
            if client.health(fatal=False):
                break
            _time.sleep(0.25)
        check("mock server reachable", client.health(fatal=False) is not None)
        check("checkpoints listed", "v1-5-pruned-emaonly.safetensors" in client.checkpoints())

        graph = load_workflow("art")
        apply_settings(graph, prompt="test", seed=3, width=128, height=96)
        paths = client.run(graph, tmp / "art", want="image", timeout=60)
        check("image downloaded", len(paths) == 1 and paths[0].exists()
              and paths[0].stat().st_size > 100, str(paths))
        check("image kept its extension", paths[0].suffix == ".png", paths[0].name)

        graph = load_workflow("acestep")
        apply_settings(graph, prompt="lofi", seed=3, seconds=3)
        paths = client.run(graph, tmp / "track", want="audio", timeout=60)
        check("audio downloaded", len(paths) == 1 and paths[0].exists(), str(paths))
        import soundfile as sf
        info = sf.info(str(paths[0]))
        check("audio is readable", info.frames > 0 and info.channels == 2,
              f"{info.frames} frames")

        try:
            client.submit({"1": {"inputs": {}}})
            check("bad workflow is rejected", False, "no error raised")
        except RuntimeError as exc:
            check("bad workflow is rejected", "class_type" in str(exc), str(exc)[:80])
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_video_loop(tmp):
    print("\nvideo loop")
    import subprocess
    from mpipe.util import ffmpeg_ok
    if not ffmpeg_ok():
        print("  skip (no ffmpeg)")
        return
    from mpipe.video import make_loop

    image = tmp / "src.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc2=size=320x180:duration=1:rate=1",
                    "-frames:v", "1", str(image)], check=True)
    width, height, fps, seconds = 320, 180, 24, 3
    out = make_loop(image, tmp / "loop.mp4", seconds=seconds, size=f"{width}x{height}",
                    fps=fps, zoom=0.12)
    check("loop file written", out.exists() and out.stat().st_size > 1000)

    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    frame_bytes = width * height * 3
    count = len(raw) // frame_bytes
    check("loop has the right frame count", abs(count - seconds * fps) <= 2,
          f"{count} vs {seconds * fps}")
    frames = np.frombuffer(raw[:count * frame_bytes], dtype=np.uint8)
    frames = frames.reshape(count, height, width, 3).astype(np.float32)
    seam = float(np.abs(frames[0] - frames[-1]).mean())
    step = float(np.abs(frames[0] - frames[1]).mean())
    mid = float(np.abs(frames[0] - frames[count // 2]).mean())
    # the wrap-around must cost no more than a couple of ordinary frame steps,
    # or the loop visibly jumps every time it repeats
    check("loop seam is invisible", seam <= max(step * 3.0, 1.0),
          f"seam {seam:.2f} vs step {step:.2f}")
    check("loop actually moves", mid > step * 3, f"mid {mid:.2f} vs step {step:.2f}")


def test_formats(tmp):
    print("\nmp3 / formats")
    from mpipe.audio import StreamWriter, audio_meta, load_audio, with_format, write_audio
    sr = 44100
    t = np.arange(sr * 3) / sr
    sig = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype("float32")
    for ext in (".wav", ".flac", ".mp3"):
        path = write_audio(tmp / f"fmt{ext}", sig, sr)
        check(f"{ext} written", path.exists() and path.stat().st_size > 500)
        back, got = load_audio(path)
        check(f"{ext} reads back", got == sr and abs(len(back) - len(sig)) < sr * 0.1,
              f"{len(back)} vs {len(sig)}")
        meta = audio_meta(path)
        check(f"{ext} metadata", meta and abs(meta["duration"] - 3.0) < 0.15,
              str(meta))
    with StreamWriter(tmp / "streamed.mp3", sr=sr) as writer:
        for i in range(0, len(sig), 4096):
            writer.write(sig[i:i + 4096])
    check("mp3 streams block by block", writer.frames == len(sig),
          f"{writer.frames} vs {len(sig)}")
    check("with_format swaps the extension",
          with_format("a/b/song.wav", "mp3").name == "song.mp3")


def test_tags(tmp):
    print("\ntags")
    from mpipe.audio import write_audio
    from mpipe.tags import have_mutagen, read_tags, write_tags
    if not have_mutagen():
        print("  skip (no mutagen)")
        return
    sr = 44100
    t = np.arange(sr * 2) / sr
    path = write_audio(tmp / "tagged.mp3",
                       np.stack([0.2 * np.sin(2 * np.pi * 330 * t)] * 2,
                                axis=1).astype("float32"), sr)
    ok = write_tags(path, title="Amber Streetlights", artist="Me", album="Vol 1",
                    year=2026, track=3, bpm=78.4, key="Am")
    check("mp3 tagged", ok)
    tags = read_tags(path)
    check("title round trips", tags.get("title") == "Amber Streetlights", str(tags))
    check("bpm round trips", tags.get("bpm") == "78", str(tags))
    check("genre defaults to lofi", tags.get("genre") == "Lofi Hip Hop", str(tags))


def test_true_peak():
    print("\ntrue-peak limiting")
    from mpipe import dsp
    from mpipe.audio import true_peak_db
    sr = 44100
    rng = np.random.default_rng(0)
    t = np.arange(sr * 3) / sr
    cases = {
        "noise": (rng.standard_normal((sr * 3, 2)) * 1.5).astype("float32"),
        "square": np.stack([np.sign(np.sin(2 * np.pi * 440 * t))] * 2,
                           axis=1).astype("float32") * 1.2,
        "tone stack": np.stack([0.3 * np.sin(2 * np.pi * 220 * t)
                                + 0.2 * np.sin(2 * np.pi * 3000 * t)] * 2,
                               axis=1).astype("float32") * 5,
    }
    for name, sig in cases.items():
        limiter = dsp.Limiter(sr, ceiling_db=-1.0, true_peak=True)
        out = np.concatenate([limiter.process(sig[i:i + 7000])
                              for i in range(0, len(sig), 7000)] + [limiter.flush()])
        tp = true_peak_db(out, sr)
        # a sample-domain limiter reconstructs at over +2 dBTP on noise, which
        # then clips in any lossy encode - this is the check that catches it
        check(f"true peak held on {name}", tp <= -0.5, f"{tp:.2f} dBTP")
    from mpipe import effects
    out = effects.brickwall(cases["tone stack"], sr, ceiling_db=-1.0, true_peak=True)
    check("one-shot brickwall holds", true_peak_db(out, sr) <= -0.5,
          f"{true_peak_db(out, sr):.2f} dBTP")


def test_lofify(tmp):
    print("\nlofify")
    from mpipe.audio import load_audio, write_audio
    from mpipe.effects import backend_name
    from mpipe.engine import SongSpec, render_song
    from mpipe.lofify import PRESETS, lofify_file, settings_from_preset
    from mpipe.stretch import analyze_file

    src = tmp / "song_in.wav"
    render_song(SongSpec(seed=5, bpm=92, key="C Major"), src, minutes=0.7, progress=False)
    audio, sr = load_audio(src)
    # a dead-centre tone stands in for a lead vocal
    t = np.arange(len(audio)) / sr
    vox = (0.25 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    audio[:, 0] += vox
    audio[:, 1] += vox
    write_audio(src, audio, sr)

    def centre_level(path):
        data, rate = load_audio(path)
        mono = data.mean(axis=1)
        spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        freqs = np.fft.rfftfreq(len(mono), 1 / rate)
        i = int(np.argmin(np.abs(freqs - 440)))
        return float(spec[max(0, i - 4):i + 5].max() / (spec.max() + 1e-12))

    before = centre_level(src)
    info = analyze_file(src)
    check("source tempo detected", abs(info["bpm"] - 92) < 4, f"{info['bpm']}")

    settings = settings_from_preset("classic")
    report = lofify_file(src, tmp / "out.mp3", settings, analysis=info, progress=False)
    check("lofi output written", (tmp / "out.mp3").exists())
    check("slowed down", report["seconds"] > info["duration"] * 1.05,
          f"{report['seconds']} vs {info['duration']}")
    check("pitch follows speed", report["semitones"] < -1.0, str(report["semitones"]))
    check("output tempo reported",
          abs(report["output_bpm"] - info["bpm"] * settings.speed) < 0.5,
          str(report["output_bpm"]))
    check("loudness on target", abs(report["lufs_out"] + 14.0) < 0.6,
          str(report["lufs_out"]))
    check("true peak under ceiling", report["true_peak_db"] <= -0.9,
          str(report["true_peak_db"]))
    after = centre_level(tmp / "out.mp3")
    check("centre vocal attenuated", after < before * 0.1,
          f"{before:.3f} -> {after:.3f}")

    kept = settings_from_preset("classic", vocals="keep", speed=1.0)
    rep2 = lofify_file(src, tmp / "kept.mp3", kept, analysis=info, progress=False)
    check("--vocals keep leaves it alone",
          centre_level(tmp / "kept.mp3") > before * 0.2,
          f"{centre_level(tmp / 'kept.mp3'):.3f} vs {before:.3f}")
    check("speed 1.0 keeps the length",
          abs(rep2["seconds"] - info["duration"]) < info["duration"] * 0.08,
          f"{rep2['seconds']} vs {info['duration']}")

    for name in PRESETS:
        settings_from_preset(name)
    check(f"all {len(PRESETS)} presets build", True)
    print(f"  (effects backend: {backend_name()})")


def test_models_catalogue():
    print("\nmodel catalogue")
    from mpipe.models import BY_KEY, CATALOGUE, fits, report
    check("catalogue is populated", len(CATALOGUE) >= 5, str(len(CATALOGUE)))
    check("keys are unique", len(BY_KEY) == len(CATALOGUE))
    for m in CATALOGUE:
        check(f"{m.key} declares a commercial stance",
              m.commercial in ("yes", "no", "conditional", "check"), m.commercial)
        check(f"{m.key} has a licence", bool(m.licence))
        check(f"{m.key} has a source url or is local",
              bool(m.url) or m.key == "builtin")
    # the point of the whole module: MusicGen's weights are non-commercial even
    # though its code is MIT, and that is easy to miss
    check("MusicGen flagged non-commercial", BY_KEY["musicgen"].commercial == "no")
    check("ACE-Step flagged commercial-ok", BY_KEY["acestep"].commercial == "yes")
    check("built-in engine needs no VRAM", BY_KEY["builtin"].min_vram_gb == 0)
    check("YuE does not fit 6 GB", not fits(BY_KEY["yue"], 6.0))
    check("ACE-Step fits 6 GB with offload", fits(BY_KEY["acestep"], 6.0))
    check("ACE-Step does not fit 6 GB without offload",
          not fits(BY_KEY["acestep"], 6.0, cpu_offload=False))
    rows = report(vram_gb=6.0, commercial_only=True, verbose=False)
    check("commercial-only filter drops MusicGen",
          all(r["key"] != "musicgen" for r in rows), str([r["key"] for r in rows]))


def test_bad_inputs(tmp):
    """A folder of real music has junk in it.  None of it may stop a run."""
    print("\nawkward inputs")
    from mpipe.audio import write_audio
    from mpipe.lofify import lofify_file, settings_from_preset
    from mpipe.song import MIN_TRACK_SECONDS, build_mix, build_song
    from mpipe.util import collect_inputs

    folder = tmp / "messy"
    folder.mkdir(exist_ok=True)
    sr = 44100
    t = np.arange(sr * 8) / sr
    for i in range(3):
        write_audio(folder / f"good{i}.wav",
                    np.stack([0.3 * np.sin(2 * np.pi * (180 + 40 * i) * t)] * 2,
                             axis=1).astype("float32"), sr)
    (folder / "empty.wav").write_bytes(b"")
    (folder / "garbage.wav").write_bytes(b"definitely not audio" * 64)
    write_audio(folder / "tiny.wav",
                np.stack([0.3 * np.sin(2 * np.pi * 220 * np.arange(sr // 4) / sr)] * 2,
                         axis=1).astype("float32"), sr)
    write_audio(folder / "silent.wav", np.zeros((sr * 4, 2), dtype="float32"), sr)

    files = collect_inputs(folder)
    check("collect_inputs picks up the junk too", len(files) >= 7, str(len(files)))

    out = folder / "song.mp3"
    report = build_song(files, out, minutes=0.6, progress=False)
    check("build_song survives unreadable files", report["seconds"] > 10,
          f"{report['seconds']}s")
    check("build_song names what it skipped",
          {"empty.wav", "garbage.wav"} <= set(report["unreadable"]),
          str(report["unreadable"]))
    check("build_song skips sub-minimum tracks",
          "tiny.wav" in report["unreadable"], str(report["unreadable"]))

    mix_out = folder / "mix.mp3"
    mix = build_mix(files, mix_out, minutes=0.5, progress=False)
    check("build_mix survives unreadable files", mix["seconds"] > 5, f"{mix['seconds']}s")
    check("build_mix names what it skipped",
          {"empty.wav", "garbage.wav"} <= set(mix["unreadable"]),
          str(mix["unreadable"]))

    # the output now sits in the input folder; a second run must not eat it
    again = collect_inputs(folder)
    check("second run sees its own output", out.name in [f.name for f in again])
    report2 = build_song(again, out, minutes=0.6, progress=False)
    check("output file is excluded from its own inputs",
          report2["seconds"] > 10 and out.name not in report2["unreadable"],
          f"{report2['seconds']}s {report2['unreadable']}")

    try:
        build_song([folder / "empty.wav", folder / "garbage.wav"],
                   tmp / "nope.mp3", minutes=0.2, progress=False)
        check("all-unreadable input fails clearly", False, "no error raised")
    except RuntimeError as exc:
        check("all-unreadable input fails clearly", "could be read" in str(exc),
              str(exc)[:70])

    settings = settings_from_preset("classic")
    try:
        lofify_file(folder / "silent.wav", tmp / "sil.mp3", settings, progress=False)
        check("lofify refuses silence", False, "no error raised")
    except RuntimeError as exc:
        check("lofify refuses silence", "silent" in str(exc), str(exc)[:70])
    try:
        lofify_file(folder / "tiny.wav", tmp / "tny.mp3", settings, progress=False)
        check("lofify refuses a too-short file", False, "no error raised")
    except RuntimeError as exc:
        check("lofify refuses a too-short file", "short" in str(exc), str(exc)[:70])
    check("minimum track length is enforced", MIN_TRACK_SECONDS >= 1.0)


def test_odd_files(tmp):
    print("\nmono / unicode / odd rates")
    from mpipe.audio import load_audio, write_audio
    from mpipe.lofify import lofify_file, settings_from_preset
    from mpipe.stretch import analyze_file
    from mpipe.util import collect_inputs

    made = {}
    for name, rate, channels in (("mono.wav", 44100, 1),
                                 ("a file with spaces.wav", 44100, 2),
                                 ("trene_nihon_cafe.wav", 44100, 2),
                                 ("lowrate.wav", 8000, 2)):
        t = np.arange(int(6 * rate)) / rate
        sig = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
        data = np.stack([sig] * channels, axis=1) if channels > 1 else sig[:, None]
        made[name] = write_audio(tmp / name, data, rate)

    for name, path in made.items():
        data, got = load_audio(path)
        check(f"loads {name}", data.ndim == 2 and data.shape[1] == 2 and got > 0,
              f"{data.shape} @{got}")
        info = analyze_file(path)
        check(f"analyses {name}", info["duration"] > 0 and info["bpm"] > 0, str(info))

    settings = settings_from_preset("classic")
    report = lofify_file(made["mono.wav"], tmp / "mono_lofi.mp3", settings, progress=False)
    check("lofifies a mono source", report["seconds"] > 0 and (tmp / "mono_lofi.mp3").exists())
    report = lofify_file(made["lowrate.wav"], tmp / "low_lofi.mp3", settings, progress=False)
    check("lofifies an 8 kHz source", report["seconds"] > 0)

    listing = tmp / "list.txt"
    listing.write_text(f"{made['a file with spaces.wav']}\n# a comment\n\n"
                       f"does_not_exist.wav\n{made['mono.wav']}\n", encoding="utf-8")
    picked = collect_inputs(listing)
    check("playlist skips comments, blanks and missing files", len(picked) == 2,
          str([p.name for p in picked]))


def test_effects_reset():
    """Both backends must honour `reset`, or state leaks between tracks."""
    print("\neffects reset")
    from mpipe import effects
    sr = 44100
    rng = np.random.default_rng(0)
    sig = (rng.standard_normal((sr, 2)) * 0.2).astype("float32")
    for forced in (True, False):
        if forced and not effects.HAVE_PEDALBOARD:
            continue
        chain = (effects.lofi_chain(sr, 0.6) if forced
                 else effects._fallback_chain(sr, 0.6, 9000.0, 12, True, 0.25))
        label = "pedalboard" if forced else "fallback"
        first = chain(sig, reset=True)
        chain(sig, reset=False)
        again = chain(sig, reset=True)
        check(f"{label} chain resets to the same state",
              float(np.abs(first - again).max()) < 1e-4,
              f"max diff {float(np.abs(first - again).max()):.2e}")


def test_check_warnings(tmp):
    """`check` has to surface the policy line that lands on lofify output."""
    print("\ncheck warnings")
    import io
    import json as _json
    from contextlib import redirect_stdout
    import pipeline
    from mpipe.audio import write_audio

    run = tmp / "runs" / "20260101-000000_lofify-classic"
    (run / "raw").mkdir(parents=True, exist_ok=True)
    sr = 44100
    t = np.arange(sr * 4) / sr
    track = write_audio(run / "raw" / "01_x_lofi.mp3",
                        np.stack([0.25 * np.sin(2 * np.pi * 220 * t)] * 2,
                                 axis=1).astype("float32"), sr)
    (run / "manifest.json").write_text(_json.dumps({
        "engine": "lofify", "preset": "classic", "rights_attested": True,
        "tracks": [{"index": 1, "file": track.name, "status": "ok",
                    "source": "/somewhere/original.mp3"}]}), encoding="utf-8")

    args = argparse.Namespace(file=str(track), run=None, log=False,
                              ledger=str(tmp / "ledger.json"), quiet=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        pipeline.cmd_check(args)
    out = buf.getvalue()
    check("check names lofify as the source", "YOUR OWN recordings" in out)
    check("check names the source file", "original.mp3" in out, out[-300:])
    check("check warns about the inauthentic-content policy",
          "inauthentic content" in out, out[-400:])
    check("check points at the monetisation notes", "MONETIZATION.md" in out)
    check("long findings are wrapped",
          all(len(line) <= 78 for line in out.splitlines()),
          max(out.splitlines(), key=len)[:90])

    builtin = tmp / "runs" / "20260101-000001_lofi-lofi"
    (builtin / "raw").mkdir(parents=True, exist_ok=True)
    song = write_audio(builtin / "song.mp3",
                       np.stack([0.25 * np.sin(2 * np.pi * 220 * t)] * 2,
                                axis=1).astype("float32"), sr)
    (builtin / "manifest.json").write_text(_json.dumps({
        "engine": "builtin", "tracks": []}), encoding="utf-8")
    args = argparse.Namespace(file=str(song), run=None, log=False,
                              ledger=str(tmp / "ledger2.json"), quiet=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        pipeline.cmd_check(args)
    out = buf.getvalue()
    check("built-in output is not given the lofify warning",
          "inauthentic content" not in out, out[-300:])
    check("built-in output states it is synthesised",
          "synthesised from scratch" in out, out[-300:])


def test_hfaudio_logic():
    """Everything about the HF backend that does not need the weights."""
    print("\nhf backend (logic)")
    from mpipe import hfaudio
    from mpipe.hfaudio import (DEFAULT_MODEL, GenSettings, MODELS,
                               MUSICGEN_FRAME_RATE, estimate_minutes,
                               fp16_is_suspect, lofi_prompt)

    check("default is the smallest model", DEFAULT_MODEL == "musicgen-small")
    check("catalogue is populated", len(MODELS) >= 2)
    for key, spec in MODELS.items():
        check(f"{key} declares a licence", bool(spec.licence))
        check(f"{key} declares commercial status", isinstance(spec.commercial, bool))
        check(f"{key} has a repo id", "/" in spec.repo)
    # the licence is the thing people get wrong, so assert it rather than trust it
    check("musicgen is flagged non-commercial",
          MODELS["musicgen-small"].commercial is False)
    check("musicgen licence names CC-BY-NC",
          "CC-BY-NC" in MODELS["musicgen-small"].licence)

    for name in ("NVIDIA GeForce GTX 1660 Ti", "GTX 1650", "NVIDIA T600"):
        check(f"fp16 flagged suspect on {name}", fp16_is_suspect(name))
    for name in ("NVIDIA GeForce RTX 3060", "NVIDIA A100-SXM4-40GB", "RTX 4090"):
        check(f"fp16 fine on {name}", not fp16_is_suspect(name))

    if hfaudio.torch_available():
        import torch
        original = hfaudio.gpu_name
        try:
            hfaudio.gpu_name = lambda: "NVIDIA GeForce GTX 1660 Ti"
            dtype, label, why = hfaudio.resolve_dtype("auto")
            check("auto picks fp32 on a 1660 Ti",
                  dtype is torch.float32 and label == "fp32", f"{label}: {why}")
            check("auto explains itself", "16-series" in why, why)
            dtype, label, why = hfaudio.resolve_dtype("fp16")
            check("fp16 is still honoured when forced",
                  dtype is torch.float16 and label == "fp16", label)
            check("forced fp16 warns about the card", "selftest" in why, why)
            hfaudio.gpu_name = lambda: "NVIDIA GeForce RTX 4090"
            dtype, label, _ = hfaudio.resolve_dtype("auto")
            check("auto picks fp16 on a healthy card",
                  dtype is torch.float16 and label == "fp16", label)
            dtype, label, _ = hfaudio.resolve_dtype("auto", device="cpu")
            check("cpu always gets fp32", label == "fp32", label)
        finally:
            hfaudio.gpu_name = original
    else:
        print("  skip (no torch) - dtype resolution")

    prompt = lofi_prompt(82, extra="rain", mood="sleepy")
    check("prompt names the genre", "lofi hip hop" in prompt, prompt)
    check("prompt carries the tempo", "82 bpm" in prompt, prompt)
    check("prompt carries mood and extras",
          "sleepy" in prompt and "rain" in prompt, prompt)

    settings = GenSettings(seconds=12.5)
    check("settings round-trip", settings.to_dict()["seconds"] == 12.5)
    check("frame rate is the codec's", MUSICGEN_FRAME_RATE == 50)
    cpu = estimate_minutes(60, "cpu", "fp32")
    gpu = estimate_minutes(60, "cuda", "fp16")
    check("cpu estimated slower than gpu", cpu > gpu, f"{cpu} vs {gpu}")

    # Raw model output is not level-controlled: the first real 1-minute render
    # measured -12.9 LUFS and +1.1 dBTP, which clips once encoded to MP3.  The
    # `hf` command masters like every other generator, so guard that here.
    from mpipe.audio import integrated_lufs, true_peak_db
    from mpipe.effects import brickwall
    sr = 44100
    rng = np.random.default_rng(5)
    t = np.arange(sr * 6) / sr
    hot = np.stack([0.6 * np.sin(2 * np.pi * 180 * t)
                    + 0.3 * np.sin(2 * np.pi * 2400 * t)
                    + 0.15 * rng.standard_normal(len(t))] * 2, axis=1).astype("float32")
    hot *= 1.8
    check("the unmastered case really does overshoot",
          true_peak_db(hot, sr) > -1.0, f"{true_peak_db(hot, sr):.2f} dBTP")
    measured = integrated_lufs(hot, sr)
    fixed = hot * (10 ** ((-14.0 - measured) / 20.0))
    fixed = brickwall(fixed, sr, ceiling_db=-1.0, true_peak=True)
    after = integrated_lufs(fixed, sr)
    drift = -14.0 - after
    if abs(drift) > 0.4:
        fixed = brickwall(fixed * (10 ** (drift / 20.0)), sr,
                          ceiling_db=-1.0, true_peak=True)
        after = integrated_lufs(fixed, sr)
    check("mastering lands on the loudness target", abs(after + 14.0) < 0.6,
          f"{after:.2f} LUFS")
    check("mastering holds the true-peak ceiling",
          true_peak_db(fixed, sr) <= -0.9, f"{true_peak_db(fixed, sr):.2f} dBTP")

    # the model table is the thing people plan a download around, so the
    # numbers and the fits/does-not-fit arithmetic have to be right
    for key, spec in MODELS.items():
        check(f"{key} states its download size", spec.download_gb > 0,
              str(spec.download_gb))
        check(f"{key} states its VRAM", spec.vram_fp32 > 0 and spec.vram_fp16 > 0)
        check(f"{key} fp16 is lighter than fp32", spec.vram_fp16 < spec.vram_fp32)
    check("small fits 6 GB in fp32", MODELS["musicgen-small"].fits(6.0, "fp32"))
    check("medium does NOT fit 6 GB in fp32",
          not MODELS["musicgen-medium"].fits(6.0, "fp32"))
    check("medium fits 6 GB in fp16", MODELS["musicgen-medium"].fits(6.0, "fp16"))
    check("large fits 6 GB in neither",
          not MODELS["musicgen-large"].fits(6.0, "fp16")
          and not MODELS["musicgen-large"].fits(6.0, "fp32"))
    check("stereo-medium fits 6 GB in fp32 (the sweet spot)",
          MODELS["musicgen-stereo-medium"].fits(6.0, "fp32"))

    from mpipe.hfaudio import Progress, _clock
    check("clock formats minutes", _clock(125) == "02:05", _clock(125))
    check("clock formats hours", _clock(3725) == "1:02:05", _clock(3725))
    bar = Progress(100, enabled=False)
    bar.start -= 50
    bar.update(25)
    stats = bar.done()
    check("progress reports a realtime factor",
          stats["realtime_factor"] and stats["realtime_factor"] > 1,
          str(stats))

    import pipeline
    parser_flags = [a for a in dir(pipeline) if a == "add_hf_args"]
    check("hf exposes its own loudness controls", bool(parser_flags))
    import argparse as _ap
    probe = _ap.ArgumentParser()
    pipeline.add_hf_args(probe)
    opts = {a.dest for a in probe._actions}
    check("hf takes --lufs and --peak", {"lufs", "peak"} <= opts, str(sorted(opts))[:90])
    check("hf can download and list models",
          {"download", "list_models"} <= opts, str(sorted(opts))[:90])
    check("hf takes --hours and --no-resume",
          {"hours", "no_resume"} <= opts, str(sorted(opts))[:90])


def test_hfaudio_model(tmp):
    """Actually run the model, when the weights are already on this machine."""
    print("\nhf backend (real model)")
    from mpipe import hfaudio
    if not hfaudio.torch_available():
        print("  skip (no torch)")
        return
    try:
        import transformers  # noqa: F401
    except Exception:
        print("  skip (no transformers)")
        return
    import os
    if os.environ.get("MPIPE_SKIP_MODEL_TESTS"):
        print("  skip (MPIPE_SKIP_MODEL_TESTS set)")
        return
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache("facebook/musicgen-small", "config.json")
    if not isinstance(cached, str):
        print("  skip (weights not cached; run `pipeline.py hf --selftest` once)")
        return

    from mpipe.hfaudio import GenSettings, HFAudioGenerator
    generator = HFAudioGenerator("musicgen-small", device="cpu", quiet=True).load()
    check("sample rate reported", generator.sample_rate == 32000,
          str(generator.sample_rate))

    result = generator.selftest(seconds=1.0)
    check("selftest returns a verdict", "verdict" in result and result["seconds"] > 0)
    check("selftest says fp32 on cpu is fine", result["ok"], result["verdict"])
    check("selftest detects finite audio", result["finite"])
    check("selftest measures a real peak", result["peak"] > 1e-3, str(result["peak"]))

    audio = generator.generate(lofi_prompt_short(), GenSettings(seconds=2.0, seed=3),
                               progress=False)
    check("generate returns stereo-shaped frames", audio.ndim == 2, str(audio.shape))
    check("generate honours the length",
          abs(len(audio) / generator.sample_rate - 2.0) < 0.6,
          f"{len(audio) / generator.sample_rate:.2f}s")
    check("generated audio is finite", bool(np.isfinite(audio).all()))
    check("generated audio is not silent", float(np.abs(audio).max()) > 1e-3)

    # Longer-than-one-call output is the part most likely to break, so force the
    # continuation path with a tiny chunk limit rather than actually generating
    # 40 seconds on a CPU.
    original_limit = hfaudio.MAX_CHUNK_SECONDS
    try:
        hfaudio.MAX_CHUNK_SECONDS = 2.0
        stitched = generator.generate(lofi_prompt_short(),
                                      GenSettings(seconds=5.0, seed=7,
                                                  overlap_seconds=1.0),
                                      progress=False)
    finally:
        hfaudio.MAX_CHUNK_SECONDS = original_limit
    sr = generator.sample_rate
    check("continuation reaches the requested length",
          abs(len(stitched) / sr - 5.0) < 0.6, f"{len(stitched) / sr:.2f}s")
    check("continuation output is finite", bool(np.isfinite(stitched).all()))
    mono = stitched.mean(axis=1)
    window = int(0.25 * sr)
    levels = [float(np.sqrt((mono[i * window:(i + 1) * window] ** 2).mean()))
              for i in range(len(mono) // window)]
    check("no silent gap at the joins", all(v > 1e-4 for v in levels),
          str([round(v, 4) for v in levels]))
    steps = np.abs(np.diff(mono))
    # a hard click at a join shows up as a jump far beyond the normal maximum
    check("joins do not click",
          float(steps.max()) < float(np.percentile(steps, 99.99)) * 4.0,
          f"max {float(steps.max()):.3f} vs p99.99 "
          f"{float(np.percentile(steps, 99.99)):.3f}")

    # asking for a length just past a chunk boundary must not fire a whole
    # extra pass to cover a fraction of a second
    calls = {"n": 0}
    real_once = generator._generate_once

    def counting(*a, **kw):
        calls["n"] += 1
        return real_once(*a, **kw)

    generator._generate_once = counting
    try:
        hfaudio.MAX_CHUNK_SECONDS = 2.0
        generator.generate(lofi_prompt_short(),
                           GenSettings(seconds=2.2, seed=9, overlap_seconds=1.0),
                           progress=False)
    finally:
        generator._generate_once = real_once
        hfaudio.MAX_CHUNK_SECONDS = original_limit
    check("a sliver does not trigger another pass", calls["n"] == 1,
          f"{calls['n']} model calls for 2.2s at a 2.0s chunk limit")

    # streaming to disk: the partial must survive an interruption and resume,
    # or an hours-long job loses everything when the machine hiccups
    import soundfile as _sf
    out = tmp / "streamed.wav"
    for leftover in (out.with_suffix(".partial.wav"), out.with_suffix(".partial.json")):
        leftover.unlink(missing_ok=True)
    try:
        hfaudio.MAX_CHUNK_SECONDS = 1.5
        audio, rep = generator.generate_to_file(
            lofi_prompt_short(), GenSettings(seconds=3.0, seed=4, overlap_seconds=0.5),
            out, progress=False)
    finally:
        hfaudio.MAX_CHUNK_SECONDS = original_limit
    check("streamed generation returns audio", len(audio) > 0 and audio.ndim == 2)
    check("streamed generation reports chunks and timing",
          rep["chunks"] >= 2 and rep["elapsed"] > 0, str(rep))
    check("partial files are cleaned up on success",
          not out.with_suffix(".partial.wav").exists()
          and not out.with_suffix(".partial.json").exists())

    # now fake an interruption and check it picks up rather than restarting
    import json as _json
    partial = out.with_suffix(".partial.wav")
    sr_gen = generator.sample_rate
    _sf.write(str(partial), audio[: int(1.5 * sr_gen)], sr_gen, subtype="FLOAT")
    fingerprint = {"prompt": lofi_prompt_short(), "model": generator.spec.repo,
                   "seed": 4, "sr": sr_gen}
    out.with_suffix(".partial.json").write_text(
        _json.dumps({"fingerprint": fingerprint, "index": 1, "seconds": 1.5,
                     "target": 3.0}), encoding="utf-8")
    calls["n"] = 0
    generator._generate_once = counting
    try:
        hfaudio.MAX_CHUNK_SECONDS = 1.5
        resumed, rep2 = generator.generate_to_file(
            lofi_prompt_short(), GenSettings(seconds=3.0, seed=4, overlap_seconds=0.5),
            out, progress=False, resume=True)
    finally:
        generator._generate_once = real_once
        hfaudio.MAX_CHUNK_SECONDS = original_limit
    check("resume continues instead of restarting", calls["n"] < rep["chunks"],
          f"{calls['n']} new calls vs {rep['chunks']} from scratch")
    check("resumed output reaches the target",
          abs(len(resumed) / sr_gen - 3.0) < 0.6, f"{len(resumed) / sr_gen:.2f}s")

    # a partial from a DIFFERENT prompt must not be reused
    _sf.write(str(partial), audio[: int(1.5 * sr_gen)], sr_gen, subtype="FLOAT")
    out.with_suffix(".partial.json").write_text(
        _json.dumps({"fingerprint": {**fingerprint, "prompt": "something else"},
                     "index": 1, "seconds": 1.5, "target": 3.0}), encoding="utf-8")
    calls["n"] = 0
    generator._generate_once = counting
    try:
        hfaudio.MAX_CHUNK_SECONDS = 1.5
        generator.generate_to_file(
            lofi_prompt_short(), GenSettings(seconds=3.0, seed=4, overlap_seconds=0.5),
            out, progress=False, resume=True)
    finally:
        generator._generate_once = real_once
        hfaudio.MAX_CHUNK_SECONDS = original_limit
    check("a mismatched partial is discarded, not resumed",
          calls["n"] >= rep["chunks"],
          f"{calls['n']} calls - should have started over")


def lofi_prompt_short():
    return "lofi hip hop, mellow rhodes, soft drums"


def test_cli():
    print("\ncli")
    import pipeline
    for args in (["--help"], ["lofi", "-h"], ["song", "-h"], ["all", "-h"],
                 ["check", "-h"], ["doctor", "-h"], ["art", "-h"],
                 ["generate", "-h"], ["lofify", "-h"], ["mix", "-h"],
                 ["models", "-h"], ["hf", "-h"]):
        try:
            pipeline.main(args)
        except SystemExit as exc:
            check(f"`{' '.join(args)}` parses", exc.code == 0, f"exit {exc.code}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="mpipe_test_"))
    try:
        test_theory()
        test_synth()
        test_drums()
        test_dsp()
        test_stretch()
        test_loudness()
        test_true_peak()
        test_formats(tmp)
        test_tags(tmp)
        test_lofify(tmp)
        test_fingerprint(tmp)
        test_bad_inputs(tmp)
        test_odd_files(tmp)
        test_effects_reset()
        test_models_catalogue()
        test_check_warnings(tmp)
        test_hfaudio_logic()
        test_hfaudio_model(tmp)
        test_comfy_workflows()
        test_comfy_roundtrip(tmp)
        test_video_loop(tmp)
        test_end_to_end(tmp)
        test_cli()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 60)
    if FAILED:
        print(f"{PASSED} passed, {len(FAILED)} FAILED")
        for line in FAILED:
            print(f"  - {line}")
        return 1
    print(f"all {PASSED} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
