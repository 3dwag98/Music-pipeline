#!/usr/bin/env python3
"""Smoke + correctness tests.  Run:  python tests/test_pipeline.py

Fast enough to run before every commit (about a minute).  Covers the things
that are easy to break silently: music theory staying in key, time-stretch and
pitch-shift accuracy, limiter ceilings, loudness targeting, fingerprint
discrimination, and a full generate -> song -> check round trip.
"""

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
    from mpipe.stretch import detect_bpm, detect_key, pitch_shift, time_stretch
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
    from mpipe.audio import write_audio
    from mpipe.engine import SongSpec, render_song
    from mpipe.fingerprint import (Ledger, containment, feature_distance,
                                   feature_vector, fingerprint_file, similarity)
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
    applied = apply_settings(art, prompt="P", negative="N", seed=5, steps=9,
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
    import shutil as _shutil
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


def test_cli():
    print("\ncli")
    import pipeline
    for args in (["--help"], ["lofi", "-h"], ["song", "-h"], ["all", "-h"],
                 ["check", "-h"], ["doctor", "-h"], ["art", "-h"],
                 ["generate", "-h"]):
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
        test_fingerprint(tmp)
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
