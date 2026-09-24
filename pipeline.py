#!/usr/bin/env python3
"""Local lofi music pipeline - generate, master, mix and publish, all on your PC.

Two generators:

  lofi      the built-in engine.  Pure DSP, no model, no GPU, no downloads.
            Runs on the CPU, renders many times faster than realtime, and can
            write a single continuous song that is hours long.
  generate  ACE-Step 1.5 over its local REST API (optional, needs the GPU).

Then:

  master    trim, loudness-normalise and limit each track
  song      beat-match a list of tracks into ONE continuous lofi song
  mix       the classic crossfaded compilation + YouTube chapters
  video     put the audio under a looping clip or a still image
  check     originality + upload-readiness report
  doctor    check this machine and write tuned ACE-Step settings

Start with:  python pipeline.py doctor
Then:        python pipeline.py lofi --count 6
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np
from datetime import datetime
from pathlib import Path

from mpipe import __version__
from mpipe.util import (DEFAULT_PRESETS, audio_files, collect_inputs,
                        die, fmt_time, load_presets, log, new_run,
                        read_manifest, resolve_run, set_quiet, slug, warn,
                        write_manifest)

LEDGER_PATH = Path(__file__).resolve().parent / "upload_ledger.json"

#: MP3 is the default everywhere: it is what gets uploaded, and a 3-hour WAV is
#: 3.2 GB against 430 MB for the same thing at 320 kbps.  --format wav keeps the
#: lossless path for anyone who wants to master elsewhere.
DEFAULT_FORMAT = "mp3"


def _fmt(args):
    return str(getattr(args, "format", DEFAULT_FORMAT) or DEFAULT_FORMAT).lower()


def _quality(args):
    return getattr(args, "mp3_quality", 320)


def _out_path(base, args):
    """Give `base` the chosen output extension."""
    from mpipe.audio import with_format
    return with_format(base, _fmt(args))


def _tag(path, args, **fields):
    """Write ID3/metadata, quietly doing nothing when mutagen is missing."""
    from mpipe.tags import write_tags
    if getattr(args, "no_tags", False):
        return
    write_tags(path, quiet=True, **fields)


# ---------------------------------------------------------------------------
# lofi - the built-in engine
# ---------------------------------------------------------------------------

def preset_to_spec(presets, name, rng, args, index=0):
    """Build one SongSpec from a style block.

    `rng` is a per-track generator (see cmd_lofi): deriving one from the run
    seed and the track index keeps each track's choices independent of how many
    random draws the previous tracks happened to make, so variety does not
    depend on the order things are evaluated in.
    """
    from mpipe.engine import SongSpec
    engine = presets.get("engine", {})
    styles = engine.get("styles", {})
    if name not in styles:
        die(f"unknown style '{name}'. Available: {', '.join(styles) or '(none in presets.json)'}")
    style = styles[name]
    defaults = engine.get("defaults", {})

    def pick(key, fallback=None):
        value = style.get(key, defaults.get(key, fallback))
        return rng.choice(value) if isinstance(value, list) and value else value

    bpm_range = style.get("bpm", defaults.get("bpm", [70, 86]))
    bpm = args.bpm or rng.randint(int(bpm_range[0]), int(bpm_range[1]))
    mood = (presets.get("moods", {}) or {}).get((args.mood or "").lower())
    if mood:
        bpm += int(mood.get("bpm_shift", 0))
    keys = style.get("keys", defaults.get("keys", ["A Minor"]))
    if mood and mood.get("prefer_minor"):
        keys = [k for k in keys if "minor" in k.lower()] or keys
    elif mood and mood.get("prefer_major"):
        keys = [k for k in keys if "major" in k.lower()] or keys
    key = args.key or rng.choice(keys)

    words = style.get("title_words", defaults.get("title_words", {}))
    title = f"{rng.choice(words.get('adj', ['Quiet']))} {rng.choice(words.get('noun', ['Hour']))}" \
        if words else f"{name.title()} {index + 1}"

    return SongSpec(
        title=title,
        seed=rng.randrange(2 ** 31),
        bpm=float(max(40, min(180, bpm))),
        key=key,
        swing=float(args.swing if args.swing is not None else pick("swing", 0.16)),
        chord_instrument=pick("chord_instrument", "rhodes"),
        lead_instrument=pick("lead_instrument", "vibraphone"),
        bass_instrument=pick("bass_instrument", "sub_bass"),
        pad_instrument=pick("pad_instrument", "pad"),
        drum_style=pick("drum_style", "dusty"),
        drum_pattern=pick("drum_pattern", "boom_bap"),
        richness=float(pick("richness", 0.85)),
        vinyl=float(args.vinyl if args.vinyl is not None else pick("vinyl", 1.0)),
        tape=float(args.tape if args.tape is not None else pick("tape", 1.0)),
        reverb=float(pick("reverb", 0.35)),
        lead_density=float(pick("lead_density", 0.55)),
        bars_per_loop=int(pick("bars_per_loop", 4)),
    )


def cmd_lofi(args):
    from mpipe.engine import render_song
    from mpipe.mastering import measure_stream, normalise_stream

    presets = load_presets(args.presets)
    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    run = Path(args.run) if args.run else new_run(f"lofi_{args.style}")
    raw = run / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(run)
    manifest.update({
        "created": datetime.now().isoformat(timespec="seconds"),
        "engine": "builtin", "version": __version__, "style": args.style,
        "mood": args.mood, "run_seed": seed, "minutes_each": args.minutes,
        "source": "fully synthesised locally - no samples, no third-party audio",
    })
    manifest.setdefault("tracks", [])
    write_manifest(run, manifest)
    log(f"Run folder: {run}")
    log(f"Engine: built-in (CPU only)   style: {args.style}   seed: {seed}\n")

    made = 0
    for i in range(1, args.count + 1):
        track_rng = random.Random((seed * 1_000_003 + i * 7_919) % (2 ** 63))
        spec = preset_to_spec(presets, args.style, track_rng, args, index=i - 1)
        if args.mood and args.mood.lower() not in presets.get("moods", {}):
            spec.title = f"{spec.title}"
        stem = _out_path(raw / f"{i:02d}_{slug(spec.title)}", args)
        log(f"[{i}/{args.count}] {spec.title}  |  {spec.bpm:.0f} BPM, {spec.key}, "
            f"{args.minutes:g} min, {spec.drum_pattern}/{spec.chord_instrument}")
        try:
            report = render_song(spec, stem, minutes=args.minutes, sr=args.samplerate,
                                 progress=args.verbose, peak_db=args.peak,
                                 mp3_quality=_quality(args))
        except KeyboardInterrupt:
            write_manifest(run, manifest)
            die("stopped by user (finished tracks are kept)")
        if args.lufs is not None:
            stats = measure_stream(stem)
            tmp = stem.with_name(stem.stem + ".tmp" + stem.suffix)
            normalise_stream(stem, tmp, lufs=args.lufs, peak_db=args.peak,
                             report=stats, mp3_quality=_quality(args))
            tmp.replace(stem)
            report["lufs"] = args.lufs
        _tag(stem, args, title=spec.title, artist=args.artist, album=args.album,
             track=i, bpm=spec.bpm, key=spec.key, year=datetime.now().year)
        log(f"    -> {stem.name}  {fmt_time(report['seconds'])}  "
            f"{report['progression']}  {report['lufs']} LUFS\n")
        manifest["tracks"].append({
            "index": i, "title": spec.title, "file": stem.name, "status": "ok",
            "spec": spec.to_dict(), "report": {k: v for k, v in report.items()
                                               if k != "markers"},
        })
        write_manifest(run, manifest)
        made += 1

    log(f"Done: {made} track(s) in {raw}")
    if made:
        log(f"Next:  python pipeline.py song --run {run} --hours 1")
    return run


# ---------------------------------------------------------------------------
# generate - ACE-Step backend
# ---------------------------------------------------------------------------

def build_track_spec(presets, genre, mood, rng, used_titles, bpm=None, key=None,
                     duration=None, extra=None):
    genres = presets.get("genres", {})
    if genre not in genres:
        die(f"unknown genre '{genre}'. Available: {', '.join(genres)}")
    g = genres[genre]
    defaults = presets.get("defaults", {})

    parts = [g["base"]]
    for _, options in g.get("pools", {}).items():
        parts.append(rng.choice(options))

    bpm_shift, prefer = 0, None
    if mood:
        m = presets.get("moods", {}).get(mood.lower())
        if m:
            parts.append(m["words"])
            bpm_shift = m.get("bpm_shift", 0)
            prefer = "Minor" if m.get("prefer_minor") else "Major" if m.get("prefer_major") else None
        else:
            parts.append(mood)
    if extra:
        parts.append(extra)

    if bpm is None:
        lo, hi = g.get("bpm", [70, 90])
        bpm = rng.randint(lo, hi) + bpm_shift
    bpm = max(40, min(200, int(bpm)))

    if key is None:
        keys = g.get("keys", ["C Major", "A Minor"])
        preferred = [k for k in keys if prefer and k.endswith(prefer)]
        key = rng.choice(preferred or keys)

    words = g.get("title_words")
    title = None
    for _ in range(100):
        t = f"{rng.choice(words['adj'])} {rng.choice(words['noun'])}" if words \
            else f"{genre} {len(used_titles) + 1}"
        if t not in used_titles:
            title = t
            break
    title = title or f"{genre.title()} {len(used_titles) + 1}"
    used_titles.add(title)

    return {
        "title": title, "caption": ", ".join(parts),
        "lyrics": g.get("lyrics", defaults.get("lyrics", "")),
        "bpm": bpm, "key_scale": key,
        "time_signature": str(g.get("time_signature", defaults.get("time_signature", "4"))),
        "duration": float(duration or g.get("duration", defaults.get("duration", 150))),
        "inference_steps": int(defaults.get("inference_steps", 8)),
        "seed": rng.randint(0, 2 ** 31 - 1),
    }


def cmd_generate(args):
    from mpipe.acestep import LOW_VRAM_REQUEST, AceStepClient, is_oom

    if getattr(args, "backend", "server") == "comfy":
        return cmd_generate_comfy(args)

    presets = load_presets(args.presets)
    if args.reference and not Path(args.reference).is_file():
        die(f"reference file not found: {args.reference}")
    if args.ref_mode == "cover" and args.reference and not args.i_own_this:
        die("--ref-mode cover rebuilds the track you give it.  A cover of someone\n"
            "       else's recording is the single most likely thing here to be claimed.\n"
            "       Re-run with --i-own-this if the audio is yours (or public domain).")

    client = AceStepClient(args.server, args.api_key)
    client.health()

    bpm, key = args.bpm, args.key
    if args.reference and args.match_reference:
        from mpipe.stretch import analyze_file
        info = analyze_file(args.reference)
        bpm = bpm or int(round(info["bpm"]))
        key = key or info["key"]
        log(f"Reference: ~{info['bpm']:.0f} BPM, {info['key']} (using these unless overridden)")

    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    rng = random.Random(seed)
    run = Path(args.run) if args.run else new_run(
        f"{args.genre}" + (f"_{slug(args.mood)}" if args.mood else ""))
    (run / "raw").mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(run)
    manifest.update({"created": datetime.now().isoformat(timespec="seconds"),
                     "engine": "ace-step", "version": __version__, "genre": args.genre,
                     "mood": args.mood, "run_seed": seed, "server": args.server,
                     "reference": args.reference, "ref_mode": args.ref_mode})
    manifest.setdefault("tracks", [])
    write_manifest(run, manifest)
    log(f"Run folder: {run}")

    used, ok = set(), 0
    duration = args.duration
    for i in range(1, args.count + 1):
        spec = build_track_spec(presets, args.genre, args.mood, rng, used, bpm=bpm,
                                key=key, duration=duration, extra=args.extra)
        payload = {
            "prompt": spec["caption"], "lyrics": spec["lyrics"],
            "bpm": spec["bpm"], "key_scale": spec["key_scale"],
            "time_signature": spec["time_signature"],
            "audio_duration": spec["duration"],
            "inference_steps": spec["inference_steps"],
            "use_random_seed": False, "seed": spec["seed"],
            "audio_format": "wav", "task_type": "text2music",
            **LOW_VRAM_REQUEST,
        }
        payload["inference_steps"] = spec["inference_steps"]
        if args.model:
            payload["model"] = args.model
        reference = source = None
        if args.reference:
            if args.ref_mode == "cover":
                payload["task_type"] = "cover"
                payload["audio_cover_strength"] = args.strength
                payload.pop("audio_duration", None)
                source = args.reference
            else:
                reference = args.reference

        log(f"\n[{i}/{args.count}] {spec['title']}  |  {spec['bpm']} BPM, "
            f"{spec['key_scale']}, {int(spec['duration'])}s")
        log(f"    {spec['caption']}")
        stem = run / "raw" / f"{i:02d}_{slug(spec['title'])}"
        entry = dict(spec, index=i, request=payload, file=None, status="failed")
        import time
        for attempt in range(args.retries + 1):
            try:
                t0 = time.time()
                task_id = client.submit(payload, reference=reference, source=source)
                files = client.wait(task_id, timeout=args.task_timeout)
                dest = client.download(files[0], stem)
                entry.update(file=dest.name, status="ok", seconds=round(time.time() - t0, 1))
                log(f"    saved {dest.name} in {entry['seconds']}s")
                ok += 1
                break
            except KeyboardInterrupt:
                write_manifest(run, manifest)
                die("stopped by user (finished tracks are kept)")
            except Exception as exc:
                entry["error"] = str(exc)
                log(f"    attempt {attempt + 1} failed: {exc}")
                if is_oom(exc) and payload.get("audio_duration", 0) > 60:
                    payload["audio_duration"] = max(60.0, payload["audio_duration"] * 0.7)
                    log(f"    out of VRAM - retrying at {payload['audio_duration']:.0f}s "
                        f"(use --duration {int(payload['audio_duration'])} next time)")
        manifest["tracks"].append(entry)
        write_manifest(run, manifest)

    log(f"\nDone: {ok}/{args.count} tracks in {run / 'raw'}")
    if ok == 0:
        die("nothing was generated - check the ACE-Step server window for errors")
    return run


# ---------------------------------------------------------------------------
# ComfyUI: art (the visual) and audio (ACE-Step through ComfyUI)
# ---------------------------------------------------------------------------

def _comfy_client(args):
    from mpipe.comfy import ComfyClient
    client = ComfyClient(args.comfy_url)
    client.health()
    return client


def _check_checkpoint(client, graph):
    """Fail early and usefully when the workflow names a model ComfyUI lacks."""
    from mpipe.comfy import find_nodes
    available = client.checkpoints()
    if not available:
        return
    for nid in find_nodes(graph, "CheckpointLoaderSimple"):
        want = graph[nid].get("inputs", {}).get("ckpt_name")
        if want and want not in available:
            die(f"ComfyUI has no checkpoint called '{want}'.\n"
                f"       It can see: {', '.join(available[:12])}"
                f"{' ...' if len(available) > 12 else ''}\n"
                f"       Pick one with:  --set CHECKPOINT.ckpt_name=<name>")


def cmd_art(args):
    from mpipe.comfy import (apply_settings, describe_workflow, load_workflow)
    from mpipe.video import make_loop

    graph = load_workflow(args.workflow)
    if args.show_workflow:
        log(f"{args.workflow}: {len(graph)} nodes\n")
        log("\n".join(describe_workflow(graph)))
        log("\nPatch any of these with --set NODE.FIELD=VALUE")
        return None

    run = resolve_run(args.run) if args.run else new_run("art")
    out_dir = run / "art"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = _comfy_client(args)
    log(f"ComfyUI: {client.describe()}")

    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    width, height = _parse_size(args.size)
    made = []
    for i in range(1, args.count + 1):
        work = json.loads(json.dumps(graph))
        applied = apply_settings(
            work, prompt=args.prompt, negative=args.negative,
            seed=seed + i - 1, steps=args.steps, cfg=args.cfg,
            width=width, height=height, filename=f"lofi_art_{i:02d}",
            overrides=args.set)
        if args.prompt and not applied.get("prompt"):
            warn("could not find a prompt node to patch - "
                 "title one POSITIVE, or use --set")
        if i == 1:
            _check_checkpoint(client, work)
        log(f"[{i}/{args.count}] seed {seed + i - 1}  {width}x{height}")
        stem = out_dir / f"art_{i:02d}"
        try:
            paths = client.run(work, stem, want="image", timeout=args.task_timeout)
        except KeyboardInterrupt:
            die("stopped by user")
        for path in paths:
            log(f"    {path.name}")
            made.append(path)

    if not made:
        die("ComfyUI returned no images")
    if args.loop_seconds:
        source = made[0]
        loop = run / "loop.mp4"
        make_loop(source, loop, seconds=args.loop_seconds, size=args.size,
                  fps=args.fps or 30, zoom=args.zoom, nvenc=args.nvenc,
                  dry_run=args.dry_run)
        log(f"\nSeamless loop: {loop}")
        log(f"Next:  python pipeline.py video --loop \"{loop}\"")
    else:
        log(f"\nNext:  python pipeline.py video --image \"{made[0]}\"")
    return run


def _clock_short(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def _parse_size(size):
    try:
        width, height = (int(v) for v in str(size).lower().split("x"))
        return width, height
    except ValueError:
        die(f"--size must look like 1920x1080, got: {size}")


def cmd_generate_comfy(args):
    """Generate tracks through ComfyUI instead of the standalone ACE-Step server."""
    from mpipe.comfy import apply_settings, load_workflow

    presets = load_presets(args.presets)
    graph = load_workflow(args.workflow or "acestep")
    client = _comfy_client(args)
    log(f"ComfyUI: {client.describe()}")

    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    rng = random.Random(seed)
    run = Path(args.run) if args.run else new_run(f"comfy_{args.genre}")
    (run / "raw").mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(run)
    manifest.update({"created": datetime.now().isoformat(timespec="seconds"),
                     "engine": "ace-step", "backend": "comfyui",
                     "version": __version__, "genre": args.genre, "mood": args.mood,
                     "run_seed": seed, "server": args.comfy_url,
                     "workflow": str(args.workflow or "acestep")})
    manifest.setdefault("tracks", [])
    write_manifest(run, manifest)
    log(f"Run folder: {run}")

    used, ok = set(), 0
    for i in range(1, args.count + 1):
        spec = build_track_spec(presets, args.genre, args.mood, rng, used,
                                bpm=args.bpm, key=args.key, duration=args.duration,
                                extra=args.extra)
        tags = f"{spec['caption']}, {spec['bpm']} bpm, {spec['key_scale']}"
        work = json.loads(json.dumps(graph))
        applied = apply_settings(work, prompt=tags, seed=spec["seed"],
                                 steps=args.steps, cfg=args.cfg,
                                 seconds=spec["duration"],
                                 lyrics=spec["lyrics"] or "",
                                 filename=f"track_{i:02d}", overrides=args.set)
        if not applied.get("prompt"):
            die("could not find a prompt node in the workflow - "
                "title one POSITIVE, or use --set")
        if i == 1:
            _check_checkpoint(client, work)
        log(f"\n[{i}/{args.count}] {spec['title']}  |  {spec['bpm']} BPM, "
            f"{spec['key_scale']}, {int(spec['duration'])}s")
        log(f"    {spec['caption']}")
        stem = run / "raw" / f"{i:02d}_{slug(spec['title'])}"
        entry = dict(spec, index=i, file=None, status="failed")
        import time
        for attempt in range(args.retries + 1):
            try:
                t0 = time.time()
                paths = client.run(work, stem, want="audio", timeout=args.task_timeout)
                entry.update(file=paths[0].name, status="ok",
                             seconds=round(time.time() - t0, 1))
                log(f"    saved {paths[0].name} in {entry['seconds']}s")
                ok += 1
                break
            except KeyboardInterrupt:
                write_manifest(run, manifest)
                die("stopped by user (finished tracks are kept)")
            except Exception as exc:
                entry["error"] = str(exc)
                log(f"    attempt {attempt + 1} failed: {exc}")
        manifest["tracks"].append(entry)
        write_manifest(run, manifest)

    log(f"\nDone: {ok}/{args.count} tracks in {run / 'raw'}")
    if ok == 0:
        die("nothing was generated - check the ComfyUI window for errors")
    return run


# ---------------------------------------------------------------------------
# master
# ---------------------------------------------------------------------------

def cmd_master(args):
    from mpipe.mastering import master_file

    run = resolve_run(args.run)
    sources = audio_files(run / "raw")
    if not sources:
        die(f"no audio in {run / 'raw'}")
    out_dir = run / "mastered"
    out_dir.mkdir(exist_ok=True)

    log(f"Mastering {len(sources)} tracks -> {out_dir}  "
        f"(target {args.lufs} LUFS, peak {args.peak} dBFS)")
    kept = 0
    manifest = read_manifest(run)
    reports = {}
    for path in sources:
        report = master_file(path, _out_path(out_dir / path.stem, args),
                             lufs=args.lufs, peak_db=args.peak,
                             fade_in=args.fade_in, fade_out=args.fade_out,
                             trim=not args.no_trim, mp3_quality=_quality(args))
        if report.get("skipped"):
            log(f"  SKIP {path.name}: {report['skipped']}")
            continue
        reports[path.stem] = report
        kept += 1
        log(f"  {path.stem:<34} {report['lufs_in']:>6.1f} -> {report['lufs_out']:>6.1f} LUFS  "
            f"peak {report['true_peak_db']:>5.1f} dBTP  {fmt_time(report['seconds'])}")
    manifest["mastered"] = reports
    write_manifest(run, manifest)

    log(f"\n{kept} tracks ready in {out_dir}")
    log("Listen and DELETE the ones you don't like, then run 'song' or 'mix'.")
    return run


# ---------------------------------------------------------------------------
# song / mix
# ---------------------------------------------------------------------------

def _inputs_for(args, run):
    if args.input:
        files = collect_inputs(args.input)
        if not files:
            die(f"no audio found in {args.input}")
        return files
    for folder in ("mastered", "raw"):
        files = audio_files(run / folder)
        if files:
            if folder == "raw":
                log("(using raw/ - run 'master' first for even levels)")
            return files
    die(f"no audio in {run}/mastered or {run}/raw")


def _minutes_from(args):
    if getattr(args, "hours", 0):
        return float(args.hours) * 60.0
    return float(getattr(args, "minutes", 0) or 0)


def _titles_from(run):
    titles = {}
    for track in read_manifest(run).get("tracks", []):
        if track.get("file"):
            titles[Path(track["file"]).stem] = track.get("title") or ""
    return {k: v for k, v in titles.items() if v}


def cmd_song(args):
    from mpipe.mastering import normalise_stream
    from mpipe.song import build_song, write_tracklist

    run = resolve_run(args.run) if (args.run or not args.input) else new_run("song")
    files = _inputs_for(args, run)
    out = Path(args.out) if args.out else _out_path(run / "song", args)
    minutes = _minutes_from(args)
    log(f"Building one continuous song from {len(files)} tracks"
        + (f", target {fmt_time(minutes * 60, minutes >= 60)}" if minutes else "") + "\n")

    report = build_song(
        files, out, minutes=minutes, sr=args.samplerate, bpm=args.bpm, key=args.key,
        order=args.order, crossfade_bars=args.crossfade_bars, seed=args.seed or 0,
        vinyl=args.vinyl, spine=args.spine, tape=args.tape, width=args.width,
        lufs=args.lufs, peak_db=args.peak, tone_strength=args.tone_match,
        lofi=args.lofi, max_stretch=args.max_stretch, max_shift=args.max_shift,
        fade_out=args.final_fade, cache_path=run / "analysis.json",
        titles=_titles_from(run), progress=args.verbose,
        spine_style=args.spine_style, spine_pattern=args.spine_pattern,
        mp3_quality=_quality(args))

    if abs(report["lufs"] - args.lufs) > 0.5:
        log(f"\nNormalising {report['lufs']} -> {args.lufs} LUFS...")
        tmp = out.with_name(out.stem + ".norm" + out.suffix)
        result = normalise_stream(out, tmp, lufs=args.lufs, peak_db=args.peak,
                                  mp3_quality=_quality(args))
        tmp.replace(out)
        report["lufs"] = result["lufs_out"]
        report["peak_dbfs"] = result["peak_dbfs"]

    tracklist = out.with_name(out.stem + "_tracklist.txt")
    lines = write_tracklist(tracklist, report["chapters"], report["seconds"])
    _write_report(run, "song", report)
    _tag(out, args, title=args.title or out.stem.replace("_", " ").title(),
         artist=args.artist, album=args.album, bpm=report["target_bpm"],
         key=report["target_key"], year=datetime.now().year,
         cover=_find_cover(run))

    log(f"\nSong:      {out}")
    log(f"Length:    {fmt_time(report['seconds'], report['seconds'] >= 3600)}   "
        f"{report['target_bpm']:.1f} BPM   {report['target_key']}")
    log(f"Loudness:  {report['lufs']} LUFS, peak {report['peak_dbfs']} dBFS")
    log(f"Chapters:  {tracklist} ({len(lines)} entries)")
    if len(lines) < 3:
        log("           NOTE: YouTube needs at least 3 chapters to show a chapter list.")
    if report["passes"] > 1:
        log(f"           {report['unique_tracks']} unique tracks reused over "
            f"{report['passes']} passes - generate more for less repetition.")
    _maybe_mp3(args, out)
    log(f"\nNext:  python pipeline.py check \"{out}\"")
    return run


def cmd_mix(args):
    from mpipe.song import build_mix, write_tracklist

    run = resolve_run(args.run) if (args.run or not args.input) else new_run("mix")
    files = _inputs_for(args, run)
    out = Path(args.out) if args.out else _out_path(run / "mix", args)
    minutes = _minutes_from(args)

    report = build_mix(files, out, minutes=minutes, sr=args.samplerate,
                       crossfade=args.crossfade, final_fade=args.final_fade,
                       shuffle=args.shuffle, seed=args.seed or 0,
                       titles=_titles_from(run), progress=args.verbose,
                       peak_db=args.peak, mp3_quality=_quality(args))
    tracklist = out.with_name(out.stem + "_tracklist.txt")
    lines = write_tracklist(tracklist, report["chapters"], report["seconds"])
    _write_report(run, "mix", report)
    _tag(out, args, title=args.title or out.stem.replace("_", " ").title(),
         artist=args.artist, album=args.album, year=datetime.now().year,
         cover=_find_cover(run))

    log(f"\nMix:      {out}  ({fmt_time(report['seconds'], report['seconds'] >= 3600)}, "
        f"{report['segments']} segments)")
    log(f"Loudness: {report['lufs']} LUFS, peak {report['peak_dbfs']} dBFS")
    log(f"Chapters: {tracklist}")
    if report["repeated"]:
        log("          NOTE: tracks had to repeat to reach the target length.")
    if len(lines) < 3:
        log("          NOTE: YouTube needs at least 3 chapters to show a chapter list.")
    _maybe_mp3(args, out)
    return run


def _find_cover(run):
    """The newest ComfyUI artwork in this run, for embedding as cover art."""
    art = Path(run) / "art"
    if not art.is_dir():
        return None
    images = sorted(art.glob("*.png")) + sorted(art.glob("*.jpg"))
    return str(images[0]) if images else None


def _maybe_mp3(args, out):
    """Kept for the old --mp3 flag; MP3 is now the default output format."""
    if getattr(args, "mp3", False) and out.suffix.lower() != ".mp3":
        log("(--mp3 is the default now; use --format mp3)")


def _write_report(run, name, report):
    payload = {k: v for k, v in report.items() if k != "chapters"}
    payload["chapters"] = [[round(t, 2), n] for t, n in report.get("chapters", [])]
    path = Path(run) / f"{name}_report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# hf - local text-to-audio model (the smallest one that makes music)
# ---------------------------------------------------------------------------

def cmd_hf(args):
    from mpipe.audio import integrated_lufs, resample, true_peak_db, write_audio
    from mpipe.effects import brickwall
    from mpipe.hfaudio import (GenSettings, HFAudioGenerator, MODELS,
                               cache_size_gb, download_model, estimate_minutes,
                               is_downloaded, lofi_prompt)

    if args.list_models:
        log("MusicGen variants (download is what transformers actually pulls)\n")
        log(f"  {'model':<24}{'download':>10}{'fp32':>8}{'fp16':>8}  {'on disk':<9}fits 6 GB?")
        log(f"  {'-' * 24}{'-' * 10}{'-' * 8}{'-' * 8}  {'-' * 9}----------")
        for key, spec in MODELS.items():
            have = cache_size_gb(spec.repo, args.cache_dir)
            fits = []
            if spec.fits(6.0, "fp32"):
                fits.append("fp32")
            if spec.fits(6.0, "fp16"):
                fits.append("fp16")
            log(f"  {key:<24}{spec.download_gb:>7.2f} GB{spec.vram_fp32:>6.1f} GB"
                f"{spec.vram_fp16:>6.1f} GB  {(f'{have:.1f} GB' if have else '-'):<9}"
                f"{'/'.join(fits) if fits else 'NO'}")
        log("\n  Download one with:  python pipeline.py hf --download <model>")
        log("  All MusicGen weights are CC-BY-NC 4.0 - not for commercial use.")
        return None

    if args.download:
        download_model(args.download if args.download is not True else args.model,
                       cache_dir=args.cache_dir)
        return None

    if not is_downloaded(MODELS[args.model].repo, args.cache_dir):
        spec = MODELS[args.model]
        log(f"{spec.repo} is not downloaded yet ({spec.download_gb:.2f} GB).")
        log(f"Fetching it now - or Ctrl+C and run: "
            f"python pipeline.py hf --download {args.model}\n")
        download_model(args.model, cache_dir=args.cache_dir)
        log("")

    generator = HFAudioGenerator(args.model, dtype=args.dtype, device=args.device,
                                 cache_dir=args.cache_dir, quiet=args.quiet)
    generator.load()

    if args.selftest:
        log("\nSelf-test: generating a short clip and checking it is real audio...")
        result = generator.selftest(seconds=args.selftest_seconds)
        log("")
        for key in ("model", "device", "dtype", "seconds", "elapsed", "finite",
                    "peak", "rms"):
            log(f"  {key:<9} {result[key]}")
        log("")
        _bullet("!" if not result["ok"] else "+", result["verdict"])
        if not result["ok"]:
            log("")
            log("  This is exactly what the self-test is for: the output is not")
            log("  usable audio in this precision.  Re-run with --dtype fp32.")
            return None
        estimate = estimate_minutes(args.minutes * 60, generator.device,
                                    generator.dtype_label)
        log(f"\n  Rough guide: {args.minutes:g} minutes of audio will take around "
            f"{estimate:.0f} minutes per track on this setup.")
        return None

    presets = load_presets(args.presets)
    seed = args.seed if args.seed is not None else random.randrange(2 ** 31)
    run = Path(args.run) if args.run else new_run(f"hf_{args.model}")
    raw = run / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    spec_info = MODELS[args.model]
    manifest = read_manifest(run)
    manifest.update({
        "created": datetime.now().isoformat(timespec="seconds"),
        "engine": "hf-text-to-audio", "version": __version__,
        "model": spec_info.repo, "model_licence": spec_info.licence,
        "commercial_use": spec_info.commercial,
        "dtype": generator.dtype_label, "device": generator.device,
        "run_seed": seed, "minutes_each": args.minutes,
        "source": f"generated locally by {spec_info.repo}",
    })
    manifest.setdefault("tracks", [])
    write_manifest(run, manifest)

    minutes = (args.hours * 60.0) if args.hours else args.minutes
    log(f"\nRun folder: {run}")
    estimate = estimate_minutes(minutes * 60, generator.device, generator.dtype_label)
    log(f"Rough guide: about {_clock_short(estimate * 60)} per track, "
        f"{_clock_short(estimate * 60 * args.count)} for all {args.count}")
    if minutes * args.count > 20:
        log("  Long job: each track streams to disk as it is made, so an "
            "interruption\n  loses at most one chunk - rerun the same command "
            "to resume.")
    log("")
    run_start = __import__("time").time()

    target_sr = args.samplerate
    made = 0
    for i in range(1, args.count + 1):
        used = set()
        rng = random.Random((seed * 1_000_003 + i * 7_919) % (2 ** 63))
        track = build_track_spec(presets, args.genre, args.mood, rng, used,
                                 bpm=args.bpm, key=args.key, extra=args.extra)
        prompt = args.prompt or lofi_prompt(track["bpm"], extra=args.extra,
                                            mood=args.mood, rng=rng,
                                            texture=args.texture)
        if args.use_presets and not args.prompt:
            prompt = f"{track['caption']}, {track['bpm']} bpm"
        settings = GenSettings(seconds=minutes * 60.0, guidance=args.guidance,
                               temperature=args.temperature, top_k=args.top_k,
                               top_p=args.top_p, seed=rng.randrange(2 ** 31),
                               overlap_seconds=args.overlap)
        log(f"[{i}/{args.count}] {track['title']}  |  {track['bpm']} BPM, "
            f"{minutes:g} min")
        log(f"    {prompt}")
        try:
            audio, gen_report = generator.generate_to_file(
                prompt, settings, raw / f"{i:02d}_{slug(track['title'])}",
                progress=not args.quiet, resume=not args.no_resume,
                label=track["title"])
        except KeyboardInterrupt:
            write_manifest(run, manifest)
            die("stopped by user (finished tracks are kept)")
        except Exception as exc:
            log(f"    failed: {exc}")
            manifest["tracks"].append({"index": i, "status": "failed",
                                       "error": str(exc), "prompt": prompt})
            write_manifest(run, manifest)
            continue

        sr = generator.sample_rate
        if sr != target_sr:
            audio = resample(audio, sr, target_sr)
        if audio.ndim == 1:
            audio = audio[:, None]
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)

        # Master it like every other generator here does.  Raw model output is
        # not level-controlled - measured at -12.9 LUFS and +1.1 dBTP, which
        # clips once it is encoded to MP3.
        measured = integrated_lufs(audio, target_sr)
        if np.isfinite(measured):
            audio = audio * (10 ** ((args.lufs - measured) / 20.0))
        audio = brickwall(audio, target_sr, ceiling_db=args.peak, true_peak=True)
        after = integrated_lufs(audio, target_sr)
        drift = args.lufs - after
        if np.isfinite(after) and abs(drift) > 0.4:
            audio = brickwall(audio * (10 ** (drift / 20.0)), target_sr,
                              ceiling_db=args.peak, true_peak=True)
            after = integrated_lufs(audio, target_sr)
        peak_db = true_peak_db(audio, target_sr)

        dest = _out_path(raw / f"{i:02d}_{slug(track['title'])}", args)
        write_audio(dest, audio, target_sr, mp3_quality=_quality(args))
        _tag(dest, args, title=track["title"], artist=args.artist,
             album=args.album, track=i, bpm=track["bpm"],
             year=datetime.now().year)
        log(f"    -> {dest.name}  {fmt_time(len(audio) / target_sr)}  "
            f"{after:.1f} LUFS  {peak_db:.2f} dBTP  "
            f"({gen_report['chunks']} chunks, {_clock_short(gen_report['elapsed'])})\n")
        manifest["tracks"].append({
            "index": i, "title": track["title"], "file": dest.name, "status": "ok",
            "prompt": prompt, "bpm": track["bpm"], "lufs": round(float(after), 2),
            "true_peak_db": round(float(peak_db), 2),
            "generation": gen_report, "settings": settings.to_dict()})
        write_manifest(run, manifest)
        made += 1

    total = __import__("time").time() - run_start
    log(f"Done: {made}/{args.count} tracks in {raw}  (total {_clock_short(total)})")
    if made:
        log(f"Next:  python pipeline.py song --run {run} --hours 1")
    return run


# ---------------------------------------------------------------------------
# lofify - turn songs you already have into lofi
# ---------------------------------------------------------------------------

def cmd_lofify(args):
    from mpipe.lofify import PRESETS, lofify_file, settings_from_preset
    from mpipe.stretch import analyze_file

    sources = collect_inputs(args.input)
    if not sources:
        die(f"no audio found in: {' '.join(args.input)}")
    if not args.i_own_this:
        die("`lofify` rebuilds a recording you supply.  A lofi remix of someone\n"
            "       else's record is still their record, and re-uploading one is the\n"
            "       most reliable way to get a copyright claim.\n"
            "       Re-run with --i-own-this if the audio is yours, licensed to you,\n"
            "       or public domain.")

    overrides = dict(
        speed=args.speed, keep_pitch=True if args.keep_pitch else None,
        semitones=args.semitones, vocals=args.vocals, vocal_amount=args.vocal_amount,
        drums=args.drums, drum_level=args.drum_level, drum_pattern=args.drum_pattern,
        drum_style=args.drum_style, amount=args.amount, lowpass_hz=args.lowpass,
        bitcrush_bits=args.bitcrush, vinyl=args.vinyl, reverb=args.reverb,
        telephone=args.telephone, mp3_artifacts=args.mp3_artifacts,
        lufs=args.lufs, peak_db=args.peak, seed=args.seed)
    try:
        settings = settings_from_preset(args.preset, **overrides)
    except KeyError:
        die(f"unknown preset '{args.preset}'. Available: {', '.join(PRESETS)}")

    run = Path(args.run) if args.run else new_run(f"lofify_{args.preset}")
    out_dir = run / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(run)
    manifest.update({
        "created": datetime.now().isoformat(timespec="seconds"),
        "engine": "lofify", "version": __version__, "preset": args.preset,
        "settings": settings.to_dict(),
        "source": "user-supplied recordings, reprocessed",
        "rights_attested": True,
    })
    manifest.setdefault("tracks", [])
    write_manifest(run, manifest)

    from mpipe.effects import backend_name, have_demucs
    log(f"Run folder: {run}")
    log(f"Preset: {args.preset}   effects: {backend_name()}   "
        f"separation: {'demucs' if have_demucs() else 'centre-channel'}\n")

    done = 0
    for i, src in enumerate(sources, 1):
        log(f"[{i}/{len(sources)}] {src.name}")
        try:
            info = analyze_file(src)
            dest = _out_path(out_dir / f"{i:02d}_{slug(src.stem)}_lofi", args)
            report = lofify_file(src, dest, settings, sr=args.samplerate,
                                 analysis=info, progress=args.verbose,
                                 mp3_quality=_quality(args))
        except KeyboardInterrupt:
            write_manifest(run, manifest)
            die("stopped by user (finished tracks are kept)")
        except Exception as exc:
            log(f"    failed: {exc}")
            manifest["tracks"].append({"index": i, "source": str(src),
                                       "status": "failed", "error": str(exc)})
            write_manifest(run, manifest)
            continue
        _tag(dest, args, title=f"{src.stem} (lofi)", artist=args.artist,
             album=args.album, track=i, bpm=report.get("output_bpm"),
             year=datetime.now().year)
        log(f"    -> {dest.name}  {fmt_time(report['seconds'])}  "
            f"{report['output_bpm'] or '?'} BPM  {report['lufs_out']} LUFS  "
            f"{report['true_peak_db']} dBTP")
        log(f"    {report['vocals']}\n")
        manifest["tracks"].append({"index": i, "title": f"{src.stem} (lofi)",
                                   "file": dest.name, "status": "ok",
                                   "source": str(src), "report": report})
        write_manifest(run, manifest)
        done += 1

    log(f"Done: {done}/{len(sources)} tracks in {out_dir}")
    if done:
        log(f"Next:  python pipeline.py song --run {run} --hours 1")
    return run


# ---------------------------------------------------------------------------
# video / analyze
# ---------------------------------------------------------------------------

def cmd_video(args):
    from mpipe.video import build_video

    run = resolve_run(args.run)
    audio = Path(args.audio) if args.audio else None
    if audio is None:
        for name in ("song.wav", "mix.wav"):
            if (run / name).exists():
                audio = run / name
                break
    if audio is None or not audio.exists():
        die("no audio found - run 'song' or 'mix' first, or pass --audio FILE")
    out = Path(args.out) if args.out else run / (audio.stem + ".mp4")
    build_video(audio, out, loop=args.loop, image=args.image, nvenc=args.nvenc,
                reencode=args.reencode, watermark=args.watermark, wm_size=args.wm_size,
                wm_opacity=args.wm_opacity, wm_margin=args.wm_margin, fps=args.fps,
                dry_run=args.dry_run,
                metadata={"title": args.title or audio.stem.replace("_", " ").title(),
                          "comment": "Contains AI-assisted / synthesised music."})
    return run


def cmd_analyze(args):
    from mpipe.stretch import analyze_file
    path = Path(args.file)
    if not path.is_file():
        die(f"file not found: {path}")
    info = analyze_file(path)
    log(f"File:  {path}")
    log(f"Tempo: ~{info['bpm']:.1f} BPM  (confidence {info.get('bpm_confidence', 0):.2f})")
    log(f"Key:   {info['key']}  (confidence {info.get('key_confidence', 0):.2f})")
    log(f"First downbeat at {info.get('downbeat', 0):.2f}s, analysed {info['duration']:.0f}s")
    if args.clap:
        _clap(path)
    log("\nTry:")
    log(f'  python pipeline.py lofi --bpm {info["bpm"]:.0f} --key "{info["key"]}" --count 4')
    return None


def _clap(path):
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        warn("--clap needs: pip install transformers torch")
        return
    from mpipe.audio import load_mono
    labels = ["calm", "sad", "happy", "energetic", "romantic", "nostalgic", "dreamy",
              "dark", "peaceful", "devotional", "sleepy", "tense"]
    clf = hf_pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused")
    mono, _ = load_mono(path, sr=48000, max_seconds=60)
    scores = clf(mono, candidate_labels=[f"{m} music" for m in labels])
    log("Mood:  " + ", ".join(f"{s['label'].replace(' music', '')} {s['score']:.2f}"
                              for s in scores[:4]))


# ---------------------------------------------------------------------------
# check / dedupe / log
# ---------------------------------------------------------------------------

def cmd_check(args):
    from mpipe.fingerprint import Ledger, feature_vector, fingerprint_file, file_sha256
    from mpipe.mastering import YOUTUBE_LUFS, YOUTUBE_PEAK_DB, measure_stream

    targets = []
    if args.file:
        targets = [Path(f) for f in collect_inputs(args.file)]
    else:
        run = resolve_run(args.run)
        for name in ("song.wav", "mix.wav"):
            if (run / name).exists():
                targets.append(run / name)
        if not targets:
            die(f"nothing to check in {run} - run 'song' or 'mix' first")
    ledger = Ledger(args.ledger or LEDGER_PATH)

    overall_ok = True
    for path in targets:
        log(f"\n=== {path.name} ===")
        stats = measure_stream(path)
        problems, notes = [], []

        log(f"Length     {fmt_time(stats['seconds'], stats['seconds'] >= 3600)}")
        log(f"Loudness   {stats['lufs']:.2f} LUFS   "
            f"(YouTube normalises to about {YOUTUBE_LUFS})")
        log(f"True peak  {stats['true_peak_db']:.2f} dBTP  (keep under {YOUTUBE_PEAK_DB})")
        if abs(stats["lufs"] - YOUTUBE_LUFS) > 1.5:
            problems.append(f"loudness is {stats['lufs']:.1f} LUFS; re-run with "
                            f"--lufs {YOUTUBE_LUFS}")
        if stats["true_peak_db"] > YOUTUBE_PEAK_DB + 0.1:
            problems.append(f"true peak {stats['true_peak_db']:.2f} dBTP is above "
                            f"{YOUTUBE_PEAK_DB} - lossy encoding will clip")

        tracklist = path.with_name(path.stem + "_tracklist.txt")
        if tracklist.exists():
            lines = [l for l in tracklist.read_text(encoding="utf-8").splitlines() if l.strip()]
            log(f"Chapters   {len(lines)} in {tracklist.name}")
            if len(lines) < 3:
                problems.append("fewer than 3 chapters - YouTube will not show a chapter list")
            elif not lines[0].startswith(("00:00", "0:00")):
                problems.append("the first chapter must start at 00:00")
        else:
            notes.append(f"no tracklist next to the audio ({tracklist.name})")

        log("\nOriginality")
        provenance = _provenance_for(path)
        for line in provenance["lines"]:
            log(f"  {line}")
        problems.extend(provenance["problems"])

        log("\nChecking against everything you have exported before...")
        pairs, duration = fingerprint_file(path)
        features = feature_vector(path)
        sha = file_sha256(path)
        findings = ledger.check(pairs, sha256=sha, features=features)
        blocking = [f for f in findings if not f.get("advisory")]
        advisory = [f for f in findings if f.get("advisory")]
        for f in blocking[:6]:
            log(f"  ! {f['kind']}: {f['file']} (score {f['score']}, logged {f['added']})")
        for f in advisory[:4]:
            log(f"  - {f['kind']}: {f['file']} (score {f['score']}, logged {f['added']})")
        if blocking:
            problems.append("this is close to something you have already exported - "
                            "re-uploading your own material is the usual reason a "
                            "channel gets claimed by itself")
        elif not findings:
            log(f"  no match against {len(ledger.entries)} previous export(s)")

        if args.log:
            ledger.add(path, pairs, duration, features=features,
                       meta={"lufs": round(stats["lufs"], 2),
                             "true_peak_db": round(stats["true_peak_db"], 2),
                             "provenance": provenance["engine"]})
            ledger.save()
            log(f"  logged in {ledger.path.name}")

        log("")
        for n in notes:
            _bullet("-", n)
        if problems:
            overall_ok = False
            log("Fix before uploading:")
            for p in problems:
                _bullet("!", p)
        else:
            log("No blocking problems found.")

    log("\n" + "-" * 68)
    log("Where this can earn, and the policies that decide it: MONETIZATION.md")
    log("Disclosure: YouTube asks you to tick 'Altered or synthetic content' in")
    log("Studio when the audio is AI-generated or synthesised.  This pipeline's")
    log("output is both.  Ticking it is not a penalty; not ticking it is a policy")
    log("breach.  Nothing here can promise how any platform will treat a file -")
    log("what it does is make sure you are not shipping someone else's audio, or")
    log("your own twice.")
    return None if overall_ok else 0


def _bullet(marker, text, width=70):
    """Wrap a finding so a long policy note stays readable in a terminal."""
    words, line, first = str(text).split(), "", True
    for word in words:
        if len(line) + len(word) + 1 > width:
            log(f"  {marker if first else ' '} {line}")
            line, first = word, False
        else:
            line = f"{line} {word}".strip()
    if line:
        log(f"  {marker if first else ' '} {line}")


def _provenance_for(path):
    """Work out where a file came from by reading its run manifest."""
    lines, problems = [], []
    engine = "unknown"
    run = path.parent
    manifest = read_manifest(run)
    if not manifest.get("tracks") and (run.parent / "manifest.json").exists():
        manifest = read_manifest(run.parent)
    engine = manifest.get("engine", "unknown")
    if engine == "builtin":
        lines.append("source: built-in engine - every sound synthesised from scratch.")
        lines.append("        no samples, no loops, no third-party recordings.")
    elif engine == "lofify":
        sources = sorted({Path(t.get("source", "")).name
                          for t in manifest.get("tracks", []) if t.get("source")})
        lines.append("source: YOUR OWN recordings, reprocessed into lofi.")
        lines.append(f"        built from: {', '.join(sources[:6])}"
                     f"{' ...' if len(sources) > 6 else ''}")
        lines.append("        you attested to holding the rights with --i-own-this.")
        lines.append("        Nothing here can verify that - if any of those inputs")
        lines.append("        is someone else's recording, a lofi edit of it is still")
        lines.append("        theirs, and Content ID matches edited audio.")
        problems.append(
            "YouTube's 'inauthentic content' policy names songs that are only "
            "pitch-shifted or sped up.  That is what lofify does by default, so "
            "raw lofify output is not a finished upload - curate it, arrange it "
            "into a song, and put original visuals on it.  See MONETIZATION.md.")
    elif engine == "ace-step":
        lines.append("source: ACE-Step (local model, text-to-music).")
        if manifest.get("reference"):
            mode = manifest.get("ref_mode", "style")
            if mode == "cover":
                lines.append(f"        COVER of: {manifest['reference']}")
                problems.append("this was built as a cover of an existing recording - "
                                "only upload it if you own that recording or it is "
                                "public domain")
            else:
                lines.append(f"        style reference: {manifest['reference']} "
                             "(feel only, not the recording)")
                lines.append("        make sure you have the rights to the reference audio.")
    else:
        lines.append("source: unknown - no manifest next to this file.")
        lines.append("        if any input was someone else's recording, that is the risk.")
    return {"lines": lines, "problems": problems, "engine": engine}


def cmd_dedupe(args):
    from mpipe.fingerprint import compare_folder
    if args.folder:
        files = collect_inputs(args.folder)
    else:
        run = resolve_run(args.run)
        files = audio_files(run / "mastered") or audio_files(run / "raw")
    if len(files) < 2:
        die("need at least two files to compare")
    log(f"Comparing {len(files)} tracks (threshold {args.threshold})...")
    dupes = compare_folder(files, threshold=args.threshold)
    if not dupes:
        log("No near-duplicates found.")
    else:
        log(f"\n{len(dupes)} pair(s) above the threshold.  Delete one of each pair "
            "before mixing - repeated material makes a long upload look automated.")
    return None


def cmd_models(args):
    """What can generate the music, and what the licence lets you do with it."""
    from mpipe.models import report

    vram = args.vram
    if vram is None:
        from mpipe.doctor import nvidia_smi
        gpus = nvidia_smi()
        if gpus:
            vram = gpus[0]["vram_mb"] / 1024.0
            log(f"Detected {gpus[0]['name']} with {vram:.1f} GB of VRAM\n")
    report(vram_gb=vram, commercial_only=args.commercial_only)
    return None


def cmd_ledger(args):
    from mpipe.fingerprint import Ledger
    ledger = Ledger(args.ledger or LEDGER_PATH)
    if not ledger.entries:
        log(f"No exports logged yet ({ledger.path}).")
        log("Run:  python pipeline.py check FILE --log   after you upload something.")
        return None
    log(f"{len(ledger.entries)} export(s) logged in {ledger.path}\n")
    for entry in ledger.entries:
        meta = entry.get("meta", {})
        log(f"  {entry['added'][:16]}  {fmt_time(entry['duration'], entry['duration'] >= 3600):>8}  "
            f"{meta.get('provenance', '?'):<9} {entry['file']}")
    return None


# ---------------------------------------------------------------------------
# all
# ---------------------------------------------------------------------------

def cmd_all(args):
    import copy

    log("=== generate ===")
    gen = copy.copy(args)
    gen.minutes = args.track_minutes           # per track, not the final song
    gen.bpm = int(args.bpm) if args.bpm else None
    run = cmd_lofi(gen) if args.engine == "builtin" else cmd_generate(gen)

    args.run = str(run)
    args.input = None
    if args.engine != "builtin":
        log("\n=== master ===")
        cmd_master(copy.copy(args))
    log("\n=== song ===")
    song_args = copy.copy(args)
    song_args.out = args.out                   # final song length stays on --minutes/--hours
    cmd_song(song_args)
    if args.loop or args.image:
        log("\n=== video ===")
        video_args = copy.copy(args)
        video_args.audio = None
        video_args.out = None
        cmd_video(video_args)
    log(f"\nAll done. Everything is in {run}")
    log(f"Check it before uploading:  python pipeline.py check --run {run}")
    return run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common(p):
    p.add_argument("--run", help="run folder (default: newest in ./runs)")
    p.add_argument("--samplerate", type=int, default=44100, help="output sample rate")
    p.add_argument("-q", "--quiet", action="store_true", help="less output")
    p.add_argument("--verbose", action="store_true", default=True, help=argparse.SUPPRESS)


def add_lofi_args(p):
    p.add_argument("--style", default="lofi", help="style from presets.json engine.styles")
    p.add_argument("--count", type=int, default=6, help="how many tracks")
    p.add_argument("--minutes", type=float, default=3.0, help="minutes per track")
    p.add_argument("--mood", help="mood preset (calm, rainy, sleepy, ...)")
    p.add_argument("--bpm", type=int, help="fixed tempo")
    p.add_argument("--key", help='fixed key, e.g. "A Minor"')
    p.add_argument("--swing", type=float, help="0 = straight, 0.3 = heavy shuffle")
    p.add_argument("--vinyl", type=float, help="vinyl crackle amount (0 = off)")
    p.add_argument("--tape", type=float, help="tape wow/saturation amount (0 = off)")
    p.add_argument("--seed", type=int, help="make the run repeatable")
    p.add_argument("--lufs", type=float, default=-14.0, help="loudness target")
    p.add_argument("--peak", type=float, default=-1.0, help="peak ceiling in dBFS")
    p.add_argument("--presets", default=str(DEFAULT_PRESETS))


def add_generate_args(p):
    p.add_argument("--genre", default="lofi", help="genre preset from presets.json")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--duration", type=float, help="seconds per track (default 150)")
    p.add_argument("--mood", help="mood preset or free text")
    p.add_argument("--extra", help="extra words added to every caption")
    p.add_argument("--bpm", type=int)
    p.add_argument("--key")
    p.add_argument("--seed", type=int)
    p.add_argument("--reference", help="your own audio to guide the style")
    p.add_argument("--ref-mode", choices=["style", "cover"], default="style")
    p.add_argument("--strength", type=float, default=0.5, help="cover strength 0-1")
    p.add_argument("--match-reference", action="store_true", help="copy tempo/key from --reference")
    p.add_argument("--i-own-this", action="store_true",
                   help="confirm you hold the rights to --reference (required for cover mode)")
    p.add_argument("--model", help="ACE-Step DiT model name")
    p.add_argument("--backend", choices=["server", "comfy"], default="server",
                   help="the standalone ACE-Step API, or ComfyUI")
    p.add_argument("--server", default="http://127.0.0.1:8001")
    p.add_argument("--api-key")
    p.add_argument("--presets", default=str(DEFAULT_PRESETS))
    p.add_argument("--retries", type=int, default=1)


def add_output_args(p):
    p.add_argument("--format", choices=["mp3", "wav", "flac"], default=DEFAULT_FORMAT,
                   help="output format (default mp3)")
    p.add_argument("--mp3-quality", default=320,
                   help="MP3 bitrate (320, 256, 192) or VBR quality (V0, V2)")
    p.add_argument("--artist", help="artist tag written into the files")
    p.add_argument("--album", help="album tag written into the files")
    p.add_argument("--title", help="title tag for the final song/mix")
    p.add_argument("--no-tags", action="store_true", help="do not write metadata")


def add_hf_args(p):
    p.add_argument("--model", default="musicgen-small",
                   help="musicgen-small (default, smallest), musicgen-stereo-small, "
                        "musicgen-medium")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32"],
                   help="auto picks fp32 on 16-series cards; fp16 forces half precision")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--cache-dir", help="where to keep the downloaded weights")
    p.add_argument("--count", type=int, default=4, help="how many tracks")
    p.add_argument("--minutes", type=float, default=2.0, help="minutes per track")
    p.add_argument("--hours", type=float, default=0,
                   help="hours per track (overrides --minutes)")
    p.add_argument("--download", nargs="?", const=True, default=None,
                   metavar="MODEL", help="download weights and exit")
    p.add_argument("--list-models", action="store_true",
                   help="show every variant with download size and VRAM, then exit")
    p.add_argument("--no-resume", action="store_true",
                   help="start fresh instead of continuing an interrupted run")
    p.add_argument("--prompt", help="use this exact prompt for every track")
    p.add_argument("--use-presets", action="store_true",
                   help="build prompts from presets.json instead of the lofi template")
    p.add_argument("--genre", default="lofi", help="preset genre for prompt wording")
    p.add_argument("--mood", help="mood preset or free text")
    p.add_argument("--extra", help="extra words appended to every prompt")
    p.add_argument("--texture", action="store_true",
                   help="ask for vinyl crackle in the prompt (off by default - "
                        "naming it makes the model foreground the artefact)")
    p.add_argument("--bpm", type=int, help="fixed tempo in the prompt")
    p.add_argument("--key", help="fixed key in the prompt")
    p.add_argument("--guidance", type=float, default=3.0, help="classifier-free guidance")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=250)
    p.add_argument("--top-p", type=float, default=0.0)
    p.add_argument("--overlap", type=float, default=5.0,
                   help="seconds fed back when continuing past the model's limit")
    p.add_argument("--seed", type=int)
    p.add_argument("--presets", default=str(DEFAULT_PRESETS))
    p.add_argument("--lufs", type=float, default=-14.0, help="loudness target")
    p.add_argument("--peak", type=float, default=-1.0, help="true-peak ceiling in dBTP")
    p.add_argument("--selftest", action="store_true",
                   help="check this card produces real audio in the chosen precision")
    p.add_argument("--selftest-seconds", type=float, default=3.0)


def add_lofify_args(p):
    p.add_argument("input", nargs="+", help="songs to lofi (files, a folder, or a .txt list)")
    p.add_argument("--preset", default="classic",
                   help="classic, slowed, study, sleep, tape, instrumental")
    p.add_argument("--i-own-this", action="store_true",
                   help="confirm you hold the rights to the input audio (required)")
    p.add_argument("--speed", type=float, help="playback speed (0.88 = the usual slowdown)")
    p.add_argument("--keep-pitch", action="store_true",
                   help="slow it down without dropping the pitch")
    p.add_argument("--semitones", type=float, help="extra transposition")
    p.add_argument("--vocals", choices=["keep", "reduce", "remove"],
                   help="what to do with the lead vocal")
    p.add_argument("--vocal-amount", type=float, help="0-1, how hard to pull it down")
    p.add_argument("--drums", choices=["off", "add"], help="lay a boom-bap kit under it")
    p.add_argument("--drum-level", type=float)
    p.add_argument("--drum-pattern", help="boom_bap, lazy, halftime, swing, shuffle, brushed")
    p.add_argument("--drum-style", choices=["dusty", "soft", "punchy", "brush"])
    p.add_argument("--amount", type=float, help="0-1, overall lofi character")
    p.add_argument("--lowpass", type=float, help="lowpass in Hz (overrides --amount)")
    p.add_argument("--bitcrush", type=int, help="bit depth (overrides --amount)")
    p.add_argument("--vinyl", type=float, help="vinyl crackle bed, 0 = off")
    p.add_argument("--reverb", type=float, help="0-1 room amount")
    p.add_argument("--telephone", type=float, help="0-1 GSM/telephone grit")
    p.add_argument("--mp3-artifacts", type=float, help="0-1 deliberate codec crunch")
    p.add_argument("--lufs", type=float, default=-14.0)
    p.add_argument("--peak", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=0)


def add_comfy_args(p):
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188",
                   help="ComfyUI address")
    p.add_argument("--workflow", help="API-format workflow JSON "
                                      "(name in workflows/, or a path)")
    p.add_argument("--set", action="append", default=[], metavar="NODE.FIELD=VALUE",
                   help="patch any workflow input; NODE is a title, class or id")
    p.add_argument("--steps", type=int, help="sampler steps")
    p.add_argument("--cfg", type=float, help="sampler guidance")
    p.add_argument("--task-timeout", type=int, default=1800)


def add_art_args(p):
    p.add_argument("--prompt", default="cozy lofi study room at night, warm desk lamp, "
                                       "rain on the window, plants, soft anime "
                                       "illustration, muted colours, grain",
                   help="what the artwork should show")
    p.add_argument("--negative", help="what to keep out of it")
    p.add_argument("--count", type=int, default=1, help="how many images")
    p.add_argument("--size", default="1920x1080", help="loop resolution, e.g. 1920x1080")
    p.add_argument("--seed", type=int)
    p.add_argument("--loop-seconds", type=float, default=40.0,
                   help="length of the seamless loop to build (0 = image only)")
    p.add_argument("--zoom", type=float, default=0.12, help="how far the loop drifts in")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--nvenc", action="store_true", help="GPU-encode the loop")
    p.add_argument("--dry-run", action="store_true", help="print the ffmpeg command only")
    p.add_argument("--show-workflow", action="store_true",
                   help="list the workflow's nodes and patchable fields, then exit")


def add_master_args(p):
    p.add_argument("--lufs", type=float, default=-14.0)
    p.add_argument("--peak", type=float, default=-1.0)
    p.add_argument("--fade-in", type=float, default=0.05)
    p.add_argument("--fade-out", type=float, default=0.05)
    p.add_argument("--no-trim", action="store_true", help="keep leading/trailing silence")


def add_song_args(p):
    p.add_argument("--input", nargs="*", help="folder, files, or a .txt playlist")
    p.add_argument("--out", help="output file")
    p.add_argument("--minutes", type=float, default=0, help="target length in minutes")
    p.add_argument("--hours", type=float, default=0, help="target length in hours")
    p.add_argument("--bpm", type=float, help="force the song's tempo")
    p.add_argument("--key", help="force the song's key")
    p.add_argument("--order", choices=["harmonic", "shuffle", "asis"], default="harmonic")
    p.add_argument("--crossfade-bars", type=int, default=4, help="bars of overlap between tracks")
    p.add_argument("--spine", type=float, default=0.0,
                   help="0-1: a continuous drum groove under everything, so the beat never stops")
    p.add_argument("--spine-style", default="dusty", choices=["dusty", "soft", "punchy", "brush"])
    p.add_argument("--spine-pattern", default="lazy")
    p.add_argument("--vinyl", type=float, default=0.8, help="continuous vinyl bed (0 = off)")
    p.add_argument("--tape", type=float, default=0.7, help="master tape character (0 = off)")
    p.add_argument("--width", type=float, default=1.25)
    p.add_argument("--lofi", type=float, default=0.0, help="0-1: extra lofi crunch on each track")
    p.add_argument("--tone-match", type=float, default=0.7,
                   help="0-1: pull every track toward one tonal balance")
    p.add_argument("--max-stretch", type=float, default=18.0, help="max tempo change, percent")
    p.add_argument("--max-shift", type=int, default=4, help="max pitch shift, semitones")
    p.add_argument("--final-fade", type=float, default=12.0)
    p.add_argument("--lufs", type=float, default=-14.0)
    p.add_argument("--peak", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mp3", action="store_true", help=argparse.SUPPRESS)


def add_mix_args(p):
    p.add_argument("--input", nargs="*", help="folder, files, or a .txt playlist")
    p.add_argument("--out", help="output file")
    p.add_argument("--minutes", type=float, default=0)
    p.add_argument("--hours", type=float, default=0)
    p.add_argument("--crossfade", type=float, default=5.0, help="seconds of overlap")
    p.add_argument("--final-fade", type=float, default=8.0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--peak", type=float, default=-1.0)
    p.add_argument("--mp3", action="store_true", help=argparse.SUPPRESS)


def add_video_args(p):
    p.add_argument("--audio", help="audio file (default: song.wav or mix.wav in the run)")
    p.add_argument("--loop", help="looping clip (mp4) to repeat under the audio")
    p.add_argument("--image", help="still image instead of a loop")
    p.add_argument("--watermark", help="PNG burned into the corner (forces re-encoding)")
    p.add_argument("--wm-size", type=int, default=150)
    p.add_argument("--wm-opacity", type=float, default=0.85)
    p.add_argument("--wm-margin", type=int, default=24)
    p.add_argument("--fps", type=int, help="frame rate when using --image")
    p.add_argument("--nvenc", action="store_true", help="use the GPU video encoder")
    p.add_argument("--reencode", action="store_true")
    p.add_argument("--title", help="title written into the file metadata")
    p.add_argument("--dry-run", action="store_true", help="print the ffmpeg command only")
    p.add_argument("--out", help="output video path")


def add_all_args(p):
    """`all` merges three stages, so the overlapping flags get explicit names."""
    p.add_argument("--engine", choices=["builtin", "ace-step"], default="builtin")
    # --- generation ---
    p.add_argument("--style", default="lofi", help="built-in engine style")
    p.add_argument("--genre", default="lofi", help="ACE-Step genre preset")
    p.add_argument("--count", type=int, default=8, help="how many tracks to generate")
    p.add_argument("--track-minutes", type=float, default=3.0,
                   help="minutes per generated track")
    p.add_argument("--duration", type=float, help="ACE-Step: seconds per track")
    p.add_argument("--mood", help="mood preset (calm, rainy, sleepy, ...)")
    p.add_argument("--extra", help="ACE-Step: extra caption words")
    p.add_argument("--swing", type=float)
    p.add_argument("--presets", default=str(DEFAULT_PRESETS))
    p.add_argument("--reference", help="ACE-Step: style reference audio")
    p.add_argument("--ref-mode", choices=["style", "cover"], default="style")
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--match-reference", action="store_true")
    p.add_argument("--i-own-this", action="store_true")
    p.add_argument("--model")
    p.add_argument("--server", default="http://127.0.0.1:8001")
    p.add_argument("--api-key")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--task-timeout", type=int, default=1800)
    p.add_argument("--backend", choices=["server", "comfy"], default="server",
                   help="ACE-Step via its own API, or via ComfyUI")
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    p.add_argument("--workflow")
    p.add_argument("--set", action="append", default=[], metavar="NODE.FIELD=VALUE")
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg", type=float)
    # --- shared musical settings ---
    p.add_argument("--bpm", type=float, help="force the tempo everywhere")
    p.add_argument("--key", help='force the key, e.g. "A Minor"')
    p.add_argument("--seed", type=int)
    p.add_argument("--lufs", type=float, default=-14.0)
    p.add_argument("--peak", type=float, default=-1.0)
    p.add_argument("--vinyl", type=float, default=0.8)
    p.add_argument("--tape", type=float, default=0.7)
    # --- mastering (ACE-Step path) ---
    p.add_argument("--fade-in", type=float, default=0.05)
    p.add_argument("--fade-out", type=float, default=0.05)
    p.add_argument("--no-trim", action="store_true")
    # --- the final song ---
    p.add_argument("--minutes", type=float, default=0, help="final song length in minutes")
    p.add_argument("--hours", type=float, default=0, help="final song length in hours")
    p.add_argument("--order", choices=["harmonic", "shuffle", "asis"], default="harmonic")
    p.add_argument("--crossfade-bars", type=int, default=4)
    p.add_argument("--spine", type=float, default=0.0)
    p.add_argument("--spine-style", default="dusty", choices=["dusty", "soft", "punchy", "brush"])
    p.add_argument("--spine-pattern", default="lazy")
    p.add_argument("--width", type=float, default=1.25)
    p.add_argument("--lofi", type=float, default=0.0)
    p.add_argument("--tone-match", type=float, default=0.7)
    p.add_argument("--max-stretch", type=float, default=18.0)
    p.add_argument("--max-shift", type=int, default=4)
    p.add_argument("--final-fade", type=float, default=12.0)
    p.add_argument("--mp3", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--out", help="final song path")
    # --- video ---
    p.add_argument("--loop", help="looping clip to put under the song")
    p.add_argument("--image", help="still image instead of a loop")
    p.add_argument("--watermark")
    p.add_argument("--wm-size", type=int, default=150)
    p.add_argument("--wm-opacity", type=float, default=0.85)
    p.add_argument("--wm-margin", type=int, default=24)
    p.add_argument("--fps", type=int)
    p.add_argument("--nvenc", action="store_true")
    p.add_argument("--reencode", action="store_true")
    p.add_argument("--dry-run", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Local lofi music pipeline - runs entirely on your own PC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start with:  python pipeline.py doctor")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check this PC and write tuned ACE-Step settings")
    p.add_argument("--write-env", nargs="?", const=".env", default=None,
                   help="write low-VRAM settings to an ACE-Step .env file")
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188",
                   help="where to look for ComfyUI")
    p.set_defaults(func=lambda a: (__import__("mpipe.doctor", fromlist=["check"])
                                   .check(write_env=a.write_env,
                                          comfy_url=a.comfy_url), None)[1])

    p = sub.add_parser("lofi", help="generate tracks with the built-in engine (no GPU)")
    add_common(p)
    add_output_args(p)
    add_lofi_args(p)
    p.set_defaults(func=cmd_lofi)

    p = sub.add_parser("generate", help="generate tracks with ACE-Step (needs the GPU)")
    add_common(p)
    add_output_args(p)
    add_generate_args(p)
    add_comfy_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("hf", help="generate lofi with a local Hugging Face model")
    add_common(p)
    add_hf_args(p)
    add_output_args(p)
    p.set_defaults(func=cmd_hf)

    p = sub.add_parser("lofify", help="turn songs you already have into lofi")
    add_common(p)
    add_lofify_args(p)
    add_output_args(p)
    p.set_defaults(func=cmd_lofify)

    p = sub.add_parser("art", help="make the cover art and a seamless video loop (ComfyUI)")
    add_common(p)
    add_art_args(p)
    add_comfy_args(p)
    p.set_defaults(func=cmd_art, workflow="art")

    p = sub.add_parser("master", help="trim, loudness-normalise and limit each track")
    add_common(p)
    add_output_args(p)
    add_master_args(p)
    p.set_defaults(func=cmd_master)

    p = sub.add_parser("song", help="beat-match tracks into ONE continuous lofi song")
    add_common(p)
    add_output_args(p)
    add_song_args(p)
    p.set_defaults(func=cmd_song)

    p = sub.add_parser("mix", help="classic crossfaded compilation + chapters")
    add_common(p)
    add_output_args(p)
    add_mix_args(p)
    p.set_defaults(func=cmd_mix)

    p = sub.add_parser("video", help="put the audio under a loop or a still image")
    add_common(p)
    add_video_args(p)
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("analyze", help="estimate tempo, key and downbeat of a file")
    p.add_argument("file")
    p.add_argument("--clap", action="store_true", help="also guess mood (large download)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("check", help="originality + upload-readiness report")
    p.add_argument("file", nargs="?", help="file to check (default: the newest run's song/mix)")
    p.add_argument("--run")
    p.add_argument("--log", action="store_true", help="record this export in the ledger")
    p.add_argument("--ledger", help="path to the ledger JSON")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("dedupe", help="find near-duplicate tracks in a folder")
    p.add_argument("folder", nargs="?")
    p.add_argument("--run")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_dedupe)

    p = sub.add_parser("models", help="which generation models fit, and their licences")
    p.add_argument("--vram", type=float, help="VRAM in GB (default: detect)")
    p.add_argument("--commercial-only", action="store_true",
                   help="hide models whose weights forbid commercial use")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("ledger", help="list everything you have exported")
    p.add_argument("--ledger")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_ledger)

    p = sub.add_parser("all", help="generate + song (+ video) in one go")
    add_common(p)
    add_output_args(p)
    add_all_args(p)
    p.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    if getattr(args, "quiet", False):
        set_quiet(True)
        args.verbose = False
    try:
        args.func(args)
    except KeyboardInterrupt:
        log("\nstopped.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
