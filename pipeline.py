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
from datetime import datetime
from pathlib import Path

from mpipe import __version__
from mpipe.util import (DEFAULT_PRESETS, RUNS_DIR, audio_files, collect_inputs,
                        die, ffmpeg_ok, fmt_time, load_presets, log, new_run,
                        read_manifest, resolve_run, set_quiet, slug, warn,
                        write_manifest)

LEDGER_PATH = Path(__file__).resolve().parent / "upload_ledger.json"


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
        stem = raw / f"{i:02d}_{slug(spec.title)}.wav"
        log(f"[{i}/{args.count}] {spec.title}  |  {spec.bpm:.0f} BPM, {spec.key}, "
            f"{args.minutes:g} min, {spec.drum_pattern}/{spec.chord_instrument}")
        try:
            report = render_song(spec, stem, minutes=args.minutes, sr=args.samplerate,
                                 progress=args.verbose, peak_db=args.peak)
        except KeyboardInterrupt:
            write_manifest(run, manifest)
            die("stopped by user (finished tracks are kept)")
        if args.lufs is not None:
            stats = measure_stream(stem)
            normalise_stream(stem, stem.with_suffix(".tmp.wav"), lufs=args.lufs,
                             peak_db=args.peak, report=stats)
            stem.with_suffix(".tmp.wav").replace(stem)
            report["lufs"] = args.lufs
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
        report = master_file(path, out_dir / f"{path.stem}.wav", lufs=args.lufs,
                             peak_db=args.peak, fade_in=args.fade_in,
                             fade_out=args.fade_out, trim=not args.no_trim)
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
    from mpipe.mastering import measure_stream, normalise_stream
    from mpipe.song import build_song, write_tracklist

    run = resolve_run(args.run) if (args.run or not args.input) else new_run("song")
    files = _inputs_for(args, run)
    out = Path(args.out) if args.out else run / "song.wav"
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
        spine_style=args.spine_style, spine_pattern=args.spine_pattern)

    if abs(report["lufs"] - args.lufs) > 0.5:
        log(f"\nNormalising {report['lufs']} -> {args.lufs} LUFS...")
        tmp = out.with_suffix(".norm.wav")
        result = normalise_stream(out, tmp, lufs=args.lufs, peak_db=args.peak)
        tmp.replace(out)
        report["lufs"] = result["lufs_out"]
        report["peak_dbfs"] = result["peak_dbfs"]

    tracklist = out.with_name(out.stem + "_tracklist.txt")
    lines = write_tracklist(tracklist, report["chapters"], report["seconds"])
    _write_report(run, "song", report)

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
    out = Path(args.out) if args.out else run / "mix.wav"
    minutes = _minutes_from(args)

    report = build_mix(files, out, minutes=minutes, sr=args.samplerate,
                       crossfade=args.crossfade, final_fade=args.final_fade,
                       shuffle=args.shuffle, seed=args.seed or 0,
                       titles=_titles_from(run), progress=args.verbose,
                       peak_db=args.peak)
    tracklist = out.with_name(out.stem + "_tracklist.txt")
    lines = write_tracklist(tracklist, report["chapters"], report["seconds"])
    _write_report(run, "mix", report)

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


def _maybe_mp3(args, out):
    if not getattr(args, "mp3", False):
        return
    if not ffmpeg_ok():
        warn("--mp3 needs ffmpeg on PATH; skipping (the WAV is fine to upload)")
        return
    from mpipe.video import tag_audio
    dest = out.with_suffix(".mp3")
    tag_audio(out, dest, {"title": out.stem.replace("_", " ").title(),
                          "genre": "Lofi Hip Hop",
                          "comment": "Generated locally. Contains AI-assisted / "
                                     "synthesised music."})
    log(f"MP3:      {dest}")


def _write_report(run, name, report):
    payload = {k: v for k, v in report.items() if k != "chapters"}
    payload["chapters"] = [[round(t, 2), n] for t, n in report.get("chapters", [])]
    path = Path(run) / f"{name}_report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
        if problems:
            overall_ok = False
            log("Fix before uploading:")
            for p in problems:
                log(f"  ! {p}")
        else:
            log("No blocking problems found.")
        for n in notes:
            log(f"  - {n}")

    log("\n" + "-" * 68)
    log("Disclosure: YouTube asks you to tick 'Altered or synthetic content' in")
    log("Studio when the audio is AI-generated or synthesised.  This pipeline's")
    log("output is both.  Ticking it is not a penalty; not ticking it is a policy")
    log("breach.  Nothing here can promise how any platform will treat a file -")
    log("what it does is make sure you are not shipping someone else's audio, or")
    log("your own twice.")
    return None if overall_ok else 0


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
    p.add_argument("--server", default="http://127.0.0.1:8001")
    p.add_argument("--api-key")
    p.add_argument("--presets", default=str(DEFAULT_PRESETS))
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--task-timeout", type=int, default=1800)


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
    p.add_argument("--mp3", action="store_true", help="also write an MP3 (needs ffmpeg)")


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
    p.add_argument("--mp3", action="store_true")


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
    p.add_argument("--mp3", action="store_true")
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
    p.add_argument("--title")
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
    p.set_defaults(func=lambda a: (__import__("mpipe.doctor", fromlist=["check"])
                                   .check(write_env=a.write_env), None)[1])

    p = sub.add_parser("lofi", help="generate tracks with the built-in engine (no GPU)")
    add_common(p)
    add_lofi_args(p)
    p.set_defaults(func=cmd_lofi)

    p = sub.add_parser("generate", help="generate tracks with ACE-Step (needs the GPU)")
    add_common(p)
    add_generate_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("master", help="trim, loudness-normalise and limit each track")
    add_common(p)
    add_master_args(p)
    p.set_defaults(func=cmd_master)

    p = sub.add_parser("song", help="beat-match tracks into ONE continuous lofi song")
    add_common(p)
    add_song_args(p)
    p.set_defaults(func=cmd_song)

    p = sub.add_parser("mix", help="classic crossfaded compilation + chapters")
    add_common(p)
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

    p = sub.add_parser("ledger", help="list everything you have exported")
    p.add_argument("--ledger")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_ledger)

    p = sub.add_parser("all", help="generate + song (+ video) in one go")
    add_common(p)
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
