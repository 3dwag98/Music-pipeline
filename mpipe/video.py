"""Render the finished audio under a looping video or a still image.

Hours-long uploads are the normal case here, so the defaults avoid re-encoding
the visual whenever possible: looping a short clip with `-c:v copy` turns a
40-minute encode into a 30-second remux.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import soundfile as sf

from .util import check_disk, die, find_ffmpeg, fmt_time, log


def audio_duration(path):
    return sf.info(str(path)).duration


def build_video(audio, out, loop=None, image=None, nvenc=False, reencode=False,
                watermark=None, wm_size=150, wm_opacity=0.85, wm_margin=24,
                fps=None, crf=20, audio_bitrate="320k", dry_run=False,
                metadata=None):
    """ffmpeg command for the final upload.  Returns the path it wrote."""
    ff = find_ffmpeg(required=not dry_run) or "ffmpeg"
    audio = Path(audio)
    if not audio.exists():
        die(f"audio not found: {audio}")
    visual = loop or image
    if not visual or not Path(visual).is_file():
        die("give --loop your_loop.mp4 or --image your_picture.png")
    duration = audio_duration(audio)
    out = Path(out)
    is_image = bool(image)

    # a rough size estimate so we fail before filling the disk, not after
    est = duration * (330_000 / 8)        # audio
    est += duration * ((1_500_000 if (is_image or reencode or watermark) else 400_000) / 8)
    check_disk(out.parent, int(est * 1.2), "the video")

    cmd = [ff, "-y", "-hide_banner", "-loglevel", "warning", "-stats"]
    if is_image:
        cmd += ["-loop", "1", "-framerate", str(fps or 2), "-i", str(visual)]
    else:
        cmd += ["-stream_loop", "-1", "-i", str(visual)]
    audio_idx = 1
    if watermark:
        cmd += ["-i", str(watermark)]
        audio_idx = 2
    cmd += ["-i", str(audio)]

    even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    copy_video = not is_image and not watermark and not reencode
    if watermark:
        cmd += ["-filter_complex",
                f"[0:v]{even}[base];"
                f"[1:v]scale={wm_size}:-1,format=rgba,"
                f"colorchannelmixer=aa={wm_opacity}[wm];"
                f"[base][wm]overlay=W-w-{wm_margin}:H-h-{wm_margin},format=yuv420p[v]",
                "-map", "[v]"]
    else:
        cmd += ["-map", "0:v:0"]
        if not copy_video:
            cmd += ["-vf", f"{even},format=yuv420p"]
    cmd += ["-map", f"{audio_idx}:a:0"]

    if copy_video:
        cmd += ["-c:v", "copy"]
    elif nvenc:
        # NVENC on Turing: p5 is a good quality/speed balance and costs almost
        # no VRAM, but do not run it while ACE-Step has the GPU.
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                "-cq", str(crf + 1), "-b:v", "0", "-bf", "2"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
        if is_image:
            cmd += ["-tune", "stillimage"]
    if is_image:
        cmd += ["-r", str(fps or 2)]

    cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000", "-ac", "2"]
    for key, value in (metadata or {}).items():
        cmd += ["-metadata", f"{key}={value}"]
    cmd += ["-t", f"{duration:.3f}", "-movflags", "+faststart", str(out)]

    log(f"Rendering {fmt_time(duration, duration >= 3600)} of video -> {out}")
    if copy_video:
        log("  (looping the clip without re-encoding - fast)")
    elif nvenc:
        log("  (NVENC: do not run this while ACE-Step is generating - they share the GPU)")
    if dry_run:
        log("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return out
    subprocess.run(cmd, check=True)
    log(f"Video ready: {out}")
    return out


def make_loop(image, out, seconds=40.0, size="1920x1080", fps=30, zoom=0.12,
              nvenc=False, crf=20, dry_run=False):
    """Turn a still image into a seamless, slowly-drifting video loop.

    A multi-hour Ken Burns render would take hours to encode.  Instead this
    makes one short loop whose zoom starts and ends at exactly 1.0 with zero
    velocity (a raised cosine), so it can be repeated forever with `-c:v copy`
    and never shows a seam.
    """
    ff = find_ffmpeg(required=not dry_run) or "ffmpeg"
    image = Path(image)
    if not image.is_file():
        die(f"image not found: {image}")
    out = Path(out)
    try:
        width, height = (int(v) for v in str(size).lower().split("x"))
    except ValueError:
        die(f"--size must look like 1920x1080, got: {size}")
    frames = max(2, int(round(seconds * fps)))

    # zoompan samples the source at output resolution, so upscale first or the
    # zoomed-in frames come out soft
    over = 2.0 + zoom
    prescale = f"scale={int(width * over)}:{int(height * over)}:flags=lanczos"
    # raised cosine: z(0) = z(N) = 1 and dz/dn = 0 at both ends -> seamless
    zexpr = f"1+{zoom}*(0.5-0.5*cos(2*PI*on/{frames}))"
    pan = (f"x='iw/2-(iw/zoom/2)+{int(width * 0.02)}*sin(2*PI*on/{frames})'"
           f":y='ih/2-(ih/zoom/2)'")
    vf = (f"{prescale},zoompan=z='{zexpr}':{pan}:d={frames}:s={width}x{height}:fps={fps},"
          f"format=yuv420p")

    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-stats",
           "-loop", "1", "-i", str(image), "-vf", vf, "-frames:v", str(frames)]
    if nvenc:
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                "-cq", str(crf), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]
    # a keyframe every second keeps the later -stream_loop remux clean
    cmd += ["-g", str(fps), "-pix_fmt", "yuv420p", "-an", str(out)]

    log(f"Rendering a {seconds:g}s seamless loop at {width}x{height} -> {out}")
    if dry_run:
        log("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return out
    subprocess.run(cmd, check=True)
    return out


def tag_audio(src, dest, metadata, bitrate="320k"):
    """Write an MP3/M4A with metadata (title, artist, comment, AI disclosure)."""
    ff = find_ffmpeg()
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    for key, value in metadata.items():
        cmd += ["-metadata", f"{key}={value}"]
    suffix = Path(dest).suffix.lower()
    if suffix == ".mp3":
        cmd += ["-codec:a", "libmp3lame", "-b:a", bitrate]
    elif suffix in (".m4a", ".aac"):
        cmd += ["-codec:a", "aac", "-b:a", bitrate]
    else:
        cmd += ["-c:a", "copy"]
    cmd += [str(dest)]
    subprocess.run(cmd, check=True)
    return Path(dest)
