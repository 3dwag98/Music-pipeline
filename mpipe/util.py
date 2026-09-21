"""Small shared helpers: logging, paths, run folders, manifests."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RUNS_DIR = HERE / "runs"
DEFAULT_PRESETS = HERE / "presets.json"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".aif"}

_QUIET = False


def set_quiet(flag: bool) -> None:
    global _QUIET
    _QUIET = bool(flag)


def log(msg: str = "") -> None:
    if not _QUIET:
        print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", flush=True)


def die(msg: str):
    print(f"ERROR: {msg}", flush=True)
    sys.exit(1)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "track"


def fmt_time(seconds: float, force_hours: bool = False) -> str:
    seconds = int(round(max(0.0, seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h or force_hours:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


# ---------------------------------------------------------------- presets ---

def load_presets(path=None) -> dict:
    path = Path(path or DEFAULT_PRESETS)
    if not path.exists():
        die(f"presets file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------- runs ---

def new_run(tag: str) -> Path:
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{slug(tag)}" if tag else "")
    run = RUNS_DIR / name
    run.mkdir(parents=True, exist_ok=True)
    return run


def resolve_run(run_arg=None) -> Path:
    if run_arg:
        run = Path(run_arg)
        if not run.is_dir():
            # allow a bare folder name inside runs/
            alt = RUNS_DIR / str(run_arg)
            if alt.is_dir():
                return alt
            die(f"run folder not found: {run}")
        return run
    if not RUNS_DIR.is_dir():
        die("no runs yet - start with 'python pipeline.py lofi' or 'python pipeline.py generate'")
    runs = sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not runs:
        die("no runs yet - start with 'python pipeline.py lofi' or 'python pipeline.py generate'")
    return runs[-1]


def read_manifest(run: Path) -> dict:
    path = Path(run) / "manifest.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"tracks": []}


def write_manifest(run: Path, manifest: dict) -> None:
    tmp = Path(run) / "manifest.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp.replace(Path(run) / "manifest.json")


def audio_files(folder) -> list:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXTS and p.is_file())


def collect_inputs(spec) -> list:
    """Accept a folder, a single file, a glob, or a .txt playlist -> list of audio paths."""
    out = []
    for item in (spec if isinstance(spec, (list, tuple)) else [spec]):
        p = Path(str(item))
        if p.is_dir():
            out.extend(audio_files(p))
        elif p.is_file() and p.suffix.lower() == ".txt":
            base = p.parent
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip().strip('"')
                if not line or line.startswith("#"):
                    continue
                q = Path(line)
                q = q if q.is_absolute() else (base / q)
                if q.is_file():
                    out.append(q)
                else:
                    warn(f"playlist entry not found: {line}")
        elif p.is_file():
            out.append(p)
        else:
            matches = sorted(Path(".").glob(str(item)))
            if matches:
                out.extend(m for m in matches if m.suffix.lower() in AUDIO_EXTS)
            else:
                warn(f"input not found: {item}")
    # de-duplicate, keep order
    seen, unique = set(), []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ----------------------------------------------------------------- ffmpeg ---

def find_ffmpeg(required: bool = True):
    exe = shutil.which("ffmpeg")
    if not exe and required:
        die("ffmpeg not found on PATH.  Windows: winget install Gyan.FFmpeg  "
            "(then open a NEW terminal so PATH refreshes)")
    return exe


def ffmpeg_ok() -> bool:
    return shutil.which("ffmpeg") is not None


def run_ffmpeg(args, check=True):
    exe = find_ffmpeg()
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-stats", *args]
    return subprocess.run(cmd, check=check)


def free_disk_bytes(path) -> int:
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return 0


def check_disk(path, needed_bytes: int, label: str = "output") -> None:
    free = free_disk_bytes(path)
    if free and free < needed_bytes:
        die(f"not enough free disk for {label}: need about {human_size(needed_bytes)}, "
            f"{human_size(free)} free on {Path(path).drive or Path(path).anchor or path}")


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}
