"""`pipeline.py doctor` - check this machine and write the tuned ACE-Step config.

Targets the exact setup this pipeline was built for: Windows, 16 GB RAM, and a
GTX 1660 Ti with 6 GB of VRAM.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .acestep import LOW_VRAM_ENV
from .comfy import LOW_VRAM_FLAGS
from .util import human_size, log, warn

#: Turing TU116/TU117 (GTX 16-series) have no tensor cores and a well-known
#: half-precision path that produces NaNs/silence in diffusion models.  Full
#: precision is slower but is the difference between music and noise.
FP16_PROBLEM_CARDS = ("1650", "1660", "1630")


def nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                gpus.append({"name": parts[0], "vram_mb": int(float(parts[1])),
                             "used_mb": int(float(parts[2])), "driver": parts[3]})
            except ValueError:
                continue
    return gpus or None


def system_ram_gb():
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        pass
    try:
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MemStatus()
        stat.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 1e9
    except Exception:
        return 0.0


def probe_comfy(url="http://127.0.0.1:8188", timeout=4):
    """Is a ComfyUI server up?  Returns a short description or None."""
    try:
        import requests
    except ImportError:
        return None
    try:
        stats = requests.get(f"{url.rstrip('/')}/system_stats", timeout=timeout).json()
    except Exception:
        return None
    devices = stats.get("devices") or []
    version = (stats.get("system") or {}).get("comfyui_version", "?")
    if devices:
        dev = devices[0]
        return (f"ComfyUI {version} on {dev.get('name', 'device')} "
                f"({dev.get('vram_free', 0) / 1e9:.1f}/"
                f"{dev.get('vram_total', 0) / 1e9:.1f} GB VRAM free)")
    return f"ComfyUI {version}"


def check(write_env=None, verbose=True, comfy_url="http://127.0.0.1:8188"):
    """Run every check.  Returns (report dict, list of problems)."""
    report, problems, notes = {}, [], []

    report["platform"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    report["python"] = sys.version.split()[0]
    if sys.version_info < (3, 9):
        problems.append(f"Python {report['python']} is too old - install Python 3.11")

    ram = system_ram_gb()
    report["ram_gb"] = round(ram, 1) if ram else "unknown"
    if ram and ram < 7.5:
        problems.append(f"only {ram:.1f} GB of RAM - hours-long renders will swap")
    elif ram and ram < 15:
        notes.append(f"{ram:.1f} GB RAM: fine, but close browsers before long renders")

    for module, why in (("numpy", "required"), ("scipy", "required"),
                        ("soundfile", "required"), ("pyloudnorm", "required"),
                        ("requests", "only for the ACE-Step backend"),
                        ("numba", "optional, makes mastering ~2x faster"),
                        ("librosa", "optional, second opinion on tempo/key")):
        try:
            __import__(module)
            report[f"py:{module}"] = "ok"
        except ImportError:
            report[f"py:{module}"] = "missing"
            if why == "required":
                problems.append(f"missing Python package '{module}' - run: pip install -r requirements.txt")
            else:
                notes.append(f"'{module}' not installed ({why})")

    ffmpeg = shutil.which("ffmpeg")
    report["ffmpeg"] = ffmpeg or "missing"
    if not ffmpeg:
        notes.append("ffmpeg not on PATH - WAV/FLAC still work, but MP3 and video need it "
                     "(Windows: winget install Gyan.FFmpeg, then open a NEW terminal)")

    gpus = nvidia_smi()
    report["gpu"] = gpus or "none detected"
    tuned = dict(LOW_VRAM_ENV)
    if gpus:
        gpu = gpus[0]
        free_mb = gpu["vram_mb"] - gpu["used_mb"]
        report["vram_mb"] = gpu["vram_mb"]
        report["vram_free_mb"] = free_mb
        if any(tag in gpu["name"] for tag in FP16_PROBLEM_CARDS):
            notes.append(f"{gpu['name']}: GTX 16-series cards produce silence or noise in "
                         "half precision. The written config forces float32 - slower, but correct.")
            notes.append("  The same applies to ComfyUI: start it with "
                         f"{' '.join(LOW_VRAM_FLAGS)}")
            tuned["ACESTEP_TORCH_DTYPE"] = "float32"
        if gpu["vram_mb"] <= 6300:
            notes.append(f"{gpu['vram_mb']} MB VRAM: keep --count batches small, "
                         "--duration at 120-150s, and batch_size at 1.")
        if free_mb < 4000:
            problems.append(f"only {free_mb} MB VRAM free - close games, browsers with video, "
                            "and OBS before generating")
    else:
        notes.append("No NVIDIA GPU detected. The built-in engine ('pipeline.py lofi') "
                     "runs entirely on the CPU and does not need one.")

    comfy = probe_comfy(comfy_url)
    report["comfyui"] = comfy or "not running"
    if comfy:
        notes.append(f"{comfy} - `pipeline.py art` and `generate --backend comfy` "
                     "can use it.")
    else:
        notes.append("ComfyUI not running (optional). It makes the cover art and the "
                     "video loop, and can host ACE-Step. Start it with: "
                     f"python main.py {' '.join(LOW_VRAM_FLAGS)}")

    try:
        free = shutil.disk_usage(str(Path.cwd())).free
        report["disk_free"] = human_size(free)
        if free < 20e9:
            notes.append(f"only {human_size(free)} free here - a 3-hour 24-bit WAV is about 3 GB")
    except OSError:
        pass

    if write_env:
        path = Path(write_env)
        lines = [f"{k}={v}" for k, v in tuned.items()]
        existing = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, _, v = line.partition("=")
                    existing[k.strip()] = v.strip()
            path.with_suffix(path.suffix + ".bak").write_text(
                path.read_text(encoding="utf-8"), encoding="utf-8")
        existing.update(tuned)
        path.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n",
                        encoding="utf-8")
        report["wrote_env"] = str(path)

    if verbose:
        log("System")
        for key in ("platform", "python", "ram_gb", "disk_free", "ffmpeg"):
            if key in report:
                log(f"  {key:<14} {report[key]}")
        log(f"\nComfyUI       {report['comfyui']}")
        log("\nGPU")
        if gpus:
            for gpu in gpus:
                log(f"  {gpu['name']}  {gpu['vram_mb']} MB VRAM "
                    f"({gpu['vram_mb'] - gpu['used_mb']} MB free), driver {gpu['driver']}")
        else:
            log("  none detected (the built-in engine does not need one)")
        log("\nPython packages")
        for key, value in report.items():
            if key.startswith("py:"):
                log(f"  {key[3:]:<12} {value}")
        if write_env:
            log(f"\nWrote tuned ACE-Step settings to {report['wrote_env']}")
            for key, value in tuned.items():
                log(f"  {key}={value}")
        if notes:
            log("\nNotes")
            for note in notes:
                log(f"  - {note}")
        if problems:
            log("\nProblems")
            for problem in problems:
                log(f"  ! {problem}")
        else:
            log("\nNo blocking problems found.")
    report["notes"] = notes
    report["problems"] = problems
    return report, problems
