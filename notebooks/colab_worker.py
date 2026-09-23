"""Out-of-kernel helpers for the Colab notebook.

The notebook kernel holds ACE-Step and torch's CUDA libraries.  On some Colab
machines a compiled library (pedalboard, first) dies with SIGILL - an
instruction the CPU does not have - which killed the whole session with no
traceback.  Everything that touches the pipeline's audio stack runs here, in a
child process, so a crash comes back to the notebook as an exit code.

    python colab_worker.py process TAKE DEST FEATURES [--similar 0.15]
    python colab_worker.py pipeline song --run RUN ...      (any pipeline.py command)

Set COLAB_BLOCK_MODULES=pedalboard,numba to hide modules that crash on this CPU;
the pipeline then takes its built-in fallbacks (numpy limiter, built-in
stretcher, soundfile for WAV).

`process` trims silence at both ends, resamples to 44.1 kHz stereo, masters to
-14 LUFS / -1 dBTP like `pipeline.py hf`, writes DEST, then compares its chroma
signature with every track already in FEATURES.  A new tune is added to
FEATURES; a repeat is deleted.  The last stdout line is a JSON result.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must happen before anything imports the blocked modules.  A None entry in
# sys.modules makes `import x` raise ImportError, which the pipeline catches.
for _mod in filter(None, os.environ.get("COLAB_BLOCK_MODULES", "").split(",")):
    sys.modules[_mod.strip()] = None
    sys.modules[_mod.strip() + ".io"] = None

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import argparse  # noqa: E402
import json  # noqa: E402
import runpy  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SR = 44100
LUFS, PEAK_DB = -14.0, -1.0


def trim_edges(x, sr, floor_db=-50.0, pad_s=0.05):
    """Cut silence at the head and tail so no join has a gap in it."""
    mono = np.abs(x).mean(axis=1)
    hop = 1024
    n = len(mono) // hop
    if n < 2:
        return x
    rms = np.sqrt((mono[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    loud = np.where(20 * np.log10(rms) > floor_db)[0]
    if not len(loud):
        return x
    pad = int(pad_s * sr)
    return x[max(0, loud[0] * hop - pad): min(len(x), (loud[-1] + 1) * hop + pad)]


def master(x, sr):
    from mpipe.audio import integrated_lufs
    from mpipe.effects import brickwall
    measured = integrated_lufs(x, sr)
    if np.isfinite(measured):
        x = x * (10 ** ((LUFS - measured) / 20.0))
    x = brickwall(x, sr, ceiling_db=PEAK_DB, true_peak=True)
    after = integrated_lufs(x, sr)
    if np.isfinite(after) and abs(LUFS - after) > 0.4:
        x = brickwall(x * (10 ** ((LUFS - after) / 20.0)), sr, ceiling_db=PEAK_DB, true_peak=True)
        after = integrated_lufs(x, sr)
    return x, after


def process(take, dest, features_path, similar):
    from mpipe.audio import resample, true_peak_db, write_audio
    from mpipe.fingerprint import feature_distance, feature_vector

    take, dest, features_path = Path(take), Path(dest), Path(features_path)
    audio, sr = sf.read(take, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all() or np.abs(audio).max() < 1e-3:
        return {"status": "broken", "why": "NaN or silence in the model output"}

    audio = trim_edges(audio, sr)
    if sr != SR:
        audio = resample(audio, sr, SR)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    audio, lufs = master(audio, SR)
    write_audio(dest, audio, SR)

    features = json.loads(features_path.read_text()) if features_path.exists() else {}
    fv = feature_vector(dest)
    if fv is None:
        dest.unlink(missing_ok=True)
        return {"status": "rejected", "why": "no usable signature"}
    closest, twin = min(((feature_distance(fv, v), name) for name, v in features.items()),
                        default=(1.0, None))
    if closest <= similar:
        dest.unlink(missing_ok=True)
        return {"status": "rejected", "why": f"same tune as {twin} (distance {closest:.3f})"}

    features[dest.name] = fv
    tmp = features_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(features))
    tmp.replace(features_path)
    return {"status": "ok", "seconds": round(len(audio) / SR, 2), "lufs": round(float(lufs), 2),
            "true_peak_db": round(float(true_peak_db(audio, SR)), 2),
            "closest": closest if twin else None, "closest_to": twin}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "pipeline":
        # run pipeline.py with the same module blocks in place
        sys.argv = [str(REPO / "pipeline.py")] + sys.argv[2:]
        os.chdir(REPO)
        runpy.run_path(str(REPO / "pipeline.py"), run_name="__main__")
        return
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process")
    p.add_argument("take")
    p.add_argument("dest")
    p.add_argument("features")
    p.add_argument("--similar", type=float, default=0.15)
    args = ap.parse_args()
    print(json.dumps(process(args.take, args.dest, args.features, args.similar)))


if __name__ == "__main__":
    main()
