"""Post-process one ACE-Step take for the Colab notebook, in its own process.

The notebook kernel holds ACE-Step and torch's CUDA libraries.  The session
kept dying the moment step 10 pulled the pipeline's audio code (pedalboard,
pyloudnorm, scipy) into that same process, with no traceback to show for it.
Running that code here instead keeps the kernel alive whatever happens: a
crash in this process comes back to the notebook as an exit code.

    python colab_worker.py TAKE DEST FEATURES [--similar 0.15]

Trims silence at both ends, resamples to 44.1 kHz stereo, masters to
-14 LUFS / -1 dBTP like `pipeline.py hf`, writes DEST, then compares its
chroma signature with every track already in FEATURES.  A new tune is added
to FEATURES; a repeat is deleted.  The last stdout line is a JSON result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpipe.audio import integrated_lufs, resample, true_peak_db, write_audio  # noqa: E402
from mpipe.effects import brickwall  # noqa: E402
from mpipe.fingerprint import feature_distance, feature_vector  # noqa: E402

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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("take")
    ap.add_argument("dest")
    ap.add_argument("features")
    ap.add_argument("--similar", type=float, default=0.15)
    args = ap.parse_args()
    print(json.dumps(process(args.take, args.dest, args.features, args.similar)))


if __name__ == "__main__":
    main()
