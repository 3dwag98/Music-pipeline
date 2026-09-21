"""Audio fingerprinting, duplicate detection and an upload log.

Why this exists: when a channel's uploads get claimed, it is almost never a
mystery third party - it is usually the channel's own material coming back
(the same bed re-uploaded, a track a distributor already registered, or a
"cover" of someone else's song).  This module gives the pipeline a memory:
every exported song is fingerprinted and logged, and `check_new` refuses to
let you ship something you have effectively shipped before.

The fingerprint is a constellation hash (spectral peak pairs), the same shape
of idea audio-matching services use.  It is for your own library hygiene - it
cannot tell you what any particular platform will do.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import load_mono
from .util import log, warn

FP_SR = 11025
FRAME = 1024
HOP = 512
FAN = 3
#: logarithmic bands - one peak per band per frame keeps the constellation
#: sparse (~20-40 peaks/second, like a real audio matcher) instead of dense,
#: which is what makes unrelated tracks actually score as unrelated.
BANDS = [(1, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 320), (320, 512)]
MAX_DT = 63          # frames between paired peaks
MAX_DF = 63          # bins between paired peaks


def _spectral_peaks(mono, sr):
    """Sparse constellation: ONE peak per frame, the strongest band maximum.

    Density is the whole ballgame here.  With several peaks per frame,
    unrelated lofi tracks - same timbre, same steady grid - line up at bar
    multiples and score ~0.4 against each other.  At one peak per frame
    (~20 peaks/second, the density real audio matchers use) that drops to
    ~0.2 while an identical file still scores exactly 1.0.
    """
    n = (len(mono) - FRAME) // HOP
    if n < 4:
        return np.zeros((0, 2), dtype=np.int64)
    idx = np.arange(n)[:, None] * HOP + np.arange(FRAME)[None, :]
    frames = mono[idx] * np.hanning(FRAME)[None, :]
    spec = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-9)

    floor = spec.mean() + 0.8 * spec.std()
    best_bin = np.full(n, -1, dtype=np.int64)
    best_val = np.full(n, -np.inf)
    for lo, hi in BANDS:
        hi = min(hi, spec.shape[1])
        if hi <= lo:
            continue
        band = spec[:, lo:hi]
        arg = band.argmax(axis=1)
        val = band[np.arange(n), arg]
        better = val > best_val
        best_val = np.where(better, val, best_val)
        best_bin = np.where(better, arg + lo, best_bin)
    keep = (best_val > floor) & (best_bin >= 0)
    rows = np.where(keep)[0]
    if len(rows) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([rows, best_bin[rows]], axis=1)


def _pack(f1, df, dt):
    """(anchor bin, delta bin, delta frames) -> one 32-bit hash."""
    return ((int(f1) & 0x1FF) << 16) | ((int(df) + MAX_DF) & 0x7F) << 8 | (int(dt) & 0x3F)


def fingerprint_array(mono, sr, max_pairs=200000):
    """Constellation hashes as (hash, anchor_frame) pairs.

    The anchor time is kept because matching by *time-aligned* hash count is far
    more discriminative than set overlap: a real match lines up at one constant
    offset, coincidental collisions scatter across every offset.
    """
    peaks = _spectral_peaks(mono, sr)
    pairs = []
    n = len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        fanned = 0
        for j in range(i + 1, n):
            t2, f2 = peaks[j]
            dt = int(t2 - t1)
            if dt <= 0:
                continue
            if dt > MAX_DT:
                break
            df = int(f2) - int(f1)
            if abs(df) > MAX_DF:
                continue
            pairs.append((_pack(f1, df, dt), int(t1)))
            fanned += 1
            if fanned >= FAN:
                break
        if len(pairs) >= max_pairs:
            break
    # de-duplicate: the same (hash, anchor) twice adds nothing but skew
    return sorted(set(pairs))


def _index(pairs):
    table = {}
    for h, t in pairs:
        table.setdefault(h, set()).add(t)
    return table


def align_score(pairs_a, pairs_b):
    """Fraction of A's anchor points that match B at one consistent time offset.

    Counting DISTINCT anchors (not raw hash collisions) keeps the score in
    [0, 1]: a repeated hash at the same offset can otherwise be counted many
    times over and push an identical file above 1.0.
    """
    if not pairs_a or not pairs_b:
        return 0.0
    table = _index(pairs_b)
    anchors_a = {t for _, t in pairs_a}
    if not anchors_a:
        return 0.0
    by_offset = {}
    for h, t in pairs_a:
        for t2 in table.get(h, ()):
            by_offset.setdefault(t2 - t, set()).add(t)
    if not by_offset:
        return 0.0
    # allow +/-1 frame of jitter, unioning the anchor sets so nothing double-counts
    merged = {}
    for off, anchors in by_offset.items():
        merged[off] = len(anchors | by_offset.get(off - 1, set())
                          | by_offset.get(off + 1, set()))
    counts = np.array(sorted(merged.values(), reverse=True), dtype=float)
    best = counts[0]
    # Subtract the background: a real match is ONE tall spike in the offset
    # histogram, while two unrelated tracks with a steady grid match a little
    # at every bar-multiple offset.  Taking the spike's prominence instead of
    # its raw height is what separates the two.
    background = counts[1:].mean() if len(counts) > 1 else 0.0
    return float(np.clip((best - background) / len(anchors_a), 0.0, 1.0))


def similarity(a, b):
    """Symmetric match score between two fingerprints (0 unrelated, 1 identical)."""
    return round(max(align_score(a, b), align_score(b, a)), 4)


def containment(a, b):
    """How much of the SHORTER print appears inside the longer one.

    This is what catches a 3-minute track dropped whole into a 3-hour mix,
    where an overall similarity would be diluted away.  Note it matches only
    unmodified copies: a track that was time-stretched or pitch-shifted on its
    way into the mix will NOT line up - use `feature_distance` for those.
    """
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return round(align_score(small, large), 4)


CHROMA_HOP = 4096
CHROMA_SR = 11025


def chroma_sequence(mono, sr, frame=8192, hop=CHROMA_HOP):
    """Per-frame pitch-class profile, L2-normalised: the shape of the harmony
    over time rather than averaged away."""
    n = (len(mono) - frame) // hop
    if n < 4:
        return np.zeros((0, 12), dtype=np.float32)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    valid = (freqs > 55.0) & (freqs < 2200.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(np.where(freqs > 0, freqs, 1e-9) / 440.0)
    pcs = np.zeros(freqs.shape, dtype=np.int64)
    pcs[valid] = np.round(midi[valid]).astype(np.int64) % 12
    masks = [valid & (pcs == pc) for pc in range(12)]

    idx = np.arange(n)[:, None] * hop + np.arange(frame)[None, :]
    mags = np.abs(np.fft.rfft(mono[idx] * np.hanning(frame)[None, :], axis=1)) ** 2
    out = np.stack([mags[:, m].sum(axis=1) for m in masks], axis=1)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, 1e-12)).astype(np.float32)


def feature_vector(path, max_seconds=240.0):
    """Musical signature that survives time-stretching and transposition.

    The constellation fingerprint deliberately does not match edited audio, so
    this covers the other case: the same composition re-stretched or moved to
    another key.  It stores the chroma SEQUENCE, not an average - two different
    tracks in the same genre have near-identical average chroma, so an averaged
    profile cannot tell them apart at all.
    """
    from .stretch import detect_bpm
    mono, sr = load_mono(path, sr=CHROMA_SR, max_seconds=max_seconds)
    if len(mono) < sr:
        return None
    seq = chroma_sequence(mono, sr)
    if len(seq) < 8:
        return None
    bpm, _ = detect_bpm(mono, sr)
    # keep it small enough to sit in a JSON ledger
    step = max(1, len(seq) // 512)
    seq = seq[::step]
    return {"bpm": bpm, "hop_s": CHROMA_HOP * step / CHROMA_SR,
            "seq": [[round(float(v), 4) for v in frame] for frame in seq]}


def _resample_seq(seq, factor):
    """Stretch a chroma sequence along time by `factor` (linear interpolation)."""
    n = len(seq)
    out_n = max(1, int(round(n * factor)))
    src = np.linspace(0.0, n - 1.0, out_n)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = (src - i0).astype(np.float32)[:, None]
    out = seq[i0] * (1.0 - frac) + seq[i1] * frac
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, 1e-12)).astype(np.float32)


def feature_distance(a, b):
    """0 = the same composition (any key, any tempo), 1 = unrelated.

    Scores the best diagonal of the cross-similarity matrix over all 12 key
    rotations: the same music lines up along one diagonal, different music
    does not line up anywhere.
    """
    if not a or not b or not a.get("seq") or not b.get("seq"):
        return 1.0
    A = np.asarray(a["seq"], dtype=np.float32)
    B = np.asarray(b["seq"], dtype=np.float32)
    if len(A) < 8 or len(B) < 8:
        return 1.0
    # Put both on the same beat-relative time axis first.  A track that was
    # stretched 8% drifts a whole frame out of step within a few bars, and the
    # diagonal falls apart even though the music is identical.
    # beats covered by one chroma frame, so the grids become tempo-independent
    span_a = a.get("hop_s", 0.37) * (a.get("bpm") or 80.0) / 60.0
    span_b = b.get("hop_s", 0.37) * (b.get("bpm") or 80.0) / 60.0
    if span_a > 0 and span_b > 0 and abs(span_a / span_b - 1.0) > 0.005:
        B = _resample_seq(B, span_b / span_a)
        if len(B) < 8:
            return 1.0
    if len(A) > len(B):
        A, B = B, A
    best = 0.0
    for rot in range(12):
        Br = np.roll(B, rot, axis=1)
        # the diagonal sums of A @ Br.T for every lag are exactly the sum of
        # the per-pitch-class cross-correlations, so 12 correlations do it all
        total = None
        for k in range(12):
            corr = np.correlate(Br[:, k], A[:, k], mode="valid")
            total = corr if total is None else total + corr
        if total is not None and len(total):
            best = max(best, float(total.max()) / len(A))
    return round(float(np.clip(1.0 - best, 0.0, 1.0)), 4)


def fingerprint_file(path, max_seconds=900.0, sample_windows=4):
    """Fingerprint a file; long files are sampled at several points so checking
    a 6-hour mix does not cost more than rendering it."""
    info = sf.info(str(path))
    duration = info.duration
    if duration <= max_seconds:
        mono, sr = load_mono(path, sr=FP_SR)
        return fingerprint_array(mono, sr), duration

    from .audio import resample
    window = max_seconds / sample_windows
    pairs = []
    for k in range(sample_windows):
        start = duration * (k + 0.5) / sample_windows - window / 2
        start = max(0.0, min(duration - window, start))
        with sf.SoundFile(str(path)) as fh:
            fh.seek(int(start * fh.samplerate))
            data = fh.read(int(window * fh.samplerate), dtype="float32", always_2d=True)
        if len(data) == 0:
            continue
        mono = resample(data.mean(axis=1)[:, None], info.samplerate, FP_SR)[:, 0]
        offset = int(start * FP_SR / HOP)
        pairs.extend((h, t + offset) for h, t in fingerprint_array(mono, FP_SR))
    return pairs, duration


def file_sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- the ledger ---

class Ledger:
    """A JSON record of everything this pipeline has exported for upload."""

    def __init__(self, path):
        self.path = Path(path)
        self.entries = []
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.entries = data.get("entries", [])
            except (OSError, json.JSONDecodeError):
                warn(f"could not read {self.path}, starting a fresh ledger")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "updated": datetime.now().isoformat(timespec="seconds"),
                   "entries": self.entries}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, path, pairs, duration, meta=None, features=None):
        entry = {
            "file": str(Path(path).name),
            "path": str(Path(path).resolve()),
            "sha256": file_sha256(path),
            "duration": round(duration, 2),
            "added": datetime.now().isoformat(timespec="seconds"),
            "hash_count": len(pairs),
            # cap so the ledger stays small; an evenly-spread sample keeps the
            # time alignment meaningful instead of only covering the intro
            "pairs": [list(p) for p in pairs[:: max(1, len(pairs) // 24000)]][:24000],
            "features": features,
            "meta": meta or {},
        }
        self.entries.append(entry)
        return entry

    #: Calibrated on engine output.  The constellation print is decisive
    #: (identical 1.0, unrelated < 0.15).  The harmony check is precise but not
    #: exhaustive: it reliably catches a transposed copy (distance 0.01-0.12
    #: against 0.35+ for unrelated music) and misses a heavily time-stretched
    #: one, so its findings are advisory rather than blocking.
    FEATURE_ADVISORY = 0.15

    def check(self, pairs, sha256=None, similar_at=0.35, contained_at=0.35,
              features=None, feature_at=None):
        """Compare a new fingerprint against everything already logged."""
        findings = []
        for entry in self.entries:
            if sha256 and entry.get("sha256") == sha256:
                findings.append({"file": entry["file"], "kind": "identical file",
                                 "score": 1.0, "added": entry.get("added")})
                continue
            old = [tuple(p) for p in entry.get("pairs", [])]
            if old:
                sim = similarity(pairs, old)
                con = containment(pairs, old)
                if sim >= similar_at or con >= contained_at:
                    findings.append({"file": entry["file"],
                                     "kind": "near-duplicate" if sim >= similar_at
                                     else "one is inside the other",
                                     "score": round(max(sim, con), 3),
                                     "added": entry.get("added")})
                    continue
            # second opinion: the constellation print only matches unmodified
            # audio, so a stretched or transposed re-use needs the coarse
            # musical signature instead
            if features and entry.get("features"):
                dist = feature_distance(features, entry["features"])
                if dist <= (feature_at or self.FEATURE_ADVISORY):
                    findings.append({"file": entry["file"],
                                     "kind": "similar harmony - worth a listen",
                                     "score": round(1.0 - dist, 3),
                                     "advisory": True,
                                     "added": entry.get("added")})
        findings.sort(key=lambda f: -f["score"])
        return findings


def compare_folder(paths, threshold=0.12, quiet=False):
    """Find near-duplicate pairs inside one batch of tracks."""
    prints = {}
    for path in paths:
        try:
            prints[str(path)], _ = fingerprint_file(path, max_seconds=240)
        except Exception as exc:
            warn(f"could not fingerprint {Path(path).name}: {exc}")
    names = list(prints)
    dupes = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = similarity(prints[names[i]], prints[names[j]])
            if score >= threshold:
                dupes.append((Path(names[i]).name, Path(names[j]).name, round(score, 3)))
    dupes.sort(key=lambda d: -d[2])
    if not quiet:
        for a, b, score in dupes:
            log(f"  near-duplicate ({score:.2f}): {a}  <->  {b}")
    return dupes
