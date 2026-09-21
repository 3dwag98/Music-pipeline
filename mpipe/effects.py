"""Effects and time/pitch work, backed by Spotify's `pedalboard` when present.

pedalboard is a C++ (JUCE + Rubber Band) library, so its filters, reverb and
time-stretch are both better and far faster than the hand-rolled numpy versions
in `dsp.py`.  It is a soft dependency: every function here falls back to the
built-in implementation when it is missing, so the pipeline never stops working
because a wheel would not install.

pedalboard uses (channels, frames); the rest of this codebase uses
(frames, channels).  `_to_pb` / `_from_pb` are the only places that matters.
"""

from __future__ import annotations

import numpy as np

from . import dsp
from .util import log, warn

try:
    import pedalboard as _pb
    HAVE_PEDALBOARD = True
    PEDALBOARD_VERSION = getattr(_pb, "__version__", "?")
except Exception:                                  # pragma: no cover
    _pb = None
    HAVE_PEDALBOARD = False
    PEDALBOARD_VERSION = None


def backend_name():
    return f"pedalboard {PEDALBOARD_VERSION}" if HAVE_PEDALBOARD else "built-in numpy DSP"


def _to_pb(x):
    """(frames, channels) float32 -> (channels, frames) float32."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    return np.ascontiguousarray(x.T)


def _from_pb(x):
    """(channels, frames) -> (frames, channels)."""
    out = np.ascontiguousarray(np.asarray(x, dtype=np.float32).T)
    return out if out.ndim == 2 else out[:, None]


# ------------------------------------------------------------ time / pitch ---

def time_pitch(x, sr, stretch=1.0, semitones=0.0, crisp=True):
    """Change length and/or pitch.

    `stretch` > 1 makes the audio SHORTER (it plays faster), matching
    `stretch.time_stretch`'s convention.  pedalboard's `stretch_factor` uses the
    same sense, so it passes straight through.
    """
    x = np.asarray(x, dtype=np.float32)
    if abs(stretch - 1.0) < 1e-4 and abs(semitones) < 1e-3:
        return x
    if HAVE_PEDALBOARD:
        try:
            out = _pb.time_stretch(
                _to_pb(x), sr, stretch_factor=float(stretch),
                pitch_shift_in_semitones=float(semitones),
                high_quality=True,
                transient_mode="crisp" if crisp else "smooth")
            return _from_pb(out)
        except Exception as exc:                   # pragma: no cover
            warn(f"pedalboard time_stretch failed ({exc}); using the built-in one")
    from .stretch import pitch_shift, time_stretch
    out = time_stretch(x, stretch) if abs(stretch - 1.0) > 1e-4 else x
    if abs(semitones) > 1e-3:
        out = pitch_shift(out, semitones, sr)
    return out


# ------------------------------------------------------------- lofi colour ---

def lofi_chain(sr, amount=0.6, lowpass_hz=None, bitcrush_bits=None, wobble=True,
               room=0.25, telephone=0.0, mp3_artifacts=0.0):
    """Build the 'make it sound like a tape' chain.

    `amount` (0-1) scales the whole character; the individual arguments override
    it.  Returns a callable taking and returning (frames, channels).
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    cutoff = float(lowpass_hz if lowpass_hz is not None
                   else 16000.0 - 9000.0 * amount)
    bits = int(bitcrush_bits if bitcrush_bits is not None
               else round(16 - 5 * amount))

    if not HAVE_PEDALBOARD:
        return _fallback_chain(sr, amount, cutoff, bits, wobble, room)

    board = [
        _pb.HighpassFilter(cutoff_frequency_hz=30.0),
        _pb.LowpassFilter(cutoff_frequency_hz=cutoff),
        _pb.PeakFilter(cutoff_frequency_hz=265.0, gain_db=-2.5 * amount, q=1.0),
        _pb.PeakFilter(cutoff_frequency_hz=2600.0, gain_db=1.8, q=0.8),
        _pb.HighShelfFilter(cutoff_frequency_hz=10500.0, gain_db=-4.0 * amount, q=0.7),
    ]
    if bits < 16:
        board.append(_pb.Bitcrush(bit_depth=max(4, bits)))
    if wobble:
        # a slow, shallow chorus is the cheapest convincing tape-wow stand-in
        board.append(_pb.Chorus(rate_hz=0.6, depth=0.18 * amount,
                                mix=0.35 * amount, centre_delay_ms=6.0, feedback=0.0))
    if telephone > 0:
        board.append(_pb.Mix([_pb.Gain(gain_db=0.0),
                              _pb.Chain([_pb.GSMFullRateCompressor(),
                                         _pb.Gain(gain_db=-6.0 + 6.0 * telephone)])]))
    if mp3_artifacts > 0:
        board.append(_pb.MP3Compressor(vbr_quality=float(np.clip(
            9.0 - 7.0 * (1.0 - mp3_artifacts), 0.0, 9.0))))
    if room > 0:
        board.append(_pb.Reverb(room_size=float(np.clip(room, 0.0, 1.0)),
                                damping=0.55, wet_level=0.14 * amount + 0.04,
                                dry_level=0.95, width=0.9))
    board.append(_pb.Compressor(threshold_db=-16.0, ratio=2.2,
                                attack_ms=12.0, release_ms=220.0))
    chain = _pb.Pedalboard(board)

    def run(audio, reset=False):
        return _from_pb(chain(_to_pb(audio), sr, reset=reset))

    return run


def _fallback_chain(sr, amount, cutoff, bits, wobble, room):
    """The same idea with the built-in DSP, for installs without pedalboard."""
    hp = dsp.Biquad("highpass", sr, 30.0, 0.7)
    lp = dsp.Biquad("lowpass", sr, cutoff, 0.7)
    mud = dsp.Biquad("peak", sr, 265.0, 1.0, gain_db=-2.5 * amount)
    presence = dsp.Biquad("peak", sr, 2600.0, 0.8, gain_db=1.8)
    air = dsp.Biquad("highshelf", sr, 10500.0, 0.7, gain_db=-4.0 * amount)
    wow = dsp.TapeWow(sr, wow_ms=2.0 * amount, flutter_ms=0.25 * amount) if wobble else None
    verb = dsp.FDNReverb(sr, room=room, damping=0.5) if room > 0 else None
    comp = dsp.Compressor(sr, threshold_db=-16.0, ratio=2.2,
                          attack_ms=12.0, release_ms=220.0)

    def run(audio, reset=False):
        y = air.process(presence.process(mud.process(lp.process(hp.process(audio)))))
        if bits < 16:
            y = dsp.bit_crush(y, bits=max(4, bits))
        if wow is not None:
            y = wow.process(y)
        y = dsp.tape_saturate(y, drive=1.0 + 0.5 * amount)
        if verb is not None:
            y = y + verb.process(y * (0.12 * amount + 0.04))
        return comp.process(y)

    return run


# --------------------------------------------------------------- mastering ---

def brickwall(x, sr, ceiling_db=-1.0, true_peak=True):
    """Final peak limit.

    pedalboard's brickwall has a true-peak mode, which limits the *reconstructed*
    waveform rather than the samples.  That is what YouTube's -1 dBTP actually
    means, and it is what stops a lossy encode from clipping on playback.
    """
    if HAVE_PEDALBOARD:
        board = _pb.Pedalboard([_pb.BrickwallLimiter(
            ceiling_db=float(ceiling_db), true_peak=bool(true_peak))])
        return _from_pb(board(_to_pb(x), sr))
    limiter = dsp.Limiter(sr, ceiling_db=ceiling_db)
    return np.concatenate([limiter.process(x), limiter.flush()])


class StreamLimiter:
    """Block-wise brickwall limiting with state kept across calls."""

    def __init__(self, sr, ceiling_db=-1.0):
        self.sr = sr
        self.ceiling = 10 ** (ceiling_db / 20.0)
        if HAVE_PEDALBOARD:
            self._board = _pb.Pedalboard([_pb.BrickwallLimiter(
                ceiling_db=float(ceiling_db), true_peak=True)])
            self._fallback = None
        else:
            self._board = None
            self._fallback = dsp.Limiter(sr, ceiling_db=ceiling_db)

    def process(self, block):
        if self._board is not None:
            # No hard clip here: the limiter is already working in the true-peak
            # domain, and clipping its output would manufacture exactly the
            # inter-sample overshoot true-peak limiting exists to prevent.
            return _from_pb(self._board(_to_pb(block), self.sr, reset=False))
        return self._fallback.process(block)

    def flush(self):
        if self._fallback is not None:
            return self._fallback.flush()
        return np.zeros((0, 2), dtype=np.float32)


# ------------------------------------------------- vocal / stem separation ---

def reduce_centre(x, amount=1.0, keep_bass_hz=140.0, sr=44100):
    """Attenuate whatever is dead-centre - usually the lead vocal.

    The oldest karaoke trick: vocals sit in the middle, so subtracting the mid
    signal removes them.  It is free and needs no model, but it also thins the
    kick and snare, so the low end is put back untouched.  Real separation
    (`--vocals remove` with demucs) is better when you have it.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 2:
        return x
    amount = float(np.clip(amount, 0.0, 1.0))
    low = dsp.Biquad("lowpass", sr, keep_bass_hz, 0.7).process(x)
    high = x - low
    mid = high.mean(axis=1, keepdims=True)
    side = high - mid
    out = side + mid * (1.0 - amount)
    return (out + low).astype(np.float32)


def hpss(x, sr, margin=2.0, n_fft=2048, hop=512):
    """Split into (harmonic, percussive) with median filtering on the spectrogram.

    Implemented here rather than via librosa so the feature works on a plain
    install.  Used to soften or isolate the drums when re-lofi-ing a track.
    """
    from scipy.ndimage import median_filter
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    harm = np.zeros_like(x)
    perc = np.zeros_like(x)
    window = np.hanning(n_fft).astype(np.float32)
    for ch in range(x.shape[1]):
        spec = _stft(x[:, ch], n_fft, hop, window)
        mag = np.abs(spec)
        # smooth along time -> what is steady is harmonic
        # smooth along frequency -> what is broadband is percussive
        h = median_filter(mag, size=(1, 17), mode="nearest")
        p = median_filter(mag, size=(17, 1), mode="nearest")
        total = (h ** margin) + (p ** margin) + 1e-12
        mask_h = (h ** margin) / total
        mask_p = (p ** margin) / total
        harm[:, ch] = _istft(spec * mask_h, n_fft, hop, window, len(x))
        perc[:, ch] = _istft(spec * mask_p, n_fft, hop, window, len(x))
    return harm, perc


def _stft(sig, n_fft, hop, window):
    frames = 1 + max(0, (len(sig) - n_fft) // hop)
    if frames < 1:
        sig = np.pad(sig, (0, n_fft - len(sig)))
        frames = 1
    idx = np.arange(frames)[:, None] * hop + np.arange(n_fft)[None, :]
    return np.fft.rfft(sig[idx] * window[None, :], axis=1).T


def _istft(spec, n_fft, hop, window, length):
    frames = spec.shape[1]
    time_frames = np.fft.irfft(spec.T, n=n_fft, axis=1) * window[None, :]
    out = np.zeros(length + n_fft, dtype=np.float64)
    norm = np.zeros(length + n_fft, dtype=np.float64)
    for i in range(frames):
        start = i * hop
        out[start:start + n_fft] += time_frames[i]
        norm[start:start + n_fft] += window ** 2
    out = out[:length] / np.maximum(norm[:length], 1e-8)
    return out.astype(np.float32)


def have_demucs():
    try:
        import demucs.apply  # noqa: F401
        import demucs.pretrained  # noqa: F401
        return True
    except Exception:
        return False


def separate_stems(x, sr, model="htdemucs", device=None, progress=False):
    """Split into {drums, bass, other, vocals} with demucs, if it is installed.

    Optional on purpose: demucs pulls in torch and a few hundred MB of weights.
    Callers must handle `None` and fall back to `reduce_centre`.
    """
    if not have_demucs():
        return None
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except Exception as exc:                        # pragma: no cover
        warn(f"demucs present but unusable ({exc})")
        return None
    try:
        bundle = get_model(model)
        bundle.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        bundle.to(device)
        audio = torch.tensor(np.asarray(x, dtype=np.float32).T)      # (ch, frames)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        ref = audio.mean(0)
        audio = (audio - ref.mean()) / (ref.std() + 1e-8)
        with torch.no_grad():
            # split/overlap keep peak VRAM low enough for a 6 GB card
            sources = apply_model(bundle, audio[None], device=device, split=True,
                                  overlap=0.25, progress=progress)[0]
        sources = sources * (ref.std() + 1e-8) + ref.mean()
        names = list(bundle.sources)
        return {name: np.ascontiguousarray(sources[i].cpu().numpy().T)
                for i, name in enumerate(names)}
    except Exception as exc:
        warn(f"demucs separation failed ({exc}); falling back to centre reduction")
        return None
