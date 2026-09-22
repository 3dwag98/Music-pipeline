"""Local text-to-audio generation with a Hugging Face model.

Built for the smallest model that can actually make music: `musicgen-small`
(300M decoder) runs inside 6 GB and is the one Meta recommends when the GPU is
small.  Everything here is local - the weights download once and then the
machine is offline.

Two things this module takes seriously, because both bite on a GTX 1660 Ti:

fp16
    Turing TU116 (the 16-series) has a half-precision path that is known to
    produce NaNs in *diffusion* models.  MusicGen is an autoregressive
    transformer, not a diffusion UNet, so it may well be fine - but "may well
    be" is not something to find out after a six-hour render.  `--dtype auto`
    picks fp32 on those cards, `--dtype fp16` forces it anyway, and
    `selftest()` answers the question empirically on YOUR card in about a
    minute.

length
    MusicGen generates ~30s at a time.  Longer output is built by feeding the
    tail of what it just made back in as an audio prompt, so a five-minute
    track is one continuous take rather than clips butted together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np

from .util import die, log, warn

#: Frame rate of MusicGen's audio codec: tokens per second of output.
MUSICGEN_FRAME_RATE = 50
#: What one generate() call can produce before quality falls apart.
MAX_CHUNK_SECONDS = 30.0

#: Cards whose fp16 path is known to misbehave (Turing TU116/TU117).
FP16_SUSPECT = ("1650", "1660", "1630", "T400", "T500", "T600")


@dataclass
class ModelSpec:
    key: str
    repo: str
    params: str
    licence: str
    commercial: bool
    sr: int
    stereo: bool
    note: str = ""

    def to_dict(self):
        return asdict(self)


#: Text-to-audio models worth pointing this at, smallest first.
MODELS = {
    "musicgen-small": ModelSpec(
        key="musicgen-small", repo="facebook/musicgen-small", params="300M decoder",
        licence="CC-BY-NC 4.0 (weights)", commercial=False, sr=32000, stereo=False,
        note="The smallest model that genuinely makes music, and the one Meta "
             "names for small GPUs. Its WEIGHTS are non-commercial - the MIT "
             "licence on audiocraft's code does not carry over."),
    "musicgen-stereo-small": ModelSpec(
        key="musicgen-stereo-small", repo="facebook/musicgen-stereo-small",
        params="300M decoder", licence="CC-BY-NC 4.0 (weights)", commercial=False,
        sr=32000, stereo=True,
        note="Same size, stereo output. Same non-commercial weights."),
    "musicgen-medium": ModelSpec(
        key="musicgen-medium", repo="facebook/musicgen-medium", params="1.5B decoder",
        licence="CC-BY-NC 4.0 (weights)", commercial=False, sr=32000, stereo=False,
        note="Better, and too big to be comfortable in 6 GB at fp32."),
}
DEFAULT_MODEL = "musicgen-small"


# --------------------------------------------------------------- hardware ---

def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def gpu_vram_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return 0.0


def fp16_is_suspect(name=None):
    """Is this card one of the ones with a dodgy half-precision path?"""
    name = name if name is not None else (gpu_name() or "")
    return any(tag in name for tag in FP16_SUSPECT)


def resolve_dtype(requested="auto", device=None):
    """Pick a dtype.  Returns (torch_dtype, label, why)."""
    import torch
    name = gpu_name() or ""
    if device == "cpu" or not name:
        return torch.float32, "fp32", "running on the CPU, where fp16 is slower, not faster"
    requested = (requested or "auto").lower()
    if requested in ("fp16", "float16", "half"):
        why = "forced by --dtype fp16"
        if fp16_is_suspect(name):
            why += (f"; note {name} is a 16-series card - run `selftest` to check "
                    "the output is not silence")
        return torch.float16, "fp16", why
    if requested in ("fp32", "float32", "full"):
        return torch.float32, "fp32", "forced by --dtype fp32"
    if fp16_is_suspect(name):
        return torch.float32, "fp32", (f"{name} is a 16-series card whose fp16 path is "
                                       "unreliable; --dtype fp16 overrides this")
    return torch.float16, "fp16", f"{name} handles fp16 well"


# -------------------------------------------------------------- generation ---

@dataclass
class GenSettings:
    seconds: float = 30.0
    guidance: float = 3.0
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    seed: int = 0
    overlap_seconds: float = 5.0     # how much tail is fed back when continuing

    def to_dict(self):
        return asdict(self)


class HFAudioGenerator:
    """Wraps a Hugging Face text-to-audio model for repeated local use."""

    def __init__(self, model_key=DEFAULT_MODEL, dtype="auto", device="auto",
                 cache_dir=None, quiet=False):
        if not torch_available():
            die("this backend needs PyTorch.\n"
                "       GPU:  pip install torch --index-url "
                "https://download.pytorch.org/whl/cu121\n"
                "       CPU:  pip install torch --index-url "
                "https://download.pytorch.org/whl/cpu\n"
                "       then: pip install transformers")
        self.spec = MODELS.get(model_key)
        if self.spec is None:
            die(f"unknown model '{model_key}'. Available: {', '.join(MODELS)}")
        self.cache_dir = cache_dir
        self.quiet = quiet
        self.model = None
        self.processor = None
        self._device_pref = device
        self._dtype_pref = dtype
        self.device = None
        self.dtype = None
        self.dtype_label = None

    # ------------------------------------------------------------- loading --
    def load(self):
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        device = self._device_pref
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype, self.dtype_label, why = resolve_dtype(self._dtype_pref, device)

        vram = gpu_vram_gb()
        if device == "cuda" and vram:
            need = 2.0 if self.dtype_label == "fp16" else 3.5
            if "medium" in self.spec.key:
                need *= 3.0
            if vram < need:
                warn(f"{self.spec.repo} wants about {need:.1f} GB and this card has "
                     f"{vram:.1f} GB - expect it to be slow or to run out")

        if not self.quiet:
            log(f"Loading {self.spec.repo} ({self.spec.params}) on {device} "
                f"in {self.dtype_label}")
            log(f"  {why}")
            if not self.spec.commercial:
                warn(f"{self.spec.repo} weights are {self.spec.licence} - "
                     "NOT licensed for commercial use. See MONETIZATION.md.")

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.spec.repo,
                                                       cache_dir=self.cache_dir)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            self.spec.repo, cache_dir=self.cache_dir,
            dtype=self.dtype if device == "cuda" else torch.float32)
        self.model.to(device)
        self.model.eval()
        if not self.quiet:
            log(f"  ready in {time.time() - t0:.1f}s")
        return self

    @property
    def sample_rate(self):
        if self.model is not None:
            return int(self.model.config.audio_encoder.sampling_rate)
        return self.spec.sr

    # ---------------------------------------------------------- generation --
    def _tokens_for(self, seconds):
        return max(16, int(round(float(seconds) * MUSICGEN_FRAME_RATE)))

    def _generate_once(self, prompt, seconds, settings, audio_prompt=None):
        """One model call.  Returns (frames, channels) float32."""
        import torch
        if self.model is None:
            self.load()
        kwargs = dict(text=[prompt], padding=True, return_tensors="pt")
        if audio_prompt is not None and len(audio_prompt):
            mono = audio_prompt.mean(axis=1) if audio_prompt.ndim > 1 else audio_prompt
            kwargs["audio"] = mono.astype(np.float32)
            kwargs["sampling_rate"] = self.sample_rate
        inputs = self.processor(**kwargs)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        if settings.seed:
            torch.manual_seed(int(settings.seed))

        gen = dict(do_sample=True, guidance_scale=float(settings.guidance),
                   max_new_tokens=self._tokens_for(seconds),
                   temperature=float(settings.temperature))
        if settings.top_k:
            gen["top_k"] = int(settings.top_k)
        if settings.top_p:
            gen["top_p"] = float(settings.top_p)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen)
        audio = out[0].to(torch.float32).cpu().numpy()
        if audio.ndim == 1:
            audio = audio[None, :]
        return np.ascontiguousarray(audio.T, dtype=np.float32)

    #: Do not fire a whole continuation call for less than this much audio -
    #: each one costs a full model pass plus the re-emitted prompt.
    MIN_CONTINUATION_SECONDS = 1.0

    def generate(self, prompt, settings: GenSettings, progress=True):
        """Generate `settings.seconds` of audio, continuing past the model's limit.

        MusicGen re-emits its audio prompt (an EnCodec round-trip of it, measured
        at 0.94 correlation against 0.01 for unrelated audio), so each
        continuation asks for `overlap + what is still needed` and drops the
        re-emitted head.
        """
        target = float(settings.seconds)
        pieces = []
        made = 0.0
        tail = None
        index = 0
        while True:
            remaining = target - made
            # a sliver is not worth a whole extra pass; the trim at the end
            # deals with the last fraction of a second
            if remaining < (self.MIN_CONTINUATION_SECONDS if pieces else 0.05):
                break
            want = min(MAX_CHUNK_SECONDS, remaining)
            if tail is not None:
                # ask for the re-emitted prompt back on top of what is needed
                want = min(MAX_CHUNK_SECONDS,
                           remaining + settings.overlap_seconds)
            step = GenSettings(**{**settings.to_dict(),
                                  "seed": (settings.seed + index) if settings.seed else 0})
            t0 = time.time()
            chunk = self._generate_once(prompt, want, step, audio_prompt=tail)
            if tail is not None:
                # Drop the re-emitted prompt and butt-join.  A crossfade here was
                # tried and measured as no improvement: the join's largest sample
                # step (0.361) already sits below the track's own 99.99th
                # percentile (0.372), so there is no seam to smooth - and blending
                # our tail against the model's 0.94-correlated reconstruction of
                # it risks comb filtering rather than fixing anything.  The level
                # difference across a join is the model playing the next section
                # differently, which is music, not an artefact.
                drop = min(len(chunk) - 1, len(tail))
                chunk = chunk[drop:]
            if len(chunk) == 0:
                break
            pieces.append(chunk)
            made += len(chunk) / self.sample_rate
            if progress and not self.quiet:
                log(f"    {min(made, target):5.1f}s / {target:.0f}s  "
                    f"({len(chunk) / self.sample_rate:.1f}s in {time.time() - t0:.1f}s)")
            overlap = int(settings.overlap_seconds * self.sample_rate)
            tail = chunk[-overlap:] if len(chunk) >= overlap else chunk
            index += 1
            if index > 200:
                warn("stopping: the model is not producing audio")
                break
        if not pieces:
            raise RuntimeError("the model produced no audio")
        audio = np.concatenate(pieces, axis=0)
        want_frames = int(target * self.sample_rate)
        return audio[:want_frames] if len(audio) > want_frames else audio

    # ------------------------------------------------------------ selftest --
    def selftest(self, seconds=3.0):
        """Generate a short clip and check it is real audio, not NaNs or silence.

        This is the empirical answer to 'does fp16 work on my card'.  It costs a
        minute and it is worth running before any long job.
        """
        settings = GenSettings(seconds=seconds, seed=1234)
        t0 = time.time()
        audio = self._generate_once("lofi hip hop, mellow rhodes, soft drums",
                                    seconds, settings)
        elapsed = time.time() - t0
        finite = bool(np.isfinite(audio).all())
        peak = float(np.abs(audio).max()) if finite and audio.size else 0.0
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if finite else 0.0
        silent = peak < 1e-4
        result = {
            "model": self.spec.repo, "device": self.device, "dtype": self.dtype_label,
            "seconds": round(len(audio) / self.sample_rate, 2),
            "elapsed": round(elapsed, 1),
            "finite": finite, "peak": round(peak, 5), "rms": round(rms, 6),
            "ok": bool(finite and not silent),
        }
        if not finite:
            result["verdict"] = (f"FAILED: output contains NaN/Inf in {self.dtype_label}. "
                                 "This is the 16-series half-precision problem. "
                                 "Re-run with --dtype fp32.")
        elif silent:
            result["verdict"] = (f"FAILED: output is silent in {self.dtype_label} "
                                 f"(peak {peak:.2e}). Re-run with --dtype fp32.")
        else:
            result["verdict"] = (f"OK: {self.dtype_label} produces real audio on this "
                                 f"machine (peak {peak:.3f}).")
        return result


# ------------------------------------------------------------- prompt help ---

# ------------------------------------------------------------- prompt help ---
#
# MusicGen responds to positive musical description, not to negation.  "no
# vocals" often makes vocals *more* likely because the model sees the word, and
# naming production artefacts ("vinyl crackle", "tape hiss", "lo-fi quality")
# makes it render the artefact instead of the music.  Everything below is
# phrased as what the music IS.

#: Core identity of the genre - always present.
LOFI_CORE = "lofi hip hop, chill instrumental beat"

#: Varied pools, so ten tracks do not come out as ten takes of one idea.
PROMPT_POOLS = {
    "keys": [
        "warm Rhodes electric piano playing soft jazz chords",
        "mellow felt piano with gentle seventh chords",
        "smooth electric piano with lush major-seventh voicings",
        "soft muted electric piano, warm and rounded",
        "dreamy vibraphone over quiet piano chords",
        "gentle nylon-string guitar playing jazzy chords",
    ],
    "drums": [
        "relaxed boom bap drums with a soft kick and rimshot",
        "laid-back swung hip hop drums, brushed snare",
        "slow head-nodding drum groove, soft and unhurried",
        "gentle drums with light hi-hats and a deep kick",
    ],
    "bass": [
        "warm round bassline",
        "smooth upright bass walking gently",
        "deep mellow sub bass",
        "soft melodic bass holding the groove",
    ],
    "mood": [
        "calm and nostalgic",
        "warm, cosy and relaxed",
        "peaceful late-night mood",
        "soft, melancholic and reflective",
        "easy Sunday-afternoon feeling",
        "gentle and dreamy",
    ],
    "quality": [
        "smooth and musical, well-played",
        "warm analogue recording, clean mix",
        "rich harmony, natural groove",
    ],
}


def lofi_prompt(bpm=78, extra=None, mood=None, rng=None, texture=False):
    """Build a varied, musical lofi prompt.

    `texture=True` adds the vinyl/tape wording back in for anyone who wants
    that sound - it is off by default because it makes the model foreground the
    crackle rather than the music.
    """
    import random as _random
    rng = rng or _random.Random()
    parts = [LOFI_CORE,
             rng.choice(PROMPT_POOLS["keys"]),
             rng.choice(PROMPT_POOLS["drums"]),
             rng.choice(PROMPT_POOLS["bass"]),
             str(mood) if mood else rng.choice(PROMPT_POOLS["mood"]),
             rng.choice(PROMPT_POOLS["quality"]),
             f"{int(bpm)} bpm"]
    if texture:
        parts.insert(-1, "soft vinyl crackle in the background")
    if extra:
        parts.append(str(extra))
    return ", ".join(parts)


#: Wall-clock cost per second of audio, for the "is this worth starting"
#: question.  The CPU figure is measured (musicgen-small, ~4x realtime on a
#: laptop core); the GPU figures are estimates and vary with the card.
SPEED_FACTORS = {"cpu": 4.0, "cuda-fp16": 1.5, "cuda-fp32": 2.5}


def estimate_minutes(seconds, device, dtype_label):
    if device == "cpu":
        factor = SPEED_FACTORS["cpu"]
    else:
        factor = SPEED_FACTORS.get(f"cuda-{dtype_label}", SPEED_FACTORS["cuda-fp32"])
    return seconds * factor / 60.0
