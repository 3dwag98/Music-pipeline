"""What can actually generate the music, and what you are allowed to do with it.

The pipeline is model-agnostic: `generate` drives ACE-Step over its own REST
API, `generate --backend comfy` drives any API-format ComfyUI workflow, and the
built-in engine needs no model at all.  So "adding a model" is usually adding a
workflow file, not writing code.

What is NOT interchangeable is the licence.  Several well-known music models
ship weights under CC-BY-NC, which forbids commercial use - and a monetised
YouTube channel is commercial use.  That is a property of the model, not of
this pipeline, and it is the kind of thing that is easy to miss and expensive
to discover later, so it lives in the tool rather than only in the README.

Nothing here is legal advice, and licences change.  Each entry carries the URL
it came from; check it before you rely on it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .util import log


@dataclass
class Model:
    key: str
    name: str
    params: str
    licence: str
    commercial: str            # "yes" | "no" | "conditional"
    min_vram_gb: float         # as published, before CPU offload
    max_length: str
    backends: tuple
    url: str
    note: str = ""

    def to_dict(self):
        return asdict(self)


#: Checked 2026-09.  `commercial` refers to the WEIGHTS, which is what bites -
#: several of these have permissive code and restricted weights.
CATALOGUE = [
    Model(
        key="acestep",
        name="ACE-Step 1.5",
        params="3.5B",
        licence="Apache 2.0",
        commercial="yes",
        min_vram_gb=8.0,
        max_length="full songs",
        backends=("generate", "generate --backend comfy"),
        url="https://github.com/ace-step/ACE-Step",
        note="Already wired into this pipeline, two ways. Apache 2.0 covers the "
             "weights as well as the code, so monetising the output is fine. "
             "8 GB is the published figure; --cpu_offload brings it under that, "
             "which is what `doctor --write-env` configures.",
    ),
    Model(
        key="diffrhythm2",
        name="DiffRhythm 2",
        params="~1B",
        licence="Apache 2.0",
        commercial="yes",
        min_vram_gb=8.0,
        max_length="full-length songs",
        backends=("generate --backend comfy (needs a workflow)",),
        url="https://github.com/ASLP-lab/DiffRhythm2",
        note="The main commercially-usable alternative to ACE-Step. Latent "
             "diffusion, fast, built for full songs rather than clips.",
    ),
    Model(
        key="musicgen",
        name="MusicGen (small ... large)",
        params="300M - 3.3B",
        licence="code MIT, WEIGHTS CC-BY-NC 4.0",
        commercial="no",
        min_vram_gb=4.0,
        max_length="~30s per generation",
        backends=("generate --backend comfy (needs a workflow)",),
        url="https://huggingface.co/facebook/musicgen-small",
        note="Fits a 6 GB card easily and sounds good - and the weights are "
             "NON-COMMERCIAL. A monetised channel is commercial use. The MIT "
             "licence on the code does not carry over to the weights, which is "
             "the trap. Fine for private or non-monetised work.",
    ),
    Model(
        key="stable-audio-open",
        name="Stable Audio Open 1.0",
        params="1.1B",
        licence="Stability AI Community License",
        commercial="conditional",
        min_vram_gb=8.0,
        max_length="47s",
        backends=("generate --backend comfy (needs a workflow)",),
        url="https://huggingface.co/stabilityai/stable-audio-open-1.0",
        note="Free for commercial use below a revenue threshold; above it you "
             "need a separate licence from Stability. Notable for provenance: "
             "trained only on CC-licensed Freesound and Free Music Archive "
             "audio with suspected copyrighted material screened out. 47s is "
             "too short for songs - better for textures and one-shots.",
    ),
    Model(
        key="stable-audio-open-small",
        name="Stable Audio Open Small",
        params="0.5B",
        licence="Stability AI Community License",
        commercial="conditional",
        min_vram_gb=4.0,
        max_length="11s",
        backends=("generate --backend comfy (needs a workflow)",),
        url="https://huggingface.co/stabilityai/stable-audio-open-small",
        note="Small enough for almost anything. 11 seconds, so it is a "
             "sound-design tool, not a song generator.",
    ),
    Model(
        key="yue",
        name="YuE 7B",
        params="7B",
        licence="see the repository",
        commercial="check",
        min_vram_gb=24.0,
        max_length="full songs with vocals",
        backends=(),
        url="https://github.com/multimodal-art-projection/YuE",
        note="Full songs with real vocals, and far beyond a 6 GB card - listed "
             "so you know why it is not an option here.",
    ),
    Model(
        key="builtin",
        name="This pipeline's own engine",
        params="none",
        licence="this repository",
        commercial="yes",
        min_vram_gb=0.0,
        max_length="unlimited (streams)",
        backends=("lofi",),
        url="",
        note="No model, no weights, no download, no VRAM. Every sound is "
             "synthesised, so there is no third-party training data or "
             "recording in the output at all - which is also the cleanest "
             "provenance story of anything on this list.",
    ),
]

BY_KEY = {m.key: m for m in CATALOGUE}

COMMERCIAL_MARK = {"yes": "yes", "no": "NO", "conditional": "conditional", "check": "check"}


def fits(model, vram_gb, cpu_offload=True):
    """Would this model run on a card of `vram_gb`?  Offload buys ~2-3 GB."""
    if model.min_vram_gb <= 0:
        return True
    budget = float(vram_gb) + (2.5 if cpu_offload else 0.0)
    return budget >= model.min_vram_gb


def report(vram_gb=None, commercial_only=False, verbose=True):
    """Print the catalogue, marked up for this machine."""
    rows = [m for m in CATALOGUE
            if not commercial_only or m.commercial in ("yes", "conditional")]
    if verbose:
        log("Music generation models this pipeline can drive\n")
        if vram_gb:
            log(f"Filtering against {vram_gb:.1f} GB of VRAM "
                f"(+2.5 GB assumed from CPU offload)\n")
        log(f"  {'model':<30} {'params':<10} {'commercial':<12} {'VRAM':<7} {'length':<22} fits?")
        log(f"  {'-' * 30} {'-' * 10} {'-' * 12} {'-' * 7} {'-' * 22} -----")
        for m in rows:
            ok = "-" if not vram_gb else ("yes" if fits(m, vram_gb) else "no")
            vram = "none" if m.min_vram_gb <= 0 else f"{m.min_vram_gb:.0f} GB"
            name = m.name if len(m.name) <= 30 else m.name[:27] + "..."
            log(f"  {name:<30} {m.params:<10} {COMMERCIAL_MARK[m.commercial]:<12} "
                f"{vram:<7} {m.max_length:<22} {ok}")
        log("")
        blocked = [m for m in rows if m.commercial == "no"]
        if blocked:
            log("Commercial use is BLOCKED for:")
            for m in blocked:
                log(f"  {m.name} - {m.licence}")
            log("  A monetised YouTube channel is commercial use.  The weights'")
            log("  licence is what counts, not the code's.\n")
        log("Details:")
        for m in rows:
            log(f"\n  {m.name}  [{m.licence}]")
            if m.url:
                log(f"    {m.url}")
            if m.backends:
                log(f"    run via: {', '.join(m.backends)}")
            for line in _wrap(m.note, 72):
                log(f"    {line}")
        log("\n  Adding a model usually means adding a ComfyUI workflow, not writing")
        log("  code: build it in ComfyUI, Workflow -> Export (API), then")
        log("  `generate --backend comfy --workflow yours.json`.")
        log("\n  Licences change and this is not legal advice - check the link.")
    return [m.to_dict() for m in rows]


def _wrap(text, width):
    words, line, out = str(text).split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out
