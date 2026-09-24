# Every local option for generating lofi, checked

Researched September 2026 against a **GTX 1660 Ti (6 GB), 16 GB RAM, local only,
instrumental lofi**. Sizes and licences were read from the model repositories,
not recalled. Licences change — check the link before relying on one.

## The short version

We had been looking at one shelf: `pipeline_tag=text-to-audio`. That tag holds
models that generate **raw audio**, which is the most expensive way to make
music and the reason everything so far has been slow, large, and
non-commercial.

The shelf nobody looks at is **symbolic generation** — models that write
*notes*, not waveforms. They are 5–25× smaller, Apache 2.0 far more often, and
they hand you a score you can render with any instruments you like.

**Verified working:** `amaai-lab/text2midi` — 225M parameters, Apache 2.0,
text-conditioned, generated 11-track MIDI from a lofi caption, rendered through
this repo's own synth engine. No audio codec anywhere in the chain.

---

## Option A — raw audio generation (what we have been doing)

| Model | Size on disk | Params | Licence | Commercial | Verdict |
|---|---|---|---|---|---|
| `facebook/musicgen-small` | 5.81 GB | 300M decoder | CC-BY-NC 4.0 | **no** | works, currently wired up, cannot be monetised |
| `facebook/musicgen-medium` | ~10 GB | 1.5B | CC-BY-NC 4.0 | **no** | better sound, tight at 6 GB in fp32 |
| `stabilityai/stable-audio-open-small` | 5.03 GB | 0.5B | Stability Community | conditional | 11s cap — sound design, not songs |
| `stabilityai/stable-audio-open-1.0` | ~5 GB | 1.1B | Stability Community | conditional | 47s cap |
| `stabilityai/stable-audio-3-small-music` | — | 0.6B | Stability Community | conditional | minutes of audio, needs Stability's own libraries |
| ACE-Step 1.5 | — | 3.5B | **Apache 2.0** | **yes** | already integrated; the commercial audio option |

**The structural problem with this whole category:** these models emit
compressed audio tokens, so everything they make carries codec character, you
get whatever instruments the model decides on, and a minute of music costs
minutes of compute. The crackle you heard was partly a prompt word and partly
this.

## Option B — symbolic / MIDI generation (the one worth taking)

| Model | Size on disk | Params | Licence | Commercial | Notes |
|---|---|---|---|---|---|
| **`amaai-lab/text2midi`** | **1.06 GB** | **225M** | **Apache 2.0** | **yes** | text → multi-track MIDI. **Verified running.** |
| `skytnt/midi-model` | 2.85 GB | 0.2B | Apache 2.0 | yes | MIDI event transformer, no text conditioning |
| `Metacreation/MIDI-GPT` | 1.15 GB | — | CC-BY-NC 4.0 | **no** | multi-track with infill, but non-commercial |
| `loubb/aria-medium-base` | 8.08 GB | 1B | Apache 2.0 | yes | solo piano only, and too big here |

### Why this suits your constraints better than raw audio

- **Smaller.** 1.06 GB against 5.81 GB. It ran on CPU here; on your card it is
  not close to a constraint, and fp16 stops being a question worth asking.
- **Apache 2.0.** You can monetise the output. MusicGen you cannot.
- **No codec, so no artefacts.** The model writes notes. Nothing in the chain
  compresses audio, so crackle and codec shimmer are not merely reduced — they
  are structurally impossible.
- **You choose the instruments.** A score is not tied to a timbre. This repo
  already synthesises Rhodes, felt piano, nylon guitar, vibraphone, kalimba,
  muted trumpet, upright and sub bass, plus a full drum kit.
- **It addresses "robotic" at the right layer.** The complaint was that the
  built-in engine used one hardcoded arrangement for every track. A symbolic
  model replaces *that* — the composition — while the synths stay under your
  control.

### What it measured here

```
params            225.4M
load time         3s (CPU)
600 tokens        135s on CPU  (GPU will be far quicker)
output            11 tracks, 136 notes from one lofi caption
licence           Apache 2.0
```

### The catch, stated plainly

- **Packaging friction.** The tokenizer is pickled against an older `miditok`;
  it needs `miditok==3.0.3` pinned. Newer versions raise attribute errors on
  decode. It took three attempts to get a clean MIDI out.
- **Quality is not guaranteed.** The authors' own listening study rates it 4.62
  out of 7 for musical quality against 5.79 for human MIDI, and chord matching
  is its weakest axis at 2.50. It writes plausible music, not great music.
- **Short outputs.** 600 tokens gave under 12 beats. Long-form needs either
  more tokens or looping/arranging sections — the latter is what this repo's
  `song` builder already does.

## Option C — hybrid, and probably the best of the three

Symbolic model writes the composition → this repo's synth engine renders it →
existing mastering, `song` builder and `check` handle the rest.

That is what the attached 8-second clip is. It is a proof of concept, not a
finished feature: the MIDI-to-synth mapping is about thirty lines and the
arrangement and long-form side are not wired up.

**What would need building:** a `midi` command (load or generate a score, map
programs to the built-in instruments, render, master), section looping for
length, and the `miditok` pin.

## Ruled out

- **`unconditional-audio-generation`** — the tag returns zero models.
- **YuE 7B** — 24 GB+.
- **Riffusion-style spectrogram tricks** — superseded and lower fidelity than
  any option above.
- **Stock-library-oriented models** — irrelevant, since those libraries ban AI
  submissions anyway (see `MONETIZATION.md`).

## Recommendation

1. **Build the hybrid.** `text2midi` + the existing synth engine is smaller,
   faster, commercially usable and structurally free of the artefacts you
   heard. It is the only option that fixes the licence problem and the crackle
   problem at once.
2. **Keep `hf`/MusicGen for reference.** Useful for comparison and for
   non-monetised work.
3. **Keep ACE-Step** as the commercial *audio* path when you want a
   model's timbre rather than your own synths.

## Sources

- [text2midi](https://huggingface.co/amaai-lab/text2midi) ·
  [skytnt/midi-model](https://huggingface.co/skytnt/midi-model) ·
  [MIDI-GPT](https://huggingface.co/Metacreation/MIDI-GPT) ·
  [aria-medium-base](https://huggingface.co/loubb/aria-medium-base)
- [musicgen-small](https://huggingface.co/facebook/musicgen-small) ·
  [stable-audio-open-small](https://huggingface.co/stabilityai/stable-audio-open-small) ·
  [stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)
