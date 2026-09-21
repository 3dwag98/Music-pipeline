# Local lofi music pipeline

Everything runs on your own PC. Nothing is uploaded, nothing is streamed, no API keys.

```
   ┌─ lofi      built-in engine · CPU only · no model · no download
   ├─ lofify    YOUR existing songs → lofi versions of them
   └─ generate  ACE-Step 1.5 · its own API, or ComfyUI (GPU)
                        │
                        ▼
                  master      trim · −14 LUFS · −1 dBTP true peak
                        │
                        ▼
      ┌──────── song ───┴─── mix ────────┐
      │  ONE continuous piece:            │  classic compilation:
      │  beat-matched, key-matched,       │  tracks back to back
      │  bar-aligned transitions,         │  with crossfades
      │  continuous vinyl + drum spine    │
      └────────────────┬──────────────────┘
                       ▼
              video  →  check  →  upload
                ▲
                └─ art   ComfyUI makes the picture and a seamless loop (optional)

  Everything comes out as tagged MP3 by default.
```

Built for a **Windows laptop, 16 GB RAM, GTX 1660 Ti (6 GB)**, but the built-in
engine needs no GPU at all and runs anywhere Python does.

---

## 1. Install (Windows, ~5 minutes)

1. Install **Python 3.11** from python.org — tick *Add python.exe to PATH*.
2. Optional: **ffmpeg** — `winget install Gyan.FFmpeg`, then open a **new**
   terminal. Only the `video` step needs it now; MP3 is written by `pedalboard`
   (LAME) directly.
3. In this folder, double-click `setup_windows.bat` — or do it by hand:

   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Check the machine:

   ```bat
   python pipeline.py doctor
   ```

   It reports RAM, VRAM, disk, missing packages, and warns about the GTX
   16-series half-precision problem (see §6).

---

## 2. Make something in one command

```bat
python pipeline.py all --count 12 --hours 1 --spine 0.5
```

Generates 12 original tracks, then beat-matches them into **one continuous
1-hour lofi song** with a tracklist ready to paste into a YouTube description.
On a laptop CPU this takes roughly 15–25 minutes for an hour of finished audio.

Step by step, so you can throw out what you don't like:

```bat
python pipeline.py lofi --count 20 --minutes 3      :: 20 tracks, 3 min each
python pipeline.py dedupe                           :: flag near-identical ones
                                                    :: then delete what you dislike
python pipeline.py song --hours 2 --spine 0.5       :: one 2-hour song
python pipeline.py art  --prompt "rainy window"     :: artwork + a seamless loop
python pipeline.py video --loop runs\<newest>\loop.mp4
python pipeline.py check --log                      :: before you upload
```

Or start from music you already have:

```bat
python pipeline.py lofify "C:\my songs" --preset study --drums add --i-own-this
python pipeline.py song --hours 1
```

Every run gets its own folder under `runs\`:

| File | What it is |
|---|---|
| `raw\` | the generated (or lofi-ed) tracks, as tagged MP3 |
| `mastered\` | trimmed, −14 LUFS, safe peaks (ACE-Step path) |
| `art\` | ComfyUI artwork, if you ran `art` |
| `manifest.json` | every setting, seed, chord and instrument used |
| `song.mp3` | the single continuous song |
| `song_tracklist.txt` | chapter timestamps for the description |
| `song_report.json` | tempo, key, per-track stretch/pitch decisions |
| `song.mp4` | final upload |

---

## 3. The three generators

### `lofi` — the built-in engine (default)

Writes the music itself: chord progressions, voicings, a motif-based melody,
a synthesised boom-bap kit, then tape wow, vinyl crackle, sidechain pump and a
mastering chain. **Every sound is generated from scratch** — no samples, no
loops, no third-party recordings anywhere in the signal path.

- Runs on the CPU. It does not touch your GPU, so it can render while ACE-Step
  is generating.
- Roughly **18× realtime** on a laptop CPU — an hour of music in ~3 minutes.
- Renders section by section and streams to disk, so a 6-hour render uses a few
  hundred MB of RAM, not 8 GB.

```bat
python pipeline.py lofi --style jazzhop --count 8 --minutes 4
python pipeline.py lofi --style sleep --count 4 --minutes 10 --bpm 62
python pipeline.py lofi --minutes 180 --count 1        :: one 3-hour track
```

Styles: `lofi`, `study`, `sleep`, `jazzhop`, `boombap`, `rainy`, `piano`,
`ambient`, `devotional`. Add your own in `presets.json` under `engine.styles` —
copy a block, rename it, and `--style yourname` works immediately.

Useful knobs: `--bpm`, `--key "A Minor"`, `--swing 0.25`, `--vinyl 0`,
`--tape 0`, `--seed 123` (repeat a run exactly).

### `generate` — ACE-Step 1.5 (optional)

> Which model should you use? `python pipeline.py models` prints the options
> with their **licences** and VRAM, marked up for your card. Read it before you
> pick one — see §6.

For a neural generator's sound. Needs the ACE-Step Windows portable package
running its API server on `127.0.0.1:8001`.

```bat
python pipeline.py doctor --write-env "C:\ACE-Step\.env"   :: writes 6 GB settings
:: start_api_server.bat, wait for http://127.0.0.1:8001/health to say ok
python pipeline.py generate --genre lofi --count 10 --duration 150
python pipeline.py master
```

`doctor --write-env` writes the low-VRAM configuration (turbo checkpoint, no
caption LLM, CPU offload, float32) and backs up any existing `.env` first.

No GPU handy? `python mock_server.py` in a second terminal fakes the API so you
can rehearse the whole flow.

### `lofify` — songs you already have

Takes finished music and rebuilds it as lofi. The moves, in the order they
belong:

1. work out the tempo, key and downbeat,
2. deal with the vocal — proper separation if `demucs` is installed, the
   centre-channel trick if not,
3. slow it down, pitch dropping with it like a tape running slow,
4. run the colour chain (lowpass, wobble, bit reduction, room),
5. optionally lay a boom-bap kit under it, **beat-locked to the new tempo**,
6. sit it on a continuous vinyl bed,
7. master to −14 LUFS with a true-peak ceiling.

```bat
python pipeline.py lofify "C:\my songs" --i-own-this
python pipeline.py lofify track.mp3 --preset sleep --i-own-this
python pipeline.py lofify *.flac --preset study --drums add --i-own-this
```

**Presets** (each is just a bundle of the flags below):

| Preset | Speed | Vocals | Character |
|---|---|---|---|
| `classic` | 0.88 | reduced | the familiar one |
| `slowed` | 0.82 | kept | "slowed + reverb" |
| `study` | 0.90 | removed | + a lazy kit under it |
| `sleep` | 0.78 | removed | very dark, 4.2 kHz lowpass |
| `tape` | 0.92 | reduced | heavy wobble, 10-bit, a little GSM grit |
| `instrumental` | 1.00 | removed | same tempo, just the backing |

Anything is overridable: `--speed 0.85`, `--keep-pitch` (slow without dropping
pitch), `--vocals {keep,reduce,remove}`, `--vocal-amount 0.6`, `--drums add`,
`--drum-pattern shuffle`, `--amount 0.7`, `--lowpass 5000`, `--bitcrush 10`,
`--vinyl 1.2`, `--reverb 0.5`, `--telephone 0.3`, `--mp3-artifacts 0.4`.

Output feeds straight into `song`, so you can turn an album into one continuous
hour.

#### Vocal removal: what you actually get

Without `demucs`, vocals are removed with the **centre-channel trick** —
subtracting what is identical in both speakers. It is free, instant, and on the
test material it pulled a dead-centre vocal down by **54–68 dB**. But it only
works on vocals mixed dead centre, and it thins anything else that is centred
(the low end is put back untouched to protect the kick and bass).

With `pip install demucs` it uses real source separation instead, which handles
vocals that are not centred and does not touch the rest of the mix. It pulls in
torch and a few hundred MB of weights, and wants ~3 GB of VRAM — fine on your
card, but slow on CPU. The pipeline detects it automatically and tells you which
one it used.

#### Rights

`lofify` refuses to run without `--i-own-this`. A lofi remix of someone else's
record is still their record — and Content ID matches *edited* audio, so slowing
it down and adding crackle changes nothing legally or technically. Use it on
your own music, music licensed to you, or public domain. `check` records which
files a run was built from, and says plainly that nothing here can verify the
claim.

---

## 4. Turning a list of tracks into ONE lofi song

This is what `song` does, and it is different from `mix`.

`mix` plays your tracks one after another with a crossfade — a compilation.
`song` makes them **one piece**:

- analyses every track's tempo, key and first downbeat,
- picks one target tempo and one target key for the whole thing,
- **time-stretches** each track onto that tempo (WSOLA, so drums stay crisp),
- **pitch-shifts** each onto a compatible key (relative majors/minors are left
  alone — they already share every note),
- trims each to whole bars and joins them with **bar-aligned** transitions that
  swap the bass over so two basses never stack into mud,
- pulls every track toward one tonal balance (`--tone-match`),
- and runs a **continuous** vinyl bed, an optional never-stopping drum groove
  (`--spine`) and one shared master chain underneath the whole thing.

The result holds one tempo from beginning to end — measured across a 10-minute
build, every 60-second window reads the same BPM, transitions included.

```bat
python pipeline.py song --hours 3 --spine 0.6
python pipeline.py song --input "C:\my beats" --hours 1 --bpm 78 --key "A Minor"
python pipeline.py song --input playlist.txt --minutes 45 --order shuffle
```

`--input` takes a folder, a list of files, or a `.txt` playlist (one path per line).

| Flag | Does |
|---|---|
| `--hours N` / `--minutes N` | target length; tracks repeat (with slight tempo/pitch drift each pass) until it is reached |
| `--spine 0..1` | continuous drum groove under everything, so the beat never stops |
| `--vinyl 0..1.5` | continuous vinyl/tape noise bed |
| `--lofi 0..1` | extra crunch per track (lowpass, saturation, bit reduction) |
| `--crossfade-bars N` | length of each transition, in bars |
| `--order` | `harmonic` (least key movement), `shuffle`, `asis` |
| `--max-stretch 18` | biggest tempo change allowed, in percent |
| `--max-shift 4` | biggest pitch shift allowed, in semitones |
| `--tone-match 0..1` | how hard to pull every track toward one tonal balance |

If a track is too far from the target tempo to stretch cleanly, it is left alone
and named in the output rather than mangled.

### Hours-long output

Everything streams. Length is limited by disk, not RAM:

| Length | 24-bit WAV | `lofi` render | `song` build |
|---|---|---|---|
| 1 hour | ~1.1 GB | ~2.5 min | ~3.5 min |
| 3 hours | ~3.2 GB | ~7 min | ~10 min |
| 6 hours | ~6.4 GB | ~14 min | ~21 min |

Measured: 45 minutes of audio rendered in 104 s (26x realtime) with a **peak
memory use of 462 MB** — the same buffer held whole in RAM would have been
960 MB, and a 6-hour one would be 7.6 GB. Memory stays flat as length grows,
because audio is written out section by section and the effects keep their own
state instead of needing the whole timeline.

`song` and `mix` check free disk before they start.

---

## 5. ComfyUI (optional) — the picture, and another way to run ACE-Step

If you already have ComfyUI, the pipeline can drive it. Two uses:

### The visual — `art`

This closes the last manual gap: until now you had to supply your own
`--loop my_loop.mp4`. Now:

```bat
python pipeline.py art --prompt "cozy attic at night, rain on the window, warm lamp"
python pipeline.py video --loop runs\<newest>\loop.mp4
```

`art` generates the image in ComfyUI, then builds a **seamless video loop** from
it — a slow drift whose zoom starts and ends at exactly 1.0 with zero velocity
(a raised cosine), so it repeats forever with no visible jump. That matters
because it means the final video is a **remux, not an encode**: a 3-hour upload
is assembled with `-c:v copy` in a couple of minutes instead of re-encoding
three hours of frames.

Measured on the wrap-around frame: the seam costs 1.3× an ordinary frame step,
while the midpoint differs by 24× — real motion, invisible loop point.

```bat
python pipeline.py art --count 4                      :: pick your favourite
python pipeline.py art --loop-seconds 60 --zoom 0.18   :: longer, stronger drift
python pipeline.py art --loop-seconds 0                :: still image only
python pipeline.py art --nvenc                         :: GPU-encode the loop
```

### ACE-Step through ComfyUI — `generate --backend comfy`

```bat
python pipeline.py generate --backend comfy --genre lofi --count 10
```

Worth doing on a 6 GB card: ComfyUI's memory management is better than the
standalone ACE-Step server's, and it supports ACE-Step natively. Start it with
the flags in §6.

### Using your own workflows

The shipped `workflows/art.json` and `workflows/acestep.json` are starting
points. Node names and checkpoint filenames differ between installs, so the
reliable path is to build what you want **in ComfyUI**, confirm it runs, then
**Workflow → Export (API)** and point at it:

```bat
python pipeline.py art --workflow my_export.json --show-workflow   :: see what you can patch
python pipeline.py art --workflow my_export.json --set SAMPLER.cfg=6.5
```

Values are matched by node title first (`POSITIVE`, `NEGATIVE`), then class
type, then any node that already has that input. Inputs wired to another node
are never overwritten, and a field a node doesn't have is refused rather than
silently ignored — patching ACE-Step's prompt into `text` instead of `tags`
looks like it worked and does nothing, so the pipeline checks.

If the workflow names a checkpoint ComfyUI can't see, you get the list of what
it *can* see instead of a bare error.

No ComfyUI installed? `python mock_comfy.py` fakes it well enough to rehearse
the whole flow.

---

## 6. Which generation model

```bat
python pipeline.py models                    :: detects your VRAM
python pipeline.py models --commercial-only  :: hide what you cannot monetise
```

The pipeline is **model-agnostic**. `generate` drives ACE-Step over its REST
API, `generate --backend comfy` drives any API-format ComfyUI workflow, and the
built-in engine needs no model at all. So adding a model is usually **adding a
workflow file, not writing code**: build it in ComfyUI, *Workflow → Export
(API)*, then `generate --backend comfy --workflow yours.json`.

What is *not* interchangeable is the licence.

| Model | Licence | Commercial? | VRAM | Length |
|---|---|---|---|---|
| **ACE-Step 1.5** | Apache 2.0 | **yes** | 8 GB (less with offload) | full songs |
| **DiffRhythm 2** | Apache 2.0 | **yes** | ~8 GB | full songs |
| MusicGen (all sizes) | code MIT, **weights CC-BY-NC** | **no** | 4 GB | ~30 s |
| Stable Audio Open 1.0 | Stability Community | conditional | ~8 GB | 47 s |
| Stable Audio Open Small | Stability Community | conditional | ~4 GB | 11 s |
| YuE 7B | see repo | check | 24 GB+ | full songs |
| **this pipeline's engine** | this repo | **yes** | none | unlimited |

**The trap worth knowing about.** MusicGen fits a 6 GB card easily, sounds good,
and its *code* is MIT — but its **weights are CC-BY-NC 4.0**, which forbids
commercial use. A monetised YouTube channel is commercial use. The code licence
does not carry over to the weights. `models --commercial-only` hides it.

**For your setup, ACE-Step is already the right answer** and is already wired in
two ways: Apache 2.0 covers the weights as well as the code, and `doctor
--write-env` configures the CPU offload that brings it under 8 GB.

**DiffRhythm 2** is the one genuine alternative — also Apache 2.0, also built
for full-length songs. It needs a ComfyUI workflow; nothing in the code has to
change.

Stable Audio Open is worth knowing for a different reason: it was trained *only*
on CC-licensed Freesound and Free Music Archive audio with suspected copyrighted
material screened out, which is the clearest training-data provenance of the
bunch. But 47 seconds is too short for songs — treat it as a texture generator.

Licences change, and none of this is legal advice. Every entry in
`pipeline.py models` carries the URL it came from.

---

## 7. Output format, and the libraries doing the work

### Everything is MP3

MP3 is the default for every file the pipeline writes. A 3-hour 24-bit WAV is
**3.2 GB**; the same thing at 320 kbps is **430 MB**, which matters on a laptop.
It is written by `pedalboard` (LAME) **block by block**, so an hours-long MP3 is
never staged through a giant WAV first.

```bat
python pipeline.py lofi --count 8                    :: MP3, 320 kbps
python pipeline.py song --hours 2 --mp3-quality V0   :: VBR instead
python pipeline.py song --hours 2 --format wav       :: lossless, if you prefer
```

`--format {mp3,wav,flac}` and `--mp3-quality {320,256,192,V0,V2}` work on every
command that writes audio.

**One honest caveat.** Generating to MP3 and then building a `song` from those
MP3s means two encode generations. At 320 kbps that is inaudible for this
material, but if you intend to master elsewhere, generate with `--format wav`
and only encode at the end.

Every file gets **ID3 tags** (title, artist, album, track, BPM, key, genre, and
the AI-disclosure note), and the final song embeds your ComfyUI artwork as cover
art if you ran `art`. `--no-tags` turns that off; `--artist` / `--album` /
`--title` set the fields.

### What each library is for

| Library | Why it is here | Required? |
|---|---|---|
| `numpy`, `scipy` | the engine, the DSP, the analysis | yes |
| `soundfile` | WAV/FLAC I/O | yes |
| `pyloudnorm` | BS.1770 loudness measurement | yes |
| **`pedalboard`** | MP3 I/O (no ffmpeg), Rubber Band stretch/pitch, the effect chain, true-peak brickwall | **strongly recommended** |
| `mutagen` | ID3 tags and embedded cover art | recommended |
| `requests` | the ACE-Step and ComfyUI backends | for those only |
| `demucs` | real stem separation for `lofify --vocals remove` | optional |
| `numba` | compiles the envelope followers, ~2× faster mastering | optional |
| `librosa` | a second opinion on tempo/key | optional |

`pedalboard` is the one that earns its place twice over. It gives MP3 without
ffmpeg, and its Rubber Band time-stretch measured **0.0 cents** of pitch error
against ±0.6 cents for the built-in one — which matters because `song` pitch-
shifts every track to a common key. Everything still works without it: the
pipeline falls back to its own numpy DSP and tells you which backend it used.

### True-peak limiting

Worth knowing, because it is the difference between compliant and clipping.
A limiter that only looks at *samples* and stops at −1.0 dBFS can still
reconstruct at **+2.25 dBTP** on dense material — and that overshoot clips in
any lossy encode, which is exactly what MP3 is. The limiter's detector therefore
runs on a 4× oversampled copy, so it accounts for what the waveform does
*between* samples.

Measured after encoding to MP3 and decoding back: **−1.0 dBTP**, on the nose.

---

## 8. Before you upload — `check`

```bat
python pipeline.py check --log
```

It reports:

- **Loudness and true peak** against YouTube's −14 LUFS / −1 dBTP.
- **Chapters** — at least 3, first one at `00:00`, or YouTube won't show them.
- **Provenance** — read from the run manifest: whether the audio was fully
  synthesised locally, model-generated, or built from a reference you supplied.
- **Duplicates against everything you have exported before.** This is the part
  that matters most in practice. A channel that gets claimed is usually being
  claimed by *itself* — the same bed re-uploaded, or material a distributor
  already registered. `--log` records each export's fingerprint in
  `upload_ledger.json`, and later checks compare against it two ways: a
  constellation fingerprint (catches the same render again) and a coarse
  musical signature (flags a transposed copy; see the note below).

`python pipeline.py dedupe` does the same comparison *within* a batch, so you
don't put two near-identical tracks in one upload.

**What the duplicate check does and doesn't catch.** The constellation
fingerprint is decisive for the case that actually matters — the same render
again: an identical file scores 1.0 where unrelated tracks score under 0.15, so
it flags a re-export and never cries wolf on a genuinely new track. The second,
harmony-based check catches a *transposed* copy (0.01–0.12 against 0.35+ for
unrelated music) but will miss a heavily time-stretched one, because tempo
detection is not reliable enough to line the two up. So its findings are
reported as advisories (`-`), not blockers (`!`): treat them as "worth a
listen", not a verdict. Precision was chosen over coverage deliberately — a
check that flags your new tracks as duplicates is worse than one that
occasionally stays quiet.

### What this can and cannot do about copyright claims

Straight answer: **no tool can promise a platform will never flag a file**, and
anyone who tells you otherwise is selling something. False positives happen, and
so do disputes over material that is entirely yours.

What this pipeline actually does is remove the real causes:

- The built-in engine synthesises **every sound from scratch** — no sample packs,
  no loops, no scraped audio. There is no third-party recording in the output to
  match against.
- `lofify` is the exception, and it is gated: it rebuilds audio *you* supply, so
  it refuses to run without `--i-own-this`, and `check` names the source files
  in its report. Slowing a record down and adding crackle does not make it
  yours — Content ID matches edited audio, and the edit is a derivative work
  either way.
- `--ref-mode cover` (which rebuilds an existing recording) is **blocked** unless
  you pass `--i-own-this`. Covering someone else's record is the single most
  reliable way to get claimed, and it is a copyright question, not a detection one.
- The ledger stops you re-shipping your own material, which is the most common
  self-inflicted claim.
- `check` verifies the technical requirements so a claim is not the thing that
  goes wrong.

One policy point, not a technical one: YouTube asks you to tick **"Altered or
synthetic content"** in Studio when audio is AI-generated or synthesised. This
pipeline's output is. Ticking it costs you nothing; not ticking it is a policy
breach. Vary genres, moods and visuals between uploads too — mass-produced,
near-identical uploads are judged on their own terms regardless of who owns them.

---

## 9. GTX 1660 Ti / 6 GB notes

- **Half precision is broken on GTX 16-series cards.** Turing TU116/TU117 have no
  tensor cores and a well-known fp16 path that yields NaNs — in practice, silence
  or noise instead of music. `doctor --write-env` sets `float32`. Slower, correct.
- **6 GB VRAM:** keep `batch_size` at 1 (the pipeline forces it; the ACE-Step
  default of 2 will OOM), `--duration` around 120–150 s, and the caption LLM off.
- If a track OOMs, `generate` automatically retries it shorter and tells you what
  to pass next time.
- **Don't run `video --nvenc` while ACE-Step is generating** — they share the GPU.
- The built-in engine uses no VRAM at all, so `lofi` and ACE-Step can run together.
- **ComfyUI** needs the same treatment — start it with:

  ```bat
  python main.py --lowvram --force-fp32 --fp32-vae --use-split-cross-attention
  ```

  `--force-fp32` is the same story as ACE-Step: on a 16-series card fp16 gives
  you black images and silent audio, not speed. `doctor` prints this line for
  you and tells you whether ComfyUI is currently running.
- **16 GB RAM:** everything long-form streams; a 6-hour render stays in the low
  hundreds of MB. Close browser tabs playing video before long jobs.
- `pip install numba` is optional and makes the mastering chain roughly twice as
  fast (it compiles the envelope followers).

---

## 10. Command reference

| Command | Purpose |
|---|---|
| `doctor` | check this PC; `--write-env` writes tuned ACE-Step settings |
| `lofi` | generate tracks with the built-in engine (no GPU) |
| `lofify` | turn songs you already have into lofi (needs `--i-own-this`) |
| `generate` | generate tracks with ACE-Step (`--backend comfy` to route via ComfyUI) |
| `art` | generate cover art in ComfyUI + a seamless video loop |
| `master` | trim, loudness-normalise and limit each track |
| `song` | beat-match tracks into one continuous song |
| `mix` | crossfaded compilation + chapters |
| `video` | put audio under a looping clip or still image |
| `analyze` | tempo, key and downbeat of any file |
| `check` | originality + upload-readiness report (`--log` to record) |
| `dedupe` | find near-duplicate tracks in a folder |
| `models` | which generation models fit your card, and what their licences allow |
| `ledger` | list everything you have exported |
| `all` | generate → song → video in one go |

Shared on every command that writes audio: `--format {mp3,wav,flac}`,
`--mp3-quality`, `--artist`, `--album`, `--title`, `--no-tags`.

`python pipeline.py <command> -h` shows every option.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `ffmpeg not found` | `winget install Gyan.FFmpeg`, then open a **new** terminal |
| Silence or noise from ACE-Step | fp16 on a 16-series card — run `doctor --write-env` and restart the server |
| `CUDA out of memory` | lower `--duration`, close games/OBS/video tabs, restart the server |
| Song sounds repetitive | generate more tracks; `check` tells you how many unique ones were reused |
| Chapters don't show on YouTube | need 3+, first at `00:00` — `check` verifies this |
| A track ignored the target tempo | it was outside `--max-stretch`; raise it or set `--bpm` closer |
| Mix is muddy | lower `--spine`, or raise `--tone-match` toward 1.0 |
| Out of disk mid-render | 24-bit WAV is ~1.1 GB/hour; `song` warns up front |
| ComfyUI: black images / silent audio | fp16 on a 16-series card — restart it with `--force-fp32 --fp32-vae` |
| ComfyUI: "has no checkpoint called ..." | the workflow names a model you don't have; the error lists what you do have — pick one with `--set CHECKPOINT.ckpt_name=<name>` |
| ComfyUI: "not an API workflow" | use **Workflow → Export (API)**, not plain Save |
| "every track is shorter than a N-bar transition" | lower `--crossfade-bars`, or generate longer tracks |
| `lofify` refuses to start | it needs `--i-own-this` — see the rights note in §3 |
| Vocals still audible after `--vocals remove` | they are not mixed dead centre; `pip install demucs` for real separation |
| "writing MP3 needs pedalboard" | `pip install pedalboard`, or use `--format wav` |
| Tags missing from the MP3s | `pip install mutagen` |
| Lofi version sounds too muddy | lower `--amount`, or raise `--lowpass` |
| Not sure which model to use | `python pipeline.py models` — it checks your VRAM and flags the non-commercial weights |
