# Local lofi music pipeline

Everything runs on your own PC. Nothing is uploaded, nothing is streamed, no API keys.

```
                 ┌─ lofi      built-in engine, CPU only, no model, no download
  generate ──────┤
                 └─ generate  ACE-Step 1.5 over its local REST API (optional, GPU)
                        │
                        ▼
                  master      trim · −14 LUFS · −1 dBTP
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
```

Built for a **Windows laptop, 16 GB RAM, GTX 1660 Ti (6 GB)**, but the built-in
engine needs no GPU at all and runs anywhere Python does.

---

## 1. Install (Windows, ~5 minutes)

1. Install **Python 3.11** from python.org — tick *Add python.exe to PATH*.
2. Optional but recommended: **ffmpeg** — `winget install Gyan.FFmpeg`, then open a
   **new** terminal. (Only needed for MP3 and video; WAV/FLAC work without it.)
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
python pipeline.py video --loop my_loop.mp4 --nvenc :: put it under a video
python pipeline.py check --log                      :: before you upload
```

Every run gets its own folder under `runs\`:

| File | What it is |
|---|---|
| `raw\` | the generated tracks |
| `mastered\` | trimmed, −14 LUFS, safe peaks (ACE-Step path) |
| `manifest.json` | every setting, seed, chord and instrument used |
| `song.wav` | the single continuous song |
| `song_tracklist.txt` | chapter timestamps for the description |
| `song_report.json` | tempo, key, per-track stretch/pitch decisions |
| `song.mp4` | final upload |

---

## 3. The two generators

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

## 5. Before you upload — `check`

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

## 6. GTX 1660 Ti / 6 GB notes

- **Half precision is broken on GTX 16-series cards.** Turing TU116/TU117 have no
  tensor cores and a well-known fp16 path that yields NaNs — in practice, silence
  or noise instead of music. `doctor --write-env` sets `float32`. Slower, correct.
- **6 GB VRAM:** keep `batch_size` at 1 (the pipeline forces it; the ACE-Step
  default of 2 will OOM), `--duration` around 120–150 s, and the caption LLM off.
- If a track OOMs, `generate` automatically retries it shorter and tells you what
  to pass next time.
- **Don't run `video --nvenc` while ACE-Step is generating** — they share the GPU.
- The built-in engine uses no VRAM at all, so `lofi` and ACE-Step can run together.
- **16 GB RAM:** everything long-form streams; a 6-hour render stays in the low
  hundreds of MB. Close browser tabs playing video before long jobs.
- `pip install numba` is optional and makes the mastering chain roughly twice as
  fast (it compiles the envelope followers).

---

## 7. Command reference

| Command | Purpose |
|---|---|
| `doctor` | check this PC; `--write-env` writes tuned ACE-Step settings |
| `lofi` | generate tracks with the built-in engine (no GPU) |
| `generate` | generate tracks with ACE-Step |
| `master` | trim, loudness-normalise and limit each track |
| `song` | beat-match tracks into one continuous song |
| `mix` | crossfaded compilation + chapters |
| `video` | put audio under a looping clip or still image |
| `analyze` | tempo, key and downbeat of any file |
| `check` | originality + upload-readiness report (`--log` to record) |
| `dedupe` | find near-duplicate tracks in a folder |
| `ledger` | list everything you have exported |
| `all` | generate → song → video in one go |

`python pipeline.py <command> -h` shows every option.

---

## 8. Troubleshooting

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
