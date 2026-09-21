"""Music theory for the local lofi engine: notes, scales, jazz chords, progressions.

Everything here is plain data + small pure functions, so the generated music is
deterministic for a given seed and every choice is auditable in the manifest.
"""

from __future__ import annotations

import random

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
                 "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}

SCALES = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "minor":          [0, 2, 3, 5, 7, 8, 10],
    "natural minor":  [0, 2, 3, 5, 7, 8, 10],
    "harmonic minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic minor":  [0, 2, 3, 5, 7, 9, 11],
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
    "phrygian":       [0, 1, 3, 5, 7, 8, 10],
    "lydian":         [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":     [0, 2, 4, 5, 7, 9, 10],
    "aeolian":        [0, 2, 3, 5, 7, 8, 10],
    "pentatonic":     [0, 2, 4, 7, 9],
    "minor pentatonic": [0, 3, 5, 7, 10],
}

#: Chord quality -> semitone offsets from the chord root.
CHORDS = {
    "maj":    [0, 4, 7],
    "min":    [0, 3, 7],
    "dim":    [0, 3, 6],
    "aug":    [0, 4, 8],
    "sus2":   [0, 2, 7],
    "sus4":   [0, 5, 7],
    "maj6":   [0, 4, 7, 9],
    "min6":   [0, 3, 7, 9],
    "maj7":   [0, 4, 7, 11],
    "min7":   [0, 3, 7, 10],
    "dom7":   [0, 4, 7, 10],
    "m7b5":   [0, 3, 6, 10],
    "dim7":   [0, 3, 6, 9],
    "minmaj7": [0, 3, 7, 11],
    "maj9":   [0, 4, 7, 11, 14],
    "min9":   [0, 3, 7, 10, 14],
    "dom9":   [0, 4, 7, 10, 14],
    "add9":   [0, 4, 7, 14],
    "madd9":  [0, 3, 7, 14],
    "maj69":  [0, 4, 7, 9, 14],
    "min11":  [0, 3, 7, 10, 14, 17],
    "maj7s11": [0, 4, 7, 11, 18],
    "dom13":  [0, 4, 7, 10, 14, 21],
    "min7add11": [0, 3, 7, 10, 17],
}

#: Lofi/jazz chord colour: which qualities may stand in for a plain triad.
COLOUR_SWAPS = {
    "maj": ["maj7", "maj9", "maj69", "add9", "maj6"],
    "min": ["min7", "min9", "min11", "madd9", "min6"],
    "dom": ["dom7", "dom9", "dom13"],
    "dim": ["m7b5", "dim7"],
}

#: Roman-numeral progressions that carry the genre.  Degrees are 0-indexed
#: scale steps; "b" prefixes a borrowed (flattened) degree.
PROGRESSIONS_MAJOR = [
    ("I-vi-ii-V",        [(0, "maj"), (5, "min"), (1, "min"), (4, "dom")]),
    ("IV-V-iii-vi",      [(3, "maj"), (4, "dom"), (2, "min"), (5, "min")]),
    ("ii-V-I-vi",        [(1, "min"), (4, "dom"), (0, "maj"), (5, "min")]),
    ("I-IV-vi-V",        [(0, "maj"), (3, "maj"), (5, "min"), (4, "dom")]),
    ("vi-IV-I-V",        [(5, "min"), (3, "maj"), (0, "maj"), (4, "dom")]),
    ("I-iii-IV-iv",      [(0, "maj"), (2, "min"), (3, "maj"), (3, "min")]),
    ("IV-iii-vi-I",      [(3, "maj"), (2, "min"), (5, "min"), (0, "maj")]),
    ("I-bVII-IV-I",      [(0, "maj"), ("b6", "maj"), (3, "maj"), (0, "maj")]),
    ("ii-V-iii-vi",      [(1, "min"), (4, "dom"), (2, "min"), (5, "min")]),
    ("I-vi-IV-V",        [(0, "maj"), (5, "min"), (3, "maj"), (4, "dom")]),
]

PROGRESSIONS_MINOR = [
    ("i-VI-III-VII",     [(0, "min"), (5, "maj"), (2, "maj"), (6, "maj")]),
    ("i-iv-VII-III",     [(0, "min"), (3, "min"), (6, "maj"), (2, "maj")]),
    ("i-VII-VI-V",       [(0, "min"), (6, "maj"), (5, "maj"), (4, "dom")]),
    ("i-iv-v-i",         [(0, "min"), (3, "min"), (4, "min"), (0, "min")]),
    ("iiø-V-i-i",        [(1, "dim"), (4, "dom"), (0, "min"), (0, "min")]),
    ("i-VI-iiø-V",       [(0, "min"), (5, "maj"), (1, "dim"), (4, "dom")]),
    ("i-III-VII-iv",     [(0, "min"), (2, "maj"), (6, "maj"), (3, "min")]),
    ("i-v-VI-IV",        [(0, "min"), (4, "min"), (5, "maj"), (3, "min")]),
    ("iv-i-V-i",         [(3, "min"), (0, "min"), (4, "dom"), (0, "min")]),
    ("i-VI-VII-i",       [(0, "min"), (5, "maj"), (6, "maj"), (0, "min")]),
]


def note_to_midi(name: str, octave: int = 4) -> int:
    name = name.strip()
    name = FLAT_TO_SHARP.get(name, name)
    if name not in NOTE_NAMES:
        raise ValueError(f"unknown note {name!r}")
    return 12 * (octave + 1) + NOTE_NAMES.index(name)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"


def parse_key(text: str):
    """'A Minor' / 'Bb dorian' / 'C' -> (root_pitch_class, scale_name)."""
    if not text:
        return 9, "minor"
    parts = str(text).replace("-", " ").split()
    root = FLAT_TO_SHARP.get(parts[0].capitalize(), parts[0].capitalize())
    if root not in NOTE_NAMES:
        root = "A"
    scale = " ".join(parts[1:]).lower().strip() if len(parts) > 1 else "minor"
    scale = {"": "minor", "maj": "major", "min": "minor", "m": "minor"}.get(scale, scale)
    if scale not in SCALES:
        scale = "minor" if "min" in scale else "major"
    return NOTE_NAMES.index(root), scale


def key_name(root_pc: int, scale: str) -> str:
    return f"{pretty_note(root_pc, root_pc)} {scale.title()}"


def scale_degrees(root_pc: int, scale: str) -> list:
    return [(root_pc + s) % 12 for s in SCALES.get(scale, SCALES["minor"])]


def degree_to_semitone(degree, scale: str) -> int:
    steps = SCALES.get(scale, SCALES["minor"])
    if isinstance(degree, str) and degree.startswith("b"):
        idx = int(degree[1:]) % len(steps)
        return steps[idx] - 1
    return steps[int(degree) % len(steps)] + 12 * (int(degree) // len(steps))


def diatonic_triad(degree: int, scale: str):
    """Stack thirds inside the scale -> the triad quality actually in the key.

    This is what keeps Dorian's major IV and Mixolydian's minor v intact
    instead of forcing every progression into plain major/minor.
    """
    steps = SCALES.get(scale, SCALES["minor"])
    n = len(steps)
    def step(i):
        return steps[i % n] + 12 * (i // n)
    d = int(degree)
    root, third, fifth = step(d), step(d + 2), step(d + 4)
    t, f = third - root, fifth - root
    if t == 4 and f == 7:
        return "maj"
    if t == 3 and f == 7:
        return "min"
    if t == 3 and f == 6:
        return "dim"
    if t == 4 and f == 8:
        return "aug"
    if t == 2 or t == 5:
        return "sus4" if t == 5 else "sus2"
    return "maj" if t >= 4 else "min"


def build_chord(root_midi: int, quality: str) -> list:
    return [root_midi + s for s in CHORDS.get(quality, CHORDS["maj"])]


def colourise(quality: str, rng: random.Random, richness: float = 0.85,
              root_midi: int = 60, scale_pcs=None, allow_leading_tone: bool = False) -> str:
    """Turn a plain triad into the 7th/9th voicings lofi lives on.

    Candidates are filtered against the key so a colour never drags in a note
    from outside the scale - the one exception is the dominant chord in minor,
    which needs its raised leading tone (V7 in a minor key).
    """
    family = quality if quality in COLOUR_SWAPS else None
    if family is None:
        return quality
    plain = {"maj": "maj7", "min": "min7", "dom": "dom7", "dim": "m7b5"}[family]
    candidates = list(COLOUR_SWAPS[family])
    if scale_pcs:
        allowed = set(scale_pcs)
        if allow_leading_tone:
            allowed = allowed | {(root_midi + 4) % 12}   # major 3rd of the V chord
        candidates = [q for q in candidates
                      if all((root_midi + s) % 12 in allowed for s in CHORDS[q])]
        if all((root_midi + s) % 12 in allowed for s in CHORDS[plain]):
            candidates.append(plain)
    if not candidates:
        return plain
    if rng.random() > richness:
        return plain if plain in candidates else candidates[0]
    return rng.choice(candidates)


def voice_chord(notes, low=52, high=76, spread=True):
    """Fold a chord into a comfortable register and avoid low-interval mud."""
    voiced = []
    for n in notes:
        while n < low:
            n += 12
        while n > high:
            n -= 12
        voiced.append(n)
    voiced = sorted(set(voiced))
    # kill semitone clashes in the low half - they turn to mud on small speakers
    cleaned = []
    for n in voiced:
        if cleaned and n - cleaned[-1] == 1 and n < 64:
            continue
        cleaned.append(n)
    if spread and len(cleaned) >= 4 and cleaned[0] - 12 >= low - 12:
        cleaned[0] -= 12  # drop-2-ish: open the voicing so it does not sound blocky
        cleaned = sorted(cleaned)
    return cleaned


def make_progression(root_pc: int, scale: str, rng: random.Random, bars: int = 4,
                     richness: float = 0.85):
    """Pick a progression and realise it as concrete, in-key chords.

    Returns (label, [{root_midi, quality, notes, bass_midi, label}, ...]).
    """
    minorish = scale in {"minor", "natural minor", "harmonic minor", "aeolian",
                         "dorian", "phrygian", "minor pentatonic", "melodic minor"}
    #: only the true minor family takes the harmonic-minor V7; Dorian and
    #: Phrygian keep their own (modal) fifth, which is the whole point of them.
    takes_v7 = scale in {"minor", "natural minor", "aeolian", "harmonic minor",
                         "melodic minor", "minor pentatonic"}
    scale_set = set(SCALES.get(scale, SCALES["minor"]))
    table = PROGRESSIONS_MINOR if minorish else PROGRESSIONS_MAJOR
    label, degrees = rng.choice(table)
    if bars > len(degrees):
        reps = (bars + len(degrees) - 1) // len(degrees)
        degrees = (degrees * reps)[:bars]
    elif bars < len(degrees):
        degrees = degrees[:bars]

    scale_pcs = scale_degrees(root_pc, scale)
    base = note_to_midi(NOTE_NAMES[root_pc % 12], 4)
    chords = []
    for degree, family in degrees:
        semis = degree_to_semitone(degree, scale)
        root_midi = base + semis
        borrowed = isinstance(degree, str)
        if borrowed:
            actual = family                       # borrowed chords keep the written quality
        else:
            actual = diatonic_triad(int(degree), scale)
            # A written dominant on the 5th degree of a true minor key is the
            # classic harmonic-minor V7 - keep it.  Elsewhere only add the b7
            # when the mode actually contains it (so Lydian never gets a G7
            # with a natural 4th in it, and Dorian keeps its minor v).
            is_fifth = int(degree) % 7 == 4
            if family == "dom" and minorish and takes_v7 and is_fifth:
                actual = "dom"
            elif family == "dom" and actual == "maj":
                flat7 = (degree_to_semitone(int(degree), scale) + 10) % 12
                if flat7 in {s % 12 for s in scale_set}:
                    actual = "dom"
        v7 = (actual == "dom" and minorish and takes_v7 and not borrowed
              and int(degree) % 7 == 4)
        quality = colourise(actual, rng, richness, root_midi, scale_pcs,
                            allow_leading_tone=v7)
        if v7:
            base_notes = build_chord(root_midi, quality)
            # raise the third to the leading tone if the mode gave us a minor 3rd
            base_notes = [n + 1 if (n - root_midi) % 12 == 3 else n for n in base_notes]
        else:
            base_notes = build_chord(root_midi, quality)
        notes = voice_chord(base_notes)
        bass = root_midi - 24
        while bass < 28:
            bass += 12
        chords.append({
            "root_midi": root_midi,
            "quality": quality,
            "notes": notes,
            "bass_midi": bass,
            "label": f"{pretty_note(root_midi % 12, root_pc)}{_short(quality)}",
            "borrowed": bool(borrowed),
        })
    return label, chords


FLAT_KEYS = {1, 3, 5, 8, 10}          # keys usually spelled with flats
SHARP_TO_FLAT = {1: "Db", 3: "Eb", 6: "Gb", 8: "Ab", 10: "Bb"}


def pretty_note(pc: int, key_pc: int = 0) -> str:
    """Spell a pitch class with flats in flat keys, sharps otherwise."""
    pc %= 12
    if key_pc % 12 in FLAT_KEYS and pc in SHARP_TO_FLAT:
        return SHARP_TO_FLAT[pc]
    return NOTE_NAMES[pc]


def _short(quality: str) -> str:
    return {"maj": "", "min": "m", "maj7": "maj7", "min7": "m7", "dom7": "7",
            "maj9": "maj9", "min9": "m9", "dom9": "9", "add9": "add9",
            "madd9": "m(add9)", "maj6": "6", "min6": "m6", "maj69": "6/9",
            "m7b5": "m7b5", "dim7": "dim7", "min11": "m11", "dom13": "13",
            "sus2": "sus2", "sus4": "sus4", "minmaj7": "mMaj7",
            "maj7s11": "maj7#11", "min7add11": "m7add11"}.get(quality, quality)


def melody_pool(chord, root_pc, scale, low=64, high=86):
    """Chord tones first, scale tones as passing notes - keeps melodies singable."""
    scale_pcs = set(scale_degrees(root_pc, scale))
    chord_pcs = {n % 12 for n in chord["notes"]}
    strong, weak = [], []
    for midi in range(low, high + 1):
        pc = midi % 12
        if pc in chord_pcs:
            strong.append(midi)
        elif pc in scale_pcs:
            weak.append(midi)
    return strong, weak


def quantise_to_scale(midi: int, root_pc: int, scale: str) -> int:
    pcs = scale_degrees(root_pc, scale)
    best, best_d = midi, 99
    for delta in range(-6, 7):
        if (midi + delta) % 12 in pcs and abs(delta) < best_d:
            best, best_d = midi + delta, abs(delta)
    return best
