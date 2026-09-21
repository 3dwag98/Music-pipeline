"""ID3 / metadata writing via mutagen, including embedded cover art.

An MP3-first pipeline needs real tags: uploaders, players and your own library
all read them, and the AI-disclosure note belongs somewhere durable rather than
only in a YouTube description box.
"""

from __future__ import annotations

from pathlib import Path

from .util import log, warn

#: What gets written into every export unless overridden.
DISCLOSURE = "Contains AI-assisted / synthesised music. Generated locally."


def have_mutagen():
    try:
        import mutagen  # noqa: F401
        return True
    except Exception:
        return False


def write_tags(path, title=None, artist=None, album=None, genre="Lofi Hip Hop",
               year=None, comment=None, track=None, cover=None, bpm=None,
               key=None, quiet=False):
    """Tag an MP3/FLAC/M4A in place.  Silently does nothing without mutagen."""
    path = Path(path)
    if not have_mutagen():
        if not quiet:
            warn("mutagen not installed, skipping tags: pip install mutagen")
        return False
    suffix = path.suffix.lower()
    try:
        if suffix == ".mp3":
            return _tag_mp3(path, title, artist, album, genre, year,
                            comment or DISCLOSURE, track, cover, bpm, key)
        return _tag_generic(path, title, artist, album, genre, year,
                            comment or DISCLOSURE, track, bpm, key)
    except Exception as exc:
        warn(f"could not tag {path.name}: {exc}")
        return False


def _tag_mp3(path, title, artist, album, genre, year, comment, track, cover, bpm, key):
    from mutagen.id3 import (APIC, COMM, ID3, TALB, TBPM, TCON, TDRC, TIT2,
                             TKEY, TPE1, TRCK, ID3NoHeaderError)
    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()
    if title:
        tags.setall("TIT2", [TIT2(encoding=3, text=[str(title)])])
    if artist:
        tags.setall("TPE1", [TPE1(encoding=3, text=[str(artist)])])
    if album:
        tags.setall("TALB", [TALB(encoding=3, text=[str(album)])])
    if genre:
        tags.setall("TCON", [TCON(encoding=3, text=[str(genre)])])
    if year:
        tags.setall("TDRC", [TDRC(encoding=3, text=[str(year)])])
    if track:
        tags.setall("TRCK", [TRCK(encoding=3, text=[str(track)])])
    if bpm:
        tags.setall("TBPM", [TBPM(encoding=3, text=[str(int(round(float(bpm))))])])
    if key:
        tags.setall("TKEY", [TKEY(encoding=3, text=[str(key)])])
    if comment:
        tags.setall("COMM", [COMM(encoding=3, lang="eng", desc="", text=[str(comment)])])
    if cover:
        art = Path(cover)
        if art.is_file():
            mime = "image/png" if art.suffix.lower() == ".png" else "image/jpeg"
            tags.setall("APIC", [APIC(encoding=3, mime=mime, type=3,
                                      desc="Cover", data=art.read_bytes())])
    tags.save(str(path), v2_version=3)
    return True


def _tag_generic(path, title, artist, album, genre, year, comment, track, bpm, key):
    import mutagen
    audio = mutagen.File(str(path))
    if audio is None:
        return False
    if audio.tags is None:
        try:
            audio.add_tags()
        except Exception:
            return False
    fields = {"title": title, "artist": artist, "album": album, "genre": genre,
              "date": str(year) if year else None,
              "tracknumber": str(track) if track else None,
              "bpm": str(int(round(float(bpm)))) if bpm else None,
              "comment": comment, "initialkey": key}
    for name, value in fields.items():
        if value:
            try:
                audio[name] = [str(value)]
            except Exception:
                continue
    audio.save()
    return True


def read_tags(path):
    """Best-effort read of title/artist/album, for naming derived files."""
    if not have_mutagen():
        return {}
    try:
        import mutagen
        audio = mutagen.File(str(path), easy=True)
        if not audio:
            return {}
        return {k: (v[0] if isinstance(v, list) and v else v)
                for k, v in dict(audio).items()}
    except Exception:
        return {}
