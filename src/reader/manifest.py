"""Walk the staging folder and group files into works.

Responsibilities:

* walk ``READER_INGEST`` recursively;
* skip Synology ``@eaDir`` sidecar directories and ``.DS_Store`` noise;
* hash each file (sha256) for provenance;
* group files into *works* by (directory, stem);
* resolve a format clash (a ``book.epub`` and a ``book.pdf`` in the same
  directory) by keeping the higher-priority format and warning — real
  libraries routinely hold the same title in several formats, and refusing
  the whole manifest over that would abort the scheduled pipeline.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIR_NAMES = {"@eaDir", "#recycle", ".DS_Store", "._.DS_Store"}
SKIP_FILE_NAMES = {".DS_Store", "Thumbs.db"}

#: Extension -> canonical format name.
FORMATS = {
    ".epub": "epub",
    ".pdf": "pdf",
}

#: Preference order for a format clash (lower number wins). Epub is preferred
#: over pdf because it yields cleaner text; artbooks are pdf-only, so they are
#: unaffected.
FORMAT_PRIORITY = {"epub": 0, "pdf": 1}

_CHUNK = 1 << 20  # 1 MiB


class ManifestError(Exception):
    """Raised for structural problems in the staging tree."""


@dataclass
class SourceFile:
    path: Path
    relpath: str
    fmt: str
    sha256: str
    size: int


@dataclass
class Work:
    key: str
    directory: str
    stem: str
    format: str
    files: list[SourceFile] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def detect_format(path: Path) -> str | None:
    return FORMATS.get(path.suffix.lower())


def _work_key(rel_dir: str, stem: str) -> str:
    return f"{rel_dir}/{stem}" if rel_dir else stem


def walk(ingest_dir: Path) -> list[SourceFile]:
    """Return every supported source file under ``ingest_dir``, sorted."""
    root = Path(ingest_dir)
    out: list[SourceFile] = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.name in SKIP_FILE_NAMES or path.name.startswith("._"):
            continue
        fmt = detect_format(path)
        if fmt is None:
            continue
        rel = path.relative_to(root)
        out.append(
            SourceFile(
                path=path,
                relpath=str(rel),
                fmt=fmt,
                sha256=sha256_file(path),
                size=path.stat().st_size,
            )
        )
    return out


def group_works(files: list[SourceFile], root: Path) -> list[Work]:
    """Group files into works, resolving a format clash by preference.

    A work is keyed by (containing directory, filename stem).  Two files that
    share a stem but differ in format are a clash: the higher-priority format
    (see :data:`FORMAT_PRIORITY`) wins and the others are skipped with a
    warning, so one stray duplicate cannot abort the whole manifest.
    """
    root = Path(root)
    works: dict[str, Work] = {}
    for sf in files:
        rel = Path(sf.relpath)
        stem, directory = rel.stem, str(rel.parent)
        if directory == ".":
            directory = ""
        key = _work_key(directory, stem)
        work = works.get(key)
        if work is None:
            works[key] = Work(key=key, directory=directory, stem=stem, format=sf.fmt, files=[sf])
            continue
        if sf.fmt == work.format:
            work.files.append(sf)
            continue
        keep_new = FORMAT_PRIORITY[sf.fmt] < FORMAT_PRIORITY[work.format]
        kept = sf.fmt if keep_new else work.format
        ignored = work.files[0].relpath if keep_new else sf.relpath
        warnings.warn(
            f"work {key!r} has mixed formats; keeping {kept!r}, ignoring {ignored!r}",
            stacklevel=2,
        )
        if keep_new:
            work.format = sf.fmt
            work.files = [sf]
    return [works[k] for k in sorted(works)]


def build(ingest_dir: Path) -> list[Work]:
    """Walk + group + guard, in one call."""
    return group_works(walk(ingest_dir), Path(ingest_dir))
