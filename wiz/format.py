"""
wiz/format.py — The .WIZ file format.

A .WIZ file is a SQLite database with two core tables:

    atoms       — temporal segments (the "objects")
    atom_tags   — key-value tags attached to atoms (the "OO layer")

Every piece of intelligence extracted by the pipeline (blinks, transcript
words, emotions, captions, …) becomes a tag on an atom. This makes the
format infinitely extensible without schema migrations and naturally
supports the graph-based search layer.

Tag taxonomy
────────────
  speaker   : "SPEAKER_01"          — who is speaking
  transcript: "<full text>"          — raw speech text (searchable)
  topic     : "machine learning"    — extracted keyword/phrase (indexed)
  emotion   : "confident"           — emotional tone label
  blink     : "true"                — blink event present in this window
  breath    : "true"                — breath event present in this window
  safe_cut  : "true"                — clean cut point (no blink, no breath)
  safe_cut  : "pause"               — natural speech gap
  caption   : "<text>"              — Video MAE frame description
  summary   : "<text>"              — LLM scene narrative
  language  : "en"                  — detected language (video-level atom)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Core data types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WizTag:
    """A single tag attached to a WizAtom."""
    tag_type: str       # e.g. "speaker", "emotion", "topic"
    tag_value: str      # e.g. "SPEAKER_01", "confident", "machine learning"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<Tag {self.tag_type}={self.tag_value!r} conf={self.confidence:.2f}>"


@dataclass
class WizAtom:
    """
    A temporal segment of a video — the fundamental unit of the .WIZ format.

    An atom carries an arbitrary set of tags that describe everything the
    pipeline knows about that time window.
    """
    atom_id: str
    frame_start: int
    frame_end: int
    time_start: float
    time_end: float
    tags: List[WizTag] = field(default_factory=list)

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def duration(self) -> float:
        return self.time_end - self.time_start

    def add_tag(
        self,
        tag_type: str,
        tag_value: str,
        confidence: float = 1.0,
        **metadata: Any,
    ) -> "WizAtom":
        """Attach a tag and return self (chainable)."""
        self.tags.append(WizTag(tag_type, str(tag_value), confidence, dict(metadata)))
        return self

    def get_tags(self, tag_type: str) -> List[WizTag]:
        return [t for t in self.tags if t.tag_type == tag_type]

    def has_tag(self, tag_type: str, tag_value: Optional[str] = None) -> bool:
        return any(
            t.tag_type == tag_type and (tag_value is None or t.tag_value == tag_value)
            for t in self.tags
        )

    def first_tag_value(self, tag_type: str, default: str = "") -> str:
        for t in self.tags:
            if t.tag_type == tag_type:
                return t.tag_value
        return default

    def __repr__(self) -> str:
        tag_preview = ", ".join(
            f"{t.tag_type}={t.tag_value!r}" for t in self.tags[:3]
        )
        return (
            f"<WizAtom [{self.time_start:.1f}s–{self.time_end:.1f}s] {tag_preview}>"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SQLite schema
# ──────────────────────────────────────────────────────────────────────────────

_SCHEMA_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS wiz_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Temporal units: every analysed segment maps to exactly one atom.
CREATE TABLE IF NOT EXISTS atoms (
    atom_id     TEXT PRIMARY KEY,
    frame_start INTEGER NOT NULL,
    frame_end   INTEGER NOT NULL,
    time_start  REAL    NOT NULL,
    time_end    REAL    NOT NULL
);

-- OO tag store: unlimited key-value tags per atom.
CREATE TABLE IF NOT EXISTS atom_tags (
    tag_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    atom_id    TEXT    NOT NULL REFERENCES atoms(atom_id) ON DELETE CASCADE,
    tag_type   TEXT    NOT NULL,
    tag_value  TEXT    NOT NULL,
    confidence REAL    NOT NULL DEFAULT 1.0,
    metadata   TEXT    NOT NULL DEFAULT '{}'
);

-- Covering index: the entire graph search lives here.
CREATE INDEX IF NOT EXISTS idx_tags_type_value ON atom_tags(tag_type, tag_value);
CREATE INDEX IF NOT EXISTS idx_tags_atom_id    ON atom_tags(atom_id);
CREATE INDEX IF NOT EXISTS idx_atoms_time      ON atoms(time_start, time_end);
"""


# ──────────────────────────────────────────────────────────────────────────────
# WizFile: read / write interface
# ──────────────────────────────────────────────────────────────────────────────

class WizFile:
    """
    Read/write interface for .wiz files.

    Usage (write)::

        wf = WizFile("results/interview.wiz")
        wf.write(atoms, meta={"fps": "29.97", "language": "en"})

    Usage (read)::

        atoms = WizFile("results/interview.wiz").read()
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ── write ─────────────────────────────────────────────────────────────────

    def write(
        self,
        atoms: List[WizAtom],
        meta: Optional[Dict[str, str]] = None,
    ) -> None:
        """Persist atoms (and optional metadata) to the .wiz file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.path))
        try:
            conn.executescript(_SCHEMA_DDL)

            if meta:
                conn.executemany(
                    "INSERT OR REPLACE INTO wiz_meta(key, value) VALUES (?, ?)",
                    list(meta.items()),
                )

            conn.executemany(
                "INSERT OR REPLACE INTO atoms"
                " (atom_id, frame_start, frame_end, time_start, time_end)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (a.atom_id, a.frame_start, a.frame_end, a.time_start, a.time_end)
                    for a in atoms
                ],
            )

            tag_rows: list = []
            for atom in atoms:
                for t in atom.tags:
                    tag_rows.append(
                        (atom.atom_id, t.tag_type, t.tag_value,
                         t.confidence, json.dumps(t.metadata, ensure_ascii=False))
                    )
            conn.executemany(
                "INSERT INTO atom_tags"
                " (atom_id, tag_type, tag_value, confidence, metadata)"
                " VALUES (?, ?, ?, ?, ?)",
                tag_rows,
            )

            conn.commit()
        finally:
            conn.close()

    # ── read ──────────────────────────────────────────────────────────────────

    def read(self) -> List[WizAtom]:
        """Load all atoms and their tags from the .wiz file."""
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            atoms_by_id: Dict[str, WizAtom] = {}

            for row in conn.execute(
                "SELECT atom_id, frame_start, frame_end, time_start, time_end"
                " FROM atoms ORDER BY time_start"
            ):
                atoms_by_id[row["atom_id"]] = WizAtom(
                    atom_id=row["atom_id"],
                    frame_start=row["frame_start"],
                    frame_end=row["frame_end"],
                    time_start=row["time_start"],
                    time_end=row["time_end"],
                )

            for row in conn.execute(
                "SELECT atom_id, tag_type, tag_value, confidence, metadata"
                " FROM atom_tags"
            ):
                atom = atoms_by_id.get(row["atom_id"])
                if atom is not None:
                    atom.tags.append(
                        WizTag(
                            tag_type=row["tag_type"],
                            tag_value=row["tag_value"],
                            confidence=row["confidence"],
                            metadata=json.loads(row["metadata"] or "{}"),
                        )
                    )

            return list(atoms_by_id.values())
        finally:
            conn.close()

    def read_meta(self) -> Dict[str, str]:
        conn = sqlite3.connect(str(self.path))
        try:
            return dict(conn.execute("SELECT key, value FROM wiz_meta").fetchall())
        except sqlite3.OperationalError:
            return {}
        finally:
            conn.close()

    # ── streaming read (memory-efficient for huge files) ──────────────────────

    def iter_atoms(self) -> Iterator[WizAtom]:
        """Yield atoms one at a time without loading all tags into memory at once."""
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            for a_row in conn.execute(
                "SELECT atom_id, frame_start, frame_end, time_start, time_end"
                " FROM atoms ORDER BY time_start"
            ):
                atom = WizAtom(
                    atom_id=a_row["atom_id"],
                    frame_start=a_row["frame_start"],
                    frame_end=a_row["frame_end"],
                    time_start=a_row["time_start"],
                    time_end=a_row["time_end"],
                )
                for t_row in conn.execute(
                    "SELECT tag_type, tag_value, confidence, metadata"
                    " FROM atom_tags WHERE atom_id = ?",
                    (atom.atom_id,),
                ):
                    atom.tags.append(
                        WizTag(
                            tag_type=t_row["tag_type"],
                            tag_value=t_row["tag_value"],
                            confidence=t_row["confidence"],
                            metadata=json.loads(t_row["metadata"] or "{}"),
                        )
                    )
                yield atom
        finally:
            conn.close()

    # ── factory helpers ───────────────────────────────────────────────────────

    @staticmethod
    def make_atom(
        frame_start: int,
        frame_end: int,
        fps: float,
        atom_id: Optional[str] = None,
    ) -> WizAtom:
        """Create a WizAtom from frame numbers, converting to wall-clock time."""
        aid = atom_id or str(uuid.uuid4())
        t_start = frame_start / fps if fps else 0.0
        t_end = frame_end / fps if fps else 0.0
        return WizAtom(
            atom_id=aid,
            frame_start=frame_start,
            frame_end=frame_end,
            time_start=t_start,
            time_end=t_end,
        )

    @staticmethod
    def make_time_atom(
        time_start: float,
        time_end: float,
        fps: float,
        atom_id: Optional[str] = None,
    ) -> WizAtom:
        """Create a WizAtom from wall-clock times, computing frame numbers."""
        aid = atom_id or str(uuid.uuid4())
        return WizAtom(
            atom_id=aid,
            frame_start=int(time_start * fps),
            frame_end=int(time_end * fps),
            time_start=time_start,
            time_end=time_end,
        )