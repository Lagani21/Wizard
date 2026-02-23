"""
SQLite schema for WIZ Intelligence Pipeline output.

Each processed video produces one .db file containing all structured
intelligence extracted by the pipeline, organised around the `atom`
concept: a frame-range segment whose size varies by task type.

Atom window defaults (configurable):
  - Blink / Breath   :   1.5 s  (fine-grained physical events)
  - Speaker / Transcript : 250 s  (utterance-level blocks)
  - Sentiment        :  950 s  (macro emotional arc)
  - Context Summary  :   30 s  (scene-level narrative)
"""

import sqlite3
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────────────────────────────────────

_DDL = """
-- Language / metadata associated with a video
CREATE TABLE IF NOT EXISTS meta_data (
    meta_data_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    language      TEXT
);

-- Universal frame-range index. Every result row in every table
-- references exactly one atom, whose window size was chosen by
-- the task that produced it.
CREATE TABLE IF NOT EXISTS atom (
    atom_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_start  INTEGER NOT NULL,
    frame_end    INTEGER NOT NULL
);

-- One row per processed video.
-- atom_id references the whole-video atom (frame 0 → total_frames).
CREATE TABLE IF NOT EXISTS video (
    video_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    atom_id      INTEGER,
    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP,
    meta_data_id INTEGER,
    file_path    TEXT,
    duration_s   REAL,
    fps          REAL,
    width        INTEGER,
    height       INTEGER,
    FOREIGN KEY (atom_id)      REFERENCES atom(atom_id),
    FOREIGN KEY (meta_data_id) REFERENCES meta_data(meta_data_id)
);

-- Blink and breath flags aggregated over 1-2 s atoms.
-- confidence stored as integer 0-100.
CREATE TABLE IF NOT EXISTS no_cut (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    INTEGER NOT NULL,
    atom_id     INTEGER NOT NULL,
    is_breath   INTEGER NOT NULL DEFAULT 0,   -- boolean (0/1)
    is_blink    INTEGER NOT NULL DEFAULT 0,   -- boolean (0/1)
    confidence  INTEGER,                       -- 0-100
    FOREIGN KEY (video_id) REFERENCES video(video_id),
    FOREIGN KEY (atom_id)  REFERENCES atom(atom_id)
);

-- Dominant emotional tone over 900-1000 s atoms.
CREATE TABLE IF NOT EXISTS sentiment (
    sentiment_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        INTEGER NOT NULL,
    atom_id         INTEGER NOT NULL,
    emotional_tone  TEXT,
    confidence      REAL,
    FOREIGN KEY (video_id) REFERENCES video(video_id),
    FOREIGN KEY (atom_id)  REFERENCES atom(atom_id)
);

-- Speaker transcription over 200-300 s atoms.
-- transcription stored as JSON array of speaker turns.
CREATE TABLE IF NOT EXISTS speaker (
    speaker_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id       INTEGER NOT NULL,
    atom_id        INTEGER NOT NULL,
    speaker_label  TEXT,
    transcription  TEXT,   -- JSON: [{"speaker": "...", "text": "...", "start": 0.0, "end": 0.0}]
    FOREIGN KEY (video_id) REFERENCES video(video_id),
    FOREIGN KEY (atom_id)  REFERENCES atom(atom_id)
);

-- LLM-generated context summary over 30 s atoms.
CREATE TABLE IF NOT EXISTS context_summary (
    context_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    INTEGER NOT NULL,
    atom_id     INTEGER NOT NULL,
    context     TEXT,
    tone_label  TEXT,
    FOREIGN KEY (video_id) REFERENCES video(video_id),
    FOREIGN KEY (atom_id)  REFERENCES atom(atom_id)
);

-- Video MAE frame-level captions over ~5 s atoms.
CREATE TABLE IF NOT EXISTS video_caption (
    caption_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    INTEGER NOT NULL,
    atom_id     INTEGER NOT NULL,
    caption     TEXT,
    confidence  REAL,
    FOREIGN KEY (video_id) REFERENCES video(video_id),
    FOREIGN KEY (atom_id)  REFERENCES atom(atom_id)
);
"""


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_database(db_path: str) -> sqlite3.Connection:
    """
    Open (or create) a SQLite database at db_path and apply the schema.

    Returns an open connection with foreign-key enforcement enabled.
    The caller is responsible for closing the connection.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # safe concurrent reads
    conn.executescript(_DDL)
    conn.commit()
    return conn


def get_db_path_for_video(video_path: str, output_dir: str = "results") -> str:
    """
    Derive a DB file path from the input video path.

    Example:
        video_path = "/data/interview.mov"
        → "results/interview.db"
    """
    stem = Path(video_path).stem
    return str(Path(output_dir) / f"{stem}.db")