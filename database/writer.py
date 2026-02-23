"""
PipelineContextWriter — maps a completed PipelineContext to the WIZ SQLite schema.

Each task type uses a different atom window size:

    Task              | Window  | Table
    ------------------|---------|----------------
    Blink / Breath    | 1.5 s   | no_cut
    Speaker           | 250 s   | speaker
    Sentiment         | 950 s   | sentiment
    Context Summary   |  30 s   | context_summary
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from ..core.context import (
        PipelineContext, BlinkEvent, BreathEvent,
        SpeakerAlignedSegment, ToneEvent, SceneSummary, VideoCaption,
    )
    from .schema import create_database, get_db_path_for_video
except ImportError:
    from core.context import (
        PipelineContext, BlinkEvent, BreathEvent,
        SpeakerAlignedSegment, ToneEvent, SceneSummary, VideoCaption,
    )
    from database.schema import create_database, get_db_path_for_video


logger = logging.getLogger("wiz.database.writer")


# ──────────────────────────────────────────────────────────────────────────────
# Atom window defaults (seconds)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_BLINK_BREATH_WINDOW_S: float = 1.5
DEFAULT_SPEAKER_WINDOW_S: float = 250.0
DEFAULT_SENTIMENT_WINDOW_S: float = 950.0
DEFAULT_CONTEXT_WINDOW_S: float = 30.0
DEFAULT_VIDEO_CAPTION_WINDOW_S: float = 5.0


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _time_windows(
    total_seconds: float,
    window_s: float,
) -> List[Tuple[float, float]]:
    """Generate non-overlapping (start, end) time windows across total_seconds."""
    windows = []
    start = 0.0
    while start < total_seconds:
        end = min(start + window_s, total_seconds)
        if end > start:
            windows.append((start, end))
        start = end
    return windows


def _to_frames(time_s: float, fps: float) -> int:
    return int(time_s * fps)


def _insert_atom(conn: sqlite3.Connection, frame_start: int, frame_end: int) -> int:
    """Insert an atom and return its rowid."""
    cur = conn.execute(
        "INSERT INTO atom (frame_start, frame_end) VALUES (?, ?)",
        (frame_start, frame_end),
    )
    return cur.lastrowid


def _dominant_tone(tone_events: List[ToneEvent], start_s: float, end_s: float) -> Tuple[Optional[str], float]:
    """Return the most frequent tone label and its average confidence within a window."""
    in_window = [
        e for e in tone_events
        if e.start_time < end_s and e.end_time > start_s
    ]
    if not in_window:
        return None, 0.0

    counts: dict = {}
    conf_sum: dict = {}
    for ev in in_window:
        counts[ev.tone_label] = counts.get(ev.tone_label, 0) + 1
        conf_sum[ev.tone_label] = conf_sum.get(ev.tone_label, 0.0) + ev.confidence

    dominant = max(counts, key=counts.__getitem__)
    avg_conf = conf_sum[dominant] / counts[dominant]
    return dominant, avg_conf


def _blink_breath_in_window(
    blink_events: List[BlinkEvent],
    breath_events: List[BreathEvent],
    start_s: float,
    end_s: float,
    fps: float,
) -> Tuple[bool, bool, int]:
    """
    Check whether any blink or breath event falls inside [start_s, end_s].

    BlinkEvent times are frame-based; BreathEvent times are in seconds.
    Returns (is_blink, is_breath, confidence_0_to_100).
    """
    start_frame = _to_frames(start_s, fps)
    end_frame = _to_frames(end_s, fps)

    blink_confs = [
        e.confidence for e in blink_events
        if e.start_frame < end_frame and e.end_frame > start_frame
    ]
    breath_confs = [
        e.confidence for e in breath_events
        if e.start_time < end_s and e.end_time > start_s
    ]

    is_blink = len(blink_confs) > 0
    is_breath = len(breath_confs) > 0

    all_confs = blink_confs + breath_confs
    confidence = int((sum(all_confs) / len(all_confs)) * 100) if all_confs else 0
    return is_blink, is_breath, confidence


def _speaker_json(
    aligned_segments: List[SpeakerAlignedSegment],
    start_s: float,
    end_s: float,
) -> Tuple[List[str], str]:
    """
    Collect speaker turns within [start_s, end_s] and serialise to JSON.

    Returns (unique_speaker_labels, json_string).
    """
    in_window = [
        seg for seg in aligned_segments
        if seg.start_time < end_s and seg.end_time > start_s
    ]

    turns = []
    speakers: set = set()
    for seg in in_window:
        speakers.add(seg.speaker_id)
        word_list = [
            {"text": w.text, "start": w.start_time, "end": w.end_time, "conf": w.confidence}
            for w in seg.words
        ] if seg.words else []
        turns.append({
            "speaker": seg.speaker_id,
            "text": seg.text,
            "start": seg.start_time,
            "end": seg.end_time,
            "words": word_list,
        })

    return sorted(speakers), json.dumps(turns, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Public writer
# ──────────────────────────────────────────────────────────────────────────────

class PipelineContextWriter:
    """
    Writes a completed PipelineContext to a WIZ SQLite database.

    Usage::

        writer = PipelineContextWriter()
        db_path = writer.write(context, db_path="results/interview.db")
    """

    def __init__(
        self,
        blink_breath_window_s: float = DEFAULT_BLINK_BREATH_WINDOW_S,
        speaker_window_s: float = DEFAULT_SPEAKER_WINDOW_S,
        sentiment_window_s: float = DEFAULT_SENTIMENT_WINDOW_S,
        context_window_s: float = DEFAULT_CONTEXT_WINDOW_S,
        video_caption_window_s: float = DEFAULT_VIDEO_CAPTION_WINDOW_S,
    ) -> None:
        self.blink_breath_window_s = blink_breath_window_s
        self.speaker_window_s = speaker_window_s
        self.sentiment_window_s = sentiment_window_s
        self.context_window_s = context_window_s
        self.video_caption_window_s = video_caption_window_s

    # ── entry point ──────────────────────────────────────────────────────────

    def write(self, context: PipelineContext, db_path: str) -> str:
        """
        Persist all pipeline results to a SQLite database.

        Args:
            context:  Completed PipelineContext from Pipeline.run()
            db_path:  Path where the .db file should be created / appended.

        Returns:
            Absolute path to the written database file.
        """
        if context.video_metadata is None:
            raise ValueError("PipelineContext has no video_metadata — cannot write DB.")

        conn = create_database(db_path)
        try:
            with conn:   # transaction
                meta_id = self._write_meta_data(conn, context)
                video_id = self._write_video(conn, context, meta_id)
                self._write_no_cut(conn, context, video_id)
                self._write_speaker(conn, context, video_id)
                self._write_sentiment(conn, context, video_id)
                self._write_context_summary(conn, context, video_id)
                self._write_video_captions(conn, context, video_id)
        finally:
            conn.close()

        logger.info(f"Pipeline results written to {db_path}")
        return str(Path(db_path).resolve())

    # ── per-table writers ────────────────────────────────────────────────────

    def _write_meta_data(self, conn: sqlite3.Connection, context: PipelineContext) -> int:
        """Detect language from transcription and insert meta_data row."""
        language = None

        # Try to extract detected language from processing metadata
        transcription_meta = context.processing_metadata.get("transcription", {})
        language = transcription_meta.get("language")

        cur = conn.execute(
            "INSERT INTO meta_data (language) VALUES (?)",
            (language,),
        )
        return cur.lastrowid

    def _write_video(
        self,
        conn: sqlite3.Connection,
        context: PipelineContext,
        meta_id: int,
    ) -> int:
        """Insert a whole-video atom, then insert the video row."""
        vm = context.video_metadata

        # Whole-video atom (frame 0 → total_frames)
        root_atom_id = _insert_atom(conn, 0, vm.total_frames)

        cur = conn.execute(
            """INSERT INTO video
               (atom_id, meta_data_id, file_path, duration_s, fps, width, height)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (root_atom_id, meta_id, vm.path,
             vm.duration_seconds, vm.fps, vm.width, vm.height),
        )
        return cur.lastrowid

    def _write_no_cut(
        self,
        conn: sqlite3.Connection,
        context: PipelineContext,
        video_id: int,
    ) -> None:
        """
        Write blink/breath detection results at 1-2 s atom granularity.
        One no_cut row per atom window.
        """
        vm = context.video_metadata
        windows = _time_windows(vm.duration_seconds, self.blink_breath_window_s)

        for start_s, end_s in windows:
            is_blink, is_breath, confidence = _blink_breath_in_window(
                context.blink_events,
                context.breath_events,
                start_s, end_s,
                vm.fps,
            )

            atom_id = _insert_atom(
                conn,
                _to_frames(start_s, vm.fps),
                _to_frames(end_s, vm.fps),
            )
            conn.execute(
                """INSERT INTO no_cut (video_id, atom_id, is_breath, is_blink, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (video_id, atom_id, int(is_breath), int(is_blink), confidence),
            )

    def _write_speaker(
        self,
        conn: sqlite3.Connection,
        context: PipelineContext,
        video_id: int,
    ) -> None:
        """
        Write speaker transcription at 200-300 s atom granularity.
        One speaker row per unique speaker per atom window.
        """
        vm = context.video_metadata

        # Prefer aligned segments; fall back to raw transcript segments
        segments = context.aligned_segments if context.aligned_segments else []
        if not segments:
            logger.debug("No aligned segments — speaker table will be empty.")
            return

        windows = _time_windows(vm.duration_seconds, self.speaker_window_s)

        for start_s, end_s in windows:
            speaker_labels, transcription_json = _speaker_json(
                segments, start_s, end_s
            )
            if not speaker_labels:
                continue

            atom_id = _insert_atom(
                conn,
                _to_frames(start_s, vm.fps),
                _to_frames(end_s, vm.fps),
            )

            # One row per speaker found in this window
            for label in speaker_labels:
                # Filter transcription JSON to this speaker
                turns = json.loads(transcription_json)
                speaker_turns = [t for t in turns if t["speaker"] == label]
                conn.execute(
                    """INSERT INTO speaker (video_id, atom_id, speaker_label, transcription)
                       VALUES (?, ?, ?, ?)""",
                    (video_id, atom_id, label, json.dumps(speaker_turns, ensure_ascii=False)),
                )

    def _write_sentiment(
        self,
        conn: sqlite3.Connection,
        context: PipelineContext,
        video_id: int,
    ) -> None:
        """
        Write dominant emotional tone at 900-1000 s atom granularity.
        One sentiment row per atom window.
        """
        vm = context.video_metadata

        if not context.tone_events:
            logger.debug("No tone events — sentiment table will be empty.")
            return

        windows = _time_windows(vm.duration_seconds, self.sentiment_window_s)

        for start_s, end_s in windows:
            tone_label, confidence = _dominant_tone(
                context.tone_events, start_s, end_s
            )
            if tone_label is None:
                continue

            atom_id = _insert_atom(
                conn,
                _to_frames(start_s, vm.fps),
                _to_frames(end_s, vm.fps),
            )
            conn.execute(
                """INSERT INTO sentiment (video_id, atom_id, emotional_tone, confidence)
                   VALUES (?, ?, ?, ?)""",
                (video_id, atom_id, tone_label, confidence),
            )

    def _write_context_summary(
        self,
        conn: sqlite3.Connection,
        context: PipelineContext,
        video_id: int,
    ) -> None:
        """
        Write LLM-generated context summaries at 30 s atom granularity.
        Maps directly from SceneSummary objects.
        """
        vm = context.video_metadata

        if not context.scene_summaries:
            logger.debug("No scene summaries — context_summary table will be empty.")
            return

        for summary in context.scene_summaries:
            atom_id = _insert_atom(
                conn,
                _to_frames(summary.start_time, vm.fps),
                _to_frames(summary.end_time, vm.fps),
            )
            conn.execute(
                """INSERT INTO context_summary (video_id, atom_id, context, tone_label)
                   VALUES (?, ?, ?, ?)""",
                (video_id, atom_id, summary.summary_text, summary.tone_label),
            )

    def _write_video_captions(
        self,
        conn: sqlite3.Connection,
        context: PipelineContext,
        video_id: int,
    ) -> None:
        """
        Write Video MAE frame-level captions at ~5 s atom granularity.
        One video_caption row per VideoCaption object.
        """
        vm = context.video_metadata

        if not context.video_captions:
            logger.debug("No video captions — video_caption table will be empty.")
            return

        for cap in context.video_captions:
            atom_id = _insert_atom(
                conn,
                _to_frames(cap.start_time, vm.fps),
                _to_frames(cap.end_time, vm.fps),
            )
            conn.execute(
                """INSERT INTO video_caption (video_id, atom_id, caption, confidence)
                   VALUES (?, ?, ?, ?)""",
                (video_id, atom_id, cap.caption, cap.confidence),
            )