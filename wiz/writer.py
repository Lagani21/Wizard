"""
wiz/writer.py — Convert a completed PipelineContext to a .WIZ file.

Each data type in the pipeline becomes one or more WizAtom objects, tagged
with the appropriate OO-style tags for downstream search.

Mapping summary
───────────────
  BlinkEvent          → atom with tag  blink:true  (or safe_cut:true)
  BreathEvent         → atom with tag  breath:true (merged into blink/breath windows)
  SpeakerAlignedSeg.  → atom with tags speaker:<id>, transcript:<text>,
                                        topic:<keyword> (extracted from text)
  ToneEvent           → atom with tag  emotion:<label>
  SceneSummary        → atom with tags summary:<text>, emotion:<label>
  VideoCaption        → atom with tag  caption:<text>
  VideoMetadata       → video-level    language:<lang> atom
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from ..core.context import (
        PipelineContext,
        BlinkEvent,
        BreathEvent,
        SpeakerAlignedSegment,
        ToneEvent,
        SceneSummary,
        VideoCaption,
    )
    from .format import WizAtom, WizFile
except ImportError:
    from core.context import (
        PipelineContext,
        BlinkEvent,
        BreathEvent,
        SpeakerAlignedSegment,
        ToneEvent,
        SceneSummary,
        VideoCaption,
    )
    from wiz.format import WizAtom, WizFile


# ──────────────────────────────────────────────────────────────────────────────
# Keyword / topic extraction
# ──────────────────────────────────────────────────────────────────────────────

_STOP_WORDS: frozenset = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "into",
    "through", "during", "before", "after", "above", "below", "up", "down",
    "out", "off", "over", "under", "again", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "not", "only", "same", "so",
    "than", "too", "very", "just", "as", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she",
    "her", "it", "its", "they", "them", "their", "what", "which", "who",
    "whom", "about", "like", "think", "know", "going", "really", "actually",
    "yeah", "um", "uh", "okay", "ok", "right", "well", "mean", "kind",
    "sort", "lot", "way", "got", "get", "go", "make", "see", "come", "even",
    "still", "back", "little", "also", "just", "then", "now", "here", "s",
    "t", "re", "ve", "ll", "d", "m",
})

_MIN_KEYWORD_LEN = 3


def extract_keywords(text: str) -> Set[str]:
    """
    Extract topic keywords (unigrams + bigrams) from a transcript string.

    Returns a set of lowercase keyword strings with stop words removed.
    Bigrams are stored as "word1 word2" (space-separated).
    """
    tokens = re.findall(r"\b[a-z']{%d,}\b" % _MIN_KEYWORD_LEN, text.lower())
    # strip possessives and contractions tails
    tokens = [re.sub(r"'.*$", "", t) for t in tokens]
    clean = [t for t in tokens if len(t) >= _MIN_KEYWORD_LEN and t not in _STOP_WORDS]

    keywords: Set[str] = set(clean)
    # bigrams
    for i in range(len(clean) - 1):
        keywords.add(f"{clean[i]} {clean[i + 1]}")

    return keywords


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and a_end > b_start


def _time_windows(total_seconds: float, window_s: float) -> List[Tuple[float, float]]:
    windows: List[Tuple[float, float]] = []
    t = 0.0
    while t < total_seconds:
        end = min(t + window_s, total_seconds)
        if end > t:
            windows.append((t, end))
        t = end
    return windows


def _frame(time_s: float, fps: float) -> int:
    return int(time_s * fps)


def _make_atom(time_start: float, time_end: float, fps: float) -> WizAtom:
    return WizAtom(
        atom_id=str(uuid.uuid4()),
        frame_start=_frame(time_start, fps),
        frame_end=_frame(time_end, fps),
        time_start=time_start,
        time_end=time_end,
    )


# ──────────────────────────────────────────────────────────────────────────────
# WizWriter
# ──────────────────────────────────────────────────────────────────────────────

class WizWriter:
    """
    Converts a completed PipelineContext to a list of tagged WizAtoms and
    persists them to a .wiz file.

    Usage::

        writer = WizWriter()
        wiz_path = writer.write(context, "results/interview.wiz")
    """

    # Window size for blink/breath aggregation (seconds)
    BLINK_BREATH_WINDOW: float = 1.5
    # Minimum speech gap to mark as a safe-cut pause (seconds)
    MIN_PAUSE_FOR_CUT: float = 0.4

    def write(self, context: PipelineContext, wiz_path: str) -> str:
        """
        Produce a .wiz file from a completed PipelineContext.

        Returns the absolute path of the written file.
        """
        if context.video_metadata is None:
            raise ValueError("PipelineContext has no video_metadata.")

        atoms = self._build_atoms(context)

        fps = context.video_metadata.fps
        lang = context.processing_metadata.get("transcription", {}).get("language", "")

        wf = WizFile(wiz_path)
        wf.write(
            atoms,
            meta={
                "fps": str(fps),
                "duration_s": str(context.video_metadata.duration_seconds),
                "total_frames": str(context.video_metadata.total_frames),
                "file_path": context.video_metadata.path,
                "language": lang,
                "width": str(context.video_metadata.width),
                "height": str(context.video_metadata.height),
            },
        )
        return str(Path(wiz_path).resolve())

    # ── atom builders ─────────────────────────────────────────────────────────

    def _build_atoms(self, context: PipelineContext) -> List[WizAtom]:
        vm = context.video_metadata
        fps = vm.fps
        duration = vm.duration_seconds
        atoms: List[WizAtom] = []

        atoms.extend(self._blink_breath_atoms(context, fps, duration))
        atoms.extend(self._speaker_atoms(context, fps))
        atoms.extend(self._emotion_atoms(context, fps))
        atoms.extend(self._summary_atoms(context, fps))
        atoms.extend(self._caption_atoms(context, fps))
        atoms.extend(self._pause_atoms(context, fps))

        return atoms

    # ── blink / breath → safe_cut atoms ──────────────────────────────────────

    def _blink_breath_atoms(
        self,
        context: PipelineContext,
        fps: float,
        duration: float,
    ) -> List[WizAtom]:
        """
        Divide the video into 1.5 s windows.
        Each window is tagged with whatever physical events fall inside it.
        Windows with neither blink nor breath are tagged safe_cut:true.
        """
        atoms: List[WizAtom] = []

        for t_start, t_end in _time_windows(duration, self.BLINK_BREATH_WINDOW):
            atom = _make_atom(t_start, t_end, fps)
            f_start = _frame(t_start, fps)
            f_end = _frame(t_end, fps)

            has_blink = any(
                e.start_frame < f_end and e.end_frame > f_start
                for e in context.blink_events
            )
            has_breath = any(
                _overlaps(e.start_time, e.end_time, t_start, t_end)
                for e in context.breath_events
            )

            if has_blink:
                blink_conf = max(
                    (e.confidence for e in context.blink_events
                     if e.start_frame < f_end and e.end_frame > f_start),
                    default=1.0,
                )
                atom.add_tag("blink", "true", confidence=blink_conf)

            if has_breath:
                breath_conf = max(
                    (e.confidence for e in context.breath_events
                     if _overlaps(e.start_time, e.end_time, t_start, t_end)),
                    default=1.0,
                )
                atom.add_tag("breath", "true", confidence=breath_conf)

            if not has_blink and not has_breath:
                atom.add_tag("safe_cut", "true")

            atoms.append(atom)

        return atoms

    # ── speaker turns → speaker + topic atoms ─────────────────────────────────

    def _speaker_atoms(
        self,
        context: PipelineContext,
        fps: float,
    ) -> List[WizAtom]:
        """
        One atom per aligned speaker segment.
        Tags: speaker:<id>, transcript:<text>, topic:<keyword> × N
        """
        atoms: List[WizAtom] = []

        for seg in context.aligned_segments:
            if not seg.text.strip():
                continue

            atom = _make_atom(seg.start_time, seg.end_time, fps)
            atom.add_tag("speaker", seg.speaker_id)
            atom.add_tag("transcript", seg.text.strip())

            for kw in extract_keywords(seg.text):
                atom.add_tag("topic", kw)

            atoms.append(atom)

        return atoms

    # ── tone events → emotion atoms ───────────────────────────────────────────

    def _emotion_atoms(
        self,
        context: PipelineContext,
        fps: float,
    ) -> List[WizAtom]:
        atoms: List[WizAtom] = []
        for ev in context.tone_events:
            atom = _make_atom(ev.start_time, ev.end_time, fps)
            atom.add_tag("emotion", ev.tone_label, confidence=ev.confidence)
            atoms.append(atom)
        return atoms

    # ── scene summaries → summary + emotion atoms ─────────────────────────────

    def _summary_atoms(
        self,
        context: PipelineContext,
        fps: float,
    ) -> List[WizAtom]:
        atoms: List[WizAtom] = []
        for s in context.scene_summaries:
            atom = _make_atom(s.start_time, s.end_time, fps)
            atom.add_tag("summary", s.summary_text, confidence=s.confidence)
            if s.tone_label:
                atom.add_tag("emotion", s.tone_label)
            for spk in s.key_speakers:
                atom.add_tag("speaker", spk)
            atoms.append(atom)
        return atoms

    # ── video captions ────────────────────────────────────────────────────────

    def _caption_atoms(
        self,
        context: PipelineContext,
        fps: float,
    ) -> List[WizAtom]:
        atoms: List[WizAtom] = []
        for cap in context.video_captions:
            atom = _make_atom(cap.start_time, cap.end_time, fps)
            atom.add_tag("caption", cap.caption, confidence=cap.confidence)
            atoms.append(atom)
        return atoms

    # ── natural speech pauses → safe_cut:pause atoms ─────────────────────────

    def _pause_atoms(
        self,
        context: PipelineContext,
        fps: float,
    ) -> List[WizAtom]:
        """
        Identify silence gaps between consecutive speaker turns.
        Gaps >= MIN_PAUSE_FOR_CUT get a safe_cut:pause atom.
        """
        atoms: List[WizAtom] = []
        segs = sorted(context.aligned_segments, key=lambda s: s.start_time)

        for i in range(len(segs) - 1):
            gap_start = segs[i].end_time
            gap_end = segs[i + 1].start_time
            if gap_end - gap_start >= self.MIN_PAUSE_FOR_CUT:
                atom = _make_atom(gap_start, gap_end, fps)
                atom.add_tag("safe_cut", "pause",
                             duration_s=gap_end - gap_start)
                atoms.append(atom)

        return atoms


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def context_to_wiz(context: PipelineContext, wiz_path: str) -> str:
    """Shorthand: write a PipelineContext to a .wiz file."""
    return WizWriter().write(context, wiz_path)