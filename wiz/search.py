"""
wiz/search.py — High-level search API over a .WIZ file.

SearchEngine wraps WizGraph and exposes four editor-facing query methods
that map directly onto the use cases from the spec:

  1. find_person_topic(speaker, topic)
     "Find every moment where [person] talks about [topic]"

  2. find_emotion(emotion)
     "Find all segments with [emotion]"

  3. find_safe_cuts()
     "Show me safe cut points in this interview clip"
     (no blinks, no breaths, natural pauses)

  4. find_person_topic_no_blink(speaker, topic)
     "Find every moment where [person] talks about [topic] and is not mid-blink"

All methods return List[SearchResult] sorted by time_start.

Quick-start::

    engine = SearchEngine("results/interview.wiz")
    for r in engine.find_person_topic("SPEAKER_01", "machine learning"):
        print(r)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

try:
    from .format import WizAtom, WizTag, WizFile
    from .graph import WizGraph, TagCondition, NotTagCondition, TemporalNotCondition
except ImportError:
    from wiz.format import WizAtom, WizTag, WizFile
    from wiz.graph import WizGraph, TagCondition, NotTagCondition, TemporalNotCondition


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single match from a search query."""
    atom: WizAtom
    score: float = 1.0
    matched_tags: List[WizTag] = field(default_factory=list)
    query_label: str = ""

    @property
    def time_start(self) -> float:
        return self.atom.time_start

    @property
    def time_end(self) -> float:
        return self.atom.time_end

    @property
    def duration(self) -> float:
        return self.atom.duration

    @property
    def speaker(self) -> str:
        return self.atom.first_tag_value("speaker")

    @property
    def transcript(self) -> str:
        return self.atom.first_tag_value("transcript")

    @property
    def emotion(self) -> str:
        return self.atom.first_tag_value("emotion")

    def timecode(self, fps: float = 25.0) -> str:
        """Format time_start as HH:MM:SS:FF timecode."""
        total_frames = int(self.time_start * fps)
        ff = total_frames % int(fps)
        ss = (total_frames // int(fps)) % 60
        mm = (total_frames // int(fps * 60)) % 60
        hh = total_frames // int(fps * 3600)
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

    def __repr__(self) -> str:
        tc = f"{self.time_start:.2f}s–{self.time_end:.2f}s"
        tags = ", ".join(f"{t.tag_type}={t.tag_value!r}" for t in self.matched_tags[:3])
        return f"<SearchResult [{tc}] score={self.score:.2f} [{tags}]>"


# ──────────────────────────────────────────────────────────────────────────────
# SearchEngine
# ──────────────────────────────────────────────────────────────────────────────

class SearchEngine:
    """
    Editor-facing search API over a .wiz file.

    The graph index is built once on construction (or explicitly via
    load_file()). Subsequent queries are in-memory only.
    """

    def __init__(self, wiz_path: Optional[str] = None) -> None:
        self.graph = WizGraph()
        self._wiz_path: Optional[str] = None
        self._load_time_ms: float = 0.0
        self._fps: float = 25.0

        if wiz_path:
            self.load_file(wiz_path)

    # ── index management ──────────────────────────────────────────────────────

    def load_file(self, wiz_path: str) -> None:
        """Load a .wiz file and build the search index."""
        t0 = time.perf_counter()
        wf = WizFile(wiz_path)
        atoms = wf.read()
        meta = wf.read_meta()
        self._fps = float(meta.get("fps", 25.0))
        self.graph.build(atoms)
        self._wiz_path = wiz_path
        self._load_time_ms = (time.perf_counter() - t0) * 1000

    def load_atoms(self, atoms: list, fps: float = 25.0) -> None:
        """Build index directly from a list of WizAtoms (useful for testing)."""
        self._fps = fps
        self.graph.build(atoms)

    # ── query 1: person + topic ───────────────────────────────────────────────

    def find_person_topic(
        self,
        speaker_id: str,
        topic: str,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Find every moment where [person] talks about [topic].

        Matches speaker atoms for speaker_id whose topic tags intersect
        with the given topic phrase. Multi-word topics use AND semantics
        across individual keyword matches.

        Optional time_start/time_end (seconds) narrow results to a time window.
        """
        speaker_ids = self.graph.find_by_tag("speaker", speaker_id)
        topic_ids = self.graph.find_topic(topic)
        result_ids = speaker_ids & topic_ids
        if time_start is not None or time_end is not None:
            result_ids &= self.graph.find_overlapping(
                time_start or 0.0, time_end or float("inf")
            )

        return self._to_results(
            result_ids,
            query_label=f"person={speaker_id!r} topic={topic!r}",
            highlight_tags={"speaker", "topic", "transcript"},
        )

    # ── query 2: emotion ──────────────────────────────────────────────────────

    def find_emotion(
        self,
        emotion: str,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Find all segments with [emotion].

        emotion should match the tone_label values from ToneClassifier,
        e.g. "confident", "sad", "angry", "excited", "neutral", "thoughtful".

        Optional time_start/time_end (seconds) narrow results to a time window.
        """
        result_ids = self.graph.find_by_tag("emotion", emotion.lower())
        if time_start is not None or time_end is not None:
            result_ids = result_ids & self.graph.find_overlapping(
                time_start or 0.0, time_end or float("inf")
            )
        atoms = self.graph._ids_to_atoms(result_ids)
        return self._atoms_to_results(
            atoms,
            query_label=f"emotion={emotion!r}",
            highlight_tags={"emotion"},
        )

    # ── query 3: safe cut points ──────────────────────────────────────────────

    def find_safe_cuts(
        self,
        include_pauses: bool = True,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Find safe cut points in the clip.

        Returns all 1.5 s windows where no blink and no breath was detected
        (tagged safe_cut:true by the writer), plus natural speech-gap windows
        (tagged safe_cut:pause) if include_pauses=True.

        Optional time_start/time_end (seconds) narrow results to a time window.
        """
        true_cuts = self.graph.find_by_tag("safe_cut", "true")

        if include_pauses:
            pause_cuts = self.graph.find_by_tag("safe_cut", "pause")
            all_ids = true_cuts | pause_cuts
        else:
            all_ids = true_cuts

        if time_start is not None or time_end is not None:
            all_ids = all_ids & self.graph.find_overlapping(
                time_start or 0.0, time_end or float("inf")
            )

        return self._to_results(
            all_ids,
            query_label="safe_cuts",
            highlight_tags={"safe_cut"},
        )

    # ── query 4: person + topic + not mid-blink ───────────────────────────────

    def find_person_topic_no_blink(
        self,
        speaker_id: str,
        topic: str,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Find every moment where [person] talks about [topic] and is not mid-blink.

        Uses temporal NOT: finds speaker+topic atoms, then removes any whose
        time range overlaps with a blink atom — even though blink atoms and
        speaker atoms have different granularities.

        Optional time_start/time_end (seconds) narrow results to a time window.
        """
        # First reduce to person+topic atoms
        speaker_ids = self.graph.find_by_tag("speaker", speaker_id)
        topic_ids = self.graph.find_topic(topic)
        candidate_ids = speaker_ids & topic_ids
        if time_start is not None or time_end is not None:
            candidate_ids &= self.graph.find_overlapping(
                time_start or 0.0, time_end or float("inf")
            )

        if not candidate_ids:
            return []

        # Blink atom time ranges to check against
        blink_atom_ids = self.graph.find_by_tag("blink", "true")
        blink_ranges = [
            (self.graph._atoms[aid].time_start, self.graph._atoms[aid].time_end)
            for aid in blink_atom_ids
            if aid in self.graph._atoms
        ]

        results: List[SearchResult] = []
        for aid in candidate_ids:
            atom = self.graph._atoms.get(aid)
            if atom is None:
                continue
            # Check temporal overlap with any blink window
            overlaps_blink = any(
                atom.time_start < b_end and atom.time_end > b_start
                for b_start, b_end in blink_ranges
            )
            if not overlaps_blink:
                results.append(self._make_result(
                    atom,
                    query_label=f"person={speaker_id!r} topic={topic!r} no_blink",
                    highlight_tags={"speaker", "topic", "transcript"},
                ))

        results.sort(key=lambda r: r.time_start)
        return results

    # ── generic tag query ─────────────────────────────────────────────────────

    def query(self, **tag_conditions: str) -> List[SearchResult]:
        """
        Generic AND query by tag type→value pairs.

        Example::

            engine.query(speaker="SPEAKER_01", emotion="confident")
        """
        conditions = [
            TagCondition(tag_type, tag_value)
            for tag_type, tag_value in tag_conditions.items()
        ]
        atoms = self.graph.query(*conditions)
        label = " AND ".join(f"{k}={v!r}" for k, v in tag_conditions.items())
        return self._atoms_to_results(atoms, query_label=label)

    # ── listing helpers ───────────────────────────────────────────────────────

    def speakers(self) -> List[str]:
        """List all speaker IDs found in this file."""
        return sorted(self.graph.all_tag_values("speaker"))

    def emotions(self) -> List[str]:
        """List all emotion labels found in this file."""
        return sorted(self.graph.all_tag_values("emotion"))

    def topics(self, min_atoms: int = 2) -> List[str]:
        """
        List frequently occurring topic keywords.

        min_atoms: skip topics that appear in fewer than this many atoms.
        """
        topic_idx = self.graph._tag_index.get("topic", {})
        return sorted(
            v for v, ids in topic_idx.items()
            if v is not None and len(ids) >= min_atoms
        )

    # ── index stats ───────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, object]:
        return {
            "wiz_path": self._wiz_path,
            "total_atoms": self.graph.atom_count,
            "tag_types": self.graph.stats(),
            "speakers": self.speakers(),
            "emotions": self.emotions(),
            "index_load_ms": round(self._load_time_ms, 2),
        }

    # ── internal helpers ──────────────────────────────────────────────────────

    def _to_results(
        self,
        atom_ids: Set[str],
        query_label: str = "",
        highlight_tags: Optional[set] = None,
    ) -> List[SearchResult]:
        atoms = self.graph._ids_to_atoms(atom_ids)
        return self._atoms_to_results(atoms, query_label, highlight_tags)

    def _atoms_to_results(
        self,
        atoms: List[WizAtom],
        query_label: str = "",
        highlight_tags: Optional[set] = None,
    ) -> List[SearchResult]:
        return [
            self._make_result(a, query_label, highlight_tags)
            for a in atoms
        ]

    def _make_result(
        self,
        atom: WizAtom,
        query_label: str = "",
        highlight_tags: Optional[set] = None,
    ) -> SearchResult:
        matched = (
            [t for t in atom.tags if t.tag_type in highlight_tags]
            if highlight_tags
            else atom.tags[:]
        )
        # Score = average confidence of matched tags (or 1.0)
        score = (
            sum(t.confidence for t in matched) / len(matched)
            if matched
            else 1.0
        )
        return SearchResult(
            atom=atom,
            score=score,
            matched_tags=matched,
            query_label=query_label,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def open_wiz(wiz_path: str) -> SearchEngine:
    """Open a .wiz file and return a ready-to-query SearchEngine."""
    return SearchEngine(wiz_path)