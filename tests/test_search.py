"""
tests/test_search.py — Unit tests for SearchEngine.

All tests run fully in-memory (no .wiz file I/O, no ML models).
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wiz.format import WizAtom
from wiz.search import SearchEngine


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_atom(atom_id: str, t_start: float, t_end: float, **tags) -> WizAtom:
    atom = WizAtom(
        atom_id=atom_id,
        frame_start=int(t_start * 25),
        frame_end=int(t_end * 25),
        time_start=t_start,
        time_end=t_end,
    )
    for tag_type, tag_value in tags.items():
        atom.add_tag(tag_type, str(tag_value))
    return atom


def engine_from(*atoms: WizAtom) -> SearchEngine:
    eng = SearchEngine()
    eng.load_atoms(list(atoms), fps=25.0)
    return eng


# ──────────────────────────────────────────────────────────────────────────────
# find_person_topic
# ──────────────────────────────────────────────────────────────────────────────

class TestFindPersonTopic:
    def test_basic_match(self):
        eng = engine_from(
            make_atom("a1", 0, 5, speaker="PERSON_001", topic="revenue"),
            make_atom("a2", 5, 10, speaker="PERSON_002", topic="revenue"),
        )
        results = eng.find_person_topic("PERSON_001", "revenue")
        assert len(results) == 1
        assert results[0].atom.atom_id == "a1"

    def test_no_match(self):
        eng = engine_from(make_atom("a1", 0, 5, speaker="PERSON_001", topic="budget"))
        assert eng.find_person_topic("PERSON_001", "revenue") == []

    def test_time_start_filter(self):
        eng = engine_from(
            make_atom("a1", 0, 5,   speaker="PERSON_001", topic="revenue"),
            make_atom("a2", 60, 65, speaker="PERSON_001", topic="revenue"),
        )
        results = eng.find_person_topic("PERSON_001", "revenue", time_start=50.0)
        ids = {r.atom.atom_id for r in results}
        assert "a2" in ids
        assert "a1" not in ids

    def test_time_end_filter(self):
        eng = engine_from(
            make_atom("a1", 0, 5,   speaker="PERSON_001", topic="revenue"),
            make_atom("a2", 60, 65, speaker="PERSON_001", topic="revenue"),
        )
        results = eng.find_person_topic("PERSON_001", "revenue", time_end=10.0)
        ids = {r.atom.atom_id for r in results}
        assert "a1" in ids
        assert "a2" not in ids

    def test_time_window_both(self):
        eng = engine_from(
            make_atom("a1", 0,  5,  speaker="PERSON_001", topic="revenue"),
            make_atom("a2", 30, 35, speaker="PERSON_001", topic="revenue"),
            make_atom("a3", 90, 95, speaker="PERSON_001", topic="revenue"),
        )
        results = eng.find_person_topic("PERSON_001", "revenue", time_start=20.0, time_end=60.0)
        ids = {r.atom.atom_id for r in results}
        assert ids == {"a2"}

    def test_fuzzy_plural(self):
        eng = engine_from(make_atom("a1", 0, 5, speaker="PERSON_001", topic="revenue"))
        results = eng.find_person_topic("PERSON_001", "revenues")
        assert len(results) == 1


# ──────────────────────────────────────────────────────────────────────────────
# find_emotion
# ──────────────────────────────────────────────────────────────────────────────

class TestFindEmotion:
    def test_basic(self):
        eng = engine_from(
            make_atom("a1", 0, 5,  emotion="confident"),
            make_atom("a2", 5, 10, emotion="neutral"),
        )
        results = eng.find_emotion("confident")
        assert len(results) == 1
        assert results[0].atom.atom_id == "a1"

    def test_case_insensitive(self):
        eng = engine_from(make_atom("a1", 0, 5, emotion="confident"))
        assert len(eng.find_emotion("Confident")) == 1

    def test_time_window(self):
        eng = engine_from(
            make_atom("a1", 0,  5,  emotion="confident"),
            make_atom("a2", 60, 65, emotion="confident"),
        )
        results = eng.find_emotion("confident", time_start=50.0)
        ids = {r.atom.atom_id for r in results}
        assert ids == {"a2"}


# ──────────────────────────────────────────────────────────────────────────────
# find_safe_cuts
# ──────────────────────────────────────────────────────────────────────────────

class TestFindSafeCuts:
    def test_returns_safe_cut_true(self):
        eng = engine_from(
            make_atom("a1", 0, 1.5, safe_cut="true"),
            make_atom("a2", 1.5, 3,  blink="true"),
        )
        results = eng.find_safe_cuts(include_pauses=False)
        ids = {r.atom.atom_id for r in results}
        assert "a1" in ids
        assert "a2" not in ids

    def test_includes_pauses(self):
        eng = engine_from(
            make_atom("a1", 0, 1.5, safe_cut="true"),
            make_atom("a2", 5, 5.8, safe_cut="pause"),
        )
        results = eng.find_safe_cuts(include_pauses=True)
        ids = {r.atom.atom_id for r in results}
        assert "a1" in ids
        assert "a2" in ids

    def test_excludes_pauses(self):
        eng = engine_from(
            make_atom("a1", 0, 1.5, safe_cut="true"),
            make_atom("a2", 5, 5.8, safe_cut="pause"),
        )
        results = eng.find_safe_cuts(include_pauses=False)
        ids = {r.atom.atom_id for r in results}
        assert "a2" not in ids

    def test_time_window(self):
        eng = engine_from(
            make_atom("a1", 0,  1.5, safe_cut="true"),
            make_atom("a2", 60, 61.5, safe_cut="true"),
        )
        results = eng.find_safe_cuts(time_start=50.0)
        ids = {r.atom.atom_id for r in results}
        assert ids == {"a2"}


# ──────────────────────────────────────────────────────────────────────────────
# find_person_topic_no_blink
# ──────────────────────────────────────────────────────────────────────────────

class TestFindPersonTopicNoBlink:
    def test_excludes_blink_overlap(self):
        # a1 overlaps with a blink atom; a2 does not
        eng = engine_from(
            make_atom("a1", 0,  5,  speaker="PERSON_001", topic="revenue"),
            make_atom("a2", 10, 15, speaker="PERSON_001", topic="revenue"),
            make_atom("blink1", 2, 4, blink="true"),  # overlaps a1
        )
        results = eng.find_person_topic_no_blink("PERSON_001", "revenue")
        ids = {r.atom.atom_id for r in results}
        assert "a2" in ids
        assert "a1" not in ids

    def test_time_window_filter(self):
        eng = engine_from(
            make_atom("a1", 0,  5,  speaker="PERSON_001", topic="revenue"),
            make_atom("a2", 60, 65, speaker="PERSON_001", topic="revenue"),
        )
        results = eng.find_person_topic_no_blink("PERSON_001", "revenue", time_start=50.0)
        ids = {r.atom.atom_id for r in results}
        assert "a2" in ids
        assert "a1" not in ids


# ──────────────────────────────────────────────────────────────────────────────
# Results are sorted by time
# ──────────────────────────────────────────────────────────────────────────────

class TestResultOrdering:
    def test_results_sorted_by_time_start(self):
        eng = engine_from(
            make_atom("a3", 20, 25, emotion="confident"),
            make_atom("a1", 0,  5,  emotion="confident"),
            make_atom("a2", 10, 15, emotion="confident"),
        )
        results = eng.find_emotion("confident")
        times = [r.time_start for r in results]
        assert times == sorted(times)