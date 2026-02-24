"""
tests/test_graph.py — Unit tests for WizGraph and _stem_variants.

All tests use in-memory WizAtoms — no file I/O, no ML models.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wiz.format import WizAtom
from wiz.graph import WizGraph, TagCondition, _stem_variants


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


def build_graph(*atoms: WizAtom) -> WizGraph:
    return WizGraph().build(list(atoms))


# ──────────────────────────────────────────────────────────────────────────────
# _stem_variants
# ──────────────────────────────────────────────────────────────────────────────

class TestStemVariants:
    def test_plural_s(self):
        assert "project" in _stem_variants("projects")

    def test_plural_es(self):
        assert "revenue" in _stem_variants("revenues")

    def test_plural_ies(self):
        assert "currency" in _stem_variants("currencies")

    def test_ing(self):
        variants = _stem_variants("making")
        assert "make" in variants or "mak" in variants

    def test_ed(self):
        assert "launch" in _stem_variants("launched")

    def test_er(self):
        assert "develop" in _stem_variants("developer")

    def test_original_always_included(self):
        assert "revenues" in _stem_variants("revenues")
        assert "launched" in _stem_variants("launched")

    def test_short_word_not_over_stemmed(self):
        # Words shorter than the threshold should not produce empty variants
        variants = _stem_variants("run")
        assert all(len(v) >= 3 for v in variants)


# ──────────────────────────────────────────────────────────────────────────────
# WizGraph — build and basic lookup
# ──────────────────────────────────────────────────────────────────────────────

class TestWizGraphBuild:
    def test_atom_count(self):
        g = build_graph(
            make_atom("a1", 0, 5, speaker="PERSON_001"),
            make_atom("a2", 5, 10, speaker="PERSON_002"),
        )
        assert g.atom_count == 2

    def test_find_by_tag_exact(self):
        g = build_graph(
            make_atom("a1", 0, 5, speaker="PERSON_001"),
            make_atom("a2", 5, 10, speaker="PERSON_002"),
        )
        result = g.find_by_tag("speaker", "PERSON_001")
        assert result == {"a1"}

    def test_find_by_tag_any_value(self):
        g = build_graph(
            make_atom("a1", 0, 5, speaker="PERSON_001"),
            make_atom("a2", 5, 10, speaker="PERSON_002"),
        )
        result = g.find_by_tag("speaker")
        assert result == {"a1", "a2"}

    def test_missing_tag_returns_empty(self):
        g = build_graph(make_atom("a1", 0, 5, speaker="PERSON_001"))
        assert g.find_by_tag("emotion", "happy") == set()

    def test_multiple_tags_on_one_atom(self):
        atom = make_atom("a1", 0, 5, speaker="PERSON_001", emotion="confident")
        g = build_graph(atom)
        assert "a1" in g.find_by_tag("speaker", "PERSON_001")
        assert "a1" in g.find_by_tag("emotion", "confident")


# ──────────────────────────────────────────────────────────────────────────────
# WizGraph — prefix search
# ──────────────────────────────────────────────────────────────────────────────

class TestPrefixSearch:
    def test_prefix_match(self):
        g = build_graph(
            make_atom("a1", 0, 5, topic="machine learning"),
            make_atom("a2", 5, 10, topic="machine vision"),
            make_atom("a3", 10, 15, topic="revenue"),
        )
        result = g.prefix_search("topic", "machine")
        assert "a1" in result
        assert "a2" in result
        assert "a3" not in result

    def test_exact_prefix(self):
        g = build_graph(make_atom("a1", 0, 5, topic="revenue"))
        assert "a1" in g.prefix_search("topic", "rev")

    def test_no_match(self):
        g = build_graph(make_atom("a1", 0, 5, topic="revenue"))
        assert g.prefix_search("topic", "xyz") == set()


# ──────────────────────────────────────────────────────────────────────────────
# WizGraph — find_topic (includes stemming)
# ──────────────────────────────────────────────────────────────────────────────

class TestFindTopic:
    def test_exact_match(self):
        g = build_graph(make_atom("a1", 0, 5, topic="revenue"))
        assert "a1" in g.find_topic("revenue")

    def test_stemmed_plural(self):
        # Stored as "revenue", queried as "revenues"
        g = build_graph(make_atom("a1", 0, 5, topic="revenue"))
        assert "a1" in g.find_topic("revenues")

    def test_stemmed_past_tense(self):
        # Stored as "launch", queried as "launched"
        g = build_graph(make_atom("a1", 0, 5, topic="launch"))
        assert "a1" in g.find_topic("launched")

    def test_multiword_exact(self):
        g = build_graph(make_atom("a1", 0, 5, topic="machine learning"))
        assert "a1" in g.find_topic("machine learning")

    def test_multiword_prefix(self):
        g = build_graph(make_atom("a1", 0, 5, topic="machine learning"))
        assert "a1" in g.find_topic("machine learn")


# ──────────────────────────────────────────────────────────────────────────────
# WizGraph — temporal overlap
# ──────────────────────────────────────────────────────────────────────────────

class TestTemporalOverlap:
    def test_overlap(self):
        g = build_graph(
            make_atom("a1", 0.0, 5.0),
            make_atom("a2", 3.0, 8.0),
            make_atom("a3", 10.0, 15.0),
        )
        result = g.find_overlapping(4.0, 6.0)
        assert "a1" in result
        assert "a2" in result
        assert "a3" not in result

    def test_exact_boundary_excluded(self):
        g = build_graph(make_atom("a1", 0.0, 5.0))
        # Query starts exactly when atom ends — no overlap
        assert "a1" not in g.find_overlapping(5.0, 10.0)

    def test_contains_query(self):
        g = build_graph(make_atom("a1", 0.0, 10.0))
        # Query is entirely inside atom
        assert "a1" in g.find_overlapping(2.0, 4.0)


# ──────────────────────────────────────────────────────────────────────────────
# WizGraph — multi-condition AND query
# ──────────────────────────────────────────────────────────────────────────────

class TestQuery:
    def test_and_query(self):
        g = build_graph(
            make_atom("a1", 0, 5, speaker="PERSON_001", emotion="confident"),
            make_atom("a2", 5, 10, speaker="PERSON_001", emotion="neutral"),
            make_atom("a3", 10, 15, speaker="PERSON_002", emotion="confident"),
        )
        results = g.query(
            TagCondition("speaker", "PERSON_001"),
            TagCondition("emotion", "confident"),
        )
        ids = {a.atom_id for a in results}
        assert ids == {"a1"}

    def test_empty_result_on_no_match(self):
        g = build_graph(make_atom("a1", 0, 5, speaker="PERSON_001"))
        results = g.query(
            TagCondition("speaker", "PERSON_001"),
            TagCondition("emotion", "excited"),
        )
        assert results == []