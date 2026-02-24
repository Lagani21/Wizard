"""
tests/test_writer.py — Unit tests for WizWriter keyword extraction and atom builders.

No file I/O. No ML models. Pure Python logic only.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wiz.writer import extract_keywords


# ──────────────────────────────────────────────────────────────────────────────
# extract_keywords
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractKeywords:
    def test_returns_set(self):
        result = extract_keywords("hello world")
        assert isinstance(result, set)

    def test_stops_words_removed(self):
        keywords = extract_keywords("the quick brown fox")
        assert "the" not in keywords

    def test_meaningful_words_kept(self):
        keywords = extract_keywords("revenue margins declined sharply")
        assert "revenue" in keywords
        assert "margins" in keywords

    def test_bigrams_included(self):
        keywords = extract_keywords("machine learning pipeline")
        assert "machine learning" in keywords or "learning pipeline" in keywords

    def test_empty_string(self):
        assert extract_keywords("") == set()

    def test_only_stop_words(self):
        # Sentence of only stop words should produce no keywords
        result = extract_keywords("the and or but is are")
        assert result == set()

    def test_case_insensitive(self):
        kw1 = extract_keywords("Revenue")
        kw2 = extract_keywords("revenue")
        assert kw1 == kw2

    def test_possessive_stripped(self):
        # "company's" should strip the possessive suffix
        keywords = extract_keywords("the company's revenue")
        assert "company" in keywords

    def test_min_length_enforced(self):
        # Single-letter and two-letter tokens should be excluded
        keywords = extract_keywords("a is go to")
        assert all(len(k.split()[0]) >= 3 for k in keywords)

    def test_realistic_sentence(self):
        text = "CEO discusses Q3 revenue and expresses concern about margins"
        keywords = extract_keywords(text)
        assert "revenue" in keywords
        assert "margins" in keywords
        assert "concern" in keywords