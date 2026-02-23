"""
wiz — OO tag-based .WIZ file format and graph search layer.

Quick-start
───────────
  # Search an existing .wiz file
  from wiz import SearchEngine
  engine = SearchEngine("results/interview.wiz")
  hits = engine.find_person_topic("SPEAKER_01", "machine learning")

  # Convert a pipeline run to .wiz
  from wiz import context_to_wiz
  wiz_path = context_to_wiz(pipeline_context, "results/interview.wiz")

  # Run the benchmark
  python -m wiz.benchmark --hours 1 --runs 50
"""

from wiz.format import WizAtom, WizTag, WizFile          # noqa: F401
from wiz.graph import WizGraph, TagCondition              # noqa: F401
from wiz.search import SearchEngine, SearchResult, open_wiz  # noqa: F401
from wiz.writer import WizWriter, context_to_wiz          # noqa: F401

__all__ = [
    "WizAtom", "WizTag", "WizFile",
    "WizGraph", "TagCondition",
    "SearchEngine", "SearchResult", "open_wiz",
    "WizWriter", "context_to_wiz",
]