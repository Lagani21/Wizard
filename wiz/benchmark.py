"""
wiz/benchmark.py — Graph search vs. naive SQL search on 1+ hours of footage.

Run directly:
    python -m wiz.benchmark
    python -m wiz.benchmark --hours 2
    python -m wiz.benchmark --hours 1 --runs 100

What it measures
────────────────
  Graph (warm)   Queries run against the in-memory WizGraph after index build.
                 Simulates repeated queries in the same editing session.

  Graph (cold)   Full round-trip: load .wiz from disk, build index, then query.
                 Simulates a fresh open of a file.

  Naive SQL      Every query hits the SQLite file directly with JOINs and
                 LIKE clauses — no in-memory indexing.

The naive SQL schema is identical to the .wiz tag schema so the comparison
is a fair "graph vs. pure SQL" test on the same underlying data.
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import statistics
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .format import WizAtom, WizFile
    from .graph import WizGraph, TagCondition
    from .search import SearchEngine
    from .writer import extract_keywords
except ImportError:
    from wiz.format import WizAtom, WizFile
    from wiz.graph import WizGraph, TagCondition
    from wiz.search import SearchEngine
    from wiz.writer import extract_keywords


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ──────────────────────────────────────────────────────────────────────────────

SPEAKERS = ["SPEAKER_01", "SPEAKER_02", "SPEAKER_03"]
EMOTIONS = ["confident", "sad", "angry", "excited", "neutral", "thoughtful"]
TOPIC_CLUSTERS: Dict[str, List[str]] = {
    "machine learning": [
        "machine learning", "neural network", "deep learning", "training data",
        "model accuracy", "gradient descent", "overfitting", "backpropagation",
    ],
    "interview": [
        "interview", "experience", "background", "career", "role",
        "team", "company", "culture", "responsibilities",
    ],
    "product launch": [
        "product launch", "release date", "roadmap", "feature", "customer",
        "market", "competition", "pricing", "strategy",
    ],
    "personal story": [
        "family", "childhood", "growing up", "school", "inspiration",
        "mentor", "challenge", "overcome", "journey",
    ],
}
_ALL_TOPICS = [t for cluster in TOPIC_CLUSTERS.values() for t in cluster]

CAPTION_TEMPLATES = [
    "Two people sit across from each other at a table",
    "Interview setting with warm lighting",
    "Subject gestures while speaking",
    "Close-up of speaker's face showing expression",
    "Wide shot of interview room",
]


def _rand_turns(duration_s: float, rng: random.Random) -> List[Tuple[float, float, str]]:
    """Generate (start, end, speaker) speech turns filling most of duration_s."""
    turns: List[Tuple[float, float, str]] = []
    t = 0.0
    while t < duration_s - 1.0:
        # Optional silence gap
        gap = rng.choice([0.0, 0.0, 0.0, rng.uniform(0.3, 2.0)])
        t += gap
        spk = rng.choice(SPEAKERS)
        seg_len = rng.uniform(3.0, 45.0)
        end = min(t + seg_len, duration_s)
        turns.append((t, end, spk))
        t = end
    return turns


def generate_synthetic_wiz(
    wiz_path: str,
    hours: float = 1.0,
    seed: int = 42,
) -> Tuple[str, Dict[str, int]]:
    """
    Generate a synthetic .wiz file representing `hours` of footage.

    Returns (wiz_path, stats_dict).
    """
    rng = random.Random(seed)
    duration_s = hours * 3600.0
    fps = 29.97
    atoms: List[WizAtom] = []
    stats: Dict[str, int] = {}

    def make(t_start: float, t_end: float) -> WizAtom:
        return WizAtom(
            atom_id=str(uuid.uuid4()),
            frame_start=int(t_start * fps),
            frame_end=int(t_end * fps),
            time_start=t_start,
            time_end=t_end,
        )

    # ── 1. blink / breath / safe_cut atoms (1.5 s windows) ───────────────────
    t = 0.0
    nc_count = 0
    sc_count = 0
    while t < duration_s:
        end = min(t + 1.5, duration_s)
        atom = make(t, end)
        is_blink = rng.random() < 0.08    # ~8 % windows have a blink
        is_breath = rng.random() < 0.06   # ~6 % have a breath
        if is_blink:
            atom.add_tag("blink", "true", confidence=rng.uniform(0.7, 1.0))
        if is_breath:
            atom.add_tag("breath", "true", confidence=rng.uniform(0.6, 0.95))
        if not is_blink and not is_breath:
            atom.add_tag("safe_cut", "true")
            sc_count += 1
        atoms.append(atom)
        nc_count += 1
        t = end
    stats["no_cut_atoms"] = nc_count
    stats["safe_cut_atoms"] = sc_count

    # ── 2. speaker turns ──────────────────────────────────────────────────────
    turns = _rand_turns(duration_s, rng)
    spk_count = 0
    for t_start, t_end, spk in turns:
        topic_phrase = rng.choice(_ALL_TOPICS)
        words = topic_phrase.split() + [
            rng.choice(["important", "interesting", "challenging",
                        "fascinating", "difficult", "amazing", "key"])
        ]
        transcript = " ".join(words * rng.randint(3, 12))

        atom = make(t_start, t_end)
        atom.add_tag("speaker", spk)
        atom.add_tag("transcript", transcript)
        for kw in extract_keywords(transcript):
            atom.add_tag("topic", kw)
        atoms.append(atom)
        spk_count += 1

        # Natural pause after this turn
        if rng.random() < 0.3:
            gap_start = t_end
            gap_end = gap_start + rng.uniform(0.4, 1.5)
            if gap_end <= duration_s:
                pause = make(gap_start, gap_end)
                pause.add_tag("safe_cut", "pause")
                atoms.append(pause)
    stats["speaker_atoms"] = spk_count

    # ── 3. emotion atoms (30 s windows) ───────────────────────────────────────
    t = 0.0
    em_count = 0
    while t < duration_s:
        end = min(t + 30.0, duration_s)
        atom = make(t, end)
        emotion = rng.choice(EMOTIONS)
        atom.add_tag("emotion", emotion, confidence=rng.uniform(0.55, 0.99))
        atoms.append(atom)
        em_count += 1
        t = end
    stats["emotion_atoms"] = em_count

    # ── 4. caption atoms (5 s windows) ────────────────────────────────────────
    t = 0.0
    cap_count = 0
    while t < duration_s:
        end = min(t + 5.0, duration_s)
        atom = make(t, end)
        atom.add_tag("caption", rng.choice(CAPTION_TEMPLATES))
        atoms.append(atom)
        cap_count += 1
        t = end
    stats["caption_atoms"] = cap_count

    stats["total_atoms"] = len(atoms)
    total_tags = sum(len(a.tags) for a in atoms)
    stats["total_tags"] = total_tags

    WizFile(wiz_path).write(atoms)
    return wiz_path, stats


# ──────────────────────────────────────────────────────────────────────────────
# Naive SQL search (baseline)
# ──────────────────────────────────────────────────────────────────────────────

class NaiveSearch:
    """
    Executes search queries directly against the .wiz SQLite file —
    no in-memory index, every query hits disk.
    """

    def __init__(self, wiz_path: str) -> None:
        self._path = wiz_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def find_person_topic(self, speaker_id: str, topic: str) -> list:
        sql = """
            SELECT DISTINCT a.atom_id, a.time_start, a.time_end
            FROM atoms a
            JOIN atom_tags ts ON a.atom_id = ts.atom_id
                              AND ts.tag_type = 'speaker'
                              AND ts.tag_value = ?
            JOIN atom_tags tt ON a.atom_id = tt.atom_id
                              AND tt.tag_type = 'topic'
                              AND tt.tag_value = ?
        """
        with self._conn() as conn:
            return conn.execute(sql, (speaker_id, topic)).fetchall()

    def find_emotion(self, emotion: str) -> list:
        sql = """
            SELECT DISTINCT a.atom_id, a.time_start, a.time_end
            FROM atoms a
            JOIN atom_tags t ON a.atom_id = t.atom_id
                             AND t.tag_type = 'emotion'
                             AND t.tag_value = ?
        """
        with self._conn() as conn:
            return conn.execute(sql, (emotion,)).fetchall()

    def find_safe_cuts(self) -> list:
        sql = """
            SELECT DISTINCT a.atom_id, a.time_start, a.time_end
            FROM atoms a
            JOIN atom_tags t ON a.atom_id = t.atom_id
                             AND t.tag_type = 'safe_cut'
        """
        with self._conn() as conn:
            return conn.execute(sql).fetchall()

    def find_person_topic_no_blink(self, speaker_id: str, topic: str) -> list:
        # Get speaker+topic atoms
        candidate_sql = """
            SELECT DISTINCT a.atom_id, a.time_start, a.time_end
            FROM atoms a
            JOIN atom_tags ts ON a.atom_id = ts.atom_id
                              AND ts.tag_type = 'speaker'
                              AND ts.tag_value = ?
            JOIN atom_tags tt ON a.atom_id = tt.atom_id
                              AND tt.tag_type = 'topic'
                              AND tt.tag_value = ?
        """
        # Get blink windows
        blink_sql = """
            SELECT a.time_start, a.time_end
            FROM atoms a
            JOIN atom_tags t ON a.atom_id = t.atom_id
                             AND t.tag_type = 'blink'
                             AND t.tag_value = 'true'
        """
        with self._conn() as conn:
            candidates = conn.execute(candidate_sql, (speaker_id, topic)).fetchall()
            blink_ranges = conn.execute(blink_sql).fetchall()

        # Temporal filter in Python (same as graph approach but on SQL data)
        result = []
        for row in candidates:
            blocked = any(
                row["time_start"] < b["time_end"] and row["time_end"] > b["time_start"]
                for b in blink_ranges
            )
            if not blocked:
                result.append(row)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────

def _timed(fn, *args, runs: int = 50) -> Tuple[float, float, float]:
    """Run fn(*args) `runs` times. Return (median_ms, mean_ms, stdev_ms)."""
    times_ms: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return (
        statistics.median(times_ms),
        statistics.mean(times_ms),
        statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
    )


def run_benchmark(
    hours: float = 1.0,
    runs: int = 50,
    keep_file: bool = False,
    wiz_path: Optional[str] = None,
) -> None:
    """
    Main benchmark entry point.

    Generates synthetic footage data, then benchmarks 4 query types across
    graph (warm), graph (cold), and naive SQL approaches.
    """
    print("=" * 72)
    print(f"  WIZ Search Benchmark — {hours:.1f}h footage, {runs} runs per query")
    print("=" * 72)

    # ── generate data ─────────────────────────────────────────────────────────
    if wiz_path is None:
        tmp = tempfile.mktemp(suffix=".wiz")
        wiz_path = tmp
    else:
        tmp = None

    print(f"\n[1/4] Generating synthetic {hours:.1f}h .wiz file …")
    t0 = time.perf_counter()
    _, gen_stats = generate_synthetic_wiz(wiz_path, hours=hours)
    gen_ms = (time.perf_counter() - t0) * 1000

    print(f"       atoms         : {gen_stats['total_atoms']:,}")
    print(f"       tags          : {gen_stats['total_tags']:,}")
    print(f"       speaker atoms : {gen_stats['speaker_atoms']:,}")
    print(f"       safe_cut atoms: {gen_stats['safe_cut_atoms']:,}")
    print(f"       file size     : {Path(wiz_path).stat().st_size / 1024:.0f} KB")
    print(f"       generated in  : {gen_ms:.0f} ms")

    # ── graph: build index ────────────────────────────────────────────────────
    print(f"\n[2/4] Building graph index …")
    t0 = time.perf_counter()
    engine = SearchEngine(wiz_path)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"       index build   : {build_ms:.1f} ms")
    print(f"       tag types     : {engine.graph.tag_type_count}")
    print(f"       speakers      : {engine.speakers()}")
    print(f"       emotions      : {engine.emotions()[:4]} …")

    # ── naive baseline ────────────────────────────────────────────────────────
    naive = NaiveSearch(wiz_path)

    # ── pick realistic query parameters ───────────────────────────────────────
    test_speaker = "SPEAKER_01"
    test_topic = "machine learning"   # a common bigram in the synthetic data
    test_emotion = "confident"

    # ── run queries ───────────────────────────────────────────────────────────
    print(f"\n[3/4] Running {runs} iterations per query …")
    print()

    queries = [
        (
            "Q1: person+topic",
            f"find_person_topic({test_speaker!r}, {test_topic!r})",
            lambda: engine.find_person_topic(test_speaker, test_topic),
            lambda: naive.find_person_topic(test_speaker, test_topic),
        ),
        (
            "Q2: emotion",
            f"find_emotion({test_emotion!r})",
            lambda: engine.find_emotion(test_emotion),
            lambda: naive.find_emotion(test_emotion),
        ),
        (
            "Q3: safe cuts",
            "find_safe_cuts()",
            lambda: engine.find_safe_cuts(),
            lambda: naive.find_safe_cuts(),
        ),
        (
            "Q4: person+topic+no_blink",
            f"find_person_topic_no_blink({test_speaker!r}, {test_topic!r})",
            lambda: engine.find_person_topic_no_blink(test_speaker, test_topic),
            lambda: naive.find_person_topic_no_blink(test_speaker, test_topic),
        ),
    ]

    results_table: List[dict] = []

    for label, description, graph_fn, naive_fn in queries:
        # Warm graph results
        graph_result = graph_fn()
        graph_med, graph_mean, graph_std = _timed(graph_fn, runs=runs)

        # Cold graph (build + query each time)
        def cold_fn():
            e2 = SearchEngine(wiz_path)
            return graph_fn()
        cold_med, cold_mean, cold_std = _timed(cold_fn, runs=max(5, runs // 10))

        # Naive SQL
        naive_result = naive_fn()
        naive_med, naive_mean, naive_std = _timed(naive_fn, runs=runs)

        speedup = naive_med / graph_med if graph_med > 0 else float("inf")

        results_table.append({
            "label": label,
            "description": description,
            "graph_hits": len(graph_result),
            "naive_hits": len(naive_result),
            "graph_warm_med": graph_med,
            "graph_cold_med": cold_med,
            "naive_med": naive_med,
            "speedup": speedup,
        })

    # ── print results ─────────────────────────────────────────────────────────
    print("[4/4] Results\n")
    print(f"{'Query':<30} {'Hits':>5}  {'Graph warm':>11}  {'Graph cold':>11}  {'Naive SQL':>10}  {'Speedup':>8}")
    print("-" * 82)

    for r in results_table:
        hits = r["graph_hits"]
        g_warm = f"{r['graph_warm_med']:.3f} ms"
        g_cold = f"{r['graph_cold_med']:.1f} ms"
        naive = f"{r['naive_med']:.3f} ms"
        sp = f"{r['speedup']:.1f}×"
        print(f"{r['label']:<30} {hits:>5}  {g_warm:>11}  {g_cold:>11}  {naive:>10}  {sp:>8}")

    print()
    avg_speedup = statistics.mean(r["speedup"] for r in results_table)
    print(f"  Average speedup (graph warm vs naive SQL): {avg_speedup:.1f}×")
    print(f"  Index build cost (amortised over {runs} queries):",
          f"{build_ms / runs:.3f} ms/query")
    print()
    print("  Interpretation:")
    print("  - Graph (warm): in-memory set ops — pays build cost once per session.")
    print("  - Graph (cold): full disk load + index build + query every time.")
    print("  - Naive SQL:    disk I/O + JOIN + filter on every query call.")
    print()

    # ── cleanup ───────────────────────────────────────────────────────────────
    if tmp and not keep_file:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    else:
        print(f"  .wiz file kept at: {wiz_path}")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# JSON-returning benchmark (for API / monitoring dashboard)
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark_json(
    hours: float = 0.1,
    runs: int = 20,
) -> dict:
    """
    Run the graph-vs-SQL benchmark and return structured results as a dict.

    Designed for the monitoring API endpoint — no stdout output.
    Uses a temporary .wiz file that is deleted after the run.
    """
    import tempfile, os

    wiz_path = tempfile.mktemp(suffix=".wiz")
    try:
        t0 = time.perf_counter()
        _, gen_stats = generate_synthetic_wiz(wiz_path, hours=hours)
        gen_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        engine = SearchEngine(wiz_path)
        build_ms = (time.perf_counter() - t0) * 1000

        naive = NaiveSearch(wiz_path)

        test_speaker = "SPEAKER_01"
        test_topic   = "machine learning"
        test_emotion = "confident"

        query_defs = [
            (
                "Person + Topic",
                lambda: engine.find_person_topic(test_speaker, test_topic),
                lambda: naive.find_person_topic(test_speaker, test_topic),
            ),
            (
                "Emotion",
                lambda: engine.find_emotion(test_emotion),
                lambda: naive.find_emotion(test_emotion),
            ),
            (
                "Safe Cuts",
                lambda: engine.find_safe_cuts(),
                lambda: naive.find_safe_cuts(),
            ),
            (
                "Person + Topic + No Blink",
                lambda: engine.find_person_topic_no_blink(test_speaker, test_topic),
                lambda: naive.find_person_topic_no_blink(test_speaker, test_topic),
            ),
        ]

        queries = []
        for label, graph_fn, naive_fn in query_defs:
            graph_result = graph_fn()
            graph_med, _, _ = _timed(graph_fn, runs=runs)
            naive_med, _, _ = _timed(naive_fn, runs=runs)
            speedup = naive_med / graph_med if graph_med > 0 else 0.0
            queries.append({
                "label":        label,
                "hits":         len(graph_result),
                "graph_ms":     round(graph_med, 4),
                "sql_ms":       round(naive_med, 4),
                "speedup":      round(speedup, 1),
            })

        avg_speedup = statistics.mean(q["speedup"] for q in queries)

        return {
            "hours":        hours,
            "runs":         runs,
            "total_atoms":  gen_stats["total_atoms"],
            "total_tags":   gen_stats["total_tags"],
            "index_build_ms": round(build_ms, 1),
            "gen_ms":       round(gen_ms, 1),
            "avg_speedup":  round(avg_speedup, 1),
            "queries":      queries,
        }
    finally:
        try:
            os.unlink(wiz_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# O(n) vs O(log n) prefix-search benchmark
# ──────────────────────────────────────────────────────────────────────────────

def _linear_prefix_search(tag_values: List[str], prefix: str, tag_index: dict) -> Set[str]:
    """
    Naive O(N) prefix search: compare the prefix to every stored value.
    This is what you'd write without the sorted-value tree.
    """
    result: Set[str] = set()
    for v in tag_values:
        if v.startswith(prefix):
            result |= tag_index.get(v, set())
    return result


def run_tree_vs_linear_benchmark(
    hours: float = 1.0,
    runs: int = 200,
) -> None:
    """
    Benchmark O(log N) prefix_search (sorted array + bisect)
    vs O(N) linear scan over stored topic values.

    Shows how the gap widens as vocabulary N grows with footage length.
    """
    print()
    print("=" * 72)
    print(f"  O(log N) tree vs O(N) linear — prefix topic search")
    print(f"  {hours:.1f}h footage, {runs} iterations per prefix")
    print("=" * 72)

    with tempfile.NamedTemporaryFile(suffix=".wiz", delete=False) as f:
        wiz_path = f.name

    try:
        generate_synthetic_wiz(wiz_path, hours=hours)
        graph = WizGraph.from_file(wiz_path)

        topic_vals: List[str] = graph._sorted_vals.get("topic", [])
        topic_index: dict = {
            v: s for v, s in graph._tag_index.get("topic", {}).items()
            if v is not None
        }
        N = len(topic_vals)
        print(f"\n  Unique topic values (N): {N:,}")

        # Test prefixes of increasing specificity
        prefixes = ["m", "ma", "mac", "mach", "machine", "machine l", "machine learning"]

        print(f"\n  {'Prefix':<20} {'N matches':>9}  {'Linear O(N)':>12}  {'Tree O(logN)':>13}  {'Speedup':>8}")
        print("  " + "-" * 68)

        for prefix in prefixes:
            # Warm up
            _linear_prefix_search(topic_vals, prefix, topic_index)
            graph.prefix_search("topic", prefix)

            lin_med, _, _ = _timed(
                lambda p=prefix: _linear_prefix_search(topic_vals, p, topic_index),
                runs=runs,
            )
            tree_med, _, _ = _timed(
                lambda p=prefix: graph.prefix_search("topic", p),
                runs=runs,
            )
            hits = len(graph.prefix_search("topic", prefix))
            sp = lin_med / tree_med if tree_med > 0 else float("inf")

            print(
                f"  {prefix!r:<20} {hits:>9}  "
                f"{lin_med * 1000:.3f} µs    {tree_med * 1000:.3f} µs    {sp:>7.1f}×"
            )

        print(f"\n  Linear scan always touches all {N:,} values.")
        print(f"  Tree (bisect) touches ~log₂({N}) ≈ {N.bit_length()} positions to land,")
        print(f"  then only the k matching values (k ≪ N for specific queries).")

    finally:
        os.unlink(wiz_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark WIZ graph search vs naive SQL search."
    )
    parser.add_argument(
        "--hours", type=float, default=1.0,
        help="Hours of synthetic footage to generate (default: 1.0)",
    )
    parser.add_argument(
        "--runs", type=int, default=50,
        help="Query iterations per method (default: 50)",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="Keep the generated .wiz file after benchmarking",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Path for the generated .wiz file (default: temp file)",
    )
    args = parser.parse_args()

    run_benchmark(
        hours=args.hours,
        runs=args.runs,
        keep_file=args.keep,
        wiz_path=args.out,
    )
    run_tree_vs_linear_benchmark(hours=args.hours, runs=args.runs)


if __name__ == "__main__":
    main()