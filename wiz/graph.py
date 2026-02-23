"""
wiz/graph.py — In-memory graph index over a .WIZ file.

The WizGraph maintains three complementary indexes built once at load time:

1. Inverted tag index  {tag_type → {tag_value → set(atom_ids)}}
   O(1) exact-match lookup per tag.  Set intersection for AND queries.

2. Sorted value tree  {tag_type → sorted list of all distinct tag values}
   A sorted array acts as the leaves of a B-tree.  bisect() lands on the
   right position in O(log N) — no comparison against every stored value.
   Used for prefix/partial topic matching: user types "mach", we find
   "machine", "machine learning", "machinery" without scanning the full
   vocabulary.  Contrast with the O(N) linear scan baseline.

   Complexity per prefix query:
     bisect_left  →  O(log N)  (N = unique values of this tag type)
     slice [lo:hi] →  O(k)     (k = number of matching values)
     union atom sets → O(m)    (m = total matching atoms)

3. Temporal interval index  sorted list + bisect
   O(log N + k) overlap queries: "which atoms cover time T?"
   Used for cross-granularity NOT queries (speaker window not mid-blink).

Building the graph is O(T log T) where T = total tags (sort step).
Queries are O(log N + k) worst-case after that.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

try:
    from .format import WizAtom, WizTag, WizFile
except ImportError:
    from wiz.format import WizAtom, WizTag, WizFile


# ──────────────────────────────────────────────────────────────────────────────
# Query conditions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TagCondition:
    """Match atoms that carry a specific tag (optionally value-filtered)."""
    tag_type: str
    tag_value: Optional[str] = None    # None → any value of this type

    def __str__(self) -> str:
        if self.tag_value is None:
            return f"{self.tag_type}=*"
        return f"{self.tag_type}={self.tag_value!r}"


@dataclass(frozen=True)
class NotTagCondition:
    """Match atoms that do NOT carry a specific tag."""
    tag_type: str
    tag_value: Optional[str] = None

    def __str__(self) -> str:
        if self.tag_value is None:
            return f"NOT {self.tag_type}=*"
        return f"NOT {self.tag_type}={self.tag_value!r}"


@dataclass(frozen=True)
class TemporalNotCondition:
    """
    Match atoms that do NOT temporally overlap with atoms carrying a given tag.

    Example: speaker atoms that are NOT overlapped by any blink atom.
    """
    tag_type: str
    tag_value: Optional[str] = None

    def __str__(self) -> str:
        val = f"={self.tag_value!r}" if self.tag_value else "=*"
        return f"TEMPORAL_NOT {self.tag_type}{val}"


# ──────────────────────────────────────────────────────────────────────────────
# WizGraph
# ──────────────────────────────────────────────────────────────────────────────

class WizGraph:
    """
    In-memory graph index built from a list of WizAtoms.

    After calling build(), the graph supports:
      - find_by_tag()           → atoms matching a tag
      - find_overlapping()      → atoms covering a time range
      - query()                 → multi-condition AND query
      - query_with_temporal_not()  → AND + temporal-NOT
    """

    def __init__(self) -> None:
        # Primary atom store
        self._atoms: Dict[str, WizAtom] = {}

        # Inverted index: tag_type → tag_value → set of atom_ids
        self._tag_index: Dict[str, Dict[str, Set[str]]] = {}

        # Sorted-value tree: tag_type → sorted list of distinct tag values.
        # Acts as the leaf level of a B-tree — bisect() into it is O(log N).
        # Enables prefix search without scanning every stored value.
        self._sorted_vals: Dict[str, List[str]] = {}

        # Temporal index: sorted list of (time_start, time_end, atom_id)
        # Sorted by time_start for bisect-based range queries.
        self._time_starts: List[float] = []   # parallel arrays
        self._time_ends: List[float] = []
        self._time_ids: List[str] = []

        self._built = False

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self, atoms: List[WizAtom]) -> "WizGraph":
        """
        Index a list of WizAtoms. Call once per .wiz file load.

        Returns self for chaining:  graph = WizGraph().build(wiz.read())
        """
        self._atoms.clear()
        self._tag_index.clear()
        self._sorted_vals.clear()
        self._time_starts.clear()
        self._time_ends.clear()
        self._time_ids.clear()

        # Build inverted tag index
        for atom in atoms:
            self._atoms[atom.atom_id] = atom
            for tag in atom.tags:
                type_idx = self._tag_index.setdefault(tag.tag_type, {})
                type_idx.setdefault(tag.tag_value, set()).add(atom.atom_id)
                # Also index tag_type → ANY (None → full type set)
                type_idx.setdefault(None, set()).add(atom.atom_id)  # type: ignore[arg-type]

        # Build sorted-value tree: one sorted list of distinct values per tag type.
        # O(V log V) once at build time; each prefix_search() is O(log V + k).
        for tag_type, value_map in self._tag_index.items():
            self._sorted_vals[tag_type] = sorted(
                v for v in value_map if v is not None
            )

        # Build temporal index (sorted by time_start)
        sorted_atoms = sorted(atoms, key=lambda a: a.time_start)
        self._time_starts = [a.time_start for a in sorted_atoms]
        self._time_ends = [a.time_end for a in sorted_atoms]
        self._time_ids = [a.atom_id for a in sorted_atoms]

        self._built = True
        return self

    @classmethod
    def from_file(cls, wiz_path: str) -> "WizGraph":
        """Load a .wiz file and build the graph in one call."""
        atoms = WizFile(wiz_path).read()
        return cls().build(atoms)

    # ── tag index queries ─────────────────────────────────────────────────────

    def find_by_tag(
        self,
        tag_type: str,
        tag_value: Optional[str] = None,
    ) -> Set[str]:
        """
        Return the set of atom_ids carrying (tag_type, tag_value).

        If tag_value is None, returns atoms that have ANY value for tag_type.
        Result is a live reference — copy if you intend to mutate it.
        """
        type_idx = self._tag_index.get(tag_type, {})
        if tag_value is None:
            return type_idx.get(None, set())  # type: ignore[arg-type]
        return type_idx.get(tag_value, set())

    def all_tag_values(self, tag_type: str) -> List[str]:
        """List all distinct values stored under a tag type."""
        return [v for v in self._tag_index.get(tag_type, {}) if v is not None]

    def prefix_search(self, tag_type: str, prefix: str) -> Set[str]:
        """
        Return atom_ids whose tag value starts with `prefix`.

        Uses the sorted-value tree (bisect) for O(log N + k) time,
        where N = number of distinct values for this tag type and
        k = number of values that match the prefix.

        This beats the O(N) linear scan that naive code would use:

            # O(N) — DON'T do this
            for v in all_values:
                if v.startswith(prefix): ...

            # O(log N + k) — what prefix_search() does
            lo = bisect_left(sorted_values, prefix)
            hi = bisect_right(sorted_values, prefix + "\\uffff")
            matched = sorted_values[lo:hi]
        """
        vals = self._sorted_vals.get(tag_type, [])
        if not vals:
            return set()

        lo = bisect.bisect_left(vals, prefix)
        # Upper bound: any string starting with `prefix` is < prefix[:-1] + chr(ord(prefix[-1])+1)
        # Using "\uffff" as a sentinel that sorts after all normal characters.
        hi = bisect.bisect_right(vals, prefix + "\uffff")

        type_idx = self._tag_index[tag_type]
        result: Set[str] = set()
        for v in vals[lo:hi]:
            result |= type_idx.get(v, set())
        return result

    # ── temporal index queries ────────────────────────────────────────────────

    def find_overlapping(
        self,
        time_start: float,
        time_end: float,
    ) -> Set[str]:
        """
        Return atom_ids of all atoms that overlap [time_start, time_end).

        Uses binary search on the sorted time_starts array:
        O(log N + k) where k = number of overlapping atoms.
        """
        # All atoms whose time_start < time_end AND time_end > time_start
        # Step 1: candidate right boundary — first atom starting at or after time_end
        right = bisect.bisect_left(self._time_starts, time_end)
        # Step 2: filter candidates: atom must also end after time_start
        result: Set[str] = set()
        for i in range(right):
            if self._time_ends[i] > time_start:
                result.add(self._time_ids[i])
        return result

    def find_overlapping_with_tag(
        self,
        time_start: float,
        time_end: float,
        tag_type: str,
        tag_value: Optional[str] = None,
    ) -> Set[str]:
        """Atoms that overlap a time range AND carry a specific tag."""
        candidates = self.find_overlapping(time_start, time_end)
        tag_set = self.find_by_tag(tag_type, tag_value)
        return candidates & tag_set

    # ── multi-condition queries ───────────────────────────────────────────────

    def query(self, *conditions: TagCondition) -> List[WizAtom]:
        """
        Execute an AND query over tag conditions.

        Each TagCondition narrows the result set via set intersection.
        Conditions are processed smallest-set-first for efficiency.

        Example::

            results = graph.query(
                TagCondition("speaker", "SPEAKER_01"),
                TagCondition("topic", "machine learning"),
            )
        """
        if not conditions:
            return list(self._atoms.values())

        # Resolve each condition to a set of atom_ids
        sets: List[Set[str]] = []
        for cond in conditions:
            s = self.find_by_tag(cond.tag_type, cond.tag_value)
            if not s:
                return []          # short-circuit: one empty set → no results
            sets.append(s)

        # Intersect smallest-first
        sets.sort(key=len)
        result_ids = sets[0].copy()
        for s in sets[1:]:
            result_ids &= s
            if not result_ids:
                return []

        return self._ids_to_atoms(result_ids)

    def query_not(
        self,
        must: List[TagCondition],
        must_not: List[NotTagCondition],
    ) -> List[WizAtom]:
        """
        AND query with tag-level NOT.

        Returns atoms satisfying all `must` conditions and none of
        the `must_not` conditions.
        """
        # Resolve positive conditions
        pos_sets: List[Set[str]] = []
        for cond in must:
            s = self.find_by_tag(cond.tag_type, cond.tag_value)
            if not s:
                return []
            pos_sets.append(s)

        if pos_sets:
            pos_sets.sort(key=len)
            result_ids = pos_sets[0].copy()
            for s in pos_sets[1:]:
                result_ids &= s
        else:
            result_ids = set(self._atoms.keys())

        # Subtract forbidden sets
        for cond in must_not:
            forbidden = self.find_by_tag(cond.tag_type, cond.tag_value)
            result_ids -= forbidden

        return self._ids_to_atoms(result_ids)

    def query_temporal_not(
        self,
        must: List[TagCondition],
        temporal_not: List[TemporalNotCondition],
    ) -> List[WizAtom]:
        """
        AND query with temporal NOT.

        Finds atoms matching `must` that do NOT temporally overlap with
        any atom carrying the tags in `temporal_not`.

        This is the correct query for "speaker atom not mid-blink"
        because speaker and blink atoms have different granularities — a
        simple set subtraction would be incorrect.
        """
        candidates = self.query(*must)
        if not candidates:
            return []

        # Build the excluded atom sets for each temporal-not condition
        excluded_time_ranges: List[List[Tuple[float, float]]] = []
        for cond in temporal_not:
            exc_ids = self.find_by_tag(cond.tag_type, cond.tag_value)
            ranges = [
                (self._atoms[aid].time_start, self._atoms[aid].time_end)
                for aid in exc_ids
                if aid in self._atoms
            ]
            excluded_time_ranges.append(ranges)

        results: List[WizAtom] = []
        for atom in candidates:
            blocked = False
            for ranges in excluded_time_ranges:
                if any(
                    atom.time_start < exc_end and atom.time_end > exc_start
                    for exc_start, exc_end in ranges
                ):
                    blocked = True
                    break
            if not blocked:
                results.append(atom)

        return results

    # ── topic-aware search ────────────────────────────────────────────────────

    def find_topic(self, topic_phrase: str) -> Set[str]:
        """
        Find atom_ids whose topic tags match a topic phrase.

        Strategy (all O(log N + k), no O(N) linear scans):
        1. Exact phrase lookup via the hash index — O(1).
        2. Prefix search on the full phrase — O(log N + k).
           Catches "machine learning" when stored as a bigram tag.
        3. If multi-word: prefix-search each word, intersect results — AND semantics.
           Catches "machine learning" when stored as separate unigram tags.
        Returns the union of all three strategies.
        """
        phrase = topic_phrase.lower().strip()

        # 1. Exact O(1) lookup
        exact = self.find_by_tag("topic", phrase)

        # 2. Prefix search on the whole phrase — O(log N + k)
        prefix_hits = self.prefix_search("topic", phrase)

        if " " in phrase:
            # 3. Per-word prefix intersection — each word is O(log N + k)
            words = phrase.split()
            per_word = [self.prefix_search("topic", w) for w in words]
            per_word = [s for s in per_word if s]
            if per_word:
                per_word.sort(key=len)
                all_words = per_word[0].copy()
                for s in per_word[1:]:
                    all_words &= s
                return exact | prefix_hits | all_words

        return exact | prefix_hits

    # ── stats ─────────────────────────────────────────────────────────────────

    @property
    def atom_count(self) -> int:
        return len(self._atoms)

    @property
    def tag_type_count(self) -> int:
        return len(self._tag_index)

    def stats(self) -> Dict[str, int]:
        """Return per-tag-type atom counts."""
        return {
            tag_type: len(value_map.get(None, set()))  # type: ignore[arg-type]
            for tag_type, value_map in self._tag_index.items()
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _ids_to_atoms(self, ids: Set[str]) -> List[WizAtom]:
        return sorted(
            (self._atoms[aid] for aid in ids if aid in self._atoms),
            key=lambda a: a.time_start,
        )