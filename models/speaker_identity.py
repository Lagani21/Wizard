"""
Cross-clip speaker identity registry.

Assigns consistent PERSON_XXX IDs across video files by matching
speaker embeddings via cosine similarity on a persistent registry.
"""

import json
import logging
import numpy as np
from datetime import date
from pathlib import Path
from typing import Dict, Optional

# Default registry location — lives alongside the .wiz result files
_DEFAULT_REGISTRY = Path(__file__).parent.parent / "web" / "results" / "speaker_registry.json"

# Cosine similarity threshold: ≥ this → same person
SIMILARITY_THRESHOLD = 0.75


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SpeakerIdentityRegistry:
    """
    Persistent registry that maps per-clip speaker labels (SPEAKER_00 etc.)
    to stable cross-clip identities (PERSON_001, PERSON_002 …).

    Each known person is stored as a running-mean embedding vector.
    On each call to resolve(), new speakers are either matched to an
    existing person (cosine sim ≥ threshold) or registered as a new one.

    The registry is serialised to JSON so identities persist across runs.
    """

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.registry_path = Path(registry_path or _DEFAULT_REGISTRY)
        self.threshold = similarity_threshold
        self.logger = logging.getLogger("wiz.models.speaker_identity")
        self._speakers: Dict[str, dict] = {}   # pid → {embedding, clip_count, first_seen}
        self._next_id: int = 1
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            with open(self.registry_path) as fh:
                raw = json.load(fh)
            for pid, entry in raw.get("speakers", {}).items():
                entry["embedding"] = np.array(entry["embedding"], dtype=np.float32)
                self._speakers[pid] = entry
            self._next_id = raw.get("next_id", len(self._speakers) + 1)
            self.logger.info(
                f"Loaded speaker registry: {len(self._speakers)} known persons"
            )
        except Exception as e:
            self.logger.warning(f"Could not load speaker registry, starting fresh: {e}")

    def _save(self) -> None:
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            out: dict = {"next_id": self._next_id, "speakers": {}}
            for pid, entry in self._speakers.items():
                out["speakers"][pid] = {
                    "embedding":  entry["embedding"].tolist(),
                    "clip_count": entry["clip_count"],
                    "first_seen": entry["first_seen"],
                }
            with open(self.registry_path, "w") as fh:
                json.dump(out, fh, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save speaker registry: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, str]:
        """
        Map per-clip speaker IDs to global Person-IDs.

        Args:
            embeddings: {clip_speaker_id → embedding_vector}

        Returns:
            {clip_speaker_id → person_id}
            e.g. {"SPEAKER_00": "PERSON_001", "SPEAKER_01": "PERSON_003"}
        """
        today = date.today().isoformat()
        mapping: Dict[str, str] = {}
        for clip_id, emb in embeddings.items():
            mapping[clip_id] = self._match_or_create(emb, today)
        self._save()
        return mapping

    def known_persons(self) -> list:
        """Return all registered Person-IDs."""
        return list(self._speakers.keys())

    def reset(self) -> None:
        """Clear all known persons (useful in tests)."""
        self._speakers.clear()
        self._next_id = 1
        self._save()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _match_or_create(self, embedding: np.ndarray, today: str) -> str:
        best_pid: Optional[str] = None
        best_sim: float = -1.0

        for pid, entry in self._speakers.items():
            sim = _cosine_sim(embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_pid = pid

        if best_pid is not None and best_sim >= self.threshold:
            # Update stored embedding as running mean
            entry = self._speakers[best_pid]
            n = entry["clip_count"]
            entry["embedding"] = (
                (entry["embedding"] * n + embedding) / (n + 1)
            ).astype(np.float32)
            entry["clip_count"] += 1
            self.logger.info(
                f"Matched speaker to {best_pid} (similarity={best_sim:.3f})"
            )
            return best_pid

        # Register new person
        new_id = f"PERSON_{self._next_id:03d}"
        self._next_id += 1
        self._speakers[new_id] = {
            "embedding":  embedding.copy().astype(np.float32),
            "clip_count": 1,
            "first_seen": today,
        }
        self.logger.info(
            f"New speaker registered as {new_id} "
            f"(best previous match was {best_sim:.3f})"
        )
        return new_id