"""
Video MAE (Masked Autoencoder) wrapper for frame-level video captioning.

Produces short natural-language descriptions for ~5 s video windows.
Two backends are available:

    "mock"    — deterministic captions from motion + brightness heuristics;
                no model file required; suitable for development and testing.

    "coreml"  — loads a Vision-Language .mlmodel via coremltools and runs
                inference on Apple Neural Engine; requires the model file.
"""

import gc
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


logger = logging.getLogger("wiz.models.video_mae")


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BaseVideoMAE(ABC):
    """Abstract base class for video captioning backends."""

    @abstractmethod
    def generate_caption(
        self,
        frames: List[np.ndarray],
        fps: float,
        window_start_s: float,
    ) -> str:
        """
        Generate a natural-language description for a sequence of frames.

        Args:
            frames:          List of BGR frames (numpy arrays H×W×3)
            fps:             Source video fps (informational only)
            window_start_s:  Start time of this window in seconds

        Returns:
            Short description string, e.g. "person speaking with moderate movement"
        """

    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Return metadata about the backend."""


# ──────────────────────────────────────────────────────────────────────────────
# Mock backend (motion heuristics — no model required)
# ──────────────────────────────────────────────────────────────────────────────

class MockVideoMAE(BaseVideoMAE):
    """
    Deterministic video captioning using simple motion and brightness heuristics.

    Analyses frame-to-frame differences and overall brightness to produce
    plausible placeholder captions without loading any ML model.
    """

    def generate_caption(
        self,
        frames: List[np.ndarray],
        fps: float,
        window_start_s: float,
    ) -> str:
        if not frames:
            return "no visual content"

        # ── Motion magnitude (mean absolute frame-diff) ──────────────────────
        motion = self._compute_motion(frames)

        # ── Brightness ────────────────────────────────────────────────────────
        brightness = self._compute_brightness(frames)

        # ── Face-region proxy (brightness in central strip) ───────────────────
        face_activity = self._compute_face_region_activity(frames)

        # ── Compose caption from buckets ─────────────────────────────────────
        motion_desc = self._motion_description(motion)
        brightness_desc = self._brightness_description(brightness)
        face_desc = self._face_activity_description(face_activity)

        caption = f"{face_desc}, {motion_desc}, {brightness_desc} scene"
        logger.debug(
            f"MockVideoMAE @{window_start_s:.1f}s — "
            f"motion={motion:.3f}, brightness={brightness:.1f}, "
            f"face_activity={face_activity:.3f} → '{caption}'"
        )
        return caption

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_motion(self, frames: List[np.ndarray]) -> float:
        """Mean absolute pixel difference between consecutive frames (0–255)."""
        if len(frames) < 2:
            return 0.0
        diffs = []
        for i in range(1, len(frames)):
            prev = frames[i - 1].astype(np.float32)
            curr = frames[i].astype(np.float32)
            diffs.append(float(np.mean(np.abs(curr - prev))))
        return float(np.mean(diffs))

    def _compute_brightness(self, frames: List[np.ndarray]) -> float:
        """Mean pixel brightness across all frames (0–255)."""
        total = 0.0
        for frame in frames:
            total += float(np.mean(frame))
        return total / len(frames)

    def _compute_face_region_activity(self, frames: List[np.ndarray]) -> float:
        """
        Proxy for face presence: variance in the central vertical strip
        (middle 30% width × full height), normalised to 0–1.
        Higher variance → more detail / activity in the face region.
        """
        variances = []
        for frame in frames:
            h, w = frame.shape[:2]
            left  = int(w * 0.35)
            right = int(w * 0.65)
            strip = frame[:, left:right, :]
            variances.append(float(np.var(strip.astype(np.float32))))
        mean_var = float(np.mean(variances))
        # Normalise: typical indoor video variance ~500–3000
        return min(mean_var / 3000.0, 1.0)

    def _motion_description(self, motion: float) -> str:
        if motion < 1.0:
            return "very low motion"
        elif motion < 4.0:
            return "low motion"
        elif motion < 10.0:
            return "moderate movement"
        elif motion < 20.0:
            return "high motion"
        else:
            return "very high motion"

    def _brightness_description(self, brightness: float) -> str:
        if brightness < 50:
            return "dark"
        elif brightness < 120:
            return "dim"
        elif brightness < 180:
            return "well-lit"
        else:
            return "bright"

    def _face_activity_description(self, activity: float) -> str:
        if activity < 0.1:
            return "static background"
        elif activity < 0.3:
            return "person present"
        elif activity < 0.6:
            return "person speaking"
        else:
            return "person speaking with gestures"

    def get_model_info(self) -> Dict[str, str]:
        return {
            "type": "mock",
            "description": "Motion-heuristic video captioning (no model)",
            "local": "true",
        }


# ──────────────────────────────────────────────────────────────────────────────
# CoreML backend (Vision-Language model via coremltools)
# ──────────────────────────────────────────────────────────────────────────────

class CoreMLVideoMAE(BaseVideoMAE):
    """
    Video captioning using a CoreML Vision-Language model (.mlmodel).

    Requires:
        pip install coremltools

    The model must accept an image input (or a multi-frame input) and produce
    a text output.  Resize policy is configurable to match the model's
    expected input dimensions.

    Raises:
        ImportError       — if coremltools is not installed
        FileNotFoundError — if model_path does not exist
    """

    def __init__(
        self,
        model_path: str,
        input_width: int = 224,
        input_height: int = 224,
        max_frames_per_window: int = 8,
    ) -> None:
        """
        Load a CoreML model from disk.

        Args:
            model_path:            Path to the .mlmodel or .mlpackage file
            input_width:           Frame width expected by the model
            input_height:          Frame height expected by the model
            max_frames_per_window: Number of frames sampled per window
        """
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CoreML model not found: {model_path}")

        try:
            import coremltools as ct
        except ImportError:
            raise ImportError(
                "coremltools is required for CoreMLVideoMAE. "
                "Install with: pip install coremltools"
            )

        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self.max_frames = max_frames_per_window

        logger.info(f"Loading CoreML model from {model_path}")
        self.model = ct.models.MLModel(model_path)
        self._spec = self.model.get_spec()
        logger.info("CoreML VideoMAE model loaded successfully")

    def generate_caption(
        self,
        frames: List[np.ndarray],
        fps: float,
        window_start_s: float,
    ) -> str:
        if not frames:
            return "no visual content"

        # Sub-sample frames evenly across the window
        indices = np.linspace(0, len(frames) - 1, min(self.max_frames, len(frames)), dtype=int)
        sampled = [frames[i] for i in indices]

        captions = []
        for frame_bgr in sampled:
            try:
                caption_text = self._run_inference(frame_bgr)
                if caption_text:
                    captions.append(caption_text.strip())
            except Exception as exc:
                logger.warning(f"CoreML inference failed for frame: {exc}")

        if not captions:
            return "visual content unavailable"

        # Return the most common caption (simple majority for short lists)
        from collections import Counter
        return Counter(captions).most_common(1)[0][0]

    def _run_inference(self, frame_bgr: np.ndarray) -> str:
        """Resize frame and run CoreML inference, returning the output string."""
        import cv2

        # Convert BGR → RGB and resize to model input dimensions
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized   = cv2.resize(frame_rgb, (self.input_width, self.input_height))

        # Determine the input feature name from the model spec
        input_name = self._spec.description.input[0].name

        # Run prediction — output format depends on the specific model
        prediction = self.model.predict({input_name: resized})

        # Extract first string value from prediction dict
        for value in prediction.values():
            if isinstance(value, str):
                return value
            if isinstance(value, (list, np.ndarray)):
                return str(value[0]) if len(value) > 0 else ""

        return ""

    def get_model_info(self) -> Dict[str, str]:
        return {
            "type": "coreml",
            "model_path": self.model_path,
            "input_size": f"{self.input_width}x{self.input_height}",
            "local": "true",
            "runtime": "Apple Neural Engine / CoreML",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Facade
# ──────────────────────────────────────────────────────────────────────────────

class VideoMAE:
    """
    Unified VideoMAE interface with swappable backends.

    Usage::

        # Development / testing
        model = VideoMAE(backend="mock")

        # Production — requires a .mlmodel file
        model = VideoMAE(backend="coreml", model_path="/models/vlm.mlmodel")

        caption = model.generate_caption(frames, fps=30.0, window_start_s=0.0)
    """

    def __init__(
        self,
        backend: str = "mock",
        model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.backend = backend

        if backend == "mock":
            self._model: BaseVideoMAE = MockVideoMAE()
        elif backend == "coreml":
            if not model_path:
                raise ValueError("model_path is required for the coreml backend")
            self._model = CoreMLVideoMAE(model_path=model_path, **kwargs)
        else:
            raise ValueError(f"Unknown VideoMAE backend: '{backend}'")

        logger.info(f"VideoMAE initialised with '{backend}' backend")

    def generate_caption(
        self,
        frames: List[np.ndarray],
        fps: float,
        window_start_s: float,
    ) -> str:
        """Generate a caption for the given frame sequence."""
        return self._model.generate_caption(frames, fps, window_start_s)

    def get_model_info(self) -> Dict[str, str]:
        """Return backend metadata."""
        info = self._model.get_model_info()
        info["backend"] = self.backend
        return info