"""
Blink detection implementation using Apple Vision framework (CoreML on-device).
"""

import cv2
import gc
import numpy as np
from typing import List, Optional

import Vision
import Quartz

try:
    from ..core.context import BlinkEvent
except ImportError:
    from core.context import BlinkEvent


def _frame_to_cgimage(frame_bgr: np.ndarray):
    """Convert OpenCV BGR frame to CGImage for Vision framework processing."""
    frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    h, w = frame_rgba.shape[:2]
    raw_bytes = frame_rgba.tobytes()  # Keep bytes alive in caller scope
    provider = Quartz.CGDataProviderCreateWithData(None, raw_bytes, len(raw_bytes), None)
    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    cg_image = Quartz.CGImageCreate(
        w, h, 8, 32, w * 4, cs,
        Quartz.kCGBitmapByteOrderDefault | Quartz.kCGImageAlphaNoneSkipLast,
        provider, None, False, Quartz.kCGRenderingIntentDefault
    )
    return cg_image, raw_bytes  # Return raw_bytes to keep it alive


def _compute_eye_ear(eye_region) -> float:
    """
    Compute Eye Aspect Ratio from a VNFaceLandmarkRegion2D eye region.

    Uses the bounding-box of the normalised eye contour points:
        EAR = vertical_extent / horizontal_extent

    Open eye  ≈ 0.25-0.40
    Blink     < ~0.15
    """
    count = eye_region.pointCount()
    if count < 3:
        return 1.0  # No reliable data — treat as open

    pts = eye_region.normalizedPoints()
    xs = [pts[i].x for i in range(count)]
    ys = [pts[i].y for i in range(count)]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    if width < 1e-6:
        return 1.0

    return height / width


class BlinkDetector:
    """
    Detects blink events from video frames using Apple Vision framework.

    Runs VNDetectFaceLandmarksRequest on each frame (CoreML / Neural Engine),
    extracts left/right eye contour points, and computes Eye Aspect Ratio (EAR)
    to identify blink events.
    """

    def __init__(self,
                 ear_threshold: float = 0.15,
                 consecutive_frames: int = 2,
                 min_blink_duration_ms: float = 50.0,
                 max_blink_duration_ms: float = 500.0) -> None:
        """
        Args:
            ear_threshold: EAR below which a blink is detected (bounding-box metric)
            consecutive_frames: Minimum consecutive below-threshold frames
            min_blink_duration_ms: Minimum blink duration in ms
            max_blink_duration_ms: Maximum blink duration in ms
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.min_blink_duration_ms = min_blink_duration_ms
        self.max_blink_duration_ms = max_blink_duration_ms

        # State tracking
        self.frame_counter = 0
        self.blink_counter = 0
        self.is_blinking = False
        self.blink_start_frame = None

    def _run_vision_request(self, frame_bgr: np.ndarray) -> list:
        """Run VNDetectFaceLandmarksRequest and return VNFaceObservation results."""
        cg_image, raw_bytes = _frame_to_cgimage(frame_bgr)

        results_holder = []

        def completion(req, err):
            if err is None and req.results():
                results_holder.extend(req.results())

        req = Vision.VNDetectFaceLandmarksRequest.alloc().initWithCompletionHandler_(completion)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})
        handler.performRequests_error_([req], None)

        # raw_bytes stays alive until here
        del raw_bytes
        return results_holder

    def detect_landmarks(self, frame_bgr: np.ndarray) -> Optional[tuple]:
        """
        Detect eye EARs in a frame.

        Returns:
            (left_ear, right_ear) tuple, or None if no face detected.
        """
        results = self._run_vision_request(frame_bgr)

        if not results:
            return None

        face_obs = results[0]
        landmarks = face_obs.landmarks()

        if landmarks is None:
            return None

        left_eye = landmarks.leftEye()
        right_eye = landmarks.rightEye()

        if left_eye is None or right_eye is None:
            return None

        return _compute_eye_ear(left_eye), _compute_eye_ear(right_eye)

    def process_frame(self, frame: np.ndarray, fps: float) -> Optional[BlinkEvent]:
        """
        Process a single frame and return a BlinkEvent if a blink completed.
        """
        self.frame_counter += 1

        ear_result = self.detect_landmarks(frame)

        if ear_result is None:
            if self.is_blinking:
                self.is_blinking = False
                self.blink_start_frame = None
            return None

        left_ear, right_ear = ear_result
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < self.ear_threshold:
            if not self.is_blinking:
                self.is_blinking = True
                self.blink_start_frame = self.frame_counter
        else:
            if self.is_blinking:
                self.is_blinking = False

                if self.blink_start_frame is not None:
                    blink_frames = self.frame_counter - self.blink_start_frame
                    duration_ms = (blink_frames / fps) * 1000

                    if (blink_frames >= self.consecutive_frames and
                            self.min_blink_duration_ms <= duration_ms <= self.max_blink_duration_ms):

                        blink_event = BlinkEvent(
                            start_frame=self.blink_start_frame,
                            end_frame=self.frame_counter,
                            duration_ms=duration_ms,
                            confidence=1.0 - avg_ear
                        )
                        self.blink_counter += 1
                        self.blink_start_frame = None
                        return blink_event

                    self.blink_start_frame = None

        return None

    def process_video_frames(self, frames: List[np.ndarray], fps: float) -> List[BlinkEvent]:
        """Process multiple frames and return all detected blink events."""
        blink_events = []
        for frame in frames:
            blink_event = self.process_frame(frame, fps)
            if blink_event is not None:
                blink_events.append(blink_event)
        return blink_events

    def reset(self) -> None:
        """Reset detector state for a new video."""
        self.frame_counter = 0
        self.blink_counter = 0
        self.is_blinking = False
        self.blink_start_frame = None

    def get_statistics(self) -> dict:
        return {
            "total_frames_processed": self.frame_counter,
            "total_blinks_detected": self.blink_counter,
            "current_state": "blinking" if self.is_blinking else "not_blinking"
        }