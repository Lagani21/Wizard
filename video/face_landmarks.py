"""
Face landmark detection utilities using Apple Vision framework (CoreML on-device).
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple

import Vision
import Quartz


def _frame_to_cgimage(frame_bgr: np.ndarray):
    """Convert OpenCV BGR frame to CGImage for Vision framework processing."""
    frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    h, w = frame_rgba.shape[:2]
    raw_bytes = frame_rgba.tobytes()
    provider = Quartz.CGDataProviderCreateWithData(None, raw_bytes, len(raw_bytes), None)
    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    cg_image = Quartz.CGImageCreate(
        w, h, 8, 32, w * 4, cs,
        Quartz.kCGBitmapByteOrderDefault | Quartz.kCGImageAlphaNoneSkipLast,
        provider, None, False, Quartz.kCGRenderingIntentDefault
    )
    return cg_image, raw_bytes


class FaceLandmarkDetector:
    """
    Face landmark detection using Apple Vision framework.

    Wraps VNDetectFaceLandmarksRequest and provides utilities for extracting
    eye, mouth, and other facial landmark regions for downstream analysis.
    """

    def __init__(self,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        self.max_num_faces = max_num_faces
        # Vision framework does not use separate tracking confidence;
        # these are stored for API compatibility only.
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def _run_request(self, frame_bgr: np.ndarray) -> list:
        """Run VNDetectFaceLandmarksRequest and return face observations."""
        cg_image, raw_bytes = _frame_to_cgimage(frame_bgr)

        results_holder = []

        def completion(req, err):
            if err is None and req.results():
                results_holder.extend(req.results())

        req = Vision.VNDetectFaceLandmarksRequest.alloc().initWithCompletionHandler_(completion)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})
        handler.performRequests_error_([req], None)

        del raw_bytes
        return results_holder[:self.max_num_faces]

    def detect_landmarks(self, frame: np.ndarray) -> Optional[dict]:
        """
        Detect face landmarks in a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Dictionary with 'face_obs' (VNFaceObservation) and helper accessors,
            or None if no face detected.
        """
        results = self._run_request(frame)
        if not results:
            return None

        face_obs = results[0]
        landmarks = face_obs.landmarks()
        if landmarks is None:
            return None

        return {
            'face_obs': face_obs,
            'landmarks': landmarks,
            'left_eye': landmarks.leftEye(),
            'right_eye': landmarks.rightEye(),
            'left_pupil': landmarks.leftPupil(),
            'right_pupil': landmarks.rightPupil(),
            'outer_lips': landmarks.outerLips(),
            'inner_lips': landmarks.innerLips(),
            'nose': landmarks.nose(),
            'bounding_box': face_obs.boundingBox(),
        }

    def get_eye_ear(self, eye_region) -> float:
        """
        Compute Eye Aspect Ratio from a VNFaceLandmarkRegion2D eye region.

        EAR = vertical_extent / horizontal_extent of the eye contour bounding box.
        Open eye ≈ 0.25-0.40, blink < ~0.15
        """
        count = eye_region.pointCount()
        if count < 3:
            return 1.0

        pts = eye_region.normalizedPoints()
        xs = [pts[i].x for i in range(count)]
        ys = [pts[i].y for i in range(count)]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        if width < 1e-6:
            return 1.0

        return height / width

    def get_region_points_normalized(self, region) -> Optional[np.ndarray]:
        """
        Extract normalised points from a VNFaceLandmarkRegion2D as a numpy array.

        Points are normalised to the face bounding box (origin bottom-left).

        Returns:
            Nx2 float32 array of (x, y) pairs, or None if region is None/empty.
        """
        if region is None or region.pointCount() == 0:
            return None

        count = region.pointCount()
        pts = region.normalizedPoints()
        points = np.array(
            [[pts[i].x, pts[i].y] for i in range(count)],
            dtype=np.float32
        )
        return points

    def draw_landmarks(self,
                       frame: np.ndarray,
                       landmark_data: dict,
                       regions: Optional[List[str]] = None) -> np.ndarray:
        """
        Draw landmark regions on a frame for visualisation.

        Args:
            frame: Input BGR frame
            landmark_data: Dict returned by detect_landmarks()
            regions: List of region names to draw (default: all)

        Returns:
            Annotated frame copy
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()

        region_names = regions or ['left_eye', 'right_eye', 'outer_lips',
                                   'inner_lips', 'nose']

        bb = landmark_data['bounding_box']
        # bounding_box is in normalised image coords (origin bottom-left)
        bb_x = bb.origin.x * w
        bb_y = (1.0 - bb.origin.y - bb.size.height) * h
        bb_w = bb.size.width * w
        bb_h = bb.size.height * h

        for name in region_names:
            region = landmark_data.get(name)
            if region is None:
                continue
            points = self.get_region_points_normalized(region)
            if points is None:
                continue

            for px, py in points:
                # Points are relative to face bounding box, origin bottom-left
                img_x = int(bb_x + px * bb_w)
                img_y = int(bb_y + (1.0 - py) * bb_h)
                cv2.circle(annotated, (img_x, img_y), 2, (0, 255, 0), -1)

        return annotated