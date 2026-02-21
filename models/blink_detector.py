"""
Blink detection implementation using MediaPipe Face Mesh.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import mediapipe as mp
from ..core.context import BlinkEvent


class BlinkDetector:
    """
    Detects blink events from video frames using face landmark detection.
    
    Uses MediaPipe Face Mesh to detect face landmarks and computes
    eye aspect ratio (EAR) to identify blink events.
    """
    
    # Eye landmark indices for MediaPipe Face Mesh
    LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # EAR calculation landmarks (6 points per eye)
    LEFT_EYE_EAR_LANDMARKS = [33, 160, 158, 133, 153, 144]  # Outer, top, bottom corners
    RIGHT_EYE_EAR_LANDMARKS = [362, 385, 387, 263, 373, 380]
    
    def __init__(self, 
                 ear_threshold: float = 0.25, 
                 consecutive_frames: int = 2,
                 min_blink_duration_ms: float = 50.0,
                 max_blink_duration_ms: float = 500.0) -> None:
        """
        Initialize the blink detector.
        
        Args:
            ear_threshold: EAR threshold below which a blink is detected
            consecutive_frames: Minimum consecutive frames for blink detection
            min_blink_duration_ms: Minimum blink duration in milliseconds
            max_blink_duration_ms: Maximum blink duration in milliseconds
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.min_blink_duration_ms = min_blink_duration_ms
        self.max_blink_duration_ms = max_blink_duration_ms
        
        # MediaPipe Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # State tracking
        self.frame_counter = 0
        self.blink_counter = 0
        self.is_blinking = False
        self.blink_start_frame = None
        
    def calculate_ear(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for given eye landmarks.
        
        Args:
            landmarks: Array of normalized face landmarks
            eye_indices: Indices of eye landmarks for EAR calculation
            
        Returns:
            Eye aspect ratio value
        """
        # Get eye landmarks
        eye_points = landmarks[eye_indices]
        
        # Calculate distances
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])  # Top to bottom
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])  # Top to bottom
        
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])   # Left to right
        
        # EAR calculation
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face landmarks in the given frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Array of normalized landmarks or None if no face detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Get the first (primary) face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            return landmarks
        
        return None
    
    def process_frame(self, frame: np.ndarray, fps: float) -> Optional[BlinkEvent]:
        """
        Process a single frame and detect blink events.
        
        Args:
            frame: Input video frame
            fps: Video frame rate for timing calculations
            
        Returns:
            BlinkEvent if a blink is completed, None otherwise
        """
        self.frame_counter += 1
        
        # Detect face landmarks
        landmarks = self.detect_landmarks(frame)
        
        if landmarks is None:
            # No face detected - reset blink state
            if self.is_blinking:
                self.is_blinking = False
                self.blink_start_frame = None
            return None
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_EAR_LANDMARKS)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_EAR_LANDMARKS)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Blink detection logic
        if avg_ear < self.ear_threshold:
            if not self.is_blinking:
                # Start of potential blink
                self.is_blinking = True
                self.blink_start_frame = self.frame_counter
        else:
            if self.is_blinking:
                # End of blink
                self.is_blinking = False
                
                if self.blink_start_frame is not None:
                    # Calculate blink duration
                    blink_frames = self.frame_counter - self.blink_start_frame
                    duration_ms = (blink_frames / fps) * 1000
                    
                    # Validate blink duration
                    if (blink_frames >= self.consecutive_frames and 
                        self.min_blink_duration_ms <= duration_ms <= self.max_blink_duration_ms):
                        
                        # Create blink event
                        blink_event = BlinkEvent(
                            start_frame=self.blink_start_frame,
                            end_frame=self.frame_counter,
                            duration_ms=duration_ms,
                            confidence=1.0 - avg_ear  # Simple confidence measure
                        )
                        
                        self.blink_counter += 1
                        self.blink_start_frame = None
                        return blink_event
                
                self.blink_start_frame = None
        
        return None
    
    def process_video_frames(self, frames: List[np.ndarray], fps: float) -> List[BlinkEvent]:
        """
        Process multiple video frames and detect all blink events.
        
        Args:
            frames: List of video frames
            fps: Video frame rate
            
        Returns:
            List of detected blink events
        """
        blink_events = []
        
        for frame in frames:
            blink_event = self.process_frame(frame, fps)
            if blink_event is not None:
                blink_events.append(blink_event)
        
        return blink_events
    
    def reset(self) -> None:
        """Reset the detector state for processing a new video."""
        self.frame_counter = 0
        self.blink_counter = 0
        self.is_blinking = False
        self.blink_start_frame = None
    
    def get_statistics(self) -> dict:
        """Get detection statistics."""
        return {
            "total_frames_processed": self.frame_counter,
            "total_blinks_detected": self.blink_counter,
            "current_state": "blinking" if self.is_blinking else "not_blinking"
        }