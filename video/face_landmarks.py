"""
Face landmark detection utilities.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple


class FaceLandmarkDetector:
    """
    Face landmark detection using MediaPipe Face Mesh.
    
    Provides utilities for detecting and processing face landmarks
    for various facial analysis tasks.
    """
    
    def __init__(self, 
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        """
        Initialize the face landmark detector.
        
        Args:
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define key landmark indices
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # EAR calculation points (6 points per eye)
        self.left_ear_indices = [33, 160, 158, 133, 153, 144]
        self.right_ear_indices = [362, 385, 387, 263, 373, 380]
        
        # Mouth landmarks
        self.mouth_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318]
    
    def detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face landmarks in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Array of normalized landmarks or None if no face detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            return landmarks
        
        return None
    
    def get_eye_landmarks(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract eye landmarks from full face landmarks.
        
        Args:
            landmarks: Full face landmarks array
            
        Returns:
            Tuple of (left_eye_landmarks, right_eye_landmarks)
        """
        left_eye = landmarks[self.left_eye_indices]
        right_eye = landmarks[self.right_eye_indices]
        return left_eye, right_eye
    
    def get_ear_landmarks(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get landmarks for EAR calculation.
        
        Args:
            landmarks: Full face landmarks array
            
        Returns:
            Tuple of (left_ear_points, right_ear_points)
        """
        left_ear_points = landmarks[self.left_ear_indices]
        right_ear_points = landmarks[self.right_ear_indices]
        return left_ear_points, right_ear_points
    
    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio from 6 eye landmarks.
        
        Args:
            eye_landmarks: Array of 6 eye landmarks for EAR calculation
            
        Returns:
            Eye Aspect Ratio value
        """
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR calculation
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def get_mouth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract mouth landmarks from full face landmarks.
        
        Args:
            landmarks: Full face landmarks array
            
        Returns:
            Mouth landmarks array
        """
        return landmarks[self.mouth_indices]
    
    def draw_landmarks(self, 
                      frame: np.ndarray, 
                      landmarks: np.ndarray,
                      indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Draw landmarks on frame for visualization.
        
        Args:
            frame: Input frame
            landmarks: Face landmarks
            indices: Specific landmark indices to draw (None for all)
            
        Returns:
            Frame with landmarks drawn
        """
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()
        
        if indices is None:
            indices = range(len(landmarks))
        
        for idx in indices:
            if idx < len(landmarks):
                x = int(landmarks[idx][0] * w)
                y = int(landmarks[idx][1] * h)
                cv2.circle(annotated_frame, (x, y), 2, (0, 255, 0), -1)
        
        return annotated_frame