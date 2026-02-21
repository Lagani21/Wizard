"""
Blink detection task for the WIZ Intelligence Pipeline.
"""

import cv2
import numpy as np
from typing import List
from ..core.base_task import BaseTask
from ..core.context import PipelineContext
from ..models.blink_detector import BlinkDetector


class BlinkTask(BaseTask):
    """
    Task for detecting blink events in video frames.
    
    Processes video frames using BlinkDetector and stores
    results in the pipeline context.
    """
    
    def __init__(self, blink_detector: BlinkDetector) -> None:
        """
        Initialize the blink task with a blink detector.
        
        Args:
            blink_detector: Configured BlinkDetector instance
        """
        super().__init__("BlinkDetection")
        self.blink_detector = blink_detector
    
    def _extract_frames_from_video(self, video_path: str) -> tuple[List[np.ndarray], float]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames, fps)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()
        
        self.logger.info(f"Extracted {len(frames)} frames at {fps} FPS from {video_path}")
        return frames, fps
    
    def _run(self, context: PipelineContext) -> None:
        """
        Execute blink detection on video frames.
        
        Args:
            context: Pipeline context containing video metadata
        """
        if context.video_metadata is None:
            raise ValueError("Video metadata not available in context")
        
        # Reset detector state for new video
        self.blink_detector.reset()
        
        # Extract frames from video
        frames, fps = self._extract_frames_from_video(context.video_metadata.path)
        
        # Validate FPS matches metadata
        if abs(fps - context.video_metadata.fps) > 0.1:
            self.logger.warning(
                f"FPS mismatch: metadata={context.video_metadata.fps}, "
                f"extracted={fps}. Using extracted FPS."
            )
        
        # Process frames for blink detection
        self.logger.info(f"Processing {len(frames)} frames for blink detection")
        
        blink_events = []
        for frame_idx, frame in enumerate(frames):
            blink_event = self.blink_detector.process_frame(frame, fps)
            if blink_event is not None:
                blink_events.append(blink_event)
                self.logger.debug(f"Blink detected: frame {blink_event.start_frame}-{blink_event.end_frame}")
        
        # Store results in context
        context.blink_events.extend(blink_events)
        
        # Store processing statistics
        stats = self.blink_detector.get_statistics()
        context.processing_metadata['blink_detection'] = {
            'detector_stats': stats,
            'total_events': len(blink_events),
            'processed_frames': len(frames),
            'fps_used': fps
        }
        
        self.logger.info(f"Blink detection completed: {len(blink_events)} events detected")
    
    @classmethod
    def create_default(cls) -> 'BlinkTask':
        """
        Create a BlinkTask with default detector configuration.
        
        Returns:
            Configured BlinkTask instance
        """
        detector = BlinkDetector(
            ear_threshold=0.25,
            consecutive_frames=2,
            min_blink_duration_ms=50.0,
            max_blink_duration_ms=500.0
        )
        return cls(detector)