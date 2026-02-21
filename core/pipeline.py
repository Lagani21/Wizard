"""
Main pipeline for the WIZ Intelligence Pipeline.
"""

import os
import logging
import cv2
import numpy as np
from typing import List, Optional
from .context import PipelineContext, VideoMetadata
from .base_task import BaseTask
from ..tasks.blink_task import BlinkTask
from ..tasks.breath_task import BreathTask
from ..tasks.transcription_task import TranscriptionTask
from ..tasks.diarization_task import DiarizationTask
from ..tasks.alignment_task import AlignmentTask
from ..audio.audio_extractor import AudioExtractor


class Pipeline:
    """
    Main pipeline coordinator for WIZ Intelligence Processing.
    
    Orchestrates the execution of detection and speech processing tasks
    in a deterministic, object-oriented manner.
    """
    
    def __init__(self, 
                 blink_task: Optional[BlinkTask] = None,
                 breath_task: Optional[BreathTask] = None,
                 transcription_task: Optional[TranscriptionTask] = None,
                 diarization_task: Optional[DiarizationTask] = None,
                 alignment_task: Optional[AlignmentTask] = None,
                 enable_speech_processing: bool = False) -> None:
        """
        Initialize the pipeline with optional task instances.
        
        Args:
            blink_task: Configured BlinkTask instance (creates default if None)
            breath_task: Configured BreathTask instance (creates default if None)
            transcription_task: Configured TranscriptionTask instance (creates default if None)
            diarization_task: Configured DiarizationTask instance (creates default if None)
            alignment_task: Configured AlignmentTask instance (creates default if None)
            enable_speech_processing: Enable speech processing tasks
        """
        self.logger = logging.getLogger("wiz.pipeline")
        self.enable_speech_processing = enable_speech_processing
        
        # Initialize core tasks with defaults if not provided
        self.blink_task = blink_task if blink_task is not None else BlinkTask.create_default()
        self.breath_task = breath_task if breath_task is not None else BreathTask.create_default()
        
        # Initialize speech processing tasks if enabled
        self.transcription_task = None
        self.diarization_task = None
        self.alignment_task = None
        
        if enable_speech_processing:
            self.transcription_task = (
                transcription_task if transcription_task is not None 
                else TranscriptionTask.create_default()
            )
            self.diarization_task = (
                diarization_task if diarization_task is not None 
                else DiarizationTask.create_default()
            )
            self.alignment_task = (
                alignment_task if alignment_task is not None 
                else AlignmentTask.create_default()
            )
        
        # Initialize audio extractor
        self.audio_extractor = AudioExtractor()
        
        # Build task execution order
        self.tasks: List[BaseTask] = [
            self.blink_task,
            self.breath_task
        ]
        
        if enable_speech_processing:
            self.tasks.extend([
                self.transcription_task,
                self.diarization_task,
                self.alignment_task
            ])
        
        processing_types = ["blink detection", "breath detection"]
        if enable_speech_processing:
            processing_types.extend(["transcription", "speaker diarization", "alignment"])
        
        self.logger.info(f"WIZ Intelligence Pipeline initialized with: {', '.join(processing_types)}")
    
    def _validate_video_path(self, video_path: str) -> None:
        """
        Validate the input video path.
        
        Args:
            video_path: Path to video file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If file is not a valid video format
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not os.path.isfile(video_path):
            raise ValueError(f"Path is not a file: {video_path}")
        
        # Check file extension (basic validation)
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        _, ext = os.path.splitext(video_path.lower())
        
        if ext not in valid_extensions:
            self.logger.warning(f"Uncommon video extension: {ext}")
    
    def _extract_video_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoMetadata object with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps if fps > 0 else 0
            
            metadata = VideoMetadata(
                path=os.path.abspath(video_path),
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration_seconds
            )
            
            self.logger.info(
                f"Video metadata: {width}x{height}, {fps:.2f}fps, "
                f"{total_frames} frames, {duration_seconds:.2f}s"
            )
            
            return metadata
            
        finally:
            cap.release()
    
    def _extract_audio_from_video(self, video_path: str) -> np.ndarray:
        """
        Extract mono audio from video file at 16kHz.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Mono audio waveform at 16kHz
        """
        return self.audio_extractor.extract_audio_from_video(video_path)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def run(self, video_path: str) -> PipelineContext:
        """
        Main entry point for the WIZ Intelligence Pipeline.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            PipelineContext containing all detection results
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video file is invalid or tasks fail
        """
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Starting WIZ Intelligence Pipeline for: {video_path}")
        
        # Validate input
        self._validate_video_path(video_path)
        
        # Initialize context
        context = PipelineContext()
        
        try:
            # Extract video metadata
            self.logger.info("Extracting video metadata...")
            context.video_metadata = self._extract_video_metadata(video_path)
            
            # Extract audio
            self.logger.info("Extracting audio from video...")
            context.audio_waveform = self._extract_audio_from_video(video_path)
            
            # Execute tasks in sequence
            for task in self.tasks:
                self.logger.info(f"Executing task: {task.task_name}")
                task.execute(context)
            
            # Log final results
            self._log_pipeline_results(context)
            
            self.logger.info("WIZ Intelligence Pipeline completed successfully")
            return context
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def _log_pipeline_results(self, context: PipelineContext) -> None:
        """
        Log the final pipeline results.
        
        Args:
            context: Pipeline context with results
        """
        self.logger.info("=== PIPELINE RESULTS ===")
        self.logger.info(f"Blink events detected: {len(context.blink_events)}")
        self.logger.info(f"Breath events detected: {len(context.breath_events)}")
        
        if context.blink_events:
            total_blink_time = sum(event.duration_ms for event in context.blink_events)
            avg_blink_confidence = np.mean([event.confidence for event in context.blink_events])
            self.logger.info(f"Total blink duration: {total_blink_time:.1f}ms")
            self.logger.info(f"Average blink confidence: {avg_blink_confidence:.3f}")
        
        if context.breath_events:
            total_breath_time = sum(event.duration_ms for event in context.breath_events)
            avg_breath_confidence = np.mean([event.confidence for event in context.breath_events])
            self.logger.info(f"Total breath duration: {total_breath_time:.1f}ms")
            self.logger.info(f"Average breath confidence: {avg_breath_confidence:.3f}")
        
        self.logger.info("========================")
    
    def get_results_summary(self, context: PipelineContext) -> dict:
        """
        Get a summary of pipeline results.
        
        Args:
            context: Pipeline context with results
            
        Returns:
            Dictionary with results summary
        """
        summary = {
            'video_metadata': {
                'path': context.video_metadata.path if context.video_metadata else None,
                'duration_s': context.video_metadata.duration_seconds if context.video_metadata else 0,
                'fps': context.video_metadata.fps if context.video_metadata else 0,
                'resolution': f"{context.video_metadata.width}x{context.video_metadata.height}" if context.video_metadata else None
            },
            'blink_detection': {
                'total_events': len(context.blink_events),
                'events': [
                    {
                        'start_frame': event.start_frame,
                        'end_frame': event.end_frame,
                        'duration_ms': event.duration_ms,
                        'confidence': event.confidence
                    }
                    for event in context.blink_events
                ]
            },
            'breath_detection': {
                'total_events': len(context.breath_events),
                'events': [
                    {
                        'start_time': event.start_time,
                        'end_time': event.end_time,
                        'duration_ms': event.duration_ms,
                        'confidence': event.confidence
                    }
                    for event in context.breath_events
                ]
            },
            'processing_metadata': context.processing_metadata
        }
        
        # Add speech processing results if available
        if self.enable_speech_processing:
            summary['speech_processing'] = {
                'transcription': {
                    'total_words': len(context.transcript_words),
                    'total_segments': len(context.transcript_segments),
                    'words': [
                        {
                            'text': word.text,
                            'start_time': word.start_time,
                            'end_time': word.end_time,
                            'confidence': word.confidence
                        }
                        for word in context.transcript_words
                    ],
                    'segments': [
                        {
                            'text': segment.text,
                            'start_time': segment.start_time,
                            'end_time': segment.end_time,
                            'word_count': len(segment.words)
                        }
                        for segment in context.transcript_segments
                    ]
                },
                'diarization': {
                    'total_segments': len(context.speaker_segments),
                    'num_speakers': len(set(seg.speaker_id for seg in context.speaker_segments)),
                    'segments': [
                        {
                            'speaker_id': segment.speaker_id,
                            'start_time': segment.start_time,
                            'end_time': segment.end_time
                        }
                        for segment in context.speaker_segments
                    ]
                },
                'alignment': {
                    'total_segments': len(context.aligned_segments),
                    'segments': [
                        {
                            'speaker_id': segment.speaker_id,
                            'text': segment.text,
                            'start_time': segment.start_time,
                            'end_time': segment.end_time,
                            'word_count': len(segment.words)
                        }
                        for segment in context.aligned_segments
                    ]
                }
            }
        
        return summary