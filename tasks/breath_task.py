"""
Breath detection task for the WIZ Intelligence Pipeline.
"""

import numpy as np
from ..core.base_task import BaseTask
from ..core.context import PipelineContext
from ..models.breath_detector import BreathDetector


class BreathTask(BaseTask):
    """
    Task for detecting breath events in audio waveform.
    
    Processes audio using BreathDetector and stores
    results in the pipeline context.
    """
    
    def __init__(self, breath_detector: BreathDetector) -> None:
        """
        Initialize the breath task with a breath detector.
        
        Args:
            breath_detector: Configured BreathDetector instance
        """
        super().__init__("BreathDetection")
        self.breath_detector = breath_detector
    
    def _validate_audio_format(self, audio: np.ndarray) -> None:
        """
        Validate that audio is in the expected format.
        
        Args:
            audio: Audio waveform to validate
            
        Raises:
            ValueError: If audio format is invalid
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio (1D array), got {audio.ndim}D array")
        
        if len(audio) == 0:
            raise ValueError("Audio waveform is empty")
        
        # Check for reasonable sample values (assuming normalized [-1, 1] or int16 range)
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            self.logger.warning("Audio appears to be silent (all zeros)")
        elif max_val > 32768:  # Likely float values outside [-1, 1]
            self.logger.warning(f"Audio values seem unnormalized (max={max_val})")
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for breath detection.
        
        Args:
            audio: Raw audio waveform
            
        Returns:
            Preprocessed audio waveform
        """
        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                # Convert int16 to float32 normalized to [-1, 1]
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)
        
        # Normalize if values are outside reasonable range
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
            self.logger.info(f"Normalized audio by factor of {max_val}")
        
        return audio
    
    def _run(self, context: PipelineContext) -> None:
        """
        Execute breath detection on audio waveform.
        
        Args:
            context: Pipeline context containing audio waveform
        """
        if context.audio_waveform is None:
            raise ValueError("Audio waveform not available in context")
        
        # Validate and preprocess audio
        self._validate_audio_format(context.audio_waveform)
        processed_audio = self._preprocess_audio(context.audio_waveform.copy())
        
        # Log audio information
        duration_s = len(processed_audio) / self.breath_detector.sample_rate
        self.logger.info(
            f"Processing audio: {len(processed_audio)} samples, "
            f"{duration_s:.2f}s duration at {self.breath_detector.sample_rate}Hz"
        )
        
        # Detect breath events
        breath_events = self.breath_detector.detect_breath_events(processed_audio)
        
        # Store results in context
        context.breath_events.extend(breath_events)
        
        # Generate and store statistics
        stats = self.breath_detector.get_statistics(processed_audio, breath_events)
        context.processing_metadata['breath_detection'] = {
            'detector_stats': stats,
            'total_events': len(breath_events),
            'audio_duration_s': duration_s,
            'sample_rate': self.breath_detector.sample_rate,
            'preprocessing': {
                'original_dtype': str(context.audio_waveform.dtype),
                'processed_dtype': str(processed_audio.dtype),
                'original_max': float(np.max(np.abs(context.audio_waveform))),
                'processed_max': float(np.max(np.abs(processed_audio)))
            }
        }
        
        # Log results
        self.logger.info(f"Breath detection completed: {len(breath_events)} events detected")
        if breath_events:
            avg_confidence = np.mean([event.confidence for event in breath_events])
            avg_duration = np.mean([event.duration_ms for event in breath_events])
            self.logger.info(
                f"Average confidence: {avg_confidence:.3f}, "
                f"Average duration: {avg_duration:.1f}ms"
            )
    
    @classmethod
    def create_default(cls) -> 'BreathTask':
        """
        Create a BreathTask with default detector configuration.
        
        Returns:
            Configured BreathTask instance
        """
        detector = BreathDetector(
            sample_rate=16000,
            min_duration_ms=200.0,
            max_duration_ms=800.0,
            energy_min_threshold=0.01,
            energy_max_threshold=0.1
        )
        return cls(detector)