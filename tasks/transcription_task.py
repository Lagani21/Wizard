"""
Transcription task for the WIZ Intelligence Pipeline.
"""

import numpy as np
try:
    # Try relative imports first
    from ..core.base_task import BaseTask
    from ..core.context import PipelineContext
    from ..models.whisper_model import WhisperModel
except ImportError:
    # Fall back to absolute imports
    from core.base_task import BaseTask
    from core.context import PipelineContext
    from models.whisper_model import WhisperModel


class TranscriptionTask(BaseTask):
    """
    Task for transcribing audio using Whisper.
    
    Processes audio waveform using WhisperModel and stores
    transcript words and segments in the pipeline context.
    """
    
    def __init__(self, whisper_model: WhisperModel) -> None:
        """
        Initialize the transcription task with a Whisper model.
        
        Args:
            whisper_model: Configured WhisperModel instance
        """
        super().__init__("Transcription")
        self.whisper_model = whisper_model
    
    def _run(self, context: PipelineContext) -> None:
        """
        Execute transcription on audio waveform.
        
        Args:
            context: Pipeline context containing audio waveform
        """
        logger = context.logger
        if context.audio_waveform is None:
            raise ValueError("Audio waveform not available in context")
        
        # Log audio information
        duration_s = len(context.audio_waveform) / 16000  # Assuming 16kHz
        logger.log_info(
            f"Processing audio: {len(context.audio_waveform)} samples, "
            f"{duration_s:.2f}s duration"
        )
        
        # Ensure model is loaded (failure is non-fatal — transcribe() has a placeholder fallback)
        if not self.whisper_model.is_loaded:
            logger.log_info("Loading Whisper model...")
            if not self.whisper_model.load_model():
                logger.log_warning("Whisper model failed to load — transcription will be unavailable")
        
        # Transcribe audio
        logger.log_info("Running Whisper transcription...")
        transcript_words, transcript_segments = self.whisper_model.transcribe(
            context.audio_waveform, 
            sample_rate=16000
        )
        
        # Store results in context
        context.transcript_words.extend(transcript_words)
        context.transcript_segments.extend(transcript_segments)
        
        # Store processing statistics
        stats = {
            'whisper_model_info': self.whisper_model.get_model_info(),
            'total_words': len(transcript_words),
            'total_segments': len(transcript_segments),
            'audio_duration_s': duration_s,
            'transcription_stats': self._calculate_transcription_stats(transcript_words, transcript_segments)
        }
        
        context.processing_metadata['transcription'] = stats
        
        # Log results
        total_text_length = sum(len(segment.text) for segment in transcript_segments)
        avg_confidence = (
            np.mean([word.confidence for word in transcript_words]) 
            if transcript_words else 0.0
        )
        
        logger.log_info(
            f"Transcription completed: {len(transcript_words)} words, "
            f"{len(transcript_segments)} segments, "
            f"{total_text_length} characters"
        )
        logger.log_info(f"Average word confidence: {avg_confidence:.3f}")
        
        # Log sample transcript
        if transcript_segments:
            sample_text = transcript_segments[0].text[:100] + "..." if len(transcript_segments[0].text) > 100 else transcript_segments[0].text
            logger.log_info(f"Sample transcript: \"{sample_text}\"")
    
    def _calculate_transcription_stats(self, words, segments) -> dict:
        """
        Calculate detailed transcription statistics.
        
        Args:
            words: List of transcript words
            segments: List of transcript segments
            
        Returns:
            Dictionary with transcription statistics
        """
        if not words and not segments:
            return {
                "words_per_minute": 0.0,
                "avg_word_duration": 0.0,
                "avg_segment_duration": 0.0,
                "confidence_distribution": {}
            }
        
        # Calculate words per minute
        if words:
            first_word_time = words[0].start_time
            last_word_time = words[-1].end_time
            duration_minutes = (last_word_time - first_word_time) / 60.0
            words_per_minute = len(words) / duration_minutes if duration_minutes > 0 else 0.0
        else:
            words_per_minute = 0.0
        
        # Calculate average durations
        avg_word_duration = (
            np.mean([word.end_time - word.start_time for word in words])
            if words else 0.0
        )
        
        avg_segment_duration = (
            np.mean([segment.end_time - segment.start_time for segment in segments])
            if segments else 0.0
        )
        
        # Calculate confidence distribution
        confidence_distribution = {}
        if words:
            confidences = [word.confidence for word in words]
            confidence_distribution = {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences)),
                "median": float(np.median(confidences))
            }
        
        return {
            "words_per_minute": words_per_minute,
            "avg_word_duration": float(avg_word_duration),
            "avg_segment_duration": float(avg_segment_duration),
            "confidence_distribution": confidence_distribution
        }
    
    @classmethod
    def create_default(cls, model_size: str = "base") -> 'TranscriptionTask':
        """
        Create a TranscriptionTask with default Whisper configuration.
        
        Args:
            model_size: Whisper model size to use
            
        Returns:
            Configured TranscriptionTask instance
        """
        whisper_model = WhisperModel(
            model_size=model_size,
            language=None,  # Auto-detect
            device=None     # Auto-detect
        )
        return cls(whisper_model)