"""
Local Whisper transcription model for the WIZ Intelligence Pipeline.
"""

import logging
import numpy as np
from typing import List, Optional, Union
from ..core.context import TranscriptWord, TranscriptSegment


class WhisperModel:
    """
    Local Whisper transcription model.
    
    Provides speech-to-text transcription with word-level timestamps
    using OpenAI's Whisper model running locally on CPU/Metal.
    """
    
    def __init__(self, 
                 model_size: str = "base",
                 language: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        """
        Initialize the Whisper model.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            language: Language code (None for auto-detection)
            device: Device to use (None for auto-detection)
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self.model = None
        self.is_loaded = False
        
        self.logger = logging.getLogger("wiz.models.whisper")
        
        # Validate model size
        valid_sizes = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if model_size not in valid_sizes:
            raise ValueError(f"Invalid model size: {model_size}. Must be one of {valid_sizes}")
    
    def load_model(self) -> bool:
        """
        Load the Whisper model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.is_loaded:
            return True
        
        try:
            import whisper
            
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Load model with appropriate device
            device = self._determine_device()
            self.model = whisper.load_model(self.model_size, device=device)
            
            self.is_loaded = True
            self.logger.info(f"Whisper model loaded successfully on {device}")
            return True
            
        except ImportError:
            self.logger.error("whisper library not installed. Install with: pip install openai-whisper")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            return False
    
    def _determine_device(self) -> str:
        """
        Determine the best device to use for Whisper.
        
        Returns:
            Device string ("cpu" or "mps" for Apple Silicon)
        """
        if self.device:
            return self.device
        
        try:
            import torch
            
            # Check for Apple Silicon Metal Performance Shaders
            if torch.backends.mps.is_available():
                self.logger.info("Using Metal Performance Shaders (MPS) for Whisper")
                return "mps"
            
        except ImportError:
            pass
        
        # Fallback to CPU
        self.logger.info("Using CPU for Whisper")
        return "cpu"
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[List[TranscriptWord], List[TranscriptSegment]]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio waveform (mono, float32)
            sample_rate: Audio sample rate (should be 16000)
            
        Returns:
            Tuple of (transcript_words, transcript_segments)
        """
        if not self.is_loaded:
            if not self.load_model():
                return self._create_fallback_transcription(audio, sample_rate)
        
        try:
            # Ensure audio is the right format
            if sample_rate != 16000:
                self.logger.warning(f"Expected 16kHz audio, got {sample_rate}Hz")
            
            # Whisper expects float32 audio normalized to [-1, 1]
            audio_normalized = self._normalize_audio(audio)
            
            # Transcribe with word timestamps
            self.logger.info(f"Transcribing {len(audio_normalized)/sample_rate:.1f}s of audio")
            
            result = self.model.transcribe(
                audio_normalized,
                language=self.language,
                word_timestamps=True,
                verbose=False
            )
            
            # Extract words and segments
            words = self._extract_words(result)
            segments = self._extract_segments(result)
            
            self.logger.info(f"Transcription complete: {len(words)} words, {len(segments)} segments")
            
            return words, segments
            
        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {e}")
            return self._create_fallback_transcription(audio, sample_rate)
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio for Whisper processing.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize to [-1, 1] if needed
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
            self.logger.debug(f"Normalized audio by factor of {max_val}")
        
        return audio
    
    def _extract_words(self, whisper_result: dict) -> List[TranscriptWord]:
        """
        Extract words from Whisper result.
        
        Args:
            whisper_result: Whisper transcription result
            
        Returns:
            List of TranscriptWord objects
        """
        words = []
        
        for segment in whisper_result.get("segments", []):
            segment_words = segment.get("words", [])
            
            for word_info in segment_words:
                word = TranscriptWord(
                    text=word_info.get("word", "").strip(),
                    start_time=word_info.get("start", 0.0),
                    end_time=word_info.get("end", 0.0),
                    confidence=word_info.get("probability", 0.0)
                )
                
                if word.text:  # Only add non-empty words
                    words.append(word)
        
        return words
    
    def _extract_segments(self, whisper_result: dict) -> List[TranscriptSegment]:
        """
        Extract segments from Whisper result.
        
        Args:
            whisper_result: Whisper transcription result
            
        Returns:
            List of TranscriptSegment objects
        """
        segments = []
        
        for segment_info in whisper_result.get("segments", []):
            # Extract words for this segment
            segment_words = []
            for word_info in segment_info.get("words", []):
                word = TranscriptWord(
                    text=word_info.get("word", "").strip(),
                    start_time=word_info.get("start", 0.0),
                    end_time=word_info.get("end", 0.0),
                    confidence=word_info.get("probability", 0.0)
                )
                if word.text:
                    segment_words.append(word)
            
            segment = TranscriptSegment(
                text=segment_info.get("text", "").strip(),
                start_time=segment_info.get("start", 0.0),
                end_time=segment_info.get("end", 0.0),
                words=segment_words
            )
            
            if segment.text:  # Only add non-empty segments
                segments.append(segment)
        
        return segments
    
    def _create_fallback_transcription(self, audio: np.ndarray, sample_rate: int) -> tuple[List[TranscriptWord], List[TranscriptSegment]]:
        """
        Create fallback transcription when Whisper fails.
        
        Args:
            audio: Audio waveform
            sample_rate: Audio sample rate
            
        Returns:
            Tuple of empty transcript words and segments
        """
        duration = len(audio) / sample_rate
        
        # Create a placeholder segment
        placeholder_segment = TranscriptSegment(
            text="[Transcription unavailable]",
            start_time=0.0,
            end_time=duration,
            words=[]
        )
        
        self.logger.warning(f"Using fallback transcription for {duration:.1f}s audio")
        
        return [], [placeholder_segment]
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_size": self.model_size,
            "language": self.language,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "backend": "openai-whisper"
        }
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.is_loaded = False
        self.logger.info("Whisper model unloaded")
    
    def estimate_memory_usage(self) -> str:
        """
        Estimate memory usage for the model.
        
        Returns:
            Estimated memory usage string
        """
        # Rough estimates for different model sizes
        memory_estimates = {
            "tiny": "39 MB",
            "base": "74 MB", 
            "small": "244 MB",
            "medium": "769 MB",
            "large": "1550 MB",
            "large-v2": "1550 MB",
            "large-v3": "1550 MB"
        }
        
        return memory_estimates.get(self.model_size, "Unknown")