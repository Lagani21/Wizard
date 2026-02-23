"""
Breath detection implementation using simple heuristic analysis.
"""

import gc
import numpy as np
from typing import List
from scipy import signal
try:
    # Try relative import first
    from ..core.context import BreathEvent
except ImportError:
    # Fall back to absolute import
    from core.context import BreathEvent


class BreathDetector:
    """
    Detects breath events from audio using simple heuristic analysis.
    
    Analyzes audio segments directly for breath-like characteristics
    based on energy, duration, and spectral properties.
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 min_duration_ms: float = 200.0,
                 max_duration_ms: float = 800.0,
                 energy_min_threshold: float = 0.01,
                 energy_max_threshold: float = 0.1,
                 window_length_ms: float = 50.0) -> None:
        """
        Initialize the breath detector.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            min_duration_ms: Minimum breath event duration
            max_duration_ms: Maximum breath event duration
            energy_min_threshold: Minimum energy for breath detection
            energy_max_threshold: Maximum energy for breath detection
            window_length_ms: Analysis window length in milliseconds
        """
        self.sample_rate = sample_rate
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.energy_min_threshold = energy_min_threshold
        self.energy_max_threshold = energy_max_threshold
        
        # Convert time parameters to samples
        self.window_length = int(window_length_ms * sample_rate / 1000)
        self.min_duration_samples = int(min_duration_ms * sample_rate / 1000)
        self.max_duration_samples = int(max_duration_ms * sample_rate / 1000)
    
    def compute_energy(self, audio: np.ndarray) -> float:
        """
        Compute RMS energy of audio segment.
        
        Args:
            audio: Audio waveform
            
        Returns:
            RMS energy value
        """
        return np.sqrt(np.mean(audio ** 2))
    
    def compute_spectral_flatness(self, audio: np.ndarray) -> float:
        """
        Compute spectral flatness (measure of noise-likeness).
        
        Args:
            audio: Audio waveform
            
        Returns:
            Spectral flatness value (0-1, higher = more noise-like)
        """
        if len(audio) < self.window_length:
            return 0.0
            
        # Compute power spectral density
        freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=min(self.window_length, len(audio)))
        
        # Remove DC component
        psd = psd[1:]
        
        # Avoid log of zero
        psd = psd + 1e-12
        
        # Spectral flatness: geometric mean / arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(psd)))
        arithmetic_mean = np.mean(psd)
        
        spectral_flatness = geometric_mean / arithmetic_mean
        return spectral_flatness
    
    def is_breath_like(self, audio_segment: np.ndarray) -> tuple[bool, float]:
        """
        Determine if an audio segment is breath-like using simple heuristics.
        
        Args:
            audio_segment: Audio segment to analyze
            
        Returns:
            Tuple of (is_breath_like, confidence_score)
        """
        # Check duration
        duration_samples = len(audio_segment)
        if duration_samples < self.min_duration_samples or duration_samples > self.max_duration_samples:
            return False, 0.0
        
        # Compute basic features
        energy = self.compute_energy(audio_segment)
        spectral_flatness = self.compute_spectral_flatness(audio_segment)
        
        # Simple breath-like criteria
        confidence_score = 0.0
        
        # Energy should be in breath range (not silence, not loud)
        if self.energy_min_threshold <= energy <= self.energy_max_threshold:
            confidence_score += 0.6
        elif energy > self.energy_min_threshold:  # At least some energy
            confidence_score += 0.2
        
        # High spectral flatness indicates noise-like (breath-like) sound
        if spectral_flatness > 0.5:
            confidence_score += 0.4
        
        # Threshold for classification
        is_breath = confidence_score > 0.5
        
        return is_breath, confidence_score
    
    def detect_breath_events(self, audio: np.ndarray) -> List[BreathEvent]:
        """
        Detect breath events using sliding window analysis.
        
        Args:
            audio: Mono audio waveform (16kHz)
            
        Returns:
            List of detected breath events
        """
        breath_events = []
        
        # Fixed-size sliding window: one evaluation per position (O(N)).
        # The previous inner loop tried every sample duration from
        # min_duration_samples to max_duration_samples, calling signal.welch
        # ~9600 times per outer step — catastrophically slow on real audio.
        # Using the midpoint of the allowed range gives representative windows
        # while keeping CPU usage stable and predictable.
        target_duration_samples = (self.min_duration_samples + self.max_duration_samples) // 2
        step_size = self.window_length // 4  # 75% overlap

        for start_sample in range(0, len(audio) - target_duration_samples, step_size):
            end_sample = start_sample + target_duration_samples
            audio_segment = audio[start_sample:end_sample]

            is_breath, confidence = self.is_breath_like(audio_segment)

            if is_breath:
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                duration_ms = (end_time - start_time) * 1000

                # Deduplicate: keep highest-confidence detection per overlapping region
                overlap = False
                for existing_event in breath_events:
                    if (start_time < existing_event.end_time and
                            end_time > existing_event.start_time):
                        if confidence > existing_event.confidence:
                            breath_events.remove(existing_event)
                        else:
                            overlap = True
                        break

                if not overlap:
                    breath_events.append(BreathEvent(
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration_ms,
                        confidence=confidence
                    ))
        
        # Sort events by start time
        breath_events.sort(key=lambda x: x.start_time)
        
        # Clean up temporary variables
        if 'audio_segment' in locals():
            del audio_segment
        gc.collect()
        
        return breath_events
    
    def get_statistics(self, audio: np.ndarray, breath_events: List[BreathEvent]) -> dict:
        """
        Get basic detection statistics.
        
        Args:
            audio: Input audio waveform
            breath_events: Detected breath events
            
        Returns:
            Dictionary with detection statistics
        """
        total_duration = len(audio) / self.sample_rate
        
        return {
            "total_audio_duration_s": total_duration,
            "total_breath_events": len(breath_events),
            "average_breath_duration_ms": np.mean([event.duration_ms for event in breath_events]) if breath_events else 0,
            "average_confidence": np.mean([event.confidence for event in breath_events]) if breath_events else 0
        }