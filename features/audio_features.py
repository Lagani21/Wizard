"""
Audio feature extraction from waveform data for emotional tone detection.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple
from scipy import signal
try:
    # Try relative imports first
    from ..core.context import SpeakerSegment
except ImportError:
    # Fall back to absolute imports
    from core.context import SpeakerSegment


class AudioFeatureExtractor:
    """
    Extracts audio-based features from waveform data for tone detection.
    
    Analyzes acoustic properties like volume, energy, and spectral characteristics
    that correlate with emotional tone.
    """
    
    def __init__(self, sample_rate: int = 16000) -> None:
        """
        Initialize the audio feature extractor.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger("wiz.features.audio")
        
        # Frame parameters for analysis
        self.frame_length_ms = 25.0
        self.hop_length_ms = 10.0
        self.frame_length = int(self.frame_length_ms * sample_rate / 1000)
        self.hop_length = int(self.hop_length_ms * sample_rate / 1000)
    
    def extract_features(self, 
                        audio_waveform: np.ndarray,
                        speaker_segments: List[SpeakerSegment],
                        time_window_start: float,
                        time_window_end: float) -> Dict[str, float]:
        """
        Extract audio features for a given time window.
        
        Args:
            audio_waveform: Full audio waveform
            speaker_segments: List of speaker segments
            time_window_start: Start time of analysis window
            time_window_end: End time of analysis window
            
        Returns:
            Dictionary of extracted audio features
        """
        # Extract audio segment for the time window
        start_sample = int(time_window_start * self.sample_rate)
        end_sample = int(time_window_end * self.sample_rate)
        
        if start_sample >= len(audio_waveform) or end_sample <= start_sample:
            return self._get_empty_features()
        
        audio_segment = audio_waveform[start_sample:min(end_sample, len(audio_waveform))]
        
        if len(audio_segment) < self.frame_length:
            return self._get_empty_features()
        
        # Get speaker segments in this window
        window_speaker_segments = self._get_speaker_segments_in_window(
            speaker_segments, time_window_start, time_window_end
        )
        
        # Extract various audio features
        features = {}
        
        # Volume and energy features
        features.update(self._extract_volume_features(audio_segment))
        
        # Spectral features
        features.update(self._extract_spectral_features(audio_segment))
        
        # Temporal features
        features.update(self._extract_temporal_features(audio_segment))
        
        # Speech activity features
        features.update(self._extract_speech_activity_features(
            audio_segment, window_speaker_segments, time_window_start, time_window_end
        ))
        
        # Prosodic features
        features.update(self._extract_prosodic_features(audio_segment))
        
        self.logger.debug(f"Extracted {len(features)} audio features for window {time_window_start:.1f}-{time_window_end:.1f}s")
        
        return features
    
    def _get_speaker_segments_in_window(self, 
                                      segments: List[SpeakerSegment],
                                      start_time: float,
                                      end_time: float) -> List[SpeakerSegment]:
        """Get speaker segments that overlap with the time window."""
        window_segments = []
        
        for segment in segments:
            # Check for overlap
            if (segment.start_time < end_time and segment.end_time > start_time):
                window_segments.append(segment)
        
        return window_segments
    
    def _extract_volume_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract volume and energy-based features."""
        # RMS energy
        rms_energy = np.sqrt(np.mean(audio ** 2))
        
        # Peak amplitude
        peak_amplitude = np.max(np.abs(audio))
        
        # Dynamic range
        if rms_energy > 1e-10:
            dynamic_range = 20 * np.log10(peak_amplitude / rms_energy)
        else:
            dynamic_range = 0.0
        
        # Frame-wise energy for variance calculation
        frame_energies = []
        for i in range(0, len(audio) - self.frame_length, self.hop_length):
            frame = audio[i:i + self.frame_length]
            frame_energy = np.sqrt(np.mean(frame ** 2))
            frame_energies.append(frame_energy)
        
        frame_energies = np.array(frame_energies)
        energy_variance = np.var(frame_energies)
        energy_mean = np.mean(frame_energies)
        
        # Energy percentiles (for distribution analysis)
        energy_percentiles = np.percentile(frame_energies, [25, 50, 75]) if len(frame_energies) > 0 else [0, 0, 0]
        
        return {
            "rms_energy": float(rms_energy),
            "peak_amplitude": float(peak_amplitude),
            "dynamic_range_db": float(dynamic_range),
            "energy_variance": float(energy_variance),
            "energy_mean": float(energy_mean),
            "energy_q25": float(energy_percentiles[0]),
            "energy_median": float(energy_percentiles[1]),
            "energy_q75": float(energy_percentiles[2])
        }
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral characteristics features."""
        # Compute power spectral density
        freqs, psd = signal.welch(audio, fs=self.sample_rate, nperseg=min(1024, len(audio)))
        
        # Remove DC component
        freqs = freqs[1:]
        psd = psd[1:]
        
        if len(psd) == 0:
            return {
                "spectral_centroid": 0.0,
                "spectral_bandwidth": 0.0,
                "spectral_flatness": 0.0,
                "spectral_rolloff": 0.0
            }
        
        # Spectral centroid (brightness)
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
        
        # Spectral flatness (measure of noise-likeness)
        psd_safe = psd + 1e-12
        geometric_mean = np.exp(np.mean(np.log(psd_safe)))
        arithmetic_mean = np.mean(psd_safe)
        spectral_flatness = geometric_mean / arithmetic_mean
        
        # Spectral rolloff (85% of energy)
        cumsum_psd = np.cumsum(psd)
        total_energy = cumsum_psd[-1]
        rolloff_idx = np.where(cumsum_psd >= 0.85 * total_energy)[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        return {
            "spectral_centroid": float(spectral_centroid),
            "spectral_bandwidth": float(spectral_bandwidth),
            "spectral_flatness": float(spectral_flatness),
            "spectral_rolloff": float(spectral_rolloff)
        }
    
    def _extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract temporal pattern features."""
        # Zero crossing rate
        signs = np.sign(audio)
        sign_changes = np.diff(signs)
        zero_crossings = np.sum(np.abs(sign_changes)) / 2
        zcr = zero_crossings / len(audio)
        
        # Autocorrelation for periodicity detection
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        
        # Find peaks in autocorrelation (potential pitch periods)
        if len(autocorr) > 100:
            peaks, _ = signal.find_peaks(autocorr[20:], height=0.3)
            periodicity = np.max(autocorr[20:]) if len(peaks) > 0 else 0.0
        else:
            periodicity = 0.0
        
        return {
            "zero_crossing_rate": float(zcr),
            "periodicity": float(periodicity)
        }
    
    def _extract_speech_activity_features(self, 
                                        audio: np.ndarray,
                                        speaker_segments: List[SpeakerSegment],
                                        window_start: float,
                                        window_end: float) -> Dict[str, float]:
        """Extract speech activity and dialogue features."""
        window_duration = window_end - window_start
        
        # Calculate speech duration within the window
        total_speech_duration = 0.0
        num_speakers_active = 0
        overlap_duration = 0.0
        
        if speaker_segments:
            # Track active speakers
            active_speakers = set()
            
            for segment in speaker_segments:
                # Calculate overlap with window
                segment_start = max(segment.start_time, window_start)
                segment_end = min(segment.end_time, window_end)
                
                if segment_end > segment_start:
                    speech_duration = segment_end - segment_start
                    total_speech_duration += speech_duration
                    active_speakers.add(segment.speaker_id)
            
            num_speakers_active = len(active_speakers)
            
            # Check for overlapping speakers (approximate)
            if len(speaker_segments) > 1:
                for i, seg1 in enumerate(speaker_segments):
                    for seg2 in speaker_segments[i+1:]:
                        if (seg1.speaker_id != seg2.speaker_id and
                            seg1.start_time < seg2.end_time and seg1.end_time > seg2.start_time):
                            overlap_start = max(seg1.start_time, seg2.start_time, window_start)
                            overlap_end = min(seg1.end_time, seg2.end_time, window_end)
                            if overlap_end > overlap_start:
                                overlap_duration += overlap_end - overlap_start
        
        # Speech activity ratio
        speech_ratio = total_speech_duration / window_duration if window_duration > 0 else 0.0
        
        # Speaker overlap ratio
        overlap_ratio = overlap_duration / window_duration if window_duration > 0 else 0.0
        
        return {
            "speech_activity_ratio": float(speech_ratio),
            "num_active_speakers": float(num_speakers_active),
            "speaker_overlap_ratio": float(overlap_ratio)
        }
    
    def _extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract prosodic features related to pitch and rhythm."""
        # Simple pitch estimation using autocorrelation
        frame_size = min(2048, len(audio))
        if len(audio) < frame_size:
            return {
                "pitch_variance": 0.0,
                "pitch_range": 0.0,
                "rhythm_regularity": 0.0
            }
        
        # Frame-based pitch estimation
        pitches = []
        for i in range(0, len(audio) - frame_size, frame_size // 2):
            frame = audio[i:i + frame_size]
            
            # Autocorrelation-based pitch estimation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for peak in reasonable pitch range (80-400 Hz)
            min_period = int(self.sample_rate / 400)  # 400 Hz
            max_period = int(self.sample_rate / 80)   # 80 Hz
            
            if max_period < len(autocorr):
                search_range = autocorr[min_period:max_period]
                if len(search_range) > 0:
                    peak_idx = np.argmax(search_range) + min_period
                    if autocorr[peak_idx] > 0.3 * autocorr[0]:  # Threshold for voiced detection
                        pitch = self.sample_rate / peak_idx
                        pitches.append(pitch)
        
        if len(pitches) > 0:
            pitch_variance = np.var(pitches)
            pitch_range = np.max(pitches) - np.min(pitches)
            
            # Rhythm regularity (consistency of pitch changes)
            if len(pitches) > 2:
                pitch_diffs = np.diff(pitches)
                rhythm_regularity = 1.0 / (1.0 + np.std(pitch_diffs))  # Higher = more regular
            else:
                rhythm_regularity = 0.0
        else:
            pitch_variance = 0.0
            pitch_range = 0.0
            rhythm_regularity = 0.0
        
        return {
            "pitch_variance": float(pitch_variance),
            "pitch_range": float(pitch_range),
            "rhythm_regularity": float(rhythm_regularity)
        }
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return empty feature set when no audio data is available."""
        return {
            "rms_energy": 0.0,
            "peak_amplitude": 0.0,
            "dynamic_range_db": 0.0,
            "energy_variance": 0.0,
            "energy_mean": 0.0,
            "energy_q25": 0.0,
            "energy_median": 0.0,
            "energy_q75": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_flatness": 0.0,
            "spectral_rolloff": 0.0,
            "zero_crossing_rate": 0.0,
            "periodicity": 0.0,
            "speech_activity_ratio": 0.0,
            "num_active_speakers": 0.0,
            "speaker_overlap_ratio": 0.0,
            "pitch_variance": 0.0,
            "pitch_range": 0.0,
            "rhythm_regularity": 0.0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names extracted by this class."""
        return list(self._get_empty_features().keys())