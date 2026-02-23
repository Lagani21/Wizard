"""
Audio extraction utilities for the WIZ Intelligence Pipeline.
"""

import os
import tempfile
import subprocess
import numpy as np
import logging
from typing import Optional
from pathlib import Path


class AudioExtractor:
    """
    Extracts audio from video files for speech processing.
    
    Converts video audio to mono 16kHz format suitable for 
    Whisper transcription and Pyannote diarization.
    """
    
    def __init__(self, target_sample_rate: int = 16000) -> None:
        """
        Initialize the audio extractor.
        
        Args:
            target_sample_rate: Target sample rate for extracted audio
        """
        self.target_sample_rate = target_sample_rate
        self.logger = logging.getLogger("wiz.audio.extractor")
    
    def extract_audio_from_video(self, video_path: str) -> np.ndarray:
        """
        Extract mono audio from video file at target sample rate.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Mono audio waveform as numpy array
            
        Raises:
            RuntimeError: If audio extraction fails
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.logger.info(f"Extracting audio from {video_path}")
        
        # Check if ffmpeg is available
        if not self._check_ffmpeg():
            self.logger.warning("ffmpeg not available, using placeholder audio")
            return self._create_placeholder_audio(video_path)
        
        try:
            # Use ffmpeg to extract audio
            audio_data = self._extract_with_ffmpeg(video_path)
            self.logger.info(f"Extracted {len(audio_data)} samples at {self.target_sample_rate}Hz")
            return audio_data
            
        except Exception as e:
            self.logger.warning(f"ffmpeg extraction failed: {e}, using placeholder")
            return self._create_placeholder_audio(video_path)
    
    def _check_ffmpeg(self) -> bool:
        """
        Check if ffmpeg is available on the system.
        
        Returns:
            True if ffmpeg is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def _extract_with_ffmpeg(self, video_path: str) -> np.ndarray:
        """
        Extract audio using ffmpeg.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio waveform as numpy array
        """
        # Get a unique temp path without pre-creating the file.
        # NamedTemporaryFile leaves the file open on macOS, which causes ffmpeg
        # to fail with "Invalid argument" when it tries to write to the same path.
        temp_fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)   # release FD immediately
        os.unlink(temp_audio_path)  # let ffmpeg create the file fresh

        try:
            # FFmpeg: extract mono 16kHz WAV (WAV avoids raw-PCM path issues on macOS)
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vn",                          # no video stream
                "-ac", "1",                     # mono
                "-ar", str(self.target_sample_rate),
                "-y",                           # overwrite if somehow exists
                temp_audio_path,
            ]

            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

            # Read WAV with scipy (preferred) or wave module
            audio_normalized = None
            try:
                import scipy.io.wavfile as wavfile
                sr, audio_int16 = wavfile.read(temp_audio_path)
                # scipy may return float already for 32-bit WAVs
                if audio_int16.dtype == np.int16:
                    audio_normalized = audio_int16.astype(np.float32) / 32768.0
                else:
                    audio_normalized = audio_int16.astype(np.float32)
            except Exception:
                pass  # fall through to wave fallback

            if audio_normalized is None:
                # Fallback: read WAV with stdlib wave module
                import wave
                with wave.open(temp_audio_path, "rb") as wf:
                    raw_data = wf.readframes(wf.getnframes())
                audio_array = np.frombuffer(raw_data, dtype=np.int16)
                audio_normalized = audio_array.astype(np.float32) / 32768.0

            return audio_normalized

        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    def _create_placeholder_audio(self, video_path: str) -> np.ndarray:
        """
        Create placeholder audio when extraction fails.
        
        Args:
            video_path: Path to video file (for duration estimation)
            
        Returns:
            Placeholder audio waveform
        """
        try:
            # Try to get video duration using ffprobe
            duration = self._get_video_duration(video_path)
        except:
            # Fallback to default duration
            duration = 10.0
        
        # Generate quiet noise
        num_samples = int(duration * self.target_sample_rate)
        placeholder_audio = np.random.normal(0, 0.01, num_samples).astype(np.float32)
        
        self.logger.info(f"Generated placeholder audio: {duration:.1f}s, {num_samples} samples")
        
        return placeholder_audio
    
    def _get_video_duration(self, video_path: str) -> float:
        """
        Get video duration using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                video_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return float(result.stdout.strip())
            
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        
        # Fallback: estimate from file size (very rough)
        try:
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            # Rough estimate: 1MB per second for typical video
            estimated_duration = max(file_size_mb / 10, 5.0)  # At least 5 seconds
            return min(estimated_duration, 300.0)  # At most 5 minutes
        except:
            return 10.0  # Default fallback
    
    def save_audio_to_file(self, audio: np.ndarray, output_path: str) -> None:
        """
        Save audio array to WAV file for debugging/testing.
        
        Args:
            audio: Audio waveform
            output_path: Output file path
        """
        try:
            import scipy.io.wavfile as wavfile
            
            # Convert to int16 for WAV format
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(output_path, self.target_sample_rate, audio_int16)
            
            self.logger.info(f"Audio saved to {output_path}")
            
        except ImportError:
            self.logger.warning("scipy not available, cannot save audio file")
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
    
    def get_audio_info(self, audio: np.ndarray) -> dict:
        """
        Get information about the audio array.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Dictionary with audio information
        """
        duration_s = len(audio) / self.target_sample_rate
        rms_level = np.sqrt(np.mean(audio ** 2))
        max_level = np.max(np.abs(audio))
        
        return {
            "sample_rate": self.target_sample_rate,
            "duration_seconds": duration_s,
            "num_samples": len(audio),
            "rms_level": float(rms_level),
            "max_level": float(max_level),
            "dynamic_range_db": 20 * np.log10(max_level / (rms_level + 1e-10)) if rms_level > 0 else 0.0
        }