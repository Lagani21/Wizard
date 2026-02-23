"""
Local Pyannote speaker diarization model for the WIZ Intelligence Pipeline.
"""

import logging
import os
import tempfile
import numpy as np
from typing import List, Optional, Dict, Any
try:
    # Try relative imports first
    from ..core.context import SpeakerSegment
except ImportError:
    # Fall back to absolute imports
    from core.context import SpeakerSegment


class DiarizationModel:
    """
    Local Pyannote speaker diarization model.
    
    Provides speaker diarization (who spoke when) using Pyannote.audio
    running locally on CPU.
    """
    
    def __init__(self, 
                 model_name: str = "pyannote/speaker-diarization-3.1",
                 auth_token: Optional[str] = None,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None) -> None:
        """
        Initialize the diarization model.
        
        Args:
            model_name: Pyannote model name
            auth_token: HuggingFace auth token (if required)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
        """
        self.model_name = model_name
        self.auth_token = auth_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        self.pipeline = None
        self.is_loaded = False
        
        self.logger = logging.getLogger("wiz.models.diarization")
    
    def load_model(self) -> bool:
        """
        Load the Pyannote diarization pipeline.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.is_loaded:
            return True
        
        try:
            from pyannote.audio import Pipeline

            self.logger.info(f"Loading Pyannote model: {self.model_name}")

            # Resolve auth token: explicit arg → HF_TOKEN env → HUGGING_FACE_HUB_TOKEN env
            token = (
                self.auth_token
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            )

            if token:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=token,
                )
            else:
                self.pipeline = Pipeline.from_pretrained(self.model_name)
            
            self.is_loaded = True
            self.logger.info("Pyannote diarization model loaded successfully")
            return True
            
        except ImportError:
            self.logger.error("pyannote.audio library not installed. Install with: pip install pyannote.audio")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load Pyannote model: {e}")
            self.logger.info("You may need to accept user conditions at https://hf.co/pyannote/speaker-diarization-3.1")
            return False
    
    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on audio.
        
        Args:
            audio: Audio waveform (mono, float32)
            sample_rate: Audio sample rate
            
        Returns:
            List of SpeakerSegment objects
        """
        if not self.is_loaded:
            if not self.load_model():
                return self._create_fallback_diarization(audio, sample_rate)
        
        try:
            self.logger.info(f"Diarizing {len(audio)/sample_rate:.1f}s of audio")
            
            # Create temporary audio file for pyannote
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_audio_path = temp_file.name
            
            try:
                # Save audio to temporary WAV file
                self._save_audio_to_wav(audio, temp_audio_path, sample_rate)
                
                # Run diarization
                diarization_result = self.pipeline(
                    temp_audio_path,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers
                )
                
                # Extract speaker segments
                speaker_segments = self._extract_speaker_segments(diarization_result)
                
                self.logger.info(f"Diarization complete: {len(speaker_segments)} segments")
                
                return speaker_segments
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
        
        except Exception as e:
            self.logger.error(f"Pyannote diarization failed: {e}")
            return self._create_fallback_diarization(audio, sample_rate)
    
    def _save_audio_to_wav(self, audio: np.ndarray, output_path: str, sample_rate: int) -> None:
        """
        Save audio array to WAV file.
        
        Args:
            audio: Audio waveform
            output_path: Output file path
            sample_rate: Audio sample rate
        """
        try:
            import scipy.io.wavfile as wavfile
            
            # Convert to int16 for WAV format
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(output_path, sample_rate, audio_int16)
            
        except ImportError:
            # Fallback: use simple WAV writer
            self._write_simple_wav(audio, output_path, sample_rate)
    
    def _write_simple_wav(self, audio: np.ndarray, output_path: str, sample_rate: int) -> None:
        """
        Simple WAV file writer without scipy dependency.
        
        Args:
            audio: Audio waveform  
            output_path: Output file path
            sample_rate: Audio sample rate
        """
        import struct
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # WAV file header
        with open(output_path, 'wb') as f:
            # RIFF header
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + len(audio_int16) * 2))  # File size - 8
            f.write(b'WAVE')
            
            # Format chunk
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # Chunk size
            f.write(struct.pack('<H', 1))   # Audio format (PCM)
            f.write(struct.pack('<H', 1))   # Channels (mono)
            f.write(struct.pack('<I', sample_rate))  # Sample rate
            f.write(struct.pack('<I', sample_rate * 2))  # Byte rate
            f.write(struct.pack('<H', 2))   # Block align
            f.write(struct.pack('<H', 16))  # Bits per sample
            
            # Data chunk
            f.write(b'data')
            f.write(struct.pack('<I', len(audio_int16) * 2))  # Data size
            f.write(audio_int16.tobytes())
    
    def _extract_speaker_segments(self, diarization_result) -> List[SpeakerSegment]:
        """
        Extract speaker segments from Pyannote diarization result.
        
        Args:
            diarization_result: Pyannote diarization output
            
        Returns:
            List of SpeakerSegment objects
        """
        segments = []
        
        for segment, _, speaker in diarization_result.itertracks(yield_label=True):
            speaker_segment = SpeakerSegment(
                speaker_id=str(speaker),
                start_time=segment.start,
                end_time=segment.end
            )
            segments.append(speaker_segment)
        
        # Sort by start time
        segments.sort(key=lambda x: x.start_time)
        
        return segments
    
    def _create_fallback_diarization(self, audio: np.ndarray, sample_rate: int) -> List[SpeakerSegment]:
        """
        Create fallback diarization when Pyannote fails.
        
        Args:
            audio: Audio waveform
            sample_rate: Audio sample rate
            
        Returns:
            List with single speaker segment
        """
        duration = len(audio) / sample_rate
        
        # Create a single speaker for the entire audio
        fallback_segment = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start_time=0.0,
            end_time=duration
        )
        
        self.logger.warning(f"Using fallback diarization (single speaker) for {duration:.1f}s audio")
        
        return [fallback_segment]
    
    def get_speaker_statistics(self, segments: List[SpeakerSegment]) -> Dict[str, Any]:
        """
        Get statistics about speaker segments.
        
        Args:
            segments: List of speaker segments
            
        Returns:
            Dictionary with speaker statistics
        """
        if not segments:
            return {"num_speakers": 0, "total_duration": 0.0, "speaker_durations": {}}
        
        # Calculate total duration and per-speaker durations
        speaker_durations = {}
        total_duration = 0.0
        
        for segment in segments:
            duration = segment.end_time - segment.start_time
            total_duration += duration
            
            if segment.speaker_id not in speaker_durations:
                speaker_durations[segment.speaker_id] = 0.0
            speaker_durations[segment.speaker_id] += duration
        
        # Calculate speaker percentages
        speaker_percentages = {}
        if total_duration > 0:
            for speaker_id, duration in speaker_durations.items():
                speaker_percentages[speaker_id] = (duration / total_duration) * 100
        
        return {
            "num_speakers": len(speaker_durations),
            "total_duration": total_duration,
            "speaker_durations": speaker_durations,
            "speaker_percentages": speaker_percentages,
            "num_segments": len(segments)
        }
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "is_loaded": self.is_loaded,
            "backend": "pyannote.audio"
        }
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        self.is_loaded = False
        self.logger.info("Pyannote diarization model unloaded")
    
    def extract_speaker_embeddings(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: List[SpeakerSegment],
    ) -> Dict[str, np.ndarray]:
        """
        Extract a mean embedding vector for each unique speaker.

        Tries pyannote's internal WeSpeaker/ECAPA embedding model first.
        Falls back to a numpy-only log-mel spectral fingerprint if pyannote
        internals are inaccessible or torch is unavailable.

        Args:
            audio:       Mono float32 waveform
            sample_rate: Audio sample rate (Hz)
            segments:    Speaker segments returned by diarize()

        Returns:
            {speaker_id → embedding_vector (float32 ndarray)}
        """
        # Collect audio chunks per speaker (≥ 0.5 s only)
        speaker_chunks: Dict[str, List[np.ndarray]] = {}
        for seg in segments:
            s = int(seg.start_time * sample_rate)
            e = int(seg.end_time * sample_rate)
            chunk = audio[s:e]
            if len(chunk) > sample_rate * 0.5:
                speaker_chunks.setdefault(seg.speaker_id, []).append(chunk)

        if not speaker_chunks:
            return {}

        # Attempt 1: pyannote internal embedding model
        result = self._embed_with_pyannote(speaker_chunks, sample_rate)
        if result:
            return result

        # Attempt 2: lightweight spectral fingerprint (numpy only)
        return self._embed_spectral(speaker_chunks, sample_rate)

    def _embed_with_pyannote(
        self,
        speaker_chunks: Dict[str, List[np.ndarray]],
        sample_rate: int,
    ) -> Dict[str, np.ndarray]:
        """Use pyannote's internal embedding model if accessible."""
        if not self.is_loaded or self.pipeline is None:
            return {}
        try:
            import torch
            from pyannote.audio import Inference

            emb_model = getattr(self.pipeline, "_embedding", None)
            if emb_model is None:
                return {}

            inference = Inference(emb_model, window="whole")
            result: Dict[str, np.ndarray] = {}
            for sid, chunks in speaker_chunks.items():
                chunk = max(chunks, key=len)
                waveform = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
                emb = inference({"waveform": waveform, "sample_rate": sample_rate})
                result[sid] = np.array(emb, dtype=np.float32).flatten()

            self.logger.info(
                f"Extracted pyannote embeddings for {len(result)} speakers"
            )
            return result
        except Exception as e:
            self.logger.debug(f"pyannote embedding extraction unavailable: {e}")
            return {}

    def _embed_spectral(
        self,
        speaker_chunks: Dict[str, List[np.ndarray]],
        sample_rate: int,
    ) -> Dict[str, np.ndarray]:
        """
        Fallback: 80-dim log-mel spectral fingerprint (mean + std of 40 bands).
        Uses only numpy — no extra dependencies.
        """
        n_bands = 40
        fft_size = 512

        def _mel_filterbank(n_fft: int, n_mel: int, sr: int) -> np.ndarray:
            low = 2595 * np.log10(1 + 80 / 700)
            high = 2595 * np.log10(1 + (sr / 2) / 700)
            mel_pts = np.linspace(low, high, n_mel + 2)
            hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
            bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
            fb = np.zeros((n_mel, n_fft // 2 + 1))
            for m in range(1, n_mel + 1):
                for k in range(bins[m - 1], bins[m]):
                    fb[m - 1, k] = (k - bins[m - 1]) / max(bins[m] - bins[m - 1], 1)
                for k in range(bins[m], bins[m + 1]):
                    fb[m - 1, k] = (bins[m + 1] - k) / max(bins[m + 1] - bins[m], 1)
            return fb

        fb = _mel_filterbank(fft_size, n_bands, sample_rate)
        window = np.hanning(fft_size)
        hop = fft_size // 2
        result: Dict[str, np.ndarray] = {}

        for sid, chunks in speaker_chunks.items():
            frames_list = []
            for chunk in chunks:
                for i in range(0, len(chunk) - fft_size, hop):
                    frames_list.append(chunk[i: i + fft_size] * window)

            if not frames_list:
                result[sid] = np.zeros(n_bands * 2, dtype=np.float32)
                continue

            frames_np = np.stack(frames_list)
            spectrum = np.abs(np.fft.rfft(frames_np, n=fft_size)) ** 2
            log_mel = np.log(fb @ spectrum.T + 1e-8)   # (n_bands, T)
            result[sid] = np.concatenate(
                [log_mel.mean(axis=1), log_mel.std(axis=1)]
            ).astype(np.float32)

        self.logger.info(
            f"Extracted spectral fingerprints for {len(result)} speakers"
        )
        return result

    def merge_adjacent_segments(self, segments: List[SpeakerSegment], gap_threshold: float = 0.5) -> List[SpeakerSegment]:
        """
        Merge adjacent segments from the same speaker.
        
        Args:
            segments: List of speaker segments
            gap_threshold: Maximum gap to merge (seconds)
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        merged_segments = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # Check if segments are from same speaker and close in time
            if (current_segment.speaker_id == next_segment.speaker_id and
                next_segment.start_time - current_segment.end_time <= gap_threshold):
                
                # Merge segments
                current_segment = SpeakerSegment(
                    speaker_id=current_segment.speaker_id,
                    start_time=current_segment.start_time,
                    end_time=next_segment.end_time
                )
            else:
                # Add current segment and start new one
                merged_segments.append(current_segment)
                current_segment = next_segment
        
        # Add the last segment
        merged_segments.append(current_segment)
        
        return merged_segments