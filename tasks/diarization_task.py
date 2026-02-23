"""
Speaker diarization task for the WIZ Intelligence Pipeline.
"""

import numpy as np
from typing import Optional

try:
    # Try relative imports first
    from ..core.base_task import BaseTask
    from ..core.context import PipelineContext
    from ..models.diarization_model import DiarizationModel
    from ..models.speaker_identity import SpeakerIdentityRegistry
except ImportError:
    # Fall back to absolute imports
    from core.base_task import BaseTask
    from core.context import PipelineContext
    from models.diarization_model import DiarizationModel
    from models.speaker_identity import SpeakerIdentityRegistry


class DiarizationTask(BaseTask):
    """
    Task for speaker diarization using Pyannote.

    After diarizing, resolves per-clip SPEAKER_XX labels to stable
    cross-clip PERSON_XXX identities via SpeakerIdentityRegistry.
    """

    def __init__(
        self,
        diarization_model: DiarizationModel,
        identity_registry: Optional[SpeakerIdentityRegistry] = None,
    ) -> None:
        """
        Args:
            diarization_model: Configured DiarizationModel instance
            identity_registry: Optional shared registry for cross-clip IDs.
                               Defaults to the project-level registry file.
        """
        super().__init__("SpeakerDiarization")
        self.diarization_model = diarization_model
        self.identity_registry = identity_registry or SpeakerIdentityRegistry()
    
    def _run(self, context: PipelineContext) -> None:
        """
        Execute speaker diarization on audio waveform.
        
        Args:
            context: Pipeline context containing audio waveform
        """
        logger = context.logger
        if context.audio_waveform is None:
            raise ValueError("Audio waveform not available in context")
        
        # Log audio information
        duration_s = len(context.audio_waveform) / 16000  # Assuming 16kHz
        logger.log_info(
            f"Processing audio for diarization: {len(context.audio_waveform)} samples, "
            f"{duration_s:.2f}s duration"
        )
        
        # Ensure model is loaded (failure is non-fatal — diarize() has a single-speaker fallback)
        if not self.diarization_model.is_loaded:
            logger.log_info("Loading Pyannote diarization model...")
            if not self.diarization_model.load_model():
                logger.log_warning(
                    "Pyannote model failed to load — falling back to single-speaker diarization. "
                    "To enable full diarization set the HF_TOKEN environment variable and accept "
                    "the model license at https://hf.co/pyannote/speaker-diarization-3.1"
                )
        
        # Perform diarization
        logger.log_info("Running speaker diarization...")
        speaker_segments = self.diarization_model.diarize(
            context.audio_waveform, 
            sample_rate=16000
        )
        
        # Merge adjacent segments from same speaker (optional optimization)
        if len(speaker_segments) > 1:
            merged_segments = self.diarization_model.merge_adjacent_segments(speaker_segments)
            logger.log_info(f"Merged {len(speaker_segments)} segments into {len(merged_segments)}")
            speaker_segments = merged_segments
        
        # Store results in context
        context.speaker_segments.extend(speaker_segments)

        # ── Cross-clip speaker identity resolution ────────────────────────
        id_map: dict = {}
        try:
            embeddings = self.diarization_model.extract_speaker_embeddings(
                context.audio_waveform, 16000, speaker_segments
            )
            if embeddings:
                id_map = self.identity_registry.resolve(embeddings)
                # Remap every segment in context to its stable PERSON_XXX ID
                for seg in context.speaker_segments:
                    if seg.speaker_id in id_map:
                        seg.speaker_id = id_map[seg.speaker_id]
                logger.log_info(
                    f"Speaker identity resolved: "
                    + ", ".join(f"{k}→{v}" for k, v in id_map.items())
                )
            else:
                logger.log_info(
                    "Speaker embeddings unavailable — keeping per-clip SPEAKER_XX IDs"
                )
        except Exception as exc:
            logger.log_warning(f"Cross-clip identity resolution skipped: {exc}")

        # Generate statistics (uses remapped IDs if resolution succeeded)
        speaker_stats = self.diarization_model.get_speaker_statistics(speaker_segments)
        
        # Store processing statistics
        stats = {
            'diarization_model_info': self.diarization_model.get_model_info(),
            'speaker_statistics': speaker_stats,
            'total_segments': len(speaker_segments),
            'audio_duration_s': duration_s,
            'identity_mapping': id_map,
            'known_persons': self.identity_registry.known_persons(),
        }

        context.processing_metadata['diarization'] = stats
        
        # Log results
        num_speakers = speaker_stats['num_speakers']
        logger.log_info(
            f"Diarization completed: {num_speakers} speakers, "
            f"{len(speaker_segments)} segments"
        )
        
        # Log speaker breakdown
        if speaker_stats['speaker_percentages']:
            logger.log_info("Speaker breakdown:")
            for speaker_id, percentage in speaker_stats['speaker_percentages'].items():
                duration = speaker_stats['speaker_durations'][speaker_id]
                logger.log_info(f"  {speaker_id}: {duration:.1f}s ({percentage:.1f}%)")
    
    @classmethod
    def create_default(
        cls,
        min_speakers: int = None,
        max_speakers: int = None,
        auth_token: str = None,
        identity_registry: Optional[SpeakerIdentityRegistry] = None,
    ) -> 'DiarizationTask':
        """
        Create a DiarizationTask with default Pyannote configuration.

        Args:
            min_speakers:      Minimum number of speakers to expect
            max_speakers:      Maximum number of speakers to expect
            auth_token:        HuggingFace auth token if required
            identity_registry: Shared SpeakerIdentityRegistry for cross-clip IDs

        Returns:
            Configured DiarizationTask instance
        """
        diarization_model = DiarizationModel(
            model_name="pyannote/speaker-diarization-3.1",
            auth_token=auth_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        return cls(diarization_model, identity_registry=identity_registry)