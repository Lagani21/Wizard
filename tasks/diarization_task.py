"""
Speaker diarization task for the WIZ Intelligence Pipeline.
"""

import numpy as np
from ..core.base_task import BaseTask
from ..core.context import PipelineContext
from ..models.diarization_model import DiarizationModel


class DiarizationTask(BaseTask):
    """
    Task for speaker diarization using Pyannote.
    
    Processes audio waveform using DiarizationModel and stores
    speaker segments in the pipeline context.
    """
    
    def __init__(self, diarization_model: DiarizationModel) -> None:
        """
        Initialize the diarization task with a diarization model.
        
        Args:
            diarization_model: Configured DiarizationModel instance
        """
        super().__init__("SpeakerDiarization")
        self.diarization_model = diarization_model
    
    def _run(self, context: PipelineContext) -> None:
        """
        Execute speaker diarization on audio waveform.
        
        Args:
            context: Pipeline context containing audio waveform
        """
        if context.audio_waveform is None:
            raise ValueError("Audio waveform not available in context")
        
        # Log audio information
        duration_s = len(context.audio_waveform) / 16000  # Assuming 16kHz
        self.logger.info(
            f"Processing audio for diarization: {len(context.audio_waveform)} samples, "
            f"{duration_s:.2f}s duration"
        )
        
        # Ensure model is loaded
        if not self.diarization_model.is_loaded:
            self.logger.info("Loading Pyannote diarization model...")
            if not self.diarization_model.load_model():
                raise RuntimeError("Failed to load Pyannote diarization model")
        
        # Perform diarization
        self.logger.info("Running speaker diarization...")
        speaker_segments = self.diarization_model.diarize(
            context.audio_waveform, 
            sample_rate=16000
        )
        
        # Merge adjacent segments from same speaker (optional optimization)
        if len(speaker_segments) > 1:
            merged_segments = self.diarization_model.merge_adjacent_segments(speaker_segments)
            self.logger.info(f"Merged {len(speaker_segments)} segments into {len(merged_segments)}")
            speaker_segments = merged_segments
        
        # Store results in context
        context.speaker_segments.extend(speaker_segments)
        
        # Generate statistics
        speaker_stats = self.diarization_model.get_speaker_statistics(speaker_segments)
        
        # Store processing statistics
        stats = {
            'diarization_model_info': self.diarization_model.get_model_info(),
            'speaker_statistics': speaker_stats,
            'total_segments': len(speaker_segments),
            'audio_duration_s': duration_s
        }
        
        context.processing_metadata['diarization'] = stats
        
        # Log results
        num_speakers = speaker_stats['num_speakers']
        self.logger.info(
            f"Diarization completed: {num_speakers} speakers, "
            f"{len(speaker_segments)} segments"
        )
        
        # Log speaker breakdown
        if speaker_stats['speaker_percentages']:
            self.logger.info("Speaker breakdown:")
            for speaker_id, percentage in speaker_stats['speaker_percentages'].items():
                duration = speaker_stats['speaker_durations'][speaker_id]
                self.logger.info(f"  {speaker_id}: {duration:.1f}s ({percentage:.1f}%)")
    
    @classmethod
    def create_default(cls, 
                      min_speakers: int = None, 
                      max_speakers: int = None,
                      auth_token: str = None) -> 'DiarizationTask':
        """
        Create a DiarizationTask with default Pyannote configuration.
        
        Args:
            min_speakers: Minimum number of speakers to expect
            max_speakers: Maximum number of speakers to expect
            auth_token: HuggingFace auth token if required
            
        Returns:
            Configured DiarizationTask instance
        """
        diarization_model = DiarizationModel(
            model_name="pyannote/speaker-diarization-3.1",
            auth_token=auth_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        return cls(diarization_model)