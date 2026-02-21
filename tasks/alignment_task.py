"""
Speaker-transcript alignment task for the WIZ Intelligence Pipeline.
"""

from ..core.base_task import BaseTask
from ..core.context import PipelineContext
from ..audio.speaker_alignment import SpeakerAligner


class AlignmentTask(BaseTask):
    """
    Task for aligning transcript words with speaker segments.
    
    Merges transcription results with diarization results to create
    speaker-aligned segments in the pipeline context.
    """
    
    def __init__(self, speaker_aligner: SpeakerAligner) -> None:
        """
        Initialize the alignment task with a speaker aligner.
        
        Args:
            speaker_aligner: Configured SpeakerAligner instance
        """
        super().__init__("SpeakerAlignment")
        self.speaker_aligner = speaker_aligner
    
    def _run(self, context: PipelineContext) -> None:
        """
        Execute speaker-transcript alignment.
        
        Args:
            context: Pipeline context containing transcript words and speaker segments
        """
        transcript_words = context.transcript_words
        speaker_segments = context.speaker_segments
        
        self.logger.info(
            f"Aligning {len(transcript_words)} words with {len(speaker_segments)} speaker segments"
        )
        
        # Perform alignment
        aligned_segments = self.speaker_aligner.align_transcript_with_speakers(
            transcript_words, speaker_segments
        )
        
        # Store results in context
        context.aligned_segments.extend(aligned_segments)
        
        # Generate and store statistics
        alignment_stats = self.speaker_aligner.get_alignment_statistics(aligned_segments)
        validation_errors = self.speaker_aligner.validate_alignment(aligned_segments)
        
        stats = {
            'alignment_statistics': alignment_stats,
            'validation_errors': validation_errors,
            'total_aligned_segments': len(aligned_segments)
        }
        
        context.processing_metadata['alignment'] = stats
        
        # Log results
        self.logger.info(f"Alignment completed: {len(aligned_segments)} aligned segments")
        
        if alignment_stats['num_speakers'] > 0:
            self.logger.info(f"Speakers identified: {alignment_stats['num_speakers']}")
            
            # Log speaker breakdown
            speaker_durations = alignment_stats.get('speaker_durations', {})
            total_duration = alignment_stats.get('total_duration', 0)
            
            for speaker_id, duration in speaker_durations.items():
                percentage = (duration / total_duration * 100) if total_duration > 0 else 0
                word_count = alignment_stats.get('speaker_word_counts', {}).get(speaker_id, 0)
                self.logger.info(
                    f"  {speaker_id}: {duration:.1f}s ({percentage:.1f}%), {word_count} words"
                )
        
        # Log validation results
        if validation_errors:
            self.logger.warning(f"Alignment validation found {len(validation_errors)} issues:")
            for error in validation_errors[:3]:  # Show first 3 errors
                self.logger.warning(f"  - {error}")
            if len(validation_errors) > 3:
                self.logger.warning(f"  ... and {len(validation_errors) - 3} more issues")
        else:
            self.logger.info("Alignment validation passed")
    
    @classmethod
    def create_default(cls, overlap_threshold: float = 0.5) -> 'AlignmentTask':
        """
        Create an AlignmentTask with default configuration.
        
        Args:
            overlap_threshold: Minimum overlap ratio to assign word to speaker
            
        Returns:
            Configured AlignmentTask instance
        """
        speaker_aligner = SpeakerAligner(overlap_threshold=overlap_threshold)
        return cls(speaker_aligner)