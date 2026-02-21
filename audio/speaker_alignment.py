"""
Speaker-transcript alignment utilities for the WIZ Intelligence Pipeline.
"""

import logging
from typing import List, Optional, Dict, Any
from ..core.context import TranscriptWord, SpeakerSegment, SpeakerAlignedSegment


class SpeakerAligner:
    """
    Aligns transcript words with speaker diarization segments.
    
    Merges word-level transcription with speaker identification
    to create speaker-aligned transcript segments.
    """
    
    def __init__(self, overlap_threshold: float = 0.5) -> None:
        """
        Initialize the speaker aligner.
        
        Args:
            overlap_threshold: Minimum overlap ratio to assign word to speaker
        """
        self.overlap_threshold = overlap_threshold
        self.logger = logging.getLogger("wiz.audio.speaker_alignment")
    
    def align_transcript_with_speakers(self, 
                                     transcript_words: List[TranscriptWord],
                                     speaker_segments: List[SpeakerSegment]) -> List[SpeakerAlignedSegment]:
        """
        Align transcript words with speaker segments.
        
        Args:
            transcript_words: List of transcript words with timing
            speaker_segments: List of speaker segments with timing
            
        Returns:
            List of speaker-aligned segments
        """
        if not transcript_words:
            self.logger.info("No transcript words to align")
            return []
        
        if not speaker_segments:
            self.logger.info("No speaker segments, creating single-speaker alignment")
            return self._create_single_speaker_alignment(transcript_words)
        
        self.logger.info(f"Aligning {len(transcript_words)} words with {len(speaker_segments)} speaker segments")
        
        # Assign each word to a speaker
        word_speaker_assignments = self._assign_words_to_speakers(transcript_words, speaker_segments)
        
        # Group consecutive words by speaker
        aligned_segments = self._group_words_by_speaker(transcript_words, word_speaker_assignments)
        
        self.logger.info(f"Created {len(aligned_segments)} aligned segments")
        
        return aligned_segments
    
    def _assign_words_to_speakers(self, 
                                transcript_words: List[TranscriptWord],
                                speaker_segments: List[SpeakerSegment]) -> List[str]:
        """
        Assign each word to a speaker based on temporal overlap.
        
        Args:
            transcript_words: List of transcript words
            speaker_segments: List of speaker segments
            
        Returns:
            List of speaker IDs for each word
        """
        word_assignments = []
        
        for word in transcript_words:
            assigned_speaker = self._find_best_speaker_for_word(word, speaker_segments)
            word_assignments.append(assigned_speaker)
        
        return word_assignments
    
    def _find_best_speaker_for_word(self, 
                                  word: TranscriptWord,
                                  speaker_segments: List[SpeakerSegment]) -> str:
        """
        Find the best speaker for a word based on temporal overlap.
        
        Args:
            word: Transcript word to assign
            speaker_segments: List of speaker segments
            
        Returns:
            Speaker ID for the word
        """
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        
        word_duration = word.end_time - word.start_time
        
        for segment in speaker_segments:
            overlap = self._calculate_temporal_overlap(word, segment)
            
            if overlap > 0:
                # Calculate overlap ratio relative to word duration
                overlap_ratio = overlap / word_duration if word_duration > 0 else 0.0
                
                if overlap_ratio > best_overlap and overlap_ratio >= self.overlap_threshold:
                    best_overlap = overlap_ratio
                    best_speaker = segment.speaker_id
        
        return best_speaker
    
    def _calculate_temporal_overlap(self, 
                                  word: TranscriptWord,
                                  segment: SpeakerSegment) -> float:
        """
        Calculate temporal overlap between a word and speaker segment.
        
        Args:
            word: Transcript word
            segment: Speaker segment
            
        Returns:
            Overlap duration in seconds
        """
        # Find overlap boundaries
        overlap_start = max(word.start_time, segment.start_time)
        overlap_end = min(word.end_time, segment.end_time)
        
        # Calculate overlap duration
        overlap = max(0.0, overlap_end - overlap_start)
        
        return overlap
    
    def _group_words_by_speaker(self, 
                              transcript_words: List[TranscriptWord],
                              word_assignments: List[str]) -> List[SpeakerAlignedSegment]:
        """
        Group consecutive words by speaker into aligned segments.
        
        Args:
            transcript_words: List of transcript words
            word_assignments: List of speaker assignments for each word
            
        Returns:
            List of speaker-aligned segments
        """
        if not transcript_words or not word_assignments:
            return []
        
        aligned_segments = []
        current_speaker = None
        current_words = []
        
        for word, speaker_id in zip(transcript_words, word_assignments):
            if speaker_id != current_speaker:
                # Speaker changed, finalize current segment
                if current_words:
                    segment = self._create_aligned_segment(current_speaker, current_words)
                    aligned_segments.append(segment)
                
                # Start new segment
                current_speaker = speaker_id
                current_words = [word]
            else:
                # Same speaker, add word to current segment
                current_words.append(word)
        
        # Finalize last segment
        if current_words:
            segment = self._create_aligned_segment(current_speaker, current_words)
            aligned_segments.append(segment)
        
        return aligned_segments
    
    def _create_aligned_segment(self, 
                              speaker_id: str,
                              words: List[TranscriptWord]) -> SpeakerAlignedSegment:
        """
        Create a speaker-aligned segment from a list of words.
        
        Args:
            speaker_id: Speaker ID for the segment
            words: List of words in the segment
            
        Returns:
            SpeakerAlignedSegment object
        """
        if not words:
            raise ValueError("Cannot create segment with no words")
        
        # Calculate segment timing
        start_time = words[0].start_time
        end_time = words[-1].end_time
        
        # Combine text from all words
        text = " ".join(word.text for word in words)
        
        return SpeakerAlignedSegment(
            speaker_id=speaker_id,
            text=text,
            start_time=start_time,
            end_time=end_time,
            words=words.copy()
        )
    
    def _create_single_speaker_alignment(self, 
                                       transcript_words: List[TranscriptWord]) -> List[SpeakerAlignedSegment]:
        """
        Create alignment with single speaker when no diarization available.
        
        Args:
            transcript_words: List of transcript words
            
        Returns:
            List with single speaker-aligned segment
        """
        if not transcript_words:
            return []
        
        # Create one segment with all words assigned to default speaker
        segment = self._create_aligned_segment("SPEAKER_00", transcript_words)
        
        return [segment]
    
    def get_alignment_statistics(self, aligned_segments: List[SpeakerAlignedSegment]) -> Dict[str, Any]:
        """
        Get statistics about the speaker alignment.
        
        Args:
            aligned_segments: List of aligned segments
            
        Returns:
            Dictionary with alignment statistics
        """
        if not aligned_segments:
            return {
                "num_segments": 0,
                "num_speakers": 0,
                "total_duration": 0.0,
                "speaker_word_counts": {},
                "speaker_durations": {}
            }
        
        # Calculate statistics
        total_duration = 0.0
        speaker_word_counts = {}
        speaker_durations = {}
        
        for segment in aligned_segments:
            duration = segment.end_time - segment.start_time
            total_duration += duration
            
            speaker_id = segment.speaker_id
            
            # Update speaker word counts
            word_count = len(segment.words)
            if speaker_id not in speaker_word_counts:
                speaker_word_counts[speaker_id] = 0
            speaker_word_counts[speaker_id] += word_count
            
            # Update speaker durations
            if speaker_id not in speaker_durations:
                speaker_durations[speaker_id] = 0.0
            speaker_durations[speaker_id] += duration
        
        return {
            "num_segments": len(aligned_segments),
            "num_speakers": len(speaker_word_counts),
            "total_duration": total_duration,
            "speaker_word_counts": speaker_word_counts,
            "speaker_durations": speaker_durations,
            "avg_segment_duration": total_duration / len(aligned_segments) if aligned_segments else 0.0
        }
    
    def validate_alignment(self, aligned_segments: List[SpeakerAlignedSegment]) -> List[str]:
        """
        Validate the alignment quality and return any issues found.
        
        Args:
            aligned_segments: List of aligned segments to validate
            
        Returns:
            List of validation error messages (empty if no issues)
        """
        errors = []
        
        if not aligned_segments:
            return ["No aligned segments found"]
        
        # Check for temporal consistency
        for i, segment in enumerate(aligned_segments):
            # Check segment timing
            if segment.start_time >= segment.end_time:
                errors.append(f"Segment {i}: Invalid timing (start >= end)")
            
            # Check word timing consistency
            if segment.words:
                first_word_start = segment.words[0].start_time
                last_word_end = segment.words[-1].end_time
                
                if abs(segment.start_time - first_word_start) > 1.0:
                    errors.append(f"Segment {i}: Large gap between segment and first word timing")
                
                if abs(segment.end_time - last_word_end) > 1.0:
                    errors.append(f"Segment {i}: Large gap between segment and last word timing")
            
            # Check for overlaps with next segment
            if i < len(aligned_segments) - 1:
                next_segment = aligned_segments[i + 1]
                if segment.end_time > next_segment.start_time:
                    overlap = segment.end_time - next_segment.start_time
                    errors.append(f"Segments {i} and {i+1}: Overlap of {overlap:.2f}s")
        
        return errors