"""
Text feature extraction from transcription data for emotional tone detection.
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from ..core.context import TranscriptWord, TranscriptSegment, SpeakerAlignedSegment


class TextFeatureExtractor:
    """
    Extracts text-based features from transcription data for tone detection.
    
    Analyzes speech patterns, sentiment cues, and linguistic markers
    that correlate with emotional tone.
    """
    
    def __init__(self, 
                 sentiment_words_positive: Optional[List[str]] = None,
                 sentiment_words_negative: Optional[List[str]] = None) -> None:
        """
        Initialize the text feature extractor.
        
        Args:
            sentiment_words_positive: List of positive sentiment words
            sentiment_words_negative: List of negative sentiment words
        """
        self.logger = logging.getLogger("wiz.features.text")
        
        # Basic sentiment word lists (can be expanded)
        self.positive_words = sentiment_words_positive or [
            "good", "great", "excellent", "wonderful", "amazing", "fantastic",
            "love", "like", "enjoy", "happy", "excited", "pleased", "glad",
            "perfect", "awesome", "brilliant", "superb", "outstanding",
            "yes", "absolutely", "definitely", "sure", "right", "correct"
        ]
        
        self.negative_words = sentiment_words_negative or [
            "bad", "terrible", "awful", "horrible", "hate", "dislike",
            "angry", "mad", "furious", "upset", "sad", "disappointed",
            "wrong", "incorrect", "no", "never", "not", "cannot", "can't",
            "problem", "issue", "trouble", "difficult", "hard", "impossible"
        ]
        
        # Convert to sets for faster lookup
        self.positive_words_set = set(word.lower() for word in self.positive_words)
        self.negative_words_set = set(word.lower() for word in self.negative_words)
        
        # Intensity words that amplify emotion
        self.intensity_words = {
            "very", "extremely", "really", "quite", "totally", "completely",
            "absolutely", "incredibly", "amazingly", "terribly", "horribly"
        }
    
    def extract_features(self, 
                        transcript_segments: List[TranscriptSegment],
                        aligned_segments: List[SpeakerAlignedSegment],
                        time_window_start: float,
                        time_window_end: float) -> Dict[str, float]:
        """
        Extract text features for a given time window.
        
        Args:
            transcript_segments: List of transcript segments
            aligned_segments: List of speaker-aligned segments
            time_window_start: Start time of analysis window
            time_window_end: End time of analysis window
            
        Returns:
            Dictionary of extracted text features
        """
        # Find segments that overlap with the time window
        window_segments = self._get_segments_in_window(
            aligned_segments, time_window_start, time_window_end
        )
        
        if not window_segments:
            return self._get_empty_features()
        
        # Extract various text features
        features = {}
        
        # Speech rate features
        features.update(self._extract_speech_rate_features(window_segments))
        
        # Sentiment features
        features.update(self._extract_sentiment_features(window_segments))
        
        # Linguistic pattern features
        features.update(self._extract_linguistic_features(window_segments))
        
        # Speaker interaction features
        features.update(self._extract_speaker_features(window_segments))
        
        # Completion and fluency features
        features.update(self._extract_fluency_features(window_segments))
        
        self.logger.debug(f"Extracted {len(features)} text features for window {time_window_start:.1f}-{time_window_end:.1f}s")
        
        return features
    
    def _get_segments_in_window(self, 
                               segments: List[SpeakerAlignedSegment],
                               start_time: float,
                               end_time: float) -> List[SpeakerAlignedSegment]:
        """Get segments that overlap with the time window."""
        window_segments = []
        
        for segment in segments:
            # Check for overlap
            if (segment.start_time < end_time and segment.end_time > start_time):
                window_segments.append(segment)
        
        return window_segments
    
    def _extract_speech_rate_features(self, segments: List[SpeakerAlignedSegment]) -> Dict[str, float]:
        """Extract speech rate and tempo features."""
        if not segments:
            return {"speech_rate_wps": 0.0, "speech_tempo_variance": 0.0}
        
        total_words = 0
        total_duration = 0.0
        segment_rates = []
        
        for segment in segments:
            duration = segment.end_time - segment.start_time
            if duration > 0:
                word_count = len(segment.words)
                rate = word_count / duration  # words per second
                
                total_words += word_count
                total_duration += duration
                segment_rates.append(rate)
        
        # Overall speech rate
        overall_rate = total_words / total_duration if total_duration > 0 else 0.0
        
        # Speech rate variance (indicator of excitement/nervousness)
        rate_variance = np.var(segment_rates) if len(segment_rates) > 1 else 0.0
        
        return {
            "speech_rate_wps": overall_rate,
            "speech_tempo_variance": float(rate_variance)
        }
    
    def _extract_sentiment_features(self, segments: List[SpeakerAlignedSegment]) -> Dict[str, float]:
        """Extract sentiment-based features."""
        total_words = 0
        positive_count = 0
        negative_count = 0
        intensity_count = 0
        
        for segment in segments:
            words = self._extract_words_from_text(segment.text)
            
            for word in words:
                word_lower = word.lower()
                total_words += 1
                
                if word_lower in self.positive_words_set:
                    positive_count += 1
                elif word_lower in self.negative_words_set:
                    negative_count += 1
                
                if word_lower in self.intensity_words:
                    intensity_count += 1
        
        if total_words == 0:
            return {
                "sentiment_positive_ratio": 0.0,
                "sentiment_negative_ratio": 0.0,
                "sentiment_polarity": 0.0,
                "intensity_word_ratio": 0.0
            }
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        polarity = positive_ratio - negative_ratio  # Range: -1 to +1
        intensity_ratio = intensity_count / total_words
        
        return {
            "sentiment_positive_ratio": positive_ratio,
            "sentiment_negative_ratio": negative_ratio,
            "sentiment_polarity": polarity,
            "intensity_word_ratio": intensity_ratio
        }
    
    def _extract_linguistic_features(self, segments: List[SpeakerAlignedSegment]) -> Dict[str, float]:
        """Extract linguistic pattern features."""
        total_sentences = 0
        question_count = 0
        exclamation_count = 0
        interruption_count = 0
        
        for segment in segments:
            text = segment.text
            
            # Count sentences (rough approximation)
            sentence_markers = re.findall(r'[.!?]+', text)
            total_sentences += len(sentence_markers)
            
            # Count questions
            question_count += text.count('?')
            
            # Count exclamations
            exclamation_count += text.count('!')
            
            # Count potential interruptions (incomplete sentences)
            if text.endswith('...') or text.endswith('-'):
                interruption_count += 1
        
        # Normalize by number of segments
        num_segments = len(segments) if segments else 1
        
        return {
            "question_density": question_count / num_segments,
            "exclamation_density": exclamation_count / num_segments,
            "interruption_rate": interruption_count / num_segments,
            "sentence_completion_rate": total_sentences / num_segments if num_segments > 0 else 0.0
        }
    
    def _extract_speaker_features(self, segments: List[SpeakerAlignedSegment]) -> Dict[str, float]:
        """Extract speaker interaction features."""
        if not segments:
            return {"speaker_count": 0.0, "speaker_turn_rate": 0.0}
        
        unique_speakers = set(segment.speaker_id for segment in segments)
        num_speakers = len(unique_speakers)
        
        # Count speaker turns
        speaker_turns = 0
        prev_speaker = None
        
        for segment in segments:
            if prev_speaker is not None and segment.speaker_id != prev_speaker:
                speaker_turns += 1
            prev_speaker = segment.speaker_id
        
        # Calculate turn rate per minute
        total_duration = segments[-1].end_time - segments[0].start_time
        turn_rate = (speaker_turns / (total_duration / 60)) if total_duration > 0 else 0.0
        
        return {
            "speaker_count": float(num_speakers),
            "speaker_turn_rate": turn_rate
        }
    
    def _extract_fluency_features(self, segments: List[SpeakerAlignedSegment]) -> Dict[str, float]:
        """Extract speech fluency and hesitation features."""
        total_segments = len(segments)
        if total_segments == 0:
            return {"hesitation_rate": 0.0, "filler_word_ratio": 0.0}
        
        hesitation_markers = 0
        filler_words = 0
        total_words = 0
        
        # Common filler words and hesitation markers
        fillers = {"um", "uh", "er", "ah", "like", "you know", "sort of", "kind of"}
        hesitation_patterns = ["...", "--", "uh", "um", "er"]
        
        for segment in segments:
            text = segment.text.lower()
            
            # Count hesitation markers
            for pattern in hesitation_patterns:
                hesitation_markers += text.count(pattern)
            
            # Count filler words
            words = self._extract_words_from_text(text)
            total_words += len(words)
            
            for word in words:
                if word.lower() in fillers:
                    filler_words += 1
        
        hesitation_rate = hesitation_markers / total_segments
        filler_ratio = filler_words / total_words if total_words > 0 else 0.0
        
        return {
            "hesitation_rate": hesitation_rate,
            "filler_word_ratio": filler_ratio
        }
    
    def _extract_words_from_text(self, text: str) -> List[str]:
        """Extract words from text, removing punctuation."""
        # Simple word extraction
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return empty feature set when no data is available."""
        return {
            "speech_rate_wps": 0.0,
            "speech_tempo_variance": 0.0,
            "sentiment_positive_ratio": 0.0,
            "sentiment_negative_ratio": 0.0,
            "sentiment_polarity": 0.0,
            "intensity_word_ratio": 0.0,
            "question_density": 0.0,
            "exclamation_density": 0.0,
            "interruption_rate": 0.0,
            "sentence_completion_rate": 0.0,
            "speaker_count": 0.0,
            "speaker_turn_rate": 0.0,
            "hesitation_rate": 0.0,
            "filler_word_ratio": 0.0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names extracted by this class."""
        return list(self._get_empty_features().keys())