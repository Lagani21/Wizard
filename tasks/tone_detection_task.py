"""
Tone detection task for the WIZ Intelligence Pipeline.
"""

import numpy as np
from typing import List, Dict, Any
from ..core.base_task import BaseTask
from ..core.context import PipelineContext, ToneEvent, VisualEmbedding
from ..models.tone_classifier import ToneClassifier
from ..features.text_features import TextFeatureExtractor
from ..features.audio_features import AudioFeatureExtractor
from ..features.visual_features import VisualFeatureExtractor


class ToneDetectionTask(BaseTask):
    """
    Task for detecting emotional tone using multimodal features.
    
    Combines text, audio, and visual features to classify emotional tone
    in fixed time windows and stores results in the pipeline context.
    """
    
    def __init__(self, 
                 tone_classifier: ToneClassifier,
                 text_extractor: TextFeatureExtractor = None,
                 audio_extractor: AudioFeatureExtractor = None,
                 visual_extractor: VisualFeatureExtractor = None,
                 window_size_seconds: float = 8.0,
                 window_overlap_seconds: float = 2.0) -> None:
        """
        Initialize the tone detection task.
        
        Args:
            tone_classifier: Configured ToneClassifier instance
            text_extractor: Text feature extractor (creates default if None)
            audio_extractor: Audio feature extractor (creates default if None)
            visual_extractor: Visual feature extractor (creates default if None)
            window_size_seconds: Size of analysis windows in seconds
            window_overlap_seconds: Overlap between windows in seconds
        """
        super().__init__("ToneDetection")
        self.tone_classifier = tone_classifier
        self.window_size = window_size_seconds
        self.window_overlap = window_overlap_seconds
        
        # Initialize feature extractors
        self.text_extractor = text_extractor if text_extractor is not None else TextFeatureExtractor()
        self.audio_extractor = audio_extractor if audio_extractor is not None else AudioFeatureExtractor()
        self.visual_extractor = visual_extractor if visual_extractor is not None else VisualFeatureExtractor()
    
    def _run(self, context: PipelineContext) -> None:
        """
        Execute tone detection on multimodal data.
        
        Args:
            context: Pipeline context containing multimodal data
        """
        # Validate required data is available
        self._validate_context_data(context)
        
        # Create placeholder visual embeddings if not available
        if not context.visual_embeddings and context.video_metadata:
            self.logger.info("Creating placeholder visual embeddings...")
            context.visual_embeddings = self.visual_extractor.create_placeholder_embeddings(
                context.video_metadata.duration_seconds
            )
        
        # Determine analysis windows
        windows = self._create_analysis_windows(context)
        
        self.logger.info(f"Analyzing {len(windows)} time windows for tone detection")
        
        # Process each window
        tone_events = []
        
        for i, (start_time, end_time) in enumerate(windows):
            self.logger.debug(f"Processing window {i+1}/{len(windows)}: {start_time:.1f}-{end_time:.1f}s")
            
            # Extract features for this window
            features = self._extract_multimodal_features(context, start_time, end_time)
            
            # Classify tone
            tone_label, confidence = self.tone_classifier.predict(features)
            
            # Create tone event
            tone_event = ToneEvent(
                scene_id=f"scene_{i:03d}",
                start_time=start_time,
                end_time=end_time,
                tone_label=tone_label,
                confidence=confidence
            )
            
            tone_events.append(tone_event)
            
            self.logger.debug(f"Window {i+1}: {tone_label} (confidence: {confidence:.3f})")
        
        # Store results in context
        context.tone_events.extend(tone_events)
        
        # Generate and store statistics
        stats = self._calculate_tone_statistics(tone_events)
        
        context.processing_metadata['tone_detection'] = {
            'classifier_info': self.tone_classifier.get_model_info(),
            'window_config': {
                'window_size_seconds': self.window_size,
                'window_overlap_seconds': self.window_overlap,
                'total_windows': len(windows)
            },
            'detection_stats': stats,
            'total_events': len(tone_events)
        }
        
        # Log results
        self.logger.info(f"Tone detection completed: {len(tone_events)} events")
        
        # Log tone distribution
        if stats['tone_distribution']:
            self.logger.info("Tone distribution:")
            for tone, count in stats['tone_distribution'].items():
                percentage = (count / len(tone_events)) * 100 if tone_events else 0
                avg_conf = stats['avg_confidence_by_tone'].get(tone, 0)
                self.logger.info(f"  {tone}: {count} events ({percentage:.1f}%, avg conf: {avg_conf:.3f})")
    
    def _validate_context_data(self, context: PipelineContext) -> None:
        """Validate that required data is available in the context."""
        if context.audio_waveform is None:
            raise ValueError("Audio waveform not available in context")
        
        if not context.transcript_segments and not context.aligned_segments:
            self.logger.warning("No transcript data available - text features will be limited")
        
        if not context.speaker_segments:
            self.logger.warning("No speaker segments available - speaker features will be limited")
        
        if context.video_metadata is None:
            raise ValueError("Video metadata not available in context")
    
    def _create_analysis_windows(self, context: PipelineContext) -> List[tuple[float, float]]:
        """Create time windows for analysis."""
        if context.video_metadata is None:
            raise ValueError("Video metadata required for window creation")
        
        total_duration = context.video_metadata.duration_seconds
        step_size = self.window_size - self.window_overlap
        
        windows = []
        current_start = 0.0
        
        while current_start < total_duration:
            window_end = min(current_start + self.window_size, total_duration)
            
            # Only add windows that are at least 50% of the target size
            if window_end - current_start >= self.window_size * 0.5:
                windows.append((current_start, window_end))
            
            current_start += step_size
        
        return windows
    
    def _extract_multimodal_features(self, 
                                   context: PipelineContext,
                                   start_time: float,
                                   end_time: float) -> Dict[str, float]:
        """Extract multimodal features for a time window."""
        all_features = {}
        
        # Extract text features
        try:
            text_features = self.text_extractor.extract_features(
                context.transcript_segments,
                context.aligned_segments,
                start_time,
                end_time
            )
            all_features.update(text_features)
            self.logger.debug(f"Extracted {len(text_features)} text features")
        except Exception as e:
            self.logger.warning(f"Failed to extract text features: {e}")
        
        # Extract audio features
        try:
            audio_features = self.audio_extractor.extract_features(
                context.audio_waveform,
                context.speaker_segments,
                start_time,
                end_time
            )
            all_features.update(audio_features)
            self.logger.debug(f"Extracted {len(audio_features)} audio features")
        except Exception as e:
            self.logger.warning(f"Failed to extract audio features: {e}")
        
        # Extract visual features
        try:
            visual_features = self.visual_extractor.extract_features(
                context.visual_embeddings,
                start_time,
                end_time
            )
            all_features.update(visual_features)
            self.logger.debug(f"Extracted {len(visual_features)} visual features")
        except Exception as e:
            self.logger.warning(f"Failed to extract visual features: {e}")
        
        return all_features
    
    def _calculate_tone_statistics(self, tone_events: List[ToneEvent]) -> Dict[str, Any]:
        """Calculate statistics about tone detection results."""
        if not tone_events:
            return {
                'tone_distribution': {},
                'avg_confidence_by_tone': {},
                'overall_avg_confidence': 0.0,
                'dominant_tone': None,
                'tone_changes': 0
            }
        
        # Tone distribution
        tone_counts = {}
        tone_confidences = {}
        
        for event in tone_events:
            tone = event.tone_label
            
            if tone not in tone_counts:
                tone_counts[tone] = 0
                tone_confidences[tone] = []
            
            tone_counts[tone] += 1
            tone_confidences[tone].append(event.confidence)
        
        # Average confidence by tone
        avg_confidence_by_tone = {}
        for tone, confidences in tone_confidences.items():
            avg_confidence_by_tone[tone] = np.mean(confidences)
        
        # Overall statistics
        overall_avg_confidence = np.mean([event.confidence for event in tone_events])
        
        # Dominant tone
        dominant_tone = max(tone_counts.items(), key=lambda x: x[1])[0] if tone_counts else None
        
        # Count tone changes
        tone_changes = 0
        if len(tone_events) > 1:
            for i in range(1, len(tone_events)):
                if tone_events[i].tone_label != tone_events[i-1].tone_label:
                    tone_changes += 1
        
        return {
            'tone_distribution': tone_counts,
            'avg_confidence_by_tone': avg_confidence_by_tone,
            'overall_avg_confidence': float(overall_avg_confidence),
            'dominant_tone': dominant_tone,
            'tone_changes': tone_changes
        }
    
    @classmethod
    def create_default(cls, 
                      classifier_type: str = "rule_based",
                      window_size_seconds: float = 8.0) -> 'ToneDetectionTask':
        """
        Create a ToneDetectionTask with default configuration.
        
        Args:
            classifier_type: Type of classifier ("rule_based" or "mlp")
            window_size_seconds: Analysis window size
            
        Returns:
            Configured ToneDetectionTask instance
        """
        tone_classifier = ToneClassifier(classifier_type=classifier_type)
        
        return cls(
            tone_classifier=tone_classifier,
            window_size_seconds=window_size_seconds,
            window_overlap_seconds=window_size_seconds * 0.25  # 25% overlap
        )