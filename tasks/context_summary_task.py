"""
Context summary task for generating scene-level summaries using local LLM.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from ..core.base_task import BaseTask
from ..core.context import (
    PipelineContext, SceneSummary, ToneEvent, TranscriptSegment, 
    SpeakerAlignedSegment, SpeakerSegment, VisualEmbedding
)
from ..models.local_llm import LocalLLM


class SceneDataExtractor:
    """
    Extracts and structures scene-level data from pipeline outputs.
    
    Collects multimodal features for each scene without including
    raw embeddings or unstructured data.
    """
    
    def __init__(self) -> None:
        """Initialize the scene data extractor."""
        self.logger = logging.getLogger("wiz.tasks.context_summary.data_extractor")
    
    def extract_scene_data(self, 
                          context: PipelineContext,
                          scene_start: float,
                          scene_end: float,
                          scene_id: str) -> Dict[str, Any]:
        """
        Extract structured scene-level data from pipeline context.
        
        Args:
            context: Pipeline context containing all processing results
            scene_start: Scene start time in seconds
            scene_end: Scene end time in seconds
            scene_id: Unique scene identifier
            
        Returns:
            Dictionary of structured scene data
        """
        scene_data = {
            "scene_id": scene_id,
            "start_time": scene_start,
            "end_time": scene_end,
            "duration": scene_end - scene_start
        }
        
        # Extract transcript data
        transcript_data = self._extract_transcript_data(
            context.transcript_segments, 
            context.aligned_segments, 
            scene_start, scene_end
        )
        scene_data.update(transcript_data)
        
        # Extract speaker data
        speaker_data = self._extract_speaker_data(
            context.speaker_segments, 
            context.aligned_segments,
            scene_start, scene_end
        )
        scene_data.update(speaker_data)
        
        # Extract tone data
        tone_data = self._extract_tone_data(context.tone_events, scene_start, scene_end)
        scene_data.update(tone_data)
        
        # Extract visual features (structured, not raw embeddings)
        visual_data = self._extract_visual_features(
            context.visual_embeddings, 
            scene_start, scene_end
        )
        scene_data.update(visual_data)
        
        return scene_data
    
    def _extract_transcript_data(self, 
                               transcript_segments: List[TranscriptSegment],
                               aligned_segments: List[SpeakerAlignedSegment],
                               start_time: float,
                               end_time: float) -> Dict[str, Any]:
        """Extract transcript-related features for the scene."""
        
        # Find overlapping segments
        scene_segments = []
        for segment in aligned_segments:
            if segment.start_time < end_time and segment.end_time > start_time:
                scene_segments.append(segment)
        
        if not scene_segments:
            return {
                "transcript_excerpt": "",
                "word_count": 0,
                "speech_rate": 0.0,
                "avg_word_confidence": 0.0
            }
        
        # Combine transcript text
        combined_text = " ".join(segment.text for segment in scene_segments)
        transcript_excerpt = combined_text[:300] + "..." if len(combined_text) > 300 else combined_text
        
        # Calculate speech metrics
        total_words = sum(len(segment.words) for segment in scene_segments)
        total_duration = sum(segment.end_time - segment.start_time for segment in scene_segments)
        speech_rate = total_words / total_duration if total_duration > 0 else 0.0
        
        # Calculate average word confidence
        all_confidences = []
        for segment in scene_segments:
            all_confidences.extend(word.confidence for word in segment.words)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        return {
            "transcript_excerpt": transcript_excerpt,
            "word_count": total_words,
            "speech_rate": speech_rate,
            "avg_word_confidence": avg_confidence
        }
    
    def _extract_speaker_data(self, 
                            speaker_segments: List[SpeakerSegment],
                            aligned_segments: List[SpeakerAlignedSegment],
                            start_time: float,
                            end_time: float) -> Dict[str, Any]:
        """Extract speaker-related features for the scene."""
        
        # Find speakers active in this scene
        active_speakers = set()
        speaker_durations = {}
        overlapping_time = 0.0
        
        scene_speaker_segments = []
        for segment in speaker_segments:
            if segment.start_time < end_time and segment.end_time > start_time:
                scene_speaker_segments.append(segment)
                active_speakers.add(segment.speaker_id)
                
                # Calculate duration within scene
                segment_start = max(segment.start_time, start_time)
                segment_end = min(segment.end_time, end_time)
                duration = segment_end - segment_start
                
                if segment.speaker_id not in speaker_durations:
                    speaker_durations[segment.speaker_id] = 0.0
                speaker_durations[segment.speaker_id] += duration
        
        # Calculate speaker overlap
        scene_duration = end_time - start_time
        total_speech_time = sum(speaker_durations.values())
        overlap_ratio = max(0, (total_speech_time - scene_duration) / scene_duration) if scene_duration > 0 else 0.0
        
        # Identify dominant speakers
        dominant_speakers = []
        if speaker_durations:
            max_duration = max(speaker_durations.values())
            dominant_speakers = [
                speaker for speaker, duration in speaker_durations.items()
                if duration >= max_duration * 0.5  # Speakers with >50% of max speaking time
            ]
        
        return {
            "active_speakers": list(active_speakers),
            "dominant_speakers": dominant_speakers,
            "speaker_count": len(active_speakers),
            "speaker_overlap_ratio": overlap_ratio,
            "speaker_durations": speaker_durations
        }
    
    def _extract_tone_data(self, 
                          tone_events: List[ToneEvent],
                          start_time: float,
                          end_time: float) -> Dict[str, Any]:
        """Extract tone-related features for the scene."""
        
        # Find tone events overlapping with scene
        scene_tone_events = []
        for event in tone_events:
            if event.start_time < end_time and event.end_time > start_time:
                scene_tone_events.append(event)
        
        if not scene_tone_events:
            return {
                "tone_label": "neutral",
                "tone_confidence": 0.5,
                "tone_stability": 1.0  # Default to stable if no data
            }
        
        # Determine dominant tone
        tone_weights = {}
        total_weight = 0.0
        
        for event in scene_tone_events:
            # Weight by overlap duration and confidence
            overlap_start = max(event.start_time, start_time)
            overlap_end = min(event.end_time, end_time)
            overlap_duration = overlap_end - overlap_start
            weight = overlap_duration * event.confidence
            
            if event.tone_label not in tone_weights:
                tone_weights[event.tone_label] = 0.0
            tone_weights[event.tone_label] += weight
            total_weight += weight
        
        # Get dominant tone
        dominant_tone = max(tone_weights.items(), key=lambda x: x[1])[0] if tone_weights else "neutral"
        dominant_confidence = tone_weights[dominant_tone] / total_weight if total_weight > 0 else 0.5
        
        # Calculate tone stability (lower variance = more stable)
        if len(scene_tone_events) > 1:
            unique_tones = len(set(event.tone_label for event in scene_tone_events))
            tone_stability = 1.0 / unique_tones  # More unique tones = less stable
        else:
            tone_stability = 1.0
        
        return {
            "tone_label": dominant_tone,
            "tone_confidence": dominant_confidence,
            "tone_stability": tone_stability,
            "all_scene_tones": list(tone_weights.keys())
        }
    
    def _extract_visual_features(self, 
                               visual_embeddings: List[VisualEmbedding],
                               start_time: float,
                               end_time: float) -> Dict[str, Any]:
        """Extract structured visual features (not raw embeddings)."""
        
        # Find visual embeddings overlapping with scene
        scene_embeddings = []
        for embedding in visual_embeddings:
            if embedding.start_time < end_time and embedding.end_time > start_time:
                scene_embeddings.append(embedding)
        
        if not scene_embeddings:
            return {
                "motion_intensity": 0.0,
                "visual_change_intensity": 0.0,
                "shot_count": 0
            }
        
        # Calculate motion intensity (based on embedding changes)
        motion_values = []
        for i in range(len(scene_embeddings) - 1):
            emb1 = scene_embeddings[i].embedding
            emb2 = scene_embeddings[i + 1].embedding
            
            # Calculate similarity/distance as motion proxy
            if emb1.shape == emb2.shape:
                # Simple L2 distance as motion indicator
                import numpy as np
                motion = float(np.linalg.norm(emb2 - emb1))
                motion_values.append(motion)
        
        motion_intensity = sum(motion_values) / len(motion_values) if motion_values else 0.0
        
        # Visual change intensity (variance in embeddings)
        if len(scene_embeddings) > 1:
            import numpy as np
            all_norms = [float(np.linalg.norm(emb.embedding)) for emb in scene_embeddings]
            visual_change_intensity = float(np.var(all_norms))
        else:
            visual_change_intensity = 0.0
        
        return {
            "motion_intensity": motion_intensity,
            "visual_change_intensity": visual_change_intensity,
            "shot_count": len(scene_embeddings)
        }


class PromptFormatter:
    """
    Formats structured scene data into LLM prompts.
    
    Creates deterministic, structured prompts for scene summarization
    without including raw data or embeddings.
    """
    
    def __init__(self) -> None:
        """Initialize the prompt formatter."""
        self.logger = logging.getLogger("wiz.tasks.context_summary.prompt_formatter")
    
    def format_scene_prompt(self, scene_data: Dict[str, Any]) -> str:
        """
        Format scene data into a structured LLM prompt.
        
        Args:
            scene_data: Structured scene data dictionary
            
        Returns:
            Formatted prompt string for LLM
        """
        # Format time range
        start_time = scene_data["start_time"]
        end_time = scene_data["end_time"]
        time_str = f"{start_time:.1f}–{end_time:.1f}s"
        
        # Format speakers
        dominant_speakers = scene_data.get("dominant_speakers", [])
        if dominant_speakers:
            speakers_str = ", ".join(dominant_speakers)
        else:
            speakers_str = "None identified"
        
        # Format speech rate
        speech_rate = scene_data.get("speech_rate", 0.0)
        speech_rate_desc = self._describe_speech_rate(speech_rate)
        
        # Format motion intensity
        motion_intensity = scene_data.get("motion_intensity", 0.0)
        motion_desc = self._describe_motion_intensity(motion_intensity)
        
        # Format tone information
        tone_label = scene_data.get("tone_label", "neutral")
        tone_confidence = scene_data.get("tone_confidence", 0.5)
        
        # Get transcript excerpt
        transcript_excerpt = scene_data.get("transcript_excerpt", "No transcript available.")
        
        # Format speaker overlap
        overlap_ratio = scene_data.get("speaker_overlap_ratio", 0.0)
        overlap_desc = self._describe_speaker_overlap(overlap_ratio)
        
        # Construct structured prompt
        prompt = f"""You are an editorial assistant analyzing a video scene.

Scene Time: {time_str}
Tone: {tone_label} (confidence: {tone_confidence:.2f})
Primary Speakers: {speakers_str}
Speech Rate: {speech_rate_desc} ({speech_rate:.1f} words/second)
Motion Intensity: {motion_desc}
Speaker Interaction: {overlap_desc}
Transcript Excerpt:
{transcript_excerpt}

Write a concise 2–3 sentence summary describing:
- What is happening in the scene
- The emotional tone and atmosphere
- Key speaker dynamics or interactions

Be objective and editorial. Focus on observable patterns."""
        
        self.logger.debug(f"Formatted prompt for scene {scene_data['scene_id']}")
        return prompt
    
    def _describe_speech_rate(self, rate: float) -> str:
        """Convert speech rate to descriptive text."""
        if rate < 1.0:
            return "Very slow"
        elif rate < 2.0:
            return "Slow"
        elif rate < 3.0:
            return "Normal"
        elif rate < 4.0:
            return "Fast"
        else:
            return "Very fast"
    
    def _describe_motion_intensity(self, intensity: float) -> str:
        """Convert motion intensity to descriptive text."""
        if intensity < 0.5:
            return "Low visual activity"
        elif intensity < 1.5:
            return "Moderate visual activity" 
        elif intensity < 3.0:
            return "High visual activity"
        else:
            return "Very high visual activity"
    
    def _describe_speaker_overlap(self, ratio: float) -> str:
        """Convert speaker overlap ratio to descriptive text."""
        if ratio < 0.05:
            return "Clear turn-taking"
        elif ratio < 0.15:
            return "Minimal overlap"
        elif ratio < 0.3:
            return "Moderate overlap"
        else:
            return "Significant interruption/overlap"


class ContextSummaryTask(BaseTask):
    """
    Task for generating scene-level summaries using local LLM.
    
    Collects structured scene data from pipeline outputs, formats prompts,
    and generates editorial-quality summaries for each scene.
    """
    
    def __init__(self, 
                 local_llm: LocalLLM,
                 scene_duration_seconds: float = 30.0,
                 min_scene_duration_seconds: float = 10.0,
                 max_tokens: int = 150) -> None:
        """
        Initialize the context summary task.
        
        Args:
            local_llm: Configured LocalLLM instance
            scene_duration_seconds: Target scene duration in seconds
            min_scene_duration_seconds: Minimum scene duration
            max_tokens: Maximum tokens for LLM generation
        """
        super().__init__("ContextSummary")
        self.local_llm = local_llm
        self.scene_duration = scene_duration_seconds
        self.min_scene_duration = min_scene_duration_seconds
        self.max_tokens = max_tokens
        
        # Initialize helper components
        self.data_extractor = SceneDataExtractor()
        self.prompt_formatter = PromptFormatter()
    
    def _run(self, context: PipelineContext) -> None:
        """
        Execute context summarization on structured pipeline data.
        
        Args:
            context: Pipeline context containing multimodal processing results
        """
        # Validate required data
        self._validate_context_data(context)
        
        # Determine scene boundaries
        scenes = self._create_scene_boundaries(context)
        
        self.logger.info(f"Generating summaries for {len(scenes)} scenes")
        
        # Process each scene
        scene_summaries = []
        
        for i, (start_time, end_time) in enumerate(scenes):
            scene_id = f"scene_{i:03d}"
            
            self.logger.debug(f"Processing {scene_id}: {start_time:.1f}-{end_time:.1f}s")
            
            try:
                # Extract structured scene data
                scene_data = self.data_extractor.extract_scene_data(
                    context, start_time, end_time, scene_id
                )
                
                # Format prompt
                prompt = self.prompt_formatter.format_scene_prompt(scene_data)
                
                # Generate summary using LLM
                summary_text = self.local_llm.generate(prompt, max_tokens=self.max_tokens)
                
                # Create scene summary object
                scene_summary = SceneSummary(
                    scene_id=scene_id,
                    start_time=start_time,
                    end_time=end_time,
                    summary_text=summary_text.strip(),
                    tone_label=scene_data.get("tone_label", "neutral"),
                    key_speakers=scene_data.get("dominant_speakers", []),
                    confidence=scene_data.get("tone_confidence", 0.5)
                )
                
                scene_summaries.append(scene_summary)
                
                self.logger.debug(f"Generated summary for {scene_id}: {len(summary_text)} characters")
                
            except Exception as e:
                self.logger.error(f"Failed to process {scene_id}: {e}")
                
                # Create fallback summary
                fallback_summary = SceneSummary(
                    scene_id=scene_id,
                    start_time=start_time,
                    end_time=end_time,
                    summary_text=f"Scene summary generation failed: {str(e)}",
                    tone_label="neutral",
                    key_speakers=[],
                    confidence=0.0
                )
                scene_summaries.append(fallback_summary)
        
        # Store results in context
        context.scene_summaries.extend(scene_summaries)
        
        # Store metadata
        model_info = self.local_llm.get_model_info()
        context.processing_metadata['context_summary'] = {
            'model_info': model_info,
            'scene_config': {
                'scene_duration_seconds': self.scene_duration,
                'min_scene_duration_seconds': self.min_scene_duration,
                'max_tokens': self.max_tokens,
                'total_scenes': len(scenes)
            },
            'summary_stats': {
                'successful_summaries': len([s for s in scene_summaries if not s.summary_text.startswith("Scene summary generation failed")]),
                'failed_summaries': len([s for s in scene_summaries if s.summary_text.startswith("Scene summary generation failed")]),
                'avg_summary_length': sum(len(s.summary_text) for s in scene_summaries) / len(scene_summaries) if scene_summaries else 0
            }
        }
        
        self.logger.info(f"Context summarization completed: {len(scene_summaries)} summaries generated")
    
    def _validate_context_data(self, context: PipelineContext) -> None:
        """Validate that required data is available in the context."""
        if context.video_metadata is None:
            raise ValueError("Video metadata not available in context")
        
        # Speech processing is required for meaningful summaries
        if not context.transcript_segments:
            self.logger.warning("No transcript segments available - summaries will be limited")
        
        if not context.aligned_segments:
            self.logger.warning("No speaker-aligned segments available - speaker analysis will be limited")
    
    def _create_scene_boundaries(self, context: PipelineContext) -> List[Tuple[float, float]]:
        """Create scene time boundaries for summarization."""
        if context.video_metadata is None:
            raise ValueError("Video metadata required for scene boundary creation")
        
        total_duration = context.video_metadata.duration_seconds
        scenes = []
        
        current_start = 0.0
        while current_start < total_duration:
            scene_end = min(current_start + self.scene_duration, total_duration)
            
            # Only add scenes that meet minimum duration
            if scene_end - current_start >= self.min_scene_duration:
                scenes.append((current_start, scene_end))
            
            current_start = scene_end
        
        return scenes
    
    @classmethod
    def create_default(cls, 
                      llm_backend: str = "mock",
                      model_path: Optional[str] = None,
                      scene_duration_seconds: float = 30.0) -> 'ContextSummaryTask':
        """
        Create a ContextSummaryTask with default configuration.
        
        Args:
            llm_backend: LLM backend type ("mock", "llama_cpp", "mlx")
            model_path: Path to model file (for llama_cpp)
            scene_duration_seconds: Target scene duration
            
        Returns:
            Configured ContextSummaryTask instance
        """
        local_llm = LocalLLM(backend=llm_backend, model_path=model_path)
        
        return cls(
            local_llm=local_llm,
            scene_duration_seconds=scene_duration_seconds,
            max_tokens=150
        )