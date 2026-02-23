"""
WIZ Intelligence Pipeline

A modular, object-oriented pipeline for detecting blink, breath, and speech events
from video input using local ML inference.
"""

from .core.pipeline import Pipeline
from .core.context import (
    PipelineContext, VideoMetadata, BlinkEvent, BreathEvent,
    TranscriptWord, TranscriptSegment, SpeakerSegment, SpeakerAlignedSegment,
    VisualEmbedding, ToneEvent, SceneSummary
)
from .models.blink_detector import BlinkDetector
from .models.breath_detector import BreathDetector
from .models.speaker_identity import SpeakerIdentityRegistry
from .models.whisper_model import WhisperModel
from .models.diarization_model import DiarizationModel
from .tasks.blink_task import BlinkTask
from .tasks.breath_task import BreathTask
from .tasks.transcription_task import TranscriptionTask
from .tasks.diarization_task import DiarizationTask
from .tasks.alignment_task import AlignmentTask
from .tasks.tone_detection_task import ToneDetectionTask
from .tasks.context_summary_task import ContextSummaryTask
from .models.tone_classifier import ToneClassifier
from .models.local_llm import LocalLLM
from .features.text_features import TextFeatureExtractor
from .features.audio_features import AudioFeatureExtractor
from .features.visual_features import VisualFeatureExtractor
from .audio.audio_extractor import AudioExtractor
from .audio.speaker_alignment import SpeakerAligner

__version__ = "1.1.0"
__author__ = "WIZ Intelligence Team"

__all__ = [
    "Pipeline",
    "PipelineContext", 
    "VideoMetadata",
    "BlinkEvent",
    "BreathEvent",
    "TranscriptWord",
    "TranscriptSegment", 
    "SpeakerSegment",
    "SpeakerAlignedSegment",
    "VisualEmbedding",
    "ToneEvent",
    "SceneSummary",
    "BlinkDetector",
    "BreathDetector",
    "SpeakerIdentityRegistry",
    "WhisperModel",
    "DiarizationModel",
    "ToneClassifier",
    "LocalLLM",
    "BlinkTask",
    "BreathTask",
    "TranscriptionTask",
    "DiarizationTask",
    "AlignmentTask",
    "ToneDetectionTask",
    "ContextSummaryTask",
    "TextFeatureExtractor",
    "AudioFeatureExtractor", 
    "VisualFeatureExtractor",
    "AudioExtractor",
    "SpeakerAligner"
]