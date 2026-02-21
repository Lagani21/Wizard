# WIZ Intelligence Pipeline

A modular, object-oriented pipeline for multimodal analysis including blink detection, breath detection, speech processing, and emotional tone detection using local ML inference.

## Overview

The WIZ Intelligence Pipeline is designed to process video files and detect multiple types of events:

### Core Detection
1. **Blink Detection**: Detects eye blink events using MediaPipe Face Mesh and Eye Aspect Ratio (EAR) calculations
2. **Breath Detection**: Detects breath sounds from audio using simple heuristic analysis

### Speech Processing (Optional)
3. **Speech Transcription**: Converts speech to text with word-level timestamps using Whisper
4. **Speaker Diarization**: Identifies "who spoke when" using Pyannote.audio
5. **Speaker-Transcript Alignment**: Merges transcription with speaker identification

### Emotional Tone Detection (Optional)
6. **Multimodal Tone Classification**: Analyzes emotional tone using text, audio, and visual features
   - Supports 6 tone categories: calm, tense, excited, somber, neutral, confrontational
   - Rule-based classifier with optional MLP architecture
   - Combines speech patterns, acoustic features, and visual cues

### AI Scene Summaries (Optional)
7. **Context-Aware Scene Summarization**: Generates editorial-quality scene summaries using local LLM
   - Analyzes structured multimodal data (transcript, tone, speaker interactions)
   - Supports multiple LLM backends: Mock, llama.cpp, Apple MLX
   - Creates narrative analysis and scene-level descriptions

## Key Features

- **Local Processing Only**: All inference runs locally on Apple Silicon (CPU/Metal)
- **No External Dependencies**: No cloud APIs, CUDA, or external web requests
- **Independent Detection**: Two standalone detectors with no cross-signal reasoning
- **Simple Heuristics**: Clean detection algorithms without complex fusion or scoring
- **Object-Oriented Architecture**: Modular design with single responsibility classes
- **Deterministic Execution**: Reproducible results with no global state
- **Type-Safe**: Full type hints and explicit return types

## Architecture

### Core Components

- `Pipeline`: Main entry point and task coordinator
- `PipelineContext`: Shared state container for all tasks
- `BaseTask`: Abstract base class for all processing tasks

### Detection Models

- `BlinkDetector`: MediaPipe-based blink detection using EAR
- `BreathDetector`: Heuristic breath detection from audio segments
- `ToneClassifier`: Multimodal emotional tone classifier

### Feature Extractors

- `TextFeatureExtractor`: Speech patterns, sentiment, speaker interactions
- `AudioFeatureExtractor`: Volume, spectral, prosodic, and temporal features
- `VisualFeatureExtractor`: Motion patterns, scene intensity, visual stability

### Tasks

- `BlinkTask`: Processes video frames for blink detection
- `BreathTask`: Processes audio waveform for breath detection
- `ToneDetectionTask`: Analyzes multimodal features for emotional tone classification

## Installation

### Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+

### Setup

```bash
# Clone/download the project
cd WIZ-Intelligence-Pipeline

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from core.pipeline import Pipeline

# Create pipeline with default configuration
pipeline = Pipeline()

# Process video file
context = pipeline.run("input_video.mp4")

# Access results
print(f"Blinks detected: {len(context.blink_events)}")
print(f"Breaths detected: {len(context.breath_events)}")
```

### With Speech Processing

```python
from core.pipeline import Pipeline

# Create pipeline with speech processing enabled
pipeline = Pipeline(enable_speech_processing=True)

results = pipeline.run("input_video.mp4")

# Access speech processing results
print(f"Transcript words: {results['speech_processing']['transcription']['total_words']}")
print(f"Unique speakers: {results['speech_processing']['diarization']['num_speakers']}")
```

### With Emotional Tone Detection

```python
from core.pipeline import Pipeline

# Create pipeline with tone detection enabled (requires speech processing)
pipeline = Pipeline(
    enable_speech_processing=True,
    enable_tone_detection=True
)

results = pipeline.run("input_video.mp4")

# Access tone detection results
tone_events = results['tone_detection']['events']
for event in tone_events:
    print(f"{event['start_time']:.1f}s: {event['tone_label']} (confidence: {event['confidence']:.3f})")
```

### With AI Scene Summaries

```python
from core.pipeline import Pipeline

# Create pipeline with AI scene summarization enabled
pipeline = Pipeline(
    enable_speech_processing=True,  # Required for meaningful summaries
    enable_tone_detection=True,     # Enhances summary quality  
    enable_context_summary=True     # Enable AI summaries
)

results = pipeline.run("input_video.mp4")

# Access scene summaries
scene_summaries = results['context_summary']['summaries']
for summary in scene_summaries:
    print(f"Scene {summary['scene_id']} ({summary['start_time']:.1f}-{summary['end_time']:.1f}s):")
    print(f"  Tone: {summary['tone_label']}")
    print(f"  Summary: {summary['summary_text']}")
```

### Command Line Examples

```bash
# Basic detection (blink + breath)
python example.py input_video.mp4

# With speech processing (transcription + diarization)
python example_speech.py input_video.mp4

# With emotional tone detection
python example_tone.py input_video.mp4

# With AI scene summaries
python example_summary.py input_video.mp4
```

## Testing

### Quick Testing During Development

```bash
# Test a single video quickly
python quick_test.py path/to/your/video.mp4
```

### Comprehensive Testing

```bash
# Add your test videos
mkdir test_videos
cp /path/to/your/videos/*.mp4 test_videos/

# Run interactive test menu
python run_tests.py

# Or run full automated test suite
python test_pipeline.py
```

See [TEST_INSTRUCTIONS.md](TEST_INSTRUCTIONS.md) for detailed testing documentation.

### Custom Configuration

```python
from models.blink_detector import BlinkDetector
from models.breath_detector import BreathDetector
from tasks.blink_task import BlinkTask
from tasks.breath_task import BreathTask
from core.pipeline import Pipeline

# Configure custom detectors
blink_detector = BlinkDetector(
    ear_threshold=0.22,
    consecutive_frames=3,
    min_blink_duration_ms=80.0
)

breath_detector = BreathDetector(
    min_duration_ms=150.0,
    max_duration_ms=1000.0,
    energy_min_threshold=0.005,
    energy_max_threshold=0.15
)

# Create custom tasks
blink_task = BlinkTask(blink_detector)
breath_task = BreathTask(breath_detector)

# Create pipeline with custom tasks
pipeline = Pipeline(blink_task, breath_task)

# Process video
context = pipeline.run("input_video.mp4")
```

### Speech Processing

#### Prerequisites

Install additional dependencies for speech processing:

```bash
pip install openai-whisper pyannote.audio torch
```

**Note**: You may need to accept Hugging Face model conditions at [https://hf.co/pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)

#### Basic Speech Processing

```python
from core.pipeline import Pipeline

# Enable speech processing
pipeline = Pipeline(enable_speech_processing=True)
context = pipeline.run("video_with_speech.mp4")

# Access results
print(f"Words transcribed: {len(context.transcript_words)}")
print(f"Speakers detected: {len(set(seg.speaker_id for seg in context.speaker_segments))}")
print(f"Aligned segments: {len(context.aligned_segments)}")
```

#### Advanced Speech Configuration

```python
from models.whisper_model import WhisperModel
from models.diarization_model import DiarizationModel
from tasks.transcription_task import TranscriptionTask
from tasks.diarization_task import DiarizationTask
from tasks.alignment_task import AlignmentTask
from core.pipeline import Pipeline

# Configure Whisper model
whisper_model = WhisperModel(
    model_size="base",  # "tiny", "base", "small", "medium", "large"
    language="en",      # Force English, or None for auto-detect
    device=None         # Auto-detect (CPU/Metal)
)

# Configure diarization model
diarization_model = DiarizationModel(
    min_speakers=2,     # Expected minimum speakers
    max_speakers=4,     # Expected maximum speakers
    auth_token=None     # HuggingFace token if required
)

# Create tasks
transcription_task = TranscriptionTask(whisper_model)
diarization_task = DiarizationTask(diarization_model)
alignment_task = AlignmentTask.create_default()

# Create pipeline
pipeline = Pipeline(
    transcription_task=transcription_task,
    diarization_task=diarization_task,
    alignment_task=alignment_task,
    enable_speech_processing=True
)

context = pipeline.run("meeting_video.mp4")
```

## Output Format

### Blink Events

```python
BlinkEvent(
    start_frame=150,
    end_frame=157,
    duration_ms=233.3,
    confidence=0.875
)
```

### Breath Events

```python
BreathEvent(
    start_time=2.45,
    end_time=2.89,
    duration_ms=440.0,
    confidence=0.692
)
```

### Speech Processing Events

#### Transcript Words
```python
TranscriptWord(
    text="hello",
    start_time=1.23,
    end_time=1.56,
    confidence=0.95
)
```

#### Speaker Segments
```python
SpeakerSegment(
    speaker_id="SPEAKER_00",
    start_time=0.0,
    end_time=5.2
)
```

#### Speaker-Aligned Segments
```python
SpeakerAlignedSegment(
    speaker_id="SPEAKER_01",
    text="Hello, how are you today?",
    start_time=1.2,
    end_time=3.4,
    words=[TranscriptWord(...), ...]
)
```

### Tone Detection Events

```python
ToneEvent(
    scene_id="scene_003",
    start_time=15.2,
    end_time=23.2,
    tone_label="excited", 
    confidence=0.847
)
```

#### Supported Tone Labels
- `calm`: Stable, composed emotional state
- `tense`: Nervous, strained, or anxious state  
- `excited`: High energy, enthusiastic state
- `somber`: Serious, subdued, or melancholic state
- `neutral`: Baseline emotional state
- `confrontational`: Aggressive or argumentative state

### AI Scene Summaries

```python
SceneSummary(
    scene_id="scene_002",
    start_time=30.0,
    end_time=60.0,
    summary_text="A heated discussion unfolds between the two main speakers with overlapping dialogue and raised intensity. The confrontational tone suggests disagreement on key points.",
    tone_label="confrontational",
    key_speakers=["SPEAKER_00", "SPEAKER_01"],
    confidence=0.824
)
```

#### LLM Backend Support
- **Mock**: Development/testing mode (no model required)
- **llama.cpp**: GGUF quantized models for CPU inference
- **Apple MLX**: Optimized for Apple Silicon devices

## Configuration Parameters

### Blink Detection

- `ear_threshold`: EAR threshold below which a blink is detected (default: 0.25)
- `consecutive_frames`: Minimum consecutive frames for blink detection (default: 2)
- `min_blink_duration_ms`: Minimum blink duration in milliseconds (default: 50.0)
- `max_blink_duration_ms`: Maximum blink duration in milliseconds (default: 500.0)

### Breath Detection

- `sample_rate`: Audio sample rate in Hz (default: 16000)
- `min_duration_ms`: Minimum breath event duration (default: 200.0)
- `max_duration_ms`: Maximum breath event duration (default: 800.0)
- `energy_min_threshold`: Minimum energy for breath detection (default: 0.01)
- `energy_max_threshold`: Maximum energy for breath detection (default: 0.1)

### Tone Detection

- `window_size_seconds`: Analysis window size for tone detection (default: 8.0)
- `window_overlap_seconds`: Overlap between analysis windows (default: 2.0)
- `classifier_type`: Type of classifier to use - "rule_based" or "mlp" (default: "rule_based")

#### Feature Categories
- **Text Features**: Speech rate, sentiment polarity, speaker interactions, linguistic patterns
- **Audio Features**: Energy distribution, spectral characteristics, prosodic patterns
- **Visual Features**: Motion magnitude, scene intensity, visual stability (placeholder)

### Context Summary

- `scene_duration_seconds`: Target scene duration for summarization (default: 30.0)
- `min_scene_duration_seconds`: Minimum scene duration (default: 10.0)
- `max_tokens`: Maximum tokens for LLM generation (default: 150)
- `llm_backend`: LLM backend type - "mock", "llama_cpp", "mlx" (default: "mock")

#### LLM Backend Configuration
- **Mock Backend**: No additional setup required, uses template-based responses
- **llama.cpp Backend**: Requires `model_path` parameter pointing to GGUF model file
- **MLX Backend**: Requires `model_name` parameter (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")

## Constraints

### Technical Constraints

- All ML inference must be local
- CoreML only for learned models
- No PyTorch/TensorFlow runtime in production
- No CUDA or GPU assumptions
- Must run on Apple Silicon (CPU/Metal only)
- No external web requests

### Architecture Constraints

- Object-oriented design only
- No procedural scripts or global state
- Single responsibility classes
- Dependency injection via constructors
- Deterministic execution

## Future Enhancements

1. **CoreML Integration**: Replace heuristic breath detection with trained CoreML models
2. **Audio Extraction**: Implement proper audio extraction from video using ffmpeg
3. **Performance Optimization**: Add frame-skipping and multi-threading support
4. **Real-time Processing**: Add support for live video streams

## File Structure

```
/
├── core/
│   ├── __init__.py
│   ├── pipeline.py          # Main pipeline coordinator
│   ├── context.py           # Shared state container
│   └── base_task.py         # Abstract task base class
├── models/
│   ├── __init__.py
│   ├── blink_detector.py    # Blink detection implementation
│   ├── breath_detector.py   # Breath detection implementation
│   ├── whisper_model.py     # Whisper transcription model
│   ├── diarization_model.py # Pyannote diarization model
│   ├── tone_classifier.py   # Emotional tone classifier
│   ├── local_llm.py         # Local LLM wrapper for summarization
│   └── coreml_model.py      # CoreML model wrapper
├── tasks/
│   ├── __init__.py
│   ├── blink_task.py        # Blink detection task
│   ├── breath_task.py       # Breath detection task
│   ├── transcription_task.py # Speech transcription task
│   ├── diarization_task.py  # Speaker diarization task
│   ├── alignment_task.py    # Speaker-transcript alignment task
│   ├── tone_detection_task.py # Emotional tone detection task
│   └── context_summary_task.py # AI scene summarization task
├── features/
│   ├── __init__.py
│   ├── text_features.py     # Text feature extraction
│   ├── audio_features.py    # Audio feature extraction
│   └── visual_features.py   # Visual feature extraction
├── audio/
│   ├── __init__.py
│   ├── audio_extractor.py   # Audio extraction from video
│   └── speaker_alignment.py # Speaker-transcript alignment utilities
├── video/
│   ├── __init__.py
│   └── face_landmarks.py    # Face landmark utilities
├── example.py               # Basic usage example
├── example_speech.py        # Speech processing example
├── example_tone.py          # Tone detection example
├── example_summary.py       # AI scene summarization example
├── quick_test.py            # Fast single-video testing
├── test_pipeline.py         # Comprehensive test suite
├── run_tests.py             # Interactive test runner
├── TEST_INSTRUCTIONS.md     # Testing documentation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## License

Internal WIZ Intelligence Project - All Rights Reserved

## Support

For questions or issues, contact the WIZ Intelligence development team.