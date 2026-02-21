# WIZ Intelligence Pipeline

A modular, object-oriented pipeline for detecting blink and breath events from video input using local ML inference.

## Overview

The WIZ Intelligence Pipeline is designed to process video files and detect multiple types of events:

### Core Detection
1. **Blink Detection**: Detects eye blink events using MediaPipe Face Mesh and Eye Aspect Ratio (EAR) calculations
2. **Breath Detection**: Detects breath sounds from audio using simple heuristic analysis

### Speech Processing (Optional)
3. **Speech Transcription**: Converts speech to text with word-level timestamps using Whisper
4. **Speaker Diarization**: Identifies "who spoke when" using Pyannote.audio
5. **Speaker-Transcript Alignment**: Merges transcription with speaker identification

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

### Tasks

- `BlinkTask`: Processes video frames for blink detection
- `BreathTask`: Processes audio waveform for breath detection

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

### Command Line Examples

```bash
# Basic detection (blink + breath)
python example.py input_video.mp4

# With speech processing (transcription + diarization)
python example_speech.py input_video.mp4
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
│   └── coreml_model.py      # CoreML model wrapper
├── tasks/
│   ├── __init__.py
│   ├── blink_task.py        # Blink detection task
│   ├── breath_task.py       # Breath detection task
│   ├── transcription_task.py # Speech transcription task
│   ├── diarization_task.py  # Speaker diarization task
│   └── alignment_task.py    # Speaker-transcript alignment task
├── audio/
│   ├── __init__.py
│   ├── audio_extractor.py   # Audio extraction from video
│   └── speaker_alignment.py # Speaker-transcript alignment utilities
├── video/
│   ├── __init__.py
│   └── face_landmarks.py    # Face landmark utilities
├── example.py               # Basic usage example
├── example_speech.py        # Speech processing example
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