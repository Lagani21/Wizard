# Speech Processing Implementation

This document describes the speech processing capabilities added to the WIZ Intelligence Pipeline.

## Overview

The speech processing pipeline adds three major capabilities:

1. **Speech Transcription** - Convert speech to text with word-level timestamps
2. **Speaker Diarization** - Identify "who spoke when" 
3. **Speaker-Transcript Alignment** - Merge transcription with speaker identification

## Architecture

### Models
- **WhisperModel**: Local Whisper transcription (CPU/Metal)
- **DiarizationModel**: Local Pyannote speaker diarization (CPU)

### Tasks
- **TranscriptionTask**: Runs Whisper transcription
- **DiarizationTask**: Runs Pyannote diarization  
- **AlignmentTask**: Merges transcription with speaker segments

### Utilities
- **AudioExtractor**: Extracts mono 16kHz audio from video
- **SpeakerAligner**: Aligns transcript words with speaker segments

## Data Flow

```
Video File
    ↓
AudioExtractor → Audio Waveform (16kHz mono)
    ↓
├── TranscriptionTask (Whisper)
│   └── TranscriptWords + TranscriptSegments
└── DiarizationTask (Pyannote)  
    └── SpeakerSegments
        ↓
AlignmentTask
    └── SpeakerAlignedSegments
```

## Key Features

### Local Processing Only
- ✅ No cloud APIs or external requests
- ✅ Whisper runs on CPU/Metal (Apple Silicon optimized)
- ✅ Pyannote runs on CPU only
- ✅ All models downloaded and cached locally

### Parallel Execution Support
- Tasks can run independently
- TranscriptionTask and DiarizationTask are parallelizable
- AlignmentTask requires both to complete first

### Flexible Configuration
- Whisper model size selection (`tiny`, `base`, `small`, `medium`, `large`)
- Language detection or forced language
- Speaker count hints for diarization
- Customizable alignment thresholds

## Usage Examples

### Basic Speech Processing

```python
from core.pipeline import Pipeline

# Enable speech processing with defaults
pipeline = Pipeline(enable_speech_processing=True)
context = pipeline.run("video_with_speech.mp4")

# Access results
print(f"Transcript: {len(context.transcript_words)} words")
print(f"Speakers: {len(set(seg.speaker_id for seg in context.speaker_segments))}")
print(f"Aligned segments: {len(context.aligned_segments)}")
```

### Custom Configuration

```python
from models.whisper_model import WhisperModel
from models.diarization_model import DiarizationModel
from tasks.transcription_task import TranscriptionTask
from tasks.diarization_task import DiarizationTask

# Custom Whisper setup
whisper_model = WhisperModel(
    model_size="base",
    language="en",
    device="mps"  # Force Metal Performance Shaders
)

# Custom diarization setup  
diarization_model = DiarizationModel(
    min_speakers=2,
    max_speakers=4,
    auth_token="your_hf_token"
)

# Create pipeline with custom models
transcription_task = TranscriptionTask(whisper_model)
diarization_task = DiarizationTask(diarization_model)

pipeline = Pipeline(
    transcription_task=transcription_task,
    diarization_task=diarization_task,
    enable_speech_processing=True
)

context = pipeline.run("meeting_video.mp4")
```

## Output Format

### TranscriptWord
```python
TranscriptWord(
    text="hello",
    start_time=1.23,
    end_time=1.56, 
    confidence=0.95
)
```

### SpeakerSegment
```python
SpeakerSegment(
    speaker_id="SPEAKER_00",
    start_time=0.0,
    end_time=5.2
)
```

### SpeakerAlignedSegment
```python
SpeakerAlignedSegment(
    speaker_id="SPEAKER_01",
    text="Hello, how are you today?",
    start_time=1.2,
    end_time=3.4,
    words=[TranscriptWord(...), ...]
)
```

## Dependencies

### Required for Speech Processing
```bash
pip install openai-whisper pyannote.audio torch
```

### Model Requirements
- **Whisper**: Models downloaded automatically on first use
- **Pyannote**: Requires accepting terms at https://hf.co/pyannote/speaker-diarization-3.1

## Performance Considerations

### Model Sizes (Whisper)
- `tiny`: 39 MB, fastest but least accurate
- `base`: 74 MB, good balance (recommended)
- `small`: 244 MB, better accuracy
- `medium`: 769 MB, high accuracy
- `large`: 1550 MB, best accuracy but slowest

### Processing Time
- Whisper: ~1-3x real-time depending on model size
- Pyannote: ~2-5x real-time depending on audio length
- Overall: Expect 3-10x real-time for complete speech processing

### Memory Usage
- Whisper: 100MB - 2GB depending on model size
- Pyannote: ~500MB - 1GB
- Peak usage during parallel processing: ~1-3GB

## Fallback Behavior

The pipeline gracefully handles missing dependencies:

1. **Whisper not available**: Creates placeholder transcript
2. **Pyannote not available**: Creates single-speaker segments  
3. **FFmpeg not available**: Uses placeholder audio
4. **Model loading fails**: Falls back to simpler approaches

## Integration with Existing Pipeline

Speech processing is **optional** and **additive**:

- Basic pipeline (blink + breath) works unchanged
- Speech processing adds new capabilities without affecting existing features
- Can enable/disable speech processing per pipeline instance
- Results are stored in separate context fields

## Testing

Use the existing test framework with speech-enabled videos:

```bash
# Test speech processing on single video
python quick_test.py test_videos/speech_video.mp4

# Run comprehensive speech tests  
python test_pipeline.py

# Speech-specific example
python example_speech.py test_videos/meeting.mp4
```

## Troubleshooting

### Common Issues

1. **"pyannote.audio not installed"**
   - Install: `pip install pyannote.audio`

2. **"You must accept user conditions"**
   - Visit: https://hf.co/pyannote/speaker-diarization-3.1
   - Accept terms and get auth token

3. **"FFmpeg not found"**
   - Install FFmpeg: `brew install ffmpeg` (macOS)
   - Pipeline will use placeholder audio if FFmpeg unavailable

4. **Slow processing**
   - Use smaller Whisper model (`tiny` or `base`)
   - Consider shorter video clips for testing
   - Processing is CPU-intensive - use adequate hardware

5. **Poor transcription quality**
   - Try larger Whisper model (`medium` or `large`)
   - Ensure clear audio with minimal background noise
   - Check audio extraction is working properly

### Performance Optimization

- Use `base` Whisper model for good speed/accuracy balance
- Enable Metal Performance Shaders on Apple Silicon
- Process shorter video segments for faster iteration
- Consider parallel processing for batch jobs