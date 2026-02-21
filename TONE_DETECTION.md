# Emotional Tone Detection Documentation

## Overview

The Emotional Tone Detection module analyzes multimodal features from video content to classify emotional tone in fixed time windows. It combines text analysis, audio characteristics, and visual cues to provide structured tone classification.

## Architecture

### Core Components

1. **Feature Extractors** (`features/`)
   - `TextFeatureExtractor`: Analyzes speech patterns, sentiment, and speaker interactions
   - `AudioFeatureExtractor`: Extracts acoustic features like energy, spectral characteristics, and prosodics
   - `VisualFeatureExtractor`: Processes VideoMAE embeddings for motion and scene analysis

2. **Classifier** (`models/tone_classifier.py`)
   - `ToneClassifier`: Main classifier interface supporting rule-based and MLP approaches
   - `RuleBasedToneClassifier`: Hand-crafted rules for tone detection
   - `MLPToneClassifier`: Neural network classifier (placeholder for future training)

3. **Task Integration** (`tasks/tone_detection_task.py`)
   - `ToneDetectionTask`: Pipeline task that orchestrates feature extraction and classification

## Supported Tone Categories

The system classifies content into 6 emotional tone categories:

- **Calm**: Stable, composed emotional state with moderate speech patterns and stable visual cues
- **Tense**: Nervous, strained state indicated by hesitation, speaker overlap, and energy variance
- **Excited**: High energy state with fast speech, high volume, and positive sentiment
- **Somber**: Serious, subdued state with slow speech, low energy, and negative sentiment
- **Neutral**: Baseline emotional state when no strong patterns are detected
- **Confrontational**: Aggressive state with speaker overlap, negative sentiment, and high intensity

## Feature Engineering

### Text Features (from Whisper transcription)

- **Speech Rate**: Words per second, tempo variance
- **Sentiment Analysis**: Positive/negative word ratios, polarity scores, intensity words
- **Linguistic Patterns**: Question density, exclamation usage, interruption rates
- **Speaker Interaction**: Number of speakers, turn-taking rates, completion rates
- **Fluency Indicators**: Hesitation markers, filler word ratios

### Audio Features (from waveform)

- **Volume & Energy**: RMS energy, peak amplitude, dynamic range, energy distribution
- **Spectral Characteristics**: Spectral centroid, bandwidth, flatness, rolloff
- **Temporal Patterns**: Zero crossing rate, periodicity detection
- **Speech Activity**: Activity ratios, speaker overlap detection
- **Prosodic Features**: Pitch variance, rhythm regularity (basic estimation)

### Visual Features (from VideoMAE embeddings)

- **Motion Analysis**: Motion magnitude between shots, motion acceleration
- **Scene Intensity**: Embedding magnitude, energy peaks, intensity variance  
- **Temporal Consistency**: Shot similarity, visual stability, smoothness
- **Distribution Analysis**: Effective dimensionality, sparsity, entropy measures

## Classification Approaches

### Rule-Based Classifier (Default)

Uses hand-crafted thresholds and logical combinations:

```python
# Example rules
if speech_rate > 3.0 and energy > 0.1 and motion > 2.0:
    → EXCITED (high confidence)

if hesitation > 0.3 and speaker_count > 1 and overlap > 0.1:
    → TENSE (moderate confidence)
```

**Advantages:**
- Interpretable decisions
- No training data required
- Deterministic results
- Fast inference

### MLP Classifier (Placeholder)

Small neural network for learning complex patterns:

```python
# Architecture: Input → [64, 32] → 6 classes
ToneClassifier(classifier_type="mlp", hidden_sizes=[64, 32])
```

**Note**: Currently falls back to rule-based logic. Future versions will support trained models.

## Configuration

### Window Parameters

```python
ToneDetectionTask(
    window_size_seconds=8.0,      # Analysis window size
    window_overlap_seconds=2.0     # Overlap between windows
)
```

### Classifier Selection

```python
# Rule-based (default)
Pipeline(enable_tone_detection=True)

# MLP classifier
tone_task = ToneDetectionTask.create_default(classifier_type="mlp")
Pipeline(tone_detection_task=tone_task, enable_tone_detection=True)
```

## Usage Examples

### Basic Usage

```python
from Wizard import Pipeline

# Enable tone detection (requires speech processing)
pipeline = Pipeline(
    enable_speech_processing=True,
    enable_tone_detection=True
)

results = pipeline.run("video.mp4")

# Access tone events
tone_events = results['tone_detection']['events']
for event in tone_events:
    print(f"{event['start_time']:.1f}s: {event['tone_label']} "
          f"(confidence: {event['confidence']:.3f})")
```

### Advanced Configuration

```python
from Wizard import ToneDetectionTask, ToneClassifier
from Wizard import TextFeatureExtractor, AudioFeatureExtractor

# Custom feature extractors
text_extractor = TextFeatureExtractor(
    sentiment_words_positive=["great", "awesome", "excellent"],
    sentiment_words_negative=["bad", "terrible", "awful"]
)

audio_extractor = AudioFeatureExtractor(sample_rate=16000)

# Custom classifier
tone_classifier = ToneClassifier(classifier_type="rule_based")

# Custom task
tone_task = ToneDetectionTask(
    tone_classifier=tone_classifier,
    text_extractor=text_extractor,
    audio_extractor=audio_extractor,
    window_size_seconds=10.0,
    window_overlap_seconds=3.0
)

# Use in pipeline
pipeline = Pipeline(
    tone_detection_task=tone_task,
    enable_speech_processing=True,
    enable_tone_detection=True
)
```

## Output Format

### ToneEvent Structure

```python
@dataclass
class ToneEvent:
    scene_id: str           # Unique scene identifier (e.g., "scene_003")
    start_time: float       # Start time in seconds
    end_time: float         # End time in seconds  
    tone_label: str         # One of: calm, tense, excited, somber, neutral, confrontational
    confidence: float       # Classification confidence (0.0 to 1.0)
```

### Results Summary

```python
results['tone_detection'] = {
    'total_events': 15,
    'events': [
        {
            'scene_id': 'scene_000',
            'start_time': 0.0,
            'end_time': 8.0,
            'tone_label': 'neutral',
            'confidence': 0.623
        },
        # ... more events
    ]
}
```

### Metadata

```python
results['processing_metadata']['tone_detection'] = {
    'classifier_info': {
        'type': 'rule_based',
        'version': '1.0',
        'num_classes': '6'
    },
    'window_config': {
        'window_size_seconds': 8.0,
        'window_overlap_seconds': 2.0,
        'total_windows': 15
    },
    'detection_stats': {
        'tone_distribution': {'neutral': 8, 'calm': 4, 'excited': 3},
        'overall_avg_confidence': 0.742,
        'dominant_tone': 'neutral',
        'tone_changes': 7
    }
}
```

## Dependencies

### Required for Tone Detection

```python
# Core dependencies (already included)
numpy >= 1.21.0
scipy >= 1.7.0  # For spectral analysis

# Speech processing (required for tone detection)
openai-whisper >= 20231117
pyannote.audio >= 3.1.0
torch >= 1.13.0
```

### Visual Features (Future)

```python
# For actual VideoMAE embeddings (not yet implemented)
# transformers
# torch-audio
# torchvision
```

## Performance Considerations

### Computational Complexity

- **Text Features**: O(n) where n = number of words/segments
- **Audio Features**: O(m) where m = audio samples (with windowing)
- **Visual Features**: O(k) where k = number of embeddings
- **Classification**: O(1) for rule-based, O(features) for MLP

### Memory Usage

- Feature vectors: ~50-100 floats per window
- Audio processing: Temporary spectral arrays
- Visual embeddings: 768 floats per shot (placeholder)

### Typical Processing Times

- 1-minute video: ~2-5 seconds for tone detection
- Dominated by speech processing overhead (Whisper + Pyannote)
- Feature extraction: <100ms per window

## Limitations

### Current Limitations

1. **Visual Features**: Currently uses placeholder embeddings, not actual VideoMAE
2. **MLP Classifier**: Not trained, falls back to rule-based logic
3. **Language Support**: English-only sentiment analysis
4. **Speaker Overlap**: Simple overlap detection, not sophisticated audio separation

### Future Improvements

1. **Real VideoMAE Integration**: Actual visual embedding extraction
2. **Trained Classifiers**: MLP and other ML models with real training data
3. **Multilingual Support**: Sentiment analysis for other languages
4. **Advanced Audio Features**: Pitch tracking, voice quality analysis
5. **Temporal Modeling**: Consider tone history and transitions

## Debugging and Troubleshooting

### Common Issues

1. **No Tone Events Generated**
   - Check if speech processing is enabled
   - Verify audio extraction worked
   - Check video has sufficient duration (>4 seconds)

2. **Low Confidence Scores**
   - Normal for ambiguous content
   - Rule-based classifier is conservative
   - Consider adjusting window size

3. **Unexpected Classifications**
   - Check feature extraction debug logs
   - Verify audio quality and transcription accuracy
   - Rule thresholds may need adjustment

### Debug Logging

```python
import logging

# Enable debug logging for tone detection
logging.getLogger("wiz.features").setLevel(logging.DEBUG)
logging.getLogger("wiz.models.tone_classifier").setLevel(logging.DEBUG)
logging.getLogger("wiz.tasks.tone_detection").setLevel(logging.DEBUG)
```

### Feature Inspection

```python
# Access extracted features for debugging
from Wizard.features import TextFeatureExtractor, AudioFeatureExtractor

text_extractor = TextFeatureExtractor()
audio_extractor = AudioFeatureExtractor()

# Extract features for specific window
text_features = text_extractor.extract_features(
    transcript_segments, aligned_segments, start_time, end_time
)

print("Text features:", text_features)
print("Feature names:", text_extractor.get_feature_names())
```

## Integration Notes

### Pipeline Integration

The tone detection task integrates seamlessly into the existing pipeline:

1. **Dependencies**: Requires completed speech processing (transcription + diarization)
2. **Execution Order**: Runs after alignment task
3. **Data Flow**: Uses PipelineContext for input/output
4. **Error Handling**: Graceful degradation if dependencies missing

### Extending the System

To add new tone categories:

```python
# In models/tone_classifier.py
class ToneLabel(Enum):
    # ... existing tones
    AGGRESSIVE = "aggressive"  # Add new tone
    
# Update classifier rules and MLP output dimensions accordingly
```

To add new features:

```python
# In features/text_features.py
def _extract_new_features(self, segments):
    # Implement new feature extraction
    return {"new_feature": value}

# Update _get_empty_features() and get_feature_names()
```

## Testing

### Unit Testing

```bash
# Test individual components
python -m pytest features/ -v
python -m pytest models/tone_classifier.py -v
python -m pytest tasks/tone_detection_task.py -v
```

### Integration Testing

```bash
# Test with tone detection enabled
python example_tone.py test_videos/sample.mp4

# Run comprehensive tests
python run_tests.py
```

### Validation

```bash
# Check feature extraction
python -c "
from Wizard.features import TextFeatureExtractor
extractor = TextFeatureExtractor()
print('Text features:', len(extractor.get_feature_names()))
"
```

This documentation provides comprehensive coverage of the emotional tone detection system, from architecture to usage and troubleshooting.