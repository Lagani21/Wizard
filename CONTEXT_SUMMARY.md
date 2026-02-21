# Context Box Summary Model Documentation

## Overview

The Context Box Summary Model generates editorial-quality scene summaries using structured outputs from the WIZ Intelligence Pipeline. It analyzes multimodal data and produces narrative summaries using local LLM inference.

## Architecture

### Core Components

1. **LocalLLM Wrapper** (`models/local_llm.py`)
   - Unified interface for different LLM backends
   - Support for Mock, llama.cpp, and Apple MLX
   - Clean separation between model logic and pipeline integration

2. **Scene Data Extraction** (`tasks/context_summary_task.py`)
   - `SceneDataExtractor`: Collects structured multimodal features
   - `PromptFormatter`: Creates deterministic LLM prompts
   - `ContextSummaryTask`: Pipeline task orchestrating the process

3. **Data Models** (`core/context.py`)
   - `SceneSummary`: Structured summary output format
   - Integration with existing pipeline context

## Supported LLM Backends

### Mock Backend (Default)
- **Purpose**: Development and testing
- **Requirements**: None
- **Behavior**: Template-based responses using prompt analysis
- **Performance**: Fast, deterministic

```python
LocalLLM(backend="mock")
```

### llama.cpp Backend
- **Purpose**: Production CPU inference with quantized models
- **Requirements**: `llama-cpp-python` package, GGUF model file
- **Models**: Any GGUF quantized model (Llama, Mistral, etc.)
- **Performance**: CPU optimized, good for Apple Silicon

```python
LocalLLM(
    backend="llama_cpp", 
    model_path="/path/to/model.gguf",
    n_ctx=2048,
    n_threads=4
)
```

### Apple MLX Backend
- **Purpose**: Production inference optimized for Apple Silicon
- **Requirements**: `mlx` and `mlx-lm` packages
- **Models**: MLX-compatible models from HuggingFace
- **Performance**: Metal acceleration, memory efficient

```python
LocalLLM(
    backend="mlx",
    model_name="mlx-community/Llama-3.2-3B-Instruct-4bit"
)
```

## Input Data Processing

The system processes structured scene-level data from pipeline outputs:

### From Speech Processing
- **Transcript Data**: Word count, speech rate, transcript excerpts
- **Speaker Analysis**: Active speakers, dominant speakers, overlap ratios
- **Word Confidence**: Average transcription confidence scores

### From Tone Detection  
- **Emotional Tone**: Dominant tone label and confidence
- **Tone Stability**: Consistency of tone throughout scene
- **Tone Transitions**: Changes between scenes

### From Visual Processing
- **Motion Analysis**: Motion intensity based on embedding changes
- **Visual Activity**: Scene change intensity and shot counts
- **Structured Features**: No raw embeddings, only derived metrics

### Scene Boundaries
- **Fixed Duration**: Configurable scene length (default: 30 seconds)
- **Minimum Duration**: Prevents very short scenes (default: 10 seconds)
- **Temporal Alignment**: Aligned with other pipeline outputs

## Prompt Structure

The system uses deterministic, structured prompts:

```
You are an editorial assistant analyzing a video scene.

Scene Time: {start}–{end}
Tone: {tone_label} (confidence: {confidence})
Primary Speakers: {speaker_names}
Speech Rate: {speech_rate_desc} ({speech_rate} words/second)
Motion Intensity: {motion_desc}
Speaker Interaction: {overlap_desc}
Transcript Excerpt:
{transcript}

Write a concise 2–3 sentence summary describing:
- What is happening in the scene
- The emotional tone and atmosphere  
- Key speaker dynamics or interactions

Be objective and editorial. Focus on observable patterns.
```

## Output Format

### SceneSummary Data Structure

```python
@dataclass
class SceneSummary:
    scene_id: str           # Unique scene identifier
    start_time: float       # Scene start time in seconds
    end_time: float         # Scene end time in seconds
    summary_text: str       # Generated narrative summary
    tone_label: str         # Dominant emotional tone
    key_speakers: List[str] # Primary speakers in scene
    confidence: float       # Overall confidence score
```

### Pipeline Results Integration

```python
results['context_summary'] = {
    'total_summaries': 8,
    'summaries': [
        {
            'scene_id': 'scene_001',
            'start_time': 0.0,
            'end_time': 30.0,
            'summary_text': 'A calm introduction begins with...',
            'tone_label': 'calm',
            'key_speakers': ['SPEAKER_00'],
            'confidence': 0.762
        },
        # ... more summaries
    ]
}
```

## Configuration

### Basic Configuration

```python
# Using default mock LLM
pipeline = Pipeline(
    enable_speech_processing=True,
    enable_tone_detection=True,
    enable_context_summary=True
)
```

### Custom Configuration

```python
from Wizard import ContextSummaryTask, LocalLLM

# Configure custom LLM
local_llm = LocalLLM(
    backend="llama_cpp",
    model_path="/path/to/model.gguf",
    temperature=0.7
)

# Configure custom task
summary_task = ContextSummaryTask(
    local_llm=local_llm,
    scene_duration_seconds=45.0,
    max_tokens=200
)

# Use in pipeline
pipeline = Pipeline(
    context_summary_task=summary_task,
    enable_context_summary=True
)
```

### Scene Configuration

- `scene_duration_seconds`: Target scene length (default: 30.0)
- `min_scene_duration_seconds`: Minimum scene length (default: 10.0)  
- `max_tokens`: LLM generation limit (default: 150)

## Dependencies

### Core Dependencies (Always Required)
```python
numpy >= 1.21.0
scipy >= 1.7.0
```

### Optional LLM Dependencies
```python
# For llama.cpp backend
llama-cpp-python >= 0.2.0

# For Apple MLX backend  
mlx >= 0.12.0
mlx-lm >= 0.8.0
```

### Model Requirements

#### llama.cpp Models
- **Format**: GGUF quantized models
- **Size**: 3B-7B parameters recommended for scene summaries
- **Sources**: HuggingFace, TheBloke quantizations
- **Storage**: Local file system

#### MLX Models
- **Format**: MLX-compatible models
- **Size**: 3B-7B parameters recommended
- **Sources**: mlx-community on HuggingFace
- **Storage**: Downloaded automatically via HuggingFace

## Performance Characteristics

### Processing Speed
- **Mock Backend**: ~100ms per scene (instant)
- **llama.cpp Backend**: ~1-5 seconds per scene (depends on model size)
- **MLX Backend**: ~0.5-2 seconds per scene (Apple Silicon optimized)

### Memory Usage
- **Mock Backend**: Minimal (<10MB)
- **llama.cpp Backend**: 2-8GB (depends on model size)
- **MLX Backend**: 2-6GB (more memory efficient)

### Typical Video Processing
- **5-minute video**: 10 scenes, 30-60 seconds total processing
- **30-minute video**: 60 scenes, 3-10 minutes total processing
- **Bottleneck**: LLM inference time, not feature extraction

## Usage Examples

### Development/Testing
```python
# Uses mock LLM, no model downloads required
pipeline = Pipeline(enable_context_summary=True)
results = pipeline.run("video.mp4")

for summary in results['context_summary']['summaries']:
    print(f"{summary['scene_id']}: {summary['summary_text']}")
```

### Production with llama.cpp
```python
# Download a GGUF model first
pipeline = Pipeline(
    enable_speech_processing=True,
    enable_context_summary=True,
    context_summary_task=ContextSummaryTask.create_default(
        llm_backend="llama_cpp",
        model_path="/models/llama-3.2-3b-instruct-q4_k_m.gguf"
    )
)
```

### Production with Apple MLX
```python  
# Model downloaded automatically on first use
pipeline = Pipeline(
    enable_speech_processing=True,
    enable_context_summary=True,
    context_summary_task=ContextSummaryTask.create_default(
        llm_backend="mlx",
        model_name="mlx-community/Llama-3.2-3B-Instruct-4bit"
    )
)
```

## Error Handling

### Graceful Degradation
- **Model Loading Failures**: Falls back to mock backend with warnings
- **Generation Failures**: Creates error summaries with context
- **Missing Dependencies**: Clear error messages with installation instructions

### Validation
- **Input Validation**: Checks for required pipeline outputs
- **Scene Validation**: Ensures minimum scene duration requirements
- **Output Validation**: Ensures summary format consistency

## Integration Notes

### Pipeline Dependencies
- **Required**: Video metadata, audio extraction
- **Enhanced by**: Speech processing (transcription, diarization) 
- **Enhanced by**: Tone detection (emotional context)
- **Independent of**: Blink/breath detection

### Execution Order
1. Core detection tasks (blink, breath)
2. Speech processing tasks (transcription, diarization, alignment)
3. Tone detection task
4. **Context summary task (runs last)**

### Data Flow
```
Video Input
    ↓
Core Processing (blink, breath, speech, tone)
    ↓
Scene Data Extraction (structured features)
    ↓  
Prompt Formatting (deterministic templates)
    ↓
Local LLM Inference (scene summaries)
    ↓
Pipeline Results (structured output)
```

## Troubleshooting

### Common Issues

1. **"No context summary results available"**
   - Check `enable_context_summary=True` in Pipeline
   - Verify speech processing is enabled for meaningful summaries

2. **Model loading failures**
   - Check file paths for llama.cpp models
   - Verify internet connection for MLX model downloads
   - Ensure sufficient disk space for model storage

3. **Poor summary quality**
   - Enable tone detection for better context
   - Increase `max_tokens` for longer summaries  
   - Try different LLM models or backends
   - Check transcript quality (speech processing accuracy)

4. **Performance issues**
   - Use smaller models (3B vs 7B parameters)
   - Reduce scene duration to decrease processing time
   - Use MLX backend on Apple Silicon for best performance

### Debug Logging
```python
import logging

# Enable debug logging
logging.getLogger("wiz.tasks.context_summary").setLevel(logging.DEBUG)
logging.getLogger("wiz.models.local_llm").setLevel(logging.DEBUG)

# Run pipeline with detailed logs
pipeline = Pipeline(enable_context_summary=True)
results = pipeline.run("video.mp4")
```

## Future Enhancements

### Planned Features
1. **Better Visual Integration**: Real VideoMAE embeddings instead of placeholders
2. **Conversation Analysis**: Multi-turn dialogue understanding  
3. **Temporal Modeling**: Scene transition analysis and narrative flow
4. **Custom Prompts**: User-defined prompt templates
5. **Batch Processing**: Multiple video summarization
6. **Export Formats**: JSON, markdown, PDF summary exports

### Model Support Expansion
1. **Ollama Integration**: Local model serving via Ollama
2. **Quantization Options**: Different quantization levels for speed/quality tradeoffs
3. **Fine-tuned Models**: Domain-specific summarization models
4. **Multi-modal Models**: Vision-language models for better video understanding

This documentation provides comprehensive coverage of the Context Box Summary Model implementation, from architecture to usage and troubleshooting.