# WIZ Video Intelligence - Offline Setup

## 🔒 Complete Offline Video Analysis

Your WIZ Intelligence Pipeline can run completely offline with no internet connection required after initial setup.

## Quick Start

### 1. Initial Setup (requires internet once)

```bash
# Install dependencies
pip install -r web/requirements.txt

# Run offline setup (downloads models)
python web/offline_setup.py
```

### 2. Choose Your Interface

```bash
# Easy launcher
python start_offline.py
```

Then choose:
- **Option 1**: Web Interface (like the timeline in your screenshot)
- **Option 2**: Desktop App (simple drag & drop GUI)
- **Option 3**: Setup models (if you need to re-download)

## Interface Options

### 🌐 Web Interface (Recommended)
- Full timeline view with waveforms
- Professional video editor-like interface
- Drag & drop video upload
- Interactive timeline with clickable events
- Detailed analysis results

**To start**: `python start_offline.py` → Choose option 1

### 🖥️ Desktop App
- Simple drag & drop interface
- No browser required
- Perfect for quick analysis
- Lightweight and fast

**To start**: `python start_offline.py` → Choose option 2

## What Gets Analyzed (All Offline)

✅ **Blink Detection** - Eye blink events with timestamps  
✅ **Breath Detection** - Breath sounds from audio  
✅ **Speech Transcription** - Convert speech to text (Whisper)  
✅ **Speaker Identification** - "Who spoke when"  
✅ **Emotional Tone** - 6 categories (calm, tense, excited, etc.)  
✅ **Scene Summaries** - AI-generated scene descriptions  
✅ **Timeline View** - Visual timeline like professional video editors  

## File Support

- **MP4** (recommended)
- **AVI**
- **MOV** 
- **MKV**
- **WebM**

## System Requirements

- **macOS** (Apple Silicon preferred)
- **Python 3.8+**
- **8GB RAM** minimum (16GB+ recommended)
- **5GB disk space** for models

## Offline Model Cache

Models are cached in `~/.wiz_models/`:
- Whisper models (speech recognition)
- MediaPipe models (face detection) 
- Configuration files

## Troubleshooting

### "No module found" errors
```bash
pip install -r web/requirements.txt
```

### "Models not found" errors  
```bash
python web/offline_setup.py
```

### Web interface won't start
```bash
cd web
python run.py
```

### Desktop app won't start
Make sure you ran offline setup first:
```bash
python web/offline_setup.py
```

## Advanced Usage

### Command Line Processing
```bash
# Basic analysis
python example.py your_video.mp4

# Full analysis with all features  
python example_summary.py your_video.mp4
```

### Programmatic Usage
```python
from core.pipeline import Pipeline

# Create offline-optimized pipeline
pipeline = Pipeline(
    enable_speech_processing=True,
    enable_tone_detection=True,
    enable_context_summary=True
)

# Process video
results = pipeline.run("your_video.mp4")

# Access results
print(f"Blinks: {len(results.blink_events)}")
print(f"Speech segments: {len(results.aligned_segments)}")
```

## Privacy & Security

🔒 **100% Offline Processing**  
- No data sent to external servers
- No internet required after setup
- All analysis happens on your machine
- Your videos never leave your computer

## Performance Tips

1. **Use smaller Whisper models** for faster processing:
   - `tiny`: Fastest, less accurate
   - `base`: Good balance (default)  
   - `small`: Better accuracy, slower

2. **Close other apps** during processing for best performance

3. **Use MP4 format** for optimal compatibility

## File Structure

```
web/
├── index.html          # Web interface
├── style.css          # Interface styling  
├── script.js          # Timeline and interactions
├── app.py            # Flask backend
├── run.py            # Web server launcher
├── desktop_app.py    # Desktop GUI (auto-created)
├── offline_setup.py  # Model setup script
└── requirements.txt  # Dependencies
```

## Need Help?

1. Run the offline setup: `python web/offline_setup.py`
2. Use the simple launcher: `python start_offline.py`
3. Check the main README.md for detailed documentation

---

**🧙‍♂️ Happy analyzing! Your videos stay private and secure.**