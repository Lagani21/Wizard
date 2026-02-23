# WIZ Video Intelligence - Quick Start Guide

## 🚀 Fast Setup (3 Steps)

### 1. Install Core Dependencies
```bash
pip install flask numpy opencv-python mediapipe
```

### 2. Quick Test
```bash
python quick_test.py
```

### 3. Launch Interface
```bash
# Option A: Web Interface (Recommended)
python run_web.py

# Option B: Desktop App (if tkinter works)
python desktop_app.py
```

## 🎯 Interface Options

### 🌐 **Web Interface** (Always Works)
- **Professional timeline** like video editors
- **Interactive waveform** visualization
- **Click timeline** to jump to moments
- **Beautiful visual results**
- **Works in any browser**

**Launch:** `python run_web.py`

### 🖥️ **Desktop App** (Requires tkinter)
- **Simple drag & drop**
- **Native application feel**
- **Text-based results**
- **No browser needed**

**Launch:** `python desktop_app.py`

## ⚠️ tkinter Issues?

If you get "tkinter not found" errors:

**macOS:** tkinter is usually included - try updating Python
**Linux:** `sudo apt-get install python3-tk`
**Windows:** Reinstall Python with tkinter option checked

**Quick Fix:** Just use the Web Interface instead!

## 🎬 Test with Sample Video

The system can create a test video automatically:

```bash
python test_system.py
# Choose option 2 to create test video
```

## 🔧 Troubleshooting

### "Module not found" errors:
```bash
pip install -r requirements_core.txt
```

### "Pipeline import failed":
Make sure you're in the project root directory

### "Flask not found":
```bash
pip install flask
```

### Web interface won't start:
```bash
cd web
python app.py
```

## 📁 File Support

✅ **MP4** (best compatibility)  
✅ **AVI**  
✅ **MOV**  
✅ **MKV**  
✅ **WebM**  

## 🎯 What Gets Analyzed

- **👁️ Blink Detection** - Eye blink events
- **🫁 Breath Detection** - Breath sounds  
- **🎤 Speech Transcription** - Convert speech to text
- **👥 Speaker ID** - "Who spoke when"
- **😊 Emotional Tone** - 6 categories (calm, excited, etc.)
- **📽️ Scene Summaries** - AI descriptions
- **⏰ Timeline View** - Visual timeline with events

## 🔒 Privacy

- **100% Offline** - No data sent anywhere
- **Local Processing** - Everything on your machine
- **No Internet** - Works without WiFi after setup

## 💡 Tips

1. **Start with short videos** (under 2 minutes)
2. **MP4 format works best**  
3. **Close other apps** during processing
4. **Web interface** is more reliable than desktop app

## 🎉 Ready to Go!

Just run `python run_web.py` and start analyzing videos!

---
Need help? Check the main README.md for detailed documentation.