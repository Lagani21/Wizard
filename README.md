# WIZ Video Intelligence

Multimodal video analysis pipeline — blink detection, breath detection, speech transcription, speaker diarization, emotional tone, scene summaries, and a graph-accelerated search layer. Runs entirely on Apple Silicon, no cloud APIs.

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- ffmpeg installed via Homebrew: `brew install ffmpeg`

## Install

```bash
git clone <repo>
cd Wizard
pip install -r requirements.txt
```

For full speech processing (transcription + diarization), install the optional deps:

```bash
pip install openai-whisper pyannote.audio torch
```

Pyannote requires a HuggingFace token and model licence acceptance:
1. Create a free account at https://huggingface.co
2. Accept the licence at https://hf.co/pyannote/speaker-diarization-3.1
3. Generate a token at https://hf.co/settings/tokens
4. Set it: `export HF_TOKEN=hf_...`

For LLM scene summaries, install one of:
```bash
pip install llama-cpp-python   # llama.cpp backend (CPU/Metal)
pip install mlx mlx-lm         # Apple MLX backend (faster on M-series)
```

## Run the web app

```bash
python main.py
# or with options:
python main.py --port 8080 --debug
```

Open **http://localhost:5555** in your browser.

| Page | URL | What it does |
|---|---|---|
| Upload | `/` | Drop a video, watch it process, explore results |
| Monitoring | `/monitoring` | Live view of all processing sessions |
| Benchmark | `/benchmark` | Graph vs SQL search performance comparison |

## Process a folder of videos (batch mode)

```bash
python main.py --folder ./clips/
# with custom output directory and lite mode:
python main.py --folder ./clips/ --output ./results/ --mode lite
```

Processes every `.mp4 / .mov / .avi / .mkv` in the folder sequentially. A single `SpeakerIdentityRegistry` is shared across all clips so the same physical person receives a consistent `PERSON_001 / PERSON_002` label in every clip's `.wiz` file. Prints a summary table when done.

## Run the pipeline from the command line (single file)

```bash
python main.py path/to/video.mp4
```

Results are written to `web/results/` as a `.wiz` file (SQLite) and a `.json` summary.

## Run the search benchmark standalone

```bash
python -m wiz.benchmark --hours 0.1 --runs 20
```

Generates synthetic footage, builds the WizGraph index, and prints median latency for graph search vs naive SQL across four query types.

---

## Why these models

Every model was chosen for three constraints: runs on Apple Silicon without CUDA, good accuracy on interview-style footage, and available under a permissive licence.

### Blink detection — Apple Vision framework + EAR (CoreML / Neural Engine)

Blink detection runs through Apple's `Vision.VNDetectFaceLandmarksRequest`, which executes on the Neural Engine or GPU via CoreML — the fastest possible path on M-series hardware, with zero CUDA dependency. Each frame is converted to a `CGImage` and passed to a `VNImageRequestHandler`. The framework returns normalised 2D eye contour points for both eyes.

We compute a bounding-box Eye Aspect Ratio (EAR) from those points:

```
EAR = height_of_eye_bounding_box / width_of_eye_bounding_box
```

An open eye has EAR ≈ 0.25–0.40. A blink closes the eye, collapsing the vertical extent: EAR drops below 0.25 for at least two consecutive frames. A `BlinkEvent` is recorded with frame-level `start_frame` / `end_frame` and a confidence score derived from the depth of the EAR dip.

Why Apple Vision over MediaPipe or dlib:
- **CoreML/Neural Engine acceleration** — runs faster than any CPU-bound Python face model on M-series
- **No extra model download** — `Vision.framework` ships with macOS
- **Frame-precise output** — the framework processes frames synchronously, so every event has an exact frame number the editor can trust

### Breath detection — energy-based audio segmentation

Breath sounds are short, broadband, high-energy transients in the 100–3000 Hz range. We detect them by:
1. Computing short-time RMS energy in 20 ms windows
2. Applying a bandpass filter (100–3000 Hz) to isolate breath frequencies from speech
3. Flagging frames where energy exceeds a dynamic threshold (mean + 1.5σ) with minimum duration 0.1 s

This approach has no model to download and runs in milliseconds. The alternative — a VAD (Voice Activity Detector) like Silero — detects speech pauses but not breath sounds specifically. Breath happens at the edges of speech turns and inside sentences; a VAD would miss most of them. An ML breath classifier (e.g. trained on respiratory audio datasets) would be more accurate but is overkill for production use where recall matters more than precision — it is better to flag a questionable cut point than to miss one.

### Transcription — OpenAI Whisper (local)

Whisper `base` (74M params) runs at roughly 10–20× real-time on M-series CPU and produces word-level timestamps via `word_timestamps=True`. It handles accented speech, technical vocabulary, and noisy interview audio better than any other open-weight model at this size. The `small` model (244M) is used in `--mode lite` to reduce memory pressure.

### Speaker diarization — Pyannote 3.1

Pyannote speaker-diarization-3.1 is the highest-accuracy open-weight diarization model (DER ~18% on AMI, ~12% on VoxConverse). It runs on CPU/MPS without CUDA. We inject a `SpeakerIdentityRegistry` that maps each clip's `SPEAKER_XX` labels to stable `PERSON_XXX` identities across clips using cosine similarity on speaker embeddings — so the same person gets the same ID whether they appear in clip 1 or clip 47.

### Emotional tone — rule-based + MLP

A lightweight 3-layer MLP trained on prosodic and lexical features (speech rate, pitch variance, energy envelope, sentiment polarity). No heavy transformer required. Labels: `confident`, `concerned`, `excited`, `neutral`, `thoughtful`, `sad`. Fast enough to run per-segment in real time.

### Scene summaries — local LLM (llama.cpp / MLX / mock)

Aligned transcript segments are chunked into 30–60 s windows and summarised with a prompt. Supports three backends: llama.cpp (any GGUF model), Apple MLX (`mlx-lm`), or a deterministic mock for testing without a model. The mock produces readable summaries so the pipeline is fully demonstrable without downloading a multi-GB model.