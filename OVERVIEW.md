# Project Overview

## What it is

WIZ is a local-first video intelligence pipeline. You drop in a video and it produces a structured analysis: every blink, every breath, a full transcript with speaker labels, emotional tone per scene, AI scene summaries, and visual captions — all processed on-device using Apple Silicon.

The output is stored in a `.wiz` file (SQLite) and queried through a graph-accelerated search engine (WizGraph) that runs 60–80× faster than naive SQL joins.

---

## File structure

```
Wizard/
├── core/               # Pipeline engine
│   ├── pipeline.py     # Orchestrates all tasks in sequence
│   ├── context.py      # Shared state object passed through every task
│   ├── base_task.py    # Abstract base class all tasks implement
│   ├── logger.py       # Structured file + console logging
│   ├── monitor.py      # Task timing and success/failure tracking
│   └── metrics.py      # Metric collection helpers
│
├── models/             # ML model wrappers
│   ├── blink_detector.py      # MediaPipe Face Mesh + EAR threshold
│   ├── breath_detector.py     # Energy-based breath sound detector
│   ├── whisper_model.py       # OpenAI Whisper (local) transcription
│   ├── diarization_model.py   # Pyannote speaker diarization + embedding extraction
│   ├── speaker_identity.py    # Cross-clip PERSON_XXX identity registry
│   ├── tone_classifier.py     # Rule-based + MLP emotional tone
│   ├── video_mae.py           # VideoMAE visual captioning
│   └── local_llm.py           # Local LLM (llama.cpp / MLX / mock)
│
├── tasks/              # One task per detection stage
│   ├── blink_task.py
│   ├── breath_task.py
│   ├── transcription_task.py
│   ├── diarization_task.py
│   ├── alignment_task.py       # Merges transcript words with speaker segments
│   ├── tone_detection_task.py
│   ├── video_mae_task.py       # Visual caption generation
│   └── context_summary_task.py # AI scene summarization
│
├── features/           # Feature extractors used by tone detection
│   ├── text_features.py    # Speech rate, sentiment, speaker patterns
│   ├── audio_features.py   # Energy, spectral, prosodic
│   └── visual_features.py  # Motion, scene intensity
│
├── audio/
│   ├── audio_extractor.py      # Extracts PCM audio from video via ffmpeg
│   └── speaker_alignment.py    # Word-to-speaker alignment utilities
│
├── wiz/                # Search layer
│   ├── format.py       # Atom / AtomTag data classes and .wiz schema
│   ├── writer.py       # WizWriter: converts PipelineContext → .wiz file
│   ├── graph.py        # WizGraph: builds in-memory set-intersection index
│   ├── search.py       # SearchEngine: high-level query API over WizGraph
│   └── benchmark.py    # Synthetic benchmark comparing graph vs SQL
│
├── web/                # Flask web application
│   ├── app.py          # All routes: upload, progress, results, search, monitoring
│   ├── index.html      # Main UI — upload, results, timeline, search tab
│   ├── script.js       # Client-side app logic
│   ├── style.css       # Dark-theme stylesheet
│   ├── monitoring.html # Live session dashboard
│   └── benchmark.html  # Standalone graph vs SQL benchmark page
│
├── main.py             # CLI entry point
└── requirements.txt
```

---

## Pipeline stages

Each stage reads from and writes back to a `PipelineContext` object.

| Stage | What it produces |
|---|---|
| Video decode | Frames + raw audio waveform, video metadata |
| Blink detection | `blink_events` — frame range, duration, confidence |
| Breath detection | `breath_events` — time range, duration, confidence |
| Transcription | `transcript_words` — text, start/end time, confidence |
| Diarization | `speaker_segments` — speaker ID, time range |
| Alignment | `aligned_segments` — speaker ID + text + word timings |
| Tone detection | `tone_events` — scene ID, time range, tone label, confidence |
| Scene summary | `scene_summaries` — narrative text, tone, key speakers |
| Video captioning | `video_captions` — window ID, time range, caption text |
| WizWriter | Writes all of the above to a `.wiz` SQLite file |

---

## WizGraph search layer

`.wiz` files store atoms (time-ranged segments) and atom_tags (key/value pairs attached to atoms).

`WizGraph` loads a `.wiz` file once and builds a set-of-atom-IDs index keyed by `(tag_key, tag_value)`. Multi-dimensional queries resolve as Python set intersections — no SQL joins, no disk I/O after load.

`SearchEngine` exposes named query methods on top of WizGraph:

| Method | Finds |
|---|---|
| `find_person_topic(speaker, topic)` | Segments where a speaker discusses a topic |
| `find_emotion(label)` | Segments with a given tone label |
| `find_safe_cuts()` | Windows with no blinks and speech pauses (clean edit points) |
| `find_person_topic_no_blink(speaker, topic)` | Person+topic, excluding blink windows |
| `query(**kwargs)` | Generic tag intersection |

Typical speedup over naive SQL: **60–80×** on 6-minute footage, rising with dataset size.

---

## Web interface

The Flask app (`web/app.py`) serves three pages and exposes a JSON API.

**Upload page** (`/`)
- Drag-and-drop or browse to upload a video
- Live progress bar as the pipeline runs
- Results view: video player, waveform timeline with Speech / Tone / Captions / Events tracks, and a tabbed detail panel (Transcript, Scenes, Captions, Events, Search)
- Session is persisted in `sessionStorage` — the video survives a page refresh

**Monitoring page** (`/monitoring`)
- Summary stats: videos processed, completed, total blinks/breaths/speakers
- Per-session table with all detection counts
- Live pipeline log tail

**Benchmark page** (`/benchmark`)
- Select synthetic footage length (3 min → 1 hour) and iteration count
- Runs `run_benchmark_json()` on the server and animates bar charts
- Compares WizGraph (µs range) vs naive SQL (ms range) for four query types