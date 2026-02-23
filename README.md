# WIZ Video Intelligence

Multimodal video analysis pipeline — blink detection, breath detection, speech transcription, speaker diarization, emotional tone, scene summaries, and a graph-accelerated search layer. Runs entirely on Apple Silicon, no cloud APIs.

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+

## Install

```bash
git clone <repo>
cd Wizard
pip install -r requirements.txt
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

## Run the pipeline from the command line

```bash
python main.py path/to/video.mp4
```

Results are written to `web/results/` as a `.wiz` file (SQLite) and a `.json` summary.

## Run the search benchmark standalone

```bash
python -m wiz.benchmark --hours 0.1 --runs 20
```

Generates synthetic footage, builds the WizGraph index, and prints median latency for graph search vs naive SQL across four query types.