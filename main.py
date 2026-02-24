#!/usr/bin/env python3
"""
WIZ Video Intelligence Pipeline — single entry point.

Usage:
    python main.py                        # start web UI (default, port 5555)
    python main.py --port 8080            # custom port
    python main.py --folder ./clips/      # batch-process a folder of videos
    python main.py --folder ./clips/ --output ./results/ --mode lite
"""

import sys
import argparse
from pathlib import Path

# Ensure the project root is always on sys.path regardless of cwd
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Folder batch processing
# ──────────────────────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"}


def run_folder(folder: str, output_dir: str, mode: str) -> None:
    """
    Process every video file in `folder` through the full WIZ pipeline.

    A single SpeakerIdentityRegistry is shared across all clips so that
    the same physical person receives a consistent PERSON_XXX label in every
    clip's .wiz file — even if pyannote assigns different SPEAKER_XX labels
    per clip.

    Results
    -------
    One .wiz file per clip written to `output_dir`.
    A summary table is printed to stdout when all clips are done.
    """
    from core.pipeline import Pipeline
    from models.speaker_identity import SpeakerIdentityRegistry
    from tasks.diarization_task import DiarizationTask

    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"Error: '{folder}' is not a directory.")
        sys.exit(1)

    video_files = sorted(
        f for f in folder_path.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        print(f"No video files found in '{folder}'")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  WIZ Batch Pipeline")
    print("=" * 60)
    print(f"  Folder : {folder_path.resolve()}")
    print(f"  Output : {output_path.resolve()}")
    print(f"  Mode   : {mode}")
    print(f"  Clips  : {len(video_files)}")
    print("=" * 60)

    # One registry instance — shared across every clip so Person-IDs are stable
    registry = SpeakerIdentityRegistry()
    shared_diarization = DiarizationTask.create_default(identity_registry=registry)

    summary: list = []

    for idx, video_path in enumerate(video_files, start=1):
        print(f"\n[{idx}/{len(video_files)}] {video_path.name}")
        print("-" * 50)

        try:
            # Inject shared diarization task so the registry accumulates
            # speaker embeddings across clips
            pipeline = Pipeline(
                diarization_task=shared_diarization,
                run_mode=mode,
            )
            ctx = pipeline.run(str(video_path))

            # Write .wiz to output_dir (pipeline also writes to results/ but
            # here we write a second copy in the caller-specified location)
            wiz_path = output_path / f"{video_path.stem}.wiz"
            if ctx.video_metadata:
                try:
                    from wiz.writer import WizWriter
                    WizWriter().write(ctx, str(wiz_path))
                except Exception as exc:
                    print(f"  Warning: .wiz write failed: {exc}")

            speakers = sorted({s.speaker_id for s in ctx.speaker_segments})
            summary.append({
                "file":    video_path.name,
                "dur":     f"{ctx.video_metadata.duration_seconds:.1f}s" if ctx.video_metadata else "?",
                "speakers": ", ".join(speakers) or "—",
                "blinks":  len(ctx.blink_events),
                "breaths": len(ctx.breath_events),
                "words":   len(ctx.transcript_words),
                "wiz":     str(wiz_path),
                "ok":      True,
            })

        except Exception as exc:
            print(f"  FAILED: {exc}")
            summary.append({"file": video_path.name, "ok": False, "error": str(exc)})

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BATCH SUMMARY")
    print("=" * 60)
    for row in summary:
        if row["ok"]:
            print(f"  ✓ {row['file']}")
            print(f"      {row['dur']}  |  speakers: {row['speakers']}")
            print(f"      blinks: {row['blinks']}  breaths: {row['breaths']}  words: {row['words']}")
            print(f"      → {row['wiz']}")
        else:
            print(f"  ✗ {row['file']}  —  {row['error']}")

    known = registry.known_persons()
    print("=" * 60)
    if known:
        print(f"  Cross-clip persons identified: {', '.join(known)}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WIZ Video Intelligence Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # start web UI on port 5555
  python main.py --port 8080              # custom port
  python main.py --folder ./clips/        # batch-process a folder
  python main.py --folder ./clips/ --output ./out/ --mode lite
        """,
    )

    # Web server args
    parser.add_argument("--port",   type=int, default=5555,    help="Web server port (default: 5555)")
    parser.add_argument("--host",   default="0.0.0.0",         help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--debug",  action="store_true",        help="Enable Flask debug mode")

    # Batch folder args
    parser.add_argument("--folder", metavar="DIR",              help="Process all videos in this folder")
    parser.add_argument("--output", metavar="DIR", default="results/batch",
                        help="Output directory for batch .wiz files (default: results/batch)")
    parser.add_argument("--mode",   choices=["full", "lite"], default="full",
                        help="Pipeline mode: full (all tasks) or lite (core only, default: full)")

    args = parser.parse_args()

    if args.folder:
        run_folder(args.folder, args.output, args.mode)
        return

    # ── Web UI ─────────────────────────────────────────────────────────────────
    from web.app import app, WEB_DIR

    (WEB_DIR / "uploads").mkdir(parents=True, exist_ok=True)
    (WEB_DIR / "results").mkdir(parents=True, exist_ok=True)
    (WEB_DIR / "logs").mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  WIZ Video Intelligence Pipeline")
    print("=" * 55)
    print(f"  Open your browser → http://localhost:{args.port}")
    print("  Press Ctrl+C to stop")
    print("=" * 55)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()