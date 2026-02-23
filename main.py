#!/usr/bin/env python3
"""
WIZ Video Intelligence Pipeline — single entry point.

Usage:
    python main.py                  # start web UI (default, port 5555)
    python main.py --port 8080      # custom port
"""

import sys
import argparse
from pathlib import Path

# Ensure the project root is always on sys.path regardless of cwd
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from web.app import app, WEB_DIR


def main():
    parser = argparse.ArgumentParser(description="WIZ Video Intelligence Pipeline")
    parser.add_argument("--port", type=int, default=5555, help="Web server port (default: 5555)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    # Ensure required directories exist
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