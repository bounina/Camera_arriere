#!/usr/bin/env python3
"""Phase 0 camera preview using Picamera2 native QtGL preview only."""

import argparse
import signal
import sys
import time
from pathlib import Path

from picamera2 import Picamera2, Preview


running = True


def _handle_signal(_signum, _frame):
    global running
    running = False


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Picamera2 preview (Phase 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Preview duration in seconds (0 means infinite)",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Save a JPEG snapshot after 2 seconds warmup",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": (args.width, args.height)}
    )
    picam2.configure(preview_config)
    picam2.start_preview(Preview.QTGL)
    picam2.start()
    print("preview started …")

    start_time = time.monotonic()
    snapshot_done = False

    try:
        while running:
            elapsed = time.monotonic() - start_time

            if args.snapshot and not snapshot_done and elapsed >= 2.0:
                output_dir = Path("data/screenshots")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "phase0.jpg"
                picam2.capture_file(str(output_path))
                print("saved snapshot …")
                snapshot_done = True

            if args.timeout > 0 and elapsed >= args.timeout:
                break

            time.sleep(0.05)
    finally:
        picam2.stop_preview()
        picam2.stop()


if __name__ == "__main__":
    sys.exit(main())
