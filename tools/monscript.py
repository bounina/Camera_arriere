#!/usr/bin/env python3
"""Camera preview MVP: Picamera2 + OpenCV + parking overlay."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.overlay import generate_curve_points

MAX_STEERING_DEG = 35
STEERING_STEP_DEG = 2


def draw_overlay(frame_bgr: np.ndarray, steering_deg: int, fps: float) -> np.ndarray:
    """Draw guidelines, steering curve and debug text on top of the frame."""
    h, w = frame_bgr.shape[:2]

    # Static guidelines.
    bottom_y = h - 20
    horizon_y = int(h * 0.48)
    cv2.line(frame_bgr, (int(w * 0.33), bottom_y), (int(w * 0.44), horizon_y), (0, 255, 0), 2)
    cv2.line(frame_bgr, (int(w * 0.67), bottom_y), (int(w * 0.56), horizon_y), (0, 255, 0), 2)

    # Dynamic center curve.
    curve_points = np.array(generate_curve_points(w, h, steering_deg), dtype=np.int32)
    cv2.polylines(frame_bgr, [curve_points], isClosed=False, color=(0, 165, 255), thickness=3)

    # HUD text.
    cv2.putText(
        frame_bgr,
        f"FPS: {fps:5.1f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"RES: {w}x{h}",
        (12, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"STEERING: {steering_deg:+d} deg",
        (12, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rear-camera preview with simple AR overlay")
    parser.add_argument("--width", type=int, default=1280, help="Preview width")
    parser.add_argument("--height", type=int, default=720, help="Preview height")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display and save a few processed frames to data/screenshots/",
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=30,
        help="Number of frames to process in headless mode before exiting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steering_deg = 0

    screenshot_dir = Path("data/screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    frame_count = 0
    total_frames = 0
    fps = 0.0
    fps_window_start = time.perf_counter()

    try:
        while True:
            rgb_frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            frame_count += 1
            total_frames += 1
            now = time.perf_counter()
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_window_start = now

            frame_bgr = draw_overlay(frame_bgr, steering_deg, fps)

            if args.headless:
                if total_frames % 10 == 0:
                    file_path = screenshot_dir / f"headless_{int(time.time())}.jpg"
                    cv2.imwrite(str(file_path), frame_bgr)
                    print(f"saved: {file_path}")
                if args.headless_frames > 0 and total_frames >= args.headless_frames:
                    break
                continue

            cv2.imshow("Camera arriere - Phase 1", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("a"):
                steering_deg = max(-MAX_STEERING_DEG, steering_deg - STEERING_STEP_DEG)
            elif key == ord("d"):
                steering_deg = min(MAX_STEERING_DEG, steering_deg + STEERING_STEP_DEG)
            elif key == ord("r"):
                steering_deg = 0
            elif key == ord("s"):
                filename = screenshot_dir / f"capture_{int(time.time())}.jpg"
                cv2.imwrite(str(filename), frame_bgr)
                print(f"capture saved: {filename}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
