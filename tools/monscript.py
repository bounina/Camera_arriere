#!/usr/bin/env python3
"""Camera preview MVP: Picamera2 + OpenCV + parking overlay."""

from __future__ import annotations

import argparse
import os
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
WARMUP_SECONDS = 1.0
WARMUP_SKIP_FRAMES = 10


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
    parser.add_argument("--width", type=int, default=None, help="Preview width")
    parser.add_argument("--height", type=int, default=None, help="Preview height")
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.5,
        help="Scale factor applied before cv2.imshow (e.g. 0.5 halves display size)",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Display raw frames without parking overlay/HUD",
    )
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
    parser.add_argument(
        "--dump-first-frame",
        action="store_true",
        help="Save raw.npy and bgr.png for the first captured frame before overlay",
    )
    return parser.parse_args()


def can_use_opencv_gui() -> bool:
    """Return True when OpenCV highgui can create windows on this host."""
    try:
        cv2.namedWindow("__camera_arriere_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__camera_arriere_probe__")
    except cv2.error:
        return False
    return True


def configure_camera(
    picam2: Picamera2,
    width: int,
    height: int,
) -> tuple[str, bool, tuple[int, int]]:
    """Configure camera preview stream and infer whether RGB->BGR conversion is needed.

    Returns (effective_format, needs_rgb_to_bgr_conversion, effective_main_size).
    """
    requested_size = (width, height)

    rgb_config = picam2.create_preview_configuration(
        main={"size": requested_size, "format": "RGB888"}
    )
    picam2.configure(rgb_config)

    configured_main = picam2.camera_configuration().get("main", {})
    effective_format = str(configured_main.get("format", "RGB888")).upper()
    effective_size = tuple(configured_main.get("size", requested_size))

    rgb_like_formats = {"RGB888", "RGBX8888", "RGBA8888", "XBGR8888"}
    bgr_like_formats = {"BGR888", "BGRX8888", "BGRA8888", "XRGB8888"}

    if effective_format in rgb_like_formats:
        needs_rgb_to_bgr = True
    elif effective_format in bgr_like_formats:
        needs_rgb_to_bgr = False
    else:
        # Keep previous behavior for unknown 3-channel formats.
        needs_rgb_to_bgr = True

    return effective_format, needs_rgb_to_bgr, effective_size


def to_bgr(frame: np.ndarray, needs_rgb_to_bgr: bool) -> np.ndarray:
    """Convert camera frame into BGR safely across 1/3/4-channel formats."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    channels = frame.shape[2]
    if channels == 3:
        if needs_rgb_to_bgr:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame.copy()
    if channels == 4:
        if needs_rgb_to_bgr:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Unsupported channel count: {channels}")


def main() -> None:
    args = parse_args()
    ssh_detected = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))
    display_env = os.environ.get("DISPLAY", "")
    x11_forwarded = ssh_detected and "localhost:" in display_env

    width_explicit = "--width" in sys.argv
    height_explicit = "--height" in sys.argv

    if args.width is None:
        args.width = 640 if x11_forwarded else 1280
    if args.height is None:
        args.height = 360 if x11_forwarded else 720

    if not width_explicit and not height_explicit and x11_forwarded:
        # Ensure both dimensions are aligned to a lightweight X11 forwarding profile.
        args.width, args.height = 640, 360

    auto_headless = False
    if not args.headless and not can_use_opencv_gui():
        auto_headless = True
        args.headless = True

    steering_deg = 0

    screenshot_dir = Path("data/screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()
    effective_format, needs_rgb_to_bgr, effective_size = configure_camera(
        picam2, args.width, args.height
    )

    stream_config = picam2.camera_configuration().get("main", {})
    stream_size = stream_config.get("size", effective_size)
    stream_format = stream_config.get("format", effective_format)

    print(
        "[startup] "
        f"DISPLAY={display_env or '<unset>'} "
        f"resolution={args.width}x{args.height} "
        f"stream_main_size={stream_size[0]}x{stream_size[1]} "
        f"stream_main_format={stream_format} "
        f"conversion_mode={'RGB->BGR' if needs_rgb_to_bgr else 'none'} "
        f"display_scale={args.display_scale:.2f} "
        f"overlay={'off' if args.no_overlay else 'on'} "
        f"ssh_detected={ssh_detected} x11_forwarded={x11_forwarded}"
    )

    picam2.start()
    time.sleep(WARMUP_SECONDS)

    frame_count = 0
    total_frames = 0
    fps = 0.0
    fps_window_start = time.perf_counter()
    first_frame_logged = False
    first_frame_dumped = False

    if not args.headless:
        cv2.namedWindow("Camera arriere - Phase 1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera arriere - Phase 1", 800, 450)
        cv2.moveWindow("Camera arriere - Phase 1", 50, 50)

    try:
        warmup_remaining = WARMUP_SKIP_FRAMES
        while True:
            raw_frame = picam2.capture_array("main")
            frame_bgr = to_bgr(raw_frame, needs_rgb_to_bgr)

            if not first_frame_logged:
                print(f"[startup] first_frame_shape={raw_frame.shape} dtype={raw_frame.dtype}")
                first_frame_logged = True

            if args.dump_first_frame and not first_frame_dumped:
                raw_dump_path = screenshot_dir / "raw.npy"
                bgr_dump_path = screenshot_dir / "bgr.png"
                np.save(raw_dump_path, raw_frame)
                cv2.imwrite(str(bgr_dump_path), frame_bgr)
                print(f"dumped first frame: {raw_dump_path} and {bgr_dump_path}")
                first_frame_dumped = True

            if warmup_remaining > 0:
                warmup_remaining -= 1
                continue

            frame_count += 1
            total_frames += 1
            now = time.perf_counter()
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_window_start = now

            if not args.no_overlay:
                frame_bgr = draw_overlay(frame_bgr, steering_deg, fps)

            if args.headless:
                if auto_headless and total_frames == 1:
                    print(
                        "OpenCV GUI indisponible (GTK/X11). Bascule automatique en mode --headless."
                    )
                if total_frames % 10 == 0:
                    file_path = screenshot_dir / f"headless_{int(time.time())}.jpg"
                    cv2.imwrite(str(file_path), frame_bgr)
                    print(f"saved: {file_path}")
                if args.headless_frames > 0 and total_frames >= args.headless_frames:
                    break
                continue

            display_frame = frame_bgr
            if args.display_scale != 1.0:
                scaled_w = max(1, int(frame_bgr.shape[1] * args.display_scale))
                scaled_h = max(1, int(frame_bgr.shape[0] * args.display_scale))
                display_frame = cv2.resize(
                    frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
                )

            cv2.imshow("Camera arriere - Phase 1", display_frame)
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
    except KeyboardInterrupt:
        print("\n[shutdown] KeyboardInterrupt reçu, arrêt propre.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
