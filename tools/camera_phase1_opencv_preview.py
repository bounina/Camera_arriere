#!/usr/bin/env python3
"""Phase 1 OpenCV preview with robust color conversion from Picamera2."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from picamera2 import Picamera2

WINDOW_NAME = "phase1-opencv"
SCREENSHOT_DIR = Path("data/screenshots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 OpenCV preview with color-safe conversion")
    parser.add_argument(
        "--force-conversion",
        choices=["rgb2bgr", "bgr", "rgba2bgr", "bgra2bgr", "swaprb"],
        default=None,
        help="Force a conversion strategy instead of automatic selection.",
    )
    parser.add_argument(
        "--dump-first-frame",
        action="store_true",
        help="Dump first raw frame (npy) and converted bgr.png to data/screenshots/.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI and save frames to disk.")
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=30,
        help="Number of frames to save in headless mode before exit.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="Warmup delay before reading preview frames.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=10,
        help="Frames to skip after warmup.",
    )
    return parser.parse_args()


def ensure_output_dir() -> Path:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return SCREENSHOT_DIR


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def try_preview_configuration(picam2: Picamera2) -> tuple[str, object]:
    attempts = ["XRGB8888", "RGBX", "RGB888"]
    last_error: Exception | None = None

    for fmt in attempts:
        try:
            cfg = picam2.create_preview_configuration(main={"format": fmt, "size": (1280, 720)})
            picam2.configure(cfg)
            print(f"[INFO] Configured preview format: {fmt}")
            return fmt, cfg
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            print(f"[WARN] Preview format {fmt} unsupported: {exc}")

    raise RuntimeError(f"Unable to configure preview format from {attempts}: {last_error}")


def score_bgr_candidate(image_bgr: np.ndarray) -> float:
    h, w = image_bgr.shape[:2]
    y0, y1 = h // 3, (2 * h) // 3
    x0, x1 = w // 3, (2 * w) // 3
    roi = image_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        roi = image_bgr

    mean_bgr = roi.mean(axis=(0, 1))
    mean_b, _, mean_r = mean_bgr
    return float(mean_r - mean_b)


def build_converter(raw: np.ndarray, forced: str | None) -> tuple[str, Callable[[np.ndarray], np.ndarray]]:
    channels = raw.shape[2] if raw.ndim == 3 else 1

    if forced == "swaprb":
        return "swaprb", lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if forced == "rgb2bgr":
        return "rgb2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if forced == "bgr":
        return "bgr", lambda frame: frame
    if forced == "rgba2bgr":
        return "rgba2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if forced == "bgra2bgr":
        return "bgra2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if channels == 4:
        conv_rgba = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
        conv_bgra = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        score_rgba = score_bgr_candidate(conv_rgba)
        score_bgra = score_bgr_candidate(conv_bgra)
        if score_rgba >= score_bgra:
            return "auto:rgba2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return "auto:bgra2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if channels == 3:
        conv_rgb = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        score_rgb = score_bgr_candidate(conv_rgb)
        score_bgr = score_bgr_candidate(raw)
        if score_rgb >= score_bgr:
            return "auto:rgb2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return "auto:bgr", lambda frame: frame

    if channels == 1:
        return "auto:gray2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    raise RuntimeError(f"Unsupported frame shape for conversion: {raw.shape}")


def draw_hud(image_bgr: np.ndarray, fps: float, raw_shape: tuple[int, ...], conversion: str) -> np.ndarray:
    out = image_bgr.copy()
    lines = [
        f"FPS: {fps:.1f}",
        f"raw: {raw_shape}",
        f"conv: {conversion}",
    ]
    y = 30
    for text in lines:
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += 28
    return out


def save_bgr_frame(prefix: str, image_bgr: np.ndarray) -> Path:
    out_dir = ensure_output_dir()
    path = out_dir / f"{prefix}_{now_stamp()}.jpg"
    cv2.imwrite(str(path), image_bgr)
    print(f"[INFO] Saved screenshot: {path}")
    return path


def main() -> int:
    args = parse_args()
    auto_headless = not bool(sys.platform.startswith("win")) and not bool(os.environ.get("DISPLAY"))
    headless = args.headless or auto_headless
    if auto_headless and not args.headless:
        print("[INFO] DISPLAY not set, switching to headless mode.")

    picam2 = Picamera2()
    _fmt, _cfg = try_preview_configuration(picam2)

    stop_requested = False

    def _handle_signal(_sig: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print("[INFO] Stop requested.")

    signal.signal(signal.SIGINT, _handle_signal)

    try:
        picam2.start()
        print(f"[INFO] Warmup for {args.warmup_seconds:.1f}s...")
        time.sleep(args.warmup_seconds)

        for _ in range(max(0, args.skip_frames)):
            picam2.capture_array()

        raw_first = picam2.capture_array()
        conversion_name, converter = build_converter(raw_first, args.force_conversion)
        print(f"[INFO] Conversion selected: {conversion_name}")

        bgr_first = converter(raw_first)

        if args.dump_first_frame:
            out_dir = ensure_output_dir()
            raw_path = out_dir / "raw.npy"
            bgr_path = out_dir / "bgr.png"
            np.save(raw_path, raw_first)
            cv2.imwrite(str(bgr_path), bgr_first)
            print(f"[INFO] Dumped first frame raw -> {raw_path}")
            print(f"[INFO] Dumped first frame bgr -> {bgr_path}")

        if not headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        prev_t = time.perf_counter()
        fps = 0.0
        frame_index = 0

        while not stop_requested:
            raw = raw_first if frame_index == 0 else picam2.capture_array()
            bgr = bgr_first if frame_index == 0 else converter(raw)

            now_t = time.perf_counter()
            dt = now_t - prev_t
            prev_t = now_t
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps <= 0 else (0.9 * fps + 0.1 * inst_fps)

            hud_frame = draw_hud(bgr, fps, tuple(raw.shape), conversion_name)

            if headless:
                save_bgr_frame("phase1_headless", hud_frame)
                frame_index += 1
                if frame_index >= max(1, args.headless_frames):
                    print(f"[INFO] Headless frame budget reached: {args.headless_frames}")
                    break
                continue

            cv2.imshow(WINDOW_NAME, hud_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit key pressed.")
                break
            if key == ord("s"):
                save_bgr_frame("phase1", hud_frame)

            frame_index += 1

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Camera stopped and OpenCV windows destroyed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
