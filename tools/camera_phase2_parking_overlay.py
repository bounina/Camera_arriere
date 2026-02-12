#!/usr/bin/env python3
# Tests:
#   python3 tools/camera_phase2_parking_overlay.py
#   python3 tools/camera_phase2_parking_overlay.py --dump-first-frame
#   python3 tools/camera_phase2_parking_overlay.py --headless --headless-frames 5
"""Phase 2 OpenCV preview with OEM-style parking overlay and robust color conversion."""

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

WINDOW_NAME = "phase2-parking-overlay"
SCREENSHOT_DIR = Path("data/screenshots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 OpenCV preview with OEM-style parking overlay")
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
    parser.add_argument(
        "--max-steering-deg",
        type=float,
        default=35.0,
        help="Maximum simulated steering angle in degrees.",
    )
    parser.add_argument(
        "--steering-step-deg",
        type=float,
        default=2.0,
        help="Steering increment/decrement in degrees for a/d keys.",
    )
    parser.add_argument(
        "--overlay-style",
        choices=["solid", "dashed"],
        default="solid",
        help="Line style for OEM overlay guides.",
    )
    parser.add_argument(
        "--show-distance-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable distance markers for fixed guide lines.",
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


def draw_dashed_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int = 18,
    gap_length: int = 12,
) -> None:
    for idx in range(len(points) - 1):
        p0 = points[idx].astype(float)
        p1 = points[idx + 1].astype(float)
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1.0:
            continue
        direction = seg / seg_len
        cursor = 0.0
        while cursor < seg_len:
            start = p0 + direction * cursor
            end = p0 + direction * min(cursor + dash_length, seg_len)
            cv2.line(
                image,
                tuple(np.round(start).astype(int)),
                tuple(np.round(end).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )
            cursor += dash_length + gap_length


def draw_polyline(
    image: np.ndarray,
    points: np.ndarray,
    style: str,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if style == "dashed":
        draw_dashed_polyline(image, points, color, thickness)
    else:
        cv2.polylines(image, [points.astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def distance_to_y(distance_m: float, height: int) -> int:
    horizon_y = int(height * 0.36)
    bottom_y = int(height * 0.95)
    max_distance_m = 3.5
    t = max(0.0, min(1.0, distance_m / max_distance_m))
    eased = t**0.8
    y = int(bottom_y - (bottom_y - horizon_y) * eased)
    return y


def draw_oem_overlay(
    image_bgr: np.ndarray,
    steering_deg: float,
    max_steering_deg: float,
    overlay_style: str,
    show_distance_markers: bool,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]

    guide_color = (0, 235, 255)
    marker_color = (255, 255, 255)
    dynamic_color = (0, 200, 0)

    bottom_y = int(h * 0.95)
    top_y = int(h * 0.42)
    center_x = w // 2

    left_bottom_x = int(w * 0.32)
    right_bottom_x = int(w * 0.68)
    left_top_x = int(w * 0.44)
    right_top_x = int(w * 0.56)

    left_points = np.array([[left_bottom_x, bottom_y], [left_top_x, top_y]], dtype=np.int32)
    right_points = np.array([[right_bottom_x, bottom_y], [right_top_x, top_y]], dtype=np.int32)

    draw_polyline(out, left_points, overlay_style, guide_color, 3)
    draw_polyline(out, right_points, overlay_style, guide_color, 3)

    if show_distance_markers:
        for dist_m in (0.5, 1.0, 2.0, 3.0):
            y = distance_to_y(dist_m, h)
            line_points = np.array([[left_bottom_x, y], [right_bottom_x, y]], dtype=np.int32)
            draw_polyline(out, line_points, overlay_style, marker_color, 2)
            cv2.putText(
                out,
                f"{dist_m:.1f}m",
                (right_bottom_x + 10, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                marker_color,
                1,
                cv2.LINE_AA,
            )

    max_abs = max(1e-6, max_steering_deg)
    turn_ratio = max(-1.0, min(1.0, steering_deg / max_abs))

    curve_top_y = int(h * 0.40)
    y_values = np.linspace(bottom_y, curve_top_y, 80)
    points = []
    max_shift_px = int(w * 0.22)
    for y in y_values:
        progress = (bottom_y - y) / max(1.0, bottom_y - curve_top_y)
        shift = turn_ratio * max_shift_px * (progress**2)
        x = int(center_x + shift)
        points.append([x, int(y)])
    dynamic_points = np.array(points, dtype=np.int32)
    draw_polyline(out, dynamic_points, overlay_style, dynamic_color, 3)

    return out


def draw_hud(
    image_bgr: np.ndarray,
    fps: float,
    raw_shape: tuple[int, ...],
    conversion: str,
    steering_deg: float,
    overlay_enabled: bool,
) -> np.ndarray:
    out = image_bgr.copy()
    lines = [
        f"FPS: {fps:.1f}",
        f"raw: {raw_shape}",
        f"conv: {conversion}",
        f"steering: {steering_deg:+.1f} deg (a/d/r)",
        f"overlay: {'on' if overlay_enabled else 'off'} (o)",
        "keys: a d r o s q",
    ]
    y = 30
    for text in lines:
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
        y += 26
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

    steering_deg = 0.0
    overlay_enabled = True

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

            view = bgr
            if overlay_enabled:
                view = draw_oem_overlay(
                    view,
                    steering_deg,
                    args.max_steering_deg,
                    args.overlay_style,
                    args.show_distance_markers,
                )
            hud_frame = draw_hud(view, fps, tuple(raw.shape), conversion_name, steering_deg, overlay_enabled)

            if headless:
                save_bgr_frame("phase2_headless", hud_frame)
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
                save_bgr_frame("phase2", hud_frame)
            if key == ord("o"):
                overlay_enabled = not overlay_enabled
                print(f"[INFO] Overlay {'enabled' if overlay_enabled else 'disabled'}")
            if key == ord("a"):
                steering_deg = max(-args.max_steering_deg, steering_deg - args.steering_step_deg)
            if key == ord("d"):
                steering_deg = min(args.max_steering_deg, steering_deg + args.steering_step_deg)
            if key == ord("r"):
                steering_deg = 0.0

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
