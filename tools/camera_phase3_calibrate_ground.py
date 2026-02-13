#!/usr/bin/env python3
"""Phase 3: ground-projected parking overlay with keyboard homography calibration."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from picamera2 import Picamera2

WINDOW_NAME = "phase3-ground-calibration"
CALIB_PATH = Path("configs/parking_calib.json")
STEP_CHOICES = [1, 5, 20]
POINT_NAMES = ["Near-L", "Near-R", "Far-R", "Far-L"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 calibration: project overlay on ground plane with homography")
    parser.add_argument("--force-conversion", choices=["rgb2bgr", "bgr", "rgba2bgr", "bgra2bgr", "swaprb"], default=None)
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=180)
    parser.add_argument("--flip", choices=["none", "h", "v", "hv"], default="none")
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--skip-frames", type=int, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--headless-frames", type=int, default=30)
    parser.add_argument("--width-m", type=float, default=2.6, help="Calibration rectangle width in meters")
    parser.add_argument("--near-y", type=float, default=0.5, help="Near edge distance in meters")
    parser.add_argument("--far-y", type=float, default=3.0, help="Far edge distance in meters")
    parser.add_argument("--lane-width-m", type=float, default=2.6, help="Overlay lane/corridor width in meters")
    parser.add_argument("--max-distance-m", type=float, default=3.0, help="Max distance rendered in meters")
    parser.add_argument("--zone-alpha", type=float, default=0.35)
    return parser.parse_args()


def try_preview_configuration(picam2: Picamera2) -> None:
    attempts = ["XRGB8888", "RGBX", "RGB888"]
    last_error: Exception | None = None
    for fmt in attempts:
        try:
            cfg = picam2.create_preview_configuration(main={"format": fmt, "size": (1280, 720)})
            picam2.configure(cfg)
            print(f"[INFO] Configured preview format: {fmt}")
            return
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            print(f"[WARN] Preview format {fmt} unsupported: {exc}")
    raise RuntimeError(f"Unable to configure preview format from {attempts}: {last_error}")


def score_bgr_candidate(image_bgr: np.ndarray) -> float:
    h, w = image_bgr.shape[:2]
    roi = image_bgr[h // 3 : (2 * h) // 3, w // 3 : (2 * w) // 3]
    if roi.size == 0:
        roi = image_bgr
    mean_bgr = roi.mean(axis=(0, 1))
    return float(mean_bgr[2] - mean_bgr[0])


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
        rgba = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
        bgra = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        if score_bgr_candidate(rgba) >= score_bgr_candidate(bgra):
            return "auto:rgba2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return "auto:bgra2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if channels == 3:
        rgb = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        if score_bgr_candidate(rgb) >= score_bgr_candidate(raw):
            return "auto:rgb2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return "auto:bgr", lambda frame: frame
    if channels == 1:
        return "auto:gray2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported frame shape for conversion: {raw.shape}")


def apply_orientation(frame_bgr: np.ndarray, rotate: int, flip: str) -> np.ndarray:
    oriented = frame_bgr
    if rotate == 90:
        oriented = cv2.rotate(oriented, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        oriented = cv2.rotate(oriented, cv2.ROTATE_180)
    elif rotate == 270:
        oriented = cv2.rotate(oriented, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if flip == "h":
        oriented = cv2.flip(oriented, 1)
    elif flip == "v":
        oriented = cv2.flip(oriented, 0)
    elif flip == "hv":
        oriented = cv2.flip(oriented, -1)
    return oriented


def default_image_points(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [0.25 * width, 0.78 * height],
            [0.75 * width, 0.78 * height],
            [0.60 * width, 0.44 * height],
            [0.40 * width, 0.44 * height],
        ],
        dtype=np.float32,
    )


def world_rect_points(width_m: float, near_y: float, far_y: float) -> np.ndarray:
    half = width_m / 2.0
    return np.array(
        [
            [-half, near_y],
            [half, near_y],
            [half, far_y],
            [-half, far_y],
        ],
        dtype=np.float32,
    )


def compute_h(world_pts: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(world_pts.astype(np.float32), image_pts.astype(np.float32))


def project_world(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = points_xy.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def draw_ground_overlay(
    frame: np.ndarray,
    H: np.ndarray,
    lane_width_m: float,
    near_y: float,
    max_distance_m: float,
    zone_alpha: float,
) -> np.ndarray:
    out = frame.copy()
    if max_distance_m <= near_y + 1e-6:
        return out

    left_x = -lane_width_m / 2.0
    right_x = lane_width_m / 2.0
    y_vals = np.linspace(near_y, max_distance_m, 100)
    left_world = np.stack([np.full_like(y_vals, left_x), y_vals], axis=1)
    right_world = np.stack([np.full_like(y_vals, right_x), y_vals], axis=1)

    left_img = project_world(H, left_world).astype(np.int32)
    right_img = project_world(H, right_world).astype(np.int32)

    polygon = np.vstack([left_img, right_img[::-1]])
    fill = np.zeros_like(frame)
    cv2.fillPoly(fill, [polygon], (150, 115, 70), lineType=cv2.LINE_AA)
    out = cv2.addWeighted(fill, max(0.0, min(1.0, zone_alpha)), out, 1.0, 0.0)

    cv2.polylines(out, [left_img], False, (0, 220, 255), 4, cv2.LINE_AA)
    cv2.polylines(out, [right_img], False, (0, 220, 255), 4, cv2.LINE_AA)

    for dist in (0.5, 1.0, 2.0, 3.0):
        if dist < near_y or dist > max_distance_m:
            continue
        seg_world = np.array([[left_x, dist], [right_x, dist]], dtype=np.float32)
        seg_img = project_world(H, seg_world).astype(np.int32)
        p0, p1 = tuple(seg_img[0]), tuple(seg_img[1])
        cv2.line(out, p0, p1, (210, 210, 210), 2, cv2.LINE_AA)
        cv2.putText(out, f"{dist:.1f}m", (p1[0] + 8, p1[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    return out


def draw_calibration(frame: np.ndarray, image_points: np.ndarray, selected_idx: int, calibration_mode: bool) -> np.ndarray:
    out = frame.copy()
    pts = image_points.astype(np.int32)
    cv2.polylines(out, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
    for idx, (x, y) in enumerate(pts):
        color = (0, 0, 255) if idx == selected_idx else (0, 255, 255)
        radius = 8 if idx == selected_idx else 6
        cv2.circle(out, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        cv2.putText(out, f"{idx+1}:{POINT_NAMES[idx]}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if calibration_mode:
        cv2.putText(out, "CALIBRATION MODE", (10, out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 50, 255), 2, cv2.LINE_AA)
    return out


def draw_hud(frame: np.ndarray, fps: float, selected_idx: int, step_px: int, max_distance_m: float, rotate: int, flip: str, calibration_mode: bool) -> np.ndarray:
    out = frame.copy()
    lines = [
        f"FPS: {fps:.1f}",
        f"selected: {selected_idx + 1} {POINT_NAMES[selected_idx]}",
        f"step_px: {step_px} (+/-)",
        f"max_distance_m: {max_distance_m:.2f} ([/])",
        f"rot={rotate} flip={flip}",
        f"calibration: {'ON' if calibration_mode else 'OFF'} (c)",
        "keys: c 1..4 i/j/k/l +/- [ ] p o q",
    ]
    y = 28
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 24
    return out


def save_calibration(path: Path, image_points: np.ndarray, width_m: float, near_y: float, far_y: float, max_distance_m: float, rotate: int, flip: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_points": image_points.tolist(),
        "width_m": width_m,
        "near_y": near_y,
        "far_y": far_y,
        "max_distance_m": max_distance_m,
        "rotate": rotate,
        "flip": flip,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] Calibration saved: {path}")


def load_calibration(path: Path) -> dict | None:
    if not path.exists():
        print(f"[WARN] Calibration file not found: {path}")
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"[INFO] Calibration loaded: {path}")
    return data


def main() -> int:
    args = parse_args()
    auto_headless = not bool(sys.platform.startswith("win")) and not bool(os.environ.get("DISPLAY"))
    headless = args.headless or auto_headless
    if auto_headless and not args.headless:
        print("[INFO] DISPLAY not set, switching to headless mode.")

    picam2 = Picamera2()
    try_preview_configuration(picam2)

    stop_requested = False

    def _handle_signal(_sig: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)

    selected_idx = 0
    step_idx = 1
    calibration_mode = True

    width_m = float(args.width_m)
    near_y = float(args.near_y)
    far_y = float(args.far_y)
    max_distance_m = float(args.max_distance_m)

    try:
        picam2.start()
        time.sleep(args.warmup_seconds)
        for _ in range(max(0, args.skip_frames)):
            picam2.capture_array()

        raw_first = picam2.capture_array()
        conv_name, converter = build_converter(raw_first, args.force_conversion)
        bgr_first = apply_orientation(converter(raw_first), args.rotate, args.flip)
        h, w = bgr_first.shape[:2]
        image_points = default_image_points(w, h)
        print(f"[INFO] Conversion selected: {conv_name}")

        loaded = load_calibration(CALIB_PATH)
        if loaded is not None:
            try:
                image_points = np.array(loaded["image_points"], dtype=np.float32)
                width_m = float(loaded.get("width_m", width_m))
                near_y = float(loaded.get("near_y", near_y))
                far_y = float(loaded.get("far_y", far_y))
                max_distance_m = float(loaded.get("max_distance_m", max_distance_m))
                print("[INFO] Applied calibration file values.")
            except (KeyError, TypeError, ValueError) as exc:
                print(f"[WARN] Invalid calibration file data: {exc}")

        if not headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        prev_t = time.perf_counter()
        fps = 0.0
        frame_index = 0

        while not stop_requested:
            raw = raw_first if frame_index == 0 else picam2.capture_array()
            bgr = bgr_first if frame_index == 0 else apply_orientation(converter(raw), args.rotate, args.flip)

            now_t = time.perf_counter()
            dt = now_t - prev_t
            prev_t = now_t
            if dt > 0:
                fps_inst = 1.0 / dt
                fps = fps_inst if fps <= 0 else (0.9 * fps + 0.1 * fps_inst)

            near_for_overlay = max(0.0, near_y)
            max_distance_m = max(max_distance_m, near_for_overlay + 0.1)
            world_pts = world_rect_points(width_m, near_y, far_y)
            H = compute_h(world_pts, image_points)

            view = draw_ground_overlay(bgr, H, args.lane_width_m, near_for_overlay, max_distance_m, args.zone_alpha)
            view = draw_calibration(view, image_points, selected_idx, calibration_mode)
            view = draw_hud(view, fps, selected_idx, STEP_CHOICES[step_idx], max_distance_m, args.rotate, args.flip, calibration_mode)

            if headless:
                frame_index += 1
                if frame_index >= max(1, args.headless_frames):
                    print(f"[INFO] Headless frame budget reached: {args.headless_frames}")
                    break
                continue

            cv2.imshow(WINDOW_NAME, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calibration_mode = not calibration_mode
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                selected_idx = key - ord("1")
            if key in (ord("+"), ord("=")):
                step_idx = min(step_idx + 1, len(STEP_CHOICES) - 1)
            if key in (ord("-"), ord("_")):
                step_idx = max(step_idx - 1, 0)
            if key == ord("["):
                max_distance_m = max(near_for_overlay + 0.1, max_distance_m - 0.1)
            if key == ord("]"):
                max_distance_m = min(10.0, max_distance_m + 0.1)
            if key == ord("p"):
                save_calibration(CALIB_PATH, image_points, width_m, near_y, far_y, max_distance_m, args.rotate, args.flip)
            if key == ord("o"):
                loaded = load_calibration(CALIB_PATH)
                if loaded is not None:
                    try:
                        image_points = np.array(loaded["image_points"], dtype=np.float32)
                        width_m = float(loaded.get("width_m", width_m))
                        near_y = float(loaded.get("near_y", near_y))
                        far_y = float(loaded.get("far_y", far_y))
                        max_distance_m = float(loaded.get("max_distance_m", max_distance_m))
                    except (KeyError, TypeError, ValueError) as exc:
                        print(f"[WARN] Could not apply calibration: {exc}")

            if calibration_mode:
                dx = dy = 0
                step = STEP_CHOICES[step_idx]
                if key == ord("i"):
                    dy = -step
                elif key == ord("k"):
                    dy = step
                elif key == ord("j"):
                    dx = -step
                elif key == ord("l"):
                    dx = step
                if dx or dy:
                    image_points[selected_idx, 0] = np.clip(image_points[selected_idx, 0] + dx, 0, w - 1)
                    image_points[selected_idx, 1] = np.clip(image_points[selected_idx, 1] + dy, 0, h - 1)

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
