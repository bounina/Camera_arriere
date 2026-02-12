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
        "--horizon-y",
        type=int,
        default=None,
        help="Overlay vanishing horizon (px). Defaults to 45% of frame height.",
    )
    parser.add_argument(
        "--vanish-x",
        type=int,
        default=None,
        help="Overlay vanishing x-center (px). Defaults to frame center.",
    )
    parser.add_argument(
        "--roi-start-y",
        type=int,
        default=None,
        help="Start y for overlay rendering ROI. Defaults to 45% of frame height.",
    )
    parser.add_argument(
        "--lane-width-bottom",
        type=int,
        default=520,
        help="Trajectory width in pixels near the bottom of the frame.",
    )
    parser.add_argument(
        "--lane-width-top",
        type=int,
        default=160,
        help="Trajectory width in pixels near the top of the frame.",
    )
    parser.add_argument(
        "--edge-thickness-near",
        type=int,
        default=8,
        help="Edge line thickness close to vehicle.",
    )
    parser.add_argument(
        "--edge-thickness-far",
        type=int,
        default=3,
        help="Edge line thickness near horizon.",
    )
    parser.add_argument(
        "--zone-alpha-near",
        type=float,
        default=0.40,
        help="Trajectory fill alpha near vehicle.",
    )
    parser.add_argument(
        "--zone-alpha-far",
        type=float,
        default=0.10,
        help="Trajectory fill alpha near horizon.",
    )
    parser.add_argument(
        "--style",
        choices=["tesla", "simple"],
        default="tesla",
        help="Overlay rendering style.",
    )
    parser.add_argument(
        "--show-distance-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable distance markers for fixed guide lines.",
    )
    parser.add_argument(
        "--danger-band",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Highlight danger zone near rear bumper.",
    )
    parser.add_argument(
        "--danger-distance",
        type=float,
        default=0.5,
        help="Danger band distance in meters.",
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


def distance_to_y(distance_m: float, height: int, horizon_y: int) -> int:
    bottom_y = int(height * 0.95)
    max_distance_m = 3.5
    t = max(0.0, min(1.0, distance_m / max_distance_m))
    eased = t**0.8
    y = int(bottom_y - (bottom_y - horizon_y) * eased)
    return y


def resolve_overlay_anchors(args: argparse.Namespace, width: int, height: int) -> tuple[int, int, int]:
    horizon_y = args.horizon_y if args.horizon_y is not None else int(height * 0.45)
    vanish_x = args.vanish_x if args.vanish_x is not None else width // 2
    roi_start_y = args.roi_start_y if args.roi_start_y is not None else int(height * 0.45)

    horizon_y = int(np.clip(horizon_y, 0, height - 1))
    vanish_x = int(np.clip(vanish_x, 0, width - 1))
    roi_start_y = int(np.clip(roi_start_y, 0, height - 1))
    return horizon_y, vanish_x, roi_start_y


def build_trajectory_points(
    width: int,
    height: int,
    steering_deg: float,
    max_steering_deg: float,
    lane_width_bottom_px: int,
    lane_width_top_px: int,
    horizon_y: int,
    vanish_x: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bottom_y = int(height * 0.95)
    sample_count = 90
    y_values = np.linspace(bottom_y, horizon_y, sample_count)

    max_abs = max(1e-6, max_steering_deg)
    turn_ratio = max(-1.0, min(1.0, steering_deg / max_abs))
    max_shift_px = int(width * 0.18)

    left_points: list[list[int]] = []
    right_points: list[list[int]] = []
    progress_values = np.linspace(0.0, 1.0, sample_count)
    for progress, y in zip(progress_values, y_values, strict=False):
        smooth = 3.0 * progress**2 - 2.0 * progress**3
        shift = turn_ratio * max_shift_px * smooth
        center_x = vanish_x + shift * (1.0 - 0.65 * progress)
        lane_width = lane_width_bottom_px + (lane_width_top_px - lane_width_bottom_px) * smooth
        half_w = lane_width * 0.5

        left_anchor = vanish_x - half_w
        right_anchor = vanish_x + half_w
        x_left = int((1.0 - progress) * (center_x - half_w) + progress * left_anchor)
        x_right = int((1.0 - progress) * (center_x + half_w) + progress * right_anchor)

        y_int = int(y)
        left_points.append([x_left, y_int])
        right_points.append([x_right, y_int])

    left_arr = np.array(left_points, dtype=np.int32)
    right_arr = np.array(right_points, dtype=np.int32)
    polygon = np.vstack([left_arr, right_arr[::-1]]).astype(np.int32)
    return left_arr, right_arr, polygon, y_values


def draw_tapered_edge(
    canvas: np.ndarray,
    points: np.ndarray,
    thickness_near: int,
    thickness_far: int,
    glow_color: tuple[int, int, int],
    edge_color: tuple[int, int, int],
) -> None:
    segment_count = len(points) - 1
    for idx in range(segment_count):
        p0 = tuple(points[idx])
        p1 = tuple(points[idx + 1])
        t = idx / max(1, segment_count)
        thickness = int(round(thickness_near + (thickness_far - thickness_near) * t))
        thickness = max(1, thickness)
        cv2.line(canvas, p0, p1, glow_color, thickness + 4, cv2.LINE_AA)
        cv2.line(canvas, p0, p1, edge_color, thickness, cv2.LINE_AA)


def draw_oem_overlay(
    image_bgr: np.ndarray,
    steering_deg: float,
    max_steering_deg: float,
    style: str,
    zone_alpha_near: float,
    zone_alpha_far: float,
    lane_width_bottom_px: int,
    lane_width_top_px: int,
    edge_thickness_near: int,
    edge_thickness_far: int,
    show_distance_markers: bool,
    danger_band: bool,
    danger_distance: float,
    horizon_y: int,
    vanish_x: int,
    roi_start_y: int,
    cached_geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]

    zone_alpha_near = max(0.0, min(1.0, zone_alpha_near))
    zone_alpha_far = max(0.0, min(1.0, zone_alpha_far))
    edge_thickness_near = max(1, edge_thickness_near)
    edge_thickness_far = max(1, edge_thickness_far)
    lane_width_bottom_px = max(40, lane_width_bottom_px)
    lane_width_top_px = max(20, lane_width_top_px)
    roi_start_y = int(np.clip(roi_start_y, 0, h - 1))
    horizon_y = int(np.clip(horizon_y, roi_start_y, h - 1))

    zone_fill_color = (150, 115, 70) if style == "tesla" else (120, 120, 120)
    edge_outer_color = (25, 95, 150) if style == "tesla" else (30, 30, 30)
    edge_inner_color = (0, 190, 255) if style == "tesla" else (220, 220, 220)
    marker_color = (225, 225, 225)
    marker_bg_color = (35, 35, 35)

    left_arr, right_arr, polygon, y_values = cached_geometry
    roi = out[roi_start_y:, :]
    roi_h = roi.shape[0]
    if roi_h <= 1:
        return out

    gradient_bands = 8
    for band in range(gradient_bands):
        y0 = int(roi_start_y + (roi_h * band) / gradient_bands)
        y1 = int(roi_start_y + (roi_h * (band + 1)) / gradient_bands)
        y1 = max(y0 + 1, y1)
        y1 = min(h, y1)
        if y0 >= y1:
            continue
        progress = band / max(1, gradient_bands - 1)
        alpha = zone_alpha_far + (zone_alpha_near - zone_alpha_far) * progress
        alpha = float(np.clip(alpha, 0.0, 1.0))

        mask = np.zeros((y1 - y0, w), dtype=np.uint8)
        shifted_poly = polygon.copy()
        shifted_poly[:, 1] -= y0
        cv2.fillPoly(mask, [shifted_poly], 255, lineType=cv2.LINE_AA)

        band_slice = out[y0:y1, :]
        blend_target = np.full_like(band_slice, zone_fill_color)
        blended = cv2.addWeighted(blend_target, alpha, band_slice, 1.0 - alpha, 0.0)
        band_slice[mask > 0] = blended[mask > 0]

    edge_overlay = np.zeros_like(out)
    draw_tapered_edge(edge_overlay, left_arr, edge_thickness_near, edge_thickness_far, edge_outer_color, edge_inner_color)
    draw_tapered_edge(edge_overlay, right_arr, edge_thickness_near, edge_thickness_far, edge_outer_color, edge_inner_color)

    edge_roi = edge_overlay[roi_start_y:, :]
    out_roi = out[roi_start_y:, :]
    edge_mask = np.any(edge_roi > 0, axis=2)
    out_roi[edge_mask] = edge_roi[edge_mask]

    if danger_band:
        danger_y = distance_to_y(max(0.05, danger_distance), h, horizon_y)
        danger_y = int(np.clip(danger_y, roi_start_y, h - 1))
        danger_overlay = np.zeros((h - danger_y, w, 3), dtype=np.uint8)
        danger_overlay[:, :] = (20, 70, 200)
        danger_roi = out[danger_y:, :]
        danger_alpha = 0.16
        out[danger_y:, :] = cv2.addWeighted(danger_overlay, danger_alpha, danger_roi, 1.0 - danger_alpha, 0.0)

    if show_distance_markers:
        for dist_m in (0.5, 1.0, 2.0, 3.0):
            y = distance_to_y(dist_m, h, horizon_y)
            if y < roi_start_y:
                continue
            idx = int(np.clip(np.argmin(np.abs(y_values - y)), 0, len(y_values) - 1))
            marker_p0 = tuple(left_arr[idx])
            marker_p1 = tuple(right_arr[idx])
            cv2.line(out, marker_p0, marker_p1, marker_color, 1, cv2.LINE_AA)

            text = f"{dist_m:.1f}m"
            text_pos = (marker_p1[0] + 10, marker_p1[1] + 5)
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            rect_tl = (text_pos[0] - 4, text_pos[1] - th - 4)
            rect_br = (text_pos[0] + tw + 4, text_pos[1] + baseline + 2)
            if rect_br[0] >= w:
                shift = rect_br[0] - w + 2
                rect_tl = (rect_tl[0] - shift, rect_tl[1])
                rect_br = (rect_br[0] - shift, rect_br[1])
                text_pos = (text_pos[0] - shift, text_pos[1])

            x0 = max(0, rect_tl[0])
            y0 = max(0, rect_tl[1])
            x1 = min(w, rect_br[0])
            y1 = min(h, rect_br[1])
            if x1 > x0 and y1 > y0:
                bg_patch = np.full((y1 - y0, x1 - x0, 3), marker_bg_color, dtype=np.uint8)
                out[y0:y1, x0:x1] = cv2.addWeighted(bg_patch, 0.45, out[y0:y1, x0:x1], 0.55, 0.0)
            cv2.putText(
                out,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                marker_color,
                1,
                cv2.LINE_AA,
            )

    return out


def draw_hud(
    image_bgr: np.ndarray,
    fps: float,
    raw_shape: tuple[int, ...],
    conversion: str,
    steering_deg: float,
    overlay_enabled: bool,
    style: str,
) -> np.ndarray:
    out = image_bgr.copy()
    lines = [
        f"FPS: {fps:.1f}",
        f"raw: {raw_shape}",
        f"conv: {conversion}",
        f"steering: {steering_deg:+.1f} deg (a/d/r)",
        f"style: {style}",
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
        cached_geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        cached_signature: tuple[float, int, int, int, int, int, int, int] | None = None

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
                h, w = view.shape[:2]
                horizon_y, vanish_x, roi_start_y = resolve_overlay_anchors(args, w, h)
                steering_signature = (
                    round(steering_deg, 3),
                    w,
                    h,
                    args.lane_width_bottom,
                    args.lane_width_top,
                    horizon_y,
                    vanish_x,
                    int(args.max_steering_deg * 1000),
                )
                if cached_geometry is None or cached_signature != steering_signature:
                    cached_geometry = build_trajectory_points(
                        w,
                        h,
                        steering_deg,
                        args.max_steering_deg,
                        args.lane_width_bottom,
                        args.lane_width_top,
                        horizon_y,
                        vanish_x,
                    )
                    cached_signature = steering_signature

                view = draw_oem_overlay(
                    view,
                    steering_deg,
                    args.max_steering_deg,
                    args.style,
                    args.zone_alpha_near,
                    args.zone_alpha_far,
                    args.lane_width_bottom,
                    args.lane_width_top,
                    args.edge_thickness_near,
                    args.edge_thickness_far,
                    args.show_distance_markers,
                    args.danger_band,
                    args.danger_distance,
                    horizon_y,
                    vanish_x,
                    roi_start_y,
                    cached_geometry,
                )
            hud_frame = draw_hud(view, fps, tuple(raw.shape), conversion_name, steering_deg, overlay_enabled, args.style)

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
