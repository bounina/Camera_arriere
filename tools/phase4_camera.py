#!/usr/bin/env python3
"""Phase 3: ground-projected parking overlay with keyboard homography calibration + TCP Teleoperation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import socket
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from picamera2 import Picamera2

WINDOW_NAME = "phase3-ground-calibration"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIB_PATH = PROJECT_ROOT / "configs" / "parking_calib.json"
STEP_CHOICES = [1, 5, 20]
POINT_NAMES = ["Near-L", "Near-R", "Far-R", "Far-L"]

# --- (Les fonctions de parsing et d'homographie restent identiques) ---
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 calibration & Teleoperation.")
    parser.add_argument("--force-conversion", choices=["rgb2bgr", "bgr", "rgba2bgr", "bgra2bgr", "swaprb"], default=None)
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None)
    parser.add_argument("--flip", choices=["none", "h", "v", "hv"], default=None)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--skip-frames", type=int, default=10)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--headless-frames", type=int, default=30)
    parser.add_argument("--width-m", type=float, default=None)
    parser.add_argument("--near-y", type=float, default=None)
    parser.add_argument("--far-y", type=float, default=None)
    parser.add_argument("--lane-width-m", type=float, default=None)
    parser.add_argument("--max-distance-m", type=float, default=None)
    parser.add_argument("--zone-alpha", type=float, default=None)
    parser.add_argument("--distance-markers", type=str, default=None)
    parser.add_argument("--marker-step-m", type=float, default=None)
    parser.add_argument("--units", choices=["m", "cm"], default=None)
    return parser.parse_args()

def parse_distance_markers(raw: str | None) -> list[float] | None:
    if not raw: return None
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if token: values.append(float(token))
    return values

def clamp_world_params(width_m, near_y, far_y, lane_width_m, max_distance_m, zone_alpha):
    width_m = max(0.01, float(width_m))
    lane_width_m = max(0.01, float(lane_width_m))
    near_y = max(0.0, float(near_y))
    far_y = max(float(far_y), near_y + 0.05)
    max_distance_m = max(float(max_distance_m), far_y)
    zone_alpha = min(1.0, max(0.0, float(zone_alpha)))
    return width_m, near_y, far_y, lane_width_m, max_distance_m, zone_alpha

def marker_values(near_y, max_distance_m, distance_markers_m, marker_step_m, marker_mode):
    if marker_mode == "list" and distance_markers_m:
        return sorted([v for v in distance_markers_m if near_y <= v <= max_distance_m])
    if marker_step_m is not None and marker_step_m > 0:
        eps = marker_step_m * 0.5
        generated = np.arange(near_y, max_distance_m + eps, marker_step_m)
        return [float(v) for v in generated]
    return []

def value_to_units(value_m: float, units: str) -> float:
    return value_m * 100.0 if units == "cm" else value_m

def units_suffix(units: str) -> str:
    return "cm" if units == "cm" else "m"

def draw_label_with_bg(frame, text, org, scale=0.5):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    x0, y0 = max(0, x - 3), max(0, y - th - 5)
    x1, y1 = min(frame.shape[1] - 1, x + tw + 4), min(frame.shape[0] - 1, y + baseline + 3)
    layer = frame.copy()
    cv2.rectangle(layer, (x0, y0), (x1, y1), (30, 30, 30), -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), 1, cv2.LINE_AA)

def try_preview_configuration(picam2: Picamera2) -> None:
    attempts = ["XRGB8888", "RGBX", "RGB888"]
    last_error = None
    for fmt in attempts:
        try:
            cfg = picam2.create_preview_configuration(main={"format": fmt, "size": (640, 480)}) # Résolution allégée pour la fluidité
            picam2.configure(cfg)
            print(f"[INFO] Configured preview format: {fmt}")
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to configure preview format from {attempts}: {last_error}")

def score_bgr_candidate(image_bgr: np.ndarray) -> float:
    h, w = image_bgr.shape[:2]
    roi = image_bgr[h // 3 : (2 * h) // 3, w // 3 : (2 * w) // 3]
    if roi.size == 0: roi = image_bgr
    mean_bgr = roi.mean(axis=(0, 1))
    return float(mean_bgr[2] - mean_bgr[0])

def build_converter(raw: np.ndarray, forced: str | None) -> tuple[str, Callable[[np.ndarray], np.ndarray]]:
    channels = raw.shape[2] if raw.ndim == 3 else 1
    if forced == "swaprb": return "swaprb", lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if forced == "rgb2bgr": return "rgb2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if forced == "bgr": return "bgr", lambda frame: frame
    if forced == "rgba2bgr": return "rgba2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if forced == "bgra2bgr": return "bgra2bgr", lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

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
    raise RuntimeError(f"Unsupported frame shape: {raw.shape}")

def apply_orientation(frame_bgr: np.ndarray, rotate: int, flip: str) -> np.ndarray:
    oriented = frame_bgr
    if rotate == 90: oriented = cv2.rotate(oriented, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180: oriented = cv2.rotate(oriented, cv2.ROTATE_180)
    elif rotate == 270: oriented = cv2.rotate(oriented, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if flip == "h": oriented = cv2.flip(oriented, 1)
    elif flip == "v": oriented = cv2.flip(oriented, 0)
    elif flip == "hv": oriented = cv2.flip(oriented, -1)
    return oriented

def default_image_points(width: int, height: int) -> np.ndarray:
    return np.array([
        [0.25 * width, 0.78 * height], [0.75 * width, 0.78 * height],
        [0.60 * width, 0.44 * height], [0.40 * width, 0.44 * height],
    ], dtype=np.float32)

def world_rect_points(width_m: float, near_y: float, far_y: float) -> np.ndarray:
    half = width_m / 2.0
    return np.array([[-half, near_y], [half, near_y], [half, far_y], [-half, far_y]], dtype=np.float32)

def compute_h(world_pts: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(world_pts.astype(np.float32), image_pts.astype(np.float32))

def project_world(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = points_xy.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)

def draw_ground_overlay(frame, H, lane_width_m, near_y, max_distance_m, zone_alpha, distance_markers_m, marker_step_m, marker_mode, units):
    out = frame.copy()
    if max_distance_m <= near_y + 1e-6: return out

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

    for dist in marker_values(near_y, max_distance_m, distance_markers_m, marker_step_m, marker_mode):
        seg_world = np.array([[left_x, dist], [right_x, dist]], dtype=np.float32)
        seg_img = project_world(H, seg_world).astype(np.int32)
        p0, p1 = tuple(seg_img[0]), tuple(seg_img[1])
        cv2.line(out, p0, p1, (210, 210, 210), 2, cv2.LINE_AA)
        label_val = value_to_units(dist, units)
        draw_label_with_bg(out, f"{label_val:.2f} {units_suffix(units)}", (p1[0] + 8, p1[1] + 4))
    return out

def draw_calibration(frame, image_points, selected_idx, calibration_mode):
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

def draw_hud(frame, fps, selected_idx, step_px, calibration_mode, tcp_status):
    out = frame.copy()
    lines = [
        f"FPS: {fps:.1f} | TCP: {tcp_status}",
        "DRIVE: Z(Avance) S(Recule) Q(Gauche) D(Droite) ESPACE(Stop)",
        f"CALIB: {'ON' if calibration_mode else 'OFF'} (c) | EDIT: (t)",
        f"Selected: {selected_idx + 1} {POINT_NAMES[selected_idx]} (Move: i/j/k/l)",
        "SAVE: (p) | RELOAD: (o) | QUIT: (ESC)"
    ]
    y = 28
    for line in lines:
        color = (0, 255, 0) if "TCP: OK" in line else (0, 200, 255)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 24
    return out

def save_calibration(path, image_points, width_m, near_y, far_y, lane_width_m, max_distance_m, zone_alpha, distance_markers_m, marker_step_m, marker_mode, rotate, flip, units, force_conversion):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_points": image_points.tolist(), "width_m": width_m, "near_y": near_y, "far_y": far_y,
        "lane_width_m": lane_width_m, "max_distance_m": max_distance_m, "zone_alpha": zone_alpha,
        "distance_markers_m": distance_markers_m, "marker_step_m": marker_step_m, "marker_mode": marker_mode,
        "rotate": rotate, "flip": flip, "units": units, "force_conversion": force_conversion,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] Saved calibration: {path}")

def load_calibration(path: Path) -> dict | None:
    if not path.exists(): return None
    return json.loads(path.read_text(encoding="utf-8"))

def as_float(data: dict, key: str, fallback: float) -> float:
    return float(data.get(key)) if data.get(key) is not None else fallback

def main() -> int:
    args = parse_args()
    calibration_path = CALIB_PATH
    loaded = load_calibration(calibration_path)

    headless = args.headless
    selected_idx = 0
    step_idx = 1
    calibration_mode = True
    param_edit_mode = False

    # Default values
    rotate, flip, units, forced_conversion = 180, "none", "m", None
    width_m, near_y, far_y = 2.6, 0.5, 3.0
    lane_width_m, max_distance_m, zone_alpha = 2.6, 3.0, 0.35
    distance_markers_m, marker_step_m, marker_mode = None, 0.5, "step"

    if loaded is not None:
        try:
            width_m = as_float(loaded, "width_m", width_m)
            near_y = as_float(loaded, "near_y", near_y)
            far_y = as_float(loaded, "far_y", far_y)
            lane_width_m = as_float(loaded, "lane_width_m", lane_width_m)
            max_distance_m = as_float(loaded, "max_distance_m", max_distance_m)
            zone_alpha = as_float(loaded, "zone_alpha", zone_alpha)
            marker_step_m = as_float(loaded, "marker_step_m", marker_step_m)
            rotate = int(loaded.get("rotate", rotate))
            units = str(loaded.get("units", units))
            loaded_force = loaded.get("force_conversion")
            if isinstance(loaded_force, str) and loaded_force: forced_conversion = loaded_force
        except Exception as exc:
            print(f"[WARN] Invalid calibration file defaults: {exc}")

    # Override with args
    if args.rotate is not None: rotate = args.rotate
    if args.units is not None: units = args.units
    if args.force_conversion is not None: forced_conversion = args.force_conversion

    # --- INIT TCP CLIENT POUR LA VOITURE ---
    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_status = "DISCONNECTED"
    try:
        tcp_client.connect(('127.0.0.1', 8884))
        tcp_client.setblocking(False) # Non-bloquant pour la vidéo
        tcp_status = "OK"
        print("[INFO] Connecté au serveur C++ (Voiture) sur le port 8884")
    except Exception as e:
        print(f"[WARN] Serveur C++ introuvable : {e}")

    picam2 = Picamera2()
    try_preview_configuration(picam2)

    stop_requested = False
    def _handle_signal(_sig: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        picam2.start()
        time.sleep(args.warmup_seconds)
        for _ in range(max(0, args.skip_frames)): picam2.capture_array()

        raw_first = picam2.capture_array()
        conv_name, converter = build_converter(raw_first, forced_conversion)
        bgr_first = apply_orientation(converter(raw_first), rotate, flip)
        h, w = bgr_first.shape[:2]
        image_points = default_image_points(w, h)

        if loaded is not None and "image_points" in loaded:
            image_points = np.array(loaded["image_points"], dtype=np.float32)

        if not headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        prev_t = time.perf_counter()
        fps = 0.0
        frame_index = 0

        while not stop_requested:
            raw = raw_first if frame_index == 0 else picam2.capture_array()
            bgr = bgr_first if frame_index == 0 else apply_orientation(converter(raw), rotate, flip)

            now_t = time.perf_counter()
            dt, prev_t = now_t - prev_t, now_t
            if dt > 0: fps = (0.9 * fps + 0.1 * (1.0 / dt)) if fps > 0 else (1.0 / dt)

            width_m, near_y, far_y, lane_width_m, max_distance_m, zone_alpha = clamp_world_params(
                width_m, near_y, far_y, lane_width_m, max_distance_m, zone_alpha)
            
            world_pts = world_rect_points(width_m, near_y, far_y)
            H = compute_h(world_pts, image_points)

            view = draw_ground_overlay(bgr, H, lane_width_m, max(0.0, near_y), max_distance_m, zone_alpha, distance_markers_m, marker_step_m, marker_mode, units)
            view = draw_calibration(view, image_points, selected_idx, calibration_mode)
            view = draw_hud(view, fps, selected_idx, STEP_CHOICES[step_idx], calibration_mode, tcp_status)

            cv2.imshow(WINDOW_NAME, view)
            key = cv2.waitKey(1) & 0xFF
            
            # --- GESTION DES TOUCHES ---
            if key == 27: # Echap (ESC) pour Quitter
                break
                
            # 1. TELEOPERATION (Envoi C++)
            cmd = None
            if key == ord('z'): cmd = "up"
            elif key == ord('s'): cmd = "down"
            elif key == ord('q'): cmd = "left"
            elif key == ord('d'): cmd = "right"
            elif key == ord(' '): cmd = "stop"

            if cmd and tcp_status == "OK":
                try:
                    tcp_client.sendall((cmd + "\n").encode('utf-8'))
                    print(f"[TCP] Sent: {cmd}")
                except BlockingIOError:
                    pass # Normal en non-bloquant si le buffer est plein
                except Exception as e:
                    tcp_status = "ERROR"
                    print(f"[TCP ERREUR] {e}")

            # 2. CALIBRATION & IHM
            if key == ord("c"): calibration_mode = not calibration_mode
            if key == ord("t"): param_edit_mode = not param_edit_mode
            if key in (ord("1"), ord("2"), ord("3"), ord("4")): selected_idx = key - ord("1")
            
            if key == ord("p"):
                save_calibration(calibration_path, image_points, width_m, near_y, far_y, lane_width_m, max_distance_m, zone_alpha, distance_markers_m, marker_step_m, marker_mode, rotate, flip, units, forced_conversion)

            if calibration_mode:
                dx = dy = 0
                step = STEP_CHOICES[step_idx]
                if key == ord("i"): dy = -step
                elif key == ord("k"): dy = step
                elif key == ord("j"): dx = -step
                elif key == ord("l"): dx = step
                if dx or dy:
                    image_points[selected_idx, 0] = np.clip(image_points[selected_idx, 0] + dx, 0, w - 1)
                    image_points[selected_idx, 1] = np.clip(image_points[selected_idx, 1] + dy, 0, h - 1)

            frame_index += 1

    except KeyboardInterrupt:
        print("[INFO] Arrêt demandé.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        if tcp_status == "OK": tcp_client.close()
        print("[INFO] Caméra et OpenCV fermés proprement.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
