#!/usr/bin/env python3
"""
video_streamer.py — Serveur de streaming vidéo JPEG-over-TCP.

Tourne sur la Raspberry Pi. Capture les frames via Picamera2,
les encode en JPEG et les envoie à un client TCP unique sur le port 8885.

Protocole : [4 bytes uint32 big-endian = taille JPEG] [N bytes = données JPEG]

Usage (sur la Pi via SSH) :
    python3 video_streamer.py
    python3 video_streamer.py --port 8885 --width 640 --height 480 --quality 70 --rotate 180
"""

from __future__ import annotations

import argparse
import signal
import socket
import struct
import sys
import time

import cv2
import numpy as np
from picamera2 import Picamera2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JPEG-over-TCP video streamer for Raspberry Pi.")
    parser.add_argument("--port", type=int, default=8885, help="TCP port to listen on (default: 8885)")
    parser.add_argument("--width", type=int, default=640, help="Output width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Output height (default: 480)")
    parser.add_argument("--quality", type=int, default=70, help="JPEG quality 1-100 (default: 70)")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=180,
                        help="Rotation in degrees (default: 180)")
    parser.add_argument("--flip", choices=["none", "h", "v", "hv"], default="none",
                        help="Flip mode (default: none)")
    parser.add_argument("--fps-cap", type=int, default=30, help="Max FPS (default: 30)")
    parser.add_argument("--swap-rb", action="store_true", default=False,
                        help="Force swap Red/Blue channels (use if colors are wrong)")
    parser.add_argument("--fov", choices=["normal", "wide"], default="wide",
                        help="FOV mode: 'wide' uses full sensor for max angle (default: wide)")
    return parser.parse_args()


def apply_orientation(frame: np.ndarray, rotate: int, flip: str) -> np.ndarray:
    """Apply rotation and flip to a frame."""
    if rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotate == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if flip == "h":
        frame = cv2.flip(frame, 1)
    elif flip == "v":
        frame = cv2.flip(frame, 0)
    elif flip == "hv":
        frame = cv2.flip(frame, -1)
    return frame


def try_configure(picam2: Picamera2, width: int, height: int, fov_mode: str) -> None:
    """Configure the camera. In 'wide' mode, use full sensor for widest FOV."""
    sensor_res = picam2.sensor_resolution
    print(f"[STREAMER] Sensor native resolution: {sensor_res[0]}x{sensor_res[1]}")

    # RGB888 first — gives true RGB, easy to convert to BGR for OpenCV
    for fmt in ["RGB888", "XRGB8888", "RGBX"]:
        try:
            if fov_mode == "wide":
                # Use full sensor → widest FOV, downscaled to output size
                cfg = picam2.create_preview_configuration(
                    main={"format": fmt, "size": (width, height)},
                    raw={"size": sensor_res},
                )
            else:
                cfg = picam2.create_preview_configuration(
                    main={"format": fmt, "size": (width, height)}
                )
            picam2.configure(cfg)
            print(f"[STREAMER] Configured: {fmt} @ {width}x{height} (FOV: {fov_mode})")
            return
        except Exception as exc:
            print(f"[STREAMER] Format {fmt} unavailable: {exc}")
    raise RuntimeError("No compatible camera format found")


def convert_to_bgr(frame: np.ndarray, swap_rb: bool = False) -> np.ndarray:
    """Auto-convert captured frame to BGR for JPEG encoding."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    channels = frame.shape[2]
    if channels == 4:
        # XRGB8888 from Picamera2 is actually BGRA in memory
        result = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif channels == 3:
        # Direct copy — Picamera2 RGB888 may already be BGR in memory
        result = frame.copy()
    else:
        result = frame
    # If --swap-rb is set, force a channel swap
    if swap_rb:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


def main() -> int:
    args = parse_args()

    # --- Signal handling ---
    stop = False
    def _on_signal(_sig, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # --- Camera init ---
    picam2 = Picamera2()
    try_configure(picam2, args.width, args.height, args.fov)
    picam2.start()

    # In wide mode, ensure full sensor is used (no digital crop)
    if args.fov == "wide":
        sensor_res = picam2.sensor_resolution
        picam2.set_controls({"ScalerCrop": (0, 0, sensor_res[0], sensor_res[1])})
        print(f"[STREAMER] ScalerCrop set to full sensor: {sensor_res}")

    time.sleep(1.0)  # warmup
    # Flush initial frames
    for _ in range(5):
        picam2.capture_array()
    print("[STREAMER] Camera ready.")

    # --- TCP server ---
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.settimeout(1.0)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print(f"[STREAMER] Listening on 0.0.0.0:{args.port}")

    min_frame_time = 1.0 / args.fps_cap if args.fps_cap > 0 else 0.0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]

    try:
        while not stop:
            # Wait for a client connection
            print("[STREAMER] Waiting for client...")
            client = None
            while not stop and client is None:
                try:
                    client, addr = server.accept()
                    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    print(f"[STREAMER] Client connected: {addr}")
                except socket.timeout:
                    continue

            if stop or client is None:
                break

            # Stream frames to this client
            frame_count = 0
            t_start = time.perf_counter()
            try:
                while not stop:
                    t0 = time.perf_counter()

                    raw = picam2.capture_array()
                    bgr = convert_to_bgr(raw, swap_rb=args.swap_rb)
                    bgr = apply_orientation(bgr, args.rotate, args.flip)

                    # Print diagnostic info for the very first frame
                    if frame_count == 0:
                        print(f"[DIAG] Raw frame: shape={raw.shape}, dtype={raw.dtype}")
                        print(f"[DIAG] Pixel [100,100] raw = {raw[100,100]}")
                        print(f"[DIAG] Pixel [100,100] bgr = {bgr[100,100]}")
                        print(f"[DIAG] swap_rb = {args.swap_rb}")
                        print(f"[DIAG] Si couleurs inversées, relance avec: --swap-rb")

                    ok, jpeg = cv2.imencode(".jpg", bgr, encode_params)
                    if not ok:
                        continue

                    data = jpeg.tobytes()
                    header = struct.pack(">I", len(data))

                    try:
                        client.sendall(header + data)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        print("[STREAMER] Client disconnected.")
                        break

                    frame_count += 1
                    if frame_count % 100 == 0:
                        elapsed = time.perf_counter() - t_start
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"[STREAMER] Streaming... {fps:.1f} FPS, frame size ~{len(data)//1024}KB")

                    # FPS cap
                    elapsed = time.perf_counter() - t0
                    if elapsed < min_frame_time:
                        time.sleep(min_frame_time - elapsed)

            finally:
                client.close()

    finally:
        picam2.stop()
        server.close()
        print("[STREAMER] Stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
