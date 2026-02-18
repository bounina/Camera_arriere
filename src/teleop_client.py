#!/usr/bin/env python3
"""
teleop_client.py — Client de téléopération manuelle (PC Windows).

Se connecte à la Raspberry Pi pour :
  1. Recevoir le flux vidéo JPEG-over-TCP (port 8885)
  2. Envoyer les commandes clavier vers le serveur C++ (port 8884)

Affiche la vidéo nativement avec cv2.imshow (pas de VNC !).

Touches :
    Z   → Avancer    (TELEOP:FWD)
    S   → Reculer    (TELEOP:BWD)
    Q   → Gauche     (TELEOP:LEFT)  — direction progressive
    D   → Droite     (TELEOP:RIGHT) — direction progressive
    ESPACE → Arrêt   (TELEOP:STOP)  — remet vitesse ET angle à zéro
    ESC → Quitter

Usage (sur le PC Windows) :
    python teleop_client.py --pi-ip 192.168.1.42
    python teleop_client.py --pi-ip 192.168.1.42 --video-port 8885 --cmd-port 8884
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

import cv2
import numpy as np

WINDOW_NAME = "Teleop - Voiture Autonome"

# Constantes de pilotage (miroir du C++)
ANGLE_STEP = 0.1
MAX_ANGLE = 1.0
FWD_SPEED = 0.15
BWD_SPEED = -0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual teleoperation client for the autonomous car.")
    parser.add_argument("--pi-ip", type=str, required=True,
                        help="IP address of the Raspberry Pi (e.g. 192.168.1.42)")
    parser.add_argument("--video-port", type=int, default=8885,
                        help="Video streaming port (default: 8885)")
    parser.add_argument("--cmd-port", type=int, default=8884,
                        help="C++ command server port (default: 8884)")
    return parser.parse_args()


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly `size` bytes from socket, or raise on disconnect."""
    buf = b""
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed by remote")
        buf += chunk
    return buf


def connect_video(ip: str, port: int) -> socket.socket:
    """Connect to the Pi video streamer with retries."""
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((ip, port))
            s.settimeout(2.0)
            print(f"[VIDEO] Connected to {ip}:{port}")
            return s
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            print(f"[VIDEO] Waiting for streamer ({e})... retry in 2s")
            time.sleep(2.0)


def connect_cmd(ip: str, port: int) -> socket.socket | None:
    """Connect to the C++ command server (non-blocking, best effort)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect((ip, port))
        s.setblocking(False)
        print(f"[CMD] Connected to C++ server {ip}:{port}")
        return s
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        print(f"[CMD] Could not connect to C++ server: {e}")
        return None


def send_command(sock: socket.socket | None, cmd: str) -> bool:
    """Send a TELEOP command to the C++ server. Returns False on error."""
    if sock is None:
        return False
    try:
        sock.sendall((cmd + "\n").encode("utf-8"))
        return True
    except (BrokenPipeError, ConnectionResetError, BlockingIOError, OSError):
        return False


def draw_hud(frame: np.ndarray, fps: float, speed: float, angle: float,
             video_ok: bool, cmd_ok: bool) -> np.ndarray:
    """Draw a heads-up display overlay on the frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Semi-transparent dark bar at top
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    # Status line
    vid_status = "OK" if video_ok else "LOST"
    cmd_status = "OK" if cmd_ok else "DISCONNECTED"
    vid_color = (0, 255, 0) if video_ok else (0, 0, 255)
    cmd_color = (0, 255, 0) if cmd_ok else (0, 0, 255)

    cv2.putText(out, f"FPS: {fps:.0f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"VIDEO: {vid_status}", (150, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, vid_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"CMD: {cmd_status}", (350, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cmd_color, 2, cv2.LINE_AA)

    # Speed & angle
    speed_pct = abs(speed) * 100
    direction = "FWD" if speed > 0 else ("BWD" if speed < 0 else "STOP")
    cv2.putText(out, f"Speed: {speed_pct:.0f}% ({direction})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Angle: {angle:+.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Steering bar
    bar_cx = w // 2
    bar_y = 108
    bar_half_w = 150
    cv2.line(out, (bar_cx - bar_half_w, bar_y), (bar_cx + bar_half_w, bar_y),
             (100, 100, 100), 3, cv2.LINE_AA)
    needle_x = int(bar_cx + angle * bar_half_w)
    cv2.circle(out, (needle_x, bar_y), 8, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.putText(out, "L", (bar_cx - bar_half_w - 15, bar_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(out, "R", (bar_cx + bar_half_w + 5, bar_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Controls reminder at bottom
    cv2.putText(out, "Z:Avance  S:Recule  Q:Gauche  D:Droite  ESPACE:Stop  ESC:Quitter",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return out


def main() -> int:
    args = parse_args()

    # --- Connect to video stream ---
    video_sock = connect_video(args.pi_ip, args.video_port)

    # --- Connect to C++ command server ---
    cmd_sock = connect_cmd(args.pi_ip, args.cmd_port)
    cmd_ok = cmd_sock is not None

    # --- State ---
    current_speed = 0.0
    current_angle = 0.0
    fps = 0.0
    prev_t = time.perf_counter()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("[TELEOP] Ready. Use Z/Q/S/D/ESPACE to drive. ESC to quit.")

    try:
        while True:
            # --- Receive one JPEG frame ---
            try:
                header = recv_exact(video_sock, 4)
                frame_size = struct.unpack(">I", header)[0]
                if frame_size > 5_000_000:  # sanity check: 5MB max
                    print(f"[VIDEO] Abnormal frame size: {frame_size}, reconnecting...")
                    raise ConnectionError("Frame too large")
                jpeg_data = recv_exact(video_sock, frame_size)
            except (ConnectionError, socket.timeout, OSError) as e:
                print(f"[VIDEO] Lost connection: {e}. Reconnecting...")
                video_sock.close()
                video_sock = connect_video(args.pi_ip, args.video_port)
                continue

            # Decode JPEG
            frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # FPS calculation
            now_t = time.perf_counter()
            dt = now_t - prev_t
            prev_t = now_t
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps <= 0 else (0.9 * fps + 0.1 * inst_fps)

            # --- Draw HUD & display ---
            view = draw_hud(frame, fps, current_speed, current_angle,
                            True, cmd_ok)
            cv2.imshow(WINDOW_NAME, view)

            # --- Keyboard input ---
            key = cv2.waitKey(1) & 0xFF

            cmd = None
            if key == 27:  # ESC
                # Send stop before quitting
                send_command(cmd_sock, "TELEOP:STOP")
                break
            elif key == ord("z"):
                cmd = "TELEOP:FWD"
                current_speed = FWD_SPEED
            elif key == ord("s"):
                cmd = "TELEOP:BWD"
                current_speed = BWD_SPEED
            elif key == ord("q"):
                cmd = "TELEOP:LEFT"
                current_angle = max(-MAX_ANGLE, current_angle - ANGLE_STEP)
            elif key == ord("d"):
                cmd = "TELEOP:RIGHT"
                current_angle = min(MAX_ANGLE, current_angle + ANGLE_STEP)
            elif key == ord(" "):
                cmd = "TELEOP:STOP"
                current_speed = 0.0
                current_angle = 0.0

            if cmd is not None:
                ok = send_command(cmd_sock, cmd)
                if not ok and cmd_ok:
                    # Lost connection, try to reconnect
                    print("[CMD] Connection lost, attempting reconnect...")
                    cmd_sock = connect_cmd(args.pi_ip, args.cmd_port)
                    cmd_ok = cmd_sock is not None
                    if cmd_ok:
                        send_command(cmd_sock, cmd)
                elif ok:
                    cmd_ok = True

    except KeyboardInterrupt:
        send_command(cmd_sock, "TELEOP:STOP")
        print("\n[TELEOP] Interrupted.")
    finally:
        video_sock.close()
        if cmd_sock:
            cmd_sock.close()
        cv2.destroyAllWindows()
        print("[TELEOP] Stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
