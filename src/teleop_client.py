#!/usr/bin/env python3
"""
teleop_client.py — Client de téléopération manuelle (PC Windows).

Se connecte à la Raspberry Pi pour :
  1. Recevoir le flux vidéo JPEG-over-TCP (port 8885)
  2. Envoyer les commandes clavier vers le serveur C++ (port 8884)

Contrôle style télécommande RC :
  - Z maintenu  → accélération progressive (rampe)
  - S maintenu  → accélération arrière progressive
  - Relâché     → décélération automatique (freinage moteur)
  - Q/D         → direction progressive gauche/droite
  - ESPACE      → arrêt d'urgence immédiat
  - T           → test servo (sweep direction)
  - ESC         → quitter

Usage :
    python teleop_client.py --pi-ip 192.168.1.42
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

# ======== PARAMÈTRES DE CONDUITE RC ========
MAX_FWD_SPEED  = 0.20     # vitesse max avant
MAX_BWD_SPEED  = -0.15    # vitesse max arrière
ACCEL_STEP     = 0.008    # accélération par frame (~0.24/s à 30fps)
DECEL_FACTOR   = 0.92     # décélération auto quand aucune touche (multiplicateur)
DEAD_ZONE      = 0.02     # en-dessous de ça, on met à 0

ANGLE_STEP     = 0.1      # incrément direction par appui Q/D
MAX_ANGLE      = 1.0
ANGLE_RETURN   = 0.03     # rappel au centre automatique par frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RC-style teleoperation client.")
    parser.add_argument("--pi-ip", type=str, required=True,
                        help="IP address of the Raspberry Pi")
    parser.add_argument("--video-port", type=int, default=8885,
                        help="Video streaming port (default: 8885)")
    parser.add_argument("--cmd-port", type=int, default=8884,
                        help="C++ command server port (default: 8884)")
    return parser.parse_args()


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly `size` bytes from socket."""
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
    """Connect to the C++ command server."""
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
    """Send a command to the C++ server. Returns False on error."""
    if sock is None:
        return False
    try:
        sock.sendall((cmd + "\n").encode("utf-8"))
        return True
    except (BrokenPipeError, ConnectionResetError, BlockingIOError, OSError):
        return False


def draw_hud(frame: np.ndarray, fps: float, speed: float, angle: float,
             video_ok: bool, cmd_ok: bool, throttle_state: str) -> np.ndarray:
    """Draw a heads-up display overlay."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Semi-transparent dark bar at top
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    # Status line
    vid_color = (0, 255, 0) if video_ok else (0, 0, 255)
    cmd_color = (0, 255, 0) if cmd_ok else (0, 0, 255)
    cv2.putText(out, f"FPS: {fps:.0f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"VIDEO: {'OK' if video_ok else 'LOST'}", (150, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, vid_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"CMD: {'OK' if cmd_ok else 'DISCONNECTED'}", (350, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cmd_color, 2, cv2.LINE_AA)

    # Speed bar (vertical, left side)
    speed_pct = speed / MAX_FWD_SPEED * 100 if MAX_FWD_SPEED != 0 else 0
    direction = throttle_state
    speed_color = (0, 255, 0) if speed > 0 else ((0, 100, 255) if speed < 0 else (200, 200, 200))
    cv2.putText(out, f"Speed: {abs(speed_pct):.0f}% ({direction})", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, speed_color, 2, cv2.LINE_AA)

    # Angle
    cv2.putText(out, f"Angle: {angle:+.2f}", (10, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Steering bar
    bar_cx = w // 2
    bar_y = 118
    bar_half_w = 150
    cv2.line(out, (bar_cx - bar_half_w, bar_y), (bar_cx + bar_half_w, bar_y),
             (100, 100, 100), 3, cv2.LINE_AA)
    needle_x = int(bar_cx + angle * bar_half_w)
    cv2.circle(out, (needle_x, bar_y), 8, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.putText(out, "L", (bar_cx - bar_half_w - 15, bar_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(out, "R", (bar_cx + bar_half_w + 5, bar_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Controls at bottom
    cv2.putText(out, "Z:Accel  S:Recule  Q:Gauche  D:Droite  ESPACE:Stop  T:TestServo  ESC:Quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    return out


def main() -> int:
    args = parse_args()

    # --- Connect ---
    video_sock = connect_video(args.pi_ip, args.video_port)
    cmd_sock = connect_cmd(args.pi_ip, args.cmd_port)
    cmd_ok = cmd_sock is not None

    # --- State ---
    speed = 0.0
    angle = 0.0
    fps = 0.0
    prev_t = time.perf_counter()
    last_cmd_time = 0.0
    CMD_INTERVAL = 0.05  # envoie commandes max 20x/s pour ne pas saturer

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("[TELEOP] Ready. Z/S=throttle, Q/D=steer, SPACE=stop, T=test servo, ESC=quit.")

    try:
        while True:
            # --- Receive video frame ---
            try:
                header = recv_exact(video_sock, 4)
                frame_size = struct.unpack(">I", header)[0]
                if frame_size > 5_000_000:
                    raise ConnectionError("Frame too large")
                jpeg_data = recv_exact(video_sock, frame_size)
            except (ConnectionError, socket.timeout, OSError) as e:
                print(f"[VIDEO] Lost: {e}. Reconnecting...")
                video_sock.close()
                video_sock = connect_video(args.pi_ip, args.video_port)
                continue

            frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # FPS
            now_t = time.perf_counter()
            dt = now_t - prev_t
            prev_t = now_t
            if dt > 0:
                fps = (0.9 * fps + 0.1 / dt) if fps > 0 else (1.0 / dt)

            # --- Keyboard ---
            key = cv2.waitKey(1) & 0xFF
            throttle_active = False
            throttle_state = "COAST"

            if key == 27:  # ESC
                send_command(cmd_sock, "TELEOP:STOP")
                break

            elif key == ord("z"):
                # Accélération avant
                speed = min(speed + ACCEL_STEP, MAX_FWD_SPEED)
                throttle_active = True
                throttle_state = "FWD"

            elif key == ord("s"):
                # Accélération arrière
                speed = max(speed - ACCEL_STEP, MAX_BWD_SPEED)
                throttle_active = True
                throttle_state = "BWD"

            elif key == ord("q"):
                angle = max(-MAX_ANGLE, angle - ANGLE_STEP)

            elif key == ord("d"):
                angle = min(MAX_ANGLE, angle + ANGLE_STEP)

            elif key == ord(" "):
                # Arrêt d'urgence
                speed = 0.0
                angle = 0.0
                throttle_state = "STOP"
                send_command(cmd_sock, "TELEOP:STOP")

            elif key == ord("t"):
                # Test servo sweep
                print("[TEST] Sending servo test command...")
                send_command(cmd_sock, "TELEOP:TEST_SERVO")
                throttle_state = "TEST"

            # --- Décélération automatique ---
            if not throttle_active and key != ord(" ") and key != ord("t"):
                # Décélération progressive quand aucune touche gaz
                speed *= DECEL_FACTOR
                if abs(speed) < DEAD_ZONE:
                    speed = 0.0

                # Rappel direction au centre (léger)
                if abs(angle) > 0.01:
                    if angle > 0:
                        angle = max(0.0, angle - ANGLE_RETURN)
                    else:
                        angle = min(0.0, angle + ANGLE_RETURN)

            # Déterminer throttle_state pour l'affichage
            if throttle_state == "COAST":
                if speed > DEAD_ZONE:
                    throttle_state = "FWD"
                elif speed < -DEAD_ZONE:
                    throttle_state = "BWD"
                else:
                    throttle_state = "STOP"

            # --- Envoi continu des commandes ---
            if now_t - last_cmd_time >= CMD_INTERVAL:
                cmd = f"TELEOP:DRIVE,{speed:.4f},{angle:.4f}"
                ok = send_command(cmd_sock, cmd)
                last_cmd_time = now_t

                if not ok and cmd_ok:
                    print("[CMD] Connection lost, reconnecting...")
                    cmd_sock = connect_cmd(args.pi_ip, args.cmd_port)
                    cmd_ok = cmd_sock is not None
                elif ok:
                    cmd_ok = True

            # --- HUD & display ---
            view = draw_hud(frame, fps, speed, angle, True, cmd_ok, throttle_state)
            cv2.imshow(WINDOW_NAME, view)

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
