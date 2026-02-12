#!/usr/bin/env python3
"""Phase 0 camera smoke test: clean colors and minimal HUD, no project overlays."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 camera smoke test (Picamera2 + OpenCV)")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Scale factor before display (0.5 is often better over VNC)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI; save a few frames to data/screenshots then quit",
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=30,
        help="How many frames to save/process before exiting in headless mode",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Alias for headless mode (also auto-enabled when DISPLAY is missing)",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=1.0,
        help="Delay after camera start before processing frames",
    )
    parser.add_argument(
        "--warmup-skip-frames",
        type=int,
        default=10,
        help="Frames to skip after warmup delay before measurements/saves",
    )
    parser.add_argument(
        "--dump-first-frame",
        action="store_true",
        help="Save raw first frame (raw.npy) and converted first frame (bgr.png)",
    )
    return parser.parse_args()


def configure_camera(picam2: Picamera2, width: int, height: int) -> tuple[str, tuple[int, int]]:
    """Configure Picamera2 and return announced main format and size."""
    config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(config)

    main = picam2.camera_configuration().get("main", {})
    announced_format = str(main.get("format", "UNKNOWN"))
    announced_size = tuple(main.get("size", (width, height)))
    return announced_format, announced_size


def convert_to_bgr(frame: np.ndarray, announced_format: str) -> tuple[np.ndarray, str]:
    """Convert capture_array output to BGR for OpenCV display/write.

    Conversion is based on the *actual* frame shape/channels, with format used only as a hint.
    """
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), "GRAY->BGR"

    if frame.ndim != 3:
        raise ValueError(f"Unsupported frame ndim={frame.ndim}, shape={frame.shape}")

    channels = frame.shape[2]
    fmt = announced_format.upper()

    if channels == 2:
        if "UYVY" in fmt:
            return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY), "YUV422(UYVY)->BGR"
        if "YVYU" in fmt:
            return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YVYU), "YUV422(YVYU)->BGR"
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV), "YUV422(YUYV/default)->BGR"

    if channels == 3:
        if "BGR" in fmt:
            return frame.copy(), "BGR(3ch)->BGR"
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), "RGB(3ch)->BGR"

    if channels == 4:
        if "BGR" in fmt or "XBGR" in fmt or "BGRA" in fmt:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), "BGRA/XBGR(4ch)->BGR"
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR), "RGBA/RGBX(4ch)->BGR"

    raise ValueError(f"Unsupported channel count={channels}, shape={frame.shape}")


def add_hud(frame_bgr: np.ndarray, fps: float, announced_format: str, raw_shape: tuple[int, ...]) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    cv2.putText(out, f"FPS: {fps:5.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(out, f"RES: {w}x{h}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        out,
        f"FMT: {announced_format} RAW:{raw_shape}",
        (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return out


def main() -> None:
    args = parse_args()

    screenshot_dir = Path("data/screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    if args.no_gui:
        args.headless = True

    if not os.environ.get("DISPLAY"):
        args.headless = True

    picam2 = Picamera2()
    announced_format, announced_size = configure_camera(picam2, args.width, args.height)

    print(
        "[startup] "
        f"requested={args.width}x{args.height} "
        f"announced={announced_size[0]}x{announced_size[1]} "
        f"format={announced_format} "
        f"display_scale={args.display_scale:.2f} "
        f"headless={args.headless}"
    )

    shown_windows = False
    first_frame_logged = False
    first_dump_done = False
    total_frames = 0
    fps_frames = 0
    fps = 0.0
    fps_t0 = time.perf_counter()

    try:
        picam2.start()

        if args.warmup_seconds > 0:
            time.sleep(args.warmup_seconds)

        for _ in range(max(0, args.warmup_skip_frames)):
            _ = picam2.capture_array()

        while True:
            raw = picam2.capture_array()
            bgr, conversion = convert_to_bgr(raw, announced_format)
            total_frames += 1
            fps_frames += 1

            now = time.perf_counter()
            dt = now - fps_t0
            if dt >= 1.0:
                fps = fps_frames / dt
                fps_frames = 0
                fps_t0 = now

            if not first_frame_logged:
                print(
                    "[first-frame] "
                    f"announced_format={announced_format} "
                    f"raw_shape={raw.shape} raw_dtype={raw.dtype} "
                    f"conversion={conversion}"
                )
                first_frame_logged = True

            if args.dump_first_frame and not first_dump_done:
                np.save(screenshot_dir / "raw.npy", raw)
                cv2.imwrite(str(screenshot_dir / "bgr.png"), bgr)
                print(f"[dump] saved {screenshot_dir / 'raw.npy'} and {screenshot_dir / 'bgr.png'}")
                first_dump_done = True

            hud = add_hud(bgr, fps, announced_format, tuple(raw.shape))

            if args.headless:
                cv2.imwrite(str(screenshot_dir / f"headless_{total_frames:03d}.png"), hud)
                if total_frames >= args.headless_frames:
                    print(f"[headless] completed after {total_frames} frame(s)")
                    break
                continue

            if args.display_scale != 1.0:
                hud = cv2.resize(hud, dsize=None, fx=args.display_scale, fy=args.display_scale)

            if not shown_windows:
                cv2.namedWindow("camera-smoke-test", cv2.WINDOW_NORMAL)
                shown_windows = True

            cv2.imshow("camera-smoke-test", hud)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                shot = screenshot_dir / f"smoke_{int(time.time())}.png"
                cv2.imwrite(str(shot), bgr)
                print(f"[screenshot] saved {shot}")

    except KeyboardInterrupt:
        print("\n[shutdown] interrupted by user (Ctrl+C)")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("[shutdown] camera stopped, windows destroyed")


if __name__ == "__main__":
    main()
