"""Overlay helpers for parking guidelines."""

from __future__ import annotations

from typing import List, Tuple


Point = Tuple[int, int]


def generate_curve_points(
    width: int,
    height: int,
    steering_deg: float,
    samples: int = 60,
) -> List[Point]:
    """Return curve points in image coordinates for a simulated steering angle.

    The model is intentionally simple for MVP:
    - 0° gives a mostly straight trajectory.
    - Positive angle bends the curve to the right.
    - Negative angle bends the curve to the left.
    """
    if samples < 2:
        raise ValueError("samples must be >= 2")

    base_x = width // 2
    y_bottom = height - 20
    y_top = int(height * 0.42)
    span = max(1, y_bottom - y_top)

    # Normalize steering to [-1, 1] with ±35° expected input range.
    steer_norm = max(-1.0, min(1.0, steering_deg / 35.0))

    curve: List[Point] = []
    for i in range(samples):
        t = i / (samples - 1)
        y = int(y_bottom - t * span)

        # Quadratic displacement: higher displacement near image top.
        x_offset = int(steer_norm * (t**2) * width * 0.25)
        x = max(0, min(width - 1, base_x + x_offset))
        curve.append((x, y))

    return curve
