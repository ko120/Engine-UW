#!/usr/bin/env python3
"""
Minimal PET/TTC sample extracted from the RTSSI near-miss logic.

Trajectory format:
    [(frame_index, x, y), ...]

Use the same point for each object that your tracker uses for conflict analysis
in production, usually the bbox center or foot point.
"""

from math import hypot
from typing import Optional, Sequence, Tuple

TrackPoint = Tuple[int, float, float]


def compute_pet(
    track_a: Sequence[TrackPoint],
    track_b: Sequence[TrackPoint],
    crossing_point: Tuple[float, float],
    anchor_frame: int,
    future_steps: int,
    fps: float,
) -> Optional[float]:
    """Post-Encroachment Time in seconds.

    PET is the time gap between object A and object B passing closest to the
    same crossing/conflict point.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")

    end_frame = anchor_frame + future_steps
    traj_a = [p for p in track_a if anchor_frame <= p[0] <= end_frame]
    traj_b = [p for p in track_b if anchor_frame <= p[0] <= end_frame]
    if len(traj_a) < 2 or len(traj_b) < 2:
        return None

    px, py = crossing_point
    closest_a = min(traj_a, key=lambda p: hypot(p[1] - px, p[2] - py))[0]
    closest_b = min(traj_b, key=lambda p: hypot(p[1] - px, p[2] - py))[0]

    return round(abs(closest_a - closest_b) / fps, 2)


def compute_ttc(
    track_a: Sequence[TrackPoint],
    track_b: Sequence[TrackPoint],
    anchor_frame: int,
    future_steps: int,
    fps: float,
    meters_per_pixel: float = 0.03,
) -> Tuple[Optional[float], Optional[float]]:
    """Time-to-Collision/proximity in seconds, plus minimum distance in meters.

    This follows our production fallback: compare object positions at common
    frame indices, find the frame where they are closest, then convert pixels
    to meters using `meters_per_pixel`.

    Returns:
        (ttc_seconds, min_distance_meters)
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be > 0")

    end_frame = anchor_frame + future_steps
    pos_a = {
        frame: (x, y)
        for frame, x, y in track_a
        if anchor_frame <= frame <= end_frame
    }
    pos_b = {
        frame: (x, y)
        for frame, x, y in track_b
        if anchor_frame <= frame <= end_frame
    }
    common_frames = sorted(set(pos_a) & set(pos_b))
    if not common_frames:
        return None, None

    min_frame = anchor_frame
    min_dist_m = float("inf")
    for frame in common_frames:
        ax, ay = pos_a[frame]
        bx, by = pos_b[frame]
        dist_m = hypot(ax - bx, ay - by) * meters_per_pixel
        if dist_m < min_dist_m:
            min_dist_m = dist_m
            min_frame = frame

    ttc_sec = (min_frame - anchor_frame) / fps
    return round(ttc_sec, 2), round(min_dist_m, 2)


if __name__ == "__main__":
    fps = 10.0
    anchor_frame = 100
    future_steps = 60
    crossing_point = (50.0, 50.0)

    object_a = [(100 + i, 20.0 + 2.0 * i, 50.0) for i in range(30)]
    object_b = [(100 + i, 50.0, 90.0 - 2.0 * i) for i in range(30)]

    pet = compute_pet(
        object_a,
        object_b,
        crossing_point,
        anchor_frame,
        future_steps,
        fps,
    )
    ttc, min_dist = compute_ttc(
        object_a,
        object_b,
        anchor_frame,
        future_steps,
        fps,
        meters_per_pixel=0.03,
    )

    print(f"PET: {pet}s")
    print(f"TTC: {ttc}s")
    print(f"Minimum distance: {min_dist}m")
