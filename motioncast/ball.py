"""Ball candidate selection (online) and trajectory clean-up (offline)."""

from __future__ import annotations

import numpy as np


def pick_ball(candidates_xy: np.ndarray, candidates_conf: np.ndarray,
              last_xy: np.ndarray | None, max_jump_px: float = 250.0) -> int | None:
    """Choose one ball among this frame's detections.

    Prefers the candidate closest to the last known position (if within max_jump_px),
    otherwise the most confident one.
    """
    if len(candidates_xy) == 0:
        return None
    if last_xy is not None:
        d = np.linalg.norm(candidates_xy - last_xy, axis=1)
        i = int(np.argmin(d))
        if d[i] <= max_jump_px:
            return i
    return int(np.argmax(candidates_conf))


def clean_trajectory(xy: np.ndarray, fps: float, max_speed: float, max_gap: int) -> np.ndarray:
    """Remove implausible jumps and linearly interpolate short gaps.

    Args:
        xy: (T, 2) positions in any metric unit (NaN where missing).
        max_speed: max allowed speed in units/second; points arriving faster are dropped.
        max_gap: gaps up to this many frames are interpolated; longer gaps stay NaN.
    """
    xy = np.array(xy, dtype=float)
    t = len(xy)
    valid = ~np.isnan(xy).any(axis=1)

    # Drop outliers: a point is bad if it is far from both valid neighbours relative to time elapsed.
    idx = np.flatnonzero(valid)
    for _ in range(2):
        if len(idx) < 3:
            break
        keep = np.ones(len(idx), dtype=bool)
        for k in range(len(idx)):
            neighbours = [j for j in (k - 1, k + 1) if 0 <= j < len(idx)]
            speeds = [np.linalg.norm(xy[idx[k]] - xy[idx[j]]) / (abs(idx[k] - idx[j]) / fps)
                      for j in neighbours]
            if speeds and min(speeds) > max_speed:
                keep[k] = False
        xy[idx[~keep]] = np.nan
        idx = idx[keep]

    # Interpolate short gaps.
    out = xy.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                out[a + k] = xy[a] + (xy[b] - xy[a]) * k / (b - a)
    assert len(out) == t
    return out
