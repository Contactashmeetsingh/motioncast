"""Per-track kinematics (distance, speed) and heatmaps from pitch-space positions."""

from __future__ import annotations

import numpy as np


def smooth_segments(xy: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average over each contiguous non-NaN run of a (T, 2) track."""
    xy = np.asarray(xy, dtype=float)
    out = np.full_like(xy, np.nan)
    if window <= 1:
        return xy.copy()
    valid = ~np.isnan(xy).any(axis=1)
    t = 0
    n = len(xy)
    while t < n:
        if not valid[t]:
            t += 1
            continue
        start = t
        while t < n and valid[t]:
            t += 1
        seg = xy[start:t]
        half = window // 2
        padded = np.pad(seg, ((half, half), (0, 0)), mode="edge")
        kernel = np.ones(window) / window
        out[start:t, 0] = np.convolve(padded[:, 0], kernel, mode="valid")
        out[start:t, 1] = np.convolve(padded[:, 1], kernel, mode="valid")
    return out


def kinematics(xy_m: np.ndarray, fps: float, max_speed_mps: float) -> dict:
    """Distance covered and speed stats for one smoothed track (metres, NaN = absent).

    Steps implying more than max_speed_mps are treated as homography/ID glitches and
    excluded from both distance and speed.
    """
    xy_m = np.asarray(xy_m, dtype=float)
    step = np.linalg.norm(np.diff(xy_m, axis=0), axis=1)  # NaN across gaps
    speed = step * fps
    ok = np.isfinite(speed) & (speed <= max_speed_mps)
    distance = float(step[ok].sum())
    seconds_tracked = float((~np.isnan(xy_m).any(axis=1)).sum() / fps)
    if ok.sum() >= 5:
        # Top speed over a 1 s window so single-frame noise does not dominate.
        w = max(1, int(round(fps)))
        s = np.where(ok, speed, np.nan)
        if len(s) >= w:
            windows = np.lib.stride_tricks.sliding_window_view(s, w)
            full = ~np.isnan(windows).any(axis=1)
            top = float(np.nanmax(windows[full].mean(axis=1))) if full.any() else float(np.nanmax(s))
        else:
            top = float(np.nanmax(s))
        mean = float(np.nanmean(speed[ok]))
    else:
        top = mean = 0.0
    return {
        "distance_m": round(distance, 1),
        "top_speed_mps": round(top, 2),
        "top_speed_kmh": round(top * 3.6, 1),
        "mean_speed_mps": round(mean, 2),
        "seconds_tracked": round(seconds_tracked, 2),
    }


def heatmap(xy_m: np.ndarray, length_m: float = 120.0, width_m: float = 70.0,
            bins: tuple[int, int] = (24, 14)) -> np.ndarray:
    """2D occupancy histogram (rows = width bins, cols = length bins), normalised to sum 1."""
    xy = np.asarray(xy_m, dtype=float).reshape(-1, 2)
    xy = xy[~np.isnan(xy).any(axis=1)]
    h, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=(bins[1], bins[0]),
                             range=((0, width_m), (0, length_m)))
    total = h.sum()
    return h / total if total else h
