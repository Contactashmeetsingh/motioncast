"""Image -> pitch homography from detected pitch keypoints."""

from __future__ import annotations

import cv2
import numpy as np

from motioncast.pitch import PITCH, SoccerPitch


def estimate_homography(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray | None,
    pitch: SoccerPitch = PITCH,
    min_conf: float = 0.5,
    min_points: int = 4,
    max_reprojection_error_cm: float = 150.0,
) -> np.ndarray | None:
    """Fit a 3x3 image->pitch homography, or return None if the view is unreliable.

    Args:
        keypoints_xy: (32, 2) predicted image coordinates, in pitch-vertex order.
        keypoints_conf: (32,) confidences, or None to accept every non-zero point.
    """
    keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32).reshape(-1, 2)
    mask = (keypoints_xy[:, 0] > 1) & (keypoints_xy[:, 1] > 1)
    if keypoints_conf is not None:
        mask &= np.asarray(keypoints_conf).reshape(-1) >= min_conf
    if int(mask.sum()) < min_points:
        return None

    src = keypoints_xy[mask]
    dst = pitch.vertices[mask]
    # Collinear keypoints (e.g. only the halfway line) give a degenerate fit.
    if np.linalg.matrix_rank(np.c_[src, np.ones(len(src))]) < 3:
        return None
    method = cv2.RANSAC if len(src) >= 6 else 0
    h, _ = cv2.findHomography(src, dst, method, ransacReprojThreshold=100.0)
    if h is None or not np.all(np.isfinite(h)):
        return None
    err = np.linalg.norm(transform_points(h, src) - dst, axis=1)
    if np.median(err) > max_reprojection_error_cm:
        return None
    return h


def transform_points(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return points
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), h).reshape(-1, 2)


def fill_homographies(hs: list[np.ndarray | None], max_gap: int = 50) -> list[np.ndarray | None]:
    """Fill frames with no homography from the nearest frame that has one (within max_gap)."""
    n = len(hs)
    valid = [i for i, h in enumerate(hs) if h is not None]
    if not valid:
        return list(hs)
    out: list[np.ndarray | None] = list(hs)
    valid_arr = np.array(valid)
    for i in range(n):
        if out[i] is not None:
            continue
        j = valid_arr[np.argmin(np.abs(valid_arr - i))]
        if abs(j - i) <= max_gap:
            out[i] = hs[j]
    return out
