"""Soccer pitch geometry and minimap drawing.

Pitch geometry adapted from Roboflow `sports` (MIT License, Copyright (c) 2024 Roboflow).
Coordinates are in centimetres: x along the length (0..12000), y across the width (0..7000).
The 32 vertices are in the same order as the keypoints predicted by
football-pitch-detection.pt.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np


@dataclass(frozen=True)
class SoccerPitch:
    width: int = 7000
    length: int = 12000
    penalty_box_width: int = 4100
    penalty_box_length: int = 2015
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @cached_property
    def vertices(self) -> np.ndarray:
        w, ln = self.width, self.length
        pbw, pbl = self.penalty_box_width, self.penalty_box_length
        gbw, gbl = self.goal_box_width, self.goal_box_length
        r, ps = self.centre_circle_radius, self.penalty_spot_distance
        return np.array([
            (0, 0), (0, (w - pbw) / 2), (0, (w - gbw) / 2), (0, (w + gbw) / 2),
            (0, (w + pbw) / 2), (0, w),
            (gbl, (w - gbw) / 2), (gbl, (w + gbw) / 2),
            (ps, w / 2),
            (pbl, (w - pbw) / 2), (pbl, (w - gbw) / 2), (pbl, (w + gbw) / 2), (pbl, (w + pbw) / 2),
            (ln / 2, 0), (ln / 2, w / 2 - r), (ln / 2, w / 2 + r), (ln / 2, w),
            (ln - pbl, (w - pbw) / 2), (ln - pbl, (w - gbw) / 2),
            (ln - pbl, (w + gbw) / 2), (ln - pbl, (w + pbw) / 2),
            (ln - ps, w / 2),
            (ln - gbl, (w - gbw) / 2), (ln - gbl, (w + gbw) / 2),
            (ln, 0), (ln, (w - pbw) / 2), (ln, (w - gbw) / 2), (ln, (w + gbw) / 2),
            (ln, (w + pbw) / 2), (ln, w),
            (ln / 2 - r, w / 2), (ln / 2 + r, w / 2),
        ], dtype=np.float32)

    edges: tuple = (
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
    )

    def contains(self, xy: np.ndarray, margin_cm: float = 300.0) -> np.ndarray:
        """Boolean mask: which pitch-space points lie on (or just off) the pitch."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return (
            (xy[:, 0] >= -margin_cm) & (xy[:, 0] <= self.length + margin_cm)
            & (xy[:, 1] >= -margin_cm) & (xy[:, 1] <= self.width + margin_cm)
        )


PITCH = SoccerPitch()


class PitchCanvas:
    """Maps pitch centimetres to minimap pixels and draws on it."""

    def __init__(self, pitch: SoccerPitch = PITCH, scale: float = 0.1, padding: int = 50,
                 background_bgr=(46, 110, 43), line_bgr=(235, 240, 235), line_thickness: int = 3):
        self.pitch, self.scale, self.padding = pitch, scale, padding
        h = int(pitch.width * scale) + 2 * padding
        w = int(pitch.length * scale) + 2 * padding
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[:] = background_bgr
        # Alternating mowing stripes, purely cosmetic.
        stripe = max(1, int(pitch.length * scale / 12))
        for i in range(0, w, 2 * stripe):
            img[:, i:i + stripe] = np.clip(np.array(background_bgr) * 1.08, 0, 255)
        for a, b in pitch.edges:
            cv2.line(img, self.to_px(pitch.vertices[a - 1]), self.to_px(pitch.vertices[b - 1]),
                     line_bgr, line_thickness, cv2.LINE_AA)
        cv2.circle(img, self.to_px((pitch.length / 2, pitch.width / 2)),
                   int(pitch.centre_circle_radius * scale), line_bgr, line_thickness, cv2.LINE_AA)
        for x in (pitch.penalty_spot_distance, pitch.length - pitch.penalty_spot_distance):
            cv2.circle(img, self.to_px((x, pitch.width / 2)), 5, line_bgr, -1, cv2.LINE_AA)
        self.base = img

    def to_px(self, xy) -> tuple[int, int]:
        return (int(round(xy[0] * self.scale)) + self.padding,
                int(round(xy[1] * self.scale)) + self.padding)

    def blank(self) -> np.ndarray:
        return self.base.copy()
