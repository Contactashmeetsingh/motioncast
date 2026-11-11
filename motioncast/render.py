"""Draw pipeline results onto the source video, plus a minimap inset and heatmaps."""

from __future__ import annotations

import shutil
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from motioncast.pitch import PITCH, PitchCanvas

# BGR
TEAM_COLORS = {0: (255, 191, 0), 1: (147, 20, 255), -1: (0, 215, 255)}  # sky blue, pink, gold (ref)
BALL_COLOR = (255, 255, 255)
DARK = (20, 20, 20)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class VideoWriter:
    """H.264 via ffmpeg when available (plays in browsers), else OpenCV mp4v."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.path, self.size = Path(path), size
        w, h = size
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "fast",
                 "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
                stdin=subprocess.PIPE)
            self.cv = None
        else:
            self.proc = None
            self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")
        else:
            self.cv.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _panel(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.6) -> None:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha) + np.array(DARK) * alpha).astype(np.uint8)


def draw_player(img, box, color, label, holder: bool) -> None:
    x1, y1, x2, y2 = box
    cx, cy = int((x1 + x2) / 2), int(y2)
    w = max(12, int((x2 - x1) * 0.7))
    cv2.ellipse(img, (cx, cy), (w, max(4, int(0.35 * w))), 0, -45, 235, color, 2, cv2.LINE_AA)
    if holder:
        cv2.ellipse(img, (cx, cy), (w + 6, max(6, int(0.35 * w) + 4)), 0, 0, 360, (255, 255, 255), 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    bx1, by1 = cx - tw // 2 - 4, cy + 8
    cv2.rectangle(img, (bx1, by1), (bx1 + tw + 8, by1 + th + 8), color, -1)
    cv2.putText(img, label, (bx1 + 4, by1 + th + 3), FONT, 0.45, DARK, 1, cv2.LINE_AA)


def draw_ball(img, xy) -> None:
    x, y = int(xy[0]), int(xy[1])
    pts = np.array([[x, y - 12], [x - 9, y - 26], [x + 9, y - 26]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 230, 255), cv2.LINE_AA)
    cv2.polylines(img, [pts], True, DARK, 1, cv2.LINE_AA)


def draw_hud(img, team_pct: dict, t_s: float) -> None:
    h, w = img.shape[:2]
    pw, ph = 380, 70
    x1, y1 = w // 2 - pw // 2, 16
    _panel(img, x1, y1, x1 + pw, y1 + ph)
    cv2.putText(img, "POSSESSION", (x1 + 14, y1 + 22), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{t_s:5.1f}s", (x1 + pw - 70, y1 + 22), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    a, b = team_pct.get(0, 0.0), team_pct.get(1, 0.0)
    bx1, bx2, by1, by2 = x1 + 14, x1 + pw - 14, y1 + 34, y1 + 56
    total = a + b
    split = bx1 + int((bx2 - bx1) * (a / total)) if total else (bx1 + bx2) // 2
    cv2.rectangle(img, (bx1, by1), (split, by2), TEAM_COLORS[0], -1)
    cv2.rectangle(img, (split, by1), (bx2, by2), TEAM_COLORS[1], -1)
    if total:
        cv2.putText(img, f"{a:.0f}%", (bx1 + 6, by2 - 6), FONT, 0.5, DARK, 1, cv2.LINE_AA)
        txt = f"{b:.0f}%"
        tw = cv2.getTextSize(txt, FONT, 0.5, 1)[0][0]
        cv2.putText(img, txt, (bx2 - tw - 6, by2 - 6), FONT, 0.5, DARK, 1, cv2.LINE_AA)


def draw_radar(canvas: PitchCanvas, dots: list[tuple[np.ndarray, int, bool]],
               ball_trail: list[np.ndarray]) -> np.ndarray:
    img = canvas.blank()
    for i, p in enumerate(ball_trail[:-1]):
        q = ball_trail[i + 1]
        cv2.line(img, canvas.to_px(p * 100), canvas.to_px(q * 100), (255, 255, 255), 2, cv2.LINE_AA)
    for xy_m, team_id, holder in dots:
        c = canvas.to_px(xy_m * 100)
        cv2.circle(img, c, 16, TEAM_COLORS[team_id], -1, cv2.LINE_AA)
        cv2.circle(img, c, 16, (255, 255, 255) if holder else DARK, 3 if holder else 2, cv2.LINE_AA)
    if ball_trail:
        cv2.circle(img, canvas.to_px(ball_trail[-1] * 100), 9, (0, 230, 255), -1, cv2.LINE_AA)
        cv2.circle(img, canvas.to_px(ball_trail[-1] * 100), 9, DARK, 2, cv2.LINE_AA)
    return img


def render_video(res, path: Path, cfg, progress: bool = True) -> None:
    import supervision as sv
    from tqdm import tqdm

    an = res.analysis
    T = len(an.frames)
    canvas = PitchCanvas()
    rw = int(an.width * cfg.radar_scale)
    rh = int(canvas.base.shape[0] * rw / canvas.base.shape[1])
    counts = {0: 0, 1: 0}
    ball_trail: deque = deque(maxlen=int(an.fps))
    gen = sv.get_video_frames_generator(an.video_path, start=an.start_frame, end=an.start_frame + T)
    with VideoWriter(path, an.fps, (an.width, an.height)) as writer:
        for t, frame in enumerate(tqdm(gen, total=T, desc="render", disable=not progress)):
            f = an.frames[t]
            img = frame.copy()
            holder = res.holders[t]
            for tid, box in zip(f.ids, f.xyxy):
                tid = int(tid)
                if tid not in res.display_id:
                    continue
                draw_player(img, box, TEAM_COLORS[res.team[tid]], str(res.display_id[tid]), tid == holder)
            if not np.isnan(res.ball_img[t]).any():
                draw_ball(img, res.ball_img[t])

            if res.poss_team[t] is not None:
                counts[res.poss_team[t]] += 1
            total = counts[0] + counts[1]
            pct = {k: 100 * v / total for k, v in counts.items()} if total else {}
            draw_hud(img, pct, t / an.fps)

            if res.has_h[t]:
                dots = [(res.pitch_xy_m[tid][t], res.team[tid], tid == holder)
                        for tid in res.track_ids if not np.isnan(res.pitch_xy_m[tid][t]).any()]
                if not np.isnan(res.ball_m[t]).any():
                    ball_trail.append(res.ball_m[t])
                else:
                    ball_trail.clear()
                radar = cv2.resize(draw_radar(canvas, dots, list(ball_trail)), (rw, rh),
                                   interpolation=cv2.INTER_AREA)
                x0, y0 = (an.width - rw) // 2, an.height - rh - 12
                roi = img[y0:y0 + rh, x0:x0 + rw]
                roi[:] = cv2.addWeighted(radar, 0.82, roi, 0.18, 0)
            else:
                ball_trail.clear()
            writer.write(img)


def save_heatmaps(res, path: Path) -> None:
    """Side-by-side team occupancy heatmaps over the pitch drawing."""
    canvas = PitchCanvas(scale=0.05, padding=20, line_thickness=2)
    panels = []
    for k in (0, 1):
        pts = np.concatenate([res.pitch_xy_m[tid] for tid in res.track_ids if res.team[tid] == k]
                             or [np.zeros((0, 2))])
        pts = pts[~np.isnan(pts).any(axis=1)]
        base = canvas.blank()
        h, w = base.shape[:2]
        density = np.zeros((h, w), np.float32)
        for p in pts:
            x, y = canvas.to_px(p * 100)
            if 0 <= x < w and 0 <= y < h:
                density[y, x] += 1
        if density.max() > 0:
            density = cv2.GaussianBlur(density, (0, 0), sigmaX=12)
            density /= density.max()
            colored = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            alpha = (np.clip(density * 1.4, 0, 1) * 0.85)[..., None]
            base = (base * (1 - alpha) + colored * alpha).astype(np.uint8)
        bar = np.full((36, w, 3), DARK, np.uint8)
        cv2.circle(bar, (18, 18), 9, TEAM_COLORS[k], -1, cv2.LINE_AA)
        cv2.putText(bar, f"Team {k}  ({len(pts)} player-frames)", (36, 24), FONT, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.vstack([bar, base]))
    gap = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
    cv2.imwrite(str(path), np.hstack([panels[0], gap, panels[1]]))


__all__ = ["render_video", "save_heatmaps", "VideoWriter", "PITCH"]
