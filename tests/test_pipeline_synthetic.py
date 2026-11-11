"""Stages 2-3 (postprocess + outputs) on a synthetic clip, with no model weights needed."""

import json

import cv2
import numpy as np
import pytest

from motioncast import pipeline
from motioncast.config import GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, REFEREE_CLASS_ID, PipelineConfig

W, H, FPS, T = 640, 360, 25.0, 40
# Image -> pitch(cm): the frame shows the whole pitch.
IMG_TO_PITCH = np.diag([12000 / W, 7000 / H, 1.0])
RED, BLUE, YELLOW = (40, 40, 220), (220, 60, 40), (0, 220, 220)

# track id -> (class, color, start xy, velocity px/frame)
ACTORS = {
    1: (PLAYER_CLASS_ID, RED, (100, 100), (2, 0)),
    2: (PLAYER_CLASS_ID, RED, (150, 250), (1, -1)),
    3: (PLAYER_CLASS_ID, BLUE, (400, 120), (-1, 1)),
    4: (PLAYER_CLASS_ID, BLUE, (450, 260), (-2, 0)),
    5: (GOALKEEPER_CLASS_ID, (80, 80, 80), (20, 180), (0, 0)),
    6: (REFEREE_CLASS_ID, YELLOW, (320, 300), (0, 0)),
    99: (PLAYER_CLASS_ID, BLUE, (300, 50), (0, 0)),  # appears for 3 frames only -> dropped
}


class ColorEmbedder:
    """Stand-in for SigLIP: the crop's mean colour is a perfect team feature here."""

    def __call__(self, crops):
        return np.array([c.reshape(-1, 3).mean(axis=0) for c in crops])


@pytest.fixture
def synthetic(tmp_path):
    video = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    frames, crops = [], []
    for t in range(T):
        img = np.full((H, W, 3), (40, 140, 40), np.uint8)
        ids, boxes, cls = [], [], []
        for tid, (c, color, (x, y), (vx, vy)) in ACTORS.items():
            if tid == 99 and t >= 3:
                continue
            cx, cy = x + vx * t, y + vy * t
            box = (cx - 8, cy - 30, cx + 8, cy)
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, -1)
            ids.append(tid)
            boxes.append(box)
            cls.append(c)
            if c == PLAYER_CLASS_ID and t % 5 == 0:
                crops.append((tid, img[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy()))
        # Ball sits at player 1's feet.
        bx, by = 100 + 2 * t, 100
        vw.write(img)
        frames.append(pipeline.FrameRecord(
            ids=np.array(ids), xyxy=np.array(boxes, np.float32), cls=np.array(cls),
            conf=np.full(len(ids), 0.9, np.float32), ball_xy=np.array([bx, by], np.float32),
            h=IMG_TO_PITCH if t != 10 else None))  # one frame with no homography
    vw.release()
    an = pipeline.Analysis(str(video), FPS, W, H, 0, frames, crops, timings={"device": "cpu"})
    return an, tmp_path


def test_postprocess_and_outputs(synthetic):
    an, tmp = synthetic
    cfg = PipelineConfig(min_track_frames=10, smoothing_window=3)
    res = pipeline.postprocess(an, cfg, embedder=ColorEmbedder(), progress=False)

    assert 99 not in res.display_id                       # short track dropped
    assert res.role[5] == "goalkeeper" and res.role[6] == "referee"
    assert res.team[1] == res.team[2] != res.team[3] == res.team[4]
    assert res.team[5] == res.team[1]                     # keeper sits by the red team's end
    assert res.team[6] == -1
    assert res.has_h.all()                                # frame 10 filled from a neighbour
    assert res.holders[-1] == 1 and res.poss_team[-1] == res.team[1]

    # Player 1 moves 2 px/frame = 37.5 cm/frame along x -> 9.375 m/s.
    p1 = next(p for p in res.stats["players"] if p["id"] == res.display_id[1])
    assert abs(p1["top_speed_mps"] - 9.375) < 0.2
    assert res.stats["possession"]["team_pct"][str(res.team[1])] == 100.0

    paths = pipeline.write_outputs(res, tmp / "out", cfg, progress=False)
    stats = json.loads(paths["stats"].read_text())
    assert stats["frames"] == T and len(stats["players"]) == 6
    lines = paths["tracks"].read_text().strip().splitlines()
    assert lines[0].startswith("frame,track_id") and len(lines) > T * 5
    cap = cv2.VideoCapture(str(paths["video"]))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == T
    assert cv2.imread(str(paths["heatmaps"])) is not None


def test_analysis_cache_roundtrip(synthetic, tmp_path):
    an, _ = synthetic
    pipeline.save_analysis(an, tmp_path / "a.pkl")
    back = pipeline.load_analysis(tmp_path / "a.pkl")
    assert len(back.frames) == T and np.array_equal(back.frames[5].xyxy, an.frames[5].xyxy)
