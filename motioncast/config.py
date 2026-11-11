"""Pipeline configuration: paths, model class ids, and tunable thresholds."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MOTIONCAST_DATA", REPO_ROOT / "data"))
MODELS_DIR = DATA_DIR / "models"
VIDEOS_DIR = DATA_DIR / "videos"

# Class ids of Roboflow's football-player-detection model.
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Team ids used everywhere downstream. Referees are not on a team.
TEAM_A, TEAM_B, NO_TEAM = 0, 1, -1


def pick_device(requested: str = "auto") -> str:
    """Resolve 'auto' to the best available torch device."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PipelineConfig:
    device: str = "auto"
    models_dir: Path = MODELS_DIR

    # Detection
    player_imgsz: int = 1280
    player_conf: float = 0.3
    ball_imgsz: int = 1280
    ball_conf: float = 0.2
    use_ball_model: bool = True  # False: use the player model's ball class (faster, weaker)

    # Pitch keypoints / homography
    keypoint_conf: float = 0.5
    min_keypoints: int = 4
    pitch_every: int = 1  # run the keypoint model every N frames; reuse H in between
    max_reprojection_error_cm: float = 150.0

    # Tracking
    track_buffer_s: float = 1.5
    min_track_frames: int = 10  # drop tracks shorter than this (mostly false positives)

    # Team classification
    team_crop_every: int = 5  # sample a crop for each track every N frames
    team_max_crops_per_track: int = 12

    # Ball post-processing
    ball_max_gap_frames: int = 12  # interpolate missing ball positions across gaps this long
    ball_max_speed_mps: float = 40.0  # jumps faster than this are treated as false detections

    # Analytics
    smoothing_window: int = 9  # frames, odd
    max_player_speed_mps: float = 11.0
    possession_radius_m: float = 2.0
    possession_hysteresis: int = 3

    # Limits
    max_frames: int | None = None
    start_frame: int = 0

    # Rendering
    render: bool = True
    radar_scale: float = 0.36  # fraction of frame width used for the minimap inset

    extra: dict = field(default_factory=dict)

    def model_path(self, name: str) -> Path:
        return self.models_dir / f"football-{name}-detection.pt"
