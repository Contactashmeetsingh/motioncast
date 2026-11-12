"""End-to-end pipeline: video -> detections/tracks -> teams -> pitch coords -> stats -> rendered video.

Three stages:
  1. analyze():     one pass of model inference over the video (players, ball, pitch keypoints),
                    ByteTrack IDs, and a sample of player crops for team classification.
  2. postprocess(): offline clean-up using the whole clip: drop short tracks, vote each
                    track's role and team, fill homography gaps, project to the pitch, smooth,
                    clean the ball trajectory, compute possession and per-player stats.
  3. render():      draw the results back onto the video with a minimap inset (no inference).
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from motioncast import analytics, ball, homography, possession, team
from motioncast.config import (
    BALL_CLASS_ID,
    GOALKEEPER_CLASS_ID,
    NO_TEAM,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    PipelineConfig,
    pick_device,
)
from motioncast.pitch import PITCH

ROLE_NAMES = {PLAYER_CLASS_ID: "player", GOALKEEPER_CLASS_ID: "goalkeeper", REFEREE_CLASS_ID: "referee"}


@dataclass
class FrameRecord:
    ids: np.ndarray            # (n,) ByteTrack ids
    xyxy: np.ndarray           # (n, 4) image boxes
    cls: np.ndarray            # (n,) detector class per box
    conf: np.ndarray           # (n,)
    ball_xy: np.ndarray | None  # (2,) image centre of the chosen ball detection
    h: np.ndarray | None       # 3x3 image->pitch(cm) homography


@dataclass
class Analysis:
    video_path: str
    fps: float
    width: int
    height: int
    start_frame: int
    frames: list[FrameRecord]
    crops: list[tuple[int, np.ndarray]]  # (track id, torso crop)
    timings: dict = field(default_factory=dict)


@dataclass
class Result:
    analysis: Analysis
    track_ids: list[int]                   # kept tracks, in display order
    display_id: dict[int, int]             # ByteTrack id -> 1..K
    role: dict[int, str]
    team: dict[int, int]
    pitch_xy_m: dict[int, np.ndarray]      # smoothed (T, 2), NaN when absent
    ball_img: np.ndarray                   # (T, 2) cleaned image positions
    ball_m: np.ndarray                     # (T, 2) pitch metres
    holders: list[int | None]
    poss_team: list[int | None]
    has_h: np.ndarray                      # (T,) bool, homography available
    stats: dict


# --------------------------------------------------------------------------- stage 1
def _foot_points(xyxy: np.ndarray) -> np.ndarray:
    return np.c_[(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]]


def analyze(video_path: str | Path, cfg: PipelineConfig, progress: bool = True) -> Analysis:
    import supervision as sv
    from tqdm import tqdm
    from ultralytics import YOLO

    video_path = str(video_path)
    device = pick_device(cfg.device)
    for name in ("player", "pitch") + (("ball",) if cfg.use_ball_model else ()):
        if not cfg.model_path(name).exists():
            raise FileNotFoundError(
                f"Missing {cfg.model_path(name)}. Run `python -m motioncast download` first.")

    player_model = YOLO(str(cfg.model_path("player")))
    pitch_model = YOLO(str(cfg.model_path("pitch")))
    ball_model = YOLO(str(cfg.model_path("ball"))) if cfg.use_ball_model else None

    info = sv.VideoInfo.from_video_path(video_path)
    fps = float(info.fps)
    end = info.total_frames
    if cfg.max_frames:
        end = min(end, cfg.start_frame + cfg.max_frames)
    n_frames = max(0, end - cfg.start_frame)

    tracker = sv.ByteTrack(
        frame_rate=int(round(fps)),
        lost_track_buffer=int(round(fps * cfg.track_buffer_s)),
        minimum_consecutive_frames=3,
    )

    frames: list[FrameRecord] = []
    crops: list[tuple[int, np.ndarray]] = []
    crop_counts: Counter = Counter()
    last_ball = None
    last_h = None
    t0 = time.time()
    gen = sv.get_video_frames_generator(video_path, start=cfg.start_frame, end=end)
    for i, frame in enumerate(tqdm(gen, total=n_frames, desc="analyze", disable=not progress)):
        res = player_model(frame, imgsz=cfg.player_imgsz, conf=cfg.player_conf,
                           device=device, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)
        people = det[det.class_id != BALL_CLASS_ID].with_nms(threshold=0.5, class_agnostic=True)
        people = tracker.update_with_detections(people)

        # Ball
        if ball_model is not None:
            bres = ball_model(frame, imgsz=cfg.ball_imgsz, conf=cfg.ball_conf,
                              device=device, verbose=False)[0]
            bdet = sv.Detections.from_ultralytics(bres)
        else:
            bdet = det[det.class_id == BALL_CLASS_ID]
        bxy = bdet.get_anchors_coordinates(sv.Position.CENTER) if len(bdet) else np.zeros((0, 2))
        bconf = bdet.confidence if len(bdet) else np.zeros(0)
        k = ball.pick_ball(bxy, bconf, last_ball)
        ball_xy = bxy[k].astype(np.float32) if k is not None else None
        if ball_xy is not None:
            last_ball = ball_xy

        # Pitch homography
        if i % cfg.pitch_every == 0:
            kres = pitch_model(frame, device=device, verbose=False)[0]
            h = None
            if kres.keypoints is not None and len(kres.keypoints.xy):
                kp = kres.keypoints
                xy = kp.xy[0].cpu().numpy()
                kconf = kp.conf[0].cpu().numpy() if kp.conf is not None else None
                h = homography.estimate_homography(
                    xy, kconf, min_conf=cfg.keypoint_conf, min_points=cfg.min_keypoints,
                    max_reprojection_error_cm=cfg.max_reprojection_error_cm)
            last_h = h
        h = last_h

        ids = people.tracker_id if people.tracker_id is not None else np.zeros(0, dtype=int)
        rec = FrameRecord(
            ids=np.asarray(ids, dtype=int),
            xyxy=people.xyxy.astype(np.float32),
            cls=np.asarray(people.class_id, dtype=int),
            conf=np.asarray(people.confidence, dtype=np.float32),
            ball_xy=ball_xy,
            h=h,
        )
        frames.append(rec)

        # Sample torso crops of outfield players for team classification.
        for tid, box, c in zip(rec.ids, rec.xyxy, rec.cls):
            if c != PLAYER_CLASS_ID or crop_counts[tid] >= cfg.team_max_crops_per_track:
                continue
            if crop_counts[tid] == 0 or i % cfg.team_crop_every == 0:
                x1, y1, x2, y2 = box.astype(int)
                crop = team.center_crop(frame[max(0, y1):y2, max(0, x1):x2])
                if crop.size and crop.shape[0] >= 8 and crop.shape[1] >= 4:
                    crops.append((int(tid), crop.copy()))
                    crop_counts[tid] += 1

    return Analysis(video_path, fps, info.width, info.height, cfg.start_frame, frames, crops,
                    timings={"analyze_s": round(time.time() - t0, 1), "device": device})


# --------------------------------------------------------------------------- stage 2
GOALKEEPER_MIN_SHARE = 0.3


def _vote_role(votes: Counter) -> str:
    """Majority role, except goalkeeper wins at >=30%: the detector often calls keepers 'player'."""
    if votes[GOALKEEPER_CLASS_ID] >= GOALKEEPER_MIN_SHARE * sum(votes.values()):
        return ROLE_NAMES[GOALKEEPER_CLASS_ID]
    return ROLE_NAMES[votes.most_common(1)[0][0]]


def postprocess(an: Analysis, cfg: PipelineConfig, embedder=None, progress: bool = True) -> Result:
    t0 = time.time()
    T = len(an.frames)
    fps = an.fps

    # Track membership and role votes.
    appear: dict[int, list[int]] = defaultdict(list)
    role_votes: dict[int, Counter] = defaultdict(Counter)
    for t, f in enumerate(an.frames):
        for tid, c in zip(f.ids, f.cls):
            appear[int(tid)].append(t)
            role_votes[int(tid)][int(c)] += 1
    kept = sorted((tid for tid, ts in appear.items() if len(ts) >= cfg.min_track_frames),
                  key=lambda tid: (appear[tid][0], tid))
    keep_set = set(kept)
    display_id = {tid: i + 1 for i, tid in enumerate(kept)}
    role = {tid: _vote_role(role_votes[tid]) for tid in kept}

    # Homographies: fill gaps from the nearest frame that had one.
    hs = homography.fill_homographies([f.h for f in an.frames], max_gap=int(fps * 2))
    has_h = np.array([h is not None for h in hs])

    # Raw pitch positions (metres) per track.
    raw = {tid: np.full((T, 2), np.nan) for tid in kept}
    for t, f in enumerate(an.frames):
        if hs[t] is None or len(f.ids) == 0:
            continue
        pts = homography.transform_points(hs[t], _foot_points(f.xyxy))
        on_pitch = PITCH.contains(pts)
        for tid, p, ok in zip(f.ids, pts, on_pitch):
            if int(tid) in keep_set and ok:
                raw[int(tid)][t] = p / 100.0
    pitch_xy = {tid: analytics.smooth_segments(raw[tid], cfg.smoothing_window) for tid in kept}

    # Teams: cluster crop embeddings, vote per track.
    team_of: dict[int, int] = {tid: NO_TEAM for tid in kept}
    player_tracks = [tid for tid in kept if role[tid] == "player"]
    crops = [(tid, c) for tid, c in an.crops if tid in set(player_tracks)]
    if crops:
        if embedder is None:
            embedder = team.CropEmbedder(device=an.timings.get("device", "cpu"))
        feats = embedder([c for _, c in crops])
        labels = team.cluster_two_teams(feats)
        team_of.update(team.vote_track_teams([tid for tid, _ in crops], labels))

    def mean_team_x() -> dict[int, float]:
        out = {}
        for k in (0, 1):
            xs = [np.nanmean(pitch_xy[tid][:, 0]) for tid in player_tracks
                  if team_of.get(tid) == k and not np.isnan(pitch_xy[tid][:, 0]).all()]
            if xs:
                out[k] = float(np.mean(xs))
        return out

    # Cluster labels are arbitrary; make team 0 the team positioned further left on the pitch
    # so ids are stable across runs.
    team_x = mean_team_x()
    if len(team_x) == 2 and team_x[1] < team_x[0]:
        team_of = {tid: (1 - k if k in (0, 1) else k) for tid, k in team_of.items()}
        team_x = {0: team_x[1], 1: team_x[0]}
    # Goalkeepers: assign to the team whose players sit nearer their end.
    gk_x = {tid: float(np.nanmean(pitch_xy[tid][:, 0])) for tid in kept
            if role[tid] == "goalkeeper" and not np.isnan(pitch_xy[tid][:, 0]).all()}
    team_of.update(team.assign_goalkeepers(gk_x, team_x))

    # Ball: clean in image space (for drawing), then project each frame.
    ball_raw = np.array([f.ball_xy if f.ball_xy is not None else (np.nan, np.nan) for f in an.frames],
                        dtype=float).reshape(T, 2)
    max_px_per_s = cfg.ball_max_speed_mps * an.width / 40.0  # ~40 m visible across a broadcast frame
    ball_img = ball.clean_trajectory(ball_raw, fps, max_px_per_s, cfg.ball_max_gap_frames)
    ball_m = np.full((T, 2), np.nan)
    for t in range(T):
        if hs[t] is not None and not np.isnan(ball_img[t]).any():
            p = homography.transform_points(hs[t], ball_img[t])[0]
            if PITCH.contains(p, margin_cm=500)[0]:
                ball_m[t] = p / 100.0

    # Possession: outfield players + goalkeepers with a team.
    field_tracks = [tid for tid in kept if role[tid] != "referee" and team_of[tid] != NO_TEAM]
    per_frame = []
    for t in range(T):
        ids = np.array([tid for tid in field_tracks if not np.isnan(pitch_xy[tid][t]).any()], dtype=int)
        per_frame.append((
            ids,
            np.array([team_of[tid] for tid in ids], dtype=int),
            np.array([pitch_xy[tid][t] for tid in ids]).reshape(-1, 2),
        ))
    holders, poss_team = possession.compute_possession(
        ball_m, per_frame, radius_m=cfg.possession_radius_m, hysteresis=cfg.possession_hysteresis)

    stats = _build_stats(an, cfg, kept, display_id, role, team_of, pitch_xy, ball_raw, ball_m,
                         holders, poss_team, has_h)
    an.timings["postprocess_s"] = round(time.time() - t0, 1)
    stats["timings"] = an.timings
    return Result(an, kept, display_id, role, team_of, pitch_xy, ball_img, ball_m,
                  holders, poss_team, has_h, stats)


def _build_stats(an, cfg, kept, display_id, role, team_of, pitch_xy, ball_raw, ball_m,
                 holders, poss_team, has_h) -> dict:
    T, fps = len(an.frames), an.fps
    poss = possession.possession_summary(poss_team, holders)
    held = Counter(h for h in holders if h is not None)
    players = []
    for tid in kept:
        k = analytics.kinematics(pitch_xy[tid], fps, cfg.max_player_speed_mps)
        players.append({
            "id": display_id[tid],
            "role": role[tid],
            "team": team_of[tid],
            **k,
            "touches_s": round(held.get(tid, 0) / fps, 2),
            "heatmap": analytics.heatmap(pitch_xy[tid]).round(4).tolist(),
        })
    teams = {}
    for k in (0, 1):
        members = [p for p in players if p["team"] == k]
        team_xy = np.concatenate([pitch_xy[tid] for tid in kept if team_of[tid] == k] or [np.zeros((0, 2))])
        teams[str(k)] = {
            "players": len(members),
            "distance_m": round(sum(p["distance_m"] for p in members), 1),
            "possession_pct": poss["team_pct"].get(str(k), 0.0),
            "heatmap": analytics.heatmap(team_xy).round(4).tolist(),
        }
    return {
        "video": Path(an.video_path).name,
        "fps": fps,
        "frames": T,
        "start_frame": an.start_frame,
        "duration_s": round(T / fps, 2),
        "resolution": [an.width, an.height],
        "pitch_m": [PITCH.length / 100, PITCH.width / 100],
        "coverage": {
            "homography_pct": round(100 * float(has_h.mean()), 1) if T else 0.0,
            "ball_detected_pct": round(100 * float((~np.isnan(ball_raw).any(axis=1)).mean()), 1) if T else 0.0,
            "ball_on_pitch_pct": round(100 * float((~np.isnan(ball_m).any(axis=1)).mean()), 1) if T else 0.0,
        },
        "possession": {"team_pct": poss["team_pct"],
                       "by_player_s": {str(display_id[int(t)]): round(n / fps, 2)
                                       for t, n in held.most_common()}},
        "teams": teams,
        "players": players,
        "notes": [
            "Teams are unsupervised clusters (0/1); which real team each is cannot be known from video alone.",
            "Track IDs come from ByteTrack and can switch when players occlude each other.",
            "Ball pitch position is projected from the image, so it is inaccurate while the ball is in the air.",
        ],
    }


# --------------------------------------------------------------------------- outputs
def write_outputs(res: Result, out_dir: str | Path, cfg: PipelineConfig, progress: bool = True) -> dict:
    from motioncast import render

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"stats": out / "stats.json", "tracks": out / "tracks.csv", "heatmaps": out / "heatmaps.png"}

    # tracks.csv: one row per (frame, track).
    rows = ["frame,track_id,role,team,x1,y1,x2,y2,conf,pitch_x_m,pitch_y_m,holder"]
    for t, f in enumerate(res.analysis.frames):
        for tid, box, c in zip(f.ids, f.xyxy, f.conf):
            tid = int(tid)
            if tid not in res.display_id:
                continue
            px, py = res.pitch_xy_m[tid][t]
            rows.append(
                f"{res.analysis.start_frame + t},{res.display_id[tid]},{res.role[tid]},{res.team[tid]},"
                f"{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f},{c:.3f},"
                f"{'' if np.isnan(px) else f'{px:.2f}'},{'' if np.isnan(py) else f'{py:.2f}'},"
                f"{int(res.holders[t] == tid)}")
    paths["tracks"].write_text("\n".join(rows) + "\n")

    render.save_heatmaps(res, paths["heatmaps"])
    if cfg.render:
        t0 = time.time()
        paths["video"] = out / "annotated.mp4"
        render.render_video(res, paths["video"], cfg, progress=progress)
        res.stats["timings"]["render_s"] = round(time.time() - t0, 1)
    res.stats["outputs"] = {k: v.name for k, v in paths.items()}
    paths["stats"].write_text(json.dumps(res.stats, indent=2))
    return paths


def save_analysis(an: Analysis, path: Path) -> None:
    import pickle

    with open(path, "wb") as f:
        pickle.dump(an, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_analysis(path: Path) -> Analysis:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def run(video_path: str | Path, out_dir: str | Path, cfg: PipelineConfig | None = None,
        progress: bool = True, reuse_analysis: bool = False) -> dict:
    """Run all stages. With reuse_analysis, skip inference if out_dir/analysis.pkl matches."""
    cfg = cfg or PipelineConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "analysis.pkl"
    an = None
    if reuse_analysis and cache.exists():
        an = load_analysis(cache)
        if Path(an.video_path).name != Path(video_path).name or an.start_frame != cfg.start_frame:
            an = None
        else:
            an.timings = {"analyze_s": 0.0, "analysis_reused": True, "device": an.timings.get("device", "cpu")}
    if an is None:
        an = analyze(video_path, cfg, progress=progress)
        save_analysis(an, cache)
    res = postprocess(an, cfg, progress=progress)
    return write_outputs(res, out_dir, cfg, progress=progress)
