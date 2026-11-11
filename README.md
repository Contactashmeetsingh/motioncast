# MotionCast

Soccer broadcast video in, match analysis out. MotionCast detects and tracks every player,
goalkeeper, referee and the ball, splits players into teams without labels, maps everyone onto
a 2D pitch, and computes possession, distance covered, speed and heatmaps.

```
video ─► detect (players/GK/refs, ball, 32 pitch keypoints)
      ─► ByteTrack IDs
      ─► homography per frame (keypoints → pitch, RANSAC + reprojection check, gap fill)
      ─► team split (SigLIP crop embeddings → UMAP → KMeans, majority vote per track)
      ─► pitch positions in metres, smoothed; ball trajectory cleaned + interpolated
      ─► possession (nearest player within 2 m, with hysteresis), distance, speed, heatmaps
      ─► annotated video + minimap, stats.json, tracks.csv, heatmaps.png
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m motioncast download                  # 3 pretrained models (~410 MB) + a 30 s sample clip
python -m motioncast run data/videos/0bfacc_0.mp4
```

Outputs land in `outputs/<video name>/`:

| File | What |
|---|---|
| `annotated.mp4` | Players ringed in team colour with IDs, ball marker, possession bar, live minimap (H.264) |
| `stats.json` | Per-player distance / top speed / time on ball / heatmap; per-team possession and distance; coverage metrics |
| `tracks.csv` | One row per frame per track: box, role, team, pitch position in metres, ball holder flag |
| `heatmaps.png` | Occupancy heatmap for each team |
| `analysis.pkl` | Cached model outputs. `--reuse` re-runs post-processing and rendering without inference |

Useful flags: `--max-frames 250` (quick test), `--start N`, `--device {auto,mps,cuda,cpu}`,
`--pitch-every 3` (run the keypoint model every 3rd frame), `--fast-ball` (skip the dedicated
ball model), `--no-render`, `--reuse`.

## Layout

```
motioncast/
  cli.py          download + run commands
  config.py       all thresholds in one dataclass
  pipeline.py     analyze() → postprocess() → write_outputs()
  homography.py   keypoints → image-to-pitch homography, validation, gap filling
  pitch.py        pitch geometry (32 keypoints, cm) + minimap canvas
  team.py         SigLIP embeddings, clustering, per-track vote, goalkeeper assignment
  ball.py         per-frame ball choice, outlier removal, gap interpolation
  possession.py   holder/team per frame with hysteresis
  analytics.py    smoothing, distance, speed, heatmaps
  render.py       annotated video, minimap, heatmap image
tests/            unit tests + a full postprocess/render run on a synthetic clip (no weights needed)
```

Run tests with `pytest` and lint with `ruff check motioncast tests`.

## Models

All three detectors are pretrained checkpoints from Roboflow's
[`sports`](https://github.com/roboflow/sports) project (MIT), downloaded at runtime and never
committed: `football-player-detection.pt` (ball / goalkeeper / player / referee),
`football-ball-detection.pt`, and `football-pitch-detection.pt` (32 pitch keypoints). Team
embeddings use `google/siglip-base-patch16-224`. Pitch geometry is adapted from the same project.

## Limitations

- Built for broadcast-style soccer footage similar to the models' training data. Other sports
  or camera setups need new detectors and pitch geometry.
- Teams are unlabeled clusters (0 / 1). The video alone can't tell you which real club each one is.
- ByteTrack IDs can swap when players overlap. There is no re-identification across long occlusions.
- The ball's pitch position comes from projecting its image position onto the ground, so it is
  wrong while the ball is in the air. This affects possession during long passes.
- Distance and speed depend on homography quality. Frames where the keypoint fit fails reuse
  the nearest good homography (up to 2 s away).

## Credits

Started as a Cal Hacks 2026 team project with Vincent Mendy, Dawood Khan and Shaaz Saleem
([original repo](https://github.com/wheredawoodat949/AI-Hackathon)). This repo is a rebuild of that
work into a single end-to-end pipeline. The full commit history is preserved.
