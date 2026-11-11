# CLAUDE.md

MotionCast: soccer broadcast video → tracking, teams, pitch positions, stats. See README.md.

- Setup: `source .venv/bin/activate` (or `pip install -e ".[dev]"`), then `python -m motioncast download`.
- Run: `python -m motioncast run <video> [--max-frames N] [--reuse]`. `--reuse` skips inference
  using `outputs/<name>/analysis.pkl`, which is the fast loop when changing post-processing or rendering.
- Test: `pytest` (no weights or GPU needed). Lint: `ruff check motioncast tests`.
- Stage boundaries: `analyze()` is the only stage that runs models. Everything after it is pure
  numpy over `Analysis` and must stay testable with the synthetic clip in
  `tests/test_pipeline_synthetic.py`.
- Units: pitch geometry is in centimetres (`pitch.py`). Analytics, stats and CSV are in metres.
- Pin `supervision<0.27` (0.27+ removed `sv.ByteTrack`).
- Never commit weights, videos or `outputs/` (all gitignored). Never invent metrics. Report what
  a real run printed.
