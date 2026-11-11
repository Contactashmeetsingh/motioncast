"""Ball possession from per-frame ball and player pitch positions (metres)."""

from __future__ import annotations

from collections import Counter

import numpy as np


def compute_possession(
    ball_xy: np.ndarray,
    players: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    radius_m: float = 2.0,
    hysteresis: int = 3,
) -> tuple[list[int | None], list[int | None]]:
    """Per-frame ball holder and possessing team.

    Args:
        ball_xy: (T, 2) ball positions in metres, NaN where unknown.
        players: per frame, (track_ids, team_ids, xy_m) for outfield players + goalkeepers.
        radius_m: a player controls the ball if nearest to it and within this distance.
        hysteresis: a new holder must be nearest for this many consecutive frames
            before possession switches (suppresses flicker when players cross).

    Returns:
        (holder_track_id per frame, possessing team per frame). The team stays with the
        last holder while the ball is loose or in flight, so team possession is continuous.
    """
    holders: list[int | None] = []
    teams: list[int | None] = []
    holder = holder_team = None
    candidate, streak = None, 0
    for t, (ids, team_ids, xy) in enumerate(players):
        nearest = nearest_team = None
        b = ball_xy[t]
        if len(ids) and not np.isnan(b).any():
            d = np.linalg.norm(xy - b, axis=1)
            d = np.where(np.isnan(d), np.inf, d)
            i = int(np.argmin(d))
            if d[i] <= radius_m:
                nearest, nearest_team = int(ids[i]), int(team_ids[i])

        if nearest is None:
            candidate, streak = None, 0
            holders.append(None)
        else:
            if nearest == holder:
                candidate, streak = None, 0
            else:
                streak = streak + 1 if nearest == candidate else 1
                candidate = nearest
                if holder is None or streak >= hysteresis:
                    holder, holder_team = nearest, nearest_team
                    candidate, streak = None, 0
            holders.append(holder if nearest == holder else None)
        teams.append(holder_team)
    return holders, teams


def possession_summary(teams: list[int | None], holders: list[int | None]) -> dict:
    team_counts = Counter(t for t in teams if t is not None)
    total = sum(team_counts.values())
    return {
        "team_pct": {str(k): round(100 * v / total, 1) for k, v in sorted(team_counts.items())} if total else {},
        "frames_with_possession": total,
        "player_frames": {str(k): v for k, v in Counter(h for h in holders if h is not None).most_common()},
    }
