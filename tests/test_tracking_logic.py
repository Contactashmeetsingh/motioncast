import numpy as np

from motioncast import analytics, ball, possession, team

NAN = np.nan


def test_pick_ball_prefers_continuity_then_confidence():
    xy = np.array([[100.0, 100.0], [900.0, 500.0]])
    conf = np.array([0.3, 0.9])
    assert ball.pick_ball(xy, conf, last_xy=np.array([110.0, 95.0])) == 0
    assert ball.pick_ball(xy, conf, last_xy=None) == 1
    assert ball.pick_ball(np.zeros((0, 2)), np.zeros(0), None) is None


def test_clean_trajectory_drops_outlier_and_fills_short_gaps():
    xy = np.array([[0, 0], [1, 0], [NAN, NAN], [3, 0], [500, 500], [5, 0],
                   [NAN, NAN], [NAN, NAN], [NAN, NAN], [NAN, NAN], [10, 0]], dtype=float)
    out = ball.clean_trajectory(xy, fps=1.0, max_speed=10.0, max_gap=2)
    np.testing.assert_allclose(out[2], [2, 0])     # 1-frame gap interpolated
    np.testing.assert_allclose(out[4], [4, 0])     # outlier removed, then interpolated
    assert np.isnan(out[6:10]).all()               # 4-frame gap left alone


def test_possession_hysteresis_and_loose_ball():
    ids = np.array([7, 9])
    teams = np.array([0, 1])
    frames, ball_xy = [], []
    # Ball at player 7 for 4 frames, then at player 9 for 1 frame (flicker), back to 7,
    # then loose, then at player 9 for 4 frames.
    pos7, pos9 = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    seq = [pos7] * 4 + [pos9] + [pos7] * 2 + [np.array([5.0, 0.0])] * 2 + [pos9] * 4
    for b in seq:
        frames.append((ids, teams, np.stack([pos7, pos9])))
        ball_xy.append(b)
    holders, poss = possession.compute_possession(np.array(ball_xy), frames, radius_m=2.0, hysteresis=3)
    assert holders[:4] == [7] * 4
    assert holders[4] is None and poss[4] == 0      # one-frame flicker does not switch
    assert holders[7] is None and poss[7] == 0      # loose ball keeps last team
    assert poss[-1] == 1 and holders[-1] == 9       # switches after 3 consecutive frames
    summary = possession.possession_summary(poss, holders)
    assert sum(summary["team_pct"].values()) == 100.0


def test_smoothing_respects_gaps():
    xy = np.array([[0, 0], [2, 0], [4, 0], [NAN, NAN], [10, 0], [10, 0]], dtype=float)
    out = analytics.smooth_segments(xy, window=3)
    assert np.isnan(out[3]).all()
    np.testing.assert_allclose(out[1], [2, 0])
    np.testing.assert_allclose(out[4], [10, 0])


def test_kinematics_constant_speed_and_glitch_rejection():
    fps = 25.0
    t = np.arange(100)
    xy = np.c_[t * 0.2, np.zeros(100)]  # 0.2 m/frame = 5 m/s
    xy[50] = [80, 30]                   # teleport glitch
    k = analytics.kinematics(xy, fps, max_speed_mps=11)
    assert abs(k["top_speed_mps"] - 5.0) < 0.05
    assert abs(k["distance_m"] - 0.2 * 97) < 0.5  # 99 steps minus the 2 glitch steps
    assert k["seconds_tracked"] == 4.0


def test_heatmap_normalised():
    hm = analytics.heatmap(np.array([[10, 10], [10, 10], [110, 60], [NAN, NAN]]))
    assert hm.shape == (14, 24)
    assert abs(hm.sum() - 1) < 1e-9


def test_team_vote_and_goalkeeper_assignment():
    votes = team.vote_track_teams([1, 1, 1, 2, 2], np.array([0, 0, 1, 1, 1]))
    assert votes == {1: 0, 2: 1}
    gk = team.assign_goalkeepers({50: 5.0, 51: 115.0}, {0: 40.0, 1: 80.0})
    assert gk == {50: 0, 51: 1}


def test_cluster_two_teams_separates_obvious_groups():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, size=(20, 16))
    b = rng.normal(5, 0.1, size=(20, 16))
    labels = team.cluster_two_teams(np.vstack([a, b]))
    assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1
    assert labels[0] != labels[-1]


def test_goalkeeper_role_wins_with_minority_share():
    from collections import Counter

    from motioncast.pipeline import _vote_role

    assert _vote_role(Counter({1: 103, 2: 115})) == "goalkeeper"
    assert _vote_role(Counter({1: 10, 2: 90})) == "player"
    assert _vote_role(Counter({3: 60, 2: 40})) == "referee"
