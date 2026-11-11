import numpy as np

from motioncast import homography
from motioncast.pitch import PITCH, PitchCanvas

# A plausible broadcast-style homography: pitch(cm) -> image(px).
PITCH_TO_IMG = np.array([[0.12, 0.03, 200.0], [0.0, 0.05, 300.0], [0.0, 0.00002, 1.0]])


def project(h, pts):
    return homography.transform_points(h, pts)


def test_vertices_shape_and_bounds():
    v = PITCH.vertices
    assert v.shape == (32, 2)
    assert PITCH.contains(v, margin_cm=0).all()


def test_estimate_homography_recovers_pitch_coords():
    img_pts = project(PITCH_TO_IMG, PITCH.vertices)
    conf = np.ones(32)
    conf[[0, 5, 24, 29]] = 0.1  # low-confidence corners get ignored
    h = homography.estimate_homography(img_pts, conf)
    assert h is not None
    back = project(h, img_pts)
    assert np.abs(back - PITCH.vertices).max() < 1.0


def test_estimate_homography_rejects_too_few_or_collinear_points():
    img_pts = project(PITCH_TO_IMG, PITCH.vertices)
    conf = np.zeros(32)
    conf[[13, 14, 15]] = 1.0
    assert homography.estimate_homography(img_pts, conf) is None
    conf[16] = 1.0  # 4 points, but all on the halfway line
    assert homography.estimate_homography(img_pts, conf) is None


def test_estimate_homography_rejects_inconsistent_keypoints():
    rng = np.random.default_rng(0)
    img_pts = rng.uniform(100, 1800, size=(32, 2)).astype(np.float32)
    assert homography.estimate_homography(img_pts, np.ones(32), max_reprojection_error_cm=50) is None


def test_fill_homographies_uses_nearest_within_gap():
    a, b = np.eye(3), 2 * np.eye(3)
    filled = homography.fill_homographies([None, a, None, None, b, None, None, None], max_gap=2)
    assert filled[0] is a and filled[2] is a and filled[3] is b and filled[6] is b
    assert filled[7] is None  # 3 frames from b, beyond max_gap


def test_canvas_maps_corners_inside_image():
    c = PitchCanvas()
    h, w = c.base.shape[:2]
    x, y = c.to_px((PITCH.length, PITCH.width))
    assert 0 < x < w and 0 < y < h
