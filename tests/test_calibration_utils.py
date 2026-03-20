import numpy as np
from camera_lidar_fusion.utils.calibration_utils import cart2hom


def test_cart2hom_shape():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = cart2hom(pts)
    assert result.shape == (2, 4)


def test_cart2hom_ones_appended():
    pts = np.array([[1.0, 2.0, 3.0]])
    result = cart2hom(pts)
    assert result[0, 3] == 1.0


def test_cart2hom_original_values_preserved():
    pts = np.array([[7.0, 8.0, 9.0]])
    result = cart2hom(pts)
    np.testing.assert_array_equal(result[0, :3], [7.0, 8.0, 9.0])
