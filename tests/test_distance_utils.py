import pytest
from camera_lidar_fusion.utils.distance_utils import filter_outliers, get_best_distance


def test_filter_outliers_removes_extreme():
    distances = [10.0, 10.2, 9.8, 50.0, 10.1]
    result = filter_outliers(distances)
    assert 50.0 not in result


def test_filter_outliers_single_item():
    assert filter_outliers([5.0]) == [5.0]


def test_get_best_distance_closest():
    assert get_best_distance([5.0, 3.0, 8.0], technique="closest") == 3.0


def test_get_best_distance_average():
    assert get_best_distance([4.0, 6.0], technique="average") == 5.0


def test_get_best_distance_median():
    assert get_best_distance([1.0, 3.0, 5.0], technique="median") == 3.0


def test_get_best_distance_empty_raises():
    with pytest.raises(ValueError):
        get_best_distance([])


def test_get_best_distance_invalid_technique():
    with pytest.raises(ValueError):
        get_best_distance([1.0, 2.0], technique="unknown")
