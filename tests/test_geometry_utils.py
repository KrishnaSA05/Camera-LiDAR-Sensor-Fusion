from camera_lidar_fusion.utils.geometry_utils import rect_contains


def test_point_inside_rect():
    assert rect_contains([0, 0, 100, 100], [50, 50]) is True


def test_point_outside_rect():
    assert rect_contains([0, 0, 100, 100], [150, 50]) is False


def test_point_on_edge_excluded():
    assert rect_contains([0, 0, 100, 100], [0, 50]) is False


def test_shrink_excludes_border():
    # 10% shrink → effective box [10,10,90,90]; point [5,50] is outside
    assert rect_contains([0, 0, 100, 100], [5, 50], shrink_factor=0.1) is False


def test_shrink_includes_center():
    assert rect_contains([0, 0, 100, 100], [50, 50], shrink_factor=0.1) is True
