import pytest
import numpy as np


@pytest.fixture
def dummy_point_cloud():
    """Synthetic point cloud: 100 random points beyond 2m clip distance."""
    return np.random.uniform(2.5, 50.0, size=(100, 3)).astype(np.float32)


@pytest.fixture
def dummy_image():
    """Blank 480x640 RGB image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)
