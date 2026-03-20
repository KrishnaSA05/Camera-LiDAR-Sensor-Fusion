import numpy as np
from camera_lidar_fusion.utils.calibration_utils import read_calib_file


class CalibrationData:
    """Loads and stores KITTI-format calibration matrices."""

    def __init__(self, calib_file: str):
        calibs = read_calib_file(calib_file)
        self.P   = np.reshape(calibs["P2"],             [3, 4])
        self.V2C = np.reshape(calibs["Tr_velo_to_cam"], [3, 4])
        self.R0  = np.reshape(calibs["R0_rect"],        [3, 3])
