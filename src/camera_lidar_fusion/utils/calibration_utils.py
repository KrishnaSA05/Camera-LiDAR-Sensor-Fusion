import numpy as np


def read_calib_file(filepath: str) -> dict:
    """Read KITTI calibration file and return a dictionary of numpy arrays."""
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(":", 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def cart2hom(pts_3d: np.ndarray) -> np.ndarray:
    """Convert Nx3 Cartesian coordinates to Nx4 Homogeneous coordinates."""
    n = pts_3d.shape[0]
    return np.hstack((pts_3d, np.ones((n, 1))))


def project_velo_to_image(
    pts_3d_velo: np.ndarray,
    P: np.ndarray,
    V2C: np.ndarray,
    R0: np.ndarray
) -> np.ndarray:
    """
    Project LiDAR points (Nx3) to image plane (Nx2).
    Uses the standard KITTI formula: Y_2D = P x R0 x V2C x X_3D
    """
    R0_homo   = np.vstack([R0, [0, 0, 0]])
    R0_homo_2 = np.column_stack([R0_homo, [0, 0, 0, 1]])
    p_r0      = np.dot(P, R0_homo_2)
    p_r0_rt   = np.dot(p_r0, np.vstack((V2C, [0, 0, 0, 1])))

    pts_3d_homo = np.column_stack([pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))])
    pts_2d      = np.transpose(np.dot(p_r0_rt, np.transpose(pts_3d_homo)))

    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def get_lidar_in_image_fov(
    pc_velo: np.ndarray,
    P: np.ndarray,
    V2C: np.ndarray,
    R0: np.ndarray,
    xmin: float, ymin: float,
    xmax: float, ymax: float,
    clip_distance: float = 2.0
):
    """
    Filter LiDAR points to only those visible in the camera Field of View.
    Returns (filtered_points, all_2d_projections, fov_boolean_mask).
    """
    pts_2d  = project_velo_to_image(pc_velo, P, V2C, R0)
    fov_inds = (
        (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) &
        (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    ) & (pc_velo[:, 0] > clip_distance)
    return pc_velo[fov_inds, :], pts_2d, fov_inds
