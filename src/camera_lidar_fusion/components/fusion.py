import cv2
import numpy as np
import matplotlib.pyplot as plt

from camera_lidar_fusion.utils.calibration_utils import get_lidar_in_image_fov
from camera_lidar_fusion.utils.visualization import draw_lidar_on_image
from camera_lidar_fusion.utils.geometry_utils import rect_contains
from camera_lidar_fusion.utils.distance_utils import filter_outliers, get_best_distance


class EarlyFusion:
    """Fuses LiDAR point clouds with YOLOv8 2D bounding box detections."""

    def __init__(self, calib_data):
        self.calib = calib_data
        self.imgfov_pts_2d  = None
        self.imgfov_pc_velo = None

    def project_lidar_to_image(self, pc_velo: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Project LiDAR points into camera image plane and overlay on image."""
        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
            pc_velo,
            self.calib.P, self.calib.V2C, self.calib.R0,
            xmin=0, ymin=0, xmax=img.shape[1], ymax=img.shape[0]
        )
        self.imgfov_pts_2d  = pts_2d[fov_inds, :]
        self.imgfov_pc_velo = imgfov_pc_velo
        return draw_lidar_on_image(img.copy(), self.imgfov_pts_2d, self.imgfov_pc_velo)

    def fuse(self, pred_bboxes: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Match LiDAR depth points inside each YOLO bounding box and annotate distance."""
        img_out = image.copy()
        cmap = plt.colormaps["hsv"].resampled(256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        for box in pred_bboxes:
            distances = []
            for i in range(self.imgfov_pts_2d.shape[0]):
                depth = self.imgfov_pc_velo[i, 0]
                color = cmap[int(510.0 / depth), :]
                cv2.circle(
                    img_out,
                    (int(np.round(self.imgfov_pts_2d[i, 0])),
                     int(np.round(self.imgfov_pts_2d[i, 1]))),
                    2, color=tuple(color), thickness=-1
                )
                if rect_contains(box, self.imgfov_pts_2d[i], shrink_factor=0.1):
                    distances.append(depth)

            if len(distances) > 2:
                distances = filter_outliers(distances)
                best_distance = get_best_distance(distances, technique="average")
                cx = int(box[0] + (box[2] - box[0]) / 2)
                cy = int(box[1] + (box[3] - box[1]) / 2)
                cv2.putText(
                    img_out, f"{round(best_distance, 2)}m",
                    (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 3, cv2.LINE_AA
                )

        return img_out
