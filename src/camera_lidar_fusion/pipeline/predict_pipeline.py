import cv2
import numpy as np
from camera_lidar_fusion.components.data_ingestion import CalibrationData
from camera_lidar_fusion.components.detector import run_obstacle_detection
from camera_lidar_fusion.components.fusion import EarlyFusion


def run_pipeline(image: np.ndarray, point_cloud: np.ndarray, model, calib_file: str) -> np.ndarray:
    """
    Full inference pipeline:
      1. Load calibration
      2. Project LiDAR onto image plane
      3. Run YOLOv8 object detection
      4. Fuse detections with LiDAR depth
    Returns annotated output image.
    """
    calib  = CalibrationData(calib_file)
    fusion = EarlyFusion(calib)

    # Step 1: project LiDAR onto image
    fusion.project_lidar_to_image(point_cloud[:, :3], image.copy())

    # Step 2: run YOLO detection
    result, pred_bboxes = run_obstacle_detection(image.copy(), model)

    # Step 3: fuse
    return fusion.fuse(pred_bboxes, result)
