import cv2
import glob
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from camera_lidar_fusion.pipeline.predict_pipeline import run_pipeline


def run_on_video(
    img_dir: str = "data/img",
    pcd_dir: str = "data/velodyne",
    calib_dir: str = "data/calib",
    model_path: str = "yolov8s.pt",
    output_path: str = "output/video_result.avi"
):
    model = YOLO(model_path)
    image_files = sorted(glob.glob(f"{img_dir}/*.png"))
    point_files  = sorted(glob.glob(f"{pcd_dir}/*.pcd"))
    calib_files  = sorted(glob.glob(f"{calib_dir}/*.txt"))

    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    first_img = cv2.imread(image_files[0])
    h, w = first_img.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), 10, (w, h))

    total = min(len(image_files), len(point_files), len(calib_files))
    for i in range(total):
        image  = cv2.cvtColor(cv2.imread(image_files[i]), cv2.COLOR_BGR2RGB)
        cloud  = o3d.io.read_point_cloud(point_files[i])
        points = np.asarray(cloud.points)
        result = run_pipeline(image, points, model, calib_files[i])
        writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Processed frame {i + 1}/{total}")

    writer.release()
    print(f"[INFO] Video saved to {output_path}")


if __name__ == "__main__":
    run_on_video()
