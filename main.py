import argparse
import glob
import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from camera_lidar_fusion.components.data_ingestion import CalibrationData
from camera_lidar_fusion.pipeline.predict_pipeline import run_pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description="Camera-LiDAR Early Fusion Inference")
    parser.add_argument("--model",      type=str, default="yolov8s.pt",       help="Path to YOLO model weights")
    parser.add_argument("--img_path",   type=str, default="data/img",          help="Path to image directory")
    parser.add_argument("--pcd_path",   type=str, default="data/velodyne",     help="Path to point cloud directory")
    parser.add_argument("--calib_path", type=str, default="data/calib",        help="Path to calibration directory")
    parser.add_argument("--index",      type=int, default=0,                   help="File index to process")
    parser.add_argument("--output",     type=str, default="output/result.jpg", help="Output image path")
    return parser.parse_args()


def main():
    args = parse_arguments()

    image_files = sorted(glob.glob(f"{args.img_path}/*.png"))
    point_files  = sorted(glob.glob(f"{args.pcd_path}/*.pcd"))
    calib_files  = sorted(glob.glob(f"{args.calib_path}/*.txt"))

    if args.index >= min(len(image_files), len(point_files), len(calib_files)):
        raise IndexError(f"Index {args.index} is out of range. Available files: {len(image_files)}")

    model  = YOLO(args.model)
    cloud  = o3d.io.read_point_cloud(point_files[args.index])
    points = np.asarray(cloud.points)
    image  = cv2.cvtColor(cv2.imread(image_files[args.index]), cv2.COLOR_BGR2RGB)

    result = run_pipeline(image, points, model, calib_files[args.index])
    cv2.imwrite(args.output, result)
    print(f"[INFO] Result saved to {args.output}")


if __name__ == "__main__":
    main()
