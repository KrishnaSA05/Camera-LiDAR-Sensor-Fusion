# 🚗 Camera-LiDAR Early Fusion for 3D Object Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-yellow)
![Dataset](https://img.shields.io/badge/Dataset-KITTI-green)
![License](https://img.shields.io/badge/License-MIT-red)
![Tests](https://img.shields.io/badge/Tests-15%20passed-brightgreen)

A production-ready implementation of **early fusion** between Camera and LiDAR sensors
for 3D object detection and real-world distance estimation. LiDAR point clouds are
projected onto the camera image plane using KITTI calibration matrices, and fused with
YOLOv8 2D bounding box detections to estimate real-world distances of detected objects.

![Fusion Demo](<div align="Left">
    <img src="output/test-ezgif.com-video-to-gif-converter.gif" width="1000" height="400">
</div>)

---

## 🎯 Key Features

- **Early Sensor Fusion** — Projects LiDAR point clouds onto the camera image plane using KITTI calibration matrices (P2, R0, Velo2Cam)
- **Real-time Distance Estimation** — Estimates depth of each detected object using LiDAR points inside YOLO bounding boxes
- **Outlier Filtering** — 1-sigma statistical filtering to remove noisy depth measurements
- **YOLOv8 Integration** — Detects cyclists, cars, and trucks using pretrained YOLOv8s (~105–231ms inference per frame)
- **Batch Video Pipeline** — Process full sequences of KITTI frames into an annotated output video
- **Modular Codebase** — Clean separation of components, pipeline, and utilities for production readiness

---

## Camera detections & point cloud visualization

<div align="Left">
    <img src="output/object detection and lidar projection 3D points.png" width="1000">
</div>

---

## 🏗️ Project Structure

```
camera_lidar_sensor_fusion/
├── main.py                              # CLI entry point for single-frame inference
├── setup.py                             # Package installer
├── requirements.txt                     # All dependencies
├── scripts/
│   └── run_video.py                     # Batch inference over a sequence of frames
├── notebooks/
│   └── Visual_Fusion_Early_Fusion.ipynb # Step-by-step research walkthrough
├── src/
│   └── camera_lidar_fusion/
│       ├── components/
│       │   ├── data_ingestion.py        # KITTI calibration matrix loader
│       │   ├── detector.py              # YOLOv8 inference wrapper
│       │   └── fusion.py               # Core early fusion logic
│       ├── pipeline/
│       │   └── predict_pipeline.py      # End-to-end orchestration
│       └── utils/
│           ├── calibration_utils.py     # LiDAR → Camera projection math
│           ├── visualization.py         # LiDAR point & bounding box rendering
│           ├── geometry_utils.py        # Point-in-box geometry
│           └── distance_utils.py        # Outlier filtering & distance aggregation
├── tests/
│   ├── conftest.py
│   ├── test_calibration_utils.py
│   ├── test_geometry_utils.py
│   └── test_distance_utils.py
└── data/
    ├── img/        # KITTI camera images (.png)
    ├── velodyne/   # LiDAR point clouds (.pcd)
    ├── calib/      # Calibration files (.txt)
    └── label/      # Ground truth labels (.txt)
```

---

## 🧠 How It Works

```
LiDAR Point Cloud (.pcd)
        │
        ▼
 Calibration (P2 × R0 × Velo2Cam)
        │
        ▼
 Project 3D → 2D Image Plane
        │
        ▼
Camera Image ──► YOLOv8 Detection ──► 2D Bounding Boxes
        │                                     │
        └──────────────┬───────────────────────┘
                       ▼
           Match LiDAR points inside boxes
                       │
                       ▼
           Filter outliers (1-sigma)
                       │
                       ▼
         Estimate distance (average depth)
                       │
                       ▼
         Annotated Output Image 🎯
```

### Projection Formula

```
Y_2D = P2 × R0_rect × Tr_velo_to_cam × X_3D
```

Where:
- `P2` — Camera projection matrix (3×4)
- `R0_rect` — Rectification rotation matrix (3×3)
- `Tr_velo_to_cam` — LiDAR to camera rigid transform (3×4)

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/KrishnaSA05/Camera-LiDAR-Sensor-Fusion.git
cd camera_lidar_sensor_fusion

# Create and activate conda environment (Python 3.11 required for open3d)
conda create -n fusion_env python=3.11 -y
conda activate fusion_env

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

---

## 🗂️ Dataset Setup (KITTI)

Download the [KITTI Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php) and organize as follows:

```
data/
├── img/          # Left color images  (image_2)
├── velodyne/     # Point clouds       (.pcd format)
├── calib/        # Calibration files  (calib)
└── label/        # Labels             (label_2)
```

> **Note:** If your LiDAR files are in `.bin` format (KITTI raw), convert them to `.pcd` first — see the notebook for a conversion script.

---

## 🚀 Usage

### Single Frame Inference

```bash
python main.py --model yolov8s.pt --img_path data/img --pcd_path data/velodyne --calib_path data/calib --index 0 --output output/result.jpg
```

### Batch Video Inference

```bash
python scripts/run_video.py
```

### Run Unit Tests

```bash
pytest tests/ -v
```

---

## 📦 Requirements

```
ultralytics
opencv-python
open3d
numpy
matplotlib
Pillow
pytest
PyYAML
```

> Requires **Python 3.11** — open3d does not support Python 3.12+

---

## 📊 Results

| Frame | Detections | Inference Speed |
|---|---|---|
| Frame 0 | 1 truck (77% conf) | 231ms |
| Frame 1 | 3 cars | 231ms |
| Frame 2 | 1 car | 171ms |
| Frame 3 | 3 cars | 179ms |
| Frame 4 | 2 cars | 105ms |

**Distance estimation accuracy:** LiDAR-based depth estimation with 1-sigma outlier filtering.
Example output: truck detected at **20.32m**, adjacent car at **~4m**.

---

## ✅ Test Results

```
tests/test_calibration_utils.py::test_cart2hom_shape                  PASSED
tests/test_calibration_utils.py::test_cart2hom_ones_appended          PASSED
tests/test_calibration_utils.py::test_cart2hom_original_values        PASSED
tests/test_geometry_utils.py::test_point_inside_rect                   PASSED
tests/test_geometry_utils.py::test_point_outside_rect                  PASSED
tests/test_geometry_utils.py::test_point_on_edge_excluded              PASSED
tests/test_geometry_utils.py::test_shrink_excludes_border              PASSED
tests/test_geometry_utils.py::test_shrink_includes_center              PASSED
tests/test_distance_utils.py::test_filter_outliers_removes_extreme     PASSED
tests/test_distance_utils.py::test_filter_outliers_single_item         PASSED
tests/test_distance_utils.py::test_get_best_distance_closest           PASSED
tests/test_distance_utils.py::test_get_best_distance_average           PASSED
tests/test_distance_utils.py::test_get_best_distance_median            PASSED
tests/test_distance_utils.py::test_get_best_distance_empty_raises      PASSED
tests/test_distance_utils.py::test_get_best_distance_invalid           PASSED

============= 15 passed ✅ =============
```

---

## 🔍 Notebook Walkthrough

The notebook `notebooks/Visual_Fusion_Early_Fusion.ipynb` provides a step-by-step explanation of:

1. Reading KITTI calibration files (P2, R0, Velo2Cam)
2. Projecting LiDAR points to the image plane
3. Filtering points to the camera Field of View (FOV)
4. Running YOLOv8 object detection
5. Fusing 2D detections with LiDAR depth estimates
6. Visualising results and comparing with ground truth labels

---

## 🙏 Acknowledgements

- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Open3D](http://www.open3d.org/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Krishna Sanjay Ambekar**
🔗 [GitHub](https://github.com/KrishnaSA05) • [LinkedIn](www.linkedin.com/in/krishna-ambekar-b4a2641b2)
