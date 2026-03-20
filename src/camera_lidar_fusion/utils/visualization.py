import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


CLASS_LABELS = {1: "cyclist", 2: "car", 7: "truck"}


def draw_lidar_on_image(
    img: np.ndarray,
    imgfov_pts_2d: np.ndarray,
    imgfov_pc_velo: np.ndarray
) -> np.ndarray:
    """Overlay depth-coloured LiDAR points onto a camera image."""
    cmap = plt.colormaps["hsv"].resampled(256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_velo[i, 0]
        color = cmap[int(510.0 / depth), :]
        cv2.circle(
            img,
            (int(np.round(imgfov_pts_2d[i, 0])),
             int(np.round(imgfov_pts_2d[i, 1]))),
            2, color=tuple(color), thickness=-1
        )
    return img


def draw_boxes_cv(img, detections):
    """Draw YOLO bounding boxes with class label and confidence on image."""
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, Image.Image):
        img = np.array(img)

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        label = f"{CLASS_LABELS.get(int(class_id), str(int(class_id)))}: {confidence:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (int(x1), int(y1) - 20),
                      (int(x1) + w, int(y1) - 20 + h), (0, 255, 0), -1)
        cv2.putText(img, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return Image.fromarray(img)
