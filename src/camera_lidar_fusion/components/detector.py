import cv2


def run_obstacle_detection(img, model, conf: float = 0.5, classes: list = None):
    """
    Run YOLOv8 inference on a BGR image.
    Returns annotated image (numpy array) and predicted bounding boxes.
    """
    if classes is None:
        classes = [1, 2, 7]   # cyclist, car, truck (KITTI YOLO mapping)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictions = model(img_rgb, conf=conf, classes=classes)

    result = None
    pred_bboxes = None
    for r in predictions:
        pred_bboxes = r.boxes.data.detach().cpu().numpy()
        result = r.plot()

    return result, pred_bboxes
