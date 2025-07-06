"""
this module provides functionality to detect persons in images using a YOLO model.
"""

from ultralytics import YOLO

person_detector = YOLO("yolo11n.pt")  # COCO pretrained


def detect_persons(image):
    """Detect persons in an image using YOLO."""
    results = person_detector(image)[0]
    return [
        (int(x1), int(y1), int(x2), int(y2))
        for x1, y1, x2, y2, conf, cls in results.boxes.data.tolist()
        if int(cls) == 0
    ]
