import re

import easyocr
from ultralytics import YOLO

bib_model = YOLO("yolov8n.pt")  # fine-tune for bibs for better accuracy
ocr = easyocr.Reader(["en"])


def detect_bib_in_crop(crop, debug=False):
    res = bib_model(crop)[0]
    for x1, y1, x2, y2, conf, cls in res.boxes.data.tolist():
        text = ocr.readtext(crop[int(y1) : int(y2), int(x1) : int(x2)])
        for _, t, score in text:
            if re.fullmatch(r"\d{1,5}", t):
                if debug:
                    print(f"Detected bib: {t} at box {x1, y1, x2, y2}")
                return t, (int(x1), int(y1), int(x2), int(y2))
    return None, None
