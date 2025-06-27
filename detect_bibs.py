import re

import easyocr
from ultralytics import YOLO

bib_model = YOLO("yolov8n.pt")
ocr_reader = easyocr.Reader(["en"])


def detect_numeric_bibs_with_boxes(img, debug=False):
    results = bib_model(img)[0]
    bib_regions = []

    for box in results.boxes.data:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped = img[y1:y2, x1:x2]
        text = ocr_reader.readtext(cropped)
        for t in text:
            candidate = t[1]
            if re.fullmatch(r"\d{1,5}", candidate):  # 1–5 digit bibs
                if debug:
                    print(f"Detected bib: {candidate} at box {x1, y1, x2, y2}")
                bib_regions.append((candidate, (x1, y1, x2, y2)))
                break  # one number per region is enough

    return bib_regions
