import re

import easyocr
from ultralytics import YOLO

bib_model = YOLO("yolov8n.pt")
ocr_reader = easyocr.Reader(["en"])


def detect_numeric_bibs(img, debug=False):
    results = bib_model(img)[0]
    bib_numbers = []

    for box in results.boxes.data:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped = img[y1:y2, x1:x2]
        text = ocr_reader.readtext(cropped)
        for t in text:
            candidate = t[1]
            if re.fullmatch(r"\d{2,5}", candidate):  # Accept 2–5 digit numbers
                if debug:
                    print(f"Valid bib detected: {candidate}")
                bib_numbers.append(candidate)
                break  # Assume one bib per person per image

    return bib_numbers
