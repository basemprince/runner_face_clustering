from ultralytics import YOLO
import easyocr
import cv2

bib_model = YOLO('yolov8n.pt')
ocr_reader = easyocr.Reader(['en'])

def detect_bibs_and_numbers(img, debug=False):
    results = bib_model(img)[0]
    bib_regions = []

    for box in results.boxes.data:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped = img[y1:y2, x1:x2]
        text = ocr_reader.readtext(cropped)
        bib_texts = [t[1] for t in text]

        if debug:
            print("OCR Result:", text)

        if bib_texts:
            bib_regions.append((bib_texts[0], (x1, y1, x2, y2)))

    return bib_regions
