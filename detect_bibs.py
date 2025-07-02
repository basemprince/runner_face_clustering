import re

import easyocr

ocr = easyocr.Reader(["en"])


def detect_bib_in_crop(crop, debug=False):
    h, w = crop.shape[:2]
    roi = crop#[int(h * 0.1) : int(h * 0.9), :]  # middle 50% of height
    results = ocr.readtext(roi)
    non_bib_texts = []
    for _, t, _ in results:
        if re.fullmatch(r"\d{1,5}", t):
            if debug:
                print(f"Detected bib: {t}")
            return t, None
        non_bib_texts.append(t)
        if debug:
            print(f"Non-bib text detected: {t}")
    return None, " ".join(non_bib_texts).strip()
