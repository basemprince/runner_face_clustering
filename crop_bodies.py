def crop_person(image, box, margin=0.1):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
    x2, y2 = min(w, x2 + dx), min(h, y2 + dy)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)
