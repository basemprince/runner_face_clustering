import insightface

face_model = insightface.app.FaceAnalysis(name="buffalo_l")
face_model.prepare(ctx_id=0)


def extract_faces_and_embeddings(img, debug=False):
    h, w = img.shape[:2]
    faces = face_model.get(img)
    results = []

    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)

        # Clip coordinates to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            if debug:
                print(f"Skipped invalid face box: {(x1, y1, x2, y2)}")
            continue

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            if debug:
                print(f"Skipped empty crop at: {(x1, y1, x2, y2)}")
            continue

        results.append({"face_img": face_crop, "embedding": f.embedding, "bbox": (x1, y1, x2, y2)})

        if debug:
            print(f"Face detected at {(x1, y1, x2, y2)}")

    return results
