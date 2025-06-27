import insightface
import cv2

face_model = insightface.app.FaceAnalysis(name='buffalo_l')
face_model.prepare(ctx_id=0)  # Use CPU

def extract_faces_and_embeddings(img, debug=False):
    faces = face_model.get(img)
    results = []

    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        emb = f.embedding
        face_crop = img[y1:y2, x1:x2]
        results.append((face_crop, emb))

        if debug:
            print(f"Face detected at {(x1, y1, x2, y2)}")

    return results
