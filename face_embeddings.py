"""
this module provides functionality to extract face embeddings from images using the InsightFace library.
"""

from insightface.app import FaceAnalysis

face_model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_model.prepare(ctx_id=0)


def extract_face_embeddings(image):
    """Extract face embeddings from an image."""
    faces = face_model.get(image)
    results = []
    for f in faces:
        box = tuple(map(int, f.bbox))
        results.append({"embedding": f.embedding, "bbox": box})
    return results
