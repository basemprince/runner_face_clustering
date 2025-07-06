"""
this module provides functionality to extract face embeddings from images using the InsightFace library.
"""

import onnxruntime as ort
from insightface.app import FaceAnalysis

# Check available ONNX providers
available_providers = ort.get_available_providers()

# Choose best provider in priority order
if "CUDAExecutionProvider" in available_providers:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    CTX_ID = 0  # GPU context
elif "CoreMLExecutionProvider" in available_providers:
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    CTX_ID = 0  # Apple Neural Engine
else:
    providers = ["CPUExecutionProvider"]
    CTX_ID = -1  # CPU context

# Initialize and prepare the face model
face_model = FaceAnalysis(name="buffalo_l", providers=providers)
face_model.prepare(ctx_id=CTX_ID)

print(f"Using providers: {providers}, ctx_id: {CTX_ID}")


def extract_face_embeddings(image):
    """Extract face embeddings from an image."""
    faces = face_model.get(image)
    results = []
    for f in faces:
        box = tuple(map(int, f.bbox))
        results.append({"embedding": f.embedding, "bbox": box})
    return results
