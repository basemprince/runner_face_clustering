"""Detection utilities for runner processing."""

from .crop_bodies import crop_person
from .detect_bibs import detect_bib_in_crop
from .detect_runners import detect_persons
from .extract_faces import extract_faces_and_embeddings
from .face_embeddings import extract_face_embeddings

__all__ = [
    "crop_person",
    "detect_bib_in_crop",
    "detect_persons",
    "extract_face_embeddings",
    "extract_faces_and_embeddings",
]
