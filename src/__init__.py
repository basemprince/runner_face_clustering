"""Face clustering utilities package."""

from .clustering.cluster_faces import cluster_face_embeddings
from .detection.crop_bodies import crop_person
from .detection.detect_bibs import detect_bib_in_crop
from .detection.detect_runners import detect_persons
from .detection.face_embeddings import extract_face_embeddings
from .visualization.visualize_embeddings import plot_embeddings, reduce_embeddings

__all__ = [
    "cluster_face_embeddings",
    "crop_person",
    "detect_bib_in_crop",
    "detect_persons",
    "extract_face_embeddings",
    "plot_embeddings",
    "reduce_embeddings",
]
