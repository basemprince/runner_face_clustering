import os
import shutil
from collections import defaultdict

import cv2
from sklearn.cluster import DBSCAN


def cluster_faces(face_entries):
    embeddings = [f["embedding"] for f in face_entries]
    clusterer = DBSCAN(eps=0.5, min_samples=1, metric="cosine")
    labels = clusterer.fit_predict(embeddings)
    return labels


def save_faces_grouped_by_id(face_entries, labels, bib_map, output_dir="output", debug=False):
    os.makedirs(output_dir, exist_ok=True)
    grouped = defaultdict(list)

    for face, label in zip(face_entries, labels):
        grouped[label].append(face)

    for cluster_id, faces in grouped.items():
        bib = bib_map.get(cluster_id)
        folder_name = f"bib#{bib}" if bib else f"person#{cluster_id + 1}"
        person_dir = os.path.join(output_dir, folder_name)
        os.makedirs(person_dir, exist_ok=True)

        for i, face_data in enumerate(faces):
            img_path = face_data["img_path"]
            filename = os.path.basename(img_path)
            save_path = os.path.join(person_dir, f"image_{i}_{filename}")
            shutil.copy(img_path, save_path)

            if debug:
                face_img = face_data["face_img"]
                cv2.imwrite(os.path.join(person_dir, f"face_{i}.jpg"), face_img)
