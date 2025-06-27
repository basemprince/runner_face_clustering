from sklearn.cluster import DBSCAN
import numpy as np
import os
import cv2
from collections import defaultdict

def cluster_faces(face_data):
    X = [e for _, e in face_data]
    clusterer = DBSCAN(eps=0.5, min_samples=1, metric='cosine')
    labels = clusterer.fit_predict(X)
    return labels

def save_faces(clustered_faces, labels, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    label_map = defaultdict(list)

    for (face_img, _), label in zip(clustered_faces, labels):
        label_map[label].append(face_img)

    for label, face_imgs in label_map.items():
        person_dir = os.path.join(output_dir, f'person#{label + 1}')
        os.makedirs(person_dir, exist_ok=True)
        for i, face in enumerate(face_imgs):
            cv2.imwrite(os.path.join(person_dir, f'face_{i}.jpg'), face)
