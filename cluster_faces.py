import hdbscan
import numpy as np


def cluster_face_embeddings(embeddings):
    X = np.vstack(embeddings)
    model = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
    labels = model.fit_predict(X)
    return labels
