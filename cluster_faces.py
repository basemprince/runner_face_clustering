import hdbscan
import numpy as np


def cluster_face_embeddings(embeddings):
    X = np.vstack(embeddings)
    cov = np.cov(X, rowvar=False)
    model = hdbscan.HDBSCAN(
        metric='mahalanobis',
        V=cov,
        min_cluster_size=2,
        cluster_selection_epsilon=0.0,
        min_samples=5,
    )

    labels = model.fit_predict(X)
    return labels
