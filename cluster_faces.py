"""
This module contains the function to cluster face embeddings using HDBSCAN.
"""

import hdbscan
import numpy as np


def cluster_face_embeddings(embeddings):
    """Cluster face embeddings using HDBSCAN."""
    stacked_embed = np.vstack(embeddings)
    cov = np.cov(stacked_embed, rowvar=False)
    model = hdbscan.HDBSCAN(
        metric="mahalanobis",
        V=cov,
        min_cluster_size=2,
        cluster_selection_epsilon=0.0,
        min_samples=5,
    )

    labels = model.fit_predict(stacked_embed)

    # Treat each HDBSCAN outlier (-1) as its own cluster label
    next_label = labels.max() + 1 if labels.size else 0
    for idx, lbl in enumerate(labels):
        if lbl == -1:
            labels[idx] = next_label
            next_label += 1

    return labels
