"""
This module contains the function to cluster face embeddings using HDBSCAN.
"""

import hdbscan
import numpy as np


def cluster_face_embeddings(embeddings):
    """Cluster face embeddings using HDBSCAN with safety checks."""
    stacked_embed = np.vstack(embeddings)

    # Remove rows with NaNs
    stacked_embed = stacked_embed[~np.isnan(stacked_embed).any(axis=1)]

    if stacked_embed.shape[0] < 2:
        return np.array([-1] * len(embeddings))  # Not enough valid points

    # Compute covariance safely
    cov = np.cov(stacked_embed, rowvar=False)
    if np.any(np.isnan(cov)) or np.linalg.matrix_rank(cov) < cov.shape[0]:
        cov += np.eye(cov.shape[0]) * 1e-6  # Regularize

    model = hdbscan.HDBSCAN(
        metric="mahalanobis",
        V=cov,
        min_cluster_size=2,
        cluster_selection_epsilon=0.0,
        min_samples=5,
    )

    labels = model.fit_predict(stacked_embed)

    # Treat each outlier as its own cluster
    next_label = labels.max() + 1 if labels.size else 0
    for idx, lbl in enumerate(labels):
        if lbl == -1:
            labels[idx] = next_label
            next_label += 1

    return labels
