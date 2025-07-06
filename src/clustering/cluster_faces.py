"""
This module contains the function to cluster face embeddings using HDBSCAN.
"""

from typing import Sequence

import hdbscan
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from visualization import _auto_pca_components


def cluster_face_embeddings(
    embeddings: Sequence[np.ndarray],
    reduce_method: str | None = None,
    n_components: int | str = 2,
    auto_pca_threshold: float = 0.7,
):
    """Cluster face embeddings using HDBSCAN with optional dimensionality reduction.

    Parameters
    ----------
    embeddings : Iterable[np.ndarray]
        Input embedding vectors.
    reduce_method : {"pca", "tsne"} | None, optional
        If provided, reduce the embeddings before clustering using the specified method.
    n_components : int | str, optional
        Number of dimensions for the reducer. If ``"auto"`` with PCA, the number
        of components explaining at least ``auto_pca_threshold`` variance is
        chosen. Defaults to ``2``.
    auto_pca_threshold : float, optional
        Proportion of variance to explain when automatically selecting PCA
        components. Defaults to ``0.7`` (70%%).

    Returns
    -------
    np.ndarray
        Cluster labels for each embedding. If ``embeddings`` is empty, an empty
        array is returned.
    """
    if not embeddings:
        return np.array([], dtype=int)

    if len(embeddings) == 1:
        return np.array([0], dtype=int)

    stacked_embed = np.vstack(embeddings)

    # Remove rows with NaNs
    stacked_embed = stacked_embed[~np.isnan(stacked_embed).any(axis=1)]

    if reduce_method:
        if reduce_method == "pca":
            if n_components == "auto":
                n_components = _auto_pca_components(
                    stacked_embed,
                    threshold=auto_pca_threshold,
                )
            reducer = PCA(n_components=int(n_components))
        elif reduce_method == "tsne":
            n_components = 2 if n_components == "auto" else n_components
            reducer = TSNE(n_components=int(n_components), init="random", random_state=42)
        else:
            raise ValueError("reduce_method must be 'pca' or 'tsne'")
        stacked_embed = reducer.fit_transform(stacked_embed)

    # Compute covariance safely
    cov = np.cov(stacked_embed, rowvar=False)

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
