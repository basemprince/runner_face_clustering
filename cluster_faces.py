import warnings

import hdbscan
import numpy as np
from sklearn.neighbors import BallTree


def cluster_face_embeddings(
    embeddings,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    metric: str = "cosine",
):
    """Cluster face embeddings using HDBSCAN.

    The embeddings are L2-normalized before clustering. ``metric`` defaults to
    ``cosine`` which generally works well for face embeddings but can be changed
    to ``euclidean`` or ``mahalanobis`` if desired. Outliers labelled ``-1`` by
    HDBSCAN are treated as individual clusters so every sample receives a label.
    """

    X = np.vstack(embeddings)

    # Normalize for metrics like cosine/euclidean
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    if metric == "cosine" and "cosine" not in BallTree.valid_metrics:
        warnings.warn(
            "Cosine metric unsupported by this scikit-learn; falling back to euclidean."
        )
        metric = "euclidean"

    cluster_kwargs = dict(
        metric=metric,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    if metric == "mahalanobis":
        # Mahalanobis requires the covariance matrix
        cluster_kwargs["V"] = np.cov(X, rowvar=False)

    model = hdbscan.HDBSCAN(**cluster_kwargs)
    labels = model.fit_predict(X)

    # Assign each outlier to its own cluster label
    next_label = labels.max() + 1 if labels.size else 0
    for i, lbl in enumerate(labels):
        if lbl == -1:
            labels[i] = next_label
            next_label += 1

    return labels
