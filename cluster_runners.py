"""
this module contains the function to cluster feature vectors using DBSCAN.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_embeddings(embeddings, eps=0.1, min_samples=1):
    """
    Cluster feature vectors using DBSCAN (cosine distance).
    """
    stacked_embed = np.vstack(embeddings)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = model.fit_predict(stacked_embed)

    print(f"🧩 DBSCAN found {len(set(labels) - {-1})} clusters, with {list(labels).count(-1)} noise samples.")
    return labels
