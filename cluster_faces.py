from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import inspect


def cluster_face_embeddings(embeddings, threshold: float = 0.3):
    """Cluster face embeddings using agglomerative clustering.

    A cosine distance matrix is computed and then ``AgglomerativeClustering``
    is run with ``distance_threshold`` so clusters are formed solely based on
    similarity.  The default threshold of ``0.3`` works reasonably well for
    embeddings produced by ``insightface`` but can be tuned by callers.
    """

    X = np.vstack(embeddings)

    # Compute pairwise cosine distances and let the sklearn implementation
    # operate on the pre-computed matrix. Using cosine distance tends to
    # perform better for normalized face embeddings than Euclidean distance.
    distance_matrix = cosine_distances(X)

    clustering_kwargs = {
        "linkage": "average",
        "distance_threshold": threshold,
        "n_clusters": None,
    }
    # scikit-learn 1.2 renamed the ``affinity`` parameter to ``metric`` and
    # removed it entirely in later versions. Detect which keyword is supported
    # at runtime for better compatibility across versions.
    if "metric" in inspect.signature(AgglomerativeClustering).parameters:
        clustering_kwargs["metric"] = "precomputed"
    else:
        clustering_kwargs["affinity"] = "precomputed"

    model = AgglomerativeClustering(**clustering_kwargs)

    labels = model.fit_predict(distance_matrix)
    return labels
