from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import numpy as np


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

    model = AgglomerativeClustering(
        affinity="precomputed",
        linkage="average",
        distance_threshold=threshold,
        n_clusters=None,
    )

    labels = model.fit_predict(distance_matrix)
    return labels
