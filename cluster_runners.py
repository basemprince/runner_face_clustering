from sklearn.cluster import DBSCAN
import numpy as np

def cluster_embeddings(embeddings, eps=0.1, min_samples=1):
    """
    Cluster feature vectors using DBSCAN (cosine distance).
    """
    X = np.vstack(embeddings)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = model.fit_predict(X)

    print(f"🧩 DBSCAN found {len(set(labels) - {-1})} clusters, with {list(labels).count(-1)} noise samples.")
    return labels
