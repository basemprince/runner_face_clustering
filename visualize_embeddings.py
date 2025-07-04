"""Utility functions to visualize face embeddings with dimensionality reduction."""

from __future__ import annotations

import json
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_embeddings(embeddings: Iterable[np.ndarray], method: str = "pca", n_components: int = 2) -> np.ndarray:
    """Reduce embedding dimensions using PCA or t-SNE."""
    data = np.vstack(list(embeddings))

    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, init="random", random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    return reducer.fit_transform(data)


def plot_embeddings(reduced: np.ndarray, labels: Iterable[int] | None = None, out_path: str | None = None) -> None:
    """Plot 2D embeddings with optional cluster labels."""
    plt.figure(figsize=(8, 8))
    if labels is None:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7)
    else:
        labels_array = np.array(list(labels))
        for lbl in np.unique(labels_array):
            mask = labels_array == lbl
            plt.scatter(reduced[mask, 0], reduced[mask, 1], label=str(lbl), s=10, alpha=0.7)
        plt.legend(title="Cluster")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    else:
        plt.show()


def main() -> None:  # pragma: no cover - convenience script
    """CLI entry point for embedding visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize embeddings")
    parser.add_argument("embeddings", help="Path to JSON file containing a list of embeddings")
    parser.add_argument("--labels", help="Optional JSON file with cluster labels", default=None)
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--output", help="Path to save the plot")
    args = parser.parse_args()

    with open(args.embeddings, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = None
    if args.labels:
        with open(args.labels, "r", encoding="utf-8") as f:
            labels = json.load(f)

    reduced = reduce_embeddings(data, method=args.method)
    plot_embeddings(reduced, labels=labels, out_path=args.output)


if __name__ == "__main__":  # pragma: no cover - convenience script
    main()

