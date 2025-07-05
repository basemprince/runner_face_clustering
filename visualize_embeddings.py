"""Utility functions to visualize face embeddings with dimensionality reduction."""

# pylint: disable=import-outside-toplevel, duplicate-code

from __future__ import annotations

import json
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _auto_pca_components(data: np.ndarray, threshold: float = 0.7) -> int:
    """Return the smallest number of PCA components explaining ``threshold`` variance.

    Parameters
    ----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    threshold : float, optional
        Proportion of variance to explain. Defaults to ``0.7`` (70%%).
    """
    n_components_values = list(range(1, min(data.shape[0], data.shape[1]) + 1))
    explained_variances = []
    for n_component in n_components_values:
        pca = PCA(n_components=n_component)
        pca.fit(data)
        explained_variances.append(np.sum(pca.explained_variance_ratio_))
    for n_component, variance in zip(n_components_values, explained_variances):
        if variance >= threshold:
            print(f"Auto-selected {n_component} PCA components to explain {variance:.2%} variance")
            return n_component
    print(f"Using all {n_components_values[-1]} components to explain {explained_variances[-1]:.2%} variance")
    return n_components_values[-1]


def reduce_embeddings(
    embeddings: Iterable[np.ndarray],
    method: str = "pca",
    n_components: int | str = 2,
    auto_pca_threshold: float = 0.7,
) -> np.ndarray:
    """Reduce embedding dimensions using PCA or t-SNE.

    When ``method`` is ``"pca"`` and ``n_components`` is ``"auto```, the number
    of components is chosen to explain at least ``auto_pca_threshold`` of the
    variance.
    """
    data = np.vstack(list(embeddings))

    if method == "pca":
        if n_components == "auto":
            n_components = _auto_pca_components(data, threshold=auto_pca_threshold)
        reducer = PCA(n_components=int(n_components))
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, init="random", random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    return reducer.fit_transform(data)


def plot_embeddings(
    reduced: np.ndarray,
    labels: Iterable[int] | None = None,
    out_path: str | None = None,
) -> None:
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
    parser.add_argument(
        "--n-components",
        default="2",
        help="Output dimensionality or 'auto' for PCA",
    )
    parser.add_argument(
        "--auto-pca-threshold",
        type=float,
        default=0.7,
        help="Variance proportion for automatic PCA component selection",
    )
    parser.add_argument("--output", help="Path to save the plot")
    args = parser.parse_args()

    with open(args.embeddings, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = None
    if args.labels:
        with open(args.labels, "r", encoding="utf-8") as f:
            labels = json.load(f)

    n_components: int | str = "auto" if args.n_components == "auto" else int(args.n_components)
    reduced = reduce_embeddings(
        data,
        method=args.method,
        n_components=n_components,
        auto_pca_threshold=args.auto_pca_threshold,
    )
    plot_embeddings(reduced, labels=labels, out_path=args.output)


if __name__ == "__main__":  # pragma: no cover - convenience script
    main()
