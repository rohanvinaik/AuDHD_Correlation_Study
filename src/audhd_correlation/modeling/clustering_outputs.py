#!/usr/bin/env python3
"""
Clustering Output Utilities

Save co-assignment matrices, UMAP/t-SNE projections, and other clustering
artifacts for inspection and reporting.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple
import json


def save_coassignment_matrix(
    coassignment: np.ndarray,
    sample_ids: np.ndarray,
    output_path: str,
    format: str = "both"
) -> None:
    """
    Save co-assignment matrix to disk

    Args:
        coassignment: Co-assignment matrix (n_samples, n_samples)
        sample_ids: Sample IDs
        output_path: Output file path (without extension)
        format: "csv", "npy", or "both"
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format in ["csv", "both"]:
        # Save as CSV with sample IDs
        df = pd.DataFrame(coassignment, index=sample_ids, columns=sample_ids)
        df.to_csv(f"{output_path}.csv")
        print(f"Co-assignment matrix saved: {output_path}.csv")

    if format in ["npy", "both"]:
        # Save as NumPy array (more efficient for large matrices)
        np.savez(
            f"{output_path}.npz",
            coassignment=coassignment,
            sample_ids=sample_ids
        )
        print(f"Co-assignment matrix saved: {output_path}.npz")


def load_coassignment_matrix(
    input_path: str,
    format: str = "auto"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load co-assignment matrix from disk

    Args:
        input_path: Input file path (with or without extension)
        format: "csv", "npy", or "auto" (detect from extension)

    Returns:
        Tuple of (coassignment, sample_ids)
    """
    input_path = Path(input_path)

    if format == "auto":
        if input_path.suffix == ".csv":
            format = "csv"
        elif input_path.suffix == ".npz":
            format = "npy"
        else:
            # Try .csv first
            if input_path.with_suffix(".csv").exists():
                format = "csv"
            elif input_path.with_suffix(".npz").exists():
                format = "npy"
            else:
                raise FileNotFoundError(f"Co-assignment matrix not found: {input_path}")

    if format == "csv":
        df = pd.read_csv(input_path, index_col=0)
        coassignment = df.values
        sample_ids = df.index.values
    elif format == "npy":
        data = np.load(input_path)
        coassignment = data["coassignment"]
        sample_ids = data["sample_ids"]
    else:
        raise ValueError(f"Unknown format: {format}")

    return coassignment, sample_ids


def plot_coassignment_matrix(
    coassignment: np.ndarray,
    labels: Optional[np.ndarray] = None,
    sample_ids: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "viridis",
) -> None:
    """
    Plot co-assignment matrix as heatmap

    Args:
        coassignment: Co-assignment matrix (n_samples, n_samples)
        labels: Optional cluster labels for ordering
        sample_ids: Optional sample IDs for axis labels
        output_path: Optional path to save figure
        figsize: Figure size
        cmap: Colormap
    """
    n_samples = coassignment.shape[0]

    # Sort by cluster labels if provided
    if labels is not None:
        # Sort by label, then by co-assignment within label
        sort_idx = np.argsort(labels)
        coassignment_sorted = coassignment[sort_idx][:, sort_idx]

        if sample_ids is not None:
            sample_ids_sorted = sample_ids[sort_idx]
        else:
            sample_ids_sorted = None

        labels_sorted = labels[sort_idx]
    else:
        coassignment_sorted = coassignment
        sample_ids_sorted = sample_ids
        labels_sorted = None

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(coassignment_sorted, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Co-assignment Frequency", rotation=270, labelpad=20)

    # Labels
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")
    ax.set_title("Co-assignment Matrix")

    # Add cluster boundaries if labels provided
    if labels_sorted is not None:
        # Find boundaries
        boundaries = []
        current_label = labels_sorted[0]
        for i, label in enumerate(labels_sorted):
            if label != current_label:
                boundaries.append(i)
                current_label = label

        # Draw lines
        for boundary in boundaries:
            ax.axhline(boundary, color='white', linewidth=2, linestyle='--')
            ax.axvline(boundary, color='white', linewidth=2, linestyle='--')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Co-assignment plot saved: {output_path}")
    else:
        plt.show()

    plt.close()


def save_projection(
    embedding: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    method: str,
    output_path: str,
    confidence: Optional[np.ndarray] = None,
) -> None:
    """
    Save 2D projection (t-SNE/UMAP) to disk

    Args:
        embedding: 2D embedding (n_samples, 2)
        labels: Cluster labels
        sample_ids: Sample IDs
        method: Method name (e.g., "tsne_perp30", "umap_n15")
        output_path: Output file path (without extension)
        confidence: Optional confidence scores
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame({
        "sample_id": sample_ids,
        f"{method}_1": embedding[:, 0],
        f"{method}_2": embedding[:, 1],
        "cluster": labels,
    })

    if confidence is not None:
        df["confidence"] = confidence

    # Save
    df.to_csv(f"{output_path}_{method}.csv", index=False)
    print(f"Projection saved: {output_path}_{method}.csv")


def save_all_projections(
    embeddings: Dict[str, np.ndarray],
    labels: np.ndarray,
    sample_ids: np.ndarray,
    output_dir: str,
    confidence: Optional[np.ndarray] = None,
) -> None:
    """
    Save all projections to disk

    Args:
        embeddings: Dictionary of {method_name: embedding}
        labels: Cluster labels
        sample_ids: Sample IDs
        output_dir: Output directory
        confidence: Optional confidence scores
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, embedding in embeddings.items():
        if embedding.shape[1] == 2:  # Only save 2D projections
            save_projection(
                embedding=embedding,
                labels=labels,
                sample_ids=sample_ids,
                method=method,
                output_path=output_dir / "projections",
                confidence=confidence,
            )


def plot_projection(
    embedding: np.ndarray,
    labels: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    method: str = "Projection",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """
    Plot 2D projection with clusters and confidence

    Args:
        embedding: 2D embedding (n_samples, 2)
        labels: Cluster labels
        confidence: Optional confidence scores
        method: Method name for title
        output_path: Optional path to save figure
        figsize: Figure size
    """
    if confidence is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        axes = [ax]

    # Plot 1: Clusters
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        label_name = "Noise" if label == -1 else f"Cluster {label}"

        axes[0].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            label=label_name,
            s=50,
            alpha=0.6,
            edgecolors='k',
            linewidths=0.5
        )

    axes[0].set_xlabel(f"{method} 1")
    axes[0].set_ylabel(f"{method} 2")
    axes[0].set_title(f"{method} - Clusters")
    axes[0].legend(loc='best')

    # Plot 2: Confidence (if provided)
    if confidence is not None:
        scatter = axes[1].scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=confidence,
            s=50,
            alpha=0.6,
            cmap='viridis',
            edgecolors='k',
            linewidths=0.5
        )
        plt.colorbar(scatter, ax=axes[1], label="Confidence")

        axes[1].set_xlabel(f"{method} 1")
        axes[1].set_ylabel(f"{method} 2")
        axes[1].set_title(f"{method} - Confidence")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Projection plot saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_all_projections(
    embeddings: Dict[str, np.ndarray],
    labels: np.ndarray,
    output_dir: str,
    confidence: Optional[np.ndarray] = None,
) -> None:
    """
    Plot all 2D projections

    Args:
        embeddings: Dictionary of {method_name: embedding}
        labels: Cluster labels
        output_dir: Output directory
        confidence: Optional confidence scores
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, embedding in embeddings.items():
        if embedding.shape[1] == 2:
            plot_projection(
                embedding=embedding,
                labels=labels,
                confidence=confidence,
                method=method,
                output_path=output_dir / f"projection_{method}.png",
            )


def save_clustering_summary(
    labels: np.ndarray,
    metrics: dict,
    output_path: str,
    confidence: Optional[np.ndarray] = None,
    coassignment: Optional[np.ndarray] = None,
) -> None:
    """
    Save clustering summary to JSON

    Args:
        labels: Cluster labels
        metrics: Clustering metrics dictionary
        output_path: Output file path
        confidence: Optional confidence scores
        coassignment: Optional co-assignment matrix
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Cluster sizes
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    cluster_sizes = {int(label): int(count) for label, count in zip(unique, counts)}

    # Build summary
    summary = {
        "n_samples": len(labels),
        "n_clusters": len(unique),
        "n_noise": int((labels == -1).sum()),
        "cluster_sizes": cluster_sizes,
        "metrics": metrics,
    }

    if confidence is not None:
        summary["mean_confidence"] = float(confidence.mean())
        summary["min_confidence"] = float(confidence.min())
        summary["max_confidence"] = float(confidence.max())

        # Per-cluster confidence
        cluster_confidence = {}
        for label in unique:
            mask = labels == label
            cluster_confidence[int(label)] = {
                "mean": float(confidence[mask].mean()),
                "std": float(confidence[mask].std()),
            }
        summary["cluster_confidence"] = cluster_confidence

    if coassignment is not None:
        # Summary statistics
        summary["coassignment_stats"] = {
            "mean": float(coassignment.mean()),
            "min": float(coassignment.min()),
            "max": float(coassignment.max()),
        }

    # Save
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Clustering summary saved: {output_path}")


def save_all_clustering_outputs(
    labels: np.ndarray,
    sample_ids: np.ndarray,
    output_dir: str,
    coassignment: Optional[np.ndarray] = None,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
    confidence: Optional[np.ndarray] = None,
    metrics: Optional[dict] = None,
) -> None:
    """
    Save all clustering outputs to disk

    Args:
        labels: Cluster labels
        sample_ids: Sample IDs
        output_dir: Output directory
        coassignment: Optional co-assignment matrix
        embeddings: Optional dictionary of embeddings
        confidence: Optional confidence scores
        metrics: Optional clustering metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save labels
    labels_df = pd.DataFrame({
        "sample_id": sample_ids,
        "cluster": labels,
    })
    if confidence is not None:
        labels_df["confidence"] = confidence

    labels_df.to_csv(output_dir / "cluster_labels.csv", index=False)
    print(f"Cluster labels saved: {output_dir}/cluster_labels.csv")

    # Save co-assignment matrix
    if coassignment is not None:
        save_coassignment_matrix(
            coassignment=coassignment,
            sample_ids=sample_ids,
            output_path=output_dir / "coassignment_matrix",
            format="both"
        )

        # Plot co-assignment matrix
        plot_coassignment_matrix(
            coassignment=coassignment,
            labels=labels,
            sample_ids=sample_ids,
            output_path=output_dir / "coassignment_matrix.png"
        )

    # Save projections
    if embeddings is not None:
        save_all_projections(
            embeddings=embeddings,
            labels=labels,
            sample_ids=sample_ids,
            output_dir=output_dir,
            confidence=confidence,
        )

        # Plot projections
        plot_all_projections(
            embeddings=embeddings,
            labels=labels,
            output_dir=output_dir,
            confidence=confidence,
        )

    # Save summary
    if metrics is not None:
        save_clustering_summary(
            labels=labels,
            metrics=metrics,
            output_path=output_dir / "clustering_summary.json",
            confidence=confidence,
            coassignment=coassignment,
        )

    print(f"\n✓ All clustering outputs saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate synthetic data
    n_samples = 100
    sample_ids = np.array([f"SAMPLE_{i:03d}" for i in range(n_samples)])

    # Fake clustering results
    labels = np.random.choice([0, 1, 2, -1], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    confidence = np.random.rand(n_samples) * 0.5 + 0.5

    # Fake co-assignment matrix
    coassignment = np.random.rand(n_samples, n_samples)
    coassignment = (coassignment + coassignment.T) / 2  # Symmetrize
    np.fill_diagonal(coassignment, 1.0)

    # Fake embeddings
    embeddings = {
        "tsne_perp30": np.random.randn(n_samples, 2),
        "umap_n15": np.random.randn(n_samples, 2),
    }

    # Fake metrics
    metrics = {
        "silhouette": 0.45,
        "calinski_harabasz": 120.5,
        "davies_bouldin": 0.89,
        "n_clusters": 3,
        "n_noise": (labels == -1).sum(),
    }

    # Save all
    save_all_clustering_outputs(
        labels=labels,
        sample_ids=sample_ids,
        output_dir="outputs/clustering_example",
        coassignment=coassignment,
        embeddings=embeddings,
        confidence=confidence,
        metrics=metrics,
    )