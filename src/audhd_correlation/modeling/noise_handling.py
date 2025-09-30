#!/usr/bin/env python3
"""
Noise Handling Strategies for HDBSCAN

Implements strategies for handling HDBSCAN noise points (label -1):
1. Keep: Retain -1 as separate noise cluster
2. Reassign: Assign to nearest cluster
3. Filter: Remove from analysis

See docs/clustering_configuration.md for details.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.neighbors import NearestNeighbors


def reassign_noise_nearest_cluster(
    X: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    Reassign noise points to nearest cluster centroid

    Args:
        X: Data matrix (n_samples, n_features)
        labels: Cluster labels with -1 for noise

    Returns:
        Labels with noise reassigned
    """
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels

    # Compute cluster centroids
    centroids = {}
    for cluster in set(labels) - {-1}:
        cluster_mask = labels == cluster
        centroids[cluster] = X[cluster_mask].mean(axis=0)

    if not centroids:
        # No valid clusters, return original
        return labels

    # Assign noise to nearest centroid
    labels_reassigned = labels.copy()
    for i in np.where(noise_mask)[0]:
        distances = {
            cluster: np.linalg.norm(X[i] - centroid)
            for cluster, centroid in centroids.items()
        }
        labels_reassigned[i] = min(distances, key=distances.get)

    return labels_reassigned


def reassign_noise_knn_vote(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 5
) -> np.ndarray:
    """
    Reassign noise points by majority vote of k nearest neighbors

    Args:
        X: Data matrix (n_samples, n_features)
        labels: Cluster labels with -1 for noise
        k: Number of neighbors for voting

    Returns:
        Labels with noise reassigned
    """
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels

    # Fit k-NN on non-noise points
    valid_mask = ~noise_mask
    if valid_mask.sum() < k:
        # Not enough valid points, fall back to nearest cluster
        return reassign_noise_nearest_cluster(X, labels)

    knn = NearestNeighbors(n_neighbors=min(k, valid_mask.sum()))
    knn.fit(X[valid_mask])

    # Find neighbors for noise points
    _, indices = knn.kneighbors(X[noise_mask])

    # Vote
    labels_reassigned = labels.copy()
    valid_labels = labels[valid_mask]

    for i, neighbors_idx in zip(np.where(noise_mask)[0], indices):
        neighbor_labels = valid_labels[neighbors_idx]
        # Majority vote
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        labels_reassigned[i] = unique[counts.argmax()]

    return labels_reassigned


def filter_noise(
    X: np.ndarray,
    labels: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    min_confidence: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out noise points and optionally low-confidence samples

    Args:
        X: Data matrix (n_samples, n_features)
        labels: Cluster labels with -1 for noise
        confidence: Optional confidence scores (n_samples,)
        min_confidence: Minimum confidence threshold

    Returns:
        Tuple of (X_filtered, labels_filtered, indices_kept)
    """
    # Filter noise
    valid_mask = labels >= 0

    # Additional confidence filtering if provided
    if confidence is not None:
        confidence_mask = confidence >= min_confidence
        valid_mask = valid_mask & confidence_mask

    X_filtered = X[valid_mask]
    labels_filtered = labels[valid_mask]
    indices_kept = np.where(valid_mask)[0]

    return X_filtered, labels_filtered, indices_kept


def handle_noise(
    X: np.ndarray,
    labels: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    strategy: str = "keep",
    reassign_method: str = "nearest_cluster",
    knn_k: int = 5,
    min_confidence: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Handle noise points according to specified strategy

    Args:
        X: Data matrix (n_samples, n_features)
        labels: Cluster labels with -1 for noise
        confidence: Optional confidence scores
        strategy: "keep", "reassign", or "filter"
        reassign_method: "nearest_cluster" or "knn_vote" (if strategy=reassign)
        knn_k: Number of neighbors for knn_vote
        min_confidence: Confidence threshold (if strategy=filter)

    Returns:
        Tuple of (X_out, labels_out, indices_kept)
        - X_out: Data (possibly filtered)
        - labels_out: Labels (possibly reassigned)
        - indices_kept: Indices of kept samples (None if strategy != filter)
    """
    if strategy == "keep":
        # No modification
        return X, labels, None

    elif strategy == "reassign":
        if reassign_method == "nearest_cluster":
            labels_out = reassign_noise_nearest_cluster(X, labels)
        elif reassign_method == "knn_vote":
            labels_out = reassign_noise_knn_vote(X, labels, k=knn_k)
        else:
            raise ValueError(f"Unknown reassign_method: {reassign_method}")

        return X, labels_out, None

    elif strategy == "filter":
        X_out, labels_out, indices = filter_noise(
            X, labels, confidence, min_confidence
        )
        return X_out, labels_out, indices

    else:
        raise ValueError(f"Unknown noise handling strategy: {strategy}")


def detect_imbalance(
    labels: np.ndarray,
    threshold: float = 0.1,
    ignore_noise: bool = True
) -> Tuple[bool, dict]:
    """
    Detect imbalanced clusters

    Args:
        labels: Cluster labels
        threshold: Imbalance threshold (min_size / max_size)
        ignore_noise: Ignore noise points (label -1)

    Returns:
        Tuple of (is_imbalanced, info_dict)
    """
    # Filter noise if requested
    if ignore_noise:
        labels_filtered = labels[labels >= 0]
    else:
        labels_filtered = labels

    if len(labels_filtered) == 0:
        return False, {"reason": "no_valid_samples"}

    # Count cluster sizes
    unique, counts = np.unique(labels_filtered, return_counts=True)

    if len(counts) < 2:
        return False, {"reason": "single_cluster"}

    min_size = counts.min()
    max_size = counts.max()
    ratio = min_size / max_size

    is_imbalanced = ratio < threshold

    info = {
        "ratio": ratio,
        "min_size": int(min_size),
        "max_size": int(max_size),
        "n_clusters": len(counts),
        "threshold": threshold,
        "cluster_sizes": {
            int(cluster): int(count)
            for cluster, count in zip(unique, counts)
        }
    }

    return is_imbalanced, info


def merge_small_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    min_size: int = 10
) -> np.ndarray:
    """
    Merge clusters smaller than min_size into nearest cluster

    Args:
        X: Data matrix (n_samples, n_features)
        labels: Cluster labels
        min_size: Minimum cluster size threshold

    Returns:
        Labels with small clusters merged
    """
    # Identify small clusters
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    small_clusters = set(unique[counts < min_size])

    if not small_clusters:
        return labels

    # Get large clusters
    large_clusters = set(unique) - small_clusters

    if not large_clusters:
        # All clusters are small, keep as-is
        return labels

    # Compute centroids for large clusters
    centroids = {}
    for cluster in large_clusters:
        cluster_mask = labels == cluster
        centroids[cluster] = X[cluster_mask].mean(axis=0)

    # Merge small clusters
    labels_merged = labels.copy()
    for small_cluster in small_clusters:
        mask = labels == small_cluster
        cluster_center = X[mask].mean(axis=0)

        # Find nearest large cluster
        distances = {
            cluster: np.linalg.norm(cluster_center - centroid)
            for cluster, centroid in centroids.items()
        }
        nearest = min(distances, key=distances.get)

        labels_merged[mask] = nearest

    return labels_merged


def get_noise_statistics(
    labels: np.ndarray,
    confidence: Optional[np.ndarray] = None
) -> dict:
    """
    Get statistics about noise points

    Args:
        labels: Cluster labels
        confidence: Optional confidence scores

    Returns:
        Dictionary with statistics
    """
    noise_mask = labels == -1
    n_noise = noise_mask.sum()
    n_total = len(labels)

    stats = {
        "n_noise": int(n_noise),
        "n_total": int(n_total),
        "noise_fraction": float(n_noise / n_total) if n_total > 0 else 0.0,
        "n_valid": int(n_total - n_noise),
    }

    if confidence is not None:
        stats["mean_confidence_noise"] = float(confidence[noise_mask].mean()) if n_noise > 0 else None
        stats["mean_confidence_valid"] = float(confidence[~noise_mask].mean()) if (n_total - n_noise) > 0 else None

    # Cluster sizes
    valid_labels = labels[~noise_mask]
    if len(valid_labels) > 0:
        unique, counts = np.unique(valid_labels, return_counts=True)
        stats["n_clusters"] = len(unique)
        stats["cluster_sizes"] = {
            f"cluster_{int(cluster)}": int(count)
            for cluster, count in zip(unique, counts)
        }

    return stats


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate synthetic data with noise
    n_samples = 200
    n_features = 10

    # 3 clusters + noise
    cluster1 = np.random.randn(60, n_features) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(60, n_features) + np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(60, n_features) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    noise = np.random.randn(20, n_features) * 5  # Outliers

    X = np.vstack([cluster1, cluster2, cluster3, noise])
    labels = np.array([0] * 60 + [1] * 60 + [2] * 60 + [-1] * 20)
    confidence = np.random.rand(n_samples) * 0.5 + 0.5  # 0.5 to 1.0
    confidence[labels == -1] *= 0.3  # Lower confidence for noise

    print("="*60)
    print("NOISE HANDLING DEMONSTRATION")
    print("="*60)

    # Statistics
    stats = get_noise_statistics(labels, confidence)
    print(f"\nOriginal Statistics:")
    print(f"  Total samples: {stats['n_total']}")
    print(f"  Noise points: {stats['n_noise']} ({stats['noise_fraction']*100:.1f}%)")
    print(f"  Valid points: {stats['n_valid']}")
    print(f"  Number of clusters: {stats['n_clusters']}")
    print(f"  Mean confidence (noise): {stats['mean_confidence_noise']:.3f}")
    print(f"  Mean confidence (valid): {stats['mean_confidence_valid']:.3f}")

    # Strategy 1: Keep
    print(f"\n{'='*60}")
    print("Strategy 1: KEEP NOISE")
    print("="*60)
    X_keep, labels_keep, _ = handle_noise(X, labels, strategy="keep")
    print(f"Unique labels: {sorted(set(labels_keep))}")
    print(f"Sample count: {len(labels_keep)}")

    # Strategy 2: Reassign (nearest cluster)
    print(f"\n{'='*60}")
    print("Strategy 2: REASSIGN (Nearest Cluster)")
    print("="*60)
    X_reassign, labels_reassign, _ = handle_noise(
        X, labels, strategy="reassign", reassign_method="nearest_cluster"
    )
    print(f"Unique labels: {sorted(set(labels_reassign))}")
    print(f"Sample count: {len(labels_reassign)}")
    print(f"Noise points reassigned: {(labels == -1).sum()}")

    # Strategy 3: Reassign (k-NN vote)
    print(f"\n{'='*60}")
    print("Strategy 3: REASSIGN (k-NN Vote)")
    print("="*60)
    X_knn, labels_knn, _ = handle_noise(
        X, labels, strategy="reassign", reassign_method="knn_vote", knn_k=5
    )
    print(f"Unique labels: {sorted(set(labels_knn))}")
    print(f"Sample count: {len(labels_knn)}")

    # Strategy 4: Filter
    print(f"\n{'='*60}")
    print("Strategy 4: FILTER NOISE")
    print("="*60)
    X_filter, labels_filter, indices_filter = handle_noise(
        X, labels, confidence, strategy="filter", min_confidence=0.5
    )
    print(f"Unique labels: {sorted(set(labels_filter))}")
    print(f"Sample count: {len(labels_filter)} (filtered {n_samples - len(labels_filter)})")
    print(f"Indices kept: {len(indices_filter)}")

    # Imbalance detection
    print(f"\n{'='*60}")
    print("IMBALANCE DETECTION")
    print("="*60)
    is_imbalanced, info = detect_imbalance(labels, threshold=0.1)
    print(f"Imbalanced: {is_imbalanced}")
    print(f"Cluster sizes: {info.get('cluster_sizes', {})}")
    print(f"Min/Max ratio: {info.get('ratio', 0):.3f}")

    # Merge small clusters
    print(f"\n{'='*60}")
    print("MERGE SMALL CLUSTERS")
    print("="*60)
    # Create artificial small cluster
    labels_with_small = labels.copy()
    labels_with_small[180:185] = 3  # Small cluster of size 5

    print(f"Before merge: {sorted(set(labels_with_small[labels_with_small >= 0]))}")
    unique, counts = np.unique(labels_with_small[labels_with_small >= 0], return_counts=True)
    print(f"Cluster sizes: {dict(zip(unique, counts))}")

    labels_merged = merge_small_clusters(X, labels_with_small, min_size=10)
    print(f"After merge: {sorted(set(labels_merged[labels_merged >= 0]))}")
    unique, counts = np.unique(labels_merged[labels_merged >= 0], return_counts=True)
    print(f"Cluster sizes: {dict(zip(unique, counts))}")

    print(f"\n{'='*60}")
    print("✓ All strategies demonstrated successfully")
    print("="*60)