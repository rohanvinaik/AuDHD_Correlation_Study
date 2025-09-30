"""Internal validation metrics for clustering results

Comprehensive metrics to assess clustering quality, including handling
of imbalanced cluster sizes and robustness to outliers.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)


@dataclass
class InternalValidationMetrics:
    """Container for internal validation metrics"""
    silhouette: float
    silhouette_per_cluster: Dict[int, float]
    davies_bouldin: float
    calinski_harabasz: float
    dunn_index: float
    generalized_dunn_index: float
    within_cluster_variance: float
    between_cluster_variance: float
    variance_ratio: float
    compactness: float
    separation: float
    # Robust variants
    robust_silhouette: Optional[float] = None
    median_silhouette: Optional[float] = None
    # Imbalanced handling
    balanced_accuracy: Optional[float] = None
    # Overall quality
    overall_quality: Optional[float] = None


def compute_internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean',
    handle_outliers: bool = True,
    outlier_percentile: float = 5.0,
) -> InternalValidationMetrics:
    """
    Compute comprehensive internal validation metrics

    Args:
        X: Data matrix (n_samples, n_features)
        labels: Cluster labels
        metric: Distance metric
        handle_outliers: Whether to compute robust metrics
        outlier_percentile: Percentile threshold for outlier detection

    Returns:
        InternalValidationMetrics object
    """
    # Filter out noise points (-1 labels)
    valid_mask = labels >= 0
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    if len(np.unique(labels_valid)) < 2:
        warnings.warn("Less than 2 clusters, returning default metrics")
        return _default_metrics()

    # Standard sklearn metrics
    try:
        sil = silhouette_score(X_valid, labels_valid, metric=metric)
        sil_samples = silhouette_samples(X_valid, labels_valid, metric=metric)
    except:
        sil = 0.0
        sil_samples = np.zeros(len(X_valid))

    try:
        db = davies_bouldin_score(X_valid, labels_valid)
    except:
        db = np.inf

    try:
        ch = calinski_harabasz_score(X_valid, labels_valid)
    except:
        ch = 0.0

    # Per-cluster silhouette
    sil_per_cluster = {}
    for label in np.unique(labels_valid):
        cluster_mask = labels_valid == label
        sil_per_cluster[int(label)] = np.mean(sil_samples[cluster_mask])

    # Dunn indices
    dunn = compute_dunn_index(X_valid, labels_valid, metric=metric)
    gen_dunn = compute_generalized_dunn_index(X_valid, labels_valid, metric=metric)

    # Variance metrics
    within_var, between_var = compute_variance_metrics(X_valid, labels_valid)
    variance_ratio = between_var / (within_var + 1e-10)

    # Compactness and separation
    compactness = compute_compactness(X_valid, labels_valid, metric=metric)
    separation = compute_separation(X_valid, labels_valid, metric=metric)

    # Robust metrics (if requested)
    robust_sil = None
    median_sil = None
    if handle_outliers:
        robust_sil = compute_robust_silhouette(
            X_valid, labels_valid, sil_samples, outlier_percentile
        )
        median_sil = np.median(sil_samples)

    # Overall quality score (weighted combination)
    overall_quality = compute_overall_quality(
        sil, db, ch, dunn, variance_ratio
    )

    return InternalValidationMetrics(
        silhouette=sil,
        silhouette_per_cluster=sil_per_cluster,
        davies_bouldin=db,
        calinski_harabasz=ch,
        dunn_index=dunn,
        generalized_dunn_index=gen_dunn,
        within_cluster_variance=within_var,
        between_cluster_variance=between_var,
        variance_ratio=variance_ratio,
        compactness=compactness,
        separation=separation,
        robust_silhouette=robust_sil,
        median_silhouette=median_sil,
        overall_quality=overall_quality,
    )


def compute_dunn_index(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """
    Compute Dunn index (ratio of minimum separation to maximum diameter)

    Higher is better.

    Args:
        X: Data matrix
        labels: Cluster labels
        metric: Distance metric

    Returns:
        Dunn index
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    # Compute pairwise distances
    distances = squareform(pdist(X, metric=metric))

    # Minimum inter-cluster distance
    min_inter = np.inf
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            mask1 = labels == label1
            mask2 = labels == label2

            inter_dists = distances[np.ix_(mask1, mask2)]
            min_inter = min(min_inter, np.min(inter_dists))

    # Maximum intra-cluster distance (diameter)
    max_intra = 0.0
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) < 2:
            continue

        intra_dists = distances[np.ix_(mask, mask)]
        max_intra = max(max_intra, np.max(intra_dists))

    if max_intra == 0:
        return 0.0

    return min_inter / max_intra


def compute_generalized_dunn_index(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean',
    p: int = 2,
) -> float:
    """
    Compute generalized Dunn index using cluster centroids

    Args:
        X: Data matrix
        labels: Cluster labels
        metric: Distance metric
        p: Norm order

    Returns:
        Generalized Dunn index
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    # Compute centroids
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroids.append(np.mean(X[mask], axis=0))
    centroids = np.array(centroids)

    # Minimum centroid distance
    centroid_dists = pdist(centroids, metric=metric)
    min_centroid_dist = np.min(centroid_dists) if len(centroid_dists) > 0 else 0

    # Maximum cluster radius
    max_radius = 0.0
    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_points = X[mask]

        if len(cluster_points) < 2:
            continue

        # Distance from centroid
        dists = np.linalg.norm(cluster_points - centroids[i], ord=p, axis=1)
        max_radius = max(max_radius, np.max(dists))

    if max_radius == 0:
        return 0.0

    return min_centroid_dist / (2 * max_radius)


def compute_variance_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute within-cluster and between-cluster variance

    Args:
        X: Data matrix
        labels: Cluster labels

    Returns:
        Tuple of (within_variance, between_variance)
    """
    unique_labels = np.unique(labels)
    n_samples = len(X)

    # Overall mean
    overall_mean = np.mean(X, axis=0)

    # Within-cluster variance
    within_var = 0.0
    for label in unique_labels:
        mask = labels == label
        cluster_points = X[mask]

        if len(cluster_points) < 2:
            continue

        cluster_mean = np.mean(cluster_points, axis=0)
        within_var += np.sum((cluster_points - cluster_mean) ** 2)

    within_var /= n_samples

    # Between-cluster variance
    between_var = 0.0
    for label in unique_labels:
        mask = labels == label
        n_cluster = np.sum(mask)
        cluster_mean = np.mean(X[mask], axis=0)

        between_var += n_cluster * np.sum((cluster_mean - overall_mean) ** 2)

    between_var /= n_samples

    return within_var, between_var


def compute_compactness(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean',
) -> float:
    """
    Compute average compactness (average within-cluster distance)

    Lower is better.

    Args:
        X: Data matrix
        labels: Cluster labels
        metric: Distance metric

    Returns:
        Compactness score
    """
    unique_labels = np.unique(labels)
    compactness = 0.0
    n_clusters = 0

    for label in unique_labels:
        mask = labels == label
        cluster_points = X[mask]

        if len(cluster_points) < 2:
            continue

        # Average pairwise distance within cluster
        dists = pdist(cluster_points, metric=metric)
        compactness += np.mean(dists)
        n_clusters += 1

    if n_clusters == 0:
        return 0.0

    return compactness / n_clusters


def compute_separation(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean',
) -> float:
    """
    Compute average separation (average between-cluster centroid distance)

    Higher is better.

    Args:
        X: Data matrix
        labels: Cluster labels
        metric: Distance metric

    Returns:
        Separation score
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    # Compute centroids
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroids.append(np.mean(X[mask], axis=0))
    centroids = np.array(centroids)

    # Average pairwise centroid distance
    centroid_dists = pdist(centroids, metric=metric)

    return np.mean(centroid_dists)


def compute_robust_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    silhouette_samples: np.ndarray,
    outlier_percentile: float = 5.0,
) -> float:
    """
    Compute robust silhouette score by trimming outliers

    Args:
        X: Data matrix
        labels: Cluster labels
        silhouette_samples: Pre-computed silhouette samples
        outlier_percentile: Percentile threshold for outliers

    Returns:
        Robust silhouette score
    """
    # Remove bottom percentile (outliers with poor silhouette)
    threshold = np.percentile(silhouette_samples, outlier_percentile)
    robust_samples = silhouette_samples[silhouette_samples >= threshold]

    if len(robust_samples) == 0:
        return 0.0

    return np.mean(robust_samples)


def compute_overall_quality(
    silhouette: float,
    davies_bouldin: float,
    calinski_harabasz: float,
    dunn_index: float,
    variance_ratio: float,
) -> float:
    """
    Compute overall quality score (0-1, higher is better)

    Weighted combination of normalized metrics.

    Args:
        silhouette: Silhouette score (-1 to 1)
        davies_bouldin: Davies-Bouldin index (lower is better)
        calinski_harabasz: Calinski-Harabasz score (higher is better)
        dunn_index: Dunn index (higher is better)
        variance_ratio: Between/within variance ratio (higher is better)

    Returns:
        Overall quality score (0-1)
    """
    # Normalize silhouette from [-1, 1] to [0, 1]
    sil_norm = (silhouette + 1) / 2

    # Normalize Davies-Bouldin (inverse, since lower is better)
    # Typical range is [0, inf], we use 1/(1+x) to map to [0, 1]
    db_norm = 1 / (1 + davies_bouldin)

    # Normalize Calinski-Harabasz (log scale)
    # Typical range is [0, inf]
    ch_norm = np.log(calinski_harabasz + 1) / 10  # Rough normalization
    ch_norm = min(1.0, ch_norm)

    # Dunn index is already roughly in [0, 1] range
    dunn_norm = min(1.0, dunn_index)

    # Variance ratio (log scale)
    var_ratio_norm = np.log(variance_ratio + 1) / 5
    var_ratio_norm = min(1.0, var_ratio_norm)

    # Weighted average
    weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Emphasize silhouette
    scores = [sil_norm, db_norm, ch_norm, dunn_norm, var_ratio_norm]

    overall = np.average(scores, weights=weights)

    return float(overall)


def _default_metrics() -> InternalValidationMetrics:
    """Return default metrics when clustering is invalid"""
    return InternalValidationMetrics(
        silhouette=0.0,
        silhouette_per_cluster={},
        davies_bouldin=np.inf,
        calinski_harabasz=0.0,
        dunn_index=0.0,
        generalized_dunn_index=0.0,
        within_cluster_variance=0.0,
        between_cluster_variance=0.0,
        variance_ratio=0.0,
        compactness=0.0,
        separation=0.0,
        overall_quality=0.0,
    )


def compute_stability_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean',
) -> Dict[str, float]:
    """
    Compute stability-related metrics

    Args:
        X: Data matrix
        labels: Cluster labels
        metric: Distance metric

    Returns:
        Dictionary of stability metrics
    """
    unique_labels = np.unique(labels[labels >= 0])

    if len(unique_labels) < 2:
        return {'coefficient_of_variation': 0.0, 'imbalance_ratio': 0.0}

    # Cluster size variation
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    cv = np.std(cluster_sizes) / (np.mean(cluster_sizes) + 1e-10)

    # Imbalance ratio (max size / min size)
    imbalance = np.max(cluster_sizes) / (np.min(cluster_sizes) + 1e-10)

    # Density variation
    densities = []
    for label in unique_labels:
        mask = labels == label
        cluster_points = X[mask]

        if len(cluster_points) < 2:
            continue

        dists = pdist(cluster_points, metric=metric)
        densities.append(1.0 / (np.mean(dists) + 1e-10))

    density_cv = np.std(densities) / (np.mean(densities) + 1e-10) if len(densities) > 1 else 0.0

    return {
        'coefficient_of_variation': cv,
        'imbalance_ratio': imbalance,
        'density_variation': density_cv,
        'min_cluster_size': np.min(cluster_sizes),
        'max_cluster_size': np.max(cluster_sizes),
        'median_cluster_size': np.median(cluster_sizes),
    }


def compute_outlier_robustness(
    X: np.ndarray,
    labels: np.ndarray,
    contamination: float = 0.1,
    n_iterations: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Test robustness to outliers

    Adds synthetic outliers and measures metric degradation.

    Args:
        X: Data matrix
        labels: Cluster labels
        contamination: Fraction of outliers to add
        n_iterations: Number of test iterations
        random_state: Random seed

    Returns:
        Dictionary with robustness metrics
    """
    np.random.seed(random_state)

    # Baseline metrics
    baseline = compute_internal_metrics(X, labels, handle_outliers=False)

    # Test with outliers
    sil_scores = []
    db_scores = []

    for i in range(n_iterations):
        # Add random outliers
        n_outliers = int(len(X) * contamination)

        # Generate outliers far from data
        data_min = X.min(axis=0)
        data_max = X.max(axis=0)
        data_range = data_max - data_min

        outliers = np.random.uniform(
            data_min - 2 * data_range,
            data_max + 2 * data_range,
            size=(n_outliers, X.shape[1])
        )

        # Combine with original data
        X_contaminated = np.vstack([X, outliers])
        labels_contaminated = np.hstack([labels, np.full(n_outliers, -1)])

        # Compute metrics
        try:
            metrics = compute_internal_metrics(
                X_contaminated, labels_contaminated, handle_outliers=False
            )
            sil_scores.append(metrics.silhouette)
            db_scores.append(metrics.davies_bouldin)
        except:
            continue

    # Compute degradation
    if sil_scores:
        sil_degradation = baseline.silhouette - np.mean(sil_scores)
        db_degradation = np.mean(db_scores) - baseline.davies_bouldin

        return {
            'silhouette_degradation': sil_degradation,
            'davies_bouldin_degradation': db_degradation,
            'silhouette_robustness': 1.0 - abs(sil_degradation) / (abs(baseline.silhouette) + 1e-10),
            'baseline_silhouette': baseline.silhouette,
            'contaminated_silhouette_mean': np.mean(sil_scores),
            'contaminated_silhouette_std': np.std(sil_scores),
        }
    else:
        return {
            'silhouette_degradation': 0.0,
            'davies_bouldin_degradation': 0.0,
            'silhouette_robustness': 0.0,
        }


def compute_balanced_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean',
) -> Dict[str, float]:
    """
    Compute metrics adjusted for imbalanced cluster sizes

    Args:
        X: Data matrix
        labels: Cluster labels
        metric: Distance metric

    Returns:
        Dictionary of balanced metrics
    """
    valid_mask = labels >= 0
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    unique_labels = np.unique(labels_valid)

    if len(unique_labels) < 2:
        return {'balanced_silhouette': 0.0, 'weighted_compactness': 0.0}

    # Balanced silhouette (equal weight per cluster)
    sil_samples = silhouette_samples(X_valid, labels_valid, metric=metric)

    cluster_silhouettes = []
    for label in unique_labels:
        mask = labels_valid == label
        cluster_silhouettes.append(np.mean(sil_samples[mask]))

    balanced_sil = np.mean(cluster_silhouettes)

    # Weighted compactness (weight by cluster size)
    cluster_sizes = [np.sum(labels_valid == label) for label in unique_labels]
    total_size = np.sum(cluster_sizes)

    weighted_compact = 0.0
    for label, size in zip(unique_labels, cluster_sizes):
        mask = labels_valid == label
        cluster_points = X_valid[mask]

        if len(cluster_points) < 2:
            continue

        dists = pdist(cluster_points, metric=metric)
        weighted_compact += (size / total_size) * np.mean(dists)

    return {
        'balanced_silhouette': balanced_sil,
        'weighted_compactness': weighted_compact,
        'silhouette_std_across_clusters': np.std(cluster_silhouettes),
    }