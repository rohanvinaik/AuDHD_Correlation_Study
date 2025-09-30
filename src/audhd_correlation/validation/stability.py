"""Stability analysis via bootstrap and subsampling

Tests clustering robustness through resampling and perturbation.
"""
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score


@dataclass
class StabilityResult:
    """Container for stability analysis results

    Uses standardized naming convention: {metric}_{statistic}
    See docs/validation_metric_names.md for full specification.
    """
    # Raw scores from all bootstraps
    ari_scores: np.ndarray
    ami_scores: np.ndarray
    fmi_scores: np.ndarray  # Renamed from fowlkes_mallows_scores
    jaccard_scores: np.ndarray

    # Summary statistics (STANDARD NAMING)
    ari_mean: float  # Renamed from mean_ari
    ari_std: float   # Renamed from std_ari
    ari_ci: Tuple[float, float]  # Renamed from confidence_interval_ari

    ami_mean: float  # Renamed from mean_ami
    ami_std: float   # Renamed from std_ami
    ami_ci: Tuple[float, float]  # Renamed from confidence_interval_ami

    fmi_mean: float
    fmi_std: float

    jaccard_mean: float
    jaccard_std: float

    # Overall assessment
    stability_score: float  # 0-1, higher = more stable
    interpretation: str  # "excellent", "good", "moderate", "poor"

    def to_dict(self) -> Dict:
        """
        Convert to dictionary with standard field names

        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            # Summary statistics
            "ari_mean": float(self.ari_mean),
            "ari_std": float(self.ari_std),
            "ari_ci": [float(self.ari_ci[0]), float(self.ari_ci[1])],

            "ami_mean": float(self.ami_mean),
            "ami_std": float(self.ami_std),
            "ami_ci": [float(self.ami_ci[0]), float(self.ami_ci[1])],

            "fmi_mean": float(self.fmi_mean),
            "fmi_std": float(self.fmi_std),

            "jaccard_mean": float(self.jaccard_mean),
            "jaccard_std": float(self.jaccard_std),

            # Overall
            "stability_score": float(self.stability_score),
            "interpretation": self.interpretation,

            # Raw scores (for inspection)
            "ari_scores": self.ari_scores.tolist(),
            "ami_scores": self.ami_scores.tolist(),
            "fmi_scores": self.fmi_scores.tolist(),
            "jaccard_scores": self.jaccard_scores.tolist(),
        }


def bootstrap_stability(
    X: np.ndarray,
    labels: np.ndarray,
    clustering_func: Callable,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.8,
    random_state: int = 42,
) -> StabilityResult:
    """
    Assess clustering stability via bootstrap resampling

    Args:
        X: Data matrix
        labels: Original cluster labels
        clustering_func: Function that takes X and returns labels
        n_bootstrap: Number of bootstrap iterations
        sample_fraction: Fraction of samples to use in each bootstrap
        random_state: Random seed

    Returns:
        StabilityResult object
    """
    np.random.seed(random_state)

    n_samples = len(X)
    sample_size = int(n_samples * sample_fraction)

    ari_scores = []
    ami_scores = []
    fm_scores = []
    jaccard_scores = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, sample_size, replace=True)
        X_boot = X[indices]
        labels_boot_true = labels[indices]

        try:
            # Cluster bootstrap sample
            labels_boot_pred = clustering_func(X_boot)

            # Filter out noise points if present
            valid_mask = (labels_boot_true >= 0) & (labels_boot_pred >= 0)

            if np.sum(valid_mask) < 2:
                continue

            labels_true_valid = labels_boot_true[valid_mask]
            labels_pred_valid = labels_boot_pred[valid_mask]

            # Compute agreement metrics
            ari = adjusted_rand_score(labels_true_valid, labels_pred_valid)
            ami = adjusted_mutual_info_score(labels_true_valid, labels_pred_valid)
            fm = fowlkes_mallows_score(labels_true_valid, labels_pred_valid)

            # Jaccard index (proportion of sample pairs with same cluster assignment)
            jaccard = compute_jaccard_index(labels_true_valid, labels_pred_valid)

            ari_scores.append(ari)
            ami_scores.append(ami)
            fm_scores.append(fm)
            jaccard_scores.append(jaccard)

        except Exception as e:
            warnings.warn(f"Bootstrap iteration {i} failed: {e}")
            continue

    if len(ari_scores) == 0:
        return _default_stability_result()

    # Convert to arrays
    ari_scores = np.array(ari_scores)
    ami_scores = np.array(ami_scores)
    fmi_scores = np.array(fm_scores)
    jaccard_scores = np.array(jaccard_scores)

    # Compute statistics (STANDARD NAMING)
    ari_mean = np.mean(ari_scores)
    ari_std = np.std(ari_scores)
    ami_mean = np.mean(ami_scores)
    ami_std = np.std(ami_scores)
    fmi_mean = np.mean(fmi_scores)
    fmi_std = np.std(fmi_scores)
    jaccard_mean = np.mean(jaccard_scores)
    jaccard_std = np.std(jaccard_scores)

    # Confidence intervals (95%)
    ari_ci = (
        np.percentile(ari_scores, 2.5),
        np.percentile(ari_scores, 97.5)
    )
    ami_ci = (
        np.percentile(ami_scores, 2.5),
        np.percentile(ami_scores, 97.5)
    )

    # Overall stability score (weighted average)
    stability = 0.5 * ari_mean + 0.5 * ami_mean

    # Interpretation
    if stability > 0.8:
        interpretation = "highly_stable"
    elif stability > 0.6:
        interpretation = "stable"
    elif stability > 0.4:
        interpretation = "moderately_stable"
    else:
        interpretation = "unstable"

    return StabilityResult(
        ari_scores=ari_scores,
        ami_scores=ami_scores,
        fmi_scores=fmi_scores,  # RENAMED
        jaccard_scores=jaccard_scores,
        ari_mean=ari_mean,  # RENAMED
        ari_std=ari_std,    # RENAMED
        ari_ci=ari_ci,      # RENAMED
        ami_mean=ami_mean,  # RENAMED
        ami_std=ami_std,    # RENAMED
        ami_ci=ami_ci,      # RENAMED
        fmi_mean=fmi_mean,  # NEW
        fmi_std=fmi_std,    # NEW
        jaccard_mean=jaccard_mean,  # NEW
        jaccard_std=jaccard_std,    # NEW
        stability_score=stability,
        interpretation=interpretation,
    )


def subsampling_stability(
    X: np.ndarray,
    labels: np.ndarray,
    clustering_func: Callable,
    subsample_fractions: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
    n_iterations: int = 20,
    random_state: int = 42,
) -> Dict[str, StabilityResult]:
    """
    Assess stability across different subsample sizes

    Args:
        X: Data matrix
        labels: Original cluster labels
        clustering_func: Clustering function
        subsample_fractions: List of fractions to test
        n_iterations: Iterations per fraction
        random_state: Random seed

    Returns:
        Dictionary mapping fraction to StabilityResult
    """
    results = {}

    for fraction in subsample_fractions:
        result = bootstrap_stability(
            X, labels, clustering_func,
            n_bootstrap=n_iterations,
            sample_fraction=fraction,
            random_state=random_state,
        )

        results[f"fraction_{fraction:.1f}"] = result

    return results


def noise_stability(
    X: np.ndarray,
    labels: np.ndarray,
    clustering_func: Callable,
    noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2],
    n_iterations: int = 20,
    random_state: int = 42,
) -> Dict[str, StabilityResult]:
    """
    Assess stability to additive noise

    Args:
        X: Data matrix
        labels: Original cluster labels
        clustering_func: Clustering function
        noise_levels: List of noise levels (std as fraction of data std)
        n_iterations: Iterations per noise level
        random_state: Random seed

    Returns:
        Dictionary mapping noise level to StabilityResult
    """
    np.random.seed(random_state)

    data_std = np.std(X, axis=0)
    results = {}

    for noise_level in noise_levels:
        ari_scores = []
        ami_scores = []
        fm_scores = []
        jaccard_scores = []

        for i in range(n_iterations):
            # Add noise
            noise = np.random.randn(*X.shape) * (data_std * noise_level)
            X_noisy = X + noise

            try:
                # Cluster noisy data
                labels_noisy = clustering_func(X_noisy)

                # Filter valid labels
                valid_mask = (labels >= 0) & (labels_noisy >= 0)

                if np.sum(valid_mask) < 2:
                    continue

                labels_valid = labels[valid_mask]
                labels_noisy_valid = labels_noisy[valid_mask]

                # Compute metrics
                ari = adjusted_rand_score(labels_valid, labels_noisy_valid)
                ami = adjusted_mutual_info_score(labels_valid, labels_noisy_valid)
                fm = fowlkes_mallows_score(labels_valid, labels_noisy_valid)
                jaccard = compute_jaccard_index(labels_valid, labels_noisy_valid)

                ari_scores.append(ari)
                ami_scores.append(ami)
                fm_scores.append(fm)
                jaccard_scores.append(jaccard)

            except Exception as e:
                warnings.warn(f"Noise iteration {i} at level {noise_level} failed: {e}")
                continue

        if len(ari_scores) == 0:
            results[f"noise_{noise_level:.2f}"] = _default_stability_result()
            continue

        # Create StabilityResult
        ari_scores = np.array(ari_scores)
        ami_scores = np.array(ami_scores)
        fm_scores = np.array(fm_scores)
        jaccard_scores = np.array(jaccard_scores)

        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        mean_ami = np.mean(ami_scores)
        std_ami = np.std(ami_scores)

        ci_ari = (np.percentile(ari_scores, 2.5), np.percentile(ari_scores, 97.5))
        ci_ami = (np.percentile(ami_scores, 2.5), np.percentile(ami_scores, 97.5))

        stability = 0.5 * mean_ari + 0.5 * mean_ami

        if stability > 0.8:
            interpretation = "highly_stable"
        elif stability > 0.6:
            interpretation = "stable"
        elif stability > 0.4:
            interpretation = "moderately_stable"
        else:
            interpretation = "unstable"

        results[f"noise_{noise_level:.2f}"] = StabilityResult(
            ari_scores=ari_scores,
            ami_scores=ami_scores,
            fmi_scores=fm_scores,
            jaccard_scores=jaccard_scores,
            ari_mean=mean_ari,
            ari_std=std_ari,
            ami_mean=mean_ami,
            ami_std=std_ami,
            ari_ci=ci_ari,
            ami_ci=ci_ami,
            fmi_mean=np.mean(fm_scores),
            fmi_std=np.std(fm_scores),
            jaccard_mean=np.mean(jaccard_scores),
            jaccard_std=np.std(jaccard_scores),
            stability_score=stability,
            interpretation=interpretation,
        )

    return results


def feature_stability(
    X: np.ndarray,
    labels: np.ndarray,
    clustering_func: Callable,
    feature_fractions: List[float] = [0.5, 0.7, 0.9],
    n_iterations: int = 20,
    random_state: int = 42,
) -> Dict[str, StabilityResult]:
    """
    Assess stability when using subset of features

    Args:
        X: Data matrix
        labels: Original cluster labels
        clustering_func: Clustering function
        feature_fractions: Fractions of features to use
        n_iterations: Iterations per fraction
        random_state: Random seed

    Returns:
        Dictionary mapping feature fraction to StabilityResult
    """
    np.random.seed(random_state)

    n_features = X.shape[1]
    results = {}

    for fraction in feature_fractions:
        n_select = int(n_features * fraction)
        ari_scores = []
        ami_scores = []
        fm_scores = []
        jaccard_scores = []

        for i in range(n_iterations):
            # Select random features
            feature_indices = np.random.choice(n_features, n_select, replace=False)
            X_subset = X[:, feature_indices]

            try:
                # Cluster with subset
                labels_subset = clustering_func(X_subset)

                # Filter valid labels
                valid_mask = (labels >= 0) & (labels_subset >= 0)

                if np.sum(valid_mask) < 2:
                    continue

                labels_valid = labels[valid_mask]
                labels_subset_valid = labels_subset[valid_mask]

                # Compute metrics
                ari = adjusted_rand_score(labels_valid, labels_subset_valid)
                ami = adjusted_mutual_info_score(labels_valid, labels_subset_valid)
                fm = fowlkes_mallows_score(labels_valid, labels_subset_valid)
                jaccard = compute_jaccard_index(labels_valid, labels_subset_valid)

                ari_scores.append(ari)
                ami_scores.append(ami)
                fm_scores.append(fm)
                jaccard_scores.append(jaccard)

            except Exception as e:
                warnings.warn(f"Feature subset iteration {i} at fraction {fraction} failed: {e}")
                continue

        if len(ari_scores) == 0:
            results[f"features_{fraction:.1f}"] = _default_stability_result()
            continue

        # Create StabilityResult
        ari_scores = np.array(ari_scores)
        ami_scores = np.array(ami_scores)
        fm_scores = np.array(fm_scores)
        jaccard_scores = np.array(jaccard_scores)

        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        mean_ami = np.mean(ami_scores)
        std_ami = np.std(ami_scores)

        ci_ari = (np.percentile(ari_scores, 2.5), np.percentile(ari_scores, 97.5))
        ci_ami = (np.percentile(ami_scores, 2.5), np.percentile(ami_scores, 97.5))

        stability = 0.5 * mean_ari + 0.5 * mean_ami

        if stability > 0.8:
            interpretation = "highly_stable"
        elif stability > 0.6:
            interpretation = "stable"
        elif stability > 0.4:
            interpretation = "moderately_stable"
        else:
            interpretation = "unstable"

        results[f"features_{fraction:.1f}"] = StabilityResult(
            ari_scores=ari_scores,
            ami_scores=ami_scores,
            fmi_scores=fm_scores,
            jaccard_scores=jaccard_scores,
            ari_mean=mean_ari,
            ari_std=std_ari,
            ami_mean=mean_ami,
            ami_std=std_ami,
            ari_ci=ci_ari,
            ami_ci=ci_ami,
            fmi_mean=np.mean(fm_scores),
            fmi_std=np.std(fm_scores),
            jaccard_mean=np.mean(jaccard_scores),
            jaccard_std=np.std(jaccard_scores),
            stability_score=stability,
            interpretation=interpretation,
        )

    return results


def compute_jaccard_index(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """
    Compute Jaccard index for cluster agreement

    Fraction of sample pairs with same cluster relationship in both labelings.

    Args:
        labels1: First labeling
        labels2: Second labeling

    Returns:
        Jaccard index (0-1)
    """
    n = len(labels1)

    if n < 2:
        return 0.0

    # Count agreements and disagreements
    same_in_both = 0
    same_in_either = 0

    # Sample pairs to avoid O(n^2) computation for large n
    if n > 1000:
        n_pairs = 10000
        indices = np.random.choice(n, (n_pairs, 2), replace=True)
    else:
        # All pairs
        indices = np.array([(i, j) for i in range(n) for j in range(i+1, n)])

    for i, j in indices:
        same1 = labels1[i] == labels1[j]
        same2 = labels2[i] == labels2[j]

        if same1 and same2:
            same_in_both += 1

        if same1 or same2:
            same_in_either += 1

    if same_in_either == 0:
        return 0.0

    return same_in_both / same_in_either


def permutation_test_stability(
    X: np.ndarray,
    labels: np.ndarray,
    clustering_func: Callable,
    n_permutations: int = 100,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Test if clustering is significantly better than random

    Permutes labels and compares to original clustering.

    Args:
        X: Data matrix
        labels: Cluster labels
        clustering_func: Clustering function
        n_permutations: Number of permutations
        random_state: Random seed

    Returns:
        Dictionary with test results
    """
    np.random.seed(random_state)

    # Recompute clustering on original data
    try:
        labels_recomputed = clustering_func(X)
        valid_mask = (labels >= 0) & (labels_recomputed >= 0)

        if np.sum(valid_mask) < 2:
            return {'p_value': 1.0, 'observed_ari': 0.0, 'random_ari_mean': 0.0}

        labels_valid = labels[valid_mask]
        labels_recomputed_valid = labels_recomputed[valid_mask]

        observed_ari = adjusted_rand_score(labels_valid, labels_recomputed_valid)

    except:
        return {'p_value': 1.0, 'observed_ari': 0.0, 'random_ari_mean': 0.0}

    # Permutation test
    random_ari_scores = []

    for i in range(n_permutations):
        # Permute labels
        labels_permuted = np.random.permutation(labels)

        try:
            # Recompute clustering
            labels_pred = clustering_func(X)

            valid_mask = (labels_permuted >= 0) & (labels_pred >= 0)

            if np.sum(valid_mask) < 2:
                continue

            labels_perm_valid = labels_permuted[valid_mask]
            labels_pred_valid = labels_pred[valid_mask]

            random_ari = adjusted_rand_score(labels_perm_valid, labels_pred_valid)
            random_ari_scores.append(random_ari)

        except:
            continue

    if len(random_ari_scores) == 0:
        return {'p_value': 1.0, 'observed_ari': observed_ari, 'random_ari_mean': 0.0}

    random_ari_scores = np.array(random_ari_scores)

    # P-value: proportion of random scores >= observed
    p_value = np.mean(random_ari_scores >= observed_ari)

    return {
        'p_value': p_value,
        'observed_ari': observed_ari,
        'random_ari_mean': np.mean(random_ari_scores),
        'random_ari_std': np.std(random_ari_scores),
        'random_ari_95_percentile': np.percentile(random_ari_scores, 95),
    }


def _default_stability_result() -> StabilityResult:
    """Return default stability result"""
    return StabilityResult(
        ari_scores=np.array([]),
        ami_scores=np.array([]),
        fmi_scores=np.array([]),
        jaccard_scores=np.array([]),
        ari_mean=0.0,
        ari_std=0.0,
        ami_mean=0.0,
        ami_std=0.0,
        ari_ci=(0.0, 0.0),
        ami_ci=(0.0, 0.0),
        fmi_mean=0.0,
        fmi_std=0.0,
        jaccard_mean=0.0,
        jaccard_std=0.0,
        stability_score=0.0,
        interpretation="poor",
    )