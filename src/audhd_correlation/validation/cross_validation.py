"""Cross-validation across sites and cohorts

Tests clustering generalizability and robustness.
"""
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneGroupOut


@dataclass
class CrossValidationResult:
    """Result of cross-validation"""
    fold_ari_scores: List[float]
    fold_ami_scores: List[float]
    mean_ari: float
    std_ari: float
    mean_ami: float
    std_ami: float
    confidence_interval_ari: Tuple[float, float]
    confidence_interval_ami: Tuple[float, float]
    generalization_score: float  # Overall score (0-1)
    interpretation: str


def cross_site_validation(
    X: np.ndarray,
    labels: np.ndarray,
    site_labels: np.ndarray,
    clustering_func: Callable,
) -> CrossValidationResult:
    """
    Leave-one-site-out cross-validation

    Tests if clustering generalizes across different sites.

    Args:
        X: Data matrix
        labels: Original cluster labels
        site_labels: Site assignments for each sample
        clustering_func: Clustering function

    Returns:
        CrossValidationResult
    """
    unique_sites = np.unique(site_labels)

    if len(unique_sites) < 2:
        warnings.warn("Less than 2 sites, cannot perform cross-site validation")
        return _default_cv_result()

    ari_scores = []
    ami_scores = []

    for site in unique_sites:
        # Split into train (other sites) and test (current site)
        test_mask = site_labels == site
        train_mask = ~test_mask

        X_train = X[train_mask]
        X_test = X[test_mask]
        labels_train = labels[train_mask]
        labels_test = labels[test_mask]

        if len(X_train) < 10 or len(X_test) < 5:
            continue

        try:
            # Train on other sites
            labels_train_pred = clustering_func(X_train)

            # Test on held-out site
            labels_test_pred = clustering_func(X_test)

            # Compare test predictions to true labels
            valid_mask = (labels_test >= 0) & (labels_test_pred >= 0)

            if np.sum(valid_mask) < 2:
                continue

            labels_test_valid = labels_test[valid_mask]
            labels_test_pred_valid = labels_test_pred[valid_mask]

            ari = adjusted_rand_score(labels_test_valid, labels_test_pred_valid)
            ami = adjusted_mutual_info_score(labels_test_valid, labels_test_pred_valid)

            ari_scores.append(ari)
            ami_scores.append(ami)

        except Exception as e:
            warnings.warn(f"Cross-site validation failed for site {site}: {e}")
            continue

    if len(ari_scores) == 0:
        return _default_cv_result()

    return _compute_cv_result(ari_scores, ami_scores)


def cross_cohort_validation(
    X_cohorts: Dict[str, np.ndarray],
    labels_cohorts: Dict[str, np.ndarray],
    clustering_func: Callable,
) -> CrossValidationResult:
    """
    Leave-one-cohort-out cross-validation

    Args:
        X_cohorts: Dictionary mapping cohort names to data matrices
        labels_cohorts: Dictionary mapping cohort names to labels
        clustering_func: Clustering function

    Returns:
        CrossValidationResult
    """
    cohort_names = list(X_cohorts.keys())

    if len(cohort_names) < 2:
        warnings.warn("Less than 2 cohorts, cannot perform cross-cohort validation")
        return _default_cv_result()

    ari_scores = []
    ami_scores = []

    for test_cohort in cohort_names:
        # Train on other cohorts
        X_train_list = []
        labels_train_list = []

        for cohort in cohort_names:
            if cohort != test_cohort:
                X_train_list.append(X_cohorts[cohort])
                labels_train_list.append(labels_cohorts[cohort])

        X_train = np.vstack(X_train_list)
        labels_train = np.concatenate(labels_train_list)

        X_test = X_cohorts[test_cohort]
        labels_test = labels_cohorts[test_cohort]

        if len(X_train) < 10 or len(X_test) < 5:
            continue

        try:
            # Train on other cohorts
            labels_train_pred = clustering_func(X_train)

            # Test on held-out cohort
            labels_test_pred = clustering_func(X_test)

            # Compare
            valid_mask = (labels_test >= 0) & (labels_test_pred >= 0)

            if np.sum(valid_mask) < 2:
                continue

            labels_test_valid = labels_test[valid_mask]
            labels_test_pred_valid = labels_test_pred[valid_mask]

            ari = adjusted_rand_score(labels_test_valid, labels_test_pred_valid)
            ami = adjusted_mutual_info_score(labels_test_valid, labels_test_pred_valid)

            ari_scores.append(ari)
            ami_scores.append(ami)

        except Exception as e:
            warnings.warn(f"Cross-cohort validation failed for cohort {test_cohort}: {e}")
            continue

    if len(ari_scores) == 0:
        return _default_cv_result()

    return _compute_cv_result(ari_scores, ami_scores)


def stratified_cross_validation(
    X: np.ndarray,
    labels: np.ndarray,
    clustering_func: Callable,
    n_folds: int = 5,
    random_state: int = 42,
) -> CrossValidationResult:
    """
    Stratified k-fold cross-validation

    Maintains cluster proportions in each fold.

    Args:
        X: Data matrix
        labels: Cluster labels
        clustering_func: Clustering function
        n_folds: Number of folds
        random_state: Random seed

    Returns:
        CrossValidationResult
    """
    # Filter valid labels
    valid_mask = labels >= 0
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    if len(X_valid) < n_folds * 2:
        warnings.warn("Too few samples for stratified cross-validation")
        return _default_cv_result()

    ari_scores = []
    ami_scores = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_valid, labels_valid)):
        X_train = X_valid[train_idx]
        X_test = X_valid[test_idx]
        labels_train = labels_valid[train_idx]
        labels_test = labels_valid[test_idx]

        try:
            # Train
            labels_train_pred = clustering_func(X_train)

            # Test
            labels_test_pred = clustering_func(X_test)

            # Compare
            valid_test_mask = (labels_test >= 0) & (labels_test_pred >= 0)

            if np.sum(valid_test_mask) < 2:
                continue

            labels_test_valid = labels_test[valid_test_mask]
            labels_test_pred_valid = labels_test_pred[valid_test_mask]

            ari = adjusted_rand_score(labels_test_valid, labels_test_pred_valid)
            ami = adjusted_mutual_info_score(labels_test_valid, labels_test_pred_valid)

            ari_scores.append(ari)
            ami_scores.append(ami)

        except Exception as e:
            warnings.warn(f"Fold {fold} failed: {e}")
            continue

    if len(ari_scores) == 0:
        return _default_cv_result()

    return _compute_cv_result(ari_scores, ami_scores)


def temporal_validation(
    X_timepoints: Dict[str, np.ndarray],
    labels_timepoints: Dict[str, np.ndarray],
    clustering_func: Callable,
    temporal_order: Optional[List[str]] = None,
) -> CrossValidationResult:
    """
    Temporal cross-validation

    Train on earlier timepoints, test on later timepoints.

    Args:
        X_timepoints: Dictionary mapping timepoint names to data
        labels_timepoints: Dictionary mapping timepoint names to labels
        clustering_func: Clustering function
        temporal_order: Ordered list of timepoint names (if None, use sorted keys)

    Returns:
        CrossValidationResult
    """
    if temporal_order is None:
        temporal_order = sorted(X_timepoints.keys())

    if len(temporal_order) < 2:
        warnings.warn("Need at least 2 timepoints for temporal validation")
        return _default_cv_result()

    ari_scores = []
    ami_scores = []

    # Train on each prefix, test on next timepoint
    for i in range(1, len(temporal_order)):
        train_timepoints = temporal_order[:i]
        test_timepoint = temporal_order[i]

        # Combine training data
        X_train_list = [X_timepoints[tp] for tp in train_timepoints]
        labels_train_list = [labels_timepoints[tp] for tp in train_timepoints]

        X_train = np.vstack(X_train_list)
        labels_train = np.concatenate(labels_train_list)

        X_test = X_timepoints[test_timepoint]
        labels_test = labels_timepoints[test_timepoint]

        try:
            # Train
            labels_train_pred = clustering_func(X_train)

            # Test
            labels_test_pred = clustering_func(X_test)

            # Compare
            valid_mask = (labels_test >= 0) & (labels_test_pred >= 0)

            if np.sum(valid_mask) < 2:
                continue

            labels_test_valid = labels_test[valid_mask]
            labels_test_pred_valid = labels_test_pred[valid_mask]

            ari = adjusted_rand_score(labels_test_valid, labels_test_pred_valid)
            ami = adjusted_mutual_info_score(labels_test_valid, labels_test_pred_valid)

            ari_scores.append(ari)
            ami_scores.append(ami)

        except Exception as e:
            warnings.warn(f"Temporal validation failed at timepoint {test_timepoint}: {e}")
            continue

    if len(ari_scores) == 0:
        return _default_cv_result()

    return _compute_cv_result(ari_scores, ami_scores)


def batch_effect_validation(
    X: np.ndarray,
    labels: np.ndarray,
    batch_labels: np.ndarray,
    clustering_func: Callable,
) -> Dict[str, float]:
    """
    Test if clustering is robust to batch effects

    Compares within-batch and cross-batch clustering agreement.

    Args:
        X: Data matrix
        labels: Cluster labels
        batch_labels: Batch assignments
        clustering_func: Clustering function

    Returns:
        Dictionary with validation metrics
    """
    unique_batches = np.unique(batch_labels)

    if len(unique_batches) < 2:
        return {'within_batch_ari': 0.0, 'cross_batch_ari': 0.0, 'batch_robustness': 0.0}

    # Within-batch agreement
    within_batch_ari_scores = []

    for batch in unique_batches:
        batch_mask = batch_labels == batch
        X_batch = X[batch_mask]
        labels_batch = labels[batch_mask]

        if len(X_batch) < 10:
            continue

        try:
            labels_pred = clustering_func(X_batch)

            valid_mask = (labels_batch >= 0) & (labels_pred >= 0)

            if np.sum(valid_mask) < 2:
                continue

            labels_valid = labels_batch[valid_mask]
            labels_pred_valid = labels_pred[valid_mask]

            ari = adjusted_rand_score(labels_valid, labels_pred_valid)
            within_batch_ari_scores.append(ari)

        except:
            continue

    # Cross-batch agreement (leave-one-batch-out)
    cross_batch_ari_scores = []

    for test_batch in unique_batches:
        train_mask = batch_labels != test_batch
        test_mask = batch_labels == test_batch

        X_train = X[train_mask]
        X_test = X[test_mask]
        labels_train = labels[train_mask]
        labels_test = labels[test_mask]

        if len(X_train) < 10 or len(X_test) < 5:
            continue

        try:
            labels_train_pred = clustering_func(X_train)
            labels_test_pred = clustering_func(X_test)

            valid_mask = (labels_test >= 0) & (labels_test_pred >= 0)

            if np.sum(valid_mask) < 2:
                continue

            labels_test_valid = labels_test[valid_mask]
            labels_test_pred_valid = labels_test_pred[valid_mask]

            ari = adjusted_rand_score(labels_test_valid, labels_test_pred_valid)
            cross_batch_ari_scores.append(ari)

        except:
            continue

    # Compute metrics
    within_batch_ari = np.mean(within_batch_ari_scores) if len(within_batch_ari_scores) > 0 else 0.0
    cross_batch_ari = np.mean(cross_batch_ari_scores) if len(cross_batch_ari_scores) > 0 else 0.0

    # Robustness: ratio of cross-batch to within-batch
    if within_batch_ari > 0:
        batch_robustness = cross_batch_ari / within_batch_ari
    else:
        batch_robustness = 0.0

    return {
        'within_batch_ari': within_batch_ari,
        'cross_batch_ari': cross_batch_ari,
        'batch_robustness': batch_robustness,
        'n_batches': len(unique_batches),
    }


def _compute_cv_result(
    ari_scores: List[float],
    ami_scores: List[float],
) -> CrossValidationResult:
    """Compute CrossValidationResult from scores"""
    ari_scores = np.array(ari_scores)
    ami_scores = np.array(ami_scores)

    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    mean_ami = np.mean(ami_scores)
    std_ami = np.std(ami_scores)

    # Confidence intervals
    ci_ari = (
        np.percentile(ari_scores, 2.5),
        np.percentile(ari_scores, 97.5)
    )
    ci_ami = (
        np.percentile(ami_scores, 2.5),
        np.percentile(ami_scores, 97.5)
    )

    # Generalization score
    gen_score = 0.5 * mean_ari + 0.5 * mean_ami

    # Interpretation
    if gen_score > 0.7:
        interpretation = "excellent_generalization"
    elif gen_score > 0.5:
        interpretation = "good_generalization"
    elif gen_score > 0.3:
        interpretation = "moderate_generalization"
    else:
        interpretation = "poor_generalization"

    return CrossValidationResult(
        fold_ari_scores=list(ari_scores),
        fold_ami_scores=list(ami_scores),
        mean_ari=mean_ari,
        std_ari=std_ari,
        mean_ami=mean_ami,
        std_ami=std_ami,
        confidence_interval_ari=ci_ari,
        confidence_interval_ami=ci_ami,
        generalization_score=gen_score,
        interpretation=interpretation,
    )


def _default_cv_result() -> CrossValidationResult:
    """Return default cross-validation result"""
    return CrossValidationResult(
        fold_ari_scores=[],
        fold_ami_scores=[],
        mean_ari=0.0,
        std_ari=0.0,
        mean_ami=0.0,
        std_ami=0.0,
        confidence_interval_ari=(0.0, 0.0),
        confidence_interval_ami=(0.0, 0.0),
        generalization_score=0.0,
        interpretation="poor_generalization",
    )