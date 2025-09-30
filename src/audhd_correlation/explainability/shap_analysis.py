"""SHAP analysis for cluster explainability

Computes SHAP values and extracts feature importance and interactions.
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import shap

from .classifier import ClusterClassifierResult


@dataclass
class ShapResult:
    """Result of SHAP analysis"""
    shap_values: np.ndarray  # (n_samples, n_features, n_clusters)
    base_values: np.ndarray  # (n_samples, n_clusters)
    data: np.ndarray  # (n_samples, n_features)
    feature_names: List[str]
    cluster_labels: np.ndarray
    explainer: shap.TreeExplainer

    # Summary statistics
    mean_abs_shap_per_cluster: Dict[int, Dict[str, float]]
    top_features_per_cluster: Dict[int, List[Tuple[str, float]]]


def compute_shap_values(
    classifier_result: ClusterClassifierResult,
    X: np.ndarray,
    check_additivity: bool = False,
    approximate: bool = False,
    feature_perturbation: str = 'interventional',
) -> ShapResult:
    """
    Compute SHAP values for cluster predictions

    Args:
        classifier_result: Trained classifier result
        X: Feature matrix (n_samples, n_features)
        check_additivity: Check SHAP additivity property
        approximate: Use approximate SHAP values (faster)
        feature_perturbation: 'interventional' or 'tree_path_dependent'

    Returns:
        ShapResult with SHAP values and analysis
    """
    # Create SHAP explainer
    explainer = shap.TreeExplainer(
        classifier_result.classifier,
        feature_perturbation=feature_perturbation,
    )

    # Compute SHAP values
    if approximate:
        shap_values = explainer.shap_values(X, check_additivity=False, approximate=True)
    else:
        shap_values = explainer.shap_values(X, check_additivity=check_additivity)

    # Get base values
    base_values = explainer.expected_value

    # Convert to numpy arrays
    if isinstance(shap_values, list):
        # Multi-class case: list of arrays
        shap_values = np.stack(shap_values, axis=-1)
    else:
        # Binary case: add dimension
        shap_values = shap_values[:, :, np.newaxis]

    if isinstance(base_values, (list, np.ndarray)):
        base_values = np.array(base_values)
    else:
        base_values = np.array([base_values])

    # Broadcast base values to match samples
    if base_values.ndim == 1:
        base_values = np.tile(base_values, (X.shape[0], 1))

    # Get unique clusters
    unique_clusters = np.unique(classifier_result.cluster_labels)

    # Compute mean absolute SHAP values per cluster
    mean_abs_shap_per_cluster = {}

    for cluster_idx, cluster_id in enumerate(unique_clusters):
        # Get SHAP values for this cluster output (use index, not label)
        cluster_shap = shap_values[:, :, cluster_idx]

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(cluster_shap), axis=0)

        # Ensure mean_abs_shap is 1D
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.ravel()

        mean_abs_shap_per_cluster[int(cluster_id)] = {
            classifier_result.feature_names[i]: float(mean_abs_shap[i])
            for i in range(len(classifier_result.feature_names))
        }

    # Get top features per cluster
    top_features_per_cluster = {}

    for cluster_id in unique_clusters:
        sorted_features = sorted(
            mean_abs_shap_per_cluster[int(cluster_id)].items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_features_per_cluster[int(cluster_id)] = sorted_features[:20]

    return ShapResult(
        shap_values=shap_values,
        base_values=base_values,
        data=X,
        feature_names=classifier_result.feature_names,
        cluster_labels=classifier_result.cluster_labels,
        explainer=explainer,
        mean_abs_shap_per_cluster=mean_abs_shap_per_cluster,
        top_features_per_cluster=top_features_per_cluster,
    )


def get_top_features_per_cluster(
    shap_result: ShapResult,
    n_features: int = 20,
    cluster_id: Optional[int] = None,
) -> Union[List[Tuple[str, float]], Dict[int, List[Tuple[str, float]]]]:
    """
    Get top features by mean absolute SHAP value

    Args:
        shap_result: ShapResult
        n_features: Number of top features
        cluster_id: Specific cluster (if None, return all clusters)

    Returns:
        Top features for cluster or all clusters
    """
    if cluster_id is not None:
        sorted_features = sorted(
            shap_result.mean_abs_shap_per_cluster[cluster_id].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n_features]
    else:
        result = {}
        for cid in shap_result.mean_abs_shap_per_cluster.keys():
            sorted_features = sorted(
                shap_result.mean_abs_shap_per_cluster[cid].items(),
                key=lambda x: x[1],
                reverse=True
            )
            result[cid] = sorted_features[:n_features]
        return result


def get_feature_interactions(
    shap_result: ShapResult,
    cluster_id: int,
    n_top_features: int = 10,
    sample_size: Optional[int] = 100,
) -> pd.DataFrame:
    """
    Compute feature interaction effects using SHAP interaction values

    Args:
        shap_result: ShapResult
        cluster_id: Target cluster
        n_top_features: Number of top features to analyze
        sample_size: Number of samples to use (None = all)

    Returns:
        DataFrame with pairwise interaction strengths
    """
    # Get top features
    top_features = get_top_features_per_cluster(
        shap_result, n_features=n_top_features, cluster_id=cluster_id
    )
    top_feature_names = [f[0] for f in top_features]
    top_feature_indices = [shap_result.feature_names.index(f) for f in top_feature_names]

    # Sample data if requested
    if sample_size is not None and sample_size < shap_result.data.shape[0]:
        sample_indices = np.random.choice(
            shap_result.data.shape[0], size=sample_size, replace=False
        )
        X_sample = shap_result.data[sample_indices]
    else:
        X_sample = shap_result.data

    # Compute SHAP interaction values (computationally expensive)
    try:
        interaction_values = shap_result.explainer.shap_interaction_values(X_sample)

        # interaction_values: (n_samples, n_features, n_features, n_classes)
        # or (n_samples, n_features, n_features) for binary

        if isinstance(interaction_values, list):
            # Multi-class: select cluster
            interaction_cluster = interaction_values[cluster_id]
        else:
            interaction_cluster = interaction_values

        # Average absolute interaction values across samples
        # interaction_cluster: (n_samples, n_features, n_features)
        mean_abs_interaction = np.mean(np.abs(interaction_cluster), axis=0)

        # Extract submatrix for top features
        interaction_matrix = mean_abs_interaction[np.ix_(top_feature_indices, top_feature_indices)]

        # Create DataFrame
        interaction_df = pd.DataFrame(
            interaction_matrix,
            index=top_feature_names,
            columns=top_feature_names,
        )

        return interaction_df

    except Exception as e:
        warnings.warn(f"Failed to compute SHAP interaction values: {e}")
        # Return empty DataFrame
        return pd.DataFrame(
            np.zeros((len(top_feature_names), len(top_feature_names))),
            index=top_feature_names,
            columns=top_feature_names,
        )


def identify_cluster_prototypes(
    shap_result: ShapResult,
    cluster_id: int,
    n_prototypes: int = 5,
    method: str = 'highest_probability',
) -> Dict[str, np.ndarray]:
    """
    Identify prototype samples for a cluster

    Args:
        shap_result: ShapResult
        cluster_id: Target cluster
        n_prototypes: Number of prototypes to identify
        method: 'highest_probability', 'most_consistent', or 'median'

    Returns:
        Dictionary with prototype information
    """
    # Get samples from this cluster
    cluster_mask = shap_result.cluster_labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]

    if len(cluster_indices) == 0:
        return {
            'indices': np.array([]),
            'shap_values': np.array([]),
            'data': np.array([]),
        }

    # Get SHAP values for this cluster
    cluster_shap = shap_result.shap_values[cluster_mask, :, cluster_id]
    cluster_data = shap_result.data[cluster_mask]

    if method == 'highest_probability':
        # Samples with highest predicted probability for this cluster
        # Use sum of positive SHAP values as proxy
        cluster_scores = np.sum(cluster_shap, axis=1)
        top_indices_within = np.argsort(cluster_scores)[-n_prototypes:][::-1]

    elif method == 'most_consistent':
        # Samples with most consistent SHAP pattern (low variance)
        mean_shap = np.mean(cluster_shap, axis=0)
        shap_distances = np.sum((cluster_shap - mean_shap) ** 2, axis=1)
        top_indices_within = np.argsort(shap_distances)[:n_prototypes]

    elif method == 'median':
        # Samples closest to median in feature space
        median_features = np.median(cluster_data, axis=0)
        distances = np.sum((cluster_data - median_features) ** 2, axis=1)
        top_indices_within = np.argsort(distances)[:n_prototypes]

    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert to original indices
    prototype_indices = cluster_indices[top_indices_within]

    return {
        'indices': prototype_indices,
        'shap_values': shap_result.shap_values[prototype_indices, :, cluster_id],
        'data': shap_result.data[prototype_indices],
        'base_value': shap_result.base_values[prototype_indices, cluster_id],
        'feature_names': shap_result.feature_names,
    }


def get_cluster_signature(
    shap_result: ShapResult,
    cluster_id: int,
    n_features: int = 10,
    threshold: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """
    Get cluster signature: features that characterize this cluster

    Args:
        shap_result: ShapResult
        cluster_id: Target cluster
        n_features: Number of signature features
        threshold: Minimum mean SHAP value to include

    Returns:
        Dictionary with positive and negative signature features
    """
    # Get samples from this cluster
    cluster_mask = shap_result.cluster_labels == cluster_id
    cluster_shap = shap_result.shap_values[cluster_mask, :, cluster_id]

    # Mean SHAP value per feature (signed)
    mean_shap = np.mean(cluster_shap, axis=0)

    # Separate positive and negative contributors
    positive_features = {}
    negative_features = {}

    for i, feature_name in enumerate(shap_result.feature_names):
        shap_val = float(mean_shap[i])

        if shap_val > threshold:
            positive_features[feature_name] = shap_val
        elif shap_val < -threshold:
            negative_features[feature_name] = shap_val

    # Sort and take top features
    positive_sorted = sorted(positive_features.items(), key=lambda x: x[1], reverse=True)
    negative_sorted = sorted(negative_features.items(), key=lambda x: x[1])

    return {
        'positive': dict(positive_sorted[:n_features]),
        'negative': dict(negative_sorted[:n_features]),
    }


def explain_sample(
    shap_result: ShapResult,
    sample_index: int,
    cluster_id: Optional[int] = None,
    n_features: int = 10,
) -> Dict[str, any]:
    """
    Get explanation for individual sample

    Args:
        shap_result: ShapResult
        sample_index: Sample index
        cluster_id: Target cluster (if None, use predicted cluster)
        n_features: Number of top features to return

    Returns:
        Dictionary with sample explanation
    """
    if cluster_id is None:
        # Use cluster with highest SHAP sum
        shap_sums = np.sum(shap_result.shap_values[sample_index], axis=0)
        cluster_id = int(np.argmax(shap_sums))

    # Get SHAP values for this sample and cluster
    sample_shap = shap_result.shap_values[sample_index, :, cluster_id]
    sample_data = shap_result.data[sample_index]
    base_value = shap_result.base_values[sample_index, cluster_id]

    # Sort features by absolute SHAP value
    feature_importance_indices = np.argsort(np.abs(sample_shap))[::-1]
    top_indices = feature_importance_indices[:n_features]

    explanation = {
        'sample_index': sample_index,
        'cluster_id': cluster_id,
        'base_value': float(base_value),
        'prediction': float(base_value + np.sum(sample_shap)),
        'top_features': [
            {
                'feature': shap_result.feature_names[i],
                'value': float(sample_data[i]),
                'shap_value': float(sample_shap[i]),
                'contribution': 'positive' if sample_shap[i] > 0 else 'negative',
            }
            for i in top_indices
        ],
    }

    return explanation


def compare_clusters(
    shap_result: ShapResult,
    cluster_a: int,
    cluster_b: int,
    n_features: int = 20,
) -> pd.DataFrame:
    """
    Compare discriminative features between two clusters

    Args:
        shap_result: ShapResult
        cluster_a: First cluster
        cluster_b: Second cluster
        n_features: Number of top features

    Returns:
        DataFrame with feature comparisons
    """
    # Get mean SHAP values for each cluster
    mask_a = shap_result.cluster_labels == cluster_a
    mask_b = shap_result.cluster_labels == cluster_b

    mean_shap_a = np.mean(shap_result.shap_values[mask_a, :, cluster_a], axis=0)
    mean_shap_b = np.mean(shap_result.shap_values[mask_b, :, cluster_b], axis=0)

    # Difference in mean SHAP values
    shap_diff = mean_shap_a - mean_shap_b

    # Sort by absolute difference
    top_indices = np.argsort(np.abs(shap_diff))[-n_features:][::-1]

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'feature': [shap_result.feature_names[i] for i in top_indices],
        f'mean_shap_cluster_{cluster_a}': [mean_shap_a[i] for i in top_indices],
        f'mean_shap_cluster_{cluster_b}': [mean_shap_b[i] for i in top_indices],
        'shap_difference': [shap_diff[i] for i in top_indices],
        'abs_difference': [abs(shap_diff[i]) for i in top_indices],
    })

    return comparison_df