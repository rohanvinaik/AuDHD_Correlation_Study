"""
Enhanced Methods for Iterative Refinement System

These methods will be integrated into iterative_refinement.py

Enhanced capabilities:
1. Subtype-specific discriminative power (one-vs-rest AUC per cluster)
2. Feature-level refinement within domains
3. Non-linear relationship detection via kernel methods
4. Domain interaction effects (synergistic pairs)
5. Bootstrap stability with confidence intervals
6. Subtype homogeneity metrics
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.utils import resample
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


# ENHANCEMENT 1: Subtype-Specific Discriminative Power
def evaluate_subtype_specific_discriminative_power(
    domain_features: Dict[str, pd.DataFrame],
    cluster_labels: np.ndarray
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate discriminative power per domain per subtype.

    Different domains may discriminate different subtypes. For example,
    genetic factors might discriminate one ASD/ADHD subtype while
    autonomic features discriminate another.

    Parameters
    ----------
    domain_features : dict
        Domain name -> feature DataFrame
    cluster_labels : np.ndarray
        Current cluster assignments

    Returns
    -------
    dict
        Domain -> Subtype -> Discriminative score (AUC)
        Higher AUC means domain is better at discriminating that subtype from others
    """
    n_clusters = len(np.unique(cluster_labels))
    subtype_powers = {}

    for domain, df in domain_features.items():
        if df is None or df.empty:
            continue

        subtype_powers[domain] = {}
        try:
            features_scaled = StandardScaler().fit_transform(df)

            # Evaluate one-vs-rest discrimination for each subtype
            for cluster_id in range(n_clusters):
                # Binary labels: current cluster vs rest
                binary_labels = (cluster_labels == cluster_id).astype(int)

                # Skip if too few positive samples
                if np.sum(binary_labels) < 5 or np.sum(1 - binary_labels) < 5:
                    subtype_powers[domain][cluster_id] = 0.5  # Random baseline
                    continue

                try:
                    # Logistic regression with cross-validation
                    lr = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42, n_jobs=-1)
                    auc_scores = cross_val_score(
                        lr, features_scaled, binary_labels,
                        cv=5, scoring='roc_auc', n_jobs=-1
                    )
                    subtype_powers[domain][cluster_id] = np.mean(auc_scores)

                except Exception as e:
                    logger.debug(f"Subtype discrimination failed for {domain}, "
                               f"cluster {cluster_id}: {e}")
                    subtype_powers[domain][cluster_id] = 0.5

        except Exception as e:
            logger.debug(f"Failed to process domain {domain}: {e}")
            subtype_powers[domain] = {i: 0.5 for i in range(n_clusters)}

    return subtype_powers


# ENHANCEMENT 2: Feature-Level Refinement
def identify_discriminative_features_within_domain(
    domain_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    top_k: int = 10
) -> List[str]:
    """
    Identify most discriminative features within a domain.

    Uses mutual information and recursive feature elimination to find
    features that contribute most to subtype discrimination.

    Parameters
    ----------
    domain_df : pd.DataFrame
        Feature matrix for this domain
    cluster_labels : np.ndarray
        Cluster assignments
    top_k : int, default=10
        Number of top features to return

    Returns
    -------
    list
        Names of top discriminative features
    """
    if domain_df.empty or len(domain_df.columns) == 0:
        return []

    try:
        features_scaled = StandardScaler().fit_transform(domain_df)

        # 1. Mutual information scores
        mi_scores = mutual_info_classif(
            features_scaled, cluster_labels, random_state=42
        )

        # 2. Recursive feature elimination (if enough features and time)
        if features_scaled.shape[1] > top_k and features_scaled.shape[1] < 100:
            try:
                rf = RandomForestClassifier(
                    n_estimators=50, random_state=42, max_depth=10, n_jobs=-1
                )
                rfecv = RFECV(rf, step=1, cv=3, scoring='accuracy', n_jobs=-1)
                rfecv.fit(features_scaled, cluster_labels)
                rfe_support = rfecv.support_
            except Exception as e:
                logger.debug(f"RFECV failed: {e}")
                rfe_support = np.ones(features_scaled.shape[1], dtype=bool)
        else:
            rfe_support = np.ones(features_scaled.shape[1], dtype=bool)

        # 3. Combine rankings
        # MI ranking (higher is better)
        mi_ranking = np.argsort(mi_scores)[::-1]
        feature_importance = np.zeros(len(domain_df.columns))

        # Assign scores based on MI rank
        for rank, idx in enumerate(mi_ranking):
            feature_importance[idx] += (len(mi_ranking) - rank)

        # Boost features selected by RFE
        feature_importance[rfe_support] += len(mi_ranking) * 0.5

        # Get top features
        top_indices = np.argsort(feature_importance)[::-1][:top_k]
        return domain_df.columns[top_indices].tolist()

    except Exception as e:
        logger.warning(f"Feature selection failed: {e}")
        # Fallback: return first k features
        return domain_df.columns[:min(top_k, len(domain_df.columns))].tolist()


# ENHANCEMENT 3: Non-linear Discrimination
def calculate_nonlinear_discriminative_power(
    domain_features: pd.DataFrame,
    cluster_labels: np.ndarray,
    timeout_seconds: int = 30
) -> float:
    """
    Calculate non-linear discriminative power using kernel methods.

    Tests multiple non-linear classifiers to detect domains with
    non-linear relationships to subtypes.

    Parameters
    ----------
    domain_features : pd.DataFrame
        Feature matrix
    cluster_labels : np.ndarray
        Cluster assignments
    timeout_seconds : int, default=30
        Max time per classifier

    Returns
    -------
    float
        Best non-linear classification accuracy (0-1)
    """
    if domain_features.empty:
        return 0.0

    try:
        features_scaled = StandardScaler().fit_transform(domain_features)

        # Limit features if too many (for computational efficiency)
        if features_scaled.shape[1] > 50:
            # Use PCA to reduce
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50, random_state=42)
            features_scaled = pca.fit_transform(features_scaled)

        # Try multiple non-linear classifiers
        classifiers = [
            ('RBF SVM', SVC(kernel='rbf', C=1.0, random_state=42)),
            ('Poly SVM', SVC(kernel='poly', degree=2, C=1.0, random_state=42)),
        ]

        # For small datasets, also try Gaussian Process (expensive)
        if features_scaled.shape[0] < 200:
            try:
                from sklearn.gaussian_process import GaussianProcessClassifier
                classifiers.append(
                    ('GP', GaussianProcessClassifier(random_state=42))
                )
            except ImportError:
                pass

        scores = []
        for name, clf in classifiers:
            try:
                # Use 3-fold CV for speed
                cv_scores = cross_val_score(
                    clf, features_scaled, cluster_labels,
                    cv=3, scoring='accuracy', n_jobs=-1
                )
                score = np.mean(cv_scores)
                scores.append(score)
                logger.debug(f"  {name}: {score:.3f}")

            except Exception as e:
                logger.debug(f"  {name} failed: {e}")
                scores.append(0.0)

        # Return best non-linear score
        return max(scores) if scores else 0.0

    except Exception as e:
        logger.warning(f"Non-linear scoring failed: {e}")
        return 0.0


# ENHANCEMENT 4: Domain Interaction Effects
def evaluate_domain_interactions(
    domain_features: Dict[str, pd.DataFrame],
    cluster_labels: np.ndarray,
    max_pairs: int = 10
) -> List[Tuple[str, str, float]]:
    """
    Find domain pairs with synergistic discriminative power.

    Identifies cases where two domains together are more discriminative
    than either alone (positive interaction/synergy).

    Parameters
    ----------
    domain_features : dict
        Domain name -> feature DataFrame
    cluster_labels : np.ndarray
        Cluster assignments
    max_pairs : int, default=10
        Maximum interaction pairs to return

    Returns
    -------
    list of tuple
        (domain1, domain2, interaction_strength) sorted by strength
        Interaction strength > 0 indicates synergy
    """
    interaction_scores = []
    n_clusters = len(np.unique(cluster_labels))

    # Limit to domains with reasonable number of features
    valid_domains = {
        d: df for d, df in domain_features.items()
        if df is not None and not df.empty and df.shape[1] < 100
    }

    if len(valid_domains) < 2:
        return []

    for domain1, domain2 in combinations(valid_domains.keys(), 2):
        df1 = domain_features[domain1]
        df2 = domain_features[domain2]

        if df1 is None or df2 is None or df1.empty or df2.empty:
            continue

        try:
            # Scale features
            scaler = StandardScaler()
            f1_scaled = scaler.fit_transform(df1.iloc[:, :20])  # Limit features
            f2_scaled = scaler.fit_transform(df2.iloc[:, :20])

            # Individual discriminative power (using adjusted mutual info)
            kmeans_params = {'n_clusters': n_clusters, 'random_state': 42, 'n_init': 10}

            labels1 = KMeans(**kmeans_params).fit_predict(f1_scaled)
            mi1 = adjusted_mutual_info_score(labels1, cluster_labels)

            labels2 = KMeans(**kmeans_params).fit_predict(f2_scaled)
            mi2 = adjusted_mutual_info_score(labels2, cluster_labels)

            # Combined discriminative power
            combined = np.hstack([f1_scaled, f2_scaled])
            labels_combined = KMeans(**kmeans_params).fit_predict(combined)
            mi_combined = adjusted_mutual_info_score(labels_combined, cluster_labels)

            # Interaction effect (synergy beyond additive)
            # We compare to the max (not sum) to identify true synergy
            expected = max(mi1, mi2)
            interaction = mi_combined - expected

            if interaction > 0.01:  # Small threshold to avoid noise
                interaction_scores.append((domain1, domain2, interaction))

        except Exception as e:
            logger.debug(f"Interaction calculation failed for {domain1}-{domain2}: {e}")
            continue

    # Return top interactions
    interaction_scores.sort(key=lambda x: x[2], reverse=True)
    return interaction_scores[:max_pairs]


# ENHANCEMENT 5: Bootstrap Stability
def bootstrap_discriminative_power(
    domain_features: pd.DataFrame,
    cluster_labels: np.ndarray,
    n_bootstrap: int = 100
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence intervals for discriminative power.

    Provides stability assessment for discriminative scores.

    Parameters
    ----------
    domain_features : pd.DataFrame
        Feature matrix
    cluster_labels : np.ndarray
        Cluster assignments
    n_bootstrap : int, default=100
        Number of bootstrap samples

    Returns
    -------
    tuple
        (mean_score, lower_95ci, upper_95ci)
    """
    if domain_features.empty:
        return 0.0, 0.0, 0.0

    bootstrap_scores = []
    n_samples = len(cluster_labels)
    n_clusters = len(np.unique(cluster_labels))

    for _ in range(n_bootstrap):
        try:
            # Bootstrap sample
            indices = resample(range(n_samples), n_samples=n_samples, random_state=None)
            boot_features = domain_features.iloc[indices]
            boot_labels = cluster_labels[indices]

            # Check if all clusters represented
            if len(np.unique(boot_labels)) < n_clusters:
                continue

            # Calculate discriminative score
            features_scaled = StandardScaler().fit_transform(boot_features)
            score = silhouette_score(features_scaled, boot_labels)
            bootstrap_scores.append(score)

        except Exception as e:
            logger.debug(f"Bootstrap sample failed: {e}")
            continue

    if len(bootstrap_scores) > 10:
        mean_score = np.mean(bootstrap_scores)
        lower_ci = np.percentile(bootstrap_scores, 2.5)
        upper_ci = np.percentile(bootstrap_scores, 97.5)
        return mean_score, lower_ci, upper_ci
    else:
        # Not enough successful bootstrap samples
        return 0.0, 0.0, 0.0


# ENHANCEMENT 6: Subtype Homogeneity Metrics
def calculate_subtype_homogeneity(
    domain_features: pd.DataFrame,
    cluster_labels: np.ndarray
) -> Dict[str, float]:
    """
    Calculate within-subtype homogeneity metrics.

    Measures how consistent/homogeneous each subtype is for this domain.

    Parameters
    ----------
    domain_features : pd.DataFrame
        Feature matrix
    cluster_labels : np.ndarray
        Cluster assignments

    Returns
    -------
    dict
        Homogeneity metrics:
        - avg_within_distance: Average pairwise distance within clusters (lower is better)
        - calinski_harabasz: Variance ratio (between/within, higher is better)
        - avg_cohesion: Average cluster cohesion (higher is better)
    """
    if domain_features.empty:
        return {
            'avg_within_distance': np.nan,
            'calinski_harabasz': np.nan,
            'avg_cohesion': np.nan
        }

    try:
        features_scaled = StandardScaler().fit_transform(domain_features)
        n_clusters = len(np.unique(cluster_labels))

        metrics = {}

        # 1. Average within-cluster distance
        distances = squareform(pdist(features_scaled, 'euclidean'))
        within_cluster_distances = []

        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            if np.sum(mask) > 1:
                cluster_distances = distances[mask][:, mask]
                # Upper triangle only (exclude diagonal)
                upper_tri = cluster_distances[np.triu_indices_from(cluster_distances, k=1)]
                if len(upper_tri) > 0:
                    within_cluster_distances.append(np.mean(upper_tri))

        metrics['avg_within_distance'] = (
            np.mean(within_cluster_distances) if within_cluster_distances else np.nan
        )

        # 2. Calinski-Harabasz score (variance ratio)
        from sklearn.metrics import calinski_harabasz_score
        metrics['calinski_harabasz'] = calinski_harabasz_score(
            features_scaled, cluster_labels
        )

        # 3. Cluster cohesion (inverse of variance)
        cohesions = []
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            if np.sum(mask) > 1:
                cluster_features = features_scaled[mask]
                cluster_var = np.mean(np.var(cluster_features, axis=0))
                cohesions.append(1.0 / (1.0 + cluster_var))

        metrics['avg_cohesion'] = np.mean(cohesions) if cohesions else np.nan

        return metrics

    except Exception as e:
        logger.warning(f"Homogeneity calculation failed: {e}")
        return {
            'avg_within_distance': np.nan,
            'calinski_harabasz': np.nan,
            'avg_cohesion': np.nan
        }
