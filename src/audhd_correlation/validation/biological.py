"""Biological validity tests for clustering results

Tests enrichment for known biological pathways and clinical relevance.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests


@dataclass
class EnrichmentResult:
    """Result of pathway enrichment analysis"""
    pathway_name: str
    cluster_id: int
    n_in_cluster_in_pathway: int
    n_in_cluster_not_in_pathway: int
    n_not_in_cluster_in_pathway: int
    n_not_in_cluster_not_in_pathway: int
    odds_ratio: float
    p_value: float
    adjusted_p_value: float
    enriched: bool


@dataclass
class ClinicalRelevanceResult:
    """Result of clinical relevance analysis"""
    clinical_variable: str
    cluster_differences: Dict[str, float]  # cluster_id -> mean value
    test_statistic: float
    p_value: float
    adjusted_p_value: float
    effect_size: float
    significant: bool
    interpretation: str


def pathway_enrichment_analysis(
    cluster_labels: np.ndarray,
    feature_names: List[str],
    pathway_dict: Dict[str, List[str]],
    min_pathway_size: int = 5,
    max_pathway_size: int = 500,
    correction_method: str = 'fdr_bh',
) -> List[EnrichmentResult]:
    """
    Test for pathway enrichment in clusters

    Uses Fisher's exact test to test if pathway members are over-represented
    in specific clusters.

    Args:
        cluster_labels: Cluster assignments for each feature
        feature_names: Names of features
        pathway_dict: Dictionary mapping pathway names to lists of feature names
        min_pathway_size: Minimum pathway size to test
        max_pathway_size: Maximum pathway size to test
        correction_method: Multiple testing correction method

    Returns:
        List of EnrichmentResult objects
    """
    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    results = []

    # Filter pathways by size
    pathways_to_test = {
        name: genes
        for name, genes in pathway_dict.items()
        if min_pathway_size <= len(genes) <= max_pathway_size
    }

    if len(pathways_to_test) == 0:
        warnings.warn("No pathways meet size criteria")
        return results

    # Convert feature names to set for faster lookup
    feature_set = set(feature_names)

    # For each cluster and pathway
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_features = set(np.array(feature_names)[cluster_mask])

        for pathway_name, pathway_genes in pathways_to_test.items():
            # Filter pathway genes to those in our dataset
            pathway_genes_in_data = set(pathway_genes) & feature_set

            if len(pathway_genes_in_data) == 0:
                continue

            # Contingency table
            # |            | In Pathway | Not in Pathway |
            # |------------|------------|----------------|
            # | In Cluster | a          | b              |
            # | Not in Cluster | c      | d              |

            a = len(cluster_features & pathway_genes_in_data)
            b = len(cluster_features - pathway_genes_in_data)
            c = len(pathway_genes_in_data - cluster_features)
            d = len(feature_set - pathway_genes_in_data - cluster_features)

            # Fisher's exact test
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')

            results.append(EnrichmentResult(
                pathway_name=pathway_name,
                cluster_id=int(cluster_id),
                n_in_cluster_in_pathway=a,
                n_in_cluster_not_in_pathway=b,
                n_not_in_cluster_in_pathway=c,
                n_not_in_cluster_not_in_pathway=d,
                odds_ratio=odds_ratio,
                p_value=p_value,
                adjusted_p_value=p_value,  # Will be updated
                enriched=False,  # Will be updated
            ))

    # Multiple testing correction
    if len(results) > 0:
        p_values = [r.p_value for r in results]
        _, adjusted_p_values, _, _ = multipletests(p_values, method=correction_method)

        for i, result in enumerate(results):
            result.adjusted_p_value = adjusted_p_values[i]
            result.enriched = adjusted_p_values[i] < 0.05

    return results


def clinical_relevance_analysis(
    cluster_labels: np.ndarray,
    clinical_data: pd.DataFrame,
    clinical_variables: Optional[List[str]] = None,
    correction_method: str = 'fdr_bh',
    effect_size_threshold: float = 0.5,
) -> List[ClinicalRelevanceResult]:
    """
    Test if clusters differ in clinically relevant variables

    Uses Kruskal-Wallis test for continuous variables and chi-square for categorical.

    Args:
        cluster_labels: Cluster assignments
        clinical_data: DataFrame with clinical variables
        clinical_variables: Variables to test (if None, test all)
        correction_method: Multiple testing correction
        effect_size_threshold: Threshold for declaring large effect

    Returns:
        List of ClinicalRelevanceResult objects
    """
    # Align data
    if len(cluster_labels) != len(clinical_data):
        warnings.warn("Cluster labels and clinical data have different lengths")
        min_len = min(len(cluster_labels), len(clinical_data))
        cluster_labels = cluster_labels[:min_len]
        clinical_data = clinical_data.iloc[:min_len]

    # Filter to valid clusters
    valid_mask = cluster_labels >= 0
    cluster_labels = cluster_labels[valid_mask]
    clinical_data = clinical_data[valid_mask]

    if clinical_variables is None:
        clinical_variables = list(clinical_data.columns)

    unique_clusters = np.unique(cluster_labels)

    if len(unique_clusters) < 2:
        warnings.warn("Less than 2 clusters, skipping clinical relevance analysis")
        return []

    results = []

    for var in clinical_variables:
        if var not in clinical_data.columns:
            continue

        var_data = clinical_data[var].values

        # Skip if too many missing values
        if np.sum(pd.isna(var_data)) > 0.5 * len(var_data):
            continue

        # Determine if categorical or continuous
        is_categorical = (
            clinical_data[var].dtype == 'object' or
            clinical_data[var].dtype.name == 'category' or
            len(np.unique(var_data[~pd.isna(var_data)])) < 10
        )

        if is_categorical:
            result = _test_categorical_variable(
                var, var_data, cluster_labels, unique_clusters
            )
        else:
            result = _test_continuous_variable(
                var, var_data, cluster_labels, unique_clusters
            )

        if result is not None:
            results.append(result)

    # Multiple testing correction
    if len(results) > 0:
        p_values = [r.p_value for r in results]
        _, adjusted_p_values, _, _ = multipletests(p_values, method=correction_method)

        for i, result in enumerate(results):
            result.adjusted_p_value = adjusted_p_values[i]
            result.significant = (
                adjusted_p_values[i] < 0.05 and
                result.effect_size > effect_size_threshold
            )

    return results


def _test_continuous_variable(
    var_name: str,
    var_data: np.ndarray,
    cluster_labels: np.ndarray,
    unique_clusters: np.ndarray,
) -> Optional[ClinicalRelevanceResult]:
    """Test continuous variable across clusters"""
    # Remove missing values
    valid_mask = ~pd.isna(var_data)
    var_data = var_data[valid_mask]
    cluster_labels_valid = cluster_labels[valid_mask]

    if len(var_data) < 10:
        return None

    # Compute mean per cluster
    cluster_means = {}
    cluster_groups = []

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels_valid == cluster_id
        cluster_values = var_data[cluster_mask]

        if len(cluster_values) > 0:
            cluster_means[int(cluster_id)] = float(np.mean(cluster_values))
            cluster_groups.append(cluster_values)

    if len(cluster_groups) < 2:
        return None

    # Kruskal-Wallis test (non-parametric ANOVA)
    try:
        stat, p_value = kruskal(*cluster_groups)
    except:
        return None

    # Effect size (eta-squared approximation)
    # H / (n - 1) where H is Kruskal-Wallis statistic
    effect_size = stat / (len(var_data) - 1)

    # Interpretation
    if effect_size > 0.14:
        interpretation = "large_effect"
    elif effect_size > 0.06:
        interpretation = "medium_effect"
    elif effect_size > 0.01:
        interpretation = "small_effect"
    else:
        interpretation = "negligible_effect"

    return ClinicalRelevanceResult(
        clinical_variable=var_name,
        cluster_differences=cluster_means,
        test_statistic=stat,
        p_value=p_value,
        adjusted_p_value=p_value,  # Will be updated
        effect_size=effect_size,
        significant=False,  # Will be updated
        interpretation=interpretation,
    )


def _test_categorical_variable(
    var_name: str,
    var_data: np.ndarray,
    cluster_labels: np.ndarray,
    unique_clusters: np.ndarray,
) -> Optional[ClinicalRelevanceResult]:
    """Test categorical variable across clusters"""
    # Remove missing values
    valid_mask = ~pd.isna(var_data)
    var_data = var_data[valid_mask]
    cluster_labels_valid = cluster_labels[valid_mask]

    if len(var_data) < 10:
        return None

    # Create contingency table
    unique_categories = np.unique(var_data)

    if len(unique_categories) < 2:
        return None

    contingency_table = []
    cluster_distributions = {}

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels_valid == cluster_id
        cluster_values = var_data[cluster_mask]

        row = []
        for category in unique_categories:
            count = np.sum(cluster_values == category)
            row.append(count)

        # Distribution for this cluster
        cluster_distributions[int(cluster_id)] = row[0] / (np.sum(row) + 1e-10)

        contingency_table.append(row)

    contingency_table = np.array(contingency_table)

    # Chi-square test
    try:
        stat, p_value, dof, expected = chi2_contingency(contingency_table)
    except:
        return None

    # Effect size (Cramér's V)
    n = np.sum(contingency_table)
    min_dim = min(len(unique_clusters) - 1, len(unique_categories) - 1)
    effect_size = np.sqrt(stat / (n * min_dim)) if min_dim > 0 else 0.0

    # Interpretation
    if effect_size > 0.25:
        interpretation = "large_effect"
    elif effect_size > 0.15:
        interpretation = "medium_effect"
    elif effect_size > 0.05:
        interpretation = "small_effect"
    else:
        interpretation = "negligible_effect"

    return ClinicalRelevanceResult(
        clinical_variable=var_name,
        cluster_differences=cluster_distributions,
        test_statistic=stat,
        p_value=p_value,
        adjusted_p_value=p_value,  # Will be updated
        effect_size=effect_size,
        significant=False,  # Will be updated
        interpretation=interpretation,
    )


def symptom_severity_analysis(
    cluster_labels: np.ndarray,
    symptom_scores: Dict[str, np.ndarray],
    symptom_names: Optional[List[str]] = None,
) -> Dict[str, ClinicalRelevanceResult]:
    """
    Analyze symptom severity differences across clusters

    Args:
        cluster_labels: Cluster assignments
        symptom_scores: Dictionary mapping symptom names to score arrays
        symptom_names: Symptoms to analyze (if None, analyze all)

    Returns:
        Dictionary mapping symptom to ClinicalRelevanceResult
    """
    if symptom_names is None:
        symptom_names = list(symptom_scores.keys())

    results = {}

    for symptom in symptom_names:
        if symptom not in symptom_scores:
            continue

        scores = symptom_scores[symptom]

        # Create DataFrame for analysis
        df = pd.DataFrame({symptom: scores})

        # Analyze
        clinical_results = clinical_relevance_analysis(
            cluster_labels,
            df,
            clinical_variables=[symptom],
            correction_method='fdr_bh',
        )

        if len(clinical_results) > 0:
            results[symptom] = clinical_results[0]

    return results


def diagnostic_concordance(
    cluster_labels: np.ndarray,
    diagnoses: np.ndarray,
    diagnosis_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute concordance between clusters and clinical diagnoses

    Args:
        cluster_labels: Cluster assignments
        diagnoses: Clinical diagnoses (encoded or string)
        diagnosis_names: Names of diagnoses

    Returns:
        Dictionary with concordance metrics
    """
    # Filter valid clusters
    valid_mask = cluster_labels >= 0
    cluster_labels = cluster_labels[valid_mask]
    diagnoses = diagnoses[valid_mask]

    unique_clusters = np.unique(cluster_labels)
    unique_diagnoses = np.unique(diagnoses)

    # Create contingency table
    contingency = np.zeros((len(unique_clusters), len(unique_diagnoses)))

    for i, cluster in enumerate(unique_clusters):
        for j, diagnosis in enumerate(unique_diagnoses):
            count = np.sum((cluster_labels == cluster) & (diagnoses == diagnosis))
            contingency[i, j] = count

    # Purity: max proportion in each cluster
    cluster_purities = []
    for i in range(len(unique_clusters)):
        if np.sum(contingency[i]) > 0:
            purity = np.max(contingency[i]) / np.sum(contingency[i])
            cluster_purities.append(purity)

    overall_purity = np.mean(cluster_purities) if len(cluster_purities) > 0 else 0.0

    # Inverse purity: max proportion in each diagnosis
    diagnosis_purities = []
    for j in range(len(unique_diagnoses)):
        if np.sum(contingency[:, j]) > 0:
            purity = np.max(contingency[:, j]) / np.sum(contingency[:, j])
            diagnosis_purities.append(purity)

    inverse_purity = np.mean(diagnosis_purities) if len(diagnosis_purities) > 0 else 0.0

    # F-measure (harmonic mean of purity and inverse purity)
    if overall_purity + inverse_purity > 0:
        f_measure = 2 * (overall_purity * inverse_purity) / (overall_purity + inverse_purity)
    else:
        f_measure = 0.0

    return {
        'cluster_purity': overall_purity,
        'diagnosis_purity': inverse_purity,
        'f_measure': f_measure,
        'n_clusters': len(unique_clusters),
        'n_diagnoses': len(unique_diagnoses),
    }


def functional_outcome_prediction(
    cluster_labels: np.ndarray,
    outcomes: np.ndarray,
    outcome_name: str = "functional_outcome",
) -> Dict[str, float]:
    """
    Test if clusters predict functional outcomes

    Args:
        cluster_labels: Cluster assignments
        outcomes: Functional outcome scores
        outcome_name: Name of outcome

    Returns:
        Dictionary with prediction metrics
    """
    # Filter valid
    valid_mask = (cluster_labels >= 0) & (~pd.isna(outcomes))
    cluster_labels = cluster_labels[valid_mask]
    outcomes = outcomes[valid_mask]

    if len(outcomes) < 10:
        return {'predictive_r2': 0.0, 'kruskal_p_value': 1.0}

    unique_clusters = np.unique(cluster_labels)

    # Test if outcomes differ across clusters
    cluster_groups = [outcomes[cluster_labels == c] for c in unique_clusters]

    try:
        stat, p_value = kruskal(*cluster_groups)
    except:
        p_value = 1.0

    # Compute pseudo-R² (variance explained by cluster)
    overall_var = np.var(outcomes)

    within_cluster_var = 0.0
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_outcomes = outcomes[cluster_mask]

        if len(cluster_outcomes) > 1:
            within_cluster_var += np.var(cluster_outcomes) * len(cluster_outcomes)

    within_cluster_var /= len(outcomes)

    pseudo_r2 = 1.0 - (within_cluster_var / (overall_var + 1e-10))

    return {
        'predictive_r2': pseudo_r2,
        'kruskal_p_value': p_value,
        'outcome_name': outcome_name,
        'significant': p_value < 0.05,
    }