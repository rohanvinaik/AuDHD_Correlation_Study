"""Ancestry-stratified validation

Tests generalization across different ancestry groups and handles population stratification.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class AncestryValidationResult:
    """Results of ancestry-stratified validation"""
    ancestry_group: str
    n_samples: int
    cluster_distribution: Dict[int, int]
    silhouette_score: float
    replication_rate: float
    ancestry_specific_clusters: List[int]
    shared_clusters: List[int]
    metadata: Dict = field(default_factory=dict)


@dataclass
class PopulationStratificationTest:
    """Test for population stratification"""
    chi2_statistic: float
    p_value: float
    stratified: bool
    ancestry_cluster_association: pd.DataFrame
    cramers_v: float


class AncestryStratifiedValidator:
    """Validates model across ancestry groups"""

    def __init__(
        self,
        stratification_threshold: float = 0.05,
        min_group_size: int = 20,
    ):
        """
        Initialize validator

        Args:
            stratification_threshold: P-value threshold for stratification test
            min_group_size: Minimum samples per ancestry group
        """
        self.stratification_threshold = stratification_threshold
        self.min_group_size = min_group_size

    def validate_across_ancestry(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        ancestry_labels: np.ndarray,
        ancestry_names: Optional[List[str]] = None,
    ) -> List[AncestryValidationResult]:
        """
        Validate clustering across ancestry groups

        Args:
            data: Feature data
            labels: Cluster labels
            ancestry_labels: Ancestry group labels
            ancestry_names: Names of ancestry groups

        Returns:
            List of AncestryValidationResult per group
        """
        unique_ancestries = np.unique(ancestry_labels)

        if ancestry_names is None:
            ancestry_names = [f'Ancestry_{i}' for i in unique_ancestries]

        results = []

        for ancestry_id, ancestry_name in zip(unique_ancestries, ancestry_names):
            # Select samples from this ancestry
            ancestry_mask = ancestry_labels == ancestry_id

            if ancestry_mask.sum() < self.min_group_size:
                warnings.warn(
                    f"Ancestry group {ancestry_name} has only "
                    f"{ancestry_mask.sum()} samples, skipping"
                )
                continue

            ancestry_data = data[ancestry_mask]
            ancestry_clusters = labels[ancestry_mask]

            # Calculate metrics for this ancestry
            cluster_dist = {
                int(label): int((ancestry_clusters == label).sum())
                for label in np.unique(ancestry_clusters)
                if label >= 0
            }

            # Silhouette score
            if len(np.unique(ancestry_clusters)) > 1:
                sil_score = silhouette_score(
                    ancestry_data.values,
                    ancestry_clusters,
                )
            else:
                sil_score = 0.0

            # Replication rate (proportion assigned to major clusters)
            major_clusters = [
                label for label, count in cluster_dist.items()
                if count >= self.min_group_size
            ]
            replication = sum(
                count for label, count in cluster_dist.items()
                if label in major_clusters
            ) / len(ancestry_clusters)

            # Identify ancestry-specific vs shared clusters
            ancestry_specific, shared = self._identify_cluster_specificity(
                labels,
                ancestry_labels,
                ancestry_id,
            )

            results.append(AncestryValidationResult(
                ancestry_group=ancestry_name,
                n_samples=ancestry_mask.sum(),
                cluster_distribution=cluster_dist,
                silhouette_score=sil_score,
                replication_rate=replication,
                ancestry_specific_clusters=ancestry_specific,
                shared_clusters=shared,
                metadata={
                    'ancestry_id': int(ancestry_id),
                    'n_clusters': len(cluster_dist),
                },
            ))

        return results

    def _identify_cluster_specificity(
        self,
        labels: np.ndarray,
        ancestry_labels: np.ndarray,
        target_ancestry: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Identify ancestry-specific vs shared clusters

        Args:
            labels: Cluster labels
            ancestry_labels: Ancestry labels
            target_ancestry: Target ancestry ID

        Returns:
            Tuple of (ancestry_specific_clusters, shared_clusters)
        """
        unique_clusters = np.unique(labels[labels >= 0])

        ancestry_specific = []
        shared = []

        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id

            # Proportion from target ancestry
            target_prop = (
                (ancestry_labels[cluster_mask] == target_ancestry).sum()
                / cluster_mask.sum()
            )

            # If >80% from one ancestry, consider ancestry-specific
            if target_prop > 0.8:
                ancestry_specific.append(int(cluster_id))
            else:
                shared.append(int(cluster_id))

        return ancestry_specific, shared

    def test_population_stratification(
        self,
        labels: np.ndarray,
        ancestry_labels: np.ndarray,
        ancestry_names: Optional[List[str]] = None,
    ) -> PopulationStratificationTest:
        """
        Test for population stratification (ancestry-cluster association)

        Args:
            labels: Cluster labels
            ancestry_labels: Ancestry labels
            ancestry_names: Names of ancestry groups

        Returns:
            PopulationStratificationTest object
        """
        # Create contingency table
        unique_clusters = sorted(np.unique(labels[labels >= 0]))
        unique_ancestries = sorted(np.unique(ancestry_labels))

        if ancestry_names is None:
            ancestry_names = [f'Ancestry_{i}' for i in unique_ancestries]

        contingency = np.zeros((len(unique_ancestries), len(unique_clusters)))

        for i, ancestry in enumerate(unique_ancestries):
            for j, cluster in enumerate(unique_clusters):
                contingency[i, j] = (
                    (ancestry_labels == ancestry) & (labels == cluster)
                ).sum()

        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        # Cramér's V (effect size)
        n = contingency.sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        # Create association DataFrame
        association_df = pd.DataFrame(
            contingency,
            index=ancestry_names,
            columns=[f'Cluster {c}' for c in unique_clusters],
        )

        # Convert to proportions
        association_df = association_df.div(association_df.sum(axis=1), axis=0)

        return PopulationStratificationTest(
            chi2_statistic=chi2,
            p_value=p_value,
            stratified=p_value < self.stratification_threshold,
            ancestry_cluster_association=association_df,
            cramers_v=cramers_v,
        )

    def calculate_ancestry_pcs(
        self,
        genetic_data: np.ndarray,
        n_components: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ancestry principal components

        Args:
            genetic_data: Genetic variant data
            n_components: Number of PCs

        Returns:
            Tuple of (PCs, explained_variance_ratio)
        """
        pca = PCA(n_components=n_components, random_state=42)
        pcs = pca.fit_transform(genetic_data)

        return pcs, pca.explained_variance_ratio_

    def adjust_for_ancestry(
        self,
        data: pd.DataFrame,
        ancestry_pcs: np.ndarray,
        n_pcs: int = 10,
    ) -> pd.DataFrame:
        """
        Adjust features for ancestry using PCs

        Args:
            data: Feature data
            ancestry_pcs: Ancestry principal components
            n_pcs: Number of PCs to use

        Returns:
            Adjusted feature data
        """
        from sklearn.linear_model import LinearRegression

        adjusted_data = data.copy()

        for col in data.columns:
            if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Regress out ancestry PCs
                model = LinearRegression()
                model.fit(ancestry_pcs[:, :n_pcs], data[col].values)

                # Residuals are ancestry-adjusted values
                adjusted_data[col] = data[col] - model.predict(ancestry_pcs[:, :n_pcs])

        return adjusted_data

    def plot_ancestry_distribution(
        self,
        labels: np.ndarray,
        ancestry_labels: np.ndarray,
        ancestry_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ):
        """
        Plot cluster distribution across ancestries

        Args:
            labels: Cluster labels
            ancestry_labels: Ancestry labels
            ancestry_names: Names of ancestry groups
            output_path: Output path for plot
        """
        unique_clusters = sorted(np.unique(labels[labels >= 0]))
        unique_ancestries = sorted(np.unique(ancestry_labels))

        if ancestry_names is None:
            ancestry_names = [f'Ancestry {i}' for i in unique_ancestries]

        # Create proportion matrix
        proportions = np.zeros((len(unique_ancestries), len(unique_clusters)))

        for i, ancestry in enumerate(unique_ancestries):
            ancestry_mask = ancestry_labels == ancestry
            ancestry_total = ancestry_mask.sum()

            for j, cluster in enumerate(unique_clusters):
                proportions[i, j] = (
                    (labels[ancestry_mask] == cluster).sum() / ancestry_total
                )

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            proportions,
            xticklabels=[f'Cluster {c}' for c in unique_clusters],
            yticklabels=ancestry_names,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Proportion'},
            ax=ax,
        )

        ax.set_xlabel('Cluster')
        ax.set_ylabel('Ancestry Group')
        ax.set_title('Cluster Distribution Across Ancestry Groups')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def compare_ancestry_specific_effects(
    data: pd.DataFrame,
    labels: np.ndarray,
    ancestry_labels: np.ndarray,
    feature_subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare effect sizes across ancestry groups

    Args:
        data: Feature data
        labels: Cluster labels
        ancestry_labels: Ancestry labels
        feature_subset: Subset of features to analyze

    Returns:
        DataFrame with ancestry-specific effects
    """
    from scipy.stats import f_oneway

    if feature_subset is None:
        feature_subset = data.columns.tolist()

    results = []

    unique_ancestries = np.unique(ancestry_labels)

    for feature in feature_subset:
        ancestry_effects = {}

        for ancestry in unique_ancestries:
            ancestry_mask = ancestry_labels == ancestry

            # Calculate effect size within this ancestry
            groups = [
                data[feature].values[ancestry_mask & (labels == label)]
                for label in np.unique(labels)
                if label >= 0 and (ancestry_mask & (labels == label)).sum() > 0
            ]

            if len(groups) < 2:
                ancestry_effects[f'ancestry_{ancestry}'] = np.nan
                continue

            f_stat, p_val = f_oneway(*groups)

            # Eta-squared
            grand_mean = data[feature].values[ancestry_mask].mean()
            ss_between = sum(
                len(g) * (g.mean() - grand_mean) ** 2
                for g in groups
            )
            ss_total = (
                (data[feature].values[ancestry_mask] - grand_mean) ** 2
            ).sum()
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            ancestry_effects[f'ancestry_{ancestry}'] = eta_squared

        ancestry_effects['feature'] = feature
        results.append(ancestry_effects)

    return pd.DataFrame(results)


def calculate_transferability_score(
    source_ancestry_results: AncestryValidationResult,
    target_ancestry_results: AncestryValidationResult,
) -> float:
    """
    Calculate transferability score between ancestries

    Args:
        source_ancestry_results: Validation results for source ancestry
        target_ancestry_results: Validation results for target ancestry

    Returns:
        Transferability score (0-1, higher = better transfer)
    """
    # Shared clusters
    source_clusters = set(source_ancestry_results.cluster_distribution.keys())
    target_clusters = set(target_ancestry_results.cluster_distribution.keys())

    shared = source_clusters & target_clusters
    union = source_clusters | target_clusters

    cluster_overlap = len(shared) / len(union) if union else 0

    # Silhouette scores
    sil_similarity = 1.0 - abs(
        source_ancestry_results.silhouette_score
        - target_ancestry_results.silhouette_score
    )

    # Replication rates
    rep_similarity = 1.0 - abs(
        source_ancestry_results.replication_rate
        - target_ancestry_results.replication_rate
    )

    # Combined score
    transferability = (
        0.4 * cluster_overlap + 0.3 * sil_similarity + 0.3 * rep_similarity
    )

    return transferability