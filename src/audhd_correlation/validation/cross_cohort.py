"""Cross-cohort correlation and replication analysis

Analyzes correlation of effects, biomarkers, and cluster structures across cohorts.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, fisher_exact, chi2_contingency
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class CrossCohortResult:
    """Results of cross-cohort analysis"""
    effect_correlation: float
    effect_direction_agreement: float
    biomarker_correlation: Dict[str, float]
    cluster_agreement: float
    replicated_features: List[str]
    cohort_specific_features: Dict[str, List[str]]
    metadata: Dict = field(default_factory=dict)


@dataclass
class EffectSizeComparison:
    """Comparison of effect sizes across cohorts"""
    feature: str
    ref_effect_size: float
    ext_effect_size: float
    correlation: float
    direction_match: bool
    p_value_ref: float
    p_value_ext: float
    replicated: bool


class CrossCohortAnalyzer:
    """Analyzes replication across cohorts"""

    def __init__(
        self,
        replication_threshold: float = 0.05,
        correlation_threshold: float = 0.3,
        effect_size_threshold: float = 0.2,
    ):
        """
        Initialize analyzer

        Args:
            replication_threshold: P-value threshold for replication
            correlation_threshold: Minimum correlation for replication
            effect_size_threshold: Minimum effect size to consider
        """
        self.replication_threshold = replication_threshold
        self.correlation_threshold = correlation_threshold
        self.effect_size_threshold = effect_size_threshold

    def analyze_cross_cohort(
        self,
        reference_data: pd.DataFrame,
        reference_labels: np.ndarray,
        external_data: pd.DataFrame,
        external_labels: np.ndarray,
        reference_effect_sizes: Optional[pd.DataFrame] = None,
        external_effect_sizes: Optional[pd.DataFrame] = None,
    ) -> CrossCohortResult:
        """
        Perform cross-cohort analysis

        Args:
            reference_data: Reference cohort data
            reference_labels: Reference cluster labels
            external_data: External cohort data
            external_labels: External cluster labels
            reference_effect_sizes: Reference effect sizes per feature
            external_effect_sizes: External effect sizes per feature

        Returns:
            CrossCohortResult
        """
        # Find common features
        common_features = list(
            set(reference_data.columns) & set(external_data.columns)
        )

        # Calculate effect size correlation
        if reference_effect_sizes is not None and external_effect_sizes is not None:
            effect_corr, direction_agreement = self._correlate_effect_sizes(
                reference_effect_sizes,
                external_effect_sizes,
            )
        else:
            # Calculate effect sizes from data
            ref_effects = self._calculate_effect_sizes(
                reference_data[common_features],
                reference_labels,
            )
            ext_effects = self._calculate_effect_sizes(
                external_data[common_features],
                external_labels,
            )
            effect_corr, direction_agreement = self._correlate_effect_sizes(
                ref_effects,
                ext_effects,
            )

        # Calculate biomarker correlations
        biomarker_corrs = self._calculate_biomarker_correlations(
            reference_data[common_features],
            external_data[common_features],
            reference_labels,
            external_labels,
        )

        # Calculate cluster agreement
        cluster_agreement = adjusted_rand_score(reference_labels, external_labels)

        # Identify replicated features
        replicated = self._identify_replicated_features(
            reference_data[common_features],
            external_data[common_features],
            reference_labels,
            external_labels,
        )

        # Identify cohort-specific features
        cohort_specific = self._identify_cohort_specific_features(
            reference_data[common_features],
            external_data[common_features],
            reference_labels,
            external_labels,
        )

        return CrossCohortResult(
            effect_correlation=effect_corr,
            effect_direction_agreement=direction_agreement,
            biomarker_correlation=biomarker_corrs,
            cluster_agreement=cluster_agreement,
            replicated_features=replicated,
            cohort_specific_features=cohort_specific,
            metadata={
                'n_common_features': len(common_features),
                'n_reference_samples': len(reference_data),
                'n_external_samples': len(external_data),
            },
        )

    def _calculate_effect_sizes(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate effect sizes for each feature"""
        from scipy.stats import f_oneway

        effect_sizes = []

        for feature in data.columns:
            groups = [
                data[feature].values[labels == label]
                for label in np.unique(labels)
                if label >= 0
            ]

            if len(groups) < 2:
                continue

            # ANOVA F-statistic as effect size proxy
            f_stat, p_val = f_oneway(*groups)

            # Eta-squared (effect size)
            grand_mean = data[feature].mean()
            ss_between = sum(
                len(g) * (g.mean() - grand_mean) ** 2
                for g in groups
            )
            ss_total = ((data[feature] - grand_mean) ** 2).sum()
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            effect_sizes.append({
                'feature': feature,
                'effect_size': eta_squared,
                'f_statistic': f_stat,
                'p_value': p_val,
            })

        return pd.DataFrame(effect_sizes)

    def _correlate_effect_sizes(
        self,
        ref_effects: pd.DataFrame,
        ext_effects: pd.DataFrame,
    ) -> Tuple[float, float]:
        """Correlate effect sizes between cohorts"""
        # Merge on feature
        merged = ref_effects.merge(
            ext_effects,
            on='feature',
            suffixes=('_ref', '_ext'),
        )

        if len(merged) == 0:
            return 0.0, 0.0

        # Calculate correlation
        correlation, _ = pearsonr(
            merged['effect_size_ref'],
            merged['effect_size_ext'],
        )

        # Calculate direction agreement
        ref_sign = np.sign(merged['effect_size_ref'])
        ext_sign = np.sign(merged['effect_size_ext'])
        direction_agreement = (ref_sign == ext_sign).mean()

        return correlation, direction_agreement

    def _calculate_biomarker_correlations(
        self,
        ref_data: pd.DataFrame,
        ext_data: pd.DataFrame,
        ref_labels: np.ndarray,
        ext_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate per-biomarker correlations"""
        correlations = {}

        for feature in ref_data.columns:
            # Calculate cluster means
            ref_means = [
                ref_data[feature].values[ref_labels == label].mean()
                for label in np.unique(ref_labels)
                if label >= 0
            ]

            ext_means = [
                ext_data[feature].values[ext_labels == label].mean()
                for label in np.unique(ext_labels)
                if label >= 0
            ]

            # Require same number of clusters
            if len(ref_means) != len(ext_means):
                continue

            # Calculate correlation
            if len(ref_means) >= 3:  # Need at least 3 points
                r, p = pearsonr(ref_means, ext_means)
                if p < self.replication_threshold:
                    correlations[feature] = r

        return correlations

    def _identify_replicated_features(
        self,
        ref_data: pd.DataFrame,
        ext_data: pd.DataFrame,
        ref_labels: np.ndarray,
        ext_labels: np.ndarray,
    ) -> List[str]:
        """Identify features that replicate across cohorts"""
        replicated = []

        ref_effects = self._calculate_effect_sizes(ref_data, ref_labels)
        ext_effects = self._calculate_effect_sizes(ext_data, ext_labels)

        merged = ref_effects.merge(ext_effects, on='feature', suffixes=('_ref', '_ext'))

        for _, row in merged.iterrows():
            # Check if significant in both
            ref_sig = row['p_value_ref'] < self.replication_threshold
            ext_sig = row['p_value_ext'] < self.replication_threshold

            # Check if effect sizes similar direction
            same_direction = (
                np.sign(row['effect_size_ref']) == np.sign(row['effect_size_ext'])
            )

            # Check if both exceed threshold
            ref_strong = abs(row['effect_size_ref']) > self.effect_size_threshold
            ext_strong = abs(row['effect_size_ext']) > self.effect_size_threshold

            if ref_sig and ext_sig and same_direction and ref_strong and ext_strong:
                replicated.append(row['feature'])

        return replicated

    def _identify_cohort_specific_features(
        self,
        ref_data: pd.DataFrame,
        ext_data: pd.DataFrame,
        ref_labels: np.ndarray,
        ext_labels: np.ndarray,
    ) -> Dict[str, List[str]]:
        """Identify cohort-specific features"""
        ref_effects = self._calculate_effect_sizes(ref_data, ref_labels)
        ext_effects = self._calculate_effect_sizes(ext_data, ext_labels)

        merged = ref_effects.merge(
            ext_effects,
            on='feature',
            suffixes=('_ref', '_ext'),
            how='outer',
        )

        cohort_specific = {
            'reference_specific': [],
            'external_specific': [],
        }

        for _, row in merged.iterrows():
            ref_sig = (
                row['p_value_ref'] < self.replication_threshold
                if not pd.isna(row['p_value_ref'])
                else False
            )
            ext_sig = (
                row['p_value_ext'] < self.replication_threshold
                if not pd.isna(row['p_value_ext'])
                else False
            )

            if ref_sig and not ext_sig:
                cohort_specific['reference_specific'].append(row['feature'])
            elif ext_sig and not ref_sig:
                cohort_specific['external_specific'].append(row['feature'])

        return cohort_specific

    def compare_effect_sizes(
        self,
        reference_effects: pd.DataFrame,
        external_effects: pd.DataFrame,
    ) -> List[EffectSizeComparison]:
        """
        Compare effect sizes between cohorts

        Args:
            reference_effects: Reference effect sizes
            external_effects: External effect sizes

        Returns:
            List of EffectSizeComparison objects
        """
        comparisons = []

        merged = reference_effects.merge(
            external_effects,
            on='feature',
            suffixes=('_ref', '_ext'),
        )

        for _, row in merged.iterrows():
            # Check replication
            ref_sig = row['p_value_ref'] < self.replication_threshold
            ext_sig = row['p_value_ext'] < self.replication_threshold
            same_direction = (
                np.sign(row['effect_size_ref']) == np.sign(row['effect_size_ext'])
            )

            replicated = ref_sig and ext_sig and same_direction

            # Calculate correlation (if multiple features)
            correlation = pearsonr(
                [row['effect_size_ref']],
                [row['effect_size_ext']],
            )[0] if len(merged) > 1 else np.nan

            comparisons.append(EffectSizeComparison(
                feature=row['feature'],
                ref_effect_size=row['effect_size_ref'],
                ext_effect_size=row['effect_size_ext'],
                correlation=correlation,
                direction_match=same_direction,
                p_value_ref=row['p_value_ref'],
                p_value_ext=row['p_value_ext'],
                replicated=replicated,
            ))

        return comparisons

    def plot_effect_correlation(
        self,
        reference_effects: pd.DataFrame,
        external_effects: pd.DataFrame,
        output_path: Optional[str] = None,
    ):
        """
        Plot correlation of effect sizes

        Args:
            reference_effects: Reference effect sizes
            external_effects: External effect sizes
            output_path: Output path for plot
        """
        merged = reference_effects.merge(
            external_effects,
            on='feature',
            suffixes=('_ref', '_ext'),
        )

        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.scatter(
            merged['effect_size_ref'],
            merged['effect_size_ext'],
            alpha=0.6,
        )

        # Add diagonal
        lims = [
            min(merged['effect_size_ref'].min(), merged['effect_size_ext'].min()),
            max(merged['effect_size_ref'].max(), merged['effect_size_ext'].max()),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3, label='Perfect correlation')

        # Calculate and display correlation
        r, p = pearsonr(merged['effect_size_ref'], merged['effect_size_ext'])
        ax.text(
            0.05,
            0.95,
            f'r = {r:.3f}\np = {p:.2e}',
            transform=ax.transAxes,
            va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

        ax.set_xlabel('Reference Effect Size (η²)')
        ax.set_ylabel('External Effect Size (η²)')
        ax.set_title('Cross-Cohort Effect Size Correlation')
        ax.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def test_heterogeneity(
    cohort_effects: List[pd.DataFrame],
    cohort_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Test for heterogeneity across cohorts (I² statistic)

    Args:
        cohort_effects: List of effect size DataFrames per cohort
        cohort_names: Names of cohorts

    Returns:
        DataFrame with heterogeneity statistics
    """
    from scipy.stats import chi2

    if cohort_names is None:
        cohort_names = [f'Cohort {i+1}' for i in range(len(cohort_effects))]

    # Get common features
    common_features = set(cohort_effects[0]['feature'])
    for effects in cohort_effects[1:]:
        common_features &= set(effects['feature'])

    common_features = sorted(common_features)

    results = []

    for feature in common_features:
        # Extract effect sizes and sample sizes for this feature
        effects = []
        for cohort_df in cohort_effects:
            row = cohort_df[cohort_df['feature'] == feature].iloc[0]
            effects.append(row['effect_size'])

        # Calculate Q statistic (Cochran's Q)
        mean_effect = np.mean(effects)
        Q = sum((e - mean_effect) ** 2 for e in effects)

        # Degrees of freedom
        k = len(effects)
        df = k - 1

        # P-value
        p_value = 1 - chi2.cdf(Q, df)

        # I² statistic (percentage of variation due to heterogeneity)
        I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

        results.append({
            'feature': feature,
            'Q_statistic': Q,
            'df': df,
            'p_value': p_value,
            'I2': I2,
            'heterogeneity': 'High' if I2 > 75 else 'Moderate' if I2 > 50 else 'Low',
        })

    return pd.DataFrame(results)


def calculate_cross_cohort_stability(
    cohort_embeddings: List[np.ndarray],
    cohort_labels: List[np.ndarray],
) -> Dict[str, float]:
    """
    Calculate stability of clustering across cohorts

    Args:
        cohort_embeddings: List of embeddings per cohort
        cohort_labels: List of cluster labels per cohort

    Returns:
        Dictionary of stability metrics
    """
    n_cohorts = len(cohort_embeddings)

    # Pairwise comparisons
    ari_scores = []
    nmi_scores = []

    for i in range(n_cohorts):
        for j in range(i + 1, n_cohorts):
            # Match sample sizes
            n_samples = min(len(cohort_labels[i]), len(cohort_labels[j]))

            ari = adjusted_rand_score(
                cohort_labels[i][:n_samples],
                cohort_labels[j][:n_samples],
            )
            nmi = normalized_mutual_info_score(
                cohort_labels[i][:n_samples],
                cohort_labels[j][:n_samples],
            )

            ari_scores.append(ari)
            nmi_scores.append(nmi)

    return {
        'ari_mean': np.mean(ari_scores),
        'ari_std': np.std(ari_scores),
        'nmi_mean': np.mean(nmi_scores),
        'nmi_std': np.std(nmi_scores),
        'ari_min': np.min(ari_scores),
        'ari_max': np.max(ari_scores),
    }