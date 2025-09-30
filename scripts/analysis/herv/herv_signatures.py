#!/usr/bin/env python3
"""
Human Endogenous Retrovirus (HERV) Activation Signatures
Novel biomarker class for neurodevelopmental dysregulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class HERVResult:
    """Results from HERV activation analysis"""
    herv_activity: pd.Series
    correlations: Dict[str, Tuple[float, float]]
    subtypes: pd.Series
    top_hervs: pd.DataFrame


class HERVAnalyzer:
    """
    HERV activation signature analysis

    HERVs (Human Endogenous Retroviruses) are normally silenced sequences
    that can be reactivated during neurodevelopment. Activation indicates:
    - Epigenetic dysregulation
    - Immune activation
    - Neurodevelopmental perturbation

    Recent evidence links HERV expression to ASD (Balestrieri 2012, Avramopoulos 2015)
    """

    def __init__(self):
        """Initialize analyzer"""
        self.herv_families = {
            'HERV-K': ['ERVK-1', 'ERVK-2', 'ERVK-3', 'ERVK-4', 'ERVK-5'],
            'HERV-W': ['ERVW-1', 'ERVW-2', 'ERVW-3', 'ERVW-4'],
            'HERV-H': ['ERVH-1', 'ERVH-2', 'ERVH-3'],
            'HERV-FRD': ['ERVFRD-1', 'ERVFRD-2']
        }

    def load_herv_annotations(self) -> List[str]:
        """
        Load HERV gene annotations

        In practice, would use:
        - RepeatMasker annotations
        - Dfam database
        - RetroTector identification

        Returns
        -------
        herv_genes : List[str]
            HERV loci identifiers
        """
        logger.info("Loading HERV annotations")

        # Mock HERV genes (in practice, load from annotation)
        herv_genes = []
        for family, members in self.herv_families.items():
            herv_genes.extend(members)

        logger.info(f"  Loaded {len(herv_genes)} HERV loci across {len(self.herv_families)} families")

        return herv_genes

    def compute_herv_activity(
        self,
        rna_seq: pd.DataFrame,
        herv_genes: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Compute HERV activation score per sample

        Parameters
        ----------
        rna_seq : pd.DataFrame
            RNA-seq expression (samples × genes)
        herv_genes : List[str], optional
            HERV genes to include

        Returns
        -------
        herv_activity : pd.Series
            Overall HERV activation per sample
        """
        logger.info("Computing HERV activation scores")

        if herv_genes is None:
            herv_genes = self.load_herv_annotations()

        # Filter to HERVs present in data
        herv_genes_present = [g for g in herv_genes if g in rna_seq.columns]

        if len(herv_genes_present) == 0:
            logger.warning("No HERV genes found in expression data")
            return pd.Series(0, index=rna_seq.index)

        logger.info(f"  Found {len(herv_genes_present)} HERV genes in data")

        # Aggregate HERV expression
        # Normally silenced, so any expression = activation
        herv_expression = rna_seq[herv_genes_present]

        # Overall activation score: mean log-expression
        herv_activity = herv_expression.mean(axis=1)

        logger.info(f"  HERV activity range: {herv_activity.min():.2f} - {herv_activity.max():.2f}")

        return herv_activity

    def correlate_with_phenotypes(
        self,
        herv_activity: pd.Series,
        phenotypes: pd.DataFrame,
        method: str = 'spearman'
    ) -> Dict[str, Tuple[float, float]]:
        """
        Correlate HERV activation with AuDHD phenotypes

        Parameters
        ----------
        herv_activity : pd.Series
            HERV activation scores
        phenotypes : pd.DataFrame
            Clinical phenotypes
        method : str
            Correlation method

        Returns
        -------
        correlations : Dict[str, Tuple[float, float]]
            Phenotype → (correlation, p-value)
        """
        logger.info("Correlating HERV activity with phenotypes")

        # Align samples
        common_samples = herv_activity.index.intersection(phenotypes.index)
        herv_aligned = herv_activity.loc[common_samples]
        phenotypes_aligned = phenotypes.loc[common_samples]

        correlations = {}

        for phenotype_col in phenotypes_aligned.columns:
            y = phenotypes_aligned[phenotype_col].values

            # Remove missing
            mask = ~np.isnan(y)
            if mask.sum() < 10:
                continue

            x_clean = herv_aligned[mask].values
            y_clean = y[mask]

            # Correlation
            if method == 'spearman':
                corr, p_val = stats.spearmanr(x_clean, y_clean)
            else:
                corr, p_val = stats.pearsonr(x_clean, y_clean)

            correlations[phenotype_col] = (corr, p_val)

            logger.info(f"  {phenotype_col}: r={corr:.3f}, p={p_val:.4f}")

        return correlations

    def identify_herv_subtypes(
        self,
        herv_activity: pd.Series,
        correlations: Dict[str, Tuple[float, float]],
        n_clusters: int = 3
    ) -> pd.Series:
        """
        Identify HERV-based subtypes

        Stratifies samples by HERV activation patterns

        Parameters
        ----------
        herv_activity : pd.Series
        correlations : Dict
            Phenotype correlations
        n_clusters : int
            Number of subtypes

        Returns
        -------
        subtypes : pd.Series
            Subtype labels per sample
        """
        logger.info(f"Identifying {n_clusters} HERV-based subtypes")

        from sklearn.cluster import KMeans

        # Cluster by HERV activity
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        subtypes = kmeans.fit_predict(herv_activity.values.reshape(-1, 1))

        # Label subtypes by activation level
        subtype_means = []
        for i in range(n_clusters):
            subtype_means.append((i, herv_activity[subtypes == i].mean()))

        subtype_means.sort(key=lambda x: x[1])

        # Rename: Low → Medium → High
        subtype_labels = ['Low-HERV', 'Medium-HERV', 'High-HERV']
        subtype_map = {old: new for (old, _), new in zip(subtype_means, subtype_labels)}

        subtypes_labeled = pd.Series(
            [subtype_map[s] for s in subtypes],
            index=herv_activity.index
        )

        # Report subtype sizes
        for label in subtype_labels:
            count = (subtypes_labeled == label).sum()
            mean_activity = herv_activity[subtypes_labeled == label].mean()
            logger.info(f"  {label}: n={count}, mean_activity={mean_activity:.2f}")

        return subtypes_labeled

    def test_family_specificity(
        self,
        rna_seq: pd.DataFrame,
        phenotypes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Test which HERV families are most associated with phenotypes

        Parameters
        ----------
        rna_seq : pd.DataFrame
        phenotypes : pd.DataFrame

        Returns
        -------
        family_results : pd.DataFrame
            Columns: family, phenotype, correlation, p_value
        """
        logger.info("Testing HERV family specificity")

        results = []

        for family_name, family_members in self.herv_families.items():
            # Family-specific activity
            family_genes = [g for g in family_members if g in rna_seq.columns]

            if len(family_genes) == 0:
                continue

            family_activity = rna_seq[family_genes].mean(axis=1)

            # Correlate with phenotypes
            for phenotype_col in phenotypes.columns:
                common_samples = family_activity.index.intersection(phenotypes.index)

                if len(common_samples) < 10:
                    continue

                x = family_activity.loc[common_samples].values
                y = phenotypes.loc[common_samples, phenotype_col].values

                mask = ~np.isnan(y)
                if mask.sum() < 10:
                    continue

                corr, p_val = stats.spearmanr(x[mask], y[mask])

                results.append({
                    'family': family_name,
                    'phenotype': phenotype_col,
                    'correlation': corr,
                    'p_value': p_val,
                    'n_genes': len(family_genes)
                })

        family_df = pd.DataFrame(results).sort_values('p_value')

        # FDR correction
        from statsmodels.stats.multitest import multipletests
        _, qvals, _, _ = multipletests(family_df['p_value'], method='fdr_bh')
        family_df['q_value'] = qvals

        logger.info(f"  Significant family-phenotype associations (q<0.05): {(qvals < 0.05).sum()}")

        return family_df

    def identify_top_hervs(
        self,
        rna_seq: pd.DataFrame,
        case_control: pd.Series,
        herv_genes: Optional[List[str]] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Identify most differentially expressed HERVs

        Parameters
        ----------
        rna_seq : pd.DataFrame
        case_control : pd.Series
            Case/control labels
        herv_genes : List[str], optional
        top_n : int
            Number of top HERVs to return

        Returns
        -------
        top_hervs : pd.DataFrame
            Top differentially expressed HERVs
        """
        logger.info("Identifying top differentially expressed HERVs")

        if herv_genes is None:
            herv_genes = self.load_herv_annotations()

        herv_genes_present = [g for g in herv_genes if g in rna_seq.columns]

        # Test each HERV
        results = []

        for herv in herv_genes_present:
            expr = rna_seq[herv]

            # Align with case/control
            common_samples = expr.index.intersection(case_control.index)
            expr_aligned = expr.loc[common_samples]
            labels_aligned = case_control.loc[common_samples]

            # Case vs control
            case_expr = expr_aligned[labels_aligned == 'case']
            ctrl_expr = expr_aligned[labels_aligned == 'control']

            if len(case_expr) < 5 or len(ctrl_expr) < 5:
                continue

            # T-test
            t_stat, p_val = stats.ttest_ind(case_expr, ctrl_expr)

            # Log fold change
            logfc = np.log2(case_expr.mean() / ctrl_expr.mean()) if ctrl_expr.mean() > 0 else 0

            results.append({
                'herv': herv,
                'logFC': logfc,
                't_statistic': t_stat,
                'p_value': p_val,
                'mean_case': case_expr.mean(),
                'mean_control': ctrl_expr.mean()
            })

        top_df = pd.DataFrame(results).sort_values('p_value').head(top_n)

        # FDR correction
        from statsmodels.stats.multitest import multipletests
        _, qvals, _, _ = multipletests(top_df['p_value'], method='fdr_bh')
        top_df['q_value'] = qvals

        logger.info(f"  Top HERV: {top_df.iloc[0]['herv']} (logFC={top_df.iloc[0]['logFC']:.2f}, p={top_df.iloc[0]['p_value']:.2e})")

        return top_df

    def integrate_with_baseline_deviation(
        self,
        herv_activity: pd.Series,
        baseline_deviation_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Test if HERV activation correlates with deviation from baseline

        Parameters
        ----------
        herv_activity : pd.Series
        baseline_deviation_results : pd.DataFrame

        Returns
        -------
        integration_results : pd.DataFrame
        """
        logger.info("Integrating HERV activity with baseline-deviation framework")

        # Align samples
        common_samples = herv_activity.index.intersection(
            baseline_deviation_results.index
        )

        herv_aligned = herv_activity.loc[common_samples]
        deviation_scores = baseline_deviation_results.loc[common_samples, 'deviation_score']

        # Correlation
        corr, p_val = stats.spearmanr(herv_aligned, deviation_scores)

        logger.info(f"  HERV-deviation correlation: r={corr:.3f}, p={p_val:.4f}")

        # Test by deviation category
        high_deviation = deviation_scores > deviation_scores.quantile(0.75)
        low_deviation = deviation_scores < deviation_scores.quantile(0.25)

        herv_high = herv_aligned[high_deviation]
        herv_low = herv_aligned[low_deviation]

        t_stat, p_t = stats.ttest_ind(herv_high, herv_low)

        logger.info(f"  HERV in high vs low deviation: t={t_stat:.2f}, p={p_t:.4f}")

        results = pd.DataFrame([{
            'correlation': corr,
            'p_correlation': p_val,
            't_statistic': t_stat,
            'p_ttest': p_t,
            'mean_herv_high_deviation': herv_high.mean(),
            'mean_herv_low_deviation': herv_low.mean()
        }])

        return results

    def analyze_complete(
        self,
        rna_seq: pd.DataFrame,
        phenotypes: pd.DataFrame,
        case_control: Optional[pd.Series] = None
    ) -> HERVResult:
        """
        Complete HERV activation analysis

        Parameters
        ----------
        rna_seq : pd.DataFrame
        phenotypes : pd.DataFrame
        case_control : pd.Series, optional

        Returns
        -------
        HERVResult
        """
        logger.info("=== Complete HERV Activation Analysis ===")

        # 1. Compute HERV activity
        herv_genes = self.load_herv_annotations()
        herv_activity = self.compute_herv_activity(rna_seq, herv_genes)

        # 2. Correlate with phenotypes
        correlations = self.correlate_with_phenotypes(herv_activity, phenotypes)

        # 3. Identify subtypes
        subtypes = self.identify_herv_subtypes(herv_activity, correlations, n_clusters=3)

        # 4. Top HERVs
        if case_control is not None:
            top_hervs = self.identify_top_hervs(rna_seq, case_control, herv_genes, top_n=20)
        else:
            top_hervs = pd.DataFrame()

        return HERVResult(
            herv_activity=herv_activity,
            correlations=correlations,
            subtypes=subtypes,
            top_hervs=top_hervs
        )


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    logger.info("HERV Activation Signatures Module")
    logger.info("Novel biomarker class for neurodevelopmental dysregulation")
    logger.info("\nKey capabilities:")
    logger.info("  1. HERV activation scoring")
    logger.info("  2. Phenotype correlation analysis")
    logger.info("  3. HERV-based subtype identification")
    logger.info("  4. Family-specific effects (HERV-K, -W, -H, -FRD)")
    logger.info("  5. Top differentially expressed HERVs")
    logger.info("  6. Integration with baseline-deviation framework")
    logger.info("\nHERVs are normally silenced - activation indicates dysregulation")
