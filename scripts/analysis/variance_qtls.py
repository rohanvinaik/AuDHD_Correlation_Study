#!/usr/bin/env python3
"""
Variance QTL (vQTL) Analysis using MZ Twin Differences
Identifies genetic variants affecting trait variability rather than mean levels
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats
from statsmodels.stats.multitest import multipletests
import logging

logger = logging.getLogger(__name__)


@dataclass
class vQTLResult:
    """Results from variance QTL analysis"""
    results: pd.DataFrame
    significant_vqtls: pd.DataFrame
    n_tests: int
    n_significant: int
    fdr_threshold: float


def analyze_variance_qtls(
    mz_twin_differences: pd.DataFrame,
    genotypes: pd.DataFrame,
    trait_cols: List[str],
    snp_cols: List[str],
    fdr_threshold: float = 0.05,
    min_group_size: int = 5
) -> Dict[str, vQTLResult]:
    """
    Identify variance QTLs using MZ twin differences

    MZ twins share 100% of genetics, so differences between them reflect
    environmental and stochastic factors. Testing if genetic variants
    associate with variance in these differences identifies vQTLs.

    This is superior to standard vQTL approaches because:
    1. Completely controls for genetic background
    2. Removes mean genetic effects
    3. More power to detect variance effects

    Parameters
    ----------
    mz_twin_differences : pd.DataFrame
        Within-pair differences for MZ twins (one row per twin pair)
    genotypes : pd.DataFrame
        Genotype data (0/1/2 coding)
    trait_cols : List[str]
        Phenotype columns to test
    snp_cols : List[str]
        SNP columns to test
    fdr_threshold : float
        FDR threshold for significance
    min_group_size : int
        Minimum samples per genotype group

    Returns
    -------
    Dict mapping trait names to vQTLResult objects
    """
    logger.info(f"Running vQTL analysis: {len(trait_cols)} traits × {len(snp_cols)} SNPs")

    results_by_trait = {}

    for trait in trait_cols:
        logger.info(f"  Testing trait: {trait}")

        trait_diffs = mz_twin_differences[trait].values

        # Square differences = proxy for variance
        squared_diffs = trait_diffs ** 2

        vqtl_results = []

        for snp in snp_cols:
            snp_genotypes = genotypes[snp].values

            # Skip if missing data
            valid_mask = ~(np.isnan(trait_diffs) | np.isnan(snp_genotypes))
            if valid_mask.sum() < min_group_size * 2:
                continue

            squared_diffs_valid = squared_diffs[valid_mask]
            geno_valid = snp_genotypes[valid_mask]

            # Group by genotype
            unique_genos = np.unique(geno_valid)

            if len(unique_genos) < 2:
                continue

            groups = [squared_diffs_valid[geno_valid == g] for g in unique_genos]

            # Filter groups with insufficient size
            groups = [g for g in groups if len(g) >= min_group_size]

            if len(groups) < 2:
                continue

            # Levene's test for variance differences
            stat, pval = stats.levene(*groups)

            # Also compute effect size (ratio of variances)
            group_vars = [np.var(g) for g in groups]
            var_ratio = max(group_vars) / min(group_vars) if min(group_vars) > 0 else np.nan

            # Compute mean effect (should be near zero for true vQTL)
            mean_effect = stats.kruskal(*groups)[1]  # Non-parametric test of means

            vqtl_results.append({
                'snp': snp,
                'trait': trait,
                'levene_stat': stat,
                'p_value': pval,
                'variance_ratio': var_ratio,
                'mean_effect_p': mean_effect,
                'n_groups': len(groups),
                'group_sizes': [len(g) for g in groups],
                'group_variances': group_vars
            })

        if not vqtl_results:
            logger.warning(f"  No valid tests for {trait}")
            continue

        results_df = pd.DataFrame(vqtl_results)

        # FDR correction
        _, qvals, _, _ = multipletests(results_df['p_value'].values, method='fdr_bh')
        results_df['q_value'] = qvals

        # Identify significant vQTLs
        sig_mask = (results_df['q_value'] < fdr_threshold)
        significant = results_df[sig_mask].copy()
        significant = significant.sort_values('p_value')

        logger.info(f"  Found {len(significant)} significant vQTLs (FDR < {fdr_threshold})")

        results_by_trait[trait] = vQTLResult(
            results=results_df,
            significant_vqtls=significant,
            n_tests=len(results_df),
            n_significant=len(significant),
            fdr_threshold=fdr_threshold
        )

    return results_by_trait


def compute_twin_differences(
    twin_data: pd.DataFrame,
    twin_pair_id_col: str,
    twin_order_col: str,
    trait_cols: List[str]
) -> pd.DataFrame:
    """
    Compute within-pair differences for MZ twins

    Parameters
    ----------
    twin_data : pd.DataFrame
        Twin data in long format (one row per individual)
    twin_pair_id_col : str
        Column identifying twin pairs
    twin_order_col : str
        Column distinguishing twins (e.g., 'twin_1' vs 'twin_2')
    trait_cols : List[str]
        Traits to compute differences for

    Returns
    -------
    pd.DataFrame
        One row per twin pair with difference scores
    """
    logger.info(f"Computing twin differences for {len(trait_cols)} traits")

    # Pivot to wide format
    twin_wide = twin_data.pivot(
        index=twin_pair_id_col,
        columns=twin_order_col,
        values=trait_cols
    )

    # Compute differences (twin_1 - twin_2)
    differences = {}
    differences['pair_id'] = twin_wide.index

    for trait in trait_cols:
        try:
            diff = twin_wide[(trait, 'twin_1')] - twin_wide[(trait, 'twin_2')]
            differences[f'{trait}_diff'] = diff.values
        except KeyError:
            logger.warning(f"Could not compute difference for {trait}")
            continue

    diff_df = pd.DataFrame(differences)

    logger.info(f"Computed differences for {len(diff_df)} twin pairs")

    return diff_df


def compare_vqtl_vs_mqtl(
    vqtl_results: vQTLResult,
    mqtl_results: pd.DataFrame,
    trait: str
) -> pd.DataFrame:
    """
    Compare variance QTLs vs mean QTLs

    Interesting cases:
    - vQTL but not mQTL: Affects variability only (gene-environment interaction)
    - Both vQTL and mQTL: Affects both mean and variance (scale effect)
    - mQTL but not vQTL: Standard additive effect

    Parameters
    ----------
    vqtl_results : vQTLResult
        vQTL analysis results
    mqtl_results : pd.DataFrame
        Standard GWAS/QTL results with p-values
    trait : str
        Trait name

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    # Merge results
    vqtl_df = vqtl_results.results[['snp', 'p_value', 'q_value']].copy()
    vqtl_df.columns = ['snp', 'vqtl_p', 'vqtl_q']

    mqtl_df = mqtl_results[['snp', 'p_value', 'q_value']].copy()
    mqtl_df.columns = ['snp', 'mqtl_p', 'mqtl_q']

    comparison = vqtl_df.merge(mqtl_df, on='snp', how='outer')

    # Classify
    comparison['is_vqtl'] = comparison['vqtl_q'] < 0.05
    comparison['is_mqtl'] = comparison['mqtl_q'] < 0.05

    comparison['category'] = 'Neither'
    comparison.loc[comparison['is_vqtl'] & ~comparison['is_mqtl'], 'category'] = 'vQTL only (GxE)'
    comparison.loc[~comparison['is_vqtl'] & comparison['is_mqtl'], 'category'] = 'mQTL only'
    comparison.loc[comparison['is_vqtl'] & comparison['is_mqtl'], 'category'] = 'Both (scale effect)'

    logger.info(f"\nvQTL vs mQTL comparison for {trait}:")
    logger.info(comparison['category'].value_counts().to_string())

    return comparison


def estimate_gxe_contribution(
    vqtl_results: Dict[str, vQTLResult],
    variance_partition: pd.DataFrame
) -> pd.DataFrame:
    """
    Estimate contribution of GxE to phenotypic variance

    vQTLs that don't affect mean represent GxE interactions.
    Quantify how much variance they explain.

    Parameters
    ----------
    vqtl_results : Dict
        vQTL results for each trait
    variance_partition : pd.DataFrame
        Variance components from other analyses

    Returns
    -------
    pd.DataFrame
        GxE contribution estimates
    """
    gxe_estimates = []

    for trait, result in vqtl_results.items():
        # Pure vQTLs (low mean effect p-value)
        pure_vqtls = result.significant_vqtls[
            result.significant_vqtls['mean_effect_p'] > 0.05
        ]

        n_pure_vqtls = len(pure_vqtls)

        if n_pure_vqtls > 0:
            # Estimate variance explained by vQTLs
            # Sum of variance ratios weighted by significance
            weights = -np.log10(pure_vqtls['p_value'])
            weighted_var_ratio = (pure_vqtls['variance_ratio'] * weights).sum() / weights.sum()

            gxe_estimates.append({
                'trait': trait,
                'n_vqtls': n_pure_vqtls,
                'mean_variance_ratio': weighted_var_ratio,
                'estimated_gxe_contribution': (weighted_var_ratio - 1) / weighted_var_ratio
            })

    return pd.DataFrame(gxe_estimates)


if __name__ == '__main__':
    # Example usage with simulated data
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)

    # Simulate MZ twin data
    n_pairs = 200
    n_snps = 100

    # Simulate genotypes (one per pair since MZ twins share genotype)
    genotypes = pd.DataFrame({
        f'SNP_{i}': np.random.choice([0, 1, 2], size=n_pairs, p=[0.25, 0.5, 0.25])
        for i in range(n_snps)
    })

    # Simulate trait with vQTL effect at SNP_10
    # SNP_10 genotype affects variance but not mean
    trait_diffs = []
    for i in range(n_pairs):
        if genotypes.loc[i, 'SNP_10'] == 0:
            diff = np.random.randn() * 0.5  # Low variance
        elif genotypes.loc[i, 'SNP_10'] == 1:
            diff = np.random.randn() * 1.0  # Medium variance
        else:
            diff = np.random.randn() * 2.0  # High variance
        trait_diffs.append(diff)

    mz_diffs = pd.DataFrame({
        'trait_1': trait_diffs,
        'trait_2': np.random.randn(n_pairs)  # No vQTL
    })

    # Run analysis
    results = analyze_variance_qtls(
        mz_twin_differences=mz_diffs,
        genotypes=genotypes,
        trait_cols=['trait_1', 'trait_2'],
        snp_cols=[f'SNP_{i}' for i in range(n_snps)],
        fdr_threshold=0.05
    )

    print("\n=== vQTL Analysis Results ===")
    for trait, result in results.items():
        print(f"\n{trait}:")
        print(f"  Total tests: {result.n_tests}")
        print(f"  Significant: {result.n_significant}")
        if result.n_significant > 0:
            print(f"\nTop vQTLs:")
            print(result.significant_vqtls.head())
