"""Multi-omics pathway integration

Integrates pathway enrichment results across multiple omics layers.
"""
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict


@dataclass
class MultiOmicsPathwayResult:
    """Result of multi-omics pathway integration"""
    pathway_id: str
    pathway_name: str

    # Per-omics enrichment
    genomics_nes: Optional[float] = None
    transcriptomics_nes: Optional[float] = None
    proteomics_nes: Optional[float] = None
    metabolomics_nes: Optional[float] = None

    genomics_fdr: Optional[float] = None
    transcriptomics_fdr: Optional[float] = None
    proteomics_fdr: Optional[float] = None
    metabolomics_fdr: Optional[float] = None

    # Integrated scores
    combined_score: float = 0.0
    n_omics_significant: int = 0
    consistency_score: float = 0.0

    # Cross-omics evidence
    cross_omics_genes: Optional[Set[str]] = None
    omics_agreement: Optional[float] = None


@dataclass
class MultiOmicsPathwayResults:
    """Collection of multi-omics pathway results"""
    results: List[MultiOmicsPathwayResult]
    integration_method: str

    def top_pathways(self, n: int = 20) -> List[MultiOmicsPathwayResult]:
        """Return top N pathways by combined score"""
        sorted_results = sorted(
            self.results,
            key=lambda x: x.combined_score,
            reverse=True
        )
        return sorted_results[:n]

    def consistent_pathways(self, min_omics: int = 2) -> List[MultiOmicsPathwayResult]:
        """Return pathways enriched in at least min_omics layers"""
        return [
            r for r in self.results
            if r.n_omics_significant >= min_omics
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        data = []
        for r in self.results:
            data.append({
                'pathway_id': r.pathway_id,
                'pathway_name': r.pathway_name,
                'combined_score': r.combined_score,
                'n_omics_significant': r.n_omics_significant,
                'consistency_score': r.consistency_score,
                'genomics_NES': r.genomics_nes,
                'transcriptomics_NES': r.transcriptomics_nes,
                'proteomics_NES': r.proteomics_nes,
                'metabolomics_NES': r.metabolomics_nes,
                'genomics_FDR': r.genomics_fdr,
                'transcriptomics_FDR': r.transcriptomics_fdr,
                'proteomics_FDR': r.proteomics_fdr,
                'metabolomics_FDR': r.metabolomics_fdr,
            })
        return pd.DataFrame(data)


def integrate_multiomics_pathways(
    genomics_results: Optional[pd.DataFrame] = None,
    transcriptomics_results: Optional[pd.DataFrame] = None,
    proteomics_results: Optional[pd.DataFrame] = None,
    metabolomics_results: Optional[pd.DataFrame] = None,
    method: str = 'weighted_average',
    fdr_threshold: float = 0.25,
    weights: Optional[Dict[str, float]] = None,
) -> MultiOmicsPathwayResults:
    """
    Integrate pathway enrichment across multiple omics

    Args:
        genomics_results: Genomics enrichment (pathway_name, NES, FDR)
        transcriptomics_results: Transcriptomics enrichment
        proteomics_results: Proteomics enrichment
        metabolomics_results: Metabolomics enrichment
        method: Integration method ('weighted_average', 'rank_aggregation', 'stouffer')
        fdr_threshold: FDR threshold for significance
        weights: Weights for each omics layer (if None, use equal weights)

    Returns:
        MultiOmicsPathwayResults with integrated scores
    """
    # Collect all results
    omics_results = {}
    if genomics_results is not None:
        omics_results['genomics'] = genomics_results
    if transcriptomics_results is not None:
        omics_results['transcriptomics'] = transcriptomics_results
    if proteomics_results is not None:
        omics_results['proteomics'] = proteomics_results
    if metabolomics_results is not None:
        omics_results['metabolomics'] = metabolomics_results

    if not omics_results:
        raise ValueError("At least one omics layer is required")

    # Get all unique pathways
    all_pathways = set()
    for results in omics_results.values():
        if 'pathway_name' in results.columns:
            all_pathways.update(results['pathway_name'].unique())
        elif 'pathway_id' in results.columns:
            all_pathways.update(results['pathway_id'].unique())

    # Default weights
    if weights is None:
        weights = {omics: 1.0 for omics in omics_results.keys()}

    # Normalize weights
    weight_sum = sum(weights.values())
    weights = {k: v / weight_sum for k, v in weights.items()}

    # Integrate pathways
    integrated_results = []

    for pathway in all_pathways:
        # Extract pathway ID and name
        if ':' in pathway:
            pathway_id, pathway_name = pathway.split(':', 1)
        else:
            pathway_id = pathway
            pathway_name = pathway

        # Collect scores from each omics
        omics_scores = {}
        omics_fdrs = {}

        for omics_name, results in omics_results.items():
            # Find pathway in results
            if 'pathway_name' in results.columns:
                pathway_data = results[results['pathway_name'] == pathway]
            elif 'pathway_id' in results.columns:
                pathway_data = results[results['pathway_id'] == pathway_id]
            else:
                continue

            if len(pathway_data) > 0:
                row = pathway_data.iloc[0]
                nes = row.get('NES', row.get('nes', None))
                fdr = row.get('FDR', row.get('fdr', None))

                if nes is not None:
                    omics_scores[omics_name] = float(nes)
                if fdr is not None:
                    omics_fdrs[omics_name] = float(fdr)

        # Calculate combined score
        if method == 'weighted_average':
            combined_score = _weighted_average_integration(
                omics_scores, weights
            )
        elif method == 'rank_aggregation':
            combined_score = _rank_aggregation_integration(
                omics_scores, omics_results
            )
        elif method == 'stouffer':
            combined_score = _stouffer_integration(
                omics_scores, omics_fdrs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Count significant omics
        n_omics_significant = sum(
            fdr <= fdr_threshold for fdr in omics_fdrs.values()
        )

        # Calculate consistency score
        consistency_score = _calculate_consistency(omics_scores)

        # Create result
        result = MultiOmicsPathwayResult(
            pathway_id=pathway_id,
            pathway_name=pathway_name,
            genomics_nes=omics_scores.get('genomics'),
            transcriptomics_nes=omics_scores.get('transcriptomics'),
            proteomics_nes=omics_scores.get('proteomics'),
            metabolomics_nes=omics_scores.get('metabolomics'),
            genomics_fdr=omics_fdrs.get('genomics'),
            transcriptomics_fdr=omics_fdrs.get('transcriptomics'),
            proteomics_fdr=omics_fdrs.get('proteomics'),
            metabolomics_fdr=omics_fdrs.get('metabolomics'),
            combined_score=combined_score,
            n_omics_significant=n_omics_significant,
            consistency_score=consistency_score,
        )

        integrated_results.append(result)

    # Sort by combined score
    integrated_results.sort(key=lambda x: x.combined_score, reverse=True)

    return MultiOmicsPathwayResults(
        results=integrated_results,
        integration_method=method,
    )


def _weighted_average_integration(
    omics_scores: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Weighted average of NES scores"""
    if not omics_scores:
        return 0.0

    # Only use weights for omics layers with scores
    available_weights = {
        omics: weights.get(omics, 1.0)
        for omics in omics_scores.keys()
    }

    # Normalize available weights
    weight_sum = sum(available_weights.values())
    if weight_sum == 0:
        return 0.0

    available_weights = {k: v / weight_sum for k, v in available_weights.items()}

    # Calculate weighted average
    combined = sum(
        omics_scores[omics] * available_weights[omics]
        for omics in omics_scores.keys()
    )

    return float(combined)


def _rank_aggregation_integration(
    omics_scores: Dict[str, float],
    all_omics_results: Dict[str, pd.DataFrame],
) -> float:
    """Rank-based aggregation (Borda count)"""
    if not omics_scores:
        return 0.0

    ranks = []

    for omics_name, score in omics_scores.items():
        # Get all scores for this omics
        results = all_omics_results[omics_name]
        if 'NES' in results.columns:
            all_scores = results['NES'].values
        elif 'nes' in results.columns:
            all_scores = results['nes'].values
        else:
            continue

        # Rank (higher score = better rank)
        rank = (all_scores < score).sum()
        normalized_rank = rank / len(all_scores) if len(all_scores) > 0 else 0.5

        ranks.append(normalized_rank)

    if not ranks:
        return 0.0

    # Average rank
    combined_rank = np.mean(ranks)

    return float(combined_rank)


def _stouffer_integration(
    omics_scores: Dict[str, float],
    omics_fdrs: Dict[str, float],
) -> float:
    """Stouffer's Z-score method"""
    if not omics_fdrs:
        return 0.0

    z_scores = []

    for omics_name, fdr in omics_fdrs.items():
        # Convert FDR to Z-score
        # FDR is two-tailed, so convert to one-tailed p-value
        p_value = fdr / 2

        # Clip to avoid numerical issues
        p_value = np.clip(p_value, 1e-10, 1 - 1e-10)

        # Convert to Z-score
        z = stats.norm.ppf(1 - p_value)

        # Adjust sign based on NES
        if omics_name in omics_scores:
            if omics_scores[omics_name] < 0:
                z = -z

        z_scores.append(z)

    if not z_scores:
        return 0.0

    # Combined Z-score
    combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))

    return float(combined_z)


def _calculate_consistency(omics_scores: Dict[str, float]) -> float:
    """Calculate consistency of enrichment direction across omics"""
    if len(omics_scores) < 2:
        return 1.0

    scores = list(omics_scores.values())

    # All positive or all negative = high consistency
    all_positive = all(s > 0 for s in scores)
    all_negative = all(s < 0 for s in scores)

    if all_positive or all_negative:
        # Measure agreement in magnitude
        scores_abs = np.abs(scores)
        cv = np.std(scores_abs) / (np.mean(scores_abs) + 1e-10)
        consistency = 1.0 / (1.0 + cv)
    else:
        # Mixed directions = low consistency
        n_positive = sum(s > 0 for s in scores)
        n_negative = sum(s < 0 for s in scores)
        consistency = abs(n_positive - n_negative) / len(scores)

    return float(consistency)


def combined_pathway_score(
    pathway_results: List[MultiOmicsPathwayResult],
    score_type: str = 'combined',
) -> pd.Series:
    """
    Extract scores from pathway results

    Args:
        pathway_results: List of MultiOmicsPathwayResult
        score_type: Score type ('combined', 'consistency', 'n_omics')

    Returns:
        Series of scores indexed by pathway name
    """
    if score_type == 'combined':
        scores = {r.pathway_name: r.combined_score for r in pathway_results}
    elif score_type == 'consistency':
        scores = {r.pathway_name: r.consistency_score for r in pathway_results}
    elif score_type == 'n_omics':
        scores = {r.pathway_name: r.n_omics_significant for r in pathway_results}
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    return pd.Series(scores)


def cross_omics_enrichment(
    genomics_genes: Set[str],
    transcriptomics_genes: Set[str],
    proteomics_genes: Set[str],
    pathway_gene_sets: Dict[str, Set[str]],
    min_overlap: int = 3,
) -> pd.DataFrame:
    """
    Test for pathways enriched across multiple omics layers

    Args:
        genomics_genes: Genes from genomics analysis
        transcriptomics_genes: Genes from transcriptomics
        proteomics_genes: Genes from proteomics
        pathway_gene_sets: Pathway definitions
        min_overlap: Minimum genes required per pathway

    Returns:
        DataFrame with cross-omics pathway enrichment
    """
    results = []

    for pathway_name, pathway_genes in pathway_gene_sets.items():
        # Overlap with each omics
        genomics_overlap = genomics_genes & pathway_genes
        transcriptomics_overlap = transcriptomics_genes & pathway_genes
        proteomics_overlap = proteomics_genes & pathway_genes

        # Cross-omics overlap
        cross_omics = genomics_overlap & transcriptomics_overlap & proteomics_overlap

        if len(cross_omics) >= min_overlap:
            # Hypergeometric test for each omics
            # (simplified - assumes same background size)
            background_size = 20000  # Human genome size (approx)

            # Genomics
            p_genomics = stats.hypergeom.sf(
                len(genomics_overlap) - 1,
                background_size,
                len(pathway_genes),
                len(genomics_genes),
            )

            # Transcriptomics
            p_transcriptomics = stats.hypergeom.sf(
                len(transcriptomics_overlap) - 1,
                background_size,
                len(pathway_genes),
                len(transcriptomics_genes),
            )

            # Proteomics
            p_proteomics = stats.hypergeom.sf(
                len(proteomics_overlap) - 1,
                background_size,
                len(pathway_genes),
                len(proteomics_genes),
            )

            # Combined p-value (Fisher's method)
            chi2_stat = -2 * (
                np.log(p_genomics + 1e-300) +
                np.log(p_transcriptomics + 1e-300) +
                np.log(p_proteomics + 1e-300)
            )
            p_combined = stats.chi2.sf(chi2_stat, df=6)

            results.append({
                'pathway_name': pathway_name,
                'pathway_size': len(pathway_genes),
                'genomics_overlap': len(genomics_overlap),
                'transcriptomics_overlap': len(transcriptomics_overlap),
                'proteomics_overlap': len(proteomics_overlap),
                'cross_omics_overlap': len(cross_omics),
                'cross_omics_genes': ','.join(sorted(cross_omics)[:10]),  # First 10
                'p_genomics': p_genomics,
                'p_transcriptomics': p_transcriptomics,
                'p_proteomics': p_proteomics,
                'p_combined': p_combined,
            })

    df = pd.DataFrame(results)

    if len(df) > 0:
        # FDR correction
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(df['p_combined'], method='fdr_bh')
        df['fdr'] = fdr

        # Sort by combined p-value
        df = df.sort_values('p_combined')

    return df


def visualize_multiomics_heatmap(
    results: MultiOmicsPathwayResults,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 12),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize multi-omics pathway enrichment as heatmap

    Args:
        results: MultiOmicsPathwayResults
        top_n: Number of top pathways to show
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get top pathways
    top_pathways = results.top_pathways(n=top_n)

    if not top_pathways:
        warnings.warn("No pathways to visualize")
        return

    # Create matrix
    omics_columns = ['genomics', 'transcriptomics', 'proteomics', 'metabolomics']
    pathway_names = [p.pathway_name for p in top_pathways]

    matrix = []
    for pathway in top_pathways:
        row = [
            pathway.genomics_nes if pathway.genomics_nes is not None else 0,
            pathway.transcriptomics_nes if pathway.transcriptomics_nes is not None else 0,
            pathway.proteomics_nes if pathway.proteomics_nes is not None else 0,
            pathway.metabolomics_nes if pathway.metabolomics_nes is not None else 0,
        ]
        matrix.append(row)

    matrix = np.array(matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colormap range
    vmax = np.abs(matrix).max()
    vmin = -vmax

    sns.heatmap(
        matrix,
        xticklabels=omics_columns,
        yticklabels=pathway_names,
        cmap='RdBu_r',
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'NES'},
        ax=ax,
    )

    ax.set_title(
        f'Multi-Omics Pathway Enrichment (Top {top_n})\n'
        f'Integration: {results.integration_method}',
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel('Omics Layer', fontsize=12)
    ax.set_ylabel('Pathway', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()