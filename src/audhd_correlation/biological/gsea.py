"""Gene Set Enrichment Analysis (GSEA) for pathway enrichment

Implements standard GSEA and preranked GSEA for identifying enriched pathways.
"""
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict


@dataclass
class GSEAResult:
    """Result of GSEA analysis"""
    # Pathway information
    pathway_id: str
    pathway_name: str
    pathway_size: int

    # Enrichment statistics
    es: float  # Enrichment score
    nes: float  # Normalized enrichment score
    pval: float
    fdr: float

    # Leading edge
    leading_edge_genes: List[str]
    leading_edge_size: int

    # Additional info
    rank_at_max: int
    running_es: Optional[np.ndarray] = None


@dataclass
class GSEAResults:
    """Collection of GSEA results"""
    results: List[GSEAResult]
    n_permutations: int
    fdr_threshold: float = 0.25

    def significant_pathways(self) -> List[GSEAResult]:
        """Return pathways passing FDR threshold"""
        return [r for r in self.results if r.fdr <= self.fdr_threshold]

    def top_pathways(self, n: int = 10) -> List[GSEAResult]:
        """Return top N pathways by NES"""
        sorted_results = sorted(self.results, key=lambda x: abs(x.nes), reverse=True)
        return sorted_results[:n]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        data = []
        for r in self.results:
            data.append({
                'pathway_id': r.pathway_id,
                'pathway_name': r.pathway_name,
                'pathway_size': r.pathway_size,
                'ES': r.es,
                'NES': r.nes,
                'pval': r.pval,
                'FDR': r.fdr,
                'leading_edge_size': r.leading_edge_size,
                'leading_edge_genes': ','.join(r.leading_edge_genes[:5]),  # First 5
            })
        return pd.DataFrame(data)


def load_gene_sets(
    database_path: str,
    format: str = "auto",
    gene_mapper: Optional[object] = None,  # GeneIDMapper
    min_size: int = 15,
    max_size: int = 500,
) -> Dict[str, Set[str]]:
    """
    Load gene sets from pathway database file

    Args:
        database_path: Path to pathway database file (GMT, GPAD, TSV, CSV)
        format: File format ("gmt", "gpad", "tsv", "csv", "auto")
        gene_mapper: GeneIDMapper for gene normalization (optional)
        min_size: Minimum gene set size
        max_size: Maximum gene set size

    Returns:
        Dictionary mapping pathway names to gene sets

    Raises:
        FileNotFoundError: If database_path doesn't exist

    Example:
        # Load MSigDB Hallmark gene sets
        gene_sets = load_gene_sets(
            database_path="data/h.all.v2023.1.Hs.symbols.gmt",
            format="gmt"
        )

    Download pathway databases:
        • MSigDB: https://www.gsea-msigdb.org/gsea/downloads.jsp
        • Gene Ontology: http://geneontology.org/docs/download-ontology/
        • KEGG: https://www.genome.jp/kegg/pathway.html
        • Reactome: https://reactome.org/download-data

    Or use CLI:
        audhd-pipeline download-pathways --database msigdb
    """
    from .pathway_database import load_pathway_database

    # Load database
    db = load_pathway_database(
        database_path=database_path,
        format=format,
        gene_mapper=gene_mapper,
        normalize_genes=(gene_mapper is not None),
    )

    # Filter by size
    db_filtered = db.filter_by_size(min_size=min_size, max_size=max_size)

    return db_filtered.pathways


# Removed hardcoded pathway fallbacks - users must provide explicit pathway database files
# See load_gene_sets() documentation for download instructions


def run_gsea(
    expression_data: pd.DataFrame,
    labels: np.ndarray,
    gene_sets: Dict[str, Set[str]],
    cluster_id: int,
    n_permutations: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    ranking_method: str = "log2fc",
    fdr_method: str = "bh",
    random_state: int = 42,
) -> GSEAResults:
    """
    Run GSEA on expression data

    Args:
        expression_data: Gene expression matrix (samples × genes)
        labels: Cluster labels for samples
        gene_sets: Dictionary of pathway name → gene set
        cluster_id: Cluster to test (vs all others)
        n_permutations: Number of permutations for null distribution
        min_size: Minimum gene set size
        max_size: Maximum gene set size
        ranking_method: Gene ranking method ("log2fc", "signal_to_noise", "t_stat")
        fdr_method: FDR correction method ("bh", "bonferroni", "none")
        random_state: Random seed

    Returns:
        GSEAResults with enriched pathways

    Ranking methods:
        - "log2fc": Log2 fold change (default)
        - "signal_to_noise": (mean1 - mean2) / (std1 + std2)
        - "t_stat": T-statistic from t-test
    """
    np.random.seed(random_state)

    # Create binary labels (cluster vs rest)
    y = (labels == cluster_id).astype(int)

    # Rank genes by specified method
    ranked_genes = _rank_genes(
        expression_data=expression_data,
        labels=y,
        method=ranking_method,
    )

    # Run preranked GSEA
    results = prerank_gsea(
        ranked_genes=ranked_genes,
        gene_sets=gene_sets,
        n_permutations=n_permutations,
        min_size=min_size,
        max_size=max_size,
        fdr_method=fdr_method,
        random_state=random_state,
    )

    return results


def _rank_genes(
    expression_data: pd.DataFrame,
    labels: np.ndarray,
    method: str = "log2fc",
) -> pd.Series:
    """Rank genes by differential expression

    Args:
        expression_data: Gene expression matrix
        labels: Binary labels (0/1)
        method: Ranking method

    Returns:
        Series of gene ranks (higher = more upregulated in class 1)
    """
    cluster_expr = expression_data[labels == 1]
    other_expr = expression_data[labels == 0]

    if method == "log2fc":
        # Log2 fold change
        cluster_mean = cluster_expr.mean(axis=0)
        other_mean = other_expr.mean(axis=0)
        ranks = np.log2((cluster_mean + 1) / (other_mean + 1))

    elif method == "signal_to_noise":
        # Signal-to-noise ratio
        cluster_mean = cluster_expr.mean(axis=0)
        other_mean = other_expr.mean(axis=0)
        cluster_std = cluster_expr.std(axis=0)
        other_std = other_expr.std(axis=0)

        # Avoid division by zero
        denom = cluster_std + other_std + 1e-8
        ranks = (cluster_mean - other_mean) / denom

    elif method == "t_stat":
        # T-statistic
        from scipy.stats import ttest_ind

        t_stats = []
        for gene in expression_data.columns:
            t_stat, _ = ttest_ind(
                cluster_expr[gene],
                other_expr[gene],
                equal_var=False,  # Welch's t-test
            )
            t_stats.append(t_stat)

        ranks = pd.Series(t_stats, index=expression_data.columns)

    else:
        raise ValueError(f"Unknown ranking method: {method}")

    # Sort descending
    return ranks.sort_values(ascending=False)


def prerank_gsea(
    ranked_genes: pd.Series,
    gene_sets: Dict[str, Set[str]],
    n_permutations: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    fdr_method: str = "bh",
    random_state: int = 42,
) -> GSEAResults:
    """
    Run preranked GSEA

    Args:
        ranked_genes: Ranked gene list (gene name → rank metric)
        gene_sets: Dictionary of pathway name → gene set
        n_permutations: Number of permutations
        min_size: Minimum gene set size
        max_size: Maximum gene set size
        fdr_method: FDR correction method ("bh", "bonferroni", "none")
        random_state: Random seed

    Returns:
        GSEAResults with enrichment statistics

    FDR methods:
        - "bh": Benjamini-Hochberg (False Discovery Rate)
        - "bonferroni": Bonferroni correction (more conservative)
        - "none": No correction (not recommended)
    """
    np.random.seed(random_state)

    results = []

    # Filter gene sets by size
    filtered_gene_sets = {
        name: genes for name, genes in gene_sets.items()
        if min_size <= len(genes & set(ranked_genes.index)) <= max_size
    }

    if not filtered_gene_sets:
        warnings.warn("No gene sets passed size filters")
        return GSEAResults(results=[], n_permutations=n_permutations)

    # Calculate ES for each pathway
    pathway_es = {}
    pathway_info = {}

    for pathway_name, pathway_genes in filtered_gene_sets.items():
        # Calculate enrichment score
        es, nes, pval, leading_edge, rank_at_max, running_es = _calculate_enrichment_score(
            ranked_genes=ranked_genes,
            gene_set=pathway_genes,
            n_permutations=n_permutations,
        )

        pathway_es[pathway_name] = nes
        pathway_info[pathway_name] = {
            'es': es,
            'nes': nes,
            'pval': pval,
            'leading_edge': leading_edge,
            'rank_at_max': rank_at_max,
            'running_es': running_es,
            'size': len(pathway_genes & set(ranked_genes.index)),
        }

    # FDR correction
    pvals = [info['pval'] for info in pathway_info.values()]
    fdrs = _calculate_fdr(pvals, method=fdr_method)

    # Create results
    for i, (pathway_name, info) in enumerate(pathway_info.items()):
        # Split pathway ID and name
        if ':' in pathway_name:
            pathway_id, pathway_display = pathway_name.split(':', 1)
        else:
            pathway_id = pathway_name
            pathway_display = pathway_name

        result = GSEAResult(
            pathway_id=pathway_id,
            pathway_name=pathway_display,
            pathway_size=info['size'],
            es=info['es'],
            nes=info['nes'],
            pval=info['pval'],
            fdr=fdrs[i],
            leading_edge_genes=info['leading_edge'],
            leading_edge_size=len(info['leading_edge']),
            rank_at_max=info['rank_at_max'],
            running_es=info['running_es'],
        )
        results.append(result)

    # Sort by NES
    results.sort(key=lambda x: abs(x.nes), reverse=True)

    return GSEAResults(results=results, n_permutations=n_permutations)


def _calculate_enrichment_score(
    ranked_genes: pd.Series,
    gene_set: Set[str],
    n_permutations: int = 1000,
    weight: float = 1.0,
) -> Tuple[float, float, float, List[str], int, np.ndarray]:
    """
    Calculate enrichment score for a gene set

    Returns:
        es: Enrichment score
        nes: Normalized enrichment score
        pval: P-value
        leading_edge: Leading edge genes
        rank_at_max: Rank position of maximum ES
        running_es: Running ES curve
    """
    # Get gene ranks
    gene_list = ranked_genes.index.tolist()
    gene_ranks = ranked_genes.values

    # Mark which genes are in the set
    in_set = np.array([gene in gene_set for gene in gene_list])

    n_genes = len(gene_list)
    n_in_set = in_set.sum()

    if n_in_set == 0:
        return 0.0, 0.0, 1.0, [], 0, np.zeros(n_genes)

    # Calculate running sum
    # Weighted by gene rank metric
    hit_weights = np.abs(gene_ranks) ** weight * in_set
    hit_sum = hit_weights.sum()

    if hit_sum == 0:
        hit_weights = in_set.astype(float)
        hit_sum = hit_weights.sum()

    miss_weights = (1 - in_set).astype(float)
    miss_sum = miss_weights.sum()

    # Running ES
    running_es = np.zeros(n_genes)
    for i in range(n_genes):
        if in_set[i]:
            running_es[i] = hit_weights[i] / hit_sum
        else:
            running_es[i] = -miss_weights[i] / miss_sum

    running_es = np.cumsum(running_es)

    # ES is maximum deviation from zero
    max_pos = np.argmax(np.abs(running_es))
    es = running_es[max_pos]

    # Leading edge genes
    if es > 0:
        leading_edge_idx = np.where(in_set[:max_pos + 1])[0]
    else:
        leading_edge_idx = np.where(in_set[max_pos:])[0] + max_pos

    leading_edge = [gene_list[i] for i in leading_edge_idx]

    # Permutation test for significance
    null_es = []

    for _ in range(n_permutations):
        # Shuffle gene set membership
        shuffled_in_set = np.random.permutation(in_set)

        # Recalculate ES
        hit_weights_null = np.abs(gene_ranks) ** weight * shuffled_in_set
        hit_sum_null = hit_weights_null.sum()

        if hit_sum_null == 0:
            hit_weights_null = shuffled_in_set.astype(float)
            hit_sum_null = hit_weights_null.sum()

        miss_weights_null = (1 - shuffled_in_set).astype(float)
        miss_sum_null = miss_weights_null.sum()

        running_es_null = np.zeros(n_genes)
        for i in range(n_genes):
            if shuffled_in_set[i]:
                running_es_null[i] = hit_weights_null[i] / hit_sum_null
            else:
                running_es_null[i] = -miss_weights_null[i] / miss_sum_null

        running_es_null = np.cumsum(running_es_null)
        es_null = running_es_null[np.argmax(np.abs(running_es_null))]
        null_es.append(es_null)

    null_es = np.array(null_es)

    # Normalize ES
    if es >= 0:
        null_pos = null_es[null_es >= 0]
        nes = es / (null_pos.mean() if len(null_pos) > 0 else 1.0)
    else:
        null_neg = null_es[null_es < 0]
        nes = -es / (-null_neg.mean() if len(null_neg) > 0 else 1.0)

    # P-value
    if es >= 0:
        pval = (null_es >= es).sum() / n_permutations
    else:
        pval = (null_es <= es).sum() / n_permutations

    pval = max(pval, 1.0 / n_permutations)  # Avoid p=0

    return float(es), float(nes), float(pval), leading_edge, int(max_pos), running_es


def _calculate_fdr(pvals: List[float], method: str = "bh") -> List[float]:
    """Calculate FDR using specified method

    Args:
        pvals: List of p-values
        method: Correction method ("bh", "bonferroni", "none")

    Returns:
        List of corrected p-values
    """
    n = len(pvals)
    if n == 0:
        return []

    pvals_arr = np.array(pvals)

    if method == "bh":
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(pvals_arr)
        sorted_pvals = pvals_arr[sorted_idx]

        # Calculate FDR
        fdrs = np.zeros(n)
        for i in range(n):
            fdrs[i] = sorted_pvals[i] * n / (i + 1)

        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            fdrs[i] = min(fdrs[i], fdrs[i + 1])

        # Unsort
        unsorted_fdrs = np.zeros(n)
        unsorted_fdrs[sorted_idx] = fdrs

        # Cap at 1.0
        unsorted_fdrs = np.minimum(unsorted_fdrs, 1.0)

        return unsorted_fdrs.tolist()

    elif method == "bonferroni":
        # Bonferroni correction
        corrected = np.minimum(pvals_arr * n, 1.0)
        return corrected.tolist()

    elif method == "none":
        # No correction
        return pvals

    else:
        raise ValueError(
            f"Unknown FDR method: {method}. "
            f"Supported: 'bh', 'bonferroni', 'none'"
        )


def compare_cluster_enrichments(
    expression_data: pd.DataFrame,
    labels: np.ndarray,
    gene_sets: Dict[str, Set[str]],
    cluster_ids: Optional[List[int]] = None,
    n_permutations: int = 1000,
    fdr_threshold: float = 0.25,
    ranking_method: str = "log2fc",
    fdr_method: str = "bh",
) -> pd.DataFrame:
    """
    Compare pathway enrichments across clusters

    Args:
        expression_data: Gene expression matrix
        labels: Cluster labels
        gene_sets: Pathway gene sets
        cluster_ids: Clusters to compare (default: all)
        n_permutations: Number of permutations
        fdr_threshold: FDR threshold for significance
        ranking_method: Gene ranking method
        fdr_method: FDR correction method

    Returns:
        DataFrame with pathway enrichments per cluster (NES values)
    """
    if cluster_ids is None:
        cluster_ids = np.unique(labels[labels >= 0])  # Exclude noise (-1)

    all_results = {}

    for cluster_id in cluster_ids:
        results = run_gsea(
            expression_data=expression_data,
            labels=labels,
            gene_sets=gene_sets,
            cluster_id=cluster_id,
            n_permutations=n_permutations,
            ranking_method=ranking_method,
            fdr_method=fdr_method,
        )

        # Keep only significant pathways
        sig_results = results.significant_pathways()
        all_results[f'cluster_{cluster_id}'] = {
            r.pathway_name: r.nes for r in sig_results
        }

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results).fillna(0)

    return comparison_df


def generate_enrichment_table(
    expression_data: pd.DataFrame,
    labels: np.ndarray,
    gene_sets: Dict[str, Set[str]],
    cluster_ids: Optional[List[int]] = None,
    n_permutations: int = 1000,
    fdr_threshold: float = 0.25,
    ranking_method: str = "log2fc",
    fdr_method: str = "bh",
    include_leading_edge: bool = True,
    max_leading_edge_genes: int = 10,
) -> pd.DataFrame:
    """
    Generate comprehensive enrichment table for all clusters

    Creates a detailed table with:
    - Pathway information
    - Cluster-specific enrichment scores
    - FDR values
    - Leading edge genes
    - Pathway sizes

    Args:
        expression_data: Gene expression matrix
        labels: Cluster labels
        gene_sets: Pathway gene sets
        cluster_ids: Clusters to analyze (default: all)
        n_permutations: Number of permutations
        fdr_threshold: FDR threshold for significance
        ranking_method: Gene ranking method
        fdr_method: FDR correction method
        include_leading_edge: Include leading edge genes in output
        max_leading_edge_genes: Maximum leading edge genes to show per pathway

    Returns:
        DataFrame with comprehensive enrichment results

    Output columns:
        - pathway_id: Pathway identifier
        - pathway_name: Pathway name
        - cluster_X_NES: Normalized enrichment score for cluster X
        - cluster_X_FDR: FDR q-value for cluster X
        - cluster_X_leading_edge: Leading edge genes for cluster X
        - pathway_size: Number of genes in pathway
        - n_clusters_enriched: Number of clusters with significant enrichment
    """
    if cluster_ids is None:
        cluster_ids = np.unique(labels[labels >= 0])

    # Store all results
    all_results = []

    for cluster_id in cluster_ids:
        results = run_gsea(
            expression_data=expression_data,
            labels=labels,
            gene_sets=gene_sets,
            cluster_id=cluster_id,
            n_permutations=n_permutations,
            ranking_method=ranking_method,
            fdr_method=fdr_method,
        )

        # Add cluster ID to each result
        for r in results.results:
            all_results.append({
                'cluster_id': cluster_id,
                'pathway_id': r.pathway_id,
                'pathway_name': r.pathway_name,
                'pathway_size': r.pathway_size,
                'NES': r.nes,
                'FDR': r.fdr,
                'pval': r.pval,
                'leading_edge': r.leading_edge_genes[:max_leading_edge_genes],
                'leading_edge_size': r.leading_edge_size,
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Pivot to wide format
    rows = []
    for pathway_id in df['pathway_id'].unique():
        pathway_df = df[df['pathway_id'] == pathway_id]

        row = {
            'pathway_id': pathway_id,
            'pathway_name': pathway_df['pathway_name'].iloc[0],
            'pathway_size': pathway_df['pathway_size'].iloc[0],
        }

        # Add cluster-specific columns
        for cluster_id in cluster_ids:
            cluster_data = pathway_df[pathway_df['cluster_id'] == cluster_id]

            if len(cluster_data) > 0:
                row[f'cluster_{cluster_id}_NES'] = cluster_data['NES'].iloc[0]
                row[f'cluster_{cluster_id}_FDR'] = cluster_data['FDR'].iloc[0]

                if include_leading_edge:
                    leading_edge = cluster_data['leading_edge'].iloc[0]
                    row[f'cluster_{cluster_id}_leading_edge'] = ','.join(leading_edge)
            else:
                row[f'cluster_{cluster_id}_NES'] = 0.0
                row[f'cluster_{cluster_id}_FDR'] = 1.0
                if include_leading_edge:
                    row[f'cluster_{cluster_id}_leading_edge'] = ""

        # Count significant enrichments
        n_enriched = sum(
            pathway_df['FDR'] <= fdr_threshold
        )
        row['n_clusters_enriched'] = n_enriched

        rows.append(row)

    enrichment_table = pd.DataFrame(rows)

    # Sort by number of enriched clusters, then by best NES
    nes_cols = [col for col in enrichment_table.columns if col.endswith('_NES')]
    enrichment_table['max_abs_NES'] = enrichment_table[nes_cols].abs().max(axis=1)

    enrichment_table = enrichment_table.sort_values(
        ['n_clusters_enriched', 'max_abs_NES'],
        ascending=[False, False]
    )

    enrichment_table = enrichment_table.drop('max_abs_NES', axis=1)

    return enrichment_table