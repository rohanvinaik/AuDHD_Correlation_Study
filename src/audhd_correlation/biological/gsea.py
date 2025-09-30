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
    database: str = 'GO_Biological_Process',
    species: str = 'human',
    custom_gmt: Optional[str] = None,
) -> Dict[str, Set[str]]:
    """
    Load gene sets from pathway databases

    Args:
        database: Database name ('GO_Biological_Process', 'KEGG', 'Reactome', 'MSigDB')
        species: Species ('human', 'mouse')
        custom_gmt: Path to custom GMT file

    Returns:
        Dictionary mapping pathway names to gene sets
    """
    if custom_gmt:
        return _load_gmt_file(custom_gmt)

    # For now, return example gene sets
    # In production, would integrate with GSEApy or load from databases
    gene_sets = _get_example_gene_sets(database, species)

    return gene_sets


def _load_gmt_file(gmt_path: str) -> Dict[str, Set[str]]:
    """Load gene sets from GMT file"""
    gene_sets = {}

    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            pathway_name = parts[0]
            # parts[1] is description (skip)
            genes = set(parts[2:])

            gene_sets[pathway_name] = genes

    return gene_sets


def _get_example_gene_sets(database: str, species: str) -> Dict[str, Set[str]]:
    """Get example gene sets for demonstration"""
    # These are simplified examples
    # In production, integrate with actual databases

    if database == 'GO_Biological_Process':
        return {
            'GO:0006955_immune_response': {
                'IL6', 'TNF', 'IFNG', 'IL1B', 'IL10', 'CD4', 'CD8A', 'FOXP3'
            },
            'GO:0006954_inflammatory_response': {
                'IL6', 'TNF', 'IL1B', 'NFKB1', 'PTGS2', 'CCL2', 'CXCL8'
            },
            'GO:0006412_translation': {
                'RPS6', 'RPL13', 'EIF4E', 'EIF2S1', 'MTOR', 'RPS6KB1'
            },
            'GO:0007268_chemical_synaptic_transmission': {
                'SYT1', 'SLC17A7', 'DLG4', 'GRIN1', 'GRIN2A', 'GRIA1', 'SYN1'
            },
            'GO:0007399_nervous_system_development': {
                'NEUROD1', 'NEUROG2', 'SOX2', 'PAX6', 'NKX2-1', 'ASCL1'
            },
            'GO:0006030_chitin_metabolic_process': {
                'CHIA', 'CHIT1', 'CHI3L1', 'CHI3L2', 'OVGP1'
            },
        }

    elif database == 'KEGG':
        return {
            'hsa04610:Complement_and_coagulation_cascades': {
                'C1QA', 'C1QB', 'C3', 'C5', 'F2', 'F10', 'SERPINE1'
            },
            'hsa04060:Cytokine-cytokine_receptor_interaction': {
                'IL6', 'IL6R', 'TNF', 'TNFRSF1A', 'IL1B', 'IL1R1', 'IFNG', 'IFNGR1'
            },
            'hsa04080:Neuroactive_ligand-receptor_interaction': {
                'DRD1', 'DRD2', 'HTR1A', 'HTR2A', 'GRIA1', 'GRIN1', 'GABRA1'
            },
        }

    elif database == 'Reactome':
        return {
            'R-HSA-168256:Immune_System': {
                'CD4', 'CD8A', 'IL6', 'TNF', 'IFNG', 'IL1B', 'IL10'
            },
            'R-HSA-112316:Neuronal_System': {
                'SLC17A7', 'GRIN1', 'GRIN2A', 'DLG4', 'SYT1', 'SYN1'
            },
        }

    else:
        warnings.warn(f"Unknown database: {database}. Using example gene sets.")
        return _get_example_gene_sets('GO_Biological_Process', species)


def run_gsea(
    expression_data: pd.DataFrame,
    labels: np.ndarray,
    gene_sets: Dict[str, Set[str]],
    cluster_id: int,
    n_permutations: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
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
        random_state: Random seed

    Returns:
        GSEAResults with enriched pathways
    """
    np.random.seed(random_state)

    # Create binary labels (cluster vs rest)
    y = (labels == cluster_id).astype(int)

    # Calculate fold change and rank genes
    cluster_mean = expression_data[y == 1].mean(axis=0)
    other_mean = expression_data[y == 0].mean(axis=0)

    # Log2 fold change
    fc = np.log2((cluster_mean + 1) / (other_mean + 1))

    # Rank genes by fold change
    ranked_genes = fc.sort_values(ascending=False)

    # Run preranked GSEA
    results = prerank_gsea(
        ranked_genes=ranked_genes,
        gene_sets=gene_sets,
        n_permutations=n_permutations,
        min_size=min_size,
        max_size=max_size,
        random_state=random_state,
    )

    return results


def prerank_gsea(
    ranked_genes: pd.Series,
    gene_sets: Dict[str, Set[str]],
    n_permutations: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
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
        random_state: Random seed

    Returns:
        GSEAResults with enrichment statistics
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
    fdrs = _calculate_fdr(pvals)

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


def _calculate_fdr(pvals: List[float]) -> List[float]:
    """Calculate FDR using Benjamini-Hochberg procedure"""
    n = len(pvals)
    if n == 0:
        return []

    # Sort p-values
    sorted_idx = np.argsort(pvals)
    sorted_pvals = np.array(pvals)[sorted_idx]

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

    return unsorted_fdrs.tolist()


def compare_cluster_enrichments(
    expression_data: pd.DataFrame,
    labels: np.ndarray,
    gene_sets: Dict[str, Set[str]],
    cluster_ids: Optional[List[int]] = None,
    n_permutations: int = 1000,
    fdr_threshold: float = 0.25,
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

    Returns:
        DataFrame with pathway enrichments per cluster
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
        )

        # Keep only significant pathways
        sig_results = results.significant_pathways()
        all_results[f'cluster_{cluster_id}'] = {
            r.pathway_name: r.nes for r in sig_results
        }

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results).fillna(0)

    return comparison_df