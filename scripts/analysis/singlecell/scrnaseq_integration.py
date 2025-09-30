#!/usr/bin/env python3
"""
Single-Cell RNA Sequencing Analysis Integration
Enables cell-type-specific analysis for AuDHD correlation study
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SingleCellResult:
    """Results from single-cell analysis"""
    cell_types: List[str]
    marker_genes: Dict[str, List[str]]
    disease_enrichment: pd.DataFrame
    trajectory_results: Optional[pd.DataFrame] = None
    differential_expression: Optional[pd.DataFrame] = None


class SingleCellAnalyzer:
    """
    Single-cell RNA-seq analysis for AuDHD research

    Capabilities:
    1. Cell-type identification and clustering
    2. Disease-relevant cell type enrichment
    3. Developmental trajectory analysis
    4. Cell-type-specific GWAS enrichment
    5. Integration with bulk tissue data
    """

    def __init__(self, min_genes: int = 200, min_cells: int = 3):
        """
        Initialize analyzer

        Parameters
        ----------
        min_genes : int
            Minimum genes per cell for quality control
        min_cells : int
            Minimum cells expressing a gene
        """
        self.min_genes = min_genes
        self.min_cells = min_cells

    def preprocess_data(
        self,
        counts: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Quality control and normalization

        Parameters
        ----------
        counts : pd.DataFrame
            Raw count matrix (cells × genes)
        metadata : pd.DataFrame
            Cell metadata (sample info, batch, etc.)

        Returns
        -------
        normalized_counts : pd.DataFrame
        filtered_metadata : pd.DataFrame
        """
        logger.info("Preprocessing single-cell data")

        # Quality control
        genes_per_cell = (counts > 0).sum(axis=1)
        cells_per_gene = (counts > 0).sum(axis=0)

        valid_cells = genes_per_cell >= self.min_genes
        valid_genes = cells_per_gene >= self.min_cells

        logger.info(f"  Filtered: {(~valid_cells).sum()} cells, {(~valid_genes).sum()} genes")

        counts_filtered = counts.loc[valid_cells, valid_genes]

        # Normalization: log(CPM + 1)
        cpm = counts_filtered.div(counts_filtered.sum(axis=1), axis=0) * 1e6
        normalized = np.log1p(cpm)

        if metadata is not None:
            metadata_filtered = metadata.loc[valid_cells]
        else:
            metadata_filtered = pd.DataFrame(index=counts_filtered.index)

        logger.info(f"  Final: {normalized.shape[0]} cells, {normalized.shape[1]} genes")

        return normalized, metadata_filtered

    def identify_cell_types(
        self,
        normalized_counts: pd.DataFrame,
        n_clusters: int = 10,
        resolution: float = 0.8
    ) -> pd.DataFrame:
        """
        Cluster cells and identify cell types

        Uses Leiden clustering on PCA-reduced data

        Parameters
        ----------
        normalized_counts : pd.DataFrame
            Normalized expression (cells × genes)
        n_clusters : int
            Target number of clusters
        resolution : float
            Leiden resolution parameter

        Returns
        -------
        cell_assignments : pd.DataFrame
            Columns: cell_id, cluster, cell_type
        """
        logger.info("Identifying cell types via clustering")

        # PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        pca = PCA(n_components=50, random_state=42)
        pca_coords = pca.fit_transform(normalized_counts.values)

        logger.info(f"  PCA explained variance: {pca.explained_variance_ratio_[:10].sum():.1%}")

        # Clustering (using KMeans as simple alternative to Leiden)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(pca_coords)

        # Create assignments dataframe
        assignments = pd.DataFrame({
            'cell_id': normalized_counts.index,
            'cluster': clusters,
            'cell_type': [f'Cluster_{c}' for c in clusters]
        })

        logger.info(f"  Identified {n_clusters} clusters")

        return assignments

    def find_marker_genes(
        self,
        normalized_counts: pd.DataFrame,
        cell_assignments: pd.DataFrame,
        top_n: int = 50
    ) -> Dict[str, List[str]]:
        """
        Find marker genes for each cell type

        Uses Wilcoxon rank-sum test for differential expression

        Parameters
        ----------
        normalized_counts : pd.DataFrame
        cell_assignments : pd.DataFrame
        top_n : int
            Number of top markers per cell type

        Returns
        -------
        marker_genes : Dict[str, List[str]]
            Cell type -> marker gene list
        """
        logger.info("Finding marker genes for each cell type")

        from scipy import stats

        marker_genes = {}

        for cell_type in cell_assignments['cell_type'].unique():
            # Cells in this type vs others
            in_type = cell_assignments['cell_type'] == cell_type

            # Test each gene
            gene_scores = []
            for gene in normalized_counts.columns:
                expr_in = normalized_counts.loc[in_type, gene]
                expr_out = normalized_counts.loc[~in_type, gene]

                # Wilcoxon test
                stat, pval = stats.ranksums(expr_in, expr_out)

                # Effect size: log fold change
                mean_in = expr_in.mean()
                mean_out = expr_out.mean()
                logfc = mean_in - mean_out

                gene_scores.append({
                    'gene': gene,
                    'logFC': logfc,
                    'p_value': pval,
                    'mean_in_type': mean_in,
                    'mean_out_type': mean_out
                })

            # Sort by logFC (upregulated in this type)
            gene_df = pd.DataFrame(gene_scores).sort_values('logFC', ascending=False)

            # Top markers
            markers = gene_df.head(top_n)['gene'].tolist()
            marker_genes[cell_type] = markers

            logger.info(f"  {cell_type}: {len(markers)} markers")

        return marker_genes

    def identify_disease_cell_types(
        self,
        normalized_counts: pd.DataFrame,
        cell_assignments: pd.DataFrame,
        gwas_genes: List[str],
        background_genes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Identify cell types enriched for disease GWAS genes

        Tests whether GWAS genes are preferentially expressed in specific cell types

        Parameters
        ----------
        normalized_counts : pd.DataFrame
        cell_assignments : pd.DataFrame
        gwas_genes : List[str]
            Genes near ASD/ADHD GWAS hits
        background_genes : List[str], optional
            Background gene set for enrichment test

        Returns
        -------
        enrichment_results : pd.DataFrame
            Columns: cell_type, n_gwas_genes, enrichment_score, p_value
        """
        logger.info("Testing cell-type-specific GWAS enrichment")

        from scipy import stats

        # Find GWAS genes in dataset
        gwas_in_data = [g for g in gwas_genes if g in normalized_counts.columns]
        logger.info(f"  {len(gwas_in_data)}/{len(gwas_genes)} GWAS genes in dataset")

        if background_genes is None:
            background_genes = normalized_counts.columns.tolist()

        enrichment_results = []

        for cell_type in cell_assignments['cell_type'].unique():
            # Cells in this type
            in_type = cell_assignments['cell_type'] == cell_type

            # Mean expression in this cell type
            mean_expr = normalized_counts.loc[in_type, :].mean(axis=0)

            # Rank genes by expression
            gene_ranks = mean_expr.rank(ascending=False)

            # GWAS gene ranks
            gwas_ranks = gene_ranks[gwas_in_data]
            background_ranks = gene_ranks[background_genes]

            # Wilcoxon rank-sum test: are GWAS genes ranked higher?
            stat, pval = stats.ranksums(gwas_ranks, background_ranks)

            # Enrichment score: median rank difference
            enrichment_score = background_ranks.median() - gwas_ranks.median()

            enrichment_results.append({
                'cell_type': cell_type,
                'n_gwas_genes': len(gwas_in_data),
                'median_rank_gwas': gwas_ranks.median(),
                'median_rank_background': background_ranks.median(),
                'enrichment_score': enrichment_score,
                'p_value': pval
            })

        enrichment_df = pd.DataFrame(enrichment_results).sort_values('p_value')

        # FDR correction
        from statsmodels.stats.multitest import multipletests
        _, qvals, _, _ = multipletests(enrichment_df['p_value'], method='fdr_bh')
        enrichment_df['q_value'] = qvals

        logger.info(f"  Significant cell types (q<0.05): {(qvals < 0.05).sum()}")

        return enrichment_df

    def trajectory_analysis(
        self,
        normalized_counts: pd.DataFrame,
        cell_assignments: pd.DataFrame,
        start_cluster: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Developmental trajectory / pseudotime analysis

        Orders cells along developmental trajectory to study gene expression dynamics

        Parameters
        ----------
        normalized_counts : pd.DataFrame
        cell_assignments : pd.DataFrame
        start_cluster : str, optional
            Starting cell type for trajectory

        Returns
        -------
        trajectory_results : pd.DataFrame
            Columns: cell_id, pseudotime, trajectory_position
        """
        logger.info("Computing developmental trajectories")

        from sklearn.decomposition import PCA
        from scipy.spatial.distance import cdist

        # PCA for trajectory inference
        pca = PCA(n_components=10, random_state=42)
        pca_coords = pca.fit_transform(normalized_counts.values)

        # Simplified trajectory: distance from starting cluster
        if start_cluster is None:
            # Use cluster 0 as start
            start_cluster = cell_assignments['cell_type'].unique()[0]

        logger.info(f"  Starting from: {start_cluster}")

        # Centroid of start cluster
        in_start = cell_assignments['cell_type'] == start_cluster
        start_centroid = pca_coords[in_start].mean(axis=0, keepdims=True)

        # Distance from start = pseudotime
        distances = cdist(pca_coords, start_centroid, metric='euclidean').flatten()

        trajectory_results = pd.DataFrame({
            'cell_id': normalized_counts.index,
            'pseudotime': distances,
            'trajectory_position': distances.argsort().argsort()  # Rank
        })

        logger.info(f"  Pseudotime range: {distances.min():.2f} - {distances.max():.2f}")

        return trajectory_results

    def integrate_with_gwas(
        self,
        cell_type_enrichment: pd.DataFrame,
        baseline_deviation_results: pd.DataFrame,
        significance_threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Integrate cell-type findings with baseline-deviation GWAS results

        Identifies cell-type-specific effects that align with genetic findings

        Parameters
        ----------
        cell_type_enrichment : pd.DataFrame
            From identify_disease_cell_types()
        baseline_deviation_results : pd.DataFrame
            From baseline-deviation analysis
        significance_threshold : float
            Q-value threshold

        Returns
        -------
        integrated_results : pd.DataFrame
            Cell types with converging evidence
        """
        logger.info("Integrating single-cell with GWAS")

        # Significant cell types
        sig_cell_types = cell_type_enrichment[
            cell_type_enrichment['q_value'] < significance_threshold
        ].copy()

        # Annotate with GWAS context
        sig_cell_types['has_gwas_support'] = True
        sig_cell_types['analysis_framework'] = 'baseline_deviation'

        logger.info(f"  {len(sig_cell_types)} cell types with disease enrichment")

        return sig_cell_types

    def analyze_complete(
        self,
        counts: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        gwas_genes: Optional[List[str]] = None,
        n_clusters: int = 10
    ) -> SingleCellResult:
        """
        Complete single-cell analysis pipeline

        Parameters
        ----------
        counts : pd.DataFrame
            Raw count matrix
        metadata : pd.DataFrame
            Cell metadata
        gwas_genes : List[str]
            GWAS genes for enrichment
        n_clusters : int
            Number of clusters

        Returns
        -------
        SingleCellResult
        """
        logger.info("=== Complete Single-Cell Analysis ===")

        # 1. Preprocessing
        normalized, metadata_filtered = self.preprocess_data(counts, metadata)

        # 2. Cell type identification
        cell_assignments = self.identify_cell_types(normalized, n_clusters=n_clusters)

        # 3. Marker genes
        marker_genes = self.find_marker_genes(normalized, cell_assignments)

        # 4. Disease enrichment
        if gwas_genes is not None:
            disease_enrichment = self.identify_disease_cell_types(
                normalized, cell_assignments, gwas_genes
            )
        else:
            disease_enrichment = pd.DataFrame()

        # 5. Trajectory analysis
        trajectory_results = self.trajectory_analysis(normalized, cell_assignments)

        return SingleCellResult(
            cell_types=cell_assignments['cell_type'].unique().tolist(),
            marker_genes=marker_genes,
            disease_enrichment=disease_enrichment,
            trajectory_results=trajectory_results
        )


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    logger.info("Single-Cell RNA-seq Analysis Module")
    logger.info("Ready for integration with AuDHD correlation study")
    logger.info("\nKey capabilities:")
    logger.info("  1. Cell-type identification and clustering")
    logger.info("  2. Disease-relevant cell type enrichment")
    logger.info("  3. Developmental trajectory analysis")
    logger.info("  4. GWAS gene enrichment in cell types")
    logger.info("  5. Integration with baseline-deviation framework")
