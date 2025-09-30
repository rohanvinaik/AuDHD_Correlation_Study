"""Drug target prediction and ranking

Identifies druggable targets and maps to approved drugs.
"""
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DrugTargetResult:
    """Result of drug target prediction"""
    target_gene: str
    target_protein: str

    # Druggability scores
    druggability_score: float
    binding_site_score: float
    expression_score: float
    disease_association_score: float

    # Combined score
    combined_score: float

    # Evidence
    known_drugs: List[str] = field(default_factory=list)
    drug_classes: List[str] = field(default_factory=list)
    clinical_phase: Optional[str] = None

    # Network properties
    is_hub: bool = False
    centrality_score: float = 0.0

    # Pathway membership
    pathways: List[str] = field(default_factory=list)


@dataclass
class DrugTargetResults:
    """Collection of drug target results"""
    results: List[DrugTargetResult]
    ranking_method: str

    def top_targets(self, n: int = 20) -> List[DrugTargetResult]:
        """Return top N targets by combined score"""
        sorted_results = sorted(
            self.results,
            key=lambda x: x.combined_score,
            reverse=True
        )
        return sorted_results[:n]

    def novel_targets(self) -> List[DrugTargetResult]:
        """Return targets without known drugs"""
        return [r for r in self.results if not r.known_drugs]

    def approved_drug_targets(self) -> List[DrugTargetResult]:
        """Return targets with approved drugs"""
        return [
            r for r in self.results
            if r.clinical_phase == 'Approved'
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        data = []
        for r in self.results:
            data.append({
                'target_gene': r.target_gene,
                'target_protein': r.target_protein,
                'combined_score': r.combined_score,
                'druggability_score': r.druggability_score,
                'expression_score': r.expression_score,
                'disease_association_score': r.disease_association_score,
                'is_hub': r.is_hub,
                'centrality_score': r.centrality_score,
                'known_drugs': ','.join(r.known_drugs),
                'clinical_phase': r.clinical_phase,
                'n_pathways': len(r.pathways),
            })
        return pd.DataFrame(data)


def predict_drug_targets(
    genes: List[str],
    expression_data: Optional[pd.DataFrame] = None,
    network_result: Optional[object] = None,  # NetworkResult
    pathway_enrichment: Optional[pd.DataFrame] = None,
    druggability_database: Optional[pd.DataFrame] = None,
    ranking_method: str = 'weighted_score',
    weights: Optional[Dict[str, float]] = None,
) -> DrugTargetResults:
    """
    Predict drug targets from gene list

    Args:
        genes: List of genes to evaluate
        expression_data: Gene expression matrix
        network_result: NetworkResult with centrality measures
        pathway_enrichment: Pathway enrichment results
        druggability_database: Database of druggability scores
        ranking_method: Ranking method ('weighted_score', 'ml_based')
        weights: Weights for score components

    Returns:
        DrugTargetResults with ranked targets
    """
    # Load druggability database
    if druggability_database is None:
        druggability_database = _get_druggability_database()

    # Default weights
    if weights is None:
        weights = {
            'druggability': 0.3,
            'expression': 0.2,
            'disease_association': 0.2,
            'network': 0.15,
            'pathway': 0.15,
        }

    results = []

    for gene in genes:
        # Get druggability score
        druggability_score = _get_druggability_score(
            gene, druggability_database
        )

        # Get expression score
        expression_score = _calculate_expression_score(
            gene, expression_data
        ) if expression_data is not None else 0.5

        # Get disease association score
        disease_association_score = _get_disease_association_score(
            gene, druggability_database
        )

        # Get network centrality
        is_hub = False
        centrality_score = 0.0
        if network_result is not None:
            centrality_score = network_result.degree_centrality.get(gene, 0.0)
            is_hub = gene in network_result.hub_nodes

        # Get pathway membership
        pathways = []
        if pathway_enrichment is not None:
            pathways = _get_gene_pathways(gene, pathway_enrichment)

        pathway_score = min(len(pathways) / 10.0, 1.0)  # Normalize

        # Calculate combined score
        combined_score = (
            weights['druggability'] * druggability_score +
            weights['expression'] * expression_score +
            weights['disease_association'] * disease_association_score +
            weights['network'] * centrality_score +
            weights['pathway'] * pathway_score
        )

        # Get drug information
        known_drugs, drug_classes, clinical_phase = _get_drug_info(
            gene, druggability_database
        )

        # Create result
        result = DrugTargetResult(
            target_gene=gene,
            target_protein=_gene_to_protein(gene),
            druggability_score=float(druggability_score),
            binding_site_score=float(druggability_score),  # Simplified
            expression_score=float(expression_score),
            disease_association_score=float(disease_association_score),
            combined_score=float(combined_score),
            known_drugs=known_drugs,
            drug_classes=drug_classes,
            clinical_phase=clinical_phase,
            is_hub=is_hub,
            centrality_score=float(centrality_score),
            pathways=pathways,
        )

        results.append(result)

    # Sort by combined score
    results.sort(key=lambda x: x.combined_score, reverse=True)

    return DrugTargetResults(
        results=results,
        ranking_method=ranking_method,
    )


def _get_druggability_database() -> pd.DataFrame:
    """Get example druggability database"""
    # In production, would load from DGIdb, DrugBank, ChEMBL, etc.

    data = [
        # Gene, Druggability, Disease_Association, Known_Drugs, Clinical_Phase
        ('IL6', 0.95, 0.90, 'Tocilizumab;Siltuximab', 'Approved'),
        ('TNF', 0.95, 0.95, 'Infliximab;Adalimumab;Etanercept', 'Approved'),
        ('IL1B', 0.85, 0.85, 'Canakinumab;Anakinra', 'Approved'),
        ('NFKB1', 0.60, 0.80, '', ''),
        ('DRD2', 0.95, 0.90, 'Haloperidol;Risperidone;Aripiprazole', 'Approved'),
        ('HTR2A', 0.90, 0.85, 'Clozapine;Olanzapine', 'Approved'),
        ('SLC6A4', 0.85, 0.90, 'Fluoxetine;Sertraline;Escitalopram', 'Approved'),
        ('MTOR', 0.90, 0.85, 'Rapamycin;Everolimus', 'Approved'),
        ('GRIN1', 0.70, 0.75, 'Ketamine', 'Approved'),
        ('GRIN2A', 0.75, 0.70, '', ''),
        ('JAK1', 0.85, 0.80, 'Tofacitinib;Baricitinib', 'Approved'),
        ('STAT3', 0.65, 0.75, '', 'Phase 2'),
        ('COMT', 0.80, 0.70, 'Tolcapone;Entacapone', 'Approved'),
        ('MAOA', 0.85, 0.75, 'Phenelzine;Tranylcypromine', 'Approved'),
        ('BDNF', 0.40, 0.85, '', ''),
        ('GABA', 0.30, 0.70, '', ''),
        ('CD4', 0.50, 0.60, '', ''),
        ('IFNG', 0.75, 0.80, '', 'Phase 1'),
    ]

    df = pd.DataFrame(
        data,
        columns=[
            'gene',
            'druggability_score',
            'disease_association_score',
            'known_drugs',
            'clinical_phase'
        ]
    )

    return df


def _get_druggability_score(
    gene: str,
    database: pd.DataFrame,
) -> float:
    """Get druggability score for gene"""
    gene_data = database[database['gene'] == gene]

    if len(gene_data) > 0:
        return float(gene_data.iloc[0]['druggability_score'])
    else:
        # Default score for unknown genes
        return 0.5


def _calculate_expression_score(
    gene: str,
    expression_data: pd.DataFrame,
) -> float:
    """Calculate expression-based score"""
    if gene not in expression_data.columns:
        return 0.5

    expr = expression_data[gene]

    # Score based on:
    # 1. High expression (potential target)
    # 2. High variance (differential expression)

    mean_expr = expr.mean()
    std_expr = expr.std()

    # Normalize to 0-1
    # Assuming expression is log-normalized
    expr_score = min(mean_expr / 10.0, 1.0)
    variance_score = min(std_expr / 5.0, 1.0)

    combined = 0.6 * expr_score + 0.4 * variance_score

    return float(np.clip(combined, 0, 1))


def _get_disease_association_score(
    gene: str,
    database: pd.DataFrame,
) -> float:
    """Get disease association score"""
    gene_data = database[database['gene'] == gene]

    if len(gene_data) > 0:
        return float(gene_data.iloc[0]['disease_association_score'])
    else:
        return 0.5


def _get_drug_info(
    gene: str,
    database: pd.DataFrame,
) -> Tuple[List[str], List[str], Optional[str]]:
    """Get drug information for gene"""
    gene_data = database[database['gene'] == gene]

    if len(gene_data) > 0:
        row = gene_data.iloc[0]

        drugs_str = row['known_drugs']
        known_drugs = drugs_str.split(';') if drugs_str else []

        clinical_phase = row['clinical_phase'] if row['clinical_phase'] else None

        # Infer drug classes from drugs
        drug_classes = []
        if known_drugs:
            if any('mab' in d.lower() for d in known_drugs):
                drug_classes.append('Monoclonal antibody')
            if gene in ['DRD2', 'HTR2A']:
                drug_classes.append('Antipsychotic')
            if gene == 'SLC6A4':
                drug_classes.append('SSRI')
            if gene == 'MTOR':
                drug_classes.append('mTOR inhibitor')

        return known_drugs, drug_classes, clinical_phase
    else:
        return [], [], None


def _get_gene_pathways(
    gene: str,
    pathway_enrichment: pd.DataFrame,
) -> List[str]:
    """Get pathways containing gene"""
    # Simplified - assumes pathway_enrichment has pathway names
    # In production, would cross-reference with pathway databases

    pathways = []

    if 'pathway_name' in pathway_enrichment.columns:
        # Assume significant pathways
        pathways = pathway_enrichment['pathway_name'].tolist()

    return pathways


def _gene_to_protein(gene: str) -> str:
    """Convert gene name to protein name"""
    # Simplified mapping
    # In production, use UniProt or other databases

    return gene  # Most genes have same name as protein


def rank_druggable_targets(
    targets: DrugTargetResults,
    prioritize: str = 'novel',
) -> DrugTargetResults:
    """
    Re-rank targets with specific prioritization

    Args:
        targets: DrugTargetResults to re-rank
        prioritize: Prioritization strategy ('novel', 'approved', 'hubs')

    Returns:
        Re-ranked DrugTargetResults
    """
    results = targets.results.copy()

    if prioritize == 'novel':
        # Prioritize targets without known drugs
        results.sort(
            key=lambda x: (len(x.known_drugs) == 0, x.combined_score),
            reverse=True
        )
    elif prioritize == 'approved':
        # Prioritize targets with approved drugs
        results.sort(
            key=lambda x: (x.clinical_phase == 'Approved', x.combined_score),
            reverse=True
        )
    elif prioritize == 'hubs':
        # Prioritize hub nodes
        results.sort(
            key=lambda x: (x.is_hub, x.centrality_score, x.combined_score),
            reverse=True
        )
    else:
        # Keep original ranking
        pass

    return DrugTargetResults(
        results=results,
        ranking_method=f"{targets.ranking_method}_prioritized_{prioritize}",
    )


def find_approved_drugs(
    targets: DrugTargetResults,
    target_genes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Find approved drugs for target genes

    Args:
        targets: DrugTargetResults
        target_genes: Specific genes to query (if None, use all)

    Returns:
        DataFrame with drug-target mappings
    """
    drugs_data = []

    for result in targets.results:
        if target_genes and result.target_gene not in target_genes:
            continue

        if result.known_drugs:
            for drug in result.known_drugs:
                drugs_data.append({
                    'drug_name': drug,
                    'target_gene': result.target_gene,
                    'target_protein': result.target_protein,
                    'clinical_phase': result.clinical_phase,
                    'drug_classes': ','.join(result.drug_classes),
                    'target_score': result.combined_score,
                })

    df = pd.DataFrame(drugs_data)

    if len(df) > 0:
        df = df.sort_values('target_score', ascending=False)

    return df


def drug_repurposing_candidates(
    cluster_targets: Dict[int, List[str]],
    druggability_database: Optional[pd.DataFrame] = None,
    min_druggability: float = 0.7,
) -> pd.DataFrame:
    """
    Identify drug repurposing candidates across clusters

    Args:
        cluster_targets: Dictionary mapping cluster ID to target genes
        druggability_database: Druggability database
        min_druggability: Minimum druggability score

    Returns:
        DataFrame with repurposing candidates
    """
    if druggability_database is None:
        druggability_database = _get_druggability_database()

    candidates = []

    for cluster_id, genes in cluster_targets.items():
        # Predict targets for this cluster
        targets = predict_drug_targets(
            genes=genes,
            druggability_database=druggability_database,
        )

        # Filter by druggability
        high_drug_targets = [
            t for t in targets.results
            if t.druggability_score >= min_druggability and t.known_drugs
        ]

        for target in high_drug_targets:
            for drug in target.known_drugs:
                candidates.append({
                    'cluster_id': cluster_id,
                    'target_gene': target.target_gene,
                    'drug_name': drug,
                    'current_indication': _get_drug_indication(drug),
                    'druggability_score': target.druggability_score,
                    'combined_score': target.combined_score,
                    'clinical_phase': target.clinical_phase,
                })

    df = pd.DataFrame(candidates)

    if len(df) > 0:
        df = df.sort_values(['cluster_id', 'combined_score'], ascending=[True, False])

    return df


def _get_drug_indication(drug_name: str) -> str:
    """Get primary indication for drug"""
    # Simplified mapping
    # In production, use DrugBank or other databases

    indications = {
        'Tocilizumab': 'Rheumatoid arthritis',
        'Siltuximab': 'Castleman disease',
        'Infliximab': 'Inflammatory diseases',
        'Adalimumab': 'Rheumatoid arthritis',
        'Etanercept': 'Rheumatoid arthritis',
        'Haloperidol': 'Schizophrenia',
        'Risperidone': 'Schizophrenia',
        'Aripiprazole': 'Schizophrenia',
        'Clozapine': 'Schizophrenia',
        'Olanzapine': 'Schizophrenia',
        'Fluoxetine': 'Depression',
        'Sertraline': 'Depression',
        'Escitalopram': 'Depression',
        'Rapamycin': 'Immunosuppression',
        'Everolimus': 'Cancer',
        'Ketamine': 'Anesthesia',
        'Tofacitinib': 'Rheumatoid arthritis',
        'Baricitinib': 'Rheumatoid arthritis',
    }

    return indications.get(drug_name, 'Unknown')


def visualize_drug_targets(
    targets: DrugTargetResults,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize drug target scores

    Args:
        targets: DrugTargetResults
        top_n: Number of top targets to show
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show figure
    """
    import matplotlib.pyplot as plt

    top_targets = targets.top_targets(n=top_n)

    if not top_targets:
        warnings.warn("No targets to visualize")
        return

    # Create data
    genes = [t.target_gene for t in top_targets]
    scores = [t.combined_score for t in top_targets]
    has_drugs = ['Approved' if t.clinical_phase == 'Approved' else 'Novel' for t in top_targets]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color by drug status
    colors = ['green' if status == 'Approved' else 'steelblue' for status in has_drugs]

    bars = ax.barh(genes, scores, color=colors, alpha=0.7)

    ax.set_xlabel('Combined Druggability Score', fontsize=12)
    ax.set_ylabel('Target Gene', fontsize=12)
    ax.set_title(f'Top {top_n} Drug Targets', fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Approved drugs available'),
        Patch(facecolor='steelblue', alpha=0.7, label='Novel target'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.set_xlim([0, 1.0])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()