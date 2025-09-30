"""Biological interpretation tools for multi-omics analysis

Provides pathway enrichment, network analysis, and drug target prediction.
"""

from .gsea import (
    run_gsea,
    prerank_gsea,
    load_gene_sets,
    GSEAResult,
)

from .networks import (
    reconstruct_metabolic_network,
    analyze_ppi_network,
    find_hub_nodes,
    community_detection,
    NetworkResult,
)

from .pathway_integration import (
    integrate_multiomics_pathways,
    combined_pathway_score,
    cross_omics_enrichment,
    MultiOmicsPathwayResult,
)

from .drug_targets import (
    predict_drug_targets,
    rank_druggable_targets,
    find_approved_drugs,
    DrugTargetResult,
)

__all__ = [
    # GSEA
    'run_gsea',
    'prerank_gsea',
    'load_gene_sets',
    'GSEAResult',
    # Networks
    'reconstruct_metabolic_network',
    'analyze_ppi_network',
    'find_hub_nodes',
    'community_detection',
    'NetworkResult',
    # Pathway Integration
    'integrate_multiomics_pathways',
    'combined_pathway_score',
    'cross_omics_enrichment',
    'MultiOmicsPathwayResult',
    # Drug Targets
    'predict_drug_targets',
    'rank_druggable_targets',
    'find_approved_drugs',
    'DrugTargetResult',
]