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

# Pipeline wrapper functions
def run_pathway_enrichment(data, clusters, modality='genomic', **kwargs):
    """Wrapper for pathway enrichment analysis"""
    if modality == 'genomic':
        return run_gsea(data, clusters, **kwargs)
    elif modality == 'metabolomic':
        return integrate_multiomics_pathways({'metabolomic': data}, clusters, **kwargs)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def build_biological_networks(data_dict, clusters, **kwargs):
    """Wrapper for network analysis"""
    if 'genomic' in data_dict:
        return analyze_ppi_network(data_dict['genomic'], clusters, **kwargs)
    elif 'metabolomic' in data_dict:
        return reconstruct_metabolic_network(data_dict['metabolomic'], clusters, **kwargs)
    return None

def identify_drug_targets(pathway_results, clusters, **kwargs):
    """Wrapper for drug target identification"""
    return predict_drug_targets(pathway_results, clusters, **kwargs)

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
    # Pipeline wrappers
    'run_pathway_enrichment',
    'build_biological_networks',
    'identify_drug_targets',
]