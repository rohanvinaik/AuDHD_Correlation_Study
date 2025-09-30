"""
Analysis module for genetic lookups, multi-omics, and synthesis
"""

from .genetic_lookup import (
    GeneticAnalysisSystem,
    GeneticLookupResult,
    NCBIClient,
    PubMedClient,
    LLMSynthesizer,
    quick_gene_lookup,
    quick_variant_lookup
)

from .pipeline_integration import (
    PipelineGeneticAnalysis,
    run_integrated_pipeline
)

from .multiomics_lookup import (
    MultiOmicsAnalysisSystem,
    TranscriptLookupResult,
    ProteinLookupResult,
    MetaboliteLookupResult,
    TranscriptomicsClient,
    ProteomicsClient,
    MetabolomicsClient,
    quick_transcript_lookup,
    quick_protein_lookup,
    quick_cross_omics
)

__all__ = [
    # Genetic analysis
    'GeneticAnalysisSystem',
    'GeneticLookupResult',
    'NCBIClient',
    'PubMedClient',
    'LLMSynthesizer',
    'quick_gene_lookup',
    'quick_variant_lookup',
    # Pipeline integration
    'PipelineGeneticAnalysis',
    'run_integrated_pipeline',
    # Multi-omics analysis
    'MultiOmicsAnalysisSystem',
    'TranscriptLookupResult',
    'ProteinLookupResult',
    'MetaboliteLookupResult',
    'TranscriptomicsClient',
    'ProteomicsClient',
    'MetabolomicsClient',
    'quick_transcript_lookup',
    'quick_protein_lookup',
    'quick_cross_omics'
]
