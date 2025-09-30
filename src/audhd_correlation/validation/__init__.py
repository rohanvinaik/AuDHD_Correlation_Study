"""Validation tools for clustering results

Comprehensive validation framework including:
- Internal metrics (silhouette, Dunn, etc.)
- Stability analysis (bootstrap, subsampling)
- Biological validity (pathway enrichment)
- Clinical relevance
- Cross-validation across sites
"""

from .metrics import (
    compute_internal_metrics,
    compute_stability_metrics,
    compute_outlier_robustness,
    compute_balanced_metrics,
    InternalValidationMetrics,
)

from .stability import (
    bootstrap_stability,
    subsampling_stability,
    noise_stability,
    feature_stability,
    permutation_test_stability,
    StabilityResult,
)

from .biological import (
    pathway_enrichment_analysis,
    clinical_relevance_analysis,
    symptom_severity_analysis,
    diagnostic_concordance,
    functional_outcome_prediction,
    EnrichmentResult,
    ClinicalRelevanceResult,
)

from .cross_validation import (
    cross_site_validation,
    cross_cohort_validation,
    stratified_cross_validation,
    temporal_validation,
    batch_effect_validation,
    CrossValidationResult,
)

__all__ = [
    # Metrics
    'compute_internal_metrics',
    'compute_stability_metrics',
    'compute_outlier_robustness',
    'compute_balanced_metrics',
    'InternalValidationMetrics',
    # Stability
    'bootstrap_stability',
    'subsampling_stability',
    'noise_stability',
    'feature_stability',
    'permutation_test_stability',
    'StabilityResult',
    # Biological
    'pathway_enrichment_analysis',
    'clinical_relevance_analysis',
    'symptom_severity_analysis',
    'diagnostic_concordance',
    'functional_outcome_prediction',
    'EnrichmentResult',
    'ClinicalRelevanceResult',
    # Cross-validation
    'cross_site_validation',
    'cross_cohort_validation',
    'stratified_cross_validation',
    'temporal_validation',
    'batch_effect_validation',
    'CrossValidationResult',
]