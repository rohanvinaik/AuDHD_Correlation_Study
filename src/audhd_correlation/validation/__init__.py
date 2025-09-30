"""Validation tools for clustering results

Comprehensive validation framework including:
- Internal metrics (silhouette, Dunn, etc.)
- Stability analysis (bootstrap, subsampling)
- Biological validity (pathway enrichment)
- Clinical relevance
- Cross-validation across sites
- External validation and projection
- Ancestry-stratified validation
- Prospective outcome prediction
- Meta-analysis capabilities
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

from .report import (
    generate_validation_report,
    save_report,
    ValidationReport,
)

from .external import (
    EmbeddingProjector,
    NearestCentroidClassifier,
    validate_external_cohort,
    calculate_replication_metrics,
    ProjectionResult,
    ValidationMetrics,
)

from .cross_cohort import (
    CrossCohortAnalyzer,
    test_heterogeneity,
    calculate_cross_cohort_stability,
    CrossCohortResult,
    EffectSizeComparison,
)

from .ancestry_stratified import (
    AncestryStratifiedValidator,
    compare_ancestry_specific_effects,
    calculate_transferability_score,
    AncestryValidationResult,
    PopulationStratificationTest,
)

from .prospective import (
    OutcomePredictor,
    ProspectiveValidator,
    predict_treatment_response,
    calculate_time_to_event_predictions,
    OutcomePrediction,
    ProspectiveValidationResult,
)

from .meta_analysis import (
    MetaAnalyzer,
    test_publication_bias,
    combine_cohort_results,
    StudyResult,
    MetaAnalysisResult,
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
    # Report
    'generate_validation_report',
    'save_report',
    'ValidationReport',
    # External validation
    'EmbeddingProjector',
    'NearestCentroidClassifier',
    'validate_external_cohort',
    'calculate_replication_metrics',
    'ProjectionResult',
    'ValidationMetrics',
    # Cross-cohort
    'CrossCohortAnalyzer',
    'test_heterogeneity',
    'calculate_cross_cohort_stability',
    'CrossCohortResult',
    'EffectSizeComparison',
    # Ancestry-stratified
    'AncestryStratifiedValidator',
    'compare_ancestry_specific_effects',
    'calculate_transferability_score',
    'AncestryValidationResult',
    'PopulationStratificationTest',
    # Prospective
    'OutcomePredictor',
    'ProspectiveValidator',
    'predict_treatment_response',
    'calculate_time_to_event_predictions',
    'OutcomePrediction',
    'ProspectiveValidationResult',
    # Meta-analysis
    'MetaAnalyzer',
    'test_publication_bias',
    'combine_cohort_results',
    'StudyResult',
    'MetaAnalysisResult',
]