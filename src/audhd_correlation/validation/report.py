"""Validation report generation

Generates comprehensive validation reports combining all metrics.
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    EnrichmentResult,
    ClinicalRelevanceResult,
)
from .cross_validation import (
    cross_site_validation,
    stratified_cross_validation,
    CrossValidationResult,
)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    # Internal metrics
    internal_metrics: InternalValidationMetrics
    stability_metrics: Dict[str, float]
    outlier_robustness: Dict[str, float]
    balanced_metrics: Dict[str, float]

    # Stability analysis
    bootstrap_result: Optional[StabilityResult] = None
    subsampling_results: Optional[Dict[str, StabilityResult]] = None
    noise_results: Optional[Dict[str, StabilityResult]] = None
    feature_results: Optional[Dict[str, StabilityResult]] = None
    permutation_result: Optional[Dict[str, float]] = None

    # Biological validity
    enrichment_results: Optional[List[EnrichmentResult]] = None
    clinical_results: Optional[List[ClinicalRelevanceResult]] = None

    # Cross-validation
    cross_site_result: Optional[CrossValidationResult] = None
    stratified_cv_result: Optional[CrossValidationResult] = None

    # Summary
    overall_quality: Optional[float] = None
    recommendations: Optional[List[str]] = None


def generate_validation_report(
    X: np.ndarray,
    labels: np.ndarray,
    clustering_func: Optional[Callable] = None,
    feature_names: Optional[List[str]] = None,
    pathway_dict: Optional[Dict[str, List[str]]] = None,
    clinical_data: Optional[pd.DataFrame] = None,
    site_labels: Optional[np.ndarray] = None,
    include_bootstrap: bool = True,
    include_subsampling: bool = True,
    include_noise: bool = True,
    include_features: bool = True,
    include_permutation: bool = True,
    include_enrichment: bool = False,
    include_clinical: bool = False,
    include_cross_site: bool = False,
    include_stratified_cv: bool = True,
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> ValidationReport:
    """
    Generate comprehensive validation report

    Args:
        X: Data matrix
        labels: Cluster labels
        clustering_func: Clustering function (required for stability tests)
        feature_names: Feature names (for enrichment)
        pathway_dict: Pathway dictionary (for enrichment)
        clinical_data: Clinical data (for relevance tests)
        site_labels: Site labels (for cross-site validation)
        include_bootstrap: Include bootstrap stability
        include_subsampling: Include subsampling stability
        include_noise: Include noise stability
        include_features: Include feature stability
        include_permutation: Include permutation test
        include_enrichment: Include pathway enrichment
        include_clinical: Include clinical relevance
        include_cross_site: Include cross-site validation
        include_stratified_cv: Include stratified cross-validation
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed

    Returns:
        ValidationReport
    """
    # Internal metrics (always computed)
    internal_metrics = compute_internal_metrics(X, labels, handle_outliers=True)
    stability_metrics = compute_stability_metrics(X, labels)
    outlier_robustness = compute_outlier_robustness(X, labels, n_iterations=10)
    balanced_metrics = compute_balanced_metrics(X, labels)

    # Stability analysis
    bootstrap_result = None
    subsampling_results = None
    noise_results = None
    feature_results = None
    permutation_result = None

    if clustering_func is not None:
        if include_bootstrap:
            try:
                bootstrap_result = bootstrap_stability(
                    X, labels, clustering_func, n_bootstrap=n_bootstrap, random_state=random_state
                )
            except Exception as e:
                warnings.warn(f"Bootstrap stability failed: {e}")

        if include_subsampling:
            try:
                subsampling_results = subsampling_stability(
                    X, labels, clustering_func, n_iterations=20, random_state=random_state
                )
            except Exception as e:
                warnings.warn(f"Subsampling stability failed: {e}")

        if include_noise:
            try:
                noise_results = noise_stability(
                    X, labels, clustering_func, n_iterations=20, random_state=random_state
                )
            except Exception as e:
                warnings.warn(f"Noise stability failed: {e}")

        if include_features and X.shape[1] > 5:
            try:
                feature_results = feature_stability(
                    X, labels, clustering_func, n_iterations=20, random_state=random_state
                )
            except Exception as e:
                warnings.warn(f"Feature stability failed: {e}")

        if include_permutation:
            try:
                permutation_result = permutation_test_stability(
                    X, labels, clustering_func, n_permutations=100, random_state=random_state
                )
            except Exception as e:
                warnings.warn(f"Permutation test failed: {e}")

    # Biological validity
    enrichment_results = None
    clinical_results = None

    if include_enrichment and feature_names is not None and pathway_dict is not None:
        try:
            enrichment_results = pathway_enrichment_analysis(
                labels, feature_names, pathway_dict
            )
        except Exception as e:
            warnings.warn(f"Pathway enrichment failed: {e}")

    if include_clinical and clinical_data is not None:
        try:
            clinical_results = clinical_relevance_analysis(labels, clinical_data)
        except Exception as e:
            warnings.warn(f"Clinical relevance analysis failed: {e}")

    # Cross-validation
    cross_site_result = None
    stratified_cv_result = None

    if clustering_func is not None:
        if include_cross_site and site_labels is not None:
            try:
                cross_site_result = cross_site_validation(
                    X, labels, site_labels, clustering_func
                )
            except Exception as e:
                warnings.warn(f"Cross-site validation failed: {e}")

        if include_stratified_cv:
            try:
                stratified_cv_result = stratified_cross_validation(
                    X, labels, clustering_func, n_folds=5, random_state=random_state
                )
            except Exception as e:
                warnings.warn(f"Stratified cross-validation failed: {e}")

    # Compute overall quality and recommendations
    overall_quality = _compute_overall_quality(
        internal_metrics, stability_metrics, bootstrap_result,
        permutation_result, stratified_cv_result
    )

    recommendations = _generate_recommendations(
        internal_metrics, stability_metrics, outlier_robustness,
        bootstrap_result, permutation_result, enrichment_results,
        clinical_results, cross_site_result, stratified_cv_result
    )

    return ValidationReport(
        internal_metrics=internal_metrics,
        stability_metrics=stability_metrics,
        outlier_robustness=outlier_robustness,
        balanced_metrics=balanced_metrics,
        bootstrap_result=bootstrap_result,
        subsampling_results=subsampling_results,
        noise_results=noise_results,
        feature_results=feature_results,
        permutation_result=permutation_result,
        enrichment_results=enrichment_results,
        clinical_results=clinical_results,
        cross_site_result=cross_site_result,
        stratified_cv_result=stratified_cv_result,
        overall_quality=overall_quality,
        recommendations=recommendations,
    )


def _compute_overall_quality(
    internal_metrics: InternalValidationMetrics,
    stability_metrics: Dict[str, float],
    bootstrap_result: Optional[StabilityResult],
    permutation_result: Optional[Dict[str, float]],
    stratified_cv_result: Optional[CrossValidationResult],
) -> float:
    """Compute overall quality score (0-1)"""
    scores = []

    # Internal quality (normalized)
    if internal_metrics.overall_quality is not None:
        scores.append(internal_metrics.overall_quality)

    # Bootstrap stability
    if bootstrap_result is not None:
        scores.append(bootstrap_result.stability_score)

    # Permutation significance
    if permutation_result is not None and 'p_value' in permutation_result:
        # Significant clustering gets full score
        p_value = permutation_result['p_value']
        sig_score = 1.0 if p_value < 0.05 else 0.5
        scores.append(sig_score)

    # Cross-validation generalization
    if stratified_cv_result is not None:
        scores.append(stratified_cv_result.generalization_score)

    if len(scores) == 0:
        return 0.5

    return np.mean(scores)


def _generate_recommendations(
    internal_metrics: InternalValidationMetrics,
    stability_metrics: Dict[str, float],
    outlier_robustness: Dict[str, float],
    bootstrap_result: Optional[StabilityResult],
    permutation_result: Optional[Dict[str, float]],
    enrichment_results: Optional[List[EnrichmentResult]],
    clinical_results: Optional[List[ClinicalRelevanceResult]],
    cross_site_result: Optional[CrossValidationResult],
    stratified_cv_result: Optional[CrossValidationResult],
) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []

    # Internal quality
    if internal_metrics.silhouette < 0.25:
        recommendations.append(
            "LOW INTERNAL QUALITY: Silhouette score < 0.25. Consider increasing/decreasing "
            "number of clusters or trying a different clustering algorithm."
        )

    if internal_metrics.davies_bouldin > 2.0:
        recommendations.append(
            "HIGH CLUSTER OVERLAP: Davies-Bouldin index > 2.0. Clusters may not be well-separated. "
            "Consider refining cluster number or feature selection."
        )

    # Imbalance
    if stability_metrics.get('imbalance_ratio', 1.0) > 3.0:
        recommendations.append(
            "SEVERE IMBALANCE: Some clusters are much larger than others. "
            "Consider using balanced metrics or stratified sampling."
        )

    # Outlier sensitivity
    if outlier_robustness.get('silhouette_robustness', 1.0) < 0.7:
        recommendations.append(
            "HIGH OUTLIER SENSITIVITY: Quality degrades significantly with outliers. "
            "Consider using robust clustering methods or outlier removal."
        )

    # Bootstrap stability
    if bootstrap_result is not None:
        if bootstrap_result.stability_score < 0.6:
            recommendations.append(
                "LOW STABILITY: Bootstrap stability score < 0.6. Clustering is not reproducible. "
                "Consider collecting more data or using more stable methods."
            )

    # Statistical significance
    if permutation_result is not None:
        p_value = permutation_result.get('p_value', 0.0)
        if p_value > 0.05:
            recommendations.append(
                "NOT STATISTICALLY SIGNIFICANT: Clustering is not better than random (p > 0.05). "
                "Re-evaluate whether meaningful structure exists in the data."
            )

    # Biological/clinical validity
    if enrichment_results is not None:
        n_enriched = sum(1 for r in enrichment_results if r.enriched)
        if n_enriched == 0:
            recommendations.append(
                "NO PATHWAY ENRICHMENT: Clusters do not show enrichment for known pathways. "
                "Consider whether clusters reflect biological reality."
            )

    if clinical_results is not None:
        n_significant = sum(1 for r in clinical_results if r.significant)
        if n_significant == 0:
            recommendations.append(
                "NO CLINICAL RELEVANCE: Clusters do not differ in clinical variables. "
                "Clusters may lack clinical utility."
            )

    # Generalization
    if cross_site_result is not None:
        if cross_site_result.generalization_score < 0.5:
            recommendations.append(
                "POOR CROSS-SITE GENERALIZATION: Clustering does not replicate across sites. "
                "May be driven by site-specific artifacts or batch effects."
            )

    if stratified_cv_result is not None:
        if stratified_cv_result.generalization_score < 0.5:
            recommendations.append(
                "POOR CROSS-VALIDATION: Clustering does not generalize to held-out samples. "
                "May be overfitting to the specific dataset."
            )

    if len(recommendations) == 0:
        recommendations.append(
            "EXCELLENT VALIDATION: All metrics indicate high-quality, stable, and meaningful clustering."
        )

    return recommendations


def save_report(
    report: ValidationReport,
    output_dir: Path,
    format: str = 'json',
) -> None:
    """
    Save validation report to file

    Args:
        report: ValidationReport
        output_dir: Output directory
        format: Output format ('json', 'txt', or 'html')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        _save_json_report(report, output_dir / 'validation_report.json')
    elif format == 'txt':
        _save_text_report(report, output_dir / 'validation_report.txt')
    elif format == 'html':
        _save_html_report(report, output_dir / 'validation_report.html')
    else:
        raise ValueError(f"Unknown format: {format}")


def _save_json_report(report: ValidationReport, output_path: Path) -> None:
    """Save report as JSON"""

    def convert_to_serializable(obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return {k: convert_to_serializable(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    report_dict = convert_to_serializable(report)

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)


def _save_text_report(report: ValidationReport, output_path: Path) -> None:
    """Save report as plain text"""

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Overall quality
        if report.overall_quality is not None:
            f.write(f"Overall Quality Score: {report.overall_quality:.3f}\n\n")

        # Internal metrics
        f.write("INTERNAL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Silhouette Score: {report.internal_metrics.silhouette:.3f}\n")
        f.write(f"Davies-Bouldin Index: {report.internal_metrics.davies_bouldin:.3f}\n")
        f.write(f"Calinski-Harabasz Score: {report.internal_metrics.calinski_harabasz:.1f}\n")
        f.write(f"Dunn Index: {report.internal_metrics.dunn_index:.3f}\n")
        f.write(f"Variance Ratio: {report.internal_metrics.variance_ratio:.3f}\n")

        if report.internal_metrics.overall_quality is not None:
            f.write(f"Overall Internal Quality: {report.internal_metrics.overall_quality:.3f}\n")

        f.write("\n")

        # Stability metrics
        f.write("STABILITY METRICS\n")
        f.write("-" * 80 + "\n")
        for key, value in report.stability_metrics.items():
            f.write(f"{key}: {value:.3f}\n")
        f.write("\n")

        # Bootstrap stability
        if report.bootstrap_result is not None:
            f.write("BOOTSTRAP STABILITY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean ARI: {report.bootstrap_result.mean_ari:.3f} "
                   f"± {report.bootstrap_result.std_ari:.3f}\n")
            f.write(f"95% CI: [{report.bootstrap_result.confidence_interval_ari[0]:.3f}, "
                   f"{report.bootstrap_result.confidence_interval_ari[1]:.3f}]\n")
            f.write(f"Stability Score: {report.bootstrap_result.stability_score:.3f}\n")
            f.write(f"Interpretation: {report.bootstrap_result.interpretation}\n\n")

        # Permutation test
        if report.permutation_result is not None:
            f.write("PERMUTATION TEST\n")
            f.write("-" * 80 + "\n")
            f.write(f"Observed ARI: {report.permutation_result.get('observed_ari', 0):.3f}\n")
            f.write(f"Random ARI (mean): {report.permutation_result.get('random_ari_mean', 0):.3f}\n")
            f.write(f"P-value: {report.permutation_result.get('p_value', 1.0):.4f}\n")
            sig = "YES" if report.permutation_result.get('p_value', 1.0) < 0.05 else "NO"
            f.write(f"Significant (p < 0.05): {sig}\n\n")

        # Cross-validation
        if report.stratified_cv_result is not None:
            f.write("STRATIFIED CROSS-VALIDATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean ARI: {report.stratified_cv_result.mean_ari:.3f} "
                   f"± {report.stratified_cv_result.std_ari:.3f}\n")
            f.write(f"Generalization Score: {report.stratified_cv_result.generalization_score:.3f}\n")
            f.write(f"Interpretation: {report.stratified_cv_result.interpretation}\n\n")

        # Enrichment
        if report.enrichment_results is not None and len(report.enrichment_results) > 0:
            f.write("PATHWAY ENRICHMENT\n")
            f.write("-" * 80 + "\n")
            enriched = [r for r in report.enrichment_results if r.enriched]
            f.write(f"Total pathways tested: {len(report.enrichment_results)}\n")
            f.write(f"Significantly enriched: {len(enriched)}\n\n")

            if len(enriched) > 0:
                f.write("Top enriched pathways:\n")
                for r in sorted(enriched, key=lambda x: x.adjusted_p_value)[:10]:
                    f.write(f"  - {r.pathway_name} (Cluster {r.cluster_id}): "
                           f"OR={r.odds_ratio:.2f}, p={r.adjusted_p_value:.4f}\n")
            f.write("\n")

        # Clinical relevance
        if report.clinical_results is not None and len(report.clinical_results) > 0:
            f.write("CLINICAL RELEVANCE\n")
            f.write("-" * 80 + "\n")
            significant = [r for r in report.clinical_results if r.significant]
            f.write(f"Total variables tested: {len(report.clinical_results)}\n")
            f.write(f"Significantly different: {len(significant)}\n\n")

            if len(significant) > 0:
                f.write("Significant clinical variables:\n")
                for r in sorted(significant, key=lambda x: x.adjusted_p_value):
                    f.write(f"  - {r.clinical_variable}: {r.interpretation}, "
                           f"p={r.adjusted_p_value:.4f}\n")
            f.write("\n")

        # Recommendations
        if report.recommendations is not None:
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n\n")


def _save_html_report(report: ValidationReport, output_path: Path) -> None:
    """Save report as HTML"""

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
        .metric { margin: 10px 0; }
        .metric-name { font-weight: bold; }
        .metric-value { color: #2980b9; }
        .recommendation {
            background: #ecf0f1;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
        }
        .good { color: #27ae60; }
        .warning { color: #f39c12; }
        .bad { color: #e74c3c; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #bdc3c7; padding: 8px; text-align: left; }
        th { background-color: #3498db; color: white; }
    </style>
</head>
<body>
"""

    html += "<h1>Validation Report</h1>\n"

    # Overall quality
    if report.overall_quality is not None:
        quality_class = "good" if report.overall_quality > 0.7 else ("warning" if report.overall_quality > 0.5 else "bad")
        html += f"<p class='metric'><span class='metric-name'>Overall Quality Score:</span> "
        html += f"<span class='metric-value {quality_class}'>{report.overall_quality:.3f}</span></p>\n"

    # Internal metrics
    html += "<h2>Internal Metrics</h2>\n"
    html += f"<p class='metric'>Silhouette Score: <span class='metric-value'>{report.internal_metrics.silhouette:.3f}</span></p>\n"
    html += f"<p class='metric'>Davies-Bouldin Index: <span class='metric-value'>{report.internal_metrics.davies_bouldin:.3f}</span></p>\n"
    html += f"<p class='metric'>Calinski-Harabasz Score: <span class='metric-value'>{report.internal_metrics.calinski_harabasz:.1f}</span></p>\n"

    # Recommendations
    if report.recommendations is not None:
        html += "<h2>Recommendations</h2>\n"
        for rec in report.recommendations:
            html += f"<div class='recommendation'>{rec}</div>\n"

    html += "</body>\n</html>"

    with open(output_path, 'w') as f:
        f.write(html)