"""Explainable AI tools for cluster characterization

Provides SHAP-based explanations and feature importance analysis.
"""

from .classifier import (
    train_cluster_classifier,
    ClusterClassifierResult,
)

from .shap_analysis import (
    compute_shap_values,
    get_top_features_per_cluster,
    get_feature_interactions,
    identify_cluster_prototypes,
    ShapResult,
)

from .visualization import (
    plot_shap_waterfall,
    plot_shap_summary,
    plot_shap_beeswarm,
    plot_partial_dependence,
    plot_feature_importance,
    plot_interaction_heatmap,
)

__all__ = [
    # Classifier
    'train_cluster_classifier',
    'ClusterClassifierResult',
    # SHAP analysis
    'compute_shap_values',
    'get_top_features_per_cluster',
    'get_feature_interactions',
    'identify_cluster_prototypes',
    'ShapResult',
    # Visualization
    'plot_shap_waterfall',
    'plot_shap_summary',
    'plot_shap_beeswarm',
    'plot_partial_dependence',
    'plot_feature_importance',
    'plot_interaction_heatmap',
]