"""Interactive visualization dashboard using Plotly and Dash

Provides comprehensive interactive tools for exploring multi-omics clustering results.
"""

from .embedding_plots import (
    create_embedding_plot,
    create_multi_overlay_plot,
    EmbeddingPlotConfig,
)

from .cluster_comparison import (
    create_cluster_comparison,
    create_violin_comparison,
    create_sankey_diagram,
    ClusterComparisonResult,
)

from .trajectories import (
    create_trajectory_plot,
    create_patient_timeline,
    create_transition_matrix,
    TrajectoryResult,
)

from .heatmaps import (
    create_biomarker_heatmap,
    create_interactive_heatmap,
    create_correlation_heatmap,
    HeatmapConfig,
)

from .clinical_interface import (
    create_patient_card,
    create_risk_assessment,
    create_treatment_recommendations,
    ClinicalDecisionSupport,
)

from .dashboard import (
    create_dashboard,
    run_dashboard,
)

# Pipeline wrapper functions
def plot_embedding(embedding, clusters, output_path=None, **kwargs):
    """Wrapper for embedding plot"""
    return create_embedding_plot(
        embedding=embedding,
        labels=clusters,
        save_path=output_path,
        **kwargs
    )

def plot_heatmaps(data_dict, clusters, output_dir=None, **kwargs):
    """Wrapper for heatmap plots"""
    paths = []
    for modality, data in data_dict.items():
        path = create_biomarker_heatmap(
            data=data,
            labels=clusters,
            save_path=output_dir / f'{modality}_heatmap.png' if output_dir else None,
            **kwargs
        )
        if path:
            paths.append(path)
    return paths

def plot_cluster_comparison(data_dict, clusters, output_path=None, **kwargs):
    """Wrapper for cluster comparison"""
    return create_cluster_comparison(
        data=data_dict,
        labels=clusters,
        save_path=output_path,
        **kwargs
    )

__all__ = [
    # Embedding plots
    'create_embedding_plot',
    'create_multi_overlay_plot',
    'EmbeddingPlotConfig',
    # Cluster comparison
    'create_cluster_comparison',
    'create_violin_comparison',
    'create_sankey_diagram',
    'ClusterComparisonResult',
    # Trajectories
    'create_trajectory_plot',
    'create_patient_timeline',
    'create_transition_matrix',
    'TrajectoryResult',
    # Heatmaps
    'create_biomarker_heatmap',
    'create_interactive_heatmap',
    'create_correlation_heatmap',
    'HeatmapConfig',
    # Clinical interface
    'create_patient_card',
    'create_risk_assessment',
    'create_treatment_recommendations',
    'ClinicalDecisionSupport',
    # Dashboard
    'create_dashboard',
    'run_dashboard',
    # Pipeline wrappers
    'plot_embedding',
    'plot_heatmaps',
    'plot_cluster_comparison',
]