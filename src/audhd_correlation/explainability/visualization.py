"""Visualization tools for explainability

SHAP plots, feature importance, and partial dependence plots.
"""
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

from .classifier import ClusterClassifierResult
from .shap_analysis import ShapResult


def plot_shap_waterfall(
    shap_result: ShapResult,
    sample_index: int,
    cluster_id: Optional[int] = None,
    max_display: int = 20,
    show: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generate SHAP waterfall plot for individual sample

    Shows how each feature contributes to the prediction.

    Args:
        shap_result: ShapResult
        sample_index: Sample index
        cluster_id: Target cluster (if None, use predicted cluster)
        max_display: Maximum features to display
        show: Show plot
        save_path: Path to save figure

    Returns:
        Figure
    """
    if cluster_id is None:
        # Use cluster with highest SHAP sum
        shap_sums = np.sum(shap_result.shap_values[sample_index], axis=0)
        cluster_id = int(np.argmax(shap_sums))

    # Create SHAP Explanation object
    explanation = shap.Explanation(
        values=shap_result.shap_values[sample_index, :, cluster_id],
        base_values=shap_result.base_values[sample_index, cluster_id],
        data=shap_result.data[sample_index],
        feature_names=shap_result.feature_names,
    )

    # Create waterfall plot
    fig = plt.figure(figsize=(10, max(6, max_display * 0.3)))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)

    plt.title(f"SHAP Waterfall Plot - Sample {sample_index}, Cluster {cluster_id}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_shap_summary(
    shap_result: ShapResult,
    cluster_id: int,
    plot_type: str = 'dot',
    max_display: int = 20,
    show: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generate SHAP summary plot for cluster

    Shows feature importance and value distributions.

    Args:
        shap_result: ShapResult
        cluster_id: Target cluster
        plot_type: 'dot', 'bar', or 'violin'
        max_display: Maximum features to display
        show: Show plot
        save_path: Path to save figure

    Returns:
        Figure
    """
    # Get samples from this cluster
    cluster_mask = shap_result.cluster_labels == cluster_id

    # Create figure
    fig = plt.figure(figsize=(10, max(6, max_display * 0.3)))

    # SHAP summary plot
    shap.summary_plot(
        shap_result.shap_values[cluster_mask, :, cluster_id],
        shap_result.data[cluster_mask],
        feature_names=shap_result.feature_names,
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )

    plt.title(f"SHAP Summary Plot - Cluster {cluster_id}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_shap_beeswarm(
    shap_result: ShapResult,
    cluster_id: int,
    max_display: int = 20,
    show: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generate SHAP beeswarm plot for cluster

    Modern alternative to summary plot with better visualization.

    Args:
        shap_result: ShapResult
        cluster_id: Target cluster
        max_display: Maximum features to display
        show: Show plot
        save_path: Path to save figure

    Returns:
        Figure
    """
    cluster_mask = shap_result.cluster_labels == cluster_id

    # Create Explanation object for beeswarm
    explanation = shap.Explanation(
        values=shap_result.shap_values[cluster_mask, :, cluster_id],
        base_values=shap_result.base_values[cluster_mask, cluster_id],
        data=shap_result.data[cluster_mask],
        feature_names=shap_result.feature_names,
    )

    fig = plt.figure(figsize=(10, max(6, max_display * 0.3)))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)

    plt.title(f"SHAP Beeswarm Plot - Cluster {cluster_id}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_feature_importance(
    classifier_result: ClusterClassifierResult,
    n_features: int = 20,
    show_std: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot Random Forest feature importance

    Args:
        classifier_result: ClusterClassifierResult
        n_features: Number of top features to show
        show_std: Show standard deviation across trees
        figsize: Figure size
        show: Show plot
        save_path: Path to save figure

    Returns:
        Figure
    """
    # Get top features
    sorted_features = sorted(
        classifier_result.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n_features]

    feature_names = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]

    if show_std:
        stds = [classifier_result.feature_importance_std[f] for f in feature_names]
    else:
        stds = None

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(feature_names))

    ax.barh(y_pos, importances, xerr=stds if show_std else None, alpha=0.7, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {n_features} Features by Random Forest Importance')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_shap_importance_comparison(
    shap_result: ShapResult,
    n_features: int = 15,
    figsize: Tuple[int, int] = (12, 8),
    show: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare SHAP feature importance across clusters

    Args:
        shap_result: ShapResult
        n_features: Number of top features
        figsize: Figure size
        show: Show plot
        save_path: Path to save figure

    Returns:
        Figure
    """
    unique_clusters = np.unique(shap_result.cluster_labels)
    n_clusters = len(unique_clusters)

    # Get all unique top features across clusters
    all_top_features = set()
    for cluster_id in unique_clusters:
        top_features = shap_result.top_features_per_cluster[int(cluster_id)][:n_features]
        all_top_features.update([f[0] for f in top_features])

    all_top_features = sorted(all_top_features)

    # Create matrix of mean absolute SHAP values
    importance_matrix = np.zeros((len(all_top_features), n_clusters))

    for i, feature in enumerate(all_top_features):
        for j, cluster_id in enumerate(unique_clusters):
            importance_matrix[i, j] = shap_result.mean_abs_shap_per_cluster[int(cluster_id)].get(feature, 0.0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        importance_matrix,
        xticklabels=[f"Cluster {c}" for c in unique_clusters],
        yticklabels=all_top_features,
        cmap='YlOrRd',
        annot=False,
        fmt='.3f',
        cbar_kws={'label': 'Mean |SHAP value|'},
        ax=ax,
    )

    ax.set_title('SHAP Feature Importance Across Clusters')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Feature')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_interaction_heatmap(
    interaction_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot feature interaction heatmap

    Args:
        interaction_df: Interaction matrix from get_feature_interactions()
        figsize: Figure size
        show: Show plot
        save_path: Path to save figure

    Returns:
        Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        interaction_df,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.3f',
        square=True,
        cbar_kws={'label': 'Interaction Strength'},
        ax=ax,
    )

    ax.set_title('SHAP Feature Interaction Strength')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_partial_dependence(
    classifier_result: ClusterClassifierResult,
    X: np.ndarray,
    features: Union[List[int], List[str]],
    cluster_id: int,
    kind: str = 'average',
    grid_resolution: int = 50,
    figsize: Tuple[int, int] = (12, 4),
    show: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot partial dependence for features

    Shows how predictions change as feature values vary.

    Args:
        classifier_result: ClusterClassifierResult
        X: Feature matrix
        features: Feature indices or names
        cluster_id: Target cluster
        kind: 'average', 'individual', or 'both'
        grid_resolution: Number of grid points
        figsize: Figure size
        show: Show plot
        save_path: Path to save figure

    Returns:
        Figure
    """
    # Convert feature names to indices if needed
    if isinstance(features[0], str):
        feature_indices = [
            classifier_result.feature_names.index(f) for f in features
        ]
    else:
        feature_indices = features

    # Create figure
    fig, axes = plt.subplots(1, len(feature_indices), figsize=figsize)
    if len(feature_indices) == 1:
        axes = [axes]

    # Compute partial dependence
    pd_results = partial_dependence(
        classifier_result.classifier,
        X,
        features=feature_indices,
        grid_resolution=grid_resolution,
        kind=kind,
    )

    # Plot each feature
    for i, (feature_idx, ax) in enumerate(zip(feature_indices, axes)):
        feature_name = classifier_result.feature_names[feature_idx]

        # Get values for this cluster
        if len(pd_results['average']) > 0:
            values = pd_results['average'][i][cluster_id]
            grid = pd_results['grid_values'][i]

            ax.plot(grid, values, linewidth=2, color='steelblue')
            ax.set_xlabel(feature_name)
            ax.set_ylabel(f'Cluster {cluster_id} Probability')
            ax.set_title(f'Partial Dependence: {feature_name}')
            ax.grid(alpha=0.3)

    plt.suptitle(f'Partial Dependence Plots - Cluster {cluster_id}', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_cluster_prototypes(
    shap_result: ShapResult,
    prototype_info: Dict[str, np.ndarray],
    cluster_id: int,
    n_features: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    show: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualize cluster prototype samples

    Args:
        shap_result: ShapResult
        prototype_info: Output from identify_cluster_prototypes()
        cluster_id: Cluster ID
        n_features: Number of top features to show
        figsize: Figure size
        show: Show plot
        save_path: Path to save figure

    Returns:
        Figure
    """
    n_prototypes = len(prototype_info['indices'])

    if n_prototypes == 0:
        warnings.warn("No prototypes found")
        return plt.figure()

    fig, axes = plt.subplots(n_prototypes, 1, figsize=figsize)
    if n_prototypes == 1:
        axes = [axes]

    for i, (idx, ax) in enumerate(zip(prototype_info['indices'], axes)):
        # Get SHAP values for this prototype
        shap_vals = prototype_info['shap_values'][i]
        data_vals = prototype_info['data'][i]

        # Get top features by absolute SHAP value
        top_indices = np.argsort(np.abs(shap_vals))[-n_features:][::-1]

        feature_names = [prototype_info['feature_names'][j] for j in top_indices]
        shap_values = [shap_vals[j] for j in top_indices]
        colors = ['green' if s > 0 else 'red' for s in shap_values]

        y_pos = np.arange(len(feature_names))

        ax.barh(y_pos, shap_values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'Prototype {i+1} (Sample {idx})')
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle(f'Cluster {cluster_id} Prototypes', y=1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_cluster_comparison(
    comparison_df: pd.DataFrame,
    cluster_a: int,
    cluster_b: int,
    n_features: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    show: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualize discriminative features between clusters

    Args:
        comparison_df: Output from compare_clusters()
        cluster_a: First cluster
        cluster_b: Second cluster
        n_features: Number of features to show
        figsize: Figure size
        show: Show plot
        save_path: Path to save figure

    Returns:
        Figure
    """
    # Take top N features
    plot_df = comparison_df.head(n_features).copy()

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(plot_df))

    # Plot bars for both clusters
    width = 0.35
    ax.barh(y_pos - width/2, plot_df[f'mean_shap_cluster_{cluster_a}'],
            width, label=f'Cluster {cluster_a}', alpha=0.7, color='steelblue')
    ax.barh(y_pos + width/2, plot_df[f'mean_shap_cluster_{cluster_b}'],
            width, label=f'Cluster {cluster_b}', alpha=0.7, color='coral')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Mean SHAP Value')
    ax.set_title(f'Discriminative Features: Cluster {cluster_a} vs {cluster_b}')
    ax.legend()
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_explainability_dashboard(
    classifier_result: ClusterClassifierResult,
    shap_result: ShapResult,
    X: np.ndarray,
    output_dir: Path,
    n_features: int = 20,
    n_prototypes: int = 3,
) -> None:
    """
    Create comprehensive explainability dashboard

    Generates all plots and saves to directory.

    Args:
        classifier_result: ClusterClassifierResult
        shap_result: ShapResult
        X: Feature matrix
        output_dir: Output directory
        n_features: Number of top features
        n_prototypes: Number of prototypes per cluster
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating explainability dashboard...")

    # 1. Feature importance
    print("  - Feature importance plot")
    plot_feature_importance(
        classifier_result,
        n_features=n_features,
        show=False,
        save_path=output_dir / 'feature_importance.png'
    )

    # 2. SHAP importance comparison
    print("  - SHAP importance comparison")
    plot_shap_importance_comparison(
        shap_result,
        n_features=n_features,
        show=False,
        save_path=output_dir / 'shap_importance_comparison.png'
    )

    # 3. Per-cluster plots
    unique_clusters = np.unique(shap_result.cluster_labels)

    for cluster_id in unique_clusters:
        print(f"  - Cluster {cluster_id} plots")

        cluster_dir = output_dir / f'cluster_{cluster_id}'
        cluster_dir.mkdir(exist_ok=True)

        # SHAP summary
        plot_shap_summary(
            shap_result,
            cluster_id=int(cluster_id),
            max_display=n_features,
            show=False,
            save_path=cluster_dir / 'shap_summary.png'
        )

        # SHAP beeswarm
        plot_shap_beeswarm(
            shap_result,
            cluster_id=int(cluster_id),
            max_display=n_features,
            show=False,
            save_path=cluster_dir / 'shap_beeswarm.png'
        )

    print(f"Dashboard saved to {output_dir}")