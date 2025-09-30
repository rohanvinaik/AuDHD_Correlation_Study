"""Cluster comparison tools

Provides statistical comparisons and visualizations for cluster differences.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats


@dataclass
class ClusterComparisonResult:
    """Result of cluster comparison"""
    feature_name: str
    cluster_stats: Dict[int, Dict[str, float]]  # cluster_id -> {mean, std, median}
    pvalue: float
    effect_size: float
    test_statistic: float
    test_method: str


def create_cluster_comparison(
    data: pd.DataFrame,
    labels: np.ndarray,
    features: Optional[List[str]] = None,
    test_method: str = 'kruskal',
) -> List[ClusterComparisonResult]:
    """
    Compare features across clusters

    Args:
        data: Feature data (n_samples, n_features)
        labels: Cluster labels
        features: Feature names (if None, use all columns)
        test_method: Statistical test ('kruskal', 'anova', 'mannwhitneyu')

    Returns:
        List of ClusterComparisonResult
    """
    if features is None:
        features = data.columns.tolist()

    results = []

    for feature in features:
        feature_data = data[feature].values

        # Get stats per cluster
        cluster_stats = {}
        unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)

        for cluster_id in unique_labels:
            cluster_data = feature_data[labels == cluster_id]
            cluster_stats[int(cluster_id)] = {
                'mean': float(np.mean(cluster_data)),
                'std': float(np.std(cluster_data)),
                'median': float(np.median(cluster_data)),
                'q25': float(np.percentile(cluster_data, 25)),
                'q75': float(np.percentile(cluster_data, 75)),
                'n': int(len(cluster_data)),
            }

        # Statistical test
        cluster_groups = [
            feature_data[labels == cluster_id]
            for cluster_id in unique_labels
        ]

        if test_method == 'kruskal':
            stat, pval = stats.kruskal(*cluster_groups)
            test_name = 'Kruskal-Wallis'
        elif test_method == 'anova':
            stat, pval = stats.f_oneway(*cluster_groups)
            test_name = 'One-way ANOVA'
        elif test_method == 'mannwhitneyu' and len(unique_labels) == 2:
            stat, pval = stats.mannwhitneyu(cluster_groups[0], cluster_groups[1])
            test_name = 'Mann-Whitney U'
        else:
            stat, pval = stats.kruskal(*cluster_groups)
            test_name = 'Kruskal-Wallis'

        # Effect size (eta-squared for ANOVA-like tests)
        effect_size = _calculate_effect_size(cluster_groups, test_method)

        result = ClusterComparisonResult(
            feature_name=feature,
            cluster_stats=cluster_stats,
            pvalue=float(pval),
            effect_size=float(effect_size),
            test_statistic=float(stat),
            test_method=test_name,
        )

        results.append(result)

    # Sort by p-value
    results.sort(key=lambda x: x.pvalue)

    return results


def _calculate_effect_size(groups: List[np.ndarray], test_method: str) -> float:
    """Calculate effect size (eta-squared or similar)"""
    # Eta-squared: proportion of variance explained by group membership

    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    # Between-group variance
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)

    # Total variance
    ss_total = np.sum((all_data - grand_mean)**2)

    if ss_total == 0:
        return 0.0

    eta_squared = ss_between / ss_total

    return eta_squared


def create_violin_comparison(
    data: pd.DataFrame,
    labels: np.ndarray,
    features: List[str],
    n_cols: int = 2,
    width: int = 1200,
    height: int = 800,
) -> go.Figure:
    """
    Create violin plots comparing features across clusters

    Args:
        data: Feature data
        labels: Cluster labels
        features: Features to plot
        n_cols: Number of columns in subplot grid
        width: Figure width
        height: Figure height

    Returns:
        Plotly Figure with violin plots
    """
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=features,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    unique_labels = np.unique(labels[labels >= 0])
    colors = px.colors.qualitative.Set2

    for idx, feature in enumerate(features):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        for i, cluster_id in enumerate(unique_labels):
            cluster_data = data[feature].values[labels == cluster_id]

            fig.add_trace(
                go.Violin(
                    y=cluster_data,
                    name=f'Cluster {cluster_id}',
                    marker_color=colors[i % len(colors)],
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=(idx == 0),  # Only show legend for first plot
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        width=width,
        height=height * n_rows // 2,
        template='plotly_white',
        font=dict(size=10),
    )

    # Update y-axes
    for idx in range(n_features):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        fig.update_yaxes(title_text="Value", row=row, col=col)

    return fig


def create_box_comparison(
    data: pd.DataFrame,
    labels: np.ndarray,
    features: List[str],
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """
    Create box plots for feature comparison

    Args:
        data: Feature data
        labels: Cluster labels
        features: Features to plot
        width: Figure width
        height: Figure height

    Returns:
        Box plot figure
    """
    # Prepare data
    df_list = []

    for feature in features:
        for cluster_id in np.unique(labels[labels >= 0]):
            mask = labels == cluster_id
            values = data[feature].values[mask]

            df_list.append(pd.DataFrame({
                'Feature': feature,
                'Cluster': f'Cluster {cluster_id}',
                'Value': values,
            }))

    plot_df = pd.concat(df_list, ignore_index=True)

    # Create box plot
    fig = px.box(
        plot_df,
        x='Feature',
        y='Value',
        color='Cluster',
        title='Feature Comparison Across Clusters',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_layout(
        width=width,
        height=height,
        template='plotly_white',
        xaxis_tickangle=-45,
    )

    return fig


def create_sankey_diagram(
    labels_t1: np.ndarray,
    labels_t2: np.ndarray,
    patient_ids: Optional[List[str]] = None,
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """
    Create Sankey diagram showing cluster transitions

    Args:
        labels_t1: Cluster labels at timepoint 1
        labels_t2: Cluster labels at timepoint 2
        patient_ids: Patient identifiers
        width: Figure width
        height: Figure height

    Returns:
        Sankey diagram figure
    """
    if len(labels_t1) != len(labels_t2):
        raise ValueError("Label arrays must have same length")

    # Get unique clusters
    unique_t1 = np.unique(labels_t1[labels_t1 >= 0])
    unique_t2 = np.unique(labels_t2[labels_t2 >= 0])

    # Create node labels
    source_labels = [f'T1: Cluster {c}' for c in unique_t1]
    target_labels = [f'T2: Cluster {c}' for c in unique_t2]
    all_labels = source_labels + target_labels

    # Create mapping
    source_map = {c: i for i, c in enumerate(unique_t1)}
    target_map = {c: i + len(unique_t1) for i, c in enumerate(unique_t2)}

    # Count transitions
    transitions = {}
    for l1, l2 in zip(labels_t1, labels_t2):
        if l1 >= 0 and l2 >= 0:
            key = (l1, l2)
            transitions[key] = transitions.get(key, 0) + 1

    # Create source, target, value lists
    sources = []
    targets = []
    values = []

    for (c1, c2), count in transitions.items():
        sources.append(source_map[c1])
        targets.append(target_map[c2])
        values.append(count)

    # Create Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=px.colors.qualitative.Set2[:len(all_labels)],
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        )
    )])

    fig.update_layout(
        title="Cluster Transitions Over Time",
        width=width,
        height=height,
        font=dict(size=12),
        template='plotly_white',
    )

    return fig


def create_cluster_size_comparison(
    labels_list: List[np.ndarray],
    timepoints: List[str],
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """
    Compare cluster sizes over time

    Args:
        labels_list: List of label arrays at different timepoints
        timepoints: Names of timepoints
        width: Figure width
        height: Figure height

    Returns:
        Stacked bar chart
    """
    # Count cluster sizes at each timepoint
    data = []

    all_clusters = set()
    for labels in labels_list:
        all_clusters.update(np.unique(labels[labels >= 0]))

    all_clusters = sorted(all_clusters)

    for cluster_id in all_clusters:
        counts = []
        for labels in labels_list:
            count = np.sum(labels == cluster_id)
            counts.append(count)

        data.append(
            go.Bar(
                name=f'Cluster {cluster_id}',
                x=timepoints,
                y=counts,
            )
        )

    fig = go.Figure(data=data)

    fig.update_layout(
        title="Cluster Size Changes Over Time",
        xaxis_title="Timepoint",
        yaxis_title="Number of Patients",
        barmode='stack',
        width=width,
        height=height,
        template='plotly_white',
    )

    return fig


def create_feature_importance_comparison(
    comparison_results: List[ClusterComparisonResult],
    n_top: int = 20,
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """
    Visualize most discriminative features

    Args:
        comparison_results: Results from create_cluster_comparison
        n_top: Number of top features to show
        width: Figure width
        height: Figure height

    Returns:
        Bar chart of effect sizes
    """
    # Get top features by effect size
    sorted_results = sorted(
        comparison_results,
        key=lambda x: x.effect_size,
        reverse=True
    )[:n_top]

    features = [r.feature_name for r in sorted_results]
    effect_sizes = [r.effect_size for r in sorted_results]
    pvalues = [r.pvalue for r in sorted_results]

    # Create color based on p-value
    colors = ['red' if p < 0.01 else 'orange' if p < 0.05 else 'gray' for p in pvalues]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=features,
            x=effect_sizes,
            orientation='h',
            marker=dict(color=colors),
            text=[f'p={p:.2e}' for p in pvalues],
            textposition='auto',
        )
    )

    fig.update_layout(
        title=f"Top {n_top} Discriminative Features",
        xaxis_title="Effect Size (η²)",
        yaxis_title="Feature",
        width=width,
        height=height,
        template='plotly_white',
        showlegend=False,
    )

    # Add legend
    fig.add_annotation(
        text="Color: p < 0.01 (red), p < 0.05 (orange), p ≥ 0.05 (gray)",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.15,
        showarrow=False,
        font=dict(size=10),
    )

    return fig


def create_cluster_profile_heatmap(
    comparison_results: List[ClusterComparisonResult],
    metric: str = 'mean',
    n_features: int = 30,
    width: int = 1000,
    height: int = 800,
) -> go.Figure:
    """
    Create heatmap of cluster profiles

    Args:
        comparison_results: Comparison results
        metric: Metric to show ('mean', 'median')
        n_features: Number of features to include
        width: Figure width
        height: Figure height

    Returns:
        Heatmap figure
    """
    # Get top features
    sorted_results = sorted(
        comparison_results,
        key=lambda x: x.effect_size,
        reverse=True
    )[:n_features]

    # Get all clusters
    all_clusters = set()
    for result in sorted_results:
        all_clusters.update(result.cluster_stats.keys())
    all_clusters = sorted(all_clusters)

    # Build matrix
    matrix = []
    features = []

    for result in sorted_results:
        row = []
        for cluster_id in all_clusters:
            if cluster_id in result.cluster_stats:
                value = result.cluster_stats[cluster_id][metric]
            else:
                value = 0.0
            row.append(value)
        matrix.append(row)
        features.append(result.feature_name)

    matrix = np.array(matrix)

    # Z-score normalize by row
    matrix_normalized = (matrix - matrix.mean(axis=1, keepdims=True)) / (matrix.std(axis=1, keepdims=True) + 1e-10)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_normalized,
        x=[f'Cluster {c}' for c in all_clusters],
        y=features,
        colorscale='RdBu_r',
        zmid=0,
        text=matrix,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Z-score"),
    ))

    fig.update_layout(
        title=f"Cluster Profiles ({metric.capitalize()})",
        xaxis_title="Cluster",
        yaxis_title="Feature",
        width=width,
        height=height,
        template='plotly_white',
    )

    return fig


def create_pairwise_comparison_matrix(
    data: pd.DataFrame,
    labels: np.ndarray,
    feature: str,
    width: int = 800,
    height: int = 800,
) -> go.Figure:
    """
    Create pairwise comparison matrix for a feature

    Args:
        data: Feature data
        labels: Cluster labels
        feature: Feature to compare
        width: Figure width
        height: Figure height

    Returns:
        Heatmap of p-values
    """
    unique_labels = sorted(np.unique(labels[labels >= 0]))
    n_clusters = len(unique_labels)

    # Initialize matrix
    pval_matrix = np.ones((n_clusters, n_clusters))

    # Pairwise tests
    for i, cluster_i in enumerate(unique_labels):
        for j, cluster_j in enumerate(unique_labels):
            if i < j:
                data_i = data[feature].values[labels == cluster_i]
                data_j = data[feature].values[labels == cluster_j]

                _, pval = stats.mannwhitneyu(data_i, data_j)
                pval_matrix[i, j] = pval
                pval_matrix[j, i] = pval

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=-np.log10(pval_matrix + 1e-300),  # -log10(p)
        x=[f'C{c}' for c in unique_labels],
        y=[f'C{c}' for c in unique_labels],
        colorscale='Reds',
        text=pval_matrix,
        texttemplate='p=%{text:.2e}',
        textfont={"size": 10},
        colorbar=dict(title="-log10(p)"),
    ))

    fig.update_layout(
        title=f"Pairwise Comparisons: {feature}",
        xaxis_title="Cluster",
        yaxis_title="Cluster",
        width=width,
        height=height,
        template='plotly_white',
    )

    return fig