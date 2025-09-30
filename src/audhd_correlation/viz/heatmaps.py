"""Interactive biomarker heatmaps

Publication-quality heatmaps with clustering and annotations.
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist


@dataclass
class HeatmapConfig:
    """Configuration for heatmap plots"""
    title: str = "Biomarker Heatmap"
    width: int = 1200
    height: int = 800
    colorscale: str = "RdBu_r"
    show_dendrograms: bool = True
    cluster_rows: bool = True
    cluster_cols: bool = True
    font_size: int = 10
    cell_font_size: int = 8


def create_biomarker_heatmap(
    data: pd.DataFrame,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    patient_ids: Optional[List[str]] = None,
    config: Optional[HeatmapConfig] = None,
    normalize: str = 'zscore',
) -> go.Figure:
    """
    Create interactive biomarker heatmap with clustering

    Args:
        data: Biomarker data (n_samples, n_features)
        labels: Cluster labels for samples
        feature_names: Feature names (if None, use column names)
        patient_ids: Patient identifiers
        config: Heatmap configuration
        normalize: Normalization method ('zscore', 'minmax', 'none')

    Returns:
        Plotly Figure with heatmap
    """
    if config is None:
        config = HeatmapConfig()

    if feature_names is None:
        feature_names = data.columns.tolist()

    if patient_ids is None:
        patient_ids = [f'P{i}' for i in range(len(data))]

    # Normalize data
    matrix = data.values.copy()

    if normalize == 'zscore':
        matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-10)
    elif normalize == 'minmax':
        matrix = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0) + 1e-10)
    # else: no normalization

    # Sort by cluster labels
    sort_idx = np.argsort(labels)
    matrix_sorted = matrix[sort_idx]
    labels_sorted = labels[sort_idx]
    patient_ids_sorted = [patient_ids[i] for i in sort_idx]

    # Cluster features if requested
    if config.cluster_cols and matrix_sorted.shape[1] > 2:
        try:
            col_linkage = hierarchy.linkage(matrix_sorted.T, method='average')
            col_dendro = hierarchy.dendrogram(col_linkage, no_plot=True)
            col_order = col_dendro['leaves']
            matrix_sorted = matrix_sorted[:, col_order]
            feature_names = [feature_names[i] for i in col_order]
        except Exception as e:
            warnings.warn(f"Could not cluster columns: {e}")

    # Create figure
    if config.show_dendrograms and config.cluster_cols and matrix_sorted.shape[1] > 2:
        # Create with dendrograms
        fig = _create_heatmap_with_dendrograms(
            matrix_sorted,
            labels_sorted,
            feature_names,
            patient_ids_sorted,
            config,
        )
    else:
        # Simple heatmap
        fig = _create_simple_heatmap(
            matrix_sorted,
            labels_sorted,
            feature_names,
            patient_ids_sorted,
            config,
        )

    return fig


def _create_simple_heatmap(
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    patient_ids: List[str],
    config: HeatmapConfig,
) -> go.Figure:
    """Create simple heatmap without dendrograms"""

    # Create hover text
    hover_text = []
    for i, patient_id in enumerate(patient_ids):
        row_text = []
        for j, feature in enumerate(feature_names):
            row_text.append(
                f"Patient: {patient_id}<br>"
                f"Feature: {feature}<br>"
                f"Value: {matrix[i, j]:.2f}<br>"
                f"Cluster: {labels[i]}"
            )
        hover_text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=feature_names,
        y=patient_ids,
        colorscale=config.colorscale,
        zmid=0 if 'RdBu' in config.colorscale else None,
        hovertext=hover_text,
        hoverinfo='text',
        colorbar=dict(title="Value"),
    ))

    # Add cluster annotations
    _add_cluster_sidebar(fig, labels, patient_ids)

    fig.update_layout(
        title=config.title,
        xaxis_title="Features",
        yaxis_title="Patients",
        width=config.width,
        height=config.height,
        template='plotly_white',
        font=dict(size=config.font_size),
        xaxis=dict(tickangle=-45, tickfont=dict(size=config.cell_font_size)),
        yaxis=dict(tickfont=dict(size=config.cell_font_size)),
    )

    return fig


def _create_heatmap_with_dendrograms(
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    patient_ids: List[str],
    config: HeatmapConfig,
) -> go.Figure:
    """Create heatmap with dendrograms"""

    # Compute dendrograms
    col_linkage = hierarchy.linkage(matrix.T, method='average')
    row_linkage = hierarchy.linkage(matrix, method='average')

    col_dendro = hierarchy.dendrogram(col_linkage, no_plot=True)
    row_dendro = hierarchy.dendrogram(row_linkage, no_plot=True)

    # Reorder
    col_order = col_dendro['leaves']
    row_order = row_dendro['leaves']

    matrix_ordered = matrix[row_order][:, col_order]
    feature_names_ordered = [feature_names[i] for i in col_order]
    patient_ids_ordered = [patient_ids[i] for i in row_order]
    labels_ordered = labels[row_order]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.15, 0.85],
        column_widths=[0.9, 0.1],
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
        specs=[
            [{"type": "scatter"}, None],
            [{"type": "heatmap"}, {"type": "heatmap"}],
        ],
    )

    # Add column dendrogram
    _add_dendrogram_trace(fig, col_dendro, 'columns', row=1, col=1)

    # Add main heatmap
    fig.add_trace(
        go.Heatmap(
            z=matrix_ordered,
            x=feature_names_ordered,
            y=patient_ids_ordered,
            colorscale=config.colorscale,
            zmid=0,
            showscale=True,
            colorbar=dict(
                title="Value",
                x=1.15,
            ),
        ),
        row=2,
        col=1,
    )

    # Add cluster sidebar
    cluster_colors = _labels_to_colors(labels_ordered)
    fig.add_trace(
        go.Heatmap(
            z=cluster_colors.reshape(-1, 1),
            colorscale='Viridis',
            showscale=False,
            hoverinfo='skip',
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=config.title,
        width=config.width,
        height=config.height,
        template='plotly_white',
        showlegend=False,
    )

    # Update axes
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)

    fig.update_xaxes(tickangle=-45, tickfont=dict(size=config.cell_font_size), row=2, col=1)
    fig.update_yaxes(tickfont=dict(size=config.cell_font_size), row=2, col=1)

    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2)

    return fig


def _add_dendrogram_trace(
    fig: go.Figure,
    dendro: Dict,
    orientation: str,
    row: int,
    col: int,
) -> None:
    """Add dendrogram trace to figure"""

    icoord = np.array(dendro['icoord'])
    dcoord = np.array(dendro['dcoord'])

    for i in range(len(icoord)):
        fig.add_trace(
            go.Scatter(
                x=icoord[i],
                y=dcoord[i],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip',
            ),
            row=row,
            col=col,
        )


def _add_cluster_sidebar(
    fig: go.Figure,
    labels: np.ndarray,
    patient_ids: List[str],
) -> None:
    """Add colored sidebar showing cluster assignments"""

    # Get cluster boundaries
    unique_labels = []
    boundaries = []
    current_label = labels[0]
    start_idx = 0

    for i, label in enumerate(labels):
        if label != current_label:
            unique_labels.append(current_label)
            boundaries.append((start_idx, i))
            current_label = label
            start_idx = i

    # Add last cluster
    unique_labels.append(current_label)
    boundaries.append((start_idx, len(labels)))

    # Add rectangles
    colors = px.colors.qualitative.Set2

    for i, (label, (start, end)) in enumerate(zip(unique_labels, boundaries)):
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=1.01,
            x1=1.03,
            y0=start - 0.5,
            y1=end - 0.5,
            fillcolor=colors[int(label) % len(colors)],
            line=dict(width=0),
        )


def _labels_to_colors(labels: np.ndarray) -> np.ndarray:
    """Convert cluster labels to color indices"""
    unique_labels = np.unique(labels)
    label_to_color = {label: i for i, label in enumerate(unique_labels)}
    colors = np.array([label_to_color[l] for l in labels])
    return colors


def create_interactive_heatmap(
    data: pd.DataFrame,
    row_metadata: Optional[pd.DataFrame] = None,
    col_metadata: Optional[pd.DataFrame] = None,
    config: Optional[HeatmapConfig] = None,
) -> go.Figure:
    """
    Create interactive heatmap with metadata annotations

    Args:
        data: Data matrix
        row_metadata: Metadata for rows (samples)
        col_metadata: Metadata for columns (features)
        config: Heatmap configuration

    Returns:
        Annotated heatmap figure
    """
    if config is None:
        config = HeatmapConfig()

    matrix = data.values

    # Normalize
    matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-10)

    # Create hover text with metadata
    hover_text = []
    for i in range(matrix.shape[0]):
        row_text = []
        for j in range(matrix.shape[1]):
            text = f"Row: {data.index[i]}<br>Col: {data.columns[j]}<br>Value: {matrix[i, j]:.2f}"

            if row_metadata is not None and data.index[i] in row_metadata.index:
                for col in row_metadata.columns:
                    text += f"<br>{col}: {row_metadata.loc[data.index[i], col]}"

            if col_metadata is not None and data.columns[j] in col_metadata.index:
                for col in col_metadata.columns:
                    text += f"<br>{col}: {col_metadata.loc[data.columns[j], col]}"

            row_text.append(text)
        hover_text.append(row_text)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=data.columns,
        y=data.index,
        colorscale=config.colorscale,
        zmid=0,
        hovertext=hover_text,
        hoverinfo='text',
    ))

    fig.update_layout(
        title=config.title,
        width=config.width,
        height=config.height,
        template='plotly_white',
        xaxis=dict(tickangle=-45),
    )

    return fig


def create_correlation_heatmap(
    data: pd.DataFrame,
    method: str = 'pearson',
    cluster: bool = True,
    config: Optional[HeatmapConfig] = None,
) -> go.Figure:
    """
    Create correlation heatmap

    Args:
        data: Feature data
        method: Correlation method ('pearson', 'spearman')
        cluster: Whether to cluster features
        config: Heatmap configuration

    Returns:
        Correlation heatmap
    """
    if config is None:
        config = HeatmapConfig(title=f"{method.capitalize()} Correlation")

    # Calculate correlation
    if method == 'pearson':
        corr_matrix = data.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = data.corr(method='spearman')
    else:
        raise ValueError(f"Unknown method: {method}")

    # Cluster if requested
    if cluster and len(corr_matrix) > 2:
        try:
            linkage = hierarchy.linkage(pdist(corr_matrix), method='average')
            dendro = hierarchy.dendrogram(linkage, no_plot=True)
            order = dendro['leaves']
            corr_matrix = corr_matrix.iloc[order, order]
        except Exception as e:
            warnings.warn(f"Could not cluster: {e}")

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation"),
    ))

    fig.update_layout(
        title=config.title,
        width=config.width,
        height=config.height,
        template='plotly_white',
        xaxis=dict(tickangle=-45),
    )

    return fig


def create_clustermap(
    data: pd.DataFrame,
    labels: np.ndarray,
    top_features: int = 50,
    config: Optional[HeatmapConfig] = None,
) -> go.Figure:
    """
    Create clustermap with hierarchical clustering

    Args:
        data: Feature data
        labels: Cluster labels
        top_features: Number of top variable features to include
        config: Heatmap configuration

    Returns:
        Clustermap figure
    """
    if config is None:
        config = HeatmapConfig(title="Clustermap")

    # Select top variable features
    variances = data.var(axis=0)
    top_features_idx = variances.nlargest(top_features).index
    data_subset = data[top_features_idx]

    # Normalize
    matrix = data_subset.values
    matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-10)

    # Sort by labels
    sort_idx = np.argsort(labels)
    matrix_sorted = matrix[sort_idx]
    labels_sorted = labels[sort_idx]

    # Cluster columns
    try:
        col_linkage = hierarchy.linkage(matrix_sorted.T, method='average')
        col_dendro = hierarchy.dendrogram(col_linkage, no_plot=True)
        col_order = col_dendro['leaves']
        matrix_sorted = matrix_sorted[:, col_order]
        feature_names = [top_features_idx[i] for i in col_order]
    except Exception as e:
        warnings.warn(f"Could not cluster: {e}")
        feature_names = top_features_idx.tolist()

    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=matrix_sorted,
        x=feature_names,
        y=[f'S{i}' for i in range(len(matrix_sorted))],
        colorscale=config.colorscale,
        zmid=0,
        colorbar=dict(title="Z-score"),
    ))

    # Add cluster sidebar
    _add_cluster_sidebar(fig, labels_sorted, [f'S{i}' for i in range(len(matrix_sorted))])

    fig.update_layout(
        title=config.title,
        xaxis_title="Features",
        yaxis_title="Samples",
        width=config.width,
        height=config.height,
        template='plotly_white',
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(showticklabels=False),
    )

    return fig


def create_split_violin_heatmap(
    data: pd.DataFrame,
    labels: np.ndarray,
    features: List[str],
    width: int = 1200,
    height: int = 800,
) -> go.Figure:
    """
    Create combined heatmap and violin plot

    Args:
        data: Feature data
        labels: Cluster labels
        features: Features to plot
        width: Figure width
        height: Figure height

    Returns:
        Combined figure
    """
    n_features = len(features)
    unique_labels = sorted(np.unique(labels[labels >= 0]))

    # Create subplots
    fig = make_subplots(
        rows=n_features,
        cols=2,
        column_widths=[0.6, 0.4],
        horizontal_spacing=0.1,
        vertical_spacing=0.05,
        subplot_titles=[f"{f} - Heatmap" if i == 0 else f"{f} - Distribution"
                        for i, f in enumerate(features) for _ in range(2)],
    )

    colors = px.colors.qualitative.Set2

    for idx, feature in enumerate(features):
        row = idx + 1

        # Heatmap
        # Create matrix for this feature across clusters
        feature_matrix = []
        for cluster_id in unique_labels:
            cluster_data = data[feature].values[labels == cluster_id]
            feature_matrix.append(cluster_data)

        # Pad to same length
        max_len = max(len(d) for d in feature_matrix)
        feature_matrix_padded = []
        for cluster_data in feature_matrix:
            padded = np.full(max_len, np.nan)
            padded[:len(cluster_data)] = cluster_data
            feature_matrix_padded.append(padded)

        fig.add_trace(
            go.Heatmap(
                z=feature_matrix_padded,
                y=[f'C{c}' for c in unique_labels],
                colorscale='Viridis',
                showscale=False,
            ),
            row=row,
            col=1,
        )

        # Violin plot
        for i, cluster_id in enumerate(unique_labels):
            cluster_data = data[feature].values[labels == cluster_id]
            fig.add_trace(
                go.Violin(
                    y=cluster_data,
                    name=f'C{cluster_id}',
                    marker_color=colors[i % len(colors)],
                    showlegend=False,
                ),
                row=row,
                col=2,
            )

    fig.update_layout(
        title="Feature Distributions by Cluster",
        width=width,
        height=height,
        template='plotly_white',
        showlegend=False,
    )

    return fig