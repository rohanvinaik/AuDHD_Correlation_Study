"""Interactive embedding plots (t-SNE, UMAP) with multiple overlays

Creates publication-quality interactive scatter plots with filtering and drill-down.
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


@dataclass
class EmbeddingPlotConfig:
    """Configuration for embedding plots"""
    title: str = "Embedding Plot"
    width: int = 1000
    height: int = 800
    marker_size: int = 8
    opacity: float = 0.7
    color_scale: str = "Viridis"
    show_legend: bool = True
    font_size: int = 12


def create_embedding_plot(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    overlay_data: Optional[Union[np.ndarray, pd.Series]] = None,
    overlay_name: str = "Value",
    patient_ids: Optional[List[str]] = None,
    config: Optional[EmbeddingPlotConfig] = None,
    categorical: bool = False,
) -> go.Figure:
    """
    Create interactive embedding plot (t-SNE/UMAP)

    Args:
        embedding: 2D embedding coordinates (n_samples, 2)
        labels: Cluster labels (for coloring)
        overlay_data: Additional data to overlay (continuous or categorical)
        overlay_name: Name for overlay data
        patient_ids: Patient identifiers for hover info
        config: Plot configuration
        categorical: Whether overlay_data is categorical

    Returns:
        Plotly Figure object
    """
    if config is None:
        config = EmbeddingPlotConfig()

    n_samples = len(embedding)

    # Create DataFrame
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
    })

    if patient_ids is not None:
        df['patient_id'] = patient_ids
    else:
        df['patient_id'] = [f'P{i}' for i in range(n_samples)]

    # Determine color variable
    if overlay_data is not None:
        df[overlay_name] = overlay_data
        color_var = overlay_name
    elif labels is not None:
        df['Cluster'] = labels.astype(str)
        color_var = 'Cluster'
        categorical = True
    else:
        df['Cluster'] = '0'
        color_var = 'Cluster'
        categorical = True

    # Create plot
    if categorical:
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color=color_var,
            hover_data=['patient_id'],
            title=config.title,
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    else:
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color=color_var,
            hover_data=['patient_id'],
            title=config.title,
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
            color_continuous_scale=config.color_scale,
        )

    # Update layout
    fig.update_traces(
        marker=dict(
            size=config.marker_size,
            opacity=config.opacity,
            line=dict(width=0.5, color='white'),
        ),
    )

    fig.update_layout(
        width=config.width,
        height=config.height,
        font=dict(size=config.font_size),
        hovermode='closest',
        showlegend=config.show_legend,
        template='plotly_white',
    )

    # Add export buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"marker.size": 12}],
                        label="Larger",
                        method="restyle"
                    ),
                    dict(
                        args=[{"marker.size": 6}],
                        label="Smaller",
                        method="restyle"
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )

    return fig


def create_multi_overlay_plot(
    embedding: np.ndarray,
    labels: np.ndarray,
    overlay_dict: Dict[str, np.ndarray],
    patient_ids: Optional[List[str]] = None,
    n_cols: int = 2,
    config: Optional[EmbeddingPlotConfig] = None,
) -> go.Figure:
    """
    Create multiple embedding plots with different overlays

    Args:
        embedding: 2D embedding coordinates
        labels: Cluster labels
        overlay_dict: Dictionary of {name: data} for overlays
        patient_ids: Patient identifiers
        n_cols: Number of columns in subplot grid
        config: Plot configuration

    Returns:
        Plotly Figure with subplots
    """
    if config is None:
        config = EmbeddingPlotConfig()

    n_overlays = len(overlay_dict) + 1  # +1 for cluster labels
    n_rows = (n_overlays + n_cols - 1) // n_cols

    # Create subplots
    subplot_titles = ['Clusters'] + list(overlay_dict.keys())

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    if patient_ids is None:
        patient_ids = [f'P{i}' for i in range(len(embedding))]

    # Add cluster plot
    _add_scatter_trace(
        fig,
        embedding,
        labels.astype(str),
        patient_ids,
        row=1,
        col=1,
        categorical=True,
        config=config,
    )

    # Add overlay plots
    for idx, (name, data) in enumerate(overlay_dict.items(), start=2):
        row = (idx - 1) // n_cols + 1
        col = (idx - 1) % n_cols + 1

        _add_scatter_trace(
            fig,
            embedding,
            data,
            patient_ids,
            row=row,
            col=col,
            categorical=False,
            config=config,
        )

    # Update layout
    fig.update_layout(
        height=config.height * n_rows // 2,
        width=config.width,
        showlegend=False,
        template='plotly_white',
        font=dict(size=config.font_size),
    )

    # Update axes
    for i in range(1, n_overlays + 1):
        row = (i - 1) // n_cols + 1
        col = (i - 1) % n_cols + 1
        fig.update_xaxes(title_text="Dimension 1", row=row, col=col)
        fig.update_yaxes(title_text="Dimension 2", row=row, col=col)

    return fig


def _add_scatter_trace(
    fig: go.Figure,
    embedding: np.ndarray,
    values: np.ndarray,
    patient_ids: List[str],
    row: int,
    col: int,
    categorical: bool,
    config: EmbeddingPlotConfig,
) -> None:
    """Add scatter trace to subplot"""

    if categorical:
        # Get unique categories
        unique_vals = np.unique(values)
        colors = px.colors.qualitative.Set2

        for i, val in enumerate(unique_vals):
            mask = values == val
            fig.add_trace(
                go.Scatter(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    mode='markers',
                    marker=dict(
                        size=config.marker_size,
                        color=colors[i % len(colors)],
                        opacity=config.opacity,
                        line=dict(width=0.5, color='white'),
                    ),
                    name=str(val),
                    text=[patient_ids[j] for j in np.where(mask)[0]],
                    hovertemplate='Patient: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}',
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
    else:
        # Continuous values
        fig.add_trace(
            go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode='markers',
                marker=dict(
                    size=config.marker_size,
                    color=values,
                    colorscale=config.color_scale,
                    opacity=config.opacity,
                    line=dict(width=0.5, color='white'),
                    colorbar=dict(
                        x=1.0 + 0.05 * col,
                        len=0.3,
                    ),
                ),
                text=patient_ids,
                hovertemplate='Patient: %{text}<br>Value: %{marker.color:.2f}<br>X: %{x:.2f}<br>Y: %{y:.2f}',
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def create_3d_embedding_plot(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    overlay_data: Optional[np.ndarray] = None,
    overlay_name: str = "Value",
    patient_ids: Optional[List[str]] = None,
    config: Optional[EmbeddingPlotConfig] = None,
) -> go.Figure:
    """
    Create 3D embedding plot

    Args:
        embedding: 3D embedding coordinates (n_samples, 3)
        labels: Cluster labels
        overlay_data: Additional data to overlay
        overlay_name: Name for overlay
        patient_ids: Patient identifiers
        config: Plot configuration

    Returns:
        Plotly 3D scatter figure
    """
    if config is None:
        config = EmbeddingPlotConfig()

    if embedding.shape[1] != 3:
        raise ValueError("Embedding must be 3D (n_samples, 3)")

    n_samples = len(embedding)

    if patient_ids is None:
        patient_ids = [f'P{i}' for i in range(n_samples)]

    # Create DataFrame
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'z': embedding[:, 2],
        'patient_id': patient_ids,
    })

    # Determine color
    if overlay_data is not None:
        df[overlay_name] = overlay_data
        color_var = overlay_name
        categorical = False
    elif labels is not None:
        df['Cluster'] = labels.astype(str)
        color_var = 'Cluster'
        categorical = True
    else:
        df['Cluster'] = '0'
        color_var = 'Cluster'
        categorical = True

    # Create 3D plot
    if categorical:
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color=color_var,
            hover_data=['patient_id'],
            title=config.title,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    else:
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color=color_var,
            hover_data=['patient_id'],
            title=config.title,
            color_continuous_scale=config.color_scale,
        )

    fig.update_traces(
        marker=dict(
            size=config.marker_size,
            opacity=config.opacity,
        ),
    )

    fig.update_layout(
        width=config.width,
        height=config.height,
        font=dict(size=config.font_size),
        template='plotly_white',
    )

    return fig


def create_density_overlay(
    embedding: np.ndarray,
    labels: np.ndarray,
    config: Optional[EmbeddingPlotConfig] = None,
) -> go.Figure:
    """
    Create embedding plot with density overlay

    Args:
        embedding: 2D embedding
        labels: Cluster labels
        config: Plot configuration

    Returns:
        Figure with contour density overlay
    """
    if config is None:
        config = EmbeddingPlotConfig()

    fig = go.Figure()

    # Get unique clusters
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set2

    # Add contour for each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_embedding = embedding[mask]

        # Add scatter
        fig.add_trace(
            go.Scatter(
                x=cluster_embedding[:, 0],
                y=cluster_embedding[:, 1],
                mode='markers',
                marker=dict(
                    size=config.marker_size,
                    color=colors[i % len(colors)],
                    opacity=config.opacity,
                ),
                name=f'Cluster {label}',
            )
        )

        # Add density contour
        fig.add_trace(
            go.Histogram2dContour(
                x=cluster_embedding[:, 0],
                y=cluster_embedding[:, 1],
                colorscale='Blues',
                showscale=False,
                opacity=0.3,
                line=dict(width=2, color=colors[i % len(colors)]),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=config.title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=config.width,
        height=config.height,
        template='plotly_white',
        font=dict(size=config.font_size),
    )

    return fig


def create_animated_embedding(
    embeddings_over_time: List[np.ndarray],
    labels_over_time: List[np.ndarray],
    timepoints: List[str],
    patient_ids: Optional[List[str]] = None,
    config: Optional[EmbeddingPlotConfig] = None,
) -> go.Figure:
    """
    Create animated embedding plot showing changes over time

    Args:
        embeddings_over_time: List of embeddings at different timepoints
        labels_over_time: List of labels at different timepoints
        timepoints: Names of timepoints
        patient_ids: Patient identifiers
        config: Plot configuration

    Returns:
        Animated figure
    """
    if config is None:
        config = EmbeddingPlotConfig()

    if len(embeddings_over_time) != len(labels_over_time):
        raise ValueError("Must have same number of embeddings and labels")

    n_samples = len(embeddings_over_time[0])

    if patient_ids is None:
        patient_ids = [f'P{i}' for i in range(n_samples)]

    # Create frames
    frames = []

    for embedding, labels, timepoint in zip(
        embeddings_over_time, labels_over_time, timepoints
    ):
        df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'Cluster': labels.astype(str),
            'patient_id': patient_ids,
        })

        # Create traces for this timepoint
        traces = []
        for cluster in np.unique(labels):
            cluster_df = df[df['Cluster'] == str(cluster)]
            traces.append(
                go.Scatter(
                    x=cluster_df['x'],
                    y=cluster_df['y'],
                    mode='markers',
                    marker=dict(size=config.marker_size, opacity=config.opacity),
                    name=f'Cluster {cluster}',
                    text=cluster_df['patient_id'],
                )
            )

        frames.append(go.Frame(data=traces, name=timepoint))

    # Create initial figure
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
    )

    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 500, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
                ],
                x=0.1,
                y=0,
                xanchor="right",
                yanchor="top",
            )
        ],
        sliders=[{
            "active": 0,
            "steps": [
                {"args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate"}],
                 "label": f.name,
                 "method": "animate"}
                for f in frames
            ],
            "x": 0.1,
            "len": 0.9,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top",
        }],
        title=config.title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=config.width,
        height=config.height,
        template='plotly_white',
        font=dict(size=config.font_size),
    )

    return fig