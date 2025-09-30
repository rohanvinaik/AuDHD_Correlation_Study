"""Patient trajectory visualization

Tracks individual patient journeys through cluster space over time.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


@dataclass
class TrajectoryResult:
    """Result of trajectory analysis"""
    patient_id: str
    timepoints: List[str]
    clusters: List[int]
    coordinates: List[Tuple[float, float]]
    biomarkers: Optional[Dict[str, List[float]]] = None
    stability_score: float = 0.0
    n_transitions: int = 0


def create_trajectory_plot(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    patient_ids: List[str],
    selected_patients: Optional[List[str]] = None,
    width: int = 1000,
    height: int = 800,
) -> go.Figure:
    """
    Create trajectory plot showing patient movements over time

    Args:
        embeddings: Dict of {timepoint: embedding_array}
        labels: Dict of {timepoint: labels_array}
        patient_ids: Patient identifiers (must be consistent across timepoints)
        selected_patients: Subset of patients to highlight (if None, show all)
        width: Figure width
        height: Figure height

    Returns:
        Plotly figure with trajectories
    """
    timepoints = sorted(embeddings.keys())

    if selected_patients is None:
        selected_patients = patient_ids

    fig = go.Figure()

    # Plot all patients as background
    for timepoint in timepoints:
        embedding = embeddings[timepoint]
        fig.add_trace(
            go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode='markers',
                marker=dict(size=6, color='lightgray', opacity=0.3),
                name=f'{timepoint} (all)',
                showlegend=False,
                hoverinfo='skip',
            )
        )

    # Plot trajectories for selected patients
    colors = px.colors.qualitative.Set2

    for i, patient_id in enumerate(selected_patients):
        if patient_id not in patient_ids:
            continue

        patient_idx = patient_ids.index(patient_id)

        # Get coordinates at each timepoint
        coords = []
        for timepoint in timepoints:
            embedding = embeddings[timepoint]
            coords.append(embedding[patient_idx])

        coords = np.array(coords)

        # Plot trajectory line
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode='lines+markers',
                marker=dict(
                    size=10,
                    color=colors[i % len(colors)],
                    line=dict(width=2, color='white'),
                ),
                line=dict(width=2, color=colors[i % len(colors)]),
                name=patient_id,
                text=timepoints,
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}',
            )
        )

        # Add arrows to show direction
        for j in range(len(coords) - 1):
            fig.add_annotation(
                x=coords[j+1, 0],
                y=coords[j+1, 1],
                ax=coords[j, 0],
                ay=coords[j, 1],
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors[i % len(colors)],
            )

    fig.update_layout(
        title="Patient Trajectories Over Time",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=width,
        height=height,
        template='plotly_white',
        hovermode='closest',
    )

    return fig


def create_patient_timeline(
    patient_id: str,
    trajectory: TrajectoryResult,
    biomarker_names: Optional[List[str]] = None,
    width: int = 1200,
    height: int = 800,
) -> go.Figure:
    """
    Create detailed timeline for a single patient

    Args:
        patient_id: Patient identifier
        trajectory: TrajectoryResult for patient
        biomarker_names: Names of biomarkers to plot
        width: Figure width
        height: Figure height

    Returns:
        Multi-panel timeline figure
    """
    n_timepoints = len(trajectory.timepoints)

    # Determine number of subplots
    if trajectory.biomarkers and biomarker_names:
        n_biomarkers = len(biomarker_names)
        n_rows = 2 + n_biomarkers  # Cluster + coordinates + biomarkers
    else:
        n_rows = 2

    # Create subplots
    subplot_titles = ['Cluster Assignment', 'Embedding Coordinates']
    if trajectory.biomarkers and biomarker_names:
        subplot_titles.extend(biomarker_names)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        row_heights=[1] * n_rows,
    )

    # Plot 1: Cluster assignment
    fig.add_trace(
        go.Scatter(
            x=trajectory.timepoints,
            y=trajectory.clusters,
            mode='lines+markers',
            marker=dict(size=12, color='steelblue'),
            line=dict(width=3, color='steelblue'),
            name='Cluster',
        ),
        row=1,
        col=1,
    )

    # Plot 2: Embedding coordinates
    coords = np.array(trajectory.coordinates)
    fig.add_trace(
        go.Scatter(
            x=trajectory.timepoints,
            y=coords[:, 0],
            mode='lines+markers',
            name='Dim 1',
            marker=dict(color='red'),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trajectory.timepoints,
            y=coords[:, 1],
            mode='lines+markers',
            name='Dim 2',
            marker=dict(color='blue'),
        ),
        row=2,
        col=1,
    )

    # Plot 3+: Biomarkers
    if trajectory.biomarkers and biomarker_names:
        for i, biomarker in enumerate(biomarker_names):
            if biomarker in trajectory.biomarkers:
                values = trajectory.biomarkers[biomarker]
                fig.add_trace(
                    go.Scatter(
                        x=trajectory.timepoints,
                        y=values,
                        mode='lines+markers',
                        name=biomarker,
                        marker=dict(size=10),
                        line=dict(width=2),
                    ),
                    row=3 + i,
                    col=1,
                )

    # Update layout
    fig.update_layout(
        title=f"Patient Timeline: {patient_id}",
        width=width,
        height=height,
        template='plotly_white',
        showlegend=True,
    )

    # Update axes
    fig.update_yaxes(title_text="Cluster ID", row=1, col=1)
    fig.update_yaxes(title_text="Coordinate", row=2, col=1)

    if trajectory.biomarkers and biomarker_names:
        for i in range(n_biomarkers):
            fig.update_yaxes(title_text="Value", row=3 + i, col=1)

    for row in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Timepoint", row=row, col=1)

    return fig


def create_transition_matrix(
    labels_dict: Dict[str, np.ndarray],
    patient_ids: List[str],
    normalize: bool = True,
    width: int = 800,
    height: int = 800,
) -> go.Figure:
    """
    Create transition probability matrix

    Args:
        labels_dict: Dict of {timepoint: labels}
        patient_ids: Patient identifiers
        normalize: Whether to normalize by row
        width: Figure width
        height: Figure height

    Returns:
        Heatmap of transition probabilities
    """
    timepoints = sorted(labels_dict.keys())

    if len(timepoints) < 2:
        raise ValueError("Need at least 2 timepoints for transitions")

    # Get all clusters
    all_clusters = set()
    for labels in labels_dict.values():
        all_clusters.update(np.unique(labels[labels >= 0]))
    all_clusters = sorted(all_clusters)

    # Build transition matrix (average over all consecutive timepoint pairs)
    n_clusters = len(all_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(all_clusters)}

    transition_matrix = np.zeros((n_clusters, n_clusters))
    n_transitions = 0

    for t in range(len(timepoints) - 1):
        labels_t1 = labels_dict[timepoints[t]]
        labels_t2 = labels_dict[timepoints[t + 1]]

        for l1, l2 in zip(labels_t1, labels_t2):
            if l1 >= 0 and l2 >= 0:
                i = cluster_to_idx[l1]
                j = cluster_to_idx[l2]
                transition_matrix[i, j] += 1
                n_transitions += 1

    # Normalize
    if normalize:
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=[f'To C{c}' for c in all_clusters],
        y=[f'From C{c}' for c in all_clusters],
        colorscale='Blues',
        text=transition_matrix,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Probability" if normalize else "Count"),
    ))

    fig.update_layout(
        title="Cluster Transition Matrix",
        xaxis_title="Destination Cluster",
        yaxis_title="Source Cluster",
        width=width,
        height=height,
        template='plotly_white',
    )

    return fig


def analyze_trajectories(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    patient_ids: List[str],
    biomarkers: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[TrajectoryResult]:
    """
    Analyze patient trajectories

    Args:
        embeddings: Dict of timepoint -> embedding
        labels: Dict of timepoint -> labels
        patient_ids: Patient identifiers
        biomarkers: Optional biomarker data at each timepoint

    Returns:
        List of TrajectoryResult objects
    """
    timepoints = sorted(embeddings.keys())
    results = []

    for i, patient_id in enumerate(patient_ids):
        # Collect data for this patient
        patient_clusters = []
        patient_coords = []
        patient_biomarkers = {}

        for timepoint in timepoints:
            embedding = embeddings[timepoint]
            label_array = labels[timepoint]

            patient_clusters.append(int(label_array[i]))
            patient_coords.append((float(embedding[i, 0]), float(embedding[i, 1])))

            # Collect biomarkers
            if biomarkers and timepoint in biomarkers:
                for col in biomarkers[timepoint].columns:
                    if col not in patient_biomarkers:
                        patient_biomarkers[col] = []
                    patient_biomarkers[col].append(float(biomarkers[timepoint].iloc[i][col]))

        # Calculate stability score (fraction of time in most common cluster)
        if patient_clusters:
            cluster_counts = pd.Series(patient_clusters).value_counts()
            stability_score = cluster_counts.iloc[0] / len(patient_clusters)
        else:
            stability_score = 0.0

        # Count transitions
        n_transitions = sum(
            1 for j in range(len(patient_clusters) - 1)
            if patient_clusters[j] != patient_clusters[j + 1]
        )

        result = TrajectoryResult(
            patient_id=patient_id,
            timepoints=timepoints,
            clusters=patient_clusters,
            coordinates=patient_coords,
            biomarkers=patient_biomarkers if patient_biomarkers else None,
            stability_score=float(stability_score),
            n_transitions=n_transitions,
        )

        results.append(result)

    return results


def create_stability_plot(
    trajectories: List[TrajectoryResult],
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """
    Plot patient stability scores

    Args:
        trajectories: List of TrajectoryResult
        width: Figure width
        height: Figure height

    Returns:
        Histogram of stability scores
    """
    stability_scores = [t.stability_score for t in trajectories]
    n_transitions = [t.n_transitions for t in trajectories]
    patient_ids = [t.patient_id for t in trajectories]

    fig = go.Figure()

    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=n_transitions,
            y=stability_scores,
            mode='markers',
            marker=dict(
                size=10,
                color=stability_scores,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Stability"),
                line=dict(width=1, color='white'),
            ),
            text=patient_ids,
            hovertemplate='Patient: %{text}<br>Transitions: %{x}<br>Stability: %{y:.2f}',
        )
    )

    fig.update_layout(
        title="Patient Stability Analysis",
        xaxis_title="Number of Cluster Transitions",
        yaxis_title="Stability Score",
        width=width,
        height=height,
        template='plotly_white',
    )

    return fig


def create_trajectory_heatmap(
    trajectories: List[TrajectoryResult],
    sort_by: str = 'stability',
    width: int = 1200,
    height: int = 800,
) -> go.Figure:
    """
    Create heatmap of patient trajectories

    Args:
        trajectories: List of TrajectoryResult
        sort_by: How to sort patients ('stability', 'transitions', 'patient_id')
        width: Figure width
        height: Figure height

    Returns:
        Heatmap showing cluster membership over time
    """
    # Sort trajectories
    if sort_by == 'stability':
        trajectories = sorted(trajectories, key=lambda t: t.stability_score, reverse=True)
    elif sort_by == 'transitions':
        trajectories = sorted(trajectories, key=lambda t: t.n_transitions)
    else:  # patient_id
        trajectories = sorted(trajectories, key=lambda t: t.patient_id)

    # Build matrix
    patient_ids = [t.patient_id for t in trajectories]
    timepoints = trajectories[0].timepoints if trajectories else []

    matrix = []
    for traj in trajectories:
        matrix.append(traj.clusters)

    matrix = np.array(matrix)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=timepoints,
        y=patient_ids,
        colorscale='Viridis',
        colorbar=dict(title="Cluster ID"),
    ))

    fig.update_layout(
        title=f"Patient Trajectories (sorted by {sort_by})",
        xaxis_title="Timepoint",
        yaxis_title="Patient",
        width=width,
        height=height,
        template='plotly_white',
    )

    return fig


def create_trajectory_flow(
    labels_dict: Dict[str, np.ndarray],
    patient_ids: List[str],
    width: int = 1200,
    height: int = 800,
) -> go.Figure:
    """
    Create alluvial/flow diagram of trajectories

    Args:
        labels_dict: Dict of timepoint -> labels
        patient_ids: Patient identifiers
        width: Figure width
        height: Figure height

    Returns:
        Sankey diagram showing patient flows
    """
    timepoints = sorted(labels_dict.keys())

    if len(timepoints) < 2:
        raise ValueError("Need at least 2 timepoints")

    # Build nodes and links
    all_nodes = []
    node_labels = []

    # Create nodes for each cluster at each timepoint
    node_map = {}
    node_idx = 0

    for t_idx, timepoint in enumerate(timepoints):
        labels = labels_dict[timepoint]
        unique_clusters = np.unique(labels[labels >= 0])

        for cluster in unique_clusters:
            node_label = f'T{t_idx+1} C{cluster}'
            all_nodes.append((timepoint, cluster))
            node_labels.append(node_label)
            node_map[(timepoint, cluster)] = node_idx
            node_idx += 1

    # Build links
    sources = []
    targets = []
    values = []

    for t in range(len(timepoints) - 1):
        labels_t1 = labels_dict[timepoints[t]]
        labels_t2 = labels_dict[timepoints[t + 1]]

        # Count transitions
        transitions = {}
        for l1, l2 in zip(labels_t1, labels_t2):
            if l1 >= 0 and l2 >= 0:
                key = ((timepoints[t], l1), (timepoints[t + 1], l2))
                transitions[key] = transitions.get(key, 0) + 1

        # Add links
        for (source_node, target_node), count in transitions.items():
            if source_node in node_map and target_node in node_map:
                sources.append(node_map[source_node])
                targets.append(node_map[target_node])
                values.append(count)

    # Create Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        )
    )])

    fig.update_layout(
        title="Patient Flow Through Clusters",
        width=width,
        height=height,
        font=dict(size=10),
    )

    return fig