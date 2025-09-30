"""Clinical decision support interface

Provides patient-level insights and treatment recommendations.
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
class ClinicalDecisionSupport:
    """Clinical decision support for a patient"""
    patient_id: str
    cluster_id: int
    cluster_label: str

    # Risk scores
    risk_scores: Dict[str, float]
    risk_level: str  # 'Low', 'Medium', 'High'

    # Biomarker status
    biomarker_alerts: List[str]
    biomarker_summary: Dict[str, str]  # feature -> status

    # Treatment recommendations
    recommended_treatments: List[str]
    contraindicated_treatments: List[str]

    # Similar patients
    similar_patients: List[str]
    cluster_statistics: Dict[str, float]

    # Trajectory prediction
    predicted_trajectory: Optional[str] = None
    confidence: float = 0.0


def create_patient_card(
    patient_id: str,
    patient_data: pd.Series,
    cluster_id: int,
    cluster_profile: Dict[str, float],
    reference_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """
    Create patient summary card

    Args:
        patient_id: Patient identifier
        patient_data: Patient feature values
        cluster_id: Patient's cluster assignment
        cluster_profile: Average values for cluster
        reference_ranges: Normal reference ranges for features
        width: Figure width
        height: Figure height

    Returns:
        Patient card figure
    """
    # Select key features
    features = patient_data.index.tolist()
    values = patient_data.values

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.3, 0.7],
        subplot_titles=[
            f"Patient {patient_id} - Cluster {cluster_id}",
            "Biomarker Profile"
        ],
        specs=[[{"type": "indicator"}], [{"type": "bar"}]],
    )

    # Top: Overall risk indicator
    risk_score = _calculate_risk_score(patient_data, reference_ranges)

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ),
        row=1,
        col=1,
    )

    # Bottom: Biomarker comparison
    # Show patient values vs cluster average
    fig.add_trace(
        go.Bar(
            name='Patient',
            x=features,
            y=values,
            marker_color='steelblue',
        ),
        row=2,
        col=1,
    )

    cluster_values = [cluster_profile.get(f, 0) for f in features]
    fig.add_trace(
        go.Bar(
            name='Cluster Average',
            x=features,
            y=cluster_values,
            marker_color='lightgray',
        ),
        row=2,
        col=1,
    )

    # Add reference ranges if available
    if reference_ranges:
        for feature in features:
            if feature in reference_ranges:
                lower, upper = reference_ranges[feature]
                fig.add_hline(
                    y=lower,
                    line_dash="dash",
                    line_color="green",
                    opacity=0.5,
                    row=2,
                    col=1,
                )
                fig.add_hline(
                    y=upper,
                    line_dash="dash",
                    line_color="green",
                    opacity=0.5,
                    row=2,
                    col=1,
                )

    fig.update_layout(
        width=width,
        height=height,
        template='plotly_white',
        showlegend=True,
    )

    fig.update_xaxes(tickangle=-45, row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)

    return fig


def _calculate_risk_score(
    patient_data: pd.Series,
    reference_ranges: Optional[Dict[str, Tuple[float, float]]],
) -> float:
    """Calculate overall risk score for patient"""

    if reference_ranges is None:
        return 0.5  # Default medium risk

    # Count features outside reference range
    n_abnormal = 0
    n_total = 0

    for feature, value in patient_data.items():
        if feature in reference_ranges:
            lower, upper = reference_ranges[feature]
            n_total += 1
            if value < lower or value > upper:
                n_abnormal += 1

    if n_total == 0:
        return 0.5

    # Risk score is proportion of abnormal features
    risk_score = n_abnormal / n_total

    return risk_score


def create_risk_assessment(
    patient_id: str,
    patient_data: pd.Series,
    cluster_data: pd.DataFrame,
    risk_factors: List[str],
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """
    Create risk assessment visualization

    Args:
        patient_id: Patient identifier
        patient_data: Patient feature values
        cluster_data: Data from all patients in cluster
        risk_factors: List of risk factor features
        width: Figure width
        height: Figure height

    Returns:
        Risk assessment figure
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=["Patient Risk Factors", "Comparison to Cluster"],
        specs=[[{"type": "bar"}, {"type": "box"}]],
    )

    # Left: Patient risk factors
    risk_values = [patient_data.get(f, 0) for f in risk_factors]
    risk_colors = ['red' if v > 0.7 else 'orange' if v > 0.4 else 'green' for v in risk_values]

    fig.add_trace(
        go.Bar(
            y=risk_factors,
            x=risk_values,
            orientation='h',
            marker=dict(color=risk_colors),
            text=[f'{v:.2f}' for v in risk_values],
            textposition='auto',
        ),
        row=1,
        col=1,
    )

    # Right: Box plots comparing patient to cluster
    for i, factor in enumerate(risk_factors):
        if factor in cluster_data.columns:
            # Add box plot for cluster
            fig.add_trace(
                go.Box(
                    y=cluster_data[factor],
                    name=factor,
                    boxmean='sd',
                    marker_color='lightblue',
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            # Add patient point
            patient_value = patient_data.get(factor, 0)
            fig.add_trace(
                go.Scatter(
                    x=[factor],
                    y=[patient_value],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='diamond'),
                    name=f'{patient_id}' if i == 0 else None,
                    showlegend=(i == 0),
                ),
                row=1,
                col=2,
            )

    fig.update_layout(
        title=f"Risk Assessment: Patient {patient_id}",
        width=width,
        height=height,
        template='plotly_white',
    )

    fig.update_xaxes(title_text="Risk Score", row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=2)

    return fig


def create_treatment_recommendations(
    patient_id: str,
    cluster_id: int,
    treatment_outcomes: Dict[str, Dict[str, float]],
    patient_features: Optional[pd.Series] = None,
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    """
    Create treatment recommendation visualization

    Args:
        patient_id: Patient identifier
        cluster_id: Patient's cluster
        treatment_outcomes: Dict of {treatment: {metric: value}}
        patient_features: Patient features for personalization
        width: Figure width
        height: Figure height

    Returns:
        Treatment recommendation figure
    """
    # Extract treatments and outcomes
    treatments = list(treatment_outcomes.keys())
    success_rates = [outcomes.get('success_rate', 0) for outcomes in treatment_outcomes.values()]
    response_times = [outcomes.get('response_time', 0) for outcomes in treatment_outcomes.values()]
    side_effects = [outcomes.get('side_effects', 0) for outcomes in treatment_outcomes.values()]

    # Create figure
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Success Rate", "Response Time", "Side Effect Risk"],
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
    )

    # Success rate
    colors_success = ['green' if sr > 0.7 else 'orange' if sr > 0.5 else 'red' for sr in success_rates]
    fig.add_trace(
        go.Bar(
            x=treatments,
            y=success_rates,
            marker_color=colors_success,
            text=[f'{sr:.0%}' for sr in success_rates],
            textposition='auto',
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Response time
    fig.add_trace(
        go.Bar(
            x=treatments,
            y=response_times,
            marker_color='steelblue',
            text=[f'{rt:.1f}w' for rt in response_times],
            textposition='auto',
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Side effects
    colors_se = ['green' if se < 0.2 else 'orange' if se < 0.4 else 'red' for se in side_effects]
    fig.add_trace(
        go.Bar(
            x=treatments,
            y=side_effects,
            marker_color=colors_se,
            text=[f'{se:.0%}' for se in side_effects],
            textposition='auto',
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=f"Treatment Recommendations - Patient {patient_id} (Cluster {cluster_id})",
        width=width,
        height=height,
        template='plotly_white',
    )

    # Update axes
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    fig.update_xaxes(tickangle=-45, row=1, col=3)

    fig.update_yaxes(title_text="Rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Weeks", row=1, col=2)
    fig.update_yaxes(title_text="Risk", range=[0, 1], row=1, col=3)

    return fig


def generate_clinical_report(
    patient_id: str,
    patient_data: pd.Series,
    cluster_id: int,
    cluster_profile: Dict[str, float],
    reference_ranges: Dict[str, Tuple[float, float]],
    treatment_outcomes: Dict[str, Dict[str, float]],
) -> ClinicalDecisionSupport:
    """
    Generate comprehensive clinical decision support

    Args:
        patient_id: Patient identifier
        patient_data: Patient feature values
        cluster_id: Cluster assignment
        cluster_profile: Cluster average profile
        reference_ranges: Reference ranges for features
        treatment_outcomes: Treatment outcome statistics

    Returns:
        ClinicalDecisionSupport object
    """
    # Calculate risk scores
    risk_scores = {}
    biomarker_alerts = []
    biomarker_summary = {}

    for feature, value in patient_data.items():
        if feature in reference_ranges:
            lower, upper = reference_ranges[feature]

            # Determine status
            if value < lower:
                status = "Low"
                biomarker_alerts.append(f"{feature}: Below normal ({value:.2f} < {lower:.2f})")
            elif value > upper:
                status = "High"
                biomarker_alerts.append(f"{feature}: Above normal ({value:.2f} > {upper:.2f})")
            else:
                status = "Normal"

            biomarker_summary[feature] = status

            # Calculate risk contribution
            if value < lower:
                risk_scores[feature] = (lower - value) / (lower + 1e-10)
            elif value > upper:
                risk_scores[feature] = (value - upper) / (upper + 1e-10)
            else:
                risk_scores[feature] = 0.0

    # Overall risk level
    avg_risk = np.mean(list(risk_scores.values())) if risk_scores else 0.0

    if avg_risk < 0.3:
        risk_level = "Low"
    elif avg_risk < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Treatment recommendations
    # Rank treatments by success rate
    treatment_ranking = sorted(
        treatment_outcomes.items(),
        key=lambda x: x[1].get('success_rate', 0),
        reverse=True
    )

    recommended_treatments = [t[0] for t in treatment_ranking[:3]]

    # Contraindications (treatments with high side effects and low success)
    contraindicated_treatments = [
        t for t, outcomes in treatment_outcomes.items()
        if outcomes.get('side_effects', 0) > 0.5 and outcomes.get('success_rate', 0) < 0.5
    ]

    # Find similar patients (placeholder - would use actual similarity calculation)
    similar_patients = [f"P{i}" for i in range(1, 6)]

    # Cluster statistics
    cluster_statistics = {
        'cluster_size': 100,  # Placeholder
        'avg_age': 35.0,
        'response_rate': 0.75,
    }

    return ClinicalDecisionSupport(
        patient_id=patient_id,
        cluster_id=cluster_id,
        cluster_label=f"Cluster {cluster_id}",
        risk_scores=risk_scores,
        risk_level=risk_level,
        biomarker_alerts=biomarker_alerts,
        biomarker_summary=biomarker_summary,
        recommended_treatments=recommended_treatments,
        contraindicated_treatments=contraindicated_treatments,
        similar_patients=similar_patients,
        cluster_statistics=cluster_statistics,
        predicted_trajectory="Stable",
        confidence=0.85,
    )


def create_cohort_comparison(
    cohort1_data: pd.DataFrame,
    cohort2_data: pd.DataFrame,
    cohort1_name: str = "Cohort 1",
    cohort2_name: str = "Cohort 2",
    features: Optional[List[str]] = None,
    width: int = 1200,
    height: int = 800,
) -> go.Figure:
    """
    Compare two cohorts

    Args:
        cohort1_data: Data for first cohort
        cohort2_data: Data for second cohort
        cohort1_name: Name for first cohort
        cohort2_name: Name for second cohort
        features: Features to compare (if None, use all)
        width: Figure width
        height: Figure height

    Returns:
        Cohort comparison figure
    """
    if features is None:
        features = cohort1_data.columns.tolist()

    # Create subplots
    n_features = len(features)
    n_rows = (n_features + 1) // 2
    n_cols = 2

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=features,
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
    )

    colors = ['steelblue', 'coral']

    for idx, feature in enumerate(features):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Add violin plots
        fig.add_trace(
            go.Violin(
                y=cohort1_data[feature],
                name=cohort1_name,
                marker_color=colors[0],
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Violin(
                y=cohort2_data[feature],
                name=cohort2_name,
                marker_color=colors[1],
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Cohort Comparison: {cohort1_name} vs {cohort2_name}",
        width=width,
        height=height,
        template='plotly_white',
        showlegend=True,
    )

    return fig


def create_patient_similarity_network(
    patient_id: str,
    all_patients_data: pd.DataFrame,
    patient_ids: List[str],
    n_similar: int = 10,
    width: int = 800,
    height: int = 800,
) -> go.Figure:
    """
    Create network of similar patients

    Args:
        patient_id: Target patient
        all_patients_data: Data for all patients
        patient_ids: Patient identifiers
        n_similar: Number of similar patients to show
        width: Figure width
        height: Figure height

    Returns:
        Network visualization
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Find patient index
    patient_idx = patient_ids.index(patient_id)

    # Calculate similarities
    similarities = cosine_similarity(
        all_patients_data.iloc[patient_idx:patient_idx+1],
        all_patients_data
    )[0]

    # Get top similar patients (excluding self)
    similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
    similar_ids = [patient_ids[i] for i in similar_indices]
    similar_scores = similarities[similar_indices]

    # Create network layout (star layout)
    n_nodes = len(similar_ids) + 1
    angles = np.linspace(0, 2*np.pi, len(similar_ids), endpoint=False)

    # Center node (target patient)
    node_x = [0]
    node_y = [0]
    node_text = [patient_id]
    node_size = [30]
    node_color = ['red']

    # Similar patients (arranged in circle)
    for i, (sim_id, score) in enumerate(zip(similar_ids, similar_scores)):
        node_x.append(np.cos(angles[i]))
        node_y.append(np.sin(angles[i]))
        node_text.append(f"{sim_id}<br>Similarity: {score:.2f}")
        node_size.append(15)
        node_color.append('steelblue')

    # Create edges
    edge_x = []
    edge_y = []

    for i in range(1, n_nodes):
        edge_x.extend([0, node_x[i], None])
        edge_y.extend([0, node_y[i], None])

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=1, color='lightgray'),
            hoverinfo='none',
            showlegend=False,
        )
    )

    # Add nodes
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white'),
            ),
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Similar Patients to {patient_id}",
        width=width,
        height=height,
        template='plotly_white',
        showgrid=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig