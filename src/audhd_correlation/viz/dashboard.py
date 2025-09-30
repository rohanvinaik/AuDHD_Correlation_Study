"""Main Dash application for interactive visualization dashboard

Provides comprehensive interactive interface for exploring clustering results.
"""
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from .embedding_plots import (
    create_embedding_plot,
    create_multi_overlay_plot,
    EmbeddingPlotConfig,
)
from .cluster_comparison import (
    create_cluster_comparison,
    create_violin_comparison,
    create_sankey_diagram,
    create_feature_importance_comparison,
)
from .trajectories import (
    create_trajectory_plot,
    create_patient_timeline,
    create_transition_matrix,
    analyze_trajectories,
)
from .heatmaps import (
    create_biomarker_heatmap,
    create_correlation_heatmap,
    HeatmapConfig,
)
from .clinical_interface import (
    create_patient_card,
    create_risk_assessment,
    create_treatment_recommendations,
    generate_clinical_report,
)


def create_dashboard(
    embedding: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    data: Optional[pd.DataFrame] = None,
    patient_ids: Optional[List[str]] = None,
    app_title: str = "AuDHD Multi-Omics Dashboard",
    external_stylesheets: Optional[List] = None,
) -> Dash:
    """
    Create Dash application

    Args:
        embedding: Initial embedding data
        labels: Initial cluster labels
        data: Feature data
        patient_ids: Patient identifiers
        app_title: Application title
        external_stylesheets: External CSS stylesheets

    Returns:
        Dash application object
    """
    if external_stylesheets is None:
        external_stylesheets = [dbc.themes.BOOTSTRAP]

    app = Dash(__name__, external_stylesheets=external_stylesheets)

    # Store data in app for callbacks
    app.embedding = embedding
    app.labels = labels
    app.data = data
    app.patient_ids = patient_ids

    # Define layout
    app.layout = _create_layout(app_title)

    # Register callbacks
    _register_callbacks(app)

    return app


def _create_layout(app_title: str) -> html.Div:
    """Create dashboard layout"""

    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1(app_title, className="text-center mb-4"),
                html.Hr(),
            ])
        ]),

        # Tabs for different views
        dbc.Tabs([
            # Tab 1: Embedding View
            dbc.Tab(label="Embedding View", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Embedding Plot Controls", className="mt-3"),
                        html.Label("Overlay Variable:"),
                        dcc.Dropdown(
                            id='overlay-dropdown',
                            options=[
                                {'label': 'Clusters', 'value': 'clusters'},
                                {'label': 'Age', 'value': 'age'},
                                {'label': 'Severity', 'value': 'severity'},
                            ],
                            value='clusters',
                        ),
                        html.Label("Patient Filter:", className="mt-3"),
                        dcc.Dropdown(
                            id='patient-filter',
                            options=[{'label': 'All Patients', 'value': 'all'}],
                            value='all',
                            multi=True,
                        ),
                        html.Label("Marker Size:", className="mt-3"),
                        dcc.Slider(
                            id='marker-size-slider',
                            min=4,
                            max=20,
                            step=2,
                            value=8,
                            marks={i: str(i) for i in range(4, 21, 4)},
                        ),
                        dbc.Button(
                            "Export Figure",
                            id="export-embedding-btn",
                            color="primary",
                            className="mt-3",
                        ),
                    ], width=3),
                    dbc.Col([
                        dcc.Loading(
                            id="loading-embedding",
                            children=[dcc.Graph(id='embedding-plot')],
                            type="default",
                        ),
                    ], width=9),
                ]),
            ]),

            # Tab 2: Cluster Comparison
            dbc.Tab(label="Cluster Comparison", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Comparison Controls", className="mt-3"),
                        html.Label("Features to Compare:"),
                        dcc.Dropdown(
                            id='features-dropdown',
                            options=[],
                            value=[],
                            multi=True,
                        ),
                        html.Label("Test Method:", className="mt-3"),
                        dcc.RadioItems(
                            id='test-method-radio',
                            options=[
                                {'label': 'Kruskal-Wallis', 'value': 'kruskal'},
                                {'label': 'ANOVA', 'value': 'anova'},
                            ],
                            value='kruskal',
                        ),
                        html.Label("Top Features:", className="mt-3"),
                        dcc.Slider(
                            id='top-features-slider',
                            min=10,
                            max=50,
                            step=5,
                            value=20,
                            marks={i: str(i) for i in range(10, 51, 10)},
                        ),
                    ], width=3),
                    dbc.Col([
                        dcc.Loading(
                            id="loading-comparison",
                            children=[
                                dcc.Graph(id='violin-plot'),
                                dcc.Graph(id='feature-importance-plot'),
                            ],
                            type="default",
                        ),
                    ], width=9),
                ]),
            ]),

            # Tab 3: Patient Trajectories
            dbc.Tab(label="Patient Trajectories", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Trajectory Controls", className="mt-3"),
                        html.Label("Select Patients:"),
                        dcc.Dropdown(
                            id='trajectory-patients-dropdown',
                            options=[],
                            value=[],
                            multi=True,
                        ),
                        html.Label("Show Transitions:", className="mt-3"),
                        dbc.Checklist(
                            id='show-transitions-check',
                            options=[{'label': 'Show', 'value': 'show'}],
                            value=['show'],
                        ),
                    ], width=3),
                    dbc.Col([
                        dcc.Loading(
                            id="loading-trajectories",
                            children=[
                                dcc.Graph(id='trajectory-plot'),
                                dcc.Graph(id='transition-matrix'),
                            ],
                            type="default",
                        ),
                    ], width=9),
                ]),
            ]),

            # Tab 4: Biomarker Heatmaps
            dbc.Tab(label="Biomarker Heatmaps", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Heatmap Controls", className="mt-3"),
                        html.Label("Normalization:"),
                        dcc.RadioItems(
                            id='normalization-radio',
                            options=[
                                {'label': 'Z-score', 'value': 'zscore'},
                                {'label': 'Min-Max', 'value': 'minmax'},
                                {'label': 'None', 'value': 'none'},
                            ],
                            value='zscore',
                        ),
                        html.Label("Clustering:", className="mt-3"),
                        dbc.Checklist(
                            id='clustering-check',
                            options=[
                                {'label': 'Cluster Rows', 'value': 'rows'},
                                {'label': 'Cluster Columns', 'value': 'cols'},
                            ],
                            value=['cols'],
                        ),
                        html.Label("Colorscale:", className="mt-3"),
                        dcc.Dropdown(
                            id='colorscale-dropdown',
                            options=[
                                {'label': 'RdBu', 'value': 'RdBu_r'},
                                {'label': 'Viridis', 'value': 'Viridis'},
                                {'label': 'Plasma', 'value': 'Plasma'},
                            ],
                            value='RdBu_r',
                        ),
                    ], width=3),
                    dbc.Col([
                        dcc.Loading(
                            id="loading-heatmap",
                            children=[dcc.Graph(id='heatmap-plot')],
                            type="default",
                        ),
                    ], width=9),
                ]),
            ]),

            # Tab 5: Clinical Decision Support
            dbc.Tab(label="Clinical Decision Support", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Patient Selection", className="mt-3"),
                        html.Label("Select Patient:"),
                        dcc.Dropdown(
                            id='clinical-patient-dropdown',
                            options=[],
                            value=None,
                        ),
                        html.Div(id='patient-info-card', className="mt-3"),
                    ], width=3),
                    dbc.Col([
                        dcc.Loading(
                            id="loading-clinical",
                            children=[
                                dcc.Graph(id='patient-card'),
                                dcc.Graph(id='risk-assessment'),
                                dcc.Graph(id='treatment-recommendations'),
                            ],
                            type="default",
                        ),
                    ], width=9),
                ]),
            ]),
        ]),

        # Footer
        html.Hr(),
        html.P(
            "AuDHD Multi-Omics Correlation Study - Interactive Dashboard",
            className="text-center text-muted",
        ),
    ], fluid=True)


def _register_callbacks(app: Dash) -> None:
    """Register dashboard callbacks"""

    # Callback for embedding plot
    @app.callback(
        Output('embedding-plot', 'figure'),
        [Input('overlay-dropdown', 'value'),
         Input('marker-size-slider', 'value')],
    )
    def update_embedding_plot(overlay_var, marker_size):
        if app.embedding is None or app.labels is None:
            return go.Figure().add_annotation(
                text="No data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        config = EmbeddingPlotConfig(
            title="Embedding Visualization",
            marker_size=marker_size,
        )

        # Determine overlay
        if overlay_var == 'clusters':
            fig = create_embedding_plot(
                embedding=app.embedding,
                labels=app.labels,
                patient_ids=app.patient_ids,
                config=config,
                categorical=True,
            )
        else:
            # Use data if available
            if app.data is not None and overlay_var in app.data.columns:
                overlay_data = app.data[overlay_var].values
                fig = create_embedding_plot(
                    embedding=app.embedding,
                    overlay_data=overlay_data,
                    overlay_name=overlay_var,
                    patient_ids=app.patient_ids,
                    config=config,
                    categorical=False,
                )
            else:
                fig = create_embedding_plot(
                    embedding=app.embedding,
                    labels=app.labels,
                    patient_ids=app.patient_ids,
                    config=config,
                    categorical=True,
                )

        return fig

    # Callback for violin plot
    @app.callback(
        Output('violin-plot', 'figure'),
        [Input('features-dropdown', 'value')],
    )
    def update_violin_plot(selected_features):
        if app.data is None or app.labels is None:
            return go.Figure()

        if not selected_features:
            # Use first 5 features
            selected_features = app.data.columns[:5].tolist()

        fig = create_violin_comparison(
            data=app.data,
            labels=app.labels,
            features=selected_features,
        )

        return fig

    # Callback for feature importance
    @app.callback(
        Output('feature-importance-plot', 'figure'),
        [Input('test-method-radio', 'value'),
         Input('top-features-slider', 'value')],
    )
    def update_feature_importance(test_method, n_top):
        if app.data is None or app.labels is None:
            return go.Figure()

        # Calculate comparison results
        from .cluster_comparison import create_cluster_comparison

        results = create_cluster_comparison(
            data=app.data,
            labels=app.labels,
            test_method=test_method,
        )

        fig = create_feature_importance_comparison(
            comparison_results=results,
            n_top=n_top,
        )

        return fig


def run_dashboard(
    app: Dash,
    host: str = '127.0.0.1',
    port: int = 8050,
    debug: bool = True,
) -> None:
    """
    Run dashboard server

    Args:
        app: Dash application
        host: Host address
        port: Port number
        debug: Debug mode
    """
    print(f"Starting dashboard at http://{host}:{port}")
    print("Press Ctrl+C to stop")

    app.run_server(host=host, port=port, debug=debug)


def load_sample_data() -> tuple:
    """
    Load sample data for dashboard demo

    Returns:
        Tuple of (embedding, labels, data, patient_ids)
    """
    np.random.seed(42)

    n_samples = 200
    n_features = 20

    # Generate sample embedding (t-SNE-like)
    embedding = np.random.randn(n_samples, 2)

    # Add cluster structure
    n_clusters = 4
    cluster_centers = np.array([
        [-3, -3],
        [3, -3],
        [-3, 3],
        [3, 3],
    ])

    labels = np.random.choice(n_clusters, size=n_samples)

    for i in range(n_samples):
        cluster_id = labels[i]
        embedding[i] += cluster_centers[cluster_id] + np.random.randn(2) * 0.5

    # Generate sample feature data
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names,
    )

    # Add cluster-specific patterns
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        # Make some features higher in certain clusters
        data.loc[mask, f'Feature_{cluster_id}'] += 2.0

    # Patient IDs
    patient_ids = [f'P{i:03d}' for i in range(n_samples)]

    return embedding, labels, data, patient_ids


def create_demo_dashboard() -> Dash:
    """
    Create dashboard with sample data for demonstration

    Returns:
        Dash application with sample data
    """
    embedding, labels, data, patient_ids = load_sample_data()

    app = create_dashboard(
        embedding=embedding,
        labels=labels,
        data=data,
        patient_ids=patient_ids,
        app_title="AuDHD Multi-Omics Dashboard (Demo)",
    )

    return app


if __name__ == '__main__':
    # Run demo dashboard
    app = create_demo_dashboard()
    run_dashboard(app, debug=True)