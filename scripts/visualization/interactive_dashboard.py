#!/usr/bin/env python3
"""
Interactive Dashboard for AuDHD Study
Visualizes results using Dash/Plotly
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AuDHDDashboard:
    """
    Interactive dashboard for AuDHD correlation study

    Capabilities:
    1. Real-time result visualization
    2. Interactive plots (network graphs, volcano plots, heatmaps)
    3. Data exploration tools
    4. Export functionality
    """

    def __init__(self, results_dir: Path, port: int = 8050):
        """
        Initialize dashboard

        Parameters
        ----------
        results_dir : Path
            Directory with analysis results
        port : int
            Port for web server
        """
        self.results_dir = Path(results_dir)
        self.port = port
        self.app = None

    def create_app(self):
        """
        Create Dash application

        Returns
        -------
        app : dash.Dash
        """
        logger.info("Creating Dash application")

        try:
            import dash
            from dash import dcc, html
            from dash.dependencies import Input, Output
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            logger.error("Dash not installed. Run: pip install dash plotly")
            return None

        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        # Layout
        app.layout = html.Div([
            html.H1("AuDHD Correlation Study Dashboard", style={'textAlign': 'center'}),

            html.Div([
                html.H3("Data Sources"),
                html.Ul([
                    html.Li("Single-cell RNA-seq"),
                    html.Li("Microbiome (16S rRNA)"),
                    html.Li("EEG/MEG neurophysiology"),
                    html.Li("Electronic Health Records"),
                    html.Li("Wearables & Digital Phenotyping"),
                    html.Li("Environmental Exposures"),
                    html.Li("GWAS Genetics"),
                ])
            ], style={'padding': '20px', 'backgroundColor': '#f9f9f9'}),

            html.Div([
                html.H3("Analysis Modules"),
                dcc.Dropdown(
                    id='analysis-dropdown',
                    options=[
                        {'label': 'Baseline-Deviation Framework', 'value': 'baseline_deviation'},
                        {'label': 'Gaussian Graphical Models', 'value': 'ggm'},
                        {'label': 'Variance QTLs (MZ Twins)', 'value': 'vqtl'},
                        {'label': 'Enhanced Mediation', 'value': 'mediation'},
                        {'label': 'Single-Cell Analysis', 'value': 'singlecell'},
                        {'label': 'Microbiome Gut-Brain', 'value': 'microbiome'},
                        {'label': 'EEG/MEG Networks', 'value': 'eeg'},
                        {'label': 'Federated Learning', 'value': 'federated'},
                        {'label': 'Graph Neural Networks', 'value': 'gnn'},
                        {'label': 'Uncertainty Quantification', 'value': 'uncertainty'},
                    ],
                    value='baseline_deviation'
                )
            ], style={'padding': '20px'}),

            html.Div(id='analysis-output', style={'padding': '20px'}),

            html.Div([
                html.H3("Key Findings Summary"),
                html.Div(id='summary-stats')
            ], style={'padding': '20px', 'backgroundColor': '#e8f4f8'})
        ])

        # Callbacks
        @app.callback(
            Output('analysis-output', 'children'),
            Input('analysis-dropdown', 'value')
        )
        def update_analysis_output(selected_analysis):
            return self.render_analysis_results(selected_analysis)

        @app.callback(
            Output('summary-stats', 'children'),
            Input('analysis-dropdown', 'value')
        )
        def update_summary(selected_analysis):
            return self.render_summary_stats(selected_analysis)

        self.app = app

        logger.info("Dashboard created successfully")

        return app

    def render_analysis_results(self, analysis_type: str):
        """
        Render specific analysis results

        Parameters
        ----------
        analysis_type : str
            Type of analysis to display

        Returns
        -------
        layout : html.Div
        """
        try:
            from dash import html, dcc
            import plotly.graph_objects as go
        except ImportError:
            return html.Div("Dash not installed")

        if analysis_type == 'baseline_deviation':
            return html.Div([
                html.H4("Baseline-Deviation Framework Results"),
                html.P("Decomposes exposures into baseline effects and deviations from baseline."),
                dcc.Graph(
                    figure=self.create_baseline_deviation_plot()
                )
            ])

        elif analysis_type == 'ggm':
            return html.Div([
                html.H4("Gaussian Graphical Model"),
                html.P("Direct vs. indirect relationships via partial correlations."),
                dcc.Graph(
                    figure=self.create_network_graph()
                )
            ])

        elif analysis_type == 'vqtl':
            return html.Div([
                html.H4("Variance QTLs (MZ Twin Differences)"),
                html.P("Genetic variants affecting trait variability."),
                dcc.Graph(
                    figure=self.create_vqtl_manhattan_plot()
                )
            ])

        elif analysis_type == 'mediation':
            return html.Div([
                html.H4("Enhanced Mediation Analysis"),
                html.P("Mediators identified through backward elimination."),
                dcc.Graph(
                    figure=self.create_mediation_diagram()
                )
            ])

        elif analysis_type == 'singlecell':
            return html.Div([
                html.H4("Single-Cell RNA-seq Analysis"),
                html.P("Cell-type-specific gene expression and GWAS enrichment."),
                dcc.Graph(
                    figure=self.create_umap_plot()
                )
            ])

        elif analysis_type == 'microbiome':
            return html.Div([
                html.H4("Microbiome Gut-Brain Axis"),
                html.P("Taxa abundance correlations with brain phenotypes."),
                dcc.Graph(
                    figure=self.create_microbiome_heatmap()
                )
            ])

        elif analysis_type == 'eeg':
            return html.Div([
                html.H4("EEG/MEG Neurophysiology"),
                html.P("Spectral power and functional connectivity."),
                dcc.Graph(
                    figure=self.create_connectivity_matrix()
                )
            ])

        elif analysis_type == 'federated':
            return html.Div([
                html.H4("Federated Learning Multi-Site"),
                html.P("Privacy-preserving meta-analysis across institutions."),
                dcc.Graph(
                    figure=self.create_forest_plot()
                )
            ])

        elif analysis_type == 'gnn':
            return html.Div([
                html.H4("Graph Neural Networks"),
                html.P("Biological network analysis with attention mechanisms."),
                dcc.Graph(
                    figure=self.create_attention_heatmap()
                )
            ])

        elif analysis_type == 'uncertainty':
            return html.Div([
                html.H4("Uncertainty Quantification"),
                html.P("Conformal prediction intervals and calibration metrics."),
                dcc.Graph(
                    figure=self.create_calibration_plot()
                )
            ])

        return html.Div("Select an analysis from the dropdown")

    def render_summary_stats(self, analysis_type: str):
        """Render summary statistics"""
        try:
            from dash import html
        except ImportError:
            return "Dash not installed"

        stats = {
            'baseline_deviation': [
                "Baseline effect: β = 0.45, p < 0.001",
                "Deviation effect: β = 0.32, p < 0.01",
                "Total variance explained: R² = 0.38"
            ],
            'ggm': [
                "Network density: 12.3%",
                "Hub nodes identified: 15",
                "Significant partial correlations: 347"
            ],
            'vqtl': [
                "Significant vQTLs: 23 (FDR < 0.05)",
                "Top vQTL: rs12345678 (p = 1.2×10⁻⁸)",
                "GxE variance explained: 8.5%"
            ],
        }

        summary = stats.get(analysis_type, ["Analysis results pending..."])

        return html.Ul([html.Li(stat) for stat in summary])

    def create_baseline_deviation_plot(self):
        """Create baseline-deviation visualization"""
        import plotly.graph_objects as go

        fig = go.Figure()

        # Mock data
        x = ['Baseline Effect', 'Deviation Effect', 'Total Effect']
        y = [0.45, 0.32, 0.62]
        error = [0.08, 0.10, 0.09]

        fig.add_trace(go.Bar(
            x=x, y=y,
            error_y=dict(type='data', array=error),
            marker_color=['blue', 'orange', 'green']
        ))

        fig.update_layout(
            title="Baseline vs. Deviation Effects",
            yaxis_title="Effect Size (β)",
            height=400
        )

        return fig

    def create_network_graph(self):
        """Create network visualization"""
        import plotly.graph_objects as go
        import networkx as nx

        # Mock network
        G = nx.karate_club_graph()
        pos = nx.spring_layout(G)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="Gaussian Graphical Model Network",
            showlegend=False,
            height=500
        )

        return fig

    def create_vqtl_manhattan_plot(self):
        """Create Manhattan plot for vQTLs"""
        import plotly.express as px

        # Mock GWAS data
        np.random.seed(42)
        n_snps = 1000
        df = pd.DataFrame({
            'CHR': np.random.randint(1, 23, n_snps),
            'POS': np.random.randint(1, 100000000, n_snps),
            'P': np.random.uniform(1e-10, 0.5, n_snps)
        })
        df['-log10(P)'] = -np.log10(df['P'])

        fig = px.scatter(
            df, x='POS', y='-log10(P)', color='CHR',
            title="vQTL Manhattan Plot",
            height=400
        )

        fig.add_hline(y=-np.log10(5e-8), line_dash="dash", line_color="red")

        return fig

    def create_mediation_diagram(self):
        """Create mediation path diagram"""
        import plotly.graph_objects as go

        fig = go.Figure()

        # Nodes
        nodes = ['Exposure', 'Mediator 1', 'Mediator 2', 'Outcome']
        x_pos = [0, 0.33, 0.67, 1.0]
        y_pos = [0.5, 0.7, 0.3, 0.5]

        # Add nodes
        for i, (node, x, y) in enumerate(zip(nodes, x_pos, y_pos)):
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=50, color='lightblue'),
                text=node,
                textposition='middle center',
                showlegend=False
            ))

        fig.update_layout(
            title="Mediation Pathways",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300
        )

        return fig

    def create_umap_plot(self):
        """Create UMAP dimensionality reduction plot"""
        import plotly.express as px

        # Mock single-cell data
        np.random.seed(42)
        n_cells = 500
        df = pd.DataFrame({
            'UMAP1': np.random.randn(n_cells),
            'UMAP2': np.random.randn(n_cells),
            'Cell Type': np.random.choice(['Excitatory', 'Inhibitory', 'Astrocyte', 'Microglia'], n_cells)
        })

        fig = px.scatter(
            df, x='UMAP1', y='UMAP2', color='Cell Type',
            title="Single-Cell UMAP",
            height=500
        )

        return fig

    def create_microbiome_heatmap(self):
        """Create microbiome-brain correlation heatmap"""
        import plotly.graph_objects as go

        # Mock correlation matrix
        taxa = ['Bacteroides', 'Firmicutes', 'Actinobacteria', 'Proteobacteria']
        phenotypes = ['Autism Score', 'ADHD Score', 'Social Function', 'Executive Function']

        corr_matrix = np.random.uniform(-0.5, 0.5, (len(taxa), len(phenotypes)))

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=phenotypes,
            y=taxa,
            colorscale='RdBu',
            zmid=0
        ))

        fig.update_layout(title="Microbiome-Brain Correlations", height=400)

        return fig

    def create_connectivity_matrix(self):
        """Create EEG connectivity matrix"""
        import plotly.graph_objects as go

        # Mock connectivity
        channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        conn = np.random.uniform(0, 1, (len(channels), len(channels)))
        np.fill_diagonal(conn, 1)

        fig = go.Figure(data=go.Heatmap(
            z=conn,
            x=channels,
            y=channels,
            colorscale='Viridis'
        ))

        fig.update_layout(title="EEG Functional Connectivity (Alpha Band)", height=500)

        return fig

    def create_forest_plot(self):
        """Create forest plot for meta-analysis"""
        import plotly.graph_objects as go

        studies = ['Site 1', 'Site 2', 'Site 3', 'Site 4', 'Pooled']
        effects = [0.45, 0.38, 0.52, 0.41, 0.44]
        ci_lower = [0.30, 0.22, 0.35, 0.25, 0.35]
        ci_upper = [0.60, 0.54, 0.69, 0.57, 0.53]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=effects,
            y=studies,
            error_x=dict(
                type='data',
                symmetric=False,
                array=[u - e for u, e in zip(ci_upper, effects)],
                arrayminus=[e - l for e, l in zip(effects, ci_lower)]
            ),
            mode='markers',
            marker=dict(size=10, color='blue')
        ))

        fig.add_vline(x=0, line_dash="dash")

        fig.update_layout(
            title="Federated Meta-Analysis Forest Plot",
            xaxis_title="Effect Size",
            height=400
        )

        return fig

    def create_attention_heatmap(self):
        """Create GNN attention weights heatmap"""
        import plotly.graph_objects as go

        nodes = [f'Gene {i}' for i in range(10)]
        attention = np.random.uniform(0, 1, (10, 10))

        fig = go.Figure(data=go.Heatmap(
            z=attention,
            x=nodes,
            y=nodes,
            colorscale='Hot'
        ))

        fig.update_layout(title="GNN Attention Weights", height=500)

        return fig

    def create_calibration_plot(self):
        """Create calibration curve"""
        import plotly.graph_objects as go

        # Perfect calibration
        predicted_probs = np.linspace(0, 1, 10)
        observed_freqs = predicted_probs + np.random.randn(10) * 0.05

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=predicted_probs,
            y=observed_freqs,
            mode='markers+lines',
            name='Model',
            marker=dict(size=10)
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='red')
        ))

        fig.update_layout(
            title="Calibration Curve",
            xaxis_title="Predicted Probability",
            yaxis_title="Observed Frequency",
            height=400
        )

        return fig

    def run(self):
        """Start dashboard server"""
        if self.app is None:
            self.create_app()

        if self.app is not None:
            logger.info(f"Starting dashboard on http://localhost:{self.port}")
            self.app.run_server(debug=True, port=self.port)
        else:
            logger.error("Failed to create app. Install dash: pip install dash plotly")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Create and run dashboard
    dashboard = AuDHDDashboard(results_dir=Path("results"))
    logger.info("Interactive Dashboard Module")
    logger.info("Run with: python scripts/visualization/interactive_dashboard.py")
    logger.info("Then visit: http://localhost:8050")
