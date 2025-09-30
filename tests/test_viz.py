"""Comprehensive tests for visualization dashboard"""
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from src.audhd_correlation.viz.embedding_plots import (
    create_embedding_plot,
    create_multi_overlay_plot,
    create_3d_embedding_plot,
    create_density_overlay,
    EmbeddingPlotConfig,
)

from src.audhd_correlation.viz.cluster_comparison import (
    create_cluster_comparison,
    create_violin_comparison,
    create_sankey_diagram,
    create_feature_importance_comparison,
    create_cluster_profile_heatmap,
    ClusterComparisonResult,
)

from src.audhd_correlation.viz.trajectories import (
    create_trajectory_plot,
    create_patient_timeline,
    create_transition_matrix,
    analyze_trajectories,
    TrajectoryResult,
)

from src.audhd_correlation.viz.heatmaps import (
    create_biomarker_heatmap,
    create_correlation_heatmap,
    create_clustermap,
    HeatmapConfig,
)

from src.audhd_correlation.viz.clinical_interface import (
    create_patient_card,
    create_risk_assessment,
    create_treatment_recommendations,
    generate_clinical_report,
    ClinicalDecisionSupport,
)

from src.audhd_correlation.viz.dashboard import (
    create_dashboard,
    load_sample_data,
)


@pytest.fixture
def embedding_2d():
    """Create 2D embedding"""
    np.random.seed(42)
    return np.random.randn(100, 2)


@pytest.fixture
def embedding_3d():
    """Create 3D embedding"""
    np.random.seed(42)
    return np.random.randn(100, 3)


@pytest.fixture
def labels():
    """Create cluster labels"""
    np.random.seed(42)
    return np.random.choice([0, 1, 2], size=100)


@pytest.fixture
def feature_data():
    """Create feature data"""
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    return pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'Feature_{i}' for i in range(n_features)]
    )


@pytest.fixture
def patient_ids():
    """Create patient IDs"""
    return [f'P{i:03d}' for i in range(100)]


@pytest.fixture
def temporal_data(embedding_2d, labels):
    """Create temporal data"""
    np.random.seed(42)
    n_timepoints = 3
    timepoints = ['T1', 'T2', 'T3']

    embeddings = {}
    labels_dict = {}

    for t in timepoints:
        # Add some noise to simulate temporal changes
        embeddings[t] = embedding_2d + np.random.randn(*embedding_2d.shape) * 0.2
        # Some patients change clusters
        labels_t = labels.copy()
        change_idx = np.random.choice(100, size=10, replace=False)
        labels_t[change_idx] = np.random.choice([0, 1, 2], size=10)
        labels_dict[t] = labels_t

    return embeddings, labels_dict


class TestEmbeddingPlots:
    """Tests for embedding plot functions"""

    def test_create_embedding_plot(self, embedding_2d, labels, patient_ids):
        """Test basic embedding plot"""
        fig = create_embedding_plot(
            embedding=embedding_2d,
            labels=labels,
            patient_ids=patient_ids,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text is not None

    def test_create_embedding_plot_with_overlay(self, embedding_2d, feature_data, patient_ids):
        """Test embedding plot with continuous overlay"""
        overlay_data = feature_data['Feature_0'].values

        fig = create_embedding_plot(
            embedding=embedding_2d,
            overlay_data=overlay_data,
            overlay_name='Feature_0',
            patient_ids=patient_ids,
            categorical=False,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_embedding_plot_with_config(self, embedding_2d, labels):
        """Test embedding plot with custom config"""
        config = EmbeddingPlotConfig(
            title="Custom Title",
            width=1200,
            height=900,
            marker_size=12,
        )

        fig = create_embedding_plot(
            embedding=embedding_2d,
            labels=labels,
            config=config,
        )

        assert fig.layout.title.text == "Custom Title"
        assert fig.layout.width == 1200
        assert fig.layout.height == 900

    def test_create_multi_overlay_plot(self, embedding_2d, labels, feature_data, patient_ids):
        """Test multi-overlay plot"""
        overlay_dict = {
            'Feature_0': feature_data['Feature_0'].values,
            'Feature_1': feature_data['Feature_1'].values,
        }

        fig = create_multi_overlay_plot(
            embedding=embedding_2d,
            labels=labels,
            overlay_dict=overlay_dict,
            patient_ids=patient_ids,
            n_cols=2,
        )

        assert isinstance(fig, go.Figure)
        # Should have traces for clusters + 2 overlays
        assert len(fig.data) > 2

    def test_create_3d_embedding_plot(self, embedding_3d, labels, patient_ids):
        """Test 3D embedding plot"""
        fig = create_3d_embedding_plot(
            embedding=embedding_3d,
            labels=labels,
            patient_ids=patient_ids,
        )

        assert isinstance(fig, go.Figure)
        # 3D scatter should have specific layout
        assert 'scene' in fig.layout

    def test_create_density_overlay(self, embedding_2d, labels):
        """Test density overlay plot"""
        fig = create_density_overlay(
            embedding=embedding_2d,
            labels=labels,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestClusterComparison:
    """Tests for cluster comparison functions"""

    def test_create_cluster_comparison(self, feature_data, labels):
        """Test cluster comparison"""
        results = create_cluster_comparison(
            data=feature_data,
            labels=labels,
            test_method='kruskal',
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, ClusterComparisonResult) for r in results)

        # Check result structure
        for result in results:
            assert result.feature_name is not None
            assert result.pvalue >= 0
            assert result.pvalue <= 1
            assert result.test_method == 'Kruskal-Wallis'

    def test_create_violin_comparison(self, feature_data, labels):
        """Test violin comparison plot"""
        features = feature_data.columns[:5].tolist()

        fig = create_violin_comparison(
            data=feature_data,
            labels=labels,
            features=features,
            n_cols=2,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_sankey_diagram(self, labels):
        """Test Sankey diagram"""
        # Simulate two timepoints with transitions
        labels_t2 = labels.copy()
        change_idx = np.random.choice(100, size=10, replace=False)
        labels_t2[change_idx] = (labels_t2[change_idx] + 1) % 3

        fig = create_sankey_diagram(
            labels_t1=labels,
            labels_t2=labels_t2,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_feature_importance_comparison(self, feature_data, labels):
        """Test feature importance visualization"""
        results = create_cluster_comparison(
            data=feature_data,
            labels=labels,
        )

        fig = create_feature_importance_comparison(
            comparison_results=results,
            n_top=10,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_cluster_profile_heatmap(self, feature_data, labels):
        """Test cluster profile heatmap"""
        results = create_cluster_comparison(
            data=feature_data,
            labels=labels,
        )

        fig = create_cluster_profile_heatmap(
            comparison_results=results,
            metric='mean',
            n_features=20,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestTrajectories:
    """Tests for trajectory visualization"""

    def test_create_trajectory_plot(self, temporal_data, patient_ids):
        """Test trajectory plot"""
        embeddings, labels_dict = temporal_data

        selected_patients = patient_ids[:5]

        fig = create_trajectory_plot(
            embeddings=embeddings,
            labels=labels_dict,
            patient_ids=patient_ids,
            selected_patients=selected_patients,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_analyze_trajectories(self, temporal_data, patient_ids):
        """Test trajectory analysis"""
        embeddings, labels_dict = temporal_data

        trajectories = analyze_trajectories(
            embeddings=embeddings,
            labels=labels_dict,
            patient_ids=patient_ids,
        )

        assert isinstance(trajectories, list)
        assert len(trajectories) == len(patient_ids)
        assert all(isinstance(t, TrajectoryResult) for t in trajectories)

        # Check trajectory structure
        for traj in trajectories:
            assert traj.patient_id in patient_ids
            assert len(traj.timepoints) == 3
            assert len(traj.clusters) == 3
            assert 0 <= traj.stability_score <= 1

    def test_create_patient_timeline(self, temporal_data, patient_ids):
        """Test patient timeline"""
        embeddings, labels_dict = temporal_data

        trajectories = analyze_trajectories(
            embeddings=embeddings,
            labels=labels_dict,
            patient_ids=patient_ids,
        )

        fig = create_patient_timeline(
            patient_id=patient_ids[0],
            trajectory=trajectories[0],
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_transition_matrix(self, temporal_data, patient_ids):
        """Test transition matrix"""
        _, labels_dict = temporal_data

        fig = create_transition_matrix(
            labels_dict=labels_dict,
            patient_ids=patient_ids,
            normalize=True,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestHeatmaps:
    """Tests for heatmap functions"""

    def test_create_biomarker_heatmap(self, feature_data, labels, patient_ids):
        """Test biomarker heatmap"""
        fig = create_biomarker_heatmap(
            data=feature_data,
            labels=labels,
            patient_ids=patient_ids,
            normalize='zscore',
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_biomarker_heatmap_no_clustering(self, feature_data, labels):
        """Test heatmap without clustering"""
        config = HeatmapConfig(
            cluster_rows=False,
            cluster_cols=False,
        )

        fig = create_biomarker_heatmap(
            data=feature_data,
            labels=labels,
            config=config,
        )

        assert isinstance(fig, go.Figure)

    def test_create_correlation_heatmap(self, feature_data):
        """Test correlation heatmap"""
        fig = create_correlation_heatmap(
            data=feature_data,
            method='pearson',
            cluster=True,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_clustermap(self, feature_data, labels):
        """Test clustermap"""
        fig = create_clustermap(
            data=feature_data,
            labels=labels,
            top_features=10,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestClinicalInterface:
    """Tests for clinical decision support"""

    def test_create_patient_card(self, feature_data, labels):
        """Test patient card"""
        patient_data = feature_data.iloc[0]
        cluster_id = labels[0]

        # Create cluster profile
        cluster_mask = labels == cluster_id
        cluster_profile = feature_data[cluster_mask].mean().to_dict()

        fig = create_patient_card(
            patient_id='P000',
            patient_data=patient_data,
            cluster_id=int(cluster_id),
            cluster_profile=cluster_profile,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_risk_assessment(self, feature_data, labels):
        """Test risk assessment"""
        patient_data = feature_data.iloc[0]
        cluster_mask = labels == labels[0]
        cluster_data = feature_data[cluster_mask]

        risk_factors = feature_data.columns[:5].tolist()

        fig = create_risk_assessment(
            patient_id='P000',
            patient_data=patient_data,
            cluster_data=cluster_data,
            risk_factors=risk_factors,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_treatment_recommendations(self):
        """Test treatment recommendations"""
        treatment_outcomes = {
            'Treatment_A': {
                'success_rate': 0.75,
                'response_time': 4.0,
                'side_effects': 0.2,
            },
            'Treatment_B': {
                'success_rate': 0.60,
                'response_time': 6.0,
                'side_effects': 0.4,
            },
        }

        fig = create_treatment_recommendations(
            patient_id='P000',
            cluster_id=0,
            treatment_outcomes=treatment_outcomes,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_generate_clinical_report(self, feature_data, labels):
        """Test clinical report generation"""
        patient_data = feature_data.iloc[0]
        cluster_id = labels[0]

        cluster_mask = labels == cluster_id
        cluster_profile = feature_data[cluster_mask].mean().to_dict()

        # Create reference ranges
        reference_ranges = {
            col: (0.0, 1.0) for col in feature_data.columns
        }

        treatment_outcomes = {
            'Treatment_A': {'success_rate': 0.75, 'response_time': 4.0, 'side_effects': 0.2},
        }

        report = generate_clinical_report(
            patient_id='P000',
            patient_data=patient_data,
            cluster_id=int(cluster_id),
            cluster_profile=cluster_profile,
            reference_ranges=reference_ranges,
            treatment_outcomes=treatment_outcomes,
        )

        assert isinstance(report, ClinicalDecisionSupport)
        assert report.patient_id == 'P000'
        assert report.cluster_id == int(cluster_id)
        assert report.risk_level in ['Low', 'Medium', 'High']
        assert isinstance(report.recommended_treatments, list)


class TestDashboard:
    """Tests for dashboard application"""

    def test_load_sample_data(self):
        """Test loading sample data"""
        embedding, labels, data, patient_ids = load_sample_data()

        assert embedding.shape == (200, 2)
        assert len(labels) == 200
        assert data.shape == (200, 20)
        assert len(patient_ids) == 200

    def test_create_dashboard(self, embedding_2d, labels, feature_data, patient_ids):
        """Test dashboard creation"""
        app = create_dashboard(
            embedding=embedding_2d,
            labels=labels,
            data=feature_data,
            patient_ids=patient_ids,
        )

        assert app is not None
        assert hasattr(app, 'layout')

    def test_create_dashboard_minimal(self):
        """Test dashboard with minimal data"""
        app = create_dashboard()

        assert app is not None
        assert hasattr(app, 'layout')


class TestEdgeCases:
    """Test edge cases"""

    def test_empty_data(self):
        """Test with empty data"""
        with pytest.raises((ValueError, IndexError)):
            create_embedding_plot(
                embedding=np.array([]),
                labels=np.array([]),
            )

    def test_single_cluster(self, embedding_2d):
        """Test with single cluster"""
        labels_single = np.zeros(100, dtype=int)

        fig = create_embedding_plot(
            embedding=embedding_2d,
            labels=labels_single,
        )

        assert isinstance(fig, go.Figure)

    def test_mismatched_dimensions(self, embedding_2d, labels):
        """Test with mismatched dimensions"""
        # Wrong number of labels
        labels_wrong = labels[:50]

        # Should handle gracefully or raise error
        try:
            fig = create_embedding_plot(
                embedding=embedding_2d,
                labels=labels_wrong,
            )
        except (ValueError, IndexError):
            pass  # Expected

    def test_3d_embedding_wrong_shape(self, embedding_2d):
        """Test 3D plot with 2D data"""
        with pytest.raises(ValueError, match="must be 3D"):
            create_3d_embedding_plot(
                embedding=embedding_2d,
                labels=np.zeros(100),
            )

    def test_sankey_mismatched_labels(self, labels):
        """Test Sankey with mismatched label arrays"""
        labels_wrong = labels[:50]

        with pytest.raises(ValueError, match="same length"):
            create_sankey_diagram(
                labels_t1=labels,
                labels_t2=labels_wrong,
            )

    def test_trajectory_insufficient_timepoints(self, embedding_2d, labels, patient_ids):
        """Test trajectory with insufficient timepoints"""
        embeddings = {'T1': embedding_2d}
        labels_dict = {'T1': labels}

        with pytest.raises(ValueError, match="at least 2 timepoints"):
            create_transition_matrix(
                labels_dict=labels_dict,
                patient_ids=patient_ids,
            )