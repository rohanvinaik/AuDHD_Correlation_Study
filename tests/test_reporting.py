"""Tests for report generation"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from audhd_correlation.reporting import (
    ReportGenerator,
    ReportConfig,
    GeneratedReport,
    generate_executive_summary,
    generate_key_findings,
    create_care_map,
    create_risk_stratification_table,
    create_treatment_recommendations,
    generate_supplementary_materials,
    create_methods_section,
    create_data_dictionary,
    create_statistical_appendix,
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_cluster_results():
    """Sample cluster results"""
    return [
        {
            'id': 0,
            'name': 'High Executive Dysfunction',
            'n_patients': 50,
            'silhouette': 0.65,
            'within_ss': 125.3,
            'top_features': ['ADHD_score', 'executive_function', 'dopamine'],
            'prevalence': 0.25,
            'severity': 'moderate',
        },
        {
            'id': 1,
            'name': 'Sensory Processing Issues',
            'n_patients': 40,
            'silhouette': 0.70,
            'within_ss': 98.7,
            'top_features': ['sensory_sensitivity', 'ASD_score', 'glutamate'],
            'prevalence': 0.20,
            'severity': 'mild',
        },
        {
            'id': 2,
            'name': 'Combined High Severity',
            'n_patients': 30,
            'silhouette': 0.60,
            'within_ss': 156.2,
            'top_features': ['ADHD_score', 'ASD_score', 'anxiety_score'],
            'prevalence': 0.15,
            'severity': 'severe',
        },
    ]


@pytest.fixture
def sample_patient_data():
    """Sample patient data"""
    np.random.seed(42)
    n_patients = 120
    n_features = 50

    data = pd.DataFrame(
        np.random.randn(n_patients, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Add some named features
    data['age'] = np.random.randint(18, 65, n_patients)
    data['sex'] = np.random.choice(['M', 'F'], n_patients)
    data['ADHD_score'] = np.random.uniform(0, 100, n_patients)
    data['ASD_score'] = np.random.uniform(0, 100, n_patients)

    return data


@pytest.fixture
def sample_cluster_labels():
    """Sample cluster labels"""
    return np.array([0] * 50 + [1] * 40 + [2] * 30)


@pytest.fixture
def report_config(temp_output_dir):
    """Report configuration"""
    return ReportConfig(
        study_title='AuDHD Multi-Omics Analysis',
        report_type='clinical',
        output_dir=temp_output_dir,
        include_visualizations=True,
        include_supplementary=True,
        author='Test Author',
        institution='Test Institution',
        contact_info='test@example.com',
    )


# Content Generation Tests


class TestExecutiveSummary:
    """Test executive summary generation"""

    def test_generate_executive_summary(self, sample_cluster_results):
        """Test basic executive summary generation"""
        summary = generate_executive_summary(
            study_title='Test Study',
            n_patients=100,
            n_clusters=3,
            cluster_characteristics=sample_cluster_results,
            validation_metrics={'silhouette': 0.65},
        )

        assert summary.title == 'Test Study'
        assert 'multi-omics' in summary.background.lower()
        assert len(summary.key_findings) > 0
        assert 'precision medicine' in summary.impact_statement.lower()
        assert len(summary.recommendations) > 0

    def test_executive_summary_with_minimal_data(self):
        """Test with minimal data"""
        summary = generate_executive_summary(
            study_title='Minimal Study',
            n_patients=10,
            n_clusters=2,
            cluster_characteristics=[],
            validation_metrics={},
        )

        assert summary.title == 'Minimal Study'
        assert len(summary.key_findings) >= 0


class TestKeyFindings:
    """Test key findings generation"""

    def test_generate_key_findings(self, sample_cluster_results):
        """Test key findings generation"""
        findings = generate_key_findings(
            cluster_results=sample_cluster_results,
            differential_features=pd.DataFrame(),
            pathway_enrichment=None,
        )

        assert len(findings) > 0
        for finding in findings:
            assert 'title' in finding
            assert 'description' in finding
            assert 'significance' in finding

    def test_key_findings_with_pathways(self, sample_cluster_results):
        """Test with pathway data"""
        pathways = pd.DataFrame({
            'pathway': ['Dopamine signaling', 'Serotonin pathway'],
            'pvalue': [0.001, 0.005],
            'fdr': [0.01, 0.05],
        })

        findings = generate_key_findings(
            cluster_results=sample_cluster_results,
            differential_features=pd.DataFrame(),
            pathway_enrichment=pathways,
        )

        assert len(findings) > 0


class TestCareMap:
    """Test care map generation"""

    def test_create_care_map_mild(self):
        """Test care map for mild severity"""
        care_map = create_care_map(
            subtype_id=0,
            subtype_name='Mild Subtype',
            subtype_characteristics={'severity': 'mild'},
        )

        assert care_map.subtype_id == 0
        assert care_map.subtype_name == 'Mild Subtype'
        assert len(care_map.assessment_steps) > 0
        assert len(care_map.intervention_pathway) > 0
        assert len(care_map.monitoring_protocol) > 0
        assert len(care_map.escalation_criteria) > 0

    def test_create_care_map_moderate(self):
        """Test care map for moderate severity"""
        care_map = create_care_map(
            subtype_id=1,
            subtype_name='Moderate Subtype',
            subtype_characteristics={'severity': 'moderate'},
        )

        assert len(care_map.intervention_pathway) >= 4

    def test_create_care_map_severe(self):
        """Test care map for severe severity"""
        care_map = create_care_map(
            subtype_id=2,
            subtype_name='Severe Subtype',
            subtype_characteristics={'severity': 'severe'},
        )

        assert len(care_map.intervention_pathway) >= 5


class TestRiskStratification:
    """Test risk stratification"""

    def test_create_risk_stratification_table(self, sample_patient_data, sample_cluster_labels):
        """Test risk stratification table"""
        risk_table = create_risk_stratification_table(
            patient_data=sample_patient_data,
            cluster_labels=sample_cluster_labels,
        )

        assert isinstance(risk_table, pd.DataFrame)
        assert 'patient_id' in risk_table.columns
        assert 'cluster' in risk_table.columns
        assert 'risk_level' in risk_table.columns
        assert len(risk_table) == len(sample_patient_data)

    def test_risk_levels(self, sample_patient_data, sample_cluster_labels):
        """Test risk level categorization"""
        risk_table = create_risk_stratification_table(
            patient_data=sample_patient_data,
            cluster_labels=sample_cluster_labels,
        )

        risk_levels = risk_table['risk_level'].unique()
        assert all(level in ['Low', 'Medium', 'High'] for level in risk_levels)


class TestTreatmentRecommendations:
    """Test treatment recommendations"""

    def test_create_treatment_recommendations(self):
        """Test treatment recommendation generation"""
        treatments = create_treatment_recommendations(
            cluster_id=0,
            cluster_characteristics={'severity': 'moderate'},
        )

        assert len(treatments) > 0
        for treatment in treatments:
            assert hasattr(treatment, 'treatment_name')
            assert hasattr(treatment, 'evidence_level')
            assert hasattr(treatment, 'success_rate')
            assert 0 <= treatment.success_rate <= 1
            assert 0 <= treatment.side_effect_risk <= 1

    def test_treatment_recommendations_severity(self):
        """Test that severity affects recommendations"""
        mild_treatments = create_treatment_recommendations(
            cluster_id=0,
            cluster_characteristics={'severity': 'mild'},
        )

        severe_treatments = create_treatment_recommendations(
            cluster_id=1,
            cluster_characteristics={'severity': 'severe'},
        )

        assert len(mild_treatments) > 0
        assert len(severe_treatments) > 0


# Supplementary Materials Tests


class TestSupplementaryMaterials:
    """Test supplementary materials generation"""

    def test_generate_supplementary_materials(self, sample_cluster_results):
        """Test supplementary materials generation"""
        materials = generate_supplementary_materials(
            cluster_results=sample_cluster_results,
            feature_names=['age', 'ADHD_score', 'ASD_score'],
            methods_detail=None,
        )

        assert len(materials) == 3
        assert materials[0].title == 'Supplementary Methods'
        assert materials[1].title == 'Data Dictionary'
        assert materials[2].title == 'Statistical Appendix'

    def test_create_methods_section(self):
        """Test methods section creation"""
        methods = create_methods_section({
            'sample_collection': 'Custom collection protocol',
            'qc': 'Custom QC protocol',
        })

        assert isinstance(methods, str)
        assert 'Custom collection protocol' in methods
        assert 'Custom QC protocol' in methods

    def test_create_data_dictionary(self):
        """Test data dictionary creation"""
        feature_names = [
            'age', 'sex', 'bmi',
            'gene_expr_1', 'protein_abundance_1',
            'metabolite_1', 'snp_rs123',
        ]

        dictionary = create_data_dictionary(feature_names)

        assert isinstance(dictionary, str)
        assert '<table' in dictionary
        assert 'age' in dictionary
        assert 'Clinical' in dictionary or 'Genomics' in dictionary

    def test_create_statistical_appendix(self, sample_cluster_results):
        """Test statistical appendix creation"""
        appendix = create_statistical_appendix(sample_cluster_results)

        assert isinstance(appendix, str)
        assert 'Statistical' in appendix
        assert 'Kruskal-Wallis' in appendix


# Report Generator Tests


class TestReportGenerator:
    """Test main report generator"""

    def test_initialization(self, report_config):
        """Test generator initialization"""
        generator = ReportGenerator(report_config)

        assert generator.config == report_config
        assert report_config.output_dir.exists()

    def test_generate_clinical_report(
        self,
        report_config,
        sample_cluster_results,
        sample_patient_data,
        sample_cluster_labels,
    ):
        """Test clinical report generation"""
        generator = ReportGenerator(report_config)

        report = generator.generate_clinical_report(
            cluster_results=sample_cluster_results,
            patient_data=sample_patient_data,
            cluster_labels=sample_cluster_labels,
        )

        assert isinstance(report, GeneratedReport)
        assert report.report_type == 'clinical'
        assert len(report.html_content) > 0
        assert report.output_path.exists()
        assert report.output_path.suffix == '.html'

    def test_generate_research_report(
        self,
        report_config,
        sample_cluster_results,
    ):
        """Test research report generation"""
        generator = ReportGenerator(report_config)

        cohort_stats = pd.Series({
            'age_mean': 35.0,
            'age_sd': 12.0,
            'female_pct': 45.0,
        })

        report = generator.generate_research_report(
            cluster_results=sample_cluster_results,
            cohort_stats=cohort_stats,
        )

        assert isinstance(report, GeneratedReport)
        assert report.report_type == 'research'
        assert 'abstract' in report.html_content.lower()
        assert report.output_path.exists()

    def test_generate_patient_report(self, report_config):
        """Test patient report generation"""
        generator = ReportGenerator(report_config)

        subtype_info = {
            'name': 'Your Subtype',
            'patient_description': 'Description for patient',
            'patient_explanation': 'Explanation',
            'prevalence': 0.25,
        }

        report = generator.generate_patient_report(
            patient_id='P001',
            patient_name='John Doe',
            subtype_info=subtype_info,
            treatment_recommendations=[],
        )

        assert isinstance(report, GeneratedReport)
        assert report.report_type == 'patient'
        assert 'P001' in report.output_path.name
        assert report.output_path.exists()

    def test_generate_executive_report(
        self,
        report_config,
        sample_cluster_results,
    ):
        """Test executive report generation"""
        generator = ReportGenerator(report_config)

        financial_analysis = {
            'annual_savings': 500000,
            'outcome_improvement': 25,
            'payback_months': 18,
            'cost_reduction': 15,
            'treatment_improvement': 30,
            'resource_efficiency': 20,
        }

        implementation_plan = [
            {
                'name': 'Phase 1',
                'timeline': '3 months',
                'activities': 'Pilot',
                'investment': '50,000',
            }
        ]

        report = generator.generate_executive_report(
            cluster_results=sample_cluster_results,
            financial_analysis=financial_analysis,
            implementation_plan=implementation_plan,
        )

        assert isinstance(report, GeneratedReport)
        assert report.report_type == 'executive'
        assert report.output_path.exists()

    def test_generate_all_reports(
        self,
        report_config,
        sample_cluster_results,
        sample_patient_data,
        sample_cluster_labels,
    ):
        """Test generating all report types"""
        generator = ReportGenerator(report_config)

        reports = generator.generate_all_reports(
            cluster_results=sample_cluster_results,
            patient_data=sample_patient_data,
            cluster_labels=sample_cluster_labels,
        )

        assert 'clinical' in reports
        assert 'research' in reports
        assert 'executive' in reports
        assert all(isinstance(r, GeneratedReport) for r in reports.values())
        assert all(r.output_path.exists() for r in reports.values())


# Output Format Tests


class TestHTMLOutput:
    """Test HTML output generation"""

    def test_html_content_structure(
        self,
        report_config,
        sample_cluster_results,
        sample_patient_data,
        sample_cluster_labels,
    ):
        """Test HTML content structure"""
        generator = ReportGenerator(report_config)

        report = generator.generate_clinical_report(
            cluster_results=sample_cluster_results,
            patient_data=sample_patient_data,
            cluster_labels=sample_cluster_labels,
        )

        html = report.html_content

        # Check for essential HTML elements
        assert '<html' in html.lower()
        assert '<head' in html.lower()
        assert '<body' in html.lower()
        assert '<h1' in html.lower()
        assert '<table' in html.lower()

    def test_html_file_creation(
        self,
        report_config,
        sample_cluster_results,
        sample_patient_data,
        sample_cluster_labels,
    ):
        """Test HTML file is created"""
        generator = ReportGenerator(report_config)

        report = generator.generate_clinical_report(
            cluster_results=sample_cluster_results,
            patient_data=sample_patient_data,
            cluster_labels=sample_cluster_labels,
        )

        assert report.output_path.exists()
        assert report.output_path.is_file()
        assert report.output_path.stat().st_size > 0


# Edge Cases and Error Handling


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_cluster_results(self, report_config, sample_patient_data):
        """Test with empty cluster results"""
        generator = ReportGenerator(report_config)

        with pytest.raises(Exception):
            generator.generate_clinical_report(
                cluster_results=[],
                patient_data=sample_patient_data,
                cluster_labels=np.array([]),
            )

    def test_mismatched_data_labels(
        self,
        report_config,
        sample_cluster_results,
        sample_patient_data,
    ):
        """Test with mismatched data and labels"""
        generator = ReportGenerator(report_config)

        # Labels don't match data length
        wrong_labels = np.array([0, 1, 2])

        with pytest.raises(Exception):
            generator.generate_clinical_report(
                cluster_results=sample_cluster_results,
                patient_data=sample_patient_data,
                cluster_labels=wrong_labels,
            )

    def test_missing_output_directory(self, temp_output_dir):
        """Test with non-existent output directory"""
        config = ReportConfig(
            study_title='Test',
            report_type='clinical',
            output_dir=temp_output_dir / 'nonexistent',
        )

        # Should create directory
        generator = ReportGenerator(config)
        assert config.output_dir.exists()

    def test_special_characters_in_title(
        self,
        report_config,
        sample_cluster_results,
        sample_patient_data,
        sample_cluster_labels,
    ):
        """Test with special characters in study title"""
        report_config.study_title = 'Test <Study> & "Analysis"'
        generator = ReportGenerator(report_config)

        report = generator.generate_clinical_report(
            cluster_results=sample_cluster_results,
            patient_data=sample_patient_data,
            cluster_labels=sample_cluster_labels,
        )

        assert report.output_path.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])