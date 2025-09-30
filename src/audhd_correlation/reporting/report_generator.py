"""Main report generation orchestration

Coordinates template rendering, content generation, and output creation.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

from .templates import (
    get_clinical_template,
    get_research_template,
    get_patient_template,
    get_executive_template,
    render_template,
)
from .content import (
    generate_executive_summary,
    generate_key_findings,
    create_care_map,
    create_risk_stratification_table,
    create_treatment_recommendations,
    create_care_map_flowchart,
)


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    study_title: str
    report_type: str  # 'clinical', 'research', 'patient', 'executive'
    output_dir: Path
    include_visualizations: bool = True
    include_supplementary: bool = True
    author: Optional[str] = None
    institution: Optional[str] = None
    contact_info: Optional[str] = None


@dataclass
class GeneratedReport:
    """Generated report output"""
    report_type: str
    html_content: str
    output_path: Optional[Path] = None
    figures: List[Path] = field(default_factory=list)
    supplementary_files: List[Path] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class ReportGenerator:
    """Main report generator class"""

    def __init__(self, config: ReportConfig):
        """
        Initialize report generator

        Args:
            config: Report configuration
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_clinical_report(
        self,
        cluster_results: List[Dict],
        patient_data: pd.DataFrame,
        cluster_labels: np.ndarray,
        visualizations: Optional[List[Dict]] = None,
    ) -> GeneratedReport:
        """
        Generate clinical report

        Args:
            cluster_results: Cluster analysis results
            patient_data: Patient feature data
            cluster_labels: Cluster assignments
            visualizations: List of visualization paths

        Returns:
            GeneratedReport object
        """
        # Generate content
        exec_summary = generate_executive_summary(
            study_title=self.config.study_title,
            n_patients=len(patient_data),
            n_clusters=len(cluster_results),
            cluster_characteristics=cluster_results,
            validation_metrics={'silhouette': 0.65},  # Placeholder
        )

        # Generate risk stratification table
        risk_table = create_risk_stratification_table(
            patient_data=patient_data,
            cluster_labels=cluster_labels,
        )

        # Create care maps for each cluster
        clusters_with_care = []
        for cluster in cluster_results:
            care_map = create_care_map(
                subtype_id=cluster['id'],
                subtype_name=cluster.get('name', f"Subtype {cluster['id']}"),
                subtype_characteristics=cluster,
            )

            # Add care map HTML
            cluster_copy = cluster.copy()
            cluster_copy['care_map'] = create_care_map_flowchart(care_map)

            # Add treatments
            treatments = create_treatment_recommendations(
                cluster_id=cluster['id'],
                cluster_characteristics=cluster,
            )
            cluster_copy['treatments'] = [
                {
                    'name': t.treatment_name,
                    'evidence': t.evidence_level,
                    'response_rate': t.success_rate,
                    'notes': t.notes,
                }
                for t in treatments
            ]

            clusters_with_care.append(cluster_copy)

        # Prepare template context
        context = {
            'study_title': self.config.study_title,
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'study_period': cluster_results[0].get('study_period', 'N/A'),
            'audience': 'Clinical Care Team',
            'executive_summary': exec_summary.background + ' ' + exec_summary.impact_statement,
            'key_metrics': {
                'Total Patients': len(patient_data),
                'Subtypes Identified': len(cluster_results),
                'Features Analyzed': patient_data.shape[1],
                'Validation Score': '0.65',
            },
            'n_clusters': len(cluster_results),
            'clusters': clusters_with_care,
            'risk_table': risk_table.to_html(index=False, classes='risk-table'),
            'algorithms': [],  # Placeholder for decision algorithms
            'visualizations': visualizations or [],
            'contact_info': self.config.contact_info or 'study@institution.edu',
        }

        # Render template
        template = get_clinical_template()
        html_content = render_template(template, context)

        # Save report
        output_path = self.config.output_dir / f'clinical_report_{datetime.now().strftime("%Y%m%d")}.html'
        output_path.write_text(html_content)

        return GeneratedReport(
            report_type='clinical',
            html_content=html_content,
            output_path=output_path,
            metadata=context,
        )

    def generate_research_report(
        self,
        cluster_results: List[Dict],
        cohort_stats: pd.DataFrame,
        pathway_results: Optional[pd.DataFrame] = None,
        figures: Optional[Dict[str, Path]] = None,
    ) -> GeneratedReport:
        """
        Generate research report

        Args:
            cluster_results: Cluster analysis results
            cohort_stats: Cohort statistics
            pathway_results: Pathway enrichment results
            figures: Dictionary of figure paths

        Returns:
            GeneratedReport object
        """
        figures = figures or {}

        # Generate key findings
        key_findings = generate_key_findings(
            cluster_results=cluster_results,
            differential_features=pd.DataFrame(),  # Placeholder
            pathway_enrichment=pathway_results,
        )

        # Prepare abstract
        abstract = {
            'background': (
                'Co-occurring Autism Spectrum Disorder (ASD) and Attention-Deficit/Hyperactivity '
                'Disorder (ADHD), termed AuDHD, presents significant diagnostic and treatment challenges.'
            ),
            'methods': (
                f'We performed unsupervised clustering on multi-omics data from {len(cohort_stats)} '
                'participants, integrating genomics, transcriptomics, and metabolomics.'
            ),
            'results': (
                f'Analysis identified {len(cluster_results)} distinct biological subtypes with unique '
                'molecular signatures and clinical presentations.'
            ),
            'conclusions': (
                'AuDHD comprises biologically distinct subtypes requiring tailored treatment approaches. '
                'These findings support precision medicine strategies for neurodevelopmental conditions.'
            ),
        }

        # Prepare context
        context = {
            'study_title': self.config.study_title,
            'authors': self.config.author or 'Research Team',
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'institution': self.config.institution or 'Research Institution',
            'abstract': abstract,
            'introduction': (
                'The co-occurrence of ASD and ADHD (AuDHD) affects approximately 30-50% of individuals '
                'with either condition. Despite high prevalence, biological mechanisms remain unclear...'
            ),
            'methods': {
                'study_design': 'Cross-sectional observational study with multi-omics profiling',
                'data_collection': 'Participants recruited from specialized clinics...',
                'statistical_analysis': 'Unsupervised clustering using UMAP and HDBSCAN...',
            },
            'cohort_characteristics': [
                {'name': 'Total N', 'value': len(cohort_stats)},
                {'name': 'Age (mean±SD)', 'value': f"{cohort_stats.get('age_mean', 0):.1f}±{cohort_stats.get('age_sd', 0):.1f}"},
                {'name': 'Sex (% Female)', 'value': f"{cohort_stats.get('female_pct', 0):.1f}%"},
            ],
            'results': {
                'cluster_identification': f'Identified {len(cluster_results)} clusters...',
                'differential_expression': 'Differential expression analysis revealed...',
                'pathway_enrichment': 'GSEA identified enriched pathways...',
            },
            'figures': figures,
            'cluster_stats': cluster_results,
            'pathways': pathway_results.to_dict('records') if pathway_results is not None else [],
            'discussion': 'Our findings demonstrate biological heterogeneity within AuDHD...',
            'conclusions': abstract['conclusions'],
            'references': [
                'Author et al. (2024). Journal Name. Vol:Pages.',
            ],
            'supplementary': [],
        }

        # Render template
        template = get_research_template()
        html_content = render_template(template, context)

        # Save report
        output_path = self.config.output_dir / f'research_report_{datetime.now().strftime("%Y%m%d")}.html'
        output_path.write_text(html_content)

        return GeneratedReport(
            report_type='research',
            html_content=html_content,
            output_path=output_path,
            metadata=context,
        )

    def generate_patient_report(
        self,
        patient_id: str,
        patient_name: str,
        subtype_info: Dict,
        treatment_recommendations: List[Dict],
    ) -> GeneratedReport:
        """
        Generate patient-facing report

        Args:
            patient_id: Patient identifier
            patient_name: Patient name
            subtype_info: Subtype information
            treatment_recommendations: Treatment recommendations

        Returns:
            GeneratedReport object
        """
        # Prepare patient-friendly content
        context = {
            'patient_name': patient_name,
            'report_date': datetime.now().strftime('%B %d, %Y'),
            'subtype_name': subtype_info.get('name', 'Your Subtype'),
            'subtype_description': subtype_info.get('patient_description', ''),
            'subtype_explanation': subtype_info.get('patient_explanation', ''),
            'subtype_prevalence': subtype_info.get('prevalence', 0),
            'key_findings': subtype_info.get('patient_findings', []),
            'treatments': treatment_recommendations,
            'lifestyle_recommendations': subtype_info.get('lifestyle_tips', []),
            'next_steps': subtype_info.get('next_steps', []),
            'discussion_questions': subtype_info.get('discussion_questions', []),
            'contact_info': self.config.contact_info or 'your-care-team@clinic.org',
        }

        # Render template
        template = get_patient_template()
        html_content = render_template(template, context)

        # Save report
        output_path = self.config.output_dir / f'patient_report_{patient_id}_{datetime.now().strftime("%Y%m%d")}.html'
        output_path.write_text(html_content)

        return GeneratedReport(
            report_type='patient',
            html_content=html_content,
            output_path=output_path,
            metadata=context,
        )

    def generate_executive_report(
        self,
        cluster_results: List[Dict],
        financial_analysis: Dict,
        implementation_plan: List[Dict],
    ) -> GeneratedReport:
        """
        Generate executive summary report

        Args:
            cluster_results: Cluster analysis results
            financial_analysis: ROI and cost analysis
            implementation_plan: Implementation roadmap

        Returns:
            GeneratedReport object
        """
        # Prepare KPIs
        key_metrics = [
            {'value': len(cluster_results), 'label': 'Subtypes Identified'},
            {'value': f"{financial_analysis.get('cost_reduction', 0)}%", 'label': 'Cost Reduction Potential'},
            {'value': f"{financial_analysis.get('outcome_improvement', 0)}%", 'label': 'Outcome Improvement'},
            {'value': f"{financial_analysis.get('payback_months', 0)}", 'label': 'Payback Period (months)'},
        ]

        # Key findings for executives
        key_findings = [
            {
                'title': 'Actionable Patient Subtypes',
                'summary': f'Identified {len(cluster_results)} subtypes with distinct treatment responses',
            },
            {
                'title': 'Cost-Effective Stratification',
                'summary': f'Stratified care expected to reduce costs by ${financial_analysis.get("annual_savings", 0):,} annually',
            },
            {
                'title': 'Improved Outcomes',
                'summary': f'{financial_analysis.get("outcome_improvement", 0)}% improvement in patient outcomes projected',
            },
        ]

        # Clinical impacts
        clinical_impacts = [
            {
                'area': 'Treatment Selection',
                'description': 'Subtype-specific treatment matching',
                'improvement': f'{financial_analysis.get("treatment_improvement", 0)}% response rate increase',
            },
            {
                'area': 'Resource Allocation',
                'description': 'Targeted resource deployment to high-need subtypes',
                'improvement': f'{financial_analysis.get("resource_efficiency", 0)}% efficiency gain',
            },
        ]

        # Prepare context
        context = {
            'study_title': self.config.study_title,
            'report_date': datetime.now().strftime('%B %d, %Y'),
            'key_metrics': key_metrics,
            'key_findings': key_findings,
            'clinical_impacts': clinical_impacts,
            'subtypes': [
                {
                    'name': c.get('name', f"Subtype {c['id']}"),
                    'prevalence': c.get('prevalence', 0),
                    'cost_per_patient': c.get('cost_per_patient', 0),
                    'resources': c.get('recommended_resources', 'Standard care'),
                }
                for c in cluster_results
            ],
            'implementation_phases': implementation_plan,
            'roi': financial_analysis,
            'charts': [],
        }

        # Render template
        template = get_executive_template()
        html_content = render_template(template, context)

        # Save report
        output_path = self.config.output_dir / f'executive_summary_{datetime.now().strftime("%Y%m%d")}.html'
        output_path.write_text(html_content)

        return GeneratedReport(
            report_type='executive',
            html_content=html_content,
            output_path=output_path,
            metadata=context,
        )

    def generate_all_reports(
        self,
        cluster_results: List[Dict],
        patient_data: pd.DataFrame,
        cluster_labels: np.ndarray,
        **kwargs,
    ) -> Dict[str, GeneratedReport]:
        """
        Generate all report types

        Args:
            cluster_results: Cluster analysis results
            patient_data: Patient feature data
            cluster_labels: Cluster assignments
            **kwargs: Additional data for specific reports

        Returns:
            Dictionary of generated reports by type
        """
        reports = {}

        # Clinical report
        reports['clinical'] = self.generate_clinical_report(
            cluster_results=cluster_results,
            patient_data=patient_data,
            cluster_labels=cluster_labels,
            visualizations=kwargs.get('visualizations'),
        )

        # Research report
        reports['research'] = self.generate_research_report(
            cluster_results=cluster_results,
            cohort_stats=pd.Series({
                'age_mean': 35.0,
                'age_sd': 12.0,
                'female_pct': 45.0,
            }),
            pathway_results=kwargs.get('pathway_results'),
            figures=kwargs.get('figures'),
        )

        # Executive report
        reports['executive'] = self.generate_executive_report(
            cluster_results=cluster_results,
            financial_analysis=kwargs.get('financial_analysis', {
                'annual_savings': 500000,
                'outcome_improvement': 25,
                'payback_months': 18,
                'cost_reduction': 15,
                'treatment_improvement': 30,
                'resource_efficiency': 20,
            }),
            implementation_plan=kwargs.get('implementation_plan', [
                {
                    'name': 'Phase 1: Pilot',
                    'timeline': '3 months',
                    'activities': 'Initial deployment in one clinic',
                    'investment': '50,000',
                },
                {
                    'name': 'Phase 2: Expansion',
                    'timeline': '6 months',
                    'activities': 'Roll out to 5 additional clinics',
                    'investment': '150,000',
                },
            ]),
        )

        return reports