"""Automated report generation for multi-omics analysis

Generates customized reports for different audiences with embedded visualizations.
"""

from .report_generator import (
    ReportGenerator,
    ReportConfig,
    GeneratedReport,
)

from .templates import (
    get_clinical_template,
    get_research_template,
    get_patient_template,
    get_executive_template,
)

from .content import (
    generate_executive_summary,
    generate_key_findings,
    create_care_map,
    create_risk_stratification_table,
    create_treatment_recommendations,
)

from .supplementary import (
    generate_supplementary_materials,
    create_methods_section,
    create_data_dictionary,
    create_statistical_appendix,
)

from .pdf_generator import (
    generate_pdf_report,
    PDFConfig,
)

from .html_generator import (
    generate_html_report,
    HTMLConfig,
)

from .extended_reporting import (
    generate_extended_clinical_report,
    create_clinical_decision_support_extended,
    ExtendedClinicalReport,
    ExtendedSubtypeProfile,
    ClinicalDecisionPath,
)

# Pipeline wrapper functions
def generate_comprehensive_report(
    clustering_results,
    validation_results,
    biological_results,
    visualization_results,
    output_path=None,
    **kwargs
):
    """Wrapper for comprehensive report generation"""
    generator = ReportGenerator(config=ReportConfig(**kwargs))

    # Compile all results
    results = {
        'clustering': clustering_results,
        'validation': validation_results,
        'biological': biological_results,
        'visualizations': visualization_results,
    }

    # Generate report
    report = generator.generate(results)

    if output_path:
        return generate_html_report(
            report=report,
            output_path=output_path,
        )

    return report

__all__ = [
    # Main generator
    'ReportGenerator',
    'ReportConfig',
    'GeneratedReport',
    # Templates
    'get_clinical_template',
    'get_research_template',
    'get_patient_template',
    'get_executive_template',
    # Content generation
    'generate_executive_summary',
    'generate_key_findings',
    'create_care_map',
    'create_risk_stratification_table',
    'create_treatment_recommendations',
    # Supplementary materials
    'generate_supplementary_materials',
    'create_methods_section',
    'create_data_dictionary',
    'create_statistical_appendix',
    # Output generators
    'generate_pdf_report',
    'PDFConfig',
    'generate_html_report',
    'HTMLConfig',
    # Extended reporting with multi-modal biomarkers
    'generate_extended_clinical_report',
    'create_clinical_decision_support_extended',
    'ExtendedClinicalReport',
    'ExtendedSubtypeProfile',
    'ClinicalDecisionPath',
    # Pipeline wrappers
    'generate_comprehensive_report',
]