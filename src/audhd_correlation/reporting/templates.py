"""Jinja2 templates for different report audiences

Provides templates for clinical, research, patient, and executive reports.
"""
from typing import Dict, Optional
from jinja2 import Template


def get_clinical_template() -> Template:
    """
    Get Jinja2 template for clinical reports

    Returns:
        Jinja2 Template for clinicians
    """
    template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Report: {{ study_title }}</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .section {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { margin: 0; font-size: 28px; }
        h2 { color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        h3 { color: #764ba2; margin-top: 20px; }
        .key-finding {
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
        }
        .alert {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
        }
        .critical {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 15px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #667eea;
            color: white;
            font-weight: bold;
        }
        tr:hover { background: #f5f5f5; }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
        }
        .metric-label {
            font-weight: bold;
            color: #667eea;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            text-align: center;
            color: #666;
        }
        .visualization {
            margin: 20px 0;
            text-align: center;
        }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ study_title }}</h1>
        <p><strong>Report Date:</strong> {{ report_date }}</p>
        <p><strong>Study Period:</strong> {{ study_period }}</p>
        <p><strong>Prepared for:</strong> {{ audience }}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>{{ executive_summary }}</p>

        <h3>Key Metrics</h3>
        {% for metric_name, metric_value in key_metrics.items() %}
        <div class="metric">
            <span class="metric-label">{{ metric_name }}:</span> {{ metric_value }}
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Identified Subtypes</h2>
        <p>Analysis identified <strong>{{ n_clusters }}</strong> distinct AuDHD subtypes based on multi-omics profiling.</p>

        {% for cluster in clusters %}
        <h3>Subtype {{ cluster.id }}: {{ cluster.name }}</h3>
        <p><strong>Prevalence:</strong> {{ cluster.prevalence }}% ({{ cluster.n_patients }} patients)</p>
        <p><strong>Clinical Characteristics:</strong> {{ cluster.description }}</p>

        <div class="key-finding">
            <strong>Key Biomarkers:</strong>
            <ul>
            {% for biomarker in cluster.key_biomarkers %}
                <li>{{ biomarker.name }}: {{ biomarker.direction }} ({{ biomarker.significance }})</li>
            {% endfor %}
            </ul>
        </div>

        {% if cluster.risk_level == 'high' %}
        <div class="critical">
            <strong>⚠ High Risk Subtype:</strong> {{ cluster.risk_details }}
        </div>
        {% elif cluster.risk_level == 'medium' %}
        <div class="alert">
            <strong>⚠ Medium Risk Subtype:</strong> {{ cluster.risk_details }}
        </div>
        {% endif %}

        <h4>Clinical Care Map</h4>
        {{ cluster.care_map|safe }}

        <h4>Treatment Recommendations</h4>
        <table>
            <tr>
                <th>Treatment</th>
                <th>Evidence Level</th>
                <th>Expected Response</th>
                <th>Notes</th>
            </tr>
            {% for treatment in cluster.treatments %}
            <tr>
                <td>{{ treatment.name }}</td>
                <td>{{ treatment.evidence }}</td>
                <td>{{ treatment.response_rate }}%</td>
                <td>{{ treatment.notes }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Risk Stratification</h2>
        {{ risk_table|safe }}
    </div>

    <div class="section">
        <h2>Clinical Decision Support Algorithms</h2>
        {% for algorithm in algorithms %}
        <h3>{{ algorithm.name }}</h3>
        <p>{{ algorithm.description }}</p>
        <div class="visualization">
            {{ algorithm.flowchart|safe }}
        </div>
        {% endfor %}
    </div>

    {% if visualizations %}
    <div class="section">
        <h2>Supporting Visualizations</h2>
        {% for viz in visualizations %}
        <div class="visualization">
            <h3>{{ viz.title }}</h3>
            <img src="{{ viz.path }}" alt="{{ viz.title }}">
            <p><em>{{ viz.caption }}</em></p>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="footer">
        <p>This report was generated automatically by the AuDHD Multi-Omics Analysis System.</p>
        <p>For questions, contact: {{ contact_info }}</p>
    </div>
</body>
</html>
"""
    return Template(template_str)


def get_research_template() -> Template:
    """
    Get Jinja2 template for research reports

    Returns:
        Jinja2 Template for researchers
    """
    template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Research Report: {{ study_title }}</title>
    <style>
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.8;
            color: #000;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px;
        }
        h1 { font-size: 24px; text-align: center; margin-bottom: 30px; }
        h2 { font-size: 18px; margin-top: 30px; }
        h3 { font-size: 16px; margin-top: 20px; }
        .abstract {
            background: #f9f9f9;
            padding: 20px;
            border: 1px solid #ddd;
            margin: 30px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 12px;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border: 1px solid #000;
        }
        th { background: #e0e0e0; font-weight: bold; }
        .figure {
            margin: 30px 0;
            page-break-inside: avoid;
        }
        .figure img { max-width: 100%; }
        .figure-caption {
            font-size: 12px;
            margin-top: 10px;
        }
        .reference {
            font-size: 12px;
            margin-left: 20px;
        }
        .supplementary {
            background: #f0f0f0;
            padding: 15px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>{{ study_title }}</h1>
    <p style="text-align: center;">
        <strong>Authors:</strong> {{ authors }}<br>
        <strong>Date:</strong> {{ report_date }}<br>
        <strong>Institution:</strong> {{ institution }}
    </p>

    <div class="abstract">
        <h2>Abstract</h2>
        <p><strong>Background:</strong> {{ abstract.background }}</p>
        <p><strong>Methods:</strong> {{ abstract.methods }}</p>
        <p><strong>Results:</strong> {{ abstract.results }}</p>
        <p><strong>Conclusions:</strong> {{ abstract.conclusions }}</p>
    </div>

    <h2>Introduction</h2>
    <p>{{ introduction }}</p>

    <h2>Methods</h2>
    <h3>Study Design</h3>
    <p>{{ methods.study_design }}</p>

    <h3>Data Collection</h3>
    <p>{{ methods.data_collection }}</p>

    <h3>Statistical Analysis</h3>
    <p>{{ methods.statistical_analysis }}</p>

    <table>
        <caption>Table 1: Cohort Characteristics</caption>
        <tr>
            <th>Characteristic</th>
            <th>Value</th>
        </tr>
        {% for char in cohort_characteristics %}
        <tr>
            <td>{{ char.name }}</td>
            <td>{{ char.value }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Results</h2>

    <h3>Cluster Identification</h3>
    <p>{{ results.cluster_identification }}</p>

    <div class="figure">
        <img src="{{ figures.embedding }}" alt="Embedding plot">
        <p class="figure-caption"><strong>Figure 1:</strong> {{ figures.embedding_caption }}</p>
    </div>

    <h3>Cluster Characterization</h3>
    <table>
        <caption>Table 2: Cluster Statistics</caption>
        <tr>
            <th>Cluster</th>
            <th>N</th>
            <th>Age (mean±SD)</th>
            <th>Sex (% F)</th>
            <th>Key Features</th>
        </tr>
        {% for cluster in cluster_stats %}
        <tr>
            <td>{{ cluster.id }}</td>
            <td>{{ cluster.n }}</td>
            <td>{{ cluster.age_mean }} ± {{ cluster.age_sd }}</td>
            <td>{{ cluster.female_pct }}%</td>
            <td>{{ cluster.features }}</td>
        </tr>
        {% endfor %}
    </table>

    <h3>Differential Expression Analysis</h3>
    <p>{{ results.differential_expression }}</p>

    <div class="figure">
        <img src="{{ figures.heatmap }}" alt="Differential expression heatmap">
        <p class="figure-caption"><strong>Figure 2:</strong> {{ figures.heatmap_caption }}</p>
    </div>

    <h3>Pathway Enrichment</h3>
    <p>{{ results.pathway_enrichment }}</p>

    <table>
        <caption>Table 3: Top Enriched Pathways</caption>
        <tr>
            <th>Pathway</th>
            <th>Cluster</th>
            <th>NES</th>
            <th>FDR</th>
        </tr>
        {% for pathway in pathways %}
        <tr>
            <td>{{ pathway.name }}</td>
            <td>{{ pathway.cluster }}</td>
            <td>{{ pathway.nes }}</td>
            <td>{{ pathway.fdr }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Discussion</h2>
    <p>{{ discussion }}</p>

    <h2>Conclusions</h2>
    <p>{{ conclusions }}</p>

    <h2>References</h2>
    {% for ref in references %}
    <p class="reference">{{ loop.index }}. {{ ref }}</p>
    {% endfor %}

    <div class="supplementary">
        <h2>Supplementary Materials</h2>
        <p>Additional tables, figures, and methods are available in the supplementary materials.</p>
        <ul>
            {% for supp in supplementary %}
            <li><a href="{{ supp.url }}">{{ supp.title }}</a></li>
            {% endfor %}
        </ul>
    </div>
</body>
</html>
"""
    return Template(template_str)


def get_patient_template() -> Template:
    """
    Get Jinja2 template for patient-facing reports

    Returns:
        Jinja2 Template for patients
    """
    template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Your Personalized Report</title>
    <style>
        body {
            font-family: 'Helvetica', 'Arial', sans-serif;
            line-height: 1.8;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-size: 16px;
        }
        .header {
            background: #4CAF50;
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        h1 { font-size: 32px; margin: 0; }
        h2 {
            color: #4CAF50;
            font-size: 24px;
            margin-top: 30px;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h3 { color: #388E3C; font-size: 20px; }
        .section {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .info-box {
            background: #E8F5E9;
            border-left: 5px solid #4CAF50;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .tip-box {
            background: #FFF3E0;
            border-left: 5px solid #FF9800;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .simple-table {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .simple-table div {
            padding: 10px 0;
            border-bottom: 1px solid #ddd;
        }
        .simple-table div:last-child {
            border-bottom: none;
        }
        .label { font-weight: bold; color: #4CAF50; }
        ul { padding-left: 30px; }
        li { margin: 10px 0; }
        .footer {
            margin-top: 40px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 8px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Your Personalized AuDHD Report</h1>
        <p>{{ patient_name }}</p>
        <p>Report Date: {{ report_date }}</p>
    </div>

    <div class="section">
        <h2>About This Report</h2>
        <p>This report summarizes the findings from your multi-omics analysis. We've used advanced
        technology to understand your unique biological profile and how it relates to your AuDHD diagnosis.</p>

        <div class="info-box">
            <p><strong>What is multi-omics analysis?</strong></p>
            <p>We looked at multiple types of biological information - your genes, proteins, and metabolism -
            to get a complete picture of your health. This helps us understand you better and provide more
            personalized care.</p>
        </div>
    </div>

    <div class="section">
        <h2>Your Subtype: {{ subtype_name }}</h2>
        <p>{{ subtype_description }}</p>

        <h3>What This Means for You</h3>
        <p>{{ subtype_explanation }}</p>

        <div class="info-box">
            <h4>You're Not Alone</h4>
            <p>About {{ subtype_prevalence }}% of people in our study have a similar profile to yours.
            This means we have good information about what works well for people like you.</p>
        </div>
    </div>

    <div class="section">
        <h2>Your Key Findings</h2>
        {% for finding in key_findings %}
        <h3>{{ finding.title }}</h3>
        <p>{{ finding.explanation }}</p>

        {% if finding.action %}
        <div class="tip-box">
            <strong>What You Can Do:</strong><br>
            {{ finding.action }}
        </div>
        {% endif %}
        {% endfor %}
    </div>

    <div class="section">
        <h2>Personalized Recommendations</h2>

        <h3>Treatment Options</h3>
        <p>Based on your profile, the following treatments may be most effective:</p>

        {% for treatment in treatments %}
        <div class="simple-table">
            <div><span class="label">Treatment:</span> {{ treatment.name }}</div>
            <div><span class="label">How it helps:</span> {{ treatment.description }}</div>
            <div><span class="label">Success rate for your subtype:</span> {{ treatment.success_rate }}%</div>
            {% if treatment.side_effects %}
            <div><span class="label">Important to know:</span> {{ treatment.side_effects }}</div>
            {% endif %}
        </div>
        {% endfor %}

        <h3>Lifestyle Recommendations</h3>
        <ul>
        {% for recommendation in lifestyle_recommendations %}
            <li>{{ recommendation }}</li>
        {% endfor %}
        </ul>
    </div>

    <div class="section">
        <h2>Next Steps</h2>
        <ol>
            {% for step in next_steps %}
            <li><strong>{{ step.title }}:</strong> {{ step.description }}</li>
            {% endfor %}
        </ol>
    </div>

    <div class="section">
        <h2>Questions to Ask Your Doctor</h2>
        <ul>
        {% for question in discussion_questions %}
            <li>{{ question }}</li>
        {% endfor %}
        </ul>
    </div>

    <div class="footer">
        <p><strong>Need Help Understanding This Report?</strong></p>
        <p>Contact your care team: {{ contact_info }}</p>
        <p style="font-size: 12px; color: #666; margin-top: 20px;">
            This report is for informational purposes and should be discussed with your healthcare provider.
        </p>
    </div>
</body>
</html>
"""
    return Template(template_str)


def get_executive_template() -> Template:
    """
    Get Jinja2 template for executive summary reports

    Returns:
        Jinja2 Template for executives/administrators
    """
    template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Executive Summary: {{ study_title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        h1 { margin: 0; font-size: 36px; }
        h2 {
            color: #1e3c72;
            font-size: 24px;
            margin-top: 30px;
            border-bottom: 3px solid #2a5298;
            padding-bottom: 10px;
        }
        .kpi-container {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin: 30px 0;
        }
        .kpi {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            min-width: 200px;
            text-align: center;
            margin: 10px;
        }
        .kpi-value {
            font-size: 48px;
            font-weight: bold;
            color: #2a5298;
        }
        .kpi-label {
            font-size: 14px;
            color: #666;
            margin-top: 10px;
        }
        .highlight {
            background: #e8f4f8;
            border-left: 5px solid #2a5298;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .impact {
            background: #fff9e6;
            border-left: 5px solid #f39c12;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #2a5298;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) { background: #f8f9fa; }
        .chart {
            margin: 30px 0;
            text-align: center;
        }
        img { max-width: 100%; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Executive Summary</h1>
        <h2 style="color: white; border: none; margin-top: 10px;">{{ study_title }}</h2>
        <p>{{ report_date }}</p>
    </div>

    <div class="kpi-container">
        {% for kpi in key_metrics %}
        <div class="kpi">
            <div class="kpi-value">{{ kpi.value }}</div>
            <div class="kpi-label">{{ kpi.label }}</div>
        </div>
        {% endfor %}
    </div>

    <h2>Key Findings</h2>
    {% for finding in key_findings %}
    <div class="highlight">
        <h3 style="margin-top: 0;">{{ finding.title }}</h3>
        <p>{{ finding.summary }}</p>
    </div>
    {% endfor %}

    <h2>Clinical Impact</h2>
    {% for impact in clinical_impacts %}
    <div class="impact">
        <h3 style="margin-top: 0;">{{ impact.area }}</h3>
        <p>{{ impact.description }}</p>
        <p><strong>Expected Improvement:</strong> {{ impact.improvement }}</p>
    </div>
    {% endfor %}

    <h2>Resource Allocation</h2>
    <table>
        <tr>
            <th>Subtype</th>
            <th>Prevalence</th>
            <th>Cost per Patient</th>
            <th>Recommended Resources</th>
        </tr>
        {% for subtype in subtypes %}
        <tr>
            <td>{{ subtype.name }}</td>
            <td>{{ subtype.prevalence }}%</td>
            <td>${{ subtype.cost_per_patient }}</td>
            <td>{{ subtype.resources }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Implementation Roadmap</h2>
    <table>
        <tr>
            <th>Phase</th>
            <th>Timeline</th>
            <th>Activities</th>
            <th>Investment</th>
        </tr>
        {% for phase in implementation_phases %}
        <tr>
            <td><strong>{{ phase.name }}</strong></td>
            <td>{{ phase.timeline }}</td>
            <td>{{ phase.activities }}</td>
            <td>${{ phase.investment }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Return on Investment</h2>
    <div class="highlight">
        <p><strong>Projected Annual Savings:</strong> ${{ roi.annual_savings }}</p>
        <p><strong>Improved Patient Outcomes:</strong> {{ roi.outcome_improvement }}%</p>
        <p><strong>Payback Period:</strong> {{ roi.payback_period }} months</p>
    </div>

    {% if charts %}
    <h2>Supporting Data</h2>
    {% for chart in charts %}
    <div class="chart">
        <h3>{{ chart.title }}</h3>
        <img src="{{ chart.path }}" alt="{{ chart.title }}">
    </div>
    {% endfor %}
    {% endif %}
</body>
</html>
"""
    return Template(template_str)


def render_template(template: Template, context: Dict) -> str:
    """
    Render a Jinja2 template with context data

    Args:
        template: Jinja2 Template object
        context: Dictionary of template variables

    Returns:
        Rendered HTML string
    """
    return template.render(**context)