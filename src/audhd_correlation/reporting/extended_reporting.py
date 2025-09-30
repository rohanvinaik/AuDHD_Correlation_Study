"""
Extended clinical reporting with multi-modal biomarker panels.

Generates comprehensive reports incorporating:
- Autonomic nervous system profiling
- Circadian rhythm assessment
- Environmental exposure burden
- Sensory and interoceptive profiling
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
from jinja2 import Template

logger = logging.getLogger(__name__)


@dataclass
class ExtendedSubtypeProfile:
    """Profile for a subtype with extended biomarkers"""
    subtype_id: int
    name: str
    core_features: Dict[str, Any]
    autonomic_features: Dict[str, float]
    circadian_features: Dict[str, float]
    environmental_features: Dict[str, float]
    sensory_features: Dict[str, float]
    clinical_characteristics: Dict[str, Any]
    biomarkers: Dict[str, Any]
    treatment_implications: List[str]
    sample_size: int
    prevalence: float


@dataclass
class ClinicalDecisionPath:
    """Decision path for clinical assessment"""
    initial_assessment: Dict[str, List[str]]
    conditional_pathways: Dict[str, Dict[str, Any]]
    monitoring_schedule: Dict[str, List[str]]


@dataclass
class ExtendedClinicalReport:
    """Complete extended clinical report"""
    report_id: str
    generated_date: datetime
    n_samples: int
    subtype_profiles: List[ExtendedSubtypeProfile]
    risk_stratification: pd.DataFrame
    intervention_protocols: Dict[str, Any]
    monitoring_plan: Dict[str, Any]
    visualization_paths: List[str]
    html_content: str
    metadata: Dict[str, Any]


def generate_extended_clinical_report(
    results: Dict[str, Any],
    cluster_labels: np.ndarray,
    extended_features: pd.DataFrame,
    clinical_data: pd.DataFrame,
    output_path: Optional[str] = None
) -> ExtendedClinicalReport:
    """
    Generate comprehensive clinical report with extended biomarker panels.

    Parameters
    ----------
    results : dict
        Analysis results including clustering, causal, and biological findings
    cluster_labels : np.ndarray
        Cluster assignment for each sample
    extended_features : pd.DataFrame
        Multi-modal feature data (autonomic, circadian, environmental, sensory)
    clinical_data : pd.DataFrame
        Clinical characteristics and outcomes
    output_path : str, optional
        Path to save HTML report

    Returns
    -------
    ExtendedClinicalReport
        Complete report object with all sections

    Examples
    --------
    >>> report = generate_extended_clinical_report(
    ...     results=analysis_results,
    ...     cluster_labels=clusters,
    ...     extended_features=features_df,
    ...     clinical_data=clinical_df
    ... )
    >>> print(report.html_content)
    """
    logger.info("Generating extended clinical report")

    n_clusters = len(np.unique(cluster_labels))
    n_samples = len(cluster_labels)

    # Generate subtype profiles with extended features
    subtype_profiles = []
    for cluster_id in range(n_clusters):
        profile = _generate_subtype_profile(
            cluster_id, cluster_labels, extended_features, clinical_data
        )
        subtype_profiles.append(profile)

    # Create risk stratification
    risk_stratification = _create_risk_stratification(
        subtype_profiles, extended_features, clinical_data
    )

    # Generate intervention protocols
    intervention_protocols = _create_intervention_protocols(subtype_profiles)

    # Create monitoring plan
    monitoring_plan = _create_monitoring_plan(subtype_profiles)

    # Render HTML report
    html_content = _render_extended_report_html(
        n_samples=n_samples,
        subtype_profiles=subtype_profiles,
        risk_stratification=risk_stratification,
        intervention_protocols=intervention_protocols,
        monitoring_plan=monitoring_plan,
        results=results
    )

    # Create report object
    report = ExtendedClinicalReport(
        report_id=f"extended_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        generated_date=datetime.now(),
        n_samples=n_samples,
        subtype_profiles=subtype_profiles,
        risk_stratification=risk_stratification,
        intervention_protocols=intervention_protocols,
        monitoring_plan=monitoring_plan,
        visualization_paths=[],
        html_content=html_content,
        metadata={
            'n_clusters': n_clusters,
            'feature_domains': list(extended_features.columns),
            'clinical_variables': list(clinical_data.columns)
        }
    )

    # Save if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(html_content)
        logger.info(f"Extended clinical report saved to {output_path}")

    return report


def _generate_subtype_profile(
    cluster_id: int,
    cluster_labels: np.ndarray,
    extended_features: pd.DataFrame,
    clinical_data: pd.DataFrame
) -> ExtendedSubtypeProfile:
    """Generate comprehensive profile for a single subtype"""
    mask = cluster_labels == cluster_id

    # Autonomic features
    autonomic_cols = [c for c in extended_features.columns
                     if any(x in c.lower() for x in ['hrv', 'heart', 'scl', 'eda', 'rmssd', 'sdnn'])]
    autonomic_features = {}
    if autonomic_cols:
        for col in autonomic_cols:
            autonomic_features[col] = float(extended_features.loc[mask, col].mean())

    # Circadian features
    circadian_cols = [c for c in extended_features.columns
                     if any(x in c.lower() for x in ['circadian', 'melatonin', 'sleep', 'dlmo', 'rhythm'])]
    circadian_features = {}
    if circadian_cols:
        for col in circadian_cols:
            circadian_features[col] = float(extended_features.loc[mask, col].mean())

    # Environmental features
    env_cols = [c for c in extended_features.columns
               if any(x in c.lower() for x in ['lead', 'mercury', 'pm25', 'pollut', 'phthalate', 'bpa'])]
    environmental_features = {}
    if env_cols:
        for col in env_cols:
            environmental_features[col] = float(extended_features.loc[mask, col].mean())

    # Sensory features
    sensory_cols = [c for c in extended_features.columns
                   if any(x in c.lower() for x in ['sensory', 'interoception', 'tactile', 'auditory'])]
    sensory_features = {}
    if sensory_cols:
        for col in sensory_cols:
            sensory_features[col] = float(extended_features.loc[mask, col].mean())

    # Characterize subtype based on dominant features
    subtype_name, core_features, treatment_implications = _characterize_subtype(
        autonomic_features, circadian_features, environmental_features, sensory_features
    )

    # Clinical characteristics
    clinical_characteristics = {}
    if 'anxiety' in clinical_data.columns:
        clinical_characteristics['anxiety_percent'] = float((clinical_data.loc[mask, 'anxiety'] > 0).mean() * 100)
    if 'ADHD_symptoms' in clinical_data.columns:
        clinical_characteristics['adhd_severity'] = float(clinical_data.loc[mask, 'ADHD_symptoms'].mean())
    if 'ASD_symptoms' in clinical_data.columns:
        clinical_characteristics['asd_severity'] = float(clinical_data.loc[mask, 'ASD_symptoms'].mean())

    # Biomarkers
    biomarkers = {
        'autonomic': autonomic_features,
        'circadian': circadian_features,
        'environmental': environmental_features,
        'sensory': sensory_features
    }

    return ExtendedSubtypeProfile(
        subtype_id=cluster_id,
        name=subtype_name,
        core_features=core_features,
        autonomic_features=autonomic_features,
        circadian_features=circadian_features,
        environmental_features=environmental_features,
        sensory_features=sensory_features,
        clinical_characteristics=clinical_characteristics,
        biomarkers=biomarkers,
        treatment_implications=treatment_implications,
        sample_size=int(mask.sum()),
        prevalence=float(mask.mean())
    )


def _characterize_subtype(
    autonomic_features: Dict[str, float],
    circadian_features: Dict[str, float],
    environmental_features: Dict[str, float],
    sensory_features: Dict[str, float]
) -> tuple:
    """Characterize subtype based on dominant feature patterns"""

    # Score each domain (higher = more dysregulated)
    autonomic_score = 0
    if autonomic_features:
        # Low HRV indicates dysregulation
        hrv_features = [k for k in autonomic_features.keys() if 'sdnn' in k.lower() or 'rmssd' in k.lower()]
        if hrv_features:
            autonomic_score = -np.mean([autonomic_features[k] for k in hrv_features])  # Negative because low is bad

    circadian_score = 0
    if circadian_features:
        # Delayed phase or irregular sleep indicates disruption
        phase_features = [k for k in circadian_features.keys() if 'dlmo' in k.lower() or 'phase' in k.lower()]
        if phase_features:
            circadian_score = np.mean([abs(circadian_features[k]) for k in phase_features])

    environmental_score = 0
    if environmental_features:
        # High toxicant levels indicate burden
        environmental_score = np.mean(list(environmental_features.values()))

    sensory_score = 0
    if sensory_features:
        # Extreme values (high or low) indicate dysfunction
        sensory_score = np.mean([abs(v) for v in sensory_features.values()])

    # Determine dominant pattern
    scores = {
        'autonomic': autonomic_score,
        'circadian': circadian_score,
        'environmental': environmental_score,
        'sensory': sensory_score
    }

    dominant_domain = max(scores, key=scores.get)

    # Define subtype characteristics
    if dominant_domain == 'autonomic':
        name = "Autonomic Dysregulation Pattern"
        core_features = {
            'primary': 'Low HRV, elevated sympathetic tone',
            'secondary': 'High anxiety, orthostatic symptoms'
        }
        treatment_implications = [
            "Consider beta-blockers for autonomic symptoms",
            "Evaluate for POTS/dysautonomia",
            "Heart rate variability biofeedback",
            "Stress management and relaxation training"
        ]

    elif dominant_domain == 'circadian':
        name = "Circadian Disruption Pattern"
        core_features = {
            'primary': 'Delayed sleep phase, irregular rhythms',
            'secondary': 'Evening preference, mood variability'
        }
        treatment_implications = [
            "Morning bright light therapy (10,000 lux)",
            "Low-dose melatonin (0.5-3mg) timed 5-7h before DLMO",
            "Scheduled sleep-wake protocol with consistent timing",
            "Avoid evening blue light exposure"
        ]

    elif dominant_domain == 'environmental':
        name = "Environmental Burden Pattern"
        core_features = {
            'primary': 'Elevated toxicant exposure (multiple)',
            'secondary': 'Potential regression history, GI issues'
        }
        treatment_implications = [
            "Environmental assessment and source identification",
            "Removal/reduction of ongoing exposures",
            "Nutritional support for detoxification pathways",
            "Chelation evaluation for significant metal burden",
            "Regular monitoring of exposure biomarkers"
        ]

    else:  # sensory
        name = "Sensory-Interoceptive Dysfunction"
        core_features = {
            'primary': 'Poor interoceptive awareness, sensory sensitivity',
            'secondary': 'Emotional dysregulation, body disconnection'
        }
        treatment_implications = [
            "Interoceptive training protocols",
            "Occupational therapy for sensory integration",
            "Mindfulness-based interventions",
            "Sensory diet and environmental modifications",
            "Weighted blankets, compression garments"
        ]

    return name, core_features, treatment_implications


def _create_risk_stratification(
    subtype_profiles: List[ExtendedSubtypeProfile],
    extended_features: pd.DataFrame,
    clinical_data: pd.DataFrame
) -> pd.DataFrame:
    """Create risk stratification table with extended biomarkers"""

    risk_indicators = []

    # Autonomic risk indicators
    if 'HRV_SDNN' in extended_features.columns:
        sdnn_low = (extended_features['HRV_SDNN'] < 20).sum()
        risk_indicators.append({
            'Domain': 'Autonomic',
            'Indicator': 'HRV SDNN < 20ms',
            'N_at_Risk': sdnn_low,
            'Risk_Level': 'High',
            'Associated_Outcomes': 'Anxiety, cardiac issues, poor stress response'
        })

    # Circadian risk indicators
    if any('dlmo' in c.lower() for c in extended_features.columns):
        dlmo_col = [c for c in extended_features.columns if 'dlmo' in c.lower()][0]
        dlmo_delayed = (extended_features[dlmo_col] > 23).sum()  # After 11pm
        risk_indicators.append({
            'Domain': 'Circadian',
            'Indicator': 'DLMO > 11pm',
            'N_at_Risk': dlmo_delayed,
            'Risk_Level': 'Moderate',
            'Associated_Outcomes': 'Mood disorder, school problems, sleep deprivation'
        })

    # Environmental risk indicators
    if 'lead_level' in extended_features.columns:
        lead_high = (extended_features['lead_level'] > 5).sum()  # > 5 μg/dL
        risk_indicators.append({
            'Domain': 'Environmental',
            'Indicator': 'Lead > 5 μg/dL',
            'N_at_Risk': lead_high,
            'Risk_Level': 'High',
            'Associated_Outcomes': 'Cognitive impairment, behavioral problems'
        })

    # Sensory risk indicators
    if any('interoception' in c.lower() for c in extended_features.columns):
        intero_col = [c for c in extended_features.columns if 'interoception' in c.lower()][0]
        intero_low = (extended_features[intero_col] < 0.4).sum()
        risk_indicators.append({
            'Domain': 'Sensory',
            'Indicator': 'Interoceptive accuracy < 0.4',
            'N_at_Risk': intero_low,
            'Risk_Level': 'Moderate',
            'Associated_Outcomes': 'Emotional dysregulation, anxiety, alexithymia'
        })

    return pd.DataFrame(risk_indicators)


def _create_intervention_protocols(
    subtype_profiles: List[ExtendedSubtypeProfile]
) -> Dict[str, Any]:
    """Create personalized intervention protocols for each subtype"""

    protocols = {}

    for profile in subtype_profiles:
        protocol = {
            'subtype': profile.name,
            'immediate_actions': profile.treatment_implications[:2],
            'medium_term': profile.treatment_implications[2:] if len(profile.treatment_implications) > 2 else [],
            'monitoring_frequency': 'Monthly for first 3 months, then quarterly',
            'success_metrics': _define_success_metrics(profile)
        }
        protocols[f"subtype_{profile.subtype_id}"] = protocol

    return protocols


def _define_success_metrics(profile: ExtendedSubtypeProfile) -> List[str]:
    """Define success metrics based on subtype characteristics"""

    metrics = []

    if 'autonomic' in profile.name.lower():
        metrics.extend([
            "HRV SDNN increase > 10ms",
            "Reduction in orthostatic symptoms",
            "Anxiety score decrease > 25%"
        ])

    if 'circadian' in profile.name.lower():
        metrics.extend([
            "DLMO advancement by 1-2 hours",
            "Sleep onset latency < 30 minutes",
            "Daytime alertness improvement"
        ])

    if 'environmental' in profile.name.lower():
        metrics.extend([
            "50% reduction in primary toxicant levels",
            "Symptom severity decrease > 30%",
            "No new exposures identified"
        ])

    if 'sensory' in profile.name.lower():
        metrics.extend([
            "Interoceptive accuracy increase > 0.15",
            "Sensory symptom reduction > 40%",
            "Improved emotional regulation scores"
        ])

    return metrics


def _create_monitoring_plan(
    subtype_profiles: List[ExtendedSubtypeProfile]
) -> Dict[str, Any]:
    """Create longitudinal monitoring plan"""

    plan = {
        'quarterly': {
            'all_subtypes': [
                'Clinical symptom scales (ADHD-RS, ADOS-2)',
                'Functional assessment (school, social)',
                'Adverse events screening'
            ],
            'autonomic_subtype': [
                'HRV spot check (5-minute recording)',
                'Orthostatic vital signs',
                'Anxiety/stress questionnaire'
            ],
            'circadian_subtype': [
                'Sleep diary (2 weeks)',
                'Morning-evening questionnaire',
                'School/work performance metrics'
            ],
            'environmental_subtype': [
                'Exposure assessment questionnaire',
                'Symptom diary review',
                'Home environment checklist'
            ],
            'sensory_subtype': [
                'Sensory profile questionnaire',
                'Interoceptive awareness scale',
                'Emotional regulation assessment'
            ]
        },
        'annual': {
            'all_subtypes': [
                'Comprehensive clinical assessment',
                'Neurocognitive battery',
                'Quality of life measures',
                'Growth and development tracking'
            ],
            'biomarker_panels': {
                'autonomic': '24-hour Holter, tilt table, cortisol rhythm',
                'circadian': '2-week actigraphy, DLMO, core body temperature',
                'environmental': 'Hair heavy metals, urinary organics, nutrient minerals',
                'sensory': 'Heartbeat perception task, sensory gating EEG'
            }
        },
        'as_needed': {
            'symptom_exacerbation': 'Accelerated assessment schedule',
            'new_interventions': 'Baseline + 1 month + 3 month assessment',
            'developmental_transitions': 'Pre/post transition comprehensive evaluation'
        }
    }

    return plan


def _render_extended_report_html(
    n_samples: int,
    subtype_profiles: List[ExtendedSubtypeProfile],
    risk_stratification: pd.DataFrame,
    intervention_protocols: Dict[str, Any],
    monitoring_plan: Dict[str, Any],
    results: Dict[str, Any]
) -> str:
    """Render HTML report with extended features"""

    template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>AuDHD Extended Biomarker Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; margin-top: 30px; }
        h3 { color: #7f8c8d; margin-top: 20px; }
        .subtype { background: #ecf0f1; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .subtype-title { color: #2980b9; font-weight: bold; font-size: 1.3em; }
        .feature-list { margin-left: 20px; }
        .high-risk { color: #e74c3c; font-weight: bold; }
        .moderate-risk { color: #f39c12; font-weight: bold; }
        .low-risk { color: #27ae60; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #bdc3c7; padding: 12px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #ecf0f1; }
        .treatment-box { background: #d5f4e6; padding: 15px; margin: 15px 0; border-left: 4px solid #27ae60; }
        .warning-box { background: #fadbd8; padding: 15px; margin: 15px 0; border-left: 4px solid #e74c3c; }
        .info-box { background: #d6eaf8; padding: 15px; margin: 15px 0; border-left: 4px solid #3498db; }
        ul { line-height: 1.8; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; padding: 10px 15px;
                 background: #3498db; color: white; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🧬 AuDHD Subtype Analysis Report - Extended Biomarker Panel</h1>

    <div class="info-box">
        <strong>Report Generated:</strong> {{ generation_date }}<br>
        <strong>Analysis Sample:</strong> {{ n_samples }} participants<br>
        <strong>Number of Subtypes:</strong> {{ n_subtypes }}
    </div>

    <h2>Executive Summary</h2>
    <p>This analysis incorporates comprehensive multi-modal assessment including:</p>
    <ul>
        <li><strong>Standard multi-omics:</strong> Genetics, metabolomics, clinical phenotyping</li>
        <li><strong>Autonomic profiling:</strong> Heart rate variability, sympathetic/parasympathetic balance</li>
        <li><strong>Circadian assessment:</strong> Melatonin rhythm, sleep-wake patterns, chronotype</li>
        <li><strong>Environmental burden:</strong> Heavy metals, organic pollutants, air quality</li>
        <li><strong>Sensory-interoceptive:</strong> Sensory processing, interoceptive awareness, body perception</li>
    </ul>

    <h2>Identified Subtypes with Extended Features</h2>

    {% for profile in subtype_profiles %}
    <div class="subtype">
        <div class="subtype-title">Subtype {{ profile.subtype_id + 1 }}: {{ profile.name }}</div>
        <p><strong>Sample size:</strong> {{ profile.sample_size }} ({{ "%.1f"|format(profile.prevalence * 100) }}%)</p>

        <h3>Core Features</h3>
        <div class="feature-list">
            <p><strong>Primary:</strong> {{ profile.core_features.primary }}</p>
            <p><strong>Secondary:</strong> {{ profile.core_features.secondary }}</p>
        </div>

        <h3>Biomarker Profile</h3>
        <table>
            <tr><th>Domain</th><th>Key Findings</th></tr>
            {% if profile.autonomic_features %}
            <tr>
                <td><strong>Autonomic</strong></td>
                <td>
                    {% for key, value in profile.autonomic_features.items() %}
                        {{ key }}: {{ "%.2f"|format(value) }}{% if not loop.last %}, {% endif %}
                    {% endfor %}
                </td>
            </tr>
            {% endif %}
            {% if profile.circadian_features %}
            <tr>
                <td><strong>Circadian</strong></td>
                <td>
                    {% for key, value in profile.circadian_features.items() %}
                        {{ key }}: {{ "%.2f"|format(value) }}{% if not loop.last %}, {% endif %}
                    {% endfor %}
                </td>
            </tr>
            {% endif %}
            {% if profile.environmental_features %}
            <tr>
                <td><strong>Environmental</strong></td>
                <td>
                    {% for key, value in profile.environmental_features.items() %}
                        {{ key }}: {{ "%.2f"|format(value) }}{% if not loop.last %}, {% endif %}
                    {% endfor %}
                </td>
            </tr>
            {% endif %}
            {% if profile.sensory_features %}
            <tr>
                <td><strong>Sensory</strong></td>
                <td>
                    {% for key, value in profile.sensory_features.items() %}
                        {{ key }}: {{ "%.2f"|format(value) }}{% if not loop.last %}, {% endif %}
                    {% endfor %}
                </td>
            </tr>
            {% endif %}
        </table>

        <h3>Clinical Characteristics</h3>
        <ul>
            {% for key, value in profile.clinical_characteristics.items() %}
            <li>{{ key.replace('_', ' ').title() }}: {{ "%.1f"|format(value) }}{% if 'percent' in key %}%{% endif %}</li>
            {% endfor %}
        </ul>

        <div class="treatment-box">
            <h3>Treatment Implications</h3>
            <ul>
                {% for implication in profile.treatment_implications %}
                <li>{{ implication }}</li>
                {% endfor %}
            </ul>
        </div>
    </div>
    {% endfor %}

    <h2>Risk Stratification with Extended Biomarkers</h2>

    <div class="warning-box">
        <strong>High-Risk Indicators Identified</strong><br>
        These biomarker patterns warrant immediate clinical attention and targeted intervention.
    </div>

    <table>
        <tr>
            <th>Domain</th>
            <th>Risk Indicator</th>
            <th>N at Risk</th>
            <th>Risk Level</th>
            <th>Associated Outcomes</th>
        </tr>
        {% for _, row in risk_stratification.iterrows() %}
        <tr>
            <td>{{ row.Domain }}</td>
            <td><strong>{{ row.Indicator }}</strong></td>
            <td>{{ row.N_at_Risk }}</td>
            <td class="{{ row.Risk_Level.lower() }}-risk">{{ row.Risk_Level }}</td>
            <td>{{ row.Associated_Outcomes }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Comprehensive Biomarker Testing Panels by Subtype</h2>

    <h3>Recommended Assessment Protocol</h3>
    <table>
        <tr>
            <th>Subtype</th>
            <th>Primary Tests</th>
            <th>Secondary Tests</th>
            <th>Monitoring Frequency</th>
        </tr>
        <tr>
            <td><strong>Autonomic</strong></td>
            <td>24h Holter, Tilt table, Cortisol rhythm</td>
            <td>Inflammatory panel, Autoantibodies</td>
            <td>HRV monthly, Full panel annually</td>
        </tr>
        <tr>
            <td><strong>Circadian</strong></td>
            <td>2-week actigraphy, DLMO, Cortisol rhythm</td>
            <td>Core body temp, Polysomnography</td>
            <td>Sleep diary ongoing, Full panel bi-annually</td>
        </tr>
        <tr>
            <td><strong>Environmental</strong></td>
            <td>Hair heavy metals, Urinary organics</td>
            <td>Nutrient minerals, Glutathione status</td>
            <td>Quarterly exposures, Annual comprehensive</td>
        </tr>
        <tr>
            <td><strong>Sensory</strong></td>
            <td>Heartbeat perception, Sensory profile</td>
            <td>EEG sensory gating, Proprioception tests</td>
            <td>Quarterly assessment, Annual objective testing</td>
        </tr>
    </table>

    <h2>Personalized Intervention Protocols</h2>

    {% for subtype_key, protocol in intervention_protocols.items() %}
    <div class="treatment-box">
        <h3>{{ protocol.subtype }}</h3>

        <p><strong>Immediate Actions:</strong></p>
        <ul>
            {% for action in protocol.immediate_actions %}
            <li>{{ action }}</li>
            {% endfor %}
        </ul>

        {% if protocol.medium_term %}
        <p><strong>Medium-term Interventions:</strong></p>
        <ul>
            {% for action in protocol.medium_term %}
            <li>{{ action }}</li>
            {% endfor %}
        </ul>
        {% endif %}

        <p><strong>Monitoring:</strong> {{ protocol.monitoring_frequency }}</p>

        <p><strong>Success Metrics:</strong></p>
        <ul>
            {% for metric in protocol.success_metrics %}
            <li>{{ metric }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endfor %}

    <h2>Longitudinal Monitoring Plan</h2>

    <h3>Quarterly Assessments</h3>
    <div class="info-box">
        <strong>All Subtypes:</strong>
        <ul>
            {% for item in monitoring_plan.quarterly.all_subtypes %}
            <li>{{ item }}</li>
            {% endfor %}
        </ul>
    </div>

    <h3>Annual Comprehensive Assessment</h3>
    <div class="info-box">
        <strong>Core Battery:</strong>
        <ul>
            {% for item in monitoring_plan.annual.all_subtypes %}
            <li>{{ item }}</li>
            {% endfor %}
        </ul>

        <strong>Subtype-Specific Biomarker Panels:</strong>
        <ul>
            {% for subtype, panel in monitoring_plan.annual.biomarker_panels.items() %}
            <li><strong>{{ subtype.title() }}:</strong> {{ panel }}</li>
            {% endfor %}
        </ul>
    </div>

    <h2>Clinical Decision Support</h2>

    <div class="info-box">
        <h3>Assessment Algorithm</h3>
        <ol>
            <li><strong>Initial Screening:</strong> Core ADHD/ASD assessment + brief multi-modal screen</li>
            <li><strong>Risk Stratification:</strong> Identify high-risk biomarker patterns</li>
            <li><strong>Subtype Classification:</strong> Cluster assignment based on extended features</li>
            <li><strong>Targeted Testing:</strong> Comprehensive biomarker panel for identified subtype</li>
            <li><strong>Intervention Planning:</strong> Subtype-specific protocol implementation</li>
            <li><strong>Ongoing Monitoring:</strong> Quarterly + annual assessments with adjustments</li>
        </ol>
    </div>

    <h2>Important Considerations</h2>

    <div class="warning-box">
        <h3>Clinical Interpretation Guidelines</h3>
        <ul>
            <li>These subtypes represent dimensional patterns, not discrete categories</li>
            <li>Many individuals show features of multiple subtypes (consider hybrid approaches)</li>
            <li>Biomarker results should be interpreted in full clinical context</li>
            <li>Some interventions require specialist consultation (cardiology, sleep medicine, toxicology)</li>
            <li>Environmental remediation is critical but may require time and resources</li>
            <li>Monitor for treatment response and adjust protocols as needed</li>
        </ul>
    </div>

    <h2>References and Resources</h2>

    <p><strong>Clinical Guidelines:</strong></p>
    <ul>
        <li>AAP Guidelines for ADHD Diagnosis and Management</li>
        <li>NIH Environmental Health Perspectives - Neurodevelopmental Toxicology</li>
        <li>American Academy of Sleep Medicine - Circadian Rhythm Disorders</li>
        <li>Heart Rhythm Society - Autonomic Testing Standards</li>
    </ul>

    <p><strong>Laboratory Resources:</strong></p>
    <ul>
        <li>CLIA-certified laboratories for environmental testing</li>
        <li>Sleep medicine centers for circadian assessment</li>
        <li>Autonomic testing facilities (tilt table, HRV analysis)</li>
        <li>Occupational therapy for sensory profiling</li>
    </ul>

    <hr>
    <p style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
        <em>This report is for clinical decision support. Always interpret results in context of individual patient presentation.</em><br>
        Report ID: {{ report_id }} | Generated: {{ generation_date }}
    </p>
</body>
</html>
    """

    template = Template(template_str)

    html = template.render(
        generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        n_samples=n_samples,
        n_subtypes=len(subtype_profiles),
        subtype_profiles=subtype_profiles,
        risk_stratification=risk_stratification,
        intervention_protocols=intervention_protocols,
        monitoring_plan=monitoring_plan,
        report_id=f"EXT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    return html


def create_clinical_decision_support_extended() -> ClinicalDecisionPath:
    """
    Create comprehensive clinical decision support system with extended biomarkers.

    Returns
    -------
    ClinicalDecisionPath
        Decision tree for clinical assessment and intervention

    Examples
    --------
    >>> decision_path = create_clinical_decision_support_extended()
    >>> print(decision_path.initial_assessment)
    """

    initial_assessment = {
        'core_clinical': [
            'ADOS-2 (Autism Diagnostic Observation Schedule)',
            'ADHD-RS-5 (ADHD Rating Scale)',
            'Cognitive assessment (WISC-V or age-appropriate)',
            'Adaptive functioning (Vineland-3)'
        ],
        'extended_screening': [
            'HRV 5-minute recording',
            'Morning salivary cortisol',
            'Sleep questionnaire (CSHQ)',
            'Sensory Profile-2'
        ],
        'environmental_history': [
            'Residential history and exposures',
            'Occupational exposures (parents)',
            'Diet and water source',
            'Home environment checklist'
        ]
    }

    conditional_pathways = {
        'if_autonomic_suspected': {
            'triggers': [
                'HRV SDNN < 30ms',
                'Orthostatic symptoms present',
                'High anxiety with physical symptoms',
                'Family history of dysautonomia'
            ],
            'comprehensive_tests': [
                '24-hour Holter monitoring',
                'Tilt table test',
                'Diurnal cortisol rhythm (4-point salivary)',
                'Inflammatory markers (CRP, IL-6, TNF-α)',
                'Autoantibody panel if indicated'
            ],
            'specialist_referrals': [
                'Cardiology (dysautonomia evaluation)',
                'Autonomic disorders clinic'
            ],
            'interventions': [
                'Trial: beta-blocker (propranolol 10-20mg BID)',
                'Compression garments for orthostatic symptoms',
                'Increased fluid and salt intake',
                'HRV biofeedback training',
                'Graded exercise program'
            ],
            'monitoring': 'HRV weekly, symptoms daily, clinic visit monthly × 3'
        },

        'if_circadian_disrupted': {
            'triggers': [
                'Sleep onset latency > 30 minutes consistently',
                'DLMO > 10pm',
                'Extreme evening preference',
                'School/work performance varies by time of day'
            ],
            'comprehensive_tests': [
                '2-week actigraphy with sleep diary',
                'Dim light melatonin onset (DLMO)',
                'Core body temperature minimum (if available)',
                'Overnight polysomnography if sleep disorder suspected'
            ],
            'specialist_referrals': [
                'Sleep medicine',
                'Behavioral sleep medicine'
            ],
            'interventions': [
                'Morning bright light therapy (10,000 lux × 30min)',
                'Melatonin 0.5-3mg, timed 5-7h before desired bedtime',
                'Scheduled sleep-wake protocol with consistent timing',
                'Evening blue light blocking (>6pm)',
                'CBT-I for comorbid insomnia',
                'School start time accommodation if possible'
            ],
            'monitoring': 'Sleep diary daily, actigraphy monthly, DLMO at 3 months'
        },

        'if_environmental_burden': {
            'triggers': [
                'Exposure history positive',
                'Regression in development',
                'GI symptoms prominent',
                'Residential risk factors (old housing, industrial area)'
            ],
            'comprehensive_tests': [
                'Hair heavy metals (Pb, Hg, Al, Cd, As)',
                'Urinary organic pollutants (phthalates, BPA, pesticides)',
                'Blood lead if hair level elevated',
                'Nutrient minerals (Zn, Mg, Se, Fe)',
                'Glutathione and methylation markers'
            ],
            'specialist_referrals': [
                'Environmental medicine',
                'Toxicology consultation for significant burdens',
                'Public health for home assessment'
            ],
            'interventions': [
                'Source identification and removal',
                'Home environmental remediation',
                'Nutritional support (antioxidants, minerals)',
                'Chelation ONLY if: Pb >45 μg/dL or symptomatic + elevated',
                'Regular diet from low-exposure sources',
                'Water filtration system',
                'HEPA air filtration'
            ],
            'monitoring': 'Re-test biomarkers every 3-6 months, symptom diary'
        },

        'if_sensory_issues': {
            'triggers': [
                'Sensory Profile-2 showing definite difference',
                'Emotional dysregulation prominent',
                'Body awareness difficulties',
                'Alexithymia features'
            ],
            'comprehensive_tests': [
                'Comprehensive OT evaluation',
                'Interoceptive awareness battery',
                'Heartbeat perception task',
                'EEG sensory gating (P50, PPI) if available',
                'Proprioception and vestibular testing'
            ],
            'specialist_referrals': [
                'Occupational therapy (sensory integration trained)',
                'Developmental pediatrics'
            ],
            'interventions': [
                'Individualized sensory diet',
                'Interoceptive training protocol (8-12 weeks)',
                'Mindfulness-based stress reduction (adapted)',
                'Weighted blanket for sleep/anxiety',
                'Compression garments',
                'Environmental modifications (lighting, sound, texture)',
                'Alert Program ("How Does Your Engine Run?")'
            ],
            'monitoring': 'Sensory Profile quarterly, interoceptive accuracy at 3 months'
        }
    }

    monitoring_schedule = {
        'weekly': [
            'Symptom diary (parent + self-report if age-appropriate)',
            'Sleep diary',
            'Intervention adherence tracking'
        ],
        'monthly': [
            'Brief clinical assessment',
            'HRV spot check (autonomic subtypes)',
            'Review diaries and adjust interventions',
            'Side effects monitoring'
        ],
        'quarterly': [
            'Comprehensive clinical assessment',
            'Functional outcomes (school, social, family)',
            'Quality of life measures',
            'Subtype-specific biomarker check'
        ],
        'annually': [
            'Full assessment battery',
            'Comprehensive biomarker panel',
            'Developmental trajectory analysis',
            'Intervention optimization'
        ]
    }

    return ClinicalDecisionPath(
        initial_assessment=initial_assessment,
        conditional_pathways=conditional_pathways,
        monitoring_schedule=monitoring_schedule
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create synthetic example data
    np.random.seed(42)
    n_samples = 200

    cluster_labels = np.random.choice([0, 1, 2, 3], n_samples)

    extended_features = pd.DataFrame({
        'HRV_SDNN': np.random.normal(50, 20, n_samples),
        'HRV_RMSSD': np.random.normal(40, 15, n_samples),
        'melatonin_dlmo': np.random.normal(22, 2, n_samples),
        'sleep_duration': np.random.normal(7, 1.5, n_samples),
        'lead_level': np.abs(np.random.normal(2, 3, n_samples)),
        'mercury_level': np.abs(np.random.normal(1, 1.5, n_samples)),
        'interoception_accuracy': np.random.uniform(0.3, 0.9, n_samples)
    })

    clinical_data = pd.DataFrame({
        'anxiety': np.random.choice([0, 1], n_samples),
        'ADHD_symptoms': np.random.normal(60, 15, n_samples),
        'ASD_symptoms': np.random.normal(50, 12, n_samples)
    })

    # Generate report
    report = generate_extended_clinical_report(
        results={},
        cluster_labels=cluster_labels,
        extended_features=extended_features,
        clinical_data=clinical_data,
        output_path='test_extended_report.html'
    )

    print(f"Report generated: {report.report_id}")
    print(f"Number of subtypes: {len(report.subtype_profiles)}")

    # Create decision support
    decision_path = create_clinical_decision_support_extended()
    print("\nClinical decision support created")
    print(f"Initial assessment components: {len(decision_path.initial_assessment)}")
    print(f"Conditional pathways: {len(decision_path.conditional_pathways)}")
