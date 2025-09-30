# Extended Causal Analysis System

Comprehensive multi-modal causal inference incorporating autonomic, circadian, environmental, and sensory features.

## Overview

The extended causal analysis system implements advanced causal inference methods to identify mechanisms underlying AuDHD heterogeneity. It goes beyond traditional genetic associations to test:

- **Mediation pathways**: How intermediate variables (e.g., HRV) transmit effects
- **Gene-environment interactions**: How genetic effects vary by environmental context
- **Environmental mixtures**: Combined effects of multiple exposures
- **Critical developmental periods**: When exposures have maximal impact
- **Causal network discovery**: Data-driven identification of causal relationships

## Extended Causal DAG

The system implements a comprehensive directed acyclic graph (DAG) representing hypothesized causal pathways:

```
Distal Factors → Intermediate Mechanisms → Proximal Outcomes

Genetics → Neurotransmitter function → Symptoms
         → Circadian genes → Melatonin → Sleep → Symptoms

Prenatal exposures → Brain development → Cognitive function
                   → Autonomic development → HRV → Emotional regulation

Environmental toxins → Inflammation → Symptoms
                     → Neurotransmitter disruption

Interoception → Anxiety → Emotional regulation → Symptoms
```

## Methods

### 1. Mediation Analysis

**Purpose:** Test whether effects of a treatment (T) on outcome (Y) are transmitted through a mediator (M).

**Implementation:** Baron & Kenny approach with bootstrap confidence intervals

**Example:** Does HRV mediate the effect of genetic risk on ADHD symptoms?

```
Genetic risk → HRV → ADHD symptoms
```

**Statistical Model:**
```
Step 1: Y = c*T + e₁          (total effect)
Step 2: M = a*T + e₂          (T → M path)
Step 3: Y = c'*T + b*M + e₃   (direct effect)

Indirect effect = a*b
Proportion mediated = (c - c') / c
```

**Usage:**
```python
from audhd_correlation.causal.extended_causal import MediationAnalyzer

analyzer = MediationAnalyzer()
result = analyzer.analyze_mediation(
    treatment='genetic_prs',
    mediator='HRV_SDNN',
    outcome='ADHD_symptoms',
    covariates=['age', 'sex'],
    data=df
)

print(f"Indirect effect: {result.indirect_effect:.3f}")
print(f"Proportion mediated: {result.prop_mediated:.1%}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

**Output:**
```python
@dataclass
class MediationResult:
    total_effect: float          # c (T → Y)
    direct_effect: float         # c' (T → Y | M)
    indirect_effect: float       # a*b (through mediator)
    prop_mediated: float         # (c - c') / c
    ci_lower: float             # Bootstrap 95% CI lower
    ci_upper: float             # Bootstrap 95% CI upper
    p_value: float              # Significance of indirect effect
    n_bootstrap: int = 1000
```

### 2. Gene-Environment (G×E) Interaction Analysis

**Purpose:** Test whether genetic effects on outcomes depend on environmental context.

**Example:** Do clock gene variants have stronger effects on sleep under high evening light exposure?

```
Clock gene PRS × Evening light → Sleep quality
```

**Statistical Model:**
```
Y = β₀ + β₁*G + β₂*E + β₃*G×E + βc*C + ε

Test: H₀: β₃ = 0 (no interaction)
```

**Usage:**
```python
from audhd_correlation.causal.extended_causal import GxEAnalyzer

analyzer = GxEAnalyzer()
result = analyzer.test_interaction(
    genetic_var='clock_gene_prs',
    env_var='evening_light_lux',
    outcome='sleep_quality',
    covariates=['age', 'sex'],
    data=df
)

print(f"Interaction effect: {result.interaction_effect:.3f}")
print(f"P-value: {result.p_interaction:.3e}")

# Test heterogeneity across environmental strata
result = analyzer.test_circadian_gxe(
    clock_gene_prs='clock_prs',
    light_exposure='evening_light',
    sleep_quality='psqi_score',
    data=df
)
```

**Output:**
```python
@dataclass
class GxEResult:
    interaction_effect: float            # β₃
    main_effect_g: float                # β₁
    main_effect_e: float                # β₂
    p_interaction: float                # Permutation p-value
    heterogeneous_groups: Dict[str, float]  # Effects by strata
```

### 3. Environmental Mixture Analysis

**Purpose:** Assess combined effects of multiple correlated environmental exposures.

**Method:** Weighted Quantile Sum (WQS) regression

**Example:** Combined effect of heavy metals, air pollution, and pesticides on symptoms

```
Lead + Mercury + PM2.5 + BPA + Phthalates → ADHD symptoms
```

**Statistical Model:**
```
Y = β₀ + β₁*WQS + βc*C + ε

WQS = Σᵢ wᵢ * qᵢ

where:
  qᵢ = quantile-transformed exposure i
  wᵢ = weight for exposure i (constrained: Σwᵢ = 1, wᵢ ≥ 0)
```

**Usage:**
```python
from audhd_correlation.causal.extended_causal import MixtureAnalyzer

analyzer = MixtureAnalyzer()
result = analyzer.weighted_quantile_sum(
    exposures=['lead_level', 'mercury_level', 'pm25', 'bpa', 'phthalate_sum'],
    outcome='ADHD_symptoms',
    covariates=['age', 'sex', 'ses'],
    data=df,
    n_bootstrap=100
)

print(f"Mixture effect: {result.mixture_effect:.3f} (p={result.p_value:.3f})")
print("\nComponent weights:")
for exposure, weight in result.component_weights.items():
    print(f"  {exposure}: {weight:.2%}")
```

**Output:**
```python
@dataclass
class MixtureResult:
    mixture_index: np.ndarray           # WQS index for each subject
    mixture_effect: float               # β₁ (effect on outcome)
    component_weights: Dict[str, float] # Weight for each exposure
    p_value: float
    method: str                         # 'WQS', 'quantile_g', 'BKMR'
```

### 4. Critical Period Identification

**Purpose:** Identify developmental windows when exposures have maximal impact.

**Example:** When does prenatal/postnatal lead exposure most strongly affect ADHD risk?

```
Windows: Trimester 1, Trimester 2, Trimester 3, Neonatal, Infancy, Early childhood
```

**Statistical Model:**
```
For each window w:
  Y = β₀ + β_w * Exposure_w + βc*C + ε

Compare β_w across windows with permutation testing
```

**Usage:**
```python
from audhd_correlation.causal.extended_causal import CriticalPeriodAnalyzer

analyzer = CriticalPeriodAnalyzer()

# Windows must be in chronological order
exposure_windows = {
    'trimester1_lead': df['lead_tri1'],
    'trimester2_lead': df['lead_tri2'],
    'trimester3_lead': df['lead_tri3'],
    'postnatal_lead': df['lead_postnatal']
}

result = analyzer.identify_critical_periods(
    exposure_windows=exposure_windows,
    outcome='ADHD_symptoms',
    covariates=['maternal_age', 'ses'],
    data=df
)

print("Critical periods:")
for window, effect in result.window_effects.items():
    sig = "*" if result.window_pvals[window] < 0.05 else ""
    print(f"  {window}: β={effect:.3f} (p={result.window_pvals[window]:.3f}) {sig}")
```

**Output:**
```python
@dataclass
class CriticalPeriodResult:
    window_effects: Dict[str, float]    # Effect size for each window
    window_ci: Dict[str, Tuple[float, float]]  # 95% CI
    window_pvals: Dict[str, float]      # Permutation p-values
    critical_windows: List[str]         # Significant windows (p < 0.05)
    method: str                         # 'DLM', 'independent'
```

### 5. Causal Network Discovery

**Purpose:** Data-driven identification of causal relationships using conditional independence tests.

**Method:** PC algorithm (constraint-based causal discovery)

**Usage:**
```python
from audhd_correlation.causal.extended_causal import CausalNetworkDiscovery

analyzer = CausalNetworkDiscovery()

# Prepare data with relevant variables
features = df[['genetic_prs', 'HRV_SDNN', 'sleep_quality',
               'anxiety', 'ADHD_symptoms', 'ASD_symptoms']]

result = analyzer.discover_network(
    data=features,
    algorithm='PC',
    alpha=0.05  # Significance level for independence tests
)

print("Discovered edges:")
for source, target in result.edges:
    print(f"  {source} → {target}")

print("\nStrongly connected components:")
for i, component in enumerate(result.strongly_connected):
    print(f"  Component {i}: {component}")
```

**Output:**
```python
@dataclass
class CausalNetworkResult:
    adjacency_matrix: np.ndarray        # n×n matrix of connections
    edges: List[Tuple[str, str]]        # List of directed edges
    strongly_connected: List[List[str]] # Groups of bidirectional variables
    algorithm: str                      # 'PC', 'GES', 'correlation'
```

## Integrated Analysis Pipeline

The `extended_causal_analysis()` function orchestrates all methods:

```python
from audhd_correlation.causal.extended_causal import extended_causal_analysis

# Prepare multi-modal feature data
all_features = pd.DataFrame({
    # Genetic
    'genetic_prs': genetic_risk_scores,
    'clock_gene_prs': circadian_prs,

    # Autonomic
    'HRV_SDNN': hrv_data['sdnn'],
    'HRV_RMSSD': hrv_data['rmssd'],

    # Circadian
    'melatonin_phase': circadian_data['dlmo'],
    'sleep_duration': sleep_data['hours'],

    # Environmental
    'lead_level': biomarkers['lead'],
    'mercury_level': biomarkers['mercury'],
    'pm25': pollution_data['pm25'],

    # Covariates
    'age': demographics['age'],
    'sex': demographics['sex']
})

outcomes = pd.DataFrame({
    'ADHD_symptoms': questionnaires['adhd_total'],
    'ASD_symptoms': questionnaires['asd_total']
})

# Run comprehensive analysis
results = extended_causal_analysis(
    all_features=all_features,
    outcomes=outcomes,
    config={
        'run_mediation': True,
        'run_gxe': True,
        'run_mixtures': True,
        'run_critical_periods': False,  # Requires longitudinal data
        'run_network': True
    }
)

# Results structure
{
    'mediation': {
        'genetic_HRV_ADHD': MediationResult(...),
        'HRV_anxiety_ADHD': MediationResult(...),
        ...
    },
    'gxe': {
        'clock_prs_x_light_on_sleep': GxEResult(...),
        ...
    },
    'mixtures': {
        'environmental_mixture_on_ADHD': MixtureResult(...),
        ...
    },
    'causal_network': CausalNetworkResult(...)
}
```

## Visualization

### 1. Extended Cluster Visualization

Comprehensive 9-panel figure showing multi-modal cluster characterization:

```python
from audhd_correlation.causal.extended_causal import visualize_extended_clusters

# Get cluster assignments from clustering analysis
clusters = clustering_results['labels']

# Create visualization
fig = visualize_extended_clusters(
    clusters=clusters,
    extended_features=all_features,
    save_path='results/extended_clusters.html'
)

fig.show()
```

**Panels:**
1. **Genetic-Metabolic**: Correlation between genetic risk and metabolite levels
2. **Autonomic Profile**: Radar chart of HRV features by cluster
3. **Circadian Phase**: Box plots of melatonin/sleep timing
4. **Environmental Burden**: Stacked bar chart of toxin exposures
5. **Sensory Profile**: Scatter plot of sensory features
6. **Clinical Severity**: Bar chart of symptom severity by cluster
7. **Developmental Trajectory**: Age × symptom trajectories with trend lines
8. **Treatment Response**: Response patterns by cluster
9. **Multimodal 3D**: PCA-based 3D integration of all modalities

### 2. Causal Pathway Diagram

Interactive network visualization of significant causal relationships:

```python
from audhd_correlation.causal.extended_causal import create_causal_pathway_diagram

create_causal_pathway_diagram(
    mediation_results=results['mediation'],
    gxe_results=results['gxe'],
    output_path='results/causal_pathways.html'
)
```

**Features:**
- Red edges: Mediation pathways
- Blue edges: G×E interactions
- Edge width: Effect size
- Node size: Degree (connectivity)

## Data Requirements

### Minimal Requirements
- **Genetic data**: PRS or individual SNPs
- **Autonomic data**: At least one HRV metric (SDNN, RMSSD, LF/HF)
- **Environmental data**: Exposure measurements (biomarkers or questionnaires)
- **Outcomes**: ADHD/ASD symptom scores
- **Covariates**: Age, sex, SES

### Optimal Requirements
- **Multi-omics**: Genetics, transcriptomics, proteomics, metabolomics
- **Circadian markers**: Melatonin, sleep actigraphy, clock gene expression
- **Comprehensive environmental**: Heavy metals, air pollution, pesticides, endocrine disruptors
- **Sensory/interoception**: Sensory profile questionnaires, interoceptive accuracy tasks
- **Longitudinal data**: Repeated measures across development (for critical period analysis)

## Interpretation Guidelines

### Mediation Analysis

**Full mediation:** Indirect effect significant, direct effect becomes non-significant
- Interpretation: Effect fully explained by mediator
- Example: Genetic risk → HRV → symptoms (no direct path remains)

**Partial mediation:** Both indirect and direct effects significant
- Interpretation: Mediator explains some but not all of the effect
- Example: Genetic risk → HRV → symptoms (plus direct genetic effects)

**No mediation:** Indirect effect non-significant
- Interpretation: Proposed mediator does not transmit effect
- Example: Genetic risk ↛ HRV → symptoms

### G×E Interactions

**Positive interaction (β₃ > 0):**
- Genetic effects stronger under high environmental exposure
- Example: Clock gene variants have larger sleep effects under high evening light

**Negative interaction (β₃ < 0):**
- Environmental exposure buffers genetic risk
- Example: Structured routines reduce impact of ADHD genetic risk

**Crossover interaction:**
- Direction of effect reverses across environmental strata
- Example: Gene beneficial in one environment, harmful in another

### Environmental Mixtures

**High weights (> 20%):** Primary contributors to mixture effect
**Low weights (< 5%):** Minor contributors
**Zero weights:** No contribution (possibly protective or confounded)

### Critical Periods

**Early effects (prenatal/infancy):** May indicate neurodevelopmental programming
**Late effects (childhood/adolescence):** May indicate ongoing plasticity or accumulation

## Statistical Considerations

### Sample Size

**Mediation:** N ≥ 200 recommended for adequate power
**G×E:** N ≥ 500 recommended (interactions require larger samples)
**Mixtures:** N ≥ 300 recommended for stable weight estimation
**Critical periods:** N ≥ 100 per window recommended

### Multiple Testing

The pipeline tests multiple pathways. Consider correction:
- **Bonferroni**: Divide α by number of tests (conservative)
- **FDR (Benjamini-Hochberg)**: Control false discovery rate (recommended)
- **Stratify by analysis type**: Separate corrections for mediation, G×E, etc.

### Assumptions

**Mediation:**
- No unmeasured confounding of T→M, T→Y, M→Y
- Correct functional form (consider non-linear relationships)
- No measurement error in mediator

**G×E:**
- Main effects and interaction correctly specified
- Gene and environment measured without substantial error
- No gene-environment correlation (or controlled for)

**Mixtures (WQS):**
- Effects assumed to be in same direction (all harmful or all protective)
- Monotonic dose-response relationships
- For opposite-direction effects, use quantile g-computation or BKMR

### Sensitivity Analyses

1. **Mediation**: Test with different covariate sets
2. **G×E**: Check for gene-environment correlation
3. **Mixtures**: Compare WQS, quantile g-computation, BKMR results
4. **Critical periods**: Test adjacent window combinations
5. **Network discovery**: Vary independence test threshold (α)

## Integration with Main Pipeline

The extended causal analysis integrates with other pipeline components:

```python
# After clustering (analysis/clustering.py)
from audhd_correlation.causal.extended_causal import extended_causal_analysis

# Within-cluster causal analysis
for cluster_id in range(n_clusters):
    cluster_mask = cluster_labels == cluster_id
    cluster_features = all_features[cluster_mask]
    cluster_outcomes = outcomes[cluster_mask]

    cluster_results = extended_causal_analysis(
        cluster_features,
        cluster_outcomes
    )

    # Save cluster-specific causal findings
    save_cluster_causal_results(cluster_id, cluster_results)

# Cross-cluster comparisons
compare_causal_mechanisms_across_clusters(all_cluster_results)
```

## Advanced Features

### Longitudinal Mediation

Track how mediation changes over development:

```python
time_points = ['baseline', 'year1', 'year2']
for t in time_points:
    result = analyzer.analyze_mediation(
        treatment='genetic_prs',
        mediator=f'HRV_SDNN_{t}',
        outcome=f'ADHD_symptoms_{t}',
        data=df
    )
    longitudinal_results[t] = result
```

### Multi-level G×E

Test interactions at multiple levels:

```python
# Individual-level gene × personal environment
gxe_individual = analyzer.test_interaction(
    'clock_prs', 'personal_light_exposure', 'sleep_quality', data=df
)

# Gene × family environment
gxe_family = analyzer.test_interaction(
    'clock_prs', 'family_routines', 'sleep_quality', data=df
)

# Gene × neighborhood environment
gxe_neighborhood = analyzer.test_interaction(
    'clock_prs', 'neighborhood_light_pollution', 'sleep_quality', data=df
)
```

### Hierarchical Mixtures

Test mixtures within exposure classes:

```python
# Heavy metals mixture
metals_result = analyzer.weighted_quantile_sum(
    exposures=['lead', 'mercury', 'cadmium', 'arsenic'],
    outcome='ADHD_symptoms',
    data=df
)

# Air pollutants mixture
pollutants_result = analyzer.weighted_quantile_sum(
    exposures=['pm25', 'pm10', 'no2', 'ozone'],
    outcome='ADHD_symptoms',
    data=df
)

# Combined super-mixture
df['metals_wqs'] = metals_result.mixture_index
df['pollutants_wqs'] = pollutants_result.mixture_index

super_mixture = analyzer.weighted_quantile_sum(
    exposures=['metals_wqs', 'pollutants_wqs', 'pesticides', 'pfas'],
    outcome='ADHD_symptoms',
    data=df
)
```

## References

### Methods
- **Mediation**: Baron & Kenny (1986), Preacher & Hayes (2008)
- **G×E**: Risch (2001), Boardman et al. (2014)
- **WQS**: Carrico et al. (2015)
- **Critical periods**: Sanchez et al. (2018)
- **PC algorithm**: Spirtes et al. (2000)

### Applications
- **Autonomic mediation**: Koenig et al. (2016), Bellato et al. (2020)
- **Circadian G×E**: Lane et al. (2016)
- **Environmental mixtures**: Braun et al. (2020), Goodrich et al. (2021)

## Troubleshooting

### Error: "Insufficient sample size for bootstrap"
- Increase sample size (N < 50 not recommended)
- Reduce n_bootstrap parameter
- Check for missing data

### Error: "Mediation analysis failed: singular matrix"
- Check for multicollinearity in covariates
- Standardize variables
- Remove highly correlated covariates

### Warning: "G×E interaction p-value > 0.5"
- Likely no interaction present
- Check measurement quality
- Consider non-linear interactions
- Increase sample size

### Error: "WQS weights do not converge"
- Increase n_bootstrap
- Check for outliers in exposures
- Consider log-transforming exposures
- Try quantile g-computation instead

## Citation

When using the extended causal analysis system, please cite:

```
Extended Causal Analysis System (2025). AuDHD Correlation Study.
https://github.com/rohanvinaik/AuDHD_Correlation_Study
```

---

**Questions?** See main project documentation or open an issue on GitHub.
