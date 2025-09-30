# Adaptive Weighting System with Literature Grounding

Evidence-based feature domain weighting with adaptive optimization to maximize discriminative power while remaining grounded in published meta-analyses and diagnostic studies.

## Overview

The system addresses a critical challenge: balancing **empirical evidence** from the literature with **data-driven optimization** to maximize the discriminative power of multi-modal biomarker integration.

### Key Principles

1. **Literature-Based Initialization**: Weights start from meta-analyses, heritability studies, and diagnostic accuracy research
2. **Constrained Optimization**: Weights can be optimized but must stay within evidence-based bounds
3. **Adaptive Feedback Loop**: Iterative refinement based on clustering quality or classification accuracy
4. **Transparency**: Full provenance of weight values with literature references

## Literature-Based Initial Weights

Weights are derived from 2023-2024 meta-analyses and large-scale studies:

| Domain | Weight | Range | Evidence | Key Finding |
|--------|--------|-------|----------|-------------|
| **Genetic** | 0.30 | [0.25, 0.35] | STRONG | h²=0.74-0.80 (Tick et al. 2016 meta-analysis); Accounts for 30-40% directly + G×E |
| **Metabolomic** | 0.22 | [0.18, 0.27] | STRONG | AUC 0.90-0.96 (Liu et al. 2024); Classification accuracy 81-97%; Proximal to phenotype |
| **Environmental** | 0.10 | [0.08, 0.15] | STRONG | 25% G×E + 3% direct (NRC); Equal contribution with genetics via interactions |
| **Autonomic** | 0.10 | [0.07, 0.14] | MODERATE | AUC 0.736 for ASD; Distinct hyper-arousal (ASD) vs hypo-arousal (ADHD) profiles |
| **Toxicant** | 0.08 | [0.05, 0.12] | MODERATE | Heavy metals meta-analyses; Synergistic with genetic risk during neurodevelopment |
| **Circadian** | 0.08 | [0.06, 0.12] | MODERATE | 53-93% prevalence; Mendelian randomization shows causality; Regulatory role |
| **Microbiome** | 0.07 | [0.04, 0.10] | MODERATE | Gut-brain axis; Intermediate between environment and metabolism |
| **Sensory** | 0.04 | [0.02, 0.08] | WEAK | Core ASD features but limited quantitative diagnostic accuracy data |
| **Clinical** | 0.01 | [0.00, 0.03] | WEAK | Direct phenotype (outcome, not causal predictor) |

**Total:** 1.00 (normalized)

## Evidence Strength Categories

- **STRONG**: Multiple meta-analyses, large twin studies (N>10,000), or diagnostic studies with AUC>0.85
- **MODERATE**: Multiple independent studies, diagnostic AUC 0.70-0.85, or mechanistic evidence
- **WEAK**: Limited studies, clinical relevance but insufficient quantitative data

## Key Literature Sources

### Genetics

- **Tick et al. (2016)**: Meta-analysis of 7 twin studies → h²=0.74 for ASD
  - *Journal of Child Psychology and Psychiatry*, 57(5), 585-595
- **Bai et al. (2019)**: ~80% genetic contribution, >800 genes identified
  - Study of 22,156 children
- **Sandin et al. (2017)**: Swedish twin study, h²=0.83 for ASD
  - *JAMA*, 318(12), 1182-1184

### Metabolomics

- **Liu et al. (2024)**: Plasma metabolomics with AUC 0.935-0.963
  - *MedComm*, identification of candidate biomarkers
- **SVM Classification**: Average accuracy 86%, AUC 0.95
  - Cross-validation with independent test set: 81% accuracy
- **CAMP Study**: Children's Autism Metabolome Project
  - 53% sensitivity, 91% specificity for 18-48 month olds

### Environmental Factors

- **National Research Council**: Estimated contributions
  - 3% direct environmental toxicant effects
  - 25% gene-environment interactions
  - Total: ~28% of developmental disabilities
- **Meta-analyses (2023)**: Environmental pollutants including:
  - Nitrogen dioxide, carbon monoxide, metals
  - Phthalates (MEHP, MEOHP), BPA
  - Positive associations with ASD development

### Autonomic Function

- **Diagnostic Meta-analysis**: AUC 0.736 for ASD using biological parameters
  - Compared to 0.856 for extensive clinical assessment
- **Opposite profiles**:
  - ASD: Hyper-arousal, reduced parasympathetic activity (↓ CVI)
  - ADHD: Hypo-arousal, reduced sympathetic activity (↓ CSI)
- **RSA reactivity**: Low HRV as potential ASD biomarker, especially under social stress

### Circadian Function

- **Prevalence**: 53-93% of ASD/ADHD patients have sleep disturbances
  - 53% ASD with delayed sleep onset
  - 73.3% ADHD with circadian rhythm disturbances
- **Mendelian Randomization (2024)**: Established causal relationship
  - Circadian traits causally linked to ASD/ADHD risk
- **DLMO Studies**: Delayed dim light melatonin onset in both disorders
  - Prepubertal ASD: Later DLMO, relationship with sleep/circadian parameters
  - ADHD with insomnia: Delayed melatonin onset vs controls

### Toxicants

- **Heavy Metals**: Link investigated since late 1970s
  - Lead, mercury, arsenic, cadmium associations
- **Phthalates & Organics**: Positive associations with ASD
  - Mono-3-carboxypropyl phthalate, monobutyl phthalate
- **Synergy**: Complex G×E interactions during critical neurodevelopmental periods

## Adaptive Optimization Framework

### Algorithm

```python
from audhd_correlation.integrate.adaptive_weights import (
    LiteratureBasedWeights,
    AdaptiveWeightOptimizer
)

# 1. Initialize with literature-based constraints
constraints = LiteratureBasedWeights()

# 2. Create optimizer
optimizer = AdaptiveWeightOptimizer(
    constraints=constraints,
    optimization_metric='silhouette',  # or 'davies_bouldin', 'classification'
    max_iterations=100,
    convergence_threshold=0.001
)

# 3. Optimize with your data
optimized_weights = optimizer.optimize_weights(
    domain_features=feature_dict,
    cluster_labels=clusters
)

# 4. Run adaptive feedback loop (iterative)
results = optimizer.adaptive_feedback_loop(
    domain_features=feature_dict,
    cluster_labels=clusters,
    n_iterations=5
)
```

### Optimization Metrics

**Clustering Metrics** (unsupervised):
- **Silhouette Score**: Measures cluster cohesion and separation
- **Davies-Bouldin Index**: Lower is better, ratio of within to between cluster distances
- **Calinski-Harabasz Score**: Higher is better, ratio of between to within cluster dispersion

**Classification Metrics** (supervised):
- **Cross-Validated Accuracy**: Random Forest with 3-fold CV

### Constraints

**Hard Constraints**:
1. Weights must sum to 1.0
2. Each weight must be within literature-based bounds
3. Non-negative weights only

**Soft Guidance**:
- Evidence strength penalties: Stronger evidence = narrower bounds
- Larger deviations from initial values require better metric improvement

### Convergence Criteria

Optimization stops when:
- Maximum weight change < 0.001 (threshold)
- Maximum iterations reached (default: 100)
- Metric improvement < 0.0001

## Usage in Pipeline

### 1. Default (Literature-Based)

```python
from audhd_correlation.integrate.extended_integration import extended_integrate_multiomics

# Automatically uses literature-based weights
results = extended_integrate_multiomics(
    genetic_df=genetic_data,
    metabolomic_df=metabolomic_data,
    autonomic_df=autonomic_data,
    # ... other modalities
)
```

The system will log:
```
INFO: Using literature-based weights:
  metabolomic: 0.220
  genetic: 0.300
  autonomic: 0.100
  environmental: 0.100
  ...
```

### 2. With Adaptive Optimization

```python
from audhd_correlation.integrate.adaptive_weights import (
    LiteratureBasedWeights,
    AdaptiveWeightOptimizer
)

# Prepare domain features
domain_features = {
    'genetic': genetic_df,
    'metabolomic': metabolomic_df,
    'autonomic': autonomic_df,
    # ...
}

# Run clustering with initial weights
initial_results = extended_integrate_multiomics(...)
initial_clusters = initial_results['cluster_labels']

# Optimize weights
optimizer = AdaptiveWeightOptimizer(
    constraints=LiteratureBasedWeights(),
    optimization_metric='silhouette'
)

optimized_weights = optimizer.optimize_weights(
    domain_features=domain_features,
    cluster_labels=initial_clusters
)

# Re-run with optimized weights
# (pass custom weights to integration function)
```

### 3. Loading Pre-Optimized Weights

```python
from audhd_correlation.integrate.adaptive_weights import load_adaptive_weights

# Load previously optimized and exported weights
weights = load_adaptive_weights('path/to/optimized_weights.yaml')

# Use in integration
# ...
```

## Weight Change Analysis

After optimization, examine how weights changed:

```python
print("\nWeight Changes from Literature-Based Initial:")
initial = literature_weights.get_initial_weights()
for domain in optimized_weights:
    init = initial.get(domain, 0)
    opt = optimized_weights[domain]
    change_pct = ((opt - init) / init) * 100 if init > 0 else 0
    print(f"{domain:15s}: {init:.3f} → {opt:.3f} ({change_pct:+.1f}%)")
```

**Example Output**:
```
Weight Changes from Literature-Based Initial:
genetic        : 0.300 → 0.315 (+5.0%)
metabolomic    : 0.220 → 0.245 (+11.4%)
autonomic      : 0.100 → 0.095 (-5.0%)
environmental  : 0.100 → 0.105 (+5.0%)
toxicant       : 0.080 → 0.075 (-6.3%)
circadian      : 0.080 → 0.085 (+6.3%)
microbiome     : 0.070 → 0.065 (-7.1%)
sensory        : 0.040 → 0.045 (+12.5%)
clinical       : 0.010 → 0.010 (0.0%)
```

### Interpretation Guidelines

**Small changes (<10%)**: Optimization confirms literature-based priorities
**Moderate changes (10-20%)**: Data suggests different relative importance within bounds
**Large changes (>20%)**: Indicates potential:
  - Novel subtype-specific patterns
  - Data quality issues in some modalities
  - Mismatch between literature (general population) and study sample

**Changes hitting bounds**: Domain at maximum or minimum constraint
  - Suggests literature bounds may need updating with new evidence
  - Or data quality issues preventing contribution

## Validation

### Internal Validation

1. **Cross-Validation**: Test optimized weights on held-out data
2. **Stability Analysis**: Bootstrap resampling to assess weight stability
3. **Metric Comparison**: Compare clustering/classification metrics before/after optimization

### External Validation

1. **Literature Concordance**: Do optimized weights align with domain-specific studies?
2. **Biological Plausibility**: Do weight changes make biological sense?
3. **Replication**: Do optimized weights generalize to independent cohorts?

## Reporting

When reporting analyses, include:

1. **Initial weights** with literature references
2. **Optimization details**: Metric, iterations, convergence
3. **Final weights** with percentage changes
4. **Justification**: Why changes are biologically plausible
5. **Validation**: Internal and external validation results

### Example Methods Section

> "Feature domain weights were initialized using literature-based values derived from meta-analyses of heritability (genetics: 0.30, Tick et al. 2016), diagnostic accuracy studies (metabolomics: 0.22, AUC 0.90-0.96), and prevalence studies (circadian: 0.08, 53-93% affected). Weights were optimized using constrained optimization (SLSQP) to maximize silhouette score while respecting evidence-based bounds (±30% of initial values for strong evidence domains). Final optimized weights were: genetics 0.315 (+5%), metabolomics 0.245 (+11%), autonomic 0.095 (-5%), ..."

## Files

- **Code**: `src/audhd_correlation/integrate/adaptive_weights.py`
- **Integration**: `src/audhd_correlation/integrate/extended_integration.py`
- **Config**: `configs/features/extended.yaml`
- **Documentation**: This file

## Future Enhancements

1. **Subtype-Specific Weights**: Different weights for different subtypes
2. **Age-Dependent Weights**: Adjust weights based on developmental stage
3. **Multi-Objective Optimization**: Balance multiple metrics simultaneously
4. **Bayesian Optimization**: Incorporate prior distributions from literature
5. **Meta-Learning**: Learn optimal weight adjustment strategies across cohorts

## References

### Meta-Analyses Cited

1. Tick, B., Bolton, P., Happé, F., Rutter, M., & Rijsdijk, F. (2016). Heritability of autism spectrum disorders: a meta‐analysis of twin studies. *Journal of Child Psychology and Psychiatry*, 57(5), 585-595.

2. Bai, D., et al. (2019). Association of Genetic and Environmental Factors With Autism in a 5-Country Cohort. *JAMA Psychiatry*, 76(10), 1035–1043.

3. Liu, X., et al. (2024). Metabolomic analysis of plasma biomarkers in children with autism spectrum disorders. *MedComm*, 5(2), e488.

4. National Research Council (US) Committee on Developmental Toxicology. (2000). *Scientific Frontiers in Developmental Toxicology and Risk Assessment*. National Academies Press.

### Recent Studies (2023-2024)

5. Martinez-Cayuelas, E., et al. (2024). Sleep problems and circadian rhythm functioning in autistic children. *Autism*, 28(6).

6. Meta-analysis of genetic effects in neurodevelopmental disorders. *Nature Human Behaviour* (2023).

7. Environmental pollutants as risk factors for autism spectrum disorders: systematic review and meta-analysis. *BMC Public Health* (2024).

8. Physiological parameters to support ADHD diagnosis: multiparametric approach. *Frontiers in Psychiatry* (2024).

---

**Note**: This weighting system is designed for research use. Clinical applications should be validated in independent cohorts and reviewed by domain experts before deployment.
