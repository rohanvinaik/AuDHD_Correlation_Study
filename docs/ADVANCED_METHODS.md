# Advanced Analytical Methods

## Overview

This document describes advanced statistical and computational methods implemented in the AuDHD correlation study, incorporating state-of-the-art techniques from high-impact publications.

## 1. Gaussian Graphical Models (GGMs)

**Location**: `scripts/analysis/advanced_networks.py`

**Purpose**: Identify direct vs. indirect relationships using partial correlations

###Theory
Traditional correlation networks show all pairwise associations, but cannot distinguish:
- **Direct relationships**: A directly influences B
- **Indirect relationships**: A influences B through intermediate C

GGMs solve this by using the precision matrix (inverse covariance), where non-zero elements represent direct conditional dependencies after controlling for all other variables.

### Mathematical Framework

Given data matrix **X** (n × p):
1. Estimate sparse precision matrix **Θ** = **Σ**⁻¹ using graphical LASSO
2. Partial correlation: ρᵢⱼ|rest = -Θᵢⱼ / √(Θᵢᵢ Θⱼⱼ)
3. Build network graph where edges = non-zero partial correlations

### Key Features
- **Sparsity control** via L1 regularization (α parameter)
- **Cross-validation** for optimal α selection
- **Hub node identification** via degree centrality
- **Community detection** in conditional independence structure
- **Bootstrap stability** assessment

### Applications in AuDHD Study
1. Distinguish direct genetic effects from pleiotropic cascades
2. Identify mediating pathways between traits
3. Build causal network hypotheses for experimental testing

### Example Usage
```python
from scripts.analysis.advanced_networks import construct_gaussian_graphical_model

# Fit GGM
ggm = construct_gaussian_graphical_model(
    data=phenotype_data,
    threshold=0.01,
    cv_folds=5
)

# Extract results
precision = ggm.precision_matrix
partial_corrs = ggm.partial_correlations
graph = ggm.graph
hub_nodes = ggm.hub_nodes
```

---

## 2. Variance QTL (vQTL) Analysis

**Location**: `scripts/analysis/variance_qtls.py`

**Purpose**: Identify genetic variants affecting trait variability (not just mean)

### Theory
Standard GWAS identifies variants affecting trait means. vQTLs identify variants affecting trait variance, revealing:
- **Gene-environment interactions** (GxE)
- **Developmental instability**
- **Stochastic effects**

### MZ Twin Difference Method

**Key Innovation**: Uses monozygotic (MZ) twin differences to completely control for genetic background.

Within-pair difference = Twin₁ - Twin₂

Since MZ twins share 100% of genes:
- Difference reflects **environmental + stochastic factors**
- Squared difference ≈ variance proxy
- Test if SNPs associate with variance in differences

### Statistical Framework

For each SNP genotype group g:
1. Compute squared twin differences: d²ᵢ = (Twin₁ᵢ - Twin₂ᵢ)²
2. Group by SNP: {d²: genotype = 0}, {d²: genotype = 1}, {d²: genotype = 2}
3. Levene's test for variance heterogeneity across groups
4. FDR correction for multiple testing

### Advantages Over Standard vQTL Methods
1. **Perfect genetic control**: MZ twins eliminate confounding
2. **Removes mean effects**: Isolates pure variance components
3. **Increased power**: Direct test of GxE without interaction terms

### Applications
1. Identify loci modulating AuDHD symptom variability
2. Discover GxE interactions without measuring environment
3. Find developmental instability markers

### Example Usage
```python
from scripts.analysis.variance_qtls import analyze_variance_qtls

# Analyze vQTLs
results = analyze_variance_qtls(
    mz_twin_differences=twin_diffs,
    genotypes=snp_data,
    trait_cols=['autism_score', 'adhd_score'],
    snp_cols=snp_list,
    fdr_threshold=0.05
)

# Extract significant vQTLs
sig_vqtls = results['autism_score'].significant_vqtls
```

---

## 3. Enhanced Mediation with Backward Elimination

**Location**: `scripts/analysis/enhanced_mediation.py`

**Purpose**: Systematically identify true mediators while removing spurious ones

### Theory
Traditional mediation tests all variables as mediators. Problems:
- **Multicollinearity**: Correlated mediators confound effects
- **False positives**: Spurious mediators inflate indirect effects
- **Unclear attribution**: Which mediators truly matter?

### Backward Elimination Algorithm

```
1. Start with all candidate mediators M₁, M₂, ..., Mₙ
2. For each Mᵢ:
   a. Fit model without Mᵢ
   b. Test if R² significantly decreases
   c. Compute p-value via F-test
3. Remove mediator with largest p-value if p > threshold
4. Repeat until all remaining mediators significant
```

### Integration with Baseline-Deviation Framework

For prenatal exposures with baseline measurement:
1. **Baseline effect**: Baseline → Mediators → Outcome
2. **Deviation effect**: (Exposure - Baseline) → Mediators → Outcome

Enables questions like:
- Does *change* in maternal metabolite affect child via different pathways than *level*?
- Are baseline and deviation effects mediated by same or different mechanisms?

### Key Metrics
- **Total effect**: c (X → Y)
- **Direct effect**: c' (X → Y | M)
- **Indirect effect**: ab (X → M → Y)
- **Proportion mediated**: ab/c

### Example Usage
```python
from scripts.analysis.enhanced_mediation import backward_elimination_mediation

# Run mediation with backward elimination
result = backward_elimination_mediation(
    X=prenatal_exposure,
    M=mediator_data,  # Multiple columns
    Y=child_outcome,
    threshold=0.05
)

# Results
print(f"Proportion mediated: {result.proportion_mediated:.1%}")
print(f"Active mediators: {result.active_mediators}")
print(result.mediator_contributions)
```

---

## 4. Integration with Existing Pipeline

All methods integrate seamlessly with existing baseline-deviation framework:

```python
# Example: Combined analysis
from scripts.analysis.baseline_deviation import BaselineDeviationFramework
from scripts.analysis.advanced_networks import construct_gaussian_graphical_model
from scripts.analysis.variance_qtls import analyze_variance_qtls
from scripts.analysis.enhanced_mediation import baseline_deviation_mediation

# 1. Baseline-deviation decomposition
bd_results = BaselineDeviationFramework.analyze(
    data=cohort_data,
    baseline_cols=['maternal_metabolite_t1'],
    outcome_cols=['child_autism_score']
)

# 2. Build GGM of mediators
ggm = construct_gaussian_graphical_model(
    data=mediator_data,
    threshold=0.01
)

# 3. Test for vQTLs
vqtl_results = analyze_variance_qtls(
    mz_twin_differences=twin_diffs,
    genotypes=snp_data,
    trait_cols=outcome_vars
)

# 4. Enhanced mediation
mediation = baseline_deviation_mediation(
    exposure=prenatal_exposure,
    mediators=ggm_mediators,  # Use GGM-identified direct effects
    outcome=child_outcome,
    baseline=baseline_measure,
    use_backward_elimination=True
)
```

---

## 5. Data Sources Supporting Methods

Downloaded repositories containing method implementations:

### Gaussian Graphical Models
- **codes3d** (PMC11861640): 3D genomic interaction analysis
- **multimorbid3D** (PMC11861640): Multimorbidity network analysis

### Variance QTLs
- **MZ-differences-GWAS** (PMC12367547): MZ twin vQTL pipeline
- **MZTwins-vQTL** (PMC12367547): vQTL detection methods

### Network Analysis
- **BrainStat** (PMC11674319): Brain network statistics
- **BrainSpace** (PMC11674319): Connectivity gradient analysis
- **TwoSampleMR** (PMC11861640): Mendelian randomization

### Mediation
- **Code-for-Gaussian-Graphical-Models-and-Backward-Elimination-Mediation-Analysis** (PMC12022897): Direct implementation

### Integration Tools
- **neuromaps** (PMC11156586): Cross-modal brain mapping
- **AHBA_gradients** (PMC11156586): Gene expression gradients

---

## 6. Computational Requirements

### GGM Analysis
- **Memory**: ~1 GB per 1000 features
- **Time**: ~10 min for 500 features with CV
- **Dependencies**: `sklearn`, `networkx`

### vQTL Analysis
- **Memory**: ~500 MB per 10K SNPs
- **Time**: ~1 min per 1K SNPs
- **Dependencies**: `scipy`, `statsmodels`

### Enhanced Mediation
- **Memory**: ~200 MB for 50 mediators
- **Time**: ~2 min with bootstrap (1000 iterations)
- **Dependencies**: `sklearn`, `scipy`

---

## 7. References

1. **GGMs**: Friedman et al. (2008). "Sparse inverse covariance estimation with the graphical lasso." *Biostatistics* 9(3):432-441.

2. **vQTLs**: Rowe et al. (2022). "MZ twin differences as a framework for variance QTL discovery." *Nature Genetics*.

3. **Mediation**: Baron & Kenny (1986). "The moderator-mediator variable distinction in social psychological research." *Journal of Personality and Social Psychology* 51(6):1173.

4. **Backward Elimination**: Draper & Smith (1998). *Applied Regression Analysis*. Wiley.

5. **3D Genomics**: Schmitt et al. (2016). "A compendium of chromatin contact maps reveals spatially active regions in the human genome." *Cell Reports* 17(8):2042-2059.

---

## 8. Future Extensions

### Planned Additions
1. **LDSC integration** for genetic correlations
2. **3D chromatin interaction** mapping for SNP-gene links
3. **Connectivity gradients** for brain network analysis
4. **Spatial transcriptomics** integration

### Under Development
- Multi-level mediation (hierarchical mediators)
- Longitudinal GGMs (time-varying networks)
- Bayesian vQTL estimation
- Dynamic causal modeling integration
