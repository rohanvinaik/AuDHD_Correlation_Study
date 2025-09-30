# Iterative Refinement System

Multi-stage iterative analysis system that automatically identifies and removes non-discriminative data sources, re-weights remaining domains, and iteratively improves subtype separation until convergence.

## Overview

The iterative refinement system addresses a critical challenge in multi-modal analysis: **not all data sources contribute equally to subtype discrimination**. Some domains may add noise rather than signal, degrading clustering quality and obscuring true biological subtypes.

### Key Capabilities

1. **Discriminative Power Assessment**: Quantifies each domain's contribution using multiple metrics
2. **Automatic Domain Pruning**: Removes low-value domains below threshold
3. **Weight Re-normalization**: Adjusts relative importance after pruning
4. **Weight Re-optimization**: Optional adaptive weight optimization after each pruning
5. **Convergence Detection**: Stops when further improvements are marginal
6. **Full Provenance**: Tracks all decisions and weight changes across iterations
7. **Visualization**: Multi-panel plots showing refinement progress

## How It Works

### 1. Discriminative Power Metrics

Each domain is evaluated using a **composite discriminative score** (0-1 scale) combining:

| Metric | Weight | Description |
|--------|--------|-------------|
| **Silhouette Contribution** | 0.35 | Clustering quality using only this domain's features |
| **Classification Importance** | 0.30 | Random Forest feature importance for predicting cluster labels |
| **Between-Cluster Variance** | 0.20 | ANOVA F-statistic measuring separation |
| **Correlation with Clustering** | 0.15 | How well domain features correlate with cluster assignments |

**Composite Score Formula**:
```
score = 0.35 * silhouette_norm + 0.30 * rf_importance + 0.20 * anova_norm + 0.15 * correlation
```

Domains with `score >= threshold` (default 0.3) are retained; others are pruned.

### 2. Iterative Refinement Process

```
Start with all domains
↓
┌─────────────────────────────────────────────┐
│ ITERATION                                    │
│  1. Cluster with current domain set         │
│  2. Evaluate discriminative power per domain│
│  3. Identify non-discriminative domains     │
│  4. Remove weakest domains                  │
│  5. Re-normalize weights                    │
│  6. [Optional] Re-optimize weights          │
│  7. Check convergence                       │
└─────────────────────────────────────────────┘
↓
Converged? → YES → Final domain set
           → NO  → Next iteration
```

### 3. Convergence Criteria

Refinement stops when **any** of the following occur:

1. **No domains to remove**: All remaining domains are discriminative
2. **Quality plateau**: Silhouette improvement < threshold (default 0.01)
3. **Max iterations**: Reached iteration limit (default 10)
4. **Min domains**: At minimum domain count (default 3)

### 4. Literature-Based Protection

When `preserve_strong_evidence_domains=True` (default), domains with **"strong"** evidence strength are **never removed**, regardless of discriminative score. This ensures biologically important domains (genetics, metabolomics, environmental) remain in the analysis even if their contribution is subtle in the current dataset.

## Usage

### Basic Usage

```python
from audhd_correlation.integrate.extended_integration import integrate_extended_multiomics

# Enable iterative refinement mode
results = integrate_extended_multiomics(
    genetic_df=genetic_data,
    metabolomic_df=metabolomic_data,
    autonomic_df=autonomic_data,
    circadian_df=circadian_data,
    environmental_df=environmental_data,
    # ... other modalities

    # Iterative refinement parameters
    iterative_mode=True,                      # Enable refinement
    discriminative_threshold=0.3,             # Minimum score to keep domain
    min_domains=3,                            # Never go below 3 domains
    max_iterations=10,                        # Stop after 10 iterations
    optimize_weights_per_iteration=True,      # Re-optimize after pruning
    output_dir='results/refinement/'         # Export results here
)

# Access refinement results
print(f"Initial domains: {len(results['modalities'])}")
print(f"Final domains: {len(results['final_domains'])}")
print(f"Domains removed: {set(results['modalities']) - set(results['final_domains'])}")
print(f"Improvement: {results['refinement_improvement']:.3f}")
print(f"Iterations: {results['refinement_iterations']}")

# Detailed refinement history
refinement = results['refinement_results']
for iteration_result in refinement['iterations']:
    print(f"Iteration {iteration_result.iteration}:")
    print(f"  Active domains: {iteration_result.active_domains}")
    print(f"  Silhouette: {iteration_result.silhouette_score:.3f}")
    print(f"  Removed: {iteration_result.removed_domains}")
```

### Advanced Usage with Custom Clustering

```python
from audhd_correlation.integrate.iterative_refinement import (
    IterativeRefinementEngine,
    run_iterative_refinement
)
from audhd_correlation.integrate.adaptive_weights import LiteratureBasedWeights

# Prepare domain features
domain_features = {
    'genetic': genetic_df,
    'metabolomic': metabolomic_df,
    'autonomic': autonomic_df,
    'circadian': circadian_df,
    'environmental': environmental_df,
    'toxicant': toxicant_df
}

# Custom clustering function
def my_clustering_function(domain_features_subset, domain_weights):
    """Custom clustering with your preferred algorithm."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    # Weight and combine features
    weighted_features = []
    for domain, df in domain_features_subset.items():
        scaled = StandardScaler().fit_transform(df)
        weight = domain_weights.get(domain, 1.0)
        weighted = scaled * np.sqrt(weight)
        weighted_features.append(weighted)

    all_features = np.hstack(weighted_features)

    # Your clustering algorithm
    clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
    cluster_labels = clustering.fit_predict(all_features)

    return cluster_labels

# Run refinement with custom clustering
results = run_iterative_refinement(
    domain_features=domain_features,
    clustering_function=my_clustering_function,
    literature_weights=LiteratureBasedWeights(),
    discriminative_threshold=0.35,  # More stringent
    min_domains=4,
    max_iterations=15,
    optimize_weights=True,
    output_dir='results/custom_refinement/'
)
```

### Using Refinement Engine Directly

```python
# For maximum control
engine = IterativeRefinementEngine(
    literature_weights=LiteratureBasedWeights(),
    discriminative_threshold=0.3,
    min_domains=3,
    max_iterations=10,
    convergence_threshold=0.01,
    optimization_metric='silhouette',
    preserve_strong_evidence_domains=True,
    verbose=True
)

# Run refinement
results = engine.refine(
    domain_features=domain_features,
    clustering_function=my_clustering_function,
    weight_optimizer=AdaptiveWeightOptimizer(...)  # Optional
)

# Export detailed results
engine.export_results(Path('results/refinement/'))

# Visualize progress
engine.plot_refinement_progress(save_path='refinement_progress.png')
```

## Output Files

When `output_dir` is specified, the system exports:

### 1. `refinement_summary.txt`
Human-readable summary with:
- Total iterations and convergence reason
- Initial vs. final state (domains, silhouette, clusters)
- List of removed domains
- Final active domains with discriminative scores

### 2. `refinement_iterations.csv`
Per-iteration metrics:
```csv
iteration,n_domains,silhouette,davies_bouldin,calinski_harabasz,n_clusters,weight_change,quality_change,domains_removed
0,9,0.234,1.456,234.5,4,1.000,0.234,2
1,7,0.267,1.321,298.7,4,0.045,0.033,1
2,6,0.289,1.198,345.2,4,0.023,0.022,0
```

### 3. `refinement_weights.csv`
Weight evolution across iterations:
```csv
iteration,domain,weight
0,genetic,0.300
0,metabolomic,0.220
0,autonomic,0.100
1,genetic,0.350
1,metabolomic,0.256
1,autonomic,0.117
```

### 4. `refinement_discriminative_power.csv`
Final discriminative scores per domain:
```csv
domain,composite_score,silhouette_contribution,classification_importance,between_cluster_variance,correlation_with_clustering,is_discriminative
genetic,0.756,0.312,0.245,3.456,0.412,True
metabolomic,0.823,0.389,0.298,4.123,0.456,True
autonomic,0.645,0.234,0.189,2.345,0.301,True
sensory,0.234,0.089,0.067,0.876,0.145,False
```

### 5. `refinement_progress.png`
Multi-panel visualization:
- **Panel 1**: Silhouette and Davies-Bouldin scores over iterations
- **Panel 2**: Number of active domains with removal annotations
- **Panel 3**: Weight evolution for all domains
- **Panel 4**: Final discriminative power bar chart (green=kept, red=removed)

## Example Scenario

### Initial State
```
9 domains: genetic, metabolomic, autonomic, circadian, environmental,
           toxicant, microbiome, sensory, clinical

Initial silhouette: 0.234
Initial clustering: 4 clusters with moderate separation
```

### Iteration 1
```
Evaluating discriminative power...
  genetic:       0.756 ✓ (strong evidence - protected)
  metabolomic:   0.823 ✓
  autonomic:     0.645 ✓
  circadian:     0.456 ✓
  environmental: 0.534 ✓ (strong evidence - protected)
  toxicant:      0.312 ✓
  microbiome:    0.267 ✗ (below threshold)
  sensory:       0.198 ✗ (below threshold)
  clinical:      0.145 ✗ (below threshold)

Pruning: microbiome, sensory, clinical
Remaining: 6 domains

Re-normalizing weights...
Re-optimizing weights...
New silhouette: 0.298 (+0.064)
```

### Iteration 2
```
Evaluating discriminative power...
  genetic:       0.789 ✓
  metabolomic:   0.845 ✓
  autonomic:     0.678 ✓
  circadian:     0.489 ✓
  environmental: 0.556 ✓
  toxicant:      0.278 ✗ (below threshold)

Pruning: toxicant
Remaining: 5 domains

Re-normalizing weights...
Re-optimizing weights...
New silhouette: 0.312 (+0.014)
```

### Iteration 3
```
Evaluating discriminative power...
All 5 domains above threshold

No domains to remove
CONVERGED: No more domains to remove
```

### Final State
```
5 domains: genetic, metabolomic, autonomic, circadian, environmental

Final silhouette: 0.312
Improvement: +0.078 (+33%)
Final clustering: 4 clusters with strong separation

Domains removed: microbiome, sensory, clinical, toxicant
Iterations: 3
```

## Parameter Tuning Guide

### `discriminative_threshold` (default: 0.3)

**Lower values (0.2-0.25)**: More permissive, keeps more domains
- Use when: Limited data, exploratory analysis, concerned about over-pruning

**Default (0.3)**: Balanced approach
- Use when: Standard analysis with moderate sample size (N>50)

**Higher values (0.35-0.4)**: More stringent, aggressive pruning
- Use when: Large sample size (N>200), strong signal, want only top contributors

### `min_domains` (default: 3)

**2**: Absolute minimum for meaningful multi-modal analysis
**3** (recommended): Balance between simplicity and richness
**4-5**: Conservative, ensures diversity of biological perspectives

### `max_iterations` (default: 10)

Most datasets converge in 2-5 iterations. Increase to 15-20 if:
- Many domains (>12)
- Many domains near threshold
- Gradual quality improvement observed

### `optimize_weights_per_iteration` (default: True)

**True**: Re-optimize weights after each pruning (slower but better quality)
**False**: Only re-normalize (faster, may miss optimal configurations)

Use `False` when:
- Very large datasets (N>1000)
- Many iterations expected
- Quick exploratory analysis

## Interpreting Results

### Strong Improvement (>0.05 silhouette increase)
✅ Clear signal that removed domains were adding noise
✅ Final subtypes likely more biologically coherent
✅ High confidence in refined domain set

### Moderate Improvement (0.02-0.05)
⚠️ Modest benefit from refinement
⚠️ Consider: Are removed domains truly uninformative or just subtle?
⚠️ May want to try lower threshold or keep more domains

### Minimal Improvement (<0.02)
❌ Little benefit from refinement
❌ Either: All domains contribute similarly, or sample size too small
❌ Consider using all domains with standard integration

### Negative Improvement (decrease)
❌❌ Refinement degraded clustering quality
❌❌ Possible causes:
- Threshold too high (over-pruning)
- Sample size too small (unstable metrics)
- Clustering algorithm mismatch
- All domains genuinely important

**Action**: Disable iterative mode or lower threshold

## Best Practices

### 1. Start with Standard Integration
Always run standard integration first to establish baseline. Use iterative mode as refinement step.

### 2. Check Sample Size
Iterative refinement requires **N ≥ 50** per expected cluster for stable metrics. With smaller samples, use fixed literature-based weights.

### 3. Examine Removed Domains
Check if removed domains make biological sense:
- **Expected removals**: Clinical (outcome not predictor), sensory (limited quantitative data)
- **Unexpected removals**: Genetics, metabolomics → may indicate data quality issues

### 4. Validate on Held-Out Data
If possible, validate that refined domain set generalizes to independent samples.

### 5. Protect Key Domains
Keep `preserve_strong_evidence_domains=True` to ensure biologically important domains (even if subtle) remain.

### 6. Document Decisions
Always export results (`output_dir`) for full provenance and reproducibility.

### 7. Combine with Adaptive Weighting
Use both systems together:
1. Start with literature-based weights
2. Run iterative refinement to identify discriminative domains
3. Run adaptive weight optimization on final domain set
4. Validate final model

## Integration with Existing Systems

### With Adaptive Weighting

```python
from audhd_correlation.integrate.adaptive_weights import AdaptiveWeightOptimizer

# Run iterative refinement WITH weight optimization
results = integrate_extended_multiomics(
    ...,
    iterative_mode=True,
    optimize_weights_per_iteration=True  # Uses AdaptiveWeightOptimizer internally
)

# Optimal workflow:
# 1. Literature weights → 2. Iterative refinement → 3. Final weight optimization
```

### With Extended Clustering

```python
from audhd_correlation.modeling.extended_clustering import (
    FeatureAwareClustering,
    ConsensusClusteringWithValidation
)

# Use refined domain set with advanced clustering
final_domains = results['final_domains']
final_weights = results['final_weights']

# Filter features
refined_features = results['integrated_features'][final_domains]

# Advanced clustering on refined features
clustering = FeatureAwareClustering(
    n_clusters=4,
    feature_domains=final_domains,
    domain_weights=final_weights
)
refined_clusters = clustering.fit_predict(refined_features)
```

### With Extended Causal Analysis

```python
from audhd_correlation.causal.extended_causal import ExtendedCausalAnalyzer

# Causal analysis on refined domain set
analyzer = ExtendedCausalAnalyzer()

# Use final domains from refinement
results = analyzer.analyze(
    genetic_data=genetic_df if 'genetic' in final_domains else None,
    metabolomic_data=metabolomic_df if 'metabolomic' in final_domains else None,
    # ... only include final domains
)
```

## Configuration via YAML

Add to `configs/features/extended.yaml`:

```yaml
pipeline_integration:
  iterative_refinement:
    enabled: true
    discriminative_threshold: 0.3
    min_domains: 3
    max_iterations: 10
    convergence_threshold: 0.01
    optimize_weights_per_iteration: true
    preserve_strong_evidence_domains: true
    export_results: true
    output_directory: results/iterative_refinement/

    metric_weights:
      silhouette_contribution: 0.35
      classification_importance: 0.30
      between_cluster_variance: 0.20
      correlation_with_clustering: 0.15
```

## Troubleshooting

### Problem: All domains removed except minimum
**Cause**: Threshold too high or data quality issues
**Solution**: Lower threshold to 0.25, check data preprocessing, increase sample size

### Problem: No domains removed
**Cause**: Threshold too low or all domains genuinely discriminative
**Solution**: Increase threshold to 0.35, examine discriminative scores manually

### Problem: Unstable results across runs
**Cause**: Small sample size or stochastic clustering
**Solution**: Increase N, use deterministic clustering (set random_state), run bootstrap validation

### Problem: Protected domains have low scores
**Cause**: Strong evidence domains subtle in this dataset
**Solution**: Keep protection enabled, but examine if data quality is poor for these modalities

### Problem: Refinement takes very long
**Cause**: Many domains, many iterations, weight optimization enabled
**Solution**: Disable weight optimization, reduce max_iterations, or parallelize (future feature)

## Performance Considerations

### Computational Cost

Per iteration:
- **Discriminative evaluation**: O(N × D × F) where N=samples, D=domains, F=features
- **Clustering**: Depends on algorithm (typically O(N² × k) for k-means)
- **Weight optimization**: O(iterations × N × F) if enabled

**Typical runtime**: 2-10 minutes per iteration (N=100, 9 domains, 200 features)

### Memory Usage

- Stores all iteration results in memory
- Peak memory: ~100MB per iteration for N=100, D=9, F=200
- Exports to disk to free memory if needed

## Future Enhancements

1. **Parallel evaluation**: Evaluate discriminative power for domains in parallel
2. **Bayesian optimization**: Use Gaussian processes for hyperparameter tuning
3. **Domain interactions**: Consider synergistic effects (don't remove complementary domains)
4. **Hierarchical pruning**: Prune at feature level within retained domains
5. **Multi-objective optimization**: Balance clustering quality, interpretability, cost
6. **Transfer learning**: Use refinement results from one cohort to inform another

## References

### Methodological Basis

1. **Feature selection**: Guyon & Elisseeff (2003). *An introduction to variable and feature selection*. Journal of Machine Learning Research.

2. **Clustering validation**: Rousseeuw (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*. Journal of Computational and Applied Mathematics.

3. **Random Forest importance**: Breiman (2001). *Random forests*. Machine Learning, 45(1), 5-32.

4. **Multi-modal integration**: Argelaguet et al. (2018). *Multi-Omics Factor Analysis—a framework for unsupervised integration of multi-omics data sets*. Molecular Systems Biology.

## Related Documentation

- **[Adaptive Weighting System](ADAPTIVE_WEIGHTING.md)** - Literature-based weight initialization and optimization
- **[Extended Integration](../src/audhd_correlation/integrate/extended_integration.py)** - Main integration framework
- **[Extended Clustering](../src/audhd_correlation/modeling/extended_clustering.py)** - Advanced clustering methods

---

**File**: `src/audhd_correlation/integrate/iterative_refinement.py` (1,000 lines)
**Config**: `configs/features/extended.yaml`
**Tests**: `tests/test_iterative_refinement.py`
**Date**: 2025-09-30
