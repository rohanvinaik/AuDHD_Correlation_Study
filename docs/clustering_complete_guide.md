# Complete Clustering Guide

Comprehensive guide to clustering in the AuDHD Correlation Study pipeline, covering configuration, reproducibility, noise handling, and output management.

## Quick Start

```python
from audhd_correlation.modeling.clustering import ConsensusClusteringPipeline
from audhd_correlation.modeling.noise_handling import handle_noise
from audhd_correlation.modeling.clustering_outputs import save_all_clustering_outputs

# Load integrated data
X = integration_results["factors"].values  # (n_samples, n_factors)
sample_ids = integration_results["factors"].index.values

# Initialize pipeline with reproducible seed
pipeline = ConsensusClusteringPipeline(
    use_hdbscan=True,
    use_spectral=True,
    use_bgmm=True,
    n_bootstrap=100,
    random_state=42,  # Ensures reproducibility
)

# Fit clustering
pipeline.fit(X, generate_embeddings=True)

# Get results
cluster_assignment = pipeline.predict()
labels = cluster_assignment.labels
confidence = cluster_assignment.confidence

# Handle noise (optional)
labels, _, _ = handle_noise(
    X, labels, confidence,
    strategy="reassign",  # or "keep", "filter"
    reassign_method="nearest_cluster"
)

# Save all outputs
save_all_clustering_outputs(
    labels=labels,
    sample_ids=sample_ids,
    output_dir="outputs/clustering",
    coassignment=pipeline.coassignment_matrix_,
    embeddings=pipeline.embeddings_,
    confidence=confidence,
    metrics=pipeline.get_metrics().__dict__,
)
```

## Contents

1. [Configuration](#configuration)
2. [Auto-K Determination](#auto-k-determination)
3. [Reproducibility](#reproducibility)
4. [Noise Handling](#noise-handling)
5. [Output Management](#output-management)
6. [Reference Tests](#reference-tests)
7. [Troubleshooting](#troubleshooting)

---

## Configuration

### YAML Config

```yaml
# configs/clustering/standard.yaml

clustering:
  # Auto-k determination
  auto_k:
    enabled: true
    method: "eigengap"  # eigengap, silhouette, gap, modularity, stability
    min_k: 2
    max_k: 10

  # Consensus clustering
  consensus:
    resamples: 100
    threshold: 0.5
    random_state: 42

  # Method-specific parameters
  clusterers:
    hdbscan_main:
      min_cluster_size: 10
      min_samples: 5
      metric: "euclidean"
      cluster_selection_method: "eom"
      random_state: 42

    spectral:
      affinity: "precomputed"
      assign_labels: "kmeans"
      random_state: 42

    bgmm:
      max_components: 10
      covariance_type: "full"
      random_state: 42

  # Noise handling
  noise_handling:
    strategy: "keep"  # keep, reassign, filter
    reassign_method: "nearest_cluster"  # or knn_vote
    min_confidence: 0.5

  # Embeddings
  embeddings:
    methods: ["tsne", "umap"]
    tsne:
      perplexities: [5, 10, 30, 50]
      n_components: 2
      random_state: 42
    umap:
      n_neighbors_list: [5, 15, 30, 50]
      min_dists: [0.1, 0.3, 0.5]
      n_components: 2
      random_state: 42

  # Outputs
  save_coassignment: true
  save_projections: true
  output_dir: "outputs/clustering"
```

### Programmatic Configuration

```python
from audhd_correlation.modeling.clustering import ConsensusClusteringPipeline

pipeline = ConsensusClusteringPipeline(
    use_hdbscan=True,
    use_spectral=True,
    use_bgmm=True,
    use_tda=False,
    n_bootstrap=100,
    random_state=42,
)
```

---

## Auto-K Determination

When `n_clusters=None`, the pipeline automatically determines optimal k.

### Methods Available

| Method | Best For | Parameters |
|--------|----------|------------|
| **eigengap** | Well-separated clusters | `max_k` |
| **silhouette** | Round, convex clusters | `min_k`, `max_k`, `tolerance` |
| **gap** | Arbitrary shapes | `min_k`, `max_k`, `n_refs` |
| **modularity** | Network data | `min_k`, `max_k`, `n_neighbors` |
| **stability** | Robust assignments | `min_k`, `max_k`, `n_bootstraps` |

### Example Usage

```python
from audhd_correlation.modeling.clustering import SpectralCoAssignmentClustering

# Auto-detect using eigengap
clusterer = SpectralCoAssignmentClustering(
    n_clusters=None,  # Auto-detect
)
clusterer.fit(X)

print(f"Detected {clusterer.n_clusters} clusters")
```

See `docs/clustering_configuration.md` for detailed algorithms.

---

## Reproducibility

### Setting Random State

All random operations accept `random_state`:

```python
# Global seed (optional, for extra safety)
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Pass to pipeline (required)
pipeline = ConsensusClusteringPipeline(random_state=42)
```

### What Gets Seeded?

✅ **Bootstrap resampling** for consensus
✅ **HDBSCAN** clustering
✅ **t-SNE** embedding (all perplexities)
✅ **UMAP** embedding (all parameter combinations)
✅ **K-means** initialization in spectral clustering
✅ **Bayesian GMM** initialization

### Verification

Run deterministic reference tests:

```bash
pytest tests/reference/test_clustering_determinism.py -v
```

Tests ensure:
- Same seed → Identical labels (or ARI > 0.95)
- Different seeds → Different labels, but similar quality

---

## Noise Handling

HDBSCAN assigns label `-1` to noise points. Three strategies available:

### Strategy 1: Keep Noise (Default)

Retain `-1` as separate cluster.

```python
from audhd_correlation.modeling.noise_handling import handle_noise

labels_out, _, _ = handle_noise(X, labels, strategy="keep")
```

**Metrics impact:** Computed only on non-noise samples (labels ≥ 0)

---

### Strategy 2: Reassign Noise

Assign to nearest cluster.

#### Option A: Nearest Cluster (by centroid)

```python
labels_out, _, _ = handle_noise(
    X, labels,
    strategy="reassign",
    reassign_method="nearest_cluster"
)
```

#### Option B: k-NN Vote

```python
labels_out, _, _ = handle_noise(
    X, labels,
    strategy="reassign",
    reassign_method="knn_vote",
    knn_k=5
)
```

---

### Strategy 3: Filter Noise

Remove noise points from analysis.

```python
X_filtered, labels_filtered, indices = handle_noise(
    X, labels, confidence,
    strategy="filter",
    min_confidence=0.5
)
```

---

### Noise Statistics

```python
from audhd_correlation.modeling.noise_handling import get_noise_statistics

stats = get_noise_statistics(labels, confidence)

print(f"Noise fraction: {stats['noise_fraction']*100:.1f}%")
print(f"Mean confidence (noise): {stats['mean_confidence_noise']:.3f}")
print(f"Mean confidence (valid): {stats['mean_confidence_valid']:.3f}")
```

---

### Imbalanced Clusters

Detect and handle imbalanced clusters:

```python
from audhd_correlation.modeling.noise_handling import (
    detect_imbalance,
    merge_small_clusters
)

# Detect
is_imbalanced, info = detect_imbalance(labels, threshold=0.1)

if is_imbalanced:
    print(f"Imbalanced: min={info['min_size']}, max={info['max_size']}")

    # Option 1: Merge small clusters
    labels_merged = merge_small_clusters(X, labels, min_size=10)

    # Option 2: Re-cluster with different parameters
    # Option 3: Accept if biologically meaningful
```

---

## Output Management

### Save All Outputs

```python
from audhd_correlation.modeling.clustering_outputs import save_all_clustering_outputs

save_all_clustering_outputs(
    labels=labels,
    sample_ids=sample_ids,
    output_dir="outputs/clustering",
    coassignment=coassignment_matrix,
    embeddings=embeddings_dict,
    confidence=confidence_scores,
    metrics=metrics_dict,
)
```

**Saves:**
- `cluster_labels.csv`: Sample IDs + labels + confidence
- `coassignment_matrix.csv/.npz`: Co-assignment matrix
- `coassignment_matrix.png`: Heatmap with cluster boundaries
- `projections_*.csv`: t-SNE/UMAP coordinates
- `projection_*.png`: 2D scatter plots
- `clustering_summary.json`: Metrics and statistics

---

### Co-Assignment Matrix

```python
from audhd_correlation.modeling.clustering_outputs import (
    save_coassignment_matrix,
    load_coassignment_matrix,
    plot_coassignment_matrix
)

# Save
save_coassignment_matrix(
    coassignment=coassignment,
    sample_ids=sample_ids,
    output_path="outputs/coassignment",
    format="both"  # csv and npy
)

# Load
coassignment, sample_ids = load_coassignment_matrix("outputs/coassignment.npz")

# Plot
plot_coassignment_matrix(
    coassignment=coassignment,
    labels=labels,
    output_path="outputs/coassignment.png"
)
```

---

### Projections

```python
from audhd_correlation.modeling.clustering_outputs import (
    save_projection,
    plot_projection
)

# Save single projection
save_projection(
    embedding=tsne_embedding,
    labels=labels,
    sample_ids=sample_ids,
    method="tsne_perp30",
    output_path="outputs/projections"
)

# Plot
plot_projection(
    embedding=tsne_embedding,
    labels=labels,
    confidence=confidence,
    method="t-SNE (perplexity=30)",
    output_path="outputs/tsne_plot.png"
)
```

---

## Reference Tests

Deterministic tests ensure reproducibility across runs and library versions.

### Running Tests

```bash
# Run all reference tests
pytest tests/reference/test_clustering_determinism.py -v -m reference

# Update reference labels (after verifying changes are expected)
pytest tests/reference/test_clustering_determinism.py -v -m reference --update-references
```

### What's Tested

✅ **ConsensusClusteringPipeline** - Same seed → identical labels
✅ **HDBSCANParameterSweep** - Parameter sweep reproducibility
✅ **SpectralCoAssignment** - Bootstrap stability
✅ **BayesianGMM** - Initialization reproducibility

### Tolerance-Based Comparison

Tests use ARI ≥ 0.95 tolerance to account for:
- Floating-point precision differences
- Minor library version changes
- Non-deterministic tie-breaking

```python
from sklearn.metrics import adjusted_rand_score

def compare_labels_with_tolerance(labels1, labels2, tolerance=0.95):
    ari = adjusted_rand_score(labels1, labels2)
    return ari >= tolerance
```

---

## Troubleshooting

### Issue: Only 1 cluster found

**Causes:**
- Data not preprocessed (scale, batch correct)
- HDBSCAN parameters too strict
- Insufficient samples

**Solution:**
```python
from audhd_correlation.modeling.clustering import HDBSCANParameterSweep

# Relax parameters
sweep = HDBSCANParameterSweep(
    min_cluster_sizes=[3, 5, 10],  # Lower thresholds
    min_samples_list=[1, 3, 5],
)
sweep.fit(X)

results = sweep.get_sweep_results()
print(results.sort_values('score', ascending=False).head(10))
```

---

### Issue: Too many small clusters

**Solutions:**

1. **Merge small clusters:**
```python
from audhd_correlation.modeling.noise_handling import merge_small_clusters

labels_merged = merge_small_clusters(X, labels, min_size=10)
```

2. **Increase min_cluster_size:**
```python
clusterer = hdbscan.HDBSCAN(min_cluster_size=20)  # Larger threshold
```

---

### Issue: High noise fraction (>20%)

**Possible causes:**
- Outliers in data
- Need more preprocessing
- HDBSCAN too conservative

**Solutions:**

1. **Reassign noise:**
```python
labels, _, _ = handle_noise(X, labels, strategy="reassign")
```

2. **Check noise characteristics:**
```python
stats = get_noise_statistics(labels, confidence)
print(f"Mean confidence (noise): {stats['mean_confidence_noise']:.3f}")

# If confidence is low → True outliers
# If confidence is high → Consider reassigning
```

---

### Issue: Non-reproducible results

**Checklist:**

✅ Set `random_state` in pipeline
✅ Set `np.random.seed()` and `random.seed()` globally
✅ Check HDBSCAN has `random_state` parameter (some versions don't)
✅ Ensure no parallel processing with uncontrolled seeds

**Verification:**
```bash
pytest tests/reference/test_clustering_determinism.py::test_identical_runs_same_seed -v
```

---

### Issue: Different results across library versions

**Expected:**
- Minor ARI differences (< 0.05) acceptable
- Major differences (ARI < 0.90) require investigation

**Debugging:**
```python
# Run reference tests with tolerance
pytest tests/reference/test_clustering_determinism.py -v

# If fails, check library versions:
import hdbscan, sklearn, umap
print(f"hdbscan: {hdbscan.__version__}")
print(f"sklearn: {sklearn.__version__}")
print(f"umap: {umap.__version__}")

# Document versions that work in pyproject.toml
```

---

## Complete Example Workflow

```python
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Load integrated data
from audhd_correlation.integrate.methods import (
    integrate_omics,
    extract_primary_embedding
)

integration_results = integrate_omics(data_dict, cfg)
X = extract_primary_embedding(integration_results, method="mofa")
sample_ids = data_dict[list(data_dict.keys())[0]].index.values

# 2. Initialize clustering pipeline
from audhd_correlation.modeling.clustering import ConsensusClusteringPipeline

pipeline = ConsensusClusteringPipeline(
    use_hdbscan=True,
    use_spectral=True,
    use_bgmm=True,
    n_bootstrap=100,
    random_state=42,
)

# 3. Fit clustering
pipeline.fit(X, generate_embeddings=True)

# 4. Get results
cluster_assignment = pipeline.predict()
labels = cluster_assignment.labels
confidence = cluster_assignment.confidence

# 5. Handle noise
from audhd_correlation.modeling.noise_handling import (
    handle_noise,
    get_noise_statistics
)

stats = get_noise_statistics(labels, confidence)
print(f"\nNoise statistics:")
print(f"  Fraction: {stats['noise_fraction']*100:.1f}%")
print(f"  Mean confidence: {stats['mean_confidence_noise']:.3f}")

if stats['noise_fraction'] > 0.05:  # If >5% noise
    labels, _, _ = handle_noise(
        X, labels, confidence,
        strategy="reassign",
        reassign_method="nearest_cluster"
    )

# 6. Check for imbalance
from audhd_correlation.modeling.noise_handling import detect_imbalance

is_imbalanced, info = detect_imbalance(labels)
if is_imbalanced:
    print(f"\nWarning: Imbalanced clusters detected")
    print(f"  Min size: {info['min_size']}")
    print(f"  Max size: {info['max_size']}")
    print(f"  Ratio: {info['ratio']:.3f}")

# 7. Get metrics
metrics = pipeline.get_metrics()
print(f"\nClustering quality:")
print(f"  Silhouette: {metrics.silhouette:.3f}")
print(f"  Calinski-Harabasz: {metrics.calinski_harabasz:.1f}")
print(f"  Davies-Bouldin: {metrics.davies_bouldin:.3f}")
print(f"  Stability: {metrics.stability:.3f}")
print(f"  Clusters: {metrics.n_clusters}")

# 8. Save all outputs
from audhd_correlation.modeling.clustering_outputs import save_all_clustering_outputs

save_all_clustering_outputs(
    labels=labels,
    sample_ids=sample_ids,
    output_dir="outputs/clustering",
    coassignment=pipeline.coassignment_matrix_,
    embeddings=pipeline.embeddings_,
    confidence=confidence,
    metrics=metrics.__dict__,
)

print("\n✓ Clustering complete")
print(f"  Outputs saved to: outputs/clustering")
print(f"  Clusters found: {metrics.n_clusters}")
print(f"  Quality (silhouette): {metrics.silhouette:.3f}")
```

---

## References

- **Configuration**: `docs/clustering_configuration.md`
- **Noise Handling**: `src/audhd_correlation/modeling/noise_handling.py`
- **Output Management**: `src/audhd_correlation/modeling/clustering_outputs.py`
- **Reference Tests**: `tests/reference/test_clustering_determinism.py`
- **Clustering Implementation**: `src/audhd_correlation/modeling/clustering.py`

---

## Summary

✅ **Auto-k determination** with 5 methods (eigengap, silhouette, gap, modularity, stability)
✅ **Reproducibility** via `random_state` propagation
✅ **Noise handling** with 3 strategies (keep, reassign, filter)
✅ **Imbalance detection** and small cluster merging
✅ **Complete output management** (co-assignment, projections, summaries)
✅ **Deterministic reference tests** with tolerance-based comparison
✅ **Comprehensive troubleshooting** guide

All parameters exposed via YAML config and programmatic API.