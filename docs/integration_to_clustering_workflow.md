# Complete Integration → Clustering Workflow

This document demonstrates the complete workflow from multi-omics integration to consensus clustering, following the standardized embedding contracts.

## Architecture Overview

```
Multi-Omics Data (genomic, clinical, metabolomic, microbiome)
    ↓
Integration Methods (MOFA, Stack, Null, Group-specific, Adversarial)
    ↓
Standardized Embeddings Dict[str, np.ndarray] ← EMBEDDING CONTRACT
    ↓
MultiEmbeddingGenerator (t-SNE, UMAP variants)
    ↓
ConsensusClusteringPipeline (HDBSCAN, Spectral, BGM, Ensemble)
    ↓
Cluster Labels + Confidence Scores
```

## Step-by-Step Workflow

### Step 1: Prepare Multi-Omics Data

```python
import pandas as pd
import numpy as np

# Load preprocessed data (after QC, transform, impute, scale, batch correct)
genomic = pd.read_csv("data/processed/genomic_processed.csv", index_col=0)
clinical = pd.read_csv("data/processed/clinical_processed.csv", index_col=0)
metabolomic = pd.read_csv("data/processed/metabolomic_processed.csv", index_col=0)
microbiome = pd.read_csv("data/processed/microbiome_processed.csv", index_col=0)

data_dict = {
    "genomic": genomic,
    "clinical": clinical,
    "metabolomic": metabolomic,
    "microbiome": microbiome,
}

print(f"Samples: {len(genomic)}")
print(f"Features: genomic={genomic.shape[1]}, clinical={clinical.shape[1]}, "
      f"metabolomic={metabolomic.shape[1]}, microbiome={microbiome.shape[1]}")
```

### Step 2: Integration with Standard Output

#### Option A: MOFA (Recommended)

```python
from audhd_correlation.integrate.methods import integrate_omics
from audhd_correlation.config.schema import AppConfig

# Configure integration
cfg = AppConfig()
cfg.integrate.method = "mofa"
cfg.integrate.n_factors = 10

# Run integration
integration_results = integrate_omics(data_dict, cfg, output_dir="outputs/mofa")

# Results follow standard contract:
# {
#     "factors": pd.DataFrame (n_samples, 10),
#     "loadings": Dict[str, pd.DataFrame],
#     "variance_explained": Dict[str, Dict[str, float]],
#     "model": MOFAIntegration object
# }

print(f"Factors shape: {integration_results['factors'].shape}")
print(f"Variance explained by Factor1 in genomic: "
      f"{integration_results['variance_explained']['genomic']['Factor1']:.3f}")
```

#### Option B: Null Integration (Baseline)

```python
cfg.integrate.method = "null"

baseline_results = integrate_omics(data_dict, cfg)

# Results:
# {
#     "concatenated": pd.DataFrame (n_samples, sum(n_features)),
#     "method": "null_baseline",
#     "n_modalities": 4,
#     "total_features": int
# }

print(f"Concatenated shape: {baseline_results['concatenated'].shape}")
```

### Step 3: Extract Standardized Embeddings

```python
from audhd_correlation.integrate.methods import (
    standardize_integration_output,
    extract_primary_embedding,
)

# Method A: Standardize to dict of embeddings
embeddings_dict = standardize_integration_output(integration_results, method="mofa")
# Returns: {"mofa_factors": np.ndarray (n_samples, 10)}

# Method B: Extract primary embedding directly
X_primary = extract_primary_embedding(integration_results, method="mofa")
# Returns: np.ndarray (n_samples, 10)

print(f"Primary embedding shape: {X_primary.shape}")
print(f"Embeddings dict keys: {list(embeddings_dict.keys())}")
```

### Step 4: Generate Additional Embeddings (Optional)

```python
from audhd_correlation.modeling.clustering import MultiEmbeddingGenerator

# Generate t-SNE and UMAP variants
emb_gen = MultiEmbeddingGenerator(
    methods=["tsne", "umap"],
    n_components=2,
    random_state=42
)

additional_embeddings = emb_gen.fit_transform(X_primary)

# Returns:
# {
#     "tsne_perp5": (n_samples, 2),
#     "tsne_perp10": (n_samples, 2),
#     "tsne_perp30": (n_samples, 2),
#     "umap_n5_d0.1": (n_samples, 2),
#     "umap_n15_d0.1": (n_samples, 2),
#     ...
# }

print(f"Generated {len(additional_embeddings)} embeddings")
```

### Step 5: Consensus Clustering

```python
from audhd_correlation.modeling.clustering import ConsensusClusteringPipeline

# Initialize pipeline
pipeline = ConsensusClusteringPipeline(
    use_hdbscan=True,       # HDBSCAN with parameter sweep
    use_spectral=True,      # Spectral on co-assignment matrix
    use_bgmm=True,          # Bayesian Gaussian Mixture
    use_tda=False,          # Topological gap detection (optional)
    n_bootstrap=100,        # Bootstrap resamples for stability
    random_state=42,
)

# Fit on primary embedding
# generate_embeddings=True will create t-SNE/UMAP variants internally
pipeline.fit(X_primary, generate_embeddings=True)

# Get cluster assignments
cluster_assignment = pipeline.predict()

print(f"Cluster labels: {cluster_assignment.labels}")
print(f"Confidence scores: {cluster_assignment.confidence}")
print(f"Number of clusters: {len(set(cluster_assignment.labels))}")
```

### Step 6: Evaluate Clustering Quality

```python
# Get metrics
metrics = pipeline.get_metrics()

print(f"\nClustering Quality Metrics:")
print(f"  Silhouette score: {metrics.silhouette:.3f}")
print(f"  Calinski-Harabasz: {metrics.calinski_harabasz:.1f}")
print(f"  Davies-Bouldin: {metrics.davies_bouldin:.3f}")
print(f"  Gap statistic: {metrics.gap_statistic:.3f}")
print(f"  Stability (avg confidence): {metrics.stability:.3f}")
print(f"  Number of clusters: {metrics.n_clusters}")
print(f"  Number of noise points: {metrics.n_noise}")

# Inspect individual method results
print(f"\nMethod-specific labels:")
for method, labels in pipeline.method_labels_.items():
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  {method}: {n_clusters} clusters")
```

### Step 7: Visualize Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get 2D embedding for visualization (t-SNE)
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X_primary)

# Plot clusters
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Consensus clusters
scatter1 = axes[0].scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=cluster_assignment.labels,
    s=50,
    alpha=0.6,
    cmap='tab10'
)
axes[0].set_title("Consensus Clusters")
axes[0].set_xlabel("t-SNE 1")
axes[0].set_ylabel("t-SNE 2")
plt.colorbar(scatter1, ax=axes[0])

# Confidence scores
scatter2 = axes[1].scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=cluster_assignment.confidence,
    s=50,
    alpha=0.6,
    cmap='viridis'
)
axes[1].set_title("Cluster Confidence")
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")
plt.colorbar(scatter2, ax=axes[1], label="Confidence")

plt.tight_layout()
plt.savefig("outputs/clustering_results.png", dpi=300)
plt.close()

print("Visualization saved to outputs/clustering_results.png")
```

### Step 8: Save Results

```python
# Create results DataFrame
results_df = pd.DataFrame({
    "sample_id": genomic.index,
    "cluster": cluster_assignment.labels,
    "confidence": cluster_assignment.confidence,
})

# Add embeddings
for i in range(X_primary.shape[1]):
    results_df[f"factor_{i+1}"] = X_primary[:, i]

# Add 2D coordinates
results_df["tsne_1"] = X_2d[:, 0]
results_df["tsne_2"] = X_2d[:, 1]

# Save
results_df.to_csv("outputs/clustering_results.csv", index=False)
print(f"Results saved: {len(results_df)} samples")

# Save metrics
metrics_dict = {
    "silhouette": metrics.silhouette,
    "calinski_harabasz": metrics.calinski_harabasz,
    "davies_bouldin": metrics.davies_bouldin,
    "gap_statistic": metrics.gap_statistic,
    "stability": metrics.stability,
    "n_clusters": metrics.n_clusters,
    "n_noise": metrics.n_noise,
}

import json
with open("outputs/clustering_metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
```

## Comparing Integration Methods

```python
# Compare MOFA vs. Null baseline
from audhd_correlation.integrate.methods import integrate_omics

methods = ["mofa", "null"]
results_comparison = {}

for method in methods:
    cfg.integrate.method = method

    # Integrate
    integration_results = integrate_omics(data_dict, cfg)

    # Extract embedding
    X = extract_primary_embedding(integration_results, method=method)

    # Cluster
    pipeline = ConsensusClusteringPipeline(random_state=42)
    pipeline.fit(X, generate_embeddings=False)

    # Store metrics
    metrics = pipeline.get_metrics()
    results_comparison[method] = {
        "silhouette": metrics.silhouette,
        "n_clusters": metrics.n_clusters,
        "embedding_dims": X.shape[1],
    }

# Print comparison
print("\nIntegration Method Comparison:")
print(f"{'Method':<10} {'Silhouette':<12} {'Clusters':<10} {'Dimensions':<12}")
print("-" * 50)
for method, results in results_comparison.items():
    print(f"{method:<10} {results['silhouette']:<12.3f} "
          f"{results['n_clusters']:<10} {results['embedding_dims']:<12}")
```

## Advanced: Ensemble Across Multiple Integration Methods

```python
# Run multiple integration methods
integration_embeddings = {}

# MOFA
cfg.integrate.method = "mofa"
mofa_results = integrate_omics(data_dict, cfg)
integration_embeddings["mofa"] = extract_primary_embedding(mofa_results, "mofa")

# Null baseline
cfg.integrate.method = "null"
null_results = integrate_omics(data_dict, cfg)
integration_embeddings["null"] = extract_primary_embedding(null_results, "null")

# Cluster each embedding separately
from audhd_correlation.modeling.clustering import consensus

# Build co-assignment matrix across all embeddings
all_labels = []
for name, X in integration_embeddings.items():
    pipeline = ConsensusClusteringPipeline(random_state=42)
    pipeline.fit(X, generate_embeddings=False)
    all_labels.append(pipeline.predict().labels)

# Final consensus across methods
n_samples = len(all_labels[0])
coassign = np.zeros((n_samples, n_samples))

for labels in all_labels:
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if labels[i] >= 0 and labels[i] == labels[j]:
                coassign[i, j] += 1
                coassign[j, i] += 1

coassign /= len(all_labels)

# Spectral clustering on consensus
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=5, affinity="precomputed")
final_labels = sc.fit_predict((coassign >= 0.5).astype(float))

print(f"Final consensus: {len(set(final_labels))} clusters")
```

## Troubleshooting

### Issue: Integration method fails

**Solution:** Use null integration as fallback:

```python
try:
    cfg.integrate.method = "mofa"
    results = integrate_omics(data_dict, cfg)
except Exception as e:
    print(f"MOFA failed: {e}. Falling back to null integration.")
    cfg.integrate.method = "null"
    results = integrate_omics(data_dict, cfg)
```

### Issue: Clustering finds only 1 cluster

**Possible causes:**
1. Data not sufficiently preprocessed (scale, batch correct)
2. Too few samples or features
3. HDBSCAN parameters too strict

**Solution:** Adjust HDBSCAN parameters:

```python
from audhd_correlation.modeling.clustering import HDBSCANParameterSweep

sweep = HDBSCANParameterSweep(
    min_cluster_sizes=[3, 5, 10],  # Lower thresholds
    min_samples_list=[1, 3, 5],
)
sweep.fit(X_primary)

print("Parameter sweep results:")
print(sweep.get_sweep_results())
```

### Issue: Embeddings have different sample counts

**Solution:** Align samples before integration:

```python
# Find common samples
common_samples = set(data_dict[list(data_dict.keys())[0]].index)
for modality, df in data_dict.items():
    common_samples &= set(df.index)

# Subset to common samples
data_dict_aligned = {
    modality: df.loc[list(common_samples)]
    for modality, df in data_dict.items()
}

print(f"Common samples: {len(common_samples)}")
```

## Summary

This workflow demonstrates:

1. ✅ **Standardized integration outputs** following embedding contract
2. ✅ **Null integration baseline** for comparison and de-risking
3. ✅ **Flexible embedding generation** (t-SNE, UMAP variants)
4. ✅ **Consensus clustering** across methods and embeddings
5. ✅ **Quality metrics** and visualization
6. ✅ **Error handling** and troubleshooting

All components follow consistent APIs and data formats as specified in `docs/embedding_contracts.md`.