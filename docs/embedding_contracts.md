# Embedding Contracts for Integration → Clustering

This document defines the **standard embedding contract** between integration methods and downstream clustering, ensuring consistent interfaces across the pipeline.

## Problem

Different integration methods produce different output formats:
- **MOFA**: Multiple latent spaces (factors per sample, loadings per feature per view)
- **Concatenation**: Single combined feature matrix
- **Group-specific**: Modality-specific embeddings
- **Adversarial**: Shared + private embeddings

Clustering expects: **consistent `X` (n_samples × n_features) arrays**.

## Solution: Standardized Output Contract

All integration methods MUST return:

```python
Dict[str, np.ndarray]
```

where:
- **Keys**: Embedding names (e.g., `"shared"`, `"genomic_specific"`, `"factors"`, `"concatenated"`)
- **Values**: NumPy arrays with shape `(n_samples, n_features)`
  - `n_samples`: Number of samples (consistent across all embeddings)
  - `n_features`: Number of dimensions in this embedding (can vary)

---

## Integration Method Contracts

### 1. MOFA Integration

**Input:**
```python
data_dict: Dict[str, pd.DataFrame]  # {modality: samples × features}
```

**Output:**
```python
{
    "factors": np.ndarray,           # (n_samples, n_factors)
    "loadings": Dict[str, pd.DataFrame],  # For inspection only
    "variance_explained": Dict[str, Dict[str, float]],
    "model": MOFAIntegration,
}
```

**Standard embeddings for clustering:**
```python
embeddings = {
    "mofa_factors": results["factors"].values  # (n_samples, n_factors)
}
```

**Shape requirements:**
- `n_samples`: Total number of samples across all modalities
- `n_factors`: Number of latent factors (default: 10)

**Example:**
```python
# MOFA with 200 samples, 10 factors
results = integrate_mofa(data_dict, cfg)
X = results["factors"].values  # Shape: (200, 10)
```

---

### 2. Concatenation (Null Integration Baseline)

**Input:**
```python
data_dict: Dict[str, pd.DataFrame]  # {modality: samples × features}
```

**Output:**
```python
{
    "concatenated": pd.DataFrame,  # (n_samples, sum(n_features))
}
```

**Standard embeddings for clustering:**
```python
embeddings = {
    "concatenated": results["concatenated"].values
}
```

**Shape requirements:**
- `n_samples`: Intersection or union of samples (configurable)
- `n_features`: Sum of features across all modalities

**Example:**
```python
# Concatenation with 200 samples, 3 modalities (500 + 300 + 200 features)
results = integrate_stack(data_dict, cfg)
X = results["concatenated"].values  # Shape: (200, 1000)
```

---

### 3. Group-Specific Integration

**Input:**
```python
data_dict: Dict[str, pd.DataFrame]  # {modality: samples × features}
group_labels: pd.Series              # Group assignments per sample
```

**Output:**
```python
{
    "shared": np.ndarray,                    # (n_samples, n_shared_factors)
    "group_specific": Dict[str, np.ndarray], # {group: (n_samples, n_factors)}
}
```

**Standard embeddings for clustering:**
```python
embeddings = {
    "shared": results["shared"],  # Shared factors only
    "shared_and_group0": np.hstack([
        results["shared"],
        results["group_specific"]["group_0"]
    ]),  # Shared + group-specific
}
```

**Shape requirements:**
- `n_samples`: Same for all embeddings
- `n_shared_factors`: Number of shared latent factors
- Group-specific factors: Vary by group

**Example:**
```python
# Group-specific with 200 samples, 2 groups, 5 shared + 3 group-specific factors
results = integrate_group_specific(data_dict, group_labels, cfg)
X_shared = results["shared"]  # Shape: (200, 5)
X_group0 = results["group_specific"]["ASD"]  # Shape: (200, 3)
```

---

### 4. Adversarial Integration

**Input:**
```python
data_dict: Dict[str, pd.DataFrame]  # {modality: samples × features}
protected_attr: pd.Series            # Protected attribute (e.g., batch, site)
```

**Output:**
```python
{
    "shared": np.ndarray,                 # (n_samples, n_shared)
    "private": Dict[str, np.ndarray],     # {modality: (n_samples, n_private)}
    "protected_removed": np.ndarray,      # (n_samples, n_shared) without protected attr
}
```

**Standard embeddings for clustering:**
```python
embeddings = {
    "shared": results["shared"],
    "protected_removed": results["protected_removed"],  # Preferred for clustering
}
```

**Shape requirements:**
- `n_samples`: Consistent across all
- `n_shared`: Number of shared factors
- `n_private`: Modality-specific dimensions

**Example:**
```python
# Adversarial with 200 samples, 10 shared factors
results = integrate_adversarial(data_dict, protected_attr, cfg)
X = results["protected_removed"]  # Shape: (200, 10)
```

---

## Downstream Clustering Contract

### Input Requirements

**Clustering expects:**
```python
X: np.ndarray  # Shape: (n_samples, n_features)
```

OR

```python
embeddings: Dict[str, np.ndarray]  # Multiple embeddings for ensemble clustering
```

### Using `MultiEmbeddingGenerator`

The `MultiEmbeddingGenerator` (in `modeling/clustering.py`) can consume multiple integration outputs:

```python
from audhd_correlation.modeling.clustering import MultiEmbeddingGenerator

# From integration results
integration_embeddings = {
    "mofa_factors": mofa_results["factors"].values,
    "concatenated": stack_results["concatenated"].values,
}

# Generate additional embeddings (t-SNE, UMAP)
emb_gen = MultiEmbeddingGenerator(methods=["tsne", "umap"])
all_embeddings = emb_gen.fit_transform(integration_embeddings["mofa_factors"])

# all_embeddings now contains:
# {
#     "tsne_perp5": (n_samples, 2),
#     "tsne_perp30": (n_samples, 2),
#     "umap_n15_d0.1": (n_samples, 2),
#     ...
# }

# Add integration embeddings
all_embeddings.update(integration_embeddings)

# Consensus clustering across all embeddings
from audhd_correlation.modeling.clustering import ConsensusClusteringPipeline
pipeline = ConsensusClusteringPipeline()
pipeline.fit(integration_embeddings["mofa_factors"])  # Primary embedding
labels = pipeline.predict()
```

---

## Standard Integration Function Signature

All integration methods should follow this signature:

```python
def integrate_{method}(
    data_dict: Dict[str, pd.DataFrame],
    cfg: AppConfig,
    output_dir: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Integrate multi-omics data using {method}

    Args:
        data_dict: Dictionary mapping modality names to data matrices (samples × features)
        cfg: Application configuration
        output_dir: Optional output directory for plots/reports
        **kwargs: Method-specific parameters

    Returns:
        Dictionary containing:
        - Primary embedding key (e.g., "factors", "concatenated", "shared")
          with np.ndarray or pd.DataFrame of shape (n_samples, n_dimensions)
        - Additional outputs (loadings, metrics, model objects) as needed

    Standard keys:
    - MOFA: "factors", "loadings", "variance_explained", "model"
    - Stack: "concatenated"
    - Group-specific: "shared", "group_specific"
    - Adversarial: "shared", "private", "protected_removed"
    """
    pass
```

---

## Validation Requirements

All integration outputs MUST satisfy:

### 1. Shape Consistency
```python
def validate_embedding_shape(embedding: np.ndarray, expected_n_samples: int) -> None:
    assert embedding.ndim == 2, f"Embedding must be 2D, got {embedding.ndim}D"
    assert embedding.shape[0] == expected_n_samples, (
        f"Expected {expected_n_samples} samples, got {embedding.shape[0]}"
    )
    assert np.isfinite(embedding).all(), "Embedding contains NaN or Inf"
```

### 2. Index Alignment (for DataFrames)
```python
def validate_index_alignment(
    embeddings: Dict[str, pd.DataFrame],
    expected_index: pd.Index
) -> None:
    for name, df in embeddings.items():
        assert df.index.equals(expected_index), (
            f"Embedding '{name}' index does not match expected index"
        )
```

### 3. No Missing Values
```python
def validate_no_missing(embedding: np.ndarray) -> None:
    assert not np.isnan(embedding).any(), "Embedding contains NaN values"
    assert np.isfinite(embedding).all(), "Embedding contains Inf values"
```

---

## Helper Function: Standardize Integration Outputs

```python
def standardize_integration_output(
    results: Dict[str, Any],
    method: str
) -> Dict[str, np.ndarray]:
    """
    Convert integration results to standard embedding format

    Args:
        results: Raw integration results
        method: Integration method name

    Returns:
        Dictionary of {name: np.ndarray} embeddings
    """
    embeddings = {}

    if method == "mofa":
        # Extract factors as primary embedding
        if "factors" in results:
            factors = results["factors"]
            embeddings["mofa_factors"] = (
                factors.values if isinstance(factors, pd.DataFrame) else factors
            )

    elif method == "stack":
        # Extract concatenated features
        if "concatenated" in results:
            concat = results["concatenated"]
            embeddings["concatenated"] = (
                concat.values if isinstance(concat, pd.DataFrame) else concat
            )

    elif method == "group_specific":
        # Extract shared and group-specific embeddings
        if "shared" in results:
            embeddings["shared"] = results["shared"]
        if "group_specific" in results:
            for group, emb in results["group_specific"].items():
                embeddings[f"group_{group}"] = emb

    elif method == "adversarial":
        # Extract protected-removed embedding (preferred for clustering)
        if "protected_removed" in results:
            embeddings["adversarial"] = results["protected_removed"]
        elif "shared" in results:
            embeddings["adversarial"] = results["shared"]

    else:
        raise ValueError(f"Unknown integration method: {method}")

    # Validate all embeddings
    if not embeddings:
        raise ValueError(f"No valid embeddings extracted from {method} results")

    n_samples = next(iter(embeddings.values())).shape[0]
    for name, emb in embeddings.items():
        validate_embedding_shape(emb, n_samples)
        validate_no_missing(emb)

    return embeddings
```

---

## Complete Workflow Example

```python
from audhd_correlation.integrate.methods import integrate_omics
from audhd_correlation.modeling.clustering import ConsensusClusteringPipeline

# Step 1: Integration (produces standardized output)
integration_results = integrate_omics(data_dict, cfg, output_dir)

# Step 2: Extract embeddings
if cfg.integrate.method == "mofa":
    X_primary = integration_results["factors"].values
elif cfg.integrate.method == "stack":
    X_primary = integration_results["concatenated"].values
else:
    raise ValueError(f"Unknown method: {cfg.integrate.method}")

# Step 3: Validate shape
n_samples = len(data_dict[list(data_dict.keys())[0]])
assert X_primary.shape[0] == n_samples, "Sample count mismatch"

# Step 4: Clustering
pipeline = ConsensusClusteringPipeline(
    use_hdbscan=True,
    use_spectral=True,
    use_bgmm=True,
    n_bootstrap=100,
)

pipeline.fit(X_primary, generate_embeddings=True)
cluster_assignment = pipeline.predict()

print(f"Clusters found: {len(set(cluster_assignment.labels))}")
print(f"Metrics: {pipeline.get_metrics()}")
```

---

## Summary

**Standard embedding contract:**
- All integration methods return `Dict[str, Any]`
- Primary embedding key contains `np.ndarray` or `pd.DataFrame` with shape `(n_samples, n_features)`
- Clustering consumes embeddings as `Dict[str, np.ndarray]` via `MultiEmbeddingGenerator`
- All embeddings validated for shape consistency, no missing values, finite values

**Key patterns:**
1. MOFA: Use `"factors"` array
2. Stack: Use `"concatenated"` array
3. Group-specific: Use `"shared"` or combined arrays
4. Adversarial: Use `"protected_removed"` array

This contract ensures **seamless integration → clustering** workflows regardless of method choice.