# Clustering Configuration & Tunables

This document explains all tunable parameters for clustering, auto-k determination, noise handling, and reproducibility.

## Auto-K Determination (n_clusters=None)

When `n_clusters=None`, the optimal number of clusters is determined automatically using multiple heuristics.

### Available Heuristics

#### 1. Eigengap Heuristic (Default)

**Method:** Spectral analysis of affinity matrix eigenvalues

**Algorithm:**
```python
def estimate_n_clusters_eigengap(affinity: np.ndarray, max_k: int = 10) -> int:
    """
    Estimate clusters using eigengap heuristic

    1. Compute eigenvalues of affinity matrix
    2. Find largest gap in eigenvalue spectrum
    3. n_clusters = position of largest gap + 1
    """
    from scipy.linalg import eigh

    eigenvalues = eigh(affinity, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Compute gaps
    gaps = np.diff(eigenvalues[:max_k])

    # Largest gap indicates natural separation
    n_clusters = np.argmax(gaps) + 1

    return max(2, min(n_clusters, max_k))
```

**Parameters:**
- `max_k`: Maximum clusters to consider (default: 10)
- `min_k`: Minimum clusters (default: 2)

**Best for:** Well-separated clusters with clear boundaries

**Example:**
```python
from audhd_correlation.modeling.clustering import SpectralCoAssignmentClustering

clusterer = SpectralCoAssignmentClustering(
    n_clusters=None,  # Auto-detect
    affinity='precomputed',
)
clusterer.fit(X)
```

---

#### 2. Silhouette Optimization

**Method:** Sweep k values, maximize average silhouette score

**Algorithm:**
```python
def estimate_n_clusters_silhouette(
    X: np.ndarray,
    min_k: int = 2,
    max_k: int = 10
) -> int:
    """
    Estimate clusters by maximizing silhouette score

    1. Try k = min_k to max_k
    2. Cluster with k-means
    3. Compute silhouette score
    4. Return k with highest score
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k
```

**Parameters:**
- `min_k`: Minimum clusters (default: 2)
- `max_k`: Maximum clusters (default: 10)
- `tolerance`: Minimum improvement to accept (default: 0.01)

**Best for:** Round, convex clusters of similar size

---

#### 3. Gap Statistic

**Method:** Compare within-cluster dispersion to random reference

**Algorithm:**
```python
def estimate_n_clusters_gap(
    X: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    n_refs: int = 10
) -> int:
    """
    Estimate clusters using gap statistic

    Gap(k) = E[log(W_k)] - log(W_k)
    where W_k = within-cluster dispersion
    E[·] is expectation over random reference

    Choose k where Gap(k) ≥ Gap(k+1) - s_{k+1}
    """
    gaps = []

    for k in range(min_k, max_k + 1):
        # Within-cluster dispersion for data
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        W_k = compute_within_dispersion(X, labels)

        # Expected dispersion for random reference
        W_refs = []
        for _ in range(n_refs):
            X_ref = generate_random_reference(X)
            labels_ref = KMeans(n_clusters=k, random_state=42).fit_predict(X_ref)
            W_refs.append(compute_within_dispersion(X_ref, labels_ref))

        gap = np.mean(np.log(W_refs)) - np.log(W_k)
        gaps.append(gap)

    # Find first k where gap(k) >= gap(k+1) - s_{k+1}
    for i, k in enumerate(range(min_k, max_k)):
        if gaps[i] >= gaps[i + 1] - std_errors[i + 1]:
            return k

    return max_k
```

**Parameters:**
- `min_k`: Minimum clusters (default: 2)
- `max_k`: Maximum clusters (default: 10)
- `n_refs`: Number of reference datasets (default: 10)

**Best for:** Arbitrary cluster shapes, unknown structure

---

#### 4. Modularity Optimization (Graph-based)

**Method:** Maximize modularity score on k-NN graph

**Algorithm:**
```python
def estimate_n_clusters_modularity(
    X: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    n_neighbors: int = 15
) -> int:
    """
    Estimate clusters using modularity on k-NN graph

    Modularity Q = (edges within clusters) - (expected if random)

    Maximize Q over different resolutions
    """
    from sklearn.neighbors import kneighbors_graph

    # Build k-NN graph
    knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity')

    best_k = min_k
    best_modularity = -1

    for k in range(min_k, max_k + 1):
        labels = spectral_clustering(knn_graph, n_clusters=k)
        modularity = compute_modularity(knn_graph, labels)

        if modularity > best_modularity:
            best_modularity = modularity
            best_k = k

    return best_k
```

**Parameters:**
- `min_k`: Minimum clusters (default: 2)
- `max_k`: Maximum clusters (default: 10)
- `n_neighbors`: k-NN graph connectivity (default: 15)

**Best for:** Network-structured data, community detection

---

#### 5. Stability Plateau

**Method:** Bootstrap stability analysis, find plateau in ARI

**Algorithm:**
```python
def estimate_n_clusters_stability(
    X: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    n_bootstraps: int = 100,
    plateau_tolerance: float = 0.05
) -> int:
    """
    Estimate clusters using stability plateau

    1. For each k, compute stability (mean ARI across bootstraps)
    2. Find k where stability plateaus (small improvement)
    3. Choose smallest k on plateau
    """
    from sklearn.metrics import adjusted_rand_score

    stabilities = []

    for k in range(min_k, max_k + 1):
        # Bootstrap stability
        ari_scores = []
        base_labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)

        for _ in range(n_bootstraps):
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[idx]
            labels_boot = KMeans(n_clusters=k, random_state=None).fit_predict(X_boot)

            # Map back to original indices
            labels_full = -np.ones(len(X))
            labels_full[idx] = labels_boot

            # Compute ARI
            valid_mask = labels_full >= 0
            if valid_mask.sum() > 10:
                ari = adjusted_rand_score(
                    base_labels[valid_mask],
                    labels_full[valid_mask]
                )
                ari_scores.append(ari)

        stability = np.mean(ari_scores)
        stabilities.append(stability)

    # Find plateau (where improvement < tolerance)
    for i in range(len(stabilities) - 1):
        improvement = stabilities[i + 1] - stabilities[i]
        if improvement < plateau_tolerance:
            return min_k + i  # Return k at start of plateau

    return max_k
```

**Parameters:**
- `min_k`: Minimum clusters (default: 2)
- `max_k`: Maximum clusters (default: 10)
- `n_bootstraps`: Bootstrap samples (default: 100)
- `plateau_tolerance`: Stability improvement threshold (default: 0.05)

**Best for:** Ensuring robust cluster assignments

---

## ClusterConfig Schema

```yaml
clustering:
  # Auto-k determination
  auto_k:
    enabled: true
    method: "eigengap"  # Options: eigengap, silhouette, gap, modularity, stability
    min_k: 2
    max_k: 10

    # Method-specific parameters
    eigengap:
      tolerance: 0.1  # Minimum gap size to consider

    silhouette:
      tolerance: 0.01  # Minimum improvement

    gap:
      n_refs: 10  # Number of reference datasets

    modularity:
      n_neighbors: 15  # k-NN graph connectivity

    stability:
      n_bootstraps: 100
      plateau_tolerance: 0.05

  # Consensus clustering
  consensus:
    resamples: 100
    threshold: 0.5  # Co-assignment threshold
    random_state: 42  # Reproducibility

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
      weight_concentration_prior: 1.0
      n_init: 10
      random_state: 42

  # Noise handling
  noise_handling:
    strategy: "keep"  # Options: keep, reassign, filter
    reassign_method: "nearest_cluster"  # If strategy=reassign
    min_confidence: 0.5  # For filtering low-confidence samples

  # Embedding generation
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

  # Output
  save_coassignment: true
  save_projections: true
  output_dir: "outputs/clustering"
```

---

## Noise Handling Strategies

### Strategy 1: Keep Noise (Default)

**Description:** Retain HDBSCAN label `-1` as a separate "noise" cluster

**Use case:** Want to explicitly identify outliers

**Implementation:**
```python
# Labels: [0, 1, 2, -1, 0, -1, 1, ...]
# Keep as-is, treat -1 as cluster ID
```

**Metrics impact:**
- Silhouette: Computed only on non-noise samples (labels >= 0)
- Davies-Bouldin: Non-noise only
- Calinski-Harabasz: Non-noise only

**Configuration:**
```yaml
noise_handling:
  strategy: "keep"
```

---

### Strategy 2: Reassign Noise

**Description:** Assign noise points to nearest valid cluster

**Use case:** Need all samples assigned for downstream analysis

**Methods:**

#### 2a. Nearest Cluster (by centroid distance)
```python
def reassign_noise_nearest_cluster(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Reassign noise points to nearest cluster centroid
    """
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels

    # Compute cluster centroids
    centroids = {}
    for cluster in set(labels) - {-1}:
        centroids[cluster] = X[labels == cluster].mean(axis=0)

    # Assign noise to nearest centroid
    labels_reassigned = labels.copy()
    for i in np.where(noise_mask)[0]:
        distances = {
            cluster: np.linalg.norm(X[i] - centroid)
            for cluster, centroid in centroids.items()
        }
        labels_reassigned[i] = min(distances, key=distances.get)

    return labels_reassigned
```

#### 2b. K-Nearest Neighbors Vote
```python
def reassign_noise_knn_vote(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 5
) -> np.ndarray:
    """
    Reassign noise points by majority vote of k nearest neighbors
    """
    from sklearn.neighbors import NearestNeighbors

    noise_mask = labels == -1
    if not noise_mask.any():
        return labels

    # Fit k-NN on non-noise points
    valid_mask = ~noise_mask
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X[valid_mask])

    # Find neighbors for noise points
    _, indices = knn.kneighbors(X[noise_mask])

    # Vote
    labels_reassigned = labels.copy()
    valid_labels = labels[valid_mask]

    for i, neighbors_idx in zip(np.where(noise_mask)[0], indices):
        neighbor_labels = valid_labels[neighbors_idx]
        # Majority vote
        from scipy.stats import mode
        labels_reassigned[i] = mode(neighbor_labels, keepdims=False)[0]

    return labels_reassigned
```

**Configuration:**
```yaml
noise_handling:
  strategy: "reassign"
  reassign_method: "nearest_cluster"  # or "knn_vote"
  knn_k: 5  # For knn_vote
```

---

### Strategy 3: Filter Noise

**Description:** Remove noise points from downstream analysis

**Use case:** Focus only on confidently clustered samples

**Implementation:**
```python
# Remove noise points
valid_mask = labels >= 0
X_filtered = X[valid_mask]
labels_filtered = labels[valid_mask]
```

**Configuration:**
```yaml
noise_handling:
  strategy: "filter"
  min_confidence: 0.5  # Additional filtering by confidence score
```

---

## Randomness Control

All random operations accept `random_state` parameter:

```python
# Set global seed
import numpy as np
import random

def set_global_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)

# Use in clustering
pipeline = ConsensusClusteringPipeline(random_state=42)
pipeline.fit(X)

# All internal operations use this seed:
# - Bootstrap resampling
# - t-SNE embedding
# - UMAP embedding
# - K-means initialization
# - Spectral clustering initialization
```

### Reproducibility Checklist

✅ **Set random_state in:**
1. ConsensusClusteringPipeline
2. MultiEmbeddingGenerator
3. HDBSCANParameterSweep
4. SpectralCoAssignmentClustering
5. BayesianGaussianMixtureClustering

✅ **Config propagation:**
```yaml
clustering:
  consensus:
    random_state: 42  # Propagated to all methods
```

---

## Imbalanced Clusters

### Detection

```python
def detect_imbalance(labels: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Detect imbalanced clusters

    Returns True if smallest cluster < threshold * largest cluster
    """
    counts = np.bincount(labels[labels >= 0])
    if len(counts) < 2:
        return False

    min_size = counts.min()
    max_size = counts.max()

    return (min_size / max_size) < threshold
```

### Handling

**Option 1: Merge small clusters**
```python
def merge_small_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    min_size: int = 10
) -> np.ndarray:
    """
    Merge clusters smaller than min_size into nearest cluster
    """
    counts = np.bincount(labels[labels >= 0])
    small_clusters = np.where(counts < min_size)[0]

    # Compute centroids
    centroids = {}
    for cluster in set(labels) - {-1} - set(small_clusters):
        centroids[cluster] = X[labels == cluster].mean(axis=0)

    # Merge small clusters
    labels_merged = labels.copy()
    for small_cluster in small_clusters:
        mask = labels == small_cluster
        cluster_center = X[mask].mean(axis=0)

        # Find nearest large cluster
        distances = {
            cluster: np.linalg.norm(cluster_center - centroid)
            for cluster, centroid in centroids.items()
        }
        nearest = min(distances, key=distances.get)

        labels_merged[mask] = nearest

    return labels_merged
```

**Option 2: Re-cluster with different parameters**

**Option 3: Accept imbalance if biologically meaningful**

---

## Configuration Example

```python
from audhd_correlation.modeling.clustering import ConsensusClusteringPipeline

# Load config
import yaml
with open("configs/clustering/standard.yaml") as f:
    config = yaml.safe_load(f)

# Initialize pipeline
pipeline = ConsensusClusteringPipeline(
    use_hdbscan=config["clustering"]["use_hdbscan"],
    use_spectral=config["clustering"]["use_spectral"],
    use_bgmm=config["clustering"]["use_bgmm"],
    n_bootstrap=config["clustering"]["consensus"]["resamples"],
    random_state=config["clustering"]["consensus"]["random_state"],
)

# Fit with auto-k
pipeline.fit(
    X,
    generate_embeddings=True,
    auto_k_method=config["clustering"]["auto_k"]["method"],
    min_k=config["clustering"]["auto_k"]["min_k"],
    max_k=config["clustering"]["auto_k"]["max_k"],
)

# Handle noise
if config["clustering"]["noise_handling"]["strategy"] == "reassign":
    from audhd_correlation.modeling.clustering import reassign_noise_nearest_cluster
    labels = reassign_noise_nearest_cluster(X, pipeline.consensus_labels_)
else:
    labels = pipeline.consensus_labels_

# Check imbalance
if detect_imbalance(labels):
    print("Warning: Imbalanced clusters detected")
```

---

## Summary

| Feature | Tunable Parameters | Default |
|---------|-------------------|---------|
| **Auto-k** | `method`, `min_k`, `max_k`, `tolerance` | eigengap, 2, 10, 0.1 |
| **Noise Handling** | `strategy`, `reassign_method`, `min_confidence` | keep, nearest_cluster, 0.5 |
| **Randomness** | `random_state` (all methods) | 42 |
| **Bootstrap** | `n_resamples`, `threshold` | 100, 0.5 |
| **Embeddings** | `methods`, `perplexities`, `n_neighbors`, `min_dists` | [tsne, umap], [5,10,30,50], [5,15,30,50], [0.1,0.3,0.5] |

All parameters exposed via YAML config and programmatic API.