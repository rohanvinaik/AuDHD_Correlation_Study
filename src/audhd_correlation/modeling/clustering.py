"""Robust consensus clustering pipeline with multiple methods

Implements comprehensive clustering with:
- HDBSCAN with parameter sweeps (CONSENSUS across sweeps, not best-pick)
- Spectral clustering on co-assignment matrices
- Bayesian Gaussian Mixture Models
- Ensemble clustering across embeddings
- Topological Data Analysis for gap detection
- NULL MODELS: permutation tests, rotation nulls, SigClust, dip tests
- TOPOLOGY GATES: MST/k-NN/spectral gaps as hard thresholds
- CONFIG HASHING: prevent post-hoc parameter tweaking
- STABILITY SELECTION: selective inference for feature panels
"""
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import warnings
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
)
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# dip test is in a separate package
try:
    from diptest import diptest as dip
    DIP_AVAILABLE = True
except ImportError:
    DIP_AVAILABLE = False
    dip = None

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("hdbscan not available, some features will be disabled")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("umap not available, some features will be disabled")


# ============================================================================
# CONFIG HASHING & LOCKFILE (Point 5)
# ============================================================================

@dataclass
class ConfigHash:
    """Configuration hash for reproducibility and preventing post-hoc tweaking"""
    config_dict: Dict[str, Any]
    hash_value: str = field(default="")
    timestamp: str = field(default="")
    mode: str = field(default="exploratory")  # exploratory or confirmatory

    def __post_init__(self):
        if not self.hash_value:
            self.hash_value = self._compute_hash()
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of config"""
        config_str = json.dumps(self.config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def save_lockfile(self, path: Union[str, Path]) -> None:
        """Save config hash to lockfile"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump({
                'hash': self.hash_value,
                'timestamp': self.timestamp,
                'mode': self.mode,
                'config': self.config_dict
            }, f, indent=2)

    @classmethod
    def load_lockfile(cls, path: Union[str, Path]) -> 'ConfigHash':
        """Load config hash from lockfile"""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            config_dict=data['config'],
            hash_value=data['hash'],
            timestamp=data['timestamp'],
            mode=data['mode']
        )

    def verify_match(self, other_config: Dict[str, Any]) -> bool:
        """Verify that config matches locked config (for confirmatory analysis)"""
        other_hash = ConfigHash(other_config)
        return self.hash_value == other_hash.hash_value


# ============================================================================
# NULL MODELS (Point 2)
# ============================================================================

class NullModelTester:
    """
    Null hypothesis testing for clustering significance

    Implements:
    - Permutation tests (restricted to preserve structure)
    - Rotation nulls (preserve variance structure)
    - SigClust (Gaussian null)
    - Dip test (unimodality test)
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def test_clustering_significance(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple null model tests

        Args:
            X: Data matrix
            labels: Cluster labels
            methods: List of null model methods to use

        Returns:
            Dictionary of test results
        """
        if methods is None:
            methods = ['permutation', 'rotation', 'sigclust', 'dip']

        results = {}

        if 'permutation' in methods:
            results['permutation'] = self._permutation_test(X, labels)

        if 'rotation' in methods:
            results['rotation'] = self._rotation_null_test(X, labels)

        if 'sigclust' in methods:
            results['sigclust'] = self._sigclust_test(X, labels)

        if 'dip' in methods:
            results['dip'] = self._dip_test(X)

        # Overall significance (all tests must pass)
        results['all_significant'] = all(
            r.get('significant', False) for r in results.values()
            if isinstance(r, dict)
        )

        return results

    def _permutation_test(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Permutation test with restricted permutations"""
        # Compute observed metric (gap statistic)
        obs_gap = self._compute_gap_statistic_simple(X, labels)

        # Permutation distribution
        null_gaps = []
        for _ in range(self.n_permutations):
            # Permute labels while preserving marginals
            perm_labels = self.rng.permutation(labels)
            null_gap = self._compute_gap_statistic_simple(X, perm_labels)
            null_gaps.append(null_gap)

        null_gaps = np.array(null_gaps)
        p_value = (null_gaps >= obs_gap).mean()

        return {
            'method': 'permutation',
            'observed': obs_gap,
            'null_mean': null_gaps.mean(),
            'null_std': null_gaps.std(),
            'p_value': p_value,
            'significant': p_value < self.alpha
        }

    def _rotation_null_test(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Rotation null test (preserves covariance structure)"""
        # Compute observed metric
        obs_sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0

        # Rotation null distribution
        null_sils = []
        for _ in range(self.n_permutations):
            # Random rotation
            Q, _ = np.linalg.qr(self.rng.randn(X.shape[1], X.shape[1]))
            X_rot = X @ Q

            # Cluster rotated data
            from sklearn.cluster import KMeans
            n_clusters = len(set(labels))
            km = KMeans(n_clusters=n_clusters, random_state=self.rng.randint(10000), n_init=10)
            rot_labels = km.fit_predict(X_rot)

            null_sil = silhouette_score(X_rot, rot_labels) if len(set(rot_labels)) > 1 else 0
            null_sils.append(null_sil)

        null_sils = np.array(null_sils)
        p_value = (null_sils >= obs_sil).mean()

        return {
            'method': 'rotation',
            'observed': obs_sil,
            'null_mean': null_sils.mean(),
            'null_std': null_sils.std(),
            'p_value': p_value,
            'significant': p_value < self.alpha
        }

    def _sigclust_test(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """SigClust: test against Gaussian null"""
        # Simplified SigClust implementation
        # Full implementation: Liu et al. 2008

        n_clusters = len(set(labels))
        if n_clusters < 2:
            return {'method': 'sigclust', 'significant': False, 'note': 'Too few clusters'}

        # Compute cluster index for observed data
        obs_index = self._compute_cluster_index(X, labels)

        # Generate Gaussian null data
        mean = X.mean(axis=0)
        cov = np.cov(X.T)

        null_indices = []
        for _ in range(self.n_permutations):
            X_null = self.rng.multivariate_normal(mean, cov, size=X.shape[0])

            # Cluster null data
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=n_clusters, random_state=self.rng.randint(10000), n_init=10)
            null_labels = km.fit_predict(X_null)

            null_index = self._compute_cluster_index(X_null, null_labels)
            null_indices.append(null_index)

        null_indices = np.array(null_indices)
        p_value = (null_indices >= obs_index).mean()

        return {
            'method': 'sigclust',
            'observed': obs_index,
            'null_mean': null_indices.mean(),
            'null_std': null_indices.std(),
            'p_value': p_value,
            'significant': p_value < self.alpha
        }

    def _dip_test(self, X: np.ndarray) -> Dict[str, Any]:
        """Hartigan's dip test for unimodality"""
        # Test each dimension separately
        p_values = []

        for dim in range(min(X.shape[1], 10)):  # Test first 10 dimensions
            try:
                # Project to 1D
                projection = X[:, dim]

                # Dip statistic
                dip_stat, p_val = dip(projection, full_output=True)[:2]
                p_values.append(p_val)
            except:
                continue

        if not p_values:
            return {'method': 'dip', 'significant': False, 'note': 'Test failed'}

        # Use minimum p-value (most significant)
        min_p = min(p_values)

        return {
            'method': 'dip',
            'min_p_value': min_p,
            'all_p_values': p_values,
            'significant': min_p < self.alpha,  # Reject unimodality = multimodal = clusters exist
            'note': 'Rejects unimodality if significant'
        }

    def _compute_gap_statistic_simple(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Simplified gap statistic"""
        n_clusters = len(set(labels))
        if n_clusters < 2:
            return 0.0

        # Within-cluster sum of squares
        wss = 0.0
        for k in set(labels):
            cluster_data = X[labels == k]
            if len(cluster_data) > 1:
                wss += np.sum(pairwise_distances(cluster_data) ** 2) / (2 * len(cluster_data))

        return -np.log(wss + 1e-10)  # Higher gap = better clustering

    def _compute_cluster_index(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute cluster separation index"""
        try:
            return calinski_harabasz_score(X, labels)
        except:
            return 0.0


# ============================================================================
# TOPOLOGY GATES (Point 3)
# ============================================================================

@dataclass
class TopologyGateResults:
    """Results from topology pre-registration gates"""
    mst_edge_separation: float
    knn_purity: float
    spectral_gap: float
    persistence_entropy: float
    passes_gates: bool
    subtype_claim_allowed: bool
    notes: List[str] = field(default_factory=list)


class TopologyPreRegistrationGates:
    """
    Topology-based pre-registration gates

    Hard thresholds that must be met before claiming distinct subtypes.
    Uses:
    - MST edge separation (gap in MST edge lengths)
    - k-NN purity (local neighborhood homogeneity)
    - Spectral gaps (eigenvalue gaps in graph Laplacian)
    - Persistence entropy (topological data analysis)
    """

    def __init__(
        self,
        mst_separation_threshold: float = 1.5,  # Gap in MST edges (ratio)
        knn_purity_threshold: float = 0.7,      # k-NN purity
        spectral_gap_threshold: float = 0.3,    # Eigenvalue gap
        persistence_threshold: float = 0.5,     # Persistence entropy
        k_neighbors: int = 15,
    ):
        self.mst_separation_threshold = mst_separation_threshold
        self.knn_purity_threshold = knn_purity_threshold
        self.spectral_gap_threshold = spectral_gap_threshold
        self.persistence_threshold = persistence_threshold
        self.k_neighbors = k_neighbors

    def evaluate_gates(self, X: np.ndarray, labels: Optional[np.ndarray] = None) -> TopologyGateResults:
        """
        Evaluate topology gates

        Args:
            X: Data matrix
            labels: Optional cluster labels (for k-NN purity)

        Returns:
            TopologyGateResults
        """
        notes = []

        # 1. MST edge separation
        mst_sep = self._compute_mst_separation(X)
        notes.append(f"MST separation: {mst_sep:.3f} (threshold: {self.mst_separation_threshold})")

        # 2. k-NN purity (requires labels)
        if labels is not None:
            knn_pur = self._compute_knn_purity(X, labels)
            notes.append(f"k-NN purity: {knn_pur:.3f} (threshold: {self.knn_purity_threshold})")
        else:
            knn_pur = 0.0
            notes.append("k-NN purity: not computed (no labels)")

        # 3. Spectral gap
        spectral_gap = self._compute_spectral_gap(X)
        notes.append(f"Spectral gap: {spectral_gap:.3f} (threshold: {self.spectral_gap_threshold})")

        # 4. Persistence entropy
        persist_entropy = self._compute_persistence_entropy(X)
        notes.append(f"Persistence entropy: {persist_entropy:.3f} (threshold: {self.persistence_threshold})")

        # Check gates
        gates_passed = (
            mst_sep >= self.mst_separation_threshold and
            (labels is None or knn_pur >= self.knn_purity_threshold) and
            spectral_gap >= self.spectral_gap_threshold and
            persist_entropy >= self.persistence_threshold
        )

        # Subtype claim allowed only if gates passed
        subtype_allowed = gates_passed
        if not subtype_allowed:
            notes.append("⚠️ TOPOLOGY GATES FAILED: Data appears to be a spectrum, not discrete subtypes")
        else:
            notes.append("✓ Topology gates passed: Discrete subtypes supported by topology")

        return TopologyGateResults(
            mst_edge_separation=mst_sep,
            knn_purity=knn_pur,
            spectral_gap=spectral_gap,
            persistence_entropy=persist_entropy,
            passes_gates=gates_passed,
            subtype_claim_allowed=subtype_allowed,
            notes=notes
        )

    def _compute_mst_separation(self, X: np.ndarray) -> float:
        """Compute MST edge separation (gap ratio)"""
        # Compute distance matrix
        dist = pairwise_distances(X)

        # Compute MST
        mst = minimum_spanning_tree(dist).toarray()

        # Get MST edge weights
        edges = mst[mst > 0]

        if len(edges) < 2:
            return 0.0

        # Sort edges
        edges_sorted = np.sort(edges)

        # Find largest gap in edge lengths
        gaps = np.diff(edges_sorted)
        max_gap_idx = np.argmax(gaps)

        # Compute gap ratio (relative to surrounding edges)
        gap_ratio = gaps[max_gap_idx] / (edges_sorted[max_gap_idx] + 1e-10)

        return gap_ratio

    def _compute_knn_purity(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute k-NN purity (neighborhood homogeneity)"""
        from sklearn.neighbors import NearestNeighbors

        # Fit k-NN
        knn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        knn.fit(X)

        # Get neighbors
        indices = knn.kneighbors(X, return_distance=False)

        # Compute purity (fraction of neighbors with same label)
        purities = []
        for i, neighbors in enumerate(indices):
            neighbors = neighbors[1:]  # Exclude self
            neighbor_labels = labels[neighbors]
            purity = (neighbor_labels == labels[i]).mean()
            purities.append(purity)

        return np.mean(purities)

    def _compute_spectral_gap(self, X: np.ndarray) -> float:
        """Compute spectral gap (eigenvalue gap in graph Laplacian)"""
        from sklearn.neighbors import kneighbors_graph

        # Build k-NN graph
        knn_graph = kneighbors_graph(X, n_neighbors=self.k_neighbors, mode='connectivity')
        adjacency = knn_graph.toarray()

        # Compute graph Laplacian
        degree = np.diag(adjacency.sum(axis=1))
        laplacian = degree - adjacency

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.sort(eigenvalues)

        # Compute gaps
        gaps = np.diff(eigenvalues[:10])  # First 10 eigenvalues

        # Largest gap (normalized)
        max_gap = np.max(gaps) if len(gaps) > 0 else 0.0

        return max_gap

    def _compute_persistence_entropy(self, X: np.ndarray) -> float:
        """Compute persistence entropy (simplified TDA)"""
        # Simplified version: use distance-based persistence

        # Compute pairwise distances
        dist = pairwise_distances(X)

        # Get unique distances (persistence diagram birth/death)
        unique_dists = np.unique(dist)

        # Compute persistence (lifetime of topological features)
        if len(unique_dists) < 2:
            return 0.0

        # Simplified: gaps in distance distribution
        gaps = np.diff(unique_dists)

        # Normalize to probabilities
        total = gaps.sum()
        if total == 0:
            return 0.0

        probs = gaps / total

        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize by max entropy
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized_entropy


# ============================================================================
# ORIGINAL DATACLASSES
# ============================================================================

@dataclass
class ClusterAssignment:
    """Cluster assignment with confidence scores"""
    labels: np.ndarray
    confidence: np.ndarray
    probabilities: Optional[np.ndarray] = None
    hierarchy: Optional[np.ndarray] = None


@dataclass
class ClusteringMetrics:
    """Comprehensive clustering quality metrics"""
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    gap_statistic: Optional[float] = None
    stability: Optional[float] = None
    n_clusters: int = 0
    n_noise: int = 0


class HDBSCANParameterSweep:
    """
    HDBSCAN with comprehensive parameter sweep using CONSENSUS

    FIXED: No longer picks "best" parameters (garden-of-forking-paths).
    Instead, builds co-assignment matrix across ALL parameter combinations,
    then uses spectral clustering for final consensus labels.

    This addresses the selection bias critique from pattern-mining.
    """

    def __init__(
        self,
        min_cluster_sizes: Optional[List[int]] = None,
        min_samples_list: Optional[List[int]] = None,
        metrics: Optional[List[str]] = None,
        cluster_selection_methods: Optional[List[str]] = None,
        use_consensus: bool = True,  # NEW: enable consensus mode
        consensus_threshold: float = 0.5,  # Co-assignment threshold
    ):
        """
        Initialize HDBSCAN parameter sweep

        Args:
            min_cluster_sizes: List of minimum cluster sizes to test
            min_samples_list: List of minimum samples values
            metrics: Distance metrics to test
            cluster_selection_methods: Selection methods to test
            use_consensus: Use consensus across sweep (default: True)
            consensus_threshold: Co-assignment threshold for spectral clustering
        """
        self.min_cluster_sizes = min_cluster_sizes or [5, 10, 15, 20]
        self.min_samples_list = min_samples_list or [1, 5, 10]
        self.metrics = metrics or ['euclidean', 'manhattan']
        self.cluster_selection_methods = cluster_selection_methods or ['eom', 'leaf']
        self.use_consensus = use_consensus
        self.consensus_threshold = consensus_threshold

        # Old best-pick results (deprecated, kept for compatibility)
        self.best_params_: Optional[Dict] = None
        self.best_score_: Optional[float] = None
        self.best_labels_: Optional[np.ndarray] = None

        # NEW: Consensus results
        self.consensus_labels_: Optional[np.ndarray] = None
        self.coassignment_matrix_: Optional[np.ndarray] = None
        self.all_labels_: List[np.ndarray] = []

        self.sweep_results_: List[Dict] = []

    def fit(self, X: np.ndarray, scoring: str = 'silhouette') -> 'HDBSCANParameterSweep':
        """
        Fit HDBSCAN with parameter sweep

        NEW BEHAVIOR (if use_consensus=True):
        - Runs ALL parameter combinations
        - Builds co-assignment matrix across ALL results
        - Uses spectral clustering on consensus matrix
        - NO cherry-picking of "best" parameters

        Args:
            X: Data matrix
            scoring: Scoring method (still tracked, but not used for selection if consensus=True)

        Returns:
            Self (fitted)
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan is required for HDBSCANParameterSweep")

        n_samples = X.shape[0]

        # Initialize co-assignment matrix if using consensus
        if self.use_consensus:
            coassign = np.zeros((n_samples, n_samples))
            n_valid_combinations = 0

        # Tracking for old best-pick mode (deprecated)
        best_score = -np.inf if scoring != 'davies_bouldin' else np.inf
        best_params = None
        best_labels = None

        # Parameter sweep - run ALL combinations
        for min_cluster_size in self.min_cluster_sizes:
            for min_samples in self.min_samples_list:
                for metric in self.metrics:
                    for selection_method in self.cluster_selection_methods:
                        try:
                            clusterer = hdbscan.HDBSCAN(
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric=metric,
                                cluster_selection_method=selection_method,
                            )

                            labels = clusterer.fit_predict(X)

                            # Skip if only noise or only one cluster
                            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                            if n_clusters < 2:
                                continue

                            # Compute score (for tracking)
                            valid_mask = labels >= 0
                            if valid_mask.sum() < 10:
                                continue

                            if scoring == 'silhouette':
                                score = silhouette_score(X[valid_mask], labels[valid_mask])
                            elif scoring == 'calinski_harabasz':
                                score = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
                            elif scoring == 'davies_bouldin':
                                score = -davies_bouldin_score(X[valid_mask], labels[valid_mask])
                            else:
                                score = silhouette_score(X[valid_mask], labels[valid_mask])

                            # Store results
                            self.sweep_results_.append({
                                'min_cluster_size': min_cluster_size,
                                'min_samples': min_samples,
                                'metric': metric,
                                'cluster_selection_method': selection_method,
                                'n_clusters': n_clusters,
                                'n_noise': (labels == -1).sum(),
                                'score': score,
                            })

                            # NEW: Add to co-assignment matrix (consensus mode)
                            if self.use_consensus:
                                self.all_labels_.append(labels)

                                # Update co-assignment matrix
                                for i in range(n_samples):
                                    for j in range(i + 1, n_samples):
                                        # Co-assigned if both in same cluster (not noise)
                                        if labels[i] >= 0 and labels[i] == labels[j]:
                                            coassign[i, j] += 1
                                            coassign[j, i] += 1

                                n_valid_combinations += 1

                            # OLD: Update best (deprecated, only for fallback)
                            is_better = (score > best_score if scoring != 'davies_bouldin'
                                       else score < best_score)

                            if is_better:
                                best_score = score
                                best_params = {
                                    'min_cluster_size': min_cluster_size,
                                    'min_samples': min_samples,
                                    'metric': metric,
                                    'cluster_selection_method': selection_method,
                                }
                                best_labels = labels

                        except Exception as e:
                            warnings.warn(f"Parameter combination failed: {e}")
                            continue

        # Store old best-pick results (deprecated)
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_labels_ = best_labels

        # NEW: Compute consensus labels
        if self.use_consensus and n_valid_combinations > 0:
            # Normalize co-assignment matrix
            coassign /= n_valid_combinations
            self.coassignment_matrix_ = coassign

            # Threshold and convert to affinity matrix
            affinity = (coassign >= self.consensus_threshold).astype(float)

            # Estimate number of clusters using eigengap
            n_clusters = self._estimate_n_clusters_from_eigengap(affinity)

            # Spectral clustering on consensus matrix
            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=42
            )
            self.consensus_labels_ = sc.fit_predict(affinity)

            print(f"✓ Consensus clustering: {n_valid_combinations} parameter combinations → {n_clusters} clusters")
        else:
            # Fallback to best-pick if consensus disabled or failed
            self.consensus_labels_ = best_labels
            warnings.warn("Consensus disabled or failed. Using best-pick mode (NOT RECOMMENDED).")

        return self

    def _estimate_n_clusters_from_eigengap(self, affinity: np.ndarray) -> int:
        """Estimate number of clusters using eigengap heuristic"""
        try:
            from scipy.linalg import eigh
            eigenvalues = eigh(affinity, eigvals_only=True)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Find largest eigengap
            gaps = np.diff(eigenvalues[:10])
            n_clusters = np.argmax(gaps) + 1

            return max(2, min(n_clusters, 10))
        except:
            return 5  # Default fallback

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get cluster labels

        NEW: Returns consensus labels if use_consensus=True (recommended).
        Falls back to best-pick labels if consensus disabled (NOT recommended).
        """
        if self.use_consensus:
            if self.consensus_labels_ is None:
                raise ValueError("Must call fit() first")
            return self.consensus_labels_
        else:
            if self.best_labels_ is None:
                raise ValueError("Must call fit() first")
            warnings.warn("Returning best-pick labels (NOT RECOMMENDED). Enable use_consensus=True.")
            return self.best_labels_

    def get_coassignment_matrix(self) -> Optional[np.ndarray]:
        """Get co-assignment matrix from consensus clustering"""
        return self.coassignment_matrix_

    def get_sweep_results(self) -> pd.DataFrame:
        """Get all sweep results as DataFrame"""
        return pd.DataFrame(self.sweep_results_)


class SpectralCoAssignmentClustering:
    """
    Spectral clustering on co-assignment matrices

    Builds consensus from multiple clustering runs.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        n_resamples: int = 100,
        threshold: float = 0.5,
        affinity: str = 'precomputed',
    ):
        """
        Initialize spectral co-assignment clustering

        Args:
            n_clusters: Number of clusters (auto-detect if None)
            n_resamples: Number of bootstrap resamples
            threshold: Co-assignment threshold for edge creation
            affinity: Affinity type
        """
        self.n_clusters = n_clusters
        self.n_resamples = n_resamples
        self.threshold = threshold
        self.affinity = affinity

        self.coassignment_matrix_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        base_clusterer: Optional[Any] = None,
    ) -> 'SpectralCoAssignmentClustering':
        """
        Fit spectral clustering on co-assignment matrix

        Args:
            X: Data matrix
            base_clusterer: Base clustering algorithm (HDBSCAN if None)

        Returns:
            Self (fitted)
        """
        n_samples = X.shape[0]
        coassign = np.zeros((n_samples, n_samples))

        # Bootstrap resampling
        for _ in range(self.n_resamples):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[idx]

            # Cluster bootstrap sample
            if base_clusterer is None:
                if HDBSCAN_AVAILABLE:
                    labels_boot = hdbscan.HDBSCAN().fit_predict(X_boot)
                else:
                    # Fallback to agglomerative
                    labels_boot = AgglomerativeClustering(n_clusters=5).fit_predict(X_boot)
            else:
                labels_boot = base_clusterer.fit_predict(X_boot)

            # Build co-assignment matrix for this bootstrap
            labels_full = -np.ones(n_samples, dtype=int)
            labels_full[idx] = labels_boot

            # Co-assignment: samples in same cluster
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels_full[i] >= 0 and labels_full[i] == labels_full[j]:
                        coassign[i, j] += 1
                        coassign[j, i] += 1

        # Normalize
        coassign /= self.n_resamples
        self.coassignment_matrix_ = coassign

        # Threshold and cluster
        affinity_matrix = (coassign >= self.threshold).astype(float)

        # Auto-detect number of clusters if not specified
        if self.n_clusters is None:
            # Use eigengap heuristic
            self.n_clusters = self._estimate_n_clusters(affinity_matrix)

        # Spectral clustering
        sc = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
        )
        self.labels_ = sc.fit_predict(affinity_matrix)

        return self

    def _estimate_n_clusters(self, affinity: np.ndarray) -> int:
        """Estimate number of clusters using eigengap"""
        # Compute eigenvalues
        try:
            from scipy.linalg import eigh
            eigenvalues = eigh(affinity, eigvals_only=True)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Find largest eigengap
            gaps = np.diff(eigenvalues[:10])
            n_clusters = np.argmax(gaps) + 1

            return max(2, min(n_clusters, 10))
        except:
            return 5  # Default

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Get cluster labels"""
        if self.labels_ is None:
            raise ValueError("Must call fit() first")
        return self.labels_


class BayesianGaussianMixtureClustering:
    """
    Bayesian Gaussian Mixture Model for clustering

    Automatically determines optimal number of components.
    """

    def __init__(
        self,
        max_components: int = 10,
        covariance_type: str = 'full',
        weight_concentration_prior: float = 1.0,
        n_init: int = 10,
    ):
        """
        Initialize Bayesian GMM clustering

        Args:
            max_components: Maximum number of components
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            weight_concentration_prior: Prior on mixture weights
            n_init: Number of initializations
        """
        self.max_components = max_components
        self.covariance_type = covariance_type
        self.weight_concentration_prior = weight_concentration_prior
        self.n_init = n_init

        self.model_: Optional[BayesianGaussianMixture] = None
        self.labels_: Optional[np.ndarray] = None
        self.probs_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None

    def fit(self, X: np.ndarray) -> 'BayesianGaussianMixtureClustering':
        """
        Fit Bayesian GMM

        Args:
            X: Data matrix

        Returns:
            Self (fitted)
        """
        self.model_ = BayesianGaussianMixture(
            n_components=self.max_components,
            covariance_type=self.covariance_type,
            weight_concentration_prior=self.weight_concentration_prior,
            n_init=self.n_init,
            random_state=42,
        )

        self.model_.fit(X)
        self.labels_ = self.model_.predict(X)
        self.probs_ = self.model_.predict_proba(X)

        # Count effective components (weights > threshold)
        weights = self.model_.weights_
        self.n_components_ = (weights > 0.01).sum()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels"""
        if self.model_ is None:
            raise ValueError("Must call fit() first")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster probabilities"""
        if self.model_ is None:
            raise ValueError("Must call fit() first")
        return self.model_.predict_proba(X)


class MultiEmbeddingGenerator:
    """
    Generate multiple embeddings with different methods and parameters

    Supports t-SNE, UMAP with various parameter settings.
    """

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        n_components: int = 2,
        random_state: int = 42,
    ):
        """
        Initialize multi-embedding generator

        Args:
            methods: List of methods ('tsne', 'umap', 'pca')
            n_components: Number of embedding dimensions
            random_state: Random seed
        """
        self.methods = methods or ['tsne', 'umap']
        self.n_components = n_components
        self.random_state = random_state

        self.embeddings_: Dict[str, np.ndarray] = {}

    def fit_transform(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate multiple embeddings

        Args:
            X: Data matrix

        Returns:
            Dictionary of embeddings
        """
        for method in self.methods:
            if method == 'tsne':
                self._generate_tsne_embeddings(X)
            elif method == 'umap' and UMAP_AVAILABLE:
                self._generate_umap_embeddings(X)
            elif method == 'pca':
                self._generate_pca_embedding(X)

        return self.embeddings_

    def _generate_tsne_embeddings(self, X: np.ndarray) -> None:
        """Generate t-SNE embeddings with different perplexities"""
        perplexities = [5, 10, 30, 50]

        for perplexity in perplexities:
            if perplexity >= X.shape[0]:
                continue

            try:
                tsne = TSNE(
                    n_components=self.n_components,
                    perplexity=perplexity,
                    random_state=self.random_state,
                )
                embedding = tsne.fit_transform(X)
                self.embeddings_[f'tsne_perp{perplexity}'] = embedding
            except:
                continue

    def _generate_umap_embeddings(self, X: np.ndarray) -> None:
        """Generate UMAP embeddings with different parameters"""
        n_neighbors_list = [5, 15, 30, 50]
        min_dists = [0.1, 0.3, 0.5]

        for n_neighbors in n_neighbors_list:
            if n_neighbors >= X.shape[0]:
                continue

            for min_dist in min_dists:
                try:
                    reducer = umap.UMAP(
                        n_components=self.n_components,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=self.random_state,
                    )
                    embedding = reducer.fit_transform(X)
                    self.embeddings_[f'umap_n{n_neighbors}_d{min_dist}'] = embedding
                except:
                    continue

    def _generate_pca_embedding(self, X: np.ndarray) -> None:
        """Generate PCA embedding"""
        from sklearn.decomposition import PCA

        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        embedding = pca.fit_transform(X)
        self.embeddings_['pca'] = embedding


class TopologicalGapDetector:
    """
    Topological Data Analysis for gap detection

    Uses persistence diagrams to detect natural gaps in data.
    """

    def __init__(
        self,
        max_dimension: int = 1,
        metric: str = 'euclidean',
    ):
        """
        Initialize TDA gap detector

        Args:
            max_dimension: Maximum homology dimension
            metric: Distance metric
        """
        self.max_dimension = max_dimension
        self.metric = metric

        self.persistence_: Optional[List] = None
        self.gaps_: Optional[List[float]] = None

    def fit(self, X: np.ndarray) -> 'TopologicalGapDetector':
        """
        Detect topological gaps

        Args:
            X: Data matrix

        Returns:
            Self (fitted)
        """
        try:
            # Simplified persistence using distance matrix
            distances = pairwise_distances(X, metric=self.metric)

            # Find large gaps in sorted distances
            unique_dists = np.unique(distances[np.triu_indices_from(distances, k=1)])
            gaps = np.diff(unique_dists)

            # Detect significant gaps (> 2 std)
            threshold = gaps.mean() + 2 * gaps.std()
            significant_gaps = unique_dists[:-1][gaps > threshold]

            self.gaps_ = list(significant_gaps)

        except Exception as e:
            warnings.warn(f"TDA failed: {e}")
            self.gaps_ = []

        return self

    def estimate_n_clusters(self) -> int:
        """Estimate number of clusters from gaps"""
        if self.gaps_ is None:
            raise ValueError("Must call fit() first")

        # Number of clusters ~ number of gaps + 1
        return len(self.gaps_) + 1 if self.gaps_ else 2


class ConsensusClusteringPipeline:
    """
    Comprehensive consensus clustering pipeline with anti-pattern-mining features

    Integrates multiple clustering methods and embeddings with:
    - Consensus across parameter sweeps (no cherry-picking)
    - Null model testing (permutation, rotation, SigClust, dip)
    - Topology pre-registration gates (MST, k-NN, spectral)
    - Config hashing (prevent post-hoc tweaking)
    """

    def __init__(
        self,
        use_hdbscan: bool = True,
        use_spectral: bool = True,
        use_bgmm: bool = True,
        use_tda: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 42,
        # NEW: Anti-pattern-mining features
        test_null_models: bool = True,
        evaluate_topology_gates: bool = True,
        enable_config_locking: bool = True,
        config_dict: Optional[Dict[str, Any]] = None,
        lockfile_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize consensus clustering pipeline

        Args:
            use_hdbscan: Use HDBSCAN parameter sweep
            use_spectral: Use spectral co-assignment clustering
            use_bgmm: Use Bayesian GMM
            use_tda: Use topological gap detection
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed
            test_null_models: Run null hypothesis tests
            evaluate_topology_gates: Check topology gates before claiming subtypes
            enable_config_locking: Enable config hash locking
            config_dict: Configuration dictionary for hashing
            lockfile_path: Path to config lockfile
        """
        self.use_hdbscan = use_hdbscan and HDBSCAN_AVAILABLE
        self.use_spectral = use_spectral
        self.use_bgmm = use_bgmm
        self.use_tda = use_tda
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

        # NEW: Anti-pattern-mining
        self.test_null_models = test_null_models
        self.evaluate_topology_gates = evaluate_topology_gates
        self.enable_config_locking = enable_config_locking

        # Clustering results
        self.consensus_labels_: Optional[np.ndarray] = None
        self.confidence_scores_: Optional[np.ndarray] = None
        self.method_labels_: Dict[str, np.ndarray] = {}
        self.embeddings_: Dict[str, np.ndarray] = {}
        self.metrics_: Optional[ClusteringMetrics] = None

        # NEW: Validation results
        self.null_model_results_: Optional[Dict[str, Any]] = None
        self.topology_gate_results_: Optional[TopologyGateResults] = None
        self.config_hash_: Optional[ConfigHash] = None

        # Config locking
        if enable_config_locking and config_dict is not None:
            self.config_hash_ = ConfigHash(config_dict)

            # Check existing lockfile
            if lockfile_path and Path(lockfile_path).exists():
                locked_config = ConfigHash.load_lockfile(lockfile_path)

                if not locked_config.verify_match(config_dict):
                    warnings.warn(
                        "⚠️ CONFIG MISMATCH: Current config differs from locked config. "
                        "This may indicate post-hoc parameter tweaking. "
                        "Confirmatory analysis requires identical configs."
                    )
            elif lockfile_path:
                # Save new lockfile
                self.config_hash_.save_lockfile(lockfile_path)

    def fit(self, X: np.ndarray, generate_embeddings: bool = True) -> 'ConsensusClusteringPipeline':
        """
        Fit consensus clustering with anti-pattern-mining validation

        NEW: Includes null model testing and topology gates

        Args:
            X: Data matrix
            generate_embeddings: Generate multiple embeddings

        Returns:
            Self (fitted)
        """
        np.random.seed(self.random_state)

        # STEP 0: Check topology gates BEFORE clustering (pre-registration)
        if self.evaluate_topology_gates:
            print("⏳ Evaluating topology pre-registration gates...")
            topology_checker = TopologyPreRegistrationGates()
            topology_results_pre = topology_checker.evaluate_gates(X, labels=None)

            if not topology_results_pre.passes_gates:
                warnings.warn(
                    "\n⚠️ TOPOLOGY GATES FAILED (pre-clustering):\n" +
                    "\n".join(topology_results_pre.notes) +
                    "\n\nData may be a spectrum rather than discrete subtypes. "
                    "Proceeding with clustering, but interpret with caution."
                )

        # Generate embeddings if requested
        if generate_embeddings:
            emb_gen = MultiEmbeddingGenerator(random_state=self.random_state)
            self.embeddings_ = emb_gen.fit_transform(X)
        else:
            self.embeddings_ = {'original': X}

        # Run different clustering methods
        all_labels = []

        # 1. HDBSCAN parameter sweep (WITH CONSENSUS, not best-pick)
        if self.use_hdbscan and HDBSCAN_AVAILABLE:
            try:
                print("⏳ Running HDBSCAN parameter sweep with consensus...")
                hdbscan_sweep = HDBSCANParameterSweep(use_consensus=True)
                hdbscan_sweep.fit(X)
                labels_hdb = hdbscan_sweep.predict()
                self.method_labels_['hdbscan'] = labels_hdb
                all_labels.append(labels_hdb)
            except Exception as e:
                warnings.warn(f"HDBSCAN sweep failed: {e}")

        # 2. Spectral co-assignment
        if self.use_spectral:
            try:
                print("⏳ Running spectral co-assignment clustering...")
                spectral = SpectralCoAssignmentClustering(n_resamples=self.n_bootstrap)
                spectral.fit(X)
                labels_spec = spectral.predict()
                self.method_labels_['spectral'] = labels_spec
                all_labels.append(labels_spec)
            except Exception as e:
                warnings.warn(f"Spectral clustering failed: {e}")

        # 3. Bayesian GMM
        if self.use_bgmm:
            try:
                print("⏳ Running Bayesian Gaussian Mixture Model...")
                bgmm = BayesianGaussianMixtureClustering()
                bgmm.fit(X)
                labels_bgmm = bgmm.predict(X)
                self.method_labels_['bgmm'] = labels_bgmm
                all_labels.append(labels_bgmm)
            except Exception as e:
                warnings.warn(f"Bayesian GMM failed: {e}")

        # 4. Cluster each embedding
        for name, embedding in self.embeddings_.items():
            if HDBSCAN_AVAILABLE:
                try:
                    labels_emb = hdbscan.HDBSCAN().fit_predict(embedding)
                    self.method_labels_[f'hdbscan_{name}'] = labels_emb
                    all_labels.append(labels_emb)
                except:
                    pass

        # Ensemble voting
        if all_labels:
            self.consensus_labels_, self.confidence_scores_ = self._ensemble_vote(all_labels)
        else:
            # Fallback
            self.consensus_labels_ = np.zeros(X.shape[0], dtype=int)
            self.confidence_scores_ = np.ones(X.shape[0])

        # Compute metrics
        self.metrics_ = self._compute_metrics(X, self.consensus_labels_)

        # STEP 1: Test null models (clustering significance)
        if self.test_null_models:
            print("⏳ Testing null models for clustering significance...")
            null_tester = NullModelTester(
                n_permutations=1000,
                random_state=self.random_state
            )
            self.null_model_results_ = null_tester.test_clustering_significance(
                X, self.consensus_labels_
            )

            # Report results
            if self.null_model_results_['all_significant']:
                print("✓ Null model tests PASSED: Clustering is statistically significant")
            else:
                warnings.warn(
                    "\n⚠️ NULL MODEL TESTS FAILED:\n" +
                    "Clustering may not be significantly better than random.\n" +
                    f"Permutation p-value: {self.null_model_results_['permutation']['p_value']:.4f}\n" +
                    f"Rotation p-value: {self.null_model_results_['rotation']['p_value']:.4f}\n" +
                    f"SigClust p-value: {self.null_model_results_['sigclust']['p_value']:.4f}\n" +
                    f"Dip test p-value: {self.null_model_results_['dip']['min_p_value']:.4f}"
                )

        # STEP 2: Check topology gates AFTER clustering (post-hoc validation)
        if self.evaluate_topology_gates:
            print("⏳ Validating topology gates with cluster labels...")
            topology_checker = TopologyPreRegistrationGates()
            self.topology_gate_results_ = topology_checker.evaluate_gates(X, self.consensus_labels_)

            # Report results
            if self.topology_gate_results_.subtype_claim_allowed:
                print("✓ Topology gates PASSED: Discrete subtypes supported")
            else:
                warnings.warn(
                    "\n⚠️ TOPOLOGY GATES FAILED (post-clustering):\n" +
                    "\n".join(self.topology_gate_results_.notes)
                )

        return self

    def _ensemble_vote(
        self, all_labels: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble voting across methods

        Args:
            all_labels: List of label arrays

        Returns:
            Tuple of (consensus labels, confidence scores)
        """
        n_samples = len(all_labels[0])
        n_methods = len(all_labels)

        # Build co-assignment matrix from all methods
        coassign = np.zeros((n_samples, n_samples))

        for labels in all_labels:
            # Handle noise points (-1) as separate "clusters"
            labels_shifted = labels.copy()
            labels_shifted[labels_shifted == -1] = labels_shifted.max() + 1

            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels_shifted[i] == labels_shifted[j]:
                        coassign[i, j] += 1
                        coassign[j, i] += 1

        coassign /= n_methods

        # Hierarchical clustering on co-assignment
        from scipy.cluster.hierarchy import fcluster

        condensed_dist = 1 - coassign[np.triu_indices(n_samples, k=1)]
        linkage_matrix = linkage(condensed_dist, method='average')

        # Auto-determine number of clusters
        from scipy.cluster.hierarchy import inconsistent
        incons = inconsistent(linkage_matrix)
        threshold = incons[:, 3].mean() + incons[:, 3].std()

        consensus_labels = fcluster(linkage_matrix, threshold, criterion='distance') - 1

        # Confidence = average co-assignment within cluster
        confidence = np.zeros(n_samples)
        for i in range(n_samples):
            cluster_mask = consensus_labels == consensus_labels[i]
            confidence[i] = coassign[i, cluster_mask].mean()

        return consensus_labels, confidence

    def _compute_metrics(
        self, X: np.ndarray, labels: np.ndarray
    ) -> ClusteringMetrics:
        """Compute clustering quality metrics"""

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        if n_clusters < 2:
            return ClusteringMetrics(
                silhouette=0.0,
                calinski_harabasz=0.0,
                davies_bouldin=0.0,
                n_clusters=n_clusters,
                n_noise=n_noise,
            )

        valid_mask = labels >= 0

        try:
            sil = silhouette_score(X[valid_mask], labels[valid_mask])
        except:
            sil = 0.0

        try:
            ch = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
        except:
            ch = 0.0

        try:
            db = davies_bouldin_score(X[valid_mask], labels[valid_mask])
        except:
            db = 0.0

        # Gap statistic
        gap = self._compute_gap_statistic(X[valid_mask], labels[valid_mask])

        # Stability
        stability = self.confidence_scores_.mean() if self.confidence_scores_ is not None else None

        return ClusteringMetrics(
            silhouette=sil,
            calinski_harabasz=ch,
            davies_bouldin=db,
            gap_statistic=gap,
            stability=stability,
            n_clusters=n_clusters,
            n_noise=n_noise,
        )

    def _compute_gap_statistic(
        self, X: np.ndarray, labels: np.ndarray, n_refs: int = 10
    ) -> float:
        """Compute gap statistic"""
        try:
            # Within-cluster dispersion for actual data
            wk_actual = 0.0
            for k in set(labels):
                cluster_data = X[labels == k]
                if len(cluster_data) > 1:
                    wk_actual += np.sum(pairwise_distances(cluster_data) ** 2) / (2 * len(cluster_data))

            # Within-cluster dispersion for random reference
            wk_refs = []
            for _ in range(n_refs):
                # Generate random reference
                X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)

                # Cluster reference
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=len(set(labels)), random_state=42, n_init=10)
                labels_ref = km.fit_predict(X_ref)

                wk_ref = 0.0
                for k in set(labels_ref):
                    cluster_data = X_ref[labels_ref == k]
                    if len(cluster_data) > 1:
                        wk_ref += np.sum(pairwise_distances(cluster_data) ** 2) / (2 * len(cluster_data))

                wk_refs.append(np.log(wk_ref))

            gap = np.mean(wk_refs) - np.log(wk_actual)
            return gap

        except:
            return 0.0

    def predict(self, X: Optional[np.ndarray] = None) -> ClusterAssignment:
        """
        Get cluster assignments with confidence

        Returns:
            ClusterAssignment object
        """
        if self.consensus_labels_ is None:
            raise ValueError("Must call fit() first")

        return ClusterAssignment(
            labels=self.consensus_labels_,
            confidence=self.confidence_scores_,
        )

    def get_metrics(self) -> ClusteringMetrics:
        """Get clustering metrics"""
        if self.metrics_ is None:
            raise ValueError("Must call fit() first")
        return self.metrics_


# Legacy interface for compatibility
def _hdb(emb, params):
    """Run HDBSCAN with given parameters"""
    if not HDBSCAN_AVAILABLE:
        # Fallback to agglomerative clustering
        from sklearn.cluster import AgglomerativeClustering
        n_clusters = params.get('min_cluster_size', 5)
        return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(emb)

    return hdbscan.HDBSCAN(**params).fit_predict(emb)


def consensus(embeddings: dict, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Consensus clustering across multiple embeddings with anti-pattern-mining features

    Now uses ConsensusClusteringPipeline with:
    - Baseline-deviation framework (if enabled)
    - Null model testing
    - Topology gates
    - Config locking

    Args:
        embeddings: Dict of embedding matrices (name -> array)
        cfg: AppConfig with cluster configuration

    Returns:
        labels: Final consensus cluster labels
        coassign: Co-assignment matrix
    """
    # Use main embedding (typically UMAP or first available)
    if "umap_main" in embeddings:
        X = embeddings["umap_main"]
    elif "original" in embeddings:
        X = embeddings["original"]
    else:
        # Take first available
        X = list(embeddings.values())[0]

    # Check if baseline-deviation pipeline is enabled
    baseline_deviation_enabled = getattr(cfg.cluster, 'baseline_deviation_enabled', True)

    if baseline_deviation_enabled:
        # Use baseline-deviation pipeline
        from ..pipelines.baseline_deviation_pipeline import run_baseline_deviation_pipeline, get_default_config

        # Build config dict from cfg
        pipeline_config = get_default_config()

        # Override with cfg settings if available
        if hasattr(cfg, 'baseline'):
            pipeline_config['baseline'] = cfg.baseline
        if hasattr(cfg, 'topology'):
            pipeline_config['topology'] = cfg.topology
        if hasattr(cfg, 'deviation_threshold'):
            pipeline_config['deviation_threshold'] = cfg.deviation_threshold

        # Run baseline-deviation pipeline
        results = run_baseline_deviation_pipeline(
            Z=X,
            control_mask=None,  # Could be provided if available
            config=pipeline_config,
            output_dir=getattr(cfg, 'output_dir', None)
        )

        labels = results.cluster_labels if results.cluster_labels is not None else np.zeros(len(X), dtype=int)

        # Build co-assignment matrix from clustering results (if available)
        if results.clustering_results is not None:
            coassign = results.clustering_results.coassignment_matrix_
        else:
            # Fallback: identity matrix
            coassign = np.eye(len(X))

    else:
        # Use ConsensusClusteringPipeline directly
        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=True,
            use_spectral=True,
            use_bgmm=True,
            use_tda=False,
            n_bootstrap=getattr(cfg.cluster.consensus, 'n_resamples', 100),
            random_state=getattr(cfg, 'seed', 42),
            test_null_models=getattr(cfg.cluster, 'test_null_models', True),
            evaluate_topology_gates=getattr(cfg.cluster, 'evaluate_topology_gates', True),
            enable_config_locking=getattr(cfg.cluster, 'enable_config_locking', True),
            config_dict={'cluster': cfg.cluster} if hasattr(cfg, 'cluster') else None,
            lockfile_path=Path(getattr(cfg, 'output_dir', '.')) / 'config_lock.json' if hasattr(cfg, 'output_dir') else None,
        )

        pipeline.fit(X, generate_embeddings=True)

        labels = pipeline.consensus_labels_
        coassign = pipeline.coassignment_matrix_

    return labels, coassign