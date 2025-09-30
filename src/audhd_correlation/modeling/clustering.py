"""Robust consensus clustering pipeline with multiple methods

Implements comprehensive clustering with:
- HDBSCAN with parameter sweeps
- Spectral clustering on co-assignment matrices
- Bayesian Gaussian Mixture Models
- Ensemble clustering across embeddings
- Topological Data Analysis for gap detection
"""
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import warnings

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
    HDBSCAN with comprehensive parameter sweep

    Tests multiple parameter combinations to find optimal clustering.
    """

    def __init__(
        self,
        min_cluster_sizes: Optional[List[int]] = None,
        min_samples_list: Optional[List[int]] = None,
        metrics: Optional[List[str]] = None,
        cluster_selection_methods: Optional[List[str]] = None,
    ):
        """
        Initialize HDBSCAN parameter sweep

        Args:
            min_cluster_sizes: List of minimum cluster sizes to test
            min_samples_list: List of minimum samples values
            metrics: Distance metrics to test
            cluster_selection_methods: Selection methods to test
        """
        self.min_cluster_sizes = min_cluster_sizes or [5, 10, 15, 20]
        self.min_samples_list = min_samples_list or [1, 5, 10]
        self.metrics = metrics or ['euclidean', 'manhattan']
        self.cluster_selection_methods = cluster_selection_methods or ['eom', 'leaf']

        self.best_params_: Optional[Dict] = None
        self.best_score_: Optional[float] = None
        self.best_labels_: Optional[np.ndarray] = None
        self.sweep_results_: List[Dict] = []

    def fit(self, X: np.ndarray, scoring: str = 'silhouette') -> 'HDBSCANParameterSweep':
        """
        Fit HDBSCAN with parameter sweep

        Args:
            X: Data matrix
            scoring: Scoring method ('silhouette', 'calinski_harabasz', 'davies_bouldin')

        Returns:
            Self (fitted)
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan is required for HDBSCANParameterSweep")

        best_score = -np.inf if scoring != 'davies_bouldin' else np.inf
        best_params = None
        best_labels = None

        # Parameter sweep
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

                            # Compute score
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

                            # Update best
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

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_labels_ = best_labels

        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Get best cluster labels"""
        if self.best_labels_ is None:
            raise ValueError("Must call fit() first")
        return self.best_labels_

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
    Comprehensive consensus clustering pipeline

    Integrates multiple clustering methods and embeddings.
    """

    def __init__(
        self,
        use_hdbscan: bool = True,
        use_spectral: bool = True,
        use_bgmm: bool = True,
        use_tda: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 42,
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
        """
        self.use_hdbscan = use_hdbscan and HDBSCAN_AVAILABLE
        self.use_spectral = use_spectral
        self.use_bgmm = use_bgmm
        self.use_tda = use_tda
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

        self.consensus_labels_: Optional[np.ndarray] = None
        self.confidence_scores_: Optional[np.ndarray] = None
        self.method_labels_: Dict[str, np.ndarray] = {}
        self.embeddings_: Dict[str, np.ndarray] = {}
        self.metrics_: Optional[ClusteringMetrics] = None

    def fit(self, X: np.ndarray, generate_embeddings: bool = True) -> 'ConsensusClusteringPipeline':
        """
        Fit consensus clustering

        Args:
            X: Data matrix
            generate_embeddings: Generate multiple embeddings

        Returns:
            Self (fitted)
        """
        np.random.seed(self.random_state)

        # Generate embeddings if requested
        if generate_embeddings:
            emb_gen = MultiEmbeddingGenerator(random_state=self.random_state)
            self.embeddings_ = emb_gen.fit_transform(X)
        else:
            self.embeddings_ = {'original': X}

        # Run different clustering methods
        all_labels = []

        # 1. HDBSCAN parameter sweep
        if self.use_hdbscan and HDBSCAN_AVAILABLE:
            try:
                hdbscan_sweep = HDBSCANParameterSweep()
                hdbscan_sweep.fit(X)
                labels_hdb = hdbscan_sweep.predict()
                self.method_labels_['hdbscan'] = labels_hdb
                all_labels.append(labels_hdb)
            except:
                pass

        # 2. Spectral co-assignment
        if self.use_spectral:
            try:
                spectral = SpectralCoAssignmentClustering(n_resamples=self.n_bootstrap)
                spectral.fit(X)
                labels_spec = spectral.predict()
                self.method_labels_['spectral'] = labels_spec
                all_labels.append(labels_spec)
            except:
                pass

        # 3. Bayesian GMM
        if self.use_bgmm:
            try:
                bgmm = BayesianGaussianMixtureClustering()
                bgmm.fit(X)
                labels_bgmm = bgmm.predict(X)
                self.method_labels_['bgmm'] = labels_bgmm
                all_labels.append(labels_bgmm)
            except:
                pass

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
    Consensus clustering across multiple embeddings with resampling (legacy interface)

    Args:
        embeddings: Dict of embedding matrices (name -> array)
        cfg: AppConfig with cluster.consensus and cluster.clusterers

    Returns:
        labels: Final consensus cluster labels
        coassign: Co-assignment matrix
    """
    coassign = None
    labels_store = {}

    for name, emb in embeddings.items():
        for _ in range(cfg.cluster.consensus["resamples"]):
            idx = np.random.choice(len(emb), len(emb), replace=True)
            lab = _hdb(emb[idx], cfg.cluster.clusterers["hdbscan_main"])
            full = -np.ones(len(emb), int)
            full[idx] = lab
            labels_store.setdefault(name, []).append(full)
            mat = (full[:, None] == full[None, :]) & (full[:, None] >= 0)
            coassign = mat if coassign is None else coassign + mat

    coassign = coassign / (len(embeddings) * cfg.cluster.consensus["resamples"])

    # Spectral clustering on thresholded co-assignment matrix
    n_clusters = cfg.cluster.get('n_clusters', None)
    if n_clusters is None:
        # Auto-detect
        n_clusters = 5

    sc = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans"
    )
    labels = sc.fit_predict(
        (coassign >= cfg.cluster.consensus["threshold"]).astype(float)
    )

    return labels, coassign