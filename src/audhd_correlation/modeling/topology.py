"""Topology analysis tools for testing separation vs spectrum hypothesis

Provides comprehensive topological analysis to determine whether data exhibits
discrete clusters (separation) or continuous variation (spectrum).
"""
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.stats import kstest, mannwhitneyu, anderson_ksamp
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import pairwise_distances

# Optional dependencies
try:
    import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    warnings.warn("ripser not available, persistent homology will be disabled")


@dataclass
class GapScore:
    """Gap score quantifying separation between clusters"""
    score: float  # 0 = spectrum, 1 = separated
    within_cluster_distance: float
    between_cluster_distance: float
    gap_statistic: float
    p_value: float
    interpretation: str  # "separated", "spectrum", "intermediate"


@dataclass
class PersistenceResult:
    """Persistent homology results"""
    diagrams: List[np.ndarray]  # Persistence diagrams for each dimension
    lifetimes: Dict[int, np.ndarray]  # Lifetimes by dimension
    bottleneck_distances: Optional[np.ndarray] = None
    persistence_entropy: Optional[Dict[int, float]] = None
    interpretation: Optional[str] = None


@dataclass
class TopologyAnalysisResult:
    """Comprehensive topology analysis results"""
    gap_scores: GapScore
    mst_analysis: Dict[str, Any]
    knn_connectivity: Dict[str, Any]
    spectral_gaps: Dict[str, Any]
    persistence: Optional[PersistenceResult] = None
    hypothesis_test: Optional[Dict[str, float]] = None
    overall_interpretation: Optional[str] = None


class MinimumSpanningTreeAnalyzer:
    """
    Minimum spanning tree analysis for cluster separation

    Analyzes MST edge lengths to detect gaps between clusters vs continuous variation.
    """

    def __init__(self, metric: str = 'euclidean'):
        """
        Initialize MST analyzer

        Args:
            metric: Distance metric
        """
        self.metric = metric

        self.mst_: Optional[csr_matrix] = None
        self.edge_lengths_: Optional[np.ndarray] = None
        self.gap_edges_: Optional[List[Tuple[int, int]]] = None

    def fit(self, X: np.ndarray, labels: Optional[np.ndarray] = None) -> 'MinimumSpanningTreeAnalyzer':
        """
        Fit MST analyzer

        Args:
            X: Data matrix (n_samples, n_features)
            labels: Optional cluster labels

        Returns:
            Self (fitted)
        """
        # Compute distance matrix
        distances = pairwise_distances(X, metric=self.metric)

        # Compute MST
        self.mst_ = minimum_spanning_tree(csr_matrix(distances))

        # Extract edge lengths
        mst_edges = self.mst_.toarray()
        edge_lengths = []
        edge_pairs = []

        for i in range(mst_edges.shape[0]):
            for j in range(i + 1, mst_edges.shape[1]):
                if mst_edges[i, j] > 0:
                    edge_lengths.append(mst_edges[i, j])
                    edge_pairs.append((i, j))
                elif mst_edges[j, i] > 0:
                    edge_lengths.append(mst_edges[j, i])
                    edge_pairs.append((j, i))

        self.edge_lengths_ = np.array(edge_lengths)
        self.edge_pairs_ = edge_pairs

        # Detect gap edges if labels provided
        if labels is not None:
            self.gap_edges_ = self._detect_gap_edges(labels)

        return self

    def _detect_gap_edges(self, labels: np.ndarray) -> List[Tuple[int, int]]:
        """Detect edges connecting different clusters"""
        gap_edges = []

        for i, j in self.edge_pairs_:
            if labels[i] != labels[j]:
                gap_edges.append((i, j))

        return gap_edges

    def compute_gap_statistic(
        self,
        labels: Optional[np.ndarray] = None,
        threshold_percentile: float = 95
    ) -> Dict[str, float]:
        """
        Compute gap statistic from MST edges

        Args:
            labels: Cluster labels
            threshold_percentile: Percentile for gap detection

        Returns:
            Dictionary with gap statistics
        """
        if self.edge_lengths_ is None:
            raise ValueError("Must call fit() first")

        # Sort edge lengths
        sorted_lengths = np.sort(self.edge_lengths_)

        # Compute gaps between consecutive edges
        gaps = np.diff(sorted_lengths)

        # Find largest gaps
        threshold = np.percentile(gaps, threshold_percentile)
        large_gaps = gaps[gaps > threshold]

        # Compute statistics
        mean_within = np.mean(sorted_lengths[:len(sorted_lengths)//2])
        mean_between = np.mean(sorted_lengths[len(sorted_lengths)//2:])

        results = {
            'gap_ratio': mean_between / mean_within if mean_within > 0 else 1.0,
            'n_large_gaps': len(large_gaps),
            'max_gap': np.max(gaps) if len(gaps) > 0 else 0,
            'gap_threshold': threshold,
            'mean_within_distance': mean_within,
            'mean_between_distance': mean_between,
        }

        # If labels provided, compute within vs between cluster edges
        if labels is not None:
            within_edges = []
            between_edges = []

            for (i, j), length in zip(self.edge_pairs_, self.edge_lengths_):
                if labels[i] == labels[j]:
                    within_edges.append(length)
                else:
                    between_edges.append(length)

            if within_edges and between_edges:
                results['within_cluster_mean'] = np.mean(within_edges)
                results['between_cluster_mean'] = np.mean(between_edges)
                results['cluster_gap_ratio'] = np.mean(between_edges) / np.mean(within_edges)

        return results

    def test_separation_hypothesis(self, labels: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, float]:
        """
        Statistical test for cluster separation

        Tests whether between-cluster edges are significantly longer than within-cluster edges.

        Args:
            labels: Cluster labels
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with test results
        """
        if self.edge_lengths_ is None:
            raise ValueError("Must call fit() first")

        # Separate within and between cluster edges
        within_edges = []
        between_edges = []

        for (i, j), length in zip(self.edge_pairs_, self.edge_lengths_):
            if labels[i] == labels[j]:
                within_edges.append(length)
            else:
                between_edges.append(length)

        if not within_edges or not between_edges:
            return {
                'u_statistic': np.nan,
                'p_value': np.nan,
                'effect_size': np.nan,
                'bootstrap_p_value': np.nan,
            }

        # Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(between_edges, within_edges, alternative='greater')

        # Effect size (rank-biserial correlation)
        n1, n2 = len(between_edges), len(within_edges)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)

        # Bootstrap test
        observed_diff = np.mean(between_edges) - np.mean(within_edges)
        all_edges = within_edges + between_edges

        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            perm_labels = np.random.permutation(len(all_edges))
            perm_between = [all_edges[i] for i in perm_labels[:len(between_edges)]]
            perm_within = [all_edges[i] for i in perm_labels[len(between_edges):]]
            bootstrap_diffs.append(np.mean(perm_between) - np.mean(perm_within))

        bootstrap_p = np.mean(np.array(bootstrap_diffs) >= observed_diff)

        return {
            'u_statistic': u_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'bootstrap_p_value': bootstrap_p,
            'observed_difference': observed_diff,
            'within_mean': np.mean(within_edges),
            'between_mean': np.mean(between_edges),
        }


class PersistentHomologyAnalyzer:
    """
    Persistent homology analysis via Ripser

    Analyzes topological features across scales to detect inherent structure.
    """

    def __init__(self, maxdim: int = 2, thresh: float = np.inf):
        """
        Initialize persistent homology analyzer

        Args:
            maxdim: Maximum dimension for homology
            thresh: Maximum distance for homology computation
        """
        if not RIPSER_AVAILABLE:
            raise ImportError("ripser is required for persistent homology analysis")

        self.maxdim = maxdim
        self.thresh = thresh

        self.diagrams_: Optional[List[np.ndarray]] = None
        self.result_: Optional[Dict] = None

    def fit(self, X: np.ndarray, metric: str = 'euclidean') -> 'PersistentHomologyAnalyzer':
        """
        Compute persistent homology

        Args:
            X: Data matrix
            metric: Distance metric

        Returns:
            Self (fitted)
        """
        # Compute persistence diagrams
        self.result_ = ripser.ripser(
            X,
            maxdim=self.maxdim,
            thresh=self.thresh,
            distance_matrix=False,
            do_cocycles=False,
        )

        self.diagrams_ = self.result_['dgms']

        return self

    def compute_lifetimes(self) -> Dict[int, np.ndarray]:
        """
        Compute feature lifetimes (persistence)

        Returns:
            Dictionary mapping dimension to lifetimes
        """
        if self.diagrams_ is None:
            raise ValueError("Must call fit() first")

        lifetimes = {}

        for dim, dgm in enumerate(self.diagrams_):
            # Remove infinite bars
            finite_dgm = dgm[dgm[:, 1] < np.inf]

            if len(finite_dgm) > 0:
                lifetimes[dim] = finite_dgm[:, 1] - finite_dgm[:, 0]
            else:
                lifetimes[dim] = np.array([])

        return lifetimes

    def compute_persistence_entropy(self) -> Dict[int, float]:
        """
        Compute persistence entropy for each dimension

        Higher entropy indicates more uniform distribution of features (spectrum-like).
        Lower entropy indicates few dominant features (cluster-like).

        Returns:
            Dictionary mapping dimension to entropy
        """
        lifetimes = self.compute_lifetimes()
        entropies = {}

        for dim, L in lifetimes.items():
            if len(L) == 0:
                entropies[dim] = 0.0
                continue

            # Normalize lifetimes to probabilities
            L_sum = np.sum(L)
            if L_sum == 0:
                entropies[dim] = 0.0
                continue

            p = L / L_sum

            # Compute entropy
            p_nonzero = p[p > 0]
            entropies[dim] = -np.sum(p_nonzero * np.log(p_nonzero))

        return entropies

    def detect_significant_features(
        self,
        dimension: int = 1,
        threshold_percentile: float = 90
    ) -> np.ndarray:
        """
        Detect significant topological features

        Args:
            dimension: Homology dimension
            threshold_percentile: Percentile for significance

        Returns:
            Array of significant features (birth, death)
        """
        if self.diagrams_ is None:
            raise ValueError("Must call fit() first")

        dgm = self.diagrams_[dimension]
        finite_dgm = dgm[dgm[:, 1] < np.inf]

        if len(finite_dgm) == 0:
            return np.array([])

        lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
        threshold = np.percentile(lifetimes, threshold_percentile)

        significant = finite_dgm[lifetimes >= threshold]

        return significant

    def interpret_topology(self) -> str:
        """
        Interpret topology as separated vs spectrum

        Returns:
            Interpretation string
        """
        lifetimes = self.compute_lifetimes()
        entropies = self.compute_persistence_entropy()

        # H0 (connected components) indicates clusters
        if 0 in lifetimes and len(lifetimes[0]) > 0:
            n_significant_h0 = np.sum(lifetimes[0] > np.median(lifetimes[0]))

            # H1 (loops/holes) indicates continuous structure
            if 1 in lifetimes and len(lifetimes[1]) > 0:
                n_significant_h1 = np.sum(lifetimes[1] > np.median(lifetimes[1]))
                h1_entropy = entropies.get(1, 0)

                # High H1 with high entropy suggests spectrum
                if n_significant_h1 > 3 and h1_entropy > 1.5:
                    return "spectrum"
                # Few significant H0, many H1 suggests intermediate
                elif n_significant_h0 < 5 and n_significant_h1 > 1:
                    return "intermediate"

            # Many significant H0, few H1 suggests separation
            if n_significant_h0 > 5:
                return "separated"

        return "intermediate"


class DensityGapAnalyzer:
    """
    Density-based gap analysis

    Analyzes local density to detect gaps between high-density regions.
    """

    def __init__(self, n_neighbors: int = 10, bandwidth: float = 1.0):
        """
        Initialize density gap analyzer

        Args:
            n_neighbors: Number of neighbors for density estimation
            bandwidth: Bandwidth for density estimation
        """
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth

        self.densities_: Optional[np.ndarray] = None
        self.density_gaps_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'DensityGapAnalyzer':
        """
        Compute local densities

        Args:
            X: Data matrix

        Returns:
            Self (fitted)
        """
        # Store data for later use
        self.X_ = X

        # K-nearest neighbors
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        self.nbrs_.fit(X)

        distances, indices = self.nbrs_.kneighbors(X)

        # Store indices for boundary detection
        self.knn_indices_ = indices

        # Local density as inverse of mean k-NN distance
        # Exclude distance to self (first column)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        self.densities_ = 1.0 / (mean_distances + 1e-10)

        return self

    def compute_gap_score(self, labels: np.ndarray) -> GapScore:
        """
        Compute gap score quantifying separation

        Args:
            labels: Cluster labels

        Returns:
            GapScore object
        """
        if self.densities_ is None:
            raise ValueError("Must call fit() first")

        # Compute within-cluster and between-cluster density statistics
        unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)

        within_densities = []
        for label in unique_labels:
            cluster_mask = labels == label
            within_densities.extend(self.densities_[cluster_mask])

        # Approximate between-cluster density using boundary points
        # Points with neighbors in different clusters
        boundary_densities = []
        for i, label in enumerate(labels):
            if label == -1:
                continue

            # Find neighbors with different labels using pre-computed k-NN
            neighbor_indices = self.knn_indices_[i, 1:]  # Exclude self
            neighbor_labels = labels[neighbor_indices]

            if np.any(neighbor_labels != label):
                boundary_densities.append(self.densities_[i])

        if not boundary_densities:
            boundary_densities = within_densities

        within_mean = np.mean(within_densities)
        boundary_mean = np.mean(boundary_densities)

        # Gap score: ratio of within to boundary density
        # High score = high within, low boundary = separated
        # Low score = similar densities = spectrum
        gap_ratio = within_mean / (boundary_mean + 1e-10)
        gap_statistic = (within_mean - boundary_mean) / (within_mean + boundary_mean + 1e-10)

        # Statistical test
        u_stat, p_value = mannwhitneyu(within_densities, boundary_densities, alternative='greater')

        # Normalize score to [0, 1]
        score = min(1.0, gap_statistic) if gap_statistic > 0 else 0.0

        # Interpret
        if score > 0.5 and p_value < 0.05:
            interpretation = "separated"
        elif score < 0.2:
            interpretation = "spectrum"
        else:
            interpretation = "intermediate"

        return GapScore(
            score=score,
            within_cluster_distance=1.0 / within_mean,
            between_cluster_distance=1.0 / boundary_mean,
            gap_statistic=gap_statistic,
            p_value=p_value,
            interpretation=interpretation,
        )


class KNNGraphConnectivityAnalyzer:
    """
    k-NN graph connectivity analysis

    Analyzes connectivity patterns in k-NN graph to detect separation vs spectrum.
    """

    def __init__(self, n_neighbors: int = 10, metric: str = 'euclidean'):
        """
        Initialize k-NN graph analyzer

        Args:
            n_neighbors: Number of neighbors
            metric: Distance metric
        """
        self.n_neighbors = n_neighbors
        self.metric = metric

        self.graph_: Optional[csr_matrix] = None
        self.connectivity_: Optional[Dict[str, Any]] = None

    def fit(self, X: np.ndarray, labels: Optional[np.ndarray] = None) -> 'KNNGraphConnectivityAnalyzer':
        """
        Build k-NN graph and analyze connectivity

        Args:
            X: Data matrix
            labels: Optional cluster labels

        Returns:
            Self (fitted)
        """
        # Build k-NN graph
        self.graph_ = kneighbors_graph(
            X,
            self.n_neighbors,
            mode='distance',
            metric=self.metric,
            include_self=False,
        )

        # Analyze connectivity
        self.connectivity_ = self._analyze_connectivity(labels)

        return self

    def _analyze_connectivity(self, labels: Optional[np.ndarray]) -> Dict[str, Any]:
        """Analyze graph connectivity"""
        # Connected components
        n_components, component_labels = connected_components(
            self.graph_,
            directed=False,
            return_labels=True,
        )

        results = {
            'n_components': n_components,
            'component_labels': component_labels,
            'component_sizes': np.bincount(component_labels),
        }

        # If cluster labels provided, compute within vs between cluster edges
        if labels is not None:
            within_edges = 0
            between_edges = 0

            graph_coo = self.graph_.tocoo()

            for i, j, v in zip(graph_coo.row, graph_coo.col, graph_coo.data):
                if labels[i] == labels[j]:
                    within_edges += 1
                else:
                    between_edges += 1

            total_edges = within_edges + between_edges

            results['within_cluster_edges'] = within_edges
            results['between_cluster_edges'] = between_edges
            results['edge_purity'] = within_edges / total_edges if total_edges > 0 else 0

        return results

    def compute_modularity(self, labels: np.ndarray) -> float:
        """
        Compute modularity of clustering in k-NN graph

        High modularity indicates separated clusters.

        Args:
            labels: Cluster labels

        Returns:
            Modularity score
        """
        if self.graph_ is None:
            raise ValueError("Must call fit() first")

        # Convert to binary adjacency matrix
        adjacency = (self.graph_ > 0).astype(float)

        # Total edges
        m = adjacency.sum() / 2

        if m == 0:
            return 0.0

        # Compute modularity
        modularity = 0.0
        unique_labels = np.unique(labels[labels >= 0])

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]

            # Edges within cluster
            subgraph = adjacency[cluster_indices, :][:, cluster_indices]
            e_within = subgraph.sum() / 2

            # Expected edges
            degrees = adjacency[cluster_indices, :].sum(axis=1)
            expected = np.sum(degrees) ** 2 / (4 * m)

            modularity += (e_within - expected) / m

        return modularity

    def test_connectivity_hypothesis(self, labels: np.ndarray) -> Dict[str, float]:
        """
        Test whether clusters are well-separated in connectivity

        Args:
            labels: Cluster labels

        Returns:
            Dictionary with test results
        """
        if self.connectivity_ is None:
            raise ValueError("Must call fit() first")

        edge_purity = self.connectivity_.get('edge_purity', 0)
        modularity = self.compute_modularity(labels)

        # Interpret
        if edge_purity > 0.8 and modularity > 0.3:
            interpretation = "separated"
        elif edge_purity < 0.5 or modularity < 0.1:
            interpretation = "spectrum"
        else:
            interpretation = "intermediate"

        return {
            'edge_purity': edge_purity,
            'modularity': modularity,
            'n_components': self.connectivity_['n_components'],
            'interpretation': interpretation,
        }


class SpectralGapDetector:
    """
    Spectral gap detection in graph Laplacian

    Detects spectral gaps indicating natural cluster structure.
    """

    def __init__(self, n_neighbors: int = 10, n_eigenvalues: int = 20):
        """
        Initialize spectral gap detector

        Args:
            n_neighbors: Number of neighbors for graph construction
            n_eigenvalues: Number of eigenvalues to compute
        """
        self.n_neighbors = n_neighbors
        self.n_eigenvalues = n_eigenvalues

        self.eigenvalues_: Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None
        self.gaps_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'SpectralGapDetector':
        """
        Compute Laplacian eigenvalues

        Args:
            X: Data matrix

        Returns:
            Self (fitted)
        """
        # Build k-NN graph
        knn_graph = kneighbors_graph(
            X,
            self.n_neighbors,
            mode='connectivity',
            include_self=False,
        )

        # Symmetrize
        adjacency = (knn_graph + knn_graph.T) / 2

        # Compute degree matrix
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        D = np.diag(degrees)

        # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees + 1e-10))
        L = np.eye(X.shape[0]) - D_sqrt_inv @ adjacency.toarray() @ D_sqrt_inv

        # Compute eigenvalues
        from scipy.linalg import eigh

        n_eig = min(self.n_eigenvalues, X.shape[0])
        eigenvalues, eigenvectors = eigh(L, subset_by_index=[0, n_eig - 1])

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors

        # Compute gaps
        self.gaps_ = np.diff(eigenvalues)

        return self

    def detect_spectral_gap(self, threshold_percentile: float = 90) -> Dict[str, Any]:
        """
        Detect significant spectral gaps

        Args:
            threshold_percentile: Percentile for gap significance

        Returns:
            Dictionary with gap information
        """
        if self.gaps_ is None:
            raise ValueError("Must call fit() first")

        # Find largest gaps
        threshold = np.percentile(self.gaps_, threshold_percentile)
        significant_gaps = np.where(self.gaps_ > threshold)[0]

        # Largest gap indicates number of clusters
        largest_gap_idx = np.argmax(self.gaps_)
        n_clusters_estimate = largest_gap_idx + 1

        # Eigengap ratio
        if len(self.gaps_) > 1:
            eigengap_ratio = self.gaps_[largest_gap_idx] / np.median(self.gaps_)
        else:
            eigengap_ratio = 1.0

        return {
            'largest_gap_index': largest_gap_idx,
            'largest_gap_value': self.gaps_[largest_gap_idx],
            'n_significant_gaps': len(significant_gaps),
            'significant_gap_indices': significant_gaps,
            'n_clusters_estimate': n_clusters_estimate,
            'eigengap_ratio': eigengap_ratio,
            'all_gaps': self.gaps_,
        }

    def test_separation_hypothesis(self) -> Dict[str, Any]:
        """
        Test separation vs spectrum hypothesis using spectral gaps

        Returns:
            Dictionary with test results
        """
        gap_info = self.detect_spectral_gap()

        eigengap_ratio = gap_info['eigengap_ratio']
        n_significant_gaps = gap_info['n_significant_gaps']

        # Strong eigengap suggests separation
        # Many similar gaps suggest spectrum

        if eigengap_ratio > 3.0 and n_significant_gaps <= 3:
            interpretation = "separated"
            confidence = min(1.0, eigengap_ratio / 5.0)
        elif eigengap_ratio < 1.5 or n_significant_gaps > 5:
            interpretation = "spectrum"
            confidence = max(0.0, 1.0 - eigengap_ratio / 3.0)
        else:
            interpretation = "intermediate"
            confidence = 0.5

        return {
            'interpretation': interpretation,
            'confidence': confidence,
            'eigengap_ratio': eigengap_ratio,
            'n_significant_gaps': n_significant_gaps,
            'n_clusters_estimate': gap_info['n_clusters_estimate'],
        }


class TopologyAnalyzer:
    """
    Comprehensive topology analysis combining all methods

    Provides integrated analysis to test separation vs spectrum hypothesis.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        use_persistence: bool = True,
        n_bootstrap: int = 1000,
    ):
        """
        Initialize topology analyzer

        Args:
            n_neighbors: Number of neighbors for various analyses
            use_persistence: Whether to use persistent homology
            n_bootstrap: Number of bootstrap samples for tests
        """
        self.n_neighbors = n_neighbors
        self.use_persistence = use_persistence and RIPSER_AVAILABLE
        self.n_bootstrap = n_bootstrap

        self.mst_analyzer = MinimumSpanningTreeAnalyzer()
        self.density_analyzer = DensityGapAnalyzer(n_neighbors=n_neighbors)
        self.knn_analyzer = KNNGraphConnectivityAnalyzer(n_neighbors=n_neighbors)
        self.spectral_detector = SpectralGapDetector(n_neighbors=n_neighbors)

        if self.use_persistence:
            self.persistence_analyzer = PersistentHomologyAnalyzer(maxdim=2)
        else:
            self.persistence_analyzer = None

    def analyze(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        compute_persistence: bool = True,
    ) -> TopologyAnalysisResult:
        """
        Perform comprehensive topology analysis

        Args:
            X: Data matrix (n_samples, n_features)
            labels: Cluster labels
            compute_persistence: Whether to compute persistent homology

        Returns:
            TopologyAnalysisResult with all analyses
        """
        # MST analysis
        self.mst_analyzer.fit(X, labels)
        mst_gaps = self.mst_analyzer.compute_gap_statistic(labels)
        mst_test = self.mst_analyzer.test_separation_hypothesis(labels, self.n_bootstrap)

        mst_analysis = {
            'gap_statistics': mst_gaps,
            'separation_test': mst_test,
        }

        # Density gap analysis
        self.density_analyzer.fit(X)
        gap_score = self.density_analyzer.compute_gap_score(labels)

        # k-NN connectivity
        self.knn_analyzer.fit(X, labels)
        knn_test = self.knn_analyzer.test_connectivity_hypothesis(labels)

        # Spectral gaps
        self.spectral_detector.fit(X)
        spectral_gaps = self.spectral_detector.detect_spectral_gap()
        spectral_test = self.spectral_detector.test_separation_hypothesis()

        spectral_analysis = {
            'gaps': spectral_gaps,
            'test': spectral_test,
        }

        # Persistent homology
        persistence_result = None
        if self.use_persistence and compute_persistence:
            try:
                self.persistence_analyzer.fit(X)
                diagrams = self.persistence_analyzer.diagrams_
                lifetimes = self.persistence_analyzer.compute_lifetimes()
                entropies = self.persistence_analyzer.compute_persistence_entropy()
                interpretation = self.persistence_analyzer.interpret_topology()

                persistence_result = PersistenceResult(
                    diagrams=diagrams,
                    lifetimes=lifetimes,
                    persistence_entropy=entropies,
                    interpretation=interpretation,
                )
            except Exception as e:
                warnings.warn(f"Persistent homology failed: {e}")

        # Integrated hypothesis test
        hypothesis_test = self._integrated_hypothesis_test(
            gap_score=gap_score,
            mst_test=mst_test,
            knn_test=knn_test,
            spectral_test=spectral_test,
            persistence_result=persistence_result,
        )

        # Overall interpretation
        overall_interpretation = self._interpret_overall(hypothesis_test)

        return TopologyAnalysisResult(
            gap_scores=gap_score,
            mst_analysis=mst_analysis,
            knn_connectivity=knn_test,
            spectral_gaps=spectral_analysis,
            persistence=persistence_result,
            hypothesis_test=hypothesis_test,
            overall_interpretation=overall_interpretation,
        )

    def _integrated_hypothesis_test(
        self,
        gap_score: GapScore,
        mst_test: Dict,
        knn_test: Dict,
        spectral_test: Dict,
        persistence_result: Optional[PersistenceResult],
    ) -> Dict[str, float]:
        """Integrate all tests into unified hypothesis test"""

        # Collect evidence for separation
        separation_scores = []

        # Gap score (0-1, higher = separated)
        separation_scores.append(gap_score.score)

        # MST test (p-value, lower = separated)
        if not np.isnan(mst_test['p_value']):
            separation_scores.append(1.0 - mst_test['p_value'])

        # k-NN edge purity (0-1, higher = separated)
        separation_scores.append(knn_test['edge_purity'])

        # k-NN modularity (0-1, higher = separated)
        separation_scores.append(min(1.0, knn_test['modularity'] / 0.5))

        # Spectral test confidence
        if spectral_test['interpretation'] == 'separated':
            separation_scores.append(spectral_test['confidence'])
        elif spectral_test['interpretation'] == 'spectrum':
            separation_scores.append(1.0 - spectral_test['confidence'])
        else:
            separation_scores.append(0.5)

        # Persistence entropy (if available)
        if persistence_result is not None and persistence_result.persistence_entropy:
            # Low entropy = separated, high entropy = spectrum
            if 0 in persistence_result.persistence_entropy:
                h0_entropy = persistence_result.persistence_entropy[0]
                # Normalize entropy (typical range 0-3)
                separation_scores.append(1.0 - min(1.0, h0_entropy / 3.0))

        # Aggregate scores
        mean_score = np.mean(separation_scores)
        confidence = 1.0 - np.std(separation_scores)  # Higher agreement = higher confidence

        return {
            'separation_score': mean_score,
            'confidence': confidence,
            'individual_scores': separation_scores,
            'n_tests': len(separation_scores),
        }

    def _interpret_overall(self, hypothesis_test: Dict) -> str:
        """Interpret overall topology"""
        score = hypothesis_test['separation_score']
        confidence = hypothesis_test['confidence']

        if score > 0.6 and confidence > 0.5:
            return "separated"
        elif score < 0.4 and confidence > 0.5:
            return "spectrum"
        else:
            return "intermediate"


def analyze_separation_vs_spectrum(
    X: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 10,
    use_persistence: bool = True,
    n_bootstrap: int = 1000,
) -> TopologyAnalysisResult:
    """
    Convenience function to analyze separation vs spectrum hypothesis

    Args:
        X: Data matrix
        labels: Cluster labels
        n_neighbors: Number of neighbors for analyses
        use_persistence: Whether to use persistent homology
        n_bootstrap: Number of bootstrap samples

    Returns:
        TopologyAnalysisResult
    """
    analyzer = TopologyAnalyzer(
        n_neighbors=n_neighbors,
        use_persistence=use_persistence,
        n_bootstrap=n_bootstrap,
    )

    return analyzer.analyze(X, labels)