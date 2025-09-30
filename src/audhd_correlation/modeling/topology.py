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


# ============================================================================
# BASELINE-DEVIATION FRAMEWORK
# ============================================================================
# Implements the "baseline first, deviations second, topology gate third" approach
# to prevent false positive subtype discoveries

@dataclass
class DeviationScores:
    """Per-sample deviation scores from baseline manifold"""
    orthogonal_residual: np.ndarray  # Distance to local control subspace
    mst_delta: np.ndarray  # MST edge anomaly score
    knn_curvature: np.ndarray  # kNN graph curvature mismatch
    persistence_delta: Optional[np.ndarray] = None  # Persistence anomaly
    deviation_score: Optional[np.ndarray] = None  # Aggregated score
    z_scores: Optional[Dict[str, np.ndarray]] = None  # Z-scored components
    sample_ids: Optional[np.ndarray] = None


@dataclass
class SeparationDecision:
    """Decision from topology gate"""
    separation_score: float  # 0-1, higher = more separated
    confidence_interval: Tuple[float, float]  # Bootstrap CI
    decision: str  # 'spectrum', 'intermediate', 'separated'
    subtype_claim_allowed: bool
    metrics: Dict[str, Any]
    notes: List[str]


class BaselineManifold:
    """
    Learn baseline manifold from controls (or unsupervised)

    Two modes:
    A) Control mode: Learn baseline from control/typical samples
    B) Unsupervised mode: Learn baseline from high-density ridge

    Provides:
    - fit(Z_ctrl): Learn baseline structure
    - score(Z): Score deviations for all samples
    """

    def __init__(
        self,
        mode: str = 'control',  # 'control' or 'unsupervised'
        n_neighbors: int = 15,
        local_pca_components: int = 5,
        density_percentile: float = 75.0,  # For unsupervised mode
    ):
        """
        Initialize baseline manifold

        Args:
            mode: 'control' (learn from controls) or 'unsupervised' (learn from density ridge)
            n_neighbors: Number of neighbors for local geometry
            local_pca_components: Dimensions for local tangent PCA
            density_percentile: Percentile for baseline in unsupervised mode
        """
        self.mode = mode
        self.n_neighbors = n_neighbors
        self.local_pca_components = local_pca_components
        self.density_percentile = density_percentile

        # Fitted attributes
        self.X_baseline_: Optional[np.ndarray] = None
        self.knn_graph_: Optional[Any] = None
        self.local_bases_: Optional[Dict[int, np.ndarray]] = None
        self.mst_: Optional[Any] = None
        self.mst_edges_: Optional[np.ndarray] = None
        self.densities_: Optional[np.ndarray] = None
        self.baseline_indices_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, control_mask: Optional[np.ndarray] = None) -> 'BaselineManifold':
        """
        Fit baseline manifold

        Args:
            X: Full data matrix (n_samples, n_features)
            control_mask: Boolean mask for controls (required if mode='control')

        Returns:
            Self (fitted)
        """
        if self.mode == 'control':
            if control_mask is None:
                raise ValueError("control_mask required for mode='control'")

            # Learn baseline from controls only
            self.X_baseline_ = X[control_mask]
            self.baseline_indices_ = np.where(control_mask)[0]

        elif self.mode == 'unsupervised':
            # Learn baseline from high-density ridge
            self._fit_unsupervised_baseline(X)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Build k-NN graph on baseline
        self._build_knn_graph()

        # Fit local tangent spaces
        self._fit_local_tangent_spaces()

        # Build MST on baseline
        self._build_mst()

        return self

    def _fit_unsupervised_baseline(self, X: np.ndarray):
        """Fit baseline using density-based approach"""
        # Estimate local density
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)

        # Density = inverse of mean k-NN distance
        mean_distances = np.mean(distances[:, 1:], axis=1)
        densities = 1.0 / (mean_distances + 1e-10)

        self.densities_ = densities

        # Select high-density ridge as baseline
        threshold = np.percentile(densities, self.density_percentile)
        baseline_mask = densities >= threshold

        self.X_baseline_ = X[baseline_mask]
        self.baseline_indices_ = np.where(baseline_mask)[0]

        print(f"Unsupervised baseline: {len(self.baseline_indices_)}/{len(X)} samples ({100*len(self.baseline_indices_)/len(X):.1f}%)")

    def _build_knn_graph(self):
        """Build k-NN graph on baseline"""
        self.knn_graph_ = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(self.X_baseline_)))
        self.knn_graph_.fit(self.X_baseline_)

    def _fit_local_tangent_spaces(self):
        """Fit local tangent PCA bases"""
        from sklearn.decomposition import PCA

        self.local_bases_ = {}

        for i in range(len(self.X_baseline_)):
            # Find neighbors
            distances, indices = self.knn_graph_.kneighbors([self.X_baseline_[i]], return_distance=True)
            neighbor_indices = indices[0]

            # Get neighborhood
            neighborhood = self.X_baseline_[neighbor_indices]

            # Fit local PCA
            pca = PCA(n_components=min(self.local_pca_components, len(neighborhood)))
            pca.fit(neighborhood)

            # Store basis (principal components)
            self.local_bases_[i] = pca.components_

    def _build_mst(self):
        """Build MST on baseline"""
        distances = pairwise_distances(self.X_baseline_)
        self.mst_ = minimum_spanning_tree(csr_matrix(distances))

        # Extract edge lengths
        mst_array = self.mst_.toarray()
        edges = []
        for i in range(mst_array.shape[0]):
            for j in range(i+1, mst_array.shape[1]):
                if mst_array[i, j] > 0:
                    edges.append(mst_array[i, j])
                elif mst_array[j, i] > 0:
                    edges.append(mst_array[j, i])

        self.mst_edges_ = np.array(edges) if edges else np.array([])

    def score(self, X: np.ndarray) -> DeviationScores:
        """
        Score deviations from baseline

        Args:
            X: Data matrix (can include baseline + new samples)

        Returns:
            DeviationScores object
        """
        if self.X_baseline_ is None:
            raise ValueError("Must call fit() first")

        n_samples = X.shape[0]

        # Initialize scores
        orth_resid = np.zeros(n_samples)
        mst_delta = np.zeros(n_samples)
        knn_curv = np.zeros(n_samples)

        # Compute each deviation component
        for i in range(n_samples):
            x = X[i:i+1]

            # 1. Orthogonal residual
            orth_resid[i] = self._compute_orthogonal_residual(x)

            # 2. MST delta
            mst_delta[i] = self._compute_mst_delta(x)

            # 3. k-NN curvature
            knn_curv[i] = self._compute_knn_curvature(x)

        # Z-score each component
        z_scores = {
            'orthogonal_residual': (orth_resid - orth_resid.mean()) / (orth_resid.std() + 1e-10),
            'mst_delta': (mst_delta - mst_delta.mean()) / (mst_delta.std() + 1e-10),
            'knn_curvature': (knn_curv - knn_curv.mean()) / (knn_curv.std() + 1e-10),
        }

        # Aggregate: trimmed mean of z-scores (robust to outliers)
        z_array = np.column_stack([z_scores['orthogonal_residual'], z_scores['mst_delta'], z_scores['knn_curvature']])
        # Use median for robustness
        deviation_score = np.median(z_array, axis=1)

        return DeviationScores(
            orthogonal_residual=orth_resid,
            mst_delta=mst_delta,
            knn_curvature=knn_curv,
            deviation_score=deviation_score,
            z_scores=z_scores,
        )

    def _compute_orthogonal_residual(self, x: np.ndarray) -> float:
        """Compute distance to nearest local tangent space"""
        # Find nearest baseline point
        distances, indices = self.knn_graph_.kneighbors(x, n_neighbors=1)
        nearest_idx = indices[0, 0]

        # Get local basis
        basis = self.local_bases_[nearest_idx]

        # Project onto local subspace
        x_centered = x - self.X_baseline_[nearest_idx]
        projection = x_centered @ basis.T @ basis

        # Orthogonal residual
        residual = x_centered - projection
        return np.linalg.norm(residual)

    def _compute_mst_delta(self, x: np.ndarray) -> float:
        """Compute MST edge anomaly"""
        if len(self.mst_edges_) == 0:
            return 0.0

        # Find nearest baseline point
        distances, indices = self.knn_graph_.kneighbors(x, n_neighbors=1)
        edge_length = distances[0, 0]

        # Compare to baseline MST edge distribution
        # Z-score relative to baseline edges
        mst_mean = self.mst_edges_.mean()
        mst_std = self.mst_edges_.std() + 1e-10

        delta = (edge_length - mst_mean) / mst_std

        return max(0, delta)  # Only count anomalously long edges

    def _compute_knn_curvature(self, x: np.ndarray) -> float:
        """Compute k-NN curvature mismatch"""
        # Find k nearest baseline points
        distances, indices = self.knn_graph_.kneighbors(x, n_neighbors=self.n_neighbors)

        # Compute variance of distances (curvature proxy)
        curvature = np.var(distances[0])

        # Compare to typical baseline curvature
        # Sample baseline curvatures
        baseline_curvatures = []
        for i in range(min(100, len(self.X_baseline_))):
            d, _ = self.knn_graph_.kneighbors([self.X_baseline_[i]], n_neighbors=self.n_neighbors)
            baseline_curvatures.append(np.var(d[0]))

        baseline_curv_mean = np.mean(baseline_curvatures)
        baseline_curv_std = np.std(baseline_curvatures) + 1e-10

        delta = abs(curvature - baseline_curv_mean) / baseline_curv_std

        return delta


class RotationNull:
    """
    Rotation-preserving null models for deviation threshold

    Generates null distribution by rotating baseline while preserving
    covariance structure
    """

    def __init__(
        self,
        baseline_manifold: BaselineManifold,
        n_rotations: int = 200,
        preserve_scale: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize rotation null

        Args:
            baseline_manifold: Fitted BaselineManifold
            n_rotations: Number of random rotations
            preserve_scale: Whether to preserve scale
            random_state: Random seed
        """
        self.baseline_manifold = baseline_manifold
        self.n_rotations = n_rotations
        self.preserve_scale = preserve_scale
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.null_scores_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'RotationNull':
        """
        Generate null distribution

        Args:
            X: Full data matrix

        Returns:
            Self (fitted)
        """
        n_samples, n_features = X.shape

        null_scores_list = []

        print(f"Generating rotation null ({self.n_rotations} rotations)...")

        for r in range(self.n_rotations):
            # Generate random orthonormal matrix
            Q, _ = np.linalg.qr(self.rng.randn(n_features, n_features))

            # Rotate data
            X_rot = X @ Q

            # Score deviations on rotated data
            # Need to refit baseline on rotated data
            if self.baseline_manifold.mode == 'control':
                # Use same control indices
                control_mask_rot = np.zeros(n_samples, dtype=bool)
                control_mask_rot[self.baseline_manifold.baseline_indices_] = True

                baseline_rot = BaselineManifold(
                    mode='control',
                    n_neighbors=self.baseline_manifold.n_neighbors,
                    local_pca_components=self.baseline_manifold.local_pca_components
                )
                baseline_rot.fit(X_rot, control_mask=control_mask_rot)

            else:
                baseline_rot = BaselineManifold(
                    mode='unsupervised',
                    n_neighbors=self.baseline_manifold.n_neighbors,
                    local_pca_components=self.baseline_manifold.local_pca_components,
                    density_percentile=self.baseline_manifold.density_percentile
                )
                baseline_rot.fit(X_rot)

            # Score
            dev_scores_rot = baseline_rot.score(X_rot)
            null_scores_list.append(dev_scores_rot.deviation_score)

        # Stack all null scores
        self.null_scores_ = np.column_stack(null_scores_list)

        return self

    def quantile(self, q: float) -> float:
        """
        Get quantile of null distribution

        Args:
            q: Quantile (0-1)

        Returns:
            Threshold value
        """
        if self.null_scores_ is None:
            raise ValueError("Must call fit() first")

        # Compute quantile across all rotations and samples
        return np.quantile(self.null_scores_.ravel(), q)

    def compute_fdr_threshold(
        self,
        observed_scores: np.ndarray,
        q: float = 0.05
    ) -> float:
        """
        Compute FDR-controlled threshold using Benjamini-Hochberg

        Args:
            observed_scores: Observed deviation scores
            q: Target FDR

        Returns:
            Threshold value
        """
        if self.null_scores_ is None:
            raise ValueError("Must call fit() first")

        n = len(observed_scores)

        # For each observed score, compute empirical p-value
        p_values = np.zeros(n)
        for i, score in enumerate(observed_scores):
            # P-value = proportion of null scores >= observed
            p_values[i] = (self.null_scores_ >= score).mean()

        # Benjamini-Hochberg procedure
        sorted_indices = np.argsort(p_values)
        sorted_pvals = p_values[sorted_indices]

        # Find largest k where p_(k) <= k*q/n
        k_max = 0
        for k in range(1, n + 1):
            if sorted_pvals[k-1] <= k * q / n:
                k_max = k

        if k_max == 0:
            # No discoveries
            return np.inf

        # Threshold = observed score at k_max
        threshold_idx = sorted_indices[k_max - 1]
        threshold = observed_scores[threshold_idx]

        return threshold


class TopologyGate:
    """
    Enhanced topology gate with separation scoring and CI

    Decides whether deviants form discrete clusters or continuous spectrum
    before allowing subtype claims
    """

    def __init__(
        self,
        min_separation_score: float = 0.6,
        min_confidence: float = 0.5,
        n_bootstrap: int = 200,
        random_state: int = 42,
    ):
        """
        Initialize topology gate

        Args:
            min_separation_score: Minimum score to claim separation
            min_confidence: Minimum confidence (CI width check)
            n_bootstrap: Number of bootstrap samples for CI
            random_state: Random seed
        """
        self.min_separation_score = min_separation_score
        self.min_confidence = min_confidence
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def analyze(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> SeparationDecision:
        """
        Analyze topology and make separation decision

        Args:
            X: Data matrix (deviants only)
            labels: Optional preliminary cluster labels

        Returns:
            SeparationDecision
        """
        # If no labels, generate via quick clustering
        if labels is None:
            from sklearn.cluster import KMeans
            # Try k=3 as default
            labels = KMeans(n_clusters=3, random_state=self.random_state, n_init=10).fit_predict(X)

        # Run comprehensive topology analysis
        analyzer = TopologyAnalyzer(n_neighbors=15, use_persistence=False, n_bootstrap=100)
        result = analyzer.analyze(X, labels, compute_persistence=False)

        # Extract separation score
        separation_score = result.hypothesis_test['separation_score']

        # Bootstrap CI
        ci_lower, ci_upper = self._compute_bootstrap_ci(X, labels)

        # Decision
        if separation_score >= self.min_separation_score and ci_lower >= self.min_confidence:
            decision = 'separated'
            subtype_allowed = True
        elif separation_score < 0.4:
            decision = 'spectrum'
            subtype_allowed = False
        else:
            decision = 'intermediate'
            subtype_allowed = False  # Conservative

        notes = [
            f"Separation score: {separation_score:.3f}",
            f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]",
            f"Decision: {decision}",
        ]

        if not subtype_allowed:
            notes.append("⚠️ Topology gate: Discrete subtypes NOT supported. Data appears continuous.")
        else:
            notes.append("✓ Topology gate: Discrete subtypes supported by topology.")

        return SeparationDecision(
            separation_score=separation_score,
            confidence_interval=(ci_lower, ci_upper),
            decision=decision,
            subtype_claim_allowed=subtype_allowed,
            metrics={
                'mst_gap_ratio': result.mst_analysis['gap_statistics'].get('gap_ratio', 0),
                'knn_edge_purity': result.knn_connectivity.get('edge_purity', 0),
                'spectral_gap': result.spectral_gaps['test']['eigengap_ratio'],
            },
            notes=notes
        )

    def _compute_bootstrap_ci(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for separation score"""
        n_samples = X.shape[0]
        bootstrap_scores = []

        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            indices = self.rng.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            labels_boot = labels[indices]

            # Quick topology score
            try:
                analyzer = TopologyAnalyzer(n_neighbors=15, use_persistence=False, n_bootstrap=50)
                result = analyzer.analyze(X_boot, labels_boot, compute_persistence=False)
                bootstrap_scores.append(result.hypothesis_test['separation_score'])
            except:
                continue

        if not bootstrap_scores:
            return (0.0, 0.0)

        # Compute CI
        ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

        return (ci_lower, ci_upper)