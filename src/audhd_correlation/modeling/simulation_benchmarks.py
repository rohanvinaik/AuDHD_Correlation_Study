"""Simulation benchmarks for clustering validation

Addresses Point 7: Simulation benchmarks

Tests clustering pipeline on known ground truth scenarios:
- Spectrum (no clusters)
- Clean clusters (well-separated)
- Weakly separated clusters
- Batch-confounded data
- Noisy features
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score
)
from sklearn.datasets import make_blobs, make_classification
from scipy.stats import multivariate_normal


@dataclass
class SimulationScenario:
    """Definition of a simulation scenario"""
    name: str
    description: str
    n_samples: int
    n_features: int
    true_n_clusters: int
    separation: float  # 0-1, 0=spectrum, 1=well-separated
    noise_level: float  # 0-1
    batch_effect: bool
    random_state: int = 42


@dataclass
class BenchmarkResults:
    """Results from a single benchmark scenario"""
    scenario_name: str
    true_labels: np.ndarray
    predicted_labels: np.ndarray

    # Cluster recovery metrics
    adjusted_rand_index: float
    normalized_mutual_info: float

    # Cluster quality metrics
    silhouette_score: float
    calinski_harabasz: float

    # Topology metrics
    topology_passed: bool
    null_models_passed: bool

    # Accuracy metrics
    n_clusters_true: int
    n_clusters_predicted: int
    cluster_count_error: int

    metadata: Dict[str, Any] = field(default_factory=dict)


class SimulationGenerator:
    """Generate synthetic data for benchmarking"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def generate_scenario(self, scenario: SimulationScenario) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for a scenario

        Args:
            scenario: Simulation scenario

        Returns:
            (X, y_true) where X is data and y_true is ground truth labels
        """
        if scenario.name == 'spectrum':
            return self._generate_spectrum(scenario)
        elif scenario.name == 'clean_clusters':
            return self._generate_clean_clusters(scenario)
        elif scenario.name == 'weakly_separated':
            return self._generate_weakly_separated(scenario)
        elif scenario.name == 'batch_confounded':
            return self._generate_batch_confounded(scenario)
        elif scenario.name == 'noisy_features':
            return self._generate_noisy_features(scenario)
        else:
            raise ValueError(f"Unknown scenario: {scenario.name}")

    def _generate_spectrum(self, scenario: SimulationScenario) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data with NO true clusters (continuous spectrum)

        Ground truth: should NOT find discrete clusters
        """
        n_samples = scenario.n_samples
        n_features = scenario.n_features

        # Sample from single multivariate Gaussian
        mean = np.zeros(n_features)
        cov = np.eye(n_features)

        X = multivariate_normal.rvs(mean=mean, cov=cov, size=n_samples, random_state=self.rng)

        # Add smooth gradient (spectrum, not clusters)
        gradient = np.linspace(0, 1, n_samples)
        for f in range(min(3, n_features)):  # Apply to first 3 features
            X[:, f] += gradient * 2

        # No true clusters
        y_true = np.zeros(n_samples, dtype=int)

        return X, y_true

    def _generate_clean_clusters(self, scenario: SimulationScenario) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate well-separated clusters

        Ground truth: SHOULD find discrete clusters
        """
        cluster_std = 1.0 / (scenario.separation + 1e-6)  # High separation = low std

        X, y_true = make_blobs(
            n_samples=scenario.n_samples,
            n_features=scenario.n_features,
            centers=scenario.true_n_clusters,
            cluster_std=cluster_std,
            random_state=scenario.random_state
        )

        # Add noise
        if scenario.noise_level > 0:
            noise = self.rng.randn(*X.shape) * scenario.noise_level
            X += noise

        return X, y_true

    def _generate_weakly_separated(self, scenario: SimulationScenario) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate weakly separated clusters (overlapping)

        Ground truth: ambiguous case
        """
        # High cluster_std = weak separation
        cluster_std = 3.0

        X, y_true = make_blobs(
            n_samples=scenario.n_samples,
            n_features=scenario.n_features,
            centers=scenario.true_n_clusters,
            cluster_std=cluster_std,
            random_state=scenario.random_state
        )

        # Add significant noise
        noise = self.rng.randn(*X.shape) * 2.0
        X += noise

        return X, y_true

    def _generate_batch_confounded(self, scenario: SimulationScenario) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data with batch effects confounded with clusters

        Ground truth: clusters exist but batch correction needed
        """
        # Generate clean clusters first
        X, y_true = self._generate_clean_clusters(scenario)

        # Add batch effects
        n_batches = 3
        batch_size = scenario.n_samples // n_batches

        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, scenario.n_samples)

            # Batch shift
            batch_shift = self.rng.randn(scenario.n_features) * 5.0
            X[start_idx:end_idx] += batch_shift

            # Batch scale
            batch_scale = 1.0 + self.rng.randn(scenario.n_features) * 0.5
            X[start_idx:end_idx] *= batch_scale

        return X, y_true

    def _generate_noisy_features(self, scenario: SimulationScenario) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data with many irrelevant noisy features

        Ground truth: clusters exist in low-dim subspace
        """
        # Generate clusters in low-dimensional space
        n_informative = max(5, scenario.n_features // 10)

        X_informative, y_true = make_blobs(
            n_samples=scenario.n_samples,
            n_features=n_informative,
            centers=scenario.true_n_clusters,
            cluster_std=1.0,
            random_state=scenario.random_state
        )

        # Add many noisy features
        n_noise = scenario.n_features - n_informative
        X_noise = self.rng.randn(scenario.n_samples, n_noise)

        # Combine
        X = np.hstack([X_informative, X_noise])

        return X, y_true


class BenchmarkRunner:
    """Run clustering benchmarks on simulated data"""

    def __init__(
        self,
        clustering_pipeline: Any,
        scenarios: Optional[List[SimulationScenario]] = None,
        random_state: int = 42,
    ):
        """
        Initialize benchmark runner

        Args:
            clustering_pipeline: Clustering pipeline to test
            scenarios: List of scenarios to test (default: all standard scenarios)
            random_state: Random seed
        """
        self.clustering_pipeline = clustering_pipeline
        self.scenarios = scenarios or self._get_default_scenarios()
        self.random_state = random_state

    def _get_default_scenarios(self) -> List[SimulationScenario]:
        """Get default benchmark scenarios"""
        return [
            SimulationScenario(
                name='spectrum',
                description='Continuous spectrum (no clusters)',
                n_samples=300,
                n_features=50,
                true_n_clusters=0,
                separation=0.0,
                noise_level=0.5,
                batch_effect=False
            ),
            SimulationScenario(
                name='clean_clusters',
                description='Well-separated clusters',
                n_samples=300,
                n_features=50,
                true_n_clusters=4,
                separation=0.8,
                noise_level=0.2,
                batch_effect=False
            ),
            SimulationScenario(
                name='weakly_separated',
                description='Weakly separated clusters',
                n_samples=300,
                n_features=50,
                true_n_clusters=3,
                separation=0.3,
                noise_level=0.5,
                batch_effect=False
            ),
            SimulationScenario(
                name='batch_confounded',
                description='Clusters with batch effects',
                n_samples=300,
                n_features=50,
                true_n_clusters=3,
                separation=0.6,
                noise_level=0.3,
                batch_effect=True
            ),
            SimulationScenario(
                name='noisy_features',
                description='Clusters in low-dim subspace + noise',
                n_samples=300,
                n_features=100,
                true_n_clusters=4,
                separation=0.7,
                noise_level=0.3,
                batch_effect=False
            ),
        ]

    def run_benchmarks(self) -> Dict[str, BenchmarkResults]:
        """
        Run all benchmarks

        Returns:
            Dict mapping scenario name to results
        """
        generator = SimulationGenerator(random_state=self.random_state)
        results = {}

        for scenario in self.scenarios:
            print(f"\n{'='*60}")
            print(f"BENCHMARK: {scenario.name}")
            print(f"Description: {scenario.description}")
            print(f"{'='*60}")

            # Generate data
            X, y_true = generator.generate_scenario(scenario)

            # Run clustering
            try:
                # Disable some expensive features for benchmarking
                self.clustering_pipeline.test_null_models = False
                self.clustering_pipeline.evaluate_topology_gates = True

                self.clustering_pipeline.fit(X, generate_embeddings=False)

                y_pred = self.clustering_pipeline.consensus_labels_

                # Compute metrics
                result = self._evaluate_scenario(scenario, X, y_true, y_pred)
                results[scenario.name] = result

                # Print summary
                self._print_result_summary(result)

            except Exception as e:
                warnings.warn(f"Benchmark {scenario.name} failed: {e}")
                continue

        # Print overall summary
        self._print_overall_summary(results)

        return results

    def _evaluate_scenario(
        self,
        scenario: SimulationScenario,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> BenchmarkResults:
        """Evaluate a single scenario"""

        # Handle spectrum case (no true clusters)
        if scenario.true_n_clusters == 0:
            # For spectrum, we SHOULD NOT find clusters
            # Success = low cluster count or low silhouette
            n_pred = len(set(y_pred))
            cluster_recovery = 1.0 if n_pred <= 2 else 0.0

            ari = 0.0
            nmi = 0.0
        else:
            # For true clusters, measure recovery
            ari = adjusted_rand_score(y_true, y_pred)
            nmi = normalized_mutual_info_score(y_true, y_pred)

        # Cluster quality
        sil = silhouette_score(X, y_pred) if len(set(y_pred)) > 1 else 0.0
        ch = calinski_harabasz_score(X, y_pred) if len(set(y_pred)) > 1 else 0.0

        # Topology gates
        topology_passed = False
        if hasattr(self.clustering_pipeline, 'topology_gate_results_'):
            if self.clustering_pipeline.topology_gate_results_ is not None:
                topology_passed = self.clustering_pipeline.topology_gate_results_.passes_gates

        # Null models
        null_passed = True  # Default (disabled in benchmarks for speed)

        # Cluster count accuracy
        n_true = scenario.true_n_clusters
        n_pred = len(set(y_pred))
        count_error = abs(n_pred - n_true)

        return BenchmarkResults(
            scenario_name=scenario.name,
            true_labels=y_true,
            predicted_labels=y_pred,
            adjusted_rand_index=ari,
            normalized_mutual_info=nmi,
            silhouette_score=sil,
            calinski_harabasz=ch,
            topology_passed=topology_passed,
            null_models_passed=null_passed,
            n_clusters_true=n_true,
            n_clusters_predicted=n_pred,
            cluster_count_error=count_error,
        )

    def _print_result_summary(self, result: BenchmarkResults):
        """Print summary of a single result"""
        print(f"\nResults:")
        print(f"  True clusters: {result.n_clusters_true}")
        print(f"  Predicted clusters: {result.n_clusters_predicted}")
        print(f"  Cluster count error: {result.cluster_count_error}")

        if result.n_clusters_true > 0:
            print(f"  Adjusted Rand Index: {result.adjusted_rand_index:.3f}")
            print(f"  Normalized Mutual Info: {result.normalized_mutual_info:.3f}")

        print(f"  Silhouette score: {result.silhouette_score:.3f}")
        print(f"  Topology gates: {'✓ PASSED' if result.topology_passed else '✗ FAILED'}")

    def _print_overall_summary(self, results: Dict[str, BenchmarkResults]):
        """Print overall summary across all benchmarks"""
        print(f"\n\n{'='*60}")
        print("OVERALL BENCHMARK SUMMARY")
        print(f"{'='*60}\n")

        # Create summary table
        data = []
        for name, result in results.items():
            data.append({
                'Scenario': name,
                'True K': result.n_clusters_true,
                'Pred K': result.n_clusters_predicted,
                'Error': result.cluster_count_error,
                'ARI': f"{result.adjusted_rand_index:.3f}",
                'NMI': f"{result.normalized_mutual_info:.3f}",
                'Silhouette': f"{result.silhouette_score:.3f}",
                'Topology': '✓' if result.topology_passed else '✗'
            })

        df = pd.DataFrame(data)
        print(df.to_string(index=False))

        # Overall stats
        print(f"\n{'='*60}")
        print("Key Takeaways:")
        print(f"{'='*60}")

        # Spectrum test
        if 'spectrum' in results:
            spectrum_result = results['spectrum']
            if spectrum_result.n_clusters_predicted <= 2 or not spectrum_result.topology_passed:
                print("✓ Spectrum test: PASSED (correctly identified no discrete clusters)")
            else:
                print("✗ Spectrum test: FAILED (incorrectly found clusters in continuous data)")

        # Clean clusters test
        if 'clean_clusters' in results:
            clean_result = results['clean_clusters']
            if clean_result.adjusted_rand_index > 0.7 and clean_result.topology_passed:
                print("✓ Clean clusters: PASSED (correctly recovered well-separated clusters)")
            else:
                print("✗ Clean clusters: FAILED (failed to recover obvious clusters)")

        # Overall cluster count accuracy
        errors = [r.cluster_count_error for r in results.values() if r.n_clusters_true > 0]
        mean_error = np.mean(errors) if errors else 0.0
        print(f"\nMean cluster count error: {mean_error:.2f} clusters")


def run_simulation_benchmarks(
    clustering_pipeline: Any,
    quick: bool = False,
    random_state: int = 42
) -> Dict[str, BenchmarkResults]:
    """
    Convenience function to run simulation benchmarks

    Args:
        clustering_pipeline: Clustering pipeline to test
        quick: If True, run smaller/faster benchmarks
        random_state: Random seed

    Returns:
        Dict of benchmark results
    """
    if quick:
        # Quick benchmarks (smaller data)
        scenarios = [
            SimulationScenario(
                name='clean_clusters',
                description='Well-separated clusters (quick)',
                n_samples=150,
                n_features=20,
                true_n_clusters=3,
                separation=0.8,
                noise_level=0.2,
                batch_effect=False
            ),
            SimulationScenario(
                name='spectrum',
                description='Continuous spectrum (quick)',
                n_samples=150,
                n_features=20,
                true_n_clusters=0,
                separation=0.0,
                noise_level=0.5,
                batch_effect=False
            ),
        ]
    else:
        scenarios = None  # Use defaults

    runner = BenchmarkRunner(
        clustering_pipeline=clustering_pipeline,
        scenarios=scenarios,
        random_state=random_state
    )

    return runner.run_benchmarks()
