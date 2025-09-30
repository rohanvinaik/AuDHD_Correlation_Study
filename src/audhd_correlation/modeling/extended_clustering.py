#!/usr/bin/env python3
"""
Extended Clustering for Multi-Modal Features

Enhanced clustering methods that leverage extended feature types:
- Feature-aware distance metrics
- Multi-view clustering
- Temporal clustering for longitudinal data
- Biologically-informed clustering with constraints
- Ensemble clustering with consensus
- Extended validation using new feature modalities
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    adjusted_rand_score,
)
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import warnings
import itertools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureAwareDistanceMetrics:
    """Calculate distances with feature-type-specific metrics"""

    def __init__(self):
        """Initialize distance calculator"""
        self.feature_type_metrics = {
            'continuous': 'euclidean',
            'categorical': 'hamming',
            'cyclical': 'angular',
            'compositional': 'braycurtis',  # For microbiome/proportional data
            'binary': 'jaccard'
        }

    def angular_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Angular distance for cyclical features (e.g., circadian phase)

        Args:
            X: First vector
            Y: Second vector

        Returns:
            Angular distance (0-1 scale)
        """
        # Assumes values are in radians
        diff = np.abs(X - Y)
        # Wrap around circular distance
        diff = np.minimum(diff, 2 * np.pi - diff)
        return diff / np.pi  # Normalize to 0-1

    def create_custom_distance_matrix(
        self,
        data: pd.DataFrame,
        feature_types: Dict[str, List[str]],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute weighted distance matrix using feature-type-specific metrics

        Args:
            data: DataFrame with features
            feature_types: Dict mapping feature type to column names
            weights: Optional weights for each feature type

        Returns:
            Distance matrix (n_samples, n_samples)
        """
        if weights is None:
            # Equal weights by default
            weights = {ftype: 1.0 for ftype in feature_types.keys()}

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        partial_distances = []
        used_weights = []

        for ftype, features in feature_types.items():
            # Get features of this type that exist in data
            available_features = [f for f in features if f in data.columns]

            if len(available_features) == 0:
                logger.warning(f"No features found for type {ftype}, skipping")
                continue

            subset = data[available_features].values

            # Handle missing values
            if np.isnan(subset).any():
                logger.warning(f"Missing values in {ftype} features, imputing with median")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                subset = imputer.fit_transform(subset)

            # Get metric for this feature type
            metric = self.feature_type_metrics.get(ftype, 'euclidean')

            # Compute distances
            if metric == 'angular':
                # Custom angular distance for cyclical features
                n_samples = subset.shape[0]
                dist = np.zeros((n_samples, n_samples))
                for i in range(n_samples):
                    for j in range(i + 1, n_samples):
                        d = self.angular_distance(subset[i], subset[j]).mean()
                        dist[i, j] = d
                        dist[j, i] = d
            else:
                # Use sklearn pairwise_distances
                dist = pairwise_distances(subset, metric=metric)

            partial_distances.append(dist)
            used_weights.append(normalized_weights[ftype])

        if len(partial_distances) == 0:
            raise ValueError("No valid feature types found in data")

        # Weighted combination
        combined_distance = np.average(
            np.array(partial_distances),
            weights=np.array(used_weights),
            axis=0
        )

        logger.info(f"Combined distance matrix from {len(partial_distances)} feature types")

        return combined_distance


class MultiViewClustering:
    """Cluster using multiple data views"""

    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        """
        Initialize multi-view clustering

        Args:
            n_clusters: Number of clusters
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_predict(self, views: List[np.ndarray]) -> np.ndarray:
        """
        Fit multi-view clustering

        Args:
            views: List of data views (each is n_samples x n_features_view)

        Returns:
            Cluster labels
        """
        logger.info(f"Multi-view clustering with {len(views)} views")

        # Simple approach: concatenate views and cluster
        # More sophisticated: use multi-view learning algorithms
        concatenated = np.hstack(views)

        # Apply KMeans on concatenated views
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(concatenated)

        logger.info(f"Multi-view clustering complete: {self.n_clusters} clusters")

        return labels

    def spectral_co_clustering(self, views: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Spectral co-clustering on view pairs

        Args:
            views: List of data views

        Returns:
            Dict of co-clustering results for each view pair
        """
        from sklearn.cluster import SpectralBiclustering

        results = {}

        for i, j in itertools.combinations(range(len(views)), 2):
            view_pair = np.hstack([views[i], views[j]])

            # Spectral biclustering
            model = SpectralBiclustering(
                n_clusters=self.n_clusters,
                random_state=self.random_state
            )

            try:
                model.fit(view_pair)
                row_labels = model.row_labels_
                results[f'view_{i}_vs_{j}'] = row_labels

                logger.info(f"Co-clustered views {i} and {j}")

            except Exception as e:
                logger.warning(f"Co-clustering views {i} and {j} failed: {e}")

        return results


class TemporalClustering:
    """Cluster based on temporal trajectories"""

    def __init__(self, n_clusters: int = 5, metric: str = 'dtw'):
        """
        Initialize temporal clustering

        Args:
            n_clusters: Number of trajectory clusters
            metric: Distance metric ('dtw', 'euclidean', 'softdtw')
        """
        self.n_clusters = n_clusters
        self.metric = metric

    def fit_predict(self, trajectories: np.ndarray) -> np.ndarray:
        """
        Cluster temporal trajectories

        Args:
            trajectories: Array of shape (n_samples, n_timepoints, n_features)

        Returns:
            Cluster labels
        """
        try:
            from tslearn.clustering import TimeSeriesKMeans
            from tslearn.preprocessing import TimeSeriesScalerMeanVariance

            logger.info(f"Temporal clustering with metric: {self.metric}")

            # Normalize trajectories
            scaler = TimeSeriesScalerMeanVariance()
            normalized_trajectories = scaler.fit_transform(trajectories)

            # Time series k-means
            model = TimeSeriesKMeans(
                n_clusters=self.n_clusters,
                metric=self.metric,
                max_iter=10,
                random_state=42
            )

            labels = model.fit_predict(normalized_trajectories)

            logger.info(f"Temporal clustering complete: {self.n_clusters} trajectory types")

            return labels

        except ImportError:
            logger.error("tslearn not installed, falling back to standard k-means on flattened trajectories")

            # Fallback: flatten and use standard k-means
            flattened = trajectories.reshape(trajectories.shape[0], -1)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(flattened)


class BiologicallyInformedClustering:
    """Clustering with biological constraints"""

    def __init__(self, n_clusters: int = 8):
        """
        Initialize constrained clustering

        Args:
            n_clusters: Number of clusters
        """
        self.n_clusters = n_clusters

    def generate_biological_constraints(
        self,
        data: pd.DataFrame,
        prior_knowledge: Dict[str, Any]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Generate must-link and cannot-link constraints from biological knowledge

        Args:
            data: Feature data
            prior_knowledge: Dict with biological priors

        Returns:
            (must_link, cannot_link) constraint lists
        """
        must_link = []
        cannot_link = []

        # Example: Samples from same family should be linked
        if 'family_id' in prior_knowledge:
            family_ids = prior_knowledge['family_id']
            for family in np.unique(family_ids):
                family_indices = np.where(family_ids == family)[0]
                # Create must-link constraints within family
                for i, j in itertools.combinations(family_indices, 2):
                    must_link.append((i, j))

        # Example: Samples with known different diagnoses cannot be linked
        if 'diagnosis' in prior_knowledge:
            diagnoses = prior_knowledge['diagnosis']
            autism_indices = np.where(diagnoses == 'autism')[0]
            neurotypical_indices = np.where(diagnoses == 'neurotypical')[0]

            # Create cannot-link between autism and neurotypical (if desired)
            # This is optional and depends on research question
            for i in autism_indices[:50]:  # Limit to avoid too many constraints
                for j in neurotypical_indices[:50]:
                    cannot_link.append((i, j))

        logger.info(f"Generated {len(must_link)} must-link and {len(cannot_link)} cannot-link constraints")

        return must_link, cannot_link

    def fit_predict_with_constraints(
        self,
        data: np.ndarray,
        constraints: Optional[Tuple[List, List]] = None
    ) -> np.ndarray:
        """
        Constrained clustering

        Args:
            data: Feature matrix
            constraints: (must_link, cannot_link) tuples

        Returns:
            Cluster labels
        """
        try:
            from sklearn_extra.cluster import KMedoids

            # sklearn_extra doesn't directly support constraints,
            # so we use a penalty-based approach

            # For now, fall back to standard clustering
            logger.warning("Constraint-based clustering not fully implemented, using standard K-Medoids")

            kmedoids = KMedoids(n_clusters=self.n_clusters, random_state=42)
            labels = kmedoids.fit_predict(data)

            return labels

        except ImportError:
            logger.warning("sklearn-extra not installed, using standard K-Means")

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(data)


class EnsembleClustering:
    """Ensemble clustering with consensus"""

    def __init__(self, n_clusters: int = 8):
        """
        Initialize ensemble clustering

        Args:
            n_clusters: Target number of clusters
        """
        self.n_clusters = n_clusters

    def build_consensus_matrix(
        self,
        clustering_results: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Build consensus matrix from multiple clustering results

        Args:
            clustering_results: Dict mapping method name to cluster labels

        Returns:
            Consensus matrix (n_samples, n_samples)
        """
        n_samples = len(list(clustering_results.values())[0])
        consensus = np.zeros((n_samples, n_samples))

        for method, labels in clustering_results.items():
            # Co-assignment matrix for this method
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] == labels[j] and labels[i] >= 0:  # Same cluster, not noise
                        consensus[i, j] += 1
                        consensus[j, i] += 1

        # Normalize by number of methods
        consensus /= len(clustering_results)

        logger.info(f"Built consensus matrix from {len(clustering_results)} clustering methods")

        return consensus

    def spectral_clustering_on_consensus(
        self,
        consensus_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Apply spectral clustering to consensus matrix

        Args:
            consensus_matrix: Consensus co-assignment matrix

        Returns:
            Final cluster labels
        """
        # Treat consensus as affinity matrix
        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=42
        )

        labels = spectral.fit_predict(consensus_matrix)

        logger.info(f"Spectral clustering on consensus: {self.n_clusters} final clusters")

        return labels

    def fit_ensemble(
        self,
        data: np.ndarray,
        methods: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit ensemble of clustering methods

        Args:
            data: Feature matrix
            methods: List of methods to use

        Returns:
            (final_labels, consensus_matrix)
        """
        if methods is None:
            methods = ['kmeans', 'spectral', 'agglomerative']

        clustering_results = {}

        # K-Means
        if 'kmeans' in methods:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            clustering_results['kmeans'] = kmeans.fit_predict(data)

        # Spectral
        if 'spectral' in methods:
            spectral = SpectralClustering(n_clusters=self.n_clusters, random_state=42)
            clustering_results['spectral'] = spectral.fit_predict(data)

        # Agglomerative
        if 'agglomerative' in methods:
            agg = AgglomerativeClustering(n_clusters=self.n_clusters)
            clustering_results['agglomerative'] = agg.fit_predict(data)

        # Build consensus
        consensus = self.build_consensus_matrix(clustering_results)

        # Final clustering
        final_labels = self.spectral_clustering_on_consensus(consensus)

        return final_labels, consensus


def enhanced_clustering_with_extended_features(
    integrated_data: pd.DataFrame,
    feature_metadata: Dict[str, Any],
    n_clusters: int = 8,
    methods: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Enhanced clustering leveraging extended feature types

    Args:
        integrated_data: Integrated multi-modal features
        feature_metadata: Metadata about features (types, views, etc.)
        n_clusters: Target number of clusters
        methods: Clustering methods to use

    Returns:
        Dict with clustering results and validation metrics
    """
    logger.info("Starting enhanced clustering with extended features...")

    results = {}

    # 1. Feature-aware distance-based clustering
    if 'distance_based' in (methods or []) or methods is None:
        logger.info("Computing feature-aware distances...")

        distance_calculator = FeatureAwareDistanceMetrics()

        # Define feature types from metadata
        feature_types = feature_metadata.get('feature_types', {
            'continuous': [col for col in integrated_data.columns if 'factor' in col.lower()],
        })

        custom_distance_matrix = distance_calculator.create_custom_distance_matrix(
            integrated_data,
            feature_types
        )

        # Agglomerative clustering on custom distances
        agg_custom = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        results['distance_based'] = agg_custom.fit_predict(custom_distance_matrix)

        logger.info("Distance-based clustering complete")

    # 2. Multi-view clustering
    if 'multiview' in (methods or []) or methods is None:
        logger.info("Performing multi-view clustering...")

        # Define views from metadata
        data_views = feature_metadata.get('data_views', None)

        if data_views is not None and len(data_views) > 1:
            multiview_clusterer = MultiViewClustering(n_clusters=n_clusters)
            results['multiview'] = multiview_clusterer.fit_predict(data_views)

            logger.info("Multi-view clustering complete")
        else:
            logger.warning("Multiple views not provided, skipping multi-view clustering")

    # 3. Temporal clustering (if longitudinal data available)
    if 'temporal' in (methods or []):
        logger.info("Performing temporal clustering...")

        longitudinal_data = feature_metadata.get('longitudinal_data', None)

        if longitudinal_data is not None:
            temporal_clusterer = TemporalClustering(n_clusters=n_clusters)
            results['temporal'] = temporal_clusterer.fit_predict(longitudinal_data)

            logger.info("Temporal clustering complete")
        else:
            logger.warning("Longitudinal data not provided, skipping temporal clustering")

    # 4. Biologically-informed clustering
    if 'constrained' in (methods or []):
        logger.info("Performing biologically-informed clustering...")

        prior_knowledge = feature_metadata.get('prior_knowledge', {})

        bio_clusterer = BiologicallyInformedClustering(n_clusters=n_clusters)

        if prior_knowledge:
            constraints = bio_clusterer.generate_biological_constraints(
                integrated_data,
                prior_knowledge
            )
            results['constrained'] = bio_clusterer.fit_predict_with_constraints(
                integrated_data.values,
                constraints
            )
        else:
            results['constrained'] = bio_clusterer.fit_predict_with_constraints(
                integrated_data.values
            )

        logger.info("Biologically-informed clustering complete")

    # 5. Ensemble clustering
    if len(results) > 1:
        logger.info("Building consensus from ensemble...")

        ensemble = EnsembleClustering(n_clusters=n_clusters)
        consensus_matrix = ensemble.build_consensus_matrix(results)
        final_labels = ensemble.spectral_clustering_on_consensus(consensus_matrix)

        results['ensemble'] = final_labels
        results['consensus_matrix'] = consensus_matrix

        logger.info("Ensemble clustering complete")

    elif len(results) == 1:
        # Only one method, use it as final
        results['ensemble'] = list(results.values())[0]

    else:
        # No results, run default
        logger.warning("No clustering methods successful, running default K-Means")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        results['ensemble'] = kmeans.fit_predict(integrated_data.values)

    logger.info("Enhanced clustering complete!")

    return results


def validate_extended_clusters(
    clusters: np.ndarray,
    extended_features: Dict[str, pd.DataFrame],
    clinical_features: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validate clustering using extended feature types

    Args:
        clusters: Cluster assignments
        extended_features: Dict of feature DataFrames by modality
        clinical_features: Clinical phenotype data

    Returns:
        Dict with validation metrics
    """
    logger.info("Validating clusters with extended features...")

    validation_results = {}

    # Test autonomic differentiation
    if 'autonomic' in extended_features:
        autonomic_df = extended_features['autonomic']

        if 'SDNN' in autonomic_df.columns:
            # ANOVA: HRV differs between clusters
            cluster_groups = [autonomic_df.loc[clusters == i, 'SDNN'].values
                            for i in np.unique(clusters)]
            cluster_groups = [g for g in cluster_groups if len(g) > 0]

            if len(cluster_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*cluster_groups)
                validation_results['autonomic_hrv_differentiation'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

                logger.info(f"Autonomic differentiation: F={f_stat:.2f}, p={p_value:.4f}")

    # Test circadian phenotypes
    if 'circadian' in extended_features:
        circadian_df = extended_features['circadian']

        if 'CAR_AUCi' in circadian_df.columns:
            # Test if cortisol awakening response differs
            cluster_groups = [circadian_df.loc[clusters == i, 'CAR_AUCi'].values
                            for i in np.unique(clusters)]
            cluster_groups = [g for g in cluster_groups if len(g) > 0]

            if len(cluster_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*cluster_groups)
                validation_results['circadian_car_differentiation'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

                logger.info(f"Circadian differentiation: F={f_stat:.2f}, p={p_value:.4f}")

    # Test environmental burden
    if 'toxicants' in extended_features:
        toxicants_df = extended_features['toxicants']

        if 'toxic_metal_burden_index' in toxicants_df.columns:
            # Test burden index by cluster
            cluster_groups = [toxicants_df.loc[clusters == i, 'toxic_metal_burden_index'].values
                            for i in np.unique(clusters)]
            cluster_groups = [g for g in cluster_groups if len(g) > 0]

            if len(cluster_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*cluster_groups)
                validation_results['environmental_burden_differentiation'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

                logger.info(f"Environmental burden differentiation: F={f_stat:.2f}, p={p_value:.4f}")

    # Test sensory profiles
    if 'sensory' in extended_features:
        sensory_df = extended_features['sensory']

        if 'P50_gating_ratio' in sensory_df.columns:
            # Test sensory gating
            cluster_groups = [sensory_df.loc[clusters == i, 'P50_gating_ratio'].values
                            for i in np.unique(clusters)]
            cluster_groups = [g for g in cluster_groups if len(g) > 0]

            if len(cluster_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*cluster_groups)
                validation_results['sensory_gating_differentiation'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

                logger.info(f"Sensory gating differentiation: F={f_stat:.2f}, p={p_value:.4f}")

    # Test interoception
    if 'interoception' in extended_features:
        intero_df = extended_features['interoception']

        if 'heartbeat_counting_accuracy' in intero_df.columns:
            # Test interoceptive accuracy
            cluster_groups = [intero_df.loc[clusters == i, 'heartbeat_counting_accuracy'].values
                            for i in np.unique(clusters)]
            cluster_groups = [g for g in cluster_groups if len(g) > 0]

            if len(cluster_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*cluster_groups)
                validation_results['interoception_accuracy_differentiation'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

                logger.info(f"Interoception differentiation: F={f_stat:.2f}, p={p_value:.4f}")

    # Clinical hypothesis tests
    if 'ADHD_RS_hyperactive' in clinical_features.columns:
        # Test if clusters differ on ADHD symptoms
        cluster_groups = [clinical_features.loc[clusters == i, 'ADHD_RS_hyperactive'].values
                        for i in np.unique(clusters)]
        cluster_groups = [g for g in cluster_groups if len(g) > 0]

        if len(cluster_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*cluster_groups)
            validation_results['adhd_symptom_differentiation'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

    # Count significant validations
    n_significant = sum(1 for v in validation_results.values()
                       if isinstance(v, dict) and v.get('significant', False))

    validation_results['summary'] = {
        'n_tests': len(validation_results),
        'n_significant': n_significant,
        'proportion_significant': n_significant / len(validation_results) if len(validation_results) > 0 else 0
    }

    logger.info(f"Validation complete: {n_significant}/{len(validation_results)} tests significant")

    return validation_results


if __name__ == '__main__':
    # Example usage
    logger.info("Extended Clustering Module initialized")

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 200
    n_features = 50

    # Integrated features
    integrated_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'factor_{i}' for i in range(n_features)]
    )

    # Feature metadata
    feature_metadata = {
        'feature_types': {
            'continuous': [f'factor_{i}' for i in range(n_features)]
        }
    }

    # Run enhanced clustering
    results = enhanced_clustering_with_extended_features(
        integrated_df,
        feature_metadata,
        n_clusters=4,
        methods=['distance_based']
    )

    print("\n" + "="*70)
    print("Enhanced Clustering Results")
    print("="*70)

    for method, labels in results.items():
        if method != 'consensus_matrix' and isinstance(labels, np.ndarray):
            n_clusters = len(np.unique(labels[labels >= 0]))
            n_noise = np.sum(labels == -1)
            print(f"\n{method}:")
            print(f"  Clusters: {n_clusters}")
            if n_noise > 0:
                print(f"  Noise points: {n_noise}")

    # Create synthetic extended features for validation
    extended_features = {
        'autonomic': pd.DataFrame({
            'SDNN': np.random.uniform(20, 100, n_samples)
        }),
        'circadian': pd.DataFrame({
            'CAR_AUCi': np.random.uniform(100, 500, n_samples)
        })
    }

    clinical_features = pd.DataFrame({
        'ADHD_RS_hyperactive': np.random.randint(10, 40, n_samples)
    })

    # Validate
    if 'ensemble' in results:
        validation = validate_extended_clusters(
            results['ensemble'],
            extended_features,
            clinical_features
        )

        print("\nValidation Results:")
        print("="*70)
        for test, result in validation.items():
            if test != 'summary' and isinstance(result, dict):
                sig = "✓" if result.get('significant', False) else "✗"
                print(f"{sig} {test}: F={result.get('f_statistic', 0):.2f}, p={result.get('p_value', 1):.4f}")

        if 'summary' in validation:
            summary = validation['summary']
            print(f"\nSummary: {summary['n_significant']}/{summary['n_tests']} tests significant")
            print(f"Proportion: {summary['proportion_significant']:.1%}")
