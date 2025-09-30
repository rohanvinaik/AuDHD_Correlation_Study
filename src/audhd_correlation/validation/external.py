"""External validation pipeline

Projects new samples into existing embedding space and validates cluster assignments
across independent cohorts.
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
import umap


@dataclass
class ProjectionResult:
    """Result of projecting samples into existing space"""
    projected_embedding: np.ndarray
    cluster_assignments: np.ndarray
    assignment_confidence: np.ndarray
    distances_to_centroids: np.ndarray
    silhouette_scores: np.ndarray
    metadata: Dict = field(default_factory=dict)


@dataclass
class ValidationMetrics:
    """Validation metrics for external cohort"""
    replication_rate: float
    cluster_stability: float
    silhouette_score: float
    adjusted_rand_index: Optional[float] = None
    normalized_mutual_info: Optional[float] = None
    biomarker_correlation: Optional[Dict[str, float]] = None
    effect_size_correlation: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class EmbeddingProjector:
    """Projects new samples into existing embedding space"""

    def __init__(
        self,
        reference_data: np.ndarray,
        reference_embedding: np.ndarray,
        reference_labels: np.ndarray,
        method: str = 'umap',
        scaler: Optional[StandardScaler] = None,
    ):
        """
        Initialize projector

        Args:
            reference_data: Reference feature data (n_samples, n_features)
            reference_embedding: Reference embedding coordinates
            reference_labels: Reference cluster labels
            method: Projection method ('umap', 'pca', 'linear')
            scaler: Pre-fitted scaler for normalization
        """
        self.reference_data = reference_data
        self.reference_embedding = reference_embedding
        self.reference_labels = reference_labels
        self.method = method
        self.scaler = scaler or StandardScaler()

        # Fit scaler if not provided
        if scaler is None:
            self.scaler.fit(reference_data)

        # Calculate cluster centroids
        self.cluster_centroids = self._calculate_centroids()

        # Fit projection model
        self.projection_model = self._fit_projection_model()

    def _calculate_centroids(self) -> Dict[int, np.ndarray]:
        """Calculate centroids for each cluster in embedding space"""
        centroids = {}
        for label in np.unique(self.reference_labels):
            if label >= 0:  # Exclude noise points
                mask = self.reference_labels == label
                centroids[label] = self.reference_embedding[mask].mean(axis=0)
        return centroids

    def _fit_projection_model(self):
        """Fit projection model based on method"""
        if self.method == 'umap':
            # Fit UMAP with target metric for supervised projection
            model = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42,
            )
            model.fit(self.scaler.transform(self.reference_data))
            return model

        elif self.method == 'pca':
            # Fit PCA
            n_components = min(self.reference_embedding.shape[1], 50)
            model = PCA(n_components=n_components, random_state=42)
            model.fit(self.scaler.transform(self.reference_data))
            return model

        elif self.method == 'linear':
            # Learn linear projection (least squares)
            X_scaled = self.scaler.transform(self.reference_data)
            # W = (X^T X)^-1 X^T Y
            model = np.linalg.lstsq(X_scaled, self.reference_embedding, rcond=None)[0]
            return model

        else:
            raise ValueError(f"Unknown projection method: {self.method}")

    def project(
        self,
        new_data: np.ndarray,
        assign_clusters: bool = True,
    ) -> ProjectionResult:
        """
        Project new samples into existing embedding space

        Args:
            new_data: New feature data (n_samples, n_features)
            assign_clusters: Whether to assign clusters

        Returns:
            ProjectionResult object
        """
        # Scale new data
        X_scaled = self.scaler.transform(new_data)

        # Project into embedding space
        if self.method == 'umap':
            projected = self.projection_model.transform(X_scaled)

        elif self.method == 'pca':
            projected = self.projection_model.transform(X_scaled)

        elif self.method == 'linear':
            projected = X_scaled @ self.projection_model

        else:
            raise ValueError(f"Unknown projection method: {self.method}")

        # Assign clusters if requested
        if assign_clusters:
            assignments, confidences, distances = self._assign_clusters(projected)

            # Calculate silhouette scores
            if len(np.unique(assignments)) > 1:
                silhouette = silhouette_score(projected, assignments)
                per_sample_silhouette = self._calculate_per_sample_silhouette(
                    projected, assignments
                )
            else:
                silhouette = 0.0
                per_sample_silhouette = np.zeros(len(projected))

        else:
            assignments = np.full(len(projected), -1)
            confidences = np.zeros(len(projected))
            distances = np.full((len(projected), len(self.cluster_centroids)), np.nan)
            per_sample_silhouette = np.zeros(len(projected))

        return ProjectionResult(
            projected_embedding=projected,
            cluster_assignments=assignments,
            assignment_confidence=confidences,
            distances_to_centroids=distances,
            silhouette_scores=per_sample_silhouette,
            metadata={
                'method': self.method,
                'n_clusters': len(self.cluster_centroids),
            },
        )

    def _assign_clusters(
        self,
        embedding: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assign clusters using nearest centroid

        Args:
            embedding: Embedding coordinates

        Returns:
            Tuple of (assignments, confidences, distances)
        """
        # Calculate distances to all centroids
        centroid_labels = sorted(self.cluster_centroids.keys())
        centroid_coords = np.array([
            self.cluster_centroids[label] for label in centroid_labels
        ])

        distances = cdist(embedding, centroid_coords, metric='euclidean')

        # Assign to nearest centroid
        nearest_idx = distances.argmin(axis=1)
        assignments = np.array([centroid_labels[idx] for idx in nearest_idx])

        # Calculate confidence as inverse of distance ratio
        min_dist = distances.min(axis=1)
        sorted_dist = np.sort(distances, axis=1)

        if distances.shape[1] > 1:
            second_min_dist = sorted_dist[:, 1]
            # Confidence: how much closer is nearest vs second nearest
            confidences = 1.0 - (min_dist / (second_min_dist + 1e-10))
        else:
            confidences = np.ones(len(embedding))

        return assignments, confidences, distances

    def _calculate_per_sample_silhouette(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Calculate per-sample silhouette scores"""
        from sklearn.metrics import silhouette_samples
        return silhouette_samples(embedding, labels)


class NearestCentroidClassifier:
    """Nearest centroid classifier for cluster assignment"""

    def __init__(
        self,
        centroids: Dict[int, np.ndarray],
        distance_metric: str = 'euclidean',
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize classifier

        Args:
            centroids: Dictionary mapping cluster IDs to centroid coordinates
            distance_metric: Distance metric to use
            confidence_threshold: Minimum confidence for assignment
        """
        self.centroids = centroids
        self.distance_metric = distance_metric
        self.confidence_threshold = confidence_threshold

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster assignments

        Args:
            X: Sample coordinates (n_samples, n_dims)

        Returns:
            Tuple of (assignments, confidences)
        """
        # Calculate distances to centroids
        centroid_labels = sorted(self.centroids.keys())
        centroid_coords = np.array([
            self.centroids[label] for label in centroid_labels
        ])

        distances = cdist(X, centroid_coords, metric=self.distance_metric)

        # Assign to nearest centroid
        nearest_idx = distances.argmin(axis=1)
        assignments = np.array([centroid_labels[idx] for idx in nearest_idx])

        # Calculate confidence
        min_dist = distances.min(axis=1)
        sorted_dist = np.sort(distances, axis=1)

        if distances.shape[1] > 1:
            second_min_dist = sorted_dist[:, 1]
            confidences = 1.0 - (min_dist / (second_min_dist + 1e-10))
        else:
            confidences = np.ones(len(X))

        # Mark low-confidence assignments as unassigned
        low_confidence = confidences < self.confidence_threshold
        assignments[low_confidence] = -1

        return assignments, confidences

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict cluster probabilities using softmax of negative distances

        Args:
            X: Sample coordinates (n_samples, n_dims)

        Returns:
            Probability matrix (n_samples, n_clusters)
        """
        centroid_labels = sorted(self.centroids.keys())
        centroid_coords = np.array([
            self.centroids[label] for label in centroid_labels
        ])

        distances = cdist(X, centroid_coords, metric=self.distance_metric)

        # Convert distances to probabilities via softmax
        neg_distances = -distances
        exp_distances = np.exp(neg_distances - neg_distances.max(axis=1, keepdims=True))
        probabilities = exp_distances / exp_distances.sum(axis=1, keepdims=True)

        return probabilities


def validate_external_cohort(
    reference_data: np.ndarray,
    reference_embedding: np.ndarray,
    reference_labels: np.ndarray,
    external_data: np.ndarray,
    external_labels: Optional[np.ndarray] = None,
    external_biomarkers: Optional[pd.DataFrame] = None,
    reference_biomarkers: Optional[pd.DataFrame] = None,
    projection_method: str = 'umap',
) -> Tuple[ProjectionResult, ValidationMetrics]:
    """
    Validate model on external cohort

    Args:
        reference_data: Reference feature data
        reference_embedding: Reference embedding
        reference_labels: Reference cluster labels
        external_data: External cohort feature data
        external_labels: External cohort true labels (optional)
        external_biomarkers: External cohort biomarkers
        reference_biomarkers: Reference cohort biomarkers
        projection_method: Projection method

    Returns:
        Tuple of (ProjectionResult, ValidationMetrics)
    """
    # Project external cohort
    projector = EmbeddingProjector(
        reference_data=reference_data,
        reference_embedding=reference_embedding,
        reference_labels=reference_labels,
        method=projection_method,
    )

    projection = projector.project(external_data, assign_clusters=True)

    # Calculate validation metrics
    metrics = {}

    # Replication rate: proportion with confident assignment
    confident = projection.assignment_confidence > 0.5
    metrics['replication_rate'] = confident.mean()

    # Cluster stability: silhouette score
    if len(np.unique(projection.cluster_assignments[confident])) > 1:
        metrics['cluster_stability'] = silhouette_score(
            projection.projected_embedding[confident],
            projection.cluster_assignments[confident],
        )
    else:
        metrics['cluster_stability'] = 0.0

    metrics['silhouette_score'] = projection.silhouette_scores.mean()

    # Compare with true labels if provided
    if external_labels is not None:
        metrics['adjusted_rand_index'] = adjusted_rand_score(
            external_labels[confident],
            projection.cluster_assignments[confident],
        )
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(
            external_labels[confident],
            projection.cluster_assignments[confident],
        )

    # Compare biomarker patterns if provided
    if external_biomarkers is not None and reference_biomarkers is not None:
        biomarker_corrs = {}
        common_features = list(
            set(external_biomarkers.columns) & set(reference_biomarkers.columns)
        )

        for feature in common_features:
            r, p = pearsonr(
                reference_biomarkers[feature].values,
                external_biomarkers[feature].values[:len(reference_biomarkers)],
            )
            if p < 0.05:
                biomarker_corrs[feature] = r

        metrics['biomarker_correlation'] = biomarker_corrs

    validation = ValidationMetrics(
        replication_rate=metrics['replication_rate'],
        cluster_stability=metrics['cluster_stability'],
        silhouette_score=metrics['silhouette_score'],
        adjusted_rand_index=metrics.get('adjusted_rand_index'),
        normalized_mutual_info=metrics.get('normalized_mutual_info'),
        biomarker_correlation=metrics.get('biomarker_correlation'),
        metadata={
            'n_external_samples': len(external_data),
            'n_confident_assignments': confident.sum(),
        },
    )

    return projection, validation


def calculate_replication_metrics(
    reference_features: pd.DataFrame,
    external_features: pd.DataFrame,
    reference_labels: np.ndarray,
    external_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate replication metrics between cohorts

    Args:
        reference_features: Reference feature means per cluster
        external_features: External feature means per cluster
        reference_labels: Reference cluster labels
        external_labels: External cluster labels

    Returns:
        Dictionary of replication metrics
    """
    metrics = {}

    # Ensure same features
    common_features = list(
        set(reference_features.columns) & set(external_features.columns)
    )

    ref_features = reference_features[common_features]
    ext_features = external_features[common_features]

    # Feature correlation across cohorts
    correlations = []
    for feature in common_features:
        r, p = pearsonr(ref_features[feature], ext_features[feature])
        if p < 0.05:
            correlations.append(r)

    if correlations:
        metrics['mean_feature_correlation'] = np.mean(correlations)
        metrics['median_feature_correlation'] = np.median(correlations)
        metrics['n_replicated_features'] = len(correlations)
    else:
        metrics['mean_feature_correlation'] = 0.0
        metrics['median_feature_correlation'] = 0.0
        metrics['n_replicated_features'] = 0

    # Cluster agreement
    metrics['adjusted_rand_index'] = adjusted_rand_score(
        reference_labels, external_labels
    )
    metrics['normalized_mutual_info'] = normalized_mutual_info_score(
        reference_labels, external_labels
    )

    return metrics