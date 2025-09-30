"""Double holdout validation for clustering

Addresses Point 4: Double holdout and transport validation

Implements:
- Traditional double holdout (train/test/validation)
- Transport validation across cohorts/sites/ancestries
- Prospective validation framework
"""
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

try:
    from ..validation.external import EmbeddingProjector, ValidationMetrics
    from ..validation.ancestry_stratified import AncestryStratifiedValidator
    from ..validation.prospective import ProspectiveValidator
except ImportError:
    warnings.warn("Validation modules not found. Some features disabled.")


@dataclass
class DoubleHoldoutResults:
    """Results from double holdout validation"""
    # Training set results
    train_labels: np.ndarray
    train_metrics: Dict[str, float]

    # Test set results (first holdout)
    test_labels: np.ndarray
    test_metrics: Dict[str, float]
    test_replication_rate: float

    # Validation set results (second holdout)
    validation_labels: np.ndarray
    validation_metrics: Dict[str, float]
    validation_replication_rate: float

    # Cross-holdout agreement
    test_validation_agreement: float

    # Metadata
    n_train: int
    n_test: int
    n_validation: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransportValidationResults:
    """Results from transport validation (across sites/cohorts/ancestries)"""
    source_cohort: str
    target_cohorts: List[str]

    # Per-target results
    replication_rates: Dict[str, float]
    cluster_stability: Dict[str, float]
    effect_sizes: Dict[str, Dict[str, float]]

    # Overall transport
    mean_replication: float
    min_replication: float

    # Ancestry-specific (if applicable)
    ancestry_specific_results: Optional[Dict[str, Any]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


class DoubleHoldoutValidator:
    """
    Double holdout validation for clustering

    Splits data into 3 sets:
    1. Training: Cluster discovery
    2. Test: First holdout validation
    3. Validation: Second holdout validation (confirmatory)

    This prevents overfitting and ensures robustness.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_state: int = 42,
        stratify_by: Optional[np.ndarray] = None,
    ):
        """
        Initialize double holdout validator

        Args:
            test_size: Fraction for test set
            validation_size: Fraction for validation set (from remaining)
            random_state: Random seed
            stratify_by: Optional stratification variable
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.stratify_by = stratify_by

    def split_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/test/validation

        Args:
            X: Feature matrix
            y: Optional labels for stratification

        Returns:
            X_train, X_test, X_val, indices_train, indices_test, indices_val
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # First split: train vs (test + validation)
        stratify_first = self.stratify_by if self.stratify_by is not None else y

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=self.test_size + self.validation_size,
            random_state=self.random_state,
            stratify=stratify_first
        )

        # Second split: test vs validation
        relative_val_size = self.validation_size / (self.test_size + self.validation_size)

        stratify_second = None
        if stratify_first is not None:
            stratify_second = stratify_first[temp_idx]

        test_idx, val_idx = train_test_split(
            temp_idx,
            test_size=relative_val_size,
            random_state=self.random_state,
            stratify=stratify_second
        )

        # Extract data
        X_train = X[train_idx]
        X_test = X[test_idx]
        X_val = X[val_idx]

        return X_train, X_test, X_val, train_idx, test_idx, val_idx

    def validate(
        self,
        X: np.ndarray,
        clustering_pipeline: Any,
        embedding_generator: Optional[Any] = None,
    ) -> DoubleHoldoutResults:
        """
        Run double holdout validation

        Args:
            X: Full feature matrix
            clustering_pipeline: Fitted clustering pipeline
            embedding_generator: Optional embedding generator

        Returns:
            DoubleHoldoutResults
        """
        # Split data
        X_train, X_test, X_val, idx_train, idx_test, idx_val = self.split_data(X)

        print(f"Double holdout split: train={len(idx_train)}, test={len(idx_test)}, val={len(idx_val)}")

        # Get training labels
        train_labels = clustering_pipeline.consensus_labels_[idx_train]

        # Project test set
        if embedding_generator is not None:
            train_embedding = embedding_generator.embeddings_['original'][idx_train]
        else:
            train_embedding = X_train

        projector_test = EmbeddingProjector(
            reference_data=X_train,
            reference_embedding=train_embedding,
            reference_labels=train_labels,
            method='umap'
        )

        test_projection = projector_test.project(X_test)
        test_labels = test_projection.cluster_assignments

        # Project validation set
        projector_val = EmbeddingProjector(
            reference_data=X_train,
            reference_embedding=train_embedding,
            reference_labels=train_labels,
            method='umap'
        )

        val_projection = projector_val.project(X_val)
        val_labels = val_projection.cluster_assignments

        # Compute metrics
        from sklearn.metrics import silhouette_score

        train_metrics = {
            'silhouette': silhouette_score(X_train, train_labels),
            'n_clusters': len(set(train_labels)),
        }

        test_metrics = {
            'silhouette': silhouette_score(X_test, test_labels),
            'n_clusters': len(set(test_labels)),
            'confidence': test_projection.assignment_confidence.mean()
        }

        val_metrics = {
            'silhouette': silhouette_score(X_val, val_labels),
            'n_clusters': len(set(val_labels)),
            'confidence': val_projection.assignment_confidence.mean()
        }

        # Replication rates (cluster structure preserved?)
        test_replication = self._compute_replication_rate(train_metrics, test_metrics)
        val_replication = self._compute_replication_rate(train_metrics, val_metrics)

        # Agreement between test and validation
        # This is key: do two independent holdouts agree?
        agreement = self._compute_holdout_agreement(test_labels, val_labels, X_test, X_val)

        return DoubleHoldoutResults(
            train_labels=train_labels,
            train_metrics=train_metrics,
            test_labels=test_labels,
            test_metrics=test_metrics,
            test_replication_rate=test_replication,
            validation_labels=val_labels,
            validation_metrics=val_metrics,
            validation_replication_rate=val_replication,
            test_validation_agreement=agreement,
            n_train=len(idx_train),
            n_test=len(idx_test),
            n_validation=len(idx_val)
        )

    def _compute_replication_rate(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float]
    ) -> float:
        """
        Compute replication rate

        Simple metric: do we find similar number of clusters with similar quality?
        """
        n_clusters_match = (train_metrics['n_clusters'] == test_metrics['n_clusters'])
        silhouette_ratio = test_metrics['silhouette'] / (train_metrics['silhouette'] + 1e-10)

        # Replication rate: 0-1 score
        # 1.0 = perfect replication (same n_clusters, similar silhouette)
        # 0.0 = no replication

        cluster_score = 1.0 if n_clusters_match else 0.5
        quality_score = min(1.0, silhouette_ratio)

        replication = (cluster_score + quality_score) / 2.0

        return replication

    def _compute_holdout_agreement(
        self,
        labels1: np.ndarray,
        labels2: np.ndarray,
        X1: np.ndarray,
        X2: np.ndarray
    ) -> float:
        """
        Compute agreement between two independent holdout sets

        Can't directly compare labels (different samples), so compare cluster properties
        """
        # Compare cluster distributions
        n_clusters1 = len(set(labels1))
        n_clusters2 = len(set(labels2))

        cluster_diff = abs(n_clusters1 - n_clusters2) / max(n_clusters1, n_clusters2)

        # Compare cluster quality
        from sklearn.metrics import silhouette_score
        sil1 = silhouette_score(X1, labels1)
        sil2 = silhouette_score(X2, labels2)

        quality_diff = abs(sil1 - sil2) / max(abs(sil1), abs(sil2))

        # Agreement score (1.0 = perfect agreement, 0.0 = no agreement)
        agreement = 1.0 - (cluster_diff + quality_diff) / 2.0

        return agreement


class TransportValidator:
    """
    Transport validation across cohorts/sites/ancestries

    Tests whether clustering generalizes across:
    - Different recruitment sites
    - Different cohorts
    - Different ancestry groups
    - Prospective time periods
    """

    def __init__(
        self,
        source_cohort: str,
        target_cohorts: List[str],
        cohort_data: Dict[str, np.ndarray],
        cohort_metadata: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        """
        Initialize transport validator

        Args:
            source_cohort: Name of source cohort (discovery)
            target_cohorts: Names of target cohorts (validation)
            cohort_data: Dict mapping cohort name to feature data
            cohort_metadata: Optional metadata per cohort
        """
        self.source_cohort = source_cohort
        self.target_cohorts = target_cohorts
        self.cohort_data = cohort_data
        self.cohort_metadata = cohort_metadata or {}

    def validate_transport(
        self,
        source_labels: np.ndarray,
        source_embedding: np.ndarray,
    ) -> TransportValidationResults:
        """
        Validate transport across cohorts

        Args:
            source_labels: Cluster labels from source cohort
            source_embedding: Embedding from source cohort

        Returns:
            TransportValidationResults
        """
        source_data = self.cohort_data[self.source_cohort]

        replication_rates = {}
        cluster_stability = {}
        effect_sizes = {}

        for target_cohort in self.target_cohorts:
            print(f"Validating transport: {self.source_cohort} → {target_cohort}")

            target_data = self.cohort_data[target_cohort]

            # Project target cohort into source space
            projector = EmbeddingProjector(
                reference_data=source_data,
                reference_embedding=source_embedding,
                reference_labels=source_labels,
                method='umap'
            )

            target_projection = projector.project(target_data)

            # Compute replication rate
            from sklearn.metrics import silhouette_score

            target_sil = silhouette_score(target_data, target_projection.cluster_assignments)
            source_sil = silhouette_score(source_data, source_labels)

            replication = min(1.0, target_sil / (source_sil + 1e-10))
            replication_rates[target_cohort] = replication

            # Cluster stability (confidence)
            stability = target_projection.assignment_confidence.mean()
            cluster_stability[target_cohort] = stability

            # Effect sizes (placeholder - would compute biomarker differences)
            effect_sizes[target_cohort] = {
                'mean_confidence': stability,
                'silhouette': target_sil
            }

        # Overall transport metrics
        mean_replication = np.mean(list(replication_rates.values()))
        min_replication = np.min(list(replication_rates.values()))

        return TransportValidationResults(
            source_cohort=self.source_cohort,
            target_cohorts=self.target_cohorts,
            replication_rates=replication_rates,
            cluster_stability=cluster_stability,
            effect_sizes=effect_sizes,
            mean_replication=mean_replication,
            min_replication=min_replication
        )


def enable_double_holdout_in_pipeline(
    X: np.ndarray,
    clustering_pipeline: Any,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    external_cohorts: Optional[Dict[str, np.ndarray]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Convenience function to enable double holdout validation

    Args:
        X: Feature matrix
        clustering_pipeline: Fitted clustering pipeline
        test_size: Test set fraction
        validation_size: Validation set fraction
        external_cohorts: Optional external cohorts for transport validation
        random_state: Random seed

    Returns:
        Dict with double holdout and transport validation results
    """
    results = {}

    # Double holdout
    dh_validator = DoubleHoldoutValidator(
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state
    )

    dh_results = dh_validator.validate(X, clustering_pipeline)
    results['double_holdout'] = dh_results

    # Transport validation (if external cohorts provided)
    if external_cohorts:
        transport_validator = TransportValidator(
            source_cohort='discovery',
            target_cohorts=list(external_cohorts.keys()),
            cohort_data={'discovery': X, **external_cohorts}
        )

        transport_results = transport_validator.validate_transport(
            source_labels=clustering_pipeline.consensus_labels_,
            source_embedding=clustering_pipeline.embeddings_.get('original', X)
        )
        results['transport'] = transport_results

    return results
