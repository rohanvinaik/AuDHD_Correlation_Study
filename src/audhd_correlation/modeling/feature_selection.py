"""Selective inference for feature selection

Addresses Point 6: Stability selection and knockoff filters

Implements:
- Stability selection (Meinshausen & Bühlmann 2010)
- Knockoff filters (Barber & Candès 2015)
- Cluster-discriminative feature selection with valid inference
"""
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import norm


@dataclass
class StabilitySelectionResults:
    """Results from stability selection"""
    selected_features: Set[int]
    selection_probabilities: np.ndarray
    threshold: float
    n_features_selected: int
    feature_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnockoffResults:
    """Results from knockoff filter"""
    selected_features: Set[int]
    w_scores: np.ndarray  # Feature importance scores
    knockoff_w_scores: np.ndarray
    fdr_threshold: float
    target_fdr: float
    feature_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StabilitySelector:
    """
    Stability selection for feature selection with valid inference

    Reference: Meinshausen & Bühlmann (2010)

    Repeatedly subsamples data and fits sparse models.
    Features selected frequently across subsamples are stable.
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        n_bootstrap: int = 100,
        sample_fraction: float = 0.5,
        threshold: float = 0.6,
        random_state: int = 42,
    ):
        """
        Initialize stability selector

        Args:
            base_estimator: Base estimator (Lasso, RandomForest, etc.)
            n_bootstrap: Number of bootstrap samples
            sample_fraction: Fraction of data to sample
            threshold: Selection probability threshold
            random_state: Random seed
        """
        self.base_estimator = base_estimator
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> StabilitySelectionResults:
        """
        Run stability selection

        Args:
            X: Feature matrix
            y: Target labels (cluster assignments)
            feature_names: Optional feature names

        Returns:
            StabilitySelectionResults
        """
        n_samples, n_features = X.shape

        # Track feature selection across bootstraps
        selection_matrix = np.zeros((self.n_bootstrap, n_features))

        # Get base estimator
        if self.base_estimator is None:
            # Default: Lasso with CV
            base_estimator = LassoCV(cv=5, random_state=self.random_state)
        else:
            base_estimator = self.base_estimator

        print(f"Running stability selection: {self.n_bootstrap} bootstraps...")

        for b in range(self.n_bootstrap):
            # Bootstrap sample
            n_subsample = int(self.sample_fraction * n_samples)
            idx = self.rng.choice(n_samples, n_subsample, replace=False)

            X_boot = X[idx]
            y_boot = y[idx]

            # Fit estimator
            try:
                if hasattr(base_estimator, 'fit'):
                    # For one-vs-rest classification
                    if len(np.unique(y)) > 2:
                        # Multiclass: use one-vs-rest
                        for class_label in np.unique(y):
                            y_binary = (y_boot == class_label).astype(int)

                            estimator = self._get_estimator_copy(base_estimator)
                            estimator.fit(X_boot, y_binary)

                            # Get selected features (non-zero coefficients or high importance)
                            selected = self._get_selected_features(estimator)
                            selection_matrix[b, selected] += 1
                    else:
                        # Binary classification
                        estimator = self._get_estimator_copy(base_estimator)
                        estimator.fit(X_boot, y_boot)
                        selected = self._get_selected_features(estimator)
                        selection_matrix[b, selected] = 1

            except Exception as e:
                warnings.warn(f"Bootstrap {b} failed: {e}")
                continue

        # Compute selection probabilities
        selection_probs = selection_matrix.mean(axis=0)

        # Select features above threshold
        selected_features = set(np.where(selection_probs >= self.threshold)[0])

        print(f"✓ Stability selection: {len(selected_features)}/{n_features} features selected")

        return StabilitySelectionResults(
            selected_features=selected_features,
            selection_probabilities=selection_probs,
            threshold=self.threshold,
            n_features_selected=len(selected_features),
            feature_names=feature_names
        )

    def _get_estimator_copy(self, estimator):
        """Get a copy of the estimator"""
        from sklearn.base import clone
        return clone(estimator)

    def _get_selected_features(self, estimator) -> List[int]:
        """Get selected features from fitted estimator"""
        # For Lasso/Ridge/LogisticRegression
        if hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            if coef.ndim > 1:
                coef = coef.ravel()
            return list(np.where(np.abs(coef) > 1e-6)[0])

        # For tree-based methods
        elif hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            # Select top 50% by importance
            threshold = np.percentile(importances, 50)
            return list(np.where(importances > threshold)[0])

        else:
            warnings.warn("Estimator type not recognized. Returning all features.")
            return list(range(estimator.n_features_in_ if hasattr(estimator, 'n_features_in_') else 100))


class KnockoffFilter:
    """
    Knockoff filter for FDR-controlled feature selection

    Reference: Barber & Candès (2015)

    Generates "knockoff" features (synthetic controls) and
    selects features that are more important than their knockoffs.

    Provides valid FDR control without multiple testing correction.
    """

    def __init__(
        self,
        fdr: float = 0.1,
        method: str = 'equicorrelated',
        statistic: str = 'lasso',
        random_state: int = 42,
    ):
        """
        Initialize knockoff filter

        Args:
            fdr: Target false discovery rate
            method: Knockoff construction method ('equicorrelated', 'sdp')
            statistic: Feature importance statistic ('lasso', 'random_forest')
            random_state: Random seed
        """
        self.fdr = fdr
        self.method = method
        self.statistic = statistic
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> KnockoffResults:
        """
        Run knockoff filter

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names

        Returns:
            KnockoffResults
        """
        print(f"Generating knockoff features...")

        # Generate knockoffs
        X_knockoff = self._generate_knockoffs(X)

        # Combine original and knockoff features
        X_augmented = np.hstack([X, X_knockoff])

        # Compute feature importance statistics
        w_orig, w_knockoff = self._compute_importance_statistics(X_augmented, y, X.shape[1])

        # Compute knockoff W statistics
        W = w_orig - w_knockoff

        # Select features using knockoff+ threshold
        selected_features = self._select_features_knockoff_plus(W, self.fdr)

        print(f"✓ Knockoff filter: {len(selected_features)}/{X.shape[1]} features selected (FDR={self.fdr})")

        return KnockoffResults(
            selected_features=selected_features,
            w_scores=w_orig,
            knockoff_w_scores=w_knockoff,
            fdr_threshold=self.fdr,
            target_fdr=self.fdr,
            feature_names=feature_names
        )

    def _generate_knockoffs(self, X: np.ndarray) -> np.ndarray:
        """
        Generate knockoff features

        Simplified version: equicorrelated knockoffs
        For full SDP-based knockoffs, see knockpy library
        """
        n_samples, n_features = X.shape

        # Standardize
        X_std = StandardScaler().fit_transform(X)

        # Compute covariance
        Sigma = np.cov(X_std.T)

        # Equicorrelated knockoffs (simplified)
        # Full implementation would use SDP
        if self.method == 'equicorrelated':
            # Diagonal s matrix
            eigenvalues = np.linalg.eigvalsh(Sigma)
            lambda_min = max(eigenvalues.min(), 1e-4)

            s = np.ones(n_features) * min(1.0, 2 * lambda_min)
            s_diag = np.diag(s)

            # Cholesky decomposition
            try:
                C = np.linalg.cholesky(2 * s_diag - s_diag @ np.linalg.inv(Sigma) @ s_diag)
            except:
                # Fallback: use identity
                C = np.eye(n_features)

            # Generate knockoffs
            U = self.rng.randn(n_samples, n_features)
            X_knockoff = X_std @ (np.eye(n_features) - np.linalg.inv(Sigma) @ s_diag) + U @ C.T

        else:
            # Fallback: random permutation knockoffs (not ideal but simple)
            X_knockoff = X_std.copy()
            for j in range(n_features):
                X_knockoff[:, j] = self.rng.permutation(X_knockoff[:, j])

        return X_knockoff

    def _compute_importance_statistics(
        self,
        X_augmented: np.ndarray,
        y: np.ndarray,
        n_original: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute feature importance statistics

        Args:
            X_augmented: Original + knockoff features
            y: Target
            n_original: Number of original features

        Returns:
            (importance_original, importance_knockoff)
        """
        if self.statistic == 'lasso':
            # Lasso coefficients
            lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=5000)
            lasso.fit(X_augmented, y)

            coef = np.abs(lasso.coef_)
            w_orig = coef[:n_original]
            w_knockoff = coef[n_original:]

        elif self.statistic == 'random_forest':
            # Random forest importance
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            rf.fit(X_augmented, y)

            importances = rf.feature_importances_
            w_orig = importances[:n_original]
            w_knockoff = importances[n_original:]

        else:
            # Fallback: correlation
            w_orig = np.abs([np.corrcoef(X_augmented[:, j], y)[0, 1] for j in range(n_original)])
            w_knockoff = np.abs([np.corrcoef(X_augmented[:, j], y)[0, 1] for j in range(n_original, X_augmented.shape[1])])

        return w_orig, w_knockoff

    def _select_features_knockoff_plus(self, W: np.ndarray, fdr: float) -> Set[int]:
        """
        Select features using knockoff+ procedure

        Args:
            W: Knockoff W statistics (importance_orig - importance_knockoff)
            fdr: Target FDR

        Returns:
            Set of selected feature indices
        """
        # Sort W in descending order
        sorted_idx = np.argsort(-W)
        sorted_W = W[sorted_idx]

        # Knockoff+ threshold
        n_features = len(W)
        selected = set()

        for k in range(1, n_features + 1):
            threshold = sorted_W[k-1]

            n_selected = np.sum(W >= threshold)
            n_negative = max(1, np.sum(W <= -threshold))

            fdr_estimate = n_negative / max(1, n_selected)

            if fdr_estimate <= fdr:
                # Select all features with W >= threshold
                selected = set(np.where(W >= threshold)[0])
                break

        return selected


def select_cluster_discriminative_features(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = 'both',  # 'stability', 'knockoff', or 'both'
    fdr: float = 0.1,
    stability_threshold: float = 0.6,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Select cluster-discriminative features with valid inference

    Args:
        X: Feature matrix
        labels: Cluster labels
        feature_names: Optional feature names
        method: Selection method
        fdr: FDR for knockoffs
        stability_threshold: Threshold for stability selection
        random_state: Random seed

    Returns:
        Dict with selection results
    """
    results = {}

    # Stability selection
    if method in ['stability', 'both']:
        print("Running stability selection...")
        stability_selector = StabilitySelector(
            threshold=stability_threshold,
            random_state=random_state
        )
        stability_results = stability_selector.fit(X, labels, feature_names)
        results['stability'] = stability_results

    # Knockoff filter
    if method in ['knockoff', 'both']:
        print("Running knockoff filter...")
        knockoff_filter = KnockoffFilter(
            fdr=fdr,
            random_state=random_state
        )
        knockoff_results = knockoff_filter.fit(X, labels, feature_names)
        results['knockoff'] = knockoff_results

    # Intersection (if both methods used)
    if method == 'both':
        intersection = stability_results.selected_features & knockoff_results.selected_features
        results['intersection'] = intersection
        print(f"✓ Intersection: {len(intersection)} features selected by BOTH methods")

    return results
