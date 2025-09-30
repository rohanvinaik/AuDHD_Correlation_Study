"""Sophisticated missing data imputation system

Implements multiple imputation methods for different missingness patterns:
- Delta-adjusted MICE for MNAR data
- missForest for mixed-type data
- KNN with custom distance metrics
- Multiple imputation with proper combining
"""
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.spatial.distance import cdist
from scipy import stats

from ..config.schema import AppConfig


class MissingnessType(Enum):
    """Types of missingness mechanisms"""
    MCAR = "MCAR"  # Missing Completely At Random
    MAR = "MAR"    # Missing At Random
    MNAR = "MNAR"  # Missing Not At Random


@dataclass
class ImputationMetrics:
    """Metrics for imputation quality assessment"""
    rmse: float
    mae: float
    correlation: float
    distribution_divergence: float  # KL divergence
    coverage: float  # Proportion of missing values imputed
    method: str
    missingness_type: MissingnessType


@dataclass
class MissingnessPattern:
    """Missingness pattern diagnostics"""
    missing_rate: float
    missing_by_feature: np.ndarray
    missing_by_sample: np.ndarray
    missingness_type: MissingnessType
    little_test_pvalue: Optional[float] = None  # Little's MCAR test


class DeltaAdjustedMICE:
    """
    Delta-adjusted MICE (Multiple Imputation by Chained Equations) for MNAR data

    Uses delta adjustment to account for systematic differences in missing values,
    appropriate for data missing not at random (MNAR).
    """

    def __init__(
        self,
        n_imputations: int = 5,
        max_iter: int = 10,
        delta: Optional[float] = None,
        estimator: Optional[Any] = None,
        random_state: int = 42,
    ):
        """
        Initialize delta-adjusted MICE

        Args:
            n_imputations: Number of multiple imputations
            max_iter: Maximum MICE iterations
            delta: Adjustment factor for MNAR (auto-estimated if None)
            estimator: Base estimator for MICE
            random_state: Random seed
        """
        self.n_imputations = n_imputations
        self.max_iter = max_iter
        self.delta = delta
        self.estimator = estimator
        self.random_state = random_state
        self.imputations_: List[np.ndarray] = []
        self.delta_estimated_: Optional[float] = None

    def fit_transform(
        self, X: np.ndarray, missingness_indicator: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Fit and transform with multiple imputations

        Args:
            X: Data with missing values
            missingness_indicator: Binary indicator of missingness pattern

        Returns:
            List of imputed datasets
        """
        if missingness_indicator is None:
            missingness_indicator = np.isnan(X)

        # Estimate delta if not provided
        if self.delta is None:
            self.delta_estimated_ = self._estimate_delta(X, missingness_indicator)
        else:
            self.delta_estimated_ = self.delta

        # Create multiple imputations
        self.imputations_ = []

        for m in range(self.n_imputations):
            # Standard MICE imputation
            imputer = IterativeImputer(
                estimator=self.estimator,
                max_iter=self.max_iter,
                random_state=self.random_state + m,
                verbose=0,
            )

            X_imputed = imputer.fit_transform(X)

            # Apply delta adjustment to imputed values
            # Adjust only the values that were missing
            X_adjusted = X_imputed.copy()
            X_adjusted[missingness_indicator] += self.delta_estimated_

            self.imputations_.append(X_adjusted)

        return self.imputations_

    def _estimate_delta(
        self, X: np.ndarray, missingness_indicator: np.ndarray
    ) -> float:
        """
        Estimate delta parameter from observed data

        Uses the difference between observed and initially imputed values
        as a proxy for MNAR bias.

        Args:
            X: Data matrix
            missingness_indicator: Binary missingness indicator

        Returns:
            Estimated delta
        """
        # Simple initial imputation (mean)
        X_simple = X.copy()
        col_means = np.nanmean(X, axis=0)

        for j in range(X.shape[1]):
            X_simple[np.isnan(X_simple[:, j]), j] = col_means[j]

        # MICE imputation
        imputer = IterativeImputer(
            max_iter=5, random_state=self.random_state, verbose=0
        )
        X_mice = imputer.fit_transform(X)

        # Estimate delta from the difference in observed regions near missing values
        # This is a simplified heuristic
        deltas = []

        for j in range(X.shape[1]):
            missing_rows = missingness_indicator[:, j]
            if missing_rows.sum() > 0 and (~missing_rows).sum() > 0:
                # Compare observed values near missing vs far from missing
                observed = X[~missing_rows, j]

                # Simple heuristic: use quantile difference
                if len(observed) > 10:
                    delta_j = np.percentile(observed, 25) - np.percentile(observed, 50)
                    deltas.append(delta_j)

        if deltas:
            return np.median(deltas)
        else:
            return 0.0


class MissForest:
    """
    missForest implementation for mixed-type data

    Uses random forests to impute missing values, handling both
    continuous and categorical variables.
    """

    def __init__(
        self,
        max_iter: int = 10,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        random_state: int = 42,
        verbose: int = 0,
    ):
        """
        Initialize missForest

        Args:
            max_iter: Maximum iterations
            n_estimators: Number of trees in random forest
            criterion: Split criterion
            random_state: Random seed
            verbose: Verbosity level
        """
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.random_state = random_state
        self.verbose = verbose
        self.continuous_features_: Optional[List[int]] = None
        self.categorical_features_: Optional[List[int]] = None

    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Fit and transform with missForest

        Args:
            X: Data with missing values
            categorical_features: Indices of categorical features

        Returns:
            Imputed data
        """
        X_imputed = X.copy()

        # Determine feature types
        if categorical_features is None:
            # Auto-detect categorical features (< 10 unique values)
            self.categorical_features_ = []
            self.continuous_features_ = []

            for j in range(X.shape[1]):
                unique_vals = len(np.unique(X[~np.isnan(X[:, j]), j]))
                if unique_vals < 10:
                    self.categorical_features_.append(j)
                else:
                    self.continuous_features_.append(j)
        else:
            self.categorical_features_ = categorical_features
            self.continuous_features_ = [
                j for j in range(X.shape[1]) if j not in categorical_features
            ]

        # Initial imputation with mean/mode
        for j in range(X.shape[1]):
            missing_mask = np.isnan(X_imputed[:, j])
            if missing_mask.sum() > 0:
                if j in self.categorical_features_:
                    # Mode imputation
                    mode_val = stats.mode(X_imputed[~missing_mask, j], keepdims=True)[0][0]
                    X_imputed[missing_mask, j] = mode_val
                else:
                    # Mean imputation
                    mean_val = np.nanmean(X_imputed[:, j])
                    X_imputed[missing_mask, j] = mean_val

        # Iterative imputation
        missing_indicator = np.isnan(X)

        for iteration in range(self.max_iter):
            X_old = X_imputed.copy()

            # Sort features by amount of missingness
            missing_counts = missing_indicator.sum(axis=0)
            feature_order = np.argsort(missing_counts)

            for j in feature_order:
                if missing_counts[j] == 0:
                    continue

                missing_mask = missing_indicator[:, j]
                observed_mask = ~missing_mask

                if observed_mask.sum() == 0:
                    continue

                # Features for prediction (all except current)
                feature_cols = [k for k in range(X.shape[1]) if k != j]
                X_train = X_imputed[observed_mask][:, feature_cols]
                y_train = X_imputed[observed_mask, j]
                X_test = X_imputed[missing_mask][:, feature_cols]

                # Train appropriate model
                if j in self.categorical_features_:
                    model = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        n_jobs=-1,
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        criterion=self.criterion,
                        random_state=self.random_state,
                        n_jobs=-1,
                    )

                model.fit(X_train, y_train)
                X_imputed[missing_mask, j] = model.predict(X_test)

            # Check convergence
            diff = np.sum((X_imputed - X_old) ** 2) / np.sum(missing_indicator)

            if self.verbose > 0:
                print(f"Iteration {iteration + 1}: MSE = {diff:.6f}")

            if diff < 1e-6:
                break

        return X_imputed


class CustomDistanceKNNImputer:
    """
    KNN imputation with custom distance metrics

    Supports various distance metrics appropriate for different data types.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        distance_metric: str = "euclidean",
        weights: str = "distance",
        distance_function: Optional[Callable] = None,
        feature_weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize custom distance KNN imputer

        Args:
            n_neighbors: Number of neighbors
            distance_metric: Distance metric ('euclidean', 'manhattan', 'mahalanobis', 'correlation', 'custom')
            weights: Weighting scheme ('uniform' or 'distance')
            distance_function: Custom distance function
            feature_weights: Weights for each feature in distance calculation
        """
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.weights = weights
        self.distance_function = distance_function
        self.feature_weights = feature_weights
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform with KNN imputation

        Args:
            X: Data with missing values

        Returns:
            Imputed data
        """
        X_imputed = X.copy()
        missing_indicator = np.isnan(X)

        # Store statistics for distance calculation
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)

        # Handle columns with all missing values
        all_missing_cols = np.all(missing_indicator, axis=0)
        if np.any(all_missing_cols):
            # Set mean to 0 and std to 1 for all-missing columns
            self.mean_[all_missing_cols] = 0.0
            self.std_[all_missing_cols] = 1.0

        # For Mahalanobis distance
        if self.distance_metric == "mahalanobis":
            # Compute covariance on observed data
            X_complete_rows = X[~np.any(missing_indicator, axis=1)]
            if len(X_complete_rows) > X.shape[1]:
                cov = np.cov(X_complete_rows, rowvar=False)
                self.cov_inv_ = np.linalg.pinv(cov)
            else:
                # Fallback to correlation distance
                self.distance_metric = "correlation"

        # Impute each missing value
        for i in range(X.shape[0]):
            missing_cols = np.where(missing_indicator[i])[0]

            if len(missing_cols) == 0:
                continue

            # Find neighbors based on observed features
            observed_cols = np.where(~missing_indicator[i])[0]

            if len(observed_cols) == 0:
                # No observed features, use column mean
                X_imputed[i, missing_cols] = self.mean_[missing_cols]
                continue

            # Compute distances to all other samples
            distances = self._compute_distances(
                X_imputed[i, observed_cols],
                X_imputed[:, observed_cols],
                observed_cols,
            )

            # Exclude self and samples with missing values in target columns
            valid_neighbors = np.ones(len(distances), dtype=bool)
            valid_neighbors[i] = False

            for j in missing_cols:
                valid_neighbors &= ~missing_indicator[:, j]

            if valid_neighbors.sum() == 0:
                # No valid neighbors
                X_imputed[i, missing_cols] = self.mean_[missing_cols]
                continue

            distances[~valid_neighbors] = np.inf

            # Get k nearest neighbors
            k = min(self.n_neighbors, valid_neighbors.sum())
            neighbor_indices = np.argsort(distances)[:k]
            neighbor_distances = distances[neighbor_indices]

            # Impute missing values
            for j in missing_cols:
                neighbor_values = X_imputed[neighbor_indices, j]

                if self.weights == "distance":
                    # Distance-weighted average
                    weights = 1.0 / (neighbor_distances + 1e-10)
                    weights /= weights.sum()
                else:
                    # Uniform weights
                    weights = np.ones(k) / k

                X_imputed[i, j] = np.sum(weights * neighbor_values)

        return X_imputed

    def _compute_distances(
        self,
        x: np.ndarray,
        X_matrix: np.ndarray,
        feature_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute distances using specified metric"""

        if self.distance_metric == "custom" and self.distance_function is not None:
            return np.array([self.distance_function(x, X_matrix[i]) for i in range(len(X_matrix))])

        # Standardize for distance calculation
        x_std = (x - self.mean_[feature_indices]) / (self.std_[feature_indices] + 1e-10)
        X_std = (X_matrix - self.mean_[feature_indices]) / (self.std_[feature_indices] + 1e-10)

        # Apply feature weights if provided
        if self.feature_weights is not None:
            weights = self.feature_weights[feature_indices]
            x_std = x_std * np.sqrt(weights)
            X_std = X_std * np.sqrt(weights)

        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((X_std - x_std) ** 2, axis=1))

        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(X_std - x_std), axis=1)

        elif self.distance_metric == "correlation":
            # 1 - correlation coefficient
            distances = []
            for i in range(len(X_std)):
                if np.std(X_std[i]) > 0 and np.std(x_std) > 0:
                    corr = np.corrcoef(x_std, X_std[i])[0, 1]
                    distances.append(1 - corr)
                else:
                    distances.append(np.inf)
            return np.array(distances)

        elif self.distance_metric == "mahalanobis" and self.cov_inv_ is not None:
            # Mahalanobis distance
            # Need to subset covariance matrix for the features being used
            cov_inv_subset = self.cov_inv_[np.ix_(feature_indices, feature_indices)]
            diff = X_std - x_std
            return np.sqrt(np.sum((diff @ cov_inv_subset) * diff, axis=1))

        else:
            # Default to Euclidean
            return np.sqrt(np.sum((X_std - x_std) ** 2, axis=1))


class MultipleImputation:
    """
    Multiple imputation with proper combining using Rubin's rules

    Generates multiple plausible imputations and combines results appropriately.
    """

    def __init__(
        self,
        base_imputer: Any,
        n_imputations: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize multiple imputation

        Args:
            base_imputer: Base imputation method
            n_imputations: Number of imputations
            random_state: Random seed
        """
        self.base_imputer = base_imputer
        self.n_imputations = n_imputations
        self.random_state = random_state
        self.imputations_: List[np.ndarray] = []

    def fit_transform(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Generate multiple imputations

        Args:
            X: Data with missing values

        Returns:
            List of imputed datasets
        """
        self.imputations_ = []

        for m in range(self.n_imputations):
            # Add random noise to create variability between imputations
            X_noisy = X.copy()

            # Add small amount of noise to observed values
            observed_mask = ~np.isnan(X)
            noise_scale = np.nanstd(X, axis=0) * 0.01
            X_noisy[observed_mask] += np.random.randn(observed_mask.sum()) * noise_scale[
                np.tile(np.arange(X.shape[1]), X.shape[0])[observed_mask.ravel()]
            ]

            # Impute
            if hasattr(self.base_imputer, 'random_state'):
                self.base_imputer.random_state = self.random_state + m

            X_imputed = self.base_imputer.fit_transform(X_noisy)
            self.imputations_.append(X_imputed)

        return self.imputations_

    def pool_results(
        self,
        estimates: List[np.ndarray],
        variances: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pool results from multiple imputations using Rubin's rules

        Args:
            estimates: List of parameter estimates from each imputation
            variances: List of variance estimates from each imputation

        Returns:
            Tuple of (pooled estimate, pooled variance)
        """
        m = len(estimates)

        # Pooled estimate (average across imputations)
        pooled_estimate = np.mean(estimates, axis=0)

        # Within-imputation variance
        within_var = np.mean(variances, axis=0)

        # Between-imputation variance
        between_var = np.var(estimates, axis=0, ddof=1)

        # Total variance (Rubin's rules)
        pooled_variance = within_var + (1 + 1/m) * between_var

        return pooled_estimate, pooled_variance


class ImputationDiagnostics:
    """
    Comprehensive diagnostics for missing data and imputation quality
    """

    @staticmethod
    def analyze_missingness(X: np.ndarray) -> MissingnessPattern:
        """
        Analyze missingness pattern in data

        Args:
            X: Data matrix

        Returns:
            MissingnessPattern object with diagnostics
        """
        missing_indicator = np.isnan(X)

        missing_rate = missing_indicator.sum() / missing_indicator.size
        missing_by_feature = missing_indicator.sum(axis=0) / X.shape[0]
        missing_by_sample = missing_indicator.sum(axis=1) / X.shape[1]

        # Detect missingness type using simple heuristics
        missingness_type = ImputationDiagnostics._detect_missingness_type(
            X, missing_indicator
        )

        # Little's MCAR test (simplified)
        little_pvalue = ImputationDiagnostics._little_mcar_test(X, missing_indicator)

        return MissingnessPattern(
            missing_rate=missing_rate,
            missing_by_feature=missing_by_feature,
            missing_by_sample=missing_by_sample,
            missingness_type=missingness_type,
            little_test_pvalue=little_pvalue,
        )

    @staticmethod
    def _detect_missingness_type(
        X: np.ndarray, missing_indicator: np.ndarray
    ) -> MissingnessType:
        """Detect missingness mechanism using statistical tests"""

        # Simple heuristic: check correlation between missingness and observed values
        correlations = []

        for j in range(X.shape[1]):
            if missing_indicator[:, j].sum() > 0:
                # Check correlation with other features
                for k in range(X.shape[1]):
                    if k != j and not np.all(np.isnan(X[:, k])):
                        missing_j = missing_indicator[:, j].astype(float)
                        observed_k = X[:, k].copy()

                        # Fill missing in k for correlation calculation
                        observed_k[np.isnan(observed_k)] = np.nanmean(observed_k)

                        corr = np.corrcoef(missing_j, observed_k)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

        if not correlations:
            return MissingnessType.MCAR

        max_corr = max(correlations)

        # Thresholds for classification
        if max_corr < 0.1:
            return MissingnessType.MCAR
        elif max_corr < 0.3:
            return MissingnessType.MAR
        else:
            return MissingnessType.MNAR

    @staticmethod
    def _little_mcar_test(
        X: np.ndarray, missing_indicator: np.ndarray, alpha: float = 0.05
    ) -> Optional[float]:
        """
        Simplified Little's MCAR test

        Returns p-value (higher means more consistent with MCAR)
        """
        # Simplified implementation - full test is complex
        # This is a placeholder that does a basic variance homogeneity test

        try:
            # Group samples by missingness pattern
            patterns = {}
            for i in range(X.shape[0]):
                pattern = tuple(missing_indicator[i])
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(i)

            # Need at least 2 patterns
            if len(patterns) < 2:
                return 1.0

            # For each feature, test if mean differs across patterns
            pvalues = []

            for j in range(X.shape[1]):
                groups = []
                for pattern, indices in patterns.items():
                    if not pattern[j]:  # Feature j is observed in this pattern
                        values = X[indices, j]
                        values = values[~np.isnan(values)]
                        if len(values) > 0:
                            groups.append(values)

                if len(groups) >= 2:
                    # One-way ANOVA
                    try:
                        _, pval = stats.f_oneway(*groups)
                        if not np.isnan(pval):
                            pvalues.append(pval)
                    except:
                        pass

            if pvalues:
                # Combine p-values using Fisher's method
                chi2_stat = -2 * np.sum(np.log(pvalues))
                df = 2 * len(pvalues)
                combined_pvalue = 1 - stats.chi2.cdf(chi2_stat, df)
                return combined_pvalue
            else:
                return None

        except Exception:
            return None

    @staticmethod
    def evaluate_imputation_quality(
        X_original: np.ndarray,
        X_imputed: np.ndarray,
        missing_indicator: Optional[np.ndarray] = None,
        method_name: str = "unknown",
    ) -> ImputationMetrics:
        """
        Evaluate imputation quality using multiple metrics

        Args:
            X_original: Original data with missing values
            X_imputed: Imputed data
            missing_indicator: Binary indicator of missing values
            method_name: Name of imputation method

        Returns:
            ImputationMetrics object
        """
        if missing_indicator is None:
            missing_indicator = np.isnan(X_original)

        # For metrics, we can only evaluate on originally observed values
        # In practice, we'd use a validation set with artificially induced missingness

        observed_mask = ~missing_indicator

        if observed_mask.sum() == 0:
            return ImputationMetrics(
                rmse=np.nan,
                mae=np.nan,
                correlation=np.nan,
                distribution_divergence=np.nan,
                coverage=0.0,
                method=method_name,
                missingness_type=MissingnessType.MCAR,
            )

        # RMSE and MAE on observed values (imputed should match original)
        X_original_clean = X_original.copy()
        X_original_clean[missing_indicator] = 0
        X_imputed_clean = X_imputed.copy()
        X_imputed_clean[missing_indicator] = 0

        rmse = np.sqrt(np.mean((X_original_clean[observed_mask] - X_imputed_clean[observed_mask]) ** 2))
        mae = np.mean(np.abs(X_original_clean[observed_mask] - X_imputed_clean[observed_mask]))

        # Correlation between observed values
        try:
            corr = np.corrcoef(
                X_original_clean[observed_mask],
                X_imputed_clean[observed_mask]
            )[0, 1]
        except:
            corr = np.nan

        # Distribution divergence (KL divergence approximation)
        try:
            # Use histogram-based approximation
            hist_orig, bin_edges = np.histogram(
                X_original_clean[observed_mask], bins=20, density=True
            )
            hist_imp, _ = np.histogram(
                X_imputed_clean[observed_mask], bins=bin_edges, density=True
            )

            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            kl_div = np.sum(
                hist_orig * np.log((hist_orig + epsilon) / (hist_imp + epsilon))
            )
        except:
            kl_div = np.nan

        # Coverage
        coverage = 1.0 - (np.isnan(X_imputed).sum() / missing_indicator.sum()) if missing_indicator.sum() > 0 else 1.0

        # Detect missingness type
        pattern = ImputationDiagnostics.analyze_missingness(X_original)

        return ImputationMetrics(
            rmse=rmse,
            mae=mae,
            correlation=corr,
            distribution_divergence=kl_div,
            coverage=coverage,
            method=method_name,
            missingness_type=pattern.missingness_type,
        )


class SensitivityAnalysis:
    """
    Sensitivity analysis for imputation methods

    Compares different imputation methods and parameters to assess robustness.
    """

    def __init__(self, random_state: int = 42):
        """Initialize sensitivity analysis"""
        self.random_state = random_state
        self.results_: Optional[pd.DataFrame] = None

    def compare_methods(
        self,
        X: np.ndarray,
        methods: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple imputation methods

        Args:
            X: Data with missing values
            methods: List of method names to compare

        Returns:
            DataFrame with comparison results
        """
        if methods is None:
            methods = ["mice", "missforest", "knn", "knn_correlation"]

        results = []

        for method_name in methods:
            # Get imputer
            imputer = self._get_imputer(method_name)

            # Impute
            try:
                if method_name == "mice" or method_name == "delta_mice":
                    X_imputed_list = imputer.fit_transform(X)
                    X_imputed = np.mean(X_imputed_list, axis=0)
                else:
                    X_imputed = imputer.fit_transform(X)

                # Evaluate
                metrics = ImputationDiagnostics.evaluate_imputation_quality(
                    X, X_imputed, method_name=method_name
                )

                results.append({
                    "method": method_name,
                    "rmse": metrics.rmse,
                    "mae": metrics.mae,
                    "correlation": metrics.correlation,
                    "kl_divergence": metrics.distribution_divergence,
                    "coverage": metrics.coverage,
                })

            except Exception as e:
                warnings.warn(f"Method {method_name} failed: {str(e)}")

        self.results_ = pd.DataFrame(results)
        return self.results_

    def _get_imputer(self, method_name: str) -> Any:
        """Get imputer instance by name"""

        if method_name == "mice":
            return DeltaAdjustedMICE(
                n_imputations=3, max_iter=5, delta=0.0, random_state=self.random_state
            )

        elif method_name == "delta_mice":
            return DeltaAdjustedMICE(
                n_imputations=3, max_iter=5, random_state=self.random_state
            )

        elif method_name == "missforest":
            return MissForest(
                max_iter=5, n_estimators=50, random_state=self.random_state
            )

        elif method_name == "knn":
            return CustomDistanceKNNImputer(
                n_neighbors=5, distance_metric="euclidean"
            )

        elif method_name == "knn_correlation":
            return CustomDistanceKNNImputer(
                n_neighbors=5, distance_metric="correlation"
            )

        elif method_name == "knn_mahalanobis":
            return CustomDistanceKNNImputer(
                n_neighbors=5, distance_metric="mahalanobis"
            )

        else:
            # Default to simple KNN
            return KNNImputer(n_neighbors=5)

    def plot_comparison(self, output_path: Optional[str] = None) -> None:
        """Plot comparison results"""
        import matplotlib.pyplot as plt

        if self.results_ is None:
            raise ValueError("Must run compare_methods() first")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics = ["rmse", "mae", "correlation", "kl_divergence"]
        titles = ["RMSE", "MAE", "Correlation", "KL Divergence"]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            data = self.results_.sort_values(metric)
            ax.barh(data["method"], data[metric], alpha=0.7)
            ax.set_xlabel(title)
            ax.set_ylabel("Method")
            ax.set_title(f"Comparison: {title}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


# Legacy interface for compatibility
def run(X: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """
    Impute missing values in feature matrices (legacy interface)

    Args:
        X: Dictionary of data matrices
        cfg: Configuration

    Returns:
        Dictionary of imputed data matrices
    """
    imputed = {}

    for modality, data in X.items():
        if isinstance(data, np.ndarray):
            if cfg.preprocess.imputation == "knn":
                imputer = KNNImputer(n_neighbors=5)
                imputed[modality] = imputer.fit_transform(data)
            else:
                imputed[modality] = data
        else:
            imputed[modality] = data

    return imputed