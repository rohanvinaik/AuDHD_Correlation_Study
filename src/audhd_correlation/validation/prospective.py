"""Prospective outcome prediction and validation

Predicts clinical outcomes for new samples and validates predictions over time.
"""
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class OutcomePrediction:
    """Prediction for a single outcome"""
    sample_id: str
    cluster_id: int
    predicted_outcome: float
    prediction_confidence: float
    outcome_type: str  # 'binary', 'continuous', 'time_to_event'
    prediction_date: datetime
    follow_up_date: Optional[datetime] = None
    observed_outcome: Optional[float] = None
    prediction_error: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProspectiveValidationResult:
    """Results of prospective validation"""
    outcome_name: str
    n_predictions: int
    n_observed: int
    prediction_accuracy: float
    calibration_slope: float
    calibration_intercept: float
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    predictions: List[OutcomePrediction] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class OutcomePredictor:
    """Predicts clinical outcomes from cluster assignments"""

    def __init__(
        self,
        outcome_type: str = 'binary',
        model: Optional[BaseEstimator] = None,
        calibrate: bool = True,
    ):
        """
        Initialize predictor

        Args:
            outcome_type: Type of outcome ('binary', 'continuous', 'time_to_event')
            model: Prediction model (if None, uses default)
            calibrate: Whether to calibrate probabilities
        """
        self.outcome_type = outcome_type
        self.calibrate = calibrate

        if model is None:
            if outcome_type == 'binary':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                )
            elif outcome_type == 'continuous':
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                )
            else:
                raise ValueError(f"Unsupported outcome type: {outcome_type}")
        else:
            self.model = model

        self.calibration_model = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
    ):
        """
        Fit outcome prediction model

        Args:
            X: Feature matrix
            y: Outcome values
            cluster_labels: Cluster labels (optional, can be included in X)
        """
        # Combine features and cluster labels if provided
        if cluster_labels is not None:
            X_combined = np.column_stack([X, cluster_labels.reshape(-1, 1)])
        else:
            X_combined = X

        # Fit main model
        self.model.fit(X_combined, y)

        # Fit calibration model if requested
        if self.calibrate and self.outcome_type == 'binary':
            from sklearn.calibration import CalibratedClassifierCV
            self.calibration_model = CalibratedClassifierCV(
                self.model,
                cv=5,
                method='sigmoid',
            )
            self.calibration_model.fit(X_combined, y)

        self.is_fitted = True

    def predict(
        self,
        X: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
        return_confidence: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict outcomes

        Args:
            X: Feature matrix
            cluster_labels: Cluster labels
            return_confidence: Whether to return confidence scores

        Returns:
            Tuple of (predictions, confidences)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Combine features and cluster labels
        if cluster_labels is not None:
            X_combined = np.column_stack([X, cluster_labels.reshape(-1, 1)])
        else:
            X_combined = X

        # Get predictions
        if self.outcome_type == 'binary':
            if self.calibrate and self.calibration_model is not None:
                predictions = self.calibration_model.predict(X_combined)
                if return_confidence:
                    confidences = self.calibration_model.predict_proba(X_combined)[:, 1]
                else:
                    confidences = None
            else:
                predictions = self.model.predict(X_combined)
                if return_confidence:
                    confidences = self.model.predict_proba(X_combined)[:, 1]
                else:
                    confidences = None
        else:
            predictions = self.model.predict(X_combined)
            if return_confidence:
                # For regression, use prediction interval as confidence
                confidences = self._estimate_prediction_intervals(X_combined)
            else:
                confidences = None

        return predictions, confidences

    def _estimate_prediction_intervals(
        self,
        X: np.ndarray,
        alpha: float = 0.05,
    ) -> np.ndarray:
        """Estimate prediction intervals for regression"""
        # Use ensemble predictions if available
        if hasattr(self.model, 'estimators_'):
            predictions = np.array([
                estimator.predict(X)
                for estimator in self.model.estimators_
            ])
            std = predictions.std(axis=0)
        else:
            # Use simple residual-based estimate
            std = np.ones(len(X)) * 0.1  # Placeholder

        return std

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
        cv: int = 5,
    ) -> Dict[str, float]:
        """
        Perform cross-validation

        Args:
            X: Feature matrix
            y: Outcome values
            cluster_labels: Cluster labels
            cv: Number of folds

        Returns:
            Dictionary of CV metrics
        """
        if cluster_labels is not None:
            X_combined = np.column_stack([X, cluster_labels.reshape(-1, 1)])
        else:
            X_combined = X

        if self.outcome_type == 'binary':
            scoring = ['roc_auc', 'average_precision', 'accuracy']
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
            cv_splitter = cv

        results = {}
        for score in scoring:
            scores = cross_val_score(
                self.model,
                X_combined,
                y,
                cv=cv_splitter,
                scoring=score,
            )
            results[score] = scores.mean()
            results[f'{score}_std'] = scores.std()

        return results


class ProspectiveValidator:
    """Validates predictions prospectively as outcomes are observed"""

    def __init__(self):
        """Initialize validator"""
        self.predictions: Dict[str, List[OutcomePrediction]] = {}

    def register_predictions(
        self,
        predictions: List[OutcomePrediction],
        outcome_name: str,
    ):
        """
        Register predictions for prospective validation

        Args:
            predictions: List of outcome predictions
            outcome_name: Name of outcome
        """
        if outcome_name not in self.predictions:
            self.predictions[outcome_name] = []

        self.predictions[outcome_name].extend(predictions)

    def update_observed_outcomes(
        self,
        sample_ids: List[str],
        observed_outcomes: List[float],
        outcome_name: str,
        observation_date: Optional[datetime] = None,
    ):
        """
        Update with observed outcomes

        Args:
            sample_ids: Sample identifiers
            observed_outcomes: Observed outcome values
            outcome_name: Name of outcome
            observation_date: Date of observation
        """
        if outcome_name not in self.predictions:
            warnings.warn(f"No predictions registered for outcome {outcome_name}")
            return

        if observation_date is None:
            observation_date = datetime.now()

        # Update predictions with observed values
        for sample_id, observed in zip(sample_ids, observed_outcomes):
            for pred in self.predictions[outcome_name]:
                if pred.sample_id == sample_id:
                    pred.observed_outcome = observed
                    pred.follow_up_date = observation_date
                    pred.prediction_error = abs(pred.predicted_outcome - observed)

    def validate(
        self,
        outcome_name: str,
    ) -> Optional[ProspectiveValidationResult]:
        """
        Validate predictions for an outcome

        Args:
            outcome_name: Name of outcome

        Returns:
            ProspectiveValidationResult or None if insufficient data
        """
        if outcome_name not in self.predictions:
            return None

        preds = self.predictions[outcome_name]

        # Filter to predictions with observed outcomes
        observed_preds = [
            p for p in preds
            if p.observed_outcome is not None
        ]

        if len(observed_preds) == 0:
            return None

        # Extract values
        predicted = np.array([p.predicted_outcome for p in observed_preds])
        observed = np.array([p.observed_outcome for p in observed_preds])

        # Calculate metrics based on outcome type
        outcome_type = observed_preds[0].outcome_type

        if outcome_type == 'binary':
            # Classification metrics
            auc_roc = roc_auc_score(observed, predicted)
            auc_pr = average_precision_score(observed, predicted)

            # Brier score as accuracy
            accuracy = 1.0 - np.mean((predicted - observed) ** 2)

            # Calibration
            calib_slope, calib_intercept = self._calculate_calibration(
                predicted, observed
            )

            result = ProspectiveValidationResult(
                outcome_name=outcome_name,
                n_predictions=len(preds),
                n_observed=len(observed_preds),
                prediction_accuracy=accuracy,
                calibration_slope=calib_slope,
                calibration_intercept=calib_intercept,
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                predictions=observed_preds,
            )

        else:  # continuous
            # Regression metrics
            mae = mean_absolute_error(observed, predicted)
            rmse = np.sqrt(mean_squared_error(observed, predicted))
            r2 = r2_score(observed, predicted)

            # Calibration
            calib_slope, calib_intercept = self._calculate_calibration(
                predicted, observed
            )

            result = ProspectiveValidationResult(
                outcome_name=outcome_name,
                n_predictions=len(preds),
                n_observed=len(observed_preds),
                prediction_accuracy=r2,
                calibration_slope=calib_slope,
                calibration_intercept=calib_intercept,
                mae=mae,
                rmse=rmse,
                r2=r2,
                predictions=observed_preds,
            )

        return result

    def _calculate_calibration(
        self,
        predicted: np.ndarray,
        observed: np.ndarray,
    ) -> Tuple[float, float]:
        """Calculate calibration slope and intercept"""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(predicted.reshape(-1, 1), observed)

        return float(model.coef_[0]), float(model.intercept_)

    def plot_calibration(
        self,
        outcome_name: str,
        output_path: Optional[str] = None,
    ):
        """
        Plot calibration curve

        Args:
            outcome_name: Name of outcome
            output_path: Output path for plot
        """
        result = self.validate(outcome_name)

        if result is None:
            warnings.warn(f"No validation data for {outcome_name}")
            return

        predicted = np.array([p.predicted_outcome for p in result.predictions])
        observed = np.array([p.observed_outcome for p in result.predictions])

        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.scatter(predicted, observed, alpha=0.5, label='Predictions')

        # Calibration line
        x_range = np.linspace(predicted.min(), predicted.max(), 100)
        y_calib = result.calibration_slope * x_range + result.calibration_intercept
        ax.plot(x_range, y_calib, 'r-', label='Calibration line')

        # Perfect calibration
        ax.plot(x_range, x_range, 'k--', alpha=0.3, label='Perfect calibration')

        ax.set_xlabel('Predicted Outcome')
        ax.set_ylabel('Observed Outcome')
        ax.set_title(f'Calibration: {outcome_name}\n'
                     f'Slope={result.calibration_slope:.2f}, '
                     f'Intercept={result.calibration_intercept:.2f}')
        ax.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def predict_treatment_response(
    cluster_id: int,
    cluster_characteristics: Dict,
    treatment_history: Optional[pd.DataFrame] = None,
    baseline_severity: Optional[float] = None,
) -> Dict[str, float]:
    """
    Predict treatment response based on cluster

    Args:
        cluster_id: Cluster identifier
        cluster_characteristics: Cluster characteristics
        treatment_history: Historical treatment outcomes
        baseline_severity: Baseline severity score

    Returns:
        Dictionary of treatment response predictions
    """
    # Default response rates by cluster (would be learned from data)
    default_responses = {
        'stimulant': 0.65,
        'ssri': 0.55,
        'cbt': 0.70,
        'behavioral': 0.60,
    }

    # Adjust based on cluster characteristics
    severity = cluster_characteristics.get('severity', 'moderate')

    if severity == 'severe':
        # Lower response rates for severe cases
        responses = {k: v * 0.8 for k, v in default_responses.items()}
    elif severity == 'mild':
        # Higher response rates for mild cases
        responses = {k: v * 1.2 for k, v in default_responses.items()}
    else:
        responses = default_responses.copy()

    # Adjust based on baseline severity if provided
    if baseline_severity is not None:
        severity_factor = 1.0 - (baseline_severity / 100) * 0.3
        responses = {k: min(1.0, v * severity_factor) for k, v in responses.items()}

    # Clip to valid range
    responses = {k: max(0.0, min(1.0, v)) for k, v in responses.items()}

    return responses


def calculate_time_to_event_predictions(
    survival_data: pd.DataFrame,
    cluster_labels: np.ndarray,
    event_col: str = 'event',
    time_col: str = 'time',
) -> Dict[int, Dict[str, float]]:
    """
    Calculate time-to-event predictions per cluster

    Args:
        survival_data: Survival data with event and time columns
        cluster_labels: Cluster labels
        event_col: Name of event column
        time_col: Name of time column

    Returns:
        Dictionary mapping cluster_id to survival statistics
    """
    from lifelines import KaplanMeierFitter

    predictions = {}

    for cluster_id in np.unique(cluster_labels):
        if cluster_id < 0:
            continue

        cluster_mask = cluster_labels == cluster_id
        cluster_survival = survival_data[cluster_mask]

        # Fit Kaplan-Meier
        kmf = KaplanMeierFitter()
        kmf.fit(
            cluster_survival[time_col],
            cluster_survival[event_col],
        )

        # Extract statistics
        median_survival = kmf.median_survival_time_
        survival_at_1yr = kmf.survival_function_at_times(365).iloc[0]

        predictions[int(cluster_id)] = {
            'median_survival_days': float(median_survival),
            'survival_at_1year': float(survival_at_1yr),
            'n_events': int(cluster_survival[event_col].sum()),
            'n_censored': int((~cluster_survival[event_col]).sum()),
        }

    return predictions