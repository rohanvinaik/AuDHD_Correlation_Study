#!/usr/bin/env python3
"""
Uncertainty Quantification
Provides confidence estimates for predictions and statistical tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """Results from uncertainty quantification"""
    point_estimates: np.ndarray
    confidence_intervals: np.ndarray
    prediction_intervals: np.ndarray
    calibration_metrics: Dict[str, float]


class UncertaintyQuantifier:
    """
    Uncertainty quantification for AuDHD predictions

    Capabilities:
    1. Conformal prediction (distribution-free)
    2. Bootstrap confidence intervals
    3. Bayesian credible intervals
    4. Monte Carlo dropout
    5. Calibration assessment
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize quantifier

        Parameters
        ----------
        confidence_level : float
            Confidence level for intervals
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: callable,
        n_bootstrap: int = 1000
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Bootstrap confidence interval

        Parameters
        ----------
        data : np.ndarray
            Sample data
        statistic : callable
            Function to compute statistic (e.g., np.mean)
        n_bootstrap : int
            Number of bootstrap samples

        Returns
        -------
        point_estimate : float
        confidence_interval : Tuple[float, float]
        """
        logger.info(f"Computing bootstrap CI (n={n_bootstrap})")

        n = len(data)
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(sample))

        bootstrap_stats = np.array(bootstrap_stats)

        point_estimate = statistic(data)

        # Percentile method
        lower = np.percentile(bootstrap_stats, self.alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - self.alpha / 2) * 100)

        logger.info(f"  Point: {point_estimate:.3f}, CI: [{lower:.3f}, {upper:.3f}]")

        return point_estimate, (lower, upper)

    def conformal_prediction(
        self,
        calibration_residuals: np.ndarray,
        test_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Conformal prediction intervals (distribution-free)

        Parameters
        ----------
        calibration_residuals : np.ndarray
            Absolute residuals from calibration set
        test_predictions : np.ndarray
            Point predictions for test set

        Returns
        -------
        prediction_intervals : np.ndarray
            Shape (n_test, 2) with [lower, upper] bounds
        """
        logger.info("Computing conformal prediction intervals")

        # Quantile of calibration residuals
        n_calib = len(calibration_residuals)
        q = np.ceil((n_calib + 1) * (1 - self.alpha)) / n_calib
        quantile = np.quantile(calibration_residuals, q)

        # Prediction intervals
        lower = test_predictions - quantile
        upper = test_predictions + quantile

        prediction_intervals = np.column_stack([lower, upper])

        logger.info(f"  Interval width: {quantile * 2:.3f}")

        return prediction_intervals

    def monte_carlo_dropout(
        self,
        model: Any,
        X: np.ndarray,
        n_samples: int = 100,
        dropout_rate: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo dropout for uncertainty estimation

        Parameters
        ----------
        model : Any
            Model with dropout layers
        X : np.ndarray
            Input data
        n_samples : int
            Number of MC samples
        dropout_rate : float
            Dropout probability

        Returns
        -------
        mean_prediction : np.ndarray
        uncertainty : np.ndarray
            Predictive standard deviation
        """
        logger.info(f"Computing MC dropout uncertainty (n={n_samples})")

        # Placeholder - would use actual model with dropout
        # For demonstration: simulate predictions
        predictions = []

        for _ in range(n_samples):
            # Simulate dropout by adding noise
            noise = np.random.randn(*X.shape) * dropout_rate
            pred = X.mean(axis=1) + noise.mean(axis=1)
            predictions.append(pred)

        predictions = np.array(predictions)

        mean_prediction = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)

        logger.info(f"  Mean uncertainty: {uncertainty.mean():.3f}")

        return mean_prediction, uncertainty

    def bayesian_credible_interval(
        self,
        posterior_samples: np.ndarray
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Bayesian credible interval from posterior samples

        Parameters
        ----------
        posterior_samples : np.ndarray
            MCMC or variational samples from posterior

        Returns
        -------
        posterior_mean : float
        credible_interval : Tuple[float, float]
        """
        logger.info("Computing Bayesian credible interval")

        posterior_mean = posterior_samples.mean()

        # Highest Posterior Density interval
        sorted_samples = np.sort(posterior_samples)
        n = len(sorted_samples)

        interval_width = int(n * (1 - self.alpha))
        interval_starts = sorted_samples[:n - interval_width]
        interval_ends = sorted_samples[interval_width:]

        # Find narrowest interval
        widths = interval_ends - interval_starts
        min_idx = np.argmin(widths)

        lower = interval_starts[min_idx]
        upper = interval_ends[min_idx]

        logger.info(f"  Posterior mean: {posterior_mean:.3f}, HDI: [{lower:.3f}, {upper:.3f}]")

        return posterior_mean, (lower, upper)

    def assess_calibration(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Assess calibration of probabilistic predictions

        Parameters
        ----------
        predictions : np.ndarray
            Predicted probabilities
        actuals : np.ndarray
            Actual outcomes (0/1)
        n_bins : int
            Number of calibration bins

        Returns
        -------
        calibration_metrics : Dict
            ECE, MCE, Brier score
        """
        logger.info("Assessing calibration")

        # Expected Calibration Error (ECE)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        mce = 0.0

        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() == 0:
                continue

            bin_predictions = predictions[mask]
            bin_actuals = actuals[mask]

            bin_confidence = bin_predictions.mean()
            bin_accuracy = bin_actuals.mean()

            bin_error = abs(bin_confidence - bin_accuracy)
            bin_weight = mask.sum() / len(predictions)

            ece += bin_weight * bin_error
            mce = max(mce, bin_error)

        # Brier score
        brier = ((predictions - actuals) ** 2).mean()

        logger.info(f"  ECE: {ece:.4f}, MCE: {mce:.4f}, Brier: {brier:.4f}")

        return {
            'ECE': ece,
            'MCE': mce,
            'Brier_score': brier
        }

    def propagate_uncertainty(
        self,
        input_uncertainty: np.ndarray,
        function: callable,
        n_samples: int = 1000
    ) -> Tuple[float, float]:
        """
        Propagate input uncertainty through function

        Parameters
        ----------
        input_uncertainty : np.ndarray
            Distribution of inputs
        function : callable
            Function to apply
        n_samples : int
            Monte Carlo samples

        Returns
        -------
        output_mean : float
        output_std : float
        """
        logger.info("Propagating uncertainty through function")

        # Monte Carlo sampling
        outputs = []

        for _ in range(n_samples):
            sample = np.random.choice(input_uncertainty)
            output = function(sample)
            outputs.append(output)

        output_mean = np.mean(outputs)
        output_std = np.std(outputs)

        logger.info(f"  Output: {output_mean:.3f} ± {output_std:.3f}")

        return output_mean, output_std

    def sensitivity_analysis(
        self,
        model: callable,
        X: np.ndarray,
        perturbation_scale: float = 0.1
    ) -> pd.DataFrame:
        """
        Sensitivity analysis: how predictions change with input perturbations

        Parameters
        ----------
        model : callable
            Prediction model
        X : np.ndarray
            Input features (n_samples × n_features)
        perturbation_scale : float
            Scale of perturbations

        Returns
        -------
        sensitivity : pd.DataFrame
            Feature sensitivities
        """
        logger.info("Performing sensitivity analysis")

        n_samples, n_features = X.shape

        # Baseline predictions
        baseline_preds = model(X)

        sensitivities = []

        for feat_idx in range(n_features):
            # Perturb feature
            X_perturbed = X.copy()
            perturbation = np.random.randn(n_samples) * perturbation_scale
            X_perturbed[:, feat_idx] += perturbation

            # New predictions
            perturbed_preds = model(X_perturbed)

            # Sensitivity: mean absolute change
            sensitivity = np.abs(perturbed_preds - baseline_preds).mean()

            sensitivities.append({
                'feature': feat_idx,
                'sensitivity': sensitivity
            })

        sensitivity_df = pd.DataFrame(sensitivities).sort_values('sensitivity', ascending=False)

        logger.info(f"  Most sensitive feature: {sensitivity_df.iloc[0]['feature']}")

        return sensitivity_df

    def analyze_complete(
        self,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None,
        calibration_residuals: Optional[np.ndarray] = None
    ) -> UncertaintyResult:
        """
        Complete uncertainty quantification pipeline

        Parameters
        ----------
        predictions : np.ndarray
        actuals : np.ndarray, optional
        calibration_residuals : np.ndarray, optional

        Returns
        -------
        UncertaintyResult
        """
        logger.info("=== Complete Uncertainty Quantification ===")

        # 1. Point estimates
        point_estimates = predictions

        # 2. Confidence intervals (bootstrap)
        if actuals is not None:
            residuals = np.abs(predictions - actuals)
            _, ci = self.bootstrap_ci(residuals, np.mean)
            confidence_intervals = np.tile(ci, (len(predictions), 1))
        else:
            confidence_intervals = np.zeros((len(predictions), 2))

        # 3. Prediction intervals (conformal)
        if calibration_residuals is not None:
            prediction_intervals = self.conformal_prediction(
                calibration_residuals, predictions
            )
        else:
            prediction_intervals = np.zeros((len(predictions), 2))

        # 4. Calibration
        if actuals is not None:
            calibration_metrics = self.assess_calibration(predictions, actuals)
        else:
            calibration_metrics = {}

        return UncertaintyResult(
            point_estimates=point_estimates,
            confidence_intervals=confidence_intervals,
            prediction_intervals=prediction_intervals,
            calibration_metrics=calibration_metrics
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Uncertainty Quantification Module")
    logger.info("Ready for robust inference in AuDHD study")
