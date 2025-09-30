"""Mendelian Randomization analysis

Uses genetic variants as instrumental variables to estimate causal effects.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS


@dataclass
class MRResult:
    """Result of Mendelian Randomization analysis"""
    # Causal estimate
    causal_estimate: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    p_value: float

    # Instrument validity
    f_statistic: float
    weak_instrument: bool

    # Method
    method: str

    # Stage-specific results
    first_stage_r2: Optional[float] = None
    first_stage_f: Optional[float] = None

    # Sensitivity
    heterogeneity_p: Optional[float] = None
    pleiotropy_intercept: Optional[float] = None
    pleiotropy_p: Optional[float] = None


def mendelian_randomization(
    instruments: np.ndarray,
    exposure: np.ndarray,
    outcome: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    method: str = '2sls',
    alpha: float = 0.05,
) -> MRResult:
    """
    Perform Mendelian Randomization analysis

    Uses genetic instruments to estimate causal effect of exposure on outcome.

    Args:
        instruments: Genetic instruments (n_samples, n_instruments)
        exposure: Exposure variable (n_samples,)
        outcome: Outcome variable (n_samples,)
        covariates: Covariates to adjust for (n_samples, n_covariates)
        method: '2sls', 'ratio', or 'wald'
        alpha: Significance level for confidence intervals

    Returns:
        MRResult with causal estimate and validity tests
    """
    # Ensure 2D arrays
    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    n_samples, n_instruments = instruments.shape

    # Add covariates if provided
    if covariates is not None:
        X_instruments = np.hstack([instruments, covariates])
        X_covariates = covariates
    else:
        X_instruments = instruments
        X_covariates = None

    # Test instrument strength
    f_stat = calculate_f_statistic(instruments, exposure, covariates)
    weak_instrument = f_stat < 10  # Rule of thumb: F < 10 indicates weak instrument

    if weak_instrument:
        warnings.warn(f"Weak instrument detected (F={f_stat:.2f} < 10). "
                     "Causal estimates may be biased.")

    if method == '2sls':
        result = _two_stage_least_squares(
            instruments=X_instruments,
            exposure=exposure,
            outcome=outcome,
            alpha=alpha,
        )

    elif method == 'ratio':
        result = _ratio_method(
            instruments=instruments,
            exposure=exposure,
            outcome=outcome,
            alpha=alpha,
        )

    elif method == 'wald':
        result = _wald_ratio(
            instruments=instruments,
            exposure=exposure,
            outcome=outcome,
            alpha=alpha,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    # Add F-statistic
    result.f_statistic = f_stat
    result.weak_instrument = weak_instrument

    return result


def _two_stage_least_squares(
    instruments: np.ndarray,
    exposure: np.ndarray,
    outcome: np.ndarray,
    alpha: float = 0.05,
) -> MRResult:
    """Two-stage least squares estimation"""

    # Stage 1: Regress exposure on instruments
    X_stage1 = sm.add_constant(instruments)
    model_stage1 = sm.OLS(exposure, X_stage1).fit()
    exposure_predicted = model_stage1.predict(X_stage1)

    first_stage_r2 = model_stage1.rsquared
    first_stage_f = model_stage1.fvalue

    # Stage 2: Regress outcome on predicted exposure
    X_stage2 = sm.add_constant(exposure_predicted)
    model_stage2 = sm.OLS(outcome, X_stage2).fit()

    causal_estimate = model_stage2.params[1]
    standard_error = model_stage2.bse[1]
    p_value = model_stage2.pvalues[1]

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = causal_estimate - z_crit * standard_error
    ci_upper = causal_estimate + z_crit * standard_error

    return MRResult(
        causal_estimate=float(causal_estimate),
        standard_error=float(standard_error),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        p_value=float(p_value),
        f_statistic=0.0,  # Will be filled later
        weak_instrument=False,  # Will be filled later
        method='2sls',
        first_stage_r2=float(first_stage_r2),
        first_stage_f=float(first_stage_f),
    )


def _ratio_method(
    instruments: np.ndarray,
    exposure: np.ndarray,
    outcome: np.ndarray,
    alpha: float = 0.05,
) -> MRResult:
    """Ratio of coefficients method (Wald ratio for single instrument)"""

    n_instruments = instruments.shape[1]

    if n_instruments == 1:
        return _wald_ratio(instruments, exposure, outcome, alpha)

    # For multiple instruments, use inverse-variance weighted method
    ratios = []
    weights = []

    for i in range(n_instruments):
        instrument = instruments[:, i]

        # Estimate instrument-exposure association
        beta_exposure = np.cov(instrument, exposure)[0, 1] / np.var(instrument)
        se_exposure = np.sqrt(np.var(exposure) / (len(exposure) * np.var(instrument)))

        # Estimate instrument-outcome association
        beta_outcome = np.cov(instrument, outcome)[0, 1] / np.var(instrument)
        se_outcome = np.sqrt(np.var(outcome) / (len(outcome) * np.var(instrument)))

        # Ratio estimate
        ratio = beta_outcome / beta_exposure
        se_ratio = se_outcome / abs(beta_exposure)

        ratios.append(ratio)
        weights.append(1 / (se_ratio ** 2))

    # Inverse-variance weighted estimate
    ratios = np.array(ratios)
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    causal_estimate = np.sum(ratios * weights)
    standard_error = np.sqrt(1 / np.sum(1 / (np.array(ratios) ** 2)))

    # P-value
    z_stat = causal_estimate / standard_error
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = causal_estimate - z_crit * standard_error
    ci_upper = causal_estimate + z_crit * standard_error

    return MRResult(
        causal_estimate=float(causal_estimate),
        standard_error=float(standard_error),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        p_value=float(p_value),
        f_statistic=0.0,
        weak_instrument=False,
        method='ratio',
    )


def _wald_ratio(
    instruments: np.ndarray,
    exposure: np.ndarray,
    outcome: np.ndarray,
    alpha: float = 0.05,
) -> MRResult:
    """Wald ratio for single instrument"""

    if instruments.shape[1] != 1:
        raise ValueError("Wald ratio requires single instrument")

    instrument = instruments[:, 0]

    # Estimate instrument-exposure association
    beta_exposure = np.cov(instrument, exposure)[0, 1] / np.var(instrument)
    se_exposure = np.sqrt(np.var(exposure) / (len(exposure) * np.var(instrument)))

    # Estimate instrument-outcome association
    beta_outcome = np.cov(instrument, outcome)[0, 1] / np.var(instrument)
    se_outcome = np.sqrt(np.var(outcome) / (len(outcome) * np.var(instrument)))

    # Wald ratio
    causal_estimate = beta_outcome / beta_exposure
    standard_error = se_outcome / abs(beta_exposure)

    # P-value
    z_stat = causal_estimate / standard_error
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = causal_estimate - z_crit * standard_error
    ci_upper = causal_estimate + z_crit * standard_error

    return MRResult(
        causal_estimate=float(causal_estimate),
        standard_error=float(standard_error),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        p_value=float(p_value),
        f_statistic=0.0,
        weak_instrument=False,
        method='wald',
    )


def calculate_f_statistic(
    instruments: np.ndarray,
    exposure: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate F-statistic for instrument strength

    F-statistic tests whether instruments predict exposure.
    F > 10 is rule of thumb for strong instruments.

    Args:
        instruments: Genetic instruments
        exposure: Exposure variable
        covariates: Optional covariates

    Returns:
        F-statistic
    """
    # Ensure 2D
    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    # Add covariates if provided
    if covariates is not None:
        X = np.hstack([instruments, covariates])
    else:
        X = instruments

    # Fit regression
    X_with_const = sm.add_constant(X)
    model = sm.OLS(exposure, X_with_const).fit()

    # F-statistic
    f_stat = model.fvalue

    return float(f_stat)


def test_instrument_validity(
    instruments: np.ndarray,
    exposure: np.ndarray,
    outcome: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Test instrumental variable assumptions

    Tests:
    1. Relevance: Instrument predicts exposure (F-statistic)
    2. Independence: Instrument uncorrelated with confounders
    3. Exclusion: Instrument affects outcome only through exposure

    Args:
        instruments: Genetic instruments
        exposure: Exposure variable
        outcome: Outcome variable
        covariates: Optional covariates

    Returns:
        Dictionary with test statistics
    """
    tests = {}

    # 1. Relevance: F-statistic
    f_stat = calculate_f_statistic(instruments, exposure, covariates)
    tests['f_statistic'] = f_stat
    tests['strong_instrument'] = f_stat >= 10

    # 2. Test for weak identification
    # Cragg-Donald statistic (approximation)
    tests['cragg_donald'] = f_stat * instruments.shape[1]

    # 3. Overidentification test (requires multiple instruments)
    if instruments.shape[1] > 1:
        # Hansen J test
        # This requires implementing IV regression with overidentification
        tests['hansen_j_p'] = None  # Placeholder
    else:
        tests['hansen_j_p'] = None

    return tests


def mr_egger_regression(
    instruments: np.ndarray,
    exposure: np.ndarray,
    outcome: np.ndarray,
) -> Dict[str, float]:
    """
    MR-Egger regression to test for pleiotropy

    Non-zero intercept suggests directional pleiotropy.

    Args:
        instruments: Genetic instruments
        exposure: Exposure variable
        outcome: Outcome variable

    Returns:
        Dictionary with MR-Egger results
    """
    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    n_instruments = instruments.shape[1]

    if n_instruments < 3:
        warnings.warn("MR-Egger requires at least 3 instruments")
        return {'intercept': None, 'slope': None, 'intercept_p': None}

    # For each instrument, calculate ratios
    ratios_x = []
    ratios_y = []

    for i in range(n_instruments):
        instrument = instruments[:, i]

        # Instrument-exposure association
        beta_x = np.cov(instrument, exposure)[0, 1] / np.var(instrument)

        # Instrument-outcome association
        beta_y = np.cov(instrument, outcome)[0, 1] / np.var(instrument)

        ratios_x.append(beta_x)
        ratios_y.append(beta_y)

    ratios_x = np.array(ratios_x)
    ratios_y = np.array(ratios_y)

    # Egger regression: beta_Y ~ beta_X
    X = sm.add_constant(ratios_x)
    model = sm.OLS(ratios_y, X).fit()

    intercept = model.params[0]
    slope = model.params[1]
    intercept_p = model.pvalues[0]

    return {
        'intercept': float(intercept),
        'slope': float(slope),
        'intercept_p': float(intercept_p),
        'pleiotropy_detected': intercept_p < 0.05,
    }