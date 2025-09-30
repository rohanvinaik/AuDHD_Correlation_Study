"""Mediation analysis for multi-step causal pathways

Decomposes total effects into direct and indirect (mediated) effects.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm


@dataclass
class MediationResult:
    """Result of mediation analysis"""
    # Effects
    total_effect: float
    direct_effect: float
    indirect_effect: float

    # Standard errors
    total_effect_se: float
    direct_effect_se: float
    indirect_effect_se: float

    # Confidence intervals
    total_effect_ci: Tuple[float, float]
    direct_effect_ci: Tuple[float, float]
    indirect_effect_ci: Tuple[float, float]

    # P-values
    total_effect_p: float
    direct_effect_p: float
    indirect_effect_p: float

    # Proportion mediated
    proportion_mediated: float

    # Method
    method: str


def mediation_analysis(
    exposure: np.ndarray,
    mediator: np.ndarray,
    outcome: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    method: str = 'baron_kenny',
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> MediationResult:
    """
    Perform mediation analysis

    Decomposes total effect into direct and indirect (mediated) effects.

    Args:
        exposure: Exposure variable (n_samples,)
        mediator: Mediator variable (n_samples,)
        outcome: Outcome variable (n_samples,)
        covariates: Covariates to adjust for (n_samples, n_covariates)
        method: 'baron_kenny' or 'sobel'
        n_bootstrap: Number of bootstrap samples for CIs
        alpha: Significance level
        random_state: Random seed

    Returns:
        MediationResult with effect decomposition
    """
    np.random.seed(random_state)

    if method == 'baron_kenny':
        result = _baron_kenny_mediation(
            exposure, mediator, outcome, covariates, n_bootstrap, alpha
        )
    elif method == 'sobel':
        result = _sobel_test_mediation(
            exposure, mediator, outcome, covariates, alpha
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return result


def _baron_kenny_mediation(
    exposure: np.ndarray,
    mediator: np.ndarray,
    outcome: np.ndarray,
    covariates: Optional[np.ndarray],
    n_bootstrap: int,
    alpha: float,
) -> MediationResult:
    """Baron & Kenny (1986) mediation analysis with bootstrap CIs"""

    # Prepare data
    if covariates is not None:
        X_total = np.column_stack([exposure, covariates])
        X_mediation = np.column_stack([exposure, mediator, covariates])
    else:
        X_total = exposure.reshape(-1, 1)
        X_mediation = np.column_stack([exposure, mediator])

    # Step 1: Total effect (X -> Y)
    X_total_const = sm.add_constant(X_total)
    model_total = sm.OLS(outcome, X_total_const).fit()
    total_effect = model_total.params[1]  # Coefficient for exposure
    total_effect_se = model_total.bse[1]
    total_effect_p = model_total.pvalues[1]

    # Step 2: X -> M
    X_exp_const = sm.add_constant(X_total)
    model_mediator = sm.OLS(mediator, X_exp_const).fit()
    a = model_mediator.params[1]  # X -> M effect

    # Step 3: Direct effect (X -> Y | M)
    X_mediation_const = sm.add_constant(X_mediation)
    model_direct = sm.OLS(outcome, X_mediation_const).fit()
    direct_effect = model_direct.params[1]  # Direct effect
    direct_effect_se = model_direct.bse[1]
    direct_effect_p = model_direct.pvalues[1]

    b = model_direct.params[2]  # M -> Y effect

    # Indirect effect
    indirect_effect = a * b

    # Bootstrap confidence intervals
    indirect_effects_bootstrap = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(len(exposure), size=len(exposure), replace=True)

        exp_boot = exposure[indices]
        med_boot = mediator[indices]
        out_boot = outcome[indices]

        if covariates is not None:
            cov_boot = covariates[indices]
            X_exp_boot = np.column_stack([exp_boot, cov_boot])
            X_med_boot = np.column_stack([exp_boot, med_boot, cov_boot])
        else:
            X_exp_boot = exp_boot.reshape(-1, 1)
            X_med_boot = np.column_stack([exp_boot, med_boot])

        # Refit models
        X_exp_boot_const = sm.add_constant(X_exp_boot)
        model_med_boot = sm.OLS(med_boot, X_exp_boot_const).fit()
        a_boot = model_med_boot.params[1]

        X_med_boot_const = sm.add_constant(X_med_boot)
        model_out_boot = sm.OLS(out_boot, X_med_boot_const).fit()
        b_boot = model_out_boot.params[2]

        indirect_effects_bootstrap.append(a_boot * b_boot)

    # Bootstrap CI for indirect effect
    indirect_effects_bootstrap = np.array(indirect_effects_bootstrap)
    indirect_effect_ci = (
        np.percentile(indirect_effects_bootstrap, alpha / 2 * 100),
        np.percentile(indirect_effects_bootstrap, (1 - alpha / 2) * 100)
    )
    indirect_effect_se = np.std(indirect_effects_bootstrap)

    # P-value for indirect effect (proportion of bootstrap samples with opposite sign)
    if indirect_effect > 0:
        indirect_effect_p = np.mean(indirect_effects_bootstrap <= 0) * 2
    else:
        indirect_effect_p = np.mean(indirect_effects_bootstrap >= 0) * 2

    # CI for total effect
    z_crit = stats.norm.ppf(1 - alpha / 2)
    total_effect_ci = (
        total_effect - z_crit * total_effect_se,
        total_effect + z_crit * total_effect_se
    )

    # CI for direct effect
    direct_effect_ci = (
        direct_effect - z_crit * direct_effect_se,
        direct_effect + z_crit * direct_effect_se
    )

    # Proportion mediated
    if total_effect != 0:
        proportion_mediated = indirect_effect / total_effect
    else:
        proportion_mediated = 0.0

    return MediationResult(
        total_effect=float(total_effect),
        direct_effect=float(direct_effect),
        indirect_effect=float(indirect_effect),
        total_effect_se=float(total_effect_se),
        direct_effect_se=float(direct_effect_se),
        indirect_effect_se=float(indirect_effect_se),
        total_effect_ci=tuple(map(float, total_effect_ci)),
        direct_effect_ci=tuple(map(float, direct_effect_ci)),
        indirect_effect_ci=tuple(map(float, indirect_effect_ci)),
        total_effect_p=float(total_effect_p),
        direct_effect_p=float(direct_effect_p),
        indirect_effect_p=float(indirect_effect_p),
        proportion_mediated=float(proportion_mediated),
        method='baron_kenny',
    )


def _sobel_test_mediation(
    exposure: np.ndarray,
    mediator: np.ndarray,
    outcome: np.ndarray,
    covariates: Optional[np.ndarray],
    alpha: float,
) -> MediationResult:
    """Sobel test for mediation"""

    # Same as Baron-Kenny but with Sobel standard error for indirect effect
    # (analytical instead of bootstrap)

    if covariates is not None:
        X_total = np.column_stack([exposure, covariates])
        X_mediation = np.column_stack([exposure, mediator, covariates])
    else:
        X_total = exposure.reshape(-1, 1)
        X_mediation = np.column_stack([exposure, mediator])

    # Total effect
    X_total_const = sm.add_constant(X_total)
    model_total = sm.OLS(outcome, X_total_const).fit()
    total_effect = model_total.params[1]
    total_effect_se = model_total.bse[1]
    total_effect_p = model_total.pvalues[1]

    # X -> M
    X_exp_const = sm.add_constant(X_total)
    model_mediator = sm.OLS(mediator, X_exp_const).fit()
    a = model_mediator.params[1]
    se_a = model_mediator.bse[1]

    # Direct effect
    X_mediation_const = sm.add_constant(X_mediation)
    model_direct = sm.OLS(outcome, X_mediation_const).fit()
    direct_effect = model_direct.params[1]
    direct_effect_se = model_direct.bse[1]
    direct_effect_p = model_direct.pvalues[1]

    b = model_direct.params[2]
    se_b = model_direct.bse[2]

    # Indirect effect
    indirect_effect = a * b

    # Sobel standard error
    indirect_effect_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)

    # Z-test for indirect effect
    z_stat = indirect_effect / indirect_effect_se
    indirect_effect_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence intervals
    z_crit = stats.norm.ppf(1 - alpha / 2)

    total_effect_ci = (
        total_effect - z_crit * total_effect_se,
        total_effect + z_crit * total_effect_se
    )

    direct_effect_ci = (
        direct_effect - z_crit * direct_effect_se,
        direct_effect + z_crit * direct_effect_se
    )

    indirect_effect_ci = (
        indirect_effect - z_crit * indirect_effect_se,
        indirect_effect + z_crit * indirect_effect_se
    )

    # Proportion mediated
    if total_effect != 0:
        proportion_mediated = indirect_effect / total_effect
    else:
        proportion_mediated = 0.0

    return MediationResult(
        total_effect=float(total_effect),
        direct_effect=float(direct_effect),
        indirect_effect=float(indirect_effect),
        total_effect_se=float(total_effect_se),
        direct_effect_se=float(direct_effect_se),
        indirect_effect_se=float(indirect_effect_se),
        total_effect_ci=tuple(map(float, total_effect_ci)),
        direct_effect_ci=tuple(map(float, direct_effect_ci)),
        indirect_effect_ci=tuple(map(float, indirect_effect_ci)),
        total_effect_p=float(total_effect_p),
        direct_effect_p=float(direct_effect_p),
        indirect_effect_p=float(indirect_effect_p),
        proportion_mediated=float(proportion_mediated),
        method='sobel',
    )


def multi_step_mediation(
    exposure: np.ndarray,
    mediators: List[np.ndarray],
    outcome: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, MediationResult]:
    """
    Sequential mediation analysis for multiple mediators

    Tests mediation chain: X -> M1 -> M2 -> ... -> Y

    Args:
        exposure: Exposure variable
        mediators: List of mediator variables (in causal order)
        outcome: Outcome variable
        covariates: Covariates
        n_bootstrap: Bootstrap samples
        alpha: Significance level
        random_state: Random seed

    Returns:
        Dictionary mapping mediator pairs to MediationResult
    """
    results = {}

    # Test each sequential pair
    for i, mediator in enumerate(mediators):
        mediator_name = f"mediator_{i+1}"

        # For first mediator, exposure is the treatment
        if i == 0:
            result = mediation_analysis(
                exposure=exposure,
                mediator=mediator,
                outcome=outcome,
                covariates=covariates,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                random_state=random_state,
            )
            results[f"exposure_to_{mediator_name}"] = result

        # For subsequent mediators, previous mediator is the treatment
        if i > 0:
            result = mediation_analysis(
                exposure=mediators[i-1],
                mediator=mediator,
                outcome=outcome,
                covariates=covariates,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                random_state=random_state,
            )
            results[f"mediator_{i}_to_{mediator_name}"] = result

    return results