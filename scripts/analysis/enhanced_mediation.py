#!/usr/bin/env python3
"""
Enhanced Mediation Analysis with Backward Elimination
Combines baseline-deviation framework with systematic mediator selection
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class MediationResult:
    """Results from mediation analysis"""
    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float
    p_value_total: float
    p_value_direct: float
    p_value_indirect: float
    active_mediators: List[str]
    mediator_contributions: pd.DataFrame


def backward_elimination_mediation(
    X: np.ndarray,
    M: pd.DataFrame,
    Y: np.ndarray,
    mediator_names: Optional[List[str]] = None,
    threshold: float = 0.05,
    min_mediators: int = 1
) -> MediationResult:
    """
    Systematic removal of non-significant mediators

    Starts with all mediators and removes least significant ones
    until all remaining mediators are significant.

    Parameters
    ----------
    X : np.ndarray
        Exposure variable (n_samples,)
    M : pd.DataFrame
        Multiple mediators (n_samples × n_mediators)
    Y : np.ndarray
        Outcome variable (n_samples,)
    mediator_names : List[str], optional
        Names of mediators
    threshold : float
        P-value threshold for retention
    min_mediators : int
        Minimum number of mediators to retain

    Returns
    -------
    MediationResult
        Complete mediation analysis results
    """
    if mediator_names is None:
        mediator_names = [f'M{i}' for i in range(M.shape[1])]

    M_array = M.values if isinstance(M, pd.DataFrame) else M

    logger.info(f"Starting backward elimination with {len(mediator_names)} mediators")

    # Start with all mediators
    active_indices = list(range(M_array.shape[1]))
    active_names = mediator_names.copy()

    elimination_history = []

    while len(active_indices) > min_mediators:
        # Test each mediator's contribution
        mediator_pvals = []

        for i, idx in enumerate(active_indices):
            # Remove mediator i
            test_indices = [j for j in active_indices if j != idx]

            if len(test_indices) == 0:
                mediator_pvals.append(0.0)
                continue

            M_reduced = M_array[:, test_indices]

            # Test if removing mediator significantly reduces fit
            p_val = test_mediator_significance(X, M_reduced, Y, M_array[:, idx])
            mediator_pvals.append(p_val)

        # Find least significant mediator
        max_pval = max(mediator_pvals)
        max_idx = mediator_pvals.index(max_pval)

        # Remove if not significant
        if max_pval > threshold:
            removed_idx = active_indices[max_idx]
            removed_name = active_names[max_idx]

            elimination_history.append({
                'step': len(elimination_history) + 1,
                'removed': removed_name,
                'p_value': max_pval,
                'n_remaining': len(active_indices) - 1
            })

            logger.info(f"  Removed {removed_name} (p={max_pval:.4f})")

            active_indices.pop(max_idx)
            active_names.pop(max_idx)
        else:
            # All remaining mediators are significant
            break

    logger.info(f"Final model: {len(active_names)} mediators retained")

    # Compute final mediation effects with selected mediators
    M_final = M_array[:, active_indices]

    # Total effect (X -> Y)
    total_effect, p_total = compute_total_effect(X, Y)

    # Direct effect (X -> Y | M)
    direct_effect, p_direct = compute_direct_effect(X, M_final, Y)

    # Indirect effect (X -> M -> Y)
    indirect_effect, p_indirect = compute_indirect_effect(X, M_final, Y)

    # Proportion mediated
    if abs(total_effect) > 1e-10:
        prop_mediated = indirect_effect / total_effect
    else:
        prop_mediated = np.nan

    # Individual mediator contributions
    mediator_contribs = []
    for i, (idx, name) in enumerate(zip(active_indices, active_names)):
        # Effect through this mediator
        M_single = M_array[:, idx:idx+1]
        indirect_single, p_single = compute_indirect_effect(X, M_single, Y)

        mediator_contribs.append({
            'mediator': name,
            'indirect_effect': indirect_single,
            'p_value': p_single,
            'proportion_of_total_indirect': indirect_single / indirect_effect if abs(indirect_effect) > 1e-10 else 0
        })

    mediator_df = pd.DataFrame(mediator_contribs).sort_values('indirect_effect', ascending=False)

    return MediationResult(
        total_effect=total_effect,
        direct_effect=direct_effect,
        indirect_effect=indirect_effect,
        proportion_mediated=prop_mediated,
        p_value_total=p_total,
        p_value_direct=p_direct,
        p_value_indirect=p_indirect,
        active_mediators=active_names,
        mediator_contributions=mediator_df
    )


def test_mediator_significance(
    X: np.ndarray,
    M_reduced: np.ndarray,
    Y: np.ndarray,
    mediator: np.ndarray
) -> float:
    """
    Test if a specific mediator adds significant explanatory power

    Uses hierarchical regression:
    1. Model with M_reduced
    2. Model with M_reduced + mediator
    3. Test if R² increase is significant
    """
    # Model 1: Y ~ X + M_reduced
    if M_reduced.shape[1] > 0:
        X_combined1 = np.column_stack([X.reshape(-1, 1), M_reduced])
    else:
        X_combined1 = X.reshape(-1, 1)

    # Model 2: Y ~ X + M_reduced + mediator
    X_combined2 = np.column_stack([X_combined1, mediator.reshape(-1, 1)])

    # Fit both models
    from sklearn.linear_model import LinearRegression

    model1 = LinearRegression().fit(X_combined1, Y)
    model2 = LinearRegression().fit(X_combined2, Y)

    r2_1 = model1.score(X_combined1, Y)
    r2_2 = model2.score(X_combined2, Y)

    # F-test for R² increase
    n = len(Y)
    p1 = X_combined1.shape[1]
    p2 = X_combined2.shape[1]

    if r2_2 <= r2_1 or r2_2 >= 1.0:
        return 1.0  # No improvement

    f_stat = ((r2_2 - r2_1) / (p2 - p1)) / ((1 - r2_2) / (n - p2 - 1))
    p_val = 1 - stats.f.cdf(f_stat, p2 - p1, n - p2 - 1)

    return p_val


def compute_total_effect(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    """Compute total effect of X on Y"""
    from sklearn.linear_model import LinearRegression

    X_reshaped = X.reshape(-1, 1)
    model = LinearRegression().fit(X_reshaped, Y)

    coef = model.coef_[0]

    # Compute p-value
    y_pred = model.predict(X_reshaped)
    residuals = Y - y_pred
    mse = np.sum(residuals**2) / (len(Y) - 2)
    se = np.sqrt(mse / np.sum((X - X.mean())**2))
    t_stat = coef / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(Y) - 2))

    return coef, p_val


def compute_direct_effect(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray
) -> Tuple[float, float]:
    """Compute direct effect of X on Y controlling for M"""
    from sklearn.linear_model import LinearRegression

    X_with_M = np.column_stack([X.reshape(-1, 1), M])
    model = LinearRegression().fit(X_with_M, Y)

    coef = model.coef_[0]  # Coefficient for X

    # Compute p-value (simplified)
    y_pred = model.predict(X_with_M)
    residuals = Y - y_pred
    mse = np.sum(residuals**2) / (len(Y) - X_with_M.shape[1] - 1)

    # Standard error of coefficient (simplified)
    X_with_M_centered = X_with_M - X_with_M.mean(axis=0)
    cov_matrix = np.linalg.inv(X_with_M_centered.T @ X_with_M_centered) * mse
    se = np.sqrt(cov_matrix[0, 0])

    t_stat = coef / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(Y) - X_with_M.shape[1] - 1))

    return coef, p_val


def compute_indirect_effect(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """
    Compute indirect effect using bootstrap for significance testing

    Indirect effect = (X -> M) * (M -> Y | X)
    """
    from sklearn.linear_model import LinearRegression

    # Path a: X -> M
    X_reshaped = X.reshape(-1, 1)
    model_a = LinearRegression().fit(X_reshaped, M)
    a_coefs = model_a.coef_[:, 0]  # One per mediator

    # Path b: M -> Y | X
    X_with_M = np.column_stack([X.reshape(-1, 1), M])
    model_b = LinearRegression().fit(X_with_M, Y)
    b_coefs = model_b.coef_[1:]  # Exclude X coefficient

    # Indirect effect = sum of a*b for each mediator
    indirect = np.sum(a_coefs * b_coefs)

    # Bootstrap for significance
    indirect_boot = []
    n = len(X)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        X_boot = X[indices]
        M_boot = M[indices]
        Y_boot = Y[indices]

        try:
            model_a_boot = LinearRegression().fit(X_boot.reshape(-1, 1), M_boot)
            X_M_boot = np.column_stack([X_boot.reshape(-1, 1), M_boot])
            model_b_boot = LinearRegression().fit(X_M_boot, Y_boot)

            a_boot = model_a_boot.coef_[:, 0]
            b_boot = model_b_boot.coef_[1:]

            indirect_boot.append(np.sum(a_boot * b_boot))
        except:
            continue

    # P-value: proportion of bootstrap samples with opposite sign
    indirect_boot = np.array(indirect_boot)
    if indirect > 0:
        p_val = np.mean(indirect_boot <= 0)
    else:
        p_val = np.mean(indirect_boot >= 0)
    p_val = max(p_val, 1/n_bootstrap)  # Lower bound

    return indirect, p_val


def baseline_deviation_mediation(
    exposure: np.ndarray,
    mediators: pd.DataFrame,
    outcome: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    use_backward_elimination: bool = True,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Combine baseline-deviation framework with mediation analysis

    If baseline is provided, analyzes:
    1. Deviation from baseline -> mediators -> outcome
    2. Baseline level -> mediators -> outcome

    Parameters
    ----------
    exposure : np.ndarray
        Exposure variable
    mediators : pd.DataFrame
        Potential mediators
    outcome : np.ndarray
        Outcome variable
    baseline : np.ndarray, optional
        Baseline/reference values for deviations
    use_backward_elimination : bool
        Whether to use backward elimination for mediator selection
    threshold : float
        Significance threshold

    Returns
    -------
    Dict with mediation results for baseline and deviation components
    """
    logger.info("Running baseline-deviation mediation analysis")

    results = {}

    if baseline is not None:
        # Compute deviations
        deviations = exposure - baseline

        # Mediation for deviations
        logger.info("  Analyzing deviation component")
        if use_backward_elimination:
            deviation_result = backward_elimination_mediation(
                deviations, mediators, outcome, threshold=threshold
            )
        else:
            deviation_result = backward_elimination_mediation(
                deviations, mediators, outcome,
                threshold=threshold, min_mediators=mediators.shape[1]
            )

        results['deviation'] = deviation_result

        # Mediation for baseline
        logger.info("  Analyzing baseline component")
        if use_backward_elimination:
            baseline_result = backward_elimination_mediation(
                baseline, mediators, outcome, threshold=threshold
            )
        else:
            baseline_result = backward_elimination_mediation(
                baseline, mediators, outcome,
                threshold=threshold, min_mediators=mediators.shape[1]
            )

        results['baseline'] = baseline_result

    else:
        # Standard mediation without baseline-deviation decomposition
        logger.info("  Analyzing total effect")
        if use_backward_elimination:
            total_result = backward_elimination_mediation(
                exposure, mediators, outcome, threshold=threshold
            )
        else:
            total_result = backward_elimination_mediation(
                exposure, mediators, outcome,
                threshold=threshold, min_mediators=mediators.shape[1]
            )

        results['total'] = total_result

    return results


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    n = 300

    # Simulate data: X -> M1 -> Y, X -> M2 -> Y, M3 is noise
    X = np.random.randn(n)
    M1 = 0.6 * X + np.random.randn(n) * 0.5
    M2 = 0.4 * X + np.random.randn(n) * 0.5
    M3 = np.random.randn(n)  # Noise mediator
    Y = 0.3 * X + 0.5 * M1 + 0.3 * M2 + np.random.randn(n) * 0.5

    M = pd.DataFrame({'M1': M1, 'M2': M2, 'M3': M3})

    # Run backward elimination
    result = backward_elimination_mediation(X, M, Y, threshold=0.05)

    print("\n=== Mediation Analysis Results ===")
    print(f"Total effect: {result.total_effect:.3f} (p={result.p_value_total:.4f})")
    print(f"Direct effect: {result.direct_effect:.3f} (p={result.p_value_direct:.4f})")
    print(f"Indirect effect: {result.indirect_effect:.3f} (p={result.p_value_indirect:.4f})")
    print(f"Proportion mediated: {result.proportion_mediated:.1%}")
    print(f"\nActive mediators: {result.active_mediators}")
    print(f"\nMediator contributions:")
    print(result.mediator_contributions)
