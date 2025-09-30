"""Sensitivity analysis for unmeasured confounding

Calculates E-values and bounds for potential confounding bias.
"""
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis"""
    # E-value
    e_value: float
    e_value_ci: float

    # Interpretation
    interpretation: str

    # Confounding bounds
    confounding_strength_required: Optional[float] = None


def calculate_e_value(
    estimate: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    outcome_type: str = 'rr',
) -> Dict[str, float]:
    """
    Calculate E-value for unmeasured confounding

    E-value: minimum strength of association (on risk ratio scale) that
    unmeasured confounding would need to explain away the observed effect.

    Args:
        estimate: Point estimate (RR, OR, or HR)
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        outcome_type: 'rr' (risk ratio), 'or' (odds ratio), or 'hr' (hazard ratio)

    Returns:
        Dictionary with E-values
    """
    # Convert to risk ratio scale if needed
    if outcome_type == 'or':
        # Approximate conversion from OR to RR (assumes rare outcome)
        estimate_rr = estimate
        if ci_lower:
            ci_lower_rr = ci_lower
        if ci_upper:
            ci_upper_rr = ci_upper
    elif outcome_type == 'hr':
        # Hazard ratio approximates risk ratio
        estimate_rr = estimate
        if ci_lower:
            ci_lower_rr = ci_lower
        if ci_upper:
            ci_upper_rr = ci_upper
    else:  # 'rr'
        estimate_rr = estimate
        if ci_lower:
            ci_lower_rr = ci_lower
        if ci_upper:
            ci_upper_rr = ci_upper

    # E-value formula: RR + sqrt(RR * (RR - 1))
    if estimate_rr >= 1:
        e_value_point = estimate_rr + np.sqrt(estimate_rr * (estimate_rr - 1))
    else:
        # For protective effects (RR < 1), use 1/RR
        rr_inv = 1 / estimate_rr
        e_value_point = rr_inv + np.sqrt(rr_inv * (rr_inv - 1))

    results = {
        'e_value_point': float(e_value_point),
    }

    # E-value for confidence interval
    if ci_lower and ci_upper:
        # Use the bound closer to the null
        if estimate_rr >= 1:
            # Use lower bound
            ci_bound = ci_lower_rr
            if ci_bound >= 1:
                e_value_ci = ci_bound + np.sqrt(ci_bound * (ci_bound - 1))
            else:
                e_value_ci = 1.0  # Crosses null
        else:
            # Use upper bound
            ci_bound = ci_upper_rr
            if ci_bound <= 1:
                ci_bound_inv = 1 / ci_bound
                e_value_ci = ci_bound_inv + np.sqrt(ci_bound_inv * (ci_bound_inv - 1))
            else:
                e_value_ci = 1.0  # Crosses null

        results['e_value_ci'] = float(e_value_ci)

    return results


def sensitivity_analysis(
    estimate: float,
    standard_error: float,
    alpha: float = 0.05,
    outcome_type: str = 'rr',
) -> SensitivityResult:
    """
    Comprehensive sensitivity analysis

    Args:
        estimate: Point estimate
        standard_error: Standard error
        alpha: Significance level
        outcome_type: Type of estimate ('rr', 'or', 'hr')

    Returns:
        SensitivityResult with E-value and interpretation
    """
    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = np.exp(np.log(estimate) - z_crit * standard_error)
    ci_upper = np.exp(np.log(estimate) + z_crit * standard_error)

    # Calculate E-values
    e_values = calculate_e_value(
        estimate=estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        outcome_type=outcome_type,
    )

    e_value_point = e_values['e_value_point']
    e_value_ci = e_values.get('e_value_ci', e_value_point)

    # Interpretation
    if e_value_ci < 1.5:
        interpretation = "LOW ROBUSTNESS: Small unmeasured confounding could explain effect"
    elif e_value_ci < 2.0:
        interpretation = "MODERATE ROBUSTNESS: Moderate unmeasured confounding required"
    elif e_value_ci < 3.0:
        interpretation = "GOOD ROBUSTNESS: Strong unmeasured confounding required"
    else:
        interpretation = "HIGH ROBUSTNESS: Very strong unmeasured confounding required"

    return SensitivityResult(
        e_value=e_value_point,
        e_value_ci=e_value_ci,
        interpretation=interpretation,
        confounding_strength_required=e_value_ci,
    )


def unmeasured_confounding_bounds(
    treatment_effect: float,
    confounder_treatment_rr: float,
    confounder_outcome_rr: float,
    prevalence_confounder: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate bounds on treatment effect under unmeasured confounding

    Uses formulae from VanderWeele & Ding (2017).

    Args:
        treatment_effect: Observed treatment effect (RR)
        confounder_treatment_rr: Association between confounder and treatment
        confounder_outcome_rr: Association between confounder and outcome
        prevalence_confounder: Prevalence of unmeasured confounder

    Returns:
        Dictionary with adjusted bounds
    """
    # Bias factor
    B = confounder_treatment_rr * confounder_outcome_rr

    # Adjusted effect bounds
    if B > 0:
        # Positive confounding
        lower_bound = treatment_effect / B
        upper_bound = treatment_effect
    else:
        # Negative confounding
        lower_bound = treatment_effect
        upper_bound = treatment_effect / B

    return {
        'observed_effect': float(treatment_effect),
        'bias_factor': float(B),
        'adjusted_lower_bound': float(lower_bound),
        'adjusted_upper_bound': float(upper_bound),
        'crosses_null': (lower_bound < 1 < upper_bound),
    }


def rosenbaum_sensitivity(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    gamma_values: np.ndarray = np.array([1.0, 1.5, 2.0, 3.0, 4.0]),
) -> Dict[float, float]:
    """
    Rosenbaum sensitivity analysis for matched studies

    Tests how strong hidden bias would need to be to change conclusions.

    Args:
        treated_outcomes: Outcomes in treated group
        control_outcomes: Outcomes in control group (matched)
        gamma_values: Values of hidden bias to test

    Returns:
        Dictionary mapping gamma to p-values
    """
    if len(treated_outcomes) != len(control_outcomes):
        raise ValueError("Treated and control must have same length (matched)")

    # Differences
    differences = treated_outcomes - control_outcomes

    # Sign test under different levels of hidden bias
    results = {}

    for gamma in gamma_values:
        # Under hidden bias gamma, probability of positive difference ranges from
        # p_min = 1/(1+gamma) to p_max = gamma/(1+gamma)

        # Wilcoxon signed rank test (simplified)
        positive_diffs = np.sum(differences > 0)
        n = len(differences)

        # Worst-case p-value under gamma
        # Use binomial test
        p_plus = gamma / (1 + gamma)
        p_value_worst = stats.binom_test(positive_diffs, n, p=p_plus, alternative='greater')

        results[float(gamma)] = float(p_value_worst)

    return results


def tipping_point_analysis(
    estimate: float,
    standard_error: float,
    prevalence_range: Tuple[float, float] = (0.1, 0.5),
    association_strength_range: Tuple[float, float] = (1.5, 5.0),
    n_grid: int = 20,
) -> pd.DataFrame:
    """
    Tipping point analysis for unmeasured confounding

    Explores grid of confounder prevalence and association strength
    to find when effect would be nullified.

    Args:
        estimate: Observed effect estimate
        standard_error: Standard error
        prevalence_range: Range of confounder prevalences to test
        association_strength_range: Range of association strengths
        n_grid: Grid resolution

    Returns:
        DataFrame with tipping point grid
    """
    import pandas as pd

    prevalences = np.linspace(prevalence_range[0], prevalence_range[1], n_grid)
    strengths = np.linspace(association_strength_range[0], association_strength_range[1], n_grid)

    results = []

    for prev in prevalences:
        for strength in strengths:
            # Bias factor (simplified)
            bias_factor = strength * strength * prev

            # Adjusted estimate
            adjusted_estimate = estimate / bias_factor

            # Check if crosses null
            z_stat = np.log(adjusted_estimate) / standard_error
            crosses_null = abs(z_stat) < 1.96

            results.append({
                'prevalence': prev,
                'association_strength': strength,
                'bias_factor': bias_factor,
                'adjusted_estimate': adjusted_estimate,
                'crosses_null': crosses_null,
            })

    df = pd.DataFrame(results)

    return df