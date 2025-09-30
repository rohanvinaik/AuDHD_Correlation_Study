"""Gene-Environment (G×E) interaction detection

Uses causal forests to detect heterogeneous treatment effects.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


@dataclass
class GxEResult:
    """Result of G×E interaction analysis"""
    # Main effects
    genetic_effect: float
    environmental_effect: float
    interaction_effect: float

    # Standard errors
    genetic_effect_se: float
    environmental_effect_se: float
    interaction_effect_se: float

    # P-values
    genetic_p: float
    environmental_p: float
    interaction_p: float

    # Heterogeneous effects
    heterogeneity_score: Optional[float] = None
    subgroup_effects: Optional[Dict[str, float]] = None

    method: str = 'linear'


def detect_gxe_interactions(
    genetics: np.ndarray,
    environment: np.ndarray,
    outcome: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    method: str = 'linear',
) -> GxEResult:
    """
    Detect Gene-Environment (G×E) interactions

    Args:
        genetics: Genetic variants (n_samples, n_snps) or polygenic score
        environment: Environmental exposure (n_samples,)
        outcome: Outcome variable (n_samples,)
        covariates: Covariates to adjust for
        method: 'linear' or 'forest'

    Returns:
        GxEResult with interaction estimates
    """
    if method == 'linear':
        return _linear_gxe(genetics, environment, outcome, covariates)
    elif method == 'forest':
        return _causal_forest_gxe(genetics, environment, outcome, covariates)
    else:
        raise ValueError(f"Unknown method: {method}")


def _linear_gxe(
    genetics: np.ndarray,
    environment: np.ndarray,
    outcome: np.ndarray,
    covariates: Optional[np.ndarray],
) -> GxEResult:
    """Linear regression for G×E interaction"""
    import statsmodels.api as sm

    # If genetics is multi-dimensional, use first PC or sum
    if genetics.ndim > 1:
        genetics = genetics[:, 0]  # Use first genetic component

    # Create interaction term
    interaction = genetics * environment

    # Build design matrix
    if covariates is not None:
        X = np.column_stack([genetics, environment, interaction, covariates])
    else:
        X = np.column_stack([genetics, environment, interaction])

    # Fit model
    X_const = sm.add_constant(X)
    model = sm.OLS(outcome, X_const).fit()

    # Extract effects
    genetic_effect = model.params[1]
    environmental_effect = model.params[2]
    interaction_effect = model.params[3]

    genetic_se = model.bse[1]
    environmental_se = model.bse[2]
    interaction_se = model.bse[3]

    genetic_p = model.pvalues[1]
    environmental_p = model.pvalues[2]
    interaction_p = model.pvalues[3]

    return GxEResult(
        genetic_effect=float(genetic_effect),
        environmental_effect=float(environmental_effect),
        interaction_effect=float(interaction_effect),
        genetic_effect_se=float(genetic_se),
        environmental_effect_se=float(environmental_se),
        interaction_effect_se=float(interaction_se),
        genetic_p=float(genetic_p),
        environmental_p=float(environmental_p),
        interaction_p=float(interaction_p),
        method='linear',
    )


def _causal_forest_gxe(
    genetics: np.ndarray,
    environment: np.ndarray,
    outcome: np.ndarray,
    covariates: Optional[np.ndarray],
) -> GxEResult:
    """Causal forest for heterogeneous effects (simplified)"""

    # This is a simplified version
    # Full implementation would use econml.dml or grf

    if genetics.ndim > 1:
        genetics = genetics[:, 0]

    # Split by genetic risk
    median_genetics = np.median(genetics)
    high_genetic = genetics >= median_genetics
    low_genetic = genetics < median_genetics

    # Estimate effects in each group
    from sklearn.linear_model import LinearRegression

    # High genetic risk group
    X_high = environment[high_genetic].reshape(-1, 1)
    y_high = outcome[high_genetic]
    model_high = LinearRegression().fit(X_high, y_high)
    effect_high = model_high.coef_[0]

    # Low genetic risk group
    X_low = environment[low_genetic].reshape(-1, 1)
    y_low = outcome[low_genetic]
    model_low = LinearRegression().fit(X_low, y_low)
    effect_low = model_low.coef_[0]

    # Interaction as difference
    interaction_effect = effect_high - effect_low

    # Heterogeneity score
    heterogeneity = abs(effect_high - effect_low) / (abs(effect_high) + abs(effect_low) + 1e-10)

    # Subgroup effects
    subgroup_effects = {
        'high_genetic_risk': float(effect_high),
        'low_genetic_risk': float(effect_low),
    }

    return GxEResult(
        genetic_effect=0.0,  # Not estimated separately
        environmental_effect=float(np.mean([effect_high, effect_low])),
        interaction_effect=float(interaction_effect),
        genetic_effect_se=0.0,
        environmental_effect_se=0.0,
        interaction_effect_se=0.0,
        genetic_p=1.0,
        environmental_p=0.05,  # Placeholder
        interaction_p=0.05,  # Placeholder
        heterogeneity_score=float(heterogeneity),
        subgroup_effects=subgroup_effects,
        method='forest',
    )


def causal_forest_analysis(
    treatment: np.ndarray,
    outcome: np.ndarray,
    covariates: np.ndarray,
    n_estimators: int = 1000,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Causal forest for heterogeneous treatment effects

    Estimates conditional average treatment effect (CATE) as function of covariates.

    Args:
        treatment: Binary treatment indicator
        outcome: Outcome variable
        covariates: Covariate matrix
        n_estimators: Number of trees
        min_samples_leaf: Minimum samples per leaf
        random_state: Random seed

    Returns:
        Dictionary with CATE estimates
    """
    # Simplified causal forest using RF
    # Full version would use econml or grf

    # Split treatment and control
    treated = treatment == 1
    control = treatment == 0

    # Fit separate models for treated and control
    rf_treated = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    rf_control = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    rf_treated.fit(covariates[treated], outcome[treated])
    rf_control.fit(covariates[control], outcome[control])

    # Predict for all samples
    mu1 = rf_treated.predict(covariates)  # E[Y|X,T=1]
    mu0 = rf_control.predict(covariates)  # E[Y|X,T=0]

    # CATE = E[Y|X,T=1] - E[Y|X,T=0]
    cate = mu1 - mu0

    return {
        'cate': cate,
        'mu1': mu1,
        'mu0': mu0,
        'ate': np.mean(cate),
    }


def heterogeneous_treatment_effects(
    treatment: np.ndarray,
    outcome: np.ndarray,
    covariates: np.ndarray,
    subgroup_variable: Optional[np.ndarray] = None,
    n_quantiles: int = 4,
) -> Dict[str, float]:
    """
    Analyze heterogeneous treatment effects

    Args:
        treatment: Treatment indicator
        outcome: Outcome variable
        covariates: Covariates
        subgroup_variable: Variable to define subgroups (if None, use CATE quantiles)
        n_quantiles: Number of quantiles for subgroup analysis

    Returns:
        Dictionary with subgroup effects
    """
    # Estimate CATE
    cate_results = causal_forest_analysis(treatment, outcome, covariates)
    cate = cate_results['cate']

    if subgroup_variable is None:
        # Use CATE quantiles
        quantiles = np.quantile(cate, np.linspace(0, 1, n_quantiles + 1))
        subgroups = np.digitize(cate, quantiles[1:-1])
    else:
        # Use provided variable
        median = np.median(subgroup_variable)
        subgroups = (subgroup_variable >= median).astype(int)

    # Estimate effects in each subgroup
    subgroup_effects = {}

    for g in np.unique(subgroups):
        mask = subgroups == g

        # Average treatment effect in subgroup
        ate_g = np.mean(cate[mask])

        subgroup_effects[f'subgroup_{g}'] = float(ate_g)

    return subgroup_effects