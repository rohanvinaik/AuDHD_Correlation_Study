"""Meta-analysis across multiple cohorts

Combines results from multiple studies and cohorts with proper heterogeneity modeling.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class StudyResult:
    """Results from a single study"""
    study_id: str
    effect_size: float
    standard_error: float
    n_samples: int
    cohort_name: Optional[str] = None
    ancestry: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class MetaAnalysisResult:
    """Results of meta-analysis"""
    pooled_effect: float
    pooled_se: float
    confidence_interval: Tuple[float, float]
    p_value: float
    heterogeneity_q: float
    heterogeneity_p: float
    i_squared: float
    tau_squared: float
    method: str  # 'fixed', 'random'
    n_studies: int
    total_n: int
    study_weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


class MetaAnalyzer:
    """Performs meta-analysis across studies"""

    def __init__(
        self,
        method: str = 'random',
        heterogeneity_threshold: float = 0.1,
    ):
        """
        Initialize meta-analyzer

        Args:
            method: Meta-analysis method ('fixed' or 'random')
            heterogeneity_threshold: P-value threshold for heterogeneity
        """
        if method not in ['fixed', 'random', 'auto']:
            raise ValueError(f"Unknown method: {method}")

        self.method = method
        self.heterogeneity_threshold = heterogeneity_threshold

    def meta_analyze(
        self,
        studies: List[StudyResult],
    ) -> MetaAnalysisResult:
        """
        Perform meta-analysis

        Args:
            studies: List of study results

        Returns:
            MetaAnalysisResult
        """
        if len(studies) < 2:
            raise ValueError("Need at least 2 studies for meta-analysis")

        # Extract effect sizes and standard errors
        effects = np.array([s.effect_size for s in studies])
        ses = np.array([s.standard_error for s in studies])
        ns = np.array([s.n_samples for s in studies])

        # Test for heterogeneity
        q_stat, q_p, i2, tau2 = self._test_heterogeneity(effects, ses)

        # Determine method
        if self.method == 'auto':
            method = 'random' if q_p < self.heterogeneity_threshold else 'fixed'
        else:
            method = self.method

        # Perform meta-analysis
        if method == 'fixed':
            pooled, pooled_se, weights = self._fixed_effects_meta_analysis(
                effects, ses
            )
        else:  # random
            pooled, pooled_se, weights = self._random_effects_meta_analysis(
                effects, ses, tau2
            )

        # Calculate confidence interval
        ci_lower = pooled - 1.96 * pooled_se
        ci_upper = pooled + 1.96 * pooled_se

        # Calculate p-value
        z_score = pooled / pooled_se
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        # Create weight dictionary
        study_weights = {
            studies[i].study_id: float(weights[i])
            for i in range(len(studies))
        }

        return MetaAnalysisResult(
            pooled_effect=pooled,
            pooled_se=pooled_se,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            heterogeneity_q=q_stat,
            heterogeneity_p=q_p,
            i_squared=i2,
            tau_squared=tau2,
            method=method,
            n_studies=len(studies),
            total_n=int(ns.sum()),
            study_weights=study_weights,
        )

    def _fixed_effects_meta_analysis(
        self,
        effects: np.ndarray,
        ses: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Fixed-effects meta-analysis (inverse variance weighting)

        Args:
            effects: Effect sizes
            ses: Standard errors

        Returns:
            Tuple of (pooled_effect, pooled_se, weights)
        """
        # Weights = 1 / variance
        weights = 1 / (ses ** 2)

        # Pooled effect
        pooled = np.sum(weights * effects) / np.sum(weights)

        # Pooled SE
        pooled_se = np.sqrt(1 / np.sum(weights))

        # Normalize weights to sum to 1
        normalized_weights = weights / weights.sum()

        return pooled, pooled_se, normalized_weights

    def _random_effects_meta_analysis(
        self,
        effects: np.ndarray,
        ses: np.ndarray,
        tau2: float,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Random-effects meta-analysis (DerSimonian-Laird)

        Args:
            effects: Effect sizes
            ses: Standard errors
            tau2: Between-study variance

        Returns:
            Tuple of (pooled_effect, pooled_se, weights)
        """
        # Weights = 1 / (variance + tau^2)
        weights = 1 / (ses ** 2 + tau2)

        # Pooled effect
        pooled = np.sum(weights * effects) / np.sum(weights)

        # Pooled SE
        pooled_se = np.sqrt(1 / np.sum(weights))

        # Normalize weights
        normalized_weights = weights / weights.sum()

        return pooled, pooled_se, normalized_weights

    def _test_heterogeneity(
        self,
        effects: np.ndarray,
        ses: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Test for heterogeneity (Cochran's Q and I²)

        Args:
            effects: Effect sizes
            ses: Standard errors

        Returns:
            Tuple of (Q, p_value, I2, tau2)
        """
        k = len(effects)

        # Fixed-effects pooled estimate
        weights = 1 / (ses ** 2)
        pooled = np.sum(weights * effects) / np.sum(weights)

        # Cochran's Q
        Q = np.sum(weights * (effects - pooled) ** 2)

        # P-value
        df = k - 1
        p_value = 1 - chi2.cdf(Q, df) if df > 0 else 1.0

        # I² statistic
        I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

        # Tau² (DerSimonian-Laird estimator)
        if Q > df:
            C = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
            tau2 = (Q - df) / C
        else:
            tau2 = 0.0

        return Q, p_value, I2, tau2

    def subgroup_analysis(
        self,
        studies: List[StudyResult],
        subgroup_var: str,
    ) -> Dict[str, MetaAnalysisResult]:
        """
        Perform subgroup meta-analysis

        Args:
            studies: List of study results
            subgroup_var: Variable to subgroup by ('ancestry', 'cohort_name', etc.)

        Returns:
            Dictionary mapping subgroup to MetaAnalysisResult
        """
        # Group studies by subgroup variable
        subgroups = {}
        for study in studies:
            value = getattr(study, subgroup_var, None)
            if value is not None:
                if value not in subgroups:
                    subgroups[value] = []
                subgroups[value].append(study)

        # Perform meta-analysis for each subgroup
        results = {}
        for subgroup_name, subgroup_studies in subgroups.items():
            if len(subgroup_studies) >= 2:
                results[subgroup_name] = self.meta_analyze(subgroup_studies)

        return results

    def meta_regression(
        self,
        studies: List[StudyResult],
        moderator: np.ndarray,
        moderator_name: str = 'moderator',
    ) -> Dict[str, float]:
        """
        Meta-regression to test moderating effects

        Args:
            studies: List of study results
            moderator: Moderator variable values
            moderator_name: Name of moderator

        Returns:
            Dictionary of regression results
        """
        effects = np.array([s.effect_size for s in studies])
        ses = np.array([s.standard_error for s in studies])
        weights = 1 / (ses ** 2)

        # Weighted least squares
        X = np.column_stack([np.ones(len(moderator)), moderator])
        W = np.diag(weights)

        # Beta = (X'WX)^-1 X'Wy
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ effects

        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            return {
                'intercept': np.nan,
                'slope': np.nan,
                'p_value': np.nan,
            }

        # Standard errors
        residuals = effects - X @ beta
        var_residual = (residuals ** 2 @ weights) / (len(effects) - 2)
        var_beta = var_residual * np.linalg.inv(XtWX).diagonal()
        se_beta = np.sqrt(var_beta)

        # P-values
        z_scores = beta / se_beta
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

        return {
            'intercept': float(beta[0]),
            'intercept_se': float(se_beta[0]),
            'intercept_p': float(p_values[0]),
            'slope': float(beta[1]),
            'slope_se': float(se_beta[1]),
            'slope_p': float(p_values[1]),
            'moderator': moderator_name,
        }

    def plot_forest_plot(
        self,
        studies: List[StudyResult],
        result: MetaAnalysisResult,
        output_path: Optional[str] = None,
    ):
        """
        Create forest plot

        Args:
            studies: List of study results
            result: Meta-analysis result
            output_path: Output path for plot
        """
        fig, ax = plt.subplots(figsize=(10, max(6, len(studies) * 0.5)))

        # Plot individual studies
        y_pos = np.arange(len(studies))

        for i, study in enumerate(studies):
            # Confidence interval
            ci_lower = study.effect_size - 1.96 * study.standard_error
            ci_upper = study.effect_size + 1.96 * study.standard_error

            # Plot CI
            ax.plot(
                [ci_lower, ci_upper],
                [i, i],
                'k-',
                linewidth=2,
            )

            # Plot point estimate (size proportional to weight)
            weight = result.study_weights[study.study_id]
            ax.scatter(
                study.effect_size,
                i,
                s=weight * 200,
                c='blue',
                alpha=0.6,
                zorder=10,
            )

            # Study label
            label = f"{study.study_id} (n={study.n_samples})"
            ax.text(-0.1, i, label, ha='right', va='center', fontsize=9)

        # Add pooled estimate
        ax.plot(
            [result.confidence_interval[0], result.confidence_interval[1]],
            [len(studies) + 0.5, len(studies) + 0.5],
            'r-',
            linewidth=3,
        )
        ax.scatter(
            result.pooled_effect,
            len(studies) + 0.5,
            s=300,
            c='red',
            marker='D',
            zorder=10,
        )
        ax.text(
            -0.1,
            len(studies) + 0.5,
            f'Pooled ({result.method.capitalize()})',
            ha='right',
            va='center',
            fontsize=10,
            fontweight='bold',
        )

        # Add vertical line at null
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)

        # Labels and formatting
        ax.set_yticks([])
        ax.set_xlabel('Effect Size')
        ax.set_title(
            f'Forest Plot\n'
            f'Pooled: {result.pooled_effect:.3f} '
            f'[{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]\n'
            f'I² = {result.i_squared:.1f}%, p = {result.heterogeneity_p:.3f}'
        )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_funnel_plot(
        self,
        studies: List[StudyResult],
        result: MetaAnalysisResult,
        output_path: Optional[str] = None,
    ):
        """
        Create funnel plot to assess publication bias

        Args:
            studies: List of study results
            result: Meta-analysis result
            output_path: Output path for plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Extract data
        effects = np.array([s.effect_size for s in studies])
        ses = np.array([s.standard_error for s in studies])

        # Scatter plot
        ax.scatter(effects, ses, alpha=0.6, s=100)

        # Add pooled estimate line
        ax.axvline(
            result.pooled_effect,
            color='r',
            linestyle='--',
            label='Pooled effect',
        )

        # Add funnel (95% CI bounds)
        y_range = np.linspace(0, ses.max() * 1.1, 100)
        ci_lower = result.pooled_effect - 1.96 * y_range
        ci_upper = result.pooled_effect + 1.96 * y_range

        ax.plot(ci_lower, y_range, 'k--', alpha=0.3)
        ax.plot(ci_upper, y_range, 'k--', alpha=0.3)

        ax.set_xlabel('Effect Size')
        ax.set_ylabel('Standard Error')
        ax.set_title('Funnel Plot (Publication Bias Assessment)')
        ax.invert_yaxis()
        ax.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def test_publication_bias(
    studies: List[StudyResult],
) -> Dict[str, float]:
    """
    Test for publication bias using Egger's test

    Args:
        studies: List of study results

    Returns:
        Dictionary with test statistics
    """
    from scipy.stats import linregress

    effects = np.array([s.effect_size for s in studies])
    ses = np.array([s.standard_error for s in studies])

    # Egger's test: regress standardized effect on precision
    standardized_effects = effects / ses
    precision = 1 / ses

    slope, intercept, r_value, p_value, std_err = linregress(
        precision, standardized_effects
    )

    return {
        'egger_intercept': intercept,
        'egger_se': std_err,
        'egger_p': p_value,
        'biased': p_value < 0.1,
    }


def combine_cohort_results(
    cohort_results: List[Dict],
    metric: str = 'effect_size',
) -> MetaAnalysisResult:
    """
    Combine results across cohorts

    Args:
        cohort_results: List of cohort result dictionaries
        metric: Metric to combine

    Returns:
        MetaAnalysisResult
    """
    # Convert to StudyResult objects
    studies = []
    for i, result in enumerate(cohort_results):
        studies.append(StudyResult(
            study_id=result.get('cohort_id', f'Cohort_{i}'),
            effect_size=result[metric],
            standard_error=result.get('standard_error', result[metric] * 0.1),
            n_samples=result.get('n_samples', 100),
            cohort_name=result.get('cohort_name'),
        ))

    # Perform meta-analysis
    analyzer = MetaAnalyzer(method='random')
    return analyzer.meta_analyze(studies)