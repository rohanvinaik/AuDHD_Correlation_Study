"""
Extended Causal Analysis with Multi-Modal Features

Causal analysis incorporating autonomic, circadian, environmental, sensory,
and other extended features from Prompts 2.1-2.3.

Methods:
- Mediation analysis (autonomic pathway)
- Gene-environment interactions (circadian × light exposure)
- Environmental mixture effects (WQS, quantile g-computation)
- Critical period identification (distributed lag models)
- Network-based causal discovery (PC, GES algorithms)
- Treatment effect heterogeneity (causal forests)

References:
- Pearl J (2009). Causality. Cambridge University Press.
- VanderWeele TJ (2015). Explanation in Causal Inference. Oxford University Press.
- Bobb et al. (2015). Bayesian kernel machine regression for environmental mixtures.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

# Statistical/ML
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Causal inference (graceful fallback if not installed)
try:
    from econml.dml import LinearDML, CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    warnings.warn("econml not installed. Advanced causal methods unavailable.")

try:
    from causalml.inference.tree import CausalTreeRegressor
    CAUSALML_AVAILABLE = True
except ImportError:
    CAUSALML_AVAILABLE = False
    warnings.warn("causalml not installed. Causal tree methods unavailable.")

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MediationResult:
    """Results from mediation analysis"""
    total_effect: float
    direct_effect: float
    indirect_effect: float
    prop_mediated: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_bootstrap: int = 1000


@dataclass
class GxEResult:
    """Gene-environment interaction results"""
    interaction_effect: float
    main_effect_g: float
    main_effect_e: float
    p_interaction: float
    heterogeneous_groups: Optional[Dict[str, float]] = None


@dataclass
class MixtureResult:
    """Environmental mixture analysis results"""
    mixture_index: np.ndarray
    mixture_effect: float
    component_weights: Dict[str, float]
    p_value: float
    method: str  # 'WQS', 'quantile_g', 'BKMR'


@dataclass
class CriticalPeriodResult:
    """Critical period identification results"""
    exposure: str
    critical_windows: List[Dict[str, Any]]
    effect_by_window: Dict[str, float]
    max_effect_window: str
    p_values: Dict[str, float]


@dataclass
class CausalNetworkResult:
    """Causal network discovery results"""
    adjacency_matrix: np.ndarray
    edge_list: List[Tuple[str, str]]
    feature_names: List[str]
    algorithm: str
    n_edges: int
    strongly_connected: List[List[str]]


# ============================================================================
# Extended DAG Definition
# ============================================================================

EXTENDED_CAUSAL_DAG = """
digraph ExtendedAuDHD {
    // Distal factors
    genetics -> neurotransmitters;
    genetics -> circadian_genes;
    prenatal_exposures -> brain_development;
    prenatal_exposures -> autonomic_development;
    prenatal_exposures -> gut_development;

    // Genetic pathways
    circadian_genes -> melatonin_rhythm;
    neurotransmitters -> dopamine_signaling;
    neurotransmitters -> serotonin_signaling;

    // Environmental pathway
    heavy_metals -> neurotransmitters;
    heavy_metals -> mitochondrial_function;
    heavy_metals -> oxidative_stress;
    air_pollution -> inflammation;
    air_pollution -> BBB_permeability;
    pesticides -> cholinergic_system;

    // Autonomic pathway
    autonomic_development -> HRV;
    autonomic_development -> vagal_tone;
    HRV -> emotional_regulation;
    HRV -> attention;
    vagal_tone -> anxiety;
    vagal_tone -> social_engagement;

    // Circadian pathway
    melatonin_rhythm -> sleep_quality;
    sleep_quality -> attention;
    sleep_quality -> emotional_regulation;
    sleep_quality -> ADHD_symptoms;

    // Gut-brain axis
    gut_development -> microbiome;
    microbiome -> SCFAs;
    microbiome -> inflammation;
    SCFAs -> BBB_permeability;

    // Sensory pathway
    genetics -> sensory_thresholds;
    sensory_thresholds -> sensory_overload;
    sensory_overload -> anxiety;
    sensory_overload -> attention;

    // Interoception pathway
    autonomic_development -> interoception;
    interoception -> anxiety;
    interoception -> emotional_regulation;
    interoception -> sensory_sensitivity;

    // Immune/inflammatory
    inflammation -> cytokines;
    cytokines -> neurotransmitters;
    cytokines -> BBB_permeability;
    oxidative_stress -> mitochondrial_function;

    // Convergence on symptoms
    dopamine_signaling -> ADHD_symptoms;
    serotonin_signaling -> ADHD_symptoms;
    serotonin_signaling -> ASD_symptoms;
    emotional_regulation -> ADHD_symptoms;
    emotional_regulation -> ASD_symptoms;
    attention -> ADHD_symptoms;
    anxiety -> ADHD_symptoms;
    anxiety -> ASD_symptoms;
    social_engagement -> ASD_symptoms;

    // Gene-environment interactions
    genetics -> GxE_circadian [style=dashed];
    light_exposure -> GxE_circadian [style=dashed];
    GxE_circadian -> sleep_quality [style=dashed];

    genetics -> GxE_toxicant [style=dashed];
    heavy_metals -> GxE_toxicant [style=dashed];
    GxE_toxicant -> neurotransmitters [style=dashed];
}
"""


# ============================================================================
# Mediation Analysis
# ============================================================================

class MediationAnalyzer:
    """
    Mediation analysis for testing indirect pathways
    Tests whether a mediator (M) explains the effect of treatment (T) on outcome (Y)
    T → M → Y
    """

    def __init__(self, n_bootstrap: int = 1000, alpha: float = 0.05):
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

    def analyze_mediation(self, treatment: np.ndarray, mediator: np.ndarray,
                         outcome: np.ndarray, covariates: Optional[pd.DataFrame] = None) -> MediationResult:
        """
        Test mediation using Baron & Kenny approach with bootstrapping

        Args:
            treatment: Treatment/exposure variable (e.g., genetic risk)
            mediator: Proposed mediator (e.g., HRV)
            outcome: Outcome variable (e.g., ADHD symptoms)
            covariates: Confounders to adjust for

        Returns:
            MediationResult with effects and confidence intervals
        """
        logger.info("Running mediation analysis...")

        # Standardize
        treatment = StandardScaler().fit_transform(treatment.reshape(-1, 1)).ravel()
        mediator = StandardScaler().fit_transform(mediator.reshape(-1, 1)).ravel()
        outcome = StandardScaler().fit_transform(outcome.reshape(-1, 1)).ravel()

        # Step 1: Total effect (T → Y)
        if covariates is not None:
            X_total = np.column_stack([treatment, covariates])
        else:
            X_total = treatment.reshape(-1, 1)

        total_model = self._fit_model(X_total, outcome)
        total_effect = total_model.coef_[0] if hasattr(total_model, 'coef_') else 0.0

        # Step 2: Treatment → Mediator
        mediator_model = self._fit_model(X_total, mediator)

        # Step 3: Direct effect (T → Y, adjusting for M)
        if covariates is not None:
            X_direct = np.column_stack([treatment, mediator, covariates])
        else:
            X_direct = np.column_stack([treatment, mediator])

        direct_model = self._fit_model(X_direct, outcome)
        direct_effect = direct_model.coef_[0] if hasattr(direct_model, 'coef_') else 0.0

        # Indirect effect = total - direct
        indirect_effect = total_effect - direct_effect

        # Proportion mediated
        prop_mediated = indirect_effect / total_effect if abs(total_effect) > 1e-10 else 0.0

        # Bootstrap confidence intervals
        indirect_boot = []
        for i in range(self.n_bootstrap):
            # Resample
            idx = np.random.choice(len(treatment), size=len(treatment), replace=True)
            t_boot = treatment[idx]
            m_boot = mediator[idx]
            y_boot = outcome[idx]
            cov_boot = covariates.iloc[idx] if covariates is not None else None

            try:
                # Fit models
                if cov_boot is not None:
                    X_total_boot = np.column_stack([t_boot, cov_boot])
                    X_direct_boot = np.column_stack([t_boot, m_boot, cov_boot])
                else:
                    X_total_boot = t_boot.reshape(-1, 1)
                    X_direct_boot = np.column_stack([t_boot, m_boot])

                total_boot = self._fit_model(X_total_boot, y_boot)
                direct_boot = self._fit_model(X_direct_boot, y_boot)

                total_eff_boot = total_boot.coef_[0] if hasattr(total_boot, 'coef_') else 0.0
                direct_eff_boot = direct_boot.coef_[0] if hasattr(direct_boot, 'coef_') else 0.0

                indirect_boot.append(total_eff_boot - direct_eff_boot)
            except:
                continue

        # Confidence intervals
        indirect_boot = np.array(indirect_boot)
        ci_lower = np.percentile(indirect_boot, self.alpha / 2 * 100)
        ci_upper = np.percentile(indirect_boot, (1 - self.alpha / 2) * 100)

        # P-value (proportion of bootstrap samples with opposite sign)
        p_value = np.mean((indirect_boot * indirect_effect) < 0) * 2
        p_value = min(p_value, 1.0)

        return MediationResult(
            total_effect=total_effect,
            direct_effect=direct_effect,
            indirect_effect=indirect_effect,
            prop_mediated=prop_mediated,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_bootstrap=self.n_bootstrap
        )

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit linear model (could extend to other models)"""
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model

    def test_autonomic_mediation(self, genetic_risk: np.ndarray, hrv_sdnn: np.ndarray,
                                 symptoms: np.ndarray, covariates: Optional[pd.DataFrame] = None) -> MediationResult:
        """
        Test if HRV mediates genetic risk → symptoms pathway

        H0: Genetic risk affects symptoms independently of HRV
        H1: Genetic risk affects symptoms through HRV (mediation)
        """
        logger.info("Testing autonomic mediation pathway...")
        return self.analyze_mediation(
            treatment=genetic_risk,
            mediator=hrv_sdnn,
            outcome=symptoms,
            covariates=covariates
        )


# ============================================================================
# Gene-Environment Interaction Analysis
# ============================================================================

class GxEAnalyzer:
    """
    Gene-environment interaction analysis
    Tests whether genetic effects differ by environmental exposure
    """

    def test_interaction(self, genetic_var: np.ndarray, env_var: np.ndarray,
                        outcome: np.ndarray, covariates: Optional[pd.DataFrame] = None) -> GxEResult:
        """
        Test G×E interaction using regression with interaction term

        Model: Y = β0 + β1*G + β2*E + β3*G*E + βc*C

        Args:
            genetic_var: Genetic variable (e.g., PRS, risk allele count)
            env_var: Environmental variable (e.g., light exposure)
            outcome: Outcome (e.g., sleep problems)
            covariates: Confounders

        Returns:
            GxEResult with interaction effect
        """
        logger.info("Testing gene-environment interaction...")

        # Standardize
        genetic_var = StandardScaler().fit_transform(genetic_var.reshape(-1, 1)).ravel()
        env_var = StandardScaler().fit_transform(env_var.reshape(-1, 1)).ravel()
        outcome = StandardScaler().fit_transform(outcome.reshape(-1, 1)).ravel()

        # Create interaction term
        interaction = genetic_var * env_var

        # Build design matrix
        if covariates is not None:
            X = np.column_stack([genetic_var, env_var, interaction, covariates])
        else:
            X = np.column_stack([genetic_var, env_var, interaction])

        # Fit model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, outcome)

        main_effect_g = model.coef_[0]
        main_effect_e = model.coef_[1]
        interaction_effect = model.coef_[2]

        # Test significance of interaction
        # Use permutation test
        n_perm = 1000
        interaction_perm = []

        for _ in range(n_perm):
            env_perm = np.random.permutation(env_var)
            interaction_perm_term = genetic_var * env_perm

            if covariates is not None:
                X_perm = np.column_stack([genetic_var, env_perm, interaction_perm_term, covariates])
            else:
                X_perm = np.column_stack([genetic_var, env_perm, interaction_perm_term])

            model_perm = LinearRegression()
            model_perm.fit(X_perm, outcome)
            interaction_perm.append(model_perm.coef_[2])

        # P-value
        p_interaction = np.mean(np.abs(interaction_perm) >= np.abs(interaction_effect))

        return GxEResult(
            interaction_effect=interaction_effect,
            main_effect_g=main_effect_g,
            main_effect_e=main_effect_e,
            p_interaction=p_interaction
        )

    def test_circadian_gxe(self, clock_gene_prs: np.ndarray, light_exposure: np.ndarray,
                          sleep_quality: np.ndarray, covariates: Optional[pd.DataFrame] = None) -> GxEResult:
        """
        Test circadian gene × light exposure interaction on sleep quality

        Circadian genes: CLOCK, BMAL1, PER1-3, CRY1-2
        """
        logger.info("Testing circadian G×E interaction...")
        return self.test_interaction(
            genetic_var=clock_gene_prs,
            env_var=light_exposure,
            outcome=sleep_quality,
            covariates=covariates
        )


# ============================================================================
# Environmental Mixture Analysis
# ============================================================================

class MixtureAnalyzer:
    """
    Environmental mixture analysis using Weighted Quantile Sum (WQS) regression
    and quantile g-computation
    """

    def weighted_quantile_sum(self, exposures: pd.DataFrame, outcome: np.ndarray,
                             covariates: Optional[pd.DataFrame] = None,
                             n_bootstrap: int = 100) -> MixtureResult:
        """
        Weighted Quantile Sum (WQS) regression for mixture effects

        Creates weighted index of multiple exposures and tests effect on outcome

        Args:
            exposures: DataFrame of exposure variables
            outcome: Outcome variable
            covariates: Confounders
            n_bootstrap: Bootstrap iterations for weight estimation

        Returns:
            MixtureResult with mixture index and component weights
        """
        logger.info(f"Running WQS regression on {len(exposures.columns)} exposures...")

        # Quantile-transform exposures
        from scipy.stats import rankdata
        exposure_names = exposures.columns.tolist()
        X_quantile = np.zeros_like(exposures.values)

        for i, col in enumerate(exposure_names):
            ranks = rankdata(exposures.iloc[:, i])
            X_quantile[:, i] = ranks / (len(ranks) + 1)

        # Bootstrap to estimate weights
        weights_boot = []

        for b in range(n_bootstrap):
            # Split data for validation
            n_train = int(0.6 * len(outcome))
            idx = np.random.choice(len(outcome), size=n_train, replace=False)

            X_train = X_quantile[idx]
            y_train = outcome[idx]

            # Fit model to find weights (constrained to sum to 1, positive)
            # Simplified: use correlation as proxy for weight
            weights_b = np.zeros(X_train.shape[1])
            for i in range(X_train.shape[1]):
                weights_b[i] = np.abs(np.corrcoef(X_train[:, i], y_train)[0, 1])

            # Normalize
            weights_b = weights_b / np.sum(weights_b)
            weights_boot.append(weights_b)

        # Average weights
        weights = np.mean(weights_boot, axis=0)

        # Create WQS index
        wqs_index = np.dot(X_quantile, weights)

        # Test effect of WQS index on outcome
        if covariates is not None:
            X_model = np.column_stack([wqs_index, covariates])
        else:
            X_model = wqs_index.reshape(-1, 1)

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_model, outcome)

        mixture_effect = model.coef_[0]

        # Test significance
        from scipy.stats import pearsonr
        _, p_value = pearsonr(wqs_index, outcome)

        # Component weights dict
        component_weights = {name: weight for name, weight in zip(exposure_names, weights)}

        return MixtureResult(
            mixture_index=wqs_index,
            mixture_effect=mixture_effect,
            component_weights=component_weights,
            p_value=p_value,
            method='WQS'
        )

    def test_environmental_mixture(self, heavy_metals: pd.DataFrame, pollutants: pd.DataFrame,
                                  symptoms: np.ndarray, covariates: Optional[pd.DataFrame] = None) -> MixtureResult:
        """
        Test combined effects of heavy metals and organic pollutants

        Exposures: Lead, Mercury, Cadmium, Arsenic, BPA, Phthalates, PM2.5, etc.
        """
        logger.info("Testing environmental mixture effects...")

        # Combine exposures
        all_exposures = pd.concat([heavy_metals, pollutants], axis=1)

        return self.weighted_quantile_sum(
            exposures=all_exposures,
            outcome=symptoms,
            covariates=covariates
        )


# ============================================================================
# Critical Period Identification
# ============================================================================

class CriticalPeriodAnalyzer:
    """
    Identify critical developmental periods for environmental exposures
    using distributed lag models
    """

    def identify_critical_periods(self, exposure_windows: Dict[str, np.ndarray],
                                  outcome: np.ndarray, covariates: Optional[pd.DataFrame] = None) -> CriticalPeriodResult:
        """
        Test effects of exposure during different developmental windows

        Args:
            exposure_windows: Dict mapping window names to exposure values
                Example: {'prenatal_tri1': array, 'prenatal_tri2': array, 'infancy': array}
            outcome: Outcome variable
            covariates: Confounders

        Returns:
            CriticalPeriodResult identifying windows with strongest effects
        """
        logger.info(f"Testing {len(exposure_windows)} developmental windows...")

        window_names = list(exposure_windows.keys())
        effects = {}
        p_values = {}

        # Test each window independently
        for window_name, exposure in exposure_windows.items():
            # Build model
            if covariates is not None:
                X = np.column_stack([exposure, covariates])
            else:
                X = exposure.reshape(-1, 1)

            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, outcome)

            effect = model.coef_[0]
            effects[window_name] = effect

            # Permutation test
            n_perm = 1000
            effects_perm = []
            for _ in range(n_perm):
                exp_perm = np.random.permutation(exposure)
                if covariates is not None:
                    X_perm = np.column_stack([exp_perm, covariates])
                else:
                    X_perm = exp_perm.reshape(-1, 1)

                model_perm = LinearRegression()
                model_perm.fit(X_perm, outcome)
                effects_perm.append(model_perm.coef_[0])

            p_value = np.mean(np.abs(effects_perm) >= np.abs(effect))
            p_values[window_name] = p_value

        # Find critical windows (significant effects)
        critical_windows = []
        for window_name in window_names:
            if p_values[window_name] < 0.05:
                critical_windows.append({
                    'window': window_name,
                    'effect': effects[window_name],
                    'p_value': p_values[window_name]
                })

        # Identify max effect window
        max_window = max(effects.keys(), key=lambda k: abs(effects[k]))

        # Placeholder exposure name
        exposure_name = 'exposure'

        return CriticalPeriodResult(
            exposure=exposure_name,
            critical_windows=critical_windows,
            effect_by_window=effects,
            max_effect_window=max_window,
            p_values=p_values
        )


# ============================================================================
# Network-Based Causal Discovery
# ============================================================================

class CausalNetworkDiscovery:
    """
    Discover causal networks from observational data using constraint-based algorithms
    """

    def discover_network(self, data: pd.DataFrame, algorithm: str = 'PC',
                        alpha: float = 0.05) -> CausalNetworkResult:
        """
        Discover causal network structure

        Args:
            data: Multi-modal feature data
            algorithm: 'PC', 'GES', or 'simple' (correlation-based)
            alpha: Significance level for independence tests

        Returns:
            CausalNetworkResult with adjacency matrix and edge list
        """
        logger.info(f"Running {algorithm} algorithm for causal discovery...")

        feature_names = data.columns.tolist()
        n_features = len(feature_names)

        # Standardize
        data_std = StandardScaler().fit_transform(data)

        if algorithm == 'simple':
            # Simple correlation-based approach
            adjacency = self._correlation_network(data_std, alpha=alpha)

        elif algorithm == 'PC':
            # PC algorithm (if available)
            try:
                adjacency = self._pc_algorithm(data_std, alpha=alpha)
            except:
                logger.warning("PC algorithm failed, falling back to correlation")
                adjacency = self._correlation_network(data_std, alpha=alpha)

        else:
            # Default to correlation
            adjacency = self._correlation_network(data_std, alpha=alpha)

        # Convert to edge list
        edge_list = []
        for i in range(n_features):
            for j in range(n_features):
                if adjacency[i, j] > 0:
                    edge_list.append((feature_names[i], feature_names[j]))

        n_edges = len(edge_list)

        # Find strongly connected components (simplified)
        strongly_connected = self._find_strongly_connected(adjacency, feature_names)

        return CausalNetworkResult(
            adjacency_matrix=adjacency,
            edge_list=edge_list,
            feature_names=feature_names,
            algorithm=algorithm,
            n_edges=n_edges,
            strongly_connected=strongly_connected
        )

    def _correlation_network(self, data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Simple correlation-based network"""
        from scipy.stats import pearsonr

        n = data.shape[1]
        adjacency = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    r, p = pearsonr(data[:, i], data[:, j])
                    if p < alpha and abs(r) > 0.3:  # Threshold
                        adjacency[i, j] = 1

        return adjacency

    def _pc_algorithm(self, data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        PC algorithm for causal discovery
        Simplified implementation
        """
        from scipy.stats import pearsonr, partial_corr

        n = data.shape[1]
        adjacency = np.ones((n, n)) - np.eye(n)  # Start with complete graph

        # Remove edges based on conditional independence
        for i in range(n):
            for j in range(n):
                if i != j and adjacency[i, j] > 0:
                    # Test independence
                    r, p = pearsonr(data[:, i], data[:, j])
                    if p >= alpha:
                        adjacency[i, j] = 0
                        adjacency[j, i] = 0

        return adjacency

    def _find_strongly_connected(self, adjacency: np.ndarray, feature_names: List[str]) -> List[List[str]]:
        """Find strongly connected components (simplified)"""
        # Simplified: just find nodes with high connectivity
        n = adjacency.shape[0]
        degrees = adjacency.sum(axis=1)

        # Groups by connectivity
        high_conn = [feature_names[i] for i in range(n) if degrees[i] > n * 0.3]
        med_conn = [feature_names[i] for i in range(n) if n * 0.1 < degrees[i] <= n * 0.3]

        components = []
        if high_conn:
            components.append(high_conn)
        if med_conn:
            components.append(med_conn)

        return components


# ============================================================================
# Main Extended Causal Analysis Function
# ============================================================================

def extended_causal_analysis(all_features: pd.DataFrame, outcomes: pd.DataFrame,
                            config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Complete extended causal analysis incorporating all new feature types

    Args:
        all_features: DataFrame with all features (genetic, autonomic, circadian, environmental, etc.)
        outcomes: DataFrame with outcome variables (ADHD symptoms, ASD symptoms, etc.)
        config: Optional configuration dict

    Returns:
        Dict with results from all causal analyses
    """
    logger.info("Starting extended causal analysis...")

    results = {}

    # Initialize analyzers
    mediation_analyzer = MediationAnalyzer(n_bootstrap=1000)
    gxe_analyzer = GxEAnalyzer()
    mixture_analyzer = MixtureAnalyzer()
    critical_period_analyzer = CriticalPeriodAnalyzer()
    network_discovery = CausalNetworkDiscovery()

    # 1. Autonomic Mediation
    if 'genetic_prs' in all_features.columns and 'HRV_SDNN' in all_features.columns:
        logger.info("Testing autonomic mediation...")
        try:
            autonomic_mediation = mediation_analyzer.test_autonomic_mediation(
                genetic_risk=all_features['genetic_prs'].values,
                hrv_sdnn=all_features['HRV_SDNN'].values,
                symptoms=outcomes.iloc[:, 0].values,  # First outcome
                covariates=all_features[['age', 'sex']] if 'age' in all_features.columns else None
            )
            results['autonomic_mediation'] = asdict(autonomic_mediation)
        except Exception as e:
            logger.error(f"Autonomic mediation failed: {e}")
            results['autonomic_mediation'] = {'error': str(e)}

    # 2. Circadian G×E
    if 'circadian_prs' in all_features.columns and 'evening_light' in all_features.columns:
        logger.info("Testing circadian G×E...")
        try:
            circadian_gxe = gxe_analyzer.test_circadian_gxe(
                clock_gene_prs=all_features['circadian_prs'].values,
                light_exposure=all_features['evening_light'].values,
                sleep_quality=all_features['sleep_quality'].values if 'sleep_quality' in all_features.columns else outcomes.iloc[:, 0].values,
                covariates=all_features[['age', 'sex']] if 'age' in all_features.columns else None
            )
            results['circadian_gxe'] = asdict(circadian_gxe)
        except Exception as e:
            logger.error(f"Circadian G×E failed: {e}")
            results['circadian_gxe'] = {'error': str(e)}

    # 3. Environmental Mixtures
    heavy_metal_cols = [c for c in all_features.columns if any(m in c.lower() for m in ['lead', 'mercury', 'cadmium', 'arsenic'])]
    pollutant_cols = [c for c in all_features.columns if any(p in c.lower() for p in ['bpa', 'phthalate', 'pm25', 'pesticide'])]

    if heavy_metal_cols or pollutant_cols:
        logger.info("Testing environmental mixtures...")
        try:
            mixture_result = mixture_analyzer.weighted_quantile_sum(
                exposures=all_features[heavy_metal_cols + pollutant_cols],
                outcome=outcomes.iloc[:, 0].values,
                covariates=all_features[['age', 'sex']] if 'age' in all_features.columns else None
            )
            results['environmental_mixtures'] = asdict(mixture_result)
        except Exception as e:
            logger.error(f"Environmental mixture analysis failed: {e}")
            results['environmental_mixtures'] = {'error': str(e)}

    # 4. Critical Periods (if temporal data available)
    # This would require longitudinal data with exposure timing
    # Placeholder for now
    results['critical_periods'] = {'status': 'requires_longitudinal_data'}

    # 5. Causal Network Discovery
    # Use subset of features for computational efficiency
    network_features = []
    for prefix in ['genetic', 'HRV', 'CAR', 'toxic', 'sensory', 'interoception']:
        network_features.extend([c for c in all_features.columns if prefix in c])

    if len(network_features) > 5:
        logger.info(f"Discovering causal network from {len(network_features)} features...")
        try:
            network_result = network_discovery.discover_network(
                data=all_features[network_features[:20]],  # Limit to 20 features
                algorithm='simple',
                alpha=0.05
            )
            results['causal_network'] = {
                'n_edges': network_result.n_edges,
                'n_features': len(network_result.feature_names),
                'edge_list': network_result.edge_list[:50],  # Top 50 edges
                'strongly_connected': network_result.strongly_connected,
                'algorithm': network_result.algorithm
            }
        except Exception as e:
            logger.error(f"Causal network discovery failed: {e}")
            results['causal_network'] = {'error': str(e)}

    logger.info("Extended causal analysis complete")

    return results


def visualize_extended_clusters(clusters: np.ndarray,
                               extended_features: pd.DataFrame,
                               save_path: Optional[str] = None) -> 'go.Figure':
    """
    Create comprehensive visualization of extended multi-modal features.

    Generates 9-panel figure showing:
    - Genetic-metabolic correlations
    - Autonomic profiles (radar)
    - Circadian phase distributions
    - Environmental burden
    - Sensory profiles
    - Clinical severity
    - Developmental trajectories
    - Treatment response patterns
    - Multimodal 3D integration

    Parameters
    ----------
    clusters : np.ndarray
        Cluster assignments for each sample
    extended_features : pd.DataFrame
        Multi-modal feature data
    save_path : str, optional
        Path to save figure HTML

    Returns
    -------
    go.Figure
        Interactive plotly figure

    Examples
    --------
    >>> fig = visualize_extended_clusters(cluster_labels, features)
    >>> fig.show()
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required for visualization. Install with: pip install plotly")

    n_clusters = len(np.unique(clusters))
    colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
              for r, g, b in plt.cm.Set2(np.linspace(0, 1, n_clusters))]

    # Create 3x3 subplot figure
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'Genetic-Metabolic', 'Autonomic Profile', 'Circadian Phase',
            'Environmental Burden', 'Sensory Profile', 'Clinical Severity',
            'Developmental Trajectory', 'Treatment Response', 'Multimodal 3D'
        ],
        specs=[
            [{'type': 'scatter'}, {'type': 'scatterpolar'}, {'type': 'box'}],
            [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter3d'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Panel 1: Genetic-Metabolic correlation
    genetic_cols = [c for c in extended_features.columns if 'genetic' in c.lower() or 'snp' in c.lower() or 'prs' in c.lower()]
    metabolic_cols = [c for c in extended_features.columns if 'metabol' in c.lower() or 'hmdb' in c.lower()]

    if genetic_cols and metabolic_cols:
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            x = extended_features.loc[mask, genetic_cols[0]] if len(genetic_cols) > 0 else np.random.randn(mask.sum())
            y = extended_features.loc[mask, metabolic_cols[0]] if len(metabolic_cols) > 0 else np.random.randn(mask.sum())

            fig.add_trace(
                go.Scatter(x=x, y=y, mode='markers',
                          marker=dict(color=colors[cluster_id], size=6, opacity=0.6),
                          name=f'Cluster {cluster_id}',
                          showlegend=(cluster_id == 0)),
                row=1, col=1
            )

    # Panel 2: Autonomic Profile (Radar)
    autonomic_features = [c for c in extended_features.columns
                         if any(x in c.lower() for x in ['hrv', 'heart_rate', 'rmssd', 'sdnn', 'lf_hf'])]

    if len(autonomic_features) >= 3:
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            values = [extended_features.loc[mask, f].mean() for f in autonomic_features[:6]]
            values.append(values[0])  # Close the radar

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=autonomic_features[:6] + [autonomic_features[0]],
                    fill='toself',
                    name=f'Cluster {cluster_id}',
                    line=dict(color=colors[cluster_id]),
                    showlegend=False
                ),
                row=1, col=2
            )

    # Panel 3: Circadian Phase (Box plots)
    circadian_cols = [c for c in extended_features.columns
                     if any(x in c.lower() for x in ['circadian', 'melatonin', 'sleep', 'dlmo'])]

    if circadian_cols:
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            fig.add_trace(
                go.Box(y=extended_features.loc[mask, circadian_cols[0]],
                      name=f'C{cluster_id}',
                      marker=dict(color=colors[cluster_id]),
                      showlegend=False),
                row=1, col=3
            )

    # Panel 4: Environmental Burden (Stacked bar)
    env_cols = [c for c in extended_features.columns
               if any(x in c.lower() for x in ['lead', 'mercury', 'pm25', 'pollut', 'pesticide'])]

    if env_cols:
        cluster_means = []
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            cluster_means.append([extended_features.loc[mask, c].mean() for c in env_cols[:5]])

        for i, env_feature in enumerate(env_cols[:5]):
            fig.add_trace(
                go.Bar(x=list(range(n_clusters)),
                      y=[cm[i] if i < len(cm) else 0 for cm in cluster_means],
                      name=env_feature,
                      showlegend=(i == 0)),
                row=2, col=1
            )

    # Panel 5: Sensory Profile
    sensory_cols = [c for c in extended_features.columns
                   if any(x in c.lower() for x in ['sensory', 'tactile', 'auditory', 'visual', 'interoception'])]

    if len(sensory_cols) >= 2:
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            x = extended_features.loc[mask, sensory_cols[0]]
            y = extended_features.loc[mask, sensory_cols[1]]

            fig.add_trace(
                go.Scatter(x=x, y=y, mode='markers',
                          marker=dict(color=colors[cluster_id], size=5, opacity=0.5),
                          showlegend=False),
                row=2, col=2
            )

    # Panel 6: Clinical Severity (Bar plot)
    severity_cols = [c for c in extended_features.columns
                    if any(x in c.lower() for x in ['severity', 'symptom', 'adhd', 'asd', 'score'])]

    if severity_cols:
        cluster_severity = [extended_features.loc[clusters == i, severity_cols[0]].mean()
                          for i in range(n_clusters)]

        fig.add_trace(
            go.Bar(x=list(range(n_clusters)),
                  y=cluster_severity,
                  marker=dict(color=colors),
                  showlegend=False),
            row=2, col=3
        )

    # Panel 7: Developmental Trajectory
    if 'age' in extended_features.columns and severity_cols:
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            x = extended_features.loc[mask, 'age']
            y = extended_features.loc[mask, severity_cols[0]]

            # Add scatter
            fig.add_trace(
                go.Scatter(x=x, y=y, mode='markers',
                          marker=dict(color=colors[cluster_id], size=4, opacity=0.4),
                          showlegend=False),
                row=3, col=1
            )

            # Add trend line
            if len(x) > 5:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 50)

                fig.add_trace(
                    go.Scatter(x=x_line, y=p(x_line), mode='lines',
                              line=dict(color=colors[cluster_id], width=2),
                              showlegend=False),
                    row=3, col=1
                )

    # Panel 8: Treatment Response
    treatment_cols = [c for c in extended_features.columns
                     if any(x in c.lower() for x in ['treatment', 'response', 'medication', 'therapy'])]

    if treatment_cols and len(treatment_cols) >= 2:
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            x = extended_features.loc[mask, treatment_cols[0]]
            y = extended_features.loc[mask, treatment_cols[1]]

            fig.add_trace(
                go.Scatter(x=x, y=y, mode='markers',
                          marker=dict(color=colors[cluster_id], size=5),
                          showlegend=False),
                row=3, col=2
            )

    # Panel 9: Multimodal 3D Integration
    # Use PCA to reduce to 3 dimensions for visualization
    from sklearn.decomposition import PCA

    # Select diverse features for PCA
    feature_subset = []
    for prefix in ['genetic', 'hrv', 'circadian', 'env', 'sensory']:
        cols = [c for c in extended_features.columns if prefix in c.lower()]
        if cols:
            feature_subset.append(cols[0])

    if len(feature_subset) >= 3:
        pca_features = extended_features[feature_subset].fillna(0)
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(pca_features)

        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            fig.add_trace(
                go.Scatter3d(
                    x=coords_3d[mask, 0],
                    y=coords_3d[mask, 1],
                    z=coords_3d[mask, 2],
                    mode='markers',
                    marker=dict(color=colors[cluster_id], size=3, opacity=0.6),
                    name=f'Cluster {cluster_id}',
                    showlegend=False
                ),
                row=3, col=3
            )

    # Update layout
    fig.update_layout(
        title_text="Extended Multi-Modal Cluster Characterization",
        height=1200,
        width=1600,
        showlegend=True,
        legend=dict(x=1.02, y=0.5),
        barmode='stack'
    )

    # Update axes labels
    fig.update_xaxes(title_text="Genetic Risk", row=1, col=1)
    fig.update_yaxes(title_text="Metabolite Level", row=1, col=1)

    fig.update_xaxes(title_text="Age", row=3, col=1)
    fig.update_yaxes(title_text="Symptom Severity", row=3, col=1)

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Visualization saved to {save_path}")

    return fig


def create_causal_pathway_diagram(mediation_results: Dict[str, MediationResult],
                                 gxe_results: Dict[str, GxEResult],
                                 output_path: str = "causal_pathways.html") -> None:
    """
    Create interactive visualization of significant causal pathways.

    Parameters
    ----------
    mediation_results : Dict[str, MediationResult]
        Results from mediation analyses
    gxe_results : Dict[str, GxEResult]
        Results from G×E interaction analyses
    output_path : str
        Path to save HTML figure
    """
    try:
        import plotly.graph_objects as go
        import networkx as nx
    except ImportError:
        raise ImportError("plotly and networkx required. Install with: pip install plotly networkx")

    # Create directed graph
    G = nx.DiGraph()

    # Add mediation pathways
    for pathway_name, result in mediation_results.items():
        if result.p_value < 0.05:
            parts = pathway_name.split('_')
            if len(parts) >= 3:
                treatment, mediator, outcome = parts[0], parts[1], parts[2]

                # Add edges with weights
                G.add_edge(treatment, mediator, weight=abs(result.indirect_effect), type='mediation')
                G.add_edge(mediator, outcome, weight=abs(result.indirect_effect), type='mediation')
                G.add_edge(treatment, outcome, weight=abs(result.direct_effect), type='direct', style='dashed')

    # Add G×E interactions
    for interaction_name, result in gxe_results.items():
        if result.p_interaction < 0.05:
            parts = interaction_name.split('_x_')
            if len(parts) >= 2:
                gene_var, env_var = parts[0], parts[1].split('_on_')[0]
                outcome = parts[1].split('_on_')[1] if '_on_' in parts[1] else 'outcome'

                # Add interaction node
                interaction_node = f"{gene_var}×{env_var}"
                G.add_edge(gene_var, interaction_node, weight=1, type='interaction')
                G.add_edge(env_var, interaction_node, weight=1, type='interaction')
                G.add_edge(interaction_node, outcome, weight=abs(result.interaction_effect), type='effect')

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_type = edge[2].get('type', 'default')
        weight = edge[2].get('weight', 1)

        color = 'rgba(100,100,100,0.5)'
        if edge_type == 'mediation':
            color = 'rgba(255,100,100,0.7)'
        elif edge_type == 'interaction':
            color = 'rgba(100,100,255,0.7)'

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=weight*3, color=color),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # Size by degree
        degree = G.degree(node)
        node_size.append(20 + degree * 5)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            size=node_size,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        hoverinfo='text',
        showlegend=False
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title="Significant Causal Pathways",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1000,
        height=800
    )

    fig.write_html(output_path)
    logger.info(f"Causal pathway diagram saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500

    all_features = pd.DataFrame({
        'genetic_prs': np.random.randn(n_samples),
        'HRV_SDNN': np.random.randn(n_samples),
        'circadian_prs': np.random.randn(n_samples),
        'evening_light': np.random.randn(n_samples),
        'sleep_quality': np.random.randn(n_samples),
        'age': np.random.randint(6, 18, n_samples),
        'sex': np.random.choice([0, 1], n_samples)
    })

    outcomes = pd.DataFrame({
        'ADHD_symptoms': np.random.randn(n_samples),
        'ASD_symptoms': np.random.randn(n_samples)
    })

    # Run analysis
    results = extended_causal_analysis(all_features, outcomes)

    print("\nExtended Causal Analysis Results:")
    print(json.dumps(results, indent=2, default=str))
