"""Comprehensive tests for causal analysis framework"""
import pytest
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from src.audhd_correlation.causal.dag import (
    build_dag,
    validate_dag,
    identify_confounders,
    identify_mediators,
    identify_colliders,
    identify_adjustment_set,
    create_audhd_dag,
)

from src.audhd_correlation.causal.mendelian_randomization import (
    mendelian_randomization,
    calculate_f_statistic,
    test_instrument_validity,
    mr_egger_regression,
)

from src.audhd_correlation.causal.mediation import (
    mediation_analysis,
    multi_step_mediation,
)

from src.audhd_correlation.causal.interactions import (
    detect_gxe_interactions,
    causal_forest_analysis,
    heterogeneous_treatment_effects,
)

from src.audhd_correlation.causal.sensitivity import (
    calculate_e_value,
    sensitivity_analysis,
    unmeasured_confounding_bounds,
    rosenbaum_sensitivity,
)


@pytest.fixture
def simple_dag():
    """Create simple test DAG"""
    nodes = ['X', 'M', 'Y', 'C']
    edges = [
        ('C', 'X'),  # Confounder
        ('C', 'Y'),
        ('X', 'M'),  # Mediator
        ('M', 'Y'),
    ]
    return build_dag(nodes, edges, exposure='X', outcome='Y')


@pytest.fixture
def causal_data():
    """Generate synthetic causal data"""
    np.random.seed(42)
    n = 500

    # Confounder
    C = np.random.randn(n)

    # Exposure (affected by confounder)
    X = 0.5 * C + np.random.randn(n)

    # Mediator (affected by exposure)
    M = 0.6 * X + np.random.randn(n)

    # Outcome (affected by confounder, mediator, and direct effect)
    Y = 0.3 * C + 0.4 * M + 0.2 * X + np.random.randn(n)

    return {'X': X, 'M': M, 'Y': Y, 'C': C}


@pytest.fixture
def mr_data():
    """Generate Mendelian randomization data"""
    np.random.seed(42)
    n = 1000

    # Genetic instruments
    G1 = np.random.binomial(2, 0.3, n)  # SNP 1
    G2 = np.random.binomial(2, 0.4, n)  # SNP 2

    # Exposure (affected by genetics)
    exposure = 0.5 * G1 + 0.3 * G2 + np.random.randn(n)

    # Outcome (affected by exposure)
    outcome = 0.6 * exposure + np.random.randn(n)

    return {
        'instruments': np.column_stack([G1, G2]),
        'exposure': exposure,
        'outcome': outcome,
    }


class TestDAG:
    """Tests for DAG construction and analysis"""

    def test_build_dag(self, simple_dag):
        """Test DAG construction"""
        assert isinstance(simple_dag, nx.DiGraph)
        assert len(simple_dag.nodes()) == 4
        assert len(simple_dag.edges()) == 4
        assert simple_dag.graph['exposure'] == 'X'
        assert simple_dag.graph['outcome'] == 'Y'

    def test_validate_dag(self, simple_dag):
        """Test DAG validation"""
        validation = validate_dag(simple_dag)

        assert validation['is_acyclic'] is True
        assert validation['is_weakly_connected'] is True
        assert validation['exposure_affects_outcome'] is True
        assert validation['n_paths'] > 0

    def test_identify_confounders(self, simple_dag):
        """Test confounder identification"""
        confounders = identify_confounders(simple_dag)

        assert 'C' in confounders
        assert 'M' not in confounders  # M is mediator, not confounder

    def test_identify_mediators(self, simple_dag):
        """Test mediator identification"""
        mediators = identify_mediators(simple_dag)

        assert 'M' in mediators
        assert 'C' not in mediators

    def test_identify_adjustment_set(self, simple_dag):
        """Test adjustment set identification"""
        adjustment_set = identify_adjustment_set(simple_dag, method='backdoor')

        # Should include confounder, not mediator
        assert 'C' in adjustment_set
        assert 'M' not in adjustment_set

    def test_create_audhd_dag(self):
        """Test creation of AuDHD DAG"""
        dag = create_audhd_dag()

        assert isinstance(dag, nx.DiGraph)
        assert 'genetics' in dag.nodes()
        assert 'comorbidity' in dag.nodes()
        assert dag.graph['exposure'] == 'genetics'
        assert dag.graph['outcome'] == 'comorbidity'

    def test_cyclic_dag_error(self):
        """Test that cyclic graphs raise error"""
        nodes = ['A', 'B', 'C']
        edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]  # Cycle

        with pytest.raises(ValueError, match="cycles"):
            build_dag(nodes, edges, exposure='A', outcome='C')


class TestMendelianRandomization:
    """Tests for Mendelian randomization"""

    def test_mendelian_randomization_2sls(self, mr_data):
        """Test 2SLS MR"""
        result = mendelian_randomization(
            instruments=mr_data['instruments'],
            exposure=mr_data['exposure'],
            outcome=mr_data['outcome'],
            method='2sls',
        )

        assert result.causal_estimate is not None
        assert result.standard_error > 0
        assert 0 <= result.p_value <= 1
        assert result.f_statistic > 0
        assert result.method == '2sls'

    def test_mendelian_randomization_ratio(self, mr_data):
        """Test ratio method MR"""
        result = mendelian_randomization(
            instruments=mr_data['instruments'],
            exposure=mr_data['exposure'],
            outcome=mr_data['outcome'],
            method='ratio',
        )

        assert result.causal_estimate is not None
        assert result.method == 'ratio'

    def test_f_statistic(self, mr_data):
        """Test F-statistic calculation"""
        f_stat = calculate_f_statistic(
            mr_data['instruments'],
            mr_data['exposure'],
        )

        assert f_stat > 0
        # With our data, should be strong instrument
        assert f_stat > 10

    def test_instrument_validity(self, mr_data):
        """Test instrument validity tests"""
        tests = test_instrument_validity(
            instruments=mr_data['instruments'],
            exposure=mr_data['exposure'],
            outcome=mr_data['outcome'],
        )

        assert 'f_statistic' in tests
        assert 'strong_instrument' in tests
        assert tests['f_statistic'] > 0

    def test_mr_egger(self, mr_data):
        """Test MR-Egger regression"""
        result = mr_egger_regression(
            instruments=mr_data['instruments'],
            exposure=mr_data['exposure'],
            outcome=mr_data['outcome'],
        )

        assert 'intercept' in result
        assert 'slope' in result
        assert 'pleiotropy_detected' in result


class TestMediation:
    """Tests for mediation analysis"""

    def test_mediation_analysis(self, causal_data):
        """Test basic mediation analysis"""
        result = mediation_analysis(
            exposure=causal_data['X'],
            mediator=causal_data['M'],
            outcome=causal_data['Y'],
            n_bootstrap=100,  # Small for speed
        )

        assert result.total_effect is not None
        assert result.direct_effect is not None
        assert result.indirect_effect is not None
        assert 0 <= result.proportion_mediated <= 1

    def test_mediation_with_covariates(self, causal_data):
        """Test mediation with covariates"""
        result = mediation_analysis(
            exposure=causal_data['X'],
            mediator=causal_data['M'],
            outcome=causal_data['Y'],
            covariates=causal_data['C'].reshape(-1, 1),
            n_bootstrap=100,
        )

        assert result.total_effect is not None

    def test_mediation_sobel(self, causal_data):
        """Test Sobel test mediation"""
        result = mediation_analysis(
            exposure=causal_data['X'],
            mediator=causal_data['M'],
            outcome=causal_data['Y'],
            method='sobel',
        )

        assert result.method == 'sobel'
        assert result.indirect_effect is not None

    def test_multi_step_mediation(self, causal_data):
        """Test multi-step mediation"""
        # Create second mediator
        M2 = 0.5 * causal_data['M'] + np.random.randn(len(causal_data['M']))

        results = multi_step_mediation(
            exposure=causal_data['X'],
            mediators=[causal_data['M'], M2],
            outcome=causal_data['Y'],
            n_bootstrap=50,
        )

        assert len(results) > 0
        assert isinstance(results, dict)


class TestInteractions:
    """Tests for G×E interactions"""

    def test_detect_gxe_linear(self):
        """Test linear G×E detection"""
        np.random.seed(42)
        n = 500

        # Genetics
        G = np.random.randn(n)

        # Environment
        E = np.random.randn(n)

        # Outcome with interaction
        Y = 0.3 * G + 0.4 * E + 0.5 * G * E + np.random.randn(n)

        result = detect_gxe_interactions(
            genetics=G,
            environment=E,
            outcome=Y,
            method='linear',
        )

        assert result.genetic_effect is not None
        assert result.environmental_effect is not None
        assert result.interaction_effect is not None
        assert result.method == 'linear'

    def test_causal_forest_analysis(self):
        """Test causal forest analysis"""
        np.random.seed(42)
        n = 300

        # Treatment
        T = np.random.binomial(1, 0.5, n)

        # Covariates
        X = np.random.randn(n, 5)

        # Outcome (with heterogeneous effects)
        Y = 0.5 * T + 0.3 * T * X[:, 0] + np.random.randn(n)

        results = causal_forest_analysis(
            treatment=T,
            outcome=Y,
            covariates=X,
            n_estimators=100,
        )

        assert 'cate' in results
        assert 'ate' in results
        assert len(results['cate']) == n

    def test_heterogeneous_effects(self):
        """Test heterogeneous treatment effect analysis"""
        np.random.seed(42)
        n = 300

        T = np.random.binomial(1, 0.5, n)
        X = np.random.randn(n, 5)
        Y = 0.5 * T + 0.3 * T * X[:, 0] + np.random.randn(n)

        effects = heterogeneous_treatment_effects(
            treatment=T,
            outcome=Y,
            covariates=X,
        )

        assert isinstance(effects, dict)
        assert len(effects) > 0


class TestSensitivity:
    """Tests for sensitivity analysis"""

    def test_calculate_e_value(self):
        """Test E-value calculation"""
        e_values = calculate_e_value(
            estimate=2.0,
            ci_lower=1.5,
            ci_upper=2.5,
            outcome_type='rr',
        )

        assert 'e_value_point' in e_values
        assert 'e_value_ci' in e_values
        assert e_values['e_value_point'] > 1
        assert e_values['e_value_ci'] > 1

    def test_sensitivity_analysis(self):
        """Test comprehensive sensitivity analysis"""
        result = sensitivity_analysis(
            estimate=2.0,
            standard_error=0.2,
            outcome_type='rr',
        )

        assert result.e_value > 0
        assert result.e_value_ci > 0
        assert result.interpretation is not None

    def test_unmeasured_confounding_bounds(self):
        """Test confounding bounds"""
        bounds = unmeasured_confounding_bounds(
            treatment_effect=2.0,
            confounder_treatment_rr=1.5,
            confounder_outcome_rr=2.0,
        )

        assert 'bias_factor' in bounds
        assert 'adjusted_lower_bound' in bounds
        assert 'adjusted_upper_bound' in bounds
        assert 'crosses_null' in bounds

    def test_rosenbaum_sensitivity(self):
        """Test Rosenbaum sensitivity analysis"""
        np.random.seed(42)
        n = 100

        treated = np.random.randn(n) + 0.5  # Treatment effect
        control = np.random.randn(n)

        results = rosenbaum_sensitivity(
            treated_outcomes=treated,
            control_outcomes=control,
            gamma_values=np.array([1.0, 1.5, 2.0]),
        )

        assert isinstance(results, dict)
        assert len(results) == 3
        assert all(0 <= p <= 1 for p in results.values())


class TestEdgeCases:
    """Test edge cases"""

    def test_single_instrument_mr(self):
        """Test MR with single instrument"""
        np.random.seed(42)
        n = 500

        G = np.random.binomial(2, 0.3, n).reshape(-1, 1)
        exposure = 0.5 * G[:, 0] + np.random.randn(n)
        outcome = 0.6 * exposure + np.random.randn(n)

        result = mendelian_randomization(
            instruments=G,
            exposure=exposure,
            outcome=outcome,
            method='wald',
        )

        assert result.method == 'wald'

    def test_weak_instrument(self):
        """Test weak instrument warning"""
        np.random.seed(42)
        n = 500

        # Weak instrument
        G = np.random.randn(n, 1) * 0.01
        exposure = G[:, 0] * 0.1 + np.random.randn(n)
        outcome = exposure * 0.5 + np.random.randn(n)

        with pytest.warns(UserWarning, match="Weak instrument"):
            result = mendelian_randomization(
                instruments=G,
                exposure=exposure,
                outcome=outcome,
            )

        assert result.weak_instrument is True

    def test_small_sample_mediation(self):
        """Test mediation with small sample"""
        np.random.seed(42)
        n = 50

        X = np.random.randn(n)
        M = 0.5 * X + np.random.randn(n)
        Y = 0.4 * M + np.random.randn(n)

        result = mediation_analysis(
            exposure=X,
            mediator=M,
            outcome=Y,
            n_bootstrap=50,
        )

        assert result is not None