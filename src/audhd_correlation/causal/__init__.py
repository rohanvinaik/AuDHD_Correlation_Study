"""Causal analysis tools for AuDHD correlation study

Implements causal inference methods including:
- DAG construction and validation
- Mendelian randomization
- Mediation analysis
- G×E interaction detection
- Sensitivity analysis for unmeasured confounding
"""

from .dag import (
    build_dag,
    validate_dag,
    identify_confounders,
    identify_mediators,
    plot_dag,
    DAGSpecification,
)

from .mendelian_randomization import (
    mendelian_randomization,
    test_instrument_validity,
    calculate_f_statistic,
    MRResult,
)

from .mediation import (
    mediation_analysis,
    multi_step_mediation,
    MediationResult,
)

from .interactions import (
    detect_gxe_interactions,
    causal_forest_analysis,
    heterogeneous_treatment_effects,
    GxEResult,
)

from .sensitivity import (
    calculate_e_value,
    sensitivity_analysis,
    unmeasured_confounding_bounds,
    SensitivityResult,
)

__all__ = [
    # DAG
    'build_dag',
    'validate_dag',
    'identify_confounders',
    'identify_mediators',
    'plot_dag',
    'DAGSpecification',
    # Mendelian Randomization
    'mendelian_randomization',
    'test_instrument_validity',
    'calculate_f_statistic',
    'MRResult',
    # Mediation
    'mediation_analysis',
    'multi_step_mediation',
    'MediationResult',
    # Interactions
    'detect_gxe_interactions',
    'causal_forest_analysis',
    'heterogeneous_treatment_effects',
    'GxEResult',
    # Sensitivity
    'calculate_e_value',
    'sensitivity_analysis',
    'unmeasured_confounding_bounds',
    'SensitivityResult',
]