"""
Adaptive Weighting System with Literature-Based Initialization.

Implements evidence-based feature domain weighting with optimization
feedback loop to maximize discriminative power while remaining grounded
in literature findings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import optimize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import logging
import json
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WeightConstraints:
    """Constraints for weight optimization based on literature."""
    domain: str
    initial_weight: float
    min_weight: float
    max_weight: float
    evidence_strength: str  # 'strong', 'moderate', 'weak'
    literature_references: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class LiteratureBasedWeights:
    """Literature-derived weights with confidence bounds."""

    # Genetics: h² = 0.74-0.80, accounts for 30-40% of cases directly
    genetic: WeightConstraints = field(default_factory=lambda: WeightConstraints(
        domain='genetic',
        initial_weight=0.28,
        min_weight=0.24,
        max_weight=0.33,
        evidence_strength='strong',
        literature_references=[
            'Tick et al. 2016 (meta-analysis, h²=0.74)',
            'Bai et al. 2019 (~80% genetic contribution)',
            'Sandin et al. 2017 (ASD heritability 0.83)'
        ],
        notes='High heritability but requires G×E interactions for full effect; rebalanced for prenatal factors'
    ))

    # Metabolomics: AUC 0.90-0.96, very high discriminative power
    metabolomic: WeightConstraints = field(default_factory=lambda: WeightConstraints(
        domain='metabolomic',
        initial_weight=0.20,
        min_weight=0.17,
        max_weight=0.25,
        evidence_strength='strong',
        literature_references=[
            'Liu et al. 2024 (AUC 0.935-0.963)',
            'SVM accuracy 86%, AUC 0.95',
            'CAMP study: 53% sensitivity, 91% specificity'
        ],
        notes='Proximal to phenotype, reflects current biological state'
    ))

    # Prenatal/Maternal: Strong evidence for developmental impact
    prenatal_maternal: WeightConstraints = field(default_factory=lambda: WeightConstraints(
        domain='prenatal_maternal',
        initial_weight=0.12,
        min_weight=0.10,
        max_weight=0.16,
        evidence_strength='strong',
        literature_references=[
            'Atladóttir et al. 2010 (maternal infection: OR 1.3-2.0)',
            'Patterson 2011 (MIA hypothesis: maternal immune activation)',
            'Meyer 2006 (neurogenesis window weeks 10-20: OR 1.5-2.0)',
            'Johnson & Marlow 2017 (preterm birth: OR 1.5-2.5)',
            'Christensen 2013 (valproate exposure: OR 3.0-5.0)',
            'Brown 2017 (SSRI exposure: OR 1.2-1.5, controversial)'
        ],
        notes='Critical developmental windows; MIA during neurogenesis shows strongest effect; G×E interaction potential'
    ))

    # Environmental: ~25% via G×E, ~3% direct = ~28% total
    environmental: WeightConstraints = field(default_factory=lambda: WeightConstraints(
        domain='environmental',
        initial_weight=0.10,
        min_weight=0.08,
        max_weight=0.14,
        evidence_strength='strong',
        literature_references=[
            'NRC: 25% G×E + 3% direct environmental',
            'Meta-analysis: equal contribution with genetics via interactions'
        ],
        notes='Critical for gene-environment interactions'
    ))

    # Autonomic: AUC 0.736, moderate discriminative power
    autonomic: WeightConstraints = field(default_factory=lambda: WeightConstraints(
        domain='autonomic',
        initial_weight=0.10,
        min_weight=0.07,
        max_weight=0.14,
        evidence_strength='moderate',
        literature_references=[
            'Meta-analysis: AUC 0.736 for ASD',
            'Distinct ASD (hyper-arousal) vs ADHD (hypo-arousal) profiles'
        ],
        notes='HRV, autonomic function; distinct patterns by subtype'
    ))

    # Toxicants: Subset of environmental, synergistic effects
    toxicant: WeightConstraints = field(default_factory=lambda: WeightConstraints(
        domain='toxicant',
        initial_weight=0.08,
        min_weight=0.05,
        max_weight=0.12,
        evidence_strength='moderate',
        literature_references=[
            'Heavy metals meta-analysis 2023',
            'Synergistic with genetic risk'
        ],
        notes='Heavy metals, phthalates, air pollution'
    ))

    # Circadian: 53-93% prevalence, causal relationship
    circadian: WeightConstraints = field(default_factory=lambda: WeightConstraints(
        domain='circadian',
        initial_weight=0.07,
        min_weight=0.05,
        max_weight=0.11,
        evidence_strength='moderate',
        literature_references=[
            '53-93% sleep problems in ASD/ADHD',
            'Mendelian randomization: causal relationship',
            'Delayed DLMO in ADHD, atypical patterns in ASD'
        ],
        notes='Regulatory, high prevalence, causally implicated'
    ))

    # Microbiome: Emerging, intermediate role
    microbiome: WeightConstraints = field(default_factory=lambda: WeightConstraints(
        domain='microbiome',
        initial_weight=0.04,
        min_weight=0.03,
        max_weight=0.08,
        evidence_strength='moderate',
        literature_references=[
            'Gut-brain axis studies',
            'Intermediate biological role',
            'Maternal microbiome transfer during pregnancy'
        ],
        notes='Gut-brain axis, intermediate between environment and metabolism; overlaps with prenatal'
    ))

    # Sensory/Interoception: Core features, less discriminative data
    sensory: WeightConstraints = field(default_factory=lambda: WeightConstraints(
        domain='sensory',
        initial_weight=0.01,
        min_weight=0.00,
        max_weight=0.04,
        evidence_strength='weak',
        literature_references=[
            'Core ASD features',
            'Limited quantitative diagnostic accuracy data'
        ],
        notes='Clinically relevant but limited discriminative studies'
    ))

    # NOTE: clinical domain removed - outcome variable, not predictor

    def to_dict(self) -> Dict[str, WeightConstraints]:
        """Convert to dictionary keyed by domain name."""
        return {
            'genetic': self.genetic,
            'metabolomic': self.metabolomic,
            'prenatal_maternal': self.prenatal_maternal,
            'environmental': self.environmental,
            'autonomic': self.autonomic,
            'toxicant': self.toxicant,
            'circadian': self.circadian,
            'microbiome': self.microbiome,
            'sensory': self.sensory
        }

    def get_initial_weights(self) -> Dict[str, float]:
        """Get initial weights as dictionary."""
        return {domain: constraints.initial_weight
                for domain, constraints in self.to_dict().items()}

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get (min, max) bounds for each domain."""
        return {domain: (constraints.min_weight, constraints.max_weight)
                for domain, constraints in self.to_dict().items()}


class AdaptiveWeightOptimizer:
    """
    Optimize feature domain weights to maximize discriminative power
    while respecting literature-based constraints.
    """

    def __init__(self,
                 constraints: Optional[LiteratureBasedWeights] = None,
                 optimization_metric: str = 'silhouette',
                 max_iterations: int = 100,
                 convergence_threshold: float = 0.001):
        """
        Initialize adaptive weight optimizer.

        Parameters
        ----------
        constraints : LiteratureBasedWeights, optional
            Literature-based weight constraints
        optimization_metric : str
            Metric to optimize ('silhouette', 'davies_bouldin', 'calinski_harabasz', 'classification')
        max_iterations : int
            Maximum optimization iterations
        convergence_threshold : float
            Convergence threshold for weight changes
        """
        self.constraints = constraints or LiteratureBasedWeights()
        self.optimization_metric = optimization_metric
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        self.current_weights = self.constraints.get_initial_weights()
        self.weight_history = [self.current_weights.copy()]
        self.metric_history = []
        self.iteration = 0

        logger.info(f"Initialized AdaptiveWeightOptimizer with {optimization_metric} metric")
        logger.info(f"Initial weights: {self.current_weights}")

    def compute_discriminative_score(self,
                                    domain_features: Dict[str, pd.DataFrame],
                                    weights: Dict[str, float],
                                    cluster_labels: Optional[np.ndarray] = None,
                                    true_labels: Optional[np.ndarray] = None) -> float:
        """
        Compute discriminative score for a given set of weights.

        Parameters
        ----------
        domain_features : dict
            Dictionary of domain_name -> DataFrame features
        weights : dict
            Dictionary of domain_name -> weight
        cluster_labels : np.ndarray, optional
            Cluster assignments (for clustering metrics)
        true_labels : np.ndarray, optional
            True class labels (for classification metrics)

        Returns
        -------
        float
            Discriminative score (higher is better)
        """
        # Combine features with weights
        weighted_features = []
        for domain, features in domain_features.items():
            if domain in weights:
                weight = weights[domain]
                weighted = features * weight
                weighted_features.append(weighted)

        if not weighted_features:
            return 0.0

        # Concatenate all weighted features
        X = pd.concat(weighted_features, axis=1).values

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        if self.optimization_metric == 'silhouette':
            if cluster_labels is None:
                logger.warning("Silhouette metric requires cluster_labels")
                return 0.0
            try:
                score = silhouette_score(X, cluster_labels)
                return score
            except Exception as e:
                logger.warning(f"Silhouette score failed: {e}")
                return 0.0

        elif self.optimization_metric == 'davies_bouldin':
            if cluster_labels is None:
                logger.warning("Davies-Bouldin metric requires cluster_labels")
                return 0.0
            try:
                score = davies_bouldin_score(X, cluster_labels)
                # Invert because lower is better
                return -score
            except Exception as e:
                logger.warning(f"Davies-Bouldin score failed: {e}")
                return 0.0

        elif self.optimization_metric == 'calinski_harabasz':
            if cluster_labels is None:
                logger.warning("Calinski-Harabasz metric requires cluster_labels")
                return 0.0
            try:
                score = calinski_harabasz_score(X, cluster_labels)
                return score
            except Exception as e:
                logger.warning(f"Calinski-Harabasz score failed: {e}")
                return 0.0

        elif self.optimization_metric == 'classification':
            if true_labels is None:
                logger.warning("Classification metric requires true_labels")
                return 0.0
            try:
                clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                scores = cross_val_score(clf, X, true_labels, cv=3, scoring='accuracy')
                return scores.mean()
            except Exception as e:
                logger.warning(f"Classification score failed: {e}")
                return 0.0

        else:
            raise ValueError(f"Unknown optimization metric: {self.optimization_metric}")

    def optimize_weights(self,
                        domain_features: Dict[str, pd.DataFrame],
                        cluster_labels: Optional[np.ndarray] = None,
                        true_labels: Optional[np.ndarray] = None,
                        verbose: bool = True) -> Dict[str, float]:
        """
        Optimize weights using constrained optimization.

        Parameters
        ----------
        domain_features : dict
            Dictionary of domain_name -> DataFrame features
        cluster_labels : np.ndarray, optional
            Cluster assignments
        true_labels : np.ndarray, optional
            True class labels
        verbose : bool
            Print optimization progress

        Returns
        -------
        dict
            Optimized weights
        """
        logger.info("Starting weight optimization...")

        # Get domains present in data
        available_domains = list(domain_features.keys())
        logger.info(f"Available domains: {available_domains}")

        # Filter constraints to available domains
        domain_to_idx = {domain: i for i, domain in enumerate(available_domains)}
        bounds_dict = self.constraints.get_bounds()

        # Create bounds array
        bounds = [bounds_dict.get(domain, (0.0, 1.0)) for domain in available_domains]

        # Initial weights (only for available domains)
        x0 = np.array([self.current_weights.get(domain, 0.1) for domain in available_domains])

        # Normalize initial weights
        x0 = x0 / x0.sum()

        def objective(x):
            """Objective function: negative discriminative score."""
            # Create weight dict
            weights = {domain: x[domain_to_idx[domain]]
                      for domain in available_domains}

            # Compute score
            score = self.compute_discriminative_score(
                domain_features, weights, cluster_labels, true_labels
            )

            return -score  # Negative because we minimize

        def constraint_sum(x):
            """Constraint: weights must sum to 1."""
            return x.sum() - 1.0

        # Set up constraints
        constraints = [
            {'type': 'eq', 'fun': constraint_sum}  # Sum to 1
        ]

        # Optimize
        result = optimize.minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.convergence_threshold}
        )

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")

        # Extract optimized weights
        optimized_weights = {domain: result.x[domain_to_idx[domain]]
                            for domain in available_domains}

        # Add back domains not in data (keep initial weights, renormalized)
        all_domains = set(self.current_weights.keys())
        missing_domains = all_domains - set(available_domains)
        for domain in missing_domains:
            optimized_weights[domain] = self.current_weights[domain]

        # Renormalize
        total = sum(optimized_weights.values())
        optimized_weights = {k: v / total for k, v in optimized_weights.items()}

        # Store history
        self.current_weights = optimized_weights
        self.weight_history.append(optimized_weights.copy())
        self.metric_history.append(-result.fun)

        if verbose:
            logger.info(f"Optimization complete. Best score: {-result.fun:.4f}")
            logger.info(f"Optimized weights: {optimized_weights}")

        return optimized_weights

    def adaptive_feedback_loop(self,
                               domain_features: Dict[str, pd.DataFrame],
                               cluster_labels: Optional[np.ndarray] = None,
                               true_labels: Optional[np.ndarray] = None,
                               n_iterations: int = 5) -> Dict[str, Any]:
        """
        Run adaptive feedback loop to iteratively optimize weights.

        Parameters
        ----------
        domain_features : dict
            Dictionary of domain_name -> DataFrame features
        cluster_labels : np.ndarray, optional
            Cluster assignments (updated each iteration)
        true_labels : np.ndarray, optional
            True class labels
        n_iterations : int
            Number of feedback iterations

        Returns
        -------
        dict
            Results including final weights, history, and metrics
        """
        logger.info(f"Starting adaptive feedback loop ({n_iterations} iterations)...")

        results = {
            'iterations': [],
            'final_weights': None,
            'convergence': False
        }

        for iteration in range(n_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")

            # Optimize weights
            new_weights = self.optimize_weights(
                domain_features, cluster_labels, true_labels, verbose=True
            )

            # Check convergence
            if iteration > 0:
                prev_weights = self.weight_history[-2]
                weight_changes = {domain: abs(new_weights[domain] - prev_weights.get(domain, 0))
                                for domain in new_weights.keys()}
                max_change = max(weight_changes.values())

                logger.info(f"Max weight change: {max_change:.6f}")

                if max_change < self.convergence_threshold:
                    logger.info("Convergence reached!")
                    results['convergence'] = True
                    break

            results['iterations'].append({
                'iteration': iteration + 1,
                'weights': new_weights.copy(),
                'metric_score': self.metric_history[-1]
            })

        results['final_weights'] = self.current_weights
        results['weight_history'] = self.weight_history
        results['metric_history'] = self.metric_history

        return results

    def export_weights(self, path: str, format: str = 'yaml'):
        """
        Export current weights to file.

        Parameters
        ----------
        path : str
            Output file path
        format : str
            Output format ('yaml', 'json')
        """
        export_data = {
            'weights': self.current_weights,
            'constraints': {
                domain: {
                    'initial': c.initial_weight,
                    'min': c.min_weight,
                    'max': c.max_weight,
                    'evidence': c.evidence_strength,
                    'references': c.literature_references,
                    'notes': c.notes
                }
                for domain, c in self.constraints.to_dict().items()
            },
            'optimization_history': {
                'iterations': len(self.weight_history),
                'metric': self.optimization_metric,
                'metric_history': self.metric_history
            }
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if format == 'yaml':
            with open(path, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False)
        elif format == 'json':
            with open(path, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Weights exported to {path}")


def load_adaptive_weights(path: str) -> Dict[str, float]:
    """
    Load adaptive weights from file.

    Parameters
    ----------
    path : str
        Path to weights file (YAML or JSON)

    Returns
    -------
    dict
        Weight dictionary
    """
    path_obj = Path(path)

    if path_obj.suffix in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    elif path_obj.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unknown file format: {path_obj.suffix}")

    return data.get('weights', {})


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize with literature-based constraints
    literature_weights = LiteratureBasedWeights()

    print("\n=== Literature-Based Initial Weights ===")
    for domain, constraints in literature_weights.to_dict().items():
        print(f"\n{domain.upper()}:")
        print(f"  Weight: {constraints.initial_weight:.3f} "
              f"[{constraints.min_weight:.3f}, {constraints.max_weight:.3f}]")
        print(f"  Evidence: {constraints.evidence_strength}")
        if constraints.literature_references:
            print(f"  References: {constraints.literature_references[0]}")

    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 200

    domain_features = {
        'genetic': pd.DataFrame(np.random.randn(n_samples, 50)),
        'metabolomic': pd.DataFrame(np.random.randn(n_samples, 30)),
        'autonomic': pd.DataFrame(np.random.randn(n_samples, 10)),
        'circadian': pd.DataFrame(np.random.randn(n_samples, 5)),
        'environmental': pd.DataFrame(np.random.randn(n_samples, 15))
    }

    cluster_labels = np.random.choice([0, 1, 2, 3], n_samples)

    # Initialize optimizer
    optimizer = AdaptiveWeightOptimizer(
        constraints=literature_weights,
        optimization_metric='silhouette'
    )

    # Run optimization
    optimized_weights = optimizer.optimize_weights(
        domain_features,
        cluster_labels=cluster_labels
    )

    print("\n=== Optimized Weights ===")
    for domain, weight in optimized_weights.items():
        initial = literature_weights.to_dict()[domain].initial_weight
        change = ((weight - initial) / initial) * 100
        print(f"{domain}: {weight:.3f} (initial: {initial:.3f}, change: {change:+.1f}%)")

    # Export
    optimizer.export_weights('optimized_weights.yaml')
    print("\nWeights exported to optimized_weights.yaml")
