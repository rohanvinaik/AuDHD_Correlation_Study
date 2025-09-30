"""
Iterative Refinement System for Multi-Modal Analysis

Automatically identifies and removes non-discriminative data sources,
re-weights remaining domains, and iteratively improves subtype separation.

Features:
- Discriminative power metrics per domain (silhouette, classification importance)
- Automatic domain pruning based on contribution thresholds
- Weight re-normalization after pruning
- Convergence detection (clustering quality, weight stability)
- Full provenance tracking across iterations
- Visualization of refinement process

Author: Claude Code
Date: 2025-09-30
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings

from .adaptive_weights import (
    LiteratureBasedWeights,
    AdaptiveWeightOptimizer,
    WeightConstraints
)

# Import enhanced discriminative analysis methods
from .iterative_refinement_enhanced import (
    evaluate_subtype_specific_discriminative_power,
    identify_discriminative_features_within_domain,
    calculate_nonlinear_discriminative_power,
    evaluate_domain_interactions,
    bootstrap_discriminative_power,
    calculate_subtype_homogeneity
)

logger = logging.getLogger(__name__)


@dataclass
class DomainDiscriminativePower:
    """Measures of how much a domain contributes to subtype discrimination."""
    domain: str
    silhouette_contribution: float  # Silhouette score with only this domain
    classification_importance: float  # Random Forest feature importance
    between_cluster_variance: float  # F-statistic from ANOVA
    correlation_with_clustering: float  # How well domain predicts cluster labels
    composite_score: float  # Weighted combination of above metrics
    is_discriminative: bool  # Above threshold?

    # Enhanced metrics (optional, computed if enabled)
    subtype_specific_scores: Optional[Dict[int, float]] = None  # Per-subtype AUC scores
    nonlinear_score: Optional[float] = None  # Non-linear discrimination via kernel methods
    confidence_interval: Optional[Tuple[float, float]] = None  # Bootstrap CI (lower, upper)
    top_features: Optional[List[str]] = None  # Most discriminative features within domain
    interaction_partners: Optional[List[Tuple[str, float]]] = None  # Synergistic domains

    def __repr__(self):
        return (f"DomainDiscriminativePower(domain={self.domain}, "
                f"composite={self.composite_score:.3f}, "
                f"discriminative={self.is_discriminative})")


@dataclass
class IterationResult:
    """Results from one iteration of refinement."""
    iteration: int
    active_domains: List[str]
    removed_domains: List[str]
    weights: Dict[str, float]
    cluster_labels: np.ndarray

    # Quality metrics
    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float
    n_clusters: int

    # Discriminative power by domain
    domain_powers: Dict[str, DomainDiscriminativePower]

    # Convergence indicators
    weight_change: float  # Max weight change from previous iteration
    quality_change: float  # Change in silhouette score
    domains_removed_this_iter: int

    def __repr__(self):
        return (f"Iteration {self.iteration}: "
                f"{len(self.active_domains)} domains, "
                f"silhouette={self.silhouette_score:.3f}, "
                f"removed={self.domains_removed_this_iter}")


class IterativeRefinementEngine:
    """
    Multi-stage iterative refinement system.

    Process:
    1. Initial integration with all domains
    2. Evaluate discriminative power of each domain
    3. Remove weakest domains below threshold
    4. Re-optimize weights for remaining domains
    5. Re-cluster with updated weights
    6. Repeat until convergence

    Convergence criteria:
    - No more domains to remove
    - Quality improvement < threshold
    - Max iterations reached
    - Minimum number of domains reached
    """

    def __init__(
        self,
        literature_weights: Optional[LiteratureBasedWeights] = None,
        discriminative_threshold: float = 0.3,
        min_domains: int = 3,
        max_iterations: int = 10,
        convergence_threshold: float = 0.01,
        optimization_metric: str = 'silhouette',
        preserve_strong_evidence_domains: bool = True,
        verbose: bool = True,
        # Enhanced discriminative analysis options
        use_subtype_specific: bool = True,
        use_nonlinear: bool = True,
        use_bootstrap_ci: bool = True,
        use_feature_selection: bool = True,
        use_interaction_effects: bool = True,
        n_bootstrap: int = 100,
        top_features_per_domain: int = 10
    ):
        """
        Initialize iterative refinement engine.

        Parameters
        ----------
        literature_weights : LiteratureBasedWeights, optional
            Literature-based weight constraints. If None, uses defaults.
        discriminative_threshold : float, default=0.3
            Minimum composite discriminative score to keep a domain (0-1 scale)
        min_domains : int, default=3
            Minimum number of domains to retain
        max_iterations : int, default=10
            Maximum refinement iterations
        convergence_threshold : float, default=0.01
            Stop if silhouette improvement < this threshold
        optimization_metric : str, default='silhouette'
            Metric to optimize ('silhouette', 'davies_bouldin', 'calinski_harabasz')
        preserve_strong_evidence_domains : bool, default=True
            Never remove domains with 'strong' evidence strength
        verbose : bool, default=True
            Print progress information
        use_subtype_specific : bool, default=True
            Evaluate per-subtype discriminative power (one-vs-rest AUC)
        use_nonlinear : bool, default=True
            Include non-linear discrimination via kernel methods
        use_bootstrap_ci : bool, default=True
            Calculate bootstrap confidence intervals for stability
        use_feature_selection : bool, default=True
            Identify most discriminative features within each domain
        use_interaction_effects : bool, default=True
            Detect synergistic domain pairs
        n_bootstrap : int, default=100
            Number of bootstrap samples for CI estimation
        top_features_per_domain : int, default=10
            Number of top features to identify per domain
        """
        self.literature_weights = literature_weights or LiteratureBasedWeights()
        self.discriminative_threshold = discriminative_threshold
        self.min_domains = min_domains
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.optimization_metric = optimization_metric
        self.preserve_strong_evidence = preserve_strong_evidence_domains
        self.verbose = verbose

        # Enhanced analysis options
        self.use_subtype_specific = use_subtype_specific
        self.use_nonlinear = use_nonlinear
        self.use_bootstrap_ci = use_bootstrap_ci
        self.use_feature_selection = use_feature_selection
        self.use_interaction_effects = use_interaction_effects
        self.n_bootstrap = n_bootstrap
        self.top_features_per_domain = top_features_per_domain

        # Track results across iterations
        self.iteration_results: List[IterationResult] = []
        self.convergence_reason: Optional[str] = None
        self.domain_interactions: Optional[List[Tuple[str, str, float]]] = None

        logger.info(f"Initialized IterativeRefinementEngine:")
        logger.info(f"  Discriminative threshold: {discriminative_threshold}")
        logger.info(f"  Min domains: {min_domains}")
        logger.info(f"  Max iterations: {max_iterations}")
        logger.info(f"  Preserve strong evidence: {preserve_strong_evidence}")
        logger.info(f"  Enhanced analysis: subtype={use_subtype_specific}, "
                   f"nonlinear={use_nonlinear}, bootstrap={use_bootstrap_ci}, "
                   f"feature_selection={use_feature_selection}, interactions={use_interaction_effects}")

    def evaluate_domain_discriminative_power(
        self,
        domain_features: Dict[str, pd.DataFrame],
        cluster_labels: np.ndarray,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, DomainDiscriminativePower]:
        """
        Evaluate how much each domain contributes to cluster discrimination.

        Uses multiple metrics:
        1. Silhouette score with only this domain's features
        2. Random Forest feature importance for predicting clusters
        3. Between-cluster variance (ANOVA F-statistic)
        4. Correlation between domain features and cluster assignments

        Parameters
        ----------
        domain_features : dict
            Domain name -> feature DataFrame
        cluster_labels : np.ndarray
            Current cluster assignments
        current_weights : dict, optional
            Current domain weights for reference

        Returns
        -------
        dict
            Domain name -> DomainDiscriminativePower
        """
        n_clusters = len(np.unique(cluster_labels))
        domain_powers = {}

        # Get all features concatenated for global context
        all_features = []
        domain_slices = {}
        start_idx = 0
        for domain, df in domain_features.items():
            if df is not None and not df.empty:
                all_features.append(df)
                domain_slices[domain] = (start_idx, start_idx + df.shape[1])
                start_idx += df.shape[1]

        if not all_features:
            logger.warning("No features available for discriminative power evaluation")
            return {}

        all_features_concat = pd.concat(all_features, axis=1)
        all_features_scaled = StandardScaler().fit_transform(all_features_concat)

        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(all_features_scaled, cluster_labels)
        feature_importances = rf.feature_importances_

        for domain, df in domain_features.items():
            if df is None or df.empty:
                logger.debug(f"Skipping empty domain: {domain}")
                continue

            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(df)

            # 1. Silhouette with only this domain
            try:
                if features_scaled.shape[1] > 0 and len(np.unique(cluster_labels)) > 1:
                    sil = silhouette_score(features_scaled, cluster_labels)
                else:
                    sil = 0.0
            except Exception as e:
                logger.debug(f"Silhouette calculation failed for {domain}: {e}")
                sil = 0.0

            # 2. Random Forest feature importance (mean for domain)
            if domain in domain_slices:
                start, end = domain_slices[domain]
                domain_importance = feature_importances[start:end].mean()
            else:
                domain_importance = 0.0

            # 3. Between-cluster variance (ANOVA)
            try:
                # Compute F-statistic for each feature, take mean
                f_stats = []
                for col_idx in range(features_scaled.shape[1]):
                    feature = features_scaled[:, col_idx]
                    groups = [feature[cluster_labels == k] for k in range(n_clusters)]
                    # Simple F-statistic: between-group variance / within-group variance
                    group_means = [np.mean(g) for g in groups if len(g) > 0]
                    overall_mean = np.mean(feature)

                    if len(group_means) > 1:
                        between_var = np.sum([len(g) * (np.mean(g) - overall_mean)**2
                                             for g in groups if len(g) > 0]) / (n_clusters - 1)
                        within_var = np.sum([np.sum((g - np.mean(g))**2)
                                            for g in groups if len(g) > 0]) / (len(feature) - n_clusters)
                        if within_var > 0:
                            f_stats.append(between_var / within_var)

                between_cluster_var = np.mean(f_stats) if f_stats else 0.0
            except Exception as e:
                logger.debug(f"ANOVA calculation failed for {domain}: {e}")
                between_cluster_var = 0.0

            # 4. Correlation with clustering (using first PC if multiple features)
            try:
                from sklearn.decomposition import PCA
                if features_scaled.shape[1] > 1:
                    pca = PCA(n_components=1)
                    pc1 = pca.fit_transform(features_scaled).flatten()
                else:
                    pc1 = features_scaled.flatten()

                # Correlation between PC1 and cluster labels
                correlation = np.abs(np.corrcoef(pc1, cluster_labels)[0, 1])
            except Exception as e:
                logger.debug(f"Correlation calculation failed for {domain}: {e}")
                correlation = 0.0

            # Composite score (weighted average of metrics, normalized to 0-1)
            # Normalize metrics
            sil_norm = (sil + 1) / 2  # Silhouette is [-1, 1], map to [0, 1]
            rf_norm = domain_importance  # Already 0-1
            anova_norm = np.clip(between_cluster_var / 10, 0, 1)  # F-stats can be large
            corr_norm = correlation  # Already 0-1

            # Base composite (linear metrics)
            composite_base = (
                0.35 * sil_norm +
                0.30 * rf_norm +
                0.20 * anova_norm +
                0.15 * corr_norm
            )

            composite = composite_base  # Will be enhanced below if options enabled

            # ENHANCEMENT 1: Add non-linear discrimination score
            nonlinear_score = None
            if self.use_nonlinear:
                try:
                    nonlinear_score = calculate_nonlinear_discriminative_power(
                        df, cluster_labels
                    )
                    # Weight non-linear component at 20%, reduce linear to 80%
                    composite = 0.8 * composite_base + 0.2 * nonlinear_score
                    logger.debug(f"  {domain} nonlinear score: {nonlinear_score:.3f}")
                except Exception as e:
                    logger.debug(f"Nonlinear scoring failed for {domain}: {e}")

            # Check if discriminative
            is_discriminative = composite >= self.discriminative_threshold

            # Check if protected by strong evidence
            if self.preserve_strong_evidence:
                constraints = getattr(self.literature_weights, domain, None)
                if constraints and constraints.evidence_strength == 'strong':
                    is_discriminative = True  # Never remove strong evidence domains
                    logger.debug(f"Domain {domain} protected by strong evidence")

            # ENHANCEMENT 2: Subtype-specific scores (computed globally after loop)
            # ENHANCEMENT 3: Bootstrap confidence intervals
            confidence_interval = None
            if self.use_bootstrap_ci:
                try:
                    _, lower_ci, upper_ci = bootstrap_discriminative_power(
                        df, cluster_labels, n_bootstrap=self.n_bootstrap
                    )
                    confidence_interval = (lower_ci, upper_ci)
                    logger.debug(f"  {domain} CI: [{lower_ci:.3f}, {upper_ci:.3f}]")
                except Exception as e:
                    logger.debug(f"Bootstrap failed for {domain}: {e}")

            # ENHANCEMENT 4: Feature-level selection
            top_features = None
            if self.use_feature_selection:
                try:
                    top_features = identify_discriminative_features_within_domain(
                        df, cluster_labels, top_k=self.top_features_per_domain
                    )
                    logger.debug(f"  {domain} top features: {top_features[:3]}...")
                except Exception as e:
                    logger.debug(f"Feature selection failed for {domain}: {e}")

            domain_powers[domain] = DomainDiscriminativePower(
                domain=domain,
                silhouette_contribution=sil,
                classification_importance=domain_importance,
                between_cluster_variance=between_cluster_var,
                correlation_with_clustering=correlation,
                composite_score=composite,
                is_discriminative=is_discriminative,
                nonlinear_score=nonlinear_score,
                confidence_interval=confidence_interval,
                top_features=top_features
            )

            if self.verbose:
                logger.info(f"Domain {domain:15s}: composite={composite:.3f}, "
                          f"sil={sil:.2f}, rf_imp={domain_importance:.2f}, "
                          f"discriminative={is_discriminative}")

        # ENHANCEMENT 5: Subtype-specific discriminative power (global analysis)
        if self.use_subtype_specific:
            logger.info("Evaluating subtype-specific discriminative power...")
            try:
                subtype_scores = evaluate_subtype_specific_discriminative_power(
                    domain_features, cluster_labels
                )
                # Add subtype scores to each domain
                for domain, scores_dict in subtype_scores.items():
                    if domain in domain_powers:
                        domain_powers[domain].subtype_specific_scores = scores_dict
                        # Log best subtype for this domain
                        if scores_dict:
                            best_subtype = max(scores_dict.items(), key=lambda x: x[1])
                            logger.info(f"  {domain:15s}: best subtype {best_subtype[0]} "
                                      f"(AUC={best_subtype[1]:.3f})")
            except Exception as e:
                logger.warning(f"Subtype-specific analysis failed: {e}")

        # ENHANCEMENT 6: Domain interaction effects (global analysis)
        if self.use_interaction_effects:
            logger.info("Evaluating domain interaction effects...")
            try:
                interactions = evaluate_domain_interactions(
                    domain_features, cluster_labels, max_pairs=10
                )
                # Store for use in pruning logic
                self.domain_interactions = interactions
                # Add interaction partners to each domain
                for d1, d2, strength in interactions:
                    if d1 in domain_powers:
                        if domain_powers[d1].interaction_partners is None:
                            domain_powers[d1].interaction_partners = []
                        domain_powers[d1].interaction_partners.append((d2, strength))
                    if d2 in domain_powers:
                        if domain_powers[d2].interaction_partners is None:
                            domain_powers[d2].interaction_partners = []
                        domain_powers[d2].interaction_partners.append((d1, strength))

                if interactions:
                    logger.info(f"Found {len(interactions)} strong domain interactions:")
                    for d1, d2, strength in interactions[:5]:  # Top 5
                        logger.info(f"  {d1} ↔ {d2}: {strength:.3f}")
            except Exception as e:
                logger.warning(f"Interaction analysis failed: {e}")

        return domain_powers

    def prune_domains(
        self,
        domain_powers: Dict[str, DomainDiscriminativePower],
        current_domains: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Identify domains to remove based on discriminative power.

        Parameters
        ----------
        domain_powers : dict
            Discriminative power for each domain
        current_domains : list
            Currently active domains

        Returns
        -------
        keep_domains : list
            Domains to keep
        remove_domains : list
            Domains to remove
        """
        # Check if we're at minimum
        if len(current_domains) <= self.min_domains:
            logger.info(f"At minimum domains ({self.min_domains}), no pruning")
            return current_domains, []

        # Sort domains by composite score
        sorted_domains = sorted(
            [(d, domain_powers[d].composite_score) for d in current_domains if d in domain_powers],
            key=lambda x: x[1]
        )

        # Find non-discriminative domains
        remove_candidates = [
            d for d, score in sorted_domains
            if not domain_powers[d].is_discriminative
        ]

        # Don't remove more than would take us below min_domains
        max_removable = len(current_domains) - self.min_domains
        remove_domains = remove_candidates[:max_removable]

        # ENHANCEMENT: Check for strong interactions and preserve domains with synergy
        if self.use_interaction_effects and self.domain_interactions and remove_domains:
            logger.info("Checking for strong interactions before pruning...")
            strong_interaction_domains = set()

            for d1, d2, strength in self.domain_interactions:
                if strength > 0.1:  # Threshold for strong interaction
                    if d1 in remove_domains:
                        strong_interaction_domains.add(d1)
                        logger.info(f"  Preserving {d1} due to interaction with {d2} (strength={strength:.3f})")
                    if d2 in remove_domains:
                        strong_interaction_domains.add(d2)
                        logger.info(f"  Preserving {d2} due to interaction with {d1} (strength={strength:.3f})")

            # Update removal list to exclude domains with strong interactions
            if strong_interaction_domains:
                final_remove = [d for d in remove_domains if d not in strong_interaction_domains]
                logger.info(f"Protected {len(strong_interaction_domains)} domains due to interactions")
                remove_domains = final_remove

        keep_domains = [d for d in current_domains if d not in remove_domains]

        if remove_domains:
            logger.info(f"Pruning {len(remove_domains)} non-discriminative domains: {remove_domains}")
        else:
            logger.info("No domains to prune this iteration")

        return keep_domains, remove_domains

    def renormalize_weights(
        self,
        current_weights: Dict[str, float],
        keep_domains: List[str]
    ) -> Dict[str, float]:
        """
        Re-normalize weights after removing domains.

        Parameters
        ----------
        current_weights : dict
            Current domain weights
        keep_domains : list
            Domains to keep

        Returns
        -------
        dict
            Re-normalized weights summing to 1.0
        """
        # Keep only active domain weights
        new_weights = {d: current_weights[d] for d in keep_domains if d in current_weights}

        # Re-normalize to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {d: w / total for d, w in new_weights.items()}

        logger.info("Re-normalized weights:")
        for domain, weight in sorted(new_weights.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {domain:15s}: {weight:.3f}")

        return new_weights

    def run_iteration(
        self,
        domain_features: Dict[str, pd.DataFrame],
        clustering_function,
        active_domains: List[str],
        current_weights: Dict[str, float],
        iteration: int,
        previous_result: Optional[IterationResult] = None
    ) -> IterationResult:
        """
        Run one iteration of refinement.

        Parameters
        ----------
        domain_features : dict
            All domain features (will filter to active_domains)
        clustering_function : callable
            Function that takes (features_dict, weights) and returns cluster_labels
        active_domains : list
            Currently active domains
        current_weights : dict
            Current domain weights
        iteration : int
            Iteration number
        previous_result : IterationResult, optional
            Result from previous iteration for comparison

        Returns
        -------
        IterationResult
            Results from this iteration
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*80}")
        logger.info(f"Active domains: {active_domains}")

        # Filter to active domains
        active_features = {d: domain_features[d] for d in active_domains if d in domain_features}

        # Run clustering
        cluster_labels = clustering_function(active_features, current_weights)
        n_clusters = len(np.unique(cluster_labels))

        # Compute quality metrics
        all_features = pd.concat([df for df in active_features.values()], axis=1)
        features_scaled = StandardScaler().fit_transform(all_features)

        sil_score = silhouette_score(features_scaled, cluster_labels)
        db_score = davies_bouldin_score(features_scaled, cluster_labels)
        ch_score = calinski_harabasz_score(features_scaled, cluster_labels)

        logger.info(f"Clustering quality:")
        logger.info(f"  Silhouette: {sil_score:.3f}")
        logger.info(f"  Davies-Bouldin: {db_score:.3f}")
        logger.info(f"  Calinski-Harabasz: {ch_score:.1f}")
        logger.info(f"  N clusters: {n_clusters}")

        # Evaluate discriminative power
        domain_powers = self.evaluate_domain_discriminative_power(
            active_features, cluster_labels, current_weights
        )

        # Determine what to prune
        keep_domains, remove_domains = self.prune_domains(domain_powers, active_domains)

        # Calculate convergence metrics
        if previous_result:
            weight_change = max(
                abs(current_weights.get(d, 0) - previous_result.weights.get(d, 0))
                for d in set(current_weights.keys()) | set(previous_result.weights.keys())
            )
            quality_change = sil_score - previous_result.silhouette_score
        else:
            weight_change = 1.0  # First iteration
            quality_change = sil_score

        result = IterationResult(
            iteration=iteration,
            active_domains=active_domains,
            removed_domains=remove_domains,
            weights=current_weights.copy(),
            cluster_labels=cluster_labels.copy(),
            silhouette_score=sil_score,
            davies_bouldin_score=db_score,
            calinski_harabasz_score=ch_score,
            n_clusters=n_clusters,
            domain_powers=domain_powers,
            weight_change=weight_change,
            quality_change=quality_change,
            domains_removed_this_iter=len(remove_domains)
        )

        return result

    def refine(
        self,
        domain_features: Dict[str, pd.DataFrame],
        clustering_function,
        weight_optimizer: Optional[AdaptiveWeightOptimizer] = None
    ) -> Dict[str, Any]:
        """
        Run full iterative refinement process.

        Parameters
        ----------
        domain_features : dict
            Domain name -> feature DataFrame
        clustering_function : callable
            Function(features_dict, weights) -> cluster_labels
        weight_optimizer : AdaptiveWeightOptimizer, optional
            If provided, will optimize weights after each pruning

        Returns
        -------
        dict
            Results with:
            - 'final_domains': List of retained domains
            - 'final_weights': Final domain weights
            - 'final_labels': Final cluster assignments
            - 'iterations': List of IterationResult objects
            - 'convergence_reason': Why refinement stopped
            - 'improvement': Improvement in silhouette from initial to final
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING ITERATIVE REFINEMENT")
        logger.info("="*80)

        # Initialize with all domains
        active_domains = list(domain_features.keys())
        current_weights = self.literature_weights.get_initial_weights()

        # Filter to available domains
        active_domains = [d for d in active_domains if d in current_weights]
        current_weights = {d: w for d, w in current_weights.items() if d in active_domains}

        # Normalize
        total = sum(current_weights.values())
        current_weights = {d: w / total for d, w in current_weights.items()}

        logger.info(f"Starting with {len(active_domains)} domains")

        # Iteration loop
        for iteration in range(self.max_iterations):
            # Run iteration
            previous = self.iteration_results[-1] if self.iteration_results else None
            result = self.run_iteration(
                domain_features=domain_features,
                clustering_function=clustering_function,
                active_domains=active_domains,
                current_weights=current_weights,
                iteration=iteration,
                previous_result=previous
            )

            self.iteration_results.append(result)

            # Check convergence
            if len(result.removed_domains) == 0:
                self.convergence_reason = "No more domains to remove"
                logger.info(f"\nCONVERGED: {self.convergence_reason}")
                break

            if len(result.active_domains) <= self.min_domains:
                self.convergence_reason = f"Reached minimum domains ({self.min_domains})"
                logger.info(f"\nCONVERGED: {self.convergence_reason}")
                break

            if iteration > 0 and result.quality_change < self.convergence_threshold:
                self.convergence_reason = f"Quality improvement < {self.convergence_threshold}"
                logger.info(f"\nCONVERGED: {self.convergence_reason}")
                break

            # Update for next iteration
            active_domains = [d for d in active_domains if d not in result.removed_domains]
            current_weights = self.renormalize_weights(current_weights, active_domains)

            # Optionally re-optimize weights
            if weight_optimizer:
                logger.info("Re-optimizing weights with remaining domains...")
                active_features = {d: domain_features[d] for d in active_domains}
                try:
                    optimized = weight_optimizer.optimize_weights(
                        domain_features=active_features,
                        cluster_labels=result.cluster_labels
                    )
                    current_weights = optimized
                    logger.info("Weight optimization successful")
                except Exception as e:
                    logger.warning(f"Weight optimization failed: {e}, using re-normalized weights")

        else:
            self.convergence_reason = f"Reached maximum iterations ({self.max_iterations})"
            logger.info(f"\nSTOPPED: {self.convergence_reason}")

        # Final results
        final_result = self.iteration_results[-1]
        initial_result = self.iteration_results[0]
        improvement = final_result.silhouette_score - initial_result.silhouette_score

        logger.info("\n" + "="*80)
        logger.info("REFINEMENT COMPLETE")
        logger.info("="*80)
        logger.info(f"Iterations: {len(self.iteration_results)}")
        logger.info(f"Initial domains: {len(initial_result.active_domains)}")
        logger.info(f"Final domains: {len(final_result.active_domains)}")
        logger.info(f"Domains removed: {len(initial_result.active_domains) - len(final_result.active_domains)}")
        logger.info(f"Initial silhouette: {initial_result.silhouette_score:.3f}")
        logger.info(f"Final silhouette: {final_result.silhouette_score:.3f}")
        logger.info(f"Improvement: {improvement:+.3f}")
        logger.info(f"Convergence: {self.convergence_reason}")

        return {
            'final_domains': final_result.active_domains,
            'final_weights': final_result.weights,
            'final_labels': final_result.cluster_labels,
            'final_result': final_result,
            'iterations': self.iteration_results,
            'convergence_reason': self.convergence_reason,
            'improvement': improvement,
            'n_iterations': len(self.iteration_results)
        }

    def plot_refinement_progress(self, save_path: Optional[Path] = None):
        """
        Visualize the refinement process across iterations.

        Creates a multi-panel figure showing:
        - Silhouette score over iterations
        - Number of active domains
        - Weight evolution
        - Discriminative power by domain

        Parameters
        ----------
        save_path : Path, optional
            If provided, save figure to this path
        """
        if not self.iteration_results:
            logger.warning("No iteration results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Quality metrics over iterations
        iterations = [r.iteration for r in self.iteration_results]
        silhouette = [r.silhouette_score for r in self.iteration_results]
        db_scores = [r.davies_bouldin_score for r in self.iteration_results]

        ax1 = axes[0, 0]
        ax1.plot(iterations, silhouette, 'o-', label='Silhouette', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Silhouette Score', color='C0')
        ax1.tick_params(axis='y', labelcolor='C0')
        ax1.grid(alpha=0.3)

        ax1_twin = ax1.twinx()
        ax1_twin.plot(iterations, db_scores, 's-', color='C1', label='Davies-Bouldin', linewidth=2)
        ax1_twin.set_ylabel('Davies-Bouldin Score', color='C1')
        ax1_twin.tick_params(axis='y', labelcolor='C1')

        ax1.set_title('Clustering Quality Over Iterations')

        # Panel 2: Number of active domains
        n_domains = [len(r.active_domains) for r in self.iteration_results]
        n_removed = [r.domains_removed_this_iter for r in self.iteration_results]

        ax2 = axes[0, 1]
        ax2.plot(iterations, n_domains, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Number of Active Domains')
        ax2.set_title('Domain Pruning Progress')
        ax2.grid(alpha=0.3)

        # Annotate removed domains
        for i, (it, removed) in enumerate(zip(iterations, n_removed)):
            if removed > 0:
                ax2.annotate(f'-{removed}', xy=(it, n_domains[i]),
                           xytext=(5, -10), textcoords='offset points',
                           fontsize=9, color='red')

        # Panel 3: Weight evolution
        ax3 = axes[1, 0]

        # Track weights for each domain across iterations
        all_domains = set()
        for r in self.iteration_results:
            all_domains.update(r.weights.keys())

        for domain in sorted(all_domains):
            weights = [r.weights.get(domain, 0) for r in self.iteration_results]
            ax3.plot(iterations, weights, 'o-', label=domain, linewidth=2)

        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Weight')
        ax3.set_title('Domain Weight Evolution')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(alpha=0.3)

        # Panel 4: Final discriminative power
        ax4 = axes[1, 1]

        final_result = self.iteration_results[-1]
        domains = sorted(final_result.domain_powers.keys(),
                        key=lambda d: final_result.domain_powers[d].composite_score,
                        reverse=True)
        scores = [final_result.domain_powers[d].composite_score for d in domains]
        colors = ['green' if final_result.domain_powers[d].is_discriminative
                 else 'red' for d in domains]

        ax4.barh(range(len(domains)), scores, color=colors, alpha=0.7)
        ax4.set_yticks(range(len(domains)))
        ax4.set_yticklabels(domains, fontsize=9)
        ax4.set_xlabel('Composite Discriminative Score')
        ax4.set_title('Final Discriminative Power by Domain')
        ax4.axvline(self.discriminative_threshold, color='black',
                   linestyle='--', linewidth=2, label='Threshold')
        ax4.legend()
        ax4.grid(alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved refinement progress plot to {save_path}")

        return fig

    def export_results(self, output_dir: Path):
        """
        Export detailed refinement results.

        Creates:
        - refinement_summary.txt: Human-readable summary
        - refinement_iterations.csv: Metrics per iteration
        - refinement_weights.csv: Weight evolution
        - refinement_discriminative_power.csv: Discriminative scores
        - refinement_progress.png: Visualization

        Parameters
        ----------
        output_dir : Path
            Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary text
        summary_path = output_dir / 'refinement_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ITERATIVE REFINEMENT SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total iterations: {len(self.iteration_results)}\n")
            f.write(f"Convergence reason: {self.convergence_reason}\n\n")

            initial = self.iteration_results[0]
            final = self.iteration_results[-1]

            f.write(f"Initial state:\n")
            f.write(f"  Domains: {len(initial.active_domains)}\n")
            f.write(f"  Silhouette: {initial.silhouette_score:.3f}\n")
            f.write(f"  Clusters: {initial.n_clusters}\n\n")

            f.write(f"Final state:\n")
            f.write(f"  Domains: {len(final.active_domains)}\n")
            f.write(f"  Silhouette: {final.silhouette_score:.3f}\n")
            f.write(f"  Clusters: {final.n_clusters}\n")
            f.write(f"  Improvement: {final.silhouette_score - initial.silhouette_score:+.3f}\n\n")

            f.write(f"Removed domains ({len(initial.active_domains) - len(final.active_domains)}):\n")
            removed = set(initial.active_domains) - set(final.active_domains)
            for domain in sorted(removed):
                f.write(f"  - {domain}\n")

            f.write(f"\nFinal active domains ({len(final.active_domains)}):\n")
            for domain in sorted(final.active_domains):
                weight = final.weights.get(domain, 0)
                power = final.domain_powers.get(domain)
                if power:
                    f.write(f"  - {domain:15s}: weight={weight:.3f}, "
                          f"discriminative_score={power.composite_score:.3f}\n")

        logger.info(f"Saved summary to {summary_path}")

        # Iteration metrics CSV
        iter_data = []
        for r in self.iteration_results:
            iter_data.append({
                'iteration': r.iteration,
                'n_domains': len(r.active_domains),
                'silhouette': r.silhouette_score,
                'davies_bouldin': r.davies_bouldin_score,
                'calinski_harabasz': r.calinski_harabasz_score,
                'n_clusters': r.n_clusters,
                'weight_change': r.weight_change,
                'quality_change': r.quality_change,
                'domains_removed': r.domains_removed_this_iter
            })

        iter_df = pd.DataFrame(iter_data)
        iter_path = output_dir / 'refinement_iterations.csv'
        iter_df.to_csv(iter_path, index=False)
        logger.info(f"Saved iteration metrics to {iter_path}")

        # Weight evolution CSV
        weight_data = []
        for r in self.iteration_results:
            for domain, weight in r.weights.items():
                weight_data.append({
                    'iteration': r.iteration,
                    'domain': domain,
                    'weight': weight
                })

        weight_df = pd.DataFrame(weight_data)
        weight_path = output_dir / 'refinement_weights.csv'
        weight_df.to_csv(weight_path, index=False)
        logger.info(f"Saved weight evolution to {weight_path}")

        # Discriminative power CSV (final iteration)
        final = self.iteration_results[-1]
        power_data = []
        for domain, power in final.domain_powers.items():
            power_data.append({
                'domain': domain,
                'composite_score': power.composite_score,
                'silhouette_contribution': power.silhouette_contribution,
                'classification_importance': power.classification_importance,
                'between_cluster_variance': power.between_cluster_variance,
                'correlation_with_clustering': power.correlation_with_clustering,
                'is_discriminative': power.is_discriminative
            })

        power_df = pd.DataFrame(power_data)
        power_path = output_dir / 'refinement_discriminative_power.csv'
        power_df.to_csv(power_path, index=False)
        logger.info(f"Saved discriminative power to {power_path}")

        # Visualization
        plot_path = output_dir / 'refinement_progress.png'
        self.plot_refinement_progress(save_path=plot_path)

        logger.info(f"Exported all refinement results to {output_dir}")


def run_iterative_refinement(
    domain_features: Dict[str, pd.DataFrame],
    clustering_function,
    literature_weights: Optional[LiteratureBasedWeights] = None,
    discriminative_threshold: float = 0.3,
    min_domains: int = 3,
    max_iterations: int = 10,
    optimize_weights: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to run full iterative refinement.

    Parameters
    ----------
    domain_features : dict
        Domain name -> feature DataFrame
    clustering_function : callable
        Function(features_dict, weights) -> cluster_labels
    literature_weights : LiteratureBasedWeights, optional
        Literature-based constraints
    discriminative_threshold : float, default=0.3
        Minimum score to keep domain
    min_domains : int, default=3
        Minimum domains to retain
    max_iterations : int, default=10
        Maximum iterations
    optimize_weights : bool, default=True
        Re-optimize weights after each pruning
    output_dir : Path, optional
        Export results to this directory

    Returns
    -------
    dict
        Refinement results
    """
    # Create engine
    engine = IterativeRefinementEngine(
        literature_weights=literature_weights,
        discriminative_threshold=discriminative_threshold,
        min_domains=min_domains,
        max_iterations=max_iterations
    )

    # Create optimizer if requested
    weight_optimizer = None
    if optimize_weights:
        weight_optimizer = AdaptiveWeightOptimizer(
            constraints=literature_weights or LiteratureBasedWeights(),
            optimization_metric='silhouette'
        )

    # Run refinement
    results = engine.refine(
        domain_features=domain_features,
        clustering_function=clustering_function,
        weight_optimizer=weight_optimizer
    )

    # Export if requested
    if output_dir:
        engine.export_results(Path(output_dir))

    return results
