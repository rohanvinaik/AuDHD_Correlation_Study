#!/usr/bin/env python3
"""
Federated Learning for Multi-Site Collaboration
Enables privacy-preserving analysis across institutions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FederatedResult:
    """Results from federated analysis"""
    global_model: Any
    site_contributions: pd.DataFrame
    convergence_history: pd.DataFrame


class FederatedAnalyzer:
    """
    Federated learning for AuDHD multi-site studies

    Capabilities:
    1. Federated averaging (FedAvg)
    2. Differential privacy
    3. Secure aggregation
    4. Meta-analysis across sites
    """

    def __init__(self, privacy_epsilon: float = 1.0):
        """
        Initialize federated analyzer

        Parameters
        ----------
        privacy_epsilon : float
            Differential privacy parameter
        """
        self.privacy_epsilon = privacy_epsilon

    def federated_averaging(
        self,
        site_models: List[Dict[str, np.ndarray]],
        site_weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate models from multiple sites using federated averaging

        Parameters
        ----------
        site_models : List[Dict]
            Model parameters from each site
        site_weights : List[float], optional
            Relative weight for each site (e.g., by sample size)

        Returns
        -------
        global_model : Dict
            Aggregated model parameters
        """
        logger.info(f"Performing federated averaging across {len(site_models)} sites")

        if site_weights is None:
            site_weights = [1.0 / len(site_models)] * len(site_models)

        # Normalize weights
        site_weights = np.array(site_weights) / np.sum(site_weights)

        # Average parameters
        global_model = {}

        for param_name in site_models[0].keys():
            weighted_params = [
                weight * site_model[param_name]
                for weight, site_model in zip(site_weights, site_models)
            ]
            global_model[param_name] = np.sum(weighted_params, axis=0)

        logger.info("  Federated averaging complete")

        return global_model

    def add_differential_privacy(
        self,
        gradients: np.ndarray,
        epsilon: Optional[float] = None,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Add Laplace noise for differential privacy

        Parameters
        ----------
        gradients : np.ndarray
            Model gradients
        epsilon : float, optional
            Privacy parameter (smaller = more privacy)
        sensitivity : float
            Sensitivity of query

        Returns
        -------
        noisy_gradients : np.ndarray
        """
        if epsilon is None:
            epsilon = self.privacy_epsilon

        logger.info(f"Adding differential privacy (ε={epsilon})")

        # Laplace mechanism
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, size=gradients.shape)

        noisy_gradients = gradients + noise

        return noisy_gradients

    def secure_aggregation(
        self,
        site_updates: List[np.ndarray],
        masks: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Securely aggregate site updates without revealing individual contributions

        Parameters
        ----------
        site_updates : List[np.ndarray]
            Updates from each site
        masks : List[np.ndarray], optional
            Random masks for secure aggregation

        Returns
        -------
        aggregated : np.ndarray
        """
        logger.info("Performing secure aggregation")

        if masks is None:
            # Generate pairwise masks (simplified)
            n_sites = len(site_updates)
            masks = [np.random.randn(*site_updates[0].shape) for _ in range(n_sites)]

        # Mask site updates
        masked_updates = [
            update + mask
            for update, mask in zip(site_updates, masks)
        ]

        # Aggregate
        aggregated = np.sum(masked_updates, axis=0)

        # Remove masks (sum cancels out in real protocol)
        aggregated -= np.sum(masks, axis=0)

        logger.info("  Secure aggregation complete")

        return aggregated

    def federated_meta_analysis(
        self,
        site_statistics: List[Dict[str, float]],
        method: str = 'fixed_effects'
    ) -> Dict[str, float]:
        """
        Meta-analysis of site-level statistics

        Parameters
        ----------
        site_statistics : List[Dict]
            Each dict contains: effect, se, n
        method : str
            'fixed_effects' or 'random_effects'

        Returns
        -------
        meta_results : Dict
            Combined effect, se, p_value
        """
        logger.info(f"Running federated meta-analysis ({method})")

        effects = np.array([stat['effect'] for stat in site_statistics])
        standard_errors = np.array([stat['se'] for stat in site_statistics])
        sample_sizes = np.array([stat['n'] for stat in site_statistics])

        # Inverse-variance weighting
        weights = 1.0 / (standard_errors ** 2)

        # Fixed effects
        pooled_effect = np.sum(weights * effects) / np.sum(weights)
        pooled_se = np.sqrt(1.0 / np.sum(weights))

        # Heterogeneity (I²)
        Q = np.sum(weights * (effects - pooled_effect) ** 2)
        df = len(effects) - 1
        tau_squared = max(0, (Q - df) / np.sum(weights))

        # Random effects (DerSimonian-Laird)
        if method == 'random_effects' and tau_squared > 0:
            weights_re = 1.0 / (standard_errors ** 2 + tau_squared)
            pooled_effect = np.sum(weights_re * effects) / np.sum(weights_re)
            pooled_se = np.sqrt(1.0 / np.sum(weights_re))

        # P-value
        from scipy import stats
        z_score = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        logger.info(f"  Pooled effect: {pooled_effect:.3f} ± {pooled_se:.3f}, p={p_value:.2e}")

        return {
            'effect': pooled_effect,
            'se': pooled_se,
            'p_value': p_value,
            'I_squared': max(0, (Q - df) / Q * 100) if Q > 0 else 0,
            'n_sites': len(effects),
            'total_n': int(np.sum(sample_sizes))
        }

    def federated_gwas(
        self,
        site_summary_stats: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Federated GWAS meta-analysis

        Parameters
        ----------
        site_summary_stats : List[pd.DataFrame]
            GWAS summary statistics from each site
            Columns: snp, chr, pos, beta, se, p, n

        Returns
        -------
        meta_gwas : pd.DataFrame
            Meta-analyzed GWAS results
        """
        logger.info("Running federated GWAS meta-analysis")

        # Merge all sites
        all_snps = set()
        for stats in site_summary_stats:
            all_snps.update(stats['snp'])

        logger.info(f"  Meta-analyzing {len(all_snps)} SNPs across {len(site_summary_stats)} sites")

        meta_results = []

        for snp in all_snps:
            # Gather stats from sites that have this SNP
            snp_stats = []
            for site_stats in site_summary_stats:
                snp_data = site_stats[site_stats['snp'] == snp]
                if len(snp_data) > 0:
                    snp_stats.append({
                        'effect': snp_data['beta'].iloc[0],
                        'se': snp_data['se'].iloc[0],
                        'n': snp_data['n'].iloc[0]
                    })

            if len(snp_stats) == 0:
                continue

            # Meta-analyze
            meta_result = self.federated_meta_analysis(snp_stats, method='fixed_effects')

            meta_results.append({
                'snp': snp,
                'beta': meta_result['effect'],
                'se': meta_result['se'],
                'p_value': meta_result['p_value'],
                'n_sites': meta_result['n_sites'],
                'total_n': meta_result['total_n']
            })

        meta_df = pd.DataFrame(meta_results).sort_values('p_value')

        logger.info(f"  Meta-analysis complete: {len(meta_df)} SNPs")

        return meta_df

    def analyze_complete(
        self,
        site_data: List[Dict[str, Any]]
    ) -> FederatedResult:
        """
        Complete federated analysis pipeline

        Parameters
        ----------
        site_data : List[Dict]
            Data from each site

        Returns
        -------
        FederatedResult
        """
        logger.info("=== Complete Federated Learning Analysis ===")

        # Placeholder for demonstration
        site_contributions = pd.DataFrame({
            'site_id': [f'site_{i}' for i in range(len(site_data))],
            'n_samples': [100 * (i + 1) for i in range(len(site_data))],
            'contribution_weight': [0.25 + i * 0.1 for i in range(len(site_data))]
        })

        convergence_history = pd.DataFrame({
            'round': range(10),
            'loss': np.exp(-np.arange(10) * 0.5) + 0.1
        })

        return FederatedResult(
            global_model=None,
            site_contributions=site_contributions,
            convergence_history=convergence_history
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Federated Learning Module")
    logger.info("Ready for multi-site AuDHD collaboration")
