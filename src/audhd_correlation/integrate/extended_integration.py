#!/usr/bin/env python3
"""
Extended Multi-Omics Integration

Comprehensive integration framework incorporating all feature modalities:
- Genetic, metabolomic, microbiome (original)
- Autonomic, circadian, salivary (Prompt 2.1)
- Environmental, toxicants (Prompt 2.2)
- Interoception, sensory, voice (Prompt 2.3)

Implements hierarchical integration, time-aware adjustment, and multimodal networks.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeAwareAdjuster:
    """Adjust time-sensitive features using cosinor models"""

    def __init__(self):
        """Initialize time-aware adjuster"""
        self.fitted_models = {}

    def cosinor_model(self, t: np.ndarray, MESOR: float, amplitude: float, acrophase: float) -> np.ndarray:
        """
        Cosinor rhythmic model

        Args:
            t: Time in hours (0-24)
            MESOR: Midline-estimating statistic of rhythm
            amplitude: Peak-to-MESOR difference
            acrophase: Time of peak (radians)

        Returns:
            Predicted values
        """
        return MESOR + amplitude * np.cos(2 * np.pi * t / 24 - acrophase)

    def fit_cosinor(self, times: np.ndarray, values: np.ndarray) -> Dict[str, float]:
        """
        Fit cosinor model to data

        Args:
            times: Collection times (hours, 0-24)
            values: Measured values

        Returns:
            Dict with fitted parameters
        """
        # Remove NaN values
        mask = ~(np.isnan(times) | np.isnan(values))
        times = times[mask]
        values = values[mask]

        if len(times) < 3:
            return {'MESOR': np.nan, 'amplitude': np.nan, 'acrophase': np.nan, 'failed': True}

        try:
            # Initial guess
            MESOR_init = np.mean(values)
            amplitude_init = (np.max(values) - np.min(values)) / 2
            acrophase_init = 0

            # Fit
            popt, pcov = optimize.curve_fit(
                self.cosinor_model,
                times,
                values,
                p0=[MESOR_init, amplitude_init, acrophase_init],
                maxfev=5000
            )

            return {
                'MESOR': popt[0],
                'amplitude': popt[1],
                'acrophase': popt[2],
                'failed': False
            }

        except Exception as e:
            logger.warning(f"Cosinor fit failed: {e}")
            return {'MESOR': np.mean(values), 'amplitude': 0, 'acrophase': 0, 'failed': True}

    def adjust_for_collection_time(self,
                                   df: pd.DataFrame,
                                   time_col: str,
                                   time_sensitive_cols: List[str],
                                   standard_time: float = 9.0) -> pd.DataFrame:
        """
        Adjust circadian-sensitive features to standard collection time

        Args:
            df: DataFrame with features
            time_col: Column name for collection time (hours)
            time_sensitive_cols: Columns to adjust
            standard_time: Standard time to adjust to (hours, 0-24)

        Returns:
            Adjusted DataFrame
        """
        adjusted_df = df.copy()

        for col in time_sensitive_cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue

            times = df[time_col].values
            values = df[col].values

            # Fit cosinor model
            params = self.fit_cosinor(times, values)
            self.fitted_models[col] = params

            if params['failed']:
                logger.warning(f"Cosinor fit failed for {col}, no adjustment applied")
                continue

            # Calculate expected values at collection time and standard time
            expected_at_collection = self.cosinor_model(times, **{k: v for k, v in params.items() if k != 'failed'})
            expected_at_standard = self.cosinor_model(
                np.full_like(times, standard_time),
                **{k: v for k, v in params.items() if k != 'failed'}
            )

            # Adjust: add difference to bring to standard time
            adjustment = expected_at_standard - expected_at_collection
            adjusted_df[col] = values + adjustment

            logger.info(f"Adjusted {col} for collection time (standard time: {standard_time}h)")

        return adjusted_df


class HierarchicalIntegrator:
    """Multi-level hierarchical integration of heterogeneous data types"""

    def __init__(self, integration_levels: Dict[str, Dict], weights: Optional[Dict[str, float]] = None):
        """
        Initialize hierarchical integrator

        Args:
            integration_levels: Dict defining integration hierarchy
            weights: Optional weights for each modality
        """
        self.integration_levels = integration_levels
        self.weights = weights or {}
        self.level_results = {}

    def integrate_level(self,
                       data_dict: Dict[str, pd.DataFrame],
                       level_config: Dict) -> pd.DataFrame:
        """
        Integrate data at a single hierarchical level

        Args:
            data_dict: Dictionary of DataFrames for this level
            level_config: Configuration for this level

        Returns:
            Integrated DataFrame
        """
        method = level_config.get('method', 'PCA')
        n_factors = level_config.get('n_factors', 10)

        # Concatenate modalities at this level
        components = level_config.get('components', [])
        level_data = []
        feature_names = []

        for modality in components:
            if modality not in data_dict:
                logger.warning(f"Modality {modality} not found in data, skipping")
                continue

            df = data_dict[modality]
            if df.shape[0] == 0:
                continue

            # Apply modality weight if specified
            if modality in self.weights:
                df = df * np.sqrt(self.weights[modality])

            level_data.append(df.values)
            feature_names.extend([f"{modality}_{col}" for col in df.columns])

        if len(level_data) == 0:
            logger.error(f"No data available for level")
            return pd.DataFrame()

        # Concatenate
        X = np.hstack(level_data)

        # Apply integration method
        if method.upper() == 'PCA':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(n_factors, X.shape[1], X.shape[0]))
            X_integrated = pca.fit_transform(X)

            # Create column names
            cols = [f"level_factor_{i+1}" for i in range(X_integrated.shape[1])]
            result_df = pd.DataFrame(X_integrated, index=data_dict[components[0]].index, columns=cols)

            logger.info(f"PCA integration: {X.shape[1]} features → {X_integrated.shape[1]} factors "
                       f"({pca.explained_variance_ratio_.sum():.2%} variance)")

            return result_df

        elif method.upper() == 'MOFA2':
            # Placeholder for MOFA2 integration
            logger.warning("MOFA2 integration not yet implemented, using PCA")
            pca = PCA(n_components=min(n_factors, X.shape[1], X.shape[0]))
            X_integrated = pca.fit_transform(X)
            cols = [f"level_factor_{i+1}" for i in range(X_integrated.shape[1])]
            return pd.DataFrame(X_integrated, index=data_dict[components[0]].index, columns=cols)

        elif method.upper() == 'NONE':
            # No integration, pass through
            return data_dict[components[0]]

        else:
            raise ValueError(f"Unknown integration method: {method}")

    def hierarchical_integration(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Perform hierarchical integration across all levels

        Args:
            data_dict: Dictionary mapping modality names to DataFrames

        Returns:
            Final integrated DataFrame
        """
        integrated_levels = {}

        # Process each level
        for level_name, level_config in self.integration_levels.items():
            logger.info(f"Processing {level_name}...")

            level_result = self.integrate_level(data_dict, level_config)
            integrated_levels[level_name] = level_result
            self.level_results[level_name] = level_result

        # Final integration: concatenate all levels
        if len(integrated_levels) == 0:
            raise ValueError("No levels were successfully integrated")

        final_data = pd.concat(integrated_levels.values(), axis=1)

        logger.info(f"Hierarchical integration complete: {final_data.shape[1]} final features")

        return final_data


class MultimodalNetworkBuilder:
    """Build networks connecting different data modalities"""

    def __init__(self):
        """Initialize network builder"""
        self.networks = {}

    def build_correlation_network(self,
                                  df1: pd.DataFrame,
                                  df2: pd.DataFrame,
                                  method: str = 'spearman',
                                  threshold: float = 0.3) -> pd.DataFrame:
        """
        Build correlation-based network between two modalities

        Args:
            df1: First modality DataFrame
            df2: Second modality DataFrame
            method: Correlation method ('pearson', 'spearman')
            threshold: Correlation threshold for edge inclusion

        Returns:
            Edge list DataFrame
        """
        # Ensure same samples
        common_idx = df1.index.intersection(df2.index)
        df1_aligned = df1.loc[common_idx]
        df2_aligned = df2.loc[common_idx]

        # Calculate correlations
        edges = []

        for col1 in df1_aligned.columns:
            for col2 in df2_aligned.columns:
                x = df1_aligned[col1].values
                y = df2_aligned[col2].values

                # Remove NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 10:
                    continue

                x = x[mask]
                y = y[mask]

                if method == 'pearson':
                    r, p = stats.pearsonr(x, y)
                elif method == 'spearman':
                    r, p = stats.spearmanr(x, y)
                else:
                    raise ValueError(f"Unknown correlation method: {method}")

                if abs(r) >= threshold and p < 0.05:
                    edges.append({
                        'source': col1,
                        'target': col2,
                        'weight': r,
                        'pvalue': p
                    })

        edge_df = pd.DataFrame(edges)
        logger.info(f"Built network: {len(edges)} edges (threshold: {threshold})")

        return edge_df

    def build_gene_metabolite_network(self,
                                     genetic_df: pd.DataFrame,
                                     metabolite_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build gene-metabolite association network

        Args:
            genetic_df: Genetic features (SNPs, gene expression)
            metabolite_df: Metabolite concentrations

        Returns:
            Edge list DataFrame
        """
        logger.info("Building gene-metabolite network...")
        return self.build_correlation_network(genetic_df, metabolite_df, method='spearman', threshold=0.3)

    def build_metabolite_phenotype_network(self,
                                          metabolite_df: pd.DataFrame,
                                          clinical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build metabolite-clinical phenotype network

        Args:
            metabolite_df: Metabolite concentrations
            clinical_df: Clinical measures

        Returns:
            Edge list DataFrame
        """
        logger.info("Building metabolite-phenotype network...")
        return self.build_correlation_network(metabolite_df, clinical_df, method='spearman', threshold=0.25)

    def build_gxe_network(self,
                         genetic_df: pd.DataFrame,
                         environmental_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build gene-environment interaction network

        Args:
            genetic_df: Genetic features
            environmental_df: Environmental exposures

        Returns:
            Edge list DataFrame
        """
        logger.info("Building gene-environment network...")
        return self.build_correlation_network(genetic_df, environmental_df, method='spearman', threshold=0.2)

    def build_autonomic_symptom_network(self,
                                       autonomic_df: pd.DataFrame,
                                       clinical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build autonomic-symptom network

        Args:
            autonomic_df: Autonomic measures (HRV, EDA, etc.)
            clinical_df: Clinical symptom scores

        Returns:
            Edge list DataFrame
        """
        logger.info("Building autonomic-symptom network...")
        return self.build_correlation_network(autonomic_df, clinical_df, method='spearman', threshold=0.25)

    def integrate_networks(self, networks: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Create multi-layer network from individual networks

        Args:
            networks: Dictionary of edge list DataFrames

        Returns:
            Multi-layer network structure
        """
        multilayer = {
            'layers': networks,
            'n_layers': len(networks),
            'total_edges': sum(len(net) for net in networks.values())
        }

        # Calculate inter-layer connectivity
        # (nodes that appear in multiple layers)
        all_nodes = set()
        for net in networks.values():
            if len(net) > 0:
                all_nodes.update(net['source'].unique())
                all_nodes.update(net['target'].unique())

        multilayer['n_nodes'] = len(all_nodes)

        # Node degree across layers
        node_degrees = {}
        for node in all_nodes:
            degree = 0
            for net in networks.values():
                if len(net) > 0:
                    degree += ((net['source'] == node) | (net['target'] == node)).sum()
            node_degrees[node] = degree

        multilayer['node_degrees'] = node_degrees

        # Find hub nodes (high degree)
        if node_degrees:
            threshold = np.percentile(list(node_degrees.values()), 90)
            hubs = [node for node, degree in node_degrees.items() if degree >= threshold]
            multilayer['hub_nodes'] = hubs

        logger.info(f"Multi-layer network: {multilayer['n_nodes']} nodes, "
                   f"{multilayer['total_edges']} edges, {len(multilayer.get('hub_nodes', []))} hubs")

        return multilayer


def integrate_extended_multiomics(
    genetic_df: Optional[pd.DataFrame] = None,
    metabolomic_df: Optional[pd.DataFrame] = None,
    clinical_df: Optional[pd.DataFrame] = None,
    autonomic_df: Optional[pd.DataFrame] = None,
    circadian_df: Optional[pd.DataFrame] = None,
    environmental_df: Optional[pd.DataFrame] = None,
    toxicant_df: Optional[pd.DataFrame] = None,
    sensory_df: Optional[pd.DataFrame] = None,
    interoception_df: Optional[pd.DataFrame] = None,
    voice_df: Optional[pd.DataFrame] = None,
    microbiome_df: Optional[pd.DataFrame] = None,
    context_df: Optional[pd.DataFrame] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Extended multi-omics integration including all feature types

    Implements hierarchical integration with time-aware adjustment for
    circadian features and weighted combination based on proximity to phenotype.

    Args:
        genetic_df: Genetic features (SNPs, PRS, CNVs)
        metabolomic_df: Metabolite concentrations
        clinical_df: Clinical phenotypes and symptoms
        autonomic_df: Autonomic measures (HRV, EDA, BP, etc.)
        circadian_df: Circadian features (cortisol, melatonin, actigraphy)
        environmental_df: Environmental exposures (air quality, chemicals, etc.)
        toxicant_df: Toxicant biomarkers (heavy metals, organic pollutants)
        sensory_df: Sensory processing measures
        interoception_df: Interoceptive accuracy and awareness
        voice_df: Voice and speech acoustics
        microbiome_df: Microbiome features
        context_df: Collection context (time, fasting status, etc.)
        metadata: Additional metadata

    Returns:
        Dictionary with integrated features and networks
    """
    logger.info("Starting extended multi-omics integration...")

    # Define feature importance weights based on proximity to phenotype
    weights = {
        'genetic': 0.15,           # Distal (genetic predisposition)
        'environmental': 0.08,     # Distal modifier
        'toxicant': 0.07,         # Distal modifier
        'microbiome': 0.08,       # Intermediate
        'metabolomic': 0.20,      # Proximal (biological state)
        'autonomic': 0.12,        # Proximal (physiological state)
        'circadian': 0.10,        # Proximal (regulatory state)
        'sensory': 0.07,          # Intermediate (processing differences)
        'interoception': 0.06,    # Intermediate (awareness)
        'voice': 0.05,            # Intermediate (expression)
        'clinical': 0.02          # Direct phenotype (outcome, not predictor)
    }

    # Collect available data
    data_dict = {}
    if genetic_df is not None and len(genetic_df) > 0:
        data_dict['genetic'] = genetic_df
    if metabolomic_df is not None and len(metabolomic_df) > 0:
        data_dict['metabolomic'] = metabolomic_df
    if clinical_df is not None and len(clinical_df) > 0:
        data_dict['clinical'] = clinical_df
    if autonomic_df is not None and len(autonomic_df) > 0:
        data_dict['autonomic'] = autonomic_df
    if circadian_df is not None and len(circadian_df) > 0:
        data_dict['circadian'] = circadian_df
    if environmental_df is not None and len(environmental_df) > 0:
        data_dict['environmental'] = environmental_df
    if toxicant_df is not None and len(toxicant_df) > 0:
        data_dict['toxicant'] = toxicant_df
    if sensory_df is not None and len(sensory_df) > 0:
        data_dict['sensory'] = sensory_df
    if interoception_df is not None and len(interoception_df) > 0:
        data_dict['interoception'] = interoception_df
    if voice_df is not None and len(voice_df) > 0:
        data_dict['voice'] = voice_df
    if microbiome_df is not None and len(microbiome_df) > 0:
        data_dict['microbiome'] = microbiome_df

    if len(data_dict) == 0:
        raise ValueError("No data provided for integration")

    logger.info(f"Available modalities: {list(data_dict.keys())}")

    # Time-aware adjustment for circadian features
    if circadian_df is not None and context_df is not None:
        if 'collection_clock_time' in context_df.columns:
            logger.info("Applying time-aware adjustment to circadian features...")

            adjuster = TimeAwareAdjuster()

            # Identify time-sensitive columns (cortisol, melatonin, etc.)
            time_sensitive = [col for col in circadian_df.columns
                             if any(marker in col.lower() for marker in
                                   ['cortisol', 'melatonin', 'temperature', 'activity'])]

            if len(time_sensitive) > 0:
                # Merge context
                circ_with_context = circadian_df.copy()
                circ_with_context['collection_clock_time'] = context_df['collection_clock_time']

                adjusted = adjuster.adjust_for_collection_time(
                    circ_with_context,
                    time_col='collection_clock_time',
                    time_sensitive_cols=time_sensitive,
                    standard_time=9.0  # Adjust to 9 AM
                )

                # Remove time column
                data_dict['circadian'] = adjusted.drop(columns=['collection_clock_time'])
                logger.info(f"Adjusted {len(time_sensitive)} circadian features for collection time")

    # Define preprocessing for each modality
    preprocessing_map = {
        'genetic': StandardScaler(),
        'metabolomic': RobustScaler(),
        'autonomic': StandardScaler(),
        'circadian': StandardScaler(),
        'environmental': QuantileTransformer(output_distribution='normal'),
        'toxicant': RobustScaler(),
        'sensory': StandardScaler(),
        'interoception': StandardScaler(),
        'voice': RobustScaler(),
        'microbiome': StandardScaler(),  # Assumes CLR-transformed
        'clinical': StandardScaler()
    }

    # Apply preprocessing
    preprocessed_dict = {}
    for modality, df in data_dict.items():
        if modality in preprocessing_map:
            scaler = preprocessing_map[modality]
            scaled = scaler.fit_transform(df.values)
            preprocessed_dict[modality] = pd.DataFrame(
                scaled,
                index=df.index,
                columns=df.columns
            )
            logger.info(f"Preprocessed {modality}: {df.shape}")
        else:
            preprocessed_dict[modality] = df

    # Define hierarchical integration levels
    integration_levels = {
        'level1_biological': {
            'components': ['genetic', 'metabolomic', 'microbiome', 'autonomic', 'circadian'],
            'method': 'PCA',
            'n_factors': 30
        },
        'level2_environmental': {
            'components': ['environmental', 'toxicant'],
            'method': 'PCA',
            'n_factors': 15
        },
        'level3_cognitive_sensory': {
            'components': ['sensory', 'interoception', 'voice'],
            'method': 'PCA',
            'n_factors': 15
        },
        'level4_clinical': {
            'components': ['clinical'],
            'method': 'None',
            'n_factors': None
        }
    }

    # Filter levels to only include those with available data
    filtered_levels = {}
    for level_name, level_config in integration_levels.items():
        available_components = [c for c in level_config['components'] if c in preprocessed_dict]
        if len(available_components) > 0:
            filtered_levels[level_name] = level_config.copy()
            filtered_levels[level_name]['components'] = available_components

    # Perform hierarchical integration
    integrator = HierarchicalIntegrator(filtered_levels, weights)
    integrated_features = integrator.hierarchical_integration(preprocessed_dict)

    logger.info(f"Final integrated features: {integrated_features.shape}")

    # Build multimodal networks
    network_builder = MultimodalNetworkBuilder()
    networks = {}

    if 'genetic' in preprocessed_dict and 'metabolomic' in preprocessed_dict:
        networks['gene_metabolite'] = network_builder.build_gene_metabolite_network(
            preprocessed_dict['genetic'],
            preprocessed_dict['metabolomic']
        )

    if 'metabolomic' in preprocessed_dict and 'clinical' in preprocessed_dict:
        networks['metabolite_clinical'] = network_builder.build_metabolite_phenotype_network(
            preprocessed_dict['metabolomic'],
            preprocessed_dict['clinical']
        )

    if 'genetic' in preprocessed_dict and 'environmental' in preprocessed_dict:
        networks['environment_gene'] = network_builder.build_gxe_network(
            preprocessed_dict['genetic'],
            preprocessed_dict['environmental']
        )

    if 'autonomic' in preprocessed_dict and 'clinical' in preprocessed_dict:
        networks['autonomic_clinical'] = network_builder.build_autonomic_symptom_network(
            preprocessed_dict['autonomic'],
            preprocessed_dict['clinical']
        )

    multilayer_network = network_builder.integrate_networks(networks)

    # Return results
    results = {
        'integrated_features': integrated_features,
        'level_results': integrator.level_results,
        'networks': networks,
        'multilayer_network': multilayer_network,
        'weights': weights,
        'preprocessing': preprocessing_map,
        'n_samples': integrated_features.shape[0],
        'n_features': integrated_features.shape[1],
        'modalities': list(data_dict.keys())
    }

    logger.info("Extended multi-omics integration complete!")

    return results


# Alias for backward compatibility
integrate_multiomics = integrate_extended_multiomics


if __name__ == '__main__':
    # Example usage
    logger.info("Extended Multi-Omics Integration initialized")

    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 100

    # Genetic data
    genetic_df = pd.DataFrame(
        np.random.randn(n_samples, 50),
        columns=[f'SNP_{i}' for i in range(50)]
    )

    # Metabolomic data
    metabolomic_df = pd.DataFrame(
        np.random.randn(n_samples, 30),
        columns=[f'metabolite_{i}' for i in range(30)]
    )

    # Autonomic data
    autonomic_df = pd.DataFrame({
        'SDNN': np.random.uniform(20, 100, n_samples),
        'RMSSD': np.random.uniform(15, 80, n_samples),
        'LF_HF_ratio': np.random.uniform(0.5, 3.0, n_samples),
        'SCL_mean': np.random.uniform(2, 15, n_samples)
    })

    # Circadian data
    circadian_df = pd.DataFrame({
        'cortisol_awakening': np.random.uniform(10, 30, n_samples),
        'melatonin_evening': np.random.uniform(5, 50, n_samples),
        'actigraphy_IS': np.random.uniform(0.3, 0.9, n_samples)
    })

    # Context data (collection time)
    context_df = pd.DataFrame({
        'collection_clock_time': np.random.uniform(7, 11, n_samples)  # 7-11 AM
    })

    # Clinical data
    clinical_df = pd.DataFrame({
        'ADOS_score': np.random.randint(5, 25, n_samples),
        'ADHD_RS_total': np.random.randint(10, 50, n_samples),
        'SRS_score': np.random.randint(40, 120, n_samples)
    })

    # Run integration
    results = integrate_extended_multiomics(
        genetic_df=genetic_df,
        metabolomic_df=metabolomic_df,
        autonomic_df=autonomic_df,
        circadian_df=circadian_df,
        clinical_df=clinical_df,
        context_df=context_df
    )

    print("\n" + "="*70)
    print("Extended Multi-Omics Integration Results")
    print("="*70)
    print(f"\nIntegrated features shape: {results['integrated_features'].shape}")
    print(f"Number of modalities: {len(results['modalities'])}")
    print(f"Modalities: {', '.join(results['modalities'])}")
    print(f"\nHierarchical levels:")
    for level, df in results['level_results'].items():
        print(f"  {level}: {df.shape[1]} factors")
    print(f"\nNetworks built: {len(results['networks'])}")
    for net_name, net_df in results['networks'].items():
        print(f"  {net_name}: {len(net_df)} edges")
    if 'hub_nodes' in results['multilayer_network']:
        print(f"\nHub nodes ({len(results['multilayer_network']['hub_nodes'])}):")
        print(f"  {', '.join(results['multilayer_network']['hub_nodes'][:10])}")
