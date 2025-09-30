#!/usr/bin/env python3
"""
Demo Analysis Run
Tests the integrated pipeline with available synthetic data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_integrated_analysis import IntegratedAuDHDPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_synthetic_data():
    """Load available synthetic datasets"""
    logger.info("Loading synthetic data...")

    data_dir = project_root / "data" / "processed"

    # Load datasets
    clinical = pd.read_csv(data_dir / "clinical" / "synthetic_clinical_data.csv", index_col='sample_id')
    metabolomics = pd.read_csv(data_dir / "metabolomics" / "synthetic_metabolomics_data.csv", index_col='sample_id')
    microbiome = pd.read_csv(data_dir / "microbiome" / "synthetic_microbiome_data.csv", index_col='sample_id')

    logger.info(f"  Clinical: {clinical.shape}")
    logger.info(f"  Metabolomics: {metabolomics.shape}")
    logger.info(f"  Microbiome: {microbiome.shape}")

    return clinical, metabolomics, microbiome


def prepare_analysis_data(clinical, metabolomics, microbiome):
    """Prepare data for analysis modules"""
    logger.info("Preparing data for analysis...")

    # Encode categorical variables
    clinical_numeric = clinical.copy()
    clinical_numeric['sex'] = (clinical['sex'] == 'M').astype(int)  # 1=Male, 0=Female

    # Combine all features
    all_features = pd.concat([clinical_numeric, metabolomics, microbiome], axis=1)

    # Define baseline features (demographic + core clinical)
    baseline_cols = ['age', 'sex', 'BMI', 'IQ']

    # Outcome features (symptom scores)
    outcome_cols = ['ADHD_inattention', 'ADHD_hyperactivity', 'anxiety_score']

    # Mediators (metabolomics + microbiome)
    mediator_cols = list(metabolomics.columns) + list(microbiome.columns)

    # Create data configuration
    config = {
        'run_baseline_deviation': True,
        'run_ggm': True,
        'run_vqtl': False,  # Requires twin data
        'run_mediation': True,
        'run_microbiome': True,
        'run_uncertainty': True,

        # Data for baseline-deviation
        'baseline_deviation_data': {
            'phenotype_data': all_features,
            'baseline_cols': baseline_cols,
            'outcome_cols': outcome_cols
        },

        # Data for network analysis
        'network_data': {
            'correlation_data': metabolomics  # Use metabolomics for network
        },

        # Data for mediation
        'mediation_data': {
            'exposure': clinical_numeric['ADHD_PRS'].values,  # Genetic risk
            'mediators': all_features[mediator_cols].values,  # Convert to numpy array
            'outcome': clinical_numeric['ADHD_inattention'].values,
            'baseline': all_features[baseline_cols].values  # Convert to numpy array
        }
    }

    return config


def main():
    """Run demo analysis"""
    logger.info("=" * 100)
    logger.info(" " * 35 + "AuDHD ANALYSIS DEMO")
    logger.info("=" * 100)

    # Load data
    clinical, metabolomics, microbiome = load_synthetic_data()

    # Prepare analysis configuration
    config = prepare_analysis_data(clinical, metabolomics, microbiome)

    # Initialize pipeline
    results_dir = project_root / "results" / "demo_run"
    pipeline = IntegratedAuDHDPipeline(
        data_dir=project_root / "data",
        results_dir=results_dir
    )

    # Run analysis
    logger.info("\nStarting integrated analysis pipeline...")
    try:
        results = pipeline.run_complete_pipeline(config)

        logger.info("\n" + "=" * 100)
        logger.info(" " * 40 + "ANALYSIS COMPLETE")
        logger.info("=" * 100)

        # Summary
        logger.info("\nResults Summary:")
        for module_name, module_results in results.items():
            if module_results is not None:
                logger.info(f"  ✓ {module_name}: SUCCESS")
            else:
                logger.info(f"  ✗ {module_name}: No results")

        logger.info(f"\nResults saved to: {results_dir}")

        return 0

    except Exception as e:
        logger.error(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
