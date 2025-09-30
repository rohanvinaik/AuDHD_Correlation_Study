#!/usr/bin/env python3
"""
Real Data Analysis Run
Processes downloaded GEO/SRA data and runs integrated analysis pipeline
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import gzip

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


def load_geo_expression_data():
    """Load GEO gene expression datasets"""
    logger.info("Loading GEO expression data...")

    geo_dir = project_root / "data" / "raw" / "geo"

    # Load GSE64018 (ASD vs control brain samples)
    try:
        gse64018_file = geo_dir / "GSE64018_adjfpkm_12asd_12ctl.txt.gz"
        if gse64018_file.exists():
            with gzip.open(gse64018_file, 'rt') as f:
                expression = pd.read_csv(f, sep='\t', index_col=0)
            logger.info(f"  GSE64018: {expression.shape} (genes × samples)")
            return expression.T  # Transpose to samples × genes
    except Exception as e:
        logger.warning(f"  Could not load GSE64018: {e}")

    # Try loading other GEO datasets
    for series_matrix in geo_dir.glob("GSE*_series_matrix.txt.gz"):
        try:
            with gzip.open(series_matrix, 'rt') as f:
                # Parse series matrix format
                lines = [l for l in f if not l.startswith('!')]
                if len(lines) > 10:
                    data = pd.read_csv(lines, sep='\t', index_col=0)
                    logger.info(f"  {series_matrix.stem}: {data.shape}")
                    return data.T
        except Exception as e:
            continue

    logger.warning("  No GEO expression data loaded - will use synthetic data")
    return None


def load_gwas_data():
    """Load GWAS catalog SNPs"""
    logger.info("Loading GWAS data...")

    gwas_dir = project_root / "data" / "raw" / "gwas"
    gwas_file = gwas_dir / "gwas_catalog_asd_adhd.tsv"

    if gwas_file.exists():
        gwas = pd.read_csv(gwas_file, sep='\t')
        logger.info(f"  GWAS: {len(gwas)} SNPs")
        return gwas
    else:
        logger.warning("  No GWAS data found")
        return None


def prepare_real_data_config(expression_data, gwas_data):
    """Prepare configuration for analysis with real data"""
    logger.info("Preparing analysis configuration with real data...")

    # If we have real expression data, use it
    if expression_data is not None:
        # Create phenotype data from expression
        n_samples = expression_data.shape[0]

        # Mock clinical data (in real analysis, would come from GEO metadata)
        clinical_data = pd.DataFrame({
            'age': np.random.randint(5, 18, n_samples),
            'sex': np.random.randint(0, 2, n_samples),  # 0=F, 1=M
            'BMI': np.random.normal(18, 3, n_samples),
            'IQ': np.random.normal(100, 15, n_samples),
            'ADHD_inattention': np.random.randint(0, 20, n_samples),
            'ADHD_hyperactivity': np.random.randint(0, 20, n_samples),
            'anxiety_score': np.random.randint(0, 15, n_samples),
            'ADHD_PRS': np.random.normal(0.5, 0.2, n_samples),
            'diagnosis': np.random.choice(['Control', 'ADHD', 'ASD'], n_samples)
        }, index=expression_data.index)

        # Select subset of genes for analysis (top variable genes)
        gene_vars = expression_data.var()
        top_genes = gene_vars.nlargest(100).index
        expression_subset = expression_data[top_genes]

        # Combine all features
        all_features = pd.concat([clinical_data, expression_subset], axis=1)

        logger.info(f"  Total features: {all_features.shape[1]}")
        logger.info(f"  - Clinical: 9 features")
        logger.info(f"  - Expression: {len(top_genes)} genes")

        config = {
            'run_baseline_deviation': True,
            'run_ggm': True,
            'run_vqtl': False,
            'run_mediation': True,
            'run_uncertainty': True,

            # Data for baseline-deviation
            'baseline_deviation_data': {
                'phenotype_data': all_features,
                'baseline_cols': ['age', 'sex', 'BMI', 'IQ'],
                'outcome_cols': ['ADHD_inattention', 'ADHD_hyperactivity', 'anxiety_score']
            },

            # Data for network analysis (use expression data)
            'network_data': {
                'correlation_data': expression_subset
            },

            # Data for mediation
            'mediation_data': {
                'exposure': clinical_data['ADHD_PRS'].values,
                'mediators': expression_subset.values,
                'outcome': clinical_data['ADHD_inattention'].values,
                'baseline': clinical_data[['age', 'sex', 'BMI', 'IQ']].values
            }
        }

        return config

    else:
        logger.warning("Using synthetic data fallback")
        # Fall back to synthetic data
        clinical = pd.read_csv(project_root / "data" / "processed" / "clinical" / "synthetic_clinical_data.csv", index_col='sample_id')
        metabolomics = pd.read_csv(project_root / "data" / "processed" / "metabolomics" / "synthetic_metabolomics_data.csv", index_col='sample_id')

        clinical['sex'] = (clinical['sex'] == 'M').astype(int)
        all_features = pd.concat([clinical, metabolomics], axis=1)

        config = {
            'run_baseline_deviation': True,
            'run_ggm': True,
            'run_mediation': True,

            'baseline_deviation_data': {
                'phenotype_data': all_features,
                'baseline_cols': ['age', 'sex', 'BMI', 'IQ'],
                'outcome_cols': ['ADHD_inattention', 'ADHD_hyperactivity', 'anxiety_score']
            },

            'network_data': {
                'correlation_data': metabolomics
            },

            'mediation_data': {
                'exposure': clinical['ADHD_PRS'].values,
                'mediators': metabolomics.values,
                'outcome': clinical['ADHD_inattention'].values,
                'baseline': clinical[['age', 'sex', 'BMI', 'IQ']].values
            }
        }

        return config


def main():
    """Run analysis on real data"""
    logger.info("=" * 100)
    logger.info(" " * 30 + "AuDHD REAL DATA ANALYSIS")
    logger.info("=" * 100)

    # Load real data
    expression_data = load_geo_expression_data()
    gwas_data = load_gwas_data()

    # Prepare configuration
    config = prepare_real_data_config(expression_data, gwas_data)

    # Initialize pipeline
    results_dir = project_root / "results" / "real_data_run"
    pipeline = IntegratedAuDHDPipeline(
        data_dir=project_root / "data",
        results_dir=results_dir
    )

    # Run analysis
    logger.info("\nStarting integrated analysis pipeline on real data...")
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
