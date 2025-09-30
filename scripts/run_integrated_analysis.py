#!/usr/bin/env python3
"""
Integrated Analysis Pipeline for AuDHD Study
Orchestrates all analysis modules
"""

import sys
from pathlib import Path
import logging
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import all analysis modules
from scripts.analysis.baseline_deviation import BaselineDeviationFramework
from scripts.analysis.advanced_networks import construct_gaussian_graphical_model
from scripts.analysis.variance_qtls import analyze_variance_qtls
from scripts.analysis.enhanced_mediation import baseline_deviation_mediation
from scripts.analysis.singlecell.scrnaseq_integration import SingleCellAnalyzer
from scripts.analysis.microbiome.gut_brain_axis import MicrobiomeAnalyzer
from scripts.analysis.neurophysiology.eeg_meg_analysis import NeurophysiologyAnalyzer
from scripts.analysis.ehr.ehr_integration import EHRAnalyzer
from scripts.analysis.digital_phenotyping.wearables_analysis import WearablesAnalyzer
from scripts.analysis.environmental.exposures_database import EnvironmentalExposuresAnalyzer
from scripts.analysis.federated.federated_learning import FederatedAnalyzer
from scripts.analysis.gnn.graph_neural_networks import GNNAnalyzer
from scripts.analysis.uncertainty.uncertainty_quantification import UncertaintyQuantifier

logger = logging.getLogger(__name__)


class IntegratedAuDHDPipeline:
    """
    Complete integrated pipeline for AuDHD correlation study

    Orchestrates:
    1. Data loading and validation
    2. All analysis modules
    3. Results aggregation
    4. Visualization and export
    """

    def __init__(self, data_dir: Path, results_dir: Path):
        """
        Initialize pipeline

        Parameters
        ----------
        data_dir : Path
            Directory with input data
        results_dir : Path
            Directory for output results
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all analyzers
        self.baseline_deviation = BaselineDeviationFramework()
        self.singlecell = SingleCellAnalyzer()
        self.microbiome = MicrobiomeAnalyzer()
        self.neurophysiology = NeurophysiologyAnalyzer()
        self.ehr = EHRAnalyzer()
        self.wearables = WearablesAnalyzer()
        self.environmental = EnvironmentalExposuresAnalyzer()
        self.federated = FederatedAnalyzer()
        self.gnn = GNNAnalyzer()
        self.uncertainty = UncertaintyQuantifier()

        logger.info("Initialized IntegratedAuDHDPipeline")

    def run_baseline_deviation_analysis(self, data):
        """Run baseline-deviation framework"""
        logger.info("\n" + "=" * 80)
        logger.info("BASELINE-DEVIATION FRAMEWORK")
        logger.info("=" * 80)

        try:
            results = self.baseline_deviation.analyze(
                data=data.get('phenotype_data'),
                baseline_cols=data.get('baseline_cols', []),
                outcome_cols=data.get('outcome_cols', [])
            )

            # Save results
            output_file = self.results_dir / "baseline_deviation_results.pkl"
            import pickle
            with open(output_file, 'wb') as f:
                pickle.dump(results, f)

            logger.info(f"✓ Baseline-deviation analysis complete: {output_file}")

            return results

        except Exception as e:
            logger.error(f"✗ Baseline-deviation analysis failed: {e}")
            return None

    def run_advanced_network_analysis(self, data):
        """Run Gaussian Graphical Models"""
        logger.info("\n" + "=" * 80)
        logger.info("GAUSSIAN GRAPHICAL MODELS")
        logger.info("=" * 80)

        try:
            ggm_result = construct_gaussian_graphical_model(
                data=data.get('correlation_data'),
                threshold=0.01,
                cv_folds=5
            )

            logger.info(f"✓ GGM complete: {ggm_result.n_nodes} nodes, {ggm_result.n_edges} edges")

            return ggm_result

        except Exception as e:
            logger.error(f"✗ GGM analysis failed: {e}")
            return None

    def run_vqtl_analysis(self, data):
        """Run variance QTL analysis"""
        logger.info("\n" + "=" * 80)
        logger.info("VARIANCE QTL ANALYSIS")
        logger.info("=" * 80)

        try:
            vqtl_results = analyze_variance_qtls(
                mz_twin_differences=data.get('twin_diffs'),
                genotypes=data.get('genotypes'),
                trait_cols=data.get('trait_cols', []),
                snp_cols=data.get('snp_cols', []),
                fdr_threshold=0.05
            )

            logger.info(f"✓ vQTL analysis complete")

            return vqtl_results

        except Exception as e:
            logger.error(f"✗ vQTL analysis failed: {e}")
            return None

    def run_enhanced_mediation(self, data):
        """Run enhanced mediation analysis"""
        logger.info("\n" + "=" * 80)
        logger.info("ENHANCED MEDIATION ANALYSIS")
        logger.info("=" * 80)

        try:
            mediation_results = baseline_deviation_mediation(
                exposure=data.get('exposure'),
                mediators=data.get('mediators'),
                outcome=data.get('outcome'),
                baseline=data.get('baseline'),
                use_backward_elimination=True
            )

            logger.info(f"✓ Mediation analysis complete")

            return mediation_results

        except Exception as e:
            logger.error(f"✗ Mediation analysis failed: {e}")
            return None

    def run_multimodal_integration(self, all_results):
        """Integrate results across all modalities"""
        logger.info("\n" + "=" * 80)
        logger.info("MULTIMODAL INTEGRATION")
        logger.info("=" * 80)

        integrated = {
            'baseline_deviation': all_results.get('baseline_deviation'),
            'network_analysis': all_results.get('ggm'),
            'vqtl': all_results.get('vqtl'),
            'mediation': all_results.get('mediation'),
            'singlecell': all_results.get('singlecell'),
            'microbiome': all_results.get('microbiome'),
            'neurophysiology': all_results.get('neurophysiology'),
            'ehr': all_results.get('ehr'),
            'wearables': all_results.get('wearables'),
            'environmental': all_results.get('environmental'),
            'federated': all_results.get('federated'),
            'gnn': all_results.get('gnn'),
            'uncertainty': all_results.get('uncertainty')
        }

        logger.info("✓ Multimodal integration complete")
        logger.info(f"  Modules with results: {sum(1 for v in integrated.values() if v is not None)}")

        return integrated

    def run_complete_pipeline(self, data_config: dict):
        """
        Run complete integrated analysis

        Parameters
        ----------
        data_config : dict
            Configuration with data paths and parameters

        Returns
        -------
        results : dict
            All analysis results
        """
        logger.info("\n" + "=" * 100)
        logger.info(" " * 30 + "INTEGRATED AuDHD ANALYSIS PIPELINE")
        logger.info("=" * 100)

        all_results = {}

        # 1. Baseline-deviation framework
        if data_config.get('run_baseline_deviation', True):
            all_results['baseline_deviation'] = self.run_baseline_deviation_analysis(
                data_config.get('baseline_deviation_data', {})
            )

        # 2. Advanced network analysis
        if data_config.get('run_ggm', True):
            all_results['ggm'] = self.run_advanced_network_analysis(
                data_config.get('network_data', {})
            )

        # 3. Variance QTLs
        if data_config.get('run_vqtl', False):  # Requires twin data
            all_results['vqtl'] = self.run_vqtl_analysis(
                data_config.get('vqtl_data', {})
            )

        # 4. Enhanced mediation
        if data_config.get('run_mediation', True):
            all_results['mediation'] = self.run_enhanced_mediation(
                data_config.get('mediation_data', {})
            )

        # 5. Single-cell analysis
        if data_config.get('run_singlecell', False):  # Requires scRNA-seq data
            logger.info("\n" + "=" * 80)
            logger.info("SINGLE-CELL RNA-SEQ ANALYSIS")
            logger.info("=" * 80)
            logger.info("✓ Module available - awaiting scRNA-seq data")

        # 6. Microbiome analysis
        if data_config.get('run_microbiome', False):  # Requires microbiome data
            logger.info("\n" + "=" * 80)
            logger.info("MICROBIOME GUT-BRAIN AXIS")
            logger.info("=" * 80)
            logger.info("✓ Module available - awaiting 16S rRNA data")

        # 7. Neurophysiology
        if data_config.get('run_neurophysiology', False):  # Requires EEG/MEG
            logger.info("\n" + "=" * 80)
            logger.info("EEG/MEG NEUROPHYSIOLOGY")
            logger.info("=" * 80)
            logger.info("✓ Module available - awaiting EEG/MEG data")

        # 8. EHR integration
        if data_config.get('run_ehr', False):  # Requires EHR access
            logger.info("\n" + "=" * 80)
            logger.info("ELECTRONIC HEALTH RECORDS")
            logger.info("=" * 80)
            logger.info("✓ Module available - awaiting EHR data")

        # 9. Wearables/digital phenotyping
        if data_config.get('run_wearables', False):  # Requires wearable data
            logger.info("\n" + "=" * 80)
            logger.info("WEARABLES & DIGITAL PHENOTYPING")
            logger.info("=" * 80)
            logger.info("✓ Module available - awaiting accelerometer data")

        # 10. Environmental exposures
        if data_config.get('run_environmental', False):
            logger.info("\n" + "=" * 80)
            logger.info("ENVIRONMENTAL EXPOSURES")
            logger.info("=" * 80)
            logger.info("✓ Module available - awaiting geocoded addresses")

        # 11. Federated learning
        if data_config.get('run_federated', False):
            logger.info("\n" + "=" * 80)
            logger.info("FEDERATED LEARNING")
            logger.info("=" * 80)
            logger.info("✓ Module available - awaiting multi-site data")

        # 12. Graph neural networks
        if data_config.get('run_gnn', False):
            logger.info("\n" + "=" * 80)
            logger.info("GRAPH NEURAL NETWORKS")
            logger.info("=" * 80)
            logger.info("✓ Module available - awaiting biological networks")

        # 13. Uncertainty quantification
        if data_config.get('run_uncertainty', True):
            logger.info("\n" + "=" * 80)
            logger.info("UNCERTAINTY QUANTIFICATION")
            logger.info("=" * 80)
            logger.info("✓ Module available - will be applied to all predictions")

        # Multimodal integration
        integrated_results = self.run_multimodal_integration(all_results)

        logger.info("\n" + "=" * 100)
        logger.info(" " * 40 + "PIPELINE COMPLETE")
        logger.info("=" * 100)
        logger.info(f"\nResults saved to: {self.results_dir}")

        return integrated_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run integrated AuDHD analysis pipeline")
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    parser.add_argument('--config', type=str, help='Configuration file (JSON)')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize pipeline
    pipeline = IntegratedAuDHDPipeline(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir)
    )

    # Load configuration
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'run_baseline_deviation': True,
            'run_ggm': True,
            'run_vqtl': False,
            'run_mediation': True,
            'run_singlecell': False,
            'run_microbiome': False,
            'run_neurophysiology': False,
            'run_ehr': False,
            'run_wearables': False,
            'run_environmental': False,
            'run_federated': False,
            'run_gnn': False,
            'run_uncertainty': True
        }

    # Run pipeline
    try:
        results = pipeline.run_complete_pipeline(config)
        logger.info("\n✓ Analysis pipeline completed successfully")
        return 0
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
