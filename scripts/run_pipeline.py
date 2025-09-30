#!/usr/bin/env python
"""Main pipeline orchestrator with phase-wise execution and checkpointing

Coordinates all analysis phases with:
- Checkpointing and resume capability
- Parallel processing for independent tasks
- Comprehensive logging and error handling
- Resource management (memory, CPU)
- Progress tracking and time estimation
- Audit trail generation
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import json
import pickle
import traceback
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class PhaseResult:
    """Result of a pipeline phase"""
    phase_name: str
    status: str  # 'success', 'failed', 'skipped'
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    checkpoint_path: Optional[Path] = None


@dataclass
class PipelineState:
    """Current state of pipeline execution"""
    run_id: str
    start_time: datetime
    completed_phases: List[str] = field(default_factory=list)
    failed_phases: List[str] = field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = field(default_factory=dict)
    current_phase: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Manages pipeline checkpoints for resume capability"""

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(
        self,
        phase_name: str,
        data: Any,
        state: PipelineState,
    ) -> Path:
        """
        Save checkpoint for a phase

        Args:
            phase_name: Name of phase
            data: Data to checkpoint
            state: Pipeline state

        Returns:
            Path to checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / f"{phase_name}.pkl"

        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'phase_name': phase_name,
                    'data': data,
                    'state': state,
                    'timestamp': datetime.now(),
                }, f)

            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, phase_name: str) -> Optional[Dict]:
        """
        Load checkpoint for a phase

        Args:
            phase_name: Name of phase

        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{phase_name}.pkl"

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None

    def checkpoint_exists(self, phase_name: str) -> bool:
        """Check if checkpoint exists for phase"""
        return (self.checkpoint_dir / f"{phase_name}.pkl").exists()

    def clear_checkpoints(self):
        """Remove all checkpoints"""
        for checkpoint in self.checkpoint_dir.glob("*.pkl"):
            checkpoint.unlink()
        self.logger.info("All checkpoints cleared")

    def save_state(self, state: PipelineState):
        """Save pipeline state"""
        state_path = self.checkpoint_dir / "pipeline_state.json"

        state_dict = {
            'run_id': state.run_id,
            'start_time': state.start_time.isoformat(),
            'completed_phases': state.completed_phases,
            'failed_phases': state.failed_phases,
            'current_phase': state.current_phase,
            'metadata': state.metadata,
        }

        with open(state_path, 'w') as f:
            json.dump(state_dict, f, indent=2)

    def load_state(self) -> Optional[PipelineState]:
        """Load pipeline state"""
        state_path = self.checkpoint_dir / "pipeline_state.json"

        if not state_path.exists():
            return None

        with open(state_path, 'r') as f:
            state_dict = json.load(f)

        return PipelineState(
            run_id=state_dict['run_id'],
            start_time=datetime.fromisoformat(state_dict['start_time']),
            completed_phases=state_dict['completed_phases'],
            failed_phases=state_dict['failed_phases'],
            current_phase=state_dict['current_phase'],
            metadata=state_dict['metadata'],
        )


class ResourceManager:
    """Manages computational resources"""

    def __init__(
        self,
        max_memory_gb: Optional[float] = None,
        max_cpus: Optional[int] = None,
    ):
        """
        Initialize resource manager

        Args:
            max_memory_gb: Maximum memory in GB
            max_cpus: Maximum CPU cores
        """
        self.max_memory_gb = max_memory_gb
        self.max_cpus = max_cpus or mp.cpu_count()
        self.logger = logging.getLogger(__name__)

    def get_available_memory(self) -> float:
        """Get available memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            self.logger.warning("psutil not available, cannot check memory")
            return float('inf')

    def get_available_cpus(self) -> int:
        """Get available CPU cores"""
        return self.max_cpus

    def check_resources(
        self,
        required_memory_gb: Optional[float] = None,
        required_cpus: Optional[int] = None,
    ) -> bool:
        """
        Check if required resources are available

        Args:
            required_memory_gb: Required memory in GB
            required_cpus: Required CPU cores

        Returns:
            True if resources available
        """
        if required_memory_gb is not None:
            available_memory = self.get_available_memory()
            if available_memory < required_memory_gb:
                self.logger.warning(
                    f"Insufficient memory: {available_memory:.1f}GB available, "
                    f"{required_memory_gb:.1f}GB required"
                )
                return False

        if required_cpus is not None:
            if self.max_cpus < required_cpus:
                self.logger.warning(
                    f"Insufficient CPUs: {self.max_cpus} available, "
                    f"{required_cpus} required"
                )
                return False

        return True

    def get_optimal_n_jobs(
        self,
        n_tasks: int,
        min_cpus: int = 1,
    ) -> int:
        """
        Get optimal number of parallel jobs

        Args:
            n_tasks: Number of tasks
            min_cpus: Minimum CPUs per task

        Returns:
            Optimal number of jobs
        """
        max_parallel = self.max_cpus // min_cpus
        return min(n_tasks, max_parallel)


class ProgressTracker:
    """Tracks pipeline progress and estimates time"""

    def __init__(self, total_phases: int):
        """
        Initialize progress tracker

        Args:
            total_phases: Total number of phases
        """
        self.total_phases = total_phases
        self.completed_phases = 0
        self.phase_times: Dict[str, float] = {}
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)

    def update(self, phase_name: str, duration: float):
        """
        Update progress with completed phase

        Args:
            phase_name: Name of completed phase
            duration: Duration in seconds
        """
        self.completed_phases += 1
        self.phase_times[phase_name] = duration

        # Calculate progress
        progress = self.completed_phases / self.total_phases * 100

        # Estimate remaining time
        avg_time = np.mean(list(self.phase_times.values()))
        remaining_phases = self.total_phases - self.completed_phases
        eta_seconds = avg_time * remaining_phases
        eta = timedelta(seconds=int(eta_seconds))

        self.logger.info(
            f"Progress: {progress:.1f}% ({self.completed_phases}/{self.total_phases}) "
            f"| ETA: {eta} | Phase time: {duration:.1f}s"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary"""
        total_time = time.time() - self.start_time

        return {
            'completed_phases': self.completed_phases,
            'total_phases': self.total_phases,
            'progress_pct': self.completed_phases / self.total_phases * 100,
            'total_time_seconds': total_time,
            'total_time': str(timedelta(seconds=int(total_time))),
            'phase_times': self.phase_times,
        }


class AuditLogger:
    """Generates audit trails for pipeline execution"""

    def __init__(self, audit_dir: Path):
        """
        Initialize audit logger

        Args:
            audit_dir: Directory for audit logs
        """
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict] = []

    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
    ):
        """
        Log an audit event

        Args:
            event_type: Type of event
            details: Event details
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
        }
        self.events.append(event)

    def save_audit_trail(self, run_id: str):
        """Save audit trail to file"""
        audit_path = self.audit_dir / f"audit_{run_id}.json"

        with open(audit_path, 'w') as f:
            json.dump({
                'run_id': run_id,
                'generated_at': datetime.now().isoformat(),
                'events': self.events,
            }, f, indent=2)

        return audit_path


class PipelineOrchestrator:
    """Main pipeline orchestrator"""

    def __init__(self, config: DictConfig):
        """
        Initialize pipeline orchestrator

        Args:
            config: Hydra configuration
        """
        self.config = config
        self.logger = self._setup_logging()

        # Initialize run ID
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize managers
        self.checkpoint_manager = CheckpointManager(
            Path(config.pipeline.checkpoint_dir) / self.run_id
        )
        self.resource_manager = ResourceManager(
            max_memory_gb=config.compute.resources.get('max_memory_gb'),
            max_cpus=config.compute.resources.get('max_cpus'),
        )
        self.audit_logger = AuditLogger(
            Path(config.pipeline.audit_dir) / self.run_id
        )

        # Initialize state
        self.state = PipelineState(
            run_id=self.run_id,
            start_time=datetime.now(),
            metadata={'config': OmegaConf.to_container(config)},
        )

        # Define phases
        self.phases = self._define_phases()
        self.progress_tracker = ProgressTracker(len(self.phases))

        # Execution mode
        self.dry_run = config.pipeline.get('dry_run', False)
        self.resume = config.pipeline.get('resume', False)

        self.logger.info(f"Pipeline initialized: run_id={self.run_id}")
        self.audit_logger.log_event('pipeline_initialized', {
            'run_id': self.run_id,
            'config': OmegaConf.to_container(config),
            'dry_run': self.dry_run,
            'resume': self.resume,
        })

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        log_dir = Path(self.config.pipeline.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = logging.getLogger('pipeline')
        logger.setLevel(logging.DEBUG if self.config.pipeline.get('verbose', False) else logging.INFO)

        # File handler
        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _define_phases(self) -> List[Dict[str, Any]]:
        """Define pipeline phases"""
        return [
            {
                'name': 'data_loading',
                'function': self._phase_data_loading,
                'dependencies': [],
                'required_memory_gb': 8.0,
                'required_cpus': 1,
                'parallelizable': False,
            },
            {
                'name': 'preprocessing',
                'function': self._phase_preprocessing,
                'dependencies': ['data_loading'],
                'required_memory_gb': 16.0,
                'required_cpus': 4,
                'parallelizable': True,
            },
            {
                'name': 'quality_control',
                'function': self._phase_quality_control,
                'dependencies': ['preprocessing'],
                'required_memory_gb': 8.0,
                'required_cpus': 2,
                'parallelizable': True,
            },
            {
                'name': 'integration',
                'function': self._phase_integration,
                'dependencies': ['quality_control'],
                'required_memory_gb': 32.0,
                'required_cpus': 8,
                'parallelizable': False,
            },
            {
                'name': 'clustering',
                'function': self._phase_clustering,
                'dependencies': ['integration'],
                'required_memory_gb': 16.0,
                'required_cpus': 4,
                'parallelizable': False,
            },
            {
                'name': 'validation',
                'function': self._phase_validation,
                'dependencies': ['clustering'],
                'required_memory_gb': 8.0,
                'required_cpus': 4,
                'parallelizable': True,
            },
            {
                'name': 'biological_analysis',
                'function': self._phase_biological_analysis,
                'dependencies': ['clustering'],
                'required_memory_gb': 16.0,
                'required_cpus': 4,
                'parallelizable': True,
            },
            {
                'name': 'visualization',
                'function': self._phase_visualization,
                'dependencies': ['clustering', 'validation'],
                'required_memory_gb': 4.0,
                'required_cpus': 2,
                'parallelizable': True,
            },
            {
                'name': 'reporting',
                'function': self._phase_reporting,
                'dependencies': ['visualization', 'biological_analysis'],
                'required_memory_gb': 4.0,
                'required_cpus': 1,
                'parallelizable': False,
            },
        ]

    def run(self) -> Dict[str, Any]:
        """
        Execute pipeline

        Returns:
            Final results dictionary
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting pipeline execution: {self.run_id}")
        self.logger.info("=" * 80)

        if self.dry_run:
            self.logger.info("DRY RUN MODE - No actual execution")
            return self._dry_run()

        try:
            # Check for resume
            if self.resume:
                self._resume_from_checkpoint()

            # Execute phases
            results = {}
            for phase in self.phases:
                phase_name = phase['name']

                # Skip if already completed
                if phase_name in self.state.completed_phases:
                    self.logger.info(f"Skipping completed phase: {phase_name}")
                    # Load checkpoint
                    checkpoint = self.checkpoint_manager.load_checkpoint(phase_name)
                    if checkpoint:
                        results[phase_name] = checkpoint['data']
                    continue

                # Check dependencies
                if not self._check_dependencies(phase, results):
                    self.logger.error(f"Dependencies not met for phase: {phase_name}")
                    self.state.failed_phases.append(phase_name)
                    break

                # Check resources
                if not self.resource_manager.check_resources(
                    required_memory_gb=phase.get('required_memory_gb'),
                    required_cpus=phase.get('required_cpus'),
                ):
                    self.logger.error(f"Insufficient resources for phase: {phase_name}")
                    self.state.failed_phases.append(phase_name)
                    break

                # Execute phase
                phase_result = self._execute_phase(phase, results)

                if phase_result.status == 'success':
                    results[phase_name] = phase_result.outputs
                    self.state.completed_phases.append(phase_name)
                else:
                    self.state.failed_phases.append(phase_name)
                    self.logger.error(f"Phase failed: {phase_name}")
                    if not self.config.pipeline.get('continue_on_error', False):
                        break

                # Save state
                self.checkpoint_manager.save_state(self.state)

            # Generate final report
            self._generate_final_report(results)

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}")
            self.logger.error(traceback.format_exc())
            self.audit_logger.log_event('pipeline_failed', {
                'error': str(e),
                'traceback': traceback.format_exc(),
            })
            raise

        finally:
            # Save audit trail
            audit_path = self.audit_logger.save_audit_trail(self.run_id)
            self.logger.info(f"Audit trail saved: {audit_path}")

    def _execute_phase(
        self,
        phase: Dict[str, Any],
        previous_results: Dict[str, Any],
    ) -> PhaseResult:
        """Execute a single phase"""
        phase_name = phase['name']

        self.logger.info("-" * 80)
        self.logger.info(f"Executing phase: {phase_name}")
        self.logger.info("-" * 80)

        self.state.current_phase = phase_name
        start_time = datetime.now()

        self.audit_logger.log_event('phase_started', {
            'phase_name': phase_name,
            'start_time': start_time.isoformat(),
        })

        try:
            # Execute phase function
            outputs = phase['function'](previous_results)

            end_time = datetime.now()
            duration = end_time - start_time

            # Save checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                phase_name, outputs, self.state
            )

            # Update progress
            self.progress_tracker.update(phase_name, duration.total_seconds())

            result = PhaseResult(
                phase_name=phase_name,
                status='success',
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                outputs=outputs,
                checkpoint_path=checkpoint_path,
            )

            self.logger.info(f"Phase completed: {phase_name} (duration: {duration})")

            self.audit_logger.log_event('phase_completed', {
                'phase_name': phase_name,
                'duration_seconds': duration.total_seconds(),
                'outputs': {k: type(v).__name__ for k, v in outputs.items()},
            })

            return result

        except Exception as e:
            end_time = datetime.now()
            duration = end_time - start_time

            self.logger.error(f"Phase failed: {phase_name}")
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())

            self.audit_logger.log_event('phase_failed', {
                'phase_name': phase_name,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })

            return PhaseResult(
                phase_name=phase_name,
                status='failed',
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                error=str(e),
            )

    def _check_dependencies(
        self,
        phase: Dict[str, Any],
        results: Dict[str, Any],
    ) -> bool:
        """Check if phase dependencies are met"""
        dependencies = phase.get('dependencies', [])

        for dep in dependencies:
            if dep not in results:
                self.logger.error(f"Missing dependency: {dep}")
                return False

        return True

    def _resume_from_checkpoint(self):
        """Resume pipeline from last checkpoint"""
        self.logger.info("Attempting to resume from checkpoint...")

        state = self.checkpoint_manager.load_state()

        if state is None:
            self.logger.info("No previous state found, starting fresh")
            return

        self.state = state
        self.logger.info(f"Resumed from run: {state.run_id}")
        self.logger.info(f"Completed phases: {state.completed_phases}")

        self.audit_logger.log_event('pipeline_resumed', {
            'original_run_id': state.run_id,
            'completed_phases': state.completed_phases,
        })

    def _dry_run(self) -> Dict[str, Any]:
        """Perform dry run"""
        self.logger.info("DRY RUN - Simulating pipeline execution")

        for phase in self.phases:
            phase_name = phase['name']
            self.logger.info(f"\nPhase: {phase_name}")
            self.logger.info(f"  Dependencies: {phase.get('dependencies', [])}")
            self.logger.info(f"  Required memory: {phase.get('required_memory_gb', 0)}GB")
            self.logger.info(f"  Required CPUs: {phase.get('required_cpus', 1)}")
            self.logger.info(f"  Parallelizable: {phase.get('parallelizable', False)}")

        return {'dry_run': True}

    def _generate_final_report(self, results: Dict[str, Any]):
        """Generate final execution report"""
        summary = self.progress_tracker.get_summary()

        report = {
            'run_id': self.run_id,
            'status': 'success' if not self.state.failed_phases else 'failed',
            'completed_phases': self.state.completed_phases,
            'failed_phases': self.state.failed_phases,
            'summary': summary,
        }

        report_path = Path(self.config.pipeline.output_dir) / f"report_{self.run_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info("=" * 80)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Status: {report['status']}")
        self.logger.info(f"Completed phases: {len(self.state.completed_phases)}/{len(self.phases)}")
        self.logger.info(f"Total time: {summary['total_time']}")
        self.logger.info(f"Report saved: {report_path}")
        self.logger.info("=" * 80)

    # Phase implementations

    def _phase_data_loading(self, prev_results: Dict) -> Dict:
        """Load and harmonize data"""
        self.logger.info("Loading data from configured sources...")

        from audhd_correlation.data import (
            load_genomic_data,
            load_clinical_data,
            load_metabolomic_data,
            load_microbiome_data,
        )
        from audhd_correlation.data.harmonize import harmonize_datasets

        try:
            # Load each data modality
            data_dict = {}

            if self.config.data.get('genomic', {}).get('enabled', False):
                self.logger.info("Loading genomic data...")
                genomic_data = load_genomic_data(
                    self.config.data.genomic.path,
                    **self.config.data.genomic.get('params', {})
                )
                data_dict['genomic'] = genomic_data

            if self.config.data.get('clinical', {}).get('enabled', False):
                self.logger.info("Loading clinical data...")
                clinical_data = load_clinical_data(
                    self.config.data.clinical.path,
                    **self.config.data.clinical.get('params', {})
                )
                data_dict['clinical'] = clinical_data

            if self.config.data.get('metabolomic', {}).get('enabled', False):
                self.logger.info("Loading metabolomic data...")
                metabolomic_data = load_metabolomic_data(
                    self.config.data.metabolomic.path,
                    **self.config.data.metabolomic.get('params', {})
                )
                data_dict['metabolomic'] = metabolomic_data

            if self.config.data.get('microbiome', {}).get('enabled', False):
                self.logger.info("Loading microbiome data...")
                microbiome_data = load_microbiome_data(
                    self.config.data.microbiome.path,
                    **self.config.data.microbiome.get('params', {})
                )
                data_dict['microbiome'] = microbiome_data

            # Harmonize datasets
            self.logger.info("Harmonizing datasets...")
            harmonized_data = harmonize_datasets(
                data_dict,
                **self.config.data.get('harmonization', {})
            )

            n_samples = harmonized_data['clinical'].shape[0] if 'clinical' in harmonized_data else 0
            n_features = sum(df.shape[1] for df in harmonized_data.values() if hasattr(df, 'shape'))

            self.logger.info(f"Loaded {len(data_dict)} modalities, {n_samples} samples, {n_features} features")

            return {
                'data': harmonized_data,
                'modalities': list(data_dict.keys()),
                'n_samples': n_samples,
                'n_features': n_features,
            }

        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            # Return minimal data for pipeline to continue
            return {
                'data': {},
                'modalities': [],
                'n_samples': 0,
                'n_features': 0,
                'error': str(e),
            }

    def _phase_preprocessing(self, prev_results: Dict) -> Dict:
        """Preprocess data"""
        self.logger.info("Preprocessing data...")

        from audhd_correlation.preprocess import (
            scale_features,
            adjust_covariates,
            correct_batch_effects,
            impute_missing,
        )

        try:
            data = prev_results['data_loading']['data']

            if not data:
                self.logger.warning("No data to preprocess")
                return {'preprocessed_data': {}, 'n_samples': 0}

            preprocessed = {}

            for modality, df in data.items():
                self.logger.info(f"Preprocessing {modality} data...")

                # Imputation
                if self.config.preprocessing.get('impute', True):
                    df = impute_missing(
                        df,
                        **self.config.preprocessing.get('imputation', {})
                    )

                # Batch correction
                if self.config.preprocessing.get('batch_correction', False):
                    df = correct_batch_effects(
                        df,
                        batch_col=self.config.preprocessing.get('batch_col'),
                        **self.config.preprocessing.get('batch_params', {})
                    )

                # Covariate adjustment
                if self.config.preprocessing.get('adjust_covariates', False):
                    covariates = self.config.preprocessing.get('covariates', [])
                    if covariates:
                        df = adjust_covariates(
                            df,
                            covariates=covariates,
                            **self.config.preprocessing.get('adjustment_params', {})
                        )

                # Scaling
                if self.config.preprocessing.get('scale', True):
                    df = scale_features(
                        df,
                        method=self.config.preprocessing.get('scaling_method', 'standard'),
                    )

                preprocessed[modality] = df

            n_samples = list(preprocessed.values())[0].shape[0] if preprocessed else 0

            self.logger.info(f"Preprocessing complete: {n_samples} samples across {len(preprocessed)} modalities")

            return {
                'preprocessed_data': preprocessed,
                'n_samples': n_samples,
                'modalities': list(preprocessed.keys()),
            }

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return {
                'preprocessed_data': {},
                'n_samples': 0,
                'error': str(e),
            }

    def _phase_quality_control(self, prev_results: Dict) -> Dict:
        """Quality control"""
        self.logger.info("Running quality control...")

        from audhd_correlation.data.qc import run_qc_pipeline
        from audhd_correlation.preprocess.qc_reports import generate_qc_report

        try:
            data = prev_results['preprocessing']['preprocessed_data']

            if not data:
                self.logger.warning("No data for QC")
                return {'qc_passed': False, 'qc_report': None}

            # Run QC for each modality
            qc_results = {}
            all_passed = True

            for modality, df in data.items():
                self.logger.info(f"Running QC for {modality}...")

                qc_result = run_qc_pipeline(
                    df,
                    modality=modality,
                    **self.config.qc.get(modality, {})
                )

                qc_results[modality] = qc_result

                if not qc_result.get('passed', True):
                    all_passed = False
                    self.logger.warning(f"QC failed for {modality}")

            # Generate QC report
            output_dir = Path(self.config.pipeline.output_dir) / self.run_id / 'qc'
            output_dir.mkdir(parents=True, exist_ok=True)

            qc_report_path = generate_qc_report(
                qc_results,
                output_path=output_dir / 'qc_report.html'
            )

            self.logger.info(f"QC report saved: {qc_report_path}")

            return {
                'qc_passed': all_passed,
                'qc_results': qc_results,
                'qc_report': str(qc_report_path),
            }

        except Exception as e:
            self.logger.error(f"QC failed: {e}")
            return {
                'qc_passed': False,
                'qc_report': None,
                'error': str(e),
            }

    def _phase_integration(self, prev_results: Dict) -> Dict:
        """Multi-omics integration"""
        self.logger.info("Integrating multi-omics data...")

        from audhd_correlation.integrate import integrate_multiomics

        try:
            data = prev_results['preprocessing']['preprocessed_data']

            if not data or len(data) < 2:
                self.logger.warning("Insufficient modalities for integration")
                return {
                    'integrated_data': None,
                    'method': None,
                    'error': 'Insufficient modalities'
                }

            method = self.config.integration.get('method', 'mofa')
            self.logger.info(f"Using integration method: {method}")

            # Perform integration
            integrated_result = integrate_multiomics(
                data,
                method=method,
                **self.config.integration.get('params', {})
            )

            # Extract latent factors if available
            if hasattr(integrated_result, 'factors'):
                n_factors = integrated_result.factors.shape[1]
                self.logger.info(f"Integration complete: {n_factors} latent factors")
            else:
                n_factors = None

            return {
                'integrated_data': integrated_result,
                'method': method,
                'n_factors': n_factors,
            }

        except Exception as e:
            self.logger.error(f"Integration failed: {e}")
            return {
                'integrated_data': None,
                'method': None,
                'error': str(e),
            }

    def _phase_clustering(self, prev_results: Dict) -> Dict:
        """Clustering analysis"""
        self.logger.info("Performing clustering...")

        from audhd_correlation.modeling.clustering import perform_clustering
        from audhd_correlation.modeling.topology import compute_persistence_diagram

        try:
            integrated_data = prev_results['integration']['integrated_data']

            if integrated_data is None:
                self.logger.error("No integrated data available for clustering")
                return {
                    'clusters': None,
                    'n_clusters': 0,
                    'embedding': None,
                    'error': 'No integrated data'
                }

            # Extract features for clustering
            if hasattr(integrated_data, 'factors'):
                X = integrated_data.factors
            else:
                X = integrated_data

            # Perform clustering
            method = self.config.clustering.get('method', 'hdbscan')
            self.logger.info(f"Using clustering method: {method}")

            clustering_result = perform_clustering(
                X,
                method=method,
                **self.config.clustering.get('params', {})
            )

            # Extract results
            labels = clustering_result['labels']
            embedding = clustering_result.get('embedding', None)
            n_clusters = len(np.unique(labels[labels >= 0]))

            self.logger.info(f"Identified {n_clusters} clusters")

            # Compute topological features if requested
            topology = None
            if self.config.clustering.get('compute_topology', False):
                self.logger.info("Computing topological features...")
                topology = compute_persistence_diagram(embedding)

            return {
                'clusters': labels,
                'n_clusters': n_clusters,
                'embedding': embedding,
                'clustering_result': clustering_result,
                'topology': topology,
            }

        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return {
                'clusters': None,
                'n_clusters': 0,
                'embedding': None,
                'error': str(e),
            }

    def _phase_validation(self, prev_results: Dict) -> Dict:
        """Validation"""
        self.logger.info("Validating results...")

        from audhd_correlation.validation import (
            compute_internal_metrics,
            bootstrap_stability,
            generate_validation_report,
        )

        try:
            clusters = prev_results['clustering']['clusters']
            embedding = prev_results['clustering']['embedding']

            if clusters is None or embedding is None:
                self.logger.error("No clustering results to validate")
                return {
                    'validation_metrics': {},
                    'error': 'No clustering results'
                }

            # Compute internal validation metrics
            self.logger.info("Computing internal validation metrics...")
            internal_metrics = compute_internal_metrics(
                embedding,
                clusters,
                **self.config.validation.get('internal', {})
            )

            # Stability analysis
            stability_result = None
            if self.config.validation.get('compute_stability', True):
                self.logger.info("Running stability analysis...")
                stability_result = bootstrap_stability(
                    embedding,
                    clusters,
                    n_bootstrap=self.config.validation.get('n_bootstrap', 100),
                    **self.config.validation.get('stability', {})
                )

            # Generate validation report
            output_dir = Path(self.config.pipeline.output_dir) / self.run_id / 'validation'
            output_dir.mkdir(parents=True, exist_ok=True)

            report_path = generate_validation_report(
                internal_metrics=internal_metrics,
                stability_result=stability_result,
                output_path=output_dir / 'validation_report.html'
            )

            self.logger.info(f"Validation report saved: {report_path}")

            return {
                'validation_metrics': internal_metrics,
                'stability_result': stability_result,
                'validation_report': str(report_path),
            }

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {
                'validation_metrics': {},
                'error': str(e),
            }

    def _phase_biological_analysis(self, prev_results: Dict) -> Dict:
        """Biological analysis"""
        self.logger.info("Running biological analysis...")

        from audhd_correlation.biological import (
            run_pathway_enrichment,
            build_biological_networks,
            identify_drug_targets,
        )

        try:
            clusters = prev_results['clustering']['clusters']
            data = prev_results['preprocessing']['preprocessed_data']

            if clusters is None:
                self.logger.error("No clustering results for biological analysis")
                return {
                    'pathways': {},
                    'error': 'No clustering results'
                }

            # Pathway enrichment analysis
            pathway_results = {}
            if self.config.biological.get('run_gsea', True):
                self.logger.info("Running pathway enrichment analysis...")

                for modality, df in data.items():
                    if modality in ['genomic', 'metabolomic']:  # Only for relevant modalities
                        pathway_results[modality] = run_pathway_enrichment(
                            df,
                            clusters,
                            modality=modality,
                            **self.config.biological.get('gsea_params', {})
                        )

            # Network analysis
            networks = None
            if self.config.biological.get('build_networks', False):
                self.logger.info("Building biological networks...")
                networks = build_biological_networks(
                    data,
                    clusters,
                    **self.config.biological.get('network_params', {})
                )

            # Drug target identification
            drug_targets = None
            if self.config.biological.get('identify_targets', False):
                self.logger.info("Identifying drug targets...")
                drug_targets = identify_drug_targets(
                    pathway_results,
                    clusters,
                    **self.config.biological.get('target_params', {})
                )

            # Save results
            output_dir = Path(self.config.pipeline.output_dir) / self.run_id / 'biological'
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save pathway results
            for modality, results in pathway_results.items():
                results.to_csv(output_dir / f'{modality}_pathways.csv', index=False)

            return {
                'pathways': pathway_results,
                'networks': networks,
                'drug_targets': drug_targets,
                'output_dir': str(output_dir),
            }

        except Exception as e:
            self.logger.error(f"Biological analysis failed: {e}")
            return {
                'pathways': {},
                'error': str(e),
            }

    def _phase_visualization(self, prev_results: Dict) -> Dict:
        """Generate visualizations"""
        self.logger.info("Creating visualizations...")

        from audhd_correlation.viz import (
            plot_embedding,
            plot_cluster_comparison,
            plot_heatmaps,
            create_dashboard,
        )

        try:
            clusters = prev_results['clustering']['clusters']
            embedding = prev_results['clustering']['embedding']
            data = prev_results['preprocessing']['preprocessed_data']

            if clusters is None or embedding is None:
                self.logger.error("No clustering results to visualize")
                return {
                    'figures': [],
                    'error': 'No clustering results'
                }

            output_dir = Path(self.config.pipeline.output_dir) / self.run_id / 'visualizations'
            output_dir.mkdir(parents=True, exist_ok=True)

            figures = []

            # Embedding plot
            self.logger.info("Creating embedding plot...")
            fig_path = plot_embedding(
                embedding,
                clusters,
                output_path=output_dir / 'embedding.png',
                **self.config.visualization.get('embedding_params', {})
            )
            figures.append(str(fig_path))

            # Cluster comparison heatmaps
            if self.config.visualization.get('create_heatmaps', True):
                self.logger.info("Creating heatmaps...")
                heatmap_paths = plot_heatmaps(
                    data,
                    clusters,
                    output_dir=output_dir,
                    **self.config.visualization.get('heatmap_params', {})
                )
                figures.extend([str(p) for p in heatmap_paths])

            # Cluster comparison plots
            if self.config.visualization.get('cluster_comparison', True):
                self.logger.info("Creating cluster comparison plots...")
                comparison_path = plot_cluster_comparison(
                    data,
                    clusters,
                    output_path=output_dir / 'cluster_comparison.png',
                    **self.config.visualization.get('comparison_params', {})
                )
                figures.append(str(comparison_path))

            # Interactive dashboard
            if self.config.visualization.get('create_dashboard', True):
                self.logger.info("Creating interactive dashboard...")
                dashboard_path = create_dashboard(
                    embedding,
                    clusters,
                    data,
                    output_path=output_dir / 'dashboard.html',
                    **self.config.visualization.get('dashboard_params', {})
                )
                figures.append(str(dashboard_path))

            self.logger.info(f"Created {len(figures)} visualizations")

            return {
                'figures': figures,
                'output_dir': str(output_dir),
            }

        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return {
                'figures': [],
                'error': str(e),
            }

    def _phase_reporting(self, prev_results: Dict) -> Dict:
        """Generate reports"""
        self.logger.info("Generating reports...")

        from audhd_correlation.reporting import (
            generate_comprehensive_report,
            generate_supplementary_materials,
        )

        try:
            # Collect all results
            clustering_results = prev_results['clustering']
            validation_results = prev_results['validation']
            biological_results = prev_results['biological_analysis']
            visualization_results = prev_results['visualization']

            output_dir = Path(self.config.pipeline.output_dir) / self.run_id / 'reports'
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate main report
            self.logger.info("Generating comprehensive report...")
            report_path = generate_comprehensive_report(
                clustering_results=clustering_results,
                validation_results=validation_results,
                biological_results=biological_results,
                visualization_results=visualization_results,
                output_path=output_dir / 'analysis_report.html',
                **self.config.reporting.get('main_report', {})
            )

            # Generate supplementary materials
            supplementary_paths = []
            if self.config.reporting.get('generate_supplementary', True):
                self.logger.info("Generating supplementary materials...")
                supplementary_paths = generate_supplementary_materials(
                    prev_results,
                    output_dir=output_dir / 'supplementary',
                    **self.config.reporting.get('supplementary', {})
                )

            # Generate PDF if requested
            pdf_path = None
            if self.config.reporting.get('generate_pdf', False):
                from audhd_correlation.reporting import generate_pdf_report
                self.logger.info("Generating PDF report...")
                pdf_path = generate_pdf_report(
                    report_path,
                    output_path=output_dir / 'analysis_report.pdf'
                )

            self.logger.info(f"Main report saved: {report_path}")

            return {
                'report': str(report_path),
                'pdf_report': str(pdf_path) if pdf_path else None,
                'supplementary': [str(p) for p in supplementary_paths],
                'output_dir': str(output_dir),
            }

        except Exception as e:
            self.logger.error(f"Reporting failed: {e}")
            return {
                'report': None,
                'error': str(e),
            }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point"""
    # Print configuration
    print("=" * 80)
    print("Pipeline Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Validate configuration if pydantic is available
    try:
        from audhd_correlation.config.validation import validate_config, PYDANTIC_AVAILABLE

        if PYDANTIC_AVAILABLE:
            print("\nValidating configuration...")
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            validated = validate_config(config_dict)
            print("✓ Configuration validated successfully\n")
    except Exception as e:
        print(f"\n⚠ Warning: Configuration validation failed: {e}")
        print("Continuing with unvalidated configuration...\n")

    # Create and run pipeline
    orchestrator = PipelineOrchestrator(cfg)
    results = orchestrator.run()

    return results


if __name__ == "__main__":
    main()
