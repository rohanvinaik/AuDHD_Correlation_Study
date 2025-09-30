"""Configuration validation using dataclasses and pydantic

Provides schema validation for all configuration options with proper type checking
and constraints.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from pathlib import Path

try:
    from pydantic import BaseModel, Field, validator, root_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback to basic validation without pydantic
    PYDANTIC_AVAILABLE = False
    BaseModel = object


# Data Configuration Schemas

class GenomicDataConfig(BaseModel):
    """Genomic data configuration"""
    enabled: bool = True
    path: str
    params: Dict = Field(default_factory=dict)

    @validator('path')
    def validate_path(cls, v):
        if v and not v.startswith(('./', '/', '~')):
            raise ValueError(f"Path must be absolute or relative: {v}")
        return v


class ClinicalDataConfig(BaseModel):
    """Clinical data configuration"""
    enabled: bool = True
    path: str
    params: Dict = Field(default_factory=dict)


class MetabolomicDataConfig(BaseModel):
    """Metabolomic data configuration"""
    enabled: bool = True
    path: Optional[str] = None
    params: Dict = Field(default_factory=dict)


class MicrobiomeDataConfig(BaseModel):
    """Microbiome data configuration"""
    enabled: bool = True
    path: Optional[str] = None
    params: Dict = Field(default_factory=dict)


class DataConfig(BaseModel):
    """Data loading configuration"""
    genomic: GenomicDataConfig
    clinical: ClinicalDataConfig
    metabolomic: Optional[MetabolomicDataConfig] = None
    microbiome: Optional[MicrobiomeDataConfig] = None
    harmonization: Dict = Field(default_factory=dict)

    @root_validator(skip_on_failure=True)
    def validate_at_least_one_enabled(cls, values):
        """Ensure at least one data type is enabled"""
        enabled_types = []
        for key in ['genomic', 'clinical', 'metabolomic', 'microbiome']:
            config = values.get(key)
            if config and config.enabled:
                enabled_types.append(key)

        if not enabled_types:
            raise ValueError("At least one data type must be enabled")

        return values


# Preprocessing Configuration Schemas

class ImputationConfig(BaseModel):
    """Imputation configuration"""
    method: Literal['mean', 'median', 'knn', 'iterative'] = 'knn'
    n_neighbors: int = Field(default=5, ge=1, le=20)
    weights: Literal['uniform', 'distance'] = 'distance'


class BatchCorrectionConfig(BaseModel):
    """Batch correction configuration"""
    method: Literal['combat', 'limma', 'none'] = 'combat'
    covariates: List[str] = Field(default_factory=list)
    parametric: bool = True


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration"""
    impute: bool = True
    imputation: ImputationConfig = Field(default_factory=ImputationConfig)

    batch_correction: bool = True
    batch_col: Optional[str] = 'site'
    batch_params: BatchCorrectionConfig = Field(default_factory=BatchCorrectionConfig)

    adjust_covariates: bool = True
    covariates: List[str] = Field(default_factory=list)
    adjustment_params: Dict = Field(default_factory=dict)

    scale: bool = True
    scaling_method: Literal['standard', 'minmax', 'robust'] = 'standard'

    qc: Dict = Field(default_factory=dict)

    @validator('batch_col')
    def validate_batch_col(cls, v, values):
        """Validate batch_col is provided if batch_correction is True"""
        if values.get('batch_correction') and not v:
            raise ValueError("batch_col must be specified when batch_correction=True")
        return v


# Integration Configuration Schemas

class IntegrationParamsConfig(BaseModel):
    """Integration method parameters"""
    n_factors: int = Field(default=15, ge=2, le=100)
    convergence_mode: Literal['fast', 'medium', 'slow'] = 'fast'
    max_iter: int = Field(default=1000, ge=10, le=10000)
    tolerance: float = Field(default=0.01, gt=0, lt=1)


class IntegrationConfig(BaseModel):
    """Integration configuration"""
    method: Literal['mofa', 'pca', 'cca', 'mcia'] = 'mofa'
    params: IntegrationParamsConfig = Field(default_factory=IntegrationParamsConfig)


# Clustering Configuration Schemas

class ClusteringParamsConfig(BaseModel):
    """Clustering parameters"""
    # HDBSCAN parameters
    min_cluster_size: int = Field(default=30, ge=5, le=1000)
    min_samples: int = Field(default=10, ge=1, le=100)
    cluster_selection_epsilon: float = Field(default=0.0, ge=0, le=1)
    cluster_selection_method: Literal['eom', 'leaf'] = 'eom'
    metric: str = 'euclidean'

    # UMAP parameters
    embedding_method: Literal['umap', 'tsne', 'pca'] = 'umap'
    umap_n_neighbors: int = Field(default=15, ge=2, le=100)
    umap_min_dist: float = Field(default=0.1, ge=0, lt=1)
    umap_n_components: int = Field(default=2, ge=2, le=10)
    umap_metric: str = 'euclidean'

    @validator('min_samples')
    def validate_min_samples(cls, v, values):
        """Ensure min_samples <= min_cluster_size"""
        min_cluster_size = values.get('min_cluster_size', 30)
        if v > min_cluster_size:
            raise ValueError(
                f"min_samples ({v}) cannot exceed min_cluster_size ({min_cluster_size})"
            )
        return v


class ClusteringConfig(BaseModel):
    """Clustering configuration"""
    method: Literal['hdbscan', 'kmeans', 'dbscan', 'hierarchical'] = 'hdbscan'
    compute_topology: bool = True
    params: ClusteringParamsConfig = Field(default_factory=ClusteringParamsConfig)


# Validation Configuration Schemas

class ValidationConfig(BaseModel):
    """Validation configuration"""
    compute_stability: bool = True
    n_bootstrap: int = Field(default=100, ge=10, le=1000)

    internal: Dict = Field(default_factory=dict)
    stability: Dict = Field(default_factory=dict)
    external: Dict = Field(default_factory=dict)
    cross_cohort: Dict = Field(default_factory=dict)
    ancestry_stratified: Dict = Field(default_factory=dict)
    prospective: Dict = Field(default_factory=dict)


# Compute Configuration Schemas

class ResourceConfig(BaseModel):
    """Resource configuration"""
    max_memory_gb: int = Field(default=32, ge=1, le=1024)
    max_cpus: int = Field(default=8, ge=1, le=256)
    use_gpu: bool = False


class ParallelizationConfig(BaseModel):
    """Parallelization configuration"""
    n_jobs: int = Field(default=4, ge=1, le=256)
    backend: Literal['loky', 'threading', 'multiprocessing'] = 'loky'
    prefer: Literal['processes', 'threads'] = 'processes'
    distributed: bool = False


class ComputeConfig(BaseModel):
    """Compute configuration"""
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    parallelization: ParallelizationConfig = Field(default_factory=ParallelizationConfig)
    cache: Dict = Field(default_factory=dict)

    @validator('parallelization')
    def validate_parallelization(cls, v, values):
        """Ensure n_jobs doesn't exceed max_cpus"""
        resources = values.get('resources')
        if resources and v.n_jobs > resources.max_cpus:
            raise ValueError(
                f"n_jobs ({v.n_jobs}) cannot exceed max_cpus ({resources.max_cpus})"
            )
        return v


# Pipeline Configuration Schemas

class PipelineConfig(BaseModel):
    """Pipeline configuration"""
    name: str
    version: str = '1.0.0'
    description: Optional[str] = None

    checkpoint_dir: str = './checkpoints'
    audit_dir: str = './audit'
    output_dir: str = './outputs'
    log_dir: str = './logs'

    dry_run: bool = False
    resume: bool = False
    continue_on_error: bool = False
    verbose: bool = False


# Main Configuration Schema

class MainConfig(BaseModel):
    """Main configuration schema"""
    pipeline: PipelineConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    integration: IntegrationConfig
    clustering: ClusteringConfig
    validation: ValidationConfig
    compute: ComputeConfig

    seed: int = Field(default=42, ge=0)

    # Optional sections
    biological: Dict = Field(default_factory=dict)
    visualization: Dict = Field(default_factory=dict)
    reporting: Dict = Field(default_factory=dict)
    experiment: Dict = Field(default_factory=dict)


# Validation functions

def validate_config(config_dict: Dict) -> MainConfig:
    """
    Validate configuration dictionary

    Args:
        config_dict: Configuration dictionary

    Returns:
        Validated MainConfig object

    Raises:
        ValidationError: If configuration is invalid
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "pydantic is required for config validation. "
            "Install with: pip install pydantic"
        )

    return MainConfig(**config_dict)


def validate_config_file(config_path: Path) -> MainConfig:
    """
    Validate configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Validated MainConfig object
    """
    from omegaconf import OmegaConf

    config_dict = OmegaConf.to_container(
        OmegaConf.load(config_path),
        resolve=True
    )

    return validate_config(config_dict)


# Convenience functions for checking specific configs

def check_resource_availability(config: ComputeConfig) -> Dict[str, bool]:
    """
    Check if requested resources are available

    Args:
        config: Compute configuration

    Returns:
        Dictionary of availability checks
    """
    import psutil
    import multiprocessing as mp

    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    available_cpus = mp.cpu_count()

    return {
        'memory_available': available_memory >= config.resources.max_memory_gb,
        'cpus_available': available_cpus >= config.resources.max_cpus,
        'available_memory_gb': available_memory,
        'available_cpus': available_cpus,
        'requested_memory_gb': config.resources.max_memory_gb,
        'requested_cpus': config.resources.max_cpus,
    }


def validate_data_paths(config: DataConfig) -> Dict[str, bool]:
    """
    Check if data paths exist

    Args:
        config: Data configuration

    Returns:
        Dictionary of path existence checks
    """
    from pathlib import Path

    results = {}

    for data_type in ['genomic', 'clinical', 'metabolomic', 'microbiome']:
        data_config = getattr(config, data_type, None)
        if data_config and data_config.enabled and data_config.path:
            path = Path(data_config.path)
            results[data_type] = path.exists()
        else:
            results[data_type] = None  # Not applicable

    return results