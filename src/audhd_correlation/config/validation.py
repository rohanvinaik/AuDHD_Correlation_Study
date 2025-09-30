"""Configuration validation utilities

This module provides high-level validation functions that use the Pydantic models
defined in schema.py. It adds path validation, resource checking, and helpful error
messages on top of the base schema validation.
"""
from pathlib import Path
from typing import Dict, Any
import warnings

try:
    from .schema import PipelineConfig
    from omegaconf import OmegaConf, DictConfig
    VALIDATION_AVAILABLE = True
except ImportError as e:
    VALIDATION_AVAILABLE = False
    _import_error = e


def validate_config(config_dict: Dict[str, Any]) -> "PipelineConfig":
    """
    Validate configuration dictionary against schema

    Args:
        config_dict: Configuration dictionary from YAML/Hydra

    Returns:
        Validated PipelineConfig object

    Raises:
        ImportError: If pydantic not available
        ValidationError: If configuration is invalid
    """
    if not VALIDATION_AVAILABLE:
        raise ImportError(
            f"Configuration validation requires pydantic and omegaconf. "
            f"Original error: {_import_error}"
        )

    from .schema import PipelineConfig

    # Convert OmegaConf to dict if needed
    if isinstance(config_dict, DictConfig):
        config_dict = OmegaConf.to_container(config_dict, resolve=True)

    return PipelineConfig(**config_dict)


def validate_config_file(config_path: Path) -> "PipelineConfig":
    """
    Validate configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated PipelineConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If configuration is invalid
    """
    if not VALIDATION_AVAILABLE:
        raise ImportError(
            f"Configuration validation requires pydantic and omegaconf. "
            f"Original error: {_import_error}"
        )

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config_dict = OmegaConf.to_container(
        OmegaConf.load(config_path),
        resolve=True
    )

    return validate_config(config_dict)


def validate_data_paths(config: "PipelineConfig", strict: bool = False) -> Dict[str, Any]:
    """
    Validate that data paths exist and are accessible

    Args:
        config: Validated pipeline configuration
        strict: If True, raise error on missing paths. If False, return warnings.

    Returns:
        Dictionary with path validation results

    Raises:
        FileNotFoundError: If strict=True and paths missing
    """
    results = {
        'all_valid': True,
        'missing_paths': [],
        'warnings': []
    }

    # Check genomic data path
    if config.data.genomic.enabled and config.data.genomic.path:
        genomic_path = Path(config.data.genomic.path)
        if not genomic_path.exists():
            results['all_valid'] = False
            results['missing_paths'].append(f"genomic: {genomic_path}")
            if strict:
                raise FileNotFoundError(f"Genomic data file not found: {genomic_path}")
            else:
                results['warnings'].append(f"⚠️  Genomic data file not found: {genomic_path}")

    # Check clinical data path
    if config.data.clinical.enabled and config.data.clinical.path:
        clinical_path = Path(config.data.clinical.path)
        if not clinical_path.exists():
            results['all_valid'] = False
            results['missing_paths'].append(f"clinical: {clinical_path}")
            if strict:
                raise FileNotFoundError(f"Clinical data file not found: {clinical_path}")
            else:
                results['warnings'].append(f"⚠️  Clinical data file not found: {clinical_path}")

    # Check metabolomic data path (optional)
    if hasattr(config.data, 'metabolomic') and config.data.metabolomic:
        if config.data.metabolomic.enabled and config.data.metabolomic.path:
            metab_path = Path(config.data.metabolomic.path)
            if not metab_path.exists():
                results['warnings'].append(
                    f"⚠️  Metabolomic data file not found: {metab_path} (optional)"
                )

    # Check microbiome data path (optional)
    if hasattr(config.data, 'microbiome') and config.data.microbiome:
        if config.data.microbiome.enabled and config.data.microbiome.path:
            micro_path = Path(config.data.microbiome.path)
            if not micro_path.exists():
                results['warnings'].append(
                    f"⚠️  Microbiome data file not found: {micro_path} (optional)"
                )

    # Check output directories are writable
    for dir_attr in ['checkpoint_dir', 'audit_dir', 'output_dir', 'log_dir']:
        dir_path = Path(getattr(config.pipeline, dir_attr))
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                results['warnings'].append(f"✓ Created directory: {dir_path}")
            except PermissionError:
                results['all_valid'] = False
                results['warnings'].append(f"⚠️  Cannot create directory: {dir_path}")

    return results


def check_resource_availability(config: "PipelineConfig") -> Dict[str, Any]:
    """
    Check if requested computational resources are available

    Args:
        config: Validated pipeline configuration

    Returns:
        Dictionary with resource availability information
    """
    try:
        import psutil
        import multiprocessing as mp
    except ImportError:
        return {
            'check_available': False,
            'message': 'psutil not installed, cannot check resources'
        }

    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
    available_cpus = mp.cpu_count()

    requested_memory = config.compute.resources.max_memory_gb
    requested_cpus = config.compute.parallelization.n_jobs

    results = {
        'check_available': True,
        'memory': {
            'available_gb': round(available_memory_gb, 2),
            'requested_gb': requested_memory,
            'sufficient': available_memory_gb >= requested_memory,
        },
        'cpus': {
            'available': available_cpus,
            'requested': requested_cpus,
            'sufficient': available_cpus >= requested_cpus,
        },
        'warnings': []
    }

    if not results['memory']['sufficient']:
        results['warnings'].append(
            f"⚠️  Insufficient memory: requested {requested_memory}GB, "
            f"available {available_memory_gb:.2f}GB"
        )

    if not results['cpus']['sufficient']:
        results['warnings'].append(
            f"⚠️  Insufficient CPUs: requested {requested_cpus}, "
            f"available {available_cpus}"
        )

    if config.compute.resources.use_gpu:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            results['gpu'] = {
                'requested': True,
                'available': gpu_available,
                'count': torch.cuda.device_count() if gpu_available else 0
            }
            if not gpu_available:
                results['warnings'].append("⚠️  GPU requested but not available")
        except ImportError:
            results['gpu'] = {
                'requested': True,
                'available': False,
                'message': 'PyTorch not installed'
            }
            results['warnings'].append("⚠️  GPU requested but PyTorch not installed")

    return results


def validate_config_comprehensive(
    config: "PipelineConfig",
    check_paths: bool = True,
    check_resources: bool = True,
    strict_paths: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive configuration validation

    Performs schema validation, path checking, and resource checking.

    Args:
        config: Configuration to validate (can be dict, DictConfig, or PipelineConfig)
        check_paths: Whether to check data paths exist
        check_resources: Whether to check computational resources
        strict_paths: If True, raise error on missing paths

    Returns:
        Dictionary with all validation results

    Raises:
        ValidationError: If schema validation fails
        FileNotFoundError: If strict_paths=True and paths missing
    """
    from .schema import PipelineConfig

    # Convert to PipelineConfig if needed
    if not isinstance(config, PipelineConfig):
        config = validate_config(config)

    results = {
        'schema_valid': True,
        'all_warnings': [],
        'all_errors': []
    }

    # Path validation
    if check_paths:
        path_results = validate_data_paths(config, strict=strict_paths)
        results['paths'] = path_results
        results['all_warnings'].extend(path_results.get('warnings', []))
        if not path_results['all_valid']:
            results['all_errors'].extend([
                f"Missing data paths: {', '.join(path_results['missing_paths'])}"
            ])

    # Resource validation
    if check_resources:
        resource_results = check_resource_availability(config)
        results['resources'] = resource_results
        results['all_warnings'].extend(resource_results.get('warnings', []))

    # Print warnings
    if results['all_warnings']:
        warnings.warn(
            "Configuration validation warnings:\n" +
            "\n".join(results['all_warnings'])
        )

    return results


# Convenience function for common use case
def load_and_validate_config(
    config_path: Path,
    check_paths: bool = True,
    check_resources: bool = True,
    strict: bool = False
) -> "PipelineConfig":
    """
    Load configuration from file and perform comprehensive validation

    This is the recommended entry point for loading configurations.

    Args:
        config_path: Path to YAML configuration file
        check_paths: Whether to validate data paths exist
        check_resources: Whether to check computational resources
        strict: If True, raise errors on validation failures

    Returns:
        Validated PipelineConfig object

    Raises:
        FileNotFoundError: If config file or data files not found (when strict=True)
        ValidationError: If configuration schema is invalid

    Example:
        >>> config = load_and_validate_config('config.yaml')
        >>> print(f"Running pipeline: {config.pipeline.name}")
        >>> results = run_pipeline(config)
    """
    # Load and validate schema
    config = validate_config_file(config_path)

    # Comprehensive validation
    validation_results = validate_config_comprehensive(
        config,
        check_paths=check_paths,
        check_resources=check_resources,
        strict_paths=strict
    )

    if validation_results['all_errors'] and strict:
        raise ValueError(
            "Configuration validation failed:\n" +
            "\n".join(validation_results['all_errors'])
        )

    return config