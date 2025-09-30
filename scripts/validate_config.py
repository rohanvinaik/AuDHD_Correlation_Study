#!/usr/bin/env python
"""Validate pipeline configuration files

Usage:
    python scripts/validate_config.py configs/config.yaml
    python scripts/validate_config.py --check-resources
    python scripts/validate_config.py --check-data-paths
"""
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf, DictConfig
from src.audhd_correlation.config.validation import (
    validate_config,
    check_resource_availability,
    validate_data_paths,
    PYDANTIC_AVAILABLE,
)


def load_hydra_config(config_path: Path) -> DictConfig:
    """Load configuration with Hydra composition"""
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra

    # Clean up any existing Hydra instance
    GlobalHydra.instance().clear()

    # Initialize Hydra with config directory
    config_dir = str(config_path.parent.absolute())
    config_name = config_path.stem

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)

    return cfg


def print_validation_result(result: Dict[str, Any], title: str):
    """Pretty print validation results"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

    for key, value in result.items():
        if isinstance(value, bool):
            status = "✓" if value else "✗"
            print(f"{status} {key}: {value}")
        elif value is None:
            print(f"- {key}: N/A")
        else:
            print(f"  {key}: {value}")

    print("=" * 80)


def validate_config_file(
    config_path: Path,
    check_resources: bool = False,
    check_paths: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Validate configuration file

    Args:
        config_path: Path to config file
        check_resources: Whether to check resource availability
        check_paths: Whether to check data path existence
        verbose: Print detailed information

    Returns:
        True if validation passes
    """
    print(f"\nValidating configuration: {config_path}")

    # Check if pydantic is available
    if not PYDANTIC_AVAILABLE:
        print("\n⚠ Warning: pydantic not installed, skipping schema validation")
        print("  Install with: pip install pydantic")
    else:
        # Load and validate config
        try:
            if config_path.name in ['config.yaml', 'pipeline.yaml']:
                # Use Hydra composition for main configs
                cfg = load_hydra_config(config_path)
            else:
                # Direct load for other configs
                cfg = OmegaConf.load(config_path)

            # Resolve config, but handle Hydra-specific interpolations
            try:
                config_dict = OmegaConf.to_container(cfg, resolve=True)
            except Exception as e:
                # Try resolving without Hydra runtime values
                print(f"⚠ Warning: Some interpolations couldn't be resolved: {e}")
                print("  Using partially resolved config for validation...")
                config_dict = OmegaConf.to_container(cfg, resolve=False)

            if verbose:
                print("\nLoaded configuration:")
                print(json.dumps(config_dict, indent=2, default=str))

            # Validate schema
            validated_config = validate_config(config_dict)
            print("✓ Configuration schema is valid")

            # Check resources
            if check_resources:
                resource_check = check_resource_availability(validated_config.compute)
                print_validation_result(resource_check, "Resource Availability Check")

                if not all([
                    resource_check['memory_available'],
                    resource_check['cpus_available']
                ]):
                    print("\n⚠ Warning: Requested resources exceed available resources")

            # Check data paths
            if check_paths:
                path_check = validate_data_paths(validated_config.data)
                print_validation_result(path_check, "Data Path Existence Check")

                missing_paths = [
                    k for k, v in path_check.items()
                    if v is False  # False means enabled but missing
                ]

                if missing_paths:
                    print(f"\n⚠ Warning: Missing data paths: {', '.join(missing_paths)}")

            return True

        except Exception as e:
            print(f"\n✗ Validation failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate pipeline configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'config_path',
        type=Path,
        nargs='?',
        default=Path('configs/config.yaml'),
        help='Path to configuration file (default: configs/config.yaml)',
    )

    parser.add_argument(
        '--check-resources',
        action='store_true',
        help='Check if requested resources are available',
    )

    parser.add_argument(
        '--check-data-paths',
        action='store_true',
        help='Check if data paths exist',
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed information',
    )

    parser.add_argument(
        '--all-configs',
        action='store_true',
        help='Validate all config files in configs/ directory',
    )

    args = parser.parse_args()

    # Validate all configs if requested
    if args.all_configs:
        configs_dir = Path('configs')
        config_files = list(configs_dir.rglob('*.yaml'))

        print(f"\nFound {len(config_files)} configuration files")

        results = []
        for config_file in sorted(config_files):
            # Skip sweep configs as they have different structure
            if 'sweep' in str(config_file):
                print(f"\nSkipping sweep config: {config_file}")
                continue

            result = validate_config_file(
                config_file,
                check_resources=args.check_resources,
                check_paths=args.check_data_paths,
                verbose=args.verbose,
            )
            results.append((config_file, result))

        # Summary
        print("\n" + "=" * 80)
        print("  Validation Summary")
        print("=" * 80)

        passed = sum(1 for _, r in results if r)
        failed = len(results) - passed

        for config_file, result in results:
            status = "✓" if result else "✗"
            print(f"{status} {config_file}")

        print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
        print("=" * 80)

        sys.exit(0 if failed == 0 else 1)

    # Validate single config
    else:
        if not args.config_path.exists():
            print(f"Error: Configuration file not found: {args.config_path}")
            sys.exit(1)

        result = validate_config_file(
            args.config_path,
            check_resources=args.check_resources,
            check_paths=args.check_data_paths,
            verbose=args.verbose,
        )

        sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()