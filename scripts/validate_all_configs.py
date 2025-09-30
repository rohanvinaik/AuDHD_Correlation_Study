#!/usr/bin/env python3
"""
Validate all YAML configuration files against the Pydantic schema

This script finds all .yaml files in the configs/ directory and validates them
against the PipelineConfig schema to ensure they're valid and type-safe.
"""
import sys
from pathlib import Path
from typing import List, Tuple
import traceback

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audhd_correlation.config.validation import validate_config_file


def find_all_yaml_configs(config_dir: Path) -> List[Path]:
    """Find all YAML files in config directory"""
    yaml_files = list(config_dir.glob("**/*.yaml"))
    yaml_files.extend(config_dir.glob("**/*.yml"))
    return sorted(yaml_files)


def validate_single_config(config_path: Path) -> Tuple[bool, str]:
    """
    Validate a single config file

    Returns:
        (success, message) tuple
    """
    try:
        config = validate_config_file(config_path)
        return True, f"✓ Valid (pipeline: {config.pipeline.name})"
    except FileNotFoundError as e:
        return False, f"✗ File not found: {e}"
    except Exception as e:
        # Get the specific validation error
        error_msg = str(e)
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        return False, f"✗ Validation failed: {error_msg}"


def main():
    """Main validation script"""
    if RICH_AVAILABLE:
        console = Console()
    else:
        console = None

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_dir = project_root / "configs"

    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    # Find all YAML files
    yaml_files = find_all_yaml_configs(config_dir)

    if not yaml_files:
        print(f"No YAML files found in {config_dir}")
        sys.exit(0)

    if console:
        console.print(f"\n[bold]Validating {len(yaml_files)} configuration files[/bold]\n")
    else:
        print(f"\nValidating {len(yaml_files)} configuration files\n")

    # Validate each file
    results = []
    for config_path in yaml_files:
        rel_path = config_path.relative_to(project_root)
        success, message = validate_single_config(config_path)
        results.append((rel_path, success, message))

        # Print progress
        if not console:
            status = "✓" if success else "✗"
            print(f"{status} {rel_path}: {message}")

    # Summary
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful

    if console:
        # Create results table
        table = Table(title="Validation Results", show_header=True)
        table.add_column("Config File", style="cyan", no_wrap=False)
        table.add_column("Status", style="white", width=10)
        table.add_column("Details", style="white", no_wrap=False)

        for rel_path, success, message in results:
            status_style = "green" if success else "red"
            table.add_row(
                str(rel_path),
                f"[{status_style}]{'PASS' if success else 'FAIL'}[/{status_style}]",
                message
            )

        console.print(table)
        console.print()

        # Summary panel
        if failed == 0:
            panel_style = "green"
            summary_msg = f"✓ All {successful} configuration files are valid!"
        else:
            panel_style = "red"
            summary_msg = f"✗ {failed} of {len(results)} configuration files failed validation"

        console.print(Panel(summary_msg, style=panel_style, expand=False))
        console.print()

    else:
        print(f"\n{'=' * 60}")
        print(f"Summary: {successful} passed, {failed} failed")
        print(f"{'=' * 60}\n")

    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)