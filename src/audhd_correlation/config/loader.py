"""
Configuration loader with YAML support and Pydantic validation
"""
from pathlib import Path
from typing import Union
import yaml
from rich.console import Console

from .schema import AppConfig

console = Console()


def load_yaml(path: Union[str, Path]) -> dict:
    """Load YAML file"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return data or {}


def load_config(config_path: Union[str, Path]) -> AppConfig:
    """
    Load and validate configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Validated AppConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config validation fails
    """
    config_path = Path(config_path)

    console.print(f"[dim]Loading config: {config_path}[/dim]")

    # Load YAML
    config_dict = load_yaml(config_path)

    # Handle Hydra-style defaults
    if "defaults" in config_dict:
        defaults = config_dict.pop("defaults")
        config_dict = merge_defaults(config_path.parent, defaults, config_dict)

    # Validate with Pydantic
    try:
        config = AppConfig(**config_dict)
        console.print("[green]✓[/green] Config validated successfully")
        return config
    except Exception as e:
        console.print(f"[red]✗ Config validation failed:[/red] {e}")
        raise


def merge_defaults(config_dir: Path, defaults: list, base_config: dict) -> dict:
    """
    Merge Hydra-style default configs

    Args:
        config_dir: Base config directory
        defaults: List of default config specifications
        base_config: Base configuration dict

    Returns:
        Merged configuration dict
    """
    merged = {}

    # Load each default config
    for default in defaults:
        if isinstance(default, str):
            # Simple default: "data: spark"
            parts = default.split(":")
            if len(parts) == 2:
                category, name = parts[0].strip(), parts[1].strip()
                default_path = config_dir / category / f"{name}.yaml"
            else:
                default_path = config_dir / f"{default}.yaml"
        elif isinstance(default, dict):
            # Dict default: {"data": "spark"}
            for category, name in default.items():
                default_path = config_dir / category / f"{name}.yaml"
        else:
            continue

        if default_path.exists():
            default_config = load_yaml(default_path)
            # Merge into category
            if isinstance(default, str) and ":" in default:
                category = default.split(":")[0].strip()
                if category not in merged:
                    merged[category] = {}
                merged[category].update(default_config)
            else:
                merged.update(default_config)

    # Override with base config
    merged.update(base_config)

    return merged


def save_config(config: AppConfig, path: Union[str, Path]) -> None:
    """
    Save config to YAML file

    Args:
        config: AppConfig instance
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    config_dict = config.model_dump()

    # Write YAML
    with open(path, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] Config saved: {path}")