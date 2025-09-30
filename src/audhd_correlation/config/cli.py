"""Command-line interface for the AuDHD correlation pipeline

Provides a Typer-based CLI for running the pipeline with Hydra configuration.
"""
from pathlib import Path
from typing import Optional, List
import sys

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

from .validation import load_and_validate_config

# Create Typer app
app = typer.Typer(
    name="audhd-pipeline",
    help="AuDHD Correlation Study: Multi-Omics Integration Pipeline",
    add_completion=False
)

console = Console() if CLI_AVAILABLE else None


@app.command()
def run(
    config: Path = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to configuration file"
    ),
    steps: Optional[str] = typer.Option(
        None,
        "--steps", "-s",
        help="Comma-separated list of steps to run (default: all)"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without executing"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Strict validation mode (fail on warnings)"
    ),
):
    """
    Run the complete pipeline or specific steps

    Examples:
        audhd-pipeline run --config config.yaml
        audhd-pipeline run --config config.yaml --steps preprocess,cluster
        audhd-pipeline run --dry-run
    """
    if not CLI_AVAILABLE:
        typer.echo("Error: CLI requires typer and rich. Install with: pip install typer rich")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]AuDHD Correlation Study Pipeline[/bold blue]\n")

    # Load and validate configuration
    try:
        console.print(f"Loading configuration from: [cyan]{config}[/cyan]")
        cfg = load_and_validate_config(
            config,
            check_paths=True,
            check_resources=True,
            strict=strict
        )
        console.print("[green]✓[/green] Configuration validated successfully\n")

    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration validation failed: {e}")
        raise typer.Exit(1)

    # Parse steps
    if steps:
        step_list = [s.strip() for s in steps.split(',')]
    else:
        step_list = ['download', 'build_features', 'integrate', 'cluster', 'validate', 'report']

    # Override config with CLI options
    if dry_run:
        cfg.pipeline.dry_run = True
    if verbose:
        cfg.pipeline.verbose = True

    # Show configuration summary
    _show_config_summary(cfg, step_list)

    if dry_run:
        console.print("\n[yellow]Dry run mode - no changes will be made[/yellow]")
        return

    # Run pipeline
    try:
        from ..pipelines.run_all import run

        console.print("\n[bold]Starting pipeline execution...[/bold]\n")

        results = run(cfg, steps=step_list)

        console.print("\n[green]✓[/green] Pipeline completed successfully!")

        # Show summary
        if results:
            _show_results_summary(results)

    except Exception as e:
        console.print(f"\n[red]✗[/red] Pipeline failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def preprocess(
    config: Path = typer.Option("config.yaml", "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Preprocess data only"""
    run(config=config, steps="preprocess", verbose=verbose)


@app.command()
def integrate(
    config: Path = typer.Option("config.yaml", "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Integrate multi-omics data only"""
    run(config=config, steps="integrate", verbose=verbose)


@app.command()
def cluster(
    config: Path = typer.Option("config.yaml", "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Cluster integrated data only"""
    run(config=config, steps="cluster", verbose=verbose)


@app.command()
def validate(
    config: Path = typer.Option("config.yaml", "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Validate clustering results only"""
    run(config=config, steps="validate", verbose=verbose)


@app.command()
def report(
    config: Path = typer.Option("config.yaml", "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate report only"""
    run(config=config, steps="report", verbose=verbose)


@app.command()
def validate_config(
    config: Path = typer.Argument(..., help="Path to configuration file"),
    strict: bool = typer.Option(False, "--strict", help="Strict validation"),
):
    """
    Validate configuration file without running pipeline

    Examples:
        audhd-pipeline validate-config config.yaml
        audhd-pipeline validate-config config.yaml --strict
    """
    if not CLI_AVAILABLE:
        typer.echo("Error: CLI requires typer and rich. Install with: pip install typer rich")
        raise typer.Exit(1)

    console.print(f"\nValidating configuration: [cyan]{config}[/cyan]\n")

    try:
        cfg = load_and_validate_config(
            config,
            check_paths=True,
            check_resources=True,
            strict=strict
        )

        console.print("[green]✓[/green] Configuration is valid!\n")

        # Show summary
        _show_config_summary(cfg, [])

    except Exception as e:
        console.print(f"[red]✗[/red] Validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def download_sample_data(
    output_dir: Path = typer.Option(
        "data/",
        "--output", "-o",
        help="Output directory for sample data"
    ),
):
    """
    Download sample/synthetic data for testing

    Downloads synthetic multi-omics data to test the pipeline.
    """
    if not CLI_AVAILABLE:
        typer.echo("Error: CLI requires typer and rich")
        raise typer.Exit(1)

    console.print("\n[bold]Downloading sample data...[/bold]\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate synthetic data
        from ..data.synthetic import generate_synthetic_dataset

        console.print("Generating synthetic data...")

        dataset = generate_synthetic_dataset(
            n_samples=100,
            output_dir=output_dir
        )

        console.print(f"\n[green]✓[/green] Sample data saved to: {output_dir}")
        console.print(f"  - Genomic: {dataset['genomic_path']}")
        console.print(f"  - Clinical: {dataset['clinical_path']}")
        console.print(f"  - Metabolomic: {dataset['metabolomic_path']}")

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to generate sample data: {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information"""
    import audhd_correlation

    if CLI_AVAILABLE:
        console.print(f"\n[bold]AuDHD Correlation Pipeline[/bold]")
        console.print(f"Version: [cyan]{audhd_correlation.__version__}[/cyan]")
        console.print(f"Python: [cyan]{sys.version.split()[0]}[/cyan]\n")
    else:
        print(f"AuDHD Correlation Pipeline v{audhd_correlation.__version__}")


# Helper functions

def _show_config_summary(cfg: "PipelineConfig", steps: List[str]):
    """Display configuration summary"""
    if not console:
        return

    table = Table(title="Configuration Summary", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Pipeline Name", cfg.pipeline.name)
    table.add_row("Version", cfg.pipeline.version)
    table.add_row("Output Dir", cfg.pipeline.output_dir)

    # Data sources
    data_sources = []
    if cfg.data.genomic.enabled:
        data_sources.append("genomic")
    if cfg.data.clinical.enabled:
        data_sources.append("clinical")
    if hasattr(cfg.data, 'metabolomic') and cfg.data.metabolomic and cfg.data.metabolomic.enabled:
        data_sources.append("metabolomic")
    if hasattr(cfg.data, 'microbiome') and cfg.data.microbiome and cfg.data.microbiome.enabled:
        data_sources.append("microbiome")

    table.add_row("Data Sources", ", ".join(data_sources))
    table.add_row("Integration Method", cfg.integration.method)
    table.add_row("Clustering Method", cfg.clustering.method)

    if steps:
        table.add_row("Steps to Run", ", ".join(steps))

    console.print(table)


def _show_results_summary(results: dict):
    """Display results summary"""
    if not console:
        return

    table = Table(title="Pipeline Results", show_header=True)
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="white")

    for stage, result in results.items():
        if isinstance(result, dict):
            status = "[green]✓[/green]" if result.get('success', True) else "[red]✗[/red]"
            details = result.get('summary', '')
        else:
            status = "[green]✓[/green]"
            details = str(result)

        table.add_row(stage, status, details)

    console.print("\n")
    console.print(table)


def main():
    """Main entry point for CLI"""
    if not CLI_AVAILABLE:
        print("Error: CLI requires typer and rich. Install with:")
        print("  pip install typer rich")
        sys.exit(1)

    app()


if __name__ == "__main__":
    main()