"""
Command-line interface for the AuDHD correlation study pipeline
"""
import typer
from pathlib import Path
from rich.console import Console
from typing import Optional

app = typer.Typer(
    name="audhd-omics",
    help="Multi-omics integration pipeline for ADHD/Autism subtyping",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def download(
    cfg: str = typer.Argument("configs/defaults.yaml", help="Config file path"),
):
    """Fetch raw data & references (requires DUAs)."""
    from .pipelines import run_all

    console.print("[bold cyan]→ Downloading datasets & references[/bold cyan]")
    console.print(f"Config: {cfg}")

    try:
        run_all.download(cfg)
        console.print("[bold green]✓ Download completed[/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Download failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def build_features(
    cfg: str = typer.Argument("configs/defaults.yaml", help="Config file path"),
):
    """QC, harmonize, and assemble multi-modal feature tables."""
    from .pipelines import run_all

    console.print("[bold cyan]→ Building features[/bold cyan]")
    console.print(f"Config: {cfg}")

    try:
        run_all.build_features(cfg)
        console.print("[bold green]✓ Feature building completed[/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Feature building failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def integrate(
    cfg: str = typer.Argument("configs/defaults.yaml", help="Config file path"),
):
    """Integrate multi-omics (stack/MOFA/DIABLO/graph)."""
    from .pipelines import run_all

    console.print("[bold cyan]→ Integrating multi-omics data[/bold cyan]")
    console.print(f"Config: {cfg}")

    try:
        run_all.integrate(cfg)
        console.print("[bold green]✓ Integration completed[/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Integration failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def cluster(
    cfg: str = typer.Argument("configs/defaults.yaml", help="Config file path"),
):
    """Embeddings + consensus clustering + topology gaps."""
    from .pipelines import run_all

    console.print("[bold cyan]→ Clustering analysis[/bold cyan]")
    console.print(f"Config: {cfg}")

    try:
        run_all.cluster(cfg)
        console.print("[bold green]✓ Clustering completed[/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Clustering failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def validate(
    cfg: str = typer.Argument("configs/defaults.yaml", help="Config file path"),
):
    """Internal/external/stability/causal/sensitivity."""
    from .pipelines import run_all

    console.print("[bold cyan]→ Running validation[/bold cyan]")
    console.print(f"Config: {cfg}")

    try:
        run_all.validate(cfg)
        console.print("[bold green]✓ Validation completed[/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Validation failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def report(
    cfg: str = typer.Argument("configs/defaults.yaml", help="Config file path"),
):
    """Generate executive summary + clinician decision cards."""
    from .pipelines import run_all

    console.print("[bold cyan]→ Generating reports[/bold cyan]")
    console.print(f"Config: {cfg}")

    try:
        run_all.report(cfg)
        console.print("[bold green]✓ Report generation completed[/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Report generation failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def pipeline(
    cfg: str = typer.Argument("configs/defaults.yaml", help="Config file path"),
    steps: Optional[str] = typer.Option(
        None,
        "--steps",
        help="Comma-separated steps to run (download,build_features,integrate,cluster,validate,report)"
    ),
):
    """Run full pipeline or selected steps."""
    from .pipelines import run_all

    console.print("[bold cyan]→ Running pipeline[/bold cyan]")
    console.print(f"Config: {cfg}")

    if steps:
        step_list = [s.strip() for s in steps.split(",")]
        console.print(f"Steps: {', '.join(step_list)}")
    else:
        step_list = None
        console.print("Steps: all")

    try:
        run_all.pipeline(cfg, steps=step_list)
        console.print("[bold green]✓ Pipeline completed[/bold green]")
    except Exception as e:
        console.print(f"[bold red]✗ Pipeline failed: {e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()