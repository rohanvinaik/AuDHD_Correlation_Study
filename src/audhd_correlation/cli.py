"""
Command-line interface for the AuDHD correlation study pipeline
"""
import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from typing import Optional

app = typer.Typer(
    name="audhd",
    help="Multi-omics integration pipeline for ADHD/Autism subtyping",
    add_completion=False,
)
console = Console()


@app.command()
def download_data(
    dataset: str = typer.Option("spark", help="Dataset to download (spark/ssc/abcd/ukb)"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory"),
):
    """Download and prepare datasets"""
    console.print(f"[bold green]Downloading {dataset} dataset...[/bold green]")
    # Implementation in scripts/download_data.py


@app.command()
def preprocess(
    config: Path = typer.Option("configs/defaults.yaml", help="Config file"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory"),
):
    """Run preprocessing pipeline"""
    console.print("[bold green]Running preprocessing...[/bold green]")
    # Implementation calls preprocess modules


@app.command()
def integrate(
    config: Path = typer.Option("configs/defaults.yaml", help="Config file"),
    method: str = typer.Option("mofa2", help="Integration method"),
):
    """Integrate multi-omics data"""
    console.print(f"[bold green]Integrating data with {method}...[/bold green]")
    # Implementation calls integration modules


@app.command()
def cluster(
    config: Path = typer.Option("configs/defaults.yaml", help="Config file"),
    method: str = typer.Option("hdbscan", help="Clustering method"),
):
    """Perform clustering analysis"""
    console.print(f"[bold green]Clustering with {method}...[/bold green]")
    # Implementation calls clustering modules


@app.command()
def validate(
    config: Path = typer.Option("configs/defaults.yaml", help="Config file"),
):
    """Run validation analyses"""
    console.print("[bold green]Running validation...[/bold green]")
    # Implementation calls validation modules


@app.command()
def report(
    config: Path = typer.Option("configs/defaults.yaml", help="Config file"),
    report_type: str = typer.Option("all", help="Report type (all/executive/clinical)"),
):
    """Generate reports"""
    console.print(f"[bold green]Generating {report_type} report...[/bold green]")
    # Implementation calls reporting modules


@app.command()
def pipeline(
    config: Path = typer.Option("configs/defaults.yaml", help="Config file"),
    steps: Optional[str] = typer.Option(None, help="Steps to run (comma-separated)"),
):
    """Run full pipeline"""
    console.print("[bold green]Running full pipeline...[/bold green]")
    with Progress() as progress:
        task = progress.add_task("[cyan]Pipeline progress...", total=100)
        # Run all pipeline steps
        progress.update(task, advance=20)


if __name__ == "__main__":
    app()