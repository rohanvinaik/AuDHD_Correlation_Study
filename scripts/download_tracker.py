#!/usr/bin/env python3
"""
Real-time Download Tracker with Progress Bars
Monitors download progress and displays graphical status
"""

import json
import time
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import os
import re

try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich library for beautiful progress bars...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich"], check=True)
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text

project_root = Path("/Users/rohanvinaik/AuDHD_Correlation_Study")
console = Console()

# All datasets we're tracking
GEO_DATASETS = [
    'GSE28521', 'GSE28475', 'GSE64018', 'GSE98793', 'GSE18123',
    'GSE80655', 'GSE113834', 'GSE42133', 'GSE147314', 'GSE50759', 'GSE53162'
]

PAPER_QUERIES = [
    "ADHD microbiome", "autism microbiome", "ADHD metabolomics", "autism metabolomics",
    "ADHD gene expression", "autism transcriptome", "ADHD proteomics", "autism proteomics",
    "maternal immune activation autism", "prenatal infection autism ADHD",
    "preterm birth autism ADHD", "pregnancy complications neurodevelopment",
    "maternal SSRI autism", "birth outcomes autism ADHD",
    "maternal fever pregnancy autism", "gestational diabetes autism ADHD",
    "valproate pregnancy autism", "gene environment interaction autism ADHD",
    "G×E neurodevelopment", "critical period neurodevelopment autism",
    "developmental window autism", "autism heterogeneity subtypes",
    "ADHD subtypes clustering", "autism endophenotypes",
    "multimodal integration autism", "multi-omics neurodevelopment",
    "autonomic dysregulation autism ADHD", "heart rate variability autism",
    "circadian rhythm autism ADHD", "melatonin autism sleep",
    "cortisol autism ADHD stress", "sensory processing autism",
    "interoception autism ADHD", "voice prosody autism",
    "speech acoustics autism", "inflammatory markers autism ADHD",
    "cytokines neurodevelopment", "oxidative stress autism",
    "mitochondrial dysfunction autism", "epigenetics autism ADHD",
    "DNA methylation neurodevelopment", "polygenic risk score autism ADHD",
    "causal inference neurodevelopment", "mediation analysis autism",
    "mixtures environmental autism", "multiple exposures neurodevelopment",
    "air pollution autism ADHD", "pesticides neurodevelopment",
    "heavy metals autism", "lead mercury autism ADHD", "phthalates neurodevelopment"
]

def get_file_size_mb(filepath):
    """Get file size in MB"""
    try:
        return filepath.stat().st_size / (1024 * 1024)
    except:
        return 0

def count_completed_queries():
    """Count how many paper queries have been processed"""
    paper_log = project_root / "logs/downloads/paper_scraping.log"
    if not paper_log.exists():
        return 0, 0, 0

    try:
        with open(paper_log) as f:
            content = f.read()

        # Count query completions
        queries_complete = len(re.findall(r'Processing query:', content))
        papers_downloaded = content.count('Downloading PMC')
        supplements_downloaded = len(re.findall(r'Downloaded.*supplement', content))

        return queries_complete, papers_downloaded, supplements_downloaded
    except:
        return 0, 0, 0

def count_geo_complete():
    """Count completed GEO datasets"""
    completed = []
    for dataset in GEO_DATASETS:
        log_file = project_root / f"logs/downloads/{dataset}.log"
        if log_file.exists():
            try:
                with open(log_file) as f:
                    content = f.read()
                if 'download complete' in content.lower():
                    completed.append(dataset)
            except:
                pass
    return completed

def get_sra_progress():
    """Get SRA download progress"""
    sra_log = project_root / "logs/downloads/sra_download.log"
    if not sra_log.exists():
        return 0, 0

    try:
        with open(sra_log) as f:
            content = f.read()

        # Find progress lines
        success_match = re.search(r'Successful: (\d+)/(\d+)', content)
        if success_match:
            return int(success_match.group(1)), int(success_match.group(2))
    except:
        pass

    return 0, 0

def create_dashboard():
    """Create rich dashboard layout"""

    # Get current status
    queries_done, papers, supplements = count_completed_queries()
    geo_complete = count_geo_complete()
    sra_done, sra_total = get_sra_progress()

    # Calculate data sizes
    geo_dir = project_root / "data/raw/geo"
    sra_dir = project_root / "data/raw/sra"
    papers_dir = project_root / "data/papers"
    gwas_dir = project_root / "data/raw/gwas"

    geo_size = sum(get_file_size_mb(f) for f in geo_dir.rglob("*") if f.is_file()) / 1024 if geo_dir.exists() else 0
    sra_size = sum(get_file_size_mb(f) for f in sra_dir.rglob("*") if f.is_file()) / 1024 if sra_dir.exists() else 0
    papers_size = sum(get_file_size_mb(f) for f in papers_dir.rglob("*") if f.is_file()) / 1024 if papers_dir.exists() else 0
    gwas_size = sum(get_file_size_mb(f) for f in gwas_dir.rglob("*") if f.is_file()) / 1024 if gwas_dir.exists() else 0

    total_size = geo_size + sra_size + papers_size + gwas_size

    # Create progress table
    table = Table(title="📊 AuDHD Multi-Omics Data Acquisition", title_style="bold magenta", show_header=True, header_style="bold cyan")
    table.add_column("Source", style="cyan", width=25)
    table.add_column("Progress", width=40)
    table.add_column("Status", justify="center", width=15)
    table.add_column("Size", justify="right", width=12)

    # Paper Scraping
    paper_progress = (queries_done / len(PAPER_QUERIES)) * 100 if PAPER_QUERIES else 0
    paper_bar = "█" * int(paper_progress / 2.5) + "░" * (40 - int(paper_progress / 2.5))
    paper_status = f"{queries_done}/{len(PAPER_QUERIES)} queries"
    paper_color = "green" if queries_done == len(PAPER_QUERIES) else "yellow"
    table.add_row(
        "📄 Paper Scraping (50 queries)",
        f"[{paper_color}]{paper_bar}[/{paper_color}] {paper_progress:.1f}%",
        f"[bold]{papers} papers\n{supplements} supps",
        f"{papers_size:.2f} GB"
    )

    # GEO Datasets
    geo_progress = (len(geo_complete) / len(GEO_DATASETS)) * 100 if GEO_DATASETS else 0
    geo_bar = "█" * int(geo_progress / 2.5) + "░" * (40 - int(geo_progress / 2.5))
    geo_status = f"{len(geo_complete)}/{len(GEO_DATASETS)} complete"
    geo_color = "green" if len(geo_complete) == len(GEO_DATASETS) else "yellow"
    table.add_row(
        "🧬 GEO Expression (11 sets)",
        f"[{geo_color}]{geo_bar}[/{geo_color}] {geo_progress:.1f}%",
        f"[bold]{geo_status}",
        f"{geo_size:.2f} GB"
    )

    # SRA Microbiome
    sra_progress = (sra_done / sra_total * 100) if sra_total > 0 else 0
    sra_bar = "█" * int(sra_progress / 2.5) + "░" * (40 - int(sra_progress / 2.5))
    sra_color = "green" if sra_done == sra_total else "yellow"
    table.add_row(
        "🦠 SRA Microbiome",
        f"[{sra_color}]{sra_bar}[/{sra_color}] {sra_progress:.1f}%",
        f"[bold]{sra_done}/{sra_total} studies",
        f"{sra_size:.2f} GB"
    )

    # GWAS Catalog
    gwas_files = len(list(gwas_dir.glob("*.tsv*"))) if gwas_dir.exists() else 0
    gwas_status = "✓ Complete" if gwas_files >= 3 else "⟳ Downloading"
    gwas_color = "green" if gwas_files >= 3 else "yellow"
    gwas_progress = (gwas_files / 3) * 100
    gwas_bar = "█" * int(gwas_progress / 2.5) + "░" * (40 - int(gwas_progress / 2.5))
    table.add_row(
        "🧬 GWAS Catalog (ASD/ADHD)",
        f"[{gwas_color}]{gwas_bar}[/{gwas_color}] {gwas_progress:.1f}%",
        f"[bold]{gwas_status}",
        f"{gwas_size:.2f} GB"
    )

    # Create summary panel
    total_items = len(PAPER_QUERIES) + len(GEO_DATASETS) + (sra_total if sra_total > 0 else 50) + 3
    completed_items = queries_done + len(geo_complete) + sra_done + gwas_files
    overall_progress = (completed_items / total_items) * 100 if total_items > 0 else 0

    summary = Text()
    summary.append("📦 Total Downloaded: ", style="bold")
    summary.append(f"{total_size:.2f} GB\n", style="bold cyan")
    summary.append("🎯 Overall Progress: ", style="bold")
    summary.append(f"{overall_progress:.1f}%\n", style="bold yellow")
    summary.append("📊 Items: ", style="bold")
    summary.append(f"{completed_items}/{total_items} complete", style="bold green" if completed_items == total_items else "bold yellow")

    summary_panel = Panel(summary, title="Summary", border_style="green")

    # Recent activity
    paper_log = project_root / "logs/downloads/paper_scraping.log"
    recent_text = Text()
    if paper_log.exists():
        try:
            with open(paper_log) as f:
                lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                if ' - INFO - ' in last_line:
                    msg = last_line.split(' - INFO - ', 1)[1]
                    recent_text.append("📝 ", style="bold blue")
                    recent_text.append(msg[:90] + "..." if len(msg) > 90 else msg, style="dim")
                else:
                    recent_text.append("Waiting for updates...", style="dim")
        except:
            recent_text.append("Initializing...", style="dim")
    else:
        recent_text.append("Starting downloads...", style="dim")

    activity_panel = Panel(recent_text, title="Recent Activity", border_style="blue")

    # Layout
    layout = Layout()
    layout.split(
        Layout(table, name="main", ratio=3),
        Layout(name="bottom", ratio=1)
    )
    layout["bottom"].split_row(
        Layout(summary_panel, name="summary"),
        Layout(activity_panel, name="activity")
    )

    return layout

def main():
    """Main tracker with live updating"""

    console.print("[bold green]🚀 Starting AuDHD Data Acquisition Tracker[/bold green]")
    console.print("[dim]Updates every 2 seconds | Press Ctrl+C to exit[/dim]\n")

    try:
        with Live(create_dashboard(), refresh_per_second=0.5, console=console) as live:
            while True:
                time.sleep(2)
                live.update(create_dashboard())

    except KeyboardInterrupt:
        console.print("\n\n[bold green]✓ Tracker stopped[/bold green]")
        console.print("[dim]Status saved to: data/catalogs/download_status.json[/dim]")
        sys.exit(0)

if __name__ == '__main__':
    main()
