"""
Main pipeline orchestration - thin, readable, delegates to modules
"""
from ..config import load_config as load_validated_config, AppConfig
from ..utils.io import save_data, load_data
from rich.console import Console

console = Console()


def _cfg(path: str) -> AppConfig:
    """Load and validate config (internal helper)"""
    return load_validated_config(path)


def download(cfg_path: str) -> None:
    """
    Fetch raw data from SPARK/SSC/ABCD/UKB & references (requires DUAs)
    """
    cfg = _cfg(cfg_path)

    console.print("[bold]→ download[/bold]")

    # Import modules dynamically to avoid circular imports
    from ..data import registries, loaders

    # Check DUA compliance
    registries.ensure_sources(cfg)
    console.print("  [green]✓[/green] DUA checks passed")

    # Fetch raw datasets
    loaders.fetch_all_datasets(cfg)
    console.print("  [green]✓[/green] Datasets downloaded")

    # Fetch reference data (ontologies, pathways)
    loaders.fetch_references(cfg)
    console.print("  [green]✓[/green] References downloaded")


def build_features(cfg_path: str) -> None:
    """
    QC, harmonize, and assemble multi-modal feature tables
    """
    cfg = _cfg(cfg_path)

    console.print("[bold]→ build_features[/bold]")

    from ..data import loaders, ontology, qc, harmonize, context
    from ..features import assemble
    from ..preprocess import impute, batch_effects, scaling, adjust

    # Load raw tables
    tables = loaders.load_all(cfg)
    console.print("  [green]✓[/green] Raw data loaded")

    # Map to ontologies (HPO, SNOMED, RxNorm)
    tables = ontology.map_all(tables, cfg)
    console.print("  [green]✓[/green] Ontology mapping complete")

    # QC per modality
    tables = qc.run_all(tables, cfg)
    console.print("  [green]✓[/green] QC passed")

    # Add context tags (fasting, time-of-day, etc.)
    tables = context.add_tags(tables, cfg)
    console.print("  [green]✓[/green] Context tags added")

    # Assemble feature matrices
    X = assemble.build_feature_matrices(tables, cfg)
    console.print("  [green]✓[/green] Feature matrices assembled")

    # Imputation (delta-adjusted MICE)
    X = impute.run(X, cfg)
    console.print("  [green]✓[/green] Imputation complete")

    # Batch/site correction (ComBat/RUV)
    X = batch_effects.correct(X, cfg)
    console.print("  [green]✓[/green] Batch effects corrected")

    # Partial out covariates (LMM)
    X = adjust.partial_out(X, cfg)
    console.print("  [green]✓[/green] Covariates adjusted")

    # Apply scaling
    X = scaling.apply(X, cfg)
    console.print("  [green]✓[/green] Scaling applied")

    # Save processed features
    assemble.save(X, cfg)
    console.print("  [green]✓[/green] Features saved")


def integrate(cfg_path: str) -> None:
    """
    Integrate multi-omics (stack/MOFA/DIABLO/graph)
    """
    cfg = _cfg(cfg_path)

    console.print("[bold]→ integrate[/bold]")

    from ..features import assemble
    from ..integrate import stack, mofa, diablo, graphs

    # Load preprocessed features
    X = assemble.load_processed(cfg)
    console.print("  [green]✓[/green] Processed features loaded")

    # Run integration method(s)
    Z = {}

    if cfg.integrate.method == "stack":
        Z["stack"] = stack.run(X, cfg)
        console.print("  [green]✓[/green] Weighted stack complete")

    if cfg.integrate.method == "mofa2":
        Z["mofa2"] = mofa.run(X, cfg)
        console.print("  [green]✓[/green] MOFA2 complete")

    if cfg.integrate.method == "diablo":
        Z["diablo"] = diablo.run(X, cfg)
        console.print("  [green]✓[/green] DIABLO complete")

    if cfg.integrate.method == "gmkf":
        Z["gmkf"] = graphs.run(X, cfg)
        console.print("  [green]✓[/green] Multi-kernel fusion complete")

    # Save integrated embeddings
    assemble.save_embeddings(Z, cfg)
    console.print("  [green]✓[/green] Embeddings saved")


def cluster(cfg_path: str) -> None:
    """
    Embeddings + consensus clustering + topology gaps
    """
    cfg = _cfg(cfg_path)

    console.print("[bold]→ cluster[/bold]")

    from ..features import assemble
    from ..modeling import representation, clustering, topology

    # Load integrated embeddings
    Z = assemble.load_embeddings(cfg)
    console.print("  [green]✓[/green] Embeddings loaded")

    # Optional: additional representation layer (VAE)
    embs = representation.make_embeddings(Z, cfg)
    console.print("  [green]✓[/green] Embeddings computed (UMAP/t-SNE)")

    # Consensus clustering (HDBSCAN + LCA)
    labels, consensus = clustering.consensus(embs, cfg)
    console.print("  [green]✓[/green] Consensus clustering complete")

    # Topology analysis (gaps, persistence)
    gaps = topology.evaluate(embs["umap_main"], labels, cfg)
    console.print(f"  [green]✓[/green] Gap score: {gaps.get('gap_score', 0.0):.2f}")

    # Save clustering results
    assemble.save_clusters(labels, consensus, gaps, cfg)
    console.print("  [green]✓[/green] Clusters saved")


def validate(cfg_path: str) -> None:
    """
    Internal/external/stability/causal/sensitivity validation
    """
    cfg = _cfg(cfg_path)

    console.print("[bold]→ validate[/bold]")

    from ..features import assemble
    from ..validation import internal, external, sensitivity
    from ..validation import causal as causal_mod

    # Load data and labels
    X = assemble.load_processed(cfg)
    labels = assemble.load_labels(cfg)
    console.print("  [green]✓[/green] Data and labels loaded")

    # Internal validation (silhouette, stability, biological)
    internal_results = internal.run_all(X, labels, cfg)
    console.print(f"  [green]✓[/green] Internal validation: "
                  f"silhouette={internal_results.get('silhouette', 0.0):.3f}, "
                  f"stability={internal_results.get('stability', 0.0):.3f}")

    # External validation (holdout cohorts, leave-site-out)
    external_results = external.run_all(cfg)
    console.print("  [green]✓[/green] External validation complete")

    # Sensitivity analyses (meds, fasting, MNAR)
    sensitivity_results = sensitivity.run_all(cfg)
    console.print("  [green]✓[/green] Sensitivity analyses complete")

    # Causal inference (MR, mediation, G×E)
    causal_results = causal_mod.run_all(cfg)
    console.print("  [green]✓[/green] Causal analyses complete")


def report(cfg_path: str) -> None:
    """
    Generate executive summary + clinician decision cards
    """
    cfg = _cfg(cfg_path)

    console.print("[bold]→ report[/bold]")

    from ..reporting import report as report_mod

    # Generate all report types
    report_mod.build_all(cfg)
    console.print("  [green]✓[/green] Reports generated")

    # Print output locations
    output_dir = cfg.output_dir
    console.print(f"\n[bold cyan]Reports saved to:[/bold cyan] {output_dir}")

    for report_type in cfg.report.types:
        for fmt in cfg.report.output_formats:
            console.print(f"  • {report_type}.{fmt}")


def pipeline(cfg_path: str, steps: Optional[List[str]] = None) -> None:
    """
    Run full pipeline or selected steps

    Args:
        cfg_path: Path to config file
        steps: Optional list of steps. If None, runs all.
               Valid: download, build_features, integrate, cluster, validate, report
    """
    all_steps = {
        "download": download,
        "build_features": build_features,
        "integrate": integrate,
        "cluster": cluster,
        "validate": validate,
        "report": report,
    }

    if steps is None:
        steps_to_run = list(all_steps.keys())
    else:
        invalid = set(steps) - set(all_steps.keys())
        if invalid:
            raise ValueError(f"Invalid steps: {invalid}")
        steps_to_run = steps

    console.print(f"\n[bold]Pipeline: {', '.join(steps_to_run)}[/bold]")
    console.print(f"[dim]Config: {cfg_path}[/dim]\n")

    for step_name in steps_to_run:
        console.print(f"[bold cyan]{'─'*60}[/bold cyan]")
        try:
            all_steps[step_name](cfg_path)
        except Exception as e:
            console.print(f"[bold red]✗ Pipeline failed at '{step_name}': {e}[/bold red]")
            raise

    console.print(f"\n[bold green]Pipeline complete ✓[/bold green]\n")