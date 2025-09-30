"""Data loaders for SPARK, SSC, ABCD, UK Biobank"""
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from ..config.schema import AppConfig
from .base import DatasetSource, LoadedData
from .genomic_loader import GenomicLoader, PRSLoader
from .metabolomic_loader import MetabolomicLoader
from .clinical_loader import ClinicalLoader
from .microbiome_loader import MicrobiomeLoader, NeuroimagingLoader


def fetch_all_datasets(cfg: AppConfig) -> None:
    """Fetch raw datasets"""
    # TODO: Implement dataset downloads
    pass


def fetch_references(cfg: AppConfig) -> None:
    """Fetch reference data (ontologies, pathways)"""
    # TODO: Implement reference downloads
    pass


def load_all(cfg: AppConfig) -> Dict[str, LoadedData]:
    """
    Load all raw data tables using specialized loaders

    Args:
        cfg: Application configuration

    Returns:
        Dictionary mapping modality names to LoadedData objects
    """
    tables = {}

    # Determine dataset source
    source = _determine_source(cfg)

    # Load genetic data (VCF → PRS, variant annotations)
    if "genetic" in cfg.data.roots:
        genetic_path = Path(cfg.data.roots["genetic"])
        tables["genetic"] = _load_genetic(genetic_path, source, cfg)

    # Load metabolomic data (LC-MS, targeted panels)
    if "metabolomic" in cfg.data.roots:
        metab_path = Path(cfg.data.roots["metabolomic"])
        tables["metabolomic"] = _load_metabolomic(metab_path, source, cfg)

    # Load clinical phenotypes (questionnaires, diagnoses)
    if "clinical" in cfg.data.roots:
        clinical_path = Path(cfg.data.roots["clinical"])
        tables["clinical"] = _load_clinical(clinical_path, source, cfg)

    # Load microbiome data (16S, metagenomic)
    if "microbiome" in cfg.data.roots:
        microbiome_path = Path(cfg.data.roots["microbiome"])
        tables["microbiome"] = _load_microbiome(microbiome_path, source, cfg)

    # Load neuroimaging data (sMRI, fMRI connectivity)
    if "neuroimaging" in cfg.data.roots:
        neuro_path = Path(cfg.data.roots["neuroimaging"])
        tables["neuroimaging"] = _load_neuroimaging(neuro_path, source, cfg)

    return tables


def _determine_source(cfg: AppConfig) -> DatasetSource:
    """
    Determine dataset source from config

    Args:
        cfg: Application configuration

    Returns:
        DatasetSource enum value
    """
    # Check if datasets are specified in config
    if hasattr(cfg.data, "datasets") and cfg.data.datasets:
        primary_dataset = cfg.data.datasets[0].lower()
        source_map = {
            "spark": DatasetSource.SPARK,
            "ssc": DatasetSource.SSC,
            "abcd": DatasetSource.ABCD,
            "ukb": DatasetSource.UKB,
            "uk_biobank": DatasetSource.UKB,
            "metabolights": DatasetSource.METABOLIGHTS,
            "hcp": DatasetSource.HCP,
        }
        return source_map.get(primary_dataset, DatasetSource.SPARK)

    return DatasetSource.SPARK


def _load_genetic(path: Path, source: DatasetSource, cfg: AppConfig) -> LoadedData:
    """
    Load genetic data using GenomicLoader

    Args:
        path: Path to genetic data directory
        source: Dataset source
        cfg: Application configuration

    Returns:
        LoadedData object
    """
    # Check for PRS file first
    prs_file = path / "prs_scores.csv"
    if prs_file.exists():
        loader = PRSLoader(source=source)
        return loader.load(prs_file)

    # Check for VCF or other genetic data
    for pattern in ["*.vcf", "*.vcf.gz", "*.h5"]:
        files = list(path.glob(pattern))
        if files:
            loader = GenomicLoader(source=source)
            return loader.load(files[0])

    # Return empty LoadedData if no files found
    from .base import DataMetadata, QCMetrics

    return LoadedData(
        data=pd.DataFrame(),
        metadata=DataMetadata(
            source=source, modality="genetic", file_path=path, qc_metrics=QCMetrics(0, 0, 0.0, 0)
        ),
    )


def _load_metabolomic(
    path: Path, source: DatasetSource, cfg: AppConfig
) -> LoadedData:
    """
    Load metabolomic data using MetabolomicLoader

    Args:
        path: Path to metabolomic data directory
        source: Dataset source
        cfg: Application configuration

    Returns:
        LoadedData object
    """
    metab_file = path / "metabolites.csv"
    if not metab_file.exists():
        # Try other formats
        for pattern in ["*.h5", "*.hdf5", "*.tsv"]:
            files = list(path.glob(pattern))
            if files:
                metab_file = files[0]
                break

    if metab_file.exists():
        loader = MetabolomicLoader(source=source)
        return loader.load(metab_file)

    # Return empty LoadedData
    from .base import DataMetadata, QCMetrics

    return LoadedData(
        data=pd.DataFrame(),
        metadata=DataMetadata(
            source=source, modality="metabolomic", file_path=path, qc_metrics=QCMetrics(0, 0, 0.0, 0)
        ),
    )


def _load_clinical(path: Path, source: DatasetSource, cfg: AppConfig) -> LoadedData:
    """
    Load clinical data using ClinicalLoader

    Args:
        path: Path to clinical data directory
        source: Dataset source
        cfg: Application configuration

    Returns:
        LoadedData object
    """
    clinical_file = path / "phenotypes.csv"
    if not clinical_file.exists():
        for pattern in ["*.tsv", "*.xlsx"]:
            files = list(path.glob(pattern))
            if files:
                clinical_file = files[0]
                break

    if clinical_file.exists():
        loader = ClinicalLoader(source=source)
        return loader.load(clinical_file)

    # Return empty LoadedData
    from .base import DataMetadata, QCMetrics

    return LoadedData(
        data=pd.DataFrame(),
        metadata=DataMetadata(
            source=source, modality="clinical", file_path=path, qc_metrics=QCMetrics(0, 0, 0.0, 0)
        ),
    )


def _load_microbiome(path: Path, source: DatasetSource, cfg: AppConfig) -> LoadedData:
    """
    Load microbiome data using MicrobiomeLoader

    Args:
        path: Path to microbiome data directory
        source: Dataset source
        cfg: Application configuration

    Returns:
        LoadedData object
    """
    microbiome_file = path / "microbiome.csv"
    if not microbiome_file.exists():
        for pattern in ["*.biom", "*.tsv", "*.h5"]:
            files = list(path.glob(pattern))
            if files:
                microbiome_file = files[0]
                break

    if microbiome_file.exists():
        loader = MicrobiomeLoader(source=source)
        return loader.load(microbiome_file)

    # Return empty LoadedData
    from .base import DataMetadata, QCMetrics

    return LoadedData(
        data=pd.DataFrame(),
        metadata=DataMetadata(
            source=source, modality="microbiome", file_path=path, qc_metrics=QCMetrics(0, 0, 0.0, 0)
        ),
    )


def _load_neuroimaging(
    path: Path, source: DatasetSource, cfg: AppConfig
) -> LoadedData:
    """
    Load neuroimaging data using NeuroimagingLoader

    Args:
        path: Path to neuroimaging data directory
        source: Dataset source
        cfg: Application configuration

    Returns:
        LoadedData object
    """
    neuro_file = path / "neuroimaging.csv"
    if not neuro_file.exists():
        for pattern in ["*.h5", "*.hdf5", "*.tsv"]:
            files = list(path.glob(pattern))
            if files:
                neuro_file = files[0]
                break

    if neuro_file.exists():
        loader = NeuroimagingLoader(source=source)
        return loader.load(neuro_file)

    # Return empty LoadedData
    from .base import DataMetadata, QCMetrics

    return LoadedData(
        data=pd.DataFrame(),
        metadata=DataMetadata(
            source=source, modality="neuroimaging", file_path=path, qc_metrics=QCMetrics(0, 0, 0.0, 0)
        ),
    )