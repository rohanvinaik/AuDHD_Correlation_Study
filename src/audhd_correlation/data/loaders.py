"""Data loaders for SPARK, SSC, ABCD, UK Biobank"""
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from ..config.schema import AppConfig


def fetch_all_datasets(cfg: AppConfig) -> None:
    """Fetch raw datasets"""
    # TODO: Implement dataset downloads
    pass


def fetch_references(cfg: AppConfig) -> None:
    """Fetch reference data (ontologies, pathways)"""
    # TODO: Implement reference downloads
    pass


def load_all(cfg: AppConfig) -> Dict[str, Any]:
    """Load all raw data tables"""
    tables = {}

    # Load genetic data (VCF → PRS, variant annotations)
    if "genetic" in cfg.data.roots:
        genetic_path = Path(cfg.data.roots["genetic"])
        tables["genetic"] = _load_genetic(genetic_path, cfg)

    # Load metabolomic data (LC-MS, targeted panels)
    if "metabolomic" in cfg.data.roots:
        metab_path = Path(cfg.data.roots["metabolomic"])
        tables["metabolomic"] = _load_metabolomic(metab_path, cfg)

    # Load clinical phenotypes (questionnaires, diagnoses)
    if "clinical" in cfg.data.roots:
        clinical_path = Path(cfg.data.roots["clinical"])
        tables["clinical"] = _load_clinical(clinical_path, cfg)

    # Load microbiome data (16S, metagenomic)
    if "microbiome" in cfg.data.roots:
        microbiome_path = Path(cfg.data.roots["microbiome"])
        tables["microbiome"] = _load_microbiome(microbiome_path, cfg)

    # Load neuroimaging data (sMRI, fMRI connectivity)
    if "neuroimaging" in cfg.data.roots:
        neuro_path = Path(cfg.data.roots["neuroimaging"])
        tables["neuroimaging"] = _load_neuroimaging(neuro_path, cfg)

    return tables


def _load_genetic(path: Path, cfg: AppConfig) -> pd.DataFrame:
    """Load genetic data (PRS scores, variant annotations)"""
    prs_file = path / "prs_scores.csv"
    if prs_file.exists():
        return pd.read_csv(prs_file, index_col=0)
    return pd.DataFrame()


def _load_metabolomic(path: Path, cfg: AppConfig) -> pd.DataFrame:
    """Load metabolomic data (targeted/untargeted)"""
    metab_file = path / "metabolites.csv"
    if metab_file.exists():
        return pd.read_csv(metab_file, index_col=0)
    return pd.DataFrame()


def _load_clinical(path: Path, cfg: AppConfig) -> pd.DataFrame:
    """Load clinical phenotype data"""
    clinical_file = path / "phenotypes.csv"
    if clinical_file.exists():
        return pd.read_csv(clinical_file, index_col=0)
    return pd.DataFrame()


def _load_microbiome(path: Path, cfg: AppConfig) -> pd.DataFrame:
    """Load microbiome data (16S OTUs, metagenomic)"""
    microbiome_file = path / "microbiome.csv"
    if microbiome_file.exists():
        return pd.read_csv(microbiome_file, index_col=0)
    return pd.DataFrame()


def _load_neuroimaging(path: Path, cfg: AppConfig) -> pd.DataFrame:
    """Load neuroimaging data (ROI volumes, connectivity)"""
    neuro_file = path / "neuroimaging.csv"
    if neuro_file.exists():
        return pd.read_csv(neuro_file, index_col=0)
    return pd.DataFrame()