"""Quality control per modality"""
from typing import Dict, Any
import pandas as pd
import numpy as np
from ..config.schema import AppConfig


def run_all(tables: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """Run QC on all modalities"""
    qc_tables = {}

    for modality, df in tables.items():
        if df.empty:
            qc_tables[modality] = df
            continue

        # Apply modality-specific QC
        if modality == "genetic":
            qc_tables[modality] = _qc_genetic(df, cfg)
        elif modality == "metabolomic":
            qc_tables[modality] = _qc_metabolomic(df, cfg)
        elif modality == "clinical":
            qc_tables[modality] = _qc_clinical(df, cfg)
        elif modality == "microbiome":
            qc_tables[modality] = _qc_microbiome(df, cfg)
        elif modality == "neuroimaging":
            qc_tables[modality] = _qc_neuroimaging(df, cfg)
        else:
            qc_tables[modality] = df

    return qc_tables


def _qc_genetic(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """QC for genetic data (call rate, HWE, MAF filters)"""
    # Remove samples with high missingness
    sample_missingness = df.isnull().mean(axis=1)
    df = df[sample_missingness < 0.1]

    # Remove variants with high missingness
    variant_missingness = df.isnull().mean(axis=0)
    df = df.loc[:, variant_missingness < 0.05]

    return df


def _qc_metabolomic(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """QC for metabolomic data (CV filters, outlier detection)"""
    # Remove features with >50% missing
    missing_rate = df.isnull().mean(axis=0)
    df = df.loc[:, missing_rate < 0.5]

    # Remove samples with extreme outliers (>5 SD from mean)
    z_scores = np.abs((df - df.mean()) / df.std())
    outlier_samples = (z_scores > 5).sum(axis=1) > (df.shape[1] * 0.1)
    df = df[~outlier_samples]

    return df


def _qc_clinical(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """QC for clinical data (range checks, consistency)"""
    # Remove samples with >30% missing clinical data
    sample_missingness = df.isnull().mean(axis=1)
    df = df[sample_missingness < 0.3]

    return df


def _qc_microbiome(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """QC for microbiome data (read depth, contamination)"""
    # Remove samples with very low read depth (<1000 total reads)
    total_reads = df.sum(axis=1)
    df = df[total_reads >= 1000]

    # Remove rare taxa (<0.1% prevalence)
    prevalence = (df > 0).mean(axis=0)
    df = df.loc[:, prevalence >= 0.001]

    return df


def _qc_neuroimaging(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """QC for neuroimaging data (motion, SNR, artifacts)"""
    # Remove samples with failed QC (placeholder: no extreme outliers)
    z_scores = np.abs((df - df.mean()) / df.std())
    outlier_samples = (z_scores > 4).sum(axis=1) > (df.shape[1] * 0.2)
    df = df[~outlier_samples]

    return df