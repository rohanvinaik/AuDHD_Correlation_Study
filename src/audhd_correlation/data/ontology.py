"""Ontology mapping (HPO, SNOMED, RxNorm, ExO)"""
from typing import Dict, Any
import pandas as pd
from ..config.schema import AppConfig


def map_all(tables: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """Map all tables to ontologies"""
    mapped_tables = {}

    for modality, df in tables.items():
        if df.empty:
            mapped_tables[modality] = df
            continue

        # Apply modality-specific ontology mapping
        if modality == "clinical":
            mapped_tables[modality] = _map_clinical_to_hpo_snomed(df, cfg)
        elif modality == "metabolomic":
            mapped_tables[modality] = _map_metabolites_to_hmdb(df, cfg)
        elif modality == "genetic":
            mapped_tables[modality] = _map_genes_to_entrez(df, cfg)
        elif modality == "microbiome":
            mapped_tables[modality] = _map_taxa_to_ncbi(df, cfg)
        else:
            mapped_tables[modality] = df

    return mapped_tables


def _map_clinical_to_hpo_snomed(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Map clinical phenotypes to HPO and SNOMED-CT"""
    # Placeholder: In production, use HPO API or pre-built mappings
    # For now, just add metadata columns
    df.attrs["ontology"] = "HPO/SNOMED-CT"
    return df


def _map_metabolites_to_hmdb(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Map metabolite names to HMDB IDs"""
    # Placeholder: Use HMDB API or local database
    df.attrs["ontology"] = "HMDB"
    return df


def _map_genes_to_entrez(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Map gene symbols to Entrez Gene IDs"""
    # Placeholder: Use NCBI Gene or BioMart
    df.attrs["ontology"] = "EntrezGene"
    return df


def _map_taxa_to_ncbi(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Map microbial taxa to NCBI Taxonomy"""
    # Placeholder: Use NCBI Taxonomy database
    df.attrs["ontology"] = "NCBITaxonomy"
    return df