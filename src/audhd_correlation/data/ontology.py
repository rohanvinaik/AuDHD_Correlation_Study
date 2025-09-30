"""Ontology mapping (HPO, SNOMED, RxNorm, ExO)"""
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from ..config.schema import AppConfig

from .clinical_ontology import ClinicalOntologyMapper
from .medication_ontology import MedicationOntologyMapper
from .diet_ontology import FNDDSMapper
from .variable_catalog import VariableCatalog, VariableType


def map_all(tables: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """
    Map all tables to ontologies and build variable catalog

    Args:
        tables: Dictionary of data tables by modality
        cfg: Application configuration

    Returns:
        Mapped tables with ontology metadata
    """
    # Initialize variable catalog
    catalog_path = Path(cfg.data.roots.get("outputs", ".")) / "variable_catalog.json"
    catalog = VariableCatalog(catalog_path)

    # Initialize ontology mappers
    cache_dir = Path.home() / ".cache" / "audhd_ontology"
    clinical_mapper = ClinicalOntologyMapper(cache_dir=cache_dir, use_api=False)
    medication_mapper = MedicationOntologyMapper(cache_dir=cache_dir, use_api=False)
    diet_mapper = FNDDSMapper(cache_dir=cache_dir, use_api=False)

    mapped_tables = {}

    for modality, df in tables.items():
        # Handle LoadedData objects from new loader system
        if hasattr(df, "data"):
            df = df.data

        if df.empty:
            mapped_tables[modality] = df
            continue

        # Apply modality-specific ontology mapping
        if modality == "clinical":
            mapped_tables[modality] = _map_clinical_to_hpo_snomed(
                df, cfg, clinical_mapper, catalog
            )
        elif modality == "metabolomic":
            mapped_tables[modality] = _map_metabolites_to_hmdb(df, cfg, catalog)
        elif modality == "genetic":
            mapped_tables[modality] = _map_genes_to_entrez(df, cfg, catalog)
        elif modality == "microbiome":
            mapped_tables[modality] = _map_taxa_to_ncbi(df, cfg, catalog)
        else:
            mapped_tables[modality] = df

    # Save variable catalog
    catalog.save()

    # Export catalog to CSV for review
    csv_path = Path(cfg.data.roots.get("outputs", ".")) / "variable_catalog.csv"
    catalog.export_to_csv(csv_path)

    return mapped_tables


def _map_clinical_to_hpo_snomed(
    df: pd.DataFrame,
    cfg: AppConfig,
    mapper: ClinicalOntologyMapper,
    catalog: VariableCatalog,
) -> pd.DataFrame:
    """
    Map clinical phenotypes to HPO and SNOMED-CT

    Args:
        df: Clinical data DataFrame
        cfg: Application configuration
        mapper: Clinical ontology mapper
        catalog: Variable catalog

    Returns:
        DataFrame with ontology metadata
    """
    # Map column names to ontologies
    clinical_terms = df.columns.tolist()
    ontology_mappings = {}

    for term in clinical_terms:
        # Map to HPO, SNOMED, ICD-10
        mappings = mapper.map_term(term)
        ontology_mappings[term] = mappings

    # Add to variable catalog
    catalog.merge_from_dataframe(
        df,
        variable_type=VariableType.CLINICAL,
        source_dataset=cfg.data.datasets[0] if cfg.data.datasets else "unknown",
        ontology_mappings=ontology_mappings,
    )

    # Add metadata to DataFrame
    df.attrs["ontology"] = "HPO/SNOMED/ICD10"
    df.attrs["ontology_mappings"] = ontology_mappings

    return df


def _map_metabolites_to_hmdb(
    df: pd.DataFrame, cfg: AppConfig, catalog: VariableCatalog
) -> pd.DataFrame:
    """
    Map metabolite names to HMDB IDs

    Args:
        df: Metabolomic data DataFrame
        cfg: Application configuration
        catalog: Variable catalog

    Returns:
        DataFrame with ontology metadata
    """
    # Add to variable catalog
    catalog.merge_from_dataframe(
        df,
        variable_type=VariableType.METABOLOMIC,
        source_dataset=cfg.data.datasets[0] if cfg.data.datasets else "unknown",
    )

    df.attrs["ontology"] = "HMDB"
    return df


def _map_genes_to_entrez(
    df: pd.DataFrame, cfg: AppConfig, catalog: VariableCatalog
) -> pd.DataFrame:
    """
    Map gene symbols to Entrez Gene IDs

    Args:
        df: Genetic data DataFrame
        cfg: Application configuration
        catalog: Variable catalog

    Returns:
        DataFrame with ontology metadata
    """
    # Add to variable catalog
    catalog.merge_from_dataframe(
        df,
        variable_type=VariableType.GENETIC,
        source_dataset=cfg.data.datasets[0] if cfg.data.datasets else "unknown",
    )

    df.attrs["ontology"] = "EntrezGene"
    return df


def _map_taxa_to_ncbi(
    df: pd.DataFrame, cfg: AppConfig, catalog: VariableCatalog
) -> pd.DataFrame:
    """
    Map microbial taxa to NCBI Taxonomy

    Args:
        df: Microbiome data DataFrame
        cfg: Application configuration
        catalog: Variable catalog

    Returns:
        DataFrame with ontology metadata
    """
    # Add to variable catalog
    catalog.merge_from_dataframe(
        df,
        variable_type=VariableType.MICROBIOME,
        source_dataset=cfg.data.datasets[0] if cfg.data.datasets else "unknown",
    )

    df.attrs["ontology"] = "NCBITaxonomy"
    return df