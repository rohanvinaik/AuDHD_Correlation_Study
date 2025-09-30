"""Gene ID normalization and mapping utilities

Ensures consistent gene identifiers across different databases and formats.
Supports HGNC symbols, Ensembl IDs, Entrez IDs.
"""
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import warnings
from pathlib import Path

import pandas as pd
import numpy as np


@dataclass
class GeneMapping:
    """Gene identifier mapping"""
    hgnc_symbol: str
    ensembl_id: Optional[str] = None
    entrez_id: Optional[int] = None
    uniprot_id: Optional[str] = None
    aliases: List[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class GeneIDMapper:
    """Gene ID normalization and mapping

    Maps between different gene identifier systems:
    - HGNC symbols (e.g., "TP53")
    - Ensembl IDs (e.g., "ENSG00000141510")
    - Entrez IDs (e.g., 7157)
    - UniProt IDs (e.g., "P04637")

    Usage:
        mapper = GeneIDMapper.from_file("data/gene_mappings.tsv")
        normalized = mapper.normalize_genes(["TP53", "ENSG00000141510"])
        # Returns: ["TP53", "TP53"]
    """

    def __init__(self, mappings: Dict[str, GeneMapping]):
        """Initialize mapper with gene mappings

        Args:
            mappings: Dictionary of {any_id: GeneMapping}
        """
        self.mappings = mappings

        # Build reverse lookup indices
        self._build_indices()

    def _build_indices(self):
        """Build lookup indices for fast mapping"""
        self.hgnc_to_mapping = {}
        self.ensembl_to_mapping = {}
        self.entrez_to_mapping = {}
        self.alias_to_mapping = {}

        for gene_id, mapping in self.mappings.items():
            # Index by HGNC symbol
            self.hgnc_to_mapping[mapping.hgnc_symbol.upper()] = mapping

            # Index by Ensembl ID
            if mapping.ensembl_id:
                self.ensembl_to_mapping[mapping.ensembl_id.upper()] = mapping

            # Index by Entrez ID
            if mapping.entrez_id:
                self.entrez_to_mapping[mapping.entrez_id] = mapping

            # Index by aliases
            for alias in mapping.aliases:
                self.alias_to_mapping[alias.upper()] = mapping

    def normalize_gene(self, gene_id: str) -> Optional[str]:
        """Normalize gene ID to HGNC symbol

        Args:
            gene_id: Gene identifier (any format)

        Returns:
            HGNC symbol or None if not found
        """
        gene_id_upper = str(gene_id).upper()

        # Try direct HGNC lookup
        if gene_id_upper in self.hgnc_to_mapping:
            return self.hgnc_to_mapping[gene_id_upper].hgnc_symbol

        # Try Ensembl ID
        if gene_id_upper in self.ensembl_to_mapping:
            return self.ensembl_to_mapping[gene_id_upper].hgnc_symbol

        # Try Entrez ID (numeric)
        try:
            entrez_id = int(gene_id)
            if entrez_id in self.entrez_to_mapping:
                return self.entrez_to_mapping[entrez_id].hgnc_symbol
        except (ValueError, TypeError):
            pass

        # Try aliases
        if gene_id_upper in self.alias_to_mapping:
            return self.alias_to_mapping[gene_id_upper].hgnc_symbol

        # Not found
        return None

    def normalize_genes(
        self,
        gene_ids: List[str],
        warn_missing: bool = True,
    ) -> List[str]:
        """Normalize list of gene IDs to HGNC symbols

        Args:
            gene_ids: List of gene identifiers
            warn_missing: Warn about unmapped genes

        Returns:
            List of HGNC symbols (unmapped genes excluded)
        """
        normalized = []
        missing = []

        for gene_id in gene_ids:
            hgnc = self.normalize_gene(gene_id)
            if hgnc:
                normalized.append(hgnc)
            else:
                missing.append(gene_id)

        if warn_missing and missing:
            warnings.warn(
                f"Could not normalize {len(missing)}/{len(gene_ids)} genes. "
                f"Examples: {missing[:5]}"
            )

        return normalized

    def to_ensembl(self, gene_id: str) -> Optional[str]:
        """Convert gene ID to Ensembl ID"""
        hgnc = self.normalize_gene(gene_id)
        if not hgnc:
            return None

        mapping = self.hgnc_to_mapping.get(hgnc.upper())
        return mapping.ensembl_id if mapping else None

    def to_entrez(self, gene_id: str) -> Optional[int]:
        """Convert gene ID to Entrez ID"""
        hgnc = self.normalize_gene(gene_id)
        if not hgnc:
            return None

        mapping = self.hgnc_to_mapping.get(hgnc.upper())
        return mapping.entrez_id if mapping else None

    @classmethod
    def from_file(cls, mapping_file: str, format: str = "auto") -> "GeneIDMapper":
        """Load gene mappings from file

        Args:
            mapping_file: Path to mapping file
            format: File format ("tsv", "csv", "auto")

        Returns:
            GeneIDMapper instance

        Expected TSV/CSV format:
            hgnc_symbol  ensembl_id  entrez_id  aliases
            TP53         ENSG00000141510  7157  P53;p53
        """
        path = Path(mapping_file)

        if not path.exists():
            raise FileNotFoundError(
                f"Gene mapping file not found: {mapping_file}\n"
                f"Please download from:\n"
                f"  - HGNC: https://www.genenames.org/download/statistics-and-files/\n"
                f"  - Or run: audhd-pipeline download-gene-mappings"
            )

        # Detect format
        if format == "auto":
            if path.suffix == ".csv":
                format = "csv"
            else:
                format = "tsv"

        # Load file
        if format == "csv":
            df = pd.read_csv(mapping_file)
        else:
            df = pd.read_csv(mapping_file, sep="\t")

        # Parse mappings
        mappings = {}

        for _, row in df.iterrows():
            hgnc_symbol = row.get("hgnc_symbol") or row.get("symbol")
            if pd.isna(hgnc_symbol):
                continue

            ensembl_id = row.get("ensembl_id")
            if pd.isna(ensembl_id):
                ensembl_id = None

            entrez_id = row.get("entrez_id")
            if pd.isna(entrez_id):
                entrez_id = None
            else:
                try:
                    entrez_id = int(entrez_id)
                except (ValueError, TypeError):
                    entrez_id = None

            uniprot_id = row.get("uniprot_id")
            if pd.isna(uniprot_id):
                uniprot_id = None

            aliases_str = row.get("aliases", "")
            if pd.isna(aliases_str):
                aliases = []
            else:
                aliases = [a.strip() for a in str(aliases_str).split(";") if a.strip()]

            mapping = GeneMapping(
                hgnc_symbol=hgnc_symbol,
                ensembl_id=ensembl_id,
                entrez_id=entrez_id,
                uniprot_id=uniprot_id,
                aliases=aliases,
            )

            # Index by all identifiers
            mappings[hgnc_symbol] = mapping
            if ensembl_id:
                mappings[ensembl_id] = mapping
            if entrez_id:
                mappings[str(entrez_id)] = mapping

        return cls(mappings=mappings)

    @classmethod
    def create_minimal(cls, gene_list: List[str]) -> "GeneIDMapper":
        """Create minimal mapper from gene list

        Useful for testing or when mapping file unavailable.
        Assumes gene names are already HGNC symbols.

        Args:
            gene_list: List of gene names (assumed HGNC)

        Returns:
            GeneIDMapper with basic mappings
        """
        mappings = {}

        for gene in gene_list:
            mapping = GeneMapping(hgnc_symbol=gene)
            mappings[gene] = mapping

        return cls(mappings=mappings)


def normalize_gene_set(
    gene_set: Set[str],
    mapper: Optional[GeneIDMapper] = None,
) -> Set[str]:
    """Normalize a gene set to HGNC symbols

    Args:
        gene_set: Set of gene identifiers
        mapper: GeneIDMapper instance (if None, returns original set)

    Returns:
        Set of normalized HGNC symbols
    """
    if mapper is None:
        return gene_set

    normalized = set()
    for gene in gene_set:
        hgnc = mapper.normalize_gene(gene)
        if hgnc:
            normalized.add(hgnc)

    return normalized


def normalize_pathway_database(
    pathways: Dict[str, Set[str]],
    mapper: Optional[GeneIDMapper] = None,
) -> Dict[str, Set[str]]:
    """Normalize all genes in pathway database

    Args:
        pathways: Dictionary of {pathway_name: gene_set}
        mapper: GeneIDMapper instance

    Returns:
        Dictionary with normalized gene sets
    """
    if mapper is None:
        return pathways

    normalized_pathways = {}

    for pathway_name, gene_set in pathways.items():
        normalized_set = normalize_gene_set(gene_set, mapper)

        # Only keep pathways with mapped genes
        if normalized_set:
            normalized_pathways[pathway_name] = normalized_set

    return normalized_pathways