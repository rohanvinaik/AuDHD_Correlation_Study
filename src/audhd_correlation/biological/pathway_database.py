"""Pathway database loading and management

Loads gene sets from standard formats (GMT, GPAD, TSV) with proper validation.
NO hardcoded pathway fallbacks - requires explicit database files.
"""
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
import warnings

import pandas as pd
import numpy as np

from .gene_id_mapping import GeneIDMapper, normalize_pathway_database


@dataclass
class PathwayDatabase:
    """Container for pathway database"""
    name: str
    pathways: Dict[str, Set[str]]  # pathway_id -> gene_set
    pathway_descriptions: Dict[str, str]  # pathway_id -> description
    n_pathways: int
    n_genes: int
    source: str  # File path or database name

    def filter_by_size(
        self,
        min_size: int = 15,
        max_size: int = 500,
    ) -> "PathwayDatabase":
        """Filter pathways by gene set size"""
        filtered = {
            pathway_id: genes
            for pathway_id, genes in self.pathways.items()
            if min_size <= len(genes) <= max_size
        }

        return PathwayDatabase(
            name=self.name,
            pathways=filtered,
            pathway_descriptions=self.pathway_descriptions,
            n_pathways=len(filtered),
            n_genes=len(set.union(*filtered.values())) if filtered else 0,
            source=self.source,
        )


def load_pathway_database(
    database_path: str,
    database_name: Optional[str] = None,
    format: str = "auto",
    gene_mapper: Optional[GeneIDMapper] = None,
    normalize_genes: bool = True,
    min_genes: int = 3,
) -> PathwayDatabase:
    """Load pathway database from file

    Args:
        database_path: Path to pathway database file
        database_name: Name for the database (defaults to filename)
        format: File format ("gmt", "gpad", "tsv", "csv", "auto")
        gene_mapper: GeneIDMapper for gene normalization
        normalize_genes: Whether to normalize gene IDs
        min_genes: Minimum genes per pathway

    Returns:
        PathwayDatabase

    Raises:
        FileNotFoundError: If database file doesn't exist
        ValueError: If file format is invalid

    Supported formats:
        - GMT: Gene Matrix Transposed format (MSigDB, Enrichr)
        - GPAD: Gene Product Association Data (Gene Ontology)
        - TSV/CSV: Custom format with pathway_id, gene columns
    """
    path = Path(database_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Pathway database not found: {database_path}\n\n"
            f"Please download pathway databases:\n"
            f"  • MSigDB: https://www.gsea-msigdb.org/gsea/downloads.jsp\n"
            f"  • Gene Ontology: http://geneontology.org/docs/download-ontology/\n"
            f"  • KEGG: https://www.genome.jp/kegg/pathway.html\n"
            f"  • Reactome: https://reactome.org/download-data\n\n"
            f"Or use CLI command:\n"
            f"  audhd-pipeline download-pathways --database msigdb --output {path.parent}\n\n"
            f"Required format: GMT, GPAD, TSV, or CSV with pathway_id and gene columns."
        )

    # Detect format
    if format == "auto":
        format = _detect_format(path)

    # Load pathways
    if format == "gmt":
        pathways, descriptions = _load_gmt(path)
    elif format == "gpad":
        pathways, descriptions = _load_gpad(path)
    elif format in ["tsv", "csv"]:
        pathways, descriptions = _load_tabular(path, format)
    else:
        raise ValueError(
            f"Unknown format: {format}. "
            f"Supported: gmt, gpad, tsv, csv"
        )

    # Filter by minimum genes
    pathways = {
        pathway_id: genes
        for pathway_id, genes in pathways.items()
        if len(genes) >= min_genes
    }

    if not pathways:
        raise ValueError(
            f"No valid pathways found in {database_path}. "
            f"Check file format and min_genes parameter."
        )

    # Normalize gene IDs
    if normalize_genes and gene_mapper is not None:
        pathways = normalize_pathway_database(pathways, gene_mapper)

    # Count unique genes
    all_genes = set.union(*pathways.values()) if pathways else set()

    if database_name is None:
        database_name = path.stem

    return PathwayDatabase(
        name=database_name,
        pathways=pathways,
        pathway_descriptions=descriptions,
        n_pathways=len(pathways),
        n_genes=len(all_genes),
        source=str(path),
    )


def _detect_format(path: Path) -> str:
    """Detect file format from extension"""
    suffix = path.suffix.lower()

    if suffix == ".gmt":
        return "gmt"
    elif suffix == ".gpad":
        return "gpad"
    elif suffix == ".csv":
        return "csv"
    elif suffix in [".tsv", ".txt"]:
        return "tsv"
    else:
        # Try to detect from first line
        with open(path) as f:
            first_line = f.readline()

        if first_line.startswith("!"):
            return "gpad"
        elif "\t" in first_line and first_line.count("\t") > 2:
            return "gmt"
        elif "\t" in first_line:
            return "tsv"
        elif "," in first_line:
            return "csv"
        else:
            raise ValueError(f"Cannot detect format for {path}")


def _load_gmt(path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Load GMT format (MSigDB standard)

    Format:
        pathway_name<tab>description<tab>gene1<tab>gene2<tab>...
    """
    pathways = {}
    descriptions = {}

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split("\t")

            if len(parts) < 3:
                warnings.warn(f"Skipping invalid GMT line: {line[:50]}")
                continue

            pathway_id = parts[0]
            description = parts[1]
            genes = set(parts[2:])  # All remaining columns are genes

            # Remove empty strings
            genes = {g for g in genes if g}

            if genes:
                pathways[pathway_id] = genes
                descriptions[pathway_id] = description

    return pathways, descriptions


def _load_gpad(path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Load GPAD format (Gene Ontology)

    Format (tab-separated):
        DB  DB_Object_ID  Qualifier  GO_ID  Reference  Evidence  ...
    """
    pathways = {}
    descriptions = {}

    with open(path) as f:
        for line in f:
            if line.startswith("!"):  # Comment line
                continue

            if not line.strip():
                continue

            parts = line.strip().split("\t")

            if len(parts) < 4:
                continue

            # Extract gene and GO term
            gene = parts[1]  # DB_Object_ID (gene symbol)
            go_term = parts[3]  # GO_ID

            # Add to pathway
            if go_term not in pathways:
                pathways[go_term] = set()
                descriptions[go_term] = go_term  # Use GO ID as description

            pathways[go_term].add(gene)

    return pathways, descriptions


def _load_tabular(path: Path, format: str) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Load tabular format (TSV/CSV)

    Expected columns:
        - pathway_id (or pathway_name, pathway)
        - gene (or gene_symbol, gene_id)
        - description (optional)
    """
    if format == "csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")

    # Find pathway ID column
    pathway_col = None
    for col in ["pathway_id", "pathway_name", "pathway", "term_id"]:
        if col in df.columns:
            pathway_col = col
            break

    if pathway_col is None:
        raise ValueError(
            f"Could not find pathway ID column. "
            f"Expected one of: pathway_id, pathway_name, pathway, term_id. "
            f"Found: {list(df.columns)}"
        )

    # Find gene column
    gene_col = None
    for col in ["gene", "gene_symbol", "gene_id", "gene_name"]:
        if col in df.columns:
            gene_col = col
            break

    if gene_col is None:
        raise ValueError(
            f"Could not find gene column. "
            f"Expected one of: gene, gene_symbol, gene_id, gene_name. "
            f"Found: {list(df.columns)}"
        )

    # Find description column (optional)
    desc_col = None
    for col in ["description", "pathway_description", "name"]:
        if col in df.columns:
            desc_col = col
            break

    # Build pathways
    pathways = {}
    descriptions = {}

    for _, row in df.iterrows():
        pathway_id = row[pathway_col]
        gene = row[gene_col]

        if pd.isna(pathway_id) or pd.isna(gene):
            continue

        pathway_id = str(pathway_id)
        gene = str(gene)

        if pathway_id not in pathways:
            pathways[pathway_id] = set()

            if desc_col and not pd.isna(row[desc_col]):
                descriptions[pathway_id] = str(row[desc_col])
            else:
                descriptions[pathway_id] = pathway_id

        pathways[pathway_id].add(gene)

    return pathways, descriptions


def merge_pathway_databases(
    databases: List[PathwayDatabase],
    name: str = "merged",
) -> PathwayDatabase:
    """Merge multiple pathway databases

    Args:
        databases: List of PathwayDatabase instances
        name: Name for merged database

    Returns:
        Merged PathwayDatabase
    """
    all_pathways = {}
    all_descriptions = {}

    for db in databases:
        # Prefix pathway IDs with database name to avoid collisions
        for pathway_id, genes in db.pathways.items():
            prefixed_id = f"{db.name}:{pathway_id}"
            all_pathways[prefixed_id] = genes
            all_descriptions[prefixed_id] = db.pathway_descriptions.get(
                pathway_id, pathway_id
            )

    all_genes = set.union(*all_pathways.values()) if all_pathways else set()

    return PathwayDatabase(
        name=name,
        pathways=all_pathways,
        pathway_descriptions=all_descriptions,
        n_pathways=len(all_pathways),
        n_genes=len(all_genes),
        source="merged",
    )


def get_pathway_genes(
    pathway_id: str,
    database: PathwayDatabase,
) -> Set[str]:
    """Get genes in a pathway

    Args:
        pathway_id: Pathway identifier
        database: PathwayDatabase

    Returns:
        Set of gene symbols
    """
    return database.pathways.get(pathway_id, set())


def get_gene_pathways(
    gene: str,
    database: PathwayDatabase,
) -> List[str]:
    """Get pathways containing a gene

    Args:
        gene: Gene symbol
        database: PathwayDatabase

    Returns:
        List of pathway IDs
    """
    pathways = []

    for pathway_id, genes in database.pathways.items():
        if gene in genes:
            pathways.append(pathway_id)

    return pathways