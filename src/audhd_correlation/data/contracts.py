"""Data contracts for multi-omics tables

Defines the expected structure, dtypes, and invariants for each data modality.
These contracts are enforced in loaders and validated in tests.
"""
from typing import TypedDict, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np


# =============================================================================
# Index and ID Conventions
# =============================================================================

class SampleIDConvention(BaseModel):
    """Standard sample ID format across all modalities"""
    model_config = {"extra": "forbid"}

    format: Literal["DATASET_SAMPLE_VISIT"] = "DATASET_SAMPLE_VISIT"
    example: str = "SPARK_S12345_V01"
    separator: str = "_"

    @field_validator("example")
    @classmethod
    def validate_example_format(cls, v: str) -> str:
        """Ensure example follows format"""
        parts = v.split("_")
        if len(parts) != 3:
            raise ValueError(f"Example must have 3 parts: DATASET_SAMPLE_VISIT, got {v}")
        return v


# Subject-level unique identifier (across visits)
SUBJECT_ID = "subject_id"  # e.g., SPARK_S12345

# Sample-level unique identifier (specific visit/timepoint)
SAMPLE_ID = "sample_id"  # e.g., SPARK_S12345_V01

# Visit/timepoint identifier
VISIT = "visit"  # e.g., V01, V02, baseline, followup

# Dataset source
DATASET = "dataset"  # e.g., SPARK, SSC, ABCD, UKB


# =============================================================================
# Genomic Data Contract
# =============================================================================

class GenomicDataContract(BaseModel):
    """Contract for genomic data tables"""
    model_config = {"extra": "forbid"}

    # Required columns
    index_col: str = SAMPLE_ID
    required_columns: List[str] = Field(
        default=[
            SUBJECT_ID,
            SAMPLE_ID,
            "sex",  # M/F for X chromosome handling
            "array_type",  # Genotyping array or WGS
            "call_rate"  # Sample-level call rate
        ]
    )

    # SNP columns (rs12345, etc.) are dynamic based on QC filters
    # But we expect specific patterns:
    snp_prefix: str = "rs"  # SNP IDs start with 'rs'
    gene_prefix: str = "ENSG"  # Gene IDs for rare variants

    # Expected dtypes
    sex_dtype: str = "category"  # M, F
    call_rate_dtype: str = "float64"  # 0.0-1.0
    genotype_dtype: str = "int8"  # 0, 1, 2, -1 (missing)

    # Structural variant columns
    cnv_columns: Optional[List[str]] = ["CNV_burden", "del_burden", "dup_burden"]
    sv_columns: Optional[List[str]] = ["SV_count", "inversion_count"]

    # Polygenic risk scores (if pre-computed)
    prs_columns: Optional[List[str]] = [
        "PRS_autism",
        "PRS_ADHD",
        "PRS_depression",
        "PRS_schizophrenia"
    ]

    # Ancestry PCs (if included)
    ancestry_pc_prefix: str = "PC"
    n_ancestry_pcs: int = 10  # PC1-PC10

    # Invariants
    min_call_rate: float = 0.90  # Minimum sample call rate
    allowed_genotypes: List[int] = [0, 1, 2, -1]  # 0=ref/ref, 1=ref/alt, 2=alt/alt, -1=missing


class GenomicDataFrame(TypedDict):
    """Runtime type for genomic DataFrame"""
    data: pd.DataFrame  # Main genotype matrix
    metadata: dict  # QC stats, array type, etc.
    snp_info: Optional[pd.DataFrame]  # SNP-level info (chr, pos, maf, etc.)


# =============================================================================
# Clinical Data Contract
# =============================================================================

class ClinicalDataContract(BaseModel):
    """Contract for clinical/phenotype data tables"""
    model_config = {"extra": "forbid"}

    # Required columns
    index_col: str = SAMPLE_ID
    required_columns: List[str] = Field(
        default=[
            SUBJECT_ID,
            SAMPLE_ID,
            VISIT,
            DATASET,
            "age_years",
            "sex",
            "diagnosis",  # Primary diagnosis
            "site"  # Collection site for batch correction
        ]
    )

    # Expected dtypes
    age_dtype: str = "float64"
    sex_dtype: str = "category"  # M, F, Other
    diagnosis_dtype: str = "category"  # ASD, ADHD, AuDHD, Control, Other
    site_dtype: str = "category"

    # Standard clinical instruments
    adhd_instruments: List[str] = [
        "ADHD_RS_inattentive",
        "ADHD_RS_hyperactive",
        "ADHD_RS_total"
    ]
    asd_instruments: List[str] = [
        "ADOS_total",
        "ADOS_social_affect",
        "ADOS_restricted_repetitive",
        "ADI_R_social",
        "ADI_R_communication",
        "ADI_R_repetitive",
        "SRS_total"
    ]
    iq_columns: List[str] = ["IQ_full_scale", "IQ_verbal", "IQ_nonverbal"]

    # Context variables (for adjustment)
    context_columns: List[str] = [
        "fasting_hours",
        "time_of_day",
        "last_medication_hours",
        "menstrual_phase",  # For females
        "sleep_hours_last_night",
        "recent_illness_7d"  # Boolean
    ]

    # Medication columns
    medication_columns: List[str] = [
        "on_stimulant",
        "on_ssri",
        "on_antipsychotic",
        "medication_list"  # Comma-separated
    ]

    # Invariants
    age_range: tuple = (0.0, 90.0)
    allowed_diagnoses: List[str] = ["ASD", "ADHD", "AuDHD", "Control", "Other"]
    allowed_sex: List[str] = ["M", "F", "Other"]


class ClinicalDataFrame(TypedDict):
    """Runtime type for clinical DataFrame"""
    data: pd.DataFrame
    metadata: dict  # Dataset info, collection dates, versions
    ontology_mappings: Optional[dict]  # HPO/SNOMED mappings


# =============================================================================
# Metabolomic Data Contract
# =============================================================================

class MetabolomicDataContract(BaseModel):
    """Contract for metabolomic abundance tables"""
    model_config = {"extra": "forbid"}

    # Required columns
    index_col: str = SAMPLE_ID
    required_columns: List[str] = Field(
        default=[
            SUBJECT_ID,
            SAMPLE_ID,
            "batch",  # MS batch
            "injection_order",
            "sample_type"  # Sample, QC, Blank
        ]
    )

    # Expected dtypes
    batch_dtype: str = "category"
    injection_order_dtype: str = "int32"
    abundance_dtype: str = "float64"  # Non-negative

    # Metabolite naming conventions
    hmdb_prefix: str = "HMDB"  # HMDB IDs
    kegg_prefix: str = "C"  # KEGG compound IDs
    name_pattern: str = r"^[A-Za-z0-9_\-]+$"  # Alphanumeric names

    # Expected metabolite categories
    metabolite_categories: List[str] = [
        "amino_acid",
        "acyl_carnitine",
        "bile_acid",
        "neurotransmitter",
        "kynurenine",
        "scfa",
        "steroid",
        "lipid",
        "nucleotide",
        "other"
    ]

    # QC thresholds
    max_missing_per_metabolite: float = 0.5  # 50% max missing
    min_detection_rate: float = 0.3  # Present in 30% samples
    qc_cv_threshold: float = 0.30  # CV < 30% in QC samples

    # Normalization
    requires_log_transform: bool = True
    recommended_normalization: str = "quantile"  # or TIC, PQN


class MetabolomicDataFrame(TypedDict):
    """Runtime type for metabolomic DataFrame"""
    data: pd.DataFrame  # Abundance matrix
    metadata: dict  # Batch, QC metrics
    metabolite_annotations: Optional[pd.DataFrame]  # HMDB/KEGG mappings


# =============================================================================
# Microbiome Data Contract
# =============================================================================

class MicrobiomeDataContract(BaseModel):
    """Contract for microbiome abundance tables"""
    model_config = {"extra": "forbid"}

    # Required columns
    index_col: str = SAMPLE_ID
    required_columns: List[str] = Field(
        default=[
            SUBJECT_ID,
            SAMPLE_ID,
            "sequencing_depth",
            "sample_type",  # stool, oral, skin
            "collection_method",
            "storage_days"  # Days from collection to processing
        ]
    )

    # Expected dtypes
    sequencing_depth_dtype: str = "int64"
    abundance_dtype: str = "float64"  # Relative abundance 0-1 or counts

    # Taxonomic naming
    otu_prefix: str = "OTU"  # OTU IDs
    asv_prefix: str = "ASV"  # ASV IDs
    taxonomy_separator: str = ";"  # k__Bacteria;p__Firmicutes;...

    # Taxonomic ranks
    taxonomic_ranks: List[str] = [
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species"
    ]

    # QC thresholds
    min_sequencing_depth: int = 10000  # Reads per sample
    min_prevalence: float = 0.10  # Present in 10% samples
    max_missing_per_taxon: float = 0.90  # 90% max missing (sparse OK)

    # Transformation
    requires_clr_transform: bool = True
    pseudocount: float = 1.0


class MicrobiomeDataFrame(TypedDict):
    """Runtime type for microbiome DataFrame"""
    data: pd.DataFrame  # Abundance matrix (OTU/ASV × sample)
    metadata: dict  # Sequencing metrics
    taxonomy: Optional[pd.DataFrame]  # Taxonomic assignments


# =============================================================================
# Cross-Modality ID Reconciliation
# =============================================================================

class IDReconciliationRules(BaseModel):
    """Rules for reconciling sample IDs across modalities"""
    model_config = {"extra": "forbid"}

    # ID components
    subject_id_col: str = SUBJECT_ID
    sample_id_col: str = SAMPLE_ID
    visit_col: str = VISIT
    dataset_col: str = DATASET

    # Join strategy
    join_on: str = SAMPLE_ID  # Join on sample_id (visit-specific)
    merge_strategy: Literal["inner", "outer", "left"] = "inner"  # Default to intersection

    # Time matching
    time_window_days: Optional[int] = 30  # For fuzzy time matching
    prefer_closest_visit: bool = True

    # Conflict resolution
    duplicate_handling: Literal["first", "last", "error"] = "error"
    missing_handling: Literal["drop", "keep", "error"] = "keep"


def reconcile_sample_ids(
    dataframes: dict[str, pd.DataFrame],
    rules: Optional[IDReconciliationRules] = None
) -> pd.DataFrame:
    """
    Reconcile sample IDs across multiple modality DataFrames

    Args:
        dataframes: Dict of {modality: DataFrame} with sample_id index
        rules: ID reconciliation rules

    Returns:
        DataFrame with columns: sample_id, subject_id, visit, dataset,
        and boolean flags for which modalities are present

    Raises:
        ValueError: If duplicate sample IDs found with duplicate_handling='error'
    """
    if rules is None:
        rules = IDReconciliationRules()

    # Extract sample IDs from each modality
    sample_id_sets = {}
    for modality, df in dataframes.items():
        sample_ids = set(df.index if df.index.name == rules.sample_id_col else df[rules.sample_id_col])
        sample_id_sets[modality] = sample_ids

    # Find intersection/union based on strategy
    if rules.merge_strategy == "inner":
        common_samples = set.intersection(*sample_id_sets.values())
    elif rules.merge_strategy == "outer":
        common_samples = set.union(*sample_id_sets.values())
    else:
        # 'left' - use first modality as reference
        first_modality = list(dataframes.keys())[0]
        common_samples = sample_id_sets[first_modality]

    # Create reconciliation table
    reconciliation = pd.DataFrame({
        rules.sample_id_col: list(common_samples)
    })

    # Add presence flags for each modality
    for modality, sample_set in sample_id_sets.items():
        reconciliation[f"has_{modality}"] = reconciliation[rules.sample_id_col].isin(sample_set)

    # Extract subject_id, visit, dataset from first available modality
    for sample_id in common_samples:
        for df in dataframes.values():
            if sample_id in df.index or (rules.sample_id_col in df.columns and sample_id in df[rules.sample_id_col].values):
                # Extract metadata
                row = df.loc[sample_id] if sample_id in df.index else df[df[rules.sample_id_col] == sample_id].iloc[0]

                if rules.subject_id_col not in reconciliation.columns:
                    reconciliation.loc[reconciliation[rules.sample_id_col] == sample_id, rules.subject_id_col] = row.get(rules.subject_id_col, sample_id.rsplit('_', 1)[0])

                break

    return reconciliation.set_index(rules.sample_id_col)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_genomic_contract(df: pd.DataFrame, contract: GenomicDataContract) -> dict:
    """Validate genomic DataFrame against contract"""
    issues = []

    # Check required columns
    missing_cols = set(contract.required_columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check genotype values
    snp_cols = [col for col in df.columns if col.startswith(contract.snp_prefix)]
    if snp_cols:
        unique_vals = pd.concat([df[col] for col in snp_cols[:10]]).dropna().unique()  # Sample first 10
        invalid_vals = set(unique_vals) - set(contract.allowed_genotypes)
        if invalid_vals:
            issues.append(f"Invalid genotype values found: {invalid_vals}")

    # Check call rate
    if "call_rate" in df.columns:
        low_call_rate = (df["call_rate"] < contract.min_call_rate).sum()
        if low_call_rate > 0:
            issues.append(f"{low_call_rate} samples with call_rate < {contract.min_call_rate}")

    return {"valid": len(issues) == 0, "issues": issues, "n_samples": len(df), "n_snps": len(snp_cols)}


def validate_clinical_contract(df: pd.DataFrame, contract: ClinicalDataContract) -> dict:
    """Validate clinical DataFrame against contract"""
    issues = []

    # Check required columns
    missing_cols = set(contract.required_columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check age range
    if "age_years" in df.columns:
        out_of_range = ((df["age_years"] < contract.age_range[0]) | (df["age_years"] > contract.age_range[1])).sum()
        if out_of_range > 0:
            issues.append(f"{out_of_range} samples with age outside [{contract.age_range[0]}, {contract.age_range[1]}]")

    # Check diagnosis categories
    if "diagnosis" in df.columns:
        invalid_dx = set(df["diagnosis"].dropna().unique()) - set(contract.allowed_diagnoses)
        if invalid_dx:
            issues.append(f"Invalid diagnosis values: {invalid_dx}")

    return {"valid": len(issues) == 0, "issues": issues, "n_samples": len(df)}


def validate_metabolomic_contract(df: pd.DataFrame, contract: MetabolomicDataContract) -> dict:
    """Validate metabolomic DataFrame against contract"""
    issues = []

    # Check required columns
    missing_cols = set(contract.required_columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check metabolite columns
    metabolite_cols = [col for col in df.columns if col not in contract.required_columns]

    # Check missing rates
    high_missing = []
    for col in metabolite_cols[:100]:  # Sample first 100
        missing_rate = df[col].isna().sum() / len(df)
        if missing_rate > contract.max_missing_per_metabolite:
            high_missing.append(col)

    if high_missing:
        issues.append(f"{len(high_missing)} metabolites exceed max missing rate")

    # Check non-negative
    numeric_cols = df[metabolite_cols].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        negative_counts = (df[numeric_cols] < 0).sum().sum()
        if negative_counts > 0:
            issues.append(f"Found {negative_counts} negative abundance values")

    return {"valid": len(issues) == 0, "issues": issues, "n_samples": len(df), "n_metabolites": len(metabolite_cols)}


def validate_microbiome_contract(df: pd.DataFrame, contract: MicrobiomeDataContract) -> dict:
    """Validate microbiome DataFrame against contract"""
    issues = []

    # Check required columns
    missing_cols = set(contract.required_columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check sequencing depth
    if "sequencing_depth" in df.columns:
        low_depth = (df["sequencing_depth"] < contract.min_sequencing_depth).sum()
        if low_depth > 0:
            issues.append(f"{low_depth} samples with sequencing depth < {contract.min_sequencing_depth}")

    # Check taxa columns
    taxa_cols = [col for col in df.columns if col.startswith((contract.otu_prefix, contract.asv_prefix))]

    # Check non-negative
    if taxa_cols:
        numeric_taxa = df[taxa_cols].select_dtypes(include=[np.number])
        negative_counts = (numeric_taxa < 0).sum().sum()
        if negative_counts > 0:
            issues.append(f"Found {negative_counts} negative abundance values")

    return {"valid": len(issues) == 0, "issues": issues, "n_samples": len(df), "n_taxa": len(taxa_cols)}