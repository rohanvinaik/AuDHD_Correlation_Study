"""
Pydantic configuration schemas for type safety and validation
"""
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Optional, Literal
from pathlib import Path


class DataConfig(BaseModel):
    """Data sources and paths configuration"""
    model_config = ConfigDict(extra="forbid")

    roots: Dict[str, str] = Field(
        ...,
        description="Data root paths by name, e.g., {'raw': 'data/raw', 'processed': 'data/processed'}"
    )
    datasets: List[str] = Field(
        default=["SPARK", "SSC", "ABCD", "UKB"],
        description="Datasets to use"
    )
    context_fields: List[str] = Field(
        default=[
            "fasting",
            "clock_time",
            "last_dose_hours",
            "storage_days",
            "menstrual_phase",
            "fever_72h",
            "recent_illness",
            "antibiotics_30d"
        ],
        description="Context/covariate fields for adjustment"
    )
    ancestry_cols: List[str] = Field(
        default=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"],
        description="Genetic ancestry principal components"
    )
    dua_required: bool = Field(
        default=True,
        description="Whether DUA checks are required"
    )

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, v: List[str]) -> List[str]:
        """Validate dataset names"""
        valid = {"SPARK", "SSC", "ABCD", "UKB", "MetaboLights", "HCP"}
        invalid = set(v) - valid
        if invalid:
            raise ValueError(f"Invalid datasets: {invalid}. Valid: {valid}")
        return v


class FeatureConfig(BaseModel):
    """Feature selection configuration"""
    model_config = ConfigDict(extra="forbid")

    genetics: Dict[str, List[str]] = Field(
        default={
            "neurotransmitter_genes": [
                "TPH1", "TPH2", "TH", "DDC", "GAD1", "GAD2",
                "COMT", "MAOA", "MAOB", "SLC6A4", "SLC6A3", "SLC6A1"
            ],
            "receptor_genes": ["DRD1", "DRD2", "DRD3", "DRD4", "DRD5"],
            "prs_sets": [
                "PRS_autism", "PRS_ADHD", "PRS_depression", "PRS_schizophrenia",
                "PRS_BMI", "PRS_smoking", "PRS_inflammation_CRP", "PRS_sleep"
            ],
            "structural_variants": ["CNV_burden", "SV_burden", "repeat_expansions"],
            "rare_variants": ["LoF_burden", "missense_burden", "de_novo_variants"],
            "mitochondrial": ["mtDNA_CN", "OXPHOS_gene_sets"],
        },
        description="Genetic feature panels"
    )
    metabolomics: Dict[str, List[str]] = Field(
        default={
            "neurotransmitters": [
                "serotonin", "5-HIAA", "dopamine", "DOPAC", "HVA",
                "GABA", "glutamate", "glutamine"
            ],
            "kynurenine_pathway": [
                "kynurenine", "kynurenic_acid", "quinolinic_acid"
            ],
            "acyl_carnitines": ["C0", "C2", "C3", "C4", "C5", "C8", "C14", "C16", "C18"],
            "bile_acids": ["CA", "CDCA", "DCA", "LCA", "UDCA"],
            "inflammatory_markers": ["IL1b", "IL6", "IL10", "IFNg", "TNFalpha", "CRP"],
            "steroids": ["cortisol", "cortisone", "DHEA_S", "testosterone", "estradiol"],
            "gut_metabolites": [
                "SCFA_acetate", "SCFA_butyrate", "SCFA_propionate",
                "TMAO", "indoxyl_sulfate", "p_cresyl_sulfate"
            ],
        },
        description="Metabolomic feature panels"
    )
    clinical: Dict[str, List[str]] = Field(
        default={
            "core_instruments": [
                "ADOS_score", "ADI_R_scores", "SRS_score",
                "ADHD_RS_inattentive", "ADHD_RS_hyperactive"
            ],
            "cognitive": ["IQ_full", "IQ_verbal", "IQ_nonverbal"],
            "comorbidities": [
                "tics", "OCD", "anxiety", "depression", "epilepsy",
                "POTS", "MCAS", "IBS", "GERD"
            ],
        },
        description="Clinical feature sets"
    )
    microbiome: Dict[str, List[str]] = Field(
        default={
            "diversity": ["alpha_diversity", "beta_diversity", "enterotypes"],
            "taxa": ["phylum_level", "genus_level", "species_level"],
            "functional": ["SCFA_production", "indole_production"],
        },
        description="Microbiome features"
    )
    imaging: Dict[str, List[str]] = Field(
        default={
            "EEG": ["delta_power", "theta_power", "alpha_power", "aperiodic_1f_slope"],
            "MRI_structural": ["total_brain_volume", "gray_matter_volume"],
        },
        description="Neuroimaging/EEG features"
    )


class PreprocessConfig(BaseModel):
    """Preprocessing configuration"""
    model_config = ConfigDict(extra="forbid")

    imputation: Literal["mice_delta", "knn", "iterative"] = Field(
        default="mice_delta",
        description="Imputation method"
    )
    batch_method: Literal["combat", "ruv", "lmm", "harmonize"] = Field(
        default="combat",
        description="Batch/site correction method"
    )
    scaling: Dict[str, Literal["standard", "robust", "minmax", "clr"]] = Field(
        default={
            "genetic": "standard",
            "metabolomic": "robust",
            "clinical": "standard",
            "microbiome": "clr"
        },
        description="Scaling method per modality"
    )
    adjust_covariates: List[str] = Field(
        default=[
            "age", "sex", "BMI", "pubertal_stage", "site",
            "fasting", "time_of_day", "last_medication_dose"
        ],
        description="Covariates to adjust for"
    )
    outlier_method: Literal["robust_mahalanobis", "isolation_forest", "none"] = Field(
        default="robust_mahalanobis",
        description="Outlier detection method"
    )
    outlier_action: Literal["winsorize", "remove", "flag"] = Field(
        default="winsorize",
        description="Action to take on outliers"
    )
    missing_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maximum proportion of missing values allowed per feature"
    )


class IntegrateConfig(BaseModel):
    """Multi-omics integration configuration"""
    model_config = ConfigDict(extra="forbid")

    method: Literal["stack", "mofa2", "diablo", "gmkf", "vae"] = Field(
        default="mofa2",
        description="Integration method"
    )
    weights: Dict[str, float] = Field(
        default={
            "genetic": 0.25,
            "metabolomic": 0.35,
            "clinical": 0.25,
            "microbiome": 0.15
        },
        description="Layer weights for stack method"
    )
    n_factors: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of latent factors (MOFA2/DIABLO)"
    )
    split_by: List[str] = Field(
        default=["family_id", "site"],
        description="Variables to split by for validation"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Test set proportion"
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure weights sum to approximately 1.0"""
        total = sum(v.values())
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Weights must sum to ~1.0, got {total}")
        return v


class ClusterConfig(BaseModel):
    """Clustering configuration"""
    model_config = ConfigDict(extra="forbid")

    embeddings: Dict[str, Dict] = Field(
        default={
            "umap": {
                "n_neighbors": [15, 30, 50],
                "min_dist": [0.1, 0.25, 0.5],
                "n_components": 2,
            },
            "tsne": {
                "perplexity": [10, 30, 50, 100],
                "n_components": 2,
                "n_iter": 2000,
            }
        },
        description="Embedding parameters"
    )
    clusterers: Dict[str, Dict] = Field(
        default={
            "hdbscan": {
                "min_cluster_size": 50,
                "min_samples": 10,
            },
            "lca": {
                "n_components": 10,
                "weight_concentration_prior_type": "dirichlet_process",
            }
        },
        description="Clustering algorithm parameters"
    )
    consensus: Dict[str, int] = Field(
        default={
            "n_resamples": 100,
            "agreement_threshold": 70,  # percentage
        },
        description="Consensus clustering parameters"
    )
    topology_enabled: bool = Field(
        default=True,
        description="Enable topological data analysis"
    )
    min_gap_score: float = Field(
        default=1.5,
        ge=1.0,
        description="Minimum gap score for cluster separation"
    )
    min_stability: float = Field(
        default=0.7,
        ge=0.5,
        le=1.0,
        description="Minimum bootstrap stability score"
    )


class ValidateConfig(BaseModel):
    """Validation configuration"""
    model_config = ConfigDict(extra="forbid")

    stability_bootstrap: int = Field(
        default=100,
        ge=50,
        le=1000,
        description="Number of bootstrap iterations for stability"
    )
    external_cohorts: List[str] = Field(
        default=["ABCD_holdout", "UKB_validation"],
        description="External validation cohorts"
    )
    sensitivity_scenarios: List[str] = Field(
        default=[
            "no_meds",
            "fasting_only",
            "circadian_adjusted",
            "mnar_sensitivity",
            "within_family"
        ],
        description="Sensitivity analysis scenarios"
    )
    leave_site_out: bool = Field(
        default=True,
        description="Perform leave-site-out cross-validation"
    )
    cross_ancestry: bool = Field(
        default=True,
        description="Perform cross-ancestry validation"
    )
    biological_validation: bool = Field(
        default=True,
        description="Test biological hypotheses (metabolites, pathways)"
    )


class CausalConfig(BaseModel):
    """Causal inference configuration"""
    model_config = ConfigDict(extra="forbid")

    mr_instruments: List[str] = Field(
        default=["PRS_CRP", "PRS_BMI", "PRS_smoking", "PRS_sleep"],
        description="Mendelian randomization instruments (PRS)"
    )
    mediation_triplets: List[Dict[str, str]] = Field(
        default=[
            {
                "exposure": "rare_variants",
                "mediator": "serotonin",
                "outcome": "symptom_severity"
            },
            {
                "exposure": "PRS_ADHD",
                "mediator": "dopamine",
                "outcome": "ADHD_RS"
            },
            {
                "exposure": "genetics",
                "mediator": "microbiome_diversity",
                "outcome": "GI_symptoms"
            }
        ],
        description="Mediation analysis pathways"
    )
    gxe_pairs: List[Dict[str, str]] = Field(
        default=[
            {"genetic": "rare_variants", "environment": "maternal_SSRI"},
            {"genetic": "PRS_autism", "environment": "prenatal_infection"},
            {"genetic": "TPH2", "environment": "early_life_stress"}
        ],
        description="Gene-environment interaction pairs"
    )
    dag_enabled: bool = Field(
        default=True,
        description="Use DAG-informed adjustment sets"
    )
    e_value_enabled: bool = Field(
        default=True,
        description="Calculate E-values for unmeasured confounding"
    )

    @field_validator("mediation_triplets")
    @classmethod
    def validate_mediation(cls, v: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Ensure mediation triplets have required keys"""
        required = {"exposure", "mediator", "outcome"}
        for triplet in v:
            if not required.issubset(triplet.keys()):
                raise ValueError(f"Mediation triplet must have keys: {required}")
        return v


class VizConfig(BaseModel):
    """Visualization configuration"""
    model_config = ConfigDict(extra="forbid")

    library: Literal["plotly", "bokeh"] = Field(
        default="plotly",
        description="Interactive plotting library"
    )
    overlays: List[str] = Field(
        default=[
            "cluster", "diagnosis", "serotonin_level", "dopamine_level",
            "genetic_burden", "GI_issues", "medication_response"
        ],
        description="Embedding plot overlays"
    )
    output_formats: List[Literal["html", "png", "svg", "pdf"]] = Field(
        default=["html", "png"],
        description="Output formats"
    )
    dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="DPI for raster outputs"
    )
    theme: Literal["default", "colorblind_safe", "grayscale"] = Field(
        default="colorblind_safe",
        description="Color theme"
    )


class ReportConfig(BaseModel):
    """Report generation configuration"""
    model_config = ConfigDict(extra="forbid")

    types: List[Literal["executive_summary", "technical_report", "clinician_card"]] = Field(
        default=["executive_summary", "technical_report", "clinician_card"],
        description="Report types to generate"
    )
    output_formats: List[Literal["markdown", "html", "pdf"]] = Field(
        default=["markdown", "html"],
        description="Output formats"
    )
    include_code: bool = Field(
        default=False,
        description="Include code in technical report"
    )
    decision_support_enabled: bool = Field(
        default=True,
        description="Generate clinical decision support tables"
    )


class AppConfig(BaseModel):
    """Root configuration schema"""
    model_config = ConfigDict(extra="forbid")

    # Global settings
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )
    n_jobs: int = Field(
        default=4,
        ge=1,
        description="Number of parallel jobs"
    )
    verbose: bool = Field(
        default=True,
        description="Verbose output"
    )

    # Sub-configs
    data: DataConfig
    features: FeatureConfig
    preprocess: PreprocessConfig
    integrate: IntegrateConfig
    cluster: ClusterConfig
    validate: ValidateConfig
    causal: CausalConfig
    viz: VizConfig = Field(default_factory=VizConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    # Paths
    data_root: Path = Field(
        default=Path("data"),
        description="Root data directory"
    )
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Output directory"
    )

    @field_validator("data_root", "output_dir")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Convert string to Path if needed"""
        return Path(v) if not isinstance(v, Path) else v