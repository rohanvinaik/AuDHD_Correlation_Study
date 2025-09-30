#!/usr/bin/env Rscript
# Download autism/ADHD gene expression studies from GEO (Gene Expression Omnibus)
# Requires: Bioconductor GEOquery package

# Configuration
output_dir <- Sys.getenv("DATA_ROOT", "../../data")
output_dir <- file.path(output_dir, "raw", "geo")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Check and install required packages
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}

required_packages <- c("GEOquery", "Biobase", "limma")
for (pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        BiocManager::install(pkg)
    }
}

library(GEOquery)
library(Biobase)

cat("=== GEO Gene Expression Studies Download ===\n")
cat("Output directory:", output_dir, "\n\n")

# ============================================================================
# DEFINE KEY STUDIES
# ============================================================================

studies <- list(
    # Autism brain transcriptomics
    list(
        geo_id = "GSE113834",
        title = "Autism brain transcriptomics",
        tissue = "brain (cortex)",
        platform = "RNA-seq",
        samples = 104,
        description = "Transcriptomic analysis of autism brain tissue",
        pmid = "30545852"
    ),

    # iPSC-derived neurons from ASD patients
    list(
        geo_id = "GSE111176",
        title = "iPSC-derived neurons from ASD patients",
        tissue = "iPSC-neurons",
        platform = "RNA-seq",
        samples = 84,
        description = "iPSC-derived neurons from autism patients vs controls",
        pmid = "30545852"
    ),

    # Autism blood expression profiling
    list(
        geo_id = "GSE42133",
        title = "Autism blood expression profiling",
        tissue = "blood",
        platform = "Microarray (Illumina HumanHT-12 V4.0)",
        samples = 82,
        description = "Whole blood gene expression in autism",
        pmid = "23527830"
    ),

    # Autism brain development
    list(
        geo_id = "GSE28521",
        title = "Autism brain development",
        tissue = "brain (multiple regions)",
        platform = "Microarray (Affymetrix Human Genome U133 Plus 2.0)",
        samples = 58,
        description = "Transcriptomic analysis across brain development in autism",
        pmid = "21614001"
    ),

    # ADHD peripheral blood
    list(
        geo_id = "GSE67530",
        title = "ADHD peripheral blood gene expression",
        tissue = "blood",
        platform = "Microarray (Affymetrix Human Gene 1.0 ST)",
        samples = 40,
        description = "Gene expression in ADHD vs controls",
        pmid = "26238637"
    ),

    # Autism organoids
    list(
        geo_id = "GSE119945",
        title = "Autism patient-derived brain organoids",
        tissue = "brain organoids",
        platform = "RNA-seq",
        samples = 15,
        description = "Brain organoids from autism patients with macrocephaly",
        pmid = "30545852"
    )
)

# ============================================================================
# DOWNLOAD FUNCTION
# ============================================================================

download_geo_study <- function(study, output_dir) {
    cat("\n==================================================\n")
    cat("Downloading:", study$geo_id, "-", study$title, "\n")
    cat("Samples:", study$samples, "\n")
    cat("Platform:", study$platform, "\n")
    cat("==================================================\n\n")

    study_dir <- file.path(output_dir, study$geo_id)
    dir.create(study_dir, recursive = TRUE, showWarnings = FALSE)

    tryCatch({
        # Download series matrix (includes expression data and phenotypes)
        cat("Downloading series matrix...\n")
        gse <- getGEO(study$geo_id, destdir = study_dir, GSEMatrix = TRUE)

        # Handle list or single GSE
        if (is.list(gse)) {
            gse <- gse[[1]]
        }

        # Extract expression data
        cat("Extracting expression matrix...\n")
        expr_matrix <- exprs(gse)
        write.csv(expr_matrix, file.path(study_dir, "expression_matrix.csv"))

        # Extract phenotype data
        cat("Extracting phenotype data...\n")
        pheno_data <- pData(gse)
        write.csv(pheno_data, file.path(study_dir, "phenotype_data.csv"))

        # Extract feature data (gene annotations)
        cat("Extracting feature annotations...\n")
        feature_data <- fData(gse)
        write.csv(feature_data, file.path(study_dir, "feature_annotations.csv"))

        # Create study metadata file
        metadata <- list(
            geo_id = study$geo_id,
            title = study$title,
            tissue = study$tissue,
            platform = study$platform,
            samples = study$samples,
            description = study$description,
            pmid = study$pmid,
            download_date = Sys.Date(),
            n_samples = ncol(expr_matrix),
            n_features = nrow(expr_matrix),
            expression_range = c(min = min(expr_matrix), max = max(expr_matrix))
        )

        # Save metadata as JSON
        metadata_json <- jsonlite::toJSON(metadata, pretty = TRUE, auto_unbox = TRUE)
        writeLines(metadata_json, file.path(study_dir, "metadata.json"))

        # Create README
        readme <- sprintf("# %s

**GEO ID**: %s
**Download Date**: %s

## Study Information

- **Title**: %s
- **Tissue**: %s
- **Platform**: %s
- **Samples**: %d
- **PubMed ID**: %s

## Description

%s

## Files

- `expression_matrix.csv` - Gene expression matrix (features × samples)
- `phenotype_data.csv` - Sample phenotypes and metadata
- `feature_annotations.csv` - Gene/probe annotations
- `metadata.json` - Study metadata

## Usage

```R
# Load expression data
expr <- read.csv('expression_matrix.csv', row.names = 1)
pheno <- read.csv('phenotype_data.csv', row.names = 1)

# Basic QC
library(ggplot2)
boxplot(expr[, 1:10], main = 'Expression distribution (first 10 samples)')

# Differential expression analysis
library(limma)
design <- model.matrix(~diagnosis, data = pheno)
fit <- lmFit(expr, design)
fit <- eBayes(fit)
results <- topTable(fit, n = Inf)
```

## Citation

PubMed ID: %s

## Notes

Downloaded using GEOquery from Bioconductor.
",
            study$geo_id,
            study$geo_id,
            Sys.Date(),
            study$title,
            study$tissue,
            study$platform,
            study$samples,
            study$pmid,
            study$description,
            study$pmid
        )

        writeLines(readme, file.path(study_dir, "README.md"))

        cat("✓ Successfully downloaded", study$geo_id, "\n")
        return(TRUE)

    }, error = function(e) {
        cat("✗ Error downloading", study$geo_id, ":", conditionMessage(e), "\n")
        return(FALSE)
    })
}

# ============================================================================
# DOWNLOAD ALL STUDIES
# ============================================================================

cat("Starting downloads...\n")
cat("This may take 10-30 minutes depending on dataset sizes.\n\n")

results <- lapply(studies, function(study) {
    download_geo_study(study, output_dir)
})

# ============================================================================
# CREATE MASTER INVENTORY
# ============================================================================

cat("\nCreating master inventory...\n")

inventory <- list(
    download_date = as.character(Sys.Date()),
    total_studies = length(studies),
    studies = studies,
    successful_downloads = sum(unlist(results)),
    failed_downloads = length(studies) - sum(unlist(results))
)

inventory_json <- jsonlite::toJSON(inventory, pretty = TRUE, auto_unbox = TRUE)
writeLines(inventory_json, file.path(output_dir, "inventory.json"))

# Create master README
master_readme <- sprintf("# GEO Gene Expression Studies

**Download Date**: %s
**Total Studies**: %d
**Successful Downloads**: %d
**Failed Downloads**: %d

## Available Studies

",
    Sys.Date(),
    length(studies),
    sum(unlist(results)),
    length(studies) - sum(unlist(results))
)

for (study in studies) {
    master_readme <- paste0(master_readme, sprintf(
        "### %s\n- **GEO ID**: %s\n- **Tissue**: %s\n- **Platform**: %s\n- **Samples**: %d\n- **Description**: %s\n\n",
        study$title,
        study$geo_id,
        study$tissue,
        study$platform,
        study$samples,
        study$description
    ))
}

master_readme <- paste0(master_readme, "

## Usage in Pipeline

### 1. Load and QC

```python
from audhd_correlation.data import load_geo_data

# Load expression data
geo_data = load_geo_data('data/raw/geo/GSE113834')
expr = geo_data['expression']
pheno = geo_data['phenotype']

# QC filtering
from audhd_correlation.preprocess import qc_expression_data
expr_qc = qc_expression_data(expr, min_expression=1, min_samples=10)
```

### 2. Differential Expression Analysis

```python
from audhd_correlation.analysis import differential_expression

# Find differentially expressed genes
de_results = differential_expression(
    expr_qc,
    pheno['diagnosis'],
    method='limma',
    fdr_threshold=0.05
)

# Top genes
top_genes = de_results.query('FDR < 0.05 & abs(logFC) > 1')
```

### 3. Integration with GWAS

```python
# Map GWAS hits to expression data
from audhd_correlation.biological import integrate_gwas_expression

gwas_expr = integrate_gwas_expression(
    gwas_sumstats='data/raw/gwas_sumstats/autism/iPSYCH_PGC_ASD_Nov2017.gz',
    expression_data=expr_qc,
    window=500000  # 500kb window around SNPs
)
```

### 4. Co-expression Network Analysis

```python
from audhd_correlation.biological import build_coexpression_network

# Build weighted gene co-expression network
network = build_coexpression_network(
    expr_qc,
    method='wgcna',
    min_module_size=30
)

# Identify hub genes
hub_genes = network.get_hub_genes(top_n=50)
```

## Key Studies by Category

### Brain Tissue
- GSE113834 - Autism brain transcriptomics (cortex)
- GSE28521 - Autism brain development (multiple regions)

### Cell Models
- GSE111176 - iPSC-derived neurons from ASD patients
- GSE119945 - Autism patient-derived brain organoids

### Blood/Peripheral
- GSE42133 - Autism blood expression profiling
- GSE67530 - ADHD peripheral blood gene expression

## Quality Control Recommendations

1. **Check platform**: Microarray vs RNA-seq require different normalization
2. **Batch effects**: Use ComBat or similar for cross-study integration
3. **Outlier detection**: Remove outlier samples (PCA, hierarchical clustering)
4. **Filter low expression**: Remove genes with low/no expression
5. **Normalize**: Use RMA (microarray) or TMM/DESeq2 (RNA-seq)

## Citations

Key papers for each study are listed in individual study READMEs.

## Directory Structure

```
geo/
├── GSE113834/              # Autism brain transcriptomics
│   ├── expression_matrix.csv
│   ├── phenotype_data.csv
│   ├── feature_annotations.csv
│   ├── metadata.json
│   └── README.md
├── GSE111176/              # iPSC-derived neurons
├── GSE42133/               # Autism blood expression
├── GSE28521/               # Autism brain development
├── GSE67530/               # ADHD blood expression
├── GSE119945/              # Autism organoids
├── inventory.json          # Master study inventory
└── README.md               # This file
```

## Notes

- GEO data is open access (no DUA required)
- Expression values may be log-transformed or raw (check metadata)
- Phenotype annotations vary by study
- Some studies may have restricted access supplementary data (genomic)
")

writeLines(master_readme, file.path(output_dir, "README.md"))

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n==================================================\n")
cat("Download Summary\n")
cat("==================================================\n")
cat("Total studies:", length(studies), "\n")
cat("Successful downloads:", sum(unlist(results)), "\n")
cat("Failed downloads:", length(studies) - sum(unlist(results)), "\n")
cat("\nOutput directory:", output_dir, "\n")
cat("\nNext steps:\n")
cat("  1. Review individual study READMEs\n")
cat("  2. QC and normalize expression data\n")
cat("  3. Integrate with other omics layers\n")
cat("  4. Run differential expression analysis\n")
cat("\n")

if (sum(unlist(results)) < length(studies)) {
    cat("⚠ Some downloads failed. Check error messages above.\n")
    cat("You can re-run this script to retry failed downloads.\n")
}