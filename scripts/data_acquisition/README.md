# Data Acquisition Scripts

**Purpose**: Automated download scripts for open-access autism/ADHD datasets

## Available Scripts

### 1. `download_abide.sh`

**Dataset**: ABIDE (Autism Brain Imaging Data Exchange)
**Access**: Open (no registration)
**Size**: 100+ GB (depending on preprocessing pipeline selected)
**Sample Size**: 1,112 autism, 1,033 controls

**Usage**:
```bash
./download_abide.sh

# Or specify version
ABIDE_VERSION=ABIDE_I ./download_abide.sh
ABIDE_VERSION=ABIDE_II ./download_abide.sh
ABIDE_VERSION=both ./download_abide.sh
```

**Requirements**:
- AWS CLI (`pip install awscli` or `brew install awscli`)
- 100+ GB free disk space

**Output**: `data/raw/abide/`

---

### 2. `download_gwas_sumstats.sh`

**Dataset**: GWAS summary statistics for autism and ADHD
**Access**: Open (no registration)
**Size**: ~5 GB
**Studies**: Grove et al. 2019 (autism), Demontis et al. 2019 (ADHD), others

**Usage**:
```bash
./download_gwas_sumstats.sh
```

**Requirements**:
- curl

**Output**: `data/raw/gwas_sumstats/`

**Includes**:
- iPSYCH + PGC autism GWAS
- SPARK autism GWAS
- iPSYCH + PGC ADHD GWAS (top hits, full sumstats require application)
- GWAS Catalog curated associations
- Cross-disorder psychiatric genomics

---

### 3. `download_geo_studies.R`

**Dataset**: Gene Expression Omnibus (GEO) autism/ADHD studies
**Access**: Open (no registration)
**Size**: ~50 GB
**Studies**: 6 key autism/ADHD expression studies

**Usage**:
```R
# From R/RStudio
source('download_geo_studies.R')

# Or from command line
Rscript download_geo_studies.R
```

**Requirements**:
- R (version 4.0+)
- Bioconductor packages: `GEOquery`, `Biobase`, `limma`

**Output**: `data/raw/geo/`

**Includes**:
- GSE113834 - Autism brain transcriptomics
- GSE111176 - iPSC-derived neurons from ASD
- GSE42133 - Autism blood expression
- GSE28521 - Autism brain development
- GSE67530 - ADHD blood expression
- GSE119945 - Autism brain organoids

---

### 4. `download_adhd200.sh` (TODO)

**Dataset**: ADHD-200 neuroimaging
**Access**: Open (no registration)
**Size**: ~50 GB
**Sample Size**: 491 ADHD, 482 controls

**Usage**:
```bash
./download_adhd200.sh
```

---

### 5. `download_openeuro.sh` (TODO)

**Dataset**: OpenNeuro autism/ADHD neuroimaging studies
**Access**: Open (no registration)
**Size**: Varies by dataset

**Usage**:
```bash
./download_openeuro.sh ds000228 ds002424
```

---

## Installation

### Prerequisites

**Required for all scripts**:
```bash
# Basic tools (usually pre-installed)
curl
wget
```

**For ABIDE (AWS S3 download)**:
```bash
# Install AWS CLI
pip install awscli
# Or on macOS
brew install awscli
```

**For GEO studies (R scripts)**:
```bash
# Install R (if not already installed)
# macOS
brew install r

# Ubuntu/Debian
sudo apt-get install r-base

# Then install Bioconductor packages
R
> if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
> BiocManager::install(c("GEOquery", "Biobase", "limma"))
```

---

## Quick Start

### Download All Open-Access Data

```bash
# 1. GWAS summary statistics (fastest, ~10 minutes)
./download_gwas_sumstats.sh

# 2. GEO expression studies (~30 minutes)
Rscript download_geo_studies.R

# 3. ABIDE neuroimaging (slowest, 1-4 hours depending on pipeline)
# Note: Select phenotypes only to test, or full pipeline for analysis
./download_abide.sh
```

### Check Available Disk Space

```bash
# Check available space
df -h

# Estimate space needed:
# - GWAS sumstats: 5 GB
# - GEO studies: 50 GB
# - ABIDE (phenotypes only): 1 GB
# - ABIDE (one pipeline): 100 GB
# - ABIDE (all pipelines): 400+ GB
```

---

## Data Organization

After download, data will be organized as:

```
data/raw/
├── abide/
│   ├── ABIDE_I/
│   │   ├── Phenotypic_V1_0b_preprocessed1.csv
│   │   └── cpac/  # If preprocessed data downloaded
│   └── README.md
├── gwas_sumstats/
│   ├── autism/
│   │   ├── iPSYCH_PGC_ASD_Nov2017.gz
│   │   └── gwas_catalog_autism.tsv
│   ├── adhd/
│   │   └── gwas_catalog_adhd.tsv
│   └── inventory.json
└── geo/
    ├── GSE113834/
    │   ├── expression_matrix.csv
    │   ├── phenotype_data.csv
    │   └── README.md
    └── inventory.json
```

---

## Usage After Download

### Load Data in Pipeline

```python
from audhd_correlation import Pipeline

# Use pre-configured settings
pipeline = Pipeline(config_path="configs/datasets/abide.yaml")

# Or specify paths directly
from audhd_correlation.data import load_geo_data
geo_data = load_geo_data("data/raw/geo/GSE113834")
```

### Quick Analysis

```python
# GWAS summary statistics
from audhd_correlation.biological import load_gwas_hits

gwas_hits = load_gwas_hits(
    'data/raw/gwas_sumstats/autism/iPSYCH_PGC_ASD_Nov2017.gz',
    p_threshold=5e-8
)

# GEO expression data
import pandas as pd
expr = pd.read_csv('data/raw/geo/GSE113834/expression_matrix.csv', index_col=0)
pheno = pd.read_csv('data/raw/geo/GSE113834/phenotype_data.csv', index_col=0)
```

---

## Troubleshooting

### AWS CLI Not Found
```bash
# Install AWS CLI
pip install awscli

# Or on macOS
brew install awscli

# Verify installation
aws --version
```

### R Package Installation Fails
```bash
# Update R to latest version
# Then retry Bioconductor installation

# If still failing, install system dependencies
# macOS
brew install openssl libxml2

# Ubuntu/Debian
sudo apt-get install libssl-dev libxml2-dev libcurl4-openssl-dev
```

### Download Interrupted
```bash
# Most scripts can be re-run safely
# They will skip already-downloaded files
./download_gwas_sumstats.sh  # Re-run to resume
```

### Disk Space Issues
```bash
# Download only what you need

# For ABIDE: Download phenotypes only initially
# Edit download_abide.sh and set:
download_preprocessed="n"

# For GEO: Comment out large studies in download_geo_studies.R
```

---

## Citation

If you use data downloaded with these scripts, please cite both:

1. **This pipeline**:
   ```
   AuDHD Correlation Study. Data Acquisition Scripts.
   https://github.com/rohanvinaik/AuDHD_Correlation_Study
   ```

2. **Original data sources** (citations in each dataset's README)

---

## Data Use Policies

**Open-access data** (downloaded with these scripts) typically allows:
- ✅ Academic research
- ✅ Publication of results
- ✅ Sharing of summary statistics

But typically prohibits:
- ❌ Commercial use without permission
- ❌ Re-identification of participants
- ❌ Redistribution of raw data

Always check the specific data use policy for each dataset.

---

## Support

**Issues with scripts**: Open issue at github.com/rohanvinaik/AuDHD_Correlation_Study/issues

**Issues with data**: Contact original data providers
- ABIDE: fcon_1000.projects.nitrc.org/indi/abide/
- GEO: www.ncbi.nlm.nih.gov/geo/
- GWAS Catalog: www.ebi.ac.uk/gwas/

---

## Contributing

To add a new download script:

1. Create script: `download_[dataset].sh` or `download_[dataset].R`
2. Follow naming convention and structure of existing scripts
3. Include:
   - Dataset description
   - Requirements check
   - Error handling
   - Progress indicators
   - README generation
4. Update this README
5. Submit pull request

---

**Last Updated**: 2025-01-30