#!/bin/bash
# Download GWAS summary statistics for autism and ADHD
# From GWAS Catalog and individual study repositories
# No registration required - open access summary statistics

set -euo pipefail

# Configuration
DATASET_NAME="gwas_sumstats"
OUTPUT_DIR="${DATA_ROOT:-../../data}/raw/${DATASET_NAME}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== GWAS Summary Statistics Download ===${NC}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"/{autism,adhd,shared}
cd "${OUTPUT_DIR}"

# ============================================================================
# AUTISM GWAS SUMMARY STATISTICS
# ============================================================================

echo -e "${YELLOW}Downloading Autism GWAS summary statistics...${NC}"
cd autism

# Grove et al. 2019 - iPSYCH + PGC Autism GWAS (18,381 cases, 27,969 controls)
echo "1. Grove et al. 2019 - iPSYCH + PGC Autism"
curl -L -o iPSYCH_PGC_ASD_Nov2017.gz \
    "https://figshare.com/ndownloader/files/28169292" 2>/dev/null || {
    echo -e "${RED}Warning: Could not download Grove et al. 2019${NC}"
}

# Matoba et al. 2020 - SPARK Autism GWAS
echo "2. Matoba et al. 2020 - SPARK Autism"
curl -L -o SPARK_ASD_2020.txt.gz \
    "https://s3.amazonaws.com/imlab-open/Data/SPARK/SPARK_ASD_2020.txt.gz" 2>/dev/null || {
    echo -e "${RED}Warning: Could not download Matoba et al. 2020${NC}"
}

# Autism from GWAS Catalog
echo "3. GWAS Catalog - Autism Spectrum Disorder"
curl -o gwas_catalog_autism.tsv \
    "https://www.ebi.ac.uk/gwas/api/search/downloads/alternative?q=efo_trait:EFO_0003756&pvalfilter=&orfilter=&betafilter=&datefilter=&genomicfilter=&genotypingfilter[]=&traitfilter[]=&dateaddedfilter=&facet=association&efo=true" 2>/dev/null || {
    echo -e "${RED}Warning: Could not download GWAS Catalog autism data${NC}"
}

cd ..

# ============================================================================
# ADHD GWAS SUMMARY STATISTICS
# ============================================================================

echo -e "\n${YELLOW}Downloading ADHD GWAS summary statistics...${NC}"
cd adhd

# Demontis et al. 2019 - iPSYCH + PGC ADHD GWAS (20,183 cases, 35,191 controls)
echo "1. Demontis et al. 2019 - iPSYCH + PGC ADHD"
echo "Note: Full summary statistics require application to iPSYCH"
echo "Downloading top hits and summary data from GWAS Catalog..."

curl -o gwas_catalog_adhd.tsv \
    "https://www.ebi.ac.uk/gwas/api/search/downloads/alternative?q=efo_trait:EFO_0003888&pvalfilter=&orfilter=&betafilter=&datefilter=&genomicfilter=&genotypingfilter[]=&traitfilter[]=&dateaddedfilter=&facet=association&efo=true" 2>/dev/null || {
    echo -e "${RED}Warning: Could not download GWAS Catalog ADHD data${NC}"
}

# Attention from GWAS Catalog (related trait)
echo "2. GWAS Catalog - Attention related traits"
curl -o gwas_catalog_attention.tsv \
    "https://www.ebi.ac.uk/gwas/api/search/downloads/alternative?q=attention&pvalfilter=&orfilter=&betafilter=&datefilter=&genomicfilter=&genotypingfilter[]=&traitfilter[]=&dateaddedfilter=&facet=association&efo=true" 2>/dev/null || {
    echo -e "${RED}Warning: Could not download attention traits${NC}"
}

cd ..

# ============================================================================
# SHARED GENETIC ARCHITECTURE STUDIES
# ============================================================================

echo -e "\n${YELLOW}Downloading cross-disorder GWAS...${NC}"
cd shared

# Cross-Disorder Group 2019 - Genomic Relationships Between Psychiatric Disorders
echo "1. Cross-Disorder Group - Psychiatric Genomics Consortium"
curl -L -o cross_disorder_2019.txt.gz \
    "https://figshare.com/ndownloader/files/14671989" 2>/dev/null || {
    echo -e "${RED}Warning: Could not download cross-disorder study${NC}"
}

# Neurodevelopmental disorders
echo "2. Neurodevelopmental disorders"
curl -o gwas_catalog_neurodevelopmental.tsv \
    "https://www.ebi.ac.uk/gwas/api/search/downloads/alternative?q=neurodevelopmental&pvalfilter=&orfilter=&betafilter=&datefilter=&genomicfilter=&genotypingfilter[]=&traitfilter[]=&dateaddedfilter=&facet=association&efo=true" 2>/dev/null || {
    echo -e "${RED}Warning: Could not download neurodevelopmental traits${NC}"
}

cd ..

# ============================================================================
# CREATE INVENTORY FILE
# ============================================================================

cat > inventory.json << 'EOF'
{
  "gwas_summary_statistics": {
    "autism": [
      {
        "study": "Grove et al. 2019",
        "pmid": "31548717",
        "file": "autism/iPSYCH_PGC_ASD_Nov2017.gz",
        "cases": 18381,
        "controls": 27969,
        "ancestry": "European",
        "notes": "iPSYCH + PGC meta-analysis, 12 genome-wide significant loci"
      },
      {
        "study": "Matoba et al. 2020",
        "pmid": "32066951",
        "file": "autism/SPARK_ASD_2020.txt.gz",
        "cases": 11710,
        "controls": "multiple",
        "ancestry": "European",
        "notes": "SPARK cohort analysis"
      },
      {
        "study": "GWAS Catalog",
        "file": "autism/gwas_catalog_autism.tsv",
        "notes": "Curated associations from all autism GWAS studies"
      }
    ],
    "adhd": [
      {
        "study": "Demontis et al. 2019",
        "pmid": "30478444",
        "file": "adhd/gwas_catalog_adhd.tsv",
        "cases": 20183,
        "controls": 35191,
        "ancestry": "European",
        "notes": "iPSYCH + PGC meta-analysis, 12 genome-wide significant loci. Full sumstats require application."
      },
      {
        "study": "GWAS Catalog - Attention",
        "file": "adhd/gwas_catalog_attention.tsv",
        "notes": "Attention-related traits from GWAS Catalog"
      }
    ],
    "cross_disorder": [
      {
        "study": "Cross-Disorder Group 2019",
        "pmid": "31835030",
        "file": "shared/cross_disorder_2019.txt.gz",
        "notes": "Genomic relationships between 8 psychiatric disorders including ASD and ADHD"
      }
    ]
  },
  "citations": {
    "grove_2019": "Grove, J., et al. (2019). Identification of common genetic risk variants for autism spectrum disorder. Nature Genetics, 51(3), 431-444.",
    "demontis_2019": "Demontis, D., et al. (2019). Discovery of the first genome-wide significant risk loci for attention deficit/hyperactivity disorder. Nature Genetics, 51(1), 63-75.",
    "matoba_2020": "Matoba, N., et al. (2020). Common genetic risk variants identified in the SPARK cohort support DDHD2 as a candidate risk gene for autism. Translational Psychiatry, 10(1), 1-13.",
    "cross_disorder_2019": "Cross-Disorder Group. (2019). Genomic relationships, novel loci, and pleiotropic mechanisms across eight psychiatric disorders. Cell, 179(7), 1469-1482."
  }
}
EOF

# ============================================================================
# CREATE README
# ============================================================================

cat > README.md << 'EOF'
# GWAS Summary Statistics

**Download Date**: $(date +%Y-%m-%d)
**Source**: GWAS Catalog, iPSYCH/PGC, individual study repositories

## Dataset Description

This directory contains GWAS summary statistics for:
- **Autism Spectrum Disorder (ASD)**
- **Attention Deficit/Hyperactivity Disorder (ADHD)**
- **Cross-disorder psychiatric genetics**

## Available Studies

### Autism
1. **Grove et al. 2019** - iPSYCH + PGC ASD
   - 18,381 cases, 27,969 controls
   - 12 genome-wide significant loci
   - European ancestry

2. **Matoba et al. 2020** - SPARK ASD
   - 11,710 cases
   - SPARK cohort

3. **GWAS Catalog** - Curated autism associations

### ADHD
1. **Demontis et al. 2019** - iPSYCH + PGC ADHD
   - 20,183 cases, 35,191 controls
   - 12 genome-wide significant loci
   - European ancestry
   - **Note**: Full summary statistics require application to iPSYCH

2. **GWAS Catalog** - ADHD and attention traits

### Cross-Disorder
1. **Cross-Disorder Group 2019**
   - Genomic relationships across 8 psychiatric disorders
   - Includes ASD and ADHD

## File Formats

GWAS summary statistics typically include:
- **SNP/rsID**: Variant identifier
- **CHR**: Chromosome
- **BP/POS**: Base pair position
- **A1**: Effect allele
- **A2**: Other allele
- **BETA/OR**: Effect size (beta or odds ratio)
- **SE**: Standard error
- **P**: P-value
- **N**: Sample size
- **FRQ/MAF**: Allele frequency

Different studies may use different column names. Use format-specific parsers.

## Usage in Pipeline

### 1. Calculate Polygenic Risk Scores (PRS)

```bash
# Preprocess summary statistics
audhd-pipeline prs prepare \
    --sumstats gwas_sumstats/autism/iPSYCH_PGC_ASD_Nov2017.gz \
    --output data/processed/autism_prs_input.txt

# Calculate PRS using PRSice-2 or LDpred
audhd-pipeline prs calculate \
    --sumstats data/processed/autism_prs_input.txt \
    --genotypes data/processed/spark_genotypes.bed \
    --output results/autism_prs_scores.txt
```

### 2. Identify Risk Loci for Downstream Analysis

```python
from audhd_correlation.biological import load_gwas_hits, annotate_genes

# Load genome-wide significant variants
hits = load_gwas_hits('gwas_sumstats/autism/gwas_catalog_autism.tsv', p_threshold=5e-8)

# Map to nearest genes
gene_annotations = annotate_genes(hits)

# Use for pathway enrichment or network analysis
```

### 3. Genetic Correlation Analysis

```bash
# Calculate genetic correlation between ASD and ADHD
ldsc \
    --rg gwas_sumstats/autism/iPSYCH_PGC_ASD_Nov2017.gz,gwas_sumstats/adhd/gwas_catalog_adhd.tsv \
    --ref-ld-chr eur_w_ld_chr/ \
    --w-ld-chr eur_w_ld_chr/ \
    --out results/asd_adhd_genetic_correlation
```

## Citations

**Grove et al. 2019 (Autism)**
Grove, J., et al. (2019). Identification of common genetic risk variants for
autism spectrum disorder. Nature Genetics, 51(3), 431-444.

**Demontis et al. 2019 (ADHD)**
Demontis, D., et al. (2019). Discovery of the first genome-wide significant
risk loci for attention deficit/hyperactivity disorder. Nature Genetics, 51(1), 63-75.

**Matoba et al. 2020 (SPARK)**
Matoba, N., et al. (2020). Common genetic risk variants identified in the SPARK
cohort support DDHD2 as a candidate risk gene for autism. Translational Psychiatry, 10(1), 1-13.

**Cross-Disorder Group 2019**
Cross-Disorder Group. (2019). Genomic relationships, novel loci, and pleiotropic
mechanisms across eight psychiatric disorders. Cell, 179(7), 1469-1482.

## Data Access Notes

- ✅ **Open Access**: GWAS Catalog data, published summary statistics
- ⚠️ **Application Required**: Full iPSYCH summary statistics (contact iPSYCH consortium)
- 📧 **Contact**: For full summary statistics, contact study authors or consortia

## Quality Control Recommendations

1. **Check build**: Ensure genomic coordinates match your reference (GRCh37/hg19 or GRCh38/hg38)
2. **Allele alignment**: Verify effect allele matches your data
3. **Filter by INFO score**: Remove poorly imputed variants (INFO < 0.8)
4. **Remove MHC region**: Consider excluding chr6:25-35Mb due to complex LD
5. **Check sample overlap**: Account for overlapping samples between studies

## Directory Structure

```
gwas_sumstats/
├── autism/
│   ├── iPSYCH_PGC_ASD_Nov2017.gz          # Grove et al. 2019
│   ├── SPARK_ASD_2020.txt.gz              # Matoba et al. 2020
│   └── gwas_catalog_autism.tsv            # GWAS Catalog curated
├── adhd/
│   ├── gwas_catalog_adhd.tsv              # GWAS Catalog ADHD
│   └── gwas_catalog_attention.tsv         # Attention traits
├── shared/
│   ├── cross_disorder_2019.txt.gz         # Cross-disorder study
│   └── gwas_catalog_neurodevelopmental.tsv
├── inventory.json                          # Study metadata
└── README.md                               # This file
```
EOF

echo -e "\n${GREEN}Download complete!${NC}"
echo "Summary statistics location: ${OUTPUT_DIR}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review inventory.json for study details"
echo "  2. Check file formats and harmonize column names"
echo "  3. Calculate polygenic risk scores (PRS)"
echo "  4. Use for pathway prioritization and gene mapping"
echo ""
echo -e "${YELLOW}Note:${NC} Full iPSYCH summary statistics require application."
echo "Visit https://ipsych.dk/en/research/downloads/ for access."