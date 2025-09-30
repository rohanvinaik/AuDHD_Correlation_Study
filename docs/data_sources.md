# Data Source Discovery and Access Guide

**Version**: 1.0.0
**Last Updated**: 2025-01-30
**Maintainer**: AuDHD Correlation Study Team

## Overview

This guide provides comprehensive information on discovering, accessing, and using autism/ADHD datasets for multi-omics research. We have cataloged **28 major data sources** totaling over **850,000 participants** with various omics layers.

## Quick Links

- **Dataset Inventory**: `data/catalogs/dataset_inventory.json` - Complete structured catalog
- **Access Tracker**: `data/catalogs/access_tracker.md` - Track your application progress
- **Download Scripts**: `scripts/data_acquisition/` - Automated download for open-access datasets
- **Dataset Configs**: `configs/datasets/` - Ready-to-use pipeline configurations

## Dataset Discovery Strategy

### Search Approach

We searched the following repository types:

1. **Disease-Specific Repositories**
   - SFARI Base (SPARK, SSC, AGRE, Simons Searchlight)
   - NDAR (National Database for Autism Research)

2. **Population Cohorts**
   - ABCD Study (adolescent development)
   - UK Biobank (adult health)
   - IMAGEN (European adolescents)

3. **Genetic Repositories**
   - dbGaP (NIH genetic datasets)
   - EGA (European Genome-phenome Archive)
   - GWAS Catalog (summary statistics)

4. **Omics-Specific Repositories**
   - GEO (gene expression)
   - MetaboLights (metabolomics)
   - Metabolomics Workbench
   - PsychENCODE (brain multi-omics)

5. **Neuroimaging Repositories**
   - OpenNeuro
   - ABIDE (Autism Brain Imaging Data Exchange)
   - ADHD-200
   - LONI IDA

### Inclusion Criteria

Datasets were included if they contain:
- **Primary**: Autism and/or ADHD diagnoses or traits
- **Secondary**: Multi-omics data (2+ modalities preferred)
- **Minimum**: 50+ samples with relevant phenotypes

## Priority Datasets for Multi-Omics Analysis

### Tier 1: Essential Multi-Omics Datasets

| Dataset | Sample Size | Omics Layers | Access | Priority |
|---------|-------------|--------------|--------|----------|
| **SPARK** | 100,000+ | Genomic + Clinical | Application | ⭐⭐⭐⭐⭐ |
| **SSC** | 2,600 families | Genomic + Transcriptomic + Clinical + Imaging | Application | ⭐⭐⭐⭐⭐ |
| **ABCD** | 11,880 | Genomic + Imaging + Clinical | Application | ⭐⭐⭐⭐⭐ |
| **UK Biobank** | 500,000 | Genomic + Metabolomic + Clinical | Application (£) | ⭐⭐⭐⭐⭐ |

**Why These Are Essential**:
- **SPARK**: Largest autism genetics cohort
- **SSC**: High-quality multi-omics (WGS, RNA-seq, MRI)
- **ABCD**: Best ADHD data with neuroimaging
- **UK Biobank**: Only large-scale metabolomics dataset

### Tier 2: Specialized Datasets

| Dataset | Use Case | Sample Size | Access |
|---------|----------|-------------|--------|
| **MSSNG** | Deep autism genomics (WGS) | 11,000 | Application + GCP costs |
| **NDAR** | Federated autism repository | 200,000+ | Registration |
| **dbGaP ADHD** | ADHD GWAS data | 55,000 | Application |
| **PsychENCODE** | Brain expression + epigenomics | 2,000 samples | Open/Controlled |

### Tier 3: Open-Access Datasets

| Dataset | Type | Sample Size | Use Case |
|---------|------|-------------|----------|
| **GWAS Catalog** | Summary statistics | 100,000+ | Polygenic scores, loci mapping |
| **ABIDE** | Neuroimaging | 2,145 | Autism brain signatures |
| **ADHD-200** | Neuroimaging | 973 | ADHD brain signatures |
| **GEO** | Gene expression | 5,000+ | Transcriptomics, meta-analysis |

## Data Access Overview

### Access Types

1. **Open Access** (No registration required)
   - GWAS Catalog summary statistics
   - GEO gene expression studies
   - MetaboLights/Metabolomics Workbench
   - ABIDE, ADHD-200 neuroimaging
   - **Download**: Use scripts in `scripts/data_acquisition/`

2. **Registration Required** (Simple registration)
   - NDAR (some studies)
   - PhenomeCentral
   - **Timeline**: Immediate - 1 week

3. **Application Required** (Full proposal + DUA)
   - SPARK, SSC, AGRE (via SFARI Base)
   - ABCD Study (via NDA)
   - MSSNG
   - dbGaP studies
   - EGA studies
   - **Timeline**: 2-12 weeks

4. **Fee-Based Access**
   - UK Biobank: £2,500-6,000 GBP
   - MSSNG: Google Cloud Platform compute costs
   - **Budget**: Plan ahead for grant proposals

### Application Process

**General Steps**:
1. ✅ Obtain institutional IRB approval
2. ✅ Identify institutional signing official
3. ✅ Draft research proposal
4. ✅ Submit application to Data Access Committee (DAC)
5. ✅ Execute Data Use Agreement (DUA)
6. ✅ Wait for approval (2-12 weeks)
7. ✅ Download and verify data

**Required Documents** (typical):
- Research proposal (2-15 pages, varies by dataset)
- CV/biosketch
- IRB approval letter
- Institutional certification
- Data Use Agreement (DUA) - signed by authorized official
- Data security plan

**Tips for Success**:
- Start applications early (can take 3 months)
- Be specific about intended analyses
- Describe data security measures
- Highlight potential impact
- Request only necessary data fields (speeds approval)

## Quick Start Guides

### For Open-Access Data

```bash
# 1. Download GWAS summary statistics
cd scripts/data_acquisition/
chmod +x download_gwas_sumstats.sh
./download_gwas_sumstats.sh

# 2. Download ABIDE neuroimaging
chmod +x download_abide.sh
./download_abide.sh

# 3. Download GEO expression studies
Rscript download_geo_studies.R
```

### For Application-Required Data

```bash
# 1. Review access tracker
open data/catalogs/access_tracker.md

# 2. Use dataset-specific configuration
cp configs/datasets/spark.yaml configs/my_spark_analysis.yaml
# Edit paths after data download

# 3. Track application progress in access_tracker.md
```

## Multi-Omics Integration Recommendations

### Ideal Combination for This Study

**Primary Datasets** (Apply ASAP):
1. **SPARK** - Autism genomics + clinical (n=100,000)
2. **ABCD** - ADHD genomics + clinical + imaging (n=11,880)
3. **UK Biobank** - Metabolomics + genomics + clinical (n=500,000)

**Rationale**:
- SPARK: Largest autism cohort with deep phenotyping
- ABCD: Best ADHD representation with neuroimaging
- UKB: Only large-scale metabolomics dataset
- Combined: Genomics + metabolomics + clinical + imaging

**Alternative if Budget Limited** (No UK Biobank):
1. **SPARK** - Autism genomics + clinical
2. **ABCD** - ADHD genomics + clinical + imaging
3. **GEO** - Add transcriptomics from open-access studies
4. **GWAS Catalog** - Polygenic risk scores

### Data Harmonization Strategy

When integrating multiple datasets:

1. **Phenotype Harmonization**
   ```yaml
   # Use configs/datasets/{dataset}.yaml mappings
   # Example: ADHD diagnosis
   SPARK: "adhd_diagnosis"
   ABCD: "cbcl_adhd_combined"
   UKB: "adhd_any"
   ```

2. **Genomic Data Harmonization**
   - Ensure same reference build (GRCh37 or GRCh38)
   - Impute to common reference panel (TOPMed or 1000G)
   - Use standard QC pipeline (see `src/audhd_correlation/data/vcf_processing.py`)

3. **Batch Effect Correction**
   - Use ComBat or similar for each modality
   - Include dataset as covariate in analyses

## Dataset-Specific Considerations

### SPARK
- **Best for**: Autism genetics
- **Strength**: Largest autism cohort
- **Limitation**: Limited omics beyond genomics
- **Note**: Family structure (use probands or account for relatedness)

### SSC
- **Best for**: Rare variants, de novo mutations
- **Strength**: Multi-omics (WGS, RNA-seq, MRI)
- **Limitation**: Smaller sample size (n=2,600)
- **Note**: Simplex families (one affected child)

### ABCD
- **Best for**: ADHD, longitudinal analysis
- **Strength**: Rich neuroimaging, longitudinal
- **Limitation**: Limited autism cases
- **Note**: Ages 9-10 at baseline

### UK Biobank
- **Best for**: Metabolomics, large-scale genetics
- **Strength**: 249 NMR metabolomics biomarkers
- **Limitation**: Adult-only (ages 40-69), limited ADHD/autism cases
- **Cost**: £2,500-6,000 GBP application fee

### GWAS Catalog
- **Best for**: Polygenic risk scores, loci prioritization
- **Strength**: Open access, well-curated
- **Limitation**: Summary statistics only (no individual-level data)

## Data Storage and Security

### Storage Requirements

Estimate disk space needed:

| Dataset | Genomics | Imaging | Clinical | Total |
|---------|----------|---------|----------|-------|
| SPARK | 500 GB | - | 1 GB | 500 GB |
| ABCD | 300 GB | 2 TB | 5 GB | 2.3 TB |
| UKB | 10 TB | 1 TB | 10 GB | 11 TB |
| GEO | - | - | 50 GB | 50 GB |

**Recommendations**:
- **Minimum**: 1 TB fast SSD for analysis
- **Archive**: 10+ TB HDD for raw data
- **Backup**: Cloud backup for irreplaceable data

### Security Requirements

All controlled-access datasets require:
- ✅ Encrypted storage
- ✅ Access controls (who can access)
- ✅ Audit logging
- ✅ Secure data destruction plan
- ✅ No cloud storage without explicit approval

**Example Security Plan**:
```bash
# Encrypted storage
mount -t ecryptfs /data/secure /data/secure

# Access controls
chmod 700 /data/secure
chown $USER:$GROUP /data/secure

# Audit logging
ausearch -f /data/secure
```

## Data Analysis Workflow

### Step 1: Data Acquisition

```bash
# Start with open-access data while applications pending
./scripts/data_acquisition/download_gwas_sumstats.sh
./scripts/data_acquisition/download_abide.sh
Rscript scripts/data_acquisition/download_geo_studies.R

# Track application-based datasets
open data/catalogs/access_tracker.md
```

### Step 2: Configuration

```bash
# Copy dataset-specific config
cp configs/datasets/spark.yaml configs/my_analysis.yaml

# Update paths after data download
vim configs/my_analysis.yaml
```

### Step 3: Preprocessing

```bash
# QC and harmonization
audhd-pipeline preprocess --config configs/my_analysis.yaml
```

### Step 4: Integration

```bash
# Multi-omics integration
audhd-pipeline integrate --config configs/my_analysis.yaml
```

### Step 5: Clustering

```bash
# Subtype discovery
audhd-pipeline cluster --config configs/my_analysis.yaml
```

### Step 6: Validation

```bash
# Stability analysis
audhd-pipeline validate --config configs/my_analysis.yaml
```

### Step 7: Interpretation

```bash
# Biological interpretation
audhd-pipeline interpret --config configs/my_analysis.yaml
```

## Common Issues and Solutions

### Issue 1: Application Rejected
**Solution**: Revise and resubmit with clarifications
- Be more specific about analyses
- Add more security details
- Reduce data scope if too broad

### Issue 2: Data Format Incompatibility
**Solution**: Use format converters
```bash
# VCF to PLINK
plink2 --vcf data.vcf.gz --make-bed --out data

# BGEN to VCF
bgenix -g data.bgen -vcf -o data.vcf
```

### Issue 3: Reference Build Mismatch
**Solution**: Liftover to common build
```bash
# GRCh37 to GRCh38
CrossMap.py vcf hg19ToHg38.chain.gz input.vcf.gz hg38.fa output.vcf
```

### Issue 4: Missing Phenotypes
**Solution**:
- Check data dictionary for alternative variable names
- Contact data provider support
- Use proxy variables if necessary

## Resources

### Key Contacts

- **SFARI Base**: SFARIBase@simonsfoundation.org
- **NDA**: ndahelp@mail.nih.gov
- **UK Biobank**: access@ukbiobank.ac.uk
- **dbGaP**: dbgap-help@ncbi.nlm.nih.gov

### Documentation

- **Dataset Inventory**: `data/catalogs/dataset_inventory.json`
- **Access Tracker**: `data/catalogs/access_tracker.md`
- **Pipeline Docs**: `docs/user_guide/`
- **Dataset Configs**: `configs/datasets/`

### External Resources

- **SFARI Base**: https://base.sfari.org
- **NDA**: https://nda.nih.gov
- **UK Biobank**: https://www.ukbiobank.ac.uk
- **dbGaP**: https://www.ncbi.nlm.nih.gov/gap
- **GWAS Catalog**: https://www.ebi.ac.uk/gwas

## Citation

If you use these catalogs or scripts, please cite:

```bibtex
@software{audhd_data_catalog,
  author = {AuDHD Correlation Study Team},
  title = {Comprehensive Data Source Catalog for Autism/ADHD Multi-Omics Research},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/rohanvinaik/AuDHD_Correlation_Study}
}
```

## Updates and Maintenance

This catalog is maintained as datasets are updated and new datasets become available.

**To suggest additions**:
1. Open an issue at github.com/rohanvinaik/AuDHD_Correlation_Study
2. Include: dataset name, URL, access type, sample size, omics layers

**Update schedule**: Quarterly (January, April, July, October)

**Last catalog update**: 2025-01-30

---

## Quick Reference Card

### Open-Access Downloads (No Application)
```bash
# GWAS summary statistics
./scripts/data_acquisition/download_gwas_sumstats.sh

# ABIDE neuroimaging
./scripts/data_acquisition/download_abide.sh

# GEO expression studies
Rscript scripts/data_acquisition/download_geo_studies.R
```

### Priority Applications (Start ASAP)
1. ⭐ **SPARK**: sparkforautism.org (4-8 weeks)
2. ⭐ **ABCD**: nda.nih.gov (2-6 weeks)
3. ⭐ **UK Biobank**: ukbiobank.ac.uk (8-12 weeks, £2,500-6,000)

### Track Progress
```bash
# Update access tracker as you apply
open data/catalogs/access_tracker.md
```

### After Data Download
```bash
# Use dataset-specific config
audhd-pipeline run --config configs/datasets/spark.yaml
```