# ADHD/Autism Microbiome Data Acquisition Tools

Comprehensive pipeline for discovering, accessing, and harmonizing gut microbiome data from multiple public repositories for ADHD and autism research.

## Overview

The gut-brain axis plays a crucial role in neurodevelopment and behavior. These tools enable systematic discovery and integration of microbiome data to investigate microbial signatures associated with ADHD and autism spectrum disorder.

### Key Findings from Literature

**Consistent patterns across studies:**
- **Reduced diversity**: Both ADHD and ASD show decreased alpha diversity (Shannon, Simpson)
- **SCFA producers depleted**: Faecalibacterium, Roseburia, Coprococcus consistently reduced
- **Butyrate deficiency**: 70-85% of studies report reduced butyrate production
- **Pro-inflammatory taxa enriched**: Clostridium, Sutterella, Desulfovibrio often increased in ASD
- **GI comorbidity**: Microbiome alterations more pronounced with gastrointestinal symptoms

**Therapeutic potential:**
- Probiotic interventions (Lactobacillus, Bifidobacterium) show modest behavioral improvements
- Microbiota Transfer Therapy demonstrates sustained clinical benefits in ASD
- Dietary fiber increases SCFA production and microbial diversity

## Data Sources

### 1. NCBI Sequence Read Archive (SRA)
- **Content**: 16S rRNA amplicon and shotgun metagenomic sequencing
- **Coverage**: Thousands of gut microbiome studies
- **Access**: Public via FTP/API
- **Tool**: `sra_searcher.py`

### 2. Qiita
- **Content**: Processed microbiome data with standardized analysis
- **Coverage**: 300,000+ samples across 1,000+ studies
- **Access**: Public API (registration required for some features)
- **Tool**: `qiita_api_client.py`

### 3. MGnify (formerly EBI Metagenomics)
- **Content**: Processed metagenomic assemblies and functional annotations
- **Coverage**: Large European cohorts
- **Access**: Public via REST API
- **Tool**: Manual query via MGnify portal

### 4. curatedMetagenomicData
- **Content**: Curated, standardized microbiome datasets
- **Coverage**: Published studies with harmonized metadata
- **Access**: Bioconductor R package
- **Integration**: Export to CSV for Python analysis

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install biopython pandas numpy requests fuzzywuzzy python-Levenshtein

# For Qiita BIOM files (optional)
pip install biom-format

# For metadata parsing
pip install lxml beautifulsoup4
```

### NCBI Setup

For SRA searches, configure your email for NCBI E-utilities:

```python
# In sra_searcher.py, set:
Entrez.email = "your.email@institution.edu"

# Or use command-line:
python sra_searcher.py --email your.email@institution.edu --search adhd
```

### Qiita Setup (Optional)

For private Qiita studies, register and obtain an API token:

1. Register at https://qiita.ucsd.edu
2. Navigate to user settings → API tokens
3. Generate new token
4. Use with `--api-token` parameter

## Usage

### 1. Search NCBI SRA

```bash
# Search for ADHD microbiome studies
python scripts/microbiome/sra_searcher.py \
    --search adhd \
    --email your.email@institution.edu \
    --output data/microbiome/sra/

# Search for autism studies
python scripts/microbiome/sra_searcher.py \
    --search autism \
    --output data/microbiome/sra/

# Search both conditions
python scripts/microbiome/sra_searcher.py \
    --search both \
    --min-relevance 10.0 \
    --max-results 200 \
    --output data/microbiome/sra/

# Get details for specific BioProject
python scripts/microbiome/sra_searcher.py \
    --bioproject PRJNA392180 \
    --output data/microbiome/sra/

# Download RunInfo metadata
python scripts/microbiome/sra_searcher.py \
    --download-metadata PRJNA392180 \
    --output data/microbiome/sra/
```

**Output files:**
- `sra_study_catalog.csv`: Filtered studies with relevance scores
- `sra_studies_detailed.json`: Complete metadata with matched terms
- `{BIOPROJECT}_RunInfo.csv`: Sample-level metadata for downloads

### 2. Query Qiita

```bash
# List known ADHD/autism studies
python scripts/microbiome/qiita_api_client.py \
    --known-studies \
    --output data/microbiome/qiita/

# Search for studies with custom terms
python scripts/microbiome/qiita_api_client.py \
    --search \
    --terms "autism" "microbiome" "children" \
    --min-samples 50 \
    --output data/microbiome/qiita/

# Get study details
python scripts/microbiome/qiita_api_client.py \
    --study 10317 \
    --output data/microbiome/qiita/

# Download sample metadata
python scripts/microbiome/qiita_api_client.py \
    --study 10317 \
    --get-samples \
    --output data/microbiome/qiita/

# Filter by sample type
python scripts/microbiome/qiita_api_client.py \
    --known-studies \
    --sample-type Stool \
    --output data/microbiome/qiita/
```

**Output files:**
- `qiita_known_studies.csv`: Curated ADHD/autism studies
- `qiita_search_results.csv`: Search results with relevance
- `study_{ID}_samples.csv`: Sample metadata for analysis

### 3. Harmonize Metadata

```bash
# Harmonize single study
python scripts/microbiome/metadata_harmonizer.py \
    --study data/microbiome/raw/study1_metadata.csv \
    --output data/microbiome/harmonized/

# Harmonize multiple studies from directory
python scripts/microbiome/metadata_harmonizer.py \
    --input data/microbiome/raw/ \
    --output data/microbiome/harmonized/

# Apply quality filters
python scripts/microbiome/metadata_harmonizer.py \
    --input data/microbiome/raw/ \
    --min-reads 10000 \
    --output data/microbiome/harmonized/

# Merge harmonized studies
python scripts/microbiome/metadata_harmonizer.py \
    --merge data/microbiome/harmonized/*_harmonized.csv \
    --output data/microbiome/integrated/
```

**Harmonization features:**
- Standardize variable names (age, sex, diagnosis, sample_type)
- Map categorical variables (Male/Female → male/female)
- Extract ADHD/autism diagnoses from free text
- Parse medication data (stimulants, SSRIs, antibiotics)
- Convert units (age months → years, BMI)
- Add quality control flags (low reads, missing metadata)

**Output files:**
- `{study}_harmonized.csv`: Standardized metadata
- `{study}_stats.json`: Summary statistics
- `all_studies_merged.csv`: Integrated dataset
- `harmonization_log.json`: Record of transformations

## Workflow

### Complete Analysis Pipeline

```bash
# 1. Search SRA for relevant studies
python scripts/microbiome/sra_searcher.py \
    --search both \
    --email your.email@institution.edu \
    --min-relevance 10.0 \
    --output data/microbiome/sra/

# 2. Download metadata for top studies
# Review sra_study_catalog.csv and select studies
for bioproject in PRJNA392180 PRJNA290380 PRJNA284355; do
    python scripts/microbiome/sra_searcher.py \
        --download-metadata $bioproject \
        --output data/microbiome/raw/
done

# 3. Query Qiita for processed data
python scripts/microbiome/qiita_api_client.py \
    --known-studies \
    --output data/microbiome/qiita/

# Download sample metadata for relevant studies
python scripts/microbiome/qiita_api_client.py \
    --study 10317 \
    --get-samples \
    --output data/microbiome/raw/

# 4. Harmonize all metadata
python scripts/microbiome/metadata_harmonizer.py \
    --input data/microbiome/raw/ \
    --min-reads 5000 \
    --output data/microbiome/harmonized/

# 5. Create integrated dataset
python scripts/microbiome/metadata_harmonizer.py \
    --merge data/microbiome/harmonized/*_harmonized.csv \
    --output data/microbiome/integrated/

# 6. Review study_metadata.json for additional context
cat data/microbiome/study_metadata.json | jq '.data_access_summary'
```

## Data Quality and Filtering

### Quality Control Thresholds

**Sequencing depth:**
- Minimum reads: 5,000 (16S), 100,000 (metagenomics)
- Recommended: 10,000+ (16S), 1M+ (metagenomics)

**Metadata completeness:**
- Essential: sample_id, age, sex, diagnosis
- Recommended: BMI, diet, medications, GI symptoms
- Maximum missing: 50% for essential variables

**Contamination:**
- Maximum mitochondrial/chloroplast: 5%
- Check for common contaminants (Ralstonia, Delftia)

### Case/Control Definitions

**ADHD cases:**
- Validated diagnosis: DSM-5, ICD-10, KSADS
- Self-reported (American Gut Project): lower confidence
- Medication as proxy: check for stimulants/atomoxetine

**Autism cases:**
- Validated diagnosis: ADOS, ADI-R, DSM-5
- Specify severity: ADOS/CARS scores if available
- Note comorbid conditions (GI symptoms, anxiety)

**Controls:**
- Neurotypical/typically developing
- No psychiatric diagnoses
- Ideally age/sex-matched to cases

### Confounding Variables

**Critical confounders to collect:**
- Age (strongest predictor of microbiome composition)
- Sex (hormonal effects on microbiome)
- BMI (obesity alters microbiome)
- Diet (fiber, processed foods)
- Medications (antibiotics, SSRIs, stimulants)
- GI symptoms (diarrhea, constipation)
- Geographic location (environmental microbes)

**Statistical adjustment:**
```python
# Example covariate adjustment in Python
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Adjust for confounders using residualization
model = sm.OLS(microbiome_feature, sm.add_constant(confounders))
results = model.fit()
adjusted_feature = results.resid
```

## Key Microbiome Features

### 1. Alpha Diversity

**Metrics:**
- Shannon diversity: accounts for richness and evenness
- Simpson diversity: probability two random reads are different taxa
- Faith's PD: phylogenetic diversity
- Observed features: total number of taxa

**Expected pattern:**
- ADHD: Lower than controls
- ASD: Lower than controls, especially with GI symptoms
- Effect size: Cohen's d = 0.3-0.7

### 2. Beta Diversity

**Metrics:**
- Bray-Curtis: abundance-weighted dissimilarity
- UniFrac: phylogenetic distance (weighted/unweighted)
- Aitchison distance: compositional-aware (preferred for differential abundance)

**Expected pattern:**
- Cases separate from controls on PCoA/PCA
- GI symptoms drive stronger separation
- R² = 0.05-0.15 in PERMANOVA

### 3. Differential Abundance

**Key taxa consistently altered:**

**Depleted in ADHD/ASD:**
- Faecalibacterium prausnitzii (butyrate producer)
- Roseburia (butyrate producer)
- Coprococcus (butyrate producer)
- Prevotella (fiber fermenter)
- Bifidobacterium (GABA producer)

**Enriched in ASD:**
- Clostridium cluster XIVa
- Sutterella
- Desulfovibrio (H2S producer)
- Bacteroides (variable)

**Statistical methods:**
- DESeq2: negative binomial model
- ANCOM-BC: compositional bias correction
- LEfSe: LDA effect size
- MaAsLin2: multivariate associations

### 4. Functional Pathways

**Analyze using shotgun metagenomics:**
- HUMAnN3: functional profiling
- MetaCyc/KEGG pathways
- Enzyme abundances (EC numbers)

**Key pathways to examine:**
- Butyrate biosynthesis (decreased)
- GABA biosynthesis (decreased)
- Dopamine/serotonin degradation (increased in ADHD)
- Tryptophan metabolism (kynurenine pathway)
- Lipopolysaccharide biosynthesis (inflammation)

### 5. Short-Chain Fatty Acids (SCFAs)

**Primary SCFAs:**
- Butyrate: 10-20 mM (typical), reduced in 70-85% of ADHD/ASD studies
- Propionate: 15-25 mM, variable (some ASD studies show elevation)
- Acetate: 50-70 mM, generally unchanged

**Clinical significance:**
- Butyrate: anti-inflammatory, supports gut barrier, HDAC inhibition
- Propionate: animal models show behavioral effects at high doses
- Acetate: crosses blood-brain barrier, energy source

**Measurement:**
- GC-MS: gold standard
- NMR spectroscopy: quantitative
- Predict from metagenomes: BugBase, PICRUSt2

## Known Studies (Curated)

### High-Priority Studies

#### 1. PRJNA392180 - Kang et al. 2017
- **Design**: ASD with GI symptoms, pre/post FMT
- **N**: 18 ASD, 2 controls
- **Key finding**: FMT improves GI and ASD symptoms
- **Data**: 16S, metabolomics, clinical scores
- **Access**: Public (SRA)

#### 2. PRJNA290380 - Aarts et al. 2017
- **Design**: ADHD, ASD, and controls
- **N**: 39 ADHD, 23 ASD, 51 controls
- **Key finding**: Reduced butyrate in ADHD
- **Data**: 16S, SCFA measurements, diet/medication
- **Access**: Public (SRA)

#### 3. PRJNA545885 - Wang et al. 2020
- **Design**: ADHD shotgun metagenomics
- **N**: 17 ADHD, 34 controls
- **Key finding**: Reduced butyrate biosynthesis genes
- **Data**: WGS metagenomics, functional annotations
- **Access**: Public (SRA)

#### 4. Qiita 10317 - American Gut Project
- **Design**: Citizen science, self-reported diagnoses
- **N**: 15,000+ samples (87 ADHD, 42 ASD)
- **Key finding**: Mental health associated with lower diversity
- **Data**: 16S V4, extensive metadata (diet, lifestyle)
- **Access**: Public (Qiita/EBI)

#### 5. PRJNA400072 - Kang et al. 2019
- **Design**: Long-term FMT follow-up (2 years)
- **N**: 18 ASD
- **Key finding**: Sustained clinical improvements, microbiome normalization
- **Data**: Shotgun metagenomics, metabolomics, SCFA
- **Access**: Public (SRA)

### Additional Studies

See `data/microbiome/study_metadata.json` for complete catalog of 28 curated studies with:
- Study design and sample sizes
- Age ranges and sex distributions
- Sequencing platforms and strategies
- Key findings and effect sizes
- SCFA and metabolomics availability
- Access information and URLs

## Integration with Other Data Types

### 1. Metabolomics Integration

Combine microbiome with metabolomics data:

```bash
# After running metabolomics tools
python scripts/integration/combine_microbiome_metabolomics.py \
    --microbiome data/microbiome/integrated/all_studies_merged.csv \
    --metabolomics data/metabolomics/metabolite_reference_ranges.csv \
    --output data/integrated/
```

**Key metabolite-microbe associations:**
- Butyrate ↔ Faecalibacterium, Roseburia
- GABA ↔ Lactobacillus, Bifidobacterium
- Tryptophan metabolites ↔ Clostridium
- 4-ethylphenylsulfate ↔ Clostridium

### 2. Genetics Integration

Link microbiome to host genetics:

```bash
# Correlate microbiome features with genetic risk scores
python scripts/integration/microbiome_genetics_correlation.py \
    --microbiome data/microbiome/integrated/all_studies_merged.csv \
    --genetics data/genetics/summary_stats_inventory.csv \
    --output data/integrated/
```

**Potential associations:**
- Host genetics → microbiome composition (heritability ~10%)
- Immune genes (HLA, NOD2) → microbial diversity
- FUT2 secretor status → Bifidobacterium abundance

### 3. Clinical Phenotypes

Correlate microbiome with clinical measures:

```bash
# Link to ABCD behavioral scores
python scripts/integration/microbiome_clinical_association.py \
    --microbiome data/microbiome/integrated/all_studies_merged.csv \
    --clinical data/abcd/phenotype_mapping.json \
    --output data/integrated/
```

**Clinical correlations to test:**
- ADHD symptom severity (KSADS, CBCL attention scores)
- ASD symptom severity (ADOS, SRS)
- GI symptom severity (custom questionnaires)
- Cognitive function (NIH Toolbox)

## Troubleshooting

### Issue: NCBI API rate limiting

```bash
# Error: "HTTP 429 Too Many Requests"
# Solution: Add delays between requests (already implemented)
# Fallback: Use NCBI EDirect tools

# Install EDirect
sh -c "$(curl -fsSL ftp://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"

# Search SRA
esearch -db sra -query "autism[Title] AND microbiome[All Fields]" | efetch -format xml
```

### Issue: Large metadata files

```bash
# Error: Memory issues with RunInfo CSV
# Solution: Process in chunks

import pandas as pd

# Read in chunks
for chunk in pd.read_csv('RunInfo.csv', chunksize=1000):
    process_chunk(chunk)
```

### Issue: Qiita API access

```bash
# Error: "401 Unauthorized"
# Solution: Check API token validity

curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://qiita.ucsd.edu/api/v1/study/10317
```

### Issue: Metadata inconsistencies

```bash
# Problem: Different variable names across studies
# Solution: Check harmonization log

cat data/microbiome/harmonized/harmonization_log.json | \
    jq '.[] | select(.action=="fuzzy_rename")'

# Manually review and adjust mappings in metadata_harmonizer.py
```

## Statistical Analysis Recommendations

### Power Analysis

```python
# Estimate sample size for microbiome differential abundance
from scipy.stats import power

# Typical effect sizes from literature
effect_size = 0.5  # Cohen's d for key taxa
alpha = 0.05
power = 0.8

# Calculate n per group
# For ADHD vs control: ~50 per group
# For ASD vs control: ~40 per group
# Adjust for covariates: +20%
```

### Batch Effect Correction

```python
# Correct for sequencing platform, study site
from combat.pycombat import pycombat

corrected_data = pycombat(
    data=microbiome_counts,
    batch=study_id,
    mod=pd.DataFrame({'diagnosis': diagnosis, 'age': age})
)
```

### Multiple Testing Correction

```python
from statsmodels.stats.multitest import multipletests

# FDR correction (Benjamini-Hochberg)
reject, pvals_corrected, _, _ = multipletests(
    pvals,
    alpha=0.05,
    method='fdr_bh'
)
```

## Data Sharing and Citation

### When Using These Tools

Please cite:
- This repository: AuDHD Correlation Study
- Original data sources (see study_metadata.json for PMIDs)
- Analysis tools (QIIME2, DESeq2, etc.)

### Contributing Data

If you have ADHD/autism microbiome data to contribute:
1. Ensure proper ethics approval and consent
2. De-identify all samples
3. Share to public repositories (SRA, Qiita)
4. Include comprehensive metadata
5. Contact repository maintainers to add to catalog

## References

### Key Papers

1. Kang et al. (2017) **Microbiota Transfer Therapy alters gut ecosystem and improves gastrointestinal and autism symptoms**. *Microbiome* 5:10. PMID: 28122648

2. Kang et al. (2019) **Long-term benefit of Microbiota Transfer Therapy on autism symptoms and gut microbiota**. *Scientific Reports* 9:5821. PMID: 30971783

3. Aarts et al. (2017) **Gut microbiome in ADHD and its relation to neural reward anticipation**. *PLOS ONE* 12(9):e0183509. PMID: 28902889

4. Wang et al. (2020) **Altered Gut Microbiota and Short Chain Fatty Acids in Chinese Children With Autism Spectrum Disorder**. *Scientific Reports* 10:18616. PMID: 33122764

5. Hsiao et al. (2013) **Microbiota modulate behavioral and physiological abnormalities associated with neurodevelopmental disorders**. *Cell* 155(7):1451-63. PMID: 24315484

6. McDonald et al. (2018) **American Gut: an Open Platform for Citizen Science Microbiome Research**. *mSystems* 3(3):e00031-18. PMID: 29795809

### Reviews

- Xu et al. (2019) **The Gut Microbiome in Psychiatry and Neurological Disorders**. *PMID: 30713328*
- Iglesias-Vázquez et al. (2020) **Composition of Gut Microbiota in Children with Autism Spectrum Disorder: A Systematic Review and Meta-Analysis**. *PMID: 32531972*
- Prehn-Kristensen et al. (2018) **Reduced microbiome alpha diversity in young patients with ADHD**. *PMID: 30206657*

### Databases

- NCBI SRA: https://www.ncbi.nlm.nih.gov/sra
- Qiita: https://qiita.ucsd.edu
- MGnify: https://www.ebi.ac.uk/metagenomics
- curatedMetagenomicData: https://waldronlab.io/curatedMetagenomicData

## Support

For questions or issues:
1. Check troubleshooting section above
2. Review study_metadata.json for known limitations
3. Open GitHub issue with detailed description
4. Contact repository maintainers

---

**Last updated**: 2025-09-30
**Version**: 1.0
**Maintained by**: AuDHD Correlation Study Team