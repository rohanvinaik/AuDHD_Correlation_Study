# UK Biobank Extraction Pipeline

**Purpose**: Extract ADHD/Autism cohorts with metabolomics data from UK Biobank

## Overview

UK Biobank is the only large-scale biobank with comprehensive NMR metabolomics (249 biomarkers from Nightingale platform, ~120,000 participants). This pipeline extracts:

- **ADHD cases**: ~5,000 (ICD-10: F90.x, self-report, medications)
- **Autism cases**: ~2,000 (ICD-10: F84.x, self-report)
- **Metabolomics subset**: ~120,000 with NMR data
- **Controls**: Matched, no psychiatric diagnosis

## Prerequisites

### 1. UK Biobank Access

**Required**:
- ✅ UK Biobank application approved (8-12 weeks)
- ✅ Data Use Agreement executed
- ✅ Application fee paid (£2,500-6,000 GBP)
- ✅ Data downloaded

**Apply at**: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access

### 2. Software Requirements

```bash
# Python packages
pip install pandas numpy pyyaml

# ukbconv (UK Biobank extraction tool)
# Download from: https://biobank.ndph.ox.ac.uk/ukb/refer.cgi?id=644
# Place in PATH or current directory
```

###3. Data Files

You need:
- `ukbXXXXX.enc_ukb` - Encrypted UK Biobank dataset
- `ukbXXXXX.key` - Decryption key
- `ukbconv` - Extraction utility

---

## Quick Start

### Step 1: Extract Fields

```bash
# Extract all ADHD/Autism relevant fields
python field_extractor.py \
    --ukb-file ukb12345.enc_ukb \
    --groups diagnosis,medications,metabolomics,genetics \
    --output data/ukb/

# Or extract specific groups separately
python field_extractor.py \
    --ukb-file ukb12345.enc_ukb \
    --groups metabolomics \
    --separate \
    --output data/ukb/
```

**Output**: `data/ukb/ukb_combined.csv`

### Step 2: Build Cohorts

```bash
# Build ADHD/Autism cohorts
python cohort_builder.py \
    --input data/ukb/ukb_combined.csv \
    --output data/ukb/ \
    --export-ids \
    --export-metadata

# Or filter for metabolomics subset only
python cohort_builder.py \
    --input data/ukb/ukb_combined.csv \
    --metabolomics-only \
    --output data/ukb/
```

**Output**:
- `adhd_only_ids.txt` - ADHD case IDs
- `autism_only_ids.txt` - Autism case IDs
- `adhd_autism_ids.txt` - Comorbid case IDs
- `controls_ids.txt` - Control IDs
- `*_metadata.csv` - Detailed cohort metadata

---

## Tools

### 1. `field_extractor.py`

**Purpose**: Extract specific UK Biobank fields using ukbconv wrapper

**Field Groups**:
- `diagnosis` - ICD-10 diagnoses (F90.x, F84.x)
- `medications` - ADHD medications
- `metabolomics` - NMR metabolomics (249 biomarkers)
- `genetics` - Genetic PCs, kinship
- `demographics` - Age, sex, ethnicity
- `mental_health` - PHQ-9, GAD-7, MHQ
- `sleep` - Duration, insomnia, chronotype
- `diet` - Dietary intake
- `environmental` - Air pollution, noise
- `family_history` - Parental illness
- `ses` - Income, education, employment

**Usage**:
```bash
# List available groups
python field_extractor.py --list-groups

# Extract specific groups
python field_extractor.py \
    --ukb-file ukb12345.enc_ukb \
    --groups diagnosis,metabolomics \
    --output data/ukb/

# Extract custom fields
python field_extractor.py \
    --ukb-file ukb12345.enc_ukb \
    --fields 20544,41270,23400-23649 \
    --output data/ukb/

# Generate documentation
python field_extractor.py \
    --ukb-file ukb12345.enc_ukb \
    --groups all \
    --docs
```

**Metabolomics Fields** (Nightingale NMR):
- Fields 23400-23649 (249 biomarkers)
- Categories: Lipoproteins, fatty acids, amino acids, glycolysis, ketones, inflammation
- ~120,000 participants with data

### 2. `cohort_builder.py`

**Purpose**: Build ADHD/Autism cohorts from extracted data

**Case Definitions**:

**ADHD**:
- ICD-10: F90.0, F90.1, F90.8, F90.9
- Self-report: Code 1117
- Medications: Methylphenidate, Atomoxetine, Dexamfetamine, Lisdexamfetamine

**Autism**:
- ICD-10: F84.0, F84.1, F84.5, F84.8, F84.9
- Self-report: Code 1111

**Controls**:
- No psychiatric diagnosis (any F code)
- Matched for age/sex (optional)
- Can require metabolomics data

**Usage**:
```bash
# Build all cohorts
python cohort_builder.py \
    --input data/ukb/ukb_combined.csv \
    --output data/ukb/ \
    --n-controls 10000 \
    --export-ids \
    --export-metadata

# Metabolomics subset only
python cohort_builder.py \
    --input data/ukb/ukb_combined.csv \
    --metabolomics-only \
    --output data/ukb/
```

**Output Structure**:
```
data/ukb/
├── adhd_only_ids.txt           # ADHD-only case IDs
├── autism_only_ids.txt         # Autism-only case IDs
├── adhd_autism_ids.txt         # Comorbid case IDs
├── controls_ids.txt            # Control IDs
├── adhd_only_metadata.csv      # Detailed ADHD metadata
├── autism_only_metadata.csv    # Detailed autism metadata
├── adhd_autism_metadata.csv    # Comorbid metadata
└── controls_metadata.csv       # Control metadata
```

---

## Configuration

### `configs/ukb_extraction_config.yaml`

Complete configuration file specifying:
- Field groups to extract
- Case definitions (ICD-10 codes, self-report codes, medications)
- Control selection criteria
- Metabolomics QC parameters
- Genetic ancestry filters
- Preprocessing methods

**Usage**:
```python
import yaml
from pathlib import Path

# Load config
with open('configs/ukb_extraction_config.yaml') as f:
    config = yaml.safe_load(f)

# Use in extraction
field_groups = config['extraction']['field_groups']
```

---

## Expected Cohort Sizes

| Cohort | Expected N | With Metabolomics |
|--------|-----------|-------------------|
| **ADHD only** | ~4,000 | ~1,200 |
| **Autism only** | ~1,500 | ~400 |
| **ADHD+Autism** | ~500 | ~100 |
| **Controls** | 10,000 | ~3,000 |

**Note**: UK Biobank participants are ages 40-69, so these are adult-diagnosed cases.

---

## Metabolomics Details

### Nightingale NMR Platform

**249 biomarkers** across 6 categories:

1. **Lipoproteins** (~80 measures)
   - VLDL, LDL, IDL, HDL subclasses
   - Particle sizes and concentrations
   - Cholesterol, triglycerides, phospholipids

2. **Fatty Acids** (~26 measures)
   - Saturated, monounsaturated, polyunsaturated
   - Omega-3, Omega-6
   - DHA, linoleic acid

3. **Amino Acids** (~9 measures)
   - Branched-chain: Isoleucine, leucine, valine
   - Aromatic: Phenylalanine, tyrosine
   - Others: Alanine, glutamine, glycine, histidine

4. **Glycolysis** (~5 measures)
   - Glucose, lactate, pyruvate, citrate

5. **Ketone Bodies** (~2 measures)
   - Acetoacetate, 3-hydroxybutyrate

6. **Inflammation** (~2 measures)
   - GlycA, C-reactive protein

**QC Recommendations**:
- Remove outliers (>5 SD)
- Log-transform (many metabolites are log-normal)
- Check for batch effects (genotype_batch field 22000)

### Availability

- **Total participants**: ~500,000
- **With metabolomics**: ~120,000 (24%)
- **Release**: Added in UK Biobank data release (check your application)

---

## Complete Workflow

### 1. Prepare Environment

```bash
# Create directories
mkdir -p data/ukb

# Check ukbconv
ukbconv --help

# Install Python packages
pip install pandas numpy pyyaml
```

### 2. Extract All Relevant Fields

```bash
# Extract comprehensive field set
python field_extractor.py \
    --ukb-file ukb12345.enc_ukb \
    --groups diagnosis,medications,metabolomics,genetics,demographics,mental_health,sleep,environmental,family_history,ses \
    --output data/ukb/

# This creates: data/ukb/ukb_combined.csv (~500,000 rows × ~300 columns)
```

**Time**: 1-2 hours depending on dataset size

### 3. Build Cohorts

```bash
# Build all cohorts with metabolomics filtering
python cohort_builder.py \
    --input data/ukb/ukb_combined.csv \
    --output data/ukb/ \
    --n-controls 10000 \
    --metabolomics-only \
    --export-ids \
    --export-metadata
```

**Output**:
- Cohort ID lists (for subsetting)
- Metadata CSVs (for analysis)

### 4. Subset Data for Analysis

```python
import pandas as pd

# Load full data
data = pd.read_csv('data/ukb/ukb_combined.csv')

# Load cohort IDs
adhd_ids = pd.read_csv('data/ukb/adhd_only_ids.txt', header=None)[0]

# Subset
adhd_data = data[data['eid'].isin(adhd_ids)]

# Save
adhd_data.to_csv('data/ukb/adhd_cohort.csv', index=False)
```

### 5. Run Pipeline

```bash
# Use UK Biobank-specific config
audhd-pipeline run --config configs/datasets/ukb.yaml
```

---

## Troubleshooting

### Issue 1: ukbconv not found

**Solution**:
```bash
# Download ukbconv
wget https://biobank.ctsu.ox.ac.uk/ukb/util/ukbconv

# Make executable
chmod +x ukbconv

# Add to PATH or use full path
export PATH=$PATH:$(pwd)
```

### Issue 2: Key file not found

**Error**: `Cannot find key file`

**Solution**:
```bash
# Ensure .key file is in same directory as .enc_ukb
ls ukb12345.*
# Should show: ukb12345.enc_ukb and ukb12345.key

# Or specify key explicitly (ukbconv option)
```

### Issue 3: Extraction timeout

**Error**: Extraction takes > 1 hour

**Solution**:
```bash
# Extract groups separately
python field_extractor.py --ukb-file ukb12345.enc_ukb --groups diagnosis --separate
python field_extractor.py --ukb-file ukb12345.enc_ukb --groups metabolomics --separate

# Then merge
python -c "
import pandas as pd
df1 = pd.read_csv('data/ukb/ukb_diagnosis.csv')
df2 = pd.read_csv('data/ukb/ukb_metabolomics.csv')
merged = df1.merge(df2, on='eid')
merged.to_csv('data/ukb/ukb_combined.csv', index=False)
"
```

### Issue 4: Few metabolomics participants

**Issue**: < 120,000 participants with metabolomics

**Solution**:
- Check if metabolomics fields were included in your UK Biobank application
- Some applications pre-date metabolomics availability
- Contact UK Biobank to add metabolomics fields to your dataset

### Issue 5: No ADHD/Autism cases found

**Issue**: Cohort builder finds 0 cases

**Solution**:
```bash
# Check if diagnosis fields were extracted
python -c "
import pandas as pd
data = pd.read_csv('data/ukb/ukb_combined.csv')
print('ICD-10 fields:', [col for col in data.columns if col.startswith('41270')])
print('Self-report fields:', [col for col in data.columns if col.startswith('20002')])
"

# If missing, re-extract with diagnosis group
python field_extractor.py --ukb-file ukb12345.enc_ukb --groups diagnosis
```

---

## Field Reference

### Key Field IDs

| Category | Field ID | Description |
|----------|----------|-------------|
| **Demographics** |
| | 31 | Sex |
| | 21003 | Age at assessment |
| | 21000 | Ethnic background |
| **Diagnosis** |
| | 20544 | Mental health problems |
| | 41270 | ICD-10 main diagnoses |
| | 41271 | ICD-10 secondary diagnoses |
| **Medications** |
| | 20003 | Treatment/medication code |
| | 20004 | Date of medication |
| **Metabolomics** |
| | 23400-23649 | NMR metabolomics (249 biomarkers) |
| **Genetics** |
| | 22001 | Genetic sex |
| | 22006 | Genetic ethnic grouping |
| | 22009 | Genetic PCs (1-40) |
| | 22020 | Genetic kinship |
| **Mental Health** |
| | 20400-20408 | GAD-7 anxiety items |
| | 20510-20514 | PHQ-9 depression items |
| **Sleep** |
| | 1160 | Sleep duration |
| | 1200 | Insomnia |
| | 1180 | Chronotype |

**Complete field list**: Use `python field_extractor.py --list-groups`

---

## Integration with Pipeline

### Use Extracted Data

```bash
# 1. Extract UK Biobank data (this pipeline)
python field_extractor.py --ukb-file ukb12345.enc_ukb --groups all

# 2. Build cohorts
python cohort_builder.py --input data/ukb/ukb_combined.csv --metabolomics-only

# 3. Run main analysis pipeline
audhd-pipeline run --config configs/datasets/ukb.yaml
```

### Configuration

Update `configs/datasets/ukb.yaml`:
```yaml
dataset:
  name: "UK_Biobank"
  data_root: "data/ukb"

  paths:
    phenotypes: "${data_root}/ukb_combined.csv"
    cohort_ids: "${data_root}/adhd_only_ids.txt"
    metabolomics: "${data_root}/ukb_metabolomics.csv"
```

---

## Citation

If you use UK Biobank data, cite:

```bibtex
@article{sudlow2015uk,
  title={UK Biobank: An Open Access Resource for Identifying the Causes of a Wide Range of Complex Diseases of Middle and Old Age},
  author={Sudlow, Cathie and others},
  journal={PLoS Medicine},
  volume={12},
  number={3},
  pages={e1001779},
  year={2015}
}
```

**Metabolomics**:
```bibtex
@article{julkunen2023metabolic,
  title={Metabolic biomarker profiling for identification of susceptibility to severe pandemic influenza},
  author={Julkunen, Heli and others},
  journal={Nature Communications},
  volume={14},
  pages={1}\,
  year={2023}
}
```

---

## Support

**UK Biobank**:
- Email: access@ukbiobank.ac.uk
- Portal: https://bbams.ndph.ox.ac.uk/ams/

**Pipeline Issues**:
- GitHub: github.com/rohanvinaik/AuDHD_Correlation_Study/issues

**Related Documentation**:
- Dataset catalog: `data/catalogs/dataset_inventory.json`
- Data sources guide: `docs/data_sources.md`
- UK Biobank config: `configs/datasets/ukb.yaml`

---

**Last Updated**: 2025-01-30