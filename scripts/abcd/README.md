# ABCD Study Data Access Pipeline

**Purpose**: Automated data access, processing, and analysis pipeline for the Adolescent Brain Cognitive Development (ABCD) Study, focusing on ADHD/Autism research.

## Overview

The ABCD Study is the largest long-term study of brain development and child health in the United States:

- **Sample size**: ~11,800 children
- **Age range**: 9-10 years at baseline, now 9-17 years (through Year 4 follow-up)
- **Study duration**: 10 years (longitudinal)
- **Key data**: Comprehensive neuroimaging, clinical diagnoses (KSADS), behavioral assessments (CBCL), environmental exposures, family history
- **ADHD cases**: ~1,000 diagnosed (8-10% prevalence)
- **Autism cases**: ~250-350 diagnosed (2-3% prevalence)

### Unique Strengths for ADHD/Autism Research

1. **Gold-standard diagnoses**: KSADS-5 structured interviews (not just screening questionnaires)
2. **Comprehensive neuroimaging**:
   - Structural MRI (cortical thickness, volumes)
   - Resting-state fMRI (network connectivity)
   - Diffusion MRI (white matter microstructure)
   - Task-based fMRI
3. **Longitudinal design**: Track developmental trajectories from childhood through adolescence
4. **Environmental data**: Geocoded air quality, neighborhood characteristics
5. **Biospecimens**: Saliva (DNA, hormones) available for subset

### Limitations

- **Limited metabolomics**: Only hormone assays (DHEA, testosterone) for ~4,500 participants. No comprehensive metabolomics profiling like UK Biobank.
- **Recruitment age**: Misses early development (ages 0-8)
- **Lower autism prevalence** than specialized registries (SPARK, SSC)

**Recommendation**: Use ABCD for neuroimaging-clinical correlations and developmental trajectories. Combine with UK Biobank for metabolomics analyses.

---

## Prerequisites

### 1. NDA Data Access

**Required**:
- ✅ NDA account at https://nda.nih.gov/
- ✅ ABCD Data Use Certification (DUC) approved (2-4 weeks)
- ✅ Institutional authorization (signed by IRB or authorized official)

**Apply at**: https://nda.nih.gov/abcd/request-access

### 2. Software Requirements

```bash
# Python packages
pip install nda-tools pandas numpy scipy tqdm pyyaml

# NDA downloadcmd tool (installed automatically with nda-tools)
# Verify installation:
downloadcmd --help
```

### 3. Configure NDA Credentials

```bash
# Interactive configuration (one-time setup)
python nda_downloader.py --configure

# Or use downloadcmd directly
downloadcmd -c
```

---

## Quick Start

### Step 1: List Available ABCD Packages

```bash
# List all ADHD/Autism-relevant packages
python nda_downloader.py --list-packages

# List by category
python nda_downloader.py --list-packages --category clinical
python nda_downloader.py --list-packages --category neuroimaging
```

**Output**:
```
=== Available ABCD Packages ===

package_id       name                              category      adhd  autism  size_mb
abcd_cbcls01     Child Behavior Checklist         clinical      True  True    5
abcd_ksad01      KSADS Diagnoses                  clinical      True  True    15
abcd_medhy01     Medication History               clinical      True  True    3
abcd_betnet02    Brain Connectivity (rs-fMRI)     neuroimaging  True  True    200
abcd_smrip10201  Structural MRI (FreeSurfer)      neuroimaging  True  True    50
...

Total packages: 18
Total size: 382 MB
```

### Step 2: Download ADHD/Autism-Relevant Packages

```bash
# Download all ADHD/Autism-relevant packages
python nda_downloader.py \
    --adhd-autism-packages \
    --output data/abcd/

# Download specific packages
python nda_downloader.py \
    --packages abcd_cbcls01,abcd_ksad01,abcd_medhy01 \
    --output data/abcd/

# Download only ADHD-relevant packages
python nda_downloader.py \
    --adhd-only \
    --output data/abcd/

# Resume interrupted download
python nda_downloader.py --resume --output data/abcd/
```

**Time**: 5-30 minutes depending on package sizes and network speed

**Output structure**:
```
data/abcd/
├── abcd_cbcls01/
│   └── abcd_cbcls01.txt
├── abcd_ksad01/
│   └── abcd_ksad01.txt
├── abcd_medhy01/
│   └── abcd_medhy01.txt
├── .nda_download_progress.json
└── download_summary.csv
```

### Step 3: Process Downloaded Packages

```bash
# Process all packages (extract key variables, merge timepoints)
python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/ \
    --timepoint baseline_year_1_arm_1

# Extract only ADHD/autism cases
python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/ \
    --cases-only

# Generate imaging metrics summary
python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/ \
    --extract-imaging
```

**Output**:
```
data/abcd/processed/
├── abcd_cbcls01_processed.csv          # Processed CBCL data
├── abcd_ksad01_processed.csv           # Processed KSADS diagnoses
├── abcd_merged_baseline_year_1_arm_1.csv  # Merged baseline data
├── abcd_adhd_only.csv                  # ADHD-only cohort
├── abcd_autism_only.csv                # Autism-only cohort
├── abcd_comorbid.csv                   # Comorbid ADHD+Autism
└── imaging_metrics_extracted.csv       # Imaging summary
```

### Step 4: Extract Imaging Metrics

```bash
# Extract neuroimaging metrics
python extract_imaging_metrics.py \
    --input data/abcd/processed/abcd_merged_baseline_year_1_arm_1.csv \
    --output data/abcd/imaging_metrics_extracted.csv \
    --compare-groups \
    --network-summary
```

**Output**:
- `imaging_metrics_extracted.csv` - Subject-level imaging data
- `imaging_summary_statistics.csv` - Descriptive statistics
- `imaging_adhd_vs_controls.csv` - ADHD vs controls comparison
- `imaging_autism_vs_controls.csv` - Autism vs controls comparison

---

## Tools

### 1. `nda_downloader.py`

**Purpose**: Download ABCD data packages from NIMH Data Archive using NDA API

**Key Features**:
- Automated authentication and session management
- Resume interrupted downloads
- Progress tracking (JSON-based)
- S3 acceleration (faster downloads)
- 18 predefined ADHD/Autism-relevant packages

**Package Categories**:

| Category | Packages | Description |
|----------|----------|-------------|
| **Clinical** | `abcd_cbcls01`, `abcd_ksad01`, `abcd_medhy01`, `abcd_sscey01` | Behavioral assessments, diagnoses, medications |
| **Neuroimaging** | `abcd_betnet02`, `abcd_smrip10201`, `abcd_dmdtifp101`, `abcd_mrfindings01` | Structural MRI, fMRI, DTI |
| **Biospecimens** | `abcd_biospec01`, `abcd_hsss01` | Sample inventory, hormone assays |
| **Sleep** | `abcd_sds01`, `abcd_midacsss01` | Sleep disturbances, chronotype |
| **Diet** | `abcd_eatqp01`, `abcd_ssbpm01` | Eating behaviors, sugar intake |
| **Cognitive** | `abcd_tbss01` | NIH Toolbox (attention, working memory) |
| **Environmental** | `abcd_airsleep01`, `abcd_rhds01` | Air quality, neighborhood SES |
| **Family** | `abcd_fhxssp01`, `abcd_lpds01` | Family history, demographics |

**Usage**:
```bash
# List packages
python nda_downloader.py --list-packages

# Download all ADHD/Autism packages
python nda_downloader.py --adhd-autism-packages --output data/abcd/

# Download specific category
python nda_downloader.py \
    --packages abcd_cbcls01,abcd_ksad01,abcd_betnet02 \
    --output data/abcd/

# Check download status
python nda_downloader.py --status --output data/abcd/

# Resume incomplete downloads
python nda_downloader.py --resume --output data/abcd/
```

**Error Handling**:
- Automatic retry with exponential backoff
- Session timeout recovery
- Progress persistence (resume from interruption)
- S3 fallback if standard download fails

---

### 2. `package_processor.py`

**Purpose**: Process downloaded NDA packages to extract ADHD/Autism-relevant variables

**Key Processing Steps**:

#### CBCL Processing
Extracts:
- ADHD subscale: `cbcl_scr_syn_attention_r` (T-score ≥65 = clinical concern)
- DSM-5 ADHD scale: `cbcl_scr_dsm5_adhd_r`
- Social problems: `cbcl_scr_syn_social_r` (autism-related)
- Internalizing/externalizing scores
- Binary risk flags: `cbcl_adhd_risk`, `cbcl_social_problems`

#### KSADS Processing
Extracts:
- ADHD diagnoses (inattentive, hyperactive, combined): `ksads_adhd_diagnosis`
- ASD diagnosis: `ksads_asd_diagnosis`
- Comorbid diagnoses (depression, anxiety, OCD, ODD, conduct)
- Comorbidity count

#### Medication Processing
Extracts:
- ADHD stimulants: Methylphenidate, amphetamine, dextroamphetamine, lisdexamfetamine
- ADHD non-stimulants: Atomoxetine, guanfacine, clonidine
- Other psychotropics: SSRIs, atypical antipsychotics
- Binary flags: `med_adhd_stimulant`, `med_adhd_any`

#### Neuroimaging Processing

**Structural MRI**:
- Global: Mean cortical thickness, total surface area, total volume
- ADHD regions: Caudate, putamen, anterior cingulate
- Autism regions: Amygdala, hippocampus, superior temporal
- Bilateral averages for subcortical volumes

**Functional Connectivity**:
- Network-level summaries: DMN, salience, executive, attention
- Within-network connectivity (mean, std)
- Global connectivity metrics
- Strong edge count (|r| > 0.3)

**DTI**:
- Global FA/MD (mean, std)
- Tract-specific metrics: Corpus callosum, corona radiata, SLF, internal capsule

**Usage**:
```bash
# Process all packages
python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/

# Process specific timepoint
python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/ \
    --timepoint baseline_year_1_arm_1

# Extract cases only
python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/ \
    --cases-only

# Generate imaging summary
python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/ \
    --extract-imaging
```

**Output**:
- Individual processed packages: `{package_id}_processed.csv`
- Merged data: `abcd_merged_{timepoint}.csv`
- Case cohorts: `abcd_adhd_only.csv`, `abcd_autism_only.csv`, `abcd_comorbid.csv`
- Imaging summary: `imaging_metrics_extracted.csv`

---

### 3. `extract_imaging_metrics.py`

**Purpose**: Extract and summarize neuroimaging metrics for ADHD/Autism research

**Extracted Metrics**:

| Region/Network | Variables | Relevance |
|----------------|-----------|-----------|
| **Basal Ganglia** | Caudate, putamen, pallidum volumes | Reduced in ADHD |
| **Prefrontal Cortex** | Anterior cingulate thickness | Executive function deficits |
| **Amygdala** | Bilateral volume | Social processing (autism) |
| **Hippocampus** | Bilateral volume | Memory, emotion regulation |
| **DMN Connectivity** | Within-network correlation | Atypical in ADHD |
| **Salience Network** | Within-network correlation | Altered in ADHD/autism |
| **Executive Network** | Within-network correlation | Reduced in ADHD |
| **White Matter FA** | Global and tract-specific | Microstructural integrity |

**Usage**:
```bash
# Extract basic metrics
python extract_imaging_metrics.py \
    --input data/abcd/processed/abcd_merged_baseline_year_1_arm_1.csv \
    --output data/abcd/imaging_metrics_extracted.csv

# Include group comparisons
python extract_imaging_metrics.py \
    --input data/abcd/processed/abcd_merged_baseline_year_1_arm_1.csv \
    --output data/abcd/imaging_metrics_extracted.csv \
    --compare-groups

# Generate network summaries
python extract_imaging_metrics.py \
    --input data/abcd/processed/abcd_merged_baseline_year_1_arm_1.csv \
    --output data/abcd/imaging_metrics_extracted.csv \
    --network-summary
```

**Output**:
- `imaging_metrics_extracted.csv`: Subject-level data with all imaging metrics
- `imaging_summary_statistics.csv`: Descriptive statistics per variable
- `imaging_adhd_vs_controls.csv`: Group comparison (t-tests, effect sizes)
- `imaging_autism_vs_controls.csv`: Group comparison for autism

**Group Comparison Output Includes**:
- Sample sizes (n)
- Means and standard deviations
- T-statistics and p-values
- Cohen's d effect sizes
- Significance flags (p < 0.05)

---

## Configuration

### `phenotype_mapping.json`

Complete phenotype dictionary mapping ABCD variables to clinical constructs:

**Contents**:
- **Clinical assessments**: CBCL, KSADS, SCQ with variable names and thresholds
- **Medications**: ADHD stimulants, non-stimulants, other psychotropics
- **Neuroimaging**: Structural MRI regions, connectivity networks, DTI tracts
- **Biospecimens**: Hormone assays (DHEA, testosterone)
- **Sleep**: Sleep disturbance scales, chronotype
- **Diet**: Eating behaviors, sugar intake
- **Cognitive**: NIH Toolbox domains (attention, working memory, processing speed)
- **Environmental**: Air pollution (PM2.5, NO2), neighborhood characteristics
- **Family/genetics**: Family history, genomics availability
- **Demographics**: Parent education, income, employment

**Usage**:
```python
import json

# Load phenotype mapping
with open('data/abcd/phenotype_mapping.json') as f:
    mapping = json.load(f)

# Access ADHD-relevant CBCL scales
adhd_scales = mapping['clinical_assessments']['cbcl']['adhd_relevant_scales']
print(adhd_scales['attention_problems']['variable'])
# Output: 'cbcl_scr_syn_attention_r'

# Access structural MRI variables
adhd_regions = mapping['neuroimaging']['structural_mri']['adhd_relevant_regions']
print(adhd_regions['caudate'])
# Output: {'left': 'smri_vol_scs_caudatelh', 'right': 'smri_vol_scs_caudaterh', ...}
```

---

## Expected Cohort Sizes

| Cohort | Expected N | With Neuroimaging |
|--------|-----------|-------------------|
| **ADHD only** | ~800-900 | ~600-700 |
| **Autism only** | ~200-250 | ~150-180 |
| **ADHD+Autism** | ~100-150 | ~70-100 |
| **Controls** | ~9,000 | ~7,000 |

**Notes**:
- ADHD prevalence: ~8-10% (includes diagnosed + subclinical)
- Autism prevalence: ~2-3% (lower than specialized registries)
- Neuroimaging completion rate: ~75-80% (some participants decline MRI)

---

## Complete Workflow

### 1. Obtain ABCD Access

**Timeline**: 2-4 weeks

1. Create NDA account: https://nda.nih.gov/
2. Complete ABCD data use certification training (~1 hour)
3. Submit application with research plan
4. Obtain institutional authorization signature
5. Wait for Data Access Committee approval (2-4 weeks)
6. Configure NDA credentials locally

### 2. Download Data

```bash
# Configure credentials (one-time)
python nda_downloader.py --configure

# Download all ADHD/Autism packages (~380 MB)
python nda_downloader.py \
    --adhd-autism-packages \
    --output data/abcd/

# Monitor progress
python nda_downloader.py --status --output data/abcd/
```

**Time**: 10-30 minutes

### 3. Process Data

```bash
# Process all packages for baseline timepoint
python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/ \
    --timepoint baseline_year_1_arm_1 \
    --extract-imaging

# Extract ADHD/autism cases
python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/ \
    --cases-only
```

**Time**: 5-10 minutes

### 4. Extract Imaging Metrics

```bash
# Generate imaging summary with group comparisons
python extract_imaging_metrics.py \
    --input data/abcd/processed/abcd_merged_baseline_year_1_arm_1.csv \
    --output data/abcd/imaging_metrics_extracted.csv \
    --compare-groups \
    --network-summary
```

**Time**: 2-5 minutes

### 5. Integrate with Analysis Pipeline

```bash
# Use ABCD-specific config
audhd-pipeline run --config configs/datasets/abcd.yaml
```

---

## Troubleshooting

### Issue 1: NDA Login Fails

**Error**: "Login failed" or "Invalid credentials"

**Solutions**:
```bash
# Reconfigure credentials
python nda_downloader.py --configure

# Check NDA account status at https://nda.nih.gov/
# Verify ABCD access is approved

# Test with downloadcmd directly
downloadcmd -l  # List available studies
```

### Issue 2: Package Download Timeout

**Error**: Download stalls or times out

**Solutions**:
```bash
# Enable S3 acceleration (default, but verify)
python nda_downloader.py --adhd-autism-packages --output data/abcd/
# (S3 is enabled by default)

# If S3 fails, disable it
python nda_downloader.py --adhd-autism-packages --no-s3 --output data/abcd/

# Resume interrupted download
python nda_downloader.py --resume --output data/abcd/

# Download packages one at a time
python nda_downloader.py --packages abcd_cbcls01 --output data/abcd/
python nda_downloader.py --packages abcd_ksad01 --output data/abcd/
```

### Issue 3: Package Not Found

**Error**: "Unknown package: abcd_xxxxx"

**Solutions**:
```bash
# Check available packages
python nda_downloader.py --list-packages

# Verify package ID spelling (case-sensitive)
# Some packages may require separate NDA application
```

### Issue 4: No ADHD/Autism Cases Found

**Error**: `process_cbcl()` or `process_ksads()` returns 0 cases

**Solutions**:
```bash
# Check if diagnosis packages were downloaded
ls data/abcd/abcd_cbcls01/
ls data/abcd/abcd_ksad01/

# If missing, download them
python nda_downloader.py \
    --packages abcd_cbcls01,abcd_ksad01 \
    --output data/abcd/

# Verify data structure
python -c "
import pandas as pd
df = pd.read_csv('data/abcd/abcd_ksad01/abcd_ksad01.txt', sep='\t', skiprows=[1])
print(df.columns.tolist()[:20])
"
```

### Issue 5: Imaging Metrics Missing

**Error**: `extract_imaging_metrics.py` finds no imaging data

**Solutions**:
```bash
# Download neuroimaging packages
python nda_downloader.py \
    --packages abcd_betnet02,abcd_smrip10201,abcd_dmdtifp101 \
    --output data/abcd/

# Process imaging packages
python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/ \
    --extract-imaging

# Check merged data for imaging columns
python -c "
import pandas as pd
df = pd.read_csv('data/abcd/processed/abcd_merged_baseline_year_1_arm_1.csv')
imaging_cols = [col for col in df.columns if 'smri' in col or 'rsfmri' in col]
print(f'Found {len(imaging_cols)} imaging columns')
"
```

### Issue 6: Memory Errors with Large Files

**Error**: `MemoryError` or process killed

**Solutions**:
```bash
# Process packages separately (don't merge all at once)
python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/ \
    --categories clinical

python package_processor.py \
    --input data/abcd/ \
    --output data/abcd/processed/ \
    --categories neuroimaging

# Use low_memory option in pandas
# (Already implemented in package_processor.py)

# Or process on machine with more RAM
```

---

## Data Structure

### ABCD Data Files

All ABCD data structures follow this format:

```
Line 1: Column headers
Line 2: Column descriptions (skip during import)
Line 3+: Data rows
```

**Delimiter**: Tab-separated (`\t`)

**Key Columns**:
- `src_subject_id`: Unique subject ID (format: `NDAR_INVxxxxxxxx`)
- `eventname`: Timepoint (e.g., `baseline_year_1_arm_1`, `2_year_follow_up_y_arm_1`)
- `site_id_l`: Data collection site
- `rel_family_id`: Family ID (for sibling pairs)

**Import Example**:
```python
import pandas as pd

# Skip description row (row index 1)
df = pd.read_csv('abcd_cbcls01.txt', sep='\t', skiprows=[1], low_memory=False)
```

### Timepoints

| Timepoint Code | Age | Description |
|----------------|-----|-------------|
| `baseline_year_1_arm_1` | 9-10 | Baseline assessment |
| `1_year_follow_up_y_arm_1` | 10-11 | Year 1 follow-up |
| `2_year_follow_up_y_arm_1` | 11-12 | Year 2 follow-up |
| `3_year_follow_up_y_arm_1` | 12-13 | Year 3 follow-up |
| `4_year_follow_up_y_arm_1` | 13-14 | Year 4 follow-up (ongoing) |

**Note**: 10-year study (through age 19-20). Years 5-10 data not yet released.

---

## Integration with Main Pipeline

### Update `configs/datasets/abcd.yaml`

```yaml
dataset:
  name: "ABCD_Study"
  data_root: "data/abcd"

  paths:
    phenotypes: "${data_root}/processed/abcd_merged_baseline_year_1_arm_1.csv"
    imaging: "${data_root}/imaging_metrics_extracted.csv"
    cohort_ids:
      adhd: "${data_root}/processed/abcd_adhd_only.csv"
      autism: "${data_root}/processed/abcd_autism_only.csv"
      comorbid: "${data_root}/processed/abcd_comorbid.csv"

  variables:
    subject_id: "src_subject_id"
    adhd_diagnosis: "ksads_adhd_diagnosis"
    autism_diagnosis: "ksads_asd_diagnosis"

  timepoints:
    - baseline_year_1_arm_1
    - 2_year_follow_up_y_arm_1
    - 4_year_follow_up_y_arm_1
```

### Run Pipeline

```bash
# Process ABCD data with main analysis pipeline
audhd-pipeline run --config configs/datasets/abcd.yaml
```

---

## Citation

If you use ABCD Study data, please cite:

```bibtex
@article{volkow2018adolescent,
  title={The Conception of the ABCD Study: From Substance Use to a Broad NIH Collaboration},
  author={Volkow, Nora D and Koob, George F and Croyle, Robert T and Bianchi, Diana W and Gordon, Joshua A and Koroshetz, Walter J and P{\'e}rez-Stable, Eliseo J and Riley, William T and Bloch, Michele H and Conway, Kevin and Deeds, Bethany G and Dowling, Gayathri J and Grant, Steven and Howlett, Katia D and Matochik, John A and Morgan, Gregory J and Murray, Michael M and Noronha, Antonio and Spong, Catherine Y and Wargo, Elaine M and Warren, Kenneth R and Weiss, Susan RB},
  journal={Developmental Cognitive Neuroscience},
  volume={32},
  pages={4--7},
  year={2018}
}

@article{casey2018adolescent,
  title={The Adolescent Brain Cognitive Development (ABCD) Study: Imaging Acquisition Across 21 Sites},
  author={Casey, BJ and Cannonier, T and Conley, MI and Cohen, AO and Barch, DM and Heitzeg, MM and Soules, ME and Teslovich, T and Dellarco, DV and Garavan, H and others},
  journal={Developmental Cognitive Neuroscience},
  volume={32},
  pages={43--54},
  year={2018}
}
```

---

## Support

**ABCD Study**:
- Website: https://abcdstudy.org
- NDA Portal: https://nda.nih.gov/abcd
- Email: ndar@mail.nih.gov

**NDA Technical Support**:
- NDA Help Desk: https://nda.nih.gov/help
- Email: NDAHelp@mail.nih.gov

**Pipeline Issues**:
- GitHub: github.com/rohanvinaik/AuDHD_Correlation_Study/issues

**Related Documentation**:
- Dataset catalog: `data/catalogs/dataset_inventory.json`
- Access tracker: `data/catalogs/access_tracker.md`
- Data sources guide: `docs/data_sources.md`
- ABCD phenotype mapping: `data/abcd/phenotype_mapping.json`

---

**Last Updated**: 2025-01-30
**ABCD Release**: 5.0 (as of documentation date)