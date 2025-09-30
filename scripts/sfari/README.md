# SFARI Data Access Tools

**Purpose**: Automated tools for accessing and downloading SFARI datasets (SPARK, SSC, AGRE, Simons Searchlight)

## ⚠️ Prerequisites

**Before using these tools, you MUST:**

1. ✅ **Register** at SFARI Base (base.sfari.org)
2. ✅ **Complete training** modules
3. ✅ **Submit Data Access Request** describing your research
4. ✅ **Execute Data Use Agreement (DUA)** signed by authorized official
5. ✅ **Receive approval** from SFARI Data Access Committee (2-4 weeks)

**Without approval, these scripts will not work.**

---

## 🛠️ Tools Overview

| Script | Purpose | Requires Approval |
|--------|---------|-------------------|
| `sfari_portal_navigator.py` | Portal navigation, dataset browsing | ✅ Yes |
| `spark_data_downloader.py` | Robust file downloads with resume | ✅ Yes |
| `isec_aws_setup.sh` | AWS S3 access for SPARK iSEC | ✅ Yes + iSEC access |
| `parse_spark_phenotypes.py` | Parse phenotype dictionary | ❌ No (offline) |

---

## 📋 Installation

### Required Packages

```bash
# Python packages
pip install selenium pandas tqdm

# Browser driver (for portal navigation)
# macOS
brew install chromedriver

# Or download from: https://chromedriver.chromium.org/

# AWS CLI (for iSEC)
pip install awscli
# Or: brew install awscli

# Download accelerator (optional but recommended)
brew install aria2
```

### Verify Installation

```bash
# Check ChromeDriver
chromedriver --version

# Check AWS CLI
aws --version

# Check aria2c (optional)
aria2c --version
```

---

## 🚀 Quick Start

### 1. Navigate Portal and List Datasets

```bash
# List all available datasets
python sfari_portal_navigator.py \
    --username your@email.com \
    --list-datasets

# Explore SPARK dataset
python sfari_portal_navigator.py \
    --username your@email.com \
    --dataset SPARK \
    --explore

# Get phenotype browser data
python sfari_portal_navigator.py \
    --username your@email.com \
    --dataset SPARK \
    --phenotypes \
    --output data/manifests/spark_phenotypes.csv

# Generate download manifest
python sfari_portal_navigator.py \
    --username your@email.com \
    --dataset SPARK \
    --manifest \
    --output data/manifests/spark_manifest.csv
```

**Output**: Creates manifest of available files with download URLs

### 2. Download SPARK Data

#### Option A: From Manifest (Standard SFARI Base)

```bash
# Download all files from manifest
python spark_data_downloader.py \
    --manifest data/manifests/spark_manifest.csv \
    --output data/raw/spark/

# Download specific file types
python spark_data_downloader.py \
    --manifest data/manifests/spark_manifest.csv \
    --types phenotype,family \
    --output data/raw/spark/

# Resume interrupted download
python spark_data_downloader.py \
    --manifest data/manifests/spark_manifest.csv \
    --resume \
    --output data/raw/spark/

# Parallel downloads (faster)
python spark_data_downloader.py \
    --manifest data/manifests/spark_manifest.csv \
    --workers 8 \
    --output data/raw/spark/
```

#### Option B: From AWS S3 (iSEC)

**Note**: iSEC access requires additional approval from SPARK team.

```bash
# Setup AWS credentials for iSEC
./isec_aws_setup.sh \
    --access-key YOUR_AWS_ACCESS_KEY \
    --secret-key YOUR_AWS_SECRET_KEY

# List available data
./isec_aws_setup.sh --list

# Sync specific categories
./isec_aws_setup.sh --sync phenotypes,family,qc

# Sync all genomics (LARGE - 100+ GB)
./isec_aws_setup.sh --sync genomics

# Sync specific files from list
echo "genomics/spark_wgs_chr22.vcf.gz" > spark_files.txt
echo "phenotypes/spark_phenotypes.csv" >> spark_files.txt
./isec_aws_setup.sh --sync-files spark_files.txt

# Verify downloads
./isec_aws_setup.sh --verify

# Generate manifest of downloaded files
./isec_aws_setup.sh --manifest
```

### 3. Parse Phenotype Dictionary

```bash
# Parse data dictionary
python parse_spark_phenotypes.py \
    --input data/raw/spark/documentation/spark_data_dictionary.csv \
    --output data/dictionaries/spark_phenotype_dictionary.json \
    --full

# Generate summary CSV
python parse_spark_phenotypes.py \
    --input data/raw/spark/documentation/spark_data_dictionary.csv \
    --output data/dictionaries/spark_phenotype_dictionary.json \
    --summary data/dictionaries/spark_phenotype_summary.csv
```

**Output**: Structured JSON with:
- Categorized variables (demographics, diagnosis, medical, etc.)
- Clinical instruments (ADOS, ADI-R, SRS, Vineland)
- Family relationships
- Metabolomics subset (if available)

---

## 📊 Complete Workflow Example

### Step 1: Obtain Access

1. Apply for SFARI Base access (see prerequisites)
2. Wait for approval (2-4 weeks)
3. (Optional) Request iSEC access for S3 downloads

### Step 2: Discover Available Data

```bash
# Set credentials as environment variables (more secure)
export SFARI_USERNAME="your@email.com"
export SFARI_PASSWORD="your_password"

# List datasets
python sfari_portal_navigator.py \
    --username $SFARI_USERNAME \
    --list-datasets

# Generate manifest for SPARK
python sfari_portal_navigator.py \
    --username $SFARI_USERNAME \
    --dataset SPARK \
    --manifest \
    --output data/manifests/spark_manifest.csv
```

### Step 3: Download Data

```bash
# Download phenotype and family data (small, ~100 MB)
python spark_data_downloader.py \
    --manifest data/manifests/spark_manifest.csv \
    --types phenotype,family,documentation \
    --output data/raw/spark/ \
    --workers 4

# Check what was downloaded
ls -lh data/raw/spark/
```

### Step 4: Parse Phenotype Dictionary

```bash
# Parse and categorize variables
python parse_spark_phenotypes.py \
    --input data/raw/spark/documentation/spark_data_dictionary.csv \
    --output data/dictionaries/spark_phenotype_dictionary.json \
    --summary data/dictionaries/spark_phenotype_summary.csv \
    --full
```

### Step 5: (Optional) Download Genomics from iSEC

**Warning**: Genomic data is LARGE (100+ GB to 10+ TB)

```bash
# Setup iSEC access
./isec_aws_setup.sh \
    --access-key YOUR_KEY \
    --secret-key YOUR_SECRET

# List available genomic data
./isec_aws_setup.sh --list

# Download chromosome 22 only (test, ~5 GB)
echo "genomics/spark_wgs_chr22.vcf.gz" > test_files.txt
./isec_aws_setup.sh --sync-files test_files.txt

# If test successful, download all genomics
./isec_aws_setup.sh --sync genomics  # This will take hours!
```

---

## 🔧 Advanced Usage

### Resume Interrupted Downloads

All download scripts support resume:

```bash
# Python downloader (uses aria2c if available)
python spark_data_downloader.py --resume --output data/raw/spark/

# iSEC (AWS S3 sync is resumable by default)
./isec_aws_setup.sh --sync genomics  # Re-run to resume
```

### Parallel Downloads

```bash
# Increase workers for faster downloads (use with caution)
python spark_data_downloader.py \
    --manifest spark_manifest.csv \
    --workers 16 \
    --output data/raw/spark/
```

### Selective Downloads

```bash
# Create custom file list
cat > priority_files.txt << EOF
phenotypes/spark_phenotypes.csv
phenotypes/spark_demographics.csv
family/spark_pedigree.fam
documentation/spark_data_dictionary.csv
EOF

# Download only these files
./isec_aws_setup.sh --sync-files priority_files.txt
```

### Checksum Verification

```bash
# Verify all downloads
python spark_data_downloader.py \
    --manifest spark_manifest.csv \
    --output data/raw/spark/  # Will verify checksums

# Skip verification (faster, less safe)
python spark_data_downloader.py \
    --manifest spark_manifest.csv \
    --no-verify \
    --output data/raw/spark/
```

---

## 📁 Output Structure

After downloading, data will be organized as:

```
data/raw/spark/
├── phenotypes/
│   ├── spark_phenotypes_v2023.csv
│   ├── spark_demographics.csv
│   └── spark_assessments.csv
├── family/
│   ├── spark_pedigree.fam
│   └── spark_family_structure.csv
├── genomics/
│   ├── spark_wgs_manifest.txt
│   ├── spark_snp_array_manifest.txt
│   └── spark_variants_chr22.vcf.gz
├── qc/
│   └── spark_qc_reports.zip
├── documentation/
│   ├── spark_data_dictionary.csv
│   ├── spark_codebook.pdf
│   └── spark_release_notes.txt
└── download_manifest.csv  # Auto-generated inventory
```

---

## 🔍 Troubleshooting

### Issue 1: Login Fails

**Symptoms**: "Login failed" error

**Solutions**:
```bash
# Check credentials
python sfari_portal_navigator.py --username your@email.com --list-datasets

# Common issues:
# 1. DUA not executed - check SFARI Base portal
# 2. Access not yet approved - wait for approval email
# 3. Password incorrect - reset at base.sfari.org
# 4. ChromeDriver not installed - brew install chromedriver
```

### Issue 2: ChromeDriver Errors

**Symptoms**: "chromedriver not found" or version mismatch

**Solutions**:
```bash
# macOS - allow ChromeDriver
xattr -d com.apple.quarantine /usr/local/bin/chromedriver

# Update ChromeDriver
brew upgrade chromedriver

# Or download matching version from:
# https://chromedriver.chromium.org/
```

### Issue 3: Download Timeouts

**Symptoms**: Downloads fail or timeout

**Solutions**:
```bash
# Use aria2c for more robust downloads
brew install aria2

# Reduce parallel workers
python spark_data_downloader.py --workers 2

# Resume interrupted downloads
python spark_data_downloader.py --resume
```

### Issue 4: iSEC Access Denied

**Symptoms**: "Cannot access iSEC bucket"

**Solutions**:
```bash
# Check iSEC access approval
# Email: spark@simonsfoundation.org

# Verify AWS credentials
aws s3 ls s3://spark-isec/ --profile spark-isec

# Check AWS profile
cat ~/.aws/credentials
cat ~/.aws/config
```

### Issue 5: Large File Downloads

**Symptoms**: Genomic files (100+ GB) taking too long

**Solutions**:
```bash
# Use iSEC (AWS S3) for large files - much faster
./isec_aws_setup.sh --sync genomics

# Or download during off-peak hours
# Or request SPARK team to ship hard drive for very large requests
```

---

## 🔒 Security Best Practices

### Credential Management

```bash
# Don't hardcode credentials in scripts!

# Option 1: Environment variables
export SFARI_USERNAME="your@email.com"
export SFARI_PASSWORD="your_password"

# Option 2: Secure credential storage
# Use OS keychain
security add-generic-password -a $USER -s sfari -w "your_password"
security find-generic-password -a $USER -s sfari -w  # Retrieve

# Option 3: AWS credentials file (for iSEC)
aws configure --profile spark-isec
```

### Data Security

```bash
# SFARI data is highly sensitive - follow DUA requirements:

# 1. Encrypted storage
# diskutil apfs enableFileVault /Volumes/YourDrive  # macOS

# 2. Restrict access
chmod 700 data/raw/spark/
chmod 600 data/raw/spark/*/*.csv

# 3. No cloud storage without explicit permission
# Don't sync to Dropbox/Google Drive!

# 4. Secure deletion when done
# Use secure delete tools, not regular rm
```

---

## 📚 Additional Resources

### SFARI Documentation

- **SFARI Base Portal**: https://base.sfari.org
- **SPARK Overview**: https://sparkforautism.org
- **Data Access**: https://base.sfari.org/help/data-access
- **iSEC Guide**: https://base.sfari.org/help/isec

### Support

- **SFARI Base Support**: SFARIBase@simonsfoundation.org
- **SPARK Team**: spark@simonsfoundation.org
- **Pipeline Issues**: github.com/rohanvinaik/AuDHD_Correlation_Study/issues

### Related Documentation

- **Dataset catalog**: `data/catalogs/dataset_inventory.json`
- **Access tracker**: `data/catalogs/access_tracker.md`
- **Data sources guide**: `docs/data_sources.md`
- **Pipeline configuration**: `configs/datasets/spark.yaml`

---

## 📝 Citation

If you use SPARK data, please cite:

```bibtex
@article{spark_consortium_2018,
  title={SPARK: A US Cohort of 50,000 Families to Accelerate Autism Research},
  author={SPARK Consortium},
  journal={Neuron},
  volume={97},
  number={3},
  pages={488--493},
  year={2018},
  publisher={Elsevier}
}
```

---

## ⚖️ Data Use Agreement

**IMPORTANT**: All data downloaded using these tools is subject to the SFARI Base Data Use Certificate. You MUST:

- ✅ Use data only for approved research project
- ✅ Not attempt to re-identify participants
- ✅ Not share raw data with unauthorized individuals
- ✅ Acknowledge SFARI/SPARK in publications
- ✅ Destroy or return data when project complete
- ❌ Do not use for commercial purposes without permission

Violations may result in:
- Loss of data access
- Notification to your institution
- Legal action

---

**Last Updated**: 2025-01-30
**Maintained By**: AuDHD Correlation Study Team