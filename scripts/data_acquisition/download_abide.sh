#!/bin/bash
# Download ABIDE (Autism Brain Imaging Data Exchange) dataset
# Open access neuroimaging data: 1,112 autism, 1,033 controls
# URL: http://fcon_1000.projects.nitrc.org/indi/abide/

set -euo pipefail

# Configuration
DATASET_NAME="abide"
OUTPUT_DIR="${DATA_ROOT:-../../data}/raw/${DATASET_NAME}"
ABIDE_VERSION="${ABIDE_VERSION:-ABIDE_I}"  # ABIDE_I or ABIDE_II or both

# S3 bucket locations
ABIDE_I_S3="s3://fcp-indi/data/Projects/ABIDE_Initiative"
ABIDE_II_S3="s3://fcp-indi/data/Projects/ABIDE2"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ABIDE Dataset Download ===${NC}"
echo "Output directory: ${OUTPUT_DIR}"
echo "ABIDE version: ${ABIDE_VERSION}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

# Check for AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not found${NC}"
    echo "Install with: pip install awscli"
    echo "Or: brew install awscli (macOS)"
    exit 1
fi

# Configure AWS for anonymous access (no credentials needed for public S3)
aws configure set default.s3.signature_version s3v4
aws configure set default.region us-east-1

# Function to download ABIDE data
download_abide() {
    local version=$1
    local s3_bucket=$2

    echo -e "${YELLOW}Downloading ${version}...${NC}"

    # Create subdirectory
    mkdir -p "${version}"
    cd "${version}"

    # Download phenotypic data
    echo "Downloading phenotypic data..."
    aws s3 cp "${s3_bucket}/Phenotypic_V1_0b_preprocessed1.csv" . --no-sign-request || {
        echo -e "${RED}Warning: Could not download phenotypic data${NC}"
    }

    # List available preprocessing pipelines
    echo -e "\n${GREEN}Available preprocessing pipelines:${NC}"
    echo "1. CPAC (Configurable Pipeline for the Analysis of Connectomes)"
    echo "2. DPARSF (Data Processing Assistant for Resting-State fMRI)"
    echo "3. NIAK (Neuroimaging Analysis Kit)"
    echo "4. raw (unprocessed anatomical and functional scans)"

    read -p "Download preprocessed data? (y/n): " download_preprocessed

    if [[ "${download_preprocessed}" =~ ^[Yy]$ ]]; then
        read -p "Select pipeline (cpac/dparsf/niak/raw): " pipeline

        echo "Downloading ${pipeline} preprocessed data..."
        echo "Note: This will take significant time and disk space (100+ GB per pipeline)"

        # Download pipeline-specific data
        aws s3 sync "${s3_bucket}/Outputs/${pipeline}/" "./${pipeline}/" \
            --no-sign-request \
            --exclude "*" \
            --include "*.nii.gz" \
            --include "*.txt"
    else
        echo "Skipping preprocessed data download"
        echo "You can download later using:"
        echo "  aws s3 sync ${s3_bucket}/Outputs/[pipeline]/ ${OUTPUT_DIR}/${version}/[pipeline]/ --no-sign-request"
    fi

    cd ..
}

# Download ABIDE I
if [[ "${ABIDE_VERSION}" == "ABIDE_I" ]] || [[ "${ABIDE_VERSION}" == "both" ]]; then
    download_abide "ABIDE_I" "${ABIDE_I_S3}"
fi

# Download ABIDE II
if [[ "${ABIDE_VERSION}" == "ABIDE_II" ]] || [[ "${ABIDE_VERSION}" == "both" ]]; then
    download_abide "ABIDE_II" "${ABIDE_II_S3}"
fi

# Download site information
echo -e "\n${YELLOW}Downloading site information...${NC}"
curl -o ABIDE_sites.txt "http://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_sites.txt" 2>/dev/null || {
    echo -e "${RED}Warning: Could not download site information${NC}"
}

# Download data dictionary
echo "Downloading data dictionary..."
curl -o ABIDE_data_dictionary.pdf "http://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_data_dictionary.pdf" 2>/dev/null || {
    echo -e "${RED}Warning: Could not download data dictionary${NC}"
}

# Create README
cat > README.md << 'EOF'
# ABIDE Dataset

**Source**: Autism Brain Imaging Data Exchange (ABIDE)
**URL**: http://fcon_1000.projects.nitrc.org/indi/abide/
**Download Date**: $(date +%Y-%m-%d)

## Dataset Description

- **ABIDE I**: 1,112 individuals with autism, 1,033 controls
- **ABIDE II**: Additional samples from new sites
- **Age range**: 7-64 years
- **Data types**: Structural MRI, resting-state fMRI, phenotypic data

## Preprocessing Pipelines

Available preprocessed versions:
- **CPAC**: Configurable Pipeline for the Analysis of Connectomes
- **DPARSF**: Data Processing Assistant for Resting-State fMRI
- **NIAK**: Neuroimaging Analysis Kit
- **raw**: Unprocessed anatomical and functional scans

## Phenotypic Variables

- Diagnosis (autism/control)
- Age, sex, handedness
- ADOS scores (when available)
- ADI-R scores (when available)
- IQ measures (when available)
- Site identifier

## Citation

Di Martino, A., et al. (2017). Enhancing studies of the connectome in autism
using the autism brain imaging data exchange II. Scientific Data, 4, 170010.

Di Martino, A., et al. (2014). The autism brain imaging data exchange: towards
a large-scale evaluation of the intrinsic brain architecture in autism.
Molecular Psychiatry, 19(6), 659-667.

## Notes

- Open access dataset (no DUA required)
- BIDS-compatible format available
- Multiple preprocessing pipelines available
- Site effects should be considered in analysis

## Directory Structure

```
ABIDE_I/
├── Phenotypic_V1_0b_preprocessed1.csv  # Phenotypic data
├── cpac/                                # CPAC preprocessed (if downloaded)
├── dparsf/                              # DPARSF preprocessed (if downloaded)
├── niak/                                # NIAK preprocessed (if downloaded)
└── raw/                                 # Raw scans (if downloaded)

ABIDE_II/
└── [similar structure]
```
EOF

echo -e "\n${GREEN}Download complete!${NC}"
echo "Dataset location: ${OUTPUT_DIR}"
echo ""
echo "To use this data in the pipeline:"
echo "  1. Update configs/datasets/abide.yaml with data paths"
echo "  2. Run preprocessing: audhd-pipeline preprocess --config configs/datasets/abide.yaml"
echo ""
echo "Note: MRI data requires additional processing before integration with genomic data"