#!/bin/bash
# iSEC AWS Setup Script
#
# Configure AWS credentials and access for SPARK iSEC (SPARK data on AWS S3)
# iSEC = interactive SFARI Experimental Cloud
#
# Prerequisites:
# 1. Approved SFARI Base access
# 2. iSEC access granted by SPARK team
# 3. AWS CLI installed (pip install awscli)
#
# Usage:
#   ./isec_aws_setup.sh --access-key YOUR_KEY --secret-key YOUR_SECRET
#   ./isec_aws_setup.sh --profile spark-isec --sync

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ISEC_BUCKET="spark-isec"
ISEC_REGION="us-east-1"
AWS_PROFILE="spark-isec"
OUTPUT_DIR="${DATA_ROOT:-../../data}/raw/spark"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found"
        echo "Install with: pip install awscli"
        echo "Or: brew install awscli (macOS)"
        exit 1
    fi

    local version
    version=$(aws --version 2>&1 | cut -d' ' -f1 | cut -d'/' -f2)
    log_info "AWS CLI version: $version"
}

check_isec_access() {
    log_info "Checking iSEC access..."

    if aws s3 ls "s3://${ISEC_BUCKET}" --profile "${AWS_PROFILE}" &> /dev/null; then
        log_info "✓ iSEC access confirmed"
        return 0
    else
        log_error "✗ Cannot access iSEC bucket"
        log_error "Please ensure:"
        log_error "  1. You have approved SFARI Base access"
        log_error "  2. SPARK team has granted iSEC access"
        log_error "  3. AWS credentials are correctly configured"
        return 1
    fi
}

configure_aws_profile() {
    local access_key="$1"
    local secret_key="$2"

    log_section "Configuring AWS Profile"

    log_info "Setting up AWS profile: ${AWS_PROFILE}"

    # Configure profile
    aws configure set aws_access_key_id "${access_key}" --profile "${AWS_PROFILE}"
    aws configure set aws_secret_access_key "${secret_key}" --profile "${AWS_PROFILE}"
    aws configure set region "${ISEC_REGION}" --profile "${AWS_PROFILE}"
    aws configure set output "json" --profile "${AWS_PROFILE}"

    log_info "✓ AWS profile configured: ${AWS_PROFILE}"

    # Test access
    if check_isec_access; then
        log_info "✓ iSEC access verified"
    else
        log_error "✗ iSEC access verification failed"
        exit 1
    fi
}

list_isec_data() {
    log_section "Available iSEC Data"

    log_info "Listing SPARK iSEC bucket contents..."

    # List top-level directories
    aws s3 ls "s3://${ISEC_BUCKET}/" --profile "${AWS_PROFILE}" | while read -r line; do
        echo "  $line"
    done

    echo ""
    log_info "Data categories:"

    # List subdirectories with sizes
    for category in genomics phenotypes family qc documentation; do
        if aws s3 ls "s3://${ISEC_BUCKET}/${category}/" --profile "${AWS_PROFILE}" &> /dev/null; then
            local size
            size=$(aws s3 ls "s3://${ISEC_BUCKET}/${category}/" --recursive --profile "${AWS_PROFILE}" \
                | awk '{sum+=$3} END {print sum/1024/1024/1024 " GB"}')
            echo "  ${category}: ${size}"
        fi
    done
}

sync_isec_data() {
    local categories="$1"

    log_section "Syncing iSEC Data"

    log_info "Output directory: ${OUTPUT_DIR}"
    log_info "Categories: ${categories}"

    mkdir -p "${OUTPUT_DIR}"

    # Split categories
    IFS=',' read -ra CATS <<< "${categories}"

    for category in "${CATS[@]}"; do
        log_info "Syncing ${category}..."

        local source="s3://${ISEC_BUCKET}/${category}/"
        local dest="${OUTPUT_DIR}/${category}/"

        mkdir -p "${dest}"

        # Sync with progress
        aws s3 sync "${source}" "${dest}" \
            --profile "${AWS_PROFILE}" \
            --region "${ISEC_REGION}" \
            --no-progress \
            --exclude "*.bam.bai" \
            --exclude "*.cram.crai" \
            | while read -r line; do
                if [[ $line == download:* ]]; then
                    echo "  ↓ ${line#download: }"
                fi
            done

        log_info "✓ ${category} sync complete"
    done

    log_info "✓ All syncs complete"
    log_info "Data location: ${OUTPUT_DIR}"
}

sync_specific_files() {
    local file_list="$1"

    log_section "Syncing Specific Files"

    if [[ ! -f "${file_list}" ]]; then
        log_error "File list not found: ${file_list}"
        exit 1
    fi

    log_info "Syncing files from: ${file_list}"

    local count=0
    local total
    total=$(wc -l < "${file_list}")

    while IFS= read -r s3_path; do
        ((count++))

        # Skip empty lines and comments
        [[ -z "${s3_path}" || "${s3_path}" =~ ^# ]] && continue

        # Remove s3:// prefix if present
        s3_path="${s3_path#s3://}"
        s3_path="${s3_path#${ISEC_BUCKET}/}"

        log_info "[${count}/${total}] Downloading: ${s3_path}"

        # Determine output path
        local output_file="${OUTPUT_DIR}/${s3_path}"
        mkdir -p "$(dirname "${output_file}")"

        # Download file
        aws s3 cp "s3://${ISEC_BUCKET}/${s3_path}" "${output_file}" \
            --profile "${AWS_PROFILE}" \
            --region "${ISEC_REGION}" || {
            log_warn "Failed to download: ${s3_path}"
        }

    done < "${file_list}"

    log_info "✓ Specific file sync complete"
}

verify_downloads() {
    log_section "Verifying Downloads"

    log_info "Checking downloaded files..."

    # Check for common data types
    local genomic_count
    local phenotype_count
    local total_size

    genomic_count=$(find "${OUTPUT_DIR}/genomics" -type f 2>/dev/null | wc -l || echo 0)
    phenotype_count=$(find "${OUTPUT_DIR}/phenotypes" -type f 2>/dev/null | wc -l || echo 0)
    total_size=$(du -sh "${OUTPUT_DIR}" 2>/dev/null | cut -f1 || echo "0")

    echo ""
    echo "Download Summary:"
    echo "  Genomic files: ${genomic_count}"
    echo "  Phenotype files: ${phenotype_count}"
    echo "  Total size: ${total_size}"
    echo ""

    # Check for key files
    log_info "Checking for key files..."

    local key_files=(
        "phenotypes/spark_phenotypes.csv"
        "phenotypes/spark_demographics.csv"
        "family/spark_pedigree.fam"
        "documentation/README.md"
    )

    for file in "${key_files[@]}"; do
        if [[ -f "${OUTPUT_DIR}/${file}" ]]; then
            echo "  ✓ ${file}"
        else
            echo "  ✗ ${file} (missing)"
        fi
    done
}

generate_download_manifest() {
    log_section "Generating Download Manifest"

    local manifest_file="${OUTPUT_DIR}/download_manifest.csv"

    log_info "Creating manifest: ${manifest_file}"

    # Header
    echo "file_path,size_bytes,modified_date,md5sum" > "${manifest_file}"

    # Find all files and add to manifest
    find "${OUTPUT_DIR}" -type f -not -path "*/\.*" | while read -r file; do
        local rel_path="${file#${OUTPUT_DIR}/}"
        local size
        local modified
        local md5

        size=$(stat -f%z "${file}" 2>/dev/null || stat -c%s "${file}" 2>/dev/null || echo "0")
        modified=$(stat -f%Sm -t "%Y-%m-%d %H:%M:%S" "${file}" 2>/dev/null || \
                   stat -c%y "${file}" 2>/dev/null | cut -d'.' -f1 || echo "unknown")

        # Calculate MD5 (optional, can be slow for large files)
        # md5=$(md5sum "${file}" 2>/dev/null | cut -d' ' -f1 || echo "")
        md5=""

        echo "${rel_path},${size},${modified},${md5}" >> "${manifest_file}"
    done

    local file_count
    file_count=$(wc -l < "${manifest_file}")
    ((file_count--))  # Subtract header

    log_info "✓ Manifest created: ${file_count} files"
}

show_usage() {
    cat << EOF
Usage: ./isec_aws_setup.sh [OPTIONS]

Setup and sync SPARK iSEC data from AWS S3

OPTIONS:
    --access-key KEY        AWS access key ID
    --secret-key KEY        AWS secret access key
    --profile NAME          AWS profile name (default: spark-isec)
    --list                  List available iSEC data
    --sync CATEGORIES       Sync data categories (comma-separated)
                            Categories: genomics,phenotypes,family,qc,documentation
    --sync-files FILE       Sync specific files from list
    --verify                Verify downloaded files
    --manifest              Generate download manifest
    --output DIR            Output directory (default: data/raw/spark)
    --help                  Show this help message

EXAMPLES:
    # Configure AWS access
    ./isec_aws_setup.sh --access-key YOUR_KEY --secret-key YOUR_SECRET

    # List available data
    ./isec_aws_setup.sh --list

    # Sync phenotype and family data
    ./isec_aws_setup.sh --sync phenotypes,family

    # Sync specific files
    ./isec_aws_setup.sh --sync-files spark_files.txt

    # Verify downloads
    ./isec_aws_setup.sh --verify

    # Generate manifest
    ./isec_aws_setup.sh --manifest

NOTES:
    - Requires approved SFARI Base access
    - iSEC access must be granted by SPARK team
    - Large genomic files may take hours to download
    - Use --sync with specific categories to save time/bandwidth

EOF
}

# Main script
main() {
    local access_key=""
    local secret_key=""
    local action=""
    local sync_categories=""
    local file_list=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --access-key)
                access_key="$2"
                shift 2
                ;;
            --secret-key)
                secret_key="$2"
                shift 2
                ;;
            --profile)
                AWS_PROFILE="$2"
                shift 2
                ;;
            --list)
                action="list"
                shift
                ;;
            --sync)
                action="sync"
                sync_categories="$2"
                shift 2
                ;;
            --sync-files)
                action="sync-files"
                file_list="$2"
                shift 2
                ;;
            --verify)
                action="verify"
                shift
                ;;
            --manifest)
                action="manifest"
                shift
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Header
    log_section "SPARK iSEC AWS Setup"

    # Check AWS CLI
    check_aws_cli

    # Configure if credentials provided
    if [[ -n "${access_key}" && -n "${secret_key}" ]]; then
        configure_aws_profile "${access_key}" "${secret_key}"
    fi

    # Execute action
    case "${action}" in
        list)
            check_isec_access && list_isec_data
            ;;
        sync)
            check_isec_access && sync_isec_data "${sync_categories}"
            ;;
        sync-files)
            check_isec_access && sync_specific_files "${file_list}"
            ;;
        verify)
            verify_downloads
            ;;
        manifest)
            generate_download_manifest
            ;;
        "")
            if [[ -n "${access_key}" ]]; then
                log_info "AWS profile configured. Use --list or --sync to access data."
            else
                show_usage
            fi
            ;;
    esac
}

# Run main
main "$@"