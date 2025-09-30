#!/bin/bash
# COMPREHENSIVE DATA DOWNLOAD - EVERYTHING
# Downloads all public data + paper supplements
# Expected time: 2-6 hours depending on connection speed

set -e  # Exit on error

PROJECT_ROOT="/Users/rohanvinaik/AuDHD_Correlation_Study"
cd "$PROJECT_ROOT"

# Create all needed directories
mkdir -p data/raw/{sra,geo,metabolights,papers}
mkdir -p data/papers/{pdfs,supplements,extracted}
mkdir -p logs/downloads

echo "================================================================================"
echo "COMPREHENSIVE DATA DOWNLOAD - ALL SOURCES"
echo "Started: $(date)"
echo "================================================================================"
echo ""
echo "This will download:"
echo "  - ~350 microbiome studies from SRA"
echo "  - ~8 gene expression datasets from GEO"
echo "  - Published papers + supplements for ADHD/Autism"
echo "  - Metabolomics data from papers"
echo ""
echo "Estimated time: 2-6 hours"
echo "Estimated size: 10-50 GB"
echo ""
echo "You can monitor progress with:"
echo "  python scripts/download_tracker.py"
echo ""
echo "================================================================================"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# =============================================================================
# PART 1: PAPER SCRAPING (Fast, do first)
# =============================================================================
log "PART 1: Scraping published papers for supplementary data..."

python scripts/scrape_papers.py \
    --query "ADHD microbiome" \
    --query "autism microbiome" \
    --query "ADHD metabolomics" \
    --query "autism metabolomics" \
    --query "ADHD gene expression" \
    --query "autism transcriptome" \
    --max-papers 100 \
    --output data/papers/ \
    2>&1 | tee logs/downloads/paper_scraping.log &

PAPER_PID=$!

# =============================================================================
# PART 2: GEO GENE EXPRESSION (Parallel download)
# =============================================================================
log "PART 2: Downloading GEO gene expression datasets..."

# Download priority GEO datasets
GEO_DATASETS=(
    "GSE28521"  # Autism brain cortex
    "GSE28475"  # Autism temporal cortex
    "GSE64018"  # Autism brain regions
    "GSE98793"  # ADHD blood
    "GSE18123"  # Autism cells
)

for gse in "${GEO_DATASETS[@]}"; do
    log "Downloading $gse..."
    python scripts/download_geo.py \
        --dataset "$gse" \
        --output data/raw/geo/ \
        2>&1 | tee logs/downloads/${gse}.log &
done

# =============================================================================
# PART 3: SRA MICROBIOME DATA (Largest, most time)
# =============================================================================
log "PART 3: Downloading SRA microbiome datasets..."

# Use the catalog we created
if [ -f "data/catalogs/sra_study_catalog.csv" ]; then
    log "Found 350 SRA studies in catalog"
    log "Downloading top 50 most relevant (by relevance score)..."

    python scripts/download_sra_batch.py \
        --catalog data/catalogs/sra_study_catalog.csv \
        --top 50 \
        --output data/raw/sra/ \
        --threads 4 \
        2>&1 | tee logs/downloads/sra_download.log &

    SRA_PID=$!
else
    log "ERROR: SRA catalog not found!"
fi

# =============================================================================
# PART 4: METABOLOMICS
# =============================================================================
log "PART 4: Downloading metabolomics data..."

# MetaboLights (if we can fix the API)
python scripts/download_metabolights.py \
    --studies MTBLS2288 MTBLS373 MTBLS158 \
    --output data/raw/metabolights/ \
    2>&1 | tee logs/downloads/metabolights.log &

# =============================================================================
# WAIT FOR CRITICAL DOWNLOADS
# =============================================================================
log "Main downloads started. Waiting for completion..."

# Wait for paper scraping (should be fast)
wait $PAPER_PID
log "✓ Paper scraping complete"

# Check on SRA download (this is the long one)
if [ ! -z "$SRA_PID" ]; then
    log "Waiting for SRA downloads (this may take 1-4 hours)..."
    wait $SRA_PID
    log "✓ SRA downloads complete"
fi

# Wait for all background jobs
wait

log "✓ All downloads complete!"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "================================================================================"
echo "DOWNLOAD COMPLETE"
echo "================================================================================"

# Calculate sizes
SRA_SIZE=$(du -sh data/raw/sra 2>/dev/null | cut -f1 || echo "0")
GEO_SIZE=$(du -sh data/raw/geo 2>/dev/null | cut -f1 || echo "0")
PAPERS_SIZE=$(du -sh data/papers 2>/dev/null | cut -f1 || echo "0")

echo ""
echo "Downloaded:"
echo "  SRA Microbiome: $SRA_SIZE"
echo "  GEO Expression: $GEO_SIZE"
echo "  Papers + Supplements: $PAPERS_SIZE"
echo ""
echo "Next steps:"
echo "  1. Process raw data: python scripts/process_all_data.py"
echo "  2. Run etiology analysis: python scripts/analyze_etiology.py"
echo ""
echo "Completed: $(date)"
echo "================================================================================"