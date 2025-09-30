# Video Script: Quick Start Guide (5 minutes)

## Introduction (30 seconds)

"Welcome to the AuDHD Correlation Study Pipeline. In this 5-minute tutorial, we'll run a complete multi-omics analysis from data loading to results visualization."

**On screen:** Title slide with pipeline logo

## Installation (30 seconds)

"First, let's install the pipeline. Open your terminal and run:"

```bash
pip install audhd-correlation-study
```

**On screen:** Terminal showing installation

"Installation takes about 2 minutes. While we wait, let me show you what the pipeline does."

## Overview (1 minute)

"The pipeline analyzes four data types:"

**On screen:** Diagram showing data flow

1. **Genomic data** (VCF files with SNP genotypes)
2. **Clinical data** (CSV with phenotypes)
3. **Metabolomic data** (metabolite abundances)
4. **Microbiome data** (taxonomic abundances)

"It integrates these using multi-omics factor analysis, identifies patient subtypes through clustering, and generates comprehensive reports."

## Sample Data (30 seconds)

"Let's download sample data to test with:"

```bash
audhd-pipeline download-sample-data
```

**On screen:** Terminal showing download

"This creates a 'data/' directory with synthetic test data."

## Configuration (30 seconds)

"Create a simple configuration file:"

```yaml
data:
  input_dir: "data/"
  output_dir: "outputs/"

clustering:
  method: "hdbscan"
  min_cluster_size: 20
```

**On screen:** Text editor showing config.yaml

"Save this as config.yaml"

## Running Analysis (1 minute)

"Now let's run the pipeline:"

```bash
audhd-pipeline run --config config.yaml
```

**On screen:** Terminal showing progress

"The pipeline will:
1. Load and harmonize data (15 seconds)
2. Preprocess and normalize (30 seconds)
3. Integrate modalities (1 minute)
4. Identify clusters (30 seconds)
5. Generate report (15 seconds)"

**On screen:** Progress bar advancing

## Viewing Results (1 minute)

"Analysis complete! Let's look at the results."

**On screen:** File explorer showing outputs/

"The outputs directory contains:
- cluster_assignments.csv - Sample cluster labels
- report.html - Interactive HTML report
- figures/ - All visualizations"

"Open the HTML report:"

```bash
open outputs/report.html
```

**On screen:** Browser showing interactive report

"Here you can see:
- UMAP plot of patient clusters
- Cluster sizes and statistics
- Clinical variable distributions
- Validation metrics
- Top differentiating features"

## Next Steps (30 seconds)

"That's it! You've run your first multi-omics analysis.

Next steps:
- Try with your own data
- Explore advanced configurations
- Read the full documentation at docs.audhd-pipeline.org

Thanks for watching!"

**On screen:** Links to resources

---

## Technical Notes for Video Production

**Duration:** 5 minutes

**Required assets:**
- Terminal recordings (use asciinema)
- Screenshots of outputs
- Animated data flow diagram
- Sample report screenshots

**Editing notes:**
- Speed up installation (time-lapse)
- Add captions for all commands
- Highlight key outputs with zoom/arrows
- Use smooth transitions between sections

**Audio:**
- Clear voiceover
- Optional: Background music (low volume)
- Sound effects for completion milestones