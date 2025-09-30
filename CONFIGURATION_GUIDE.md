# AuDHD Correlation Study - Configuration Guide

This document provides a comprehensive guide to the hierarchical configuration system implemented for the AuDHD Correlation Study pipeline.

## Overview

The configuration system uses **Hydra** for flexible, composable configuration management with the following features:

- ✅ **Hierarchical structure** with config groups
- ✅ **Parameter sweeps** for hyperparameter optimization
- ✅ **Validation schemas** with Pydantic
- ✅ **Environment-specific settings** (local, HPC, cloud)
- ✅ **Config composition and overrides** at runtime

## Quick Start

### Run with Default Configuration

```bash
python scripts/run_pipeline.py
```

### Override Config Groups

```bash
# Use minimal data and preprocessing
python scripts/run_pipeline.py data=minimal preprocessing=minimal

# Use K-means instead of HDBSCAN
python scripts/run_pipeline.py clustering=kmeans

# Use HPC compute settings
python scripts/run_pipeline.py compute=hpc
```

### Override Individual Parameters

```bash
# Change specific parameters
python scripts/run_pipeline.py \
  clustering.params.min_cluster_size=25 \
  integration.params.n_factors=20

# Enable dry run
python scripts/run_pipeline.py pipeline.dry_run=true
```

## Configuration Structure

```
configs/
├── config.yaml                 # Main config with defaults
├── data/                       # Data loading
│   ├── base.yaml              # All modalities
│   └── minimal.yaml           # Clinical + genomic only
├── preprocessing/             # Preprocessing
│   ├── standard.yaml          # Full preprocessing
│   └── minimal.yaml           # Fast (for testing)
├── integration/               # Multi-omics integration
│   ├── mofa.yaml              # MOFA (default)
│   └── pca.yaml               # Simple PCA
├── clustering/                # Clustering methods
│   ├── hdbscan.yaml           # HDBSCAN (default)
│   └── kmeans.yaml            # K-means
├── validation/                # Validation
│   ├── comprehensive.yaml     # Full validation
│   └── quick.yaml             # Fast validation
├── biological/                # Biological analysis
│   ├── full.yaml              # Complete analysis
│   └── minimal.yaml           # GSEA only
├── visualization/             # Visualization
│   ├── standard.yaml          # Standard plots
│   └── publication.yaml       # High-quality
├── reporting/                 # Reports
│   ├── research.yaml          # Research report
│   └── clinical.yaml          # Clinical report
├── compute/                   # Compute environments
│   ├── local.yaml             # Local machine
│   ├── hpc.yaml               # HPC cluster
│   └── cloud.yaml             # Cloud (AWS/GCP)
├── experiment/                # Experiment presets
│   ├── default.yaml           # Baseline
│   ├── quick_test.yaml        # Fast testing
│   └── high_resolution.yaml   # Fine-grained
└── sweep/                     # Parameter sweeps
    ├── clustering_optimization.yaml
    ├── integration_optimization.yaml
    └── optuna_clustering.yaml
```

## Configuration Groups

### Data (`data=...`)

- **base**: All modalities (genomic, clinical, metabolomic, microbiome)
- **minimal**: Clinical + genomic only (faster)

### Preprocessing (`preprocessing=...`)

- **standard**: Full preprocessing with batch correction
- **minimal**: Fast preprocessing for testing

### Integration (`integration=...`)

- **mofa**: MOFA integration (default, 15 factors)
- **pca**: Simple PCA (faster, 50 components)

### Clustering (`clustering=...`)

- **hdbscan**: HDBSCAN clustering (default)
- **kmeans**: K-means clustering (requires n_clusters)

### Validation (`validation=...`)

- **comprehensive**: Full validation suite (100 bootstraps)
- **quick**: Fast validation (10 bootstraps)

### Compute (`compute=...`)

- **local**: Local machine (32GB RAM, 8 CPUs)
- **hpc**: HPC cluster (256GB RAM, 64 CPUs)
- **cloud**: Cloud computing (configurable)

### Experiment (`experiment=...`)

- **default**: Baseline analysis
- **quick_test**: Fast parameters for development
- **high_resolution**: Sensitive parameters for publication

## Examples

### 1. Quick Test Run

```bash
python scripts/run_pipeline.py \
  experiment=quick_test \
  pipeline.dry_run=true
```

### 2. High-Resolution Analysis

```bash
python scripts/run_pipeline.py \
  experiment=high_resolution \
  compute=hpc
```

### 3. Minimal Pipeline (for testing)

```bash
python scripts/run_pipeline.py \
  data=minimal \
  preprocessing=minimal \
  integration=pca \
  validation=quick
```

### 4. Custom Parameters

```bash
python scripts/run_pipeline.py \
  clustering.params.min_cluster_size=20,25,30 \
  --multirun
```

## Parameter Sweeps

### Grid Search

Run multiple parameter combinations:

```bash
python scripts/run_pipeline.py \
  --multirun \
  clustering.params.min_cluster_size=20,25,30 \
  clustering.params.min_samples=5,10,15
```

This creates 9 runs (3 × 3 combinations).

### Using Sweep Configs

```bash
# Clustering optimization (1152 combinations)
python scripts/run_pipeline.py \
  --config-name=sweep/clustering_optimization \
  --multirun

# Integration optimization (72 combinations)
python scripts/run_pipeline.py \
  --config-name=sweep/integration_optimization \
  --multirun
```

### Bayesian Optimization (Optuna)

Install: `pip install hydra-optuna-sweeper`

```bash
python scripts/run_pipeline.py \
  --config-name=sweep/optuna_clustering \
  --multirun
```

This uses Bayesian optimization for smarter parameter search (100 trials).

## Configuration Validation

### Validate Config Files

```bash
# Validate default config
python scripts/validate_config.py

# Validate specific config
python scripts/validate_config.py configs/config.yaml

# Check resource availability
python scripts/validate_config.py --check-resources

# Check data paths
python scripts/validate_config.py --check-data-paths

# Validate all configs
python scripts/validate_config.py --all-configs
```

### Validation Features

- ✅ Type checking (int, float, str, bool)
- ✅ Value constraints (min/max, choices)
- ✅ Cross-field validation
- ✅ Resource availability checks
- ✅ Path existence verification

## Environment-Specific Settings

### Local Development

```yaml
# configs/compute/local.yaml
resources:
  max_memory_gb: 32
  max_cpus: 8
  use_gpu: false

parallelization:
  n_jobs: 4
  backend: "loky"
```

### HPC Cluster

```yaml
# configs/compute/hpc.yaml
resources:
  max_memory_gb: 256
  max_cpus: 64
  nodes: 1

slurm:
  partition: "general"
  time_limit: "24:00:00"
```

### Cloud Computing

```yaml
# configs/compute/cloud.yaml
resources:
  max_memory_gb: 128
  max_cpus: 32
  instance_type: "m5.8xlarge"

cloud:
  provider: "aws"
  region: "us-east-1"
```

## Advanced Features

### 1. Config Interpolation

Reference other config values:

```yaml
pipeline:
  output_dir: "${hydra:runtime.output_dir}/results"
  checkpoint_dir: "${hydra:runtime.output_dir}/checkpoints"
```

### 2. Environment Variables

```yaml
compute:
  cache_dir: "/scratch/${oc.env:USER}/.cache"
```

### 3. Conditional Configuration

```yaml
preprocessing:
  batch_correction: ${oc.select:data.clinical.params.batch_column,false}
```

### 4. See Resolved Config

```bash
python scripts/run_pipeline.py --cfg job
```

### 5. See Available Options

```bash
python scripts/run_pipeline.py --help
```

## Output Directory Structure

Hydra automatically creates timestamped output directories:

```
outputs/audhd_correlation_pipeline/2025-09-29/23-15-30/
├── .hydra/
│   ├── config.yaml          # Resolved config
│   ├── hydra.yaml
│   └── overrides.yaml       # Applied overrides
├── checkpoints/
├── results/
├── logs/
└── audit/
```

This ensures:
- ✅ Each run is isolated
- ✅ Full reproducibility
- ✅ Easy comparison of runs

## Best Practices

### 1. Use Config Groups for Related Settings

Instead of overriding many parameters:

```bash
# ❌ Don't do this
python scripts/run_pipeline.py \
  clustering.params.min_cluster_size=25 \
  clustering.params.min_samples=8 \
  clustering.params.umap_n_neighbors=20
```

Create a config group:

```yaml
# configs/clustering/my_settings.yaml
method: "hdbscan"
params:
  min_cluster_size: 25
  min_samples: 8
  umap_n_neighbors: 20
```

```bash
# ✅ Do this
python scripts/run_pipeline.py clustering=my_settings
```

### 2. Use Experiment Configs

For reproducible experiments:

```yaml
# configs/experiment/paper_v1.yaml
name: "paper_v1"
description: "Settings for paper submission"

dataset_overrides:
  clustering:
    params:
      min_cluster_size: 25
  validation:
    n_bootstrap: 500
```

```bash
python scripts/run_pipeline.py experiment=paper_v1
```

### 3. Version Control Your Configs

```bash
git add configs/experiment/paper_v1.yaml
git commit -m "Add configuration for paper v1"
```

### 4. Document Your Changes

Add comments to config files:

```yaml
# configs/experiment/my_analysis.yaml
# Custom analysis for reviewer comments
# Uses more stringent QC and higher resolution clustering
# Author: Your Name
# Date: 2025-09-29
```

## Troubleshooting

### Config Override Not Working

Ensure correct syntax:

```bash
# ✅ Correct
python scripts/run_pipeline.py clustering.params.min_cluster_size=25

# ❌ Incorrect
python scripts/run_pipeline.py clustering.params.min_cluster_size 25
```

### Missing Config Group

Check:
1. File exists in correct subdirectory
2. Filename matches group name (without .yaml)
3. `@package` directive is correct

### Validation Errors

Run validator to see specific errors:

```bash
python scripts/validate_config.py --verbose
```

## Testing the Configuration System

Run the test suite:

```bash
bash scripts/test_config_system.sh
```

This tests:
- Default configuration
- Config group overrides
- Parameter overrides
- Environment configs
- Experiment presets
- Dry run mode

## Further Reading

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Hydra Patterns](https://hydra.cc/docs/patterns/configuring_experiments/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [configs/README.md](configs/README.md) - Detailed config system docs

## Summary

The configuration system provides:

✅ **Flexibility**: Override any parameter at runtime
✅ **Reproducibility**: Every run saved with full config
✅ **Validation**: Type-checked and constraint-validated
✅ **Organization**: Clear hierarchical structure
✅ **Optimization**: Built-in parameter sweep support
✅ **Portability**: Environment-specific settings

This enables easy experimentation, reproducible research, and production deployment.