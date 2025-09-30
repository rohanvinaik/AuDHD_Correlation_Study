# Configuration System

This directory contains the hierarchical configuration system for the AuDHD Correlation Study pipeline. The system uses [Hydra](https://hydra.cc/) for flexible configuration management with composition and overrides.

## Directory Structure

```
configs/
├── config.yaml              # Main configuration with defaults
├── pipeline.yaml            # Legacy config (for compatibility)
├── data/                    # Data loading configurations
│   ├── base.yaml           # Standard data config
│   └── minimal.yaml        # Minimal data (testing)
├── preprocessing/           # Preprocessing configurations
│   ├── standard.yaml       # Standard preprocessing
│   └── minimal.yaml        # Fast preprocessing (testing)
├── integration/             # Multi-omics integration
│   ├── mofa.yaml           # MOFA integration (default)
│   └── pca.yaml            # Simple PCA (fast)
├── clustering/              # Clustering methods
│   ├── hdbscan.yaml        # HDBSCAN (default)
│   └── kmeans.yaml         # K-means (comparison)
├── validation/              # Validation strategies
│   ├── comprehensive.yaml  # Full validation suite
│   └── quick.yaml          # Fast validation (testing)
├── biological/              # Biological analysis
│   ├── full.yaml           # Complete analysis
│   └── minimal.yaml        # GSEA only
├── visualization/           # Visualization settings
│   ├── standard.yaml       # Standard plots
│   └── publication.yaml    # Publication-quality
├── reporting/               # Report generation
│   ├── research.yaml       # Research report
│   └── clinical.yaml       # Clinical report
├── compute/                 # Compute environments
│   ├── local.yaml          # Local machine
│   ├── hpc.yaml            # HPC cluster
│   └── cloud.yaml          # Cloud computing
├── experiment/              # Experiment configs
│   ├── default.yaml        # Baseline experiment
│   ├── quick_test.yaml     # Fast testing
│   └── high_resolution.yaml # Fine-grained analysis
└── sweep/                   # Parameter sweeps
    ├── clustering_optimization.yaml
    ├── integration_optimization.yaml
    ├── preprocessing_optimization.yaml
    └── optuna_clustering.yaml
```

## Basic Usage

### Running with Default Configuration

```bash
python scripts/run_pipeline.py
```

This uses the default configuration specified in `config.yaml`.

### Using a Different Preset

Override individual config groups:

```bash
# Use minimal data and fast preprocessing
python scripts/run_pipeline.py data=minimal preprocessing=minimal

# Use K-means clustering instead of HDBSCAN
python scripts/run_pipeline.py clustering=kmeans

# Use HPC compute settings
python scripts/run_pipeline.py compute=hpc
```

### Overriding Individual Parameters

```bash
# Change number of factors
python scripts/run_pipeline.py integration.params.n_factors=20

# Change cluster size
python scripts/run_pipeline.py clustering.params.min_cluster_size=25

# Enable dry run
python scripts/run_pipeline.py pipeline.dry_run=true
```

### Combining Multiple Overrides

```bash
python scripts/run_pipeline.py \
  data=minimal \
  preprocessing=minimal \
  validation=quick \
  clustering.params.min_cluster_size=20 \
  integration.params.n_factors=10
```

## Configuration Composition

Configurations are composed using Hydra's defaults list. The main `config.yaml` specifies:

```yaml
defaults:
  - data: base
  - preprocessing: standard
  - integration: mofa
  - clustering: hdbscan
  - validation: comprehensive
  - biological: full
  - visualization: standard
  - reporting: research
  - compute: local
  - experiment: default
  - _self_
```

Each default can be overridden at runtime.

## Experiment Configurations

Experiment configs define complete analysis setups:

### Quick Test
```bash
python scripts/run_pipeline.py experiment=quick_test
```
Fast configuration for testing with minimal parameters.

### High Resolution
```bash
python scripts/run_pipeline.py experiment=high_resolution
```
Sensitive parameters for fine-grained analysis.

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
python scripts/run_pipeline.py \
  --config-name=sweep/clustering_optimization \
  --multirun
```

### Bayesian Optimization with Optuna

Requires: `pip install hydra-optuna-sweeper`

```bash
python scripts/run_pipeline.py \
  --config-name=sweep/optuna_clustering \
  --multirun
```

This uses Bayesian optimization to find optimal parameters with fewer trials.

## Environment-Specific Settings

### Local Development

```bash
python scripts/run_pipeline.py compute=local
```

- 32GB RAM, 8 CPUs
- Uses local cache
- Standard parallelization

### HPC Cluster

```bash
python scripts/run_pipeline.py compute=hpc
```

- 256GB RAM, 64 CPUs
- Uses /scratch for cache
- Distributed processing
- SLURM integration

### Cloud Computing

```bash
python scripts/run_pipeline.py compute=cloud
```

- Configurable instance types
- Auto-scaling support
- Cloud storage integration

## Validation

Validate your configuration before running:

```bash
# Validate default config
python scripts/validate_config.py

# Validate specific config
python scripts/validate_config.py configs/config.yaml

# Check resource availability
python scripts/validate_config.py --check-resources

# Check data paths exist
python scripts/validate_config.py --check-data-paths

# Validate all configs
python scripts/validate_config.py --all-configs --verbose
```

## Configuration Schema

All configurations are validated using Pydantic schemas defined in `src/audhd_correlation/config/validation.py`.

Key validation features:
- Type checking (int, float, str, bool)
- Value constraints (min/max, choices)
- Cross-field validation
- Resource availability checks
- Path existence verification

## Best Practices

### 1. Use Config Groups for Related Settings

Don't override many individual parameters. Instead, create a new config group:

```yaml
# configs/clustering/my_experiment.yaml
# @package _global_.clustering

method: "hdbscan"
compute_topology: true
params:
  min_cluster_size: 25
  min_samples: 8
  # ... other params
```

Then use: `python scripts/run_pipeline.py clustering=my_experiment`

### 2. Create Experiment Configs for Complete Setups

For reproducible experiments, create a complete experiment config:

```yaml
# configs/experiment/my_analysis.yaml
# @package _global_.experiment

name: "my_analysis"
description: "Custom analysis with specific parameters"
tags: ["custom", "experiment"]

dataset_overrides:
  data:
    genomic:
      params:
        maf_threshold: 0.005

  clustering:
    params:
      min_cluster_size: 25
```

### 3. Version Control Your Configs

Commit your config files to track parameter changes:

```bash
git add configs/experiment/my_analysis.yaml
git commit -m "Add configuration for my_analysis experiment"
```

### 4. Use Hydra's Output Directory

Hydra automatically creates timestamped output directories:

```
outputs/audhd_correlation_pipeline/2025-09-29/22-30-45/
├── .hydra/
│   ├── config.yaml          # Resolved config
│   ├── hydra.yaml
│   └── overrides.yaml
├── checkpoints/
├── results/
└── logs/
```

This ensures each run is isolated and reproducible.

### 5. Document Your Configurations

Add descriptions to your config files:

```yaml
# configs/experiment/my_analysis.yaml
# Custom analysis for paper revision
# Uses more stringent QC and higher resolution clustering
# Author: Your Name
# Date: 2025-09-29
```

## Advanced Features

### Interpolation

Reference other config values:

```yaml
pipeline:
  output_dir: "${hydra:runtime.output_dir}/results"
  checkpoint_dir: "${hydra:runtime.output_dir}/checkpoints"
```

### Environment Variables

```yaml
compute:
  cache:
    cache_dir: "/scratch/${oc.env:USER}/.cache"
```

### Conditional Configuration

```yaml
preprocessing:
  batch_correction: ${oc.select:data.clinical.params.batch_column,false}
```

### Config Resolution

See the resolved configuration:

```bash
python scripts/run_pipeline.py --cfg job
```

## Troubleshooting

### Config Override Not Working

Ensure you're using the correct syntax:

```bash
# Correct
python scripts/run_pipeline.py clustering.params.min_cluster_size=25

# Incorrect (will fail)
python scripts/run_pipeline.py clustering.params.min_cluster_size 25
```

### Missing Config Group

If you get "Could not load config group", check:
1. File exists in correct subdirectory
2. Filename matches group name
3. `@package` directive is correct

### Validation Errors

Run validation to see specific errors:

```bash
python scripts/validate_config.py --verbose
```

## Further Reading

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Hydra Patterns](https://hydra.cc/docs/patterns/configuring_experiments/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)