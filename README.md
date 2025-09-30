# Multi-Omics Integration Pipeline for ADHD/Autism Subtyping

A comprehensive pipeline for discovering biologically distinct, clinically actionable subtypes within ADHD and autism diagnoses through integrated analysis of genetics, metabolomics, microbiome, neuroimaging, and clinical data.

## Overview

This project implements a rigorous multi-omics integration pipeline to test whether diagnostic labels "ADHD" and "autism" decompose into distinct biological subtypes with different molecular mechanisms, treatment responses, and clinical trajectories.

### Key Features

- **Multi-omics integration**: Genetics (WGS/WES, PRS, CNV), metabolomics (~300 features), clinical phenotypes, microbiome, neuroimaging/EEG
- **Robust methodology**: Comprehensive confound control, sensitivity analyses, causal inference (MR, mediation, G×E)
- **Advanced clustering**: HDBSCAN, consensus clustering, topological data analysis
- **Clinical translation**: Subtype-specific care maps, risk stratification, treatment response prediction
- **Reproducible**: Hydra configuration, Docker support, pre-registered analyses

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/rohanvinaik/AuDHD_Correlation_Study.git
cd AuDHD_Correlation_Study

# Create conda environment
conda env create -f env/environment.yml
conda activate audhd-correlation

# Install package
make install
```

### Basic Usage

```bash
# Run full pipeline
audhd-omics pipeline configs/defaults.yaml

# Run full pipeline with specific steps
audhd-omics pipeline configs/defaults.yaml --steps download,build_features,integrate

# Or run individual steps
audhd-omics download configs/defaults.yaml
audhd-omics build-features configs/defaults.yaml
audhd-omics integrate configs/defaults.yaml
audhd-omics cluster configs/defaults.yaml
audhd-omics validate configs/defaults.yaml
audhd-omics report configs/defaults.yaml

# With custom config
audhd-omics cluster configs/my_custom_config.yaml
```

### Docker

```bash
# Build image
make docker-build

# Run container
make docker-run
```

## Project Structure

```
├── configs/              # Hydra-style composable configs
│   ├── data/            # Dataset configs (SPARK, SSC, ABCD, UKB)
│   ├── features/        # Feature selection configs
│   ├── preprocess/      # QC, imputation, batch correction
│   ├── integrate/       # Multi-omics integration (MOFA2, DIABLO)
│   ├── cluster/         # Clustering methods (HDBSCAN, LCA)
│   ├── validate/        # Validation strategies
│   ├── causal/          # Causal inference (MR, mediation, G×E)
│   └── viz/             # Visualization settings
├── data/
│   ├── raw/             # Original datasets (gitignored)
│   ├── interim/         # Harmonized data
│   ├── processed/       # Model-ready matrices
│   └── external/        # Reference data (ontologies, pathways)
├── src/audhd_correlation/
│   ├── data/            # Data loading and harmonization
│   ├── features/        # Feature engineering
│   ├── preprocess/      # QC and preprocessing
│   ├── integrate/       # Multi-omics integration
│   ├── modeling/        # Clustering and topology
│   ├── validation/      # Internal/external validation
│   └── reporting/       # Report generation
├── scripts/             # Pipeline scripts
├── notebooks/           # Exploratory analysis
└── tests/               # Unit tests
```

## Pipeline Overview

### Phase 0: Ethics & Data Access (Day 0)
- DUAs for SPARK, SSC, ABCD, UK Biobank
- IRB approvals
- Pre-registration of hypotheses

### Phase 1-2: Data Acquisition & Preprocessing (Days 1-4)
- Download datasets from SFARI, ABCD, UK Biobank
- Harmonize across sites (ComBat, site random effects)
- QC: genetics (VQSR, call rate), metabolomics (CV, drift correction)
- Imputation: delta-adjusted MICE with missingness indicators
- Context adjustment: fasting, time-of-day, medication timing

### Phase 3: Integration & Clustering (Day 5)
- Multi-omics integration (MOFA2, weighted stack)
- Dimensionality reduction (UMAP, t-SNE)
- Consensus clustering (HDBSCAN + spectral)
- Topology analysis (persistence diagrams, MST gaps)

### Phase 4-5: Validation (Days 6-7)
- Internal: silhouette, stability, biological validity
- External: leave-site-out CV, cross-ancestry replication
- Gap analysis: test separation vs spectrum hypothesis

### Phase 6-7: Causal Analysis (Days 8-10)
- Mendelian randomization (PRS as instruments)
- Mediation analysis (genetics → metabolites → symptoms)
- G×E interactions (prenatal exposures, medications)
- Sensitivity analyses (medication washout, MNAR, E-values)

### Phase 8: Reporting (Days 11-12)
- Executive summary
- Clinical decision support tables
- Subtype-specific care maps
- Reproducibility bundle

## Expected Subtypes

1. **Neurotransmitter-Serotonin**: Low serotonin/5-HIAA, anxiety, GI → SSRI responsive
2. **Neurotransmitter-Dopamine**: Low dopamine metabolites, ADHD-H → stimulant responsive
3. **Neurodevelopmental**: Rare de novo variants, early severe → specialized interventions
4. **Immune-Inflammatory**: Elevated cytokines, autoimmune → anti-inflammatory
5. **Mitochondrial**: Energy metabolism dysfunction → supplements
6. **Gut-Brain**: Microbiome dysbiosis, severe GI → dietary/probiotic

## Configuration

Configuration uses Hydra for composability:

```yaml
# configs/defaults.yaml
defaults:
  - data: spark
  - features: full
  - preprocess: standard
  - integrate: mofa2
  - cluster: hdbscan

seed: 42
n_jobs: 4
```

Override via Hydra config composition:
```bash
# Use different config files
audhd-omics pipeline configs/defaults.yaml

# Create custom config that overrides specific values
# configs/my_config.yaml can override defaults
audhd-omics pipeline configs/my_config.yaml
```

## Development

```bash
# Install dev dependencies
make dev-install

# Run tests
make test

# Lint and format
make format
make lint

# Type checking
make typecheck
```

## Data Access

This project uses publicly available datasets requiring Data Use Agreements:

- **SPARK**: https://base.sfari.org
- **SSC**: https://base.sfari.org
- **ABCD**: https://abcdstudy.org
- **UK Biobank**: https://www.ukbiobank.ac.uk

See `.env.example` for required credentials.

## Citation

If you use this pipeline, please cite:

```bibtex
@software{vinaik2024audhd,
  author = {Vinaik, Rohan},
  title = {Multi-Omics Integration Pipeline for ADHD/Autism Subtyping},
  year = {2024},
  url = {https://github.com/rohanvinaik/AuDHD_Correlation_Study}
}
```

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- **Author**: Rohan Vinaik
- **Email**: your.email@example.com
- **Issues**: https://github.com/rohanvinaik/AuDHD_Correlation_Study/issues

## Acknowledgments

- SFARI for SPARK and SSC datasets
- ABCD Study consortium
- UK Biobank
- All contributing researchers and families