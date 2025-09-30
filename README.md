# AuDHD Correlation Study Pipeline

[![Tests](https://github.com/rohanvinaik/AuDHD_Correlation_Study/workflows/Tests/badge.svg)](https://github.com/rohanvinaik/AuDHD_Correlation_Study/actions)
[![Documentation](https://readthedocs.org/projects/audhd-pipeline/badge/?version=latest)](https://audhd-pipeline.readthedocs.io/en/latest/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready pipeline for discovering biologically distinct patient subtypes through integrated multi-omics analysis. Designed for ADHD/Autism research but applicable to any multi-omics clustering study.

## 🎯 Key Features

- **Multi-Omics Integration**: Genomic (VCF), clinical, metabolomic, and microbiome data
- **Advanced Clustering**: HDBSCAN, K-means, hierarchical clustering with automatic parameter selection
- **Statistical Validation**: Bootstrap stability, silhouette analysis, permutation tests
- **Biological Interpretation**: Pathway enrichment, cluster signatures, network analysis
- **Production-Ready**: Comprehensive testing, CI/CD, Docker support, extensive documentation
- **Reproducible**: Hydra configuration, checkpointing, version control, audit logging

## 📚 Documentation

**Full documentation available at:** [audhd-pipeline.readthedocs.io](https://audhd-pipeline.readthedocs.io)

- **[Quick Start Guide](https://audhd-pipeline.readthedocs.io/quickstart.html)** - Get started in 5 minutes
- **[User Guides](https://audhd-pipeline.readthedocs.io/user_guide/)** - Detailed guides for each pipeline phase
- **[API Reference](https://audhd-pipeline.readthedocs.io/api/)** - Complete API documentation
- **[Tutorials](https://audhd-pipeline.readthedocs.io/tutorials/)** - Jupyter notebook tutorials
- **[FAQ](https://audhd-pipeline.readthedocs.io/faq.html)** - Frequently asked questions
- **[Troubleshooting](https://audhd-pipeline.readthedocs.io/troubleshooting.html)** - Common issues and solutions

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/rohanvinaik/AuDHD_Correlation_Study.git
cd AuDHD_Correlation_Study

# Install with pip
pip install -e .

# Or with conda
conda env create -f env/environment.yml
conda activate audhd-study
```

### Basic Usage

**Command Line:**

```bash
# Run complete pipeline
audhd-pipeline run --config config.yaml

# Or run individual stages
audhd-pipeline preprocess --config config.yaml
audhd-pipeline cluster --config config.yaml
audhd-pipeline validate --config config.yaml
```

**Python API:**

```python
from audhd_correlation import Pipeline

# Create and run pipeline
pipeline = Pipeline(config_path="config.yaml")
results = pipeline.run()

# Generate report
pipeline.generate_report(results, output_path="report.html")
```

**Sample Data:**

```bash
# Download sample data to test
audhd-pipeline download-sample-data

# Run on sample data
audhd-pipeline run --config configs/sample_analysis.yaml
```

See the **[Quick Start Guide](docs/quickstart.rst)** for a complete walkthrough.

## 📊 Pipeline Overview

```
Data Loading → Preprocessing → Integration → Clustering → Validation → Interpretation
     ↓              ↓               ↓            ↓             ↓              ↓
  VCF/CSV      Imputation      MOFA/PCA     HDBSCAN      Silhouette    Pathways
  Harmonize    Scaling         Factors      K-means      Stability     Networks
  QC Filter    Batch Fix       15-20 dim    UMAP         Bootstrap     Signatures
```

### Supported Data Types

| Modality | Format | Features |
|----------|--------|----------|
| **Genomic** | VCF v4.1/4.2 | SNP genotypes, quality filtering, MAF filtering |
| **Clinical** | CSV | Phenotypes, demographics, diagnosis, severity scores |
| **Metabolomic** | CSV/TSV/Excel | Metabolite abundances, log-transform, quantile normalization |
| **Microbiome** | TSV/BIOM | Taxonomic abundances, CLR transform, prevalence filtering |

### Integration Methods

- **MOFA** (Multi-Omics Factor Analysis) - Identifies shared and modality-specific variation
- **PCA** - Fast concatenated PCA for quick analysis
- **CCA** - Canonical Correlation Analysis for two modalities
- **NMF** - Non-negative Matrix Factorization for count data

### Clustering Methods

- **HDBSCAN** - Density-based, auto-determines k, handles noise points
- **K-means** - Fast, fixed k, spherical clusters
- **Hierarchical** - Dendrogram-based, multiple linkage methods
- **Gaussian Mixture** - Probabilistic clustering with soft assignments

## 🏗️ Project Structure

```
AuDHD_Correlation_Study/
├── src/audhd_correlation/      # Main package
│   ├── data/                   # Data loaders and harmonization
│   ├── preprocess/             # Preprocessing and normalization
│   ├── integrate/              # Multi-omics integration
│   ├── modeling/               # Clustering algorithms
│   ├── validation/             # Validation metrics
│   ├── biological/             # Pathway enrichment
│   ├── viz/                    # Visualization
│   └── reporting/              # Report generation
├── tests/                      # Comprehensive test suite
│   ├── unit/                   # Unit tests (280+ tests)
│   ├── integration/            # Integration tests
│   ├── statistical/            # Statistical validation tests
│   ├── benchmarks/             # Performance benchmarks
│   └── property/               # Property-based tests (Hypothesis)
├── docs/                       # Sphinx documentation
│   ├── user_guide/             # User guides for each phase
│   ├── api/                    # Auto-generated API docs
│   ├── tutorials/              # Tutorial documents
│   ├── data_dictionaries/      # Data format specifications
│   └── video_scripts/          # Video tutorial scripts
├── notebooks/                  # Jupyter tutorial notebooks
├── configs/                    # Hydra configuration files
│   ├── data/                   # Dataset configs
│   ├── preprocess/             # Preprocessing configs
│   ├── integrate/              # Integration configs
│   ├── cluster/                # Clustering configs
│   └── defaults.yaml           # Default configuration
├── scripts/                    # Utility scripts
├── data/                       # Data directory (gitignored)
│   ├── raw/                    # Original datasets
│   ├── interim/                # Intermediate files
│   └── processed/              # Final processed data
└── outputs/                    # Analysis outputs
    ├── preprocessed/           # Preprocessed data
    ├── integrated/             # Integration results
    ├── clusters/               # Cluster assignments
    ├── validation/             # Validation metrics
    ├── figures/                # Generated plots
    └── report.html             # Final HTML report
```

## 🧪 Testing

Comprehensive test suite with 500+ tests:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/statistical/       # Statistical tests
pytest tests/benchmarks/        # Performance benchmarks

# Run with coverage
pytest --cov=src/audhd_correlation --cov-report=html
```

**Test Coverage:**

- ✅ Unit tests for all data loaders
- ✅ Integration tests for pipeline stages
- ✅ Statistical tests for validation metrics
- ✅ Performance benchmarks for scalability
- ✅ Property-based tests with Hypothesis
- ✅ Regression tests for key metrics

## 📈 Example Results

### Cluster Visualization

```python
from audhd_correlation.viz import plot_embedding

# UMAP embedding with cluster labels
plot_embedding(
    embedding,
    labels,
    output_path='figures/clusters.png'
)
```

### Validation Metrics

```python
from audhd_correlation.validation import validate_clusters

validation = validate_clusters(
    integrated_data,
    labels,
    n_bootstrap=100
)

print(f"Silhouette: {validation['silhouette']:.3f}")
print(f"Stability (ARI): {validation['stability_ari']:.3f}")
# Output:
# Silhouette: 0.562
# Stability (ARI): 0.738
```

### Cluster Characterization

```python
from audhd_correlation.biological import compute_cluster_signatures

signatures = compute_cluster_signatures(
    preprocessed_data,
    labels,
    method='limma'
)

# Top differentiating features per cluster
for cluster_id, features in signatures.items():
    print(f"Cluster {cluster_id}: {len(features)} signatures")
```

## ⚙️ Configuration

Configuration uses YAML format with Hydra for composability:

**Basic Configuration:**

```yaml
# config.yaml
data:
  input_dir: "data/"
  output_dir: "outputs/"

processing:
  modalities:
    - genomic
    - clinical
    - metabolomic
  impute_method: "knn"
  scale_method: "standard"

integration:
  method: "mofa"
  n_factors: 15

clustering:
  method: "hdbscan"
  min_cluster_size: 20
  embedding_method: "umap"

validation:
  n_bootstrap: 100
  compute_stability: true
```

See **[Configuration Reference](docs/configuration.rst)** for all options.

## 🐳 Docker Support

```bash
# Build Docker image
docker build -t audhd-pipeline .

# Run container
docker run -v $(pwd)/data:/data -v $(pwd)/outputs:/outputs audhd-pipeline
```

## 📊 Expected Subtypes (ASD/ADHD Context)

Based on multi-omics integration, we expect to identify:

1. **Neurotransmitter-Serotonin**: Low serotonin metabolites, anxiety profile → SSRI responsive
2. **Neurotransmitter-Dopamine**: Low dopamine metabolites, ADHD-H profile → stimulant responsive
3. **Immune-Inflammatory**: Elevated cytokines, autoimmune markers → anti-inflammatory interventions
4. **Metabolic-Mitochondrial**: Energy metabolism dysfunction → metabolic supplements
5. **Gut-Brain Axis**: Microbiome dysbiosis, GI symptoms → dietary/probiotic interventions
6. **Neurodevelopmental**: Rare genetic variants, severe early onset → specialized interventions

## 🔬 Data Requirements

**Minimum Sample Size:**
- Total: ≥ 50 samples (100+ recommended)
- Per cluster: ≥ 20 samples

**Data Quality:**
- Missing rate: < 30% per feature
- Sample call rate: > 90%
- At least 2 modalities (multi-omics preferred)

**File Formats:**
- Genomic: VCF v4.1 or v4.2
- Clinical: CSV with required columns (sample_id, age, sex, diagnosis)
- Metabolomic: CSV/TSV/Excel
- Microbiome: TSV or BIOM format

See **[Data Dictionaries](docs/data_dictionaries/)** for complete specifications.

## 🤝 Contributing

Contributions are welcome! Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines.

**Development Setup:**

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see **[LICENSE](LICENSE)** for details.

## 📞 Contact & Support

- **Documentation**: https://audhd-pipeline.readthedocs.io
- **Issues**: https://github.com/rohanvinaik/AuDHD_Correlation_Study/issues
- **Discussions**: https://github.com/rohanvinaik/AuDHD_Correlation_Study/discussions
- **Author**: Rohan Vinaik

## 📖 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{vinaik2024audhd,
  author = {Vinaik, Rohan},
  title = {AuDHD Correlation Study Pipeline: Multi-Omics Integration for Patient Subtyping},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/rohanvinaik/AuDHD_Correlation_Study},
  version = {0.1.0}
}
```

## 🙏 Acknowledgments

- **Testing Framework**: pytest, hypothesis, pytest-benchmark
- **Documentation**: Sphinx, Read the Docs
- **Multi-Omics Methods**: MOFA, scikit-learn, UMAP
- **Statistical Analysis**: scipy, statsmodels, pingouin
- **Visualization**: matplotlib, seaborn, plotly

## 🎓 Learn More

- **[Quick Start Guide](docs/quickstart.rst)** - Get started in 5 minutes
- **[Complete Workflow Tutorial](notebooks/01_complete_workflow.ipynb)** - Jupyter notebook walkthrough
- **[User Guides](docs/user_guide/)** - In-depth guides for each pipeline phase:
  - [Data Loading](docs/user_guide/data_loading.rst)
  - [Preprocessing](docs/user_guide/preprocessing.rst)
  - [Integration](docs/user_guide/integration.rst)
  - [Clustering](docs/user_guide/clustering.rst)
  - [Validation](docs/user_guide/validation.rst)
  - [Biological Analysis](docs/user_guide/biological_analysis.rst)
  - [Visualization](docs/user_guide/visualization.rst)
- **[API Reference](docs/api/)** - Complete API documentation
- **[FAQ](docs/faq.rst)** - Frequently asked questions
- **[Best Practices](docs/best_practices.rst)** - Guidelines for optimal use
- **[Troubleshooting](docs/troubleshooting.rst)** - Solutions to common issues

---

**Status**: ✅ Production-Ready | **Version**: 0.1.0 | **Python**: 3.9+ | **Last Updated**: 2024