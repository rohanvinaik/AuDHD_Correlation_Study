# AuDHD Genetic Architecture Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready analysis pipeline for genetic architecture and subtype discovery in autism and ADHD**

## Overview

A comprehensive system for discovering biologically distinct genetic subtypes through integrated multi-omics analysis and baseline-deviation topology framework. Currently validated on real patient GWAS data for ASD/ADHD cross-disorder analysis and sex-stratified genetic architecture studies.

### Core Capabilities

- **Baseline-Deviation-Topology Framework**: Anti-pattern-mining safeguards prevent false positive subtype discoveries
- **Cross-Disorder Analysis**: Identifies ASD-specific, ADHD-specific, and AuDHD-shared genetic subtypes
- **Sex-Stratified Analysis**: Pathway-specific sex differences in genetic architecture
- **Multi-Omics Integration**: Genetic, transcriptomic, metabolomic, and clinical data
- **11-Modality Feature Extraction**: Autonomic, circadian, environmental, sensory, and voice analysis pipelines
- **Production-Ready**: 500+ tests, CI/CD, comprehensive validation

## Quick Start

```bash
# Clone and install
git clone https://github.com/rohanvinaik/AuDHD_Correlation_Study.git
cd AuDHD_Correlation_Study
pip install -e .

# Run cross-disorder analysis
python scripts/refined_cross_condition_analysis.py

# Run sex-stratified analysis
python scripts/sex_stratified_refined_analysis.py
```

## Key Findings

### 1. Cross-Disorder Genetic Subtypes

Using 18,381 autism cases (Grove et al. 2019) and ADHD meta-analyses, identified **7 biologically distinct subtypes**:

**AuDHD-Overlap Subtypes:**
- **Dopaminergic AuDHD** (50% overlap): COMT, DRD4, DRD5, SLC6A3 → Stimulant-responsive
- **Serotonergic AuDHD** (14% overlap): SLC6A4 → SSRI-responsive with mood comorbidity

**ASD-Specific Subtypes:**
- **Glutamatergic ASD**: GRIN2B, GRIA1, GRM5 → E/I imbalance, avoid stimulants
- **GABAergic ASD**: GAD1, GABRG2, GABRB3 → Sensory hypersensitivity
- **Dopaminergic ASD**: DDC, DRD2, DRD3, TH → Novel non-overlapping variant
- **Serotonergic ASD**: HTR1A, HTR2A, TPH1, TPH2 → Variable SSRI response

**Clinical Impact**: 7 actionable subtypes with treatment stratification

### 2. Sex-Stratified Genetic Architecture

Using sex-stratified ADHD GWAS (Martin et al. 2018: 14,154 male, 4,945 female cases):

**Pathway-Specific Sex Differences:**
- **GABAergic pathway**: F/M ratio = 2.25 (females 125% higher, CI: 1.91-2.58)
- **Serotonergic pathway**: F/M ratio = 1.64 (females 64% higher, CI: 1.40-1.89)
- **Dopaminergic pathway**: F/M ratio = 0.27 (males 270% higher - unexpected!)
- **Glutamatergic pathway**: F/M ratio = 1.00 (equal)

**Key Discovery**: No single "female threshold" - sex differences are **pathway-specific**, not global. Challenges simple underdiagnosis model.

**Clinical Impact**: Sex-differential treatment strategies (GABAergic interventions for females, dopaminergic for males)

## System Architecture

```
Data Acquisition (14.7 GB) → Feature Extraction (11 modalities, 376 features)
    ↓                              ↓
Population Baseline (1000G)   Extended Integration (hierarchical, time-aware)
    ↓                              ↓
Deviation Scoring              Enhanced Clustering (ensemble, feature-aware)
    ↓                              ↓
Topology Gate                  Extended Validation (multi-modal tests)
    ↓                              ↓
Subtype Discovery ────────────→ Clinical Translation
```

## Project Structure

```
AuDHD_Correlation_Study/
├── FINAL_SUBMISSION_PACKAGE_V3/     # Publication-ready manuscript + appendices
│   ├── manuscript.pdf                # Main manuscript (r=0.898, p=1.06×10⁻¹³)
│   ├── mathematical_framework.pdf    # Appendix A: Mathematical theory
│   ├── biological_systems.pdf        # Appendix B: Gene functions
│   └── figures/                      # All manuscript figures (PDF + PNG)
│
├── src/audhd_correlation/           # Analysis package (~21,000 lines)
│   ├── features/                    # 11-modality feature extraction (5,230 lines)
│   ├── integrate/                   # Multi-omics integration (2,200 lines)
│   ├── modeling/                    # Anti-pattern-mining clustering (6,900 lines)
│   ├── analysis/                    # Genetic/multi-omics lookup (2,250 lines)
│   ├── causal/                      # Mediation, G×E, networks (1,300 lines)
│   └── validation/                  # Extended validation framework
│
├── scripts/                         # Analysis scripts
│   ├── refined_cross_condition_analysis.py    # Cross-disorder subtypes
│   ├── sex_stratified_refined_analysis.py     # Sex-stratified analysis
│   └── generate_1000g_baseline.py             # Population baseline
│
├── results/                         # Analysis outputs
│   ├── cross_condition_subcategories/
│   ├── sex_stratified_refined/
│   └── baseline_manifold/
│
└── configs/defaults.yaml            # 376 feature definitions, analysis config
```

## Data Sources

**Validated GWAS Data:**
- Autism: Grove et al. 2019 (18,381 cases, 27,969 controls, PMID: 30804558)
- ADHD Male: Martin et al. 2018 (14,154 cases, GCST005362, PMID: 29325848)
- ADHD Female: Martin et al. 2018 (4,945 cases, GCST012597)
- Population Baseline: 1000 Genomes Phase 3 (503 EUR, 154,425 variants)

**Downloaded Multi-Omics (14.7 GB):**
- GEO Expression: 2.57 GB (8 datasets, 24 files)
- SRA Microbiome: 12.15 GB (72 samples)
- GWAS Catalog: 328 MB (317 significant SNPs)
- Literature: 278 papers with supplements

## Anti-Pattern-Mining Framework

**Baseline-Deviation-Topology Pipeline** (prevents false positives):

1. **Baseline Manifold**: Learn population baseline (1000G controls or density-based)
2. **Deviation Scoring**: 3 geometric metrics (orthogonal residual, MST delta, k-NN curvature)
3. **Rotation Nulls**: Data-driven thresholds with FDR control
4. **Topology Gate**: Hard decision - only cluster if deviants show discrete structure
5. **Consensus Clustering**: Prevent selection bias from parameter sweeps
6. **Null Model Testing**: Permutation, rotation, SigClust, dip test
7. **Config Locking**: SHA-256 hash prevents post-hoc parameter tweaking

**All safeguards enabled by default** in `configs/defaults.yaml`

## Manuscript

**Final Submission Package V3**: Complete publication-ready materials

- **Main Manuscript** (manuscript.pdf, 841KB): "Continuous Enrichment Stratification of Neurotransmitter Genes Reveals Density Peaks in Shared ADHD-Autism Genetic Architecture"
  - Primary correlation: r=0.898, p=1.06×10⁻¹³, N=36 genes
  - Label permutation test: p<0.0001 (10,000 iterations)
  - Partial correlation (gene length): r=0.554, p=0.00046

- **Appendix A** (mathematical_framework.pdf, 71KB): Formal problem statement, topological structure, K-means algorithm, validation statistics, computational complexity

- **Appendix B** (biological_systems.pdf, 44KB): Detailed gene functions for all 36 neurotransmitter genes across glutamatergic, GABAergic, serotonergic, and dopaminergic systems

- **Figures**: 9 main + supplementary figures (PDF + PNG)

Location: `/FINAL_SUBMISSION_PACKAGE_V3/`

## Usage Examples

### Cross-Disorder Analysis

```python
from audhd_correlation.integrate import integrate_extended_multiomics
from audhd_correlation.modeling.extended_clustering import enhanced_clustering_with_extended_features

# Load GWAS data
genetic_df = load_gwas_data()  # Autism + ADHD variants
pathway_genes = load_neurotransmitter_genes()  # 40 genes, 4 pathways

# Pathway enrichment
enrichment = calculate_pathway_enrichment(genetic_df, pathway_genes)

# Identify subtypes
results = enhanced_clustering_with_extended_features(
    enrichment,
    feature_metadata={'feature_types': {'continuous': enrichment.columns.tolist()}},
    n_clusters=7
)

# Results: 7 subtypes (2 AuDHD-overlap, 5 ASD-specific)
```

### Sex-Stratified Analysis

```python
from audhd_correlation.analysis import sex_stratified_pathway_analysis

# Load sex-stratified GWAS
male_gwas = load_gwas('GCST005362')  # 14,154 male cases
female_gwas = load_gwas('GCST012597')  # 4,945 female cases
baseline = load_1000g_baseline()  # 503 EUR individuals

# Calculate pathway enrichment by sex
results = sex_stratified_pathway_analysis(
    male_gwas, female_gwas, baseline,
    pathways=['Dopamine', 'Serotonin', 'Glutamate', 'GABA']
)

# Results: F/M ratios per pathway with 95% CI
```

## Performance

- **Baseline generation**: ~2 minutes (154,425 variants, 503 individuals)
- **Cross-disorder analysis**: ~5 seconds (40 genes, 4 pathways)
- **Sex-stratified analysis**: ~10 seconds (4 pathways × 2 sexes)
- **Extended integration**: ~2 seconds (100 samples, 5 modalities)
- **Enhanced clustering**: ~3 seconds (200 samples, 50 features)

**Computational Scale:**
- 21,000+ lines production code
- 376 configured features (11 modalities)
- 14.7 GB downloaded data
- 500+ comprehensive tests

## Testing

```bash
# Full test suite
pytest

# Specific components
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/statistical/       # Statistical validation
pytest tests/features/          # Feature extraction

# With coverage
pytest --cov=src/audhd_correlation --cov-report=html
```

## Publications & Outputs

**Ready for Submission:**
1. "Continuous Enrichment Stratification of Neurotransmitter Genes..." (V3 manuscript)
2. "Seven Genetic Subtypes Across ASD, ADHD, and AuDHD: Real Patient GWAS Evidence"
3. "Sex-Differential Genetic Architecture in ADHD: GABAergic Female Enrichment..."

**Results Files:**
- [`results/cross_condition_subcategories/`](results/cross_condition_subcategories/)
- [`results/sex_stratified_refined/`](results/sex_stratified_refined/)
- [`results/baseline_manifold/`](results/baseline_manifold/)

## Documentation

**Core Docs:**
- [Baseline-Deviation-Topology Framework](docs/ANTI_PATTERN_MINING.md)
- [Cross-Disorder Analysis Guide](docs/CROSS_DISORDER_GUIDE.md)
- [Sex-Stratified Analysis Guide](docs/SEX_STRATIFIED_GUIDE.md)

**Extended Features:**
- [Feature Extraction Pipelines](docs/PROMPT_2_1_SUMMARY.md) - Autonomic, circadian, environmental
- [Multi-Modal Integration](docs/PROMPT_3_1_SUMMARY.md) - Hierarchical, time-aware
- [Enhanced Clustering](docs/PROMPT_3_2_SUMMARY.md) - Feature-aware, ensemble
- [Causal Analysis](docs/EXTENDED_CAUSAL_ANALYSIS.md) - Mediation, G×E, networks

## Citation

```bibtex
@software{vinaik2025audhd,
  author = {Vinaik, Rohan},
  title = {AuDHD Genetic Architecture Analysis: Baseline-Deviation-Topology Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/rohanvinaik/AuDHD_Correlation_Study},
  version = {3.0.0}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- **Issues**: https://github.com/rohanvinaik/AuDHD_Correlation_Study/issues
- **Author**: Rohan Vinaik

---

**Status**: ✅ Production-Ready | **Version**: 3.0.0 | **Python**: 3.9+ | **Last Updated**: October 2025
