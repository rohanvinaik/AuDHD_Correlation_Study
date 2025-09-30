# AuDHD Correlation Study Pipeline

[![Tests](https://github.com/rohanvinaik/AuDHD_Correlation_Study/workflows/Tests/badge.svg)](https://github.com/rohanvinaik/AuDHD_Correlation_Study/actions)
[![Documentation](https://readthedocs.org/projects/audhd-pipeline/badge/?version=latest)](https://audhd-pipeline.readthedocs.io/en/latest/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready system for discovering biologically distinct patient subtypes through integrated multi-omics and multi-modal phenotyping. Includes complete data acquisition infrastructure, 11-modality feature extraction pipelines, hierarchical integration methods, and advanced clustering with extended validation. Designed for AuDHD (Autism + ADHD) research but applicable to any multi-modal clustering study.

## 🎯 Key Features

### Complete Feature Extraction Pipeline (NEW)
- **Autonomic Function**: HRV (time/frequency/nonlinear), EDA, cardiovascular, respiratory measures
- **Circadian Rhythms**: Cortisol awakening response, melatonin (DLMO), actigraphy, body temperature
- **Salivary Biomarkers**: Stress hormones, inflammatory markers, oral microbiome
- **Environmental Exposures**: Air/water quality, traffic, green space, socioeconomic indicators
- **Toxicant Biomarkers**: Heavy metals, organic pollutants, PFAS, body burden indices
- **Sensory Processing**: Auditory, visual, tactile, multisensory integration, sensory gating (P50)
- **Interoception**: Heartbeat detection tasks, MAIA-2, BPQ, three-dimensional framework
- **Voice & Speech**: Prosodic, spectral, temporal, pragmatic features using Praat and librosa

### Extended Multi-Modal Integration (NEW)
- **Hierarchical Integration**: 4-level structure (biological → environmental → cognitive/sensory → clinical)
- **Time-Aware Adjustment**: Cosinor models for circadian features, standardize to collection time
- **Feature Importance Weighting**: Data-driven weights based on proximity to phenotype
- **Multimodal Networks**: Gene-metabolite, metabolite-clinical, GxE, autonomic-symptom networks
- **11 Modalities**: Genetic, metabolomic, microbiome, autonomic, circadian, salivary, environmental, toxicants, sensory, interoception, voice

### Enhanced Clustering (NEW)
- **Feature-Aware Distances**: Custom metrics for continuous, categorical, cyclical, compositional data types
- **Multi-View Clustering**: Integrate multiple complementary data views
- **Temporal Clustering**: Dynamic Time Warping for longitudinal trajectories
- **Biologically-Informed**: Constrained clustering with family structure and pathway priors
- **Ensemble Consensus**: Combine multiple methods for robust subgroup discovery
- **Extended Validation**: Autonomic, circadian, environmental, sensory, interoceptive differentiation tests

### Analysis Pipeline
- **Multi-Omics Integration**: MOFA/PCA/CCA with configurable methods
- **Advanced Clustering**: HDBSCAN, K-means, hierarchical with automatic parameter selection
- **Statistical Validation**: Bootstrap stability, cross-validation, permutation tests
- **Biological Interpretation**: GSEA pathway enrichment, gene ID normalization, drug target prediction
- **Production-Ready**: 500+ tests, CI/CD, explicit error handling, reproducible with version control

### Data Acquisition Infrastructure
- **Automated Downloads**: Parallel downloads with retry logic, resume support, checksum verification
- **Literature Tracking**: Monitor PubMed, bioRxiv, Scientific Data for dataset publications (manual check mode)
- **Comprehensive Documentation**: Auto-generated READMEs, data dictionaries, quality reports
- **Provenance Tracking**: Complete data lineage from acquisition through processing

## 📚 Documentation

**Full documentation available at:** [audhd-pipeline.readthedocs.io](https://audhd-pipeline.readthedocs.io)

- **[Quick Start Guide](https://audhd-pipeline.readthedocs.io/quickstart.html)** - Get started in 5 minutes
- **[User Guides](https://audhd-pipeline.readthedocs.io/user_guide/)** - Detailed guides for each pipeline phase
- **[API Reference](https://audhd-pipeline.readthedocs.io/api/)** - Complete API documentation
- **[Tutorials](https://audhd-pipeline.readthedocs.io/tutorials/)** - Jupyter notebook tutorials

### Feature Extraction Documentation (NEW)
- **[Prompt 2.1 Summary](docs/PROMPT_2_1_SUMMARY.md)** - Autonomic, Circadian, Salivary pipelines
- **[Prompt 2.2 Summary](docs/PROMPT_2_2_SUMMARY.md)** - Environmental & Toxicant pipelines
- **[Prompt 2.3 Summary](docs/PROMPT_2_3_SUMMARY.md)** - Sensory, Interoception, Voice pipelines

### Integration & Clustering Documentation (NEW)
- **[Prompt 3.1 Summary](docs/PROMPT_3_1_SUMMARY.md)** - Extended multi-modal integration system
- **[Prompt 3.2 Summary](docs/PROMPT_3_2_SUMMARY.md)** - Enhanced clustering with extended features

### Data Acquisition Documentation
- **[Pipeline README](scripts/pipeline/README.md)** - Automated download system
- **[Access Tracker](data/catalogs/access_tracker.md)** - Dataset access status and applications

**Note:** Automated data release monitoring is currently paused. Data availability checks are performed manually.

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

# Optional: Install voice analysis dependencies
pip install praat-parselmouth librosa

# Optional: Install temporal clustering
pip install tslearn
```

### Basic Usage - Extended Multi-Modal Analysis

**Complete Workflow:**

```python
from audhd_correlation.integrate import integrate_extended_multiomics
from audhd_correlation.modeling.extended_clustering import (
    enhanced_clustering_with_extended_features,
    validate_extended_clusters
)
import pandas as pd

# 1. Load all modalities
genetic_df = pd.read_csv('data/processed/genetics.csv', index_col=0)
metabolomic_df = pd.read_csv('data/processed/metabolomics.csv', index_col=0)
autonomic_df = pd.read_csv('data/processed/autonomic.csv', index_col=0)
circadian_df = pd.read_csv('data/processed/circadian.csv', index_col=0)
environmental_df = pd.read_csv('data/processed/environmental.csv', index_col=0)
sensory_df = pd.read_csv('data/processed/sensory.csv', index_col=0)
clinical_df = pd.read_csv('data/processed/clinical.csv', index_col=0)
context_df = pd.read_csv('data/processed/context.csv', index_col=0)

# 2. Extended integration (hierarchical, time-aware)
integration_results = integrate_extended_multiomics(
    genetic_df=genetic_df,
    metabolomic_df=metabolomic_df,
    autonomic_df=autonomic_df,
    circadian_df=circadian_df,
    environmental_df=environmental_df,
    sensory_df=sensory_df,
    clinical_df=clinical_df,
    context_df=context_df  # For time adjustment
)

integrated_features = integration_results['integrated_features']

# 3. Enhanced clustering
clustering_results = enhanced_clustering_with_extended_features(
    integrated_data=integrated_features,
    feature_metadata={'feature_types': {'continuous': integrated_features.columns.tolist()}},
    n_clusters=8
)

clusters = clustering_results['ensemble']

# 4. Extended validation
extended_features = {
    'autonomic': autonomic_df,
    'circadian': circadian_df,
    'environmental': environmental_df,
    'sensory': sensory_df
}

validation = validate_extended_clusters(
    clusters=clusters,
    extended_features=extended_features,
    clinical_features=clinical_df
)

print(f"Subgroups identified: {len(np.unique(clusters))}")
print(f"Validation tests significant: {validation['summary']['n_significant']}/{validation['summary']['n_tests']}")
```

**Feature Extraction Examples:**

```python
# Autonomic processing
from audhd_correlation.features.autonomic import HRVAnalyzer

hrv_analyzer = HRVAnalyzer()
hrv_metrics = hrv_analyzer.analyze_hrv(rr_intervals, sample_rate=1000)
print(f"SDNN: {hrv_metrics['sdnn']:.2f} ms")
print(f"RMSSD: {hrv_metrics['rmssd']:.2f} ms")
print(f"LF/HF ratio: {hrv_metrics['lf_hf_ratio']:.2f}")

# Circadian analysis
from audhd_correlation.features.circadian import CircadianAnalyzer

circadian_analyzer = CircadianAnalyzer()
car_metrics = circadian_analyzer.calculate_cortisol_awakening_response(
    cortisol_samples, sample_times
)
print(f"CAR AUCi: {car_metrics['car_auci']:.2f}")

# Sensory processing
from audhd_correlation.features.sensory_detailed import SensoryProcessor

sensory_processor = SensoryProcessor()
sensory_results = sensory_processor.process_sensory_battery(sensory_data)
print(f"P50 gating ratio: {sensory_results['p50_gating_ratio']:.3f}")

# Interoception
from audhd_correlation.features.interoception import InteroceptionProcessor

intero_processor = InteroceptionProcessor()
intero_metrics = intero_processor.calculate_heartbeat_counting_accuracy(
    recorded_beats, counted_beats, confidence
)
print(f"Interoceptive accuracy: {intero_metrics['interoceptive_accuracy']:.3f}")

# Voice analysis
from audhd_correlation.features.voice_analysis import VoiceAnalyzer

voice_analyzer = VoiceAnalyzer()
voice_features = voice_analyzer.analyze_voice_sample('audio.wav')
print(f"Mean pitch: {voice_features['pitch_mean_hz']:.1f} Hz")
print(f"HNR: {voice_features['hnr_db']:.1f} dB")
```

### Data Acquisition Workflow

**Download Data:**

```bash
# Run comprehensive download script
bash scripts/download_everything.sh

# Or download specific sources
python scripts/downloaders/geo_downloader.py
python scripts/downloaders/sra_downloader.py
python scripts/papers/scrape_all.py
```

**Check Download Status:**

```bash
python scripts/download_tracker.py
```

## 📊 Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION (14.7 GB)                        │
├─────────────────────────────────────────────────────────────────────┤
│  GEO (2.6GB) → SRA (12.1GB) → Papers (278) → GWAS (328MB)          │
│  Manual monitoring for new releases and publications                │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION (11 Modalities, 376 Features)        │
├─────────────────────────────────────────────────────────────────────┤
│  Autonomic (24) → Circadian (16) → Salivary (11)                   │
│  Environmental (23) → Toxicants (26) → Sensory (13)                │
│  Interoception (10) → Voice (13) → + Original modalities            │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│           EXTENDED INTEGRATION (Hierarchical, Time-Aware)            │
├─────────────────────────────────────────────────────────────────────┤
│  Level 1: Biological (genetic, metabolic, autonomic, circadian)    │
│  Level 2: Environmental (exposures, toxicants)                      │
│  Level 3: Cognitive/Sensory (sensory, interoception, voice)        │
│  Level 4: Clinical phenotypes                                       │
│  → Multimodal networks → Time-adjusted features                    │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│      ENHANCED CLUSTERING (Feature-Aware, Multi-View, Ensemble)      │
├─────────────────────────────────────────────────────────────────────┤
│  Custom distances → Multi-view → Temporal → Constrained → Ensemble │
│  Extended validation: Autonomic, circadian, sensory, toxicant      │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│         VALIDATION & REPORTING (Bootstrap, Biological Tests)         │
└─────────────────────────────────────────────────────────────────────┘
```

## 🗂️ Project Structure

```
AuDHD_Correlation_Study/
├── src/audhd_correlation/           # Main analysis package (~7,950 lines)
│   ├── data/                        # Data loaders and harmonization
│   ├── preprocess/                  # Preprocessing and normalization
│   ├── integrate/                   # Multi-omics integration
│   │   ├── methods.py               # Standard integration (MOFA/PCA/CCA)
│   │   └── extended_integration.py  # NEW: Hierarchical + time-aware (850 lines)
│   ├── modeling/                    # Clustering algorithms
│   │   ├── clustering.py            # Standard clustering (882 lines)
│   │   └── extended_clustering.py   # NEW: Feature-aware + ensemble (850 lines)
│   ├── features/                    # NEW: Feature extraction pipelines (~4,430 lines)
│   │   ├── autonomic.py             # HRV, EDA, cardiovascular (380 lines)
│   │   ├── circadian.py             # Cortisol, melatonin, actigraphy (420 lines)
│   │   ├── salivary.py              # Stress, inflammation, microbiome (350 lines)
│   │   ├── environmental.py         # Air/water quality, exposures (630 lines)
│   │   ├── toxicants.py             # Heavy metals, organic pollutants (680 lines)
│   │   ├── sensory_detailed.py      # Auditory, visual, tactile, gating (860 lines)
│   │   ├── interoception.py         # Heartbeat tasks, questionnaires (380 lines)
│   │   └── voice_analysis.py        # Prosodic, spectral, MFCC (730 lines)
│   ├── validation/                  # Validation metrics
│   ├── biological/                  # Pathway enrichment
│   ├── viz/                         # Visualization
│   └── reporting/                   # Report generation
│
├── scripts/                         # Data acquisition & monitoring
│   ├── downloaders/                 # Dataset downloaders (GEO, SRA, dbGaP)
│   ├── papers/                      # Paper scraping scripts
│   ├── trials/                      # Clinical trials access
│   ├── registries/                  # Patient registries & biobanks
│   └── monitoring/                  # Data release monitoring (manual mode)
│
├── data/                            # Data directory (gitignored)
│   ├── raw/                         # Original datasets (14.7 GB)
│   ├── interim/                     # Intermediate files
│   ├── processed/                   # Final processed data
│   ├── papers/                      # Downloaded papers (278 papers)
│   ├── catalogs/                    # Dataset catalogs
│   │   ├── master_catalog.json
│   │   ├── download_status.json
│   │   └── access_tracker.md
│   └── documentation/               # Auto-generated docs
│
├── configs/                         # Configuration files
│   ├── defaults.yaml                # Main config (376 feature definitions)
│   ├── download_config.yaml
│   └── monitoring_config.yaml
│
├── docs/                            # Documentation
│   ├── PROMPT_2_1_SUMMARY.md        # Autonomic, Circadian, Salivary
│   ├── PROMPT_2_2_SUMMARY.md        # Environmental, Toxicants
│   ├── PROMPT_2_3_SUMMARY.md        # Sensory, Interoception, Voice
│   ├── PROMPT_3_1_SUMMARY.md        # Extended integration
│   └── PROMPT_3_2_SUMMARY.md        # Enhanced clustering
│
├── tests/                           # Comprehensive test suite (500+ tests)
├── notebooks/                       # Jupyter tutorials
└── outputs/                         # Analysis outputs
```

## 🆕 System Capabilities

### Feature Extraction Pipelines (Prompts 2.1-2.3)

**9 Specialized Pipelines | 4,430 Lines of Code | 240 New Features**

1. **Autonomic Function (380 lines)**
   - HRV: Time domain (SDNN, RMSSD, pNN50), Frequency (LF, HF, LF/HF), Nonlinear (SD1, SD2, SampEn, ApEn, DFA)
   - EDA: Skin conductance level, response frequency/amplitude
   - Cardiovascular: Baroreflex sensitivity, blood pressure variability, orthostatic response

2. **Circadian Rhythms (420 lines)**
   - Cortisol: Awakening response (CAR), diurnal slope, evening cortisol
   - Melatonin: DLMO (dim light melatonin onset), amplitude, phase
   - Actigraphy: Interdaily stability, intradaily variability, relative amplitude

3. **Salivary Biomarkers (350 lines)**
   - Stress hormones: Cortisol, testosterone, DHEA
   - Inflammatory markers: CRP, IL-1β, IL-6, TNF-α
   - Oral microbiome: Diversity, Streptococcus abundance

4. **Environmental Exposures (630 lines)**
   - Air quality: PM2.5, NO2, O3 (prenatal, early life, lifetime)
   - Water quality: Nitrate, arsenic, fluoride
   - Traffic & Green space: Proximity, density, NDVI
   - Socioeconomic: Area deprivation index, neighborhood SES

5. **Toxicant Biomarkers (680 lines)**
   - Heavy metals: Pb, Hg, Cd, As (hair, blood, urine)
   - Organic pollutants: BPA, phthalates, PFAS, pesticides
   - Body burden indices: Toxic metal index, pollutant mixture index

6. **Sensory Processing (860 lines)**
   - Auditory: Pure tone audiometry, OAE, ABR, gap detection, P50 sensory gating
   - Visual: Contrast sensitivity, motion coherence, visual search, working memory
   - Tactile: Two-point discrimination, vibrotactile threshold, proprioception
   - Multisensory: McGurk effect, temporal binding window, sound-induced flash

7. **Interoception (380 lines)**
   - Accuracy: Heartbeat counting (Schandry 1981), discrimination (d', criterion)
   - Sensibility: MAIA-2 (8 subscales), Body Perception Questionnaire
   - Awareness: Confidence-accuracy correlation (Garfinkel 2015 framework)

8. **Voice & Speech (730 lines)**
   - Prosodic: Pitch (F0), intensity, rhythm, speech rate, pauses
   - Spectral: Formants (F1-F3), voice quality (jitter, shimmer, HNR, CPP)
   - Temporal: VOT, segment durations, coarticulation
   - MFCC: 13 coefficients + deltas for machine learning

### Extended Integration (Prompt 3.1)

**850 Lines of Code | Hierarchical 4-Level Structure**

- **Time-Aware Adjustment**: Cosinor models standardize circadian features to common collection time
- **Hierarchical Integration**: 4 levels (biological → environmental → cognitive/sensory → clinical)
- **Feature Weighting**: Data-driven weights by proximity to phenotype (metabolomic 20%, genetic 15%, etc.)
- **Multimodal Networks**: Gene-metabolite, metabolite-clinical, GxE, autonomic-symptom networks
- **11 Modalities Integrated**: Genetic, metabolomic, microbiome, autonomic, circadian, salivary, environmental, toxicants, sensory, interoception, voice

### Enhanced Clustering (Prompt 3.2)

**850 Lines of Code | 5 Advanced Methods**

- **Feature-Aware Distances**: Custom metrics for continuous, categorical, cyclical, compositional, binary data
- **Multi-View Clustering**: Integrate complementary biological, physiological, environmental, cognitive views
- **Temporal Clustering**: Dynamic Time Warping for longitudinal developmental trajectories
- **Biologically-Informed**: Constrained clustering with family structure and pathway priors
- **Ensemble Consensus**: Combine multiple methods (K-means, spectral, agglomerative, HDBSCAN) for robust subgroups
- **Extended Validation**: Test autonomic, circadian, environmental, sensory, interoceptive differentiation across clusters

## 📊 Data Summary

### Downloaded Data (14.7 GB)

| Source | Status | Size | Content |
|--------|--------|------|---------|
| **GEO Expression** | ✅ Complete | 2.57 GB | 8 datasets, 24 files |
| **SRA Microbiome** | ✅ Complete | 12.15 GB | 72 samples |
| **Papers** | ✅ Complete | - | 278 papers with supplements |
| **Data Repositories** | ✅ Complete | 83 MB | 6 files from GitHub/Zenodo |
| **GWAS** | ✅ Complete | 328 MB | 317 significant SNPs |

### Feature Space

| Category | Features | Pipelines |
|----------|----------|-----------|
| **Original Features** | 136 | Genetic, metabolomic, clinical, microbiome, imaging |
| **Extended Features** | 240 | Autonomic, circadian, environmental, sensory, voice |
| **Total Configured** | **376** | 11 modalities |

## 🔬 Configuration

Full configuration in `configs/defaults.yaml` with 376 feature definitions across 11 modalities.

**Integration Configuration:**

```yaml
integrate:
  method: hierarchical
  weights:
    genetic: 0.15
    environmental: 0.08
    toxicants: 0.07
    microbiome: 0.08
    metabolomic: 0.20     # Highest weight (most proximal)
    autonomic: 0.12
    circadian: 0.10
    salivary: 0.05
    sensory: 0.07
    interoception: 0.06
    voice: 0.05
    clinical: 0.02        # Outcome, not predictor

  hierarchical_levels:
    level1_biological:
      components: [genetic, metabolomic, microbiome, autonomic, circadian, salivary]
      method: PCA
      n_factors: 30
    level2_environmental:
      components: [environmental, toxicants]
      method: PCA
      n_factors: 15
    level3_cognitive_sensory:
      components: [sensory, interoception, voice]
      method: PCA
      n_factors: 15
    level4_clinical:
      components: [clinical]
      method: None

  time_adjustment:
    enabled: true
    standard_time: 9.0  # Standardize to 9 AM
```

**Clustering Configuration:**

```yaml
cluster:
  extended_methods:
    feature_aware_distance:
      enabled: true
      feature_types:
        continuous: [factor_, hrv_, metabolite_]
        cyclical: [circadian_phase, melatonin_phase]
        compositional: [microbiome_]

    multiview:
      enabled: true
      views:
        biological: [genetic, metabolomic, microbiome]
        physiological: [autonomic, circadian, salivary]
        environmental: [environmental, toxicants]
        cognitive_sensory: [sensory, interoception, voice]

    ensemble:
      enabled: true
      base_methods: [kmeans, spectral, agglomerative, feature_aware]
```

## 🧪 Testing

Comprehensive test suite with 500+ tests covering all components:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/statistical/       # Statistical tests
pytest tests/features/          # NEW: Feature extraction tests

# Run with coverage
pytest --cov=src/audhd_correlation --cov-report=html
```

## 📝 User Action Items

### Required for Data Access

1. **Sign up for restricted-access databases:**
   - NSRR (National Sleep Research Resource) - For polysomnography and sleep data
   - PhysioNet - For physiological signal databases
   - All of Us Research Program - For multi-modal health data

2. **When available:**
   - Download NHANES data when CDC site recovers
   - Optional: Get EPA AirNow API key for live air quality data
   - Optional: Get NASA EarthData account for satellite green space data

### Optional Dependencies

For voice analysis (recommended):
```bash
pip install praat-parselmouth librosa
```

For temporal clustering (optional):
```bash
pip install tslearn
```

## 📈 Performance Metrics

**System Scale:**
- ~7,950 lines of production code
- 376 configured features across 11 modalities
- 14.7 GB downloaded data (8 GEO datasets, 72 SRA samples, 278 papers)
- 4 hierarchical integration levels
- 5 advanced clustering methods
- 500+ comprehensive tests

**Computational Performance:**
- Extended integration: ~2 seconds (100 samples, 5 modalities)
- Enhanced clustering: ~3 seconds (200 samples, 50 features)
- Feature extraction: Modality-dependent (HRV: seconds, Voice: minutes)

## 🆕 Recent Updates (January 2025)

### Major Feature Additions

**Prompts 2.1-2.3: Feature Extraction (4,430 lines)**
- ✅ Autonomic function pipeline (HRV, EDA, cardiovascular, respiratory)
- ✅ Circadian rhythm analysis (cortisol, melatonin, actigraphy, temperature)
- ✅ Salivary biomarker processing (hormones, inflammation, microbiome)
- ✅ Environmental exposure linking (air, water, traffic, green space)
- ✅ Toxicant biomarker analysis (heavy metals, organic pollutants, PFAS)
- ✅ Sensory processing assessment (auditory, visual, tactile, multisensory, gating)
- ✅ Interoception measurement (accuracy, sensibility, awareness)
- ✅ Voice & speech acoustics (prosodic, spectral, temporal, pragmatic, MFCC)

**Prompt 3.1: Extended Integration (850 lines)**
- ✅ Hierarchical 4-level integration framework
- ✅ Time-aware circadian feature adjustment using cosinor models
- ✅ Feature importance weighting by proximity to phenotype
- ✅ Multimodal network analysis (4 cross-modal networks)
- ✅ 11-modality integration with preprocessing pipeline

**Prompt 3.2: Enhanced Clustering (850 lines)**
- ✅ Feature-aware distance metrics (5 types: continuous, categorical, cyclical, compositional, binary)
- ✅ Multi-view clustering across complementary data views
- ✅ Temporal clustering with Dynamic Time Warping for trajectories
- ✅ Biologically-informed clustering with family/pathway constraints
- ✅ Ensemble consensus from multiple methods
- ✅ Extended validation framework (autonomic, circadian, environmental, sensory tests)

### System Updates

- ✅ Configuration expanded to 376 feature definitions
- ✅ Comprehensive documentation (5 detailed summary files)
- ✅ Data acquisition complete (14.7 GB across 5 sources)
- ✅ Automated monitoring paused (manual check mode)

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
black src/ tests/ scripts/
isort src/ tests/ scripts/
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
@software{vinaik2025audhd,
  author = {Vinaik, Rohan},
  title = {AuDHD Correlation Study: Complete Multi-Modal Phenotyping and Analysis System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/rohanvinaik/AuDHD_Correlation_Study},
  version = {2.0.0},
  note = {Comprehensive system for multi-modal data acquisition, feature extraction (11 modalities),
          hierarchical integration, and advanced clustering for AuDHD research}
}
```

## 🙏 Acknowledgments

### Analysis & Feature Extraction
- **Testing Framework**: pytest, hypothesis, pytest-benchmark
- **Multi-Omics Methods**: MOFA, scikit-learn, UMAP
- **Statistical Analysis**: scipy, statsmodels, pingouin
- **Signal Processing**: scipy.signal, neurokit2, hrv-analysis
- **Voice Analysis**: Praat/parselmouth, librosa
- **Temporal Analysis**: tslearn (Dynamic Time Warping)

### Data Acquisition
- **Data Sources**: SFARI, UK Biobank, ABCD Study, NIH, PGC, GEO, SRA, EPA, USGS
- **Web Technologies**: requests, feedparser, BeautifulSoup
- **APIs**: NCBI E-utilities, EPA AirNow, Census Geocoding

---

**Status**: ✅ Production-Ready | **Version**: 2.0.0 | **Python**: 3.9+ | **Last Updated**: January 2025

**Complete System:**
- ✅ Data Acquisition (14.7 GB, 5 sources)
- ✅ Feature Extraction (11 modalities, 376 features, 4,430 lines)
- ✅ Extended Integration (hierarchical, time-aware, 850 lines)
- ✅ Enhanced Clustering (feature-aware, ensemble, 850 lines)
- ✅ Comprehensive Validation (extended multi-modal tests)
- ✅ ~7,950 lines of tested production code
- ✅ 500+ comprehensive tests
- ✅ Complete documentation
