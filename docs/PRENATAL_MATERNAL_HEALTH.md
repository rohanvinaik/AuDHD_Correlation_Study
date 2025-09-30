# Prenatal and Maternal Health Data Integration

Comprehensive system for extracting, harmonizing, and analyzing prenatal exposures and maternal health conditions critical for neurodevelopmental research.

## Overview

Prenatal environment profoundly influences neurodevelopment. The **maternal immune activation (MIA) hypothesis** and extensive research on prenatal exposures demonstrate strong associations between pregnancy conditions and ASD/ADHD risk.

### Key Prenatal Risk Factors

| Factor | Evidence Strength | Critical Window | Effect Size |
|--------|-------------------|-----------------|-------------|
| **Maternal infection + fever** | STRONG | Weeks 10-20 (neurogenesis) | OR 1.3-2.0 |
| **SSRI exposure** | MODERATE | First trimester | OR 1.2-1.5 (controversial) |
| **Preterm birth (<37 weeks)** | STRONG | Third trimester | OR 1.5-2.5 |
| **Low birth weight (<2500g)** | STRONG | Throughout | OR 1.3-2.0 |
| **Gestational diabetes** | MODERATE | Throughout | OR 1.2-1.5 |
| **Maternal stress** | MODERATE | Throughout | OR 1.2-1.6 |
| **Folate deficiency** | MODERATE | First trimester (neural tube) | OR 1.2-1.5 (protective) |
| **Valproate (anticonvulsant)** | STRONG | First trimester | OR 3.0-5.0 |

## System Components

### 1. Feature Extraction (`prenatal_maternal.py`)

**800+ lines** of comprehensive feature extraction with 5 specialized analyzers:

#### Maternal Infection Analyzer
- **Trimester-specific tracking** (1st, 2nd, 3rd)
- **Critical window analysis**:
  - Neural tube closure (days 21-28)
  - Neurogenesis peak (weeks 10-20) ← **Most critical**
  - Synaptogenesis peak (weeks 20-34)
- **Severity grading** (1-3 scale + hospitalization)
- **Fever tracking** (days with fever, critical for MIA)
- **Infection types**: Influenza, bacterial, viral, UTI, COVID-19
- **Antibiotic exposure** by trimester
- **Composite infection risk score** (0-10 scale)

**Output**: 25+ infection features including timing, severity, treatment

#### Maternal Medication Analyzer
- **Medication classification**:
  - SSRIs (fluoxetine, sertraline, paroxetine, citalopram, escitalopram)
  - SNRIs (venlafaxine, duloxetine)
  - Anticonvulsants (valproate, phenytoin, carbamazepine, lamotrigine)
  - Antibiotics (amoxicillin, azithromycin, cephalexin)
  - Acetaminophen (chronic use associated with ADHD)
  - Benzodiazepines
- **Timing and duration** by trimester
- **Polypharmacy detection** (≥3 concurrent medications)
- **Risk stratification** (high/moderate/low)

**Output**: 20+ medication features by class and timing

#### Pregnancy Complications Analyzer
- **Major complications**:
  - Gestational diabetes
  - Preeclampsia / pregnancy-induced hypertension
  - Placenta previa / placental abruption
  - Chorioamnionitis (intrauterine infection)
  - IUGR (intrauterine growth restriction)
  - Oligohydramnios / polyhydramnios
- **Birth outcomes**:
  - Gestational age (with preterm/very preterm flags)
  - Birth weight (with LBW/VLBW flags)
  - APGAR scores (1 min, 5 min)
  - Delivery method (cesarean/vaginal)
  - NICU admission
- **Composite birth complication score**

**Output**: 15+ complication and birth outcome features

#### Maternal Stress & Nutrition Analyzer
- **Stress measures**:
  - Life stress events count
  - Depression scores (PHQ-9)
  - Anxiety scores
  - Cortisol levels (if available)
- **Nutritional status**:
  - Prenatal vitamin use
  - Folate supplementation (dose adequacy)
  - Vitamin D levels (deficiency detection)
  - Omega-3 supplementation

**Output**: 10+ stress and nutrition features

### 2. Data Source Mappings (`prenatal_mappings.yaml`)

Comprehensive variable mappings from 5 major cohorts:

#### SPARK (Simons Foundation)
- **50,000+ participants** with ASD diagnosis
- **Medical History Questionnaire** - Pregnancy section
- Variables:
  - Infections: Type, timing, fever, hospitalization
  - Medications: Free text + categorical
  - Complications: Gestational diabetes, preeclampsia
  - Birth outcomes: GA, birth weight, APGAR, delivery method
  - Prenatal care: Vitamins, visit count
- **Recall bias**: Moderate (variable timing, typically <5 years)

#### ABCD (Adolescent Brain Cognitive Development)
- **11,000+ participants** at age 9-10
- **Developmental History Questionnaire** (retrospective)
- Variables:
  - Infections: Binary + fever
  - Substance use: Smoking, alcohol, drugs
  - Complications: Multi-select
  - Birth outcomes: GA, birth weight category, preterm flag
- **Recall bias**: HIGH (9-10 years retrospective)
- **Note**: Higher missing data, less detail than SPARK

#### SSC (Simons Simplex Collection)
- **2,600+ families** with one ASD child
- **Structured pregnancy interview**
- Most detailed of all sources
- **Recall bias**: Low (detailed structured interview)

#### NHANES (National Health and Nutrition Examination Survey)
- **Reproductive Health Questionnaire**
- Birth outcomes from general population
- Useful for population norms

#### All of Us Research Program
- **Pregnancy Module** + **EHR linkage**
- Most objective data (EHR ICD codes, prescriptions)
- **Recall bias**: Low (EHR validation)

### 3. Data Loader (`prenatal_loader.py`)

**400+ lines** handling:

#### Harmonization
- **Gestational age**: Weeks, days, binary preterm → unified weeks
- **Birth weight**: Grams, pounds, categorical → unified grams
- **Infection timing**: Free text, trimester, gestational week → standardized trimester
- **Medication names**: Free text → classified categories

#### Imputation
- Missing gestational age: Preterm=1 → 35 weeks, Preterm=0 → 39 weeks
- Missing birth weight: Categorical → imputed grams
- Missing infection timing: "During pregnancy" → Trimester 2 (most common)
- Flags all imputations for sensitivity analysis

#### Data Quality
- **Recall bias levels**: Low/Moderate/High by source and timing
- **Completeness scores**: Proportion of critical variables present
- **Confidence scores**: Recall bias × Completeness
- **Outlier detection**: GA <20 or >44 weeks, BW <500g or >6000g

#### Multi-Source Integration
- Combines SPARK, ABCD, SSC data
- Tracks data provenance
- Handles different variable formats
- Applies source-specific quality adjustments

## Usage

### Basic Usage

```python
from audhd_correlation.data.prenatal_loader import PrenatalDataLoader, create_prenatal_feature_matrix
from audhd_correlation.features.prenatal_maternal import extract_comprehensive_prenatal_features

# 1. Load and harmonize from multiple sources
loader = PrenatalDataLoader()

prenatal_data = loader.load_and_harmonize(
    spark_path='data/spark/medical_history.csv',
    abcd_path='data/abcd/dev_history.csv'
)

# 2. Create feature matrix
features = create_prenatal_feature_matrix(prenatal_data)

print(f"Extracted {features.shape[1]} prenatal features from {features.shape[0]} participants")
```

### Advanced Usage with Individual Components

```python
from audhd_correlation.features.prenatal_maternal import (
    MaternalInfectionAnalyzer,
    MaternalMedicationAnalyzer,
    PregnancyComplicationsAnalyzer
)

# Detailed infection analysis
infection_analyzer = MaternalInfectionAnalyzer()

infection_data = pd.DataFrame({
    'infection_type': ['influenza', 'uti'],
    'gestational_week': [15, 28],  # Week 15 = neurogenesis peak!
    'severity': [3, 1],
    'fever_days': [3, 0],
    'hospitalization': [False, False],
    'antibiotics_used': [False, True]
})

infection_features = infection_analyzer.extract_infection_features(
    infection_data, birth_date=datetime(2020, 1, 15)
)

# Output includes:
# - infection_neurogenesis_peak: 1 (!!!)
# - infection_risk_score: 6.5
# - fever_days_total: 3
# - infection_trimester1: 0
# - infection_trimester2: 1
# - antibiotics_pregnancy: 1
```

### Integration with Causal Analysis

```python
from audhd_correlation.causal.extended_causal import ExtendedCausalAnalyzer

# G×E analysis: Genetic risk × Maternal infection
analyzer = ExtendedCausalAnalyzer()

results = analyzer.test_gene_environment_interactions(
    genetic_data=genetic_prs,
    environmental_data=prenatal_features[['infection_neurogenesis_peak',
                                         'infection_risk_score',
                                         'maternal_medication_ssri']],
    outcome=clinical_df['autism_diagnosis'],
    permutations=1000
)

# Test critical period hypothesis
critical_period_results = analyzer.identify_critical_periods(
    exposures={
        'maternal_infection': infection_trimester_features,
        'maternal_ssri': ssri_trimester_features
    },
    outcome=clinical_df['autism_diagnosis'],
    n_bootstrap=1000
)
```

## Data Quality Considerations

### Recall Bias

**Problem**: Parents reporting pregnancy events years later

**Mitigation**:
1. **Track recall bias level** by source and timing
2. **Weight analyses** by data confidence scores
3. **Sensitivity analyses** excluding high-bias data
4. **Triangulation** with objective records when possible

**Recall Accuracy by Variable** (from literature):
- Birth weight, gestational age: ~90% accurate up to 10 years
- Major complications: ~80% accurate
- Infection timing (trimester): ~60% accurate
- Specific medication names: ~40% accurate if >5 years

### Missing Data

**Expected Missing Rates**:
- Gestational age: 5-10%
- Birth weight: 5-10%
- Maternal infection (any): 10-20%
- Infection timing (trimester): 40-60%
- Specific medication names: 50-70%
- Infection severity: 60-80%

**Handling Strategy**:
1. **Tier 1 (critical)**: Impute with missingness indicator
2. **Tier 2 (important)**: Multiple imputation with sensitivity analysis
3. **Tier 3 (optional)**: Complete case analysis

### Data Validation

**Internal Consistency Checks**:
```python
# Preterm birth should match GA <37 weeks
assert (prenatal_df['preterm_birth'] == 1).sum() == \
       (prenatal_df['gestational_age_weeks'] < 37).sum()

# Low birth weight should correlate with preterm
correlation = prenatal_df[['preterm_birth', 'low_birth_weight']].corr()
assert correlation.iloc[0,1] > 0.3  # Should be positively correlated

# APGAR <7 should be rare (1-2%)
assert prenatal_df['low_apgar'].mean() < 0.05
```

**External Validation**:
- Compare prevalence rates to population norms (NHANES, CDC)
- Validate high-risk combinations (e.g., VLBW + low APGAR)
- Check impossible combinations (e.g., GA >44 weeks)

## Literature Basis

### Maternal Immune Activation (MIA)

**Key Finding**: Maternal infection during pregnancy, especially with fever, increases ASD risk by 30-100%

**Critical Window**: Weeks 10-20 (neurogenesis peak) show strongest association

**Mechanism**: Maternal cytokines (IL-6, IL-17) cross placenta → fetal brain inflammation → altered neurodevelopment

**References**:
1. **Patterson PH (2011)**. *Maternal infection and immune involvement in autism*. Trends Mol Med, 17(7), 389-394.
2. **Atladóttir HÓ et al. (2010)**. *Maternal infection requiring hospitalization during pregnancy and autism spectrum disorders*. JAACAP, 49(9), 713-721.
3. **Meyer U (2006)**. *The time of prenatal immune challenge determines the specificity of inflammation-mediated brain and behavioral pathology*. J Neurosci, 26(18), 4752-4762.

### Prenatal SSRI Exposure

**Key Finding**: Controversial - some studies show 20-50% increased ASD risk, others show no effect after controlling for maternal depression

**Critical Window**: First trimester (organogenesis)

**Confounding**: Maternal depression itself may increase risk

**References**:
1. **Brown HK et al. (2017)**. *Association between serotonergic antidepressant use during pregnancy and autism spectrum disorder in children*. JAMA, 317(15), 1544-1552.
2. **Sujan AC et al. (2017)**. *Associations of maternal antidepressant use during pregnancy with risk of neurodevelopmental disorders*. JAMA Psych, 74(11), 1149-1158.

### Preterm Birth and Low Birth Weight

**Key Finding**: Preterm (<37 weeks) increases ASD risk by 50-150%

**Mechanism**: Incomplete brain maturation, hypoxia, inflammation

**References**:
1. **Johnson S & Marlow N (2017)**. *Preterm birth and neurodevelopment*. Arch Dis Child Fetal Neonatal Ed, 102(6), F399-F407.

### Valproate (Anticonvulsant)

**Key Finding**: Strongest known prenatal risk factor - 3-5x increased ASD risk

**Critical Window**: First trimester

**Mechanism**: Histone deacetylase inhibition → epigenetic changes

**Note**: Well-established, contraindicated in pregnancy

## Configuration

Add to `configs/features/extended.yaml`:

```yaml
prenatal_maternal_features:
  enabled: true

  # Data sources
  data_sources:
    spark: true
    abcd: true
    ssc: false  # If available

  # Quality thresholds
  minimum_confidence: 0.3  # Exclude if confidence <0.3
  allow_imputation: true
  flag_high_recall_bias: true

  # Feature priorities
  tier_1_critical:
    - gestational_age_weeks
    - birth_weight_grams
    - maternal_infection
    - preterm_birth

  tier_2_important:
    - infection_trimester
    - infection_neurogenesis_peak  # Most critical window
    - maternal_medication_ssri
    - pregnancy_complications

  # Risk stratification
  infection_risk_threshold: 5.0  # 0-10 scale
  complication_score_threshold: 5.0

  # Analysis options
  critical_period_analysis: true
  gxe_interaction_testing: true
  mediation_analysis: true  # Test if prenatal factors mediate genetic risk
```

## Interpretation Guidelines

### Infection Risk Score (0-10 scale)

- **0-2**: No/minimal risk (no infection or late/mild)
- **3-5**: Moderate risk (infection but not critical window, or mild during critical window)
- **6-8**: High risk (moderate infection during neurogenesis, or severe anytime)
- **9-10**: Very high risk (severe infection during neurogenesis with fever + hospitalization)

### Birth Complication Score

- **0-2**: Uncomplicated birth
- **3-5**: Mild complications
- **6-8**: Moderate complications
- **9+**: Severe complications

### Data Confidence Score (0-1 scale)

- **0.8-1.0**: High confidence (low recall bias, complete data)
- **0.5-0.7**: Moderate confidence (moderate recall bias or some missing)
- **0.3-0.4**: Low confidence (high recall bias or substantial missing)
- **<0.3**: Very low confidence (recommend exclusion)

## Future Enhancements

1. **Birth Registry Linkage**: State vital statistics for objective validation
2. **EHR Integration**: Hospital delivery records, prenatal visit notes
3. **Biomarker Validation**: Cord blood cytokines, maternal serum markers
4. **Longitudinal Tracking**: Multiple pregnancies per mother
5. **Paternal Health**: Paternal age, medications, exposures
6. **Epigenetic Markers**: DNA methylation from prenatal exposures

## Files

- **Feature extraction**: `src/audhd_correlation/features/prenatal_maternal.py` (800 lines)
- **Data loader**: `src/audhd_correlation/data/prenatal_loader.py` (400 lines)
- **Mappings**: `configs/prenatal_mappings.yaml` (300 lines)
- **Documentation**: `docs/PRENATAL_MATERNAL_HEALTH.md` (this file)

**Total**: ~1,500 lines of prenatal/maternal health infrastructure

---

**Critical for AuDHD research**: Prenatal factors are among the strongest modifiable risk factors. Maternal immune activation, SSRI exposure, and birth complications show consistent associations across studies. This system enables comprehensive analysis of these critical exposures.
