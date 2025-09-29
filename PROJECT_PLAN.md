# Multi-Omics Integration Pipeline for ADHD/Autism Subtyping

## Research Questions

### Primary Question
Do the diagnostic labels "ADHD" and "autism" decompose into biologically distinct, clinically actionable subtypes when jointly modeling genome/epigenome, rare & structural variation, transcriptome/proteome/metabolome, microbiome, exposome, neuroimaging/EEG, and rich longitudinal clinical trajectories, after accounting for prenatal, perinatal/neonatal, and postnatal exposures; sex/pubertal stage/hormonal cycles; sleep/diet/activity; medication/substance use; sociodemographics/healthcare access; site/batch effects; and measurement context (fasting/circadian/collection protocols)?

### Key Sub-Questions (Pre-register)
1. **Separation vs spectrum**: Are discovered clusters discontinuous (gaps/topology) or a manifold (gradients)?
2. **Proximal biology**: Which layers (e.g., kynurenine, GABA-glutamate, mitochondrial/energy, immune) best discriminate subtypes net of confounders?
3. **Etiologic classes**: Do subtypes map to (a) neurotransmitter-centric, (b) neurodevelopmental/rare variant, (c) immune-inflammatory, (d) mitochondrial/metabolic, (e) endocrine, (f) gut–brain/microbiome, (g) sensory/EEG-network, (h) connectome/structural?
4. **Gene–environment interplay**: Which PRS/rare CNVs/LoF interact with prenatal stress/infection, medications (e.g., valproate/SSRIs), toxins (lead/PFAS), delivery/feeding/antibiotics, etc.?
5. **Clinical utility**: Do subtypes predict treatment response, comorbidity risk, trajectory, and real-world outcomes (school/occupation; quality of life)?
6. **Robustness**: Do findings replicate across sites, ancestries, sex, and platforms, and hold under MNAR missingness, batch, fasting/circadian, and medication washouts?

## Expected Outcomes
- **Primary clusters**:
  - Neurotransmitter dysfunction (serotonin/dopamine/GABA axis)
  - Neurodevelopmental disruption (rare variants, intellectual disability)
  - Immune-inflammatory
  - Mitochondrial/metabolic
  - Endocrine
  - Gut-brain/microbiome
  - Sensory/EEG-network
  - Connectome/structural
- **Potential subclusters within primary**:
  - Serotonin-dominant vs dopamine-dominant
  - With/without GI involvement
  - Early vs late onset
  - With/without inflammatory markers

## Phase 0: Ethics, Governance, and FAIR (Day 0)

### Ethics and Governance
- DUAs/IRB approvals, pediatric consent/assent
- Return-of-results policy (pathogenic CNVs)
- Privacy: de-identification; federated or site-aware training if needed
- Bias auditing: subgroup performance/coverage; communicate limits

### FAIR Principles
- Ontologized data dictionaries
- Code & pipelines reproducible (Docker/Conda; seeds logged)
- Pre-registration of hypotheses and analysis plan

## Phase 1: Data Acquisition and Harmonization (Days 1-2)

### 1.1 Public Dataset Mining

```python
# Core datasets with both genetics and some metabolomics
datasets = {
    'SPARK': {
        'url': 'https://base.sfari.org',
        'n_samples': '~50000',
        'data_types': ['WGS', 'WES', 'phenotypes', 'metabolomics_subset'],
        'requirements': 'Data use agreement'
    },
    'SSC': {
        'url': 'https://base.sfari.org',
        'n_samples': '~3000',
        'data_types': ['WGS', 'phenotypes', 'metabolomics_studies'],
    },
    'ABCD': {
        'url': 'https://abcdstudy.org',
        'n_samples': '~12000',
        'data_types': ['genetics', 'neuroimaging', 'biospecimens'],
    },
    'UK_Biobank': {
        'url': 'https://www.ukbiobank.ac.uk',
        'n_samples': '~500000',
        'metabolomics_subset': '~120000',
        'filters': ['ICD10_F90_ADHD', 'ICD10_F84_autism']
    }
}

# Metabolomics-specific datasets
metabolomics_datasets = {
    'MetaboLights': 'ADHD and autism studies',
    'Metabolomics_Workbench': 'Public metabolomics data',
    'HMDB': 'Reference ranges for metabolites'
}
```

### 1.2 Comprehensive Variable Catalogue

#### A. Prenatal (Maternal & In-Utero)

```python
prenatal_features = {
    'maternal_conditions': [
        'depression', 'anxiety', 'autoimmune_disease', 'diabetes',
        'gestational_diabetes', 'thyroid_disorder', 'preeclampsia',
        'TORCH_infection', 'influenza', 'COVID', 'fever',
        'BMI', 'weight_gain'
    ],
    'medications_exposures': [
        'SSRIs', 'SNRIs', 'benzodiazepines', 'valproate',
        'antipsychotics', 'stimulants', 'opioids', 'nicotine',
        'alcohol', 'cannabis', 'caffeine', 'acetaminophen',
        'antibiotics', 'folate', 'B12', 'choline', 'iron',
        'vitamin_D', 'omega3'
    ],
    'toxicants': [
        'lead', 'mercury', 'arsenic', 'PFAS', 'BPA', 'phthalates',
        'pesticides', 'PM25', 'NO2', 'environmental_noise',
        'night_shift', 'light_at_night'
    ],
    'stress_trauma': [
        'life_events', 'IPV', 'SES_stress', 'cortisol_measures'
    ],
    'genetics_epigenetics': [
        'maternal_fetal_genotype', 'imprinting_loci',
        'placental_methylation'
    ],
    'obstetric': [
        'IVF_ART', 'parity', 'multiples'
    ]
}
```

#### B. Perinatal/Neonatal

```python
perinatal_features = {
    'delivery': [
        'C_section_elective', 'C_section_emergency', 'instrumented',
        'anesthesia', 'meconium', 'Apgar_1min', 'Apgar_5min',
        'cord_gases'
    ],
    'birth_metrics': [
        'gestational_age', 'SGA', 'LGA', 'birthweight_z',
        'head_circumference'
    ],
    'NICU': [
        'respiratory_distress', 'hypoxia', 'CPAP', 'ventilation',
        'sepsis', 'jaundice', 'phototherapy', 'antibiotics_days',
        'steroids', 'CRP_neonatal', 'IL6_neonatal', 'ferritin_neonatal'
    ],
    'feeding': [
        'immediate_breastfeeding', 'formula', 'breastfeeding_duration',
        'breastfeeding_exclusivity', 'fortifiers', 'probiotics',
        'early_solids'
    ],
    'early_procedures': [
        'circumcision', 'analgesia', 'vaccination_timing'
    ],
    'toxicology': [
        'neonatal_opioid_screen', 'cocaine', 'THC', 'nicotine'
    ]
}
```

#### C. Early Childhood → Adolescence

```python
childhood_features = {
    'illness_burden': [
        'recurrent_infections', 'fever_frequency', 'ear_infections',
        'asthma', 'allergy', 'eczema', 'autoimmunity', 'celiac',
        'IBD', 'MCAS_suspicion', 'POTS', 'dysautonomia', 'EDS',
        'hypermobility'
    ],
    'endocrine_metabolic': [
        'TSH', 'FT4', 'ferritin', 'TSAT', 'B12', 'folate',
        'vitamin_D', 'carnitine', 'acyl_carnitines', 'ammonia',
        'organic_acids', 'lactate_pyruvate_ratio', 'peroxisomal_markers',
        'copper', 'ceruloplasmin', 'urea_cycle_flags'
    ],
    'sleep': [
        'sleep_duration', 'sleep_latency', 'apnea', 'PLMS',
        'circadian_chronotype', 'actigraphy'
    ],
    'diet': [
        'protein_intake', 'tryptophan_intake', 'tyrosine_intake',
        'elimination_diets', 'ultra_processed_foods', 'omega3_supplement',
        'magnesium', 'NAC'
    ],
    'activity': [
        'MVPA', 'sedentary_time'
    ],
    'substances': [
        'nicotine', 'vape', 'alcohol', 'cannabis', 'caffeine',
        'energy_drinks'
    ],
    'education_psychosocial': [
        'SES', 'school_supports', 'IEP', 'ACEs', 'trauma'
    ],
    'medications': [
        'stimulant_type', 'stimulant_dose', 'stimulant_timing',
        'stimulant_holidays', 'atomoxetine', 'guanfacine', 'clonidine',
        'SSRIs', 'SNRIs', 'antipsychotics', 'anticonvulsants',
        'washout_timing', 'last_dose_hours'
    ]
}
```

#### D. Clinical Phenotyping (Augmented)

```python
clinical_features = {
    'core_instruments': [
        'ADOS_score', 'ADI_R_scores', 'SRS_score',
        'ADHD_RS_inattentive', 'ADHD_RS_hyperactive',
        'Conners_scores', 'CBCL_scores', 'RBS_R',
        'SP2_total', 'SPM', 'Vineland', 'BRIEF_2',
        'PSQI', 'GAD_7', 'PHQ_9', 'CGI_S', 'CGI_I', 'EQ_5D'
    ],
    'cognitive': [
        'IQ_full', 'IQ_verbal', 'IQ_nonverbal',
        'working_memory', 'processing_speed'
    ],
    'comorbidities_expanded': [
        'tics', 'Tourette', 'OCD', 'dyslexia', 'dyscalculia',
        'migraine', 'epilepsy_subtype', 'hypermobile_spectrum',
        'POTS', 'MCAS', 'IBS_C', 'IBS_D', 'IBS_M',
        'functional_constipation', 'GERD', 'enuresis',
        'anxiety', 'depression', 'sleep_disorder'
    ],
    'developmental': [
        'age_first_words', 'age_walking', 'regression_specifics',
        'prenatal_milestones', 'pubertal_stage_Tanner'
    ],
    'family_history': [
        'ADHD_family', 'autism_family', 'mood_disorder_family',
        'psychosis_family', 'epilepsy_family', 'autoimmune_family',
        'EDS_family', 'POTS_family', 'thyroid_family',
        'metabolic_family'
    ],
    'outcomes': [
        'school_attendance', 'school_suspension', 'grades',
        'service_utilization', 'quality_of_life'
    ],
    'demographics': [
        'age', 'sex', 'BMI', 'ethnicity'
    ],
    'context': [
        'fasting', 'collection_clock_time', 'storage_days',
        'fever_72h', 'menstrual_phase', 'recent_illness',
        'antibiotics_30d'
    ]
}
```

#### E. Multi-Omics Layers (Expanded)

```python
# Genetic features (dimension ~2000)
genetic_features = {
    'neurotransmitter_synthesis': [
        'TPH1', 'TPH2',  # Serotonin synthesis
        'TH', 'DDC',      # Dopamine synthesis
        'GAD1', 'GAD2',   # GABA synthesis
        'COMT', 'MAOA', 'MAOB',  # Catabolism
    ],
    'neurotransmitter_transport': [
        'SLC6A4',  # Serotonin transporter
        'SLC6A3',  # Dopamine transporter
        'SLC6A1',  # GABA transporter
    ],
    'neurotransmitter_receptors': [
        'DRD1', 'DRD2', 'DRD3', 'DRD4', 'DRD5',
        'HTR1A', 'HTR1B', 'HTR2A', 'HTR2C', 'HTR7',
        'GABRA1', 'GABRA2', 'GABRA3', 'GABRA4', 'GABRA5', 'GABRA6',
        'GABRB1', 'GABRB2', 'GABRB3'
    ],
    'autism_risk_genes': [
        'SHANK3', 'CHD8', 'SCN2A', 'SYNGAP1',  # High confidence
        # ... top 100 SFARI genes
    ],
    'polygenic_risk_scores': [
        'PRS_autism', 'PRS_ADHD', 'PRS_depression',
        'PRS_schizophrenia', 'PRS_educational_attainment',
        'PRS_BMI', 'PRS_smoking', 'PRS_inflammation_CRP',
        'PRS_sleep'
    ],
    'copy_number_variants': ['CNV_burden', 'specific_CNV_regions'],
    'rare_variants': ['LoF_burden', 'missense_burden', 'de_novo_variants'],
    'mitochondrial': ['mtDNA_CN', 'OXPHOS_gene_sets', 'mtGenome_variants'],
    'structural': ['SV_burden', 'repeat_expansions'],
    'epigenetic': ['HLA_types', 'imprinted_loci']
}

# Metabolomic features (dimension ~300)
metabolomic_features = {
    'neurotransmitters': [
        'serotonin', '5-HIAA', 'melatonin',
        'dopamine', 'DOPAC', 'HVA', '3-MT',
        'norepinephrine', 'epinephrine', 'MHPG',
        'GABA', 'glutamate', 'glutamine'
    ],
    'neurotransmitter_precursors': [
        'tryptophan', 'tyrosine', 'phenylalanine'
    ],
    'kynurenine_pathway': [
        'kynurenine', 'kynurenic_acid', 'quinolinic_acid',
        '3-hydroxykynurenine', 'xanthurenic_acid'
    ],
    'energy_metabolism': [
        'lactate', 'pyruvate', 'citrate', 'succinate',
        'alpha-ketoglutarate', 'malate'
    ],
    'oxidative_stress': [
        'glutathione', 'GSSG', '8-OHdG', 'MDA'
    ],
    'fatty_acids': [
        'omega3_omega6_ratio', 'DHA', 'EPA', 'AA'
    ],
    'gut_metabolites': [
        'SCFA_acetate', 'SCFA_butyrate', 'SCFA_propionate',
        'indoles', 'p_cresol', 'hippurate'
    ],
    'inflammatory_markers': [
        'IL1b', 'IL6', 'IL10', 'IFNg', 'TNFalpha', 'CRP',
        'neopterin', 'hs_CRP', 'IgE'
    ],
    'acyl_carnitines': [
        'C0', 'C2', 'C3', 'C4', 'C5', 'C8', 'C14', 'C16', 'C18'
    ],
    'bile_acids': [
        'CA', 'CDCA', 'DCA', 'LCA', 'UDCA',
        'taurine_conjugated', 'glycine_conjugated'
    ],
    'steroids': [
        'cortisol', 'cortisone', 'DHEA_S', 'testosterone', 'estradiol'
    ],
    'polyamines': [
        'putrescine', 'spermidine', 'spermine'
    ],
    'purines': [
        'adenosine', 'inosine', 'uric_acid', 'hypoxanthine'
    ],
    'urea_cycle': [
        'ornithine', 'citrulline', 'arginine', 'urea', 'ammonia'
    ],
    'micrometabolites': [
        'TMAO', 'indoxyl_sulfate', 'p_cresyl_sulfate'
    ],
    'endocrine': [
        'TSH', 'FT4', 'insulin', 'HOMA_IR', 'leptin', 'adiponectin'
    ]
}

# Microbiome features
microbiome_features = {
    'diversity': ['alpha_diversity', 'beta_diversity', 'enterotypes'],
    'taxa': ['phylum_level', 'genus_level', 'species_level'],
    'functional': ['SCFA_production', 'indole_production',
                   'p_cresol_production', 'TMAO_production'],
    'context': ['antibiotic_exposure', 'PPI_use']
}

# Neuroimaging/EEG features
neuro_features = {
    'EEG': [
        'delta_power', 'theta_power', 'alpha_power', 'beta_power',
        'gamma_power', 'aperiodic_1f_slope', 'alpha_peak_frequency',
        'event_related_potentials'
    ],
    'MRI_structural': [
        'total_brain_volume', 'gray_matter_volume',
        'white_matter_volume', 'cerebellar_volume',
        'cortical_thickness', 'surface_area'
    ],
    'MRI_diffusion': [
        'FA', 'MD', 'RD', 'AD', 'tract_integrity'
    ],
    'MRI_functional': [
        'thalamocortical_connectivity', 'DMN_connectivity',
        'salience_network', 'executive_network'
    ]
}
```

### 1.3 Data Harmonization

```python
# Ontology mapping
ontologies = {
    'clinical': 'HPO/SNOMED/ICD-10',
    'diet': 'FNDDS',
    'medications': 'RxNorm/ATC',
    'exposures': 'ExO',
    'assays': 'OBI'
}

# Batch/Site correction
def harmonize_data(data, metadata):
    """
    Comprehensive data harmonization
    """
    from combat.pycombat import pycombat

    # ComBat for batch correction
    harmonized = pycombat(data, metadata['batch'])

    # Include site × platform random effects
    # ... implementation

    return harmonized

# Context tags
context_variables = [
    'fasting_status', 'time_of_day', 'menstrual_phase',
    'last_dose_timing', 'recent_illness', 'recent_fever',
    'recent_antibiotics', 'sample_storage_time', 'freeze_thaw_cycles'
]

# Missingness handling
def handle_missingness(data):
    """
    Tag MAR/MNAR; pattern-mixture sensitivity;
    delta-adjusted MICE; inverse probability weighting
    """
    # ... implementation
    pass

# Ancestry
def ancestry_control(genetic_data):
    """
    Calculate genetic PCs; within-ancestry discovery +
    across-ancestry replication
    """
    # ... implementation
    pass
```

### 1.4 QC & Preprocessing

```python
def genomics_qc(vcf_data):
    """
    Standard VQSR; Mendelian consistency; duplicate/relatedness;
    sex checks; mtDNA heteroplasmy QC
    """
    # ... implementation
    pass

def metabolomics_qc(metabolite_data):
    """
    Blank removal, pooled QC drift correction (LOESS),
    batch/plate adjustment, CV filters, isobaric resolution
    """
    # ... implementation
    pass

def microbiome_qc(microbiome_data):
    """
    Read depth thresholds, contaminant removal (decontam),
    rarefaction/CLR transforms
    """
    # ... implementation
    pass

def protein_qc(protein_data):
    """
    LOD/LOQ handling; hook effect checks
    """
    # ... implementation
    pass

def imaging_qc(imaging_data):
    """
    Motion scrubbing; head-coil/site harmonization
    """
    # ... implementation
    pass
```

## Phase 2: Data Integration and Preprocessing (Days 3-4)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.decomposition import PCA
import scanpy as sc
from combat.pycombat import pycombat

def integrate_multiomics_enhanced(genetic_df, metabolomic_df, clinical_df,
                                  context_df, metadata):
    """
    Enhanced integration with context covariates and confound removal
    """
    # Partial out context covariates using linear mixed models
    from statsmodels.regression.mixed_linear_model import MixedLM

    # Add context covariates (fasting, time-of-day, last_dose_hours, storage_days)
    # and partial out with site as random effect

    # Delta-adjusted MICE with missingness indicators
    from sklearn.experimental import enable_iterative_imputer
    genetic_imputed = IterativeImputer(add_indicator=True).fit_transform(genetic_df)
    metabolomic_imputed = IterativeImputer(add_indicator=True).fit_transform(metabolomic_df)

    # Different scaling strategies
    genetic_scaled = StandardScaler().fit_transform(genetic_imputed)
    metabolomic_scaled = RobustScaler().fit_transform(metabolomic_imputed)
    clinical_scaled = StandardScaler().fit_transform(clinical_df)

    # Weight different data types
    weights = {
        'genetic': 0.25,
        'metabolomic': 0.35,  # Highest weight - most proximal to phenotype
        'clinical': 0.25,
        'microbiome': 0.15
    }

    # Concatenate with weighting
    integrated = np.hstack([
        genetic_scaled * weights['genetic'],
        metabolomic_scaled * weights['metabolomic'],
        clinical_scaled * weights['clinical']
    ])

    return integrated

# Advanced multi-omics integration methods
from mofapy2.run.entry_point import entry_point

def mofa2_integration(data_dict):
    """
    Use MOFA2 for principled multi-omics integration
    """
    ent = entry_point()
    ent.set_data_options(scale_groups=False, scale_views=True)
    ent.set_data_matrix(data_dict)
    ent.set_model_options(factors=50)
    ent.build()
    ent.run()

    return ent.model.get_factors()

# Supervised/semi-supervised latent factors with domain adversarial site removal
def vae_integration(data_dict, site_labels):
    """
    VAE with domain adversarial site removal
    """
    # ... implementation with PyTorch/TensorFlow
    pass
```

## Phase 3: Dimensionality Reduction and Clustering (Day 5)

```python
from sklearn.manifold import TSNE
from umap import UMAP
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_embedding(integrated_data, labels):
    """
    Multiple dimensionality reduction approaches with consensus
    """
    embeddings = {}

    # t-SNE with multiple perplexities
    for perplexity in [10, 30, 50, 100]:
        embeddings[f'tsne_p{perplexity}'] = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=2000,
            init='pca'
        ).fit_transform(integrated_data)

    # UMAP with different parameters
    for n_neighbors in [15, 30, 50]:
        for min_dist in [0.1, 0.25, 0.5]:
            embeddings[f'umap_n{n_neighbors}_d{min_dist}'] = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=2
            ).fit_transform(integrated_data)

    # 3D embeddings for rotation analysis
    embeddings['tsne_3d'] = TSNE(n_components=3).fit_transform(integrated_data)
    embeddings['umap_3d'] = UMAP(n_components=3).fit_transform(integrated_data)

    return embeddings

def consensus_clustering(embeddings):
    """
    Build consensus clustering across embeddings
    Aggregate co-assignment matrix → spectral clustering
    """
    from sklearn.cluster import SpectralClustering

    n_samples = list(embeddings.values())[0].shape[0]
    co_assignment = np.zeros((n_samples, n_samples))

    for embedding in embeddings.values():
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
        labels = clusterer.fit_predict(embedding)

        # Update co-assignment matrix
        for i in range(n_samples):
            for j in range(n_samples):
                if labels[i] == labels[j] and labels[i] != -1:
                    co_assignment[i, j] += 1

    # Normalize
    co_assignment /= len(embeddings)

    # Spectral clustering on consensus
    spectral = SpectralClustering(n_clusters=None, affinity='precomputed')
    consensus_labels = spectral.fit_predict(co_assignment)

    return consensus_labels

def hierarchical_clustering(embedded_data):
    """
    Multi-level clustering: HDBSCAN + LCA + Dirichlet process mixtures
    """
    # First level: HDBSCAN for main clusters
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
    main_clusters = clusterer.fit_predict(embedded_data)

    # Latent class analysis
    from sklearn.mixture import BayesianGaussianMixture
    lca = BayesianGaussianMixture(n_components=10,
                                  weight_concentration_prior_type='dirichlet_process')
    lca_clusters = lca.fit_predict(embedded_data)

    # Second level: subclusters within main clusters
    subclusters = {}
    for cluster_id in np.unique(main_clusters):
        if cluster_id == -1:  # Skip noise
            continue
        mask = main_clusters == cluster_id
        sub_data = embedded_data[mask]

        sub_clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
        subclusters[cluster_id] = sub_clusterer.fit_predict(sub_data)

    return main_clusters, subclusters, lca_clusters

def topology_analysis(embedded_data):
    """
    k-NN graph connectivity, spectral gaps
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import connected_components

    # k-NN graph
    knn_graph = kneighbors_graph(embedded_data, n_neighbors=15, mode='connectivity')
    n_components, labels = connected_components(knn_graph, directed=False)

    # Spectral gaps
    from scipy.sparse.linalg import eigsh
    L = knn_graph.toarray()
    D = np.diag(np.sum(L, axis=1))
    laplacian = D - L
    eigenvalues, _ = eigsh(laplacian, k=10, which='SM')
    spectral_gap = np.diff(eigenvalues)

    return n_components, spectral_gap
```

## Phase 4: Cluster Validation and Characterization (Day 6)

```python
from scipy import stats
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import pingouin as pg

def validate_clusters(data, clusters, metadata):
    """
    Comprehensive cluster validation with stability and biological validity
    """
    validation_results = {}

    # 1. Internal validation metrics
    validation_results['silhouette'] = silhouette_score(data, clusters)
    validation_results['davies_bouldin'] = davies_bouldin_score(data, clusters)

    # 2. Stability analysis - bootstrap
    stability_scores = []
    for _ in range(100):
        idx = np.random.choice(len(data), len(data), replace=True)
        resampled_data = data[idx]
        new_clusters = hdbscan.HDBSCAN().fit_predict(resampled_data)
        stability_scores.append(adjusted_rand_score(clusters[idx], new_clusters))
    validation_results['stability'] = np.mean(stability_scores)

    # 3. Biological validity - test key hypotheses
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            continue
        mask = clusters == cluster_id

        # Test metabolite differences
        for metabolite in ['serotonin', 'dopamine', 'GABA', 'kynurenine']:
            if metabolite in metadata.columns:
                in_cluster = metadata.loc[mask, metabolite]
                out_cluster = metadata.loc[~mask, metabolite]
                stat, pval = stats.mannwhitneyu(in_cluster, out_cluster)
                validation_results[f'cluster_{cluster_id}_{metabolite}_pval'] = pval

        # Test genetic burden
        genetic_burden = metadata.loc[mask, 'rare_variant_burden'].mean()
        validation_results[f'cluster_{cluster_id}_genetic_burden'] = genetic_burden

        # Test clinical severity
        severity_score = metadata.loc[mask, 'total_severity'].mean()
        validation_results[f'cluster_{cluster_id}_severity'] = severity_score

    return validation_results

def characterize_clusters(data, clusters, feature_names):
    """
    Detailed characterization with SHAP values
    """
    import shap
    from sklearn.ensemble import RandomForestClassifier

    cluster_profiles = {}

    # Train classifier to predict cluster membership
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(data, clusters)

    # SHAP values for explainability
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(data)

    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            continue

        # Get SHAP values for this cluster
        cluster_shap = shap_values[cluster_id]
        feature_importance = np.abs(cluster_shap).mean(axis=0)

        # Sort features by importance
        top_features = sorted(zip(feature_names, feature_importance),
                            key=lambda x: x[1], reverse=True)[:20]
        cluster_profiles[cluster_id] = top_features

    return cluster_profiles

def cohen_d(x, y):
    """Calculate Cohen's d effect size"""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
```

## Phase 5: Gap Analysis and Discontinuity Testing (Day 7)

```python
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from ripser import ripser
from persim import plot_diagrams

def test_discontinuity(embedded_data, clusters):
    """
    Test for gaps between clusters - separation vs spectrum
    """
    # 1. Density-based gap detection
    from sklearn.neighbors import KernelDensity

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde.fit(embedded_data)

    # Create grid
    x_min, x_max = embedded_data[:, 0].min() - 1, embedded_data[:, 0].max() + 1
    y_min, y_max = embedded_data[:, 1].min() - 1, embedded_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                          np.linspace(y_min, y_max, 100))

    # Evaluate density
    Z = kde.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Find valleys (gaps)
    threshold = np.percentile(Z, 10)
    gaps = Z < threshold

    # 2. Minimum spanning tree analysis
    dist_matrix = squareform(pdist(embedded_data))
    mst = minimum_spanning_tree(dist_matrix)

    # Find edges connecting different clusters
    between_cluster_edges = []
    within_cluster_edges = []

    for i, j in zip(*mst.nonzero()):
        if clusters[i] != clusters[j] and clusters[i] != -1 and clusters[j] != -1:
            between_cluster_edges.append(mst[i, j])
        elif clusters[i] == clusters[j] and clusters[i] != -1:
            within_cluster_edges.append(mst[i, j])

    # Gap score: large between-cluster edges indicate gaps
    gap_score = np.mean(between_cluster_edges) / np.mean(within_cluster_edges) if within_cluster_edges else 0

    # 3. Topological data analysis
    diagrams = ripser(embedded_data)['dgms']

    return {
        'density_gaps': gaps,
        'gap_score': gap_score,
        'persistence_diagrams': diagrams,
        'mst_between_mean': np.mean(between_cluster_edges) if between_cluster_edges else 0,
        'mst_within_mean': np.mean(within_cluster_edges) if within_cluster_edges else 0
    }
```

## Phase 6: Causal Analysis and Biological Interpretation (Days 8-9)

```python
def causal_analysis(data, metadata):
    """
    DAG-informed adjustment, Mendelian randomization, mediation, G×E
    """
    import dowhy
    from econml.dml import CausalForestDML

    # 1. DAG construction (prenatal → neonatal → biology → phenotype)
    causal_graph = """
    digraph {
        prenatal -> neonatal;
        prenatal -> biology;
        neonatal -> biology;
        biology -> phenotype;
        genetics -> biology;
        genetics -> phenotype;
        meds -> biology [style=dashed];
    }
    """

    # 2. Mendelian randomization
    # Instrument exposures like CRP, BMI, smoking, lipids with PRS
    def mendelian_randomization(exposure_prs, exposure, outcome):
        """
        Two-stage least squares for MR
        """
        from statsmodels.sandbox.regression.gmm import IV2SLS

        # First stage: PRS → exposure
        # Second stage: predicted exposure → outcome
        model = IV2SLS(outcome, exposure, instrument=exposure_prs)
        results = model.fit()
        return results

    # 3. Mediation analysis
    # PRS → microbiome/metabolites → symptoms
    def mediation_analysis(predictor, mediator, outcome):
        """
        Test if mediator explains predictor→outcome relationship
        """
        import pingouin as pg
        results = pg.mediation_analysis(data=metadata,
                                       x=predictor,
                                       m=mediator,
                                       y=outcome)
        return results

    # 4. G×E interactions with double ML
    def gxe_interactions(genetics, environment, outcome):
        """
        Causal forest for heterogeneous treatment effects
        """
        cf = CausalForestDML(model_y=None, model_t=None)
        cf.fit(Y=outcome, T=environment, X=genetics)
        te = cf.effect(genetics)  # Heterogeneous treatment effects
        return te

    return {
        'mr_results': mendelian_randomization,
        'mediation_results': mediation_analysis,
        'gxe_effects': gxe_interactions
    }

def pathway_enrichment(cluster_profiles, gene_sets):
    """
    Test if clusters map to known biological pathways
    """
    import gseapy as gp

    enrichment_results = {}
    for cluster_id, profile in cluster_profiles.items():
        # Get top genes for this cluster
        top_genes = [f for f, score in profile if isinstance(f, str)]

        # Run enrichment
        enr = gp.enrichr(gene_list=top_genes,
                        gene_sets=['KEGG_2021_Human',
                                  'GO_Biological_Process_2021',
                                  'WikiPathway_2021_Human',
                                  'Reactome_2022'],
                        outdir=f'cluster_{cluster_id}')
        enrichment_results[cluster_id] = enr.results

    return enrichment_results

def metabolic_network_analysis(metabolite_profiles):
    """
    Map metabolite changes to metabolic networks
    """
    import networkx as nx

    # Build metabolic network from KEGG
    network = nx.DiGraph()

    for cluster_id, metabolites in metabolite_profiles.items():
        altered_metabolites = [m for m, score in metabolites if abs(score) > 1]

        # Find connecting pathways
        # (simplified - would use actual KEGG API)
        subnetwork = network.subgraph(altered_metabolites)

        # Key regulatory nodes
        centrality = nx.betweenness_centrality(subnetwork)
        key_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

    return key_nodes

def clinical_correlation(clusters, clinical_data):
    """
    Test clinical relevance and treatment response prediction
    """
    correlations = {}

    # Treatment response
    if 'medication_response' in clinical_data:
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            response_rate = clinical_data.loc[mask, 'medication_response'].mean()
            correlations[f'cluster_{cluster_id}_med_response'] = response_rate

    # Prognosis
    if 'outcome_score' in clinical_data:
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            outcome = clinical_data.loc[mask, 'outcome_score'].mean()
            correlations[f'cluster_{cluster_id}_outcome'] = outcome

    # Comorbidity patterns
    comorbidities = ['GI_issues', 'epilepsy', 'anxiety', 'sleep_disorder',
                     'POTS', 'MCAS', 'tics']
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        comorbidity_profile = clinical_data.loc[mask, comorbidities].mean()
        correlations[f'cluster_{cluster_id}_comorbidities'] = comorbidity_profile.to_dict()

    return correlations
```

## Phase 7: Sensitivity Analyses and Confound Control (Day 10)

```python
def sensitivity_analyses(data, clusters, metadata):
    """
    Comprehensive sensitivity analyses for robustness
    """
    sensitivity_results = {}

    # 1. Essential covariates adjustment
    essential_covariates = [
        'age', 'sex', 'pubertal_stage', 'ancestry_PCs', 'site',
        'SES', 'fasting', 'time_of_day', 'BMI', 'sleep_duration',
        'recent_infection', 'recent_fever', 'antibiotic_use',
        'last_medication_dose', 'sample_storage_time'
    ]

    # 2. Negative controls
    # Metabolites not plausibly linked
    # Outcomes temporally prior to exposure

    # 3. E-value for unmeasured confounding
    def calculate_e_value(RR):
        """
        Calculate E-value: minimum strength of unmeasured confounder
        needed to explain away observed association
        """
        return RR + np.sqrt(RR * (RR - 1))

    # 4. MNAR missingness sensitivity
    # Pattern-mixture models

    # 5. Medication washout sensitivity
    def washout_sensitivity(data_on_med, data_off_med):
        """
        Compare paired on/off medication contrasts
        """
        from scipy.stats import wilcoxon
        stat, pval = wilcoxon(data_on_med, data_off_med)
        return {'statistic': stat, 'pvalue': pval}

    # 6. Sibling/within-family designs
    def within_family_analysis(data, family_ids):
        """
        Control for unmeasured family-level confounders
        """
        from statsmodels.regression.mixed_linear_model import MixedLM
        model = MixedLM(data, groups=family_ids)
        results = model.fit()
        return results

    # 7. Leave-site-out cross-validation
    def leave_site_out_cv(data, sites, labels):
        """
        Test generalization across sites
        """
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()

        scores = []
        for train_idx, test_idx in logo.split(data, labels, groups=sites):
            # Train on all sites except one
            # Test on held-out site
            # ... implementation
            pass

        return np.mean(scores)

    return sensitivity_results
```

## Phase 8: Visualization and Reporting (Day 11)

```python
import plotly.graph_objects as go
import plotly.express as px
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool

def create_interactive_visualization(embedded_data, metadata, clusters):
    """
    Comprehensive interactive visualizations with clinical context
    """
    # Main t-SNE/UMAP plot with multiple overlays
    fig = go.Figure()

    overlays = ['diagnosis', 'serotonin_level', 'dopamine_level',
                'genetic_burden', 'GI_issues', 'cluster',
                'medication_response', 'comorbidities']

    for overlay in overlays:
        fig.add_trace(go.Scatter(
            x=embedded_data[:, 0],
            y=embedded_data[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=metadata[overlay],
                colorscale='Viridis',
                showscale=True
            ),
            text=metadata[['sample_id', 'diagnosis', overlay]].apply(
                lambda x: f"ID: {x[0]}<br>Dx: {x[1]}<br>{overlay}: {x[2]}", axis=1
            ),
            hovertemplate='%{text}',
            name=overlay,
            visible=(overlay == 'cluster')
        ))

    # Buttons to switch overlays
    fig.update_layout(
        updatemenus=[
            dict(buttons=[
                dict(label=overlay,
                     method="update",
                     args=[{"visible": [o == overlay for o in overlays]}])
                for overlay in overlays
            ])
        ]
    )

    return fig

def generate_comprehensive_report(results, cluster_profiles, validation_results):
    """
    Generate report with executive summary and clinical care maps
    """
    from jinja2 import Template

    template = Template("""
    # Multi-Omics Analysis of ADHD/Autism Subtypes

    ## Executive Summary
    - Identified {{ n_main_clusters }} main clusters with {{ n_subclusters }} subclusters
    - Gap score: {{ gap_score }} (>1.5 indicates clear separation)
    - Stability: {{ stability }} (>0.7 indicates robust clustering)
    - Clinical improvement: +{{ clinical_improvement }}% in treatment response prediction

    ## Confounding Section
    ### DAGs and Adjustment Sets
    {{ dag_description }}

    ### Sensitivity Analyses
    - E-value: {{ e_value }}
    - Leave-site-out CV: {{ loo_cv }}
    - Medication washout: {{ washout_pval }}

    ## Main Findings

    ### Cluster 1: Neurotransmitter Dysfunction (Serotonin-dominant)
    - N = {{ cluster1_n }}
    - Key features: {{ cluster1_features }}
    - Metabolite profile: Low serotonin (p={{ serotonin_p }}), low 5-HIAA
    - Genetic enrichment: TPH2, SLC6A4 variants
    - Clinical: High anxiety ({{ anxiety_percent }}%), GI issues ({{ gi_percent }}%)
    - **Care map**: SSRI trial, GI workup, omega-3/magnesium

    ### Cluster 2: Neurotransmitter Dysfunction (Dopamine-dominant)
    - N = {{ cluster2_n }}
    - Key features: {{ cluster2_features }}
    - Metabolite profile: Low dopamine metabolites, altered DOPAC/HVA ratio
    - Genetic enrichment: DRD4, SLC6A3 variants
    - Clinical: High ADHD-hyperactive ({{ adhd_h_percent }}%)
    - **Care map**: Stimulant trial, iron/ferritin check

    ### Cluster 3: Neurodevelopmental Disruption
    - N = {{ cluster3_n }}
    - Key features: {{ cluster3_features }}
    - Metabolite profile: Normal neurotransmitters
    - Genetic enrichment: Rare de novo variants (CHD8, SHANK3, SCN2A)
    - Clinical: Intellectual disability ({{ id_percent }}%), early diagnosis
    - **Care map**: Genetic counseling, specialized education, seizure monitoring

    ### Cluster 4: Immune-Inflammatory
    - N = {{ cluster4_n }}
    - Key features: Elevated IL-6, TNF-α, CRP, neopterin
    - Clinical: Autoimmune comorbidity, regression after illness
    - **Care map**: Immune workup, anti-inflammatory diet, monitor autoantibodies

    ### Cluster 5: Mitochondrial/Metabolic
    - N = {{ cluster5_n }}
    - Key features: Elevated lactate, abnormal acyl-carnitines
    - Clinical: Fatigue, regression with illness
    - **Care map**: Metabolic workup, carnitine/CoQ10, avoid valproate

    ### Cluster 6: Gut-Brain/Microbiome
    - N = {{ cluster6_n }}
    - Key features: Low SCFA, high p-cresol, dysbiosis
    - Clinical: Severe GI issues, dietary selectivity
    - **Care map**: GI specialist, probiotic trial, dietary intervention

    ## Validation Metrics
    - Silhouette score: {{ silhouette }}
    - Davies-Bouldin: {{ davies_bouldin }}
    - Stability (bootstrap): {{ stability }}
    - External replication: {{ replication_score }}

    ## Biological Pathways
    {{ pathway_results }}

    ## Clinical Decision Support Table

    | Subtype | First-line Medications | Labs to Check | Referrals |
    |---------|----------------------|---------------|-----------|
    | Serotonin-dominant | SSRI, omega-3 | Serotonin, 5-HIAA, ferritin | GI, psychiatry |
    | Dopamine-dominant | Stimulants | Ferritin, TSAT, dopamine metabolites | None typically |
    | Neurodevelopmental | Per phenotype | Genetic panel, EEG | Genetics, neurology |
    | Immune-inflammatory | Anti-inflammatory | CRP, IL-6, autoantibodies | Immunology, rheumatology |
    | Mitochondrial | Carnitine, CoQ10 | Lactate, pyruvate, acyl-carnitines | Metabolism |
    | Gut-brain | Probiotics, dietary | Microbiome, SCFA, p-cresol | GI, nutrition |

    ## Risk Stratification

    | Subtype | High-risk Comorbidities | Monitoring |
    |---------|------------------------|------------|
    | Serotonin-dominant | Depression, anxiety, suicidality | Mental health screening q6mo |
    | Dopamine-dominant | Tics, substance use | Tic monitoring, substance screening |
    | Neurodevelopmental | Seizures, regression | EEG, developmental assessments |
    | Immune-inflammatory | Autoimmune, PANS/PANDAS | Infection tracking, immune labs |
    | Mitochondrial | Regression with illness | Metabolic crisis plan |
    | Gut-brain | Malnutrition, FTT | Growth monitoring, nutrition |

    ## Limitations and Future Directions
    - Replication needed in independent cohorts
    - Longitudinal validation for trajectory prediction
    - Integration of additional omics layers (epigenome, single-cell)
    - Clinical trial design for subtype-specific interventions
    """)

    return template.render(**results)
```

## Phase 9: External Validation and Replication (Day 12)

```python
def external_validation(model, new_data, new_metadata):
    """
    Validate in independent cohort with cross-ancestry replication
    """
    # Project new data into existing embedding space
    new_embedded = model.transform(new_data)

    # Assign to nearest cluster
    from sklearn.neighbors import NearestCentroid
    clf = NearestCentroid()
    clf.fit(model.embedding_, model.labels_)
    predicted_clusters = clf.predict(new_embedded)

    # Test if cluster characteristics hold
    validation_metrics = validate_clusters(new_data, predicted_clusters, new_metadata)

    # Cross-cohort replication
    def compare_cluster_profiles(original, new):
        """
        Compare cluster characteristics across cohorts
        """
        correlations = []
        for cluster_id in np.unique(original.keys()):
            orig_features = dict(original[cluster_id])
            new_features = dict(new[cluster_id])

            # Correlate feature importance
            shared_features = set(orig_features.keys()) & set(new_features.keys())
            if len(shared_features) > 0:
                orig_vals = [orig_features[f] for f in shared_features]
                new_vals = [new_features[f] for f in shared_features]
                corr = np.corrcoef(orig_vals, new_vals)[0, 1]
                correlations.append(corr)

        return np.mean(correlations)

    # Ancestry-stratified validation
    ancestries = new_metadata['ancestry'].unique()
    ancestry_results = {}

    for ancestry in ancestries:
        mask = new_metadata['ancestry'] == ancestry
        ancestry_data = new_data[mask]
        ancestry_clusters = predicted_clusters[mask]
        ancestry_metadata = new_metadata[mask]

        ancestry_results[ancestry] = validate_clusters(
            ancestry_data, ancestry_clusters, ancestry_metadata
        )

    # Prospective validation: baseline subtype → future outcomes
    def prospective_validation(baseline_clusters, followup_outcomes):
        """
        Test if baseline subtype predicts future outcomes
        """
        from sklearn.metrics import roc_auc_score

        auc_scores = {}
        for outcome in followup_outcomes.columns:
            auc = roc_auc_score(followup_outcomes[outcome], baseline_clusters)
            auc_scores[outcome] = auc

        return auc_scores

    return {
        'validation_metrics': validation_metrics,
        'replication_score': compare_cluster_profiles,
        'ancestry_results': ancestry_results,
        'prospective_validation': prospective_validation
    }
```

## Risk Register with Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Batch/site confounding | High | High | ComBat + site random effects; leave-site-out CV |
| Medication/fasting/circadian distortion | High | Medium | Tag & adjust; paired on/off contrasts |
| MNAR missingness | Medium | High | Delta-adjusted MICE + pattern-mixture sensitivity |
| Ancestry confounding | High | Medium | Within-ancestry discovery + cross-ancestry replication |
| Overfitting/leakage | High | Medium | Patient/family/site-level splits; frozen pipelines |
| Clinical irrelevance | High | Low | Prospective validation + decision-curve analysis |
| Small effect sizes | Medium | Medium | Power analysis; focus on large effect subtypes |
| Replication failure | High | Medium | Pre-registration; multi-site validation |

## Power and Multiplicity Control

### Power Analysis
- Effective n per site/ancestry after QC
- Cluster detectability simulation (mixture separation δ vs n)
- Minimum detectable effect size given sample size

### Multiplicity Control
- Hierarchical FDR across omics layers
- Enrichment uses pathway-level correction
- Pre-specified stopping rules: gap score > 1.5, stability > 0.7

## Implementation Milestones (12 Days)

1. **Day 0**: DUA, IRB, ethics, pre-registration
2. **Days 1-2**: Data acquisition, ontology mapping, QC recipes, context tags
3. **Days 3-4**: Integration (MOFA2 + weighted stack), leakage-safe splits, missingness handling
4. **Day 5**: Consensus clustering across embeddings, topology analysis
5. **Day 6**: Cluster validation, characterization, SHAP explanations
6. **Day 7**: Gap analysis, discontinuity testing, spectrum vs separation
7. **Days 8-9**: Causal analysis (MR/mediation/G×E), pathway enrichment, clinical correlation
8. **Day 10**: Sensitivity analyses, confound control, robustness checks
9. **Day 11**: Visualization, interactive plots, comprehensive report generation
10. **Day 12**: External validation, cross-ancestry replication, prospective validation

## Tools and Resources

```bash
# Environment setup
conda create -n multiomics python=3.10
conda activate multiomics

# Core packages
pip install numpy pandas scipy scikit-learn statsmodels
pip install umap-learn hdbscan
pip install scanpy anndata  # Multi-omics integration
pip install plotly bokeh matplotlib seaborn

# Specialized packages
pip install mofapy2  # Multi-Omics Factor Analysis
pip install gseapy  # Pathway analysis
pip install pingouin  # Statistics
pip install ripser persim  # Topological data analysis
pip install shap  # Explainability
pip install pycombat  # Batch correction

# Causal inference
pip install dowhy econml

# For accessing databases
pip install bioservices  # Biological databases
pip install pyega3  # EGA data access

# Microbiome
pip install qiime2 decontam

# Quality control
pip install multiqc
```

## Expected Results Summary

### Primary Findings
1. **Cluster topology**: 6-8 distinct subtypes with clear separation (gap score > 1.5)
2. **Biological basis**: Each subtype maps to distinct molecular mechanisms
3. **Clinical utility**: Subtypes predict treatment response (12-25% improvement over diagnosis-only)
4. **Robustness**: Findings replicate across sites, ancestries, and platforms

### Subtype Predictions
1. **Neurotransmitter-serotonin**: Low serotonin/5-HIAA, TPH2 variants, anxiety/GI → SSRI responsive
2. **Neurotransmitter-dopamine**: Low dopamine metabolites, DRD4/SLC6A3 variants → stimulant responsive
3. **Neurodevelopmental**: Rare de novo variants, early severe → specialized interventions
4. **Immune-inflammatory**: Elevated cytokines, autoimmune → anti-inflammatory approaches
5. **Mitochondrial**: Energy metabolism dysfunction → supplements, avoid valproate
6. **Gut-brain**: Microbiome dysbiosis, severe GI → dietary/probiotic interventions

### Clinical Translation
- Minimal biomarker panel for clinic use
- Decision support algorithms
- Risk stratification for comorbidities
- Treatment response prediction
- Monitoring protocols per subtype

This comprehensive plan addresses all major methodological concerns, incorporates extensive confound control, and provides a clear path to clinically actionable subtypes. The 12-day timeline is aggressive but achievable with proper parallelization and existing public datasets.