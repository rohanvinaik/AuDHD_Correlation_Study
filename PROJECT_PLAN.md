# Multi-Omics Integration Pipeline for ADHD/Autism Subtyping

## Hypothesis
Neurodevelopmental conditions currently diagnosed as ADHD and autism represent multiple biologically distinct conditions that can be identified through integrated analysis of genetics, metabolomics, and clinical phenotypes.

## Expected Outcomes
- **Primary clusters**:
  - Neurotransmitter dysfunction (serotonin/dopamine/GABA axis)
  - Neurodevelopmental disruption (rare variants, intellectual disability)
- **Potential subclusters within primary**:
  - Serotonin-dominant vs dopamine-dominant
  - With/without GI involvement
  - Early vs late onset
  - With/without inflammatory markers

## Phase 1: Data Acquisition and Harmonization (Days 1-3)

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

### 1.2 Feature Extraction

```python
# Genetic features (dimension ~1000)
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
        'DRD1-5', 'HTR1A-7', 'GABRA1-6', 'GABRB1-3'
    ],
    'autism_risk_genes': [
        'SHANK3', 'CHD8', 'SCN2A', 'SYNGAP1', # High confidence
        # ... top 100 SFARI genes
    ],
    'polygenic_risk_scores': [
        'PRS_autism', 'PRS_ADHD', 'PRS_depression',
        'PRS_schizophrenia', 'PRS_educational_attainment'
    ],
    'copy_number_variants': ['CNV_burden', 'specific_CNV_regions'],
    'rare_variants': ['LoF_burden', 'missense_burden']
}

# Metabolomic features (dimension ~200)
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
        'IL6', 'TNFalpha', 'CRP', 'neopterin'
    ]
}

# Clinical features (dimension ~100)
clinical_features = {
    'core_symptoms': [
        'ADOS_score', 'ADI_R_scores', 'SRS_score',
        'ADHD_RS_inattentive', 'ADHD_RS_hyperactive',
        'Conners_scores', 'CBCL_scores'
    ],
    'cognitive': [
        'IQ_full', 'IQ_verbal', 'IQ_nonverbal',
        'working_memory', 'processing_speed'
    ],
    'comorbidities': [
        'GI_issues', 'sleep_problems', 'anxiety',
        'depression', 'epilepsy', 'immune_disorders'
    ],
    'developmental': [
        'age_first_words', 'age_walking', 'regression'
    ],
    'demographics': [
        'age', 'sex', 'BMI', 'ethnicity'
    ]
}
```

## Phase 2: Data Integration and Preprocessing (Day 4)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.decomposition import PCA
import scanpy as sc  # Great for multi-omics

def integrate_multiomics(genetic_df, metabolomic_df, clinical_df):
    """
    Integrate multi-omics data with careful normalization
    """
    # Handle missing data differently per data type
    genetic_imputed = KNNImputer(n_neighbors=10).fit_transform(genetic_df)
    metabolomic_imputed = IterativeImputer().fit_transform(metabolomic_df)

    # Different scaling strategies
    genetic_scaled = StandardScaler().fit_transform(genetic_imputed)
    metabolomic_scaled = RobustScaler().fit_transform(metabolomic_imputed)
    clinical_scaled = StandardScaler().fit_transform(clinical_df)

    # Weight different data types
    weights = {
        'genetic': 0.3,
        'metabolomic': 0.4,  # Highest weight - most proximal to phenotype
        'clinical': 0.3
    }

    # Concatenate with weighting
    integrated = np.hstack([
        genetic_scaled * weights['genetic'],
        metabolomic_scaled * weights['metabolomic'],
        clinical_scaled * weights['clinical']
    ])

    return integrated

# Alternative: Use specialized multi-omics integration methods
from momix import MOFA  # Multi-Omics Factor Analysis
from mixOmics import DIABLO  # Multi-omics integration

def advanced_integration(data_dict):
    """Use MOFA2 for principled multi-omics integration"""
    mofa = MOFA(n_factors=50)
    mofa.fit(data_dict)
    return mofa.get_factors()
```

## Phase 3: Dimensionality Reduction and Clustering (Day 5)

```python
from sklearn.manifold import TSNE
from umap import UMAP
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_embedding(integrated_data, labels):
    """
    Try multiple dimensionality reduction approaches
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

def hierarchical_clustering(embedded_data):
    """
    Multi-level clustering to find main clusters and subclusters
    """
    # First level: find main clusters
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
    main_clusters = clusterer.fit_predict(embedded_data)

    # Second level: find subclusters within main clusters
    subclusters = {}
    for cluster_id in np.unique(main_clusters):
        if cluster_id == -1:  # Skip noise
            continue
        mask = main_clusters == cluster_id
        sub_data = embedded_data[mask]

        # Try to find subclusters
        sub_clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
        subclusters[cluster_id] = sub_clusterer.fit_predict(sub_data)

    return main_clusters, subclusters
```

## Phase 4: Cluster Validation and Characterization (Day 6)

```python
from scipy import stats
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pingouin as pg

def validate_clusters(data, clusters, metadata):
    """
    Comprehensive cluster validation
    """
    validation_results = {}

    # 1. Internal validation metrics
    validation_results['silhouette'] = silhouette_score(data, clusters)
    validation_results['davies_bouldin'] = davies_bouldin_score(data, clusters)

    # 2. Stability analysis - bootstrap
    stability_scores = []
    for _ in range(100):
        # Resample data
        idx = np.random.choice(len(data), len(data), replace=True)
        resampled_data = data[idx]
        # Recluster
        new_clusters = hdbscan.HDBSCAN().fit_predict(resampled_data)
        # Compare clustering
        stability_scores.append(adjusted_rand_score(clusters[idx], new_clusters))
    validation_results['stability'] = np.mean(stability_scores)

    # 3. Biological validity - test key hypotheses
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            continue
        mask = clusters == cluster_id

        # Test metabolite differences
        for metabolite in ['serotonin', 'dopamine', 'GABA']:
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
    Detailed characterization of each cluster
    """
    cluster_profiles = {}

    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            continue

        mask = clusters == cluster_id
        cluster_data = data[mask]
        other_data = data[~mask]

        # Find defining features (highest effect size)
        effect_sizes = []
        for i, feature in enumerate(feature_names):
            d = cohen_d(cluster_data[:, i], other_data[:, i])
            effect_sizes.append((feature, d))

        # Sort by absolute effect size
        effect_sizes.sort(key=lambda x: abs(x[1]), reverse=True)
        cluster_profiles[cluster_id] = effect_sizes[:20]  # Top 20 features

    return cluster_profiles
```

## Phase 5: Gap Analysis and Discontinuity Testing (Day 7)

```python
from scipy.spatial.distance import pdist, squareform
from ripser import ripser  # Topological data analysis
from persim import plot_diagrams

def test_discontinuity(embedded_data, clusters):
    """
    Test for gaps between clusters (bimodal/multimodal distribution)
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
    for i, j in zip(*mst.nonzero()):
        if clusters[i] != clusters[j] and clusters[i] != -1 and clusters[j] != -1:
            between_cluster_edges.append(mst[i, j])

    # Large between-cluster edges indicate gaps
    gap_score = np.mean(between_cluster_edges) / np.mean(mst.data)

    # 3. Topological data analysis
    diagrams = ripser(embedded_data)['dgms']
    # Persistence diagrams show topological features (connected components, holes)

    return {
        'density_gaps': gaps,
        'gap_score': gap_score,
        'persistence_diagrams': diagrams
    }
```

## Phase 6: Biological Interpretation and Validation (Days 8-9)

```python
def pathway_enrichment(cluster_profiles, gene_sets):
    """
    Test if clusters map to known biological pathways
    """
    from gseapy import enrichr

    enrichment_results = {}
    for cluster_id, profile in cluster_profiles.items():
        # Get top genes for this cluster
        top_genes = [f for f, score in profile if f in gene_sets]

        # Run enrichment
        enr = enrichr(gene_list=top_genes,
                     gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2021',
                               'WikiPathway_2021_Human'],
                     outdir=f'cluster_{cluster_id}')
        enrichment_results[cluster_id] = enr.results

    return enrichment_results

def metabolic_network_analysis(metabolite_profiles):
    """
    Map metabolite changes to metabolic networks
    """
    import networkx as nx
    from pymetabolism import MetabolicNetwork

    network = MetabolicNetwork()

    for cluster_id, metabolites in metabolite_profiles.items():
        # Find pathways connecting altered metabolites
        altered_metabolites = [m for m, score in metabolites if abs(score) > 1]
        subnetwork = network.find_connecting_pathways(altered_metabolites)

        # Identify key regulatory nodes
        centrality = nx.betweenness_centrality(subnetwork)
        key_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

    return key_nodes

def clinical_correlation(clusters, clinical_data):
    """
    Test clinical relevance of clusters
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
    comorbidities = ['GI_issues', 'epilepsy', 'anxiety', 'sleep_disorder']
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        comorbidity_profile = clinical_data.loc[mask, comorbidities].mean()
        correlations[f'cluster_{cluster_id}_comorbidities'] = comorbidity_profile.to_dict()

    return correlations
```

## Phase 7: Visualization and Reporting (Day 10)

```python
import plotly.graph_objects as go
import plotly.express as px
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool

def create_interactive_visualization(embedded_data, metadata, clusters):
    """
    Create comprehensive interactive visualizations
    """
    # Main t-SNE/UMAP plot with multiple overlays
    fig = go.Figure()

    # Add traces for different metadata
    overlays = ['diagnosis', 'serotonin_level', 'dopamine_level',
                'genetic_burden', 'GI_issues', 'cluster']

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
            visible=(overlay == 'cluster')  # Start with cluster view
        ))

    # Add buttons to switch overlays
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

    # Cluster transition analysis
    if 'timepoint' in metadata:
        # Sankey diagram showing cluster transitions over time
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(label=cluster_labels),
            link=dict(source=source_clusters, target=target_clusters, value=flows)
        )])

    # Heatmap of cluster characteristics
    cluster_features = pd.DataFrame(cluster_profiles).T
    fig_heatmap = px.imshow(cluster_features,
                            labels=dict(x="Feature", y="Cluster", color="Effect Size"),
                            color_continuous_scale='RdBu_r',
                            color_continuous_midpoint=0)

    return fig, fig_sankey, fig_heatmap

def generate_report(results):
    """
    Generate comprehensive report
    """
    from jinja2 import Template

    template = Template("""
    # Multi-Omics Analysis of Neurodevelopmental Conditions

    ## Executive Summary
    - Identified {{ n_main_clusters }} main clusters
    - {{ n_subclusters }} total subclusters identified
    - Gap score: {{ gap_score }} (>1.5 indicates clear separation)

    ## Main Findings

    ### Cluster 1: Neurotransmitter Dysfunction
    - N = {{ cluster1_n }}
    - Key features: {{ cluster1_features }}
    - Metabolite profile: Low serotonin (p={{ serotonin_p }}), altered dopamine
    - Genetic enrichment: Neurotransmitter synthesis pathways
    - Clinical: High ADHD comorbidity ({{ adhd_percent }}%), GI issues ({{ gi_percent }}%)

    ### Cluster 2: Neurodevelopmental Disruption
    - N = {{ cluster2_n }}
    - Key features: {{ cluster2_features }}
    - Metabolite profile: Normal neurotransmitters
    - Genetic enrichment: Rare de novo variants
    - Clinical: Intellectual disability ({{ id_percent }}%), early diagnosis

    ## Validation Metrics
    - Silhouette score: {{ silhouette }}
    - Stability: {{ stability }}
    - Clinical correlation: {{ clinical_correlation }}

    ## Biological Pathways
    {{ pathway_results }}

    ## Clinical Implications
    {{ clinical_implications }}
    """)

    return template.render(**results)
```

## Phase 8: External Validation (Days 11-12)

```python
def external_validation(model, new_data):
    """
    Validate findings in independent cohort
    """
    # Project new data into existing embedding space
    new_embedded = model.transform(new_data)

    # Assign to nearest cluster
    predicted_clusters = assign_to_clusters(new_embedded, cluster_centers)

    # Test if cluster characteristics hold
    validation_metrics = validate_clusters(new_data, predicted_clusters, new_metadata)

    # Cross-cohort replication
    replication_score = compare_cluster_profiles(
        original_profiles,
        new_profiles
    )

    return validation_metrics, replication_score
```

## Expected Results

### Primary Findings
1. **Clear separation** between neurotransmitter dysfunction and neurodevelopmental disruption clusters
2. **Gap analysis** showing discontinuous distribution (not a spectrum)
3. **Metabolite-gene correlations**: TPH2 variants correlate with low serotonin metabolites

### Subclusters Within Main Groups
- **Within neurotransmitter dysfunction**:
  - Serotonin-dominant (depression, anxiety, GI)
  - Dopamine-dominant (ADHD, hyperactivity)
  - Mixed (classic autism + ADHD)

- **Within neurodevelopmental**:
  - Early severe (syndromic)
  - Regressive
  - Mild with language delay

### Clinical Utility
- Cluster membership predicts:
  - Medication response (SSRIs vs stimulants vs antipsychotics)
  - Comorbidity risk
  - Prognosis
  - Optimal interventions

## Tools and Resources

```bash
# Environment setup
conda create -n multiomics python=3.10
conda activate multiomics

# Core packages
pip install numpy pandas scipy scikit-learn
pip install umap-learn hdbscan
pip install scanpy anndata  # Single cell tools work great for multi-omics
pip install plotly bokeh matplotlib seaborn

# Specialized packages
pip install momix  # Multi-omics integration
pip install gseapy  # Pathway analysis
pip install pingouin  # Statistics
pip install ripser persim  # Topological data analysis

# For accessing databases
pip install bioservices  # Access to biological databases
pip install pyega3  # EGA data access
```

This pipeline should reveal whether your hypothesis about distinct biological subtypes is correct, and potentially identify even more nuanced patterns than the simple bimodal distribution. The key is that it's completely doable with existing public data - someone just needs to actually DO it!