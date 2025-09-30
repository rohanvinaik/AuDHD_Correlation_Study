#!/usr/bin/env python3
"""
Complete Multi-Omics Analysis Pipeline
Integrates clinical, metabolomics, microbiome, and GWAS data
Performs clustering and validates results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*80)
print("COMPLETE MULTI-OMICS CLUSTERING ANALYSIS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Setup
project_root = Path("/Users/rohanvinaik/AuDHD_Correlation_Study")
data_dir = project_root / "data/processed"
output_dir = project_root / "outputs" / f"analysis_{datetime.now().strftime('%Y%m%d')}"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nOutput directory: {output_dir}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Loading Multi-Omics Data")
print("="*80)

# Clinical
clinical_file = data_dir / "clinical/synthetic_clinical_data.csv"
clinical = pd.read_csv(clinical_file)
print(f"\n[1/3] Clinical data: {clinical.shape}")
print(f"  Variables: {list(clinical.columns)}")

# Metabolomics
metabolomics_file = data_dir / "metabolomics/synthetic_metabolomics_data.csv"
metabolomics = pd.read_csv(metabolomics_file)
print(f"\n[2/3] Metabolomics data: {metabolomics.shape}")
print(f"  Metabolites: {metabolomics.shape[1]-1}")  # Minus sample_id

# Microbiome
microbiome_file = data_dir / "microbiome/synthetic_microbiome_data.csv"
microbiome = pd.read_csv(microbiome_file)
print(f"\n[3/3] Microbiome data: {microbiome.shape}")
print(f"  Bacterial genera: {microbiome.shape[1]-1}")

# Verify sample IDs match
assert all(clinical['sample_id'] == metabolomics['sample_id'])
assert all(clinical['sample_id'] == microbiome['sample_id'])
print("\n✓ All datasets aligned")

# ============================================================================
# STEP 2: PREPARE FEATURES FOR INTEGRATION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Feature Preparation and Scaling")
print("="*80)

# Clinical features (quantitative only)
clinical_features = clinical[[
    'age', 'BMI', 'ADHD_inattention', 'ADHD_hyperactivity',
    'anxiety_score', 'IQ', 'ADHD_PRS'
]].values

# Convert sex to numeric
sex_numeric = (clinical['sex'] == 'M').astype(int).values.reshape(-1, 1)
clinical_features = np.hstack([clinical_features, sex_numeric])

# Metabolomics (all columns except sample_id)
metabolomics_features = metabolomics.iloc[:, 1:].values

# Microbiome (all columns except sample_id)
microbiome_features = microbiome.iloc[:, 1:].values

print(f"\nFeature dimensions:")
print(f"  Clinical: {clinical_features.shape}")
print(f"  Metabolomics: {metabolomics_features.shape}")
print(f"  Microbiome: {microbiome_features.shape}")

# Scale each modality separately
scaler_clinical = StandardScaler()
scaler_metabolomics = StandardScaler()
scaler_microbiome = StandardScaler()

clinical_scaled = scaler_clinical.fit_transform(clinical_features)
metabolomics_scaled = scaler_metabolomics.fit_transform(metabolomics_features)
microbiome_scaled = scaler_microbiome.fit_transform(microbiome_features)

print("\n✓ All modalities scaled (mean=0, std=1)")

# ============================================================================
# STEP 3: MULTI-OMICS INTEGRATION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Multi-Omics Integration")
print("="*80)

# Simple concatenation with weighting (as per project plan)
# Weights: metabolomics highest (most proximal to phenotype)
weights = {
    'clinical': 0.25,
    'metabolomics': 0.40,  # Highest - neurotransmitters most relevant
    'microbiome': 0.35
}

integrated_data = np.hstack([
    clinical_scaled * weights['clinical'],
    metabolomics_scaled * weights['metabolomics'],
    microbiome_scaled * weights['microbiome']
])

print(f"\nIntegrated data shape: {integrated_data.shape}")
print(f"  Total features: {integrated_data.shape[1]}")
print(f"\nWeighting:")
for modality, weight in weights.items():
    print(f"  {modality}: {weight:.2f}")

# PCA for dimensionality reduction
n_components = 15  # As per project plan
pca = PCA(n_components=n_components, random_state=42)
integrated_pca = pca.fit_transform(integrated_data)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\nPCA Dimensionality Reduction:")
print(f"  Components: {n_components}")
print(f"  Explained variance (first 5): {explained_var[:5]}")
print(f"  Cumulative variance: {cumulative_var[-1]:.2%}")

# Save PCA results
pca_df = pd.DataFrame(
    integrated_pca,
    columns=[f'PC{i+1}' for i in range(n_components)]
)
pca_df.insert(0, 'sample_id', clinical['sample_id'])
pca_df.to_csv(output_dir / "integrated_pca_factors.csv", index=False)

# ============================================================================
# STEP 4: CLUSTERING ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: HDBSCAN Clustering Analysis")
print("="*80)

# HDBSCAN clustering (as per project plan)
clusterer = HDBSCAN(
    min_cluster_size=20,  # Per project plan
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='eom'
)

cluster_labels = clusterer.fit_predict(integrated_pca)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"\nClustering Results:")
print(f"  Number of clusters: {n_clusters}")
print(f"  Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
print(f"\nCluster sizes:")
for cluster_id in sorted(set(cluster_labels)):
    count = (cluster_labels == cluster_id).sum()
    if cluster_id == -1:
        print(f"  Noise: {count}")
    else:
        print(f"  Cluster {cluster_id}: {count}")

# ============================================================================
# STEP 5: VALIDATION
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Cluster Validation")
print("="*80)

# Internal validation (silhouette score)
# Exclude noise points
valid_mask = cluster_labels != -1
if valid_mask.sum() > 0:
    silhouette = silhouette_score(
        integrated_pca[valid_mask],
        cluster_labels[valid_mask]
    )
    print(f"\nInternal Validation:")
    print(f"  Silhouette score: {silhouette:.3f}")
else:
    silhouette = None
    print("\nWarning: All points classified as noise!")

# External validation (against true subtypes)
true_subtypes = clinical['true_subtype'].values

# Convert true subtypes to numeric for comparison
subtype_map = {st: i for i, st in enumerate(sorted(set(true_subtypes)))}
true_labels = np.array([subtype_map[st] for st in true_subtypes])

# Calculate ARI and NMI (only for non-noise points)
if valid_mask.sum() > 0:
    ari = adjusted_rand_score(true_labels[valid_mask], cluster_labels[valid_mask])
    nmi = normalized_mutual_info_score(true_labels[valid_mask], cluster_labels[valid_mask])

    print(f"\nExternal Validation (vs. True Subtypes):")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"  Normalized Mutual Information: {nmi:.3f}")
else:
    ari = nmi = None

# ============================================================================
# STEP 6: CLUSTER CHARACTERIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Cluster Characterization")
print("="*80)

# Add cluster assignments to clinical data
results_df = clinical.copy()
results_df['predicted_cluster'] = cluster_labels

# Calculate cluster profiles
print("\nCluster Profiles:")
for cluster_id in sorted([c for c in set(cluster_labels) if c != -1]):
    mask = cluster_labels == cluster_id
    n = mask.sum()

    print(f"\n--- Cluster {cluster_id} (n={n}) ---")

    # Demographics
    avg_age = results_df.loc[mask, 'age'].mean()
    pct_male = (results_df.loc[mask, 'sex'] == 'M').mean() * 100
    print(f"  Demographics: Age={avg_age:.1f}, Male={pct_male:.0f}%")

    # Clinical
    adhd_inatt = results_df.loc[mask, 'ADHD_inattention'].mean()
    adhd_hyper = results_df.loc[mask, 'ADHD_hyperactivity'].mean()
    anxiety = results_df.loc[mask, 'anxiety_score'].mean()
    print(f"  Symptoms: Inattention={adhd_inatt:.1f}, Hyperactivity={adhd_hyper:.1f}, Anxiety={anxiety:.1f}")

    # Genetic risk
    prs = results_df.loc[mask, 'ADHD_PRS'].mean()
    print(f"  Genetic: PRS={prs:.3f}")

    # True subtype composition (for validation)
    true_composition = results_df.loc[mask, 'true_subtype'].value_counts()
    print(f"  True subtypes: {dict(true_composition)}")

# Save results
results_df.to_csv(output_dir / "clustering_results.csv", index=False)

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Generating Visualizations")
print("="*80)

# Figure 1: PCA plot colored by clusters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot by predicted clusters
scatter1 = ax1.scatter(
    integrated_pca[:, 0], integrated_pca[:, 1],
    c=cluster_labels, cmap='tab10', s=30, alpha=0.6
)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('Predicted Clusters')
plt.colorbar(scatter1, ax=ax1, label='Cluster')

# Plot by true subtypes
true_numeric = np.array([subtype_map[st] for st in true_subtypes])
scatter2 = ax2.scatter(
    integrated_pca[:, 0], integrated_pca[:, 1],
    c=true_numeric, cmap='Set3', s=30, alpha=0.6
)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title('True Subtypes')
plt.colorbar(scatter2, ax=ax2, label='Subtype')

plt.tight_layout()
plt.savefig(output_dir / "clustering_visualization.png", dpi=300)
print(f"  Saved: clustering_visualization.png")

# Figure 2: Explained variance
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, n_components+1), cumulative_var, 'o-')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('PCA Explained Variance')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "pca_explained_variance.png", dpi=300)
print(f"  Saved: pca_explained_variance.png")

# Figure 3: Cluster sizes
fig, ax = plt.subplots(figsize=(8, 6))
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
cluster_counts.plot(kind='bar', ax=ax)
ax.set_xlabel('Cluster ID')
ax.set_ylabel('Number of Samples')
ax.set_title('Cluster Size Distribution')
plt.tight_layout()
plt.savefig(output_dir / "cluster_sizes.png", dpi=300)
print(f"  Saved: cluster_sizes.png")

plt.close('all')

# ============================================================================
# STEP 8: SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Generating Summary Report")
print("="*80)

summary = {
    'analysis_date': datetime.now().isoformat(),
    'data': {
        'n_samples': len(clinical),
        'n_clinical_features': clinical_features.shape[1],
        'n_metabolites': metabolomics_features.shape[1],
        'n_bacteria': microbiome_features.shape[1],
        'total_features': integrated_data.shape[1]
    },
    'integration': {
        'method': 'Weighted concatenation + PCA',
        'n_components': n_components,
        'explained_variance': float(cumulative_var[-1]),
        'weights': weights
    },
    'clustering': {
        'method': 'HDBSCAN',
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'cluster_sizes': {int(k): int(v) for k, v in cluster_counts.items()}
    },
    'validation': {
        'silhouette_score': float(silhouette) if silhouette else None,
        'adjusted_rand_index': float(ari) if ari else None,
        'normalized_mutual_info': float(nmi) if nmi else None
    }
}

import json
with open(output_dir / "analysis_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\nSummary Report:")
print(json.dumps(summary, indent=2))

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
for f in sorted(output_dir.glob("*")):
    size_mb = f.stat().st_size / 1e6
    print(f"  - {f.name} ({size_mb:.2f} MB)")

print("\n" + "="*80)
