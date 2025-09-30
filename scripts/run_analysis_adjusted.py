#!/usr/bin/env python3
"""
Multi-Omics Clustering with Adjusted Parameters
Uses both HDBSCAN (relaxed) and K-means for comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, calinski_harabasz_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*80)
print("MULTI-OMICS CLUSTERING - ADJUSTED PARAMETERS")
print("="*80)

# Load data (already processed)
project_root = Path("/Users/rohanvinaik/AuDHD_Correlation_Study")
data_dir = project_root / "data/processed"
output_dir = project_root / "outputs" / f"analysis_{datetime.now().strftime('%Y%m%d')}"

# Load the integrated PCA data (already computed)
pca_file = output_dir / "integrated_pca_factors.csv"
if not pca_file.exists():
    print("Error: Run the main analysis first!")
    exit(1)

pca_df = pd.read_csv(pca_file)
integrated_pca = pca_df.iloc[:, 1:].values  # Drop sample_id

# Load clinical for validation
clinical = pd.read_csv(data_dir / "clinical/synthetic_clinical_data.csv")
true_subtypes = clinical['true_subtype'].values

print(f"Loaded integrated PCA data: {integrated_pca.shape}")
print(f"True subtypes: {len(set(true_subtypes))}")

# ============================================================================
# METHOD 1: HDBSCAN with Relaxed Parameters
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: HDBSCAN (Relaxed Parameters)")
print("="*80)

hdbscan_clusterer = HDBSCAN(
    min_cluster_size=15,  # Reduced from 20
    min_samples=5,  # Reduced from 10
    metric='euclidean',
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.5  # Added for less strict clustering
)

hdbscan_labels = hdbscan_clusterer.fit_predict(integrated_pca)
n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
n_hdbscan_noise = list(hdbscan_labels).count(-1)

print(f"\nHDBSCAN Results:")
print(f"  Clusters found: {n_hdbscan_clusters}")
print(f"  Noise points: {n_hdbscan_noise} ({n_hdbscan_noise/len(hdbscan_labels)*100:.1f}%)")

# ============================================================================
# METHOD 2: K-Means (k=6 to match true subtypes)
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: K-Means (k=6)")
print("="*80)

kmeans = KMeans(n_clusters=6, random_state=42, n_init=20)
kmeans_labels = kmeans.fit_predict(integrated_pca)

print(f"\nK-Means Results:")
print(f"  Clusters: 6 (fixed)")
print(f"  Cluster sizes:")
for i in range(6):
    count = (kmeans_labels == i).sum()
    print(f"    Cluster {i}: {count}")

# ============================================================================
# METHOD 3: Hierarchical Clustering
# ============================================================================
print("\n" + "="*80)
print("METHOD 3: Hierarchical Clustering (k=6)")
print("="*80)

hierarchical = AgglomerativeClustering(n_clusters=6, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(integrated_pca)

print(f"\nHierarchical Results:")
print(f"  Clusters: 6 (fixed)")
print(f"  Cluster sizes:")
for i in range(6):
    count = (hierarchical_labels == i).sum()
    print(f"    Cluster {i}: {count}")

# ============================================================================
# VALIDATION: Compare All Methods
# ============================================================================
print("\n" + "="*80)
print("VALIDATION: Comparing All Methods")
print("="*80)

# True labels for comparison
subtype_map = {st: i for i, st in enumerate(sorted(set(true_subtypes)))}
true_labels = np.array([subtype_map[st] for st in true_subtypes])

methods = {
    'K-Means': kmeans_labels,
    'Hierarchical': hierarchical_labels,
}

# Only add HDBSCAN if it found clusters
if n_hdbscan_clusters > 0:
    methods['HDBSCAN'] = hdbscan_labels

print("\nInternal Validation (Silhouette Score):")
for name, labels in methods.items():
    if -1 not in labels:  # Only for hard assignments
        sil = silhouette_score(integrated_pca, labels)
        ch = calinski_harabasz_score(integrated_pca, labels)
        print(f"  {name}:")
        print(f"    Silhouette: {sil:.3f}")
        print(f"    Calinski-Harabasz: {ch:.1f}")

print("\nExternal Validation (vs. True Subtypes):")
for name, labels in methods.items():
    mask = labels != -1  # Exclude noise if any
    if mask.sum() > 0:
        ari = adjusted_rand_score(true_labels[mask], labels[mask])
        nmi = normalized_mutual_info_score(true_labels[mask], labels[mask])
        print(f"  {name}:")
        print(f"    ARI: {ari:.3f}")
        print(f"    NMI: {nmi:.3f}")

# ============================================================================
# BEST METHOD: K-Means (most stable for demonstration)
# ============================================================================
print("\n" + "="*80)
print("SELECTED METHOD: K-Means Clustering")
print("="*80)

best_labels = kmeans_labels

# Cluster characterization
results_df = clinical.copy()
results_df['predicted_cluster'] = best_labels

print("\nCluster Profiles:")
for cluster_id in range(6):
    mask = best_labels == cluster_id
    n = mask.sum()

    print(f"\n--- Cluster {cluster_id} (n={n}) ---")

    # Demographics
    avg_age = results_df.loc[mask, 'age'].mean()
    pct_male = (results_df.loc[mask, 'sex'] == 'M').mean() * 100
    print(f"  Demographics: Age={avg_age:.1f}, Male={pct_male:.0f}%")

    # Clinical characteristics
    adhd_inatt = results_df.loc[mask, 'ADHD_inattention'].mean()
    adhd_hyper = results_df.loc[mask, 'ADHD_hyperactivity'].mean()
    anxiety = results_df.loc[mask, 'anxiety_score'].mean()
    prs = results_df.loc[mask, 'ADHD_PRS'].mean()

    print(f"  Clinical:")
    print(f"    Inattention: {adhd_inatt:.1f}")
    print(f"    Hyperactivity: {adhd_hyper:.1f}")
    print(f"    Anxiety: {anxiety:.1f}")
    print(f"    ADHD PRS: {prs:.3f}")

    # Diagnosis distribution
    diagnosis_counts = results_df.loc[mask, 'diagnosis'].value_counts()
    print(f"  Diagnosis: {dict(diagnosis_counts)}")

    # True subtype composition
    true_composition = results_df.loc[mask, 'true_subtype'].value_counts().head(3)
    print(f"  Top true subtypes: {dict(true_composition)}")

# Save
results_df.to_csv(output_dir / "clustering_results_kmeans.csv", index=False)

# ============================================================================
# ENHANCED VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("Generating Enhanced Visualizations")
print("="*80)

# Figure 1: Compare all methods
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# K-Means
ax = axes[0, 0]
scatter = ax.scatter(integrated_pca[:, 0], integrated_pca[:, 1],
                     c=kmeans_labels, cmap='tab10', s=30, alpha=0.7)
ax.set_title('K-Means Clustering (k=6)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.colorbar(scatter, ax=ax)

# Hierarchical
ax = axes[0, 1]
scatter = ax.scatter(integrated_pca[:, 0], integrated_pca[:, 1],
                     c=hierarchical_labels, cmap='tab10', s=30, alpha=0.7)
ax.set_title('Hierarchical Clustering (k=6)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.colorbar(scatter, ax=ax)

# True subtypes
ax = axes[1, 0]
scatter = ax.scatter(integrated_pca[:, 0], integrated_pca[:, 1],
                     c=true_labels, cmap='Set3', s=30, alpha=0.7)
ax.set_title('True Biological Subtypes')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.colorbar(scatter, ax=ax)

# HDBSCAN (if available)
ax = axes[1, 1]
if n_hdbscan_clusters > 0:
    scatter = ax.scatter(integrated_pca[:, 0], integrated_pca[:, 1],
                         c=hdbscan_labels, cmap='tab10', s=30, alpha=0.7)
    ax.set_title(f'HDBSCAN ({n_hdbscan_clusters} clusters, {n_hdbscan_noise} noise)')
else:
    ax.text(0.5, 0.5, 'HDBSCAN found no clusters\n(all points = noise)',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title('HDBSCAN (No clusters found)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

plt.tight_layout()
plt.savefig(output_dir / "clustering_comparison.png", dpi=300, bbox_inches='tight')
print(f"  Saved: clustering_comparison.png")

# Figure 2: K-Means heatmap of cluster characteristics
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate mean values for each cluster
cluster_profiles = []
feature_names = ['Age', 'BMI', 'Inattention', 'Hyperactivity', 'Anxiety', 'IQ', 'PRS']

for cluster_id in range(6):
    mask = best_labels == cluster_id
    profile = [
        results_df.loc[mask, 'age'].mean(),
        results_df.loc[mask, 'BMI'].mean(),
        results_df.loc[mask, 'ADHD_inattention'].mean(),
        results_df.loc[mask, 'ADHD_hyperactivity'].mean(),
        results_df.loc[mask, 'anxiety_score'].mean(),
        results_df.loc[mask, 'IQ'].mean(),
        results_df.loc[mask, 'ADHD_PRS'].mean(),
    ]
    cluster_profiles.append(profile)

# Normalize for heatmap
cluster_profiles = np.array(cluster_profiles)
cluster_profiles_norm = (cluster_profiles - cluster_profiles.mean(axis=0)) / cluster_profiles.std(axis=0)

sns.heatmap(cluster_profiles_norm.T, annot=True, fmt='.2f',
            xticklabels=[f'C{i}' for i in range(6)],
            yticklabels=feature_names,
            cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Cluster Profiles (Z-scored)')
ax.set_xlabel('Cluster')
ax.set_ylabel('Feature')

plt.tight_layout()
plt.savefig(output_dir / "cluster_profiles_heatmap.png", dpi=300, bbox_inches='tight')
print(f"  Saved: cluster_profiles_heatmap.png")

plt.close('all')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print(f"\nBest Method: K-Means (k=6)")
print(f"  Silhouette Score: {silhouette_score(integrated_pca, kmeans_labels):.3f}")
print(f"  ARI vs. True: {adjusted_rand_score(true_labels, kmeans_labels):.3f}")
print(f"  NMI vs. True: {normalized_mutual_info_score(true_labels, kmeans_labels):.3f}")

print(f"\nAll outputs saved to: {output_dir}")
print("="*80)