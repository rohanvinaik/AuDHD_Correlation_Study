"""Supplementary materials generation

Creates methods sections, data dictionaries, and statistical appendices.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings

import pandas as pd


@dataclass
class SupplementaryMaterial:
    """Supplementary material item"""
    title: str
    content: str
    format: str  # 'html', 'markdown', 'table'
    order: int


def generate_supplementary_materials(
    cluster_results: List[Dict],
    feature_names: List[str],
    methods_detail: Optional[Dict] = None,
) -> List[SupplementaryMaterial]:
    """
    Generate comprehensive supplementary materials

    Args:
        cluster_results: Cluster analysis results
        feature_names: List of feature names
        methods_detail: Detailed methods information

    Returns:
        List of SupplementaryMaterial objects
    """
    materials = []

    # Methods section
    materials.append(SupplementaryMaterial(
        title='Supplementary Methods',
        content=create_methods_section(methods_detail or {}),
        format='html',
        order=1,
    ))

    # Data dictionary
    materials.append(SupplementaryMaterial(
        title='Data Dictionary',
        content=create_data_dictionary(feature_names),
        format='table',
        order=2,
    ))

    # Statistical appendix
    materials.append(SupplementaryMaterial(
        title='Statistical Appendix',
        content=create_statistical_appendix(cluster_results),
        format='html',
        order=3,
    ))

    return materials


def create_methods_section(methods_detail: Dict) -> str:
    """
    Create detailed methods section

    Args:
        methods_detail: Dictionary of methods details

    Returns:
        HTML string with methods
    """
    html = '<div class="methods-section">'
    html += '<h2>Supplementary Methods</h2>'

    # Data collection
    html += '<h3>1. Data Collection and Quality Control</h3>'
    html += '<p><strong>Sample Collection:</strong> ' + methods_detail.get(
        'sample_collection',
        'Biospecimens collected following standardized protocols.'
    ) + '</p>'

    html += '<p><strong>Quality Control:</strong> ' + methods_detail.get(
        'qc',
        'Quality control included outlier detection, missing data assessment, '
        'and technical replicate correlation checks.'
    ) + '</p>'

    # Omics platforms
    html += '<h3>2. Multi-Omics Profiling</h3>'

    html += '<h4>2.1 Genomics</h4>'
    html += '<p>' + methods_detail.get(
        'genomics',
        'Whole genome genotyping performed using Illumina arrays. '
        'Quality control: call rate >95%, MAF >0.01, HWE p>0.001.'
    ) + '</p>'

    html += '<h4>2.2 Transcriptomics</h4>'
    html += '<p>' + methods_detail.get(
        'transcriptomics',
        'RNA-seq performed on Illumina platform. Reads aligned with STAR, '
        'quantified with featureCounts. Normalization: TMM.'
    ) + '</p>'

    html += '<h4>2.3 Metabolomics</h4>'
    html += '<p>' + methods_detail.get(
        'metabolomics',
        'Untargeted metabolomics via LC-MS/MS. Peak detection with XCMS, '
        'identification via METLIN database.'
    ) + '</p>'

    # Preprocessing
    html += '<h3>3. Data Preprocessing and Integration</h3>'
    html += '<p>' + methods_detail.get(
        'preprocessing',
        'Features normalized independently (z-score). Batch effects corrected '
        'using ComBat. Multi-omics integration via concatenation of normalized features.'
    ) + '</p>'

    # Clustering
    html += '<h3>4. Unsupervised Clustering</h3>'
    html += '<p><strong>Dimensionality Reduction:</strong> ' + methods_detail.get(
        'dim_reduction',
        'UMAP with n_neighbors=15, min_dist=0.1, metric=cosine.'
    ) + '</p>'

    html += '<p><strong>Clustering Algorithm:</strong> ' + methods_detail.get(
        'clustering',
        'HDBSCAN with min_cluster_size=20, min_samples=10, cluster_selection_epsilon=0.5.'
    ) + '</p>'

    # Validation
    html += '<h3>5. Cluster Validation</h3>'
    html += '<p>' + methods_detail.get(
        'validation',
        'Clusters validated using silhouette score, Calinski-Harabasz index, '
        'and cross-validation stability. Statistical significance assessed via '
        'Kruskal-Wallis tests with FDR correction.'
    ) + '</p>'

    # Statistical analysis
    html += '<h3>6. Statistical Analysis</h3>'
    html += '<p>' + methods_detail.get(
        'statistics',
        'Differential feature analysis: Kruskal-Wallis test. Effect sizes: '
        'eta-squared. Multiple testing correction: Benjamini-Hochberg FDR. '
        'Pathway enrichment: GSEA with 1000 permutations. Significance threshold: FDR<0.25.'
    ) + '</p>'

    html += '</div>'

    return html


def create_data_dictionary(feature_names: List[str]) -> str:
    """
    Create data dictionary table

    Args:
        feature_names: List of feature names

    Returns:
        HTML table string
    """
    # Categorize features
    categories = {
        'Genomics': [f for f in feature_names if any(x in f.lower() for x in ['snp', 'rs', 'gene', 'variant'])],
        'Transcriptomics': [f for f in feature_names if any(x in f.lower() for x in ['expr', 'mrna', 'transcript'])],
        'Proteomics': [f for f in feature_names if any(x in f.lower() for x in ['protein', 'prot_'])],
        'Metabolomics': [f for f in feature_names if any(x in f.lower() for x in ['metab', 'metabolite'])],
        'Clinical': [f for f in feature_names if any(x in f.lower() for x in ['age', 'sex', 'bmi', 'score'])],
        'Other': [],
    }

    # Collect unclassified
    classified = set(sum(categories.values(), []))
    categories['Other'] = [f for f in feature_names if f not in classified]

    html = '<table class="data-dictionary">'
    html += '<tr><th>Category</th><th>Feature</th><th>Description</th><th>Unit</th></tr>'

    for category, features in categories.items():
        if not features:
            continue

        for i, feature in enumerate(features):
            # Infer description
            desc = _infer_feature_description(feature)
            unit = _infer_unit(feature)

            if i == 0:
                html += f'<tr><td rowspan="{len(features)}"><strong>{category}</strong></td>'
            else:
                html += '<tr>'

            html += f'<td>{feature}</td><td>{desc}</td><td>{unit}</td></tr>'

    html += '</table>'

    return html


def _infer_feature_description(feature: str) -> str:
    """Infer feature description from name"""
    feature_lower = feature.lower()

    if 'age' in feature_lower:
        return 'Patient age at enrollment'
    elif 'sex' in feature_lower:
        return 'Biological sex'
    elif 'bmi' in feature_lower:
        return 'Body mass index'
    elif 'expr' in feature_lower:
        return 'Gene expression level'
    elif 'protein' in feature_lower:
        return 'Protein abundance'
    elif 'metab' in feature_lower:
        return 'Metabolite concentration'
    elif 'snp' in feature_lower or 'rs' in feature_lower:
        return 'Genetic variant'
    else:
        return 'Multi-omics feature'


def _infer_unit(feature: str) -> str:
    """Infer measurement unit"""
    feature_lower = feature.lower()

    if 'age' in feature_lower:
        return 'years'
    elif 'bmi' in feature_lower:
        return 'kg/m²'
    elif 'expr' in feature_lower:
        return 'log2(TPM)'
    elif 'protein' in feature_lower:
        return 'log2(intensity)'
    elif 'metab' in feature_lower:
        return 'log2(abundance)'
    elif 'score' in feature_lower:
        return 'points'
    else:
        return 'normalized'


def create_statistical_appendix(cluster_results: List[Dict]) -> str:
    """
    Create statistical appendix

    Args:
        cluster_results: Cluster analysis results

    Returns:
        HTML string with statistical details
    """
    html = '<div class="statistical-appendix">'
    html += '<h2>Supplementary Statistical Analysis</h2>'

    # Cluster statistics table
    html += '<h3>Table S1: Cluster Statistics</h3>'
    html += '<table>'
    html += '<tr><th>Cluster</th><th>N</th><th>Silhouette Score</th><th>Within-Cluster SS</th></tr>'

    for cluster in cluster_results:
        html += f'<tr>'
        html += f'<td>{cluster.get("name", cluster["id"])}</td>'
        html += f'<td>{cluster.get("n_patients", "N/A")}</td>'
        html += f'<td>{cluster.get("silhouette", "N/A")}</td>'
        html += f'<td>{cluster.get("within_ss", "N/A")}</td>'
        html += f'</tr>'

    html += '</table>'

    # Statistical tests
    html += '<h3>Statistical Testing Procedures</h3>'
    html += '<p><strong>Differential Feature Analysis:</strong></p>'
    html += '<ul>'
    html += '<li>Test: Kruskal-Wallis H-test (non-parametric)</li>'
    html += '<li>Effect size: Eta-squared (η²)</li>'
    html += '<li>Multiple testing correction: Benjamini-Hochberg FDR</li>'
    html += '<li>Significance threshold: FDR < 0.05</li>'
    html += '</ul>'

    html += '<p><strong>Pathway Enrichment:</strong></p>'
    html += '<ul>'
    html += '<li>Method: Gene Set Enrichment Analysis (GSEA)</li>'
    html += '<li>Permutations: 1000</li>'
    html += '<li>Gene sets: GO, KEGG, Reactome</li>'
    html += '<li>Significance: FDR < 0.25</li>'
    html += '</ul>'

    html += '<p><strong>Validation:</strong></p>'
    html += '<ul>'
    html += '<li>Internal validation: Silhouette score, Calinski-Harabasz index</li>'
    html += '<li>Stability: 100-fold bootstrap resampling</li>'
    html += '<li>Statistical significance: Permutation test (1000 permutations)</li>'
    html += '</ul>'

    html += '</div>'

    return html