#!/usr/bin/env python3
"""
Simple SNP Cluster Annotation

Analyzes cluster characteristics from genomic positions without external API calls.
Creates detailed summaries based on chromosomal distribution, effect sizes, and MAF.
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "adhd_gwas_analysis"


# Known ADHD-related genes on Chromosome 1 (literature-based)
KNOWN_ADHD_GENES_CHR1 = {
    "GNB1": (1, 1844505, 1931023),  # Near 2 Mb
    "SLC6A3": (1, 1286473, 1311959),  # DAT1 - dopamine transporter, ~1.3 Mb
    "ADGRL2": (1, 88926034, 89046000),  # ~89 Mb - ADHD candidate
    "LPHN2": (1, 88926034, 89046000),  # Alias for ADGRL2
    # Chr1 44Mb region genes (where most SNPs cluster)
    "ST3GAL3": (1, 43938195, 44116908),  # 43.9-44.1 Mb
    "PTPRF": (1, 43933766, 44567540),  # 43.9-44.6 Mb - large gene
    "CDC42BPA": (1, 227010000, 227285000),  # Different region
}


def annotate_clusters():
    """Annotate SNP clusters with available data."""
    print("="*70)
    print("SNP CLUSTER ANNOTATION REPORT")
    print("="*70)
    print(f"Results directory: {RESULTS_DIR}\n")
    
    # Load data
    assignments = pd.read_csv(RESULTS_DIR / "adhd_snps_for_clustering_assignments.csv")
    gwas_data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "gwas" / "adhd_significant_snps.tsv", sep='\t')
    
    # Merge
    snp_data = pd.merge(
        assignments,
        gwas_data[['SNP', 'CHR', 'BP', 'A1', 'A2', 'FRQ_A_19099', 'FRQ_U_34194', 'OR', 'SE', 'P']],
        left_on='sample_id',
        right_on='SNP',
        how='left'
    )
    
    print(f"Total SNPs: {len(snp_data)}")
    print(f"Clusters: {sorted(snp_data['cluster'].unique())}\n")
    
    # Analyze each cluster
    cluster_summaries = {}
    
    for cluster_id in sorted(snp_data['cluster'].unique()):
        cluster_snps = snp_data[snp_data['cluster'] == cluster_id].copy()
        
        # Chromosome distribution
        chr_dist = cluster_snps['CHR'].value_counts().to_dict()
        dominant_chr = cluster_snps['CHR'].mode()[0] if len(cluster_snps) > 0 else None
        chr_purity = (cluster_snps['CHR'] == dominant_chr).sum() / len(cluster_snps) if len(cluster_snps) > 0 else 0
        
        # Position range
        if dominant_chr is not None:
            chr_snps = cluster_snps[cluster_snps['CHR'] == dominant_chr]
            bp_min = chr_snps['BP'].min()
            bp_max = chr_snps['BP'].max()
            span_mb = (bp_max - bp_min) / 1e6
        else:
            bp_min, bp_max, span_mb = None, None, None
        
        # Effect size statistics
        or_values = cluster_snps['OR'].dropna()
        mean_or = or_values.mean()
        std_or = or_values.std()
        median_or = or_values.median()
        
        # Risk classification
        if mean_or > 1.05:
            risk_class = "High Risk"
        elif mean_or < 0.95:
            risk_class = "Protective"
        else:
            risk_class = "Balanced"
        
        # MAF statistics (case frequency)
        maf_case = cluster_snps['FRQ_A_19099'].mean()
        maf_control = cluster_snps['FRQ_U_34194'].mean()
        
        # P-value range
        p_min = cluster_snps['P'].min()
        p_max = cluster_snps['P'].max()
        
        # Top SNPs
        top_snps = cluster_snps.nsmallest(5, 'P')[['SNP', 'CHR', 'BP', 'OR', 'P']].to_dict('records')
        
        # Check for known genes (simplified - just chr1 44Mb region)
        nearby_genes = []
        if dominant_chr == 1 and bp_min is not None:
            for gene, (chr, start, end) in KNOWN_ADHD_GENES_CHR1.items():
                if chr == dominant_chr:
                    # Check if any SNP is within 50kb of gene
                    if (bp_min - 50000 <= end) and (bp_max + 50000 >= start):
                        nearby_genes.append(gene)
        
        summary = {
            'cluster_id': int(cluster_id),
            'n_snps': len(cluster_snps),
            'dominant_chromosome': int(dominant_chr) if dominant_chr is not None else None,
            'chr_purity_pct': round(chr_purity * 100, 1),
            'chr_distribution': {int(k): int(v) for k, v in chr_dist.items()},
            'genomic_span_mb': round(span_mb, 2) if span_mb is not None else None,
            'bp_range': {'start': int(bp_min), 'end': int(bp_max)} if bp_min is not None else None,
            'mean_or': round(mean_or, 4),
            'median_or': round(median_or, 4),
            'std_or': round(std_or, 4),
            'risk_classification': risk_class,
            'maf_cases': round(maf_case, 3),
            'maf_controls': round(maf_control, 3),
            'maf_diff': round(maf_case - maf_control, 3),
            'p_value_range': {'min': float(p_min), 'max': float(p_max)},
            'top_snps': top_snps,
            'nearby_known_genes': nearby_genes
        }
        
        cluster_summaries[int(cluster_id)] = summary
        
        print(f"Cluster {cluster_id}: {risk_class}")
        print(f"  SNPs: {len(cluster_snps)}")
        print(f"  Chr: {dominant_chr} ({chr_purity*100:.0f}% purity)")
        if span_mb is not None:
            print(f"  Span: {span_mb:.1f} Mb")
        print(f"  OR: {mean_or:.3f} ± {std_or:.3f}")
        print(f"  MAF: {maf_case:.3f} (cases) vs {maf_control:.3f} (controls)")
        if nearby_genes:
            print(f"  Genes: {', '.join(nearby_genes)}")
        print()
    
    # Save summaries
    summary_file = RESULTS_DIR / "cluster_summaries.json"
    with open(summary_file, 'w') as f:
        json.dump(cluster_summaries, f, indent=2)
    
    print(f"Saved summaries to {summary_file.name}\n")
    
    # Save annotated SNP data
    output_file = RESULTS_DIR / "annotated_snps.csv"
    snp_data.to_csv(output_file, index=False)
    print(f"Saved annotated SNP data to {output_file.name}\n")
    
    return snp_data, cluster_summaries


def generate_markdown_report(cluster_summaries):
    """Generate comprehensive markdown report."""
    report = []
    
    report.append("# ADHD GWAS SNP Cluster Annotation Report")
    report.append("")
    report.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Clusters**: {len(cluster_summaries)}")
    report.append("")
    report.append("---")
    report.append("")
    
    report.append("## Cluster Summary Table")
    report.append("")
    report.append("| Cluster | SNPs | Chr | Span (Mb) | Risk Class | Mean OR | MAF Δ |")
    report.append("|---------|------|-----|-----------|------------|---------|-------|")
    
    for cid, summary in sorted(cluster_summaries.items()):
        span = f"{summary['genomic_span_mb']:.1f}" if summary['genomic_span_mb'] else "N/A"
        maf_delta = f"{summary['maf_diff']:+.3f}" if summary['maf_diff'] else "N/A"
        report.append(
            f"| {cid} | {summary['n_snps']} | {summary['dominant_chromosome']} | "
            f"{span} | {summary['risk_classification']} | {summary['mean_or']:.3f} | {maf_delta} |"
        )
    
    report.append("")
    report.append("---")
    report.append("")
    
    # Detailed cluster descriptions
    for cid, summary in sorted(cluster_summaries.items()):
        report.append(f"## Cluster {cid}: {summary['risk_classification']}")
        report.append("")
        
        report.append(f"**Size**: {summary['n_snps']} SNPs")
        report.append(f"**Primary Chromosome**: Chr{summary['dominant_chromosome']} ({summary['chr_purity_pct']}% purity)")
        
        if summary['genomic_span_mb']:
            report.append(f"**Genomic Span**: {summary['genomic_span_mb']} Mb")
            report.append(f"**Position Range**: {summary['bp_range']['start']:,} - {summary['bp_range']['end']:,} bp")
        
        report.append("")
        report.append("### Effect Size")
        report.append(f"- **Mean OR**: {summary['mean_or']:.4f}")
        report.append(f"- **Median OR**: {summary['median_or']:.4f}")
        report.append(f"- **Std OR**: {summary['std_or']:.4f}")
        report.append(f"- **Classification**: {summary['risk_classification']}")
        
        report.append("")
        report.append("### Allele Frequency")
        report.append(f"- **Cases**: {summary['maf_cases']:.3f}")
        report.append(f"- **Controls**: {summary['maf_controls']:.3f}")
        report.append(f"- **Difference**: {summary['maf_diff']:+.3f}")
        
        report.append("")
        report.append("### Chromosome Distribution")
        for chr_num, count in sorted(summary['chr_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = 100 * count / summary['n_snps']
            report.append(f"- Chr{chr_num}: {count} SNPs ({pct:.1f}%)")
        
        if summary['nearby_known_genes']:
            report.append("")
            report.append("### Nearby Known ADHD Genes")
            for gene in summary['nearby_known_genes']:
                report.append(f"- **{gene}**")
        
        report.append("")
        report.append("### Top 5 SNPs (by P-value)")
        report.append("")
        report.append("| SNP | Chr | Position | OR | P-value |")
        report.append("|-----|-----|----------|-----|---------|")
        for snp in summary['top_snps']:
            report.append(
                f"| {snp['SNP']} | {snp['CHR']} | {snp['BP']:,} | "
                f"{snp['OR']:.3f} | {snp['P']:.2e} |"
            )
        
        report.append("")
        report.append("---")
        report.append("")
    
    # Save report
    report_file = RESULTS_DIR / "CLUSTER_ANNOTATION_REPORT.md"
    with open(report_file, 'w') as f:
        f.write("\n".join(report))
    
    print(f"Generated detailed report: {report_file.name}")
    return report_file


if __name__ == "__main__":
    snp_data, cluster_summaries = annotate_clusters()
    generate_markdown_report(cluster_summaries)
    
    print("\n" + "="*70)
    print("✓ Cluster annotation complete!")
    print("="*70)
