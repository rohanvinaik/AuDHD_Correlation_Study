#!/usr/bin/env python3
"""
Quick GWAS Analysis Pipeline
Extract significant SNPs and prepare for downstream analysis
"""

import gzip
import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*80)
print("GWAS Analysis Pipeline - Quick Start")
print("="*80)

# Setup paths
project_root = Path("/Users/rohanvinaik/AuDHD_Correlation_Study")
data_dir = project_root / "data/raw/genetics"
output_dir = project_root / "data/processed/gwas"
output_dir.mkdir(parents=True, exist_ok=True)

results = {}

# Process ADHD GWAS
print("\n[1/2] Processing ADHD GWAS...")
adhd_file = data_dir / "adhd_eur_jun2017.gz"

if adhd_file.exists():
    print(f"  Loading {adhd_file.name} ({adhd_file.stat().st_size / 1e6:.0f} MB)...")

    # Read with pandas - this will take a minute for 328MB
    adhd_df = pd.read_csv(adhd_file, sep='\t', compression='gzip')
    print(f"  Loaded {len(adhd_df):,} variants")

    # Extract significant SNPs (p < 5e-8)
    adhd_sig = adhd_df[adhd_df['P'] < 5e-8].copy()
    print(f"  Found {len(adhd_sig):,} genome-wide significant SNPs")

    # Save significant SNPs
    out_file = output_dir / "adhd_significant_snps.tsv"
    adhd_sig.to_csv(out_file, sep='\t', index=False)
    print(f"  Saved to {out_file}")

    # Save top 1000 for quick analysis
    adhd_top = adhd_df.nsmallest(1000, 'P')
    adhd_top.to_csv(output_dir / "adhd_top1000.tsv", sep='\t', index=False)

    results['adhd'] = {
        'total_variants': len(adhd_df),
        'significant_5e8': len(adhd_sig),
        'significant_5e6': len(adhd_df[adhd_df['P'] < 5e-6]),
        'min_pvalue': float(adhd_df['P'].min()),
        'top_snp': adhd_df.loc[adhd_df['P'].idxmin(), 'SNP']
    }

    print(f"  Min p-value: {results['adhd']['min_pvalue']:.2e}")
    print(f"  Top SNP: {results['adhd']['top_snp']}")

else:
    print(f"  ERROR: {adhd_file} not found!")
    results['adhd'] = {'error': 'file not found'}

# Process Autism GWAS
print("\n[2/2] Processing Autism GWAS...")
autism_file = data_dir / "iPSYCH-PGC_ASD_Nov2017.gz"

if autism_file.exists():
    print(f"  Loading {autism_file.name} ({autism_file.stat().st_size / 1e3:.0f} KB)...")

    # This file is much smaller
    autism_df = pd.read_csv(autism_file, sep='\t', compression='gzip')
    print(f"  Loaded {len(autism_df):,} variants")

    # Check column names (might be different)
    print(f"  Columns: {list(autism_df.columns)}")

    # Find p-value column
    pval_col = None
    for col in ['P', 'p', 'pval', 'P-value', 'PVAL']:
        if col in autism_df.columns:
            pval_col = col
            break

    if pval_col:
        autism_sig = autism_df[autism_df[pval_col] < 5e-8].copy()
        print(f"  Found {len(autism_sig):,} genome-wide significant SNPs")

        out_file = output_dir / "autism_significant_snps.tsv"
        autism_sig.to_csv(out_file, sep='\t', index=False)
        print(f"  Saved to {out_file}")

        # Save top 1000
        autism_top = autism_df.nsmallest(1000, pval_col)
        autism_top.to_csv(output_dir / "autism_top1000.tsv", sep='\t', index=False)

        results['autism'] = {
            'total_variants': len(autism_df),
            'significant_5e8': len(autism_sig),
            'significant_5e6': len(autism_df[autism_df[pval_col] < 5e-6]),
            'min_pvalue': float(autism_df[pval_col].min())
        }

        print(f"  Min p-value: {results['autism']['min_pvalue']:.2e}")
    else:
        print(f"  ERROR: Could not find p-value column!")
        results['autism'] = {'error': 'no pvalue column'}

else:
    print(f"  ERROR: {autism_file} not found!")
    results['autism'] = {'error': 'file not found'}

# Compare ADHD vs Autism
print("\n" + "="*80)
print("Comparison Summary")
print("="*80)

if 'error' not in results.get('adhd', {}) and 'error' not in results.get('autism', {}):
    print(f"ADHD:   {results['adhd']['significant_5e8']:,} genome-wide significant SNPs")
    print(f"Autism: {results['autism']['significant_5e8']:,} genome-wide significant SNPs")

    # Check for overlap
    if len(adhd_sig) > 0 and len(autism_sig) > 0:
        # Find SNP ID column
        snp_col = 'SNP' if 'SNP' in adhd_sig.columns else None
        if snp_col and snp_col in autism_sig.columns:
            adhd_snps = set(adhd_sig[snp_col])
            autism_snps = set(autism_sig[snp_col])
            overlap = adhd_snps.intersection(autism_snps)

            print(f"\nShared significant SNPs: {len(overlap)}")
            results['overlap'] = {'shared_snps': len(overlap)}

            if len(overlap) > 0:
                print(f"Examples: {list(overlap)[:5]}")

# Save summary
summary_file = output_dir / "analysis_summary.json"
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSummary saved to: {summary_file}")
print("\n" + "="*80)
print("GWAS Processing Complete!")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print("Files created:")
for f in output_dir.glob("*.tsv"):
    print(f"  - {f.name} ({f.stat().st_size / 1e6:.1f} MB)")