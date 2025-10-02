#!/usr/bin/env python3
"""
Unified Anti-Pattern Framework Pipeline Runner

A flexible, reusable pipeline for running the anti-pattern framework on any dataset.
No need to create fresh scripts for each analysis - configure everything via command line.

Usage Examples:
    # Run on single CSV file
    python scripts/unified_pipeline_runner.py --data data/my_data.csv

    # Run on directory  
    python scripts/unified_pipeline_runner.py --data data/benchmarks/

    # Run with relaxed stability threshold
    python scripts/unified_pipeline_runner.py --data data/genetics/ --stability 0.6

    # Fast mode
    python scripts/unified_pipeline_runner.py --data data/my_data.csv --fast

    # Compare with standard methods
    python scripts/unified_pipeline_runner.py --data data/my_data.csv --compare
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from optimize_pipeline import OptimizedAntiPatternFramework

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "unified_runs"


def load_data(file_path, **kwargs):
    """Auto-detect and load data."""
    file_path = Path(file_path)
    print(f"Loading {file_path.name}...")
    
    suffix = file_path.suffix.lower()
    if suffix in ['.csv', '.txt']:
        df = pd.read_csv(file_path, index_col=0)
    elif suffix == '.tsv':
        df = pd.read_csv(file_path, sep='\t', index_col=0)
    else:
        raise ValueError(f"Unsupported format: {suffix}")
    
    print(f"  Shape: {df.shape[0]} samples × {df.shape[1]} features")
    
    # Handle missing
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        print(f"  Filling {n_missing} missing values")
        df = df.fillna(0)
    
    # Remove constants
    constant_cols = df.columns[df.std() == 0]
    if len(constant_cols) > 0:
        print(f"  Removing {len(constant_cols)} constant features")
        df = df.drop(columns=constant_cols)
    
    X = df.values
    sample_ids = df.index.tolist()
    feature_names = df.columns.tolist()
    
    print(f"  Final: {X.shape[0]} samples × {X.shape[1]} features")
    return X, sample_ids, feature_names


def run_framework(X, name, stability=0.8, fast=False):
    """Run anti-pattern framework with custom threshold."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {name}")
    print(f"{'='*70}")
    
    # Custom framework with adjustable threshold
    class CustomFramework(OptimizedAntiPatternFramework):
        def _topology_gate_vectorized(self, X, k):
            from joblib import Parallel, delayed
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
            ref_labels = kmeans.fit_predict(X_scaled)
            
            def bootstrap_iter(seed):
                rng = np.random.RandomState(seed)
                indices = rng.choice(len(X), size=len(X), replace=True)
                X_boot = X_scaled[indices]
                kmeans_boot = KMeans(n_clusters=k, random_state=seed, n_init=10)
                boot_labels = kmeans_boot.fit_predict(X_boot)
                return adjusted_rand_score(ref_labels[indices], boot_labels)
            
            seeds = [self.random_state + i for i in range(self.n_bootstrap)]
            aris = Parallel(n_jobs=self.n_jobs)(delayed(bootstrap_iter)(s) for s in seeds)
            mean_ari = np.mean(aris)
            return mean_ari > stability
    
    framework = CustomFramework(
        n_permutations=50 if fast else 100,
        n_bootstrap=25 if fast else 50,
        fast_mode=fast,
        n_jobs=-1,
        verbose=True,
        random_state=42
    )
    
    print(f"Running framework (stability={stability})...")
    start = time.time()
    labels = framework.fit_predict(X, k_range=(2, 6))
    elapsed = time.time() - start
    
    results = {
        'dataset': name,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'decision': framework.decision_,
        'reason': framework.reason_,
        'n_clusters': len(np.unique(labels)),
        'cluster_labels': labels.tolist(),
        'total_time': elapsed,
        'stability_threshold': stability
    }
    
    if framework.decision_ == "ACCEPT":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        results['silhouette_score'] = silhouette_score(X_scaled, labels)
        results['calinski_harabasz'] = calinski_harabasz_score(X_scaled, labels)
        
        unique, counts = np.unique(labels, return_counts=True)
        results['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
    
    print(f"\nDecision: {results['decision']}")
    print(f"Reason: {results['reason']}")
    if results['decision'] == "ACCEPT":
        print(f"Clusters: {results['n_clusters']}")
        print(f"Silhouette: {results['silhouette_score']:.3f}")
    print(f"Runtime: {elapsed:.1f}s\n")
    
    return results, framework


def main():
    parser = argparse.ArgumentParser(description="Unified pipeline runner")
    parser.add_argument('--data', type=str, required=True, help="Data file or directory")
    parser.add_argument('--stability', type=float, default=0.8, help="Stability threshold (default: 0.8)")
    parser.add_argument('--fast', action='store_true', help="Fast mode")
    parser.add_argument('--compare', action='store_true', help="Compare with standard methods")
    parser.add_argument('--output', type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    # Setup output
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("UNIFIED PIPELINE RUNNER")
    print("="*70)
    print(f"Output: {output_dir}")
    print(f"Stability: {args.stability}\n")
    
    # Find files
    data_path = Path(args.data)
    if data_path.is_file():
        files = [data_path]
    elif data_path.is_dir():
        files = list(data_path.glob("*.csv"))
        files = [f for f in files if f.stat().st_size > 1000]
    else:
        print(f"Error: {data_path} not found")
        return
    
    print(f"Found {len(files)} file(s)\n")
    
    all_results = []
    for data_file in files:
        try:
            X, sample_ids, features = load_data(data_file)
            results, framework = run_framework(X, data_file.stem, args.stability, args.fast)
            
            # Save
            with open(output_dir / f"{data_file.stem}_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            if results['decision'] == "ACCEPT":
                assignments = pd.DataFrame({'sample_id': sample_ids, 'cluster': results['cluster_labels']})
                assignments.to_csv(output_dir / f"{data_file.stem}_assignments.csv", index=False)
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Error with {data_file.name}: {e}\n")
            continue
    
    # Report
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(f"Processed: {len(all_results)}/{len(files)} datasets\n\n")
        for r in all_results:
            f.write(f"{r['dataset']}: {r['decision']} - {r['reason']}\n")
    
    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(all_results)}/{len(files)} datasets")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
