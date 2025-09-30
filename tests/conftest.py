"""Pytest configuration and fixtures for AuDHD Correlation Study tests

Provides synthetic data generation and common fixtures for all tests.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Tuple

# Set random seed for reproducibility
np.random.seed(42)


# ============================================================================
# Data Generation Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def synthetic_genomic_data():
    """Generate synthetic genomic data (SNPs)"""
    n_samples = 100
    n_snps = 500

    # Generate genotypes (0, 1, 2 for AA, Aa, aa)
    genotypes = np.random.choice([0, 1, 2], size=(n_samples, n_snps), p=[0.5, 0.3, 0.2])

    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    snp_ids = [f"rs{i:06d}" for i in range(n_snps)]

    df = pd.DataFrame(genotypes, index=sample_ids, columns=snp_ids)

    # Add metadata
    metadata = pd.DataFrame({
        'sample_id': sample_ids,
        'chromosome': np.random.choice(range(1, 23), n_samples),
        'population': np.random.choice(['EUR', 'AFR', 'EAS', 'AMR'], n_samples),
    })

    return {'genotypes': df, 'metadata': metadata}


@pytest.fixture(scope="session")
def synthetic_clinical_data():
    """Generate synthetic clinical phenotype data"""
    n_samples = 100

    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]

    # Generate clinical variables
    df = pd.DataFrame({
        'sample_id': sample_ids,
        'age': np.random.normal(35, 10, n_samples).clip(18, 70),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'bmi': np.random.normal(25, 5, n_samples).clip(15, 45),
        'diagnosis': np.random.choice(['ASD', 'ADHD', 'AuDHD', 'Control'], n_samples),
        'severity_score': np.random.normal(50, 15, n_samples).clip(0, 100),
        'iq': np.random.normal(100, 15, n_samples).clip(70, 140),
        'site': np.random.choice(['Site_A', 'Site_B', 'Site_C'], n_samples),
    })

    return df


@pytest.fixture(scope="session")
def synthetic_metabolomic_data():
    """Generate synthetic metabolomic data"""
    n_samples = 100
    n_metabolites = 200

    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    metabolite_ids = [f"METABOLITE_{i:04d}" for i in range(n_metabolites)]

    # Generate log-normal distributed metabolite levels
    data = np.random.lognormal(0, 1, (n_samples, n_metabolites))

    # Add some missing values (5%)
    missing_mask = np.random.random((n_samples, n_metabolites)) < 0.05
    data[missing_mask] = np.nan

    df = pd.DataFrame(data, index=sample_ids, columns=metabolite_ids)

    return df


@pytest.fixture(scope="session")
def synthetic_microbiome_data():
    """Generate synthetic microbiome abundance data"""
    n_samples = 100
    n_taxa = 150

    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    taxa_ids = [f"Genus_{i:03d}" for i in range(n_taxa)]

    # Generate Dirichlet-distributed relative abundances
    alpha = np.random.gamma(2, 2, n_taxa)
    data = np.random.dirichlet(alpha, n_samples)

    df = pd.DataFrame(data, index=sample_ids, columns=taxa_ids)

    return df


@pytest.fixture
def integrated_data():
    """Synthetic integrated data (MOFA-like factors)"""
    n_samples = 100
    n_factors = 15

    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    factor_names = [f"Factor_{i+1}" for i in range(n_factors)]

    # Generate latent factors
    factors = np.random.normal(0, 1, (n_samples, n_factors))

    df = pd.DataFrame(factors, index=sample_ids, columns=factor_names)

    return df


@pytest.fixture
def clustering_result(integrated_data):
    """Synthetic clustering result"""
    n_samples = len(integrated_data)

    # Generate cluster labels (4 clusters)
    labels = np.random.choice([0, 1, 2, 3], n_samples)

    # Generate UMAP embedding
    embedding = np.random.normal(0, 2, (n_samples, 2))

    return {
        'labels': labels,
        'embedding': embedding,
        'n_clusters': 4,
    }


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# ============================================================================
# Validation Fixtures
# ============================================================================

@pytest.fixture
def validation_metrics():
    """Expected validation metric ranges"""
    return {
        'silhouette': {'min': -1.0, 'max': 1.0, 'good': 0.5},
        'calinski_harabasz': {'min': 0.0, 'max': np.inf, 'good': 100.0},
        'davies_bouldin': {'min': 0.0, 'max': np.inf, 'good': 1.0},
    }


@pytest.fixture
def baseline_metrics():
    """Baseline metrics for regression testing"""
    return {
        'silhouette_score': 0.45,
        'calinski_harabasz_score': 150.0,
        'davies_bouldin_score': 1.2,
        'ari_stability': 0.75,
    }


# ============================================================================
# Helper Functions
# ============================================================================

def generate_clustered_data(
    n_samples: int = 100,
    n_features: int = 50,
    n_clusters: int = 3,
    cluster_std: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with known cluster structure"""
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    return X, y


@pytest.fixture
def clustered_data():
    """Fixture providing synthetic clustered data"""
    X, y = generate_clustered_data()
    return {'X': X, 'y': y}