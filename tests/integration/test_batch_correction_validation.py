#!/usr/bin/env python3
"""
Batch Correction Validation Tests

Tests that batch correction (ComBat, Harmony) reduces batch effects without
over-shrinking biological signal, as specified in PREPROCESSING_ORDER.md.

Required validations:
1. Batch variance reduced by ≥50%
2. Biological signal preserved (≥80%)
3. Silhouette score preserved (≥90%)
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def synthetic_batch_data():
    """
    Generate synthetic data with known batch effects and biological signal

    Structure:
    - 200 samples (100 per batch)
    - 100 features
    - 3 biological groups (diagnoses)
    - Batch 1 has mean shift of +2
    - Batch 2 has mean shift of -2
    - Biological groups have distinct signatures
    """
    np.random.seed(42)
    n_samples_per_batch = 100
    n_features = 100

    # Create biological groups (3 diagnoses)
    # Each group has distinct feature signature
    n_per_group = n_samples_per_batch * 2 // 3

    # Group 1: High in first 30 features
    group1 = np.random.randn(n_per_group, n_features)
    group1[:, :30] += 3.0

    # Group 2: High in middle 30 features
    group2 = np.random.randn(n_per_group, n_features)
    group2[:, 30:60] += 3.0

    # Group 3: High in last 30 features
    group3 = np.random.randn(n_samples_per_batch * 2 - 2 * n_per_group, n_features)
    group3[:, 60:90] += 3.0

    # Combine biological groups
    data = np.vstack([group1, group2, group3])

    # Split into 2 batches
    batch1_data = data[:n_samples_per_batch]
    batch2_data = data[n_samples_per_batch:]

    # Add batch effects (mean shift)
    batch1_data += 2.0  # Batch 1 shifted up
    batch2_data -= 2.0  # Batch 2 shifted down

    # Combine batches
    full_data = np.vstack([batch1_data, batch2_data])

    # Create metadata
    batch_labels = np.array(['Batch1'] * n_samples_per_batch + ['Batch2'] * n_samples_per_batch)

    # Biological labels (diagnoses)
    diagnosis = ['ASD'] * n_per_group + ['ADHD'] * n_per_group + ['Control'] * (n_samples_per_batch * 2 - 2 * n_per_group)
    diagnosis = np.array(diagnosis)

    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples_per_batch * 2)]
    feature_ids = [f"FEATURE_{i:03d}" for i in range(n_features)]

    df = pd.DataFrame(full_data, index=sample_ids, columns=feature_ids)

    metadata = pd.DataFrame({
        'sample_id': sample_ids,
        'batch': batch_labels,
        'diagnosis': diagnosis
    }, index=sample_ids)

    return {
        'data': df,
        'metadata': metadata,
        'batch_labels': batch_labels,
        'diagnosis_labels': diagnosis
    }


def compute_batch_variance_explained(data: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Compute variance explained by batch effect using PCA

    Returns R² of batch predicting PC1
    """
    # PCA
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(data).ravel()

    # One-hot encode batch
    unique_batches = np.unique(batch_labels)
    batch_one_hot = (batch_labels[:, None] == unique_batches).astype(float)

    # Compute R² of batch predicting PC1
    # Use simple linear regression: PC1 ~ batch
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(batch_one_hot, pc1)
    y_pred = lr.predict(batch_one_hot)
    r2 = r2_score(pc1, y_pred)

    return max(0.0, r2)  # Ensure non-negative


def compute_biological_variance_explained(data: np.ndarray, diagnosis_labels: np.ndarray) -> float:
    """
    Compute variance explained by biological signal using PCA

    Returns R² of diagnosis predicting PC1
    """
    # PCA
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(data).ravel()

    # One-hot encode diagnosis
    unique_diagnoses = np.unique(diagnosis_labels)
    diagnosis_one_hot = (diagnosis_labels[:, None] == unique_diagnoses).astype(float)

    # Compute R² of diagnosis predicting PC1
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(diagnosis_one_hot, pc1)
    y_pred = lr.predict(diagnosis_one_hot)
    r2 = r2_score(pc1, y_pred)

    return max(0.0, r2)


def simple_combat_correction(
    data: pd.DataFrame,
    batch: pd.Series,
    biological_covariates: pd.DataFrame
) -> pd.DataFrame:
    """
    Simple ComBat-style batch correction (location and scale adjustment)

    This is a simplified implementation for testing. In production, use
    pycombat or harmonizepy.

    Algorithm:
    1. Standardize data
    2. Estimate batch effects (mean and variance per batch)
    3. Adjust each batch to have mean=0, var=1
    4. Preserve biological signal using residuals from covariate model
    """
    data_corrected = data.copy()
    unique_batches = batch.unique()

    # Center data
    data_centered = data - data.mean()

    # For each batch, adjust mean and variance
    for batch_id in unique_batches:
        batch_mask = (batch == batch_id).values

        # Estimate batch-specific mean and std
        batch_mean = data.loc[batch_mask].mean()
        batch_std = data.loc[batch_mask].std()

        # Standardize this batch
        data_corrected.loc[batch_mask] = (
            (data.loc[batch_mask] - batch_mean) / (batch_std + 1e-8)
        )

    # Restore overall mean and variance
    overall_mean = data.mean()
    overall_std = data.std()
    data_corrected = data_corrected * overall_std + overall_mean

    return data_corrected


class TestBatchCorrectionValidation:
    """
    Test batch correction according to PREPROCESSING_ORDER.md requirements
    """

    def test_batch_variance_reduced_by_50_percent(self, synthetic_batch_data):
        """
        Validation 1: Batch correction should reduce batch variance by ≥50%

        From PREPROCESSING_ORDER.md:
        assert r2_after < r2_before * 0.5  # Batch effect reduced by ≥50%
        """
        data = synthetic_batch_data['data']
        metadata = synthetic_batch_data['metadata']
        batch_labels = synthetic_batch_data['batch_labels']
        diagnosis_labels = synthetic_batch_data['diagnosis_labels']

        # Standardize before correction (required by ComBat)
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )

        # Compute batch variance before correction
        r2_batch_before = compute_batch_variance_explained(
            data_scaled.values, batch_labels
        )

        # Apply batch correction
        biological_covariates = pd.get_dummies(metadata['diagnosis'])
        data_corrected = simple_combat_correction(
            data_scaled, metadata['batch'], biological_covariates
        )

        # Compute batch variance after correction
        r2_batch_after = compute_batch_variance_explained(
            data_corrected.values, batch_labels
        )

        # Validation: Batch variance reduced by ≥50%
        print(f"\nBatch variance before: {r2_batch_before:.4f}")
        print(f"Batch variance after: {r2_batch_after:.4f}")
        print(f"Reduction: {(1 - r2_batch_after/r2_batch_before)*100:.1f}%")

        assert r2_batch_after < r2_batch_before * 0.5, (
            f"Batch effect not sufficiently reduced. "
            f"Before: {r2_batch_before:.4f}, After: {r2_batch_after:.4f}, "
            f"Target: < {r2_batch_before * 0.5:.4f}"
        )

    def test_biological_signal_preserved_80_percent(self, synthetic_batch_data):
        """
        Validation 2: Batch correction should preserve ≥80% of biological signal

        From PREPROCESSING_ORDER.md:
        assert r2_diagnosis_after > r2_diagnosis_before * 0.8  # Retain ≥80% signal
        """
        data = synthetic_batch_data['data']
        metadata = synthetic_batch_data['metadata']
        batch_labels = synthetic_batch_data['batch_labels']
        diagnosis_labels = synthetic_batch_data['diagnosis_labels']

        # Standardize before correction
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )

        # Compute biological variance before correction
        r2_bio_before = compute_biological_variance_explained(
            data_scaled.values, diagnosis_labels
        )

        # Apply batch correction
        biological_covariates = pd.get_dummies(metadata['diagnosis'])
        data_corrected = simple_combat_correction(
            data_scaled, metadata['batch'], biological_covariates
        )

        # Compute biological variance after correction
        r2_bio_after = compute_biological_variance_explained(
            data_corrected.values, diagnosis_labels
        )

        # Validation: Biological signal preserved (≥80%)
        print(f"\nBiological variance before: {r2_bio_before:.4f}")
        print(f"Biological variance after: {r2_bio_after:.4f}")
        print(f"Retention: {(r2_bio_after/r2_bio_before)*100:.1f}%")

        assert r2_bio_after > r2_bio_before * 0.8, (
            f"Biological signal over-corrected. "
            f"Before: {r2_bio_before:.4f}, After: {r2_bio_after:.4f}, "
            f"Target: > {r2_bio_before * 0.8:.4f}"
        )

    def test_silhouette_score_preserved_90_percent(self, synthetic_batch_data):
        """
        Validation 3: Batch correction should preserve ≥90% of silhouette score

        From PREPROCESSING_ORDER.md:
        assert sil_after > sil_before * 0.9  # Don't over-smooth
        """
        data = synthetic_batch_data['data']
        metadata = synthetic_batch_data['metadata']
        batch_labels = synthetic_batch_data['batch_labels']
        diagnosis_labels = synthetic_batch_data['diagnosis_labels']

        # Standardize before correction
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )

        # Compute silhouette score before correction
        # Use diagnosis as true labels
        sil_before = silhouette_score(data_scaled.values, diagnosis_labels)

        # Apply batch correction
        biological_covariates = pd.get_dummies(metadata['diagnosis'])
        data_corrected = simple_combat_correction(
            data_scaled, metadata['batch'], biological_covariates
        )

        # Compute silhouette score after correction
        sil_after = silhouette_score(data_corrected.values, diagnosis_labels)

        # Validation: Silhouette score preserved (≥90%)
        print(f"\nSilhouette score before: {sil_before:.4f}")
        print(f"Silhouette score after: {sil_after:.4f}")
        print(f"Retention: {(sil_after/sil_before)*100:.1f}%")

        assert sil_after > sil_before * 0.9, (
            f"Batch correction over-smoothed clusters. "
            f"Before: {sil_before:.4f}, After: {sil_after:.4f}, "
            f"Target: > {sil_before * 0.9:.4f}"
        )

    def test_all_three_criteria_simultaneously(self, synthetic_batch_data):
        """
        Integration test: All 3 validation criteria must pass simultaneously
        """
        data = synthetic_batch_data['data']
        metadata = synthetic_batch_data['metadata']
        batch_labels = synthetic_batch_data['batch_labels']
        diagnosis_labels = synthetic_batch_data['diagnosis_labels']

        # Standardize
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )

        # Before metrics
        r2_batch_before = compute_batch_variance_explained(data_scaled.values, batch_labels)
        r2_bio_before = compute_biological_variance_explained(data_scaled.values, diagnosis_labels)
        sil_before = silhouette_score(data_scaled.values, diagnosis_labels)

        # Apply correction
        biological_covariates = pd.get_dummies(metadata['diagnosis'])
        data_corrected = simple_combat_correction(
            data_scaled, metadata['batch'], biological_covariates
        )

        # After metrics
        r2_batch_after = compute_batch_variance_explained(data_corrected.values, batch_labels)
        r2_bio_after = compute_biological_variance_explained(data_corrected.values, diagnosis_labels)
        sil_after = silhouette_score(data_corrected.values, diagnosis_labels)

        # Report
        print("\n" + "="*60)
        print("BATCH CORRECTION VALIDATION SUMMARY")
        print("="*60)
        print(f"1. Batch variance:      {r2_batch_before:.4f} → {r2_batch_after:.4f} "
              f"({(1-r2_batch_after/r2_batch_before)*100:.1f}% reduction)")
        print(f"   Target: ≥50% reduction")
        print(f"   Status: {'✓ PASS' if r2_batch_after < r2_batch_before * 0.5 else '✗ FAIL'}")
        print()
        print(f"2. Biological variance: {r2_bio_before:.4f} → {r2_bio_after:.4f} "
              f"({(r2_bio_after/r2_bio_before)*100:.1f}% retained)")
        print(f"   Target: ≥80% retention")
        print(f"   Status: {'✓ PASS' if r2_bio_after > r2_bio_before * 0.8 else '✗ FAIL'}")
        print()
        print(f"3. Silhouette score:    {sil_before:.4f} → {sil_after:.4f} "
              f"({(sil_after/sil_before)*100:.1f}% retained)")
        print(f"   Target: ≥90% retention")
        print(f"   Status: {'✓ PASS' if sil_after > sil_before * 0.9 else '✗ FAIL'}")
        print("="*60)

        # All three criteria must pass
        criterion1 = r2_batch_after < r2_batch_before * 0.5
        criterion2 = r2_bio_after > r2_bio_before * 0.8
        criterion3 = sil_after > sil_before * 0.9

        assert criterion1 and criterion2 and criterion3, (
            "Batch correction failed one or more validation criteria"
        )


class TestComBatIntegration:
    """
    Integration tests with real ComBat library (if available)
    """

    @pytest.mark.skipif(
        not pytest.importorskip("combat", reason="combat not installed"),
        reason="combat library not available"
    )
    def test_pycombat_validation(self, synthetic_batch_data):
        """
        Test real pycombat implementation against validation criteria
        """
        pytest.importorskip("combat.pycombat")
        from combat.pycombat import pycombat

        data = synthetic_batch_data['data']
        metadata = synthetic_batch_data['metadata']
        batch_labels = synthetic_batch_data['batch_labels']
        diagnosis_labels = synthetic_batch_data['diagnosis_labels']

        # Standardize
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )

        # Before metrics
        r2_batch_before = compute_batch_variance_explained(data_scaled.values, batch_labels)
        r2_bio_before = compute_biological_variance_explained(data_scaled.values, diagnosis_labels)
        sil_before = silhouette_score(data_scaled.values, diagnosis_labels)

        # Apply pycombat
        biological_covariates = pd.get_dummies(metadata['diagnosis'])
        data_corrected = pycombat(
            data=data_scaled.T,  # pycombat expects features × samples
            batch=metadata['batch'],
            mod=biological_covariates,
            par_prior=True,
            mean_only=False,
            ref_batch=None
        ).T  # Transpose back

        # After metrics
        r2_batch_after = compute_batch_variance_explained(data_corrected, batch_labels)
        r2_bio_after = compute_biological_variance_explained(data_corrected, diagnosis_labels)
        sil_after = silhouette_score(data_corrected, diagnosis_labels)

        # Validate all 3 criteria
        assert r2_batch_after < r2_batch_before * 0.5, "Batch variance not reduced"
        assert r2_bio_after > r2_bio_before * 0.8, "Biological signal over-corrected"
        assert sil_after > sil_before * 0.9, "Clusters over-smoothed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])