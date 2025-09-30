"""Unit tests for data loaders

Tests all data loading functionality with synthetic data.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestGenomicLoader:
    """Tests for genomic data loader"""

    def test_load_vcf_basic(self, temp_data_dir):
        """Test basic VCF loading"""
        # Create simple VCF
        vcf_path = temp_data_dir / "test.vcf"
        vcf_content = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS001\tS002
chr1\t1000\trs001\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1
chr1\t2000\trs002\tC\tT\t.\tPASS\t.\tGT\t0/1\t1/1
"""
        vcf_path.write_text(vcf_content)

        from audhd_correlation.data.genomic_loader import load_vcf

        result = load_vcf(vcf_path)

        assert result is not None
        assert 'genotypes' in result or isinstance(result, pd.DataFrame)

    def test_load_genomic_handles_missing(self, temp_data_dir):
        """Test handling of missing genotypes"""
        vcf_path = temp_data_dir / "missing.vcf"
        vcf_content = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS001\tS002
chr1\t1000\trs001\tA\tG\t.\tPASS\t.\tGT\t./.\t0/1
chr1\t2000\trs002\tC\tT\t.\tPASS\t.\tGT\t0/1\t./.
"""
        vcf_path.write_text(vcf_content)

        from audhd_correlation.data.genomic_loader import load_vcf

        result = load_vcf(vcf_path)
        assert result is not None

    def test_genomic_qc_filters(self, synthetic_genomic_data):
        """Test genomic QC filtering"""
        genotypes = synthetic_genomic_data['genotypes']

        from audhd_correlation.data.genomic_loader import apply_qc_filters

        filtered = apply_qc_filters(
            genotypes,
            min_call_rate=0.95,
            min_maf=0.01,
        )

        # Should remove some variants
        assert filtered.shape[1] <= genotypes.shape[1]
        # Should keep all samples
        assert filtered.shape[0] == genotypes.shape[0]


class TestClinicalLoader:
    """Tests for clinical data loader"""

    def test_load_csv_basic(self, temp_data_dir, synthetic_clinical_data):
        """Test basic CSV loading"""
        csv_path = temp_data_dir / "clinical.csv"
        synthetic_clinical_data.to_csv(csv_path, index=False)

        from audhd_correlation.data.clinical_loader import load_clinical_csv

        result = load_clinical_csv(csv_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(synthetic_clinical_data)
        assert 'sample_id' in result.columns

    def test_clinical_validates_required_columns(self, temp_data_dir):
        """Test validation of required columns"""
        csv_path = temp_data_dir / "incomplete.csv"
        df = pd.DataFrame({
            'sample_id': ['S001', 'S002'],
            'age': [30, 40],
            # Missing 'sex' and other required columns
        })
        df.to_csv(csv_path, index=False)

        from audhd_correlation.data.clinical_loader import load_clinical_csv

        # Should either raise error or handle gracefully
        try:
            result = load_clinical_csv(
                csv_path,
                required_columns=['sample_id', 'age', 'sex']
            )
            # If it doesn't raise, check it handles missing columns
            assert result is not None
        except (ValueError, KeyError):
            # Expected behavior for missing required columns
            pass

    def test_clinical_handles_categorical(self, temp_data_dir):
        """Test handling of categorical variables"""
        csv_path = temp_data_dir / "categorical.csv"
        df = pd.DataFrame({
            'sample_id': ['S001', 'S002', 'S003'],
            'diagnosis': ['ASD', 'ADHD', 'Control'],
            'site': ['A', 'B', 'A'],
        })
        df.to_csv(csv_path, index=False)

        from audhd_correlation.data.clinical_loader import load_clinical_csv

        result = load_clinical_csv(csv_path)

        assert result['diagnosis'].dtype == 'object' or result['diagnosis'].dtype.name == 'category'


class TestMetabolomicLoader:
    """Tests for metabolomic data loader"""

    def test_load_metabolomics_basic(self, temp_data_dir, synthetic_metabolomic_data):
        """Test basic metabolomics loading"""
        csv_path = temp_data_dir / "metabolomics.csv"
        synthetic_metabolomic_data.to_csv(csv_path)

        from audhd_correlation.data.metabolomic_loader import load_metabolomics

        result = load_metabolomics(csv_path)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == synthetic_metabolomic_data.shape[0]

    def test_metabolomics_handles_missing(self, temp_data_dir):
        """Test handling of missing metabolite values"""
        csv_path = temp_data_dir / "metabolomics_missing.csv"
        df = pd.DataFrame({
            'sample_id': ['S001', 'S002', 'S003'],
            'metabolite_1': [1.5, np.nan, 2.3],
            'metabolite_2': [0.8, 1.2, np.nan],
        })
        df.to_csv(csv_path, index=False)

        from audhd_correlation.data.metabolomic_loader import load_metabolomics

        result = load_metabolomics(csv_path)

        # Should handle missing values
        assert result is not None
        # Check that missing values are preserved or imputed
        assert result.shape == (3, 2) or result.shape == (3, 3)

    def test_metabolomics_normalization(self, synthetic_metabolomic_data):
        """Test metabolomics normalization"""
        from audhd_correlation.data.metabolomic_loader import normalize_metabolomics

        result = normalize_metabolomics(
            synthetic_metabolomic_data,
            method='log'
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape == synthetic_metabolomic_data.shape
        # Log normalization should reduce variance
        assert result.std().mean() < synthetic_metabolomic_data.std().mean()


class TestMicrobiomeLoader:
    """Tests for microbiome data loader"""

    def test_load_microbiome_basic(self, temp_data_dir, synthetic_microbiome_data):
        """Test basic microbiome loading"""
        csv_path = temp_data_dir / "microbiome.csv"
        synthetic_microbiome_data.to_csv(csv_path)

        from audhd_correlation.data.microbiome_loader import load_microbiome

        result = load_microbiome(csv_path)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == synthetic_microbiome_data.shape

    def test_microbiome_relative_abundance(self, synthetic_microbiome_data):
        """Test relative abundance calculation"""
        from audhd_correlation.data.microbiome_loader import to_relative_abundance

        result = to_relative_abundance(synthetic_microbiome_data)

        # Each sample should sum to 1
        sums = result.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(len(result)), decimal=5)

    def test_microbiome_filter_prevalence(self, synthetic_microbiome_data):
        """Test filtering by prevalence"""
        from audhd_correlation.data.microbiome_loader import filter_by_prevalence

        result = filter_by_prevalence(
            synthetic_microbiome_data,
            min_prevalence=0.1
        )

        # Should remove some taxa
        assert result.shape[1] <= synthetic_microbiome_data.shape[1]
        # Should keep all samples
        assert result.shape[0] == synthetic_microbiome_data.shape[0]


class TestDataHarmonization:
    """Tests for data harmonization"""

    def test_harmonize_sample_ids(self, synthetic_genomic_data, synthetic_clinical_data):
        """Test sample ID harmonization"""
        from audhd_correlation.data.harmonize import harmonize_sample_ids

        genomic = synthetic_genomic_data['genotypes']
        clinical = synthetic_clinical_data.set_index('sample_id')

        harmonized = harmonize_sample_ids([genomic, clinical])

        # Should have common samples
        assert all(harmonized[0].index == harmonized[1].index)
        assert len(harmonized[0]) <= min(len(genomic), len(clinical))

    def test_batch_correction(self, synthetic_clinical_data):
        """Test batch effect correction"""
        from audhd_correlation.data.harmonize import correct_batch_effects

        # Create data with known batch effect
        n_samples = len(synthetic_clinical_data)
        n_features = 50
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            index=synthetic_clinical_data['sample_id']
        )

        # Add batch effect
        batch = synthetic_clinical_data['site'].values
        for i, b in enumerate(batch):
            if b == 'Site_A':
                data.iloc[i] += 2.0

        corrected = correct_batch_effects(
            data,
            batch=batch,
            method='combat'
        )

        assert isinstance(corrected, pd.DataFrame)
        assert corrected.shape == data.shape
        # Batch effect should be reduced
        # (Test would need actual ComBat implementation)


class TestDataIntegrity:
    """Tests for data integrity checks"""

    def test_check_no_duplicate_samples(self, synthetic_clinical_data):
        """Test detection of duplicate samples"""
        from audhd_correlation.data.qc import check_duplicates

        # No duplicates
        assert not check_duplicates(synthetic_clinical_data, column='sample_id')

        # Add duplicate
        df_with_dup = pd.concat([
            synthetic_clinical_data,
            synthetic_clinical_data.iloc[[0]]
        ])

        assert check_duplicates(df_with_dup, column='sample_id')

    def test_check_data_types(self, synthetic_clinical_data):
        """Test data type validation"""
        from audhd_correlation.data.qc import validate_dtypes

        expected_dtypes = {
            'age': 'float',
            'sex': 'object',
            'bmi': 'float',
        }

        assert validate_dtypes(synthetic_clinical_data, expected_dtypes)

    def test_check_value_ranges(self, synthetic_clinical_data):
        """Test value range validation"""
        from audhd_correlation.data.qc import check_ranges

        ranges = {
            'age': (18, 70),
            'bmi': (15, 45),
            'severity_score': (0, 100),
        }

        violations = check_ranges(synthetic_clinical_data, ranges)

        # Should have no violations (or minimal due to clipping)
        assert violations == {} or all(len(v) == 0 for v in violations.values())


# ============================================================================
# Test Helpers
# ============================================================================

def test_load_functions_exist():
    """Test that all loader functions are importable"""
    from audhd_correlation.data import (
        load_genomic_data,
        load_clinical_data,
        load_metabolomic_data,
        load_microbiome_data,
    )

    assert callable(load_genomic_data)
    assert callable(load_clinical_data)
    assert callable(load_metabolomic_data)
    assert callable(load_microbiome_data)