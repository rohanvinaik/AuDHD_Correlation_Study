"""Tests for data loader system"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from src.audhd_correlation.data.base import (
    BaseDataLoader,
    DatasetSource,
    MissingDataType,
    QCMetrics,
    DataMetadata,
    LoadedData,
)
from src.audhd_correlation.data.genomic_loader import GenomicLoader, PRSLoader
from src.audhd_correlation.data.metabolomic_loader import MetabolomicLoader
from src.audhd_correlation.data.clinical_loader import ClinicalLoader
from src.audhd_correlation.data.microbiome_loader import MicrobiomeLoader


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_genetic_data(temp_dir):
    """Create sample genetic data file"""
    data = pd.DataFrame(
        {
            "PRS_autism": [0.5, 0.3, 0.7, 0.2],
            "PRS_ADHD": [0.6, 0.4, 0.8, 0.3],
            "PRS_depression": [0.4, 0.5, 0.6, 0.7],
        },
        index=["sample1", "sample2", "sample3", "sample4"],
    )
    file_path = temp_dir / "prs_scores.csv"
    data.to_csv(file_path)
    return file_path


@pytest.fixture
def sample_metabolomic_data(temp_dir):
    """Create sample metabolomic data file"""
    data = pd.DataFrame(
        {
            "serotonin": [100.0, 150.0, np.nan, 120.0],
            "dopamine": [50.0, 60.0, 55.0, 58.0],
            "GABA": [200.0, 220.0, 210.0, np.nan],
        },
        index=["sample1", "sample2", "sample3", "sample4"],
    )
    file_path = temp_dir / "metabolites.csv"
    data.to_csv(file_path)
    return file_path


@pytest.fixture
def sample_clinical_data(temp_dir):
    """Create sample clinical data file"""
    data = pd.DataFrame(
        {
            "age": [8, 10, 12, 9],
            "sex": [1, 0, 1, 0],
            "ADHD_RS_total": [45, 38, 52, 41],
            "SCQ_total": [12, 18, 15, np.nan],
            "ADHD_diagnosis": [1, 1, 1, 1],
        },
        index=["sample1", "sample2", "sample3", "sample4"],
    )
    file_path = temp_dir / "phenotypes.csv"
    data.to_csv(file_path)
    return file_path


def test_qc_metrics_creation():
    """Test QCMetrics creation and serialization"""
    qc = QCMetrics(
        n_samples=100,
        n_features=50,
        missing_rate=0.1,
        duplicate_samples=2,
        outlier_samples=["sample1", "sample2"],
        warnings=["High missing rate"],
    )

    assert qc.n_samples == 100
    assert qc.n_features == 50
    assert qc.missing_rate == 0.1
    assert len(qc.outlier_samples) == 2

    # Test serialization
    qc_dict = qc.to_dict()
    assert "n_samples" in qc_dict
    assert "timestamp" in qc_dict


def test_genomic_loader_prs(sample_genetic_data):
    """Test loading PRS scores with GenomicLoader"""
    loader = PRSLoader(source=DatasetSource.SPARK)
    loaded = loader.load(sample_genetic_data)

    assert isinstance(loaded, LoadedData)
    assert loaded.data.shape == (4, 3)
    assert "PRS_autism" in loaded.data.columns
    assert loaded.metadata.modality == "genetic"
    assert loaded.metadata.source == DatasetSource.SPARK


def test_genomic_loader_qc(sample_genetic_data):
    """Test QC metrics generation for genetic data"""
    loader = PRSLoader(source=DatasetSource.SPARK, generate_qc_report=True)
    loaded = loader.load(sample_genetic_data)

    assert loaded.metadata.qc_metrics is not None
    assert loaded.metadata.qc_metrics.n_samples == 4
    assert loaded.metadata.qc_metrics.n_features == 3
    assert loaded.metadata.qc_metrics.missing_rate == 0.0


def test_metabolomic_loader(sample_metabolomic_data):
    """Test loading metabolomic data"""
    loader = MetabolomicLoader(source=DatasetSource.ABCD)
    loaded = loader.load(sample_metabolomic_data)

    assert isinstance(loaded, LoadedData)
    assert loaded.data.shape == (4, 3)
    assert loaded.metadata.modality == "metabolomic"
    assert loaded.metadata.qc_metrics is not None


def test_metabolomic_missing_data_classification(sample_metabolomic_data):
    """Test missing data classification for metabolomics"""
    loader = MetabolomicLoader(source=DatasetSource.ABCD)
    loaded = loader.load(sample_metabolomic_data)

    # Should detect missing data
    assert loaded.metadata.qc_metrics.missing_rate > 0
    # Missing data type should be classified
    assert loaded.metadata.missing_data_type in [
        MissingDataType.MCAR,
        MissingDataType.MAR,
        MissingDataType.MNAR,
        MissingDataType.UNKNOWN,
    ]


def test_metabolomic_feature_metadata(sample_metabolomic_data):
    """Test feature metadata extraction for metabolomics"""
    loader = MetabolomicLoader(source=DatasetSource.ABCD)
    loaded = loader.load(sample_metabolomic_data)

    assert loaded.feature_metadata is not None
    assert "metabolite_class" in loaded.feature_metadata.columns
    # Check neurotransmitter classification
    assert (
        loaded.feature_metadata.loc["serotonin", "metabolite_class"]
        == "neurotransmitter"
    )


def test_clinical_loader(sample_clinical_data):
    """Test loading clinical data"""
    loader = ClinicalLoader(source=DatasetSource.SSC)
    loaded = loader.load(sample_clinical_data)

    assert isinstance(loaded, LoadedData)
    assert loaded.data.shape == (4, 5)
    assert loaded.metadata.modality == "clinical"
    assert "age" in loaded.data.columns


def test_clinical_range_checks(sample_clinical_data):
    """Test range validation for clinical data"""
    loader = ClinicalLoader(source=DatasetSource.SSC, generate_qc_report=True)
    loaded = loader.load(sample_clinical_data)

    # Should not have range errors for valid data
    assert loaded.metadata.qc_metrics is not None
    # Check that no failed checks for valid ranges
    failed_checks = [
        check
        for check in loaded.metadata.qc_metrics.failed_checks
        if "out-of-range" in check.lower()
    ]
    assert len(failed_checks) == 0


def test_clinical_feature_categorization(sample_clinical_data):
    """Test feature categorization for clinical data"""
    loader = ClinicalLoader(source=DatasetSource.SSC)
    loaded = loader.load(sample_clinical_data)

    assert loaded.feature_metadata is not None
    assert "category" in loaded.feature_metadata.columns

    # Check categorization
    assert loaded.feature_metadata.loc["age", "category"] == "demographic"
    assert loaded.feature_metadata.loc["ADHD_RS_total", "category"] == "ADHD_phenotype"
    assert (
        loaded.feature_metadata.loc["SCQ_total", "category"] == "autism_phenotype"
    )


def test_loaded_data_validation():
    """Test LoadedData validation"""
    # Create mismatched data and metadata
    data = pd.DataFrame(np.random.rand(10, 5))
    sample_metadata = pd.DataFrame(np.random.rand(8, 2))  # Wrong number of samples

    metadata = DataMetadata(
        source=DatasetSource.SPARK,
        modality="test",
        file_path=Path("test.csv"),
    )

    loaded = LoadedData(
        data=data, metadata=metadata, sample_metadata=sample_metadata
    )

    errors = loaded.validate()
    assert len(errors) > 0
    assert any("does not match" in error for error in errors)


def test_qc_report_generation(sample_clinical_data):
    """Test QC report generation"""
    loader = ClinicalLoader(source=DatasetSource.SSC, generate_qc_report=True)
    loaded = loader.load(sample_clinical_data)

    report = loader.generate_qc_report(loaded)

    assert "QC Report" in report
    assert "clinical" in report
    assert "Samples:" in report
    assert "Features:" in report


def test_microbiome_loader_csv(temp_dir):
    """Test loading microbiome data from CSV"""
    # Create sample microbiome data
    data = pd.DataFrame(
        {
            "OTU1": [100, 150, 200, 120],
            "OTU2": [50, 60, 55, 58],
            "OTU3": [0, 20, 10, 5],
        },
        index=["sample1", "sample2", "sample3", "sample4"],
    )
    file_path = temp_dir / "microbiome.csv"
    data.to_csv(file_path)

    loader = MicrobiomeLoader(source=DatasetSource.ABCD)
    loaded = loader.load(file_path)

    assert loaded.data.shape == (4, 3)
    assert loaded.metadata.modality == "microbiome"
    # Microbiome missing data is always MNAR
    assert loaded.metadata.missing_data_type == MissingDataType.MNAR


def test_loader_file_format_detection(temp_dir):
    """Test that loaders handle different file formats"""
    # Create TSV file
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    tsv_file = temp_dir / "data.tsv"
    data.to_csv(tsv_file, sep="\t")

    loader = ClinicalLoader(source=DatasetSource.SPARK)
    loaded = loader.load(tsv_file)

    assert loaded.data.shape == (3, 2)


def test_empty_data_handling(temp_dir):
    """Test handling of empty data files"""
    # Create empty CSV
    empty_file = temp_dir / "empty.csv"
    pd.DataFrame().to_csv(empty_file)

    loader = MetabolomicLoader(source=DatasetSource.SPARK)
    loaded = loader.load(empty_file)

    # Should handle empty data gracefully
    assert loaded.data.empty
    if loaded.metadata.qc_metrics:
        assert loaded.metadata.qc_metrics.n_samples == 0