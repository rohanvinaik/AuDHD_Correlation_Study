"""Abstract base classes for data loaders"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np


class MissingDataType(Enum):
    """Classification of missing data mechanisms"""

    MCAR = "missing_completely_at_random"  # Random missingness
    MAR = "missing_at_random"  # Depends on observed data
    MNAR = "missing_not_at_random"  # Depends on unobserved data
    UNKNOWN = "unknown"


class DatasetSource(Enum):
    """Supported dataset sources"""

    SPARK = "spark"
    SSC = "ssc"
    ABCD = "abcd"
    UKB = "uk_biobank"
    METABOLIGHTS = "metabolights"
    HCP = "hcp"


@dataclass
class QCMetrics:
    """Quality control metrics for loaded data"""

    n_samples: int
    n_features: int
    missing_rate: float
    duplicate_samples: int
    outlier_samples: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "missing_rate": self.missing_rate,
            "duplicate_samples": self.duplicate_samples,
            "outlier_samples": self.outlier_samples,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DataMetadata:
    """Metadata for loaded data"""

    source: DatasetSource
    modality: str
    file_path: Path
    load_timestamp: datetime = field(default_factory=datetime.now)
    batch_id: Optional[str] = None
    collection_date: Optional[datetime] = None
    storage_conditions: Optional[str] = None
    context_variables: Dict[str, Any] = field(default_factory=dict)
    qc_metrics: Optional[QCMetrics] = None
    missing_data_type: MissingDataType = MissingDataType.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "source": self.source.value,
            "modality": self.modality,
            "file_path": str(self.file_path),
            "load_timestamp": self.load_timestamp.isoformat(),
            "batch_id": self.batch_id,
            "collection_date": self.collection_date.isoformat()
            if self.collection_date
            else None,
            "storage_conditions": self.storage_conditions,
            "context_variables": self.context_variables,
            "qc_metrics": self.qc_metrics.to_dict() if self.qc_metrics else None,
            "missing_data_type": self.missing_data_type.value,
        }


@dataclass
class LoadedData:
    """Container for loaded data with metadata"""

    data: pd.DataFrame
    metadata: DataMetadata
    sample_metadata: Optional[pd.DataFrame] = None
    feature_metadata: Optional[pd.DataFrame] = None

    def validate(self) -> List[str]:
        """Validate loaded data consistency"""
        errors = []

        if self.data.empty:
            errors.append("Data DataFrame is empty")

        if self.sample_metadata is not None:
            if len(self.sample_metadata) != len(self.data):
                errors.append(
                    f"Sample metadata length ({len(self.sample_metadata)}) "
                    f"does not match data length ({len(self.data)})"
                )

        if self.feature_metadata is not None:
            if len(self.feature_metadata) != self.data.shape[1]:
                errors.append(
                    f"Feature metadata length ({len(self.feature_metadata)}) "
                    f"does not match number of features ({self.data.shape[1]})"
                )

        return errors


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders"""

    def __init__(
        self,
        source: DatasetSource,
        modality: str,
        validate_on_load: bool = True,
        generate_qc_report: bool = True,
    ):
        """
        Initialize base data loader

        Args:
            source: Dataset source (SPARK, SSC, etc.)
            modality: Data modality (genetic, metabolomic, etc.)
            validate_on_load: Whether to validate data after loading
            generate_qc_report: Whether to generate QC report
        """
        self.source = source
        self.modality = modality
        self.validate_on_load = validate_on_load
        self.generate_qc_report = generate_qc_report

    @abstractmethod
    def load(self, file_path: Union[str, Path], **kwargs) -> LoadedData:
        """
        Load data from file

        Args:
            file_path: Path to data file
            **kwargs: Additional loader-specific arguments

        Returns:
            LoadedData object with data and metadata
        """
        pass

    @abstractmethod
    def _parse_file(self, file_path: Path) -> pd.DataFrame:
        """
        Parse file into DataFrame

        Args:
            file_path: Path to data file

        Returns:
            Parsed DataFrame
        """
        pass

    def _generate_qc_metrics(
        self, data: pd.DataFrame, sample_metadata: Optional[pd.DataFrame] = None
    ) -> QCMetrics:
        """
        Generate QC metrics for loaded data

        Args:
            data: Loaded data DataFrame
            sample_metadata: Optional sample metadata

        Returns:
            QCMetrics object
        """
        n_samples, n_features = data.shape
        missing_rate = data.isnull().sum().sum() / (n_samples * n_features)
        duplicate_samples = data.index.duplicated().sum()

        # Detect outliers (simple Z-score method)
        outlier_samples = []
        if n_features > 0 and n_samples > 0:
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
                outlier_mask = (z_scores > 5).any(axis=1)
                outlier_samples = data.index[outlier_mask].tolist()

        warnings = []
        if missing_rate > 0.5:
            warnings.append(f"High missing rate: {missing_rate:.2%}")
        if duplicate_samples > 0:
            warnings.append(f"Found {duplicate_samples} duplicate samples")
        if len(outlier_samples) > n_samples * 0.1:
            warnings.append(f"High outlier rate: {len(outlier_samples)} samples")

        return QCMetrics(
            n_samples=n_samples,
            n_features=n_features,
            missing_rate=missing_rate,
            duplicate_samples=duplicate_samples,
            outlier_samples=outlier_samples,
            warnings=warnings,
        )

    def _classify_missing_data(
        self, data: pd.DataFrame, sample_metadata: Optional[pd.DataFrame] = None
    ) -> MissingDataType:
        """
        Classify missing data mechanism

        Args:
            data: Data DataFrame
            sample_metadata: Optional sample metadata

        Returns:
            MissingDataType classification
        """
        missing_mask = data.isnull()

        if missing_mask.sum().sum() == 0:
            return MissingDataType.MCAR

        # Simple heuristic: if missingness correlates with metadata, likely MAR
        if sample_metadata is not None:
            for col in sample_metadata.select_dtypes(include=[np.number]).columns:
                missing_counts = missing_mask.sum(axis=1)
                corr = np.corrcoef(sample_metadata[col].fillna(0), missing_counts)[0, 1]
                if abs(corr) > 0.3:
                    return MissingDataType.MAR

        # Default to unknown
        return MissingDataType.UNKNOWN

    def _extract_context_variables(
        self, sample_metadata: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Extract context variables from sample metadata

        Args:
            sample_metadata: Sample metadata DataFrame

        Returns:
            Dictionary of context variables
        """
        context = {}

        if sample_metadata is None:
            return context

        # Extract common context variables
        context_cols = [
            "fasting_status",
            "collection_time",
            "time_of_day",
            "storage_duration",
            "freeze_thaw_cycles",
            "medication_status",
        ]

        for col in context_cols:
            if col in sample_metadata.columns:
                context[col] = sample_metadata[col].value_counts().to_dict()

        return context

    def generate_qc_report(
        self, loaded_data: LoadedData, output_path: Optional[Path] = None
    ) -> str:
        """
        Generate QC report for loaded data

        Args:
            loaded_data: LoadedData object
            output_path: Optional path to save report

        Returns:
            QC report as string
        """
        qc = loaded_data.metadata.qc_metrics
        if qc is None:
            return "No QC metrics available"

        report = []
        report.append(f"QC Report: {loaded_data.metadata.modality}")
        report.append(f"Source: {loaded_data.metadata.source.value}")
        report.append(f"Timestamp: {qc.timestamp.isoformat()}")
        report.append("-" * 50)
        report.append(f"Samples: {qc.n_samples}")
        report.append(f"Features: {qc.n_features}")
        report.append(f"Missing rate: {qc.missing_rate:.2%}")
        report.append(f"Duplicate samples: {qc.duplicate_samples}")
        report.append(f"Outlier samples: {len(qc.outlier_samples)}")

        if qc.warnings:
            report.append("\nWarnings:")
            for warning in qc.warnings:
                report.append(f"  - {warning}")

        if qc.failed_checks:
            report.append("\nFailed checks:")
            for check in qc.failed_checks:
                report.append(f"  - {check}")

        report_text = "\n".join(report)

        if output_path:
            output_path.write_text(report_text)

        return report_text