"""Metabolomic data loader with CSV/HDF5 support"""
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np

from .base import (
    BaseDataLoader,
    DataMetadata,
    DatasetSource,
    LoadedData,
    MissingDataType,
)


class MetabolomicLoader(BaseDataLoader):
    """Loader for metabolomic data (LC-MS, targeted panels)"""

    def __init__(
        self,
        source: DatasetSource,
        validate_on_load: bool = True,
        generate_qc_report: bool = True,
        max_missing_rate: float = 0.5,
        cv_threshold: float = 0.3,
    ):
        """
        Initialize metabolomic data loader

        Args:
            source: Dataset source
            validate_on_load: Whether to validate on load
            generate_qc_report: Whether to generate QC report
            max_missing_rate: Maximum allowed missing rate per feature
            cv_threshold: Maximum coefficient of variation for QC samples
        """
        super().__init__(
            source=source,
            modality="metabolomic",
            validate_on_load=validate_on_load,
            generate_qc_report=generate_qc_report,
        )
        self.max_missing_rate = max_missing_rate
        self.cv_threshold = cv_threshold

    def load(
        self,
        file_path: Union[str, Path],
        sample_metadata_path: Optional[Union[str, Path]] = None,
        feature_metadata_path: Optional[Union[str, Path]] = None,
        batch_id: Optional[str] = None,
        fasting_status: Optional[str] = None,
        collection_time: Optional[datetime] = None,
        storage_conditions: Optional[str] = None,
        **kwargs,
    ) -> LoadedData:
        """
        Load metabolomic data from file

        Args:
            file_path: Path to metabolomic data file
            sample_metadata_path: Optional path to sample metadata
            feature_metadata_path: Optional path to feature/metabolite metadata
            batch_id: Optional batch identifier
            fasting_status: Fasting status during collection
            collection_time: Sample collection time
            storage_conditions: Storage conditions (e.g., "-80C")
            **kwargs: Additional arguments

        Returns:
            LoadedData object
        """
        file_path = Path(file_path)
        data = self._parse_file(file_path)

        # Load metadata if provided
        sample_metadata = None
        if sample_metadata_path:
            sample_metadata = pd.read_csv(sample_metadata_path, index_col=0)
            # Extract context variables from metadata
            if "fasting_status" in sample_metadata.columns and fasting_status is None:
                fasting_status = sample_metadata["fasting_status"].mode().iloc[0]

        feature_metadata = None
        if feature_metadata_path:
            feature_metadata = pd.read_csv(feature_metadata_path, index_col=0)
        else:
            feature_metadata = self._extract_feature_metadata(data)

        # Generate QC metrics
        qc_metrics = None
        if self.generate_qc_report:
            qc_metrics = self._generate_qc_metrics(data, sample_metadata)
            qc_metrics = self._apply_metabolomic_qc(data, qc_metrics)

        # Classify missing data
        missing_type = self._classify_missing_data(data, sample_metadata)

        # Extract context variables
        context_vars = self._extract_context_variables(sample_metadata)
        if fasting_status:
            context_vars["fasting_status"] = fasting_status

        # Create metadata
        metadata = DataMetadata(
            source=self.source,
            modality=self.modality,
            file_path=file_path,
            batch_id=batch_id,
            collection_date=collection_time,
            storage_conditions=storage_conditions,
            context_variables=context_vars,
            qc_metrics=qc_metrics,
            missing_data_type=missing_type,
        )

        # Create loaded data object
        loaded_data = LoadedData(
            data=data,
            metadata=metadata,
            sample_metadata=sample_metadata,
            feature_metadata=feature_metadata,
        )

        # Validate if requested
        if self.validate_on_load:
            validation_errors = loaded_data.validate()
            if validation_errors and qc_metrics:
                qc_metrics.failed_checks.extend(validation_errors)

        return loaded_data

    def _parse_file(self, file_path: Path) -> pd.DataFrame:
        """
        Parse metabolomic data file

        Args:
            file_path: Path to file

        Returns:
            DataFrame with samples as rows, metabolites as columns
        """
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            data = pd.read_csv(file_path, index_col=0)
        elif suffix == ".tsv" or suffix == ".txt":
            data = pd.read_csv(file_path, sep="\t", index_col=0)
        elif suffix == ".h5" or suffix == ".hdf5":
            data = pd.read_hdf(file_path, key="metabolites")
        elif suffix == ".xlsx":
            data = pd.read_excel(file_path, index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Ensure numeric data
        data = data.select_dtypes(include=[np.number])

        return data

    def _apply_metabolomic_qc(self, data: pd.DataFrame, qc_metrics) -> None:
        """
        Apply metabolomic-specific QC checks

        Args:
            data: Metabolomic data DataFrame
            qc_metrics: QCMetrics object to update

        Returns:
            Updated QCMetrics
        """
        # Check feature missing rates
        feature_missing = data.isnull().mean(axis=0)
        high_missing_features = (feature_missing > self.max_missing_rate).sum()

        if high_missing_features > 0:
            qc_metrics.warnings.append(
                f"{high_missing_features} features with missing rate > "
                f"{self.max_missing_rate}"
            )

        # Check for zero variance features
        zero_var_features = (data.std(axis=0) == 0).sum()
        if zero_var_features > 0:
            qc_metrics.warnings.append(
                f"{zero_var_features} features with zero variance"
            )

        # Check for negative values (suspicious for metabolomics)
        negative_values = (data < 0).sum().sum()
        if negative_values > 0:
            qc_metrics.warnings.append(f"Found {negative_values} negative values")

        # Check coefficient of variation (if we have QC samples)
        # Placeholder: assumes QC samples are labeled in index
        qc_samples = [idx for idx in data.index if "QC" in str(idx).upper()]
        if qc_samples:
            qc_data = data.loc[qc_samples]
            cv = qc_data.std() / qc_data.mean()
            high_cv_features = (cv > self.cv_threshold).sum()
            if high_cv_features > 0:
                qc_metrics.warnings.append(
                    f"{high_cv_features} features with CV > {self.cv_threshold} "
                    f"in QC samples"
                )

        return qc_metrics

    def _extract_feature_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract feature metadata from column names

        Args:
            data: Metabolomic data DataFrame

        Returns:
            DataFrame with feature metadata
        """
        feature_meta = pd.DataFrame(index=data.columns)

        # Try to parse metabolite classes from column names
        # Common patterns: "AA_", "AC_", "SM_" for amino acids, acylcarnitines, sphingomyelins
        feature_meta["metabolite_class"] = "unknown"

        for idx in feature_meta.index:
            name = str(idx).upper()
            if name.startswith("AA_"):
                feature_meta.loc[idx, "metabolite_class"] = "amino_acid"
            elif name.startswith("AC_") or name.startswith("C"):
                feature_meta.loc[idx, "metabolite_class"] = "acylcarnitine"
            elif name.startswith("SM_"):
                feature_meta.loc[idx, "metabolite_class"] = "sphingomyelin"
            elif name.startswith("PC_"):
                feature_meta.loc[idx, "metabolite_class"] = "phosphatidylcholine"
            elif any(
                nt in name
                for nt in [
                    "SEROTONIN",
                    "DOPAMINE",
                    "GABA",
                    "GLUTAMATE",
                    "TRYPTOPHAN",
                ]
            ):
                feature_meta.loc[idx, "metabolite_class"] = "neurotransmitter"

        return feature_meta

    def _classify_missing_data(
        self, data: pd.DataFrame, sample_metadata: Optional[pd.DataFrame] = None
    ) -> MissingDataType:
        """
        Classify missing data mechanism for metabolomics

        Args:
            data: Metabolomic data DataFrame
            sample_metadata: Optional sample metadata

        Returns:
            MissingDataType classification
        """
        missing_mask = data.isnull()

        if missing_mask.sum().sum() == 0:
            return MissingDataType.MCAR

        # Check if missingness is related to low abundance (MNAR)
        # Features below detection limit are MNAR
        non_missing_mean = data.mean()
        missing_by_feature = missing_mask.sum(axis=0)

        # If features with more missing have lower mean values, likely MNAR
        if len(non_missing_mean) > 0:
            corr = np.corrcoef(non_missing_mean.fillna(0), missing_by_feature)[0, 1]
            if corr < -0.3:  # Negative correlation: low values more likely missing
                return MissingDataType.MNAR

        # Check if missingness correlates with metadata (MAR)
        if sample_metadata is not None:
            for col in sample_metadata.select_dtypes(include=[np.number]).columns:
                missing_counts = missing_mask.sum(axis=1)
                corr = np.corrcoef(sample_metadata[col].fillna(0), missing_counts)[
                    0, 1
                ]
                if abs(corr) > 0.3:
                    return MissingDataType.MAR

        return MissingDataType.UNKNOWN