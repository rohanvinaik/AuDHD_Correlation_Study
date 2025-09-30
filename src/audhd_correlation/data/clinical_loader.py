"""Clinical data loader with CSV/TSV support"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

from .base import (
    BaseDataLoader,
    DataMetadata,
    DatasetSource,
    LoadedData,
    MissingDataType,
)


class ClinicalLoader(BaseDataLoader):
    """Loader for clinical phenotype data"""

    def __init__(
        self,
        source: DatasetSource,
        validate_on_load: bool = True,
        generate_qc_report: bool = True,
        required_columns: Optional[List[str]] = None,
    ):
        """
        Initialize clinical data loader

        Args:
            source: Dataset source
            validate_on_load: Whether to validate on load
            generate_qc_report: Whether to generate QC report
            required_columns: List of required column names
        """
        super().__init__(
            source=source,
            modality="clinical",
            validate_on_load=validate_on_load,
            generate_qc_report=generate_qc_report,
        )
        self.required_columns = required_columns or []

    def load(
        self,
        file_path: Union[str, Path],
        sample_metadata_path: Optional[Union[str, Path]] = None,
        batch_id: Optional[str] = None,
        **kwargs,
    ) -> LoadedData:
        """
        Load clinical data from file

        Args:
            file_path: Path to clinical data file
            sample_metadata_path: Optional path to sample metadata
            batch_id: Optional batch identifier
            **kwargs: Additional arguments

        Returns:
            LoadedData object
        """
        file_path = Path(file_path)
        data = self._parse_file(file_path)

        # Load sample metadata if provided
        sample_metadata = None
        if sample_metadata_path:
            sample_metadata = pd.read_csv(sample_metadata_path, index_col=0)

        # Generate QC metrics
        qc_metrics = None
        if self.generate_qc_report:
            qc_metrics = self._generate_qc_metrics(data, sample_metadata)
            qc_metrics = self._apply_clinical_qc(data, qc_metrics)

        # Classify missing data
        missing_type = self._classify_missing_data(data, sample_metadata)

        # Extract context variables
        context_vars = self._extract_context_variables(sample_metadata)

        # Create metadata
        metadata = DataMetadata(
            source=self.source,
            modality=self.modality,
            file_path=file_path,
            batch_id=batch_id,
            context_variables=context_vars,
            qc_metrics=qc_metrics,
            missing_data_type=missing_type,
        )

        # Create loaded data object
        loaded_data = LoadedData(
            data=data,
            metadata=metadata,
            sample_metadata=sample_metadata,
            feature_metadata=self._extract_feature_metadata(data),
        )

        # Validate if requested
        if self.validate_on_load:
            validation_errors = loaded_data.validate()
            if validation_errors and qc_metrics:
                qc_metrics.failed_checks.extend(validation_errors)

        return loaded_data

    def _parse_file(self, file_path: Path) -> pd.DataFrame:
        """
        Parse clinical data file

        Args:
            file_path: Path to file

        Returns:
            DataFrame with samples as rows, clinical features as columns
        """
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            data = pd.read_csv(file_path, index_col=0)
        elif suffix == ".tsv" or suffix == ".txt":
            data = pd.read_csv(file_path, sep="\t", index_col=0)
        elif suffix == ".xlsx":
            data = pd.read_excel(file_path, index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return data

    def _apply_clinical_qc(self, data: pd.DataFrame, qc_metrics) -> None:
        """
        Apply clinical-specific QC checks

        Args:
            data: Clinical data DataFrame
            qc_metrics: QCMetrics object to update

        Returns:
            Updated QCMetrics
        """
        # Check for required columns
        missing_required = [col for col in self.required_columns if col not in data.columns]
        if missing_required:
            qc_metrics.failed_checks.append(
                f"Missing required columns: {', '.join(missing_required)}"
            )

        # Check for out-of-range values
        range_checks = self._get_range_checks()
        for col, (min_val, max_val) in range_checks.items():
            if col in data.columns:
                out_of_range = (
                    (data[col] < min_val) | (data[col] > max_val)
                ).sum()
                if out_of_range > 0:
                    qc_metrics.warnings.append(
                        f"{out_of_range} out-of-range values in {col}"
                    )

        # Check for consistency (e.g., diagnosis vs symptom scores)
        consistency_errors = self._check_consistency(data)
        if consistency_errors:
            qc_metrics.warnings.extend(consistency_errors)

        # Check age distribution
        if "age" in data.columns:
            age_dist = data["age"].describe()
            if age_dist["min"] < 0:
                qc_metrics.failed_checks.append("Negative age values found")
            if age_dist["max"] > 120:
                qc_metrics.warnings.append("Suspicious age values (>120)")

        return qc_metrics

    def _get_range_checks(self) -> Dict[str, tuple]:
        """
        Get expected ranges for clinical variables

        Returns:
            Dictionary mapping column names to (min, max) tuples
        """
        return {
            "age": (0, 120),
            "BMI": (10, 60),
            "ADHD_RS_total": (0, 72),
            "ADHD_RS_inattention": (0, 36),
            "ADHD_RS_hyperactive": (0, 36),
            "SCQ_total": (0, 39),
            "SRS_total": (0, 195),
            "CBCL_internalizing": (0, 64),
            "CBCL_externalizing": (0, 70),
            "IQ": (40, 160),
        }

    def _check_consistency(self, data: pd.DataFrame) -> List[str]:
        """
        Check for logical consistency in clinical data

        Args:
            data: Clinical data DataFrame

        Returns:
            List of consistency error messages
        """
        errors = []

        # Check ADHD diagnosis vs scores
        if "ADHD_diagnosis" in data.columns and "ADHD_RS_total" in data.columns:
            diagnosed = data["ADHD_diagnosis"] == 1
            low_scores = data["ADHD_RS_total"] < 20

            inconsistent = (diagnosed & low_scores).sum()
            if inconsistent > 0:
                errors.append(
                    f"{inconsistent} samples with ADHD diagnosis but low symptom scores"
                )

        # Check autism diagnosis vs scores
        if "ASD_diagnosis" in data.columns and "SCQ_total" in data.columns:
            diagnosed = data["ASD_diagnosis"] == 1
            low_scores = data["SCQ_total"] < 11

            inconsistent = (diagnosed & low_scores).sum()
            if inconsistent > 0:
                errors.append(
                    f"{inconsistent} samples with ASD diagnosis but low SCQ scores"
                )

        # Check medication consistency
        if "medication" in data.columns and "medication_type" in data.columns:
            no_med = data["medication"] == 0
            has_type = data["medication_type"].notna()

            inconsistent = (no_med & has_type).sum()
            if inconsistent > 0:
                errors.append(
                    f"{inconsistent} samples with no medication but medication_type specified"
                )

        return errors

    def _extract_feature_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract feature metadata for clinical variables

        Args:
            data: Clinical data DataFrame

        Returns:
            DataFrame with feature metadata
        """
        feature_meta = pd.DataFrame(index=data.columns)

        # Classify features into categories
        feature_meta["category"] = "other"
        feature_meta["data_type"] = data.dtypes.astype(str)

        for col in feature_meta.index:
            col_upper = str(col).upper()

            if "ADHD" in col_upper:
                feature_meta.loc[col, "category"] = "ADHD_phenotype"
            elif any(x in col_upper for x in ["ASD", "AUTISM", "SCQ", "SRS", "ADOS"]):
                feature_meta.loc[col, "category"] = "autism_phenotype"
            elif any(x in col_upper for x in ["ANXIETY", "DEPRESSION", "MOOD"]):
                feature_meta.loc[col, "category"] = "comorbidity"
            elif any(x in col_upper for x in ["IQ", "COGNITIVE", "WISC"]):
                feature_meta.loc[col, "category"] = "cognitive"
            elif any(x in col_upper for x in ["AGE", "SEX", "GENDER", "RACE"]):
                feature_meta.loc[col, "category"] = "demographic"
            elif "MED" in col_upper or "DRUG" in col_upper:
                feature_meta.loc[col, "category"] = "medication"
            elif "CBCL" in col_upper:
                feature_meta.loc[col, "category"] = "behavioral"

        return feature_meta

    def _classify_missing_data(
        self, data: pd.DataFrame, sample_metadata: Optional[pd.DataFrame] = None
    ) -> MissingDataType:
        """
        Classify missing data mechanism for clinical data

        Args:
            data: Clinical data DataFrame
            sample_metadata: Optional sample metadata

        Returns:
            MissingDataType classification
        """
        missing_mask = data.isnull()

        if missing_mask.sum().sum() == 0:
            return MissingDataType.MCAR

        # Check if missingness is systematic (MNAR)
        # E.g., certain questionnaires not administered based on diagnosis
        if "ADHD_diagnosis" in data.columns:
            adhd_mask = data["ADHD_diagnosis"] == 1
            non_adhd_mask = data["ADHD_diagnosis"] == 0

            adhd_cols = [col for col in data.columns if "ADHD_RS" in str(col)]
            if adhd_cols:
                # Check if non-ADHD cases have more missing ADHD scales
                missing_in_non_adhd = missing_mask.loc[non_adhd_mask, adhd_cols].mean()
                missing_in_adhd = missing_mask.loc[adhd_mask, adhd_cols].mean()

                if (missing_in_non_adhd - missing_in_adhd).mean() > 0.2:
                    return MissingDataType.MNAR

        # Check correlation with other variables (MAR)
        if sample_metadata is not None:
            for col in sample_metadata.select_dtypes(include=[np.number]).columns:
                missing_counts = missing_mask.sum(axis=1)
                corr = np.corrcoef(sample_metadata[col].fillna(0), missing_counts)[
                    0, 1
                ]
                if abs(corr) > 0.3:
                    return MissingDataType.MAR

        return MissingDataType.UNKNOWN