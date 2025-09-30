"""Microbiome data loader with BIOM/CSV support"""
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np

from .base import (
    BaseDataLoader,
    DataMetadata,
    DatasetSource,
    LoadedData,
    MissingDataType,
)


class MicrobiomeLoader(BaseDataLoader):
    """Loader for microbiome data (16S rRNA, metagenomic)"""

    def __init__(
        self,
        source: DatasetSource,
        validate_on_load: bool = True,
        generate_qc_report: bool = True,
        min_read_depth: int = 1000,
        min_prevalence: float = 0.001,
    ):
        """
        Initialize microbiome data loader

        Args:
            source: Dataset source
            validate_on_load: Whether to validate on load
            generate_qc_report: Whether to generate QC report
            min_read_depth: Minimum total reads per sample
            min_prevalence: Minimum prevalence for taxa to keep
        """
        super().__init__(
            source=source,
            modality="microbiome",
            validate_on_load=validate_on_load,
            generate_qc_report=generate_qc_report,
        )
        self.min_read_depth = min_read_depth
        self.min_prevalence = min_prevalence

    def load(
        self,
        file_path: Union[str, Path],
        sample_metadata_path: Optional[Union[str, Path]] = None,
        taxonomy_path: Optional[Union[str, Path]] = None,
        batch_id: Optional[str] = None,
        **kwargs,
    ) -> LoadedData:
        """
        Load microbiome data from file

        Args:
            file_path: Path to microbiome data file (BIOM, CSV)
            sample_metadata_path: Optional path to sample metadata
            taxonomy_path: Optional path to taxonomy mapping
            batch_id: Optional batch identifier
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

        feature_metadata = None
        if taxonomy_path:
            feature_metadata = pd.read_csv(taxonomy_path, index_col=0)
        else:
            feature_metadata = self._extract_taxonomy_metadata(data)

        # Generate QC metrics
        qc_metrics = None
        if self.generate_qc_report:
            qc_metrics = self._generate_qc_metrics(data, sample_metadata)
            qc_metrics = self._apply_microbiome_qc(data, qc_metrics)

        # Classify missing data (zeros in microbiome data)
        missing_type = MissingDataType.MNAR  # Absent taxa are MNAR by nature

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
        Parse microbiome data file

        Args:
            file_path: Path to file

        Returns:
            DataFrame with samples as rows, taxa as columns
        """
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            data = pd.read_csv(file_path, index_col=0)
        elif suffix == ".tsv" or suffix == ".txt":
            data = pd.read_csv(file_path, sep="\t", index_col=0)
        elif suffix == ".biom":
            # Placeholder: In production, use biom-format package
            data = self._parse_biom(file_path)
        elif suffix == ".h5" or suffix == ".hdf5":
            data = pd.read_hdf(file_path, key="otu_table")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Ensure numeric data
        data = data.select_dtypes(include=[np.number])

        return data

    def _parse_biom(self, file_path: Path) -> pd.DataFrame:
        """
        Parse BIOM format file

        Args:
            file_path: Path to BIOM file

        Returns:
            DataFrame with OTU table
        """
        # Placeholder: In production, use biom-format package
        # import biom
        # table = biom.load_table(str(file_path))
        # return table.to_dataframe()

        # For now, return empty DataFrame
        return pd.DataFrame()

    def _apply_microbiome_qc(self, data: pd.DataFrame, qc_metrics) -> None:
        """
        Apply microbiome-specific QC checks

        Args:
            data: Microbiome data DataFrame
            qc_metrics: QCMetrics object to update

        Returns:
            Updated QCMetrics
        """
        # Check read depth per sample
        total_reads = data.sum(axis=1)
        low_depth_samples = (total_reads < self.min_read_depth).sum()

        if low_depth_samples > 0:
            qc_metrics.warnings.append(
                f"{low_depth_samples} samples with read depth < {self.min_read_depth}"
            )

        # Check taxa prevalence
        prevalence = (data > 0).mean(axis=0)
        rare_taxa = (prevalence < self.min_prevalence).sum()

        if rare_taxa > 0:
            qc_metrics.warnings.append(
                f"{rare_taxa} rare taxa with prevalence < {self.min_prevalence}"
            )

        # Check for negative counts (should never happen)
        negative_counts = (data < 0).sum().sum()
        if negative_counts > 0:
            qc_metrics.failed_checks.append(
                f"Found {negative_counts} negative counts"
            )

        # Check alpha diversity distribution
        alpha_diversity = (data > 0).sum(axis=1)  # Simple richness
        if alpha_diversity.std() > alpha_diversity.mean():
            qc_metrics.warnings.append("High variability in alpha diversity")

        # Check for contamination (high prevalence of rare taxa)
        high_prev_rare = ((prevalence > 0.9) & (data.mean() < 10)).sum()
        if high_prev_rare > 0:
            qc_metrics.warnings.append(
                f"{high_prev_rare} potential contaminant taxa "
                f"(high prevalence, low abundance)"
            )

        return qc_metrics

    def _extract_taxonomy_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract taxonomy metadata from OTU names

        Args:
            data: Microbiome data DataFrame

        Returns:
            DataFrame with taxonomy metadata
        """
        taxonomy_meta = pd.DataFrame(index=data.columns)

        # Parse taxonomy from column names if they follow Greengenes format
        # Example: "k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales"
        taxonomy_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

        for idx in data.columns:
            name = str(idx)

            # Try to parse semicolon-separated taxonomy
            if ";" in name:
                parts = name.split(";")
                for i, part in enumerate(parts[:len(taxonomy_levels)]):
                    if "__" in part:
                        taxonomy_meta.loc[idx, taxonomy_levels[i]] = part.split("__")[1]
                    else:
                        taxonomy_meta.loc[idx, taxonomy_levels[i]] = part

            # Classify by phylum (simplified)
            name_upper = name.upper()
            if "FIRMICUTES" in name_upper:
                taxonomy_meta.loc[idx, "phylum"] = "Firmicutes"
            elif "BACTEROIDETES" in name_upper or "BACTEROIDOTA" in name_upper:
                taxonomy_meta.loc[idx, "phylum"] = "Bacteroidetes"
            elif "ACTINOBACTERIA" in name_upper:
                taxonomy_meta.loc[idx, "phylum"] = "Actinobacteria"
            elif "PROTEOBACTERIA" in name_upper:
                taxonomy_meta.loc[idx, "phylum"] = "Proteobacteria"

        return taxonomy_meta


class NeuroimagingLoader(BaseDataLoader):
    """Loader for neuroimaging data (structural MRI, functional connectivity)"""

    def __init__(
        self,
        source: DatasetSource,
        validate_on_load: bool = True,
        generate_qc_report: bool = True,
    ):
        """Initialize neuroimaging data loader"""
        super().__init__(
            source=source,
            modality="neuroimaging",
            validate_on_load=validate_on_load,
            generate_qc_report=generate_qc_report,
        )

    def load(
        self,
        file_path: Union[str, Path],
        sample_metadata_path: Optional[Union[str, Path]] = None,
        roi_metadata_path: Optional[Union[str, Path]] = None,
        batch_id: Optional[str] = None,
        **kwargs,
    ) -> LoadedData:
        """
        Load neuroimaging data from file

        Args:
            file_path: Path to neuroimaging data file
            sample_metadata_path: Optional path to sample metadata
            roi_metadata_path: Optional path to ROI metadata
            batch_id: Optional batch identifier
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

        feature_metadata = None
        if roi_metadata_path:
            feature_metadata = pd.read_csv(roi_metadata_path, index_col=0)

        # Generate QC metrics
        qc_metrics = None
        if self.generate_qc_report:
            qc_metrics = self._generate_qc_metrics(data, sample_metadata)

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
        Parse neuroimaging data file

        Args:
            file_path: Path to file

        Returns:
            DataFrame with samples as rows, ROIs/features as columns
        """
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            data = pd.read_csv(file_path, index_col=0)
        elif suffix == ".tsv" or suffix == ".txt":
            data = pd.read_csv(file_path, sep="\t", index_col=0)
        elif suffix == ".h5" or suffix == ".hdf5":
            data = pd.read_hdf(file_path, key="neuroimaging")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Ensure numeric data
        data = data.select_dtypes(include=[np.number])

        return data