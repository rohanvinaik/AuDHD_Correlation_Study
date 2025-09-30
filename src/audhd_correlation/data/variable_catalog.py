"""Unified variable catalog system"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import json

import pandas as pd

from .ontology_base import OntologyMatch, OntologyType, MatchConfidence


class VariableType(Enum):
    """Types of variables in the catalog"""

    CLINICAL = "clinical"
    MEDICATION = "medication"
    DIETARY = "dietary"
    GENETIC = "genetic"
    METABOLOMIC = "metabolomic"
    MICROBIOME = "microbiome"
    NEUROIMAGING = "neuroimaging"
    DEMOGRAPHIC = "demographic"
    ENVIRONMENTAL = "environmental"


class DataType(Enum):
    """Data types for variables"""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"
    COUNT = "count"
    TEXT = "text"


@dataclass
class VariableDefinition:
    """Definition of a variable in the catalog"""

    name: str
    variable_type: VariableType
    data_type: DataType
    description: str
    ontology_mappings: Dict[str, OntologyMatch] = field(default_factory=dict)
    synonyms: Set[str] = field(default_factory=set)
    units: Optional[str] = None
    valid_range: Optional[tuple] = None
    categories: Optional[List[str]] = None
    missing_codes: List[Any] = field(default_factory=lambda: [-999, -888, "NA", ""])
    source_datasets: Set[str] = field(default_factory=set)
    requires_review: bool = False
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "variable_type": self.variable_type.value,
            "data_type": self.data_type.value,
            "description": self.description,
            "ontology_mappings": {
                k: v.to_dict() for k, v in self.ontology_mappings.items()
            },
            "synonyms": list(self.synonyms),
            "units": self.units,
            "valid_range": self.valid_range,
            "categories": self.categories,
            "missing_codes": self.missing_codes,
            "source_datasets": list(self.source_datasets),
            "requires_review": self.requires_review,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class VariableCatalog:
    """Unified catalog of all variables across datasets"""

    def __init__(self, catalog_path: Optional[Path] = None):
        """
        Initialize variable catalog

        Args:
            catalog_path: Path to save/load catalog
        """
        self.catalog_path = catalog_path
        self.variables: Dict[str, VariableDefinition] = {}

        if catalog_path and catalog_path.exists():
            self.load(catalog_path)

    def add_variable(self, variable: VariableDefinition) -> None:
        """
        Add variable to catalog

        Args:
            variable: VariableDefinition to add
        """
        variable.updated_at = datetime.now()
        self.variables[variable.name] = variable

    def get_variable(self, name: str) -> Optional[VariableDefinition]:
        """
        Get variable by name

        Args:
            name: Variable name

        Returns:
            VariableDefinition if found
        """
        return self.variables.get(name)

    def search_variables(
        self,
        query: str = None,
        variable_type: Optional[VariableType] = None,
        has_ontology: Optional[OntologyType] = None,
        requires_review: Optional[bool] = None,
    ) -> List[VariableDefinition]:
        """
        Search variables in catalog

        Args:
            query: Text query (searches name, description, synonyms)
            variable_type: Filter by variable type
            has_ontology: Filter by presence of specific ontology mapping
            requires_review: Filter by review status

        Returns:
            List of matching variables
        """
        results = []

        for var in self.variables.values():
            # Apply filters
            if variable_type and var.variable_type != variable_type:
                continue

            if has_ontology and has_ontology.value not in var.ontology_mappings:
                continue

            if requires_review is not None and var.requires_review != requires_review:
                continue

            # Apply text query
            if query:
                query_lower = query.lower()
                if (
                    query_lower in var.name.lower()
                    or query_lower in var.description.lower()
                    or any(query_lower in syn.lower() for syn in var.synonyms)
                ):
                    results.append(var)
            else:
                results.append(var)

        return results

    def get_review_queue(self) -> List[VariableDefinition]:
        """
        Get variables that require manual review

        Returns:
            List of variables requiring review
        """
        return [var for var in self.variables.values() if var.requires_review]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get catalog statistics

        Returns:
            Dictionary with statistics
        """
        total = len(self.variables)

        if total == 0:
            return {"total_variables": 0}

        type_counts = {}
        for vtype in VariableType:
            type_counts[vtype.value] = sum(
                1 for v in self.variables.values() if v.variable_type == vtype
            )

        ontology_coverage = {}
        for onto in OntologyType:
            ontology_coverage[onto.value] = sum(
                1
                for v in self.variables.values()
                if onto.value in v.ontology_mappings
            )

        review_count = sum(1 for v in self.variables.values() if v.requires_review)

        return {
            "total_variables": total,
            "by_type": type_counts,
            "ontology_coverage": ontology_coverage,
            "requires_review": review_count,
            "review_rate": review_count / total if total > 0 else 0,
        }

    def merge_from_dataframe(
        self,
        df: pd.DataFrame,
        variable_type: VariableType,
        source_dataset: str,
        ontology_mappings: Optional[Dict[str, Dict[str, OntologyMatch]]] = None,
    ) -> int:
        """
        Merge variables from DataFrame columns

        Args:
            df: DataFrame with variables as columns
            variable_type: Type of variables
            source_dataset: Name of source dataset
            ontology_mappings: Optional pre-computed ontology mappings

        Returns:
            Number of variables added/updated
        """
        count = 0

        for col in df.columns:
            # Infer data type
            data_type = self._infer_data_type(df[col])

            # Check if variable exists
            existing = self.get_variable(col)

            if existing:
                # Update existing variable
                existing.source_datasets.add(source_dataset)
                existing.updated_at = datetime.now()

                # Add ontology mappings if provided
                if ontology_mappings and col in ontology_mappings:
                    for onto_name, match in ontology_mappings[col].items():
                        if match.confidence != MatchConfidence.UNMATCHED:
                            existing.ontology_mappings[onto_name] = match

                            # Flag for review if ambiguous
                            if match.requires_review:
                                existing.requires_review = True
            else:
                # Create new variable
                variable = VariableDefinition(
                    name=col,
                    variable_type=variable_type,
                    data_type=data_type,
                    description=f"{variable_type.value} variable from {source_dataset}",
                    source_datasets={source_dataset},
                )

                # Add ontology mappings if provided
                if ontology_mappings and col in ontology_mappings:
                    for onto_name, match in ontology_mappings[col].items():
                        if match.confidence != MatchConfidence.UNMATCHED:
                            variable.ontology_mappings[onto_name] = match

                            if match.requires_review:
                                variable.requires_review = True

                # Infer additional properties
                if data_type == DataType.CONTINUOUS:
                    variable.valid_range = (df[col].min(), df[col].max())
                elif data_type == DataType.CATEGORICAL:
                    variable.categories = df[col].unique().tolist()

                self.add_variable(variable)

            count += 1

        return count

    def _infer_data_type(self, series: pd.Series) -> DataType:
        """
        Infer data type from pandas Series

        Args:
            series: Pandas Series

        Returns:
            DataType enum value
        """
        # Check for binary
        unique_vals = series.dropna().unique()
        if len(unique_vals) == 2:
            return DataType.BINARY

        # Check numeric types
        if pd.api.types.is_numeric_dtype(series):
            # Check if all values are integers
            if pd.api.types.is_integer_dtype(series):
                # If small number of unique values, likely categorical
                if len(unique_vals) <= 10:
                    return DataType.ORDINAL
                else:
                    return DataType.COUNT
            else:
                return DataType.CONTINUOUS

        # Check for categorical
        if pd.api.types.is_categorical_dtype(series) or len(unique_vals) <= 20:
            return DataType.CATEGORICAL

        # Default to text
        return DataType.TEXT

    def export_to_csv(self, output_path: Path) -> None:
        """
        Export catalog to CSV

        Args:
            output_path: Output file path
        """
        rows = []

        for var in self.variables.values():
            row = {
                "name": var.name,
                "variable_type": var.variable_type.value,
                "data_type": var.data_type.value,
                "description": var.description,
                "synonyms": "; ".join(var.synonyms),
                "units": var.units,
                "valid_range": str(var.valid_range) if var.valid_range else "",
                "categories": "; ".join(var.categories) if var.categories else "",
                "source_datasets": "; ".join(var.source_datasets),
                "requires_review": var.requires_review,
                "notes": var.notes,
            }

            # Add ontology mappings as separate columns
            for onto_type in OntologyType:
                onto_name = onto_type.value
                if onto_name in var.ontology_mappings:
                    match = var.ontology_mappings[onto_name]
                    row[f"{onto_name}_id"] = match.ontology_id
                    row[f"{onto_name}_term"] = match.matched_term
                    row[f"{onto_name}_confidence"] = match.confidence.value
                else:
                    row[f"{onto_name}_id"] = ""
                    row[f"{onto_name}_term"] = ""
                    row[f"{onto_name}_confidence"] = ""

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def export_to_json(self, output_path: Path) -> None:
        """
        Export catalog to JSON

        Args:
            output_path: Output file path
        """
        data = {name: var.to_dict() for name, var in self.variables.items()}

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save(self, output_path: Optional[Path] = None) -> None:
        """
        Save catalog to file

        Args:
            output_path: Output file path (defaults to catalog_path)
        """
        if output_path is None:
            output_path = self.catalog_path

        if output_path is None:
            raise ValueError("No output path specified")

        self.export_to_json(output_path)

    def load(self, input_path: Path) -> None:
        """
        Load catalog from JSON file

        Args:
            input_path: Input file path
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        self.variables = {}

        for name, var_dict in data.items():
            # Reconstruct ontology mappings
            ontology_mappings = {}
            for onto_name, match_dict in var_dict.get("ontology_mappings", {}).items():
                ontology_mappings[onto_name] = OntologyMatch(
                    source_term=match_dict["source_term"],
                    matched_term=match_dict["matched_term"],
                    ontology_id=match_dict["ontology_id"],
                    ontology_type=OntologyType(match_dict["ontology_type"]),
                    confidence=MatchConfidence(match_dict["confidence"]),
                    similarity_score=match_dict["similarity_score"],
                    alternative_matches=match_dict.get("alternative_matches", []),
                    requires_review=match_dict.get("requires_review", False),
                    metadata=match_dict.get("metadata", {}),
                    matched_at=datetime.fromisoformat(match_dict["matched_at"]),
                )

            variable = VariableDefinition(
                name=var_dict["name"],
                variable_type=VariableType(var_dict["variable_type"]),
                data_type=DataType(var_dict["data_type"]),
                description=var_dict["description"],
                ontology_mappings=ontology_mappings,
                synonyms=set(var_dict.get("synonyms", [])),
                units=var_dict.get("units"),
                valid_range=tuple(var_dict["valid_range"])
                if var_dict.get("valid_range")
                else None,
                categories=var_dict.get("categories"),
                missing_codes=var_dict.get("missing_codes", []),
                source_datasets=set(var_dict.get("source_datasets", [])),
                requires_review=var_dict.get("requires_review", False),
                notes=var_dict.get("notes", ""),
                created_at=datetime.fromisoformat(var_dict["created_at"]),
                updated_at=datetime.fromisoformat(var_dict["updated_at"]),
            )

            self.variables[name] = variable