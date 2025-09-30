#!/usr/bin/env python3
"""
SPARK Phenotype Dictionary Parser

Parse SPARK phenotype data dictionary and create structured JSON output.
Identifies:
- Clinical instruments (ADOS, ADI-R, SRS, Vineland)
- Demographics
- Family relationships
- Metabolomics subsets (if available)
- Medical history
- Behavioral assessments

Usage:
    python parse_spark_phenotypes.py --input spark_data_dictionary.csv --output spark_phenotype_dictionary.json
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional, Set

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Install with: pip install pandas")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PhenotypeVariable:
    """Represents a phenotype variable"""
    name: str
    label: str
    description: str
    category: str
    subcategory: Optional[str] = None
    data_type: str = "string"
    units: Optional[str] = None
    valid_values: List = field(default_factory=list)
    missing_codes: List = field(default_factory=list)
    instrument: Optional[str] = None
    related_variables: List[str] = field(default_factory=list)
    notes: Optional[str] = None


class SPARKPhenotypeParser:
    """Parse SPARK phenotype data dictionary"""

    # Clinical instruments
    INSTRUMENTS = {
        "ADOS": r"ados",
        "ADI-R": r"adi[_-]?r",
        "SRS": r"srs",
        "Vineland": r"vineland|vabs",
        "CBCL": r"cbcl",
        "SCQ": r"scq",
        "RBS": r"rbs",
        "BRIEF": r"brief",
        "Sensory Profile": r"sensory|sp2",
    }

    # Categories
    CATEGORIES = {
        "demographics": ["age", "sex", "gender", "race", "ethnicity", "birth"],
        "diagnosis": ["diagnosis", "dx", "asd", "autism", "adhd"],
        "family": ["family", "sibling", "parent", "mother", "father", "relation"],
        "medical": ["medical", "health", "condition", "medication", "seizure", "gi"],
        "developmental": ["developmental", "milestone", "regression", "language"],
        "behavioral": ["behavior", "sleep", "anxiety", "aggression", "repetitive"],
        "cognitive": ["iq", "intelligence", "cognitive", "adaptive"],
        "metabolomics": ["metabol", "metabolite", "nmr", "biochem"],
    }

    def __init__(self):
        self.variables: List[PhenotypeVariable] = []
        self.categories: Dict[str, List[str]] = defaultdict(list)
        self.instruments: Dict[str, List[str]] = defaultdict(list)

    def parse_csv(self, csv_path: Path) -> int:
        """
        Parse SPARK data dictionary CSV

        Args:
            csv_path: Path to data dictionary CSV

        Returns:
            Number of variables parsed
        """
        logger.info(f"Parsing data dictionary: {csv_path}")

        try:
            df = pd.read_csv(csv_path)

            # Expected columns (may vary by SPARK release)
            expected_cols = ["variable_name", "label", "description"]
            if not all(col in df.columns for col in expected_cols):
                logger.warning(f"Missing expected columns: {expected_cols}")
                logger.info(f"Available columns: {list(df.columns)}")

            for _, row in df.iterrows():
                try:
                    var = self._parse_variable(row)
                    if var:
                        self.variables.append(var)
                        self.categories[var.category].append(var.name)
                        if var.instrument:
                            self.instruments[var.instrument].append(var.name)
                except Exception as e:
                    logger.warning(f"Error parsing variable: {e}")
                    continue

            logger.info(f"Parsed {len(self.variables)} variables")
            return len(self.variables)

        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            return 0

    def _parse_variable(self, row: pd.Series) -> Optional[PhenotypeVariable]:
        """Parse a single variable row"""
        # Extract basic info
        name = str(row.get("variable_name", row.get("name", "")))
        label = str(row.get("label", row.get("variable_label", "")))
        description = str(row.get("description", row.get("variable_description", "")))

        if not name or pd.isna(name):
            return None

        # Categorize variable
        category = self._categorize_variable(name, label, description)
        subcategory = self._get_subcategory(name, label, category)

        # Identify instrument
        instrument = self._identify_instrument(name, label)

        # Parse data type
        data_type = self._parse_data_type(row)

        # Parse valid values
        valid_values = self._parse_valid_values(row)

        # Parse missing codes
        missing_codes = self._parse_missing_codes(row)

        # Get units
        units = self._get_units(row)

        # Create variable
        var = PhenotypeVariable(
            name=name,
            label=label,
            description=description,
            category=category,
            subcategory=subcategory,
            data_type=data_type,
            units=units,
            valid_values=valid_values,
            missing_codes=missing_codes,
            instrument=instrument
        )

        return var

    def _categorize_variable(self, name: str, label: str, description: str) -> str:
        """Categorize variable based on name/label/description"""
        text = f"{name} {label} {description}".lower()

        for category, keywords in self.CATEGORIES.items():
            if any(kw in text for kw in keywords):
                return category

        return "other"

    def _get_subcategory(self, name: str, label: str, category: str) -> Optional[str]:
        """Get more specific subcategory"""
        text = f"{name} {label}".lower()

        if category == "diagnosis":
            if "asd" in text or "autism" in text:
                return "autism"
            elif "adhd" in text:
                return "adhd"

        elif category == "medical":
            if "gi" in text or "gastrointestinal" in text:
                return "gastrointestinal"
            elif "seizure" in text or "epilepsy" in text:
                return "seizure"
            elif "medication" in text:
                return "medication"

        elif category == "developmental":
            if "language" in text or "speech" in text:
                return "language"
            elif "motor" in text:
                return "motor"
            elif "regression" in text:
                return "regression"

        return None

    def _identify_instrument(self, name: str, label: str) -> Optional[str]:
        """Identify clinical instrument"""
        text = f"{name} {label}".lower()

        for instrument, pattern in self.INSTRUMENTS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return instrument

        return None

    def _parse_data_type(self, row: pd.Series) -> str:
        """Parse data type"""
        type_col = row.get("data_type", row.get("type", ""))

        if pd.isna(type_col):
            return "string"

        type_str = str(type_col).lower()

        if "int" in type_str or "numeric" in type_str:
            return "integer"
        elif "float" in type_str or "decimal" in type_str:
            return "float"
        elif "date" in type_str:
            return "date"
        elif "bool" in type_str or "binary" in type_str:
            return "boolean"
        else:
            return "string"

    def _parse_valid_values(self, row: pd.Series) -> List:
        """Parse valid value codes"""
        values_col = row.get("valid_values", row.get("value_labels", ""))

        if pd.isna(values_col):
            return []

        values_str = str(values_col)

        # Parse format like "1=Yes, 2=No, 3=Unknown"
        values = []
        for part in values_str.split(","):
            part = part.strip()
            if "=" in part:
                code, label = part.split("=", 1)
                values.append({"code": code.strip(), "label": label.strip()})

        return values

    def _parse_missing_codes(self, row: pd.Series) -> List:
        """Parse missing value codes"""
        missing_col = row.get("missing_codes", row.get("missing_values", ""))

        if pd.isna(missing_col):
            return []

        missing_str = str(missing_col)
        return [x.strip() for x in missing_str.split(",")]

    def _get_units(self, row: pd.Series) -> Optional[str]:
        """Get measurement units"""
        units_col = row.get("units", row.get("measurement_unit", ""))

        if pd.isna(units_col):
            return None

        return str(units_col)

    def identify_family_relationships(self) -> Dict[str, List[str]]:
        """Identify variables related to family relationships"""
        family_vars = {
            "family_id": [],
            "individual_id": [],
            "parent_id": [],
            "sibling": [],
            "relationship": [],
        }

        for var in self.variables:
            name_lower = var.name.lower()
            label_lower = var.label.lower()

            if "family" in name_lower and "id" in name_lower:
                family_vars["family_id"].append(var.name)
            elif "individual" in name_lower and "id" in name_lower:
                family_vars["individual_id"].append(var.name)
            elif "parent" in name_lower or "mother" in name_lower or "father" in name_lower:
                family_vars["parent_id"].append(var.name)
            elif "sibling" in label_lower:
                family_vars["sibling"].append(var.name)
            elif "relation" in name_lower:
                family_vars["relationship"].append(var.name)

        return family_vars

    def identify_metabolomics_subset(self) -> List[str]:
        """Identify participants with metabolomics data"""
        metabolomics_vars = []

        for var in self.variables:
            if var.category == "metabolomics":
                metabolomics_vars.append(var.name)
            elif "metabol" in var.name.lower() or "metabol" in var.label.lower():
                metabolomics_vars.append(var.name)

        return metabolomics_vars

    def get_clinical_instruments_summary(self) -> Dict[str, Dict]:
        """Get summary of clinical instruments"""
        summary = {}

        for instrument, variables in self.instruments.items():
            summary[instrument] = {
                "n_variables": len(variables),
                "variables": variables,
                "categories": list(set([
                    var.subcategory or var.category
                    for var in self.variables
                    if var.name in variables
                ]))
            }

        return summary

    def export_json(self, output_path: Path, include_full_details: bool = True):
        """
        Export parsed dictionary to JSON

        Args:
            output_path: Output JSON file path
            include_full_details: Include full variable details (vs summary only)
        """
        logger.info(f"Exporting to JSON: {output_path}")

        # Build export data
        export_data = {
            "metadata": {
                "source": "SPARK Phenotype Data Dictionary",
                "parsed_date": pd.Timestamp.now().isoformat(),
                "n_variables": len(self.variables),
                "n_categories": len(self.categories),
                "n_instruments": len(self.instruments),
            },
            "categories": {
                cat: vars_list
                for cat, vars_list in self.categories.items()
            },
            "instruments": self.get_clinical_instruments_summary(),
            "family_relationships": self.identify_family_relationships(),
            "metabolomics_subset": self.identify_metabolomics_subset(),
        }

        if include_full_details:
            export_data["variables"] = [
                asdict(var) for var in self.variables
            ]

        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"✓ Exported {len(self.variables)} variables to {output_path}")

    def export_summary_csv(self, output_path: Path):
        """Export summary CSV of key variables"""
        logger.info(f"Exporting summary CSV: {output_path}")

        # Create summary DataFrame
        summary_data = []
        for var in self.variables:
            summary_data.append({
                "variable_name": var.name,
                "label": var.label,
                "category": var.category,
                "subcategory": var.subcategory,
                "instrument": var.instrument,
                "data_type": var.data_type,
                "has_valid_values": len(var.valid_values) > 0,
            })

        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)

        logger.info(f"✓ Exported summary to {output_path}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Parse SPARK phenotype data dictionary",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input data dictionary CSV"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dictionaries/spark_phenotype_dictionary.json"),
        help="Output JSON file"
    )

    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional summary CSV output"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Include full variable details in JSON"
    )

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Parse dictionary
    parser_obj = SPARKPhenotypeParser()
    n_vars = parser_obj.parse_csv(args.input)

    if n_vars == 0:
        logger.error("No variables parsed")
        sys.exit(1)

    # Export JSON
    parser_obj.export_json(args.output, include_full_details=args.full)

    # Export summary if requested
    if args.summary:
        parser_obj.export_summary_csv(args.summary)

    # Print summary
    print("\n" + "="*50)
    print("Parsing Summary")
    print("="*50)
    print(f"Total variables: {n_vars}")
    print(f"\nCategories:")
    for cat, vars_list in sorted(parser_obj.categories.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(vars_list)}")

    print(f"\nClinical Instruments:")
    for instr, summary in parser_obj.get_clinical_instruments_summary().items():
        print(f"  {instr}: {summary['n_variables']} variables")

    metabolomics = parser_obj.identify_metabolomics_subset()
    if metabolomics:
        print(f"\nMetabolomics variables: {len(metabolomics)}")

    print("="*50)


if __name__ == "__main__":
    main()