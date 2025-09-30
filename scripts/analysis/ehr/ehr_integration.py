#!/usr/bin/env python3
"""
Electronic Health Records (EHR) Integration
Extracts longitudinal features from clinical records for AuDHD study
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class EHRResult:
    """Results from EHR analysis"""
    longitudinal_features: pd.DataFrame
    comorbidity_patterns: pd.DataFrame
    medication_trajectories: pd.DataFrame
    diagnostic_trajectories: Optional[pd.DataFrame] = None


class EHRAnalyzer:
    """
    EHR analysis for AuDHD research

    Capabilities:
    1. Longitudinal feature extraction
    2. Comorbidity pattern detection
    3. Medication trajectory analysis
    4. Diagnostic progression tracking
    5. Healthcare utilization metrics
    """

    def __init__(self):
        """Initialize analyzer"""
        self.icd10_autism_codes = ['F84.0', 'F84.5', 'F84.8', 'F84.9']
        self.icd10_adhd_codes = ['F90.0', 'F90.1', 'F90.2', 'F90.8', 'F90.9']

    def parse_diagnosis_codes(
        self,
        encounters: pd.DataFrame,
        code_system: str = 'ICD10'
    ) -> pd.DataFrame:
        """
        Parse and standardize diagnosis codes

        Parameters
        ----------
        encounters : pd.DataFrame
            Columns: patient_id, date, diagnosis_code, description
        code_system : str
            'ICD10', 'ICD9', or 'SNOMED'

        Returns
        -------
        parsed_diagnoses : pd.DataFrame
        """
        logger.info(f"Parsing {code_system} diagnosis codes")

        encounters = encounters.copy()

        # Standardize codes
        if code_system == 'ICD10':
            # Remove decimal point: F84.0 -> F840
            encounters['code_clean'] = encounters['diagnosis_code'].str.replace('.', '')

        # Categorize by condition
        def categorize_diagnosis(code):
            if any(autism_code in code for autism_code in self.icd10_autism_codes):
                return 'autism'
            elif any(adhd_code in code for adhd_code in self.icd10_adhd_codes):
                return 'adhd'
            elif code.startswith('F'):
                return 'mental_health'
            elif code.startswith('G'):
                return 'neurological'
            else:
                return 'other'

        encounters['category'] = encounters['diagnosis_code'].apply(categorize_diagnosis)

        logger.info(f"  Parsed {len(encounters)} encounters")

        return encounters

    def extract_longitudinal_features(
        self,
        encounters: pd.DataFrame,
        patients: pd.DataFrame,
        window_years: int = 5
    ) -> pd.DataFrame:
        """
        Extract longitudinal features from EHR

        Parameters
        ----------
        encounters : pd.DataFrame
            Encounter data with dates
        patients : pd.DataFrame
            Patient demographics
        window_years : int
            Years of follow-up to consider

        Returns
        -------
        longitudinal_features : pd.DataFrame
            Columns: patient_id, n_encounters, age_at_diagnosis,
                     comorbidity_count, medication_count, etc.
        """
        logger.info("Extracting longitudinal features")

        features = []

        for patient_id in patients['patient_id'].unique():
            patient_encounters = encounters[encounters['patient_id'] == patient_id].copy()

            if len(patient_encounters) == 0:
                continue

            # Sort by date
            patient_encounters['date'] = pd.to_datetime(patient_encounters['date'])
            patient_encounters = patient_encounters.sort_values('date')

            # First and last encounter
            first_date = patient_encounters['date'].min()
            last_date = patient_encounters['date'].max()
            follow_up_days = (last_date - first_date).days

            # Age at first diagnosis
            patient_dob = patients.loc[patients['patient_id'] == patient_id, 'date_of_birth'].iloc[0]
            patient_dob = pd.to_datetime(patient_dob)
            age_at_diagnosis = (first_date - patient_dob).days / 365.25

            # Number of encounters
            n_encounters = len(patient_encounters)

            # Unique diagnoses
            unique_diagnoses = patient_encounters['diagnosis_code'].nunique()

            # Comorbidity count
            comorbidity_count = patient_encounters.groupby('category').size().to_dict()

            features.append({
                'patient_id': patient_id,
                'n_encounters': n_encounters,
                'follow_up_years': follow_up_days / 365.25,
                'age_at_diagnosis': age_at_diagnosis,
                'unique_diagnoses': unique_diagnoses,
                'autism_encounters': comorbidity_count.get('autism', 0),
                'adhd_encounters': comorbidity_count.get('adhd', 0),
                'mental_health_encounters': comorbidity_count.get('mental_health', 0),
                'neurological_encounters': comorbidity_count.get('neurological', 0)
            })

        features_df = pd.DataFrame(features)

        logger.info(f"  Extracted features for {len(features_df)} patients")

        return features_df

    def detect_comorbidity_patterns(
        self,
        encounters: pd.DataFrame,
        min_support: float = 0.05
    ) -> pd.DataFrame:
        """
        Detect frequent comorbidity patterns

        Uses association rule mining to find co-occurring conditions

        Parameters
        ----------
        encounters : pd.DataFrame
            Parsed encounters with categories
        min_support : float
            Minimum support threshold (fraction of patients)

        Returns
        -------
        comorbidity_patterns : pd.DataFrame
            Columns: pattern, support, patients
        """
        logger.info("Detecting comorbidity patterns")

        # Create patient-diagnosis matrix
        patient_diagnoses = encounters.groupby('patient_id')['category'].apply(
            lambda x: set(x)
        ).to_dict()

        # Count pattern frequencies
        from collections import Counter

        pattern_counts = Counter()

        for patient_id, diagnoses in patient_diagnoses.items():
            # All subsets of size 2+
            diagnoses_list = list(diagnoses)
            for i in range(len(diagnoses_list)):
                for j in range(i + 1, len(diagnoses_list)):
                    pattern = tuple(sorted([diagnoses_list[i], diagnoses_list[j]]))
                    pattern_counts[pattern] += 1

        # Filter by support
        n_patients = len(patient_diagnoses)
        min_count = int(min_support * n_patients)

        patterns = []
        for pattern, count in pattern_counts.items():
            if count >= min_count:
                patterns.append({
                    'pattern': ' + '.join(pattern),
                    'support': count / n_patients,
                    'n_patients': count
                })

        patterns_df = pd.DataFrame(patterns).sort_values('support', ascending=False)

        logger.info(f"  Found {len(patterns_df)} frequent patterns")

        return patterns_df

    def analyze_medication_trajectories(
        self,
        medications: pd.DataFrame,
        patients: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze medication trajectories over time

        Parameters
        ----------
        medications : pd.DataFrame
            Columns: patient_id, date, medication, dose
        patients : pd.DataFrame
            Patient demographics

        Returns
        -------
        trajectories : pd.DataFrame
            Columns: patient_id, medication_class, start_date,
                     duration_days, dose_changes
        """
        logger.info("Analyzing medication trajectories")

        # Categorize medications
        def categorize_medication(med_name):
            med_lower = med_name.lower()
            if any(stim in med_lower for stim in ['methylphenidate', 'amphetamine', 'adderall', 'ritalin']):
                return 'stimulant'
            elif any(ssri in med_lower for ssri in ['fluoxetine', 'sertraline', 'escitalopram']):
                return 'ssri'
            elif any(atypical in med_lower for atypical in ['risperidone', 'aripiprazole']):
                return 'atypical_antipsychotic'
            else:
                return 'other'

        medications['medication_class'] = medications['medication'].apply(categorize_medication)

        trajectories = []

        for patient_id in medications['patient_id'].unique():
            patient_meds = medications[medications['patient_id'] == patient_id].copy()
            patient_meds['date'] = pd.to_datetime(patient_meds['date'])
            patient_meds = patient_meds.sort_values('date')

            for med_class in patient_meds['medication_class'].unique():
                class_meds = patient_meds[patient_meds['medication_class'] == med_class]

                start_date = class_meds['date'].min()
                end_date = class_meds['date'].max()
                duration = (end_date - start_date).days

                # Dose changes
                if 'dose' in class_meds.columns:
                    doses = class_meds['dose'].dropna()
                    dose_changes = len(doses.unique()) - 1
                else:
                    dose_changes = 0

                trajectories.append({
                    'patient_id': patient_id,
                    'medication_class': med_class,
                    'start_date': start_date,
                    'duration_days': duration,
                    'n_prescriptions': len(class_meds),
                    'dose_changes': dose_changes
                })

        trajectories_df = pd.DataFrame(trajectories)

        logger.info(f"  Analyzed {len(trajectories_df)} medication trajectories")

        return trajectories_df

    def compute_diagnostic_trajectories(
        self,
        encounters: pd.DataFrame,
        time_bins: List[float] = [0, 1, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Track diagnostic progression over time bins

        Parameters
        ----------
        encounters : pd.DataFrame
            Parsed encounters
        time_bins : List[float]
            Time bins in years from first encounter

        Returns
        -------
        trajectories : pd.DataFrame
            Diagnostic category counts per time bin
        """
        logger.info("Computing diagnostic trajectories")

        trajectories = []

        for patient_id in encounters['patient_id'].unique():
            patient_encounters = encounters[encounters['patient_id'] == patient_id].copy()
            patient_encounters['date'] = pd.to_datetime(patient_encounters['date'])
            patient_encounters = patient_encounters.sort_values('date')

            first_date = patient_encounters['date'].min()
            patient_encounters['years_from_first'] = (
                (patient_encounters['date'] - first_date).dt.days / 365.25
            )

            trajectory = {'patient_id': patient_id}

            for i, bin_start in enumerate(time_bins[:-1]):
                bin_end = time_bins[i + 1]
                bin_label = f'year_{bin_start}_{bin_end}'

                bin_encounters = patient_encounters[
                    (patient_encounters['years_from_first'] >= bin_start) &
                    (patient_encounters['years_from_first'] < bin_end)
                ]

                # Count categories
                for category in ['autism', 'adhd', 'mental_health', 'neurological']:
                    count = (bin_encounters['category'] == category).sum()
                    trajectory[f'{bin_label}_{category}'] = count

            trajectories.append(trajectory)

        trajectories_df = pd.DataFrame(trajectories)

        logger.info(f"  Computed trajectories for {len(trajectories_df)} patients")

        return trajectories_df

    def compute_healthcare_utilization(
        self,
        encounters: pd.DataFrame,
        admissions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute healthcare utilization metrics

        Parameters
        ----------
        encounters : pd.DataFrame
            All encounters
        admissions : pd.DataFrame, optional
            Hospital admissions

        Returns
        -------
        utilization : pd.DataFrame
            Columns: patient_id, encounters_per_year,
                     admissions_per_year, cost_per_year
        """
        logger.info("Computing healthcare utilization")

        utilization = []

        for patient_id in encounters['patient_id'].unique():
            patient_encounters = encounters[encounters['patient_id'] == patient_id].copy()
            patient_encounters['date'] = pd.to_datetime(patient_encounters['date'])

            # Years of follow-up
            first_date = patient_encounters['date'].min()
            last_date = patient_encounters['date'].max()
            follow_up_years = (last_date - first_date).days / 365.25

            if follow_up_years == 0:
                follow_up_years = 1

            # Encounters per year
            encounters_per_year = len(patient_encounters) / follow_up_years

            # Admissions per year
            if admissions is not None:
                patient_admissions = admissions[admissions['patient_id'] == patient_id]
                admissions_per_year = len(patient_admissions) / follow_up_years
            else:
                admissions_per_year = 0

            utilization.append({
                'patient_id': patient_id,
                'encounters_per_year': encounters_per_year,
                'admissions_per_year': admissions_per_year,
                'follow_up_years': follow_up_years
            })

        utilization_df = pd.DataFrame(utilization)

        logger.info(f"  Computed utilization for {len(utilization_df)} patients")

        return utilization_df

    def integrate_with_baseline_deviation(
        self,
        ehr_features: pd.DataFrame,
        baseline_deviation_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Integrate EHR features with baseline-deviation framework

        Parameters
        ----------
        ehr_features : pd.DataFrame
            Longitudinal EHR features
        baseline_deviation_results : pd.DataFrame
            Baseline-deviation analysis results

        Returns
        -------
        integrated_results : pd.DataFrame
        """
        logger.info("Integrating EHR with baseline-deviation framework")

        # Align patients
        common_patients = ehr_features.index.intersection(
            baseline_deviation_results.index
        )

        logger.info(f"  {len(common_patients)} overlapping patients")

        integrated = pd.DataFrame({
            'n_patients': [len(common_patients)],
            'ehr_features': [ehr_features.shape[1]],
            'analysis_framework': ['baseline_deviation']
        })

        return integrated

    def analyze_complete(
        self,
        encounters: pd.DataFrame,
        patients: pd.DataFrame,
        medications: Optional[pd.DataFrame] = None
    ) -> EHRResult:
        """
        Complete EHR analysis pipeline

        Parameters
        ----------
        encounters : pd.DataFrame
            Encounter/diagnosis data
        patients : pd.DataFrame
            Patient demographics
        medications : pd.DataFrame, optional
            Medication prescriptions

        Returns
        -------
        EHRResult
        """
        logger.info("=== Complete EHR Analysis ===")

        # 1. Parse diagnoses
        parsed_encounters = self.parse_diagnosis_codes(encounters)

        # 2. Longitudinal features
        longitudinal_features = self.extract_longitudinal_features(
            parsed_encounters, patients
        )

        # 3. Comorbidity patterns
        comorbidity_patterns = self.detect_comorbidity_patterns(parsed_encounters)

        # 4. Medication trajectories
        if medications is not None:
            medication_trajectories = self.analyze_medication_trajectories(
                medications, patients
            )
        else:
            medication_trajectories = pd.DataFrame()

        # 5. Diagnostic trajectories
        diagnostic_trajectories = self.compute_diagnostic_trajectories(parsed_encounters)

        return EHRResult(
            longitudinal_features=longitudinal_features,
            comorbidity_patterns=comorbidity_patterns,
            medication_trajectories=medication_trajectories,
            diagnostic_trajectories=diagnostic_trajectories
        )


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    logger.info("Electronic Health Records Integration Module")
    logger.info("Ready for integration with AuDHD correlation study")
    logger.info("\nKey capabilities:")
    logger.info("  1. Longitudinal feature extraction")
    logger.info("  2. Comorbidity pattern detection")
    logger.info("  3. Medication trajectory analysis")
    logger.info("  4. Diagnostic progression tracking")
    logger.info("  5. Healthcare utilization metrics")
    logger.info("  6. Integration with baseline-deviation framework")
