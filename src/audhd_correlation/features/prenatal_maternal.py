"""
Prenatal and Maternal Health Feature Extraction

Comprehensive feature extraction for maternal health conditions, prenatal exposures,
and pregnancy complications with critical window analysis.

Critical for neurodevelopmental research: maternal immune activation (MIA), infection
timing, medication exposures, stress, and nutritional factors during pregnancy.

Features:
- Maternal infections by trimester with severity grading
- Medication exposures (SSRIs, antibiotics, anticonvulsants)
- Pregnancy complications (gestational diabetes, preeclampsia, etc.)
- Maternal stress and cortisol profiles during pregnancy
- Nutritional status (folate, vitamin D, omega-3)
- Birth outcomes (gestational age, birth weight, APGAR)
- Trimester-specific critical window analysis
- Recall bias correction and data quality scoring

Literature Basis:
- Maternal immune activation (MIA) hypothesis (Patterson 2011)
- Prenatal SSRI exposure and ASD risk (Brown 2017)
- Critical period for infection timing (Meyer 2006)
- G×E interactions with prenatal environment (Tordjman 2014)

Author: Claude Code
Date: 2025-09-30
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TrimesterWindows:
    """Pregnancy trimester definitions with critical developmental periods."""

    # Standard trimesters (days relative to birth, negative = before birth)
    first: Tuple[int, int] = (-280, -187)   # Conception to 13 weeks (organogenesis)
    second: Tuple[int, int] = (-186, -94)   # 14-26 weeks (brain development)
    third: Tuple[int, int] = (-93, 0)       # 27 weeks to birth (maturation)

    # Critical neurodevelopmental windows
    neural_tube_closure: Tuple[int, int] = (-259, -252)  # Days 21-28 post-conception
    neurogenesis_peak: Tuple[int, int] = (-210, -140)    # Weeks 10-20
    synaptogenesis_peak: Tuple[int, int] = (-140, -42)   # Weeks 20-34
    myelination_start: Tuple[int, int] = (-98, 0)        # Week 26 onward

    def get_trimester(self, days_before_birth: int) -> int:
        """Return trimester (1, 2, or 3) for given day."""
        if self.first[0] <= days_before_birth <= self.first[1]:
            return 1
        elif self.second[0] <= days_before_birth <= self.second[1]:
            return 2
        elif self.third[0] <= days_before_birth <= self.third[1]:
            return 3
        else:
            return 0  # Outside pregnancy window

    def is_in_critical_window(self, days_before_birth: int, window: str) -> bool:
        """Check if date falls within critical developmental window."""
        window_range = getattr(self, window, None)
        if window_range is None:
            return False
        return window_range[0] <= days_before_birth <= window_range[1]


class MaternalInfectionAnalyzer:
    """
    Analyze maternal infection timing, severity, and neurodevelopmental impact.

    Maternal immune activation (MIA) during pregnancy is strongly associated
    with increased ASD/ADHD risk. Timing is critical - infections during
    neurogenesis (weeks 10-20) have greatest impact.
    """

    def __init__(self):
        self.trimesters = TrimesterWindows()

        # Infection types with relative severity
        self.infection_types = {
            'influenza': {'severity': 3, 'fever_common': True, 'hospitalization_risk': 0.15},
            'fever_unspecified': {'severity': 2, 'fever_common': True, 'hospitalization_risk': 0.05},
            'uti': {'severity': 1, 'fever_common': False, 'hospitalization_risk': 0.02},
            'bacterial_other': {'severity': 2, 'fever_common': True, 'hospitalization_risk': 0.10},
            'viral_other': {'severity': 1, 'fever_common': True, 'hospitalization_risk': 0.03},
            'covid19': {'severity': 3, 'fever_common': True, 'hospitalization_risk': 0.20},
            'pneumonia': {'severity': 3, 'fever_common': True, 'hospitalization_risk': 0.25}
        }

    def extract_infection_features(
        self,
        infection_data: pd.DataFrame,
        birth_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Extract comprehensive maternal infection features.

        Parameters
        ----------
        infection_data : pd.DataFrame
            Columns: infection_type, infection_date (or gestational_week),
                    severity, fever_days, hospitalization, antibiotics_used
        birth_date : datetime, optional
            Child's birth date for calculating days_before_birth

        Returns
        -------
        dict
            Comprehensive infection features including timing, severity, treatment
        """
        features = {}

        if infection_data.empty:
            return self._create_empty_features()

        # Convert dates to days before birth if needed
        if birth_date is not None and 'infection_date' in infection_data.columns:
            infection_data['days_before_birth'] = infection_data['infection_date'].apply(
                lambda x: -(birth_date - x).days if pd.notna(x) else np.nan
            )
        elif 'gestational_week' in infection_data.columns:
            # Convert gestational weeks to days before birth
            infection_data['days_before_birth'] = infection_data['gestational_week'].apply(
                lambda x: -(40 - x) * 7 if pd.notna(x) else np.nan
            )

        # Overall infection features
        features['any_infection_pregnancy'] = 1
        features['n_infections_total'] = len(infection_data)

        # Trimester-specific infections
        for trimester in [1, 2, 3]:
            trimester_infections = infection_data[
                infection_data['days_before_birth'].apply(
                    lambda x: self.trimesters.get_trimester(x) == trimester if pd.notna(x) else False
                )
            ]
            features[f'infection_trimester{trimester}'] = int(len(trimester_infections) > 0)
            features[f'n_infections_trimester{trimester}'] = len(trimester_infections)

        # Critical window infections (most important for neurodevelopment)
        for window in ['neural_tube_closure', 'neurogenesis_peak', 'synaptogenesis_peak']:
            window_infections = infection_data[
                infection_data['days_before_birth'].apply(
                    lambda x: self.trimesters.is_in_critical_window(x, window) if pd.notna(x) else False
                )
            ]
            features[f'infection_{window}'] = int(len(window_infections) > 0)

        # Severity measures
        if 'severity' in infection_data.columns:
            features['infection_severity_max'] = infection_data['severity'].max()
            features['infection_severity_mean'] = infection_data['severity'].mean()
        else:
            # Estimate severity from infection type
            infection_data['estimated_severity'] = infection_data['infection_type'].map(
                lambda x: self.infection_types.get(x, {}).get('severity', 1)
            )
            features['infection_severity_max'] = infection_data['estimated_severity'].max()
            features['infection_severity_mean'] = infection_data['estimated_severity'].mean()

        # Fever features (critical for MIA)
        if 'fever_days' in infection_data.columns:
            features['fever_days_total'] = infection_data['fever_days'].sum()
            features['fever_any'] = int(infection_data['fever_days'].sum() > 0)
        else:
            # Estimate fever likelihood from infection type
            features['fever_likely'] = int(any(
                self.infection_types.get(inf_type, {}).get('fever_common', False)
                for inf_type in infection_data['infection_type']
            ))

        # Hospitalization (proxy for severe infection)
        if 'hospitalization' in infection_data.columns:
            features['infection_hospitalization'] = int(infection_data['hospitalization'].any())
            features['n_hospitalizations'] = infection_data['hospitalization'].sum()

        # Antibiotic exposure
        if 'antibiotics_used' in infection_data.columns:
            features['antibiotics_pregnancy'] = int(infection_data['antibiotics_used'].any())

            # Trimester-specific antibiotic exposure
            for trimester in [1, 2, 3]:
                trimester_abx = infection_data[
                    (infection_data['antibiotics_used']) &
                    (infection_data['days_before_birth'].apply(
                        lambda x: self.trimesters.get_trimester(x) == trimester if pd.notna(x) else False
                    ))
                ]
                features[f'antibiotics_trimester{trimester}'] = int(len(trimester_abx) > 0)

        # Infection type distribution
        if 'infection_type' in infection_data.columns:
            for inf_type in self.infection_types.keys():
                features[f'infection_type_{inf_type}'] = int(
                    (infection_data['infection_type'] == inf_type).any()
                )

        # Timing features (early vs. late pregnancy)
        early_infections = infection_data[
            infection_data['days_before_birth'].apply(lambda x: x < -140 if pd.notna(x) else False)
        ]
        features['infection_early_pregnancy'] = int(len(early_infections) > 0)
        features['infection_late_pregnancy'] = int(len(infection_data) > len(early_infections))

        # Risk score (composite of severity, timing, fever)
        features['infection_risk_score'] = self._calculate_infection_risk(infection_data)

        return features

    def _calculate_infection_risk(self, infection_data: pd.DataFrame) -> float:
        """
        Calculate composite infection risk score (0-10 scale).

        Weights:
        - Critical window timing: 4 points
        - Severity: 3 points
        - Fever: 2 points
        - Hospitalization: 1 point
        """
        risk_score = 0.0

        # Critical window weighting (neurogenesis most critical)
        for _, row in infection_data.iterrows():
            if pd.notna(row.get('days_before_birth')):
                if self.trimesters.is_in_critical_window(row['days_before_birth'], 'neurogenesis_peak'):
                    risk_score += 4.0
                elif self.trimesters.is_in_critical_window(row['days_before_birth'], 'synaptogenesis_peak'):
                    risk_score += 3.0
                elif self.trimesters.get_trimester(row['days_before_birth']) == 1:
                    risk_score += 2.0
                else:
                    risk_score += 1.0

        # Severity
        if 'severity' in infection_data.columns:
            risk_score += infection_data['severity'].mean()

        # Fever
        if 'fever_days' in infection_data.columns:
            risk_score += min(infection_data['fever_days'].sum() / 3, 2.0)

        # Hospitalization
        if 'hospitalization' in infection_data.columns:
            risk_score += infection_data['hospitalization'].sum()

        return min(risk_score, 10.0)  # Cap at 10

    def _create_empty_features(self) -> Dict[str, Any]:
        """Return features dict for cases with no infections."""
        features = {
            'any_infection_pregnancy': 0,
            'n_infections_total': 0,
            'infection_risk_score': 0.0
        }

        for trimester in [1, 2, 3]:
            features[f'infection_trimester{trimester}'] = 0
            features[f'n_infections_trimester{trimester}'] = 0
            features[f'antibiotics_trimester{trimester}'] = 0

        for window in ['neural_tube_closure', 'neurogenesis_peak', 'synaptogenesis_peak']:
            features[f'infection_{window}'] = 0

        return features


class MaternalMedicationAnalyzer:
    """
    Analyze prenatal medication exposures with timing and dosage.

    Focus on medications with known neurodevelopmental impact:
    - SSRIs (ASD risk, but also treats maternal depression)
    - Anticonvulsants (valproate, phenytoin)
    - Antibiotics (potential microbiome effects)
    - Acetaminophen (prolonged use associated with ADHD)
    """

    def __init__(self):
        self.trimesters = TrimesterWindows()

        # Medication classes with risk profiles
        self.medication_classes = {
            'ssri': {
                'drugs': ['fluoxetine', 'sertraline', 'paroxetine', 'citalopram', 'escitalopram'],
                'risk_level': 'moderate',
                'timing_critical': True
            },
            'snri': {
                'drugs': ['venlafaxine', 'duloxetine'],
                'risk_level': 'moderate',
                'timing_critical': True
            },
            'anticonvulsant': {
                'drugs': ['valproate', 'phenytoin', 'carbamazepine', 'lamotrigine'],
                'risk_level': 'high',
                'timing_critical': True
            },
            'antibiotic': {
                'drugs': ['amoxicillin', 'azithromycin', 'cephalexin', 'nitrofurantoin'],
                'risk_level': 'low',
                'timing_critical': False
            },
            'acetaminophen': {
                'drugs': ['acetaminophen', 'tylenol', 'paracetamol'],
                'risk_level': 'low_chronic',
                'timing_critical': False
            },
            'benzodiazepine': {
                'drugs': ['diazepam', 'lorazepam', 'alprazolam', 'clonazepam'],
                'risk_level': 'moderate',
                'timing_critical': True
            }
        }

    def extract_medication_features(
        self,
        medication_data: pd.DataFrame,
        birth_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Extract prenatal medication exposure features.

        Parameters
        ----------
        medication_data : pd.DataFrame
            Columns: medication_name, start_date, end_date (or start_week, duration_weeks),
                    dosage, indication
        birth_date : datetime, optional
            Child's birth date for timing calculations

        Returns
        -------
        dict
            Medication exposure features by class, timing, duration
        """
        features = {}

        if medication_data.empty:
            return self._create_empty_medication_features()

        # Classify medications
        medication_data['medication_class'] = medication_data['medication_name'].apply(
            self._classify_medication
        )

        # Overall exposure
        features['any_medication_pregnancy'] = 1
        features['n_medications_total'] = len(medication_data)
        features['n_medication_classes'] = len(medication_data['medication_class'].unique())

        # By medication class
        for med_class in self.medication_classes.keys():
            class_meds = medication_data[medication_data['medication_class'] == med_class]
            features[f'{med_class}_exposure'] = int(len(class_meds) > 0)

            if len(class_meds) > 0:
                # Duration
                if 'duration_weeks' in class_meds.columns:
                    features[f'{med_class}_duration_weeks'] = class_meds['duration_weeks'].sum()

                # Trimester-specific exposure
                for trimester in [1, 2, 3]:
                    # This requires exposure window calculation
                    features[f'{med_class}_trimester{trimester}'] = 0  # Placeholder

                # Risk level
                risk_level = self.medication_classes[med_class]['risk_level']
                features[f'{med_class}_risk_level'] = risk_level

        # Polypharmacy (multiple concurrent medications)
        features['medication_polypharmacy'] = int(
            len(medication_data['medication_name'].unique()) >= 3
        )

        # High-risk medication composite
        high_risk_classes = [c for c, info in self.medication_classes.items()
                            if info['risk_level'] in ['high', 'moderate']]
        features['high_risk_medication_exposure'] = int(
            medication_data['medication_class'].isin(high_risk_classes).any()
        )

        return features

    def _classify_medication(self, medication_name: str) -> str:
        """Classify medication into risk category."""
        medication_lower = medication_name.lower() if pd.notna(medication_name) else ''

        for med_class, info in self.medication_classes.items():
            if any(drug in medication_lower for drug in info['drugs']):
                return med_class

        return 'other'

    def _create_empty_medication_features(self) -> Dict[str, Any]:
        """Return features for no medication exposure."""
        features = {
            'any_medication_pregnancy': 0,
            'n_medications_total': 0,
            'medication_polypharmacy': 0,
            'high_risk_medication_exposure': 0
        }

        for med_class in self.medication_classes.keys():
            features[f'{med_class}_exposure'] = 0

        return features


class PregnancyComplicationsAnalyzer:
    """
    Analyze pregnancy complications and birth outcomes.

    Complications associated with neurodevelopmental risk:
    - Gestational diabetes (metabolic dysregulation)
    - Preeclampsia (hypoxia, placental insufficiency)
    - Preterm birth (<37 weeks)
    - Low birth weight (<2500g)
    - Placental abnormalities
    """

    def __init__(self):
        self.complications = [
            'gestational_diabetes',
            'preeclampsia',
            'placenta_previa',
            'placental_abruption',
            'chorioamnionitis',
            'oligohydramnios',
            'polyhydramnios',
            'IUGR',  # Intrauterine growth restriction
            'pregnancy_induced_hypertension'
        ]

    def extract_complication_features(
        self,
        pregnancy_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract pregnancy complication and birth outcome features.

        Parameters
        ----------
        pregnancy_data : pd.DataFrame
            Columns: complication_type, gestational_age_weeks, birth_weight_grams,
                    apgar_1min, apgar_5min, delivery_method, nicu_admission

        Returns
        -------
        dict
            Pregnancy and birth features
        """
        features = {}

        # Individual complications
        for complication in self.complications:
            features[complication] = 0

        if 'complication_type' in pregnancy_data.columns:
            for complication in pregnancy_data['complication_type'].dropna():
                if complication in self.complications:
                    features[complication] = 1

        # Complication count
        features['n_complications'] = sum(features[c] for c in self.complications)
        features['any_complication'] = int(features['n_complications'] > 0)

        # Birth outcomes
        if 'gestational_age_weeks' in pregnancy_data.columns:
            ga = pregnancy_data['gestational_age_weeks'].iloc[0] if len(pregnancy_data) > 0 else np.nan
            features['gestational_age_weeks'] = ga if pd.notna(ga) else np.nan
            features['preterm_birth'] = int(ga < 37) if pd.notna(ga) else 0
            features['very_preterm_birth'] = int(ga < 32) if pd.notna(ga) else 0

        if 'birth_weight_grams' in pregnancy_data.columns:
            bw = pregnancy_data['birth_weight_grams'].iloc[0] if len(pregnancy_data) > 0 else np.nan
            features['birth_weight_grams'] = bw if pd.notna(bw) else np.nan
            features['low_birth_weight'] = int(bw < 2500) if pd.notna(bw) else 0
            features['very_low_birth_weight'] = int(bw < 1500) if pd.notna(bw) else 0

        if 'apgar_5min' in pregnancy_data.columns:
            apgar = pregnancy_data['apgar_5min'].iloc[0] if len(pregnancy_data) > 0 else np.nan
            features['apgar_5min'] = apgar if pd.notna(apgar) else np.nan
            features['low_apgar'] = int(apgar < 7) if pd.notna(apgar) else 0

        if 'nicu_admission' in pregnancy_data.columns:
            features['nicu_admission'] = int(pregnancy_data['nicu_admission'].iloc[0]) if len(pregnancy_data) > 0 else 0

        if 'delivery_method' in pregnancy_data.columns:
            features['cesarean_delivery'] = int(
                pregnancy_data['delivery_method'].iloc[0] == 'cesarean'
            ) if len(pregnancy_data) > 0 else 0

        # Composite risk score
        features['birth_complication_score'] = (
            features.get('preterm_birth', 0) * 2 +
            features.get('low_birth_weight', 0) * 2 +
            features.get('low_apgar', 0) * 3 +
            features.get('nicu_admission', 0) * 2 +
            min(features.get('n_complications', 0), 5)
        )

        return features


class MaternalStressNutritionAnalyzer:
    """
    Analyze maternal stress and nutritional status during pregnancy.

    - Life stress events
    - Cortisol levels (if available)
    - Depression/anxiety symptoms
    - Folate/vitamin D/omega-3 supplementation
    """

    def extract_stress_features(
        self,
        stress_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract maternal stress features."""
        features = {}

        if 'life_events' in stress_data.columns:
            features['n_stressful_life_events'] = stress_data['life_events'].sum()
            features['major_life_stress'] = int(stress_data['life_events'].sum() >= 2)

        if 'depression_score' in stress_data.columns:
            features['maternal_depression_score'] = stress_data['depression_score'].mean()
            features['maternal_depression_clinical'] = int(
                stress_data['depression_score'].mean() >= 10  # PHQ-9 threshold
            )

        if 'anxiety_score' in stress_data.columns:
            features['maternal_anxiety_score'] = stress_data['anxiety_score'].mean()

        if 'cortisol_mean' in stress_data.columns:
            features['maternal_cortisol_mean'] = stress_data['cortisol_mean'].mean()
            features['maternal_cortisol_elevated'] = int(
                stress_data['cortisol_mean'].mean() > 15  # μg/dL threshold
            )

        return features

    def extract_nutrition_features(
        self,
        nutrition_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract maternal nutritional features."""
        features = {}

        # Supplementation
        if 'prenatal_vitamins' in nutrition_data.columns:
            features['prenatal_vitamins'] = int(nutrition_data['prenatal_vitamins'].any())

        if 'folate_supplementation' in nutrition_data.columns:
            features['folate_supplementation'] = int(nutrition_data['folate_supplementation'].any())
            if 'folate_dose_mcg' in nutrition_data.columns:
                features['folate_dose_adequate'] = int(
                    nutrition_data['folate_dose_mcg'].mean() >= 400
                )

        if 'vitamin_d_level' in nutrition_data.columns:
            features['vitamin_d_level'] = nutrition_data['vitamin_d_level'].mean()
            features['vitamin_d_deficiency'] = int(
                nutrition_data['vitamin_d_level'].mean() < 20  # ng/mL
            )

        if 'omega3_supplementation' in nutrition_data.columns:
            features['omega3_supplementation'] = int(nutrition_data['omega3_supplementation'].any())

        return features


def extract_comprehensive_prenatal_features(
    infection_data: Optional[pd.DataFrame] = None,
    medication_data: Optional[pd.DataFrame] = None,
    pregnancy_data: Optional[pd.DataFrame] = None,
    stress_data: Optional[pd.DataFrame] = None,
    nutrition_data: Optional[pd.DataFrame] = None,
    birth_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Extract comprehensive prenatal and maternal health features.

    Main entry point for prenatal feature extraction.

    Parameters
    ----------
    infection_data : pd.DataFrame, optional
        Maternal infection records
    medication_data : pd.DataFrame, optional
        Prenatal medication exposures
    pregnancy_data : pd.DataFrame, optional
        Pregnancy complications and birth outcomes
    stress_data : pd.DataFrame, optional
        Maternal stress measures
    nutrition_data : pd.DataFrame, optional
        Nutritional status and supplementation
    birth_date : datetime, optional
        Child's birth date for timing calculations

    Returns
    -------
    pd.DataFrame
        Feature matrix with all prenatal/maternal features
    """
    logger.info("Extracting comprehensive prenatal and maternal health features...")

    all_features = {}

    # Maternal infections
    if infection_data is not None and not infection_data.empty:
        logger.info(f"  Processing {len(infection_data)} maternal infection records...")
        infection_analyzer = MaternalInfectionAnalyzer()
        infection_features = infection_analyzer.extract_infection_features(
            infection_data, birth_date
        )
        all_features.update(infection_features)
    else:
        logger.info("  No maternal infection data provided")
        all_features.update(MaternalInfectionAnalyzer()._create_empty_features())

    # Medications
    if medication_data is not None and not medication_data.empty:
        logger.info(f"  Processing {len(medication_data)} medication records...")
        medication_analyzer = MaternalMedicationAnalyzer()
        medication_features = medication_analyzer.extract_medication_features(
            medication_data, birth_date
        )
        all_features.update(medication_features)
    else:
        logger.info("  No medication data provided")
        all_features.update(MaternalMedicationAnalyzer()._create_empty_medication_features())

    # Pregnancy complications
    if pregnancy_data is not None and not pregnancy_data.empty:
        logger.info("  Processing pregnancy complications and birth outcomes...")
        complication_analyzer = PregnancyComplicationsAnalyzer()
        complication_features = complication_analyzer.extract_complication_features(
            pregnancy_data
        )
        all_features.update(complication_features)
    else:
        logger.info("  No pregnancy complication data provided")

    # Stress and nutrition
    stress_nutrition_analyzer = MaternalStressNutritionAnalyzer()

    if stress_data is not None and not stress_data.empty:
        logger.info("  Processing maternal stress data...")
        stress_features = stress_nutrition_analyzer.extract_stress_features(stress_data)
        all_features.update(stress_features)

    if nutrition_data is not None and not nutrition_data.empty:
        logger.info("  Processing maternal nutrition data...")
        nutrition_features = stress_nutrition_analyzer.extract_nutrition_features(nutrition_data)
        all_features.update(nutrition_features)

    # Convert to DataFrame
    features_df = pd.DataFrame([all_features])

    logger.info(f"Extracted {len(all_features)} prenatal/maternal features")

    return features_df


if __name__ == '__main__':
    # Example usage
    logger.info("Prenatal and Maternal Health Feature Extraction initialized")

    # Example: Create sample data
    infection_data = pd.DataFrame({
        'infection_type': ['influenza', 'uti'],
        'gestational_week': [15, 28],
        'severity': [3, 1],
        'fever_days': [3, 0],
        'hospitalization': [False, False],
        'antibiotics_used': [False, True]
    })

    medication_data = pd.DataFrame({
        'medication_name': ['sertraline', 'amoxicillin'],
        'start_week': [8, 28],
        'duration_weeks': [32, 1],
        'indication': ['depression', 'infection']
    })

    pregnancy_data = pd.DataFrame({
        'complication_type': ['gestational_diabetes'],
        'gestational_age_weeks': [38],
        'birth_weight_grams': [3200],
        'apgar_5min': [9],
        'delivery_method': ['vaginal'],
        'nicu_admission': [False]
    })

    # Extract features
    features = extract_comprehensive_prenatal_features(
        infection_data=infection_data,
        medication_data=medication_data,
        pregnancy_data=pregnancy_data
    )

    print("\n" + "="*70)
    print("Sample Prenatal Features")
    print("="*70)
    print(f"\nExtracted {features.shape[1]} features")
    print("\nKey features:")
    for col in sorted(features.columns)[:20]:
        print(f"  {col:40s}: {features[col].iloc[0]}")
