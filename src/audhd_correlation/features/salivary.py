#!/usr/bin/env python3
"""
Salivary Biomarker Feature Extraction

Processes salivary cortisol, alpha-amylase, and multi-analyte panels
Calculates CAR (Cortisol Awakening Response) and diurnal rhythm metrics
"""

import numpy as np
import pandas as pd
from scipy import stats, integrate
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CortisolProcessor:
    """
    Process salivary cortisol data

    Implements protocols from:
    - Pruessner et al. (2003) - CAR calculation
    - Stalder et al. (2016) - CAR consensus guidelines
    - Adam & Kumari (2009) - Diurnal slope calculation
    """

    def __init__(self):
        """Initialize cortisol processor"""
        # Quality control thresholds
        self.min_cortisol = 0.5  # nmol/L (implausibly low)
        self.max_cortisol = 100  # nmol/L (implausibly high)

        # Timing thresholds (minutes)
        self.max_car_delay = 10  # Maximum delay from awakening for baseline
        self.max_car_duration = 60  # Maximum CAR sampling window

    def load_cortisol_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load cortisol data from CSV

        Expected columns:
        - participant_id
        - collection_date
        - sample_time (HH:MM or datetime)
        - awakening_time (HH:MM or datetime)
        - cortisol_nmol_L (or cortisol_ug_dL)
        - timepoint (e.g., 'awakening', '+15min', '+30min', etc.)

        Returns:
            df: Cleaned cortisol dataframe
        """
        df = pd.read_csv(file_path)

        # Convert times to datetime if needed
        if 'sample_time' in df.columns and df['sample_time'].dtype == 'object':
            if 'collection_date' in df.columns:
                df['sample_datetime'] = pd.to_datetime(
                    df['collection_date'].astype(str) + ' ' + df['sample_time'].astype(str)
                )
            else:
                # Assume times only, add dummy date
                df['sample_datetime'] = pd.to_datetime('2024-01-01 ' + df['sample_time'].astype(str))

        # Convert units if needed (ug/dL to nmol/L)
        if 'cortisol_ug_dL' in df.columns:
            df['cortisol_nmol_L'] = df['cortisol_ug_dL'] * 27.59  # Conversion factor

        return df

    def calculate_minutes_from_awakening(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate minutes from awakening for each sample"""
        df = df.copy()

        if 'awakening_time' in df.columns and 'sample_time' in df.columns:
            # Parse times
            awakening = pd.to_datetime(df['awakening_time'], format='%H:%M', errors='coerce').dt.time
            sample = pd.to_datetime(df['sample_time'], format='%H:%M', errors='coerce').dt.time

            # Calculate difference in minutes
            df['minutes_from_awakening'] = [
                (datetime.combine(datetime.today(), s) -
                 datetime.combine(datetime.today(), a)).total_seconds() / 60
                if pd.notna(s) and pd.notna(a) else np.nan
                for s, a in zip(sample, awakening)
            ]

        elif 'sample_datetime' in df.columns and 'awakening_datetime' in df.columns:
            df['minutes_from_awakening'] = (
                df['sample_datetime'] - df['awakening_datetime']
            ).dt.total_seconds() / 60

        return df

    def validate_cortisol_values(self, cortisol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quality control for cortisol values

        Returns:
            clean_cortisol: QC'd values
            valid_mask: Boolean mask of valid samples
        """
        valid_mask = (cortisol >= self.min_cortisol) & (cortisol <= self.max_cortisol)

        # Winsorize extreme values instead of removing
        cortisol_clean = np.copy(cortisol)
        cortisol_clean[cortisol < self.min_cortisol] = self.min_cortisol
        cortisol_clean[cortisol > self.max_cortisol] = self.max_cortisol

        return cortisol_clean, valid_mask

    def calculate_car_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Cortisol Awakening Response (CAR) metrics

        Args:
            df: DataFrame with CAR samples (awakening + post-awakening)

        Returns:
            metrics: Dictionary of CAR metrics
        """
        # Filter to CAR window (0-60 min post-awakening)
        if 'minutes_from_awakening' not in df.columns:
            df = self.calculate_minutes_from_awakening(df)

        car_df = df[
            (df['minutes_from_awakening'] >= -self.max_car_delay) &
            (df['minutes_from_awakening'] <= self.max_car_duration)
        ].sort_values('minutes_from_awakening')

        if len(car_df) < 2:
            logger.warning("Insufficient CAR samples")
            return {k: np.nan for k in ['car_auc_g', 'car_auc_i', 'car_peak', 'car_reactivity']}

        times = car_df['minutes_from_awakening'].values
        cortisol = car_df['cortisol_nmol_L'].values

        # Validate
        cortisol, valid = self.validate_cortisol_values(cortisol)

        # Baseline (awakening value or interpolated to t=0)
        if 0 in times:
            baseline = cortisol[times == 0][0]
        else:
            # Interpolate to t=0
            baseline = np.interp(0, times, cortisol)

        # Peak value
        peak_cortisol = np.max(cortisol)
        peak_time = times[np.argmax(cortisol)]

        # Reactivity (peak - baseline)
        reactivity = peak_cortisol - baseline

        # AUC with respect to ground (AUCg) - total cortisol output
        auc_g = np.trapz(cortisol, times)

        # AUC with respect to increase (AUCi) - increase from baseline
        baseline_array = np.full_like(cortisol, baseline)
        increase = np.maximum(cortisol - baseline, 0)
        auc_i = np.trapz(increase, times)

        # Mean increase
        mean_increase = np.mean(cortisol) - baseline

        metrics = {
            'car_baseline': baseline,
            'car_peak': peak_cortisol,
            'car_peak_time': peak_time,
            'car_reactivity': reactivity,
            'car_auc_g': auc_g,
            'car_auc_i': auc_i,
            'car_mean_increase': mean_increase,
            'car_n_samples': len(cortisol)
        }

        return metrics

    def calculate_diurnal_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate diurnal cortisol rhythm metrics

        Args:
            df: DataFrame with samples across the day

        Returns:
            metrics: Dictionary of diurnal metrics
        """
        # Need at least morning and evening samples
        if len(df) < 2:
            logger.warning("Insufficient samples for diurnal analysis")
            return {k: np.nan for k in ['slope', 'morning_cortisol', 'evening_cortisol']}

        # Sort by time
        df = df.sort_values('sample_datetime' if 'sample_datetime' in df.columns else 'sample_time')

        times = df['sample_datetime'] if 'sample_datetime' in df.columns else pd.to_datetime(df['sample_time'], format='%H:%M')
        cortisol = df['cortisol_nmol_L'].values

        # Validate
        cortisol, valid = self.validate_cortisol_values(cortisol)

        # Convert times to hours since first sample
        time_hours = [(t - times.iloc[0]).total_seconds() / 3600 for t in times]

        # Linear regression for slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_hours, cortisol)

        # Morning cortisol (first sample)
        morning_cortisol = cortisol[0]

        # Evening cortisol (last sample)
        evening_cortisol = cortisol[-1]

        # Morning/evening ratio
        me_ratio = morning_cortisol / evening_cortisol if evening_cortisol > 0 else np.nan

        # Total AUC
        total_auc = np.trapz(cortisol, time_hours)

        # Mean cortisol
        mean_cortisol = np.mean(cortisol)

        # Coefficient of variation
        cv_cortisol = (np.std(cortisol) / mean_cortisol) * 100 if mean_cortisol > 0 else np.nan

        metrics = {
            'slope': slope,
            'slope_r': r_value,
            'slope_p': p_value,
            'morning_cortisol': morning_cortisol,
            'evening_cortisol': evening_cortisol,
            'morning_evening_ratio': me_ratio,
            'total_auc': total_auc,
            'mean_cortisol': mean_cortisol,
            'cv_cortisol': cv_cortisol,
            'n_samples': len(cortisol)
        }

        return metrics

    def fit_diurnal_curve(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Fit parametric curve to diurnal cortisol pattern

        Uses double exponential model (Kirschbaum & Hellhammer, 1989)
        """
        df = df.sort_values('sample_datetime' if 'sample_datetime' in df.columns else 'sample_time')

        times = df['sample_datetime'] if 'sample_datetime' in df.columns else pd.to_datetime(df['sample_time'], format='%H:%M')
        cortisol = df['cortisol_nmol_L'].values

        # Convert to hours since midnight
        time_hours = np.array([t.hour + t.minute/60 for t in times])

        # Double exponential model: C(t) = C0 * (exp(-k1*t) - exp(-k2*t)) + baseline
        def double_exp(t, C0, k1, k2, baseline):
            return C0 * (np.exp(-k1*t) - np.exp(-k2*t)) + baseline

        try:
            # Initial guess
            p0 = [np.max(cortisol), 0.5, 2.0, np.min(cortisol)]

            popt, pcov = curve_fit(double_exp, time_hours, cortisol, p0=p0, maxfev=5000)

            # Calculate R-squared
            residuals = cortisol - double_exp(time_hours, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((cortisol - np.mean(cortisol))**2)
            r_squared = 1 - (ss_res / ss_tot)

            metrics = {
                'curve_C0': popt[0],
                'curve_k1': popt[1],
                'curve_k2': popt[2],
                'curve_baseline': popt[3],
                'curve_rsq': r_squared
            }

        except:
            logger.warning("Could not fit diurnal curve")
            metrics = {k: np.nan for k in ['curve_C0', 'curve_k1', 'curve_k2', 'curve_baseline', 'curve_rsq']}

        return metrics

    def process_cortisol(self, file_path: Path, protocol: str = 'car') -> Dict[str, float]:
        """
        Complete cortisol processing pipeline

        Args:
            file_path: Path to cortisol data CSV
            protocol: 'car', 'diurnal', or 'both'

        Returns:
            metrics: Complete cortisol metrics
        """
        df = self.load_cortisol_data(file_path)
        metrics = {}

        if protocol in ['car', 'both']:
            # Filter to CAR samples
            if 'timepoint' in df.columns:
                car_df = df[df['timepoint'].str.contains('awakening|\\+\\d+min', case=False, na=False)]
            else:
                car_df = self.calculate_minutes_from_awakening(df)
                car_df = car_df[car_df['minutes_from_awakening'] <= 60]

            if len(car_df) >= 2:
                metrics.update(self.calculate_car_metrics(car_df))

        if protocol in ['diurnal', 'both']:
            # Use all samples for diurnal
            if len(df) >= 2:
                metrics.update(self.calculate_diurnal_metrics(df))

                # Fit curve if enough samples
                if len(df) >= 4:
                    metrics.update(self.fit_diurnal_curve(df))

        return metrics


class MultiAnalyteSaliva:
    """
    Process multiple biomarkers from saliva samples

    Handles:
    - Stress markers (cortisol, alpha-amylase, DHEA)
    - Immune markers (IL-1β, IL-6, TNF-α, CRP, IgA)
    - Metabolic markers (glucose, insulin, leptin)
    - Other (melatonin, testosterone)
    """

    def __init__(self):
        """Initialize multi-analyte processor"""
        # Reference ranges (approximate, vary by lab)
        self.reference_ranges = {
            'cortisol_nmol_L': (5, 25),
            'alpha_amylase_U_mL': (20, 120),
            'dhea_pg_mL': (100, 500),
            'il1b_pg_mL': (0, 5),
            'il6_pg_mL': (0, 10),
            'tnfa_pg_mL': (0, 8),
            'crp_mg_L': (0, 3),
            'iga_ug_mL': (50, 200),
            'glucose_mg_dL': (60, 100),
            'melatonin_pg_mL': (5, 50),
            'testosterone_pg_mL': (50, 200)
        }

    def load_multi_analyte_data(self, file_path: Path) -> pd.DataFrame:
        """Load multi-analyte saliva data"""
        df = pd.read_csv(file_path)
        return df

    def calculate_stress_panel(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate stress marker metrics

        Returns:
            metrics: Stress biomarker metrics
        """
        metrics = {}

        # Cortisol (already handled by CortisolProcessor)
        if 'cortisol_nmol_L' in df.columns:
            metrics['mean_cortisol'] = df['cortisol_nmol_L'].mean()
            metrics['cv_cortisol'] = (df['cortisol_nmol_L'].std() / metrics['mean_cortisol']) * 100

        # Alpha-amylase (sympathetic nervous system activity)
        if 'alpha_amylase_U_mL' in df.columns:
            metrics['mean_alpha_amylase'] = df['alpha_amylase_U_mL'].mean()
            metrics['peak_alpha_amylase'] = df['alpha_amylase_U_mL'].max()

        # DHEA (anabolic steroid, buffering cortisol)
        if 'dhea_pg_mL' in df.columns:
            metrics['mean_dhea'] = df['dhea_pg_mL'].mean()

            # Cortisol/DHEA ratio (stress vulnerability index)
            if 'cortisol_nmol_L' in df.columns:
                # Convert cortisol to pg/mL for ratio (1 nmol/L = ~362 pg/mL)
                cortisol_pg = df['cortisol_nmol_L'] * 362
                metrics['cortisol_dhea_ratio'] = np.mean(cortisol_pg / df['dhea_pg_mL'])

        return metrics

    def calculate_immune_panel(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate immune marker metrics

        Returns:
            metrics: Immune biomarker metrics
        """
        metrics = {}

        # Pro-inflammatory cytokines
        for cytokine in ['il1b_pg_mL', 'il6_pg_mL', 'tnfa_pg_mL']:
            if cytokine in df.columns:
                name = cytokine.split('_')[0].upper()
                metrics[f'mean_{name}'] = df[cytokine].mean()
                metrics[f'{name}_elevated'] = np.mean(
                    df[cytokine] > self.reference_ranges[cytokine][1]
                ) * 100

        # CRP (acute phase protein)
        if 'crp_mg_L' in df.columns:
            metrics['mean_CRP'] = df['crp_mg_L'].mean()
            metrics['CRP_elevated'] = np.mean(df['crp_mg_L'] > 3) * 100  # >3 mg/L = elevated

        # IgA (mucosal immunity)
        if 'iga_ug_mL' in df.columns:
            metrics['mean_IgA'] = df['iga_ug_mL'].mean()

        # Composite inflammatory index
        inflammatory_markers = []
        for marker in ['il1b_pg_mL', 'il6_pg_mL', 'tnfa_pg_mL', 'crp_mg_L']:
            if marker in df.columns:
                # Z-score relative to reference range
                ref_low, ref_high = self.reference_ranges[marker]
                ref_mean = (ref_low + ref_high) / 2
                ref_std = (ref_high - ref_low) / 4  # Approximate
                z_scores = (df[marker] - ref_mean) / ref_std
                inflammatory_markers.append(z_scores.mean())

        if inflammatory_markers:
            metrics['inflammatory_index'] = np.mean(inflammatory_markers)

        return metrics

    def calculate_metabolic_panel(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate metabolic marker metrics

        Returns:
            metrics: Metabolic biomarker metrics
        """
        metrics = {}

        # Glucose
        if 'glucose_mg_dL' in df.columns:
            metrics['mean_glucose'] = df['glucose_mg_dL'].mean()

        # Insulin
        if 'insulin_uU_mL' in df.columns:
            metrics['mean_insulin'] = df['insulin_uU_mL'].mean()

            # HOMA-IR if both glucose and insulin available
            if 'glucose_mg_dL' in df.columns:
                metrics['HOMA_IR'] = (df['glucose_mg_dL'] * df['insulin_uU_mL']) / 405

        # Leptin
        if 'leptin_ng_mL' in df.columns:
            metrics['mean_leptin'] = df['leptin_ng_mL'].mean()

        return metrics

    def calculate_circadian_hormones(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate circadian hormone metrics (melatonin)

        Returns:
            metrics: Circadian hormone metrics
        """
        metrics = {}

        if 'melatonin_pg_mL' in df.columns:
            metrics['mean_melatonin'] = df['melatonin_pg_mL'].mean()
            metrics['peak_melatonin'] = df['melatonin_pg_mL'].max()

            # DLMO (Dim Light Melatonin Onset) - threshold method
            # Typically defined as first sustained rise above threshold
            threshold = df['melatonin_pg_mL'].iloc[0] + 2 * df['melatonin_pg_mL'].std()

            for i in range(len(df) - 1):
                if df['melatonin_pg_mL'].iloc[i] > threshold:
                    if 'sample_time' in df.columns:
                        metrics['DLMO_time'] = df['sample_time'].iloc[i]
                    break

        return metrics

    def process_multi_analyte(self, file_path: Path) -> Dict[str, float]:
        """
        Complete multi-analyte processing pipeline

        Args:
            file_path: Path to multi-analyte CSV

        Returns:
            metrics: Complete biomarker panel
        """
        df = self.load_multi_analyte_data(file_path)
        metrics = {}

        metrics.update(self.calculate_stress_panel(df))
        metrics.update(self.calculate_immune_panel(df))
        metrics.update(self.calculate_metabolic_panel(df))
        metrics.update(self.calculate_circadian_hormones(df))

        return metrics


def process_batch_salivary(data_dir: Path, output_file: Path,
                           protocol: str = 'both') -> pd.DataFrame:
    """
    Batch process salivary biomarker files

    Args:
        data_dir: Directory containing salivary data files
        output_file: Where to save results CSV
        protocol: 'car', 'diurnal', 'multi_analyte', or 'both'

    Returns:
        results: DataFrame with metrics for all participants
    """
    cortisol_processor = CortisolProcessor()
    multi_processor = MultiAnalyteSaliva()

    results = []

    for file_path in Path(data_dir).glob('*.csv'):
        try:
            logger.info(f"Processing {file_path.name}")
            metrics = {'participant_id': file_path.stem}

            if protocol in ['car', 'diurnal', 'both']:
                cort_metrics = cortisol_processor.process_cortisol(file_path, protocol=protocol)
                metrics.update(cort_metrics)

            if protocol == 'multi_analyte':
                multi_metrics = multi_processor.process_multi_analyte(file_path)
                metrics.update(multi_metrics)

            results.append(metrics)

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

    return df


if __name__ == '__main__':
    # Example: Generate synthetic cortisol data
    logger.info("Salivary Biomarker Processor initialized")

    # Simulate CAR protocol
    times = [0, 15, 30, 45]  # Minutes from awakening
    # Typical CAR: rise from baseline, peak at 30min, then decline
    baseline = 15  # nmol/L
    peak = 25
    cortisol = baseline + (peak - baseline) * np.array([0, 0.6, 1.0, 0.8])
    cortisol += np.random.normal(0, 1, len(cortisol))  # Add noise

    # Create dataframe
    df = pd.DataFrame({
        'participant_id': ['P001'] * len(times),
        'minutes_from_awakening': times,
        'cortisol_nmol_L': cortisol,
        'timepoint': ['awakening', '+15min', '+30min', '+45min']
    })

    # Process
    processor = CortisolProcessor()
    car_metrics = processor.calculate_car_metrics(df)

    print("\nCAR Metrics:")
    print("="*50)
    for k, v in car_metrics.items():
        print(f"  {k}: {v:.2f}")