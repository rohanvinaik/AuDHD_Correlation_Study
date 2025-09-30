#!/usr/bin/env python3
"""
Wearables & Digital Phenotyping Analysis
Analyzes accelerometer, heart rate, sleep, and activity data for AuDHD study
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class WearablesResult:
    """Results from wearables analysis"""
    activity_features: pd.DataFrame
    sleep_features: pd.DataFrame
    circadian_features: pd.DataFrame
    heart_rate_variability: Optional[pd.DataFrame] = None
    behavioral_patterns: Optional[pd.DataFrame] = None


class WearablesAnalyzer:
    """
    Digital phenotyping analysis for AuDHD research

    Capabilities:
    1. Activity pattern extraction (accelerometer)
    2. Sleep quality metrics
    3. Circadian rhythm analysis
    4. Heart rate variability (HRV)
    5. Behavioral pattern detection
    """

    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize analyzer

        Parameters
        ----------
        sampling_rate : float
            Accelerometer sampling rate (Hz)
        """
        self.sampling_rate = sampling_rate

    def extract_activity_features(
        self,
        accelerometer_data: pd.DataFrame,
        window_minutes: int = 60
    ) -> pd.DataFrame:
        """
        Extract activity features from accelerometer data

        Parameters
        ----------
        accelerometer_data : pd.DataFrame
            Columns: timestamp, x, y, z (acceleration in g)
        window_minutes : int
            Time window for feature aggregation

        Returns
        -------
        activity_features : pd.DataFrame
            Columns: timestamp, activity_counts, sedentary_time,
                     light_activity, moderate_activity, vigorous_activity
        """
        logger.info("Extracting activity features from accelerometer")

        accel = accelerometer_data.copy()
        accel['timestamp'] = pd.to_datetime(accel['timestamp'])

        # Compute magnitude
        accel['magnitude'] = np.sqrt(
            accel['x']**2 + accel['y']**2 + accel['z']**2
        )

        # Subtract gravity
        accel['acceleration'] = np.abs(accel['magnitude'] - 1.0)

        # Define activity levels (g thresholds)
        sedentary_threshold = 0.05
        light_threshold = 0.15
        moderate_threshold = 0.3
        vigorous_threshold = 0.6

        # Resample to windows
        accel.set_index('timestamp', inplace=True)
        window_str = f'{window_minutes}T'

        features = accel.resample(window_str).agg({
            'acceleration': ['mean', 'std', 'max'],
            'magnitude': 'mean'
        })

        features.columns = ['_'.join(col) for col in features.columns]

        # Activity intensity classification
        accel_mean = features['acceleration_mean']
        features['sedentary_time'] = (accel_mean < sedentary_threshold).astype(int)
        features['light_activity'] = ((accel_mean >= sedentary_threshold) &
                                       (accel_mean < light_threshold)).astype(int)
        features['moderate_activity'] = ((accel_mean >= light_threshold) &
                                          (accel_mean < moderate_threshold)).astype(int)
        features['vigorous_activity'] = (accel_mean >= vigorous_threshold).astype(int)

        # Activity counts
        features['activity_counts'] = features['acceleration_mean'] * 1000

        features.reset_index(inplace=True)

        logger.info(f"  Extracted {len(features)} windows of {window_minutes} min")

        return features

    def extract_sleep_features(
        self,
        activity_data: pd.DataFrame,
        sleep_diary: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extract sleep quality metrics

        Parameters
        ----------
        activity_data : pd.DataFrame
            Windowed activity features
        sleep_diary : pd.DataFrame, optional
            Self-reported sleep times (columns: date, bedtime, wake_time)

        Returns
        -------
        sleep_features : pd.DataFrame
            Columns: date, sleep_duration, sleep_efficiency,
                     wake_after_sleep_onset, sleep_fragmentation
        """
        logger.info("Extracting sleep features")

        activity_data = activity_data.copy()
        activity_data['date'] = activity_data['timestamp'].dt.date

        sleep_features = []

        for date in activity_data['date'].unique():
            day_data = activity_data[activity_data['date'] == date]

            # Identify sleep periods (sustained low activity)
            sleep_threshold = 10  # activity counts
            is_sleep = day_data['activity_counts'] < sleep_threshold

            # Find consecutive sleep bouts
            sleep_bouts = []
            in_sleep = False
            start_idx = None

            for idx, asleep in enumerate(is_sleep):
                if asleep and not in_sleep:
                    # Start of sleep bout
                    start_idx = idx
                    in_sleep = True
                elif not asleep and in_sleep:
                    # End of sleep bout
                    sleep_bouts.append((start_idx, idx))
                    in_sleep = False

            if in_sleep:
                sleep_bouts.append((start_idx, len(is_sleep)))

            # Find longest bout (main sleep period)
            if len(sleep_bouts) > 0:
                longest_bout = max(sleep_bouts, key=lambda x: x[1] - x[0])
                sleep_start_idx, sleep_end_idx = longest_bout

                # Sleep duration (hours)
                sleep_duration = (sleep_end_idx - sleep_start_idx) * 60 / 60  # minutes to hours

                # Wake after sleep onset (WASO)
                sleep_period_activity = day_data.iloc[sleep_start_idx:sleep_end_idx]['activity_counts']
                waso = (sleep_period_activity > sleep_threshold).sum() * 60 / 60  # hours

                # Sleep efficiency
                sleep_efficiency = (sleep_duration - waso) / sleep_duration if sleep_duration > 0 else 0

                # Sleep fragmentation (number of wake bouts)
                wake_bouts = np.diff((sleep_period_activity > sleep_threshold).astype(int))
                sleep_fragmentation = (wake_bouts == 1).sum()

                sleep_features.append({
                    'date': date,
                    'sleep_duration': sleep_duration,
                    'sleep_efficiency': sleep_efficiency,
                    'wake_after_sleep_onset': waso,
                    'sleep_fragmentation': sleep_fragmentation
                })

        sleep_df = pd.DataFrame(sleep_features)

        logger.info(f"  Extracted sleep features for {len(sleep_df)} nights")

        return sleep_df

    def analyze_circadian_rhythms(
        self,
        activity_data: pd.DataFrame,
        days: int = 7
    ) -> pd.DataFrame:
        """
        Analyze circadian activity patterns

        Parameters
        ----------
        activity_data : pd.DataFrame
            Windowed activity features
        days : int
            Number of days to analyze

        Returns
        -------
        circadian_features : pd.DataFrame
            Columns: subject_id, M10 (most active 10h), L5 (least active 5h),
                     relative_amplitude, interdaily_stability, intradaily_variability
        """
        logger.info("Analyzing circadian rhythms")

        activity_data = activity_data.copy()
        activity_data['hour'] = activity_data['timestamp'].dt.hour

        # Aggregate by hour of day
        hourly_activity = activity_data.groupby('hour')['activity_counts'].mean()

        # M10: Most active 10 consecutive hours
        window_10h = 10
        rolling_10h = hourly_activity.rolling(window_10h, center=True).mean()
        m10_start = rolling_10h.idxmax()
        m10 = rolling_10h.max()

        # L5: Least active 5 consecutive hours
        window_5h = 5
        rolling_5h = hourly_activity.rolling(window_5h, center=True).mean()
        l5_start = rolling_5h.idxmin()
        l5 = rolling_5h.min()

        # Relative amplitude
        relative_amplitude = (m10 - l5) / (m10 + l5) if (m10 + l5) > 0 else 0

        # Interdaily stability (IS): consistency across days
        activity_data['date'] = activity_data['timestamp'].dt.date
        daily_patterns = activity_data.groupby(['date', 'hour'])['activity_counts'].mean().unstack()

        if len(daily_patterns) > 1:
            mean_pattern = daily_patterns.mean(axis=0)
            is_score = 1 - (daily_patterns.var(axis=0).mean() / mean_pattern.var())
        else:
            is_score = 0

        # Intradaily variability (IV): fragmentation within days
        activity_changes = activity_data['activity_counts'].diff().abs()
        iv_score = activity_changes.mean() / activity_data['activity_counts'].mean() if activity_data['activity_counts'].mean() > 0 else 0

        circadian_features = pd.DataFrame([{
            'M10': m10,
            'L5': l5,
            'relative_amplitude': relative_amplitude,
            'interdaily_stability': is_score,
            'intradaily_variability': iv_score,
            'M10_start_hour': m10_start,
            'L5_start_hour': l5_start
        }])

        logger.info(f"  Relative amplitude: {relative_amplitude:.3f}")

        return circadian_features

    def compute_heart_rate_variability(
        self,
        heart_rate_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute heart rate variability (HRV) metrics

        Parameters
        ----------
        heart_rate_data : pd.DataFrame
            Columns: timestamp, ibi (inter-beat interval in ms)

        Returns
        -------
        hrv_metrics : pd.DataFrame
            Columns: SDNN, RMSSD, pNN50, LF, HF, LF_HF_ratio
        """
        logger.info("Computing heart rate variability")

        hr_data = heart_rate_data.copy()
        ibi = hr_data['ibi'].values  # ms

        # Time-domain metrics
        sdnn = np.std(ibi)  # Standard deviation of NN intervals

        # RMSSD: Root mean square of successive differences
        diff_ibi = np.diff(ibi)
        rmssd = np.sqrt(np.mean(diff_ibi**2))

        # pNN50: Percentage of successive differences > 50ms
        pnn50 = (np.abs(diff_ibi) > 50).sum() / len(diff_ibi) * 100

        # Frequency-domain metrics (simplified)
        from scipy.signal import welch

        # Convert IBI to time series
        time_series = np.cumsum(ibi) / 1000  # Convert to seconds
        sampling_rate_hrv = 4.0  # Resample to 4 Hz

        # Interpolate to regular sampling
        time_regular = np.arange(0, time_series[-1], 1/sampling_rate_hrv)
        ibi_regular = np.interp(time_regular, time_series, ibi)

        # Welch PSD
        freqs, psd = welch(ibi_regular, fs=sampling_rate_hrv, nperseg=256)

        # Frequency bands
        lf_band = (freqs >= 0.04) & (freqs < 0.15)  # Low frequency
        hf_band = (freqs >= 0.15) & (freqs < 0.4)   # High frequency

        lf_power = np.trapz(psd[lf_band], freqs[lf_band])
        hf_power = np.trapz(psd[hf_band], freqs[hf_band])
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0

        hrv_metrics = pd.DataFrame([{
            'SDNN': sdnn,
            'RMSSD': rmssd,
            'pNN50': pnn50,
            'LF_power': lf_power,
            'HF_power': hf_power,
            'LF_HF_ratio': lf_hf_ratio
        }])

        logger.info(f"  SDNN: {sdnn:.1f} ms, RMSSD: {rmssd:.1f} ms")

        return hrv_metrics

    def detect_behavioral_patterns(
        self,
        activity_data: pd.DataFrame,
        sleep_data: pd.DataFrame,
        window_days: int = 7
    ) -> pd.DataFrame:
        """
        Detect behavioral patterns and anomalies

        Parameters
        ----------
        activity_data : pd.DataFrame
        sleep_data : pd.DataFrame
        window_days : int
            Rolling window for pattern detection

        Returns
        -------
        patterns : pd.DataFrame
            Detected patterns and anomalies
        """
        logger.info("Detecting behavioral patterns")

        activity_data = activity_data.copy()
        activity_data['date'] = activity_data['timestamp'].dt.date

        # Daily activity summary
        daily_summary = activity_data.groupby('date').agg({
            'activity_counts': ['sum', 'mean', 'std'],
            'sedentary_time': 'sum',
            'moderate_activity': 'sum',
            'vigorous_activity': 'sum'
        })

        daily_summary.columns = ['_'.join(col) for col in daily_summary.columns]

        # Merge with sleep
        if 'date' in sleep_data.columns:
            daily_summary = daily_summary.merge(
                sleep_data, left_index=True, right_on='date', how='left'
            )

        # Detect anomalies (activity > 2 SD from mean)
        activity_mean = daily_summary['activity_counts_sum'].mean()
        activity_std = daily_summary['activity_counts_sum'].std()

        daily_summary['anomaly_low_activity'] = (
            daily_summary['activity_counts_sum'] < activity_mean - 2 * activity_std
        ).astype(int)

        daily_summary['anomaly_high_activity'] = (
            daily_summary['activity_counts_sum'] > activity_mean + 2 * activity_std
        ).astype(int)

        # Rolling stability
        daily_summary['activity_stability'] = daily_summary['activity_counts_sum'].rolling(
            window_days
        ).std() / daily_summary['activity_counts_sum'].rolling(window_days).mean()

        logger.info(f"  Detected {daily_summary['anomaly_low_activity'].sum()} low-activity anomalies")

        return daily_summary

    def correlate_with_symptoms(
        self,
        wearable_features: pd.DataFrame,
        symptom_scores: pd.DataFrame,
        method: str = 'spearman'
    ) -> pd.DataFrame:
        """
        Correlate wearable features with symptom scores

        Parameters
        ----------
        wearable_features : pd.DataFrame
            Digital phenotyping features
        symptom_scores : pd.DataFrame
            AuDHD symptom measures
        method : str
            Correlation method

        Returns
        -------
        correlations : pd.DataFrame
        """
        logger.info("Correlating wearable features with symptoms")

        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        # Align data
        common_subjects = wearable_features.index.intersection(symptom_scores.index)
        wearable_aligned = wearable_features.loc[common_subjects]
        symptoms_aligned = symptom_scores.loc[common_subjects]

        results = []

        for wearable_col in wearable_aligned.columns:
            for symptom_col in symptoms_aligned.columns:
                x = wearable_aligned[wearable_col].values
                y = symptoms_aligned[symptom_col].values

                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]

                if len(x_clean) < 10:
                    continue

                if method == 'spearman':
                    corr, p_val = stats.spearmanr(x_clean, y_clean)
                else:
                    corr, p_val = stats.pearsonr(x_clean, y_clean)

                results.append({
                    'wearable_feature': wearable_col,
                    'symptom': symptom_col,
                    'correlation': corr,
                    'p_value': p_val,
                    'n_samples': len(x_clean)
                })

        correlations_df = pd.DataFrame(results).sort_values('p_value')

        _, qvals, _, _ = multipletests(correlations_df['p_value'], method='fdr_bh')
        correlations_df['q_value'] = qvals

        logger.info(f"  Significant correlations (q<0.05): {(qvals < 0.05).sum()}")

        return correlations_df

    def analyze_complete(
        self,
        accelerometer_data: pd.DataFrame,
        heart_rate_data: Optional[pd.DataFrame] = None,
        symptom_scores: Optional[pd.DataFrame] = None
    ) -> WearablesResult:
        """
        Complete wearables analysis pipeline

        Parameters
        ----------
        accelerometer_data : pd.DataFrame
            Raw accelerometer data
        heart_rate_data : pd.DataFrame, optional
            Heart rate/IBI data
        symptom_scores : pd.DataFrame, optional
            Clinical symptom measures

        Returns
        -------
        WearablesResult
        """
        logger.info("=== Complete Wearables & Digital Phenotyping Analysis ===")

        # 1. Activity features
        activity_features = self.extract_activity_features(accelerometer_data)

        # 2. Sleep features
        sleep_features = self.extract_sleep_features(activity_features)

        # 3. Circadian rhythms
        circadian_features = self.analyze_circadian_rhythms(activity_features)

        # 4. Heart rate variability
        if heart_rate_data is not None:
            hrv = self.compute_heart_rate_variability(heart_rate_data)
        else:
            hrv = None

        # 5. Behavioral patterns
        behavioral_patterns = self.detect_behavioral_patterns(
            activity_features, sleep_features
        )

        return WearablesResult(
            activity_features=activity_features,
            sleep_features=sleep_features,
            circadian_features=circadian_features,
            heart_rate_variability=hrv,
            behavioral_patterns=behavioral_patterns
        )


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    logger.info("Wearables & Digital Phenotyping Analysis Module")
    logger.info("Ready for integration with AuDHD correlation study")
    logger.info("\nKey capabilities:")
    logger.info("  1. Activity pattern extraction (accelerometer)")
    logger.info("  2. Sleep quality metrics")
    logger.info("  3. Circadian rhythm analysis (M10, L5, relative amplitude)")
    logger.info("  4. Heart rate variability (HRV)")
    logger.info("  5. Behavioral pattern detection and anomalies")
    logger.info("  6. Symptom correlation analysis")
