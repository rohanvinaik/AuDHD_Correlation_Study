#!/usr/bin/env python3
"""
Circadian Rhythm & Actigraphy Feature Extraction

Processes actigraphy data for circadian rhythm analysis and sleep metrics
Compatible with ActiGraph, GENEActiv, MotionWatch, and other accelerometers
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActigraphyProcessor:
    """
    Process actigraphy data for circadian and sleep metrics

    Implements algorithms from:
    - Van Someren et al. (1999) - Circadian rest-activity rhythm
    - Oakley (1997) - Sleep detection
    - Ancoli-Israel et al. (2003) - Sleep fragmentation
    """

    def __init__(self, epoch_length: int = 60):
        """
        Initialize processor

        Args:
            epoch_length: Epoch length in seconds (typically 15, 30, or 60)
        """
        self.epoch_length = epoch_length
        self.epochs_per_hour = 3600 // epoch_length
        self.epochs_per_day = 86400 // epoch_length

    def load_actigraphy(self, file_path: Path, file_format: str = 'auto') -> pd.DataFrame:
        """
        Load actigraphy data from various formats

        Args:
            file_path: Path to actigraphy file
            file_format: 'actigraph', 'geneactiv', 'csv', or 'auto'

        Returns:
            df: DataFrame with columns ['timestamp', 'activity']
        """
        if file_format == 'auto':
            suffix = file_path.suffix.lower()
            if suffix == '.csv':
                file_format = 'csv'
            elif suffix == '.agd':
                file_format = 'actigraph'
            elif suffix == '.bin':
                file_format = 'geneactiv'
            else:
                raise ValueError(f"Cannot auto-detect format for {suffix}")

        if file_format == 'csv':
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            if 'activity' not in df.columns:
                # Try to find activity column
                activity_cols = ['counts', 'activity_counts', 'axis1', 'vm']
                for col in activity_cols:
                    if col in df.columns:
                        df = df.rename(columns={col: 'activity'})
                        break
                else:
                    raise ValueError("No activity column found in CSV")

        elif file_format == 'actigraph':
            # ActiGraph AGD format (SQLite database)
            import sqlite3
            conn = sqlite3.connect(file_path)
            df = pd.read_sql_query("SELECT * FROM data", conn)
            conn.close()
            df['timestamp'] = pd.to_datetime(df['dataTimestamp'])
            df['activity'] = df['axis1']  # Or vectorMagnitude

        elif file_format == 'geneactiv':
            logger.warning("GENEActiv BIN format requires pyActigraphy or custom parser")
            raise NotImplementedError("GENEActiv format not yet implemented")

        return df[['timestamp', 'activity']].sort_values('timestamp').reset_index(drop=True)

    def detect_sleep_wake(self, activity: np.ndarray, algorithm: str = 'sadeh') -> np.ndarray:
        """
        Detect sleep/wake states from activity counts

        Args:
            activity: Activity counts per epoch
            algorithm: 'sadeh', 'cole_kripke', or 'threshold'

        Returns:
            sleep_wake: Boolean array (True = sleep, False = wake)
        """
        if algorithm == 'sadeh':
            # Sadeh algorithm (1994)
            # Sleep score = 7.601 - 0.065*NAT - 0.056*MEAN - 0.703*SD - 1.08*LOG(N)
            window = 11  # 11-minute window
            sleep_scores = []

            for i in range(len(activity)):
                start = max(0, i - window//2)
                end = min(len(activity), i + window//2 + 1)
                window_data = activity[start:end]

                nat = np.sum(window_data >= 50)  # Number of epochs >= 50 counts
                mean_act = np.mean(window_data)
                sd_act = np.std(window_data)
                log_act = np.log(window_data[window//2] + 1)  # Log of current epoch

                score = 7.601 - 0.065*nat - 0.056*mean_act - 0.703*sd_act - 1.08*log_act
                sleep_scores.append(score)

            sleep_wake = np.array(sleep_scores) >= 0  # Threshold at 0

        elif algorithm == 'cole_kripke':
            # Cole-Kripke algorithm (1992)
            # More sensitive to wake
            P = [0.00001, 0.04, 1.0, 0.04, 0.0001, 0.0001, 0.0004, 0.02]
            A = np.zeros_like(activity, dtype=float)

            for i in range(len(activity)):
                for j, weight in enumerate(P):
                    idx = i + j - 2
                    if 0 <= idx < len(activity):
                        A[i] += weight * activity[idx]

            D = A / 100
            sleep_prob = 1 / (1 + np.exp(0.37 - 0.73*D))
            sleep_wake = sleep_prob >= 0.5

        elif algorithm == 'threshold':
            # Simple threshold method
            threshold = np.percentile(activity, 20)  # Assume sleep is lowest 20%
            sleep_wake = activity < threshold

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return sleep_wake

    def identify_sleep_periods(self, sleep_wake: np.ndarray,
                              df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify main sleep periods and calculate boundaries

        Returns:
            sleep_periods: DataFrame with sleep bout information
        """
        # Find transitions
        diff = np.diff(sleep_wake.astype(int))
        sleep_onsets = np.where(diff == 1)[0] + 1
        sleep_offsets = np.where(diff == -1)[0] + 1

        # Ensure we start with onset
        if len(sleep_offsets) > 0 and len(sleep_onsets) > 0:
            if sleep_offsets[0] < sleep_onsets[0]:
                sleep_offsets = sleep_offsets[1:]

        # Match onsets with offsets
        min_length = min(len(sleep_onsets), len(sleep_offsets))
        sleep_onsets = sleep_onsets[:min_length]
        sleep_offsets = sleep_offsets[:min_length]

        # Create sleep periods dataframe
        periods = []
        for onset, offset in zip(sleep_onsets, sleep_offsets):
            duration_epochs = offset - onset
            duration_minutes = duration_epochs * (self.epoch_length / 60)

            # Only include periods >= 3 hours (main sleep)
            if duration_minutes >= 180:
                periods.append({
                    'onset_idx': onset,
                    'offset_idx': offset,
                    'onset_time': df.iloc[onset]['timestamp'],
                    'offset_time': df.iloc[offset]['timestamp'],
                    'duration_min': duration_minutes
                })

        return pd.DataFrame(periods)

    def calculate_circadian_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate circadian rhythm parameters (Van Someren et al. 1999)

        Returns:
            metrics: Dictionary of circadian rhythm metrics
        """
        # Ensure we have at least 24 hours of data
        if len(df) < self.epochs_per_day:
            logger.warning("Less than 24 hours of data. Circadian metrics may be unreliable.")

        activity = df['activity'].values

        # Calculate hourly averages
        n_hours = len(activity) // self.epochs_per_hour
        hourly_activity = []
        for i in range(n_hours):
            start = i * self.epochs_per_hour
            end = (i + 1) * self.epochs_per_hour
            hourly_activity.append(np.mean(activity[start:end]))

        hourly_activity = np.array(hourly_activity)

        # Interdaily Stability (IS)
        # Measures strength of coupling to 24h rhythm (0-1, higher = more stable)
        n_days = n_hours // 24
        if n_days >= 2:
            hourly_avg = np.mean(hourly_activity)

            # Average activity at each hour across days
            hour_of_day_avg = []
            for h in range(24):
                hour_values = [hourly_activity[d*24 + h] for d in range(n_days)
                             if d*24 + h < len(hourly_activity)]
                hour_of_day_avg.append(np.mean(hour_values))

            numerator = np.sum((np.array(hour_of_day_avg) - hourly_avg)**2) * n_days
            denominator = np.sum((hourly_activity - hourly_avg)**2)
            IS = numerator / denominator if denominator > 0 else 0
        else:
            IS = np.nan

        # Intradaily Variability (IV)
        # Measures fragmentation (0-2, higher = more fragmented)
        diff_activity = np.diff(hourly_activity)
        numerator = np.sum(diff_activity**2) * n_hours
        denominator = np.sum((hourly_activity - np.mean(hourly_activity))**2) * (n_hours - 1)
        IV = numerator / denominator if denominator > 0 else 0

        # Relative Amplitude (RA)
        # Difference between most and least active periods
        M10 = self._calculate_M10(hourly_activity)  # Most active 10 hours
        L5 = self._calculate_L5(hourly_activity)    # Least active 5 hours
        RA = (M10 - L5) / (M10 + L5) if (M10 + L5) > 0 else 0

        # Cosinor analysis for acrophase and MESOR
        try:
            acrophase, MESOR, amplitude, rsq = self._cosinor_analysis(hourly_activity)
        except:
            acrophase, MESOR, amplitude, rsq = np.nan, np.nan, np.nan, np.nan

        metrics = {
            'IS': IS,
            'IV': IV,
            'RA': RA,
            'M10': M10,
            'L5': L5,
            'acrophase': acrophase,  # Hour of peak activity
            'MESOR': MESOR,          # Rhythm-adjusted mean
            'amplitude': amplitude,   # Rhythm amplitude
            'cosinor_rsq': rsq       # Goodness of fit
        }

        return metrics

    def _calculate_M10(self, hourly_activity: np.ndarray) -> float:
        """Calculate average activity during most active 10 consecutive hours"""
        max_avg = 0
        for i in range(len(hourly_activity) - 10 + 1):
            avg = np.mean(hourly_activity[i:i+10])
            if avg > max_avg:
                max_avg = avg
        return max_avg

    def _calculate_L5(self, hourly_activity: np.ndarray) -> float:
        """Calculate average activity during least active 5 consecutive hours"""
        min_avg = np.inf
        for i in range(len(hourly_activity) - 5 + 1):
            avg = np.mean(hourly_activity[i:i+5])
            if avg < min_avg:
                min_avg = avg
        return min_avg

    def _cosinor_analysis(self, hourly_activity: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Fit cosine curve to activity data

        Returns:
            acrophase: Time of peak (hours)
            MESOR: Midline Estimating Statistic Of Rhythm
            amplitude: Amplitude of fitted curve
            rsq: R-squared of fit
        """
        hours = np.arange(len(hourly_activity))

        # Cosine model: y = MESOR + amplitude * cos(2π(t - acrophase)/24)
        def cosine_model(t, mesor, amplitude, acrophase):
            return mesor + amplitude * np.cos(2*np.pi*(t - acrophase)/24)

        # Initial guess
        p0 = [np.mean(hourly_activity),
              (np.max(hourly_activity) - np.min(hourly_activity))/2,
              12]  # Assume acrophase around noon

        try:
            popt, _ = curve_fit(cosine_model, hours, hourly_activity, p0=p0)
            MESOR, amplitude, acrophase = popt

            # Calculate R-squared
            residuals = hourly_activity - cosine_model(hours, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((hourly_activity - np.mean(hourly_activity))**2)
            rsq = 1 - (ss_res / ss_tot)

            # Ensure acrophase is in [0, 24)
            acrophase = acrophase % 24

        except:
            MESOR = np.mean(hourly_activity)
            amplitude = np.nan
            acrophase = np.nan
            rsq = 0

        return acrophase, MESOR, amplitude, rsq

    def calculate_sleep_metrics(self, df: pd.DataFrame, sleep_wake: np.ndarray) -> Dict[str, float]:
        """
        Calculate sleep quality metrics

        Returns:
            metrics: Dictionary of sleep metrics
        """
        # Identify sleep periods
        sleep_periods = self.identify_sleep_periods(sleep_wake, df)

        if len(sleep_periods) == 0:
            logger.warning("No sleep periods detected")
            return {k: np.nan for k in ['tst', 'se', 'waso', 'sol', 'sfi', 'n_awakenings']}

        # Analyze main sleep period (longest)
        main_sleep = sleep_periods.loc[sleep_periods['duration_min'].idxmax()]
        onset_idx = main_sleep['onset_idx']
        offset_idx = main_sleep['offset_idx']

        # Total Sleep Time (TST)
        sleep_epochs = np.sum(sleep_wake[onset_idx:offset_idx])
        TST = sleep_epochs * (self.epoch_length / 60)  # minutes

        # Time in Bed (TIB)
        TIB = (offset_idx - onset_idx) * (self.epoch_length / 60)

        # Sleep Efficiency (SE)
        SE = (TST / TIB) * 100 if TIB > 0 else 0

        # Wake After Sleep Onset (WASO)
        wake_epochs = np.sum(~sleep_wake[onset_idx:offset_idx])
        WASO = wake_epochs * (self.epoch_length / 60)

        # Sleep Onset Latency (SOL) - approximate as time to first sustained sleep
        # Find first 10 consecutive sleep epochs
        SOL = 0
        for i in range(onset_idx, offset_idx - 10):
            if np.all(sleep_wake[i:i+10]):
                SOL = (i - onset_idx) * (self.epoch_length / 60)
                break

        # Number of awakenings
        # Count wake bouts > 1 minute during sleep period
        in_wake_bout = False
        n_awakenings = 0
        wake_bout_length = 0
        min_wake_length = 60 // self.epoch_length  # At least 1 minute

        for i in range(onset_idx, offset_idx):
            if not sleep_wake[i]:  # Wake epoch
                if not in_wake_bout:
                    in_wake_bout = True
                    wake_bout_length = 1
                else:
                    wake_bout_length += 1
            else:  # Sleep epoch
                if in_wake_bout and wake_bout_length >= min_wake_length:
                    n_awakenings += 1
                in_wake_bout = False
                wake_bout_length = 0

        # Sleep Fragmentation Index (SFI)
        # (Number of sleep-wake transitions + % time moving) / 100
        transitions = np.sum(np.abs(np.diff(sleep_wake[onset_idx:offset_idx].astype(int))))
        activity_threshold = np.percentile(df['activity'].values, 90)
        pct_moving = np.mean(df['activity'].values[onset_idx:offset_idx] > activity_threshold) * 100
        SFI = (transitions + pct_moving) / 100

        metrics = {
            'tst': TST,                    # Total sleep time (min)
            'tib': TIB,                    # Time in bed (min)
            'se': SE,                      # Sleep efficiency (%)
            'waso': WASO,                  # Wake after sleep onset (min)
            'sol': SOL,                    # Sleep onset latency (min)
            'n_awakenings': n_awakenings,  # Number of awakenings
            'sfi': SFI,                    # Sleep fragmentation index
            'onset_time': main_sleep['onset_time'],
            'offset_time': main_sleep['offset_time']
        }

        return metrics

    def calculate_activity_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate daytime activity metrics

        Returns:
            metrics: Dictionary of activity metrics
        """
        activity = df['activity'].values

        # Basic statistics
        mean_activity = np.mean(activity)
        peak_activity = np.max(activity)
        std_activity = np.std(activity)

        # Sedentary time (< 10th percentile)
        sedentary_threshold = np.percentile(activity, 10)
        sedentary_epochs = np.sum(activity < sedentary_threshold)
        sedentary_time = sedentary_epochs * (self.epoch_length / 60)  # minutes

        # Moderate-vigorous physical activity (MVPA) (> 75th percentile)
        mvpa_threshold = np.percentile(activity, 75)
        mvpa_epochs = np.sum(activity > mvpa_threshold)
        mvpa_time = mvpa_epochs * (self.epoch_length / 60)  # minutes

        # Variability
        cv_activity = (std_activity / mean_activity) * 100 if mean_activity > 0 else 0

        metrics = {
            'mean_activity': mean_activity,
            'peak_activity': peak_activity,
            'std_activity': std_activity,
            'cv_activity': cv_activity,
            'sedentary_min': sedentary_time,
            'mvpa_min': mvpa_time
        }

        return metrics

    def process_actigraphy(self, file_path: Path,
                          file_format: str = 'auto') -> Dict[str, float]:
        """
        Complete actigraphy processing pipeline

        Args:
            file_path: Path to actigraphy file
            file_format: File format

        Returns:
            metrics: Complete set of circadian, sleep, and activity metrics
        """
        # Load data
        df = self.load_actigraphy(file_path, file_format)

        # Detect sleep/wake
        sleep_wake = self.detect_sleep_wake(df['activity'].values, algorithm='sadeh')

        # Calculate all metrics
        metrics = {}
        metrics.update(self.calculate_circadian_metrics(df))
        metrics.update(self.calculate_sleep_metrics(df, sleep_wake))
        metrics.update(self.calculate_activity_metrics(df))

        # Add quality metrics
        metrics['n_epochs'] = len(df)
        metrics['recording_duration_days'] = len(df) / self.epochs_per_day
        metrics['data_completeness'] = (len(df) / (metrics['recording_duration_days'] * self.epochs_per_day)) * 100

        return metrics


def process_batch_actigraphy(data_dir: Path, output_file: Path) -> pd.DataFrame:
    """
    Batch process actigraphy files

    Args:
        data_dir: Directory containing actigraphy files
        output_file: Where to save results CSV

    Returns:
        results: DataFrame with metrics for all participants
    """
    processor = ActigraphyProcessor()
    results = []

    for file_path in Path(data_dir).glob('*.csv'):
        try:
            logger.info(f"Processing {file_path.name}")
            metrics = processor.process_actigraphy(file_path)
            metrics['participant_id'] = file_path.stem
            results.append(metrics)
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

    return df


if __name__ == '__main__':
    # Example: Generate synthetic actigraphy data
    logger.info("Actigraphy Processor initialized")

    # Simulate 7 days of hourly activity with circadian rhythm
    n_days = 7
    hours = np.arange(n_days * 24)

    # Cosine model with noise
    MESOR = 100
    amplitude = 80
    acrophase = 14  # 2 PM peak
    activity = MESOR + amplitude * np.cos(2*np.pi*(hours - acrophase)/24)
    activity += np.random.normal(0, 20, len(activity))  # Add noise
    activity = np.maximum(activity, 0)  # Non-negative

    # Create dataframe
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=int(i)) for i in hours]
    df = pd.DataFrame({
        'timestamp': timestamps,
        'activity': activity
    })

    # Process
    processor = ActigraphyProcessor(epoch_length=3600)  # 1-hour epochs for this example

    # Calculate circadian metrics
    circadian_metrics = processor.calculate_circadian_metrics(df)

    print("\nCircadian Metrics:")
    print("="*50)
    for k, v in circadian_metrics.items():
        print(f"  {k}: {v:.3f}" if not pd.isna(v) else f"  {k}: N/A")