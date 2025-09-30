#!/usr/bin/env python3
"""
Autonomic Nervous System Feature Extraction

Processes HRV, ECG, blood pressure, and other autonomic signals
Handles multiple input formats from PhysioNet, ABCD, NSRR, and wearables
"""

import numpy as np
import pandas as pd
from scipy import signal, interpolate
from scipy.stats import entropy
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRVProcessor:
    """
    Heart Rate Variability (HRV) processing pipeline

    Handles multiple input formats:
    - Raw ECG (requires R-peak detection)
    - RR intervals (direct processing)
    - Wearable data (device-specific parsers)
    """

    def __init__(self, sampling_rate: float = 4.0):
        """
        Initialize HRV processor

        Args:
            sampling_rate: Hz for interpolated RR series (default 4Hz recommended)
        """
        self.fs = sampling_rate
        self.min_rr = 300  # ms - physiologically plausible minimum
        self.max_rr = 2000  # ms - physiologically plausible maximum

    def detect_r_peaks(self, ecg: np.ndarray, fs: float = 256) -> np.ndarray:
        """
        Detect R-peaks in raw ECG using Pan-Tompkins algorithm

        Args:
            ecg: Raw ECG signal
            fs: Sampling frequency in Hz

        Returns:
            r_peaks: Indices of detected R-peaks
        """
        # Band-pass filter (5-15 Hz)
        nyquist = fs / 2
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = signal.butter(2, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg)

        # Derivative (emphasizes QRS complex)
        diff = np.diff(filtered)

        # Squaring
        squared = diff ** 2

        # Moving average integration
        window_size = int(0.150 * fs)  # 150ms window
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

        # Adaptive thresholding
        peaks, properties = signal.find_peaks(
            integrated,
            distance=int(0.6 * fs),  # Minimum 600ms between peaks (100 bpm)
            prominence=np.mean(integrated) * 0.2
        )

        return peaks

    def extract_rr_intervals(self, ecg: np.ndarray, fs: float = 256) -> np.ndarray:
        """Extract RR intervals from raw ECG"""
        r_peaks = self.detect_r_peaks(ecg, fs)
        rr_intervals = np.diff(r_peaks) / fs * 1000  # Convert to milliseconds
        return rr_intervals

    def validate_rr_intervals(self, rr_intervals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and clean RR intervals

        Returns:
            clean_rr: Validated RR intervals
            artifact_mask: Boolean mask of artifacts
        """
        # Remove physiologically implausible values
        valid_mask = (rr_intervals >= self.min_rr) & (rr_intervals <= self.max_rr)

        # Detect ectopic beats using median absolute deviation
        median_rr = np.median(rr_intervals[valid_mask])
        mad = np.median(np.abs(rr_intervals[valid_mask] - median_rr))
        threshold = median_rr + 3 * mad

        ectopic_mask = (rr_intervals < threshold) & (rr_intervals > threshold * -1)

        artifact_mask = ~(valid_mask & ectopic_mask)

        # Interpolate artifacts
        if np.any(artifact_mask):
            x = np.where(~artifact_mask)[0]
            y = rr_intervals[~artifact_mask]
            x_all = np.arange(len(rr_intervals))

            if len(x) > 1:
                f = interpolate.interp1d(x, y, kind='cubic', fill_value='extrapolate')
                clean_rr = f(x_all)
            else:
                clean_rr = np.full_like(rr_intervals, median_rr)
        else:
            clean_rr = rr_intervals

        return clean_rr, artifact_mask

    def detrend_rr(self, rr_intervals: np.ndarray, method: str = 'smoothness_priors') -> np.ndarray:
        """
        Remove slow-varying trends from RR series

        Args:
            rr_intervals: RR intervals in ms
            method: 'linear', 'polynomial', or 'smoothness_priors'
        """
        if method == 'linear':
            return signal.detrend(rr_intervals, type='linear')
        elif method == 'polynomial':
            x = np.arange(len(rr_intervals))
            p = np.polyfit(x, rr_intervals, 3)
            trend = np.polyval(p, x)
            return rr_intervals - trend
        elif method == 'smoothness_priors':
            # Smoothness priors detrending (Tarvainen et al. 2002)
            lambda_param = 500
            N = len(rr_intervals)
            I = np.eye(N)
            D2 = np.diff(I, n=2, axis=0)
            trend = np.linalg.solve(I + lambda_param**2 * D2.T @ D2, rr_intervals)
            return rr_intervals - trend
        else:
            return rr_intervals

    def interpolate_rr(self, rr_intervals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate RR intervals to evenly-spaced time series

        Returns:
            time: Interpolated time vector
            rr_interp: Interpolated RR series
        """
        # Create cumulative time
        t_rr = np.cumsum(rr_intervals) / 1000  # Convert to seconds
        t_rr = np.insert(t_rr, 0, 0)  # Add t=0

        # Interpolate to even sampling
        t_interp = np.arange(0, t_rr[-1], 1/self.fs)

        # Cubic spline interpolation
        f = interpolate.interp1d(t_rr[:-1], rr_intervals, kind='cubic',
                                fill_value='extrapolate')
        rr_interp = f(t_interp)

        return t_interp, rr_interp

    def calculate_time_domain(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Calculate time-domain HRV metrics

        Returns:
            metrics: Dictionary of time-domain features
        """
        rr_ms = rr_intervals

        # Basic statistics
        mean_rr = np.mean(rr_ms)
        sdnn = np.std(rr_ms, ddof=1)

        # Successive differences
        diff_rr = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(diff_rr**2))
        sdsd = np.std(diff_rr, ddof=1)

        # NN50 and pNN50
        nn50 = np.sum(np.abs(diff_rr) > 50)
        pnn50 = (nn50 / len(diff_rr)) * 100

        # NN20 (more sensitive)
        nn20 = np.sum(np.abs(diff_rr) > 20)
        pnn20 = (nn20 / len(diff_rr)) * 100

        # Mean heart rate
        mean_hr = 60000 / mean_rr

        metrics = {
            'mean_rr': mean_rr,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'sdsd': sdsd,
            'nn50': nn50,
            'pnn50': pnn50,
            'nn20': nn20,
            'pnn20': pnn20,
            'mean_hr': mean_hr,
            'min_hr': 60000 / np.max(rr_ms),
            'max_hr': 60000 / np.min(rr_ms),
            'cv_rr': (sdnn / mean_rr) * 100  # Coefficient of variation
        }

        return metrics

    def calculate_frequency_domain(self, rr_interp: np.ndarray) -> Dict[str, float]:
        """
        Calculate frequency-domain HRV metrics using Welch's method

        Args:
            rr_interp: Interpolated RR series (from interpolate_rr)

        Returns:
            metrics: Dictionary of frequency-domain features
        """
        # Welch's periodogram
        nperseg = min(256, len(rr_interp))
        freqs, psd = signal.welch(rr_interp, fs=self.fs, nperseg=nperseg,
                                  nfft=4096, detrend='linear')

        # Define frequency bands (Task Force 1996)
        vlf_band = (0.003, 0.04)  # Hz
        lf_band = (0.04, 0.15)    # Hz
        hf_band = (0.15, 0.4)     # Hz

        # Calculate power in each band
        vlf_power = np.trapz(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])],
                            freqs[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])

        lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])],
                           freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])

        hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])],
                           freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])

        total_power = vlf_power + lf_power + hf_power

        # Normalized powers
        lf_nu = (lf_power / (lf_power + hf_power)) * 100
        hf_nu = (hf_power / (lf_power + hf_power)) * 100

        # LF/HF ratio
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

        # Peak frequencies
        lf_peak = freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])][
            np.argmax(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
        ] if np.any((freqs >= lf_band[0]) & (freqs < lf_band[1])) else np.nan

        hf_peak = freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])][
            np.argmax(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
        ] if np.any((freqs >= hf_band[0]) & (freqs < hf_band[1])) else np.nan

        metrics = {
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'total_power': total_power,
            'lf_nu': lf_nu,
            'hf_nu': hf_nu,
            'lf_hf_ratio': lf_hf_ratio,
            'lf_peak': lf_peak,
            'hf_peak': hf_peak
        }

        return metrics

    def calculate_nonlinear(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Calculate nonlinear HRV metrics

        Returns:
            metrics: Dictionary of nonlinear features
        """
        # Poincaré plot indices (SD1, SD2)
        diff_rr = np.diff(rr_intervals)
        sd1 = np.sqrt(np.var(diff_rr) / 2)  # Short-term variability
        sd2 = np.sqrt(2 * np.var(rr_intervals) - np.var(diff_rr) / 2)  # Long-term variability
        sd_ratio = sd1 / sd2 if sd2 > 0 else np.nan

        # Sample entropy
        try:
            sampen = self._sample_entropy(rr_intervals, m=2, r=0.2*np.std(rr_intervals))
        except:
            sampen = np.nan

        # Approximate entropy
        try:
            apen = self._approximate_entropy(rr_intervals, m=2, r=0.2*np.std(rr_intervals))
        except:
            apen = np.nan

        # Detrended fluctuation analysis (DFA)
        try:
            dfa_alpha1, dfa_alpha2 = self._detrended_fluctuation_analysis(rr_intervals)
        except:
            dfa_alpha1, dfa_alpha2 = np.nan, np.nan

        metrics = {
            'sd1': sd1,
            'sd2': sd2,
            'sd_ratio': sd_ratio,
            'sampen': sampen,
            'apen': apen,
            'dfa_alpha1': dfa_alpha1,
            'dfa_alpha2': dfa_alpha2
        }

        return metrics

    def _sample_entropy(self, signal_data: np.ndarray, m: int, r: float) -> float:
        """Calculate sample entropy (SampEn)"""
        N = len(signal_data)

        def _maxdist(x_i, x_j):
            return max(np.abs(x_i - x_j))

        def _phi(m):
            patterns = np.array([signal_data[i:i+m] for i in range(N-m)])
            count = 0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j and _maxdist(patterns[i], patterns[j]) <= r:
                        count += 1
            return count

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        if phi_m == 0 or phi_m1 == 0:
            return np.nan

        return -np.log(phi_m1 / phi_m)

    def _approximate_entropy(self, signal_data: np.ndarray, m: int, r: float) -> float:
        """Calculate approximate entropy (ApEn)"""
        def _phi(m):
            patterns = np.array([signal_data[i:i+m] for i in range(len(signal_data)-m+1)])
            C = np.zeros(len(patterns))
            for i in range(len(patterns)):
                dist = np.max(np.abs(patterns - patterns[i]), axis=1)
                C[i] = np.sum(dist <= r) / len(patterns)
            return np.mean(np.log(C))

        return _phi(m) - _phi(m+1)

    def _detrended_fluctuation_analysis(self, signal_data: np.ndarray) -> Tuple[float, float]:
        """
        Calculate DFA alpha1 (short-term) and alpha2 (long-term)
        """
        # Integrate the signal
        y = np.cumsum(signal_data - np.mean(signal_data))

        # Define scales
        scales_short = np.arange(4, 16)  # For alpha1
        scales_long = np.arange(16, min(64, len(signal_data)//4))  # For alpha2

        def calculate_fluctuation(scales):
            F = []
            for n in scales:
                # Divide into segments
                n_segments = len(y) // n
                segments = y[:n_segments*n].reshape(n_segments, n)

                # Fit polynomial to each segment
                fluctuations = []
                for segment in segments:
                    x = np.arange(n)
                    p = np.polyfit(x, segment, 1)
                    fit = np.polyval(p, x)
                    fluctuation = np.sqrt(np.mean((segment - fit)**2))
                    fluctuations.append(fluctuation)

                F.append(np.mean(fluctuations))

            # Log-log regression
            log_scales = np.log(scales)
            log_F = np.log(F)
            alpha = np.polyfit(log_scales, log_F, 1)[0]

            return alpha

        alpha1 = calculate_fluctuation(scales_short)
        alpha2 = calculate_fluctuation(scales_long)

        return alpha1, alpha2

    def process_hrv(self, input_data: Union[np.ndarray, pd.DataFrame],
                    input_type: str = 'rr_intervals',
                    fs: float = 256) -> Dict[str, float]:
        """
        Main processing pipeline

        Args:
            input_data: Raw ECG or RR intervals
            input_type: 'ecg', 'rr_intervals', or 'wearable'
            fs: Sampling frequency (for ECG only)

        Returns:
            metrics: Complete HRV feature set
        """
        # Extract RR intervals based on input type
        if input_type == 'ecg':
            rr_intervals = self.extract_rr_intervals(input_data, fs)
        elif input_type == 'rr_intervals':
            rr_intervals = np.array(input_data)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        # Validate and clean
        rr_clean, artifact_mask = self.validate_rr_intervals(rr_intervals)

        # Detrend
        rr_detrended = self.detrend_rr(rr_clean)

        # Interpolate for frequency analysis
        t_interp, rr_interp = self.interpolate_rr(rr_detrended)

        # Calculate all metrics
        metrics = {}
        metrics.update(self.calculate_time_domain(rr_clean))
        metrics.update(self.calculate_frequency_domain(rr_interp))
        metrics.update(self.calculate_nonlinear(rr_clean))

        # Add quality metrics
        metrics['n_beats'] = len(rr_intervals)
        metrics['artifact_rate'] = np.mean(artifact_mask) * 100
        metrics['recording_duration'] = np.sum(rr_intervals) / 1000 / 60  # minutes

        return metrics


class WearableParser:
    """Parse HRV data from consumer wearables"""

    @staticmethod
    def parse_apple_watch(file_path: Path) -> pd.DataFrame:
        """Parse Apple Watch Health export"""
        # Apple Watch exports as XML
        # This would need to parse the specific format
        logger.warning("Apple Watch parser not yet implemented")
        return pd.DataFrame()

    @staticmethod
    def parse_fitbit(file_path: Path) -> pd.DataFrame:
        """Parse Fitbit export"""
        logger.warning("Fitbit parser not yet implemented")
        return pd.DataFrame()

    @staticmethod
    def parse_polar(file_path: Path) -> pd.DataFrame:
        """Parse Polar HRM file format"""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find RR interval section
        rr_start = None
        for i, line in enumerate(lines):
            if '[HRData]' in line:
                rr_start = i + 1
                break

        if rr_start is None:
            raise ValueError("No RR interval data found in Polar file")

        # Parse RR intervals
        rr_intervals = []
        for line in lines[rr_start:]:
            if line.strip():
                rr_intervals.append(float(line.strip()))

        return pd.DataFrame({'rr_ms': rr_intervals})


def process_batch_hrv(data_dir: Path, output_file: Path,
                     input_type: str = 'rr_intervals') -> pd.DataFrame:
    """
    Batch process HRV from multiple files

    Args:
        data_dir: Directory containing input files
        output_file: Where to save results CSV
        input_type: Type of input data

    Returns:
        results: DataFrame with HRV metrics for all participants
    """
    processor = HRVProcessor()
    results = []

    for file_path in Path(data_dir).glob('*'):
        try:
            logger.info(f"Processing {file_path.name}")

            # Load data (format-specific)
            if file_path.suffix == '.csv':
                data = pd.read_csv(file_path)
                rr = data['rr_ms'].values
            elif file_path.suffix == '.txt':
                rr = np.loadtxt(file_path)
            elif file_path.suffix == '.hrm':
                data = WearableParser.parse_polar(file_path)
                rr = data['rr_ms'].values
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                continue

            # Process
            metrics = processor.process_hrv(rr, input_type='rr_intervals')
            metrics['participant_id'] = file_path.stem
            results.append(metrics)

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save
    df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

    return df


if __name__ == '__main__':
    # Example usage
    logger.info("HRV Processor initialized")
    logger.info("Example: Generate synthetic RR intervals")

    # Simulate RR intervals (normally distributed around 800ms)
    np.random.seed(42)
    n_beats = 300
    mean_rr = 800
    std_rr = 50
    rr_intervals = np.random.normal(mean_rr, std_rr, n_beats)

    # Process
    processor = HRVProcessor()
    metrics = processor.process_hrv(rr_intervals, input_type='rr_intervals')

    # Display
    print("\nHRV Metrics:")
    print("="*50)
    print("Time Domain:")
    for k in ['mean_rr', 'sdnn', 'rmssd', 'pnn50']:
        print(f"  {k}: {metrics[k]:.2f}")

    print("\nFrequency Domain:")
    for k in ['lf_power', 'hf_power', 'lf_hf_ratio']:
        print(f"  {k}: {metrics[k]:.2f}")

    print("\nNonlinear:")
    for k in ['sd1', 'sd2', 'sampen', 'dfa_alpha1']:
        print(f"  {k}: {metrics[k]:.2f}")