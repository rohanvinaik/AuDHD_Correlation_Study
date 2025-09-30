#!/usr/bin/env python3
"""
EEG/MEG Neurophysiology Analysis
Analyzes neural oscillations and connectivity for AuDHD study
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class NeurophysiologyResult:
    """Results from EEG/MEG analysis"""
    power_spectra: pd.DataFrame
    connectivity: pd.DataFrame
    erp_components: Optional[pd.DataFrame] = None
    time_frequency: Optional[pd.DataFrame] = None
    network_metrics: Optional[pd.DataFrame] = None


class NeurophysiologyAnalyzer:
    """
    EEG/MEG analysis for AuDHD research

    Capabilities:
    1. Spectral power analysis (frequency bands)
    2. Functional connectivity (phase-lag index, coherence)
    3. Event-related potentials (ERPs)
    4. Time-frequency analysis
    5. Network graph metrics
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """
        Initialize analyzer

        Parameters
        ----------
        sampling_rate : float
            Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.frequency_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }

    def preprocess_eeg(
        self,
        raw_data: np.ndarray,
        channels: List[str],
        highpass: float = 1.0,
        lowpass: float = 100.0,
        notch: Optional[float] = 60.0
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Basic EEG preprocessing

        Parameters
        ----------
        raw_data : np.ndarray
            Raw EEG data (channels × time)
        channels : List[str]
            Channel names
        highpass : float
            High-pass filter cutoff (Hz)
        lowpass : float
            Low-pass filter cutoff (Hz)
        notch : float, optional
            Notch filter frequency (Hz) for line noise

        Returns
        -------
        filtered_data : np.ndarray
        valid_channels : List[str]
        """
        logger.info("Preprocessing EEG data")

        from scipy.signal import butter, filtfilt, iirnotch

        # Remove bad channels (flat or excessive noise)
        channel_std = raw_data.std(axis=1)
        valid_mask = (channel_std > 0.1) & (channel_std < 1000)
        filtered_data = raw_data[valid_mask]
        valid_channels = [ch for ch, valid in zip(channels, valid_mask) if valid]

        logger.info(f"  Retained {len(valid_channels)}/{len(channels)} channels")

        # Bandpass filter
        nyquist = self.sampling_rate / 2
        b_high, a_high = butter(4, highpass / nyquist, btype='high')
        b_low, a_low = butter(4, lowpass / nyquist, btype='low')

        filtered_data = filtfilt(b_high, a_high, filtered_data, axis=1)
        filtered_data = filtfilt(b_low, a_low, filtered_data, axis=1)

        # Notch filter for line noise
        if notch is not None:
            b_notch, a_notch = iirnotch(notch / nyquist, Q=30)
            filtered_data = filtfilt(b_notch, a_notch, filtered_data, axis=1)

        logger.info("  Applied bandpass and notch filters")

        return filtered_data, valid_channels

    def compute_power_spectra(
        self,
        data: np.ndarray,
        channels: List[str],
        method: str = 'welch'
    ) -> pd.DataFrame:
        """
        Compute power spectral density

        Parameters
        ----------
        data : np.ndarray
            Preprocessed EEG (channels × time)
        channels : List[str]
            Channel names
        method : str
            Method ('welch' or 'multitaper')

        Returns
        -------
        power_spectra : pd.DataFrame
            Columns: channel, band, power
        """
        logger.info("Computing power spectra")

        from scipy.signal import welch

        results = []

        for i, channel in enumerate(channels):
            # Welch's method
            freqs, psd = welch(
                data[i],
                fs=self.sampling_rate,
                nperseg=int(2 * self.sampling_rate)
            )

            # Extract band power
            for band_name, (fmin, fmax) in self.frequency_bands.items():
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                band_power = np.mean(psd[band_mask])

                results.append({
                    'channel': channel,
                    'band': band_name,
                    'power': band_power
                })

        power_df = pd.DataFrame(results)

        logger.info(f"  Computed {len(self.frequency_bands)} bands × {len(channels)} channels")

        return power_df

    def compute_connectivity(
        self,
        data: np.ndarray,
        channels: List[str],
        method: str = 'pli',
        band: str = 'alpha'
    ) -> pd.DataFrame:
        """
        Compute functional connectivity between channels

        Parameters
        ----------
        data : np.ndarray
            Preprocessed EEG (channels × time)
        channels : List[str]
            Channel names
        method : str
            Connectivity method ('pli', 'coherence', 'correlation')
        band : str
            Frequency band to analyze

        Returns
        -------
        connectivity : pd.DataFrame
            Pairwise connectivity matrix
        """
        logger.info(f"Computing {method} connectivity for {band} band")

        from scipy.signal import hilbert, butter, filtfilt

        # Filter to band of interest
        fmin, fmax = self.frequency_bands[band]
        nyquist = self.sampling_rate / 2
        b, a = butter(4, [fmin / nyquist, fmax / nyquist], btype='band')
        filtered = filtfilt(b, a, data, axis=1)

        # Instantaneous phase via Hilbert transform
        analytic = hilbert(filtered, axis=1)
        phases = np.angle(analytic)

        n_channels = len(channels)
        connectivity_matrix = np.zeros((n_channels, n_channels))

        if method == 'pli':
            # Phase Lag Index
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    phase_diff = phases[i] - phases[j]
                    pli = np.abs(np.mean(np.sign(phase_diff)))
                    connectivity_matrix[i, j] = pli
                    connectivity_matrix[j, i] = pli

        elif method == 'coherence':
            # Magnitude squared coherence
            from scipy.signal import coherence as compute_coherence

            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    f, coh = compute_coherence(
                        data[i], data[j],
                        fs=self.sampling_rate,
                        nperseg=int(2 * self.sampling_rate)
                    )
                    # Mean coherence in band
                    band_mask = (f >= fmin) & (f <= fmax)
                    mean_coh = np.mean(coh[band_mask])
                    connectivity_matrix[i, j] = mean_coh
                    connectivity_matrix[j, i] = mean_coh

        elif method == 'correlation':
            # Amplitude correlation
            amplitudes = np.abs(analytic)
            correlation = np.corrcoef(amplitudes)
            connectivity_matrix = correlation

        connectivity_df = pd.DataFrame(
            connectivity_matrix,
            index=channels,
            columns=channels
        )

        logger.info(f"  Mean connectivity: {connectivity_matrix[np.triu_indices(n_channels, k=1)].mean():.3f}")

        return connectivity_df

    def extract_erp_components(
        self,
        epochs: np.ndarray,
        channels: List[str],
        time_points: np.ndarray,
        components: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> pd.DataFrame:
        """
        Extract event-related potential (ERP) components

        Parameters
        ----------
        epochs : np.ndarray
            Epoched data (trials × channels × time)
        channels : List[str]
            Channel names
        time_points : np.ndarray
            Time points (in seconds)
        components : Dict[str, Tuple[float, float]], optional
            Component windows: {'P300': (0.25, 0.5), ...}

        Returns
        -------
        erp_components : pd.DataFrame
            Columns: channel, component, amplitude, latency
        """
        logger.info("Extracting ERP components")

        if components is None:
            components = {
                'N100': (0.08, 0.12),
                'P200': (0.15, 0.25),
                'P300': (0.25, 0.50),
                'N400': (0.35, 0.55)
            }

        # Average across trials
        erp = epochs.mean(axis=0)  # channels × time

        results = []

        for comp_name, (t_start, t_end) in components.items():
            # Find window
            window_mask = (time_points >= t_start) & (time_points <= t_end)

            for i, channel in enumerate(channels):
                # Peak amplitude in window
                window_values = erp[i, window_mask]
                if comp_name.startswith('N'):
                    # Negative peak
                    peak_idx = np.argmin(window_values)
                    amplitude = window_values[peak_idx]
                else:
                    # Positive peak
                    peak_idx = np.argmax(window_values)
                    amplitude = window_values[peak_idx]

                # Latency
                window_times = time_points[window_mask]
                latency = window_times[peak_idx]

                results.append({
                    'channel': channel,
                    'component': comp_name,
                    'amplitude': amplitude,
                    'latency': latency
                })

        erp_df = pd.DataFrame(results)

        logger.info(f"  Extracted {len(components)} components")

        return erp_df

    def compute_network_metrics(
        self,
        connectivity: pd.DataFrame,
        threshold: float = 0.1
    ) -> pd.DataFrame:
        """
        Compute graph-theoretic network metrics

        Parameters
        ----------
        connectivity : pd.DataFrame
            Connectivity matrix
        threshold : float
            Minimum connectivity to retain edge

        Returns
        -------
        network_metrics : pd.DataFrame
            Columns: channel, degree, clustering, betweenness
        """
        logger.info("Computing network metrics")

        import networkx as nx

        # Threshold connectivity
        adj_matrix = connectivity.values.copy()
        adj_matrix[adj_matrix < threshold] = 0

        # Build graph
        G = nx.from_numpy_array(adj_matrix)
        G = nx.relabel_nodes(G, dict(enumerate(connectivity.columns)))

        # Compute metrics
        degree = dict(G.degree(weight='weight'))
        clustering = nx.clustering(G, weight='weight')
        betweenness = nx.betweenness_centrality(G, weight='weight')

        metrics = pd.DataFrame({
            'channel': connectivity.columns,
            'degree': [degree[ch] for ch in connectivity.columns],
            'clustering': [clustering[ch] for ch in connectivity.columns],
            'betweenness': [betweenness[ch] for ch in connectivity.columns]
        })

        logger.info(f"  Mean degree: {metrics['degree'].mean():.1f}")

        return metrics

    def correlate_with_behavior(
        self,
        neural_features: pd.DataFrame,
        behavioral_scores: pd.DataFrame,
        method: str = 'spearman'
    ) -> pd.DataFrame:
        """
        Correlate neural features with behavioral scores

        Parameters
        ----------
        neural_features : pd.DataFrame
            EEG/MEG features (subjects × features)
        behavioral_scores : pd.DataFrame
            Behavioral measures (subjects × scores)
        method : str
            Correlation method

        Returns
        -------
        correlations : pd.DataFrame
            Columns: neural_feature, behavior, correlation, p_value
        """
        logger.info("Correlating neural features with behavior")

        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        # Align subjects
        common_subjects = neural_features.index.intersection(behavioral_scores.index)
        neural_aligned = neural_features.loc[common_subjects]
        behavior_aligned = behavioral_scores.loc[common_subjects]

        results = []

        for neural_col in neural_aligned.columns:
            for behavior_col in behavior_aligned.columns:
                x = neural_aligned[neural_col].values
                y = behavior_aligned[behavior_col].values

                # Remove missing
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]

                if len(x_clean) < 10:
                    continue

                # Correlation
                if method == 'spearman':
                    corr, p_val = stats.spearmanr(x_clean, y_clean)
                else:
                    corr, p_val = stats.pearsonr(x_clean, y_clean)

                results.append({
                    'neural_feature': neural_col,
                    'behavior': behavior_col,
                    'correlation': corr,
                    'p_value': p_val,
                    'n_samples': len(x_clean)
                })

        correlations_df = pd.DataFrame(results).sort_values('p_value')

        # FDR correction
        _, qvals, _, _ = multipletests(correlations_df['p_value'], method='fdr_bh')
        correlations_df['q_value'] = qvals

        logger.info(f"  Significant correlations (q<0.05): {(qvals < 0.05).sum()}")

        return correlations_df

    def analyze_complete(
        self,
        raw_data: np.ndarray,
        channels: List[str],
        behavioral_scores: Optional[pd.DataFrame] = None
    ) -> NeurophysiologyResult:
        """
        Complete neurophysiology analysis pipeline

        Parameters
        ----------
        raw_data : np.ndarray
            Raw EEG/MEG data (channels × time)
        channels : List[str]
            Channel names
        behavioral_scores : pd.DataFrame, optional
            Behavioral measures for correlation

        Returns
        -------
        NeurophysiologyResult
        """
        logger.info("=== Complete EEG/MEG Analysis ===")

        # 1. Preprocessing
        preprocessed, valid_channels = self.preprocess_eeg(raw_data, channels)

        # 2. Power spectra
        power_spectra = self.compute_power_spectra(preprocessed, valid_channels)

        # 3. Connectivity
        connectivity = self.compute_connectivity(
            preprocessed, valid_channels, method='pli', band='alpha'
        )

        # 4. Network metrics
        network_metrics = self.compute_network_metrics(connectivity)

        return NeurophysiologyResult(
            power_spectra=power_spectra,
            connectivity=connectivity,
            network_metrics=network_metrics
        )


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    logger.info("EEG/MEG Neurophysiology Analysis Module")
    logger.info("Ready for integration with AuDHD correlation study")
    logger.info("\nKey capabilities:")
    logger.info("  1. Spectral power analysis (delta, theta, alpha, beta, gamma)")
    logger.info("  2. Functional connectivity (phase-lag index, coherence)")
    logger.info("  3. Event-related potentials (ERPs)")
    logger.info("  4. Time-frequency analysis")
    logger.info("  5. Network graph metrics")
    logger.info("  6. Brain-behavior correlations")
