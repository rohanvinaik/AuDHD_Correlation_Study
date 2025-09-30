#!/usr/bin/env python3
"""
Sensory Processing Detailed Assessment Pipeline

Comprehensive sensory testing across modalities:
- Auditory processing (audiometry, OAE, ABR, temporal)
- Visual processing (contrast, motion, search, detection)
- Tactile/somatosensory processing (discrimination, threshold, proprioception)
- Multisensory integration (McGurk, sound-induced flash, temporal binding)
- Sensory gating (P50 suppression)
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from typing import Dict, List, Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditoryProcessor:
    """Process auditory/hearing assessments"""

    def __init__(self):
        """Initialize auditory processor"""

        # Pure tone audiometry frequencies (Hz)
        self.pta_frequencies = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]

        # Normal hearing thresholds (dB HL)
        self.normal_threshold = 25
        self.mild_loss = 40
        self.moderate_loss = 55

    def analyze_pure_tone_audiometry(self,
                                    thresholds: Dict[str, Dict[int, float]]) -> Dict:
        """
        Analyze pure tone audiometry results

        Args:
            thresholds: Dict with 'left' and 'right' ear thresholds
                       Each ear maps frequency (Hz) to threshold (dB HL)

        Returns:
            Dict with hearing metrics
        """
        results = {}

        for ear in ['left', 'right']:
            if ear not in thresholds:
                continue

            ear_thresholds = np.array([thresholds[ear].get(f, np.nan)
                                      for f in self.pta_frequencies])

            # Pure tone average (500, 1000, 2000 Hz)
            pta_freqs = [500, 1000, 2000]
            pta_values = [thresholds[ear].get(f, np.nan) for f in pta_freqs]
            pta = np.nanmean(pta_values)

            # High frequency average (3000, 4000, 6000 Hz)
            hf_freqs = [3000, 4000, 6000]
            hf_values = [thresholds[ear].get(f, np.nan) for f in hf_freqs]
            hf_avg = np.nanmean(hf_values)

            # Hearing loss category
            if pta <= self.normal_threshold:
                category = 'normal'
            elif pta <= self.mild_loss:
                category = 'mild_loss'
            elif pta <= self.moderate_loss:
                category = 'moderate_loss'
            else:
                category = 'severe_loss'

            results[f'{ear}_pta'] = pta
            results[f'{ear}_hf_average'] = hf_avg
            results[f'{ear}_category'] = category
            results[f'{ear}_threshold_variability'] = np.nanstd(ear_thresholds)

        # Asymmetry between ears
        if 'left' in thresholds and 'right' in thresholds:
            results['binaural_asymmetry'] = abs(results['left_pta'] - results['right_pta'])
            results['asymmetry_significant'] = results['binaural_asymmetry'] > 15  # dB

        return results

    def analyze_otoacoustic_emissions(self,
                                     dpoae_data: pd.DataFrame) -> Dict:
        """
        Analyze distortion product otoacoustic emissions (DPOAE)

        Args:
            dpoae_data: DataFrame with columns ['frequency', 'amplitude', 'noise_floor', 'ear']

        Returns:
            Dict with OAE metrics
        """
        results = {}

        for ear in dpoae_data['ear'].unique():
            ear_data = dpoae_data[dpoae_data['ear'] == ear]

            # Signal-to-noise ratio
            snr = ear_data['amplitude'] - ear_data['noise_floor']

            # Pass criteria: SNR >= 6 dB
            passes = snr >= 6
            pass_rate = np.mean(passes)

            # Average response amplitude
            mean_amplitude = ear_data['amplitude'].mean()

            results[f'{ear}_oae_pass_rate'] = pass_rate
            results[f'{ear}_oae_mean_amplitude'] = mean_amplitude
            results[f'{ear}_oae_mean_snr'] = snr.mean()
            results[f'{ear}_oae_present'] = pass_rate >= 0.7  # 70% pass

        return results

    def analyze_auditory_brainstem_response(self,
                                           waveforms: np.ndarray,
                                           sample_rate: int = 10000) -> Dict:
        """
        Analyze auditory brainstem response (ABR)

        Args:
            waveforms: Array of shape (n_trials, n_samples)
            sample_rate: Sampling rate in Hz

        Returns:
            Dict with ABR metrics
        """
        # Average waveform across trials
        avg_waveform = np.mean(waveforms, axis=0)

        # Time vector (ms)
        time_ms = np.arange(len(avg_waveform)) / sample_rate * 1000

        # Identify Wave I, III, V peaks (typically between 1-8 ms)
        window_mask = (time_ms >= 1) & (time_ms <= 8)
        window_waveform = avg_waveform[window_mask]
        window_time = time_ms[window_mask]

        # Find peaks (simplified - real analysis more complex)
        peaks, properties = signal.find_peaks(window_waveform,
                                             prominence=0.1,
                                             distance=int(0.5 * sample_rate / 1000))

        results = {}

        if len(peaks) >= 3:
            # Wave I, III, V latencies (ms)
            results['wave_I_latency'] = window_time[peaks[0]]
            results['wave_III_latency'] = window_time[peaks[1]]
            results['wave_V_latency'] = window_time[peaks[2]]

            # Inter-peak latencies (ms)
            results['I_III_interval'] = results['wave_III_latency'] - results['wave_I_latency']
            results['III_V_interval'] = results['wave_V_latency'] - results['wave_III_latency']
            results['I_V_interval'] = results['wave_V_latency'] - results['wave_I_latency']

            # Wave amplitudes
            results['wave_I_amplitude'] = window_waveform[peaks[0]]
            results['wave_V_amplitude'] = window_waveform[peaks[2]]
            results['V_I_amplitude_ratio'] = results['wave_V_amplitude'] / results['wave_I_amplitude']
        else:
            logger.warning(f"Found only {len(peaks)} ABR peaks, expected at least 3")
            results['abr_quality'] = 'poor'

        return results

    def measure_gap_detection_threshold(self,
                                       gap_durations: np.ndarray,
                                       responses: np.ndarray) -> Dict:
        """
        Measure gap detection threshold (temporal processing)

        Args:
            gap_durations: Array of gap durations tested (ms)
            responses: Array of correct detections (0 or 1)

        Returns:
            Dict with gap detection metrics
        """
        # Fit psychometric curve
        # Threshold = 75% correct detection

        # Sort by gap duration
        sorted_idx = np.argsort(gap_durations)
        gaps = gap_durations[sorted_idx]
        correct = responses[sorted_idx]

        # Calculate proportion correct for each gap duration
        unique_gaps = np.unique(gaps)
        prop_correct = np.array([np.mean(correct[gaps == g]) for g in unique_gaps])

        # Find threshold (interpolate to 75% correct)
        if np.any(prop_correct >= 0.75):
            threshold_idx = np.where(prop_correct >= 0.75)[0][0]
            if threshold_idx > 0:
                # Linear interpolation
                x1, x2 = unique_gaps[threshold_idx-1], unique_gaps[threshold_idx]
                y1, y2 = prop_correct[threshold_idx-1], prop_correct[threshold_idx]
                threshold = x1 + (0.75 - y1) * (x2 - x1) / (y2 - y1)
            else:
                threshold = unique_gaps[threshold_idx]
        else:
            threshold = np.nan

        results = {
            'gap_detection_threshold_ms': threshold,
            'max_gap_tested_ms': gaps.max(),
            'overall_accuracy': np.mean(correct),
            'temporal_processing_category': (
                'normal' if threshold <= 3 else
                'impaired' if threshold <= 10 else
                'severely_impaired'
            ) if not np.isnan(threshold) else 'undetermined'
        }

        return results

    def analyze_temporal_processing(self,
                                   data: Dict[str, Union[float, np.ndarray]]) -> Dict:
        """
        Comprehensive temporal processing analysis

        Args:
            data: Dict with temporal processing test results
                - 'gap_detection_threshold'
                - 'temporal_order_threshold'
                - 'duration_discrimination_threshold'

        Returns:
            Temporal processing profile
        """
        results = {}

        if 'gap_detection_threshold' in data:
            results['gap_detection_ms'] = data['gap_detection_threshold']

        if 'temporal_order_threshold' in data:
            # Temporal order judgment threshold (ms)
            results['temporal_order_ms'] = data['temporal_order_threshold']

        if 'duration_discrimination_threshold' in data:
            # Duration discrimination (Weber fraction)
            results['duration_discrimination_weber'] = data['duration_discrimination_threshold']

        # Overall temporal processing score
        temporal_scores = []
        if 'gap_detection_ms' in results:
            # Lower is better; normalize to 0-1 scale
            temporal_scores.append(1 - min(results['gap_detection_ms'] / 10, 1))

        if temporal_scores:
            results['temporal_processing_composite'] = np.mean(temporal_scores)

        return results


class VisualProcessor:
    """Process visual processing assessments"""

    def __init__(self):
        """Initialize visual processor"""
        self.csf_frequencies = [0.5, 1, 2, 4, 8, 16]  # cycles per degree

    def analyze_contrast_sensitivity(self,
                                    thresholds: Dict[float, float]) -> Dict:
        """
        Analyze contrast sensitivity function (CSF)

        Args:
            thresholds: Dict mapping spatial frequency (cpd) to contrast threshold

        Returns:
            Dict with CSF metrics
        """
        frequencies = np.array(list(thresholds.keys()))
        contrasts = np.array(list(thresholds.values()))

        # Convert to sensitivity (1/threshold)
        sensitivities = 1 / contrasts

        # Peak sensitivity
        peak_sensitivity = sensitivities.max()
        peak_frequency = frequencies[sensitivities.argmax()]

        # Area under log CSF curve (AULCSF)
        log_freq = np.log10(frequencies)
        log_sens = np.log10(sensitivities)
        aulcsf = np.trapz(log_sens, log_freq)

        results = {
            'peak_contrast_sensitivity': peak_sensitivity,
            'peak_spatial_frequency': peak_frequency,
            'aulcsf': aulcsf,
            'low_freq_sensitivity': sensitivities[frequencies <= 2].mean(),
            'high_freq_sensitivity': sensitivities[frequencies >= 8].mean(),
        }

        return results

    def analyze_motion_coherence(self,
                                 coherence_levels: np.ndarray,
                                 responses: np.ndarray) -> Dict:
        """
        Analyze motion coherence threshold

        Args:
            coherence_levels: Array of motion coherence levels (0-1)
            responses: Array of correct responses (0 or 1)

        Returns:
            Dict with motion processing metrics
        """
        # Sort by coherence
        sorted_idx = np.argsort(coherence_levels)
        coherence = coherence_levels[sorted_idx]
        correct = responses[sorted_idx]

        # Calculate proportion correct for each coherence level
        unique_coherence = np.unique(coherence)
        prop_correct = np.array([np.mean(correct[coherence == c])
                                for c in unique_coherence])

        # Find threshold (82% correct for 2AFC task)
        threshold_target = 0.82
        if np.any(prop_correct >= threshold_target):
            threshold_idx = np.where(prop_correct >= threshold_target)[0][0]
            if threshold_idx > 0:
                # Interpolate
                x1, x2 = unique_coherence[threshold_idx-1], unique_coherence[threshold_idx]
                y1, y2 = prop_correct[threshold_idx-1], prop_correct[threshold_idx]
                threshold = x1 + (threshold_target - y1) * (x2 - x1) / (y2 - y1)
            else:
                threshold = unique_coherence[threshold_idx]
        else:
            threshold = np.nan

        results = {
            'motion_coherence_threshold': threshold,
            'motion_perception_category': (
                'normal' if threshold <= 0.15 else
                'impaired' if threshold <= 0.40 else
                'severely_impaired'
            ) if not np.isnan(threshold) else 'undetermined'
        }

        return results

    def analyze_visual_search(self,
                             trials: pd.DataFrame) -> Dict:
        """
        Analyze visual search efficiency

        Args:
            trials: DataFrame with columns ['set_size', 'target_present', 'rt', 'correct']

        Returns:
            Dict with visual search metrics
        """
        # Separate target present/absent trials
        present = trials[trials['target_present'] == True]
        absent = trials[trials['target_present'] == False]

        # Calculate search slopes (ms per item)
        # Slope = change in RT / change in set size

        def calculate_slope(data):
            if len(data) < 2:
                return np.nan
            # Linear regression: RT ~ set_size
            slope, intercept, r_value, _, _ = stats.linregress(
                data['set_size'], data.groupby('set_size')['rt'].mean()
            )
            return slope

        present_slope = calculate_slope(present)
        absent_slope = calculate_slope(absent)

        # Search efficiency categories (ms/item)
        # < 10 ms/item: efficient (parallel) search
        # > 20 ms/item: inefficient (serial) search

        results = {
            'target_present_slope': present_slope,
            'target_absent_slope': absent_slope,
            'search_asymmetry': absent_slope / present_slope if present_slope != 0 else np.nan,
            'search_efficiency': (
                'efficient' if present_slope < 10 else
                'moderately_efficient' if present_slope < 20 else
                'inefficient'
            ),
            'mean_rt': trials['rt'].mean(),
            'accuracy': trials['correct'].mean()
        }

        return results

    def analyze_change_detection(self,
                                 trials: pd.DataFrame) -> Dict:
        """
        Analyze change detection / visual working memory

        Args:
            trials: DataFrame with columns ['set_size', 'correct']

        Returns:
            Dict with change detection metrics
        """
        # Calculate capacity using Pashler's formula
        # K = set_size * (hit_rate - false_alarm_rate)

        results = {}
        capacities = []

        for set_size in trials['set_size'].unique():
            size_data = trials[trials['set_size'] == set_size]
            accuracy = size_data['correct'].mean()

            # Estimate capacity (simplified - assumes 50% target present)
            # More accurate would separate change/no-change trials
            k = set_size * (2 * accuracy - 1)
            k = max(0, k)  # Capacity can't be negative

            capacities.append(k)
            results[f'capacity_set{int(set_size)}'] = k

        # Overall capacity estimate (typically plateaus at true capacity)
        results['visual_working_memory_k'] = np.max(capacities)
        results['capacity_category'] = (
            'high' if results['visual_working_memory_k'] >= 4 else
            'average' if results['visual_working_memory_k'] >= 3 else
            'low'
        )

        return results


class TactileProcessor:
    """Process tactile/somatosensory assessments"""

    def __init__(self):
        """Initialize tactile processor"""
        pass

    def analyze_two_point_discrimination(self,
                                        body_site: str,
                                        threshold_mm: float) -> Dict:
        """
        Analyze two-point discrimination threshold

        Args:
            body_site: Body location tested (e.g., 'fingertip', 'palm', 'forearm')
            threshold_mm: Distance threshold in mm

        Returns:
            Dict with tactile discrimination metrics
        """
        # Normal two-point discrimination thresholds (mm)
        norms = {
            'fingertip': 2.5,
            'palm': 8,
            'forearm': 40,
            'back': 50,
            'foot': 15
        }

        norm = norms.get(body_site.lower(), 10)

        results = {
            f'two_point_{body_site}_threshold_mm': threshold_mm,
            f'two_point_{body_site}_percentile': (
                'normal' if threshold_mm <= norm * 1.5 else
                'impaired' if threshold_mm <= norm * 2.5 else
                'severely_impaired'
            )
        }

        return results

    def analyze_vibrotactile_threshold(self,
                                      frequencies: np.ndarray,
                                      thresholds: np.ndarray) -> Dict:
        """
        Analyze vibrotactile detection thresholds

        Args:
            frequencies: Array of vibration frequencies (Hz)
            thresholds: Array of detection thresholds (dB re: 1 μm)

        Returns:
            Dict with vibrotactile sensitivity metrics
        """
        # Pacinian channel: sensitive to ~250 Hz
        # Meissner channel: sensitive to ~30 Hz

        pacinian_mask = (frequencies >= 200) & (frequencies <= 300)
        meissner_mask = (frequencies >= 20) & (frequencies <= 50)

        results = {
            'pacinian_threshold_db': thresholds[pacinian_mask].mean() if pacinian_mask.any() else np.nan,
            'meissner_threshold_db': thresholds[meissner_mask].mean() if meissner_mask.any() else np.nan,
            'overall_sensitivity': -thresholds.mean(),  # Lower threshold = higher sensitivity
        }

        return results

    def analyze_texture_discrimination(self,
                                      trials: pd.DataFrame) -> Dict:
        """
        Analyze texture discrimination

        Args:
            trials: DataFrame with columns ['grating_width', 'correct']

        Returns:
            Dict with texture discrimination metrics
        """
        # Calculate threshold grating width for discrimination

        widths = trials.groupby('grating_width')['correct'].mean()
        threshold_accuracy = 0.75

        if np.any(widths >= threshold_accuracy):
            threshold = widths[widths >= threshold_accuracy].index[0]
        else:
            threshold = np.nan

        results = {
            'texture_discrimination_threshold_mm': threshold,
            'texture_discrimination_accuracy': trials['correct'].mean(),
        }

        return results

    def analyze_proprioception(self,
                              trials: pd.DataFrame) -> Dict:
        """
        Analyze proprioceptive accuracy

        Args:
            trials: DataFrame with columns ['target_angle', 'response_angle']

        Returns:
            Dict with proprioception metrics
        """
        # Calculate angular error
        angular_errors = np.abs(trials['target_angle'] - trials['response_angle'])

        # Handle wrap-around for circular angles
        angular_errors = np.minimum(angular_errors, 360 - angular_errors)

        results = {
            'proprioception_mean_error_deg': angular_errors.mean(),
            'proprioception_sd_deg': angular_errors.std(),
            'proprioception_accuracy': (
                'high' if angular_errors.mean() <= 5 else
                'moderate' if angular_errors.mean() <= 10 else
                'low'
            )
        }

        return results


class MultisensoryProcessor:
    """Process multisensory integration assessments"""

    def __init__(self):
        """Initialize multisensory processor"""
        pass

    def analyze_mcgurk_effect(self,
                             trials: pd.DataFrame) -> Dict:
        """
        Analyze McGurk effect (audiovisual speech integration)

        Args:
            trials: DataFrame with columns ['condition', 'response']
                   Conditions: 'congruent', 'mcgurk', 'visual_only', 'auditory_only'

        Returns:
            Dict with McGurk effect metrics
        """
        # McGurk effect: visual /ga/ + auditory /ba/ -> perceived /da/

        results = {}

        for condition in trials['condition'].unique():
            condition_data = trials[trials['condition'] == condition]

            if condition == 'mcgurk':
                # Fusion response indicates integration
                fusion_responses = condition_data[condition_data['response'] == 'fusion']
                results['mcgurk_fusion_rate'] = len(fusion_responses) / len(condition_data)

                # Higher fusion = stronger audiovisual integration
                results['mcgurk_susceptibility'] = (
                    'high' if results['mcgurk_fusion_rate'] >= 0.7 else
                    'moderate' if results['mcgurk_fusion_rate'] >= 0.4 else
                    'low'
                )

            elif condition == 'congruent':
                # Baseline accuracy
                results['congruent_accuracy'] = condition_data['response'].mean()

        return results

    def analyze_sound_induced_flash(self,
                                   trials: pd.DataFrame) -> Dict:
        """
        Analyze sound-induced flash illusion

        Args:
            trials: DataFrame with columns ['n_flashes', 'n_beeps', 'perceived_flashes']

        Returns:
            Dict with illusion metrics
        """
        # Classic: 1 flash + 2 beeps -> perceived as 2 flashes

        illusion_trials = trials[(trials['n_flashes'] == 1) & (trials['n_beeps'] == 2)]

        if len(illusion_trials) > 0:
            illusion_rate = np.mean(illusion_trials['perceived_flashes'] > 1)
        else:
            illusion_rate = np.nan

        # Control: 1 flash + 1 beep (should be veridical)
        control_trials = trials[(trials['n_flashes'] == 1) & (trials['n_beeps'] == 1)]
        if len(control_trials) > 0:
            control_accuracy = np.mean(control_trials['perceived_flashes'] == 1)
        else:
            control_accuracy = np.nan

        results = {
            'sound_induced_flash_rate': illusion_rate,
            'audiovisual_integration_strength': (
                'strong' if illusion_rate >= 0.6 else
                'moderate' if illusion_rate >= 0.3 else
                'weak'
            ),
            'baseline_accuracy': control_accuracy
        }

        return results

    def analyze_temporal_binding_window(self,
                                       trials: pd.DataFrame) -> Dict:
        """
        Analyze temporal binding window (multisensory integration)

        Args:
            trials: DataFrame with columns ['soa_ms', 'perceived_synchronous']
                   soa_ms: Stimulus onset asynchrony (negative = auditory first)

        Returns:
            Dict with temporal binding window metrics
        """
        # Calculate proportion "synchronous" responses at each SOA

        soas = trials['soa_ms'].unique()
        soas.sort()

        sync_rates = np.array([trials[trials['soa_ms'] == soa]['perceived_synchronous'].mean()
                              for soa in soas])

        # Find temporal binding window (range where >75% perceived as synchronous)
        threshold = 0.75
        synchronous_mask = sync_rates >= threshold

        if synchronous_mask.any():
            tbw_min = soas[synchronous_mask].min()
            tbw_max = soas[synchronous_mask].max()
            tbw_width = tbw_max - tbw_min
        else:
            tbw_min = tbw_max = tbw_width = np.nan

        # Point of subjective simultaneity (PSS) - 50% synchronous
        if np.any(sync_rates >= 0.5):
            idx = np.where(sync_rates >= 0.5)[0][0]
            if idx > 0:
                # Interpolate
                pss = soas[idx-1] + (0.5 - sync_rates[idx-1]) * (soas[idx] - soas[idx-1]) / (sync_rates[idx] - sync_rates[idx-1])
            else:
                pss = soas[idx]
        else:
            pss = np.nan

        results = {
            'temporal_binding_window_ms': tbw_width,
            'tbw_lower_bound': tbw_min,
            'tbw_upper_bound': tbw_max,
            'point_subjective_simultaneity': pss,
            'tbw_category': (
                'narrow' if tbw_width <= 100 else
                'typical' if tbw_width <= 250 else
                'wide'
            ) if not np.isnan(tbw_width) else 'undetermined'
        }

        return results


class SensoryGatingProcessor:
    """Process sensory gating (P50 suppression) assessments"""

    def __init__(self):
        """Initialize sensory gating processor"""
        pass

    def analyze_p50_suppression(self,
                               s1_waveforms: np.ndarray,
                               s2_waveforms: np.ndarray,
                               sample_rate: int = 1000) -> Dict:
        """
        Analyze P50 sensory gating (paired-click paradigm)

        Args:
            s1_waveforms: First stimulus EEG waveforms (n_trials, n_samples)
            s2_waveforms: Second stimulus EEG waveforms (n_trials, n_samples)
            sample_rate: Sampling rate in Hz

        Returns:
            Dict with sensory gating metrics
        """
        # Average waveforms across trials
        s1_avg = np.mean(s1_waveforms, axis=0)
        s2_avg = np.mean(s2_waveforms, axis=0)

        # Time vector (ms)
        time_ms = np.arange(len(s1_avg)) / sample_rate * 1000

        # P50 occurs 40-80 ms post-stimulus
        p50_window = (time_ms >= 40) & (time_ms <= 80)

        # Find P50 peak amplitude (most positive peak in window)
        s1_p50_amp = s1_avg[p50_window].max()
        s1_p50_latency = time_ms[p50_window][s1_avg[p50_window].argmax()]

        s2_p50_amp = s2_avg[p50_window].max()
        s2_p50_latency = time_ms[p50_window][s2_avg[p50_window].argmax()]

        # Gating ratio: S2/S1 (lower = better gating)
        gating_ratio = s2_p50_amp / s1_p50_amp if s1_p50_amp != 0 else np.nan

        # Gating difference: S1 - S2 (higher = better gating)
        gating_difference = s1_p50_amp - s2_p50_amp

        results = {
            's1_p50_amplitude': s1_p50_amp,
            's1_p50_latency': s1_p50_latency,
            's2_p50_amplitude': s2_p50_amp,
            's2_p50_latency': s2_p50_latency,
            'p50_gating_ratio': gating_ratio,
            'p50_gating_difference': gating_difference,
            'gating_status': (
                'normal' if gating_ratio <= 0.5 else
                'impaired' if gating_ratio <= 0.8 else
                'deficient'
            ) if not np.isnan(gating_ratio) else 'undetermined'
        }

        return results


class SensoryProcessor:
    """Comprehensive sensory processing assessment"""

    def __init__(self):
        """Initialize sensory processor"""
        self.auditory = AuditoryProcessor()
        self.visual = VisualProcessor()
        self.tactile = TactileProcessor()
        self.multisensory = MultisensoryProcessor()
        self.gating = SensoryGatingProcessor()

    def process_sensory_battery(self, data: Dict) -> Dict:
        """
        Complete sensory assessment battery

        Args:
            data: Dict with keys for different sensory tests:
                - 'audiometry': pure tone thresholds
                - 'oae': otoacoustic emissions data
                - 'abr': auditory brainstem response waveforms
                - 'gap_detection': gap detection data
                - 'contrast_sensitivity': CSF thresholds
                - 'motion_coherence': motion perception data
                - 'visual_search': visual search trials
                - 'two_point_discrimination': tactile discrimination
                - 'mcgurk': McGurk effect trials
                - 'temporal_binding': temporal binding window data
                - 'p50': P50 suppression waveforms

        Returns:
            Complete sensory profile
        """
        results = {}

        # Auditory processing
        if 'audiometry' in data:
            aud_results = self.auditory.analyze_pure_tone_audiometry(data['audiometry'])
            results.update(aud_results)

        if 'oae' in data:
            oae_results = self.auditory.analyze_otoacoustic_emissions(data['oae'])
            results.update(oae_results)

        if 'abr' in data:
            abr_results = self.auditory.analyze_auditory_brainstem_response(
                data['abr']['waveforms'],
                data['abr'].get('sample_rate', 10000)
            )
            results.update(abr_results)

        if 'gap_detection' in data:
            gap_results = self.auditory.measure_gap_detection_threshold(
                data['gap_detection']['durations'],
                data['gap_detection']['responses']
            )
            results.update(gap_results)

        # Visual processing
        if 'contrast_sensitivity' in data:
            csf_results = self.visual.analyze_contrast_sensitivity(data['contrast_sensitivity'])
            results.update(csf_results)

        if 'motion_coherence' in data:
            motion_results = self.visual.analyze_motion_coherence(
                data['motion_coherence']['coherence_levels'],
                data['motion_coherence']['responses']
            )
            results.update(motion_results)

        if 'visual_search' in data:
            search_results = self.visual.analyze_visual_search(data['visual_search'])
            results.update(search_results)

        # Tactile processing
        if 'two_point_discrimination' in data:
            for site, threshold in data['two_point_discrimination'].items():
                tpd_results = self.tactile.analyze_two_point_discrimination(site, threshold)
                results.update(tpd_results)

        # Multisensory integration
        if 'mcgurk' in data:
            mcgurk_results = self.multisensory.analyze_mcgurk_effect(data['mcgurk'])
            results.update(mcgurk_results)

        if 'temporal_binding' in data:
            tbw_results = self.multisensory.analyze_temporal_binding_window(data['temporal_binding'])
            results.update(tbw_results)

        # Sensory gating
        if 'p50' in data:
            p50_results = self.gating.analyze_p50_suppression(
                data['p50']['s1_waveforms'],
                data['p50']['s2_waveforms'],
                data['p50'].get('sample_rate', 1000)
            )
            results.update(p50_results)

        return results


if __name__ == '__main__':
    # Example usage
    logger.info("Sensory Processor initialized")

    processor = SensoryProcessor()

    # Simulate pure tone audiometry
    audiometry_data = {
        'left': {250: 10, 500: 15, 1000: 20, 2000: 25, 3000: 30, 4000: 35, 6000: 40, 8000: 45},
        'right': {250: 10, 500: 15, 1000: 20, 2000: 25, 3000: 30, 4000: 35, 6000: 40, 8000: 45}
    }

    # Simulate gap detection
    np.random.seed(42)
    gap_durations = np.repeat([1, 2, 3, 5, 8, 12, 20], 10)
    gap_responses = (gap_durations >= 3).astype(int)  # Threshold at 3 ms
    gap_responses = gap_responses * (np.random.rand(len(gap_responses)) > 0.15)  # Add noise

    # Simulate P50 waveforms
    sample_rate = 1000
    time = np.arange(0, 0.2, 1/sample_rate)  # 200 ms

    # S1: strong P50 response at 50 ms
    s1_waveforms = np.random.randn(30, len(time)) * 2
    s1_waveforms[:, 45:55] += 10  # P50 peak

    # S2: suppressed P50 response
    s2_waveforms = np.random.randn(30, len(time)) * 2
    s2_waveforms[:, 45:55] += 4  # Reduced P50

    # Run sensory battery
    sensory_data = {
        'audiometry': audiometry_data,
        'gap_detection': {
            'durations': gap_durations,
            'responses': gap_responses
        },
        'p50': {
            's1_waveforms': s1_waveforms,
            's2_waveforms': s2_waveforms,
            'sample_rate': sample_rate
        }
    }

    results = processor.process_sensory_battery(sensory_data)

    print("\nSensory Processing Results:")
    print("="*60)
    print("\nAudiometry:")
    print(f"  Left PTA: {results.get('left_pta', 'N/A'):.1f} dB HL")
    print(f"  Right PTA: {results.get('right_pta', 'N/A'):.1f} dB HL")
    print(f"  Category: {results.get('left_category', 'N/A')}")

    print("\nGap Detection:")
    print(f"  Threshold: {results.get('gap_detection_threshold_ms', 'N/A'):.2f} ms")
    print(f"  Category: {results.get('temporal_processing_category', 'N/A')}")

    print("\nP50 Sensory Gating:")
    print(f"  S1 amplitude: {results.get('s1_p50_amplitude', 'N/A'):.2f} µV")
    print(f"  S2 amplitude: {results.get('s2_p50_amplitude', 'N/A'):.2f} µV")
    print(f"  Gating ratio: {results.get('p50_gating_ratio', 'N/A'):.3f}")
    print(f"  Gating status: {results.get('gating_status', 'N/A')}")
