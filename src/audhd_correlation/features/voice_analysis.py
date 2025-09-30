#!/usr/bin/env python3
"""
Voice & Speech Acoustics Analysis Pipeline

Comprehensive voice analysis using Praat (parselmouth) and librosa:
- Prosodic features (pitch, intensity, rhythm)
- Spectral features (formants, voice quality)
- Temporal features (VOT, duration, coarticulation)
- Pragmatic features (turn-taking, prosodic matching)
- MFCC features for machine learning
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from typing import Dict, List, Optional, Tuple, Union
import logging

# Audio processing libraries
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    logging.warning("parselmouth not installed. Install with: pip install praat-parselmouth")

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not installed. Install with: pip install librosa")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProsodicAnalyzer:
    """Analyze prosodic features (pitch, intensity, rhythm)"""

    def __init__(self):
        """Initialize prosodic analyzer"""
        self.f0_min = 75  # Hz (typical male range)
        self.f0_max = 300  # Hz (typical female range)

    def extract_pitch_features(self,
                               audio: Union[str, 'parselmouth.Sound'],
                               time_step: float = 0.01) -> Dict:
        """
        Extract pitch (F0) features using Praat

        Args:
            audio: Path to audio file or parselmouth Sound object
            time_step: Time step for analysis (seconds)

        Returns:
            Dict with pitch features
        """
        if not PARSELMOUTH_AVAILABLE:
            raise ImportError("parselmouth required for pitch analysis")

        # Load audio if path provided
        if isinstance(audio, str):
            sound = parselmouth.Sound(audio)
        else:
            sound = audio

        # Extract pitch using autocorrelation method
        pitch = sound.to_pitch(time_step=time_step,
                              pitch_floor=self.f0_min,
                              pitch_ceiling=self.f0_max)

        # Get F0 values (excluding unvoiced frames)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]  # Remove unvoiced frames

        if len(f0_values) == 0:
            return {'error': 'No voiced segments detected'}

        # Pitch statistics
        features = {
            'pitch_mean_hz': np.mean(f0_values),
            'pitch_sd_hz': np.std(f0_values),
            'pitch_min_hz': np.min(f0_values),
            'pitch_max_hz': np.max(f0_values),
            'pitch_range_hz': np.max(f0_values) - np.min(f0_values),
            'pitch_median_hz': np.median(f0_values),
            'pitch_iqr_hz': np.percentile(f0_values, 75) - np.percentile(f0_values, 25),
            'pitch_cv': np.std(f0_values) / np.mean(f0_values),  # Coefficient of variation
        }

        # Pitch contour slope (linear regression)
        time_points = np.arange(len(f0_values))
        slope, intercept, r_value, _, _ = stats.linregress(time_points, f0_values)
        features['pitch_slope_hz_per_frame'] = slope
        features['pitch_slope_r2'] = r_value ** 2

        # Semitone range (more perceptually relevant than Hz)
        f0_semitones = 12 * np.log2(f0_values / f0_values[0])
        features['pitch_range_semitones'] = f0_semitones.max() - f0_semitones.min()

        return features

    def extract_intensity_features(self,
                                   audio: Union[str, 'parselmouth.Sound'],
                                   time_step: float = 0.01) -> Dict:
        """
        Extract intensity features using Praat

        Args:
            audio: Path to audio file or parselmouth Sound object
            time_step: Time step for analysis (seconds)

        Returns:
            Dict with intensity features
        """
        if not PARSELMOUTH_AVAILABLE:
            raise ImportError("parselmouth required for intensity analysis")

        if isinstance(audio, str):
            sound = parselmouth.Sound(audio)
        else:
            sound = audio

        # Extract intensity
        intensity = sound.to_intensity(time_step=time_step)
        intensity_values = intensity.values[0]
        intensity_values = intensity_values[intensity_values > 0]

        if len(intensity_values) == 0:
            return {'error': 'No valid intensity values'}

        features = {
            'intensity_mean_db': np.mean(intensity_values),
            'intensity_sd_db': np.std(intensity_values),
            'intensity_min_db': np.min(intensity_values),
            'intensity_max_db': np.max(intensity_values),
            'intensity_range_db': np.max(intensity_values) - np.min(intensity_values),
        }

        return features

    def extract_rhythm_features(self,
                               audio: Union[str, 'parselmouth.Sound'],
                               syllable_nuclei: Optional[List[Tuple[float, float]]] = None) -> Dict:
        """
        Extract rhythm and timing features

        Args:
            audio: Path to audio file or parselmouth Sound object
            syllable_nuclei: Optional list of (start, end) times for syllables

        Returns:
            Dict with rhythm features
        """
        if not PARSELMOUTH_AVAILABLE:
            raise ImportError("parselmouth required for rhythm analysis")

        if isinstance(audio, str):
            sound = parselmouth.Sound(audio)
        else:
            sound = audio

        duration = sound.get_total_duration()

        # Detect syllable nuclei if not provided (simplified intensity-based method)
        if syllable_nuclei is None:
            intensity = sound.to_intensity()
            intensity_values = intensity.values[0]
            intensity_times = intensity.xs()

            # Find peaks in intensity (syllable nuclei)
            threshold = np.mean(intensity_values) + 0.5 * np.std(intensity_values)
            peaks, _ = signal.find_peaks(intensity_values,
                                        height=threshold,
                                        distance=int(0.1 / intensity.get_time_step()))  # Min 100ms apart

            syllable_times = [intensity_times[p] for p in peaks]
            n_syllables = len(syllable_times)
        else:
            syllable_times = [s[0] for s in syllable_nuclei]
            n_syllables = len(syllable_nuclei)

        if n_syllables == 0:
            return {'error': 'No syllables detected'}

        # Calculate speech rate metrics
        features = {
            'n_syllables': n_syllables,
            'duration_sec': duration,
            'speech_rate_syll_per_sec': n_syllables / duration,
        }

        # Inter-syllable intervals
        if n_syllables > 1:
            intervals = np.diff(syllable_times)
            features['mean_intersyllable_interval_sec'] = np.mean(intervals)
            features['sd_intersyllable_interval_sec'] = np.std(intervals)
            features['rhythm_variability'] = np.std(intervals) / np.mean(intervals)  # nPVI-like

        # Detect pauses (gaps in intensity > threshold)
        intensity = sound.to_intensity()
        intensity_values = intensity.values[0]
        silence_threshold = 40  # dB
        is_silent = intensity_values < silence_threshold

        # Find continuous silent regions
        silent_runs = np.diff(np.concatenate([[0], is_silent.astype(int), [0]]))
        pause_starts = np.where(silent_runs == 1)[0]
        pause_ends = np.where(silent_runs == -1)[0]
        pause_durations = (pause_ends - pause_starts) * intensity.get_time_step()

        # Only count pauses > 100ms
        pauses = pause_durations[pause_durations > 0.1]

        features['n_pauses'] = len(pauses)
        features['pause_frequency'] = len(pauses) / duration
        if len(pauses) > 0:
            features['mean_pause_duration_sec'] = np.mean(pauses)
            features['total_pause_time_sec'] = np.sum(pauses)
            features['pause_to_speech_ratio'] = np.sum(pauses) / (duration - np.sum(pauses))

            # Articulation rate (excluding pauses)
            speech_time = duration - np.sum(pauses)
            features['articulation_rate_syll_per_sec'] = n_syllables / speech_time if speech_time > 0 else 0
        else:
            features['articulation_rate_syll_per_sec'] = features['speech_rate_syll_per_sec']

        return features


class SpectralAnalyzer:
    """Analyze spectral features (formants, voice quality)"""

    def __init__(self):
        """Initialize spectral analyzer"""
        pass

    def extract_formant_features(self,
                                 audio: Union[str, 'parselmouth.Sound'],
                                 max_formants: int = 5,
                                 time_step: float = 0.01) -> Dict:
        """
        Extract formant features using Praat

        Args:
            audio: Path to audio file or parselmouth Sound object
            max_formants: Maximum number of formants to extract
            time_step: Time step for analysis (seconds)

        Returns:
            Dict with formant features
        """
        if not PARSELMOUTH_AVAILABLE:
            raise ImportError("parselmouth required for formant analysis")

        if isinstance(audio, str):
            sound = parselmouth.Sound(audio)
        else:
            sound = audio

        # Extract formants
        formants = sound.to_formant_burg(time_step=time_step,
                                        max_number_of_formants=max_formants)

        features = {}

        # Extract F1, F2, F3 statistics
        for formant_num in range(1, 4):  # F1, F2, F3
            formant_values = []

            for time in np.arange(0, sound.duration, time_step):
                f_value = formants.get_value_at_time(formant_num, time)
                if f_value is not None and not np.isnan(f_value):
                    formant_values.append(f_value)

            if len(formant_values) > 0:
                features[f'f{formant_num}_mean_hz'] = np.mean(formant_values)
                features[f'f{formant_num}_sd_hz'] = np.std(formant_values)
                features[f'f{formant_num}_range_hz'] = np.max(formant_values) - np.min(formant_values)

        # Formant dispersion (acoustic measure of vocal tract length)
        if 'f1_mean_hz' in features and 'f2_mean_hz' in features and 'f3_mean_hz' in features:
            # Average formant spacing
            features['formant_dispersion'] = (
                features['f1_mean_hz'] +
                features['f2_mean_hz'] +
                features['f3_mean_hz']
            ) / 3

            # F1-F2 distance (vowel space area proxy)
            features['f1_f2_distance'] = features['f2_mean_hz'] - features['f1_mean_hz']

        return features

    def extract_voice_quality_features(self,
                                       audio: Union[str, 'parselmouth.Sound']) -> Dict:
        """
        Extract voice quality features (jitter, shimmer, HNR, CPP)

        Args:
            audio: Path to audio file or parselmouth Sound object

        Returns:
            Dict with voice quality features
        """
        if not PARSELMOUTH_AVAILABLE:
            raise ImportError("parselmouth required for voice quality analysis")

        if isinstance(audio, str):
            sound = parselmouth.Sound(audio)
        else:
            sound = audio

        # Extract pitch object for jitter/shimmer
        pitch = sound.to_pitch()
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 300)

        features = {}

        # Jitter (pitch period variability)
        try:
            features['jitter_local'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter_rap'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter_ppq5'] = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        except Exception as e:
            logger.warning(f"Jitter calculation failed: {e}")
            features['jitter_local'] = np.nan

        # Shimmer (amplitude variability)
        try:
            features['shimmer_local'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_apq3'] = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_apq5'] = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception as e:
            logger.warning(f"Shimmer calculation failed: {e}")
            features['shimmer_local'] = np.nan

        # Harmonics-to-Noise Ratio (HNR)
        try:
            harmonicity = sound.to_harmonicity()
            features['hnr_db'] = call(harmonicity, "Get mean", 0, 0)
        except Exception as e:
            logger.warning(f"HNR calculation failed: {e}")
            features['hnr_db'] = np.nan

        # Cepstral Peak Prominence (CPP) - robust voice quality measure
        try:
            # Create power cepstrogram
            power_cepstrogram = sound.to_power_cepstrogram(60, 0.002, 5000, 50)
            features['cpp'] = call(power_cepstrogram, "Get CPPS", "yes", 0.01, 0.001, 60, 330, 0.05, "Parabolic", 0.001, 0.0, "Straight", "Robust")
        except Exception as e:
            logger.warning(f"CPP calculation failed: {e}")
            features['cpp'] = np.nan

        return features

    def extract_spectral_tilt_features(self,
                                       audio: Union[str, 'parselmouth.Sound']) -> Dict:
        """
        Extract spectral tilt features (H1-H2, H1-A1, H1-A3)

        Args:
            audio: Path to audio file or parselmouth Sound object

        Returns:
            Dict with spectral tilt features
        """
        if not PARSELMOUTH_AVAILABLE:
            raise ImportError("parselmouth required for spectral tilt analysis")

        if isinstance(audio, str):
            sound = parselmouth.Sound(audio)
        else:
            sound = audio

        features = {}

        # This is a simplified placeholder - full implementation requires
        # harmonic extraction and formant amplitude measurement
        # Typically done on vowels only

        try:
            # Get spectrum
            spectrum = sound.to_spectrum()

            # Get first two harmonics (simplified)
            pitch = sound.to_pitch()
            mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

            if not np.isnan(mean_f0) and mean_f0 > 0:
                # H1 amplitude (first harmonic = F0)
                h1_amp = call(spectrum, "Get real value in bin", mean_f0)

                # H2 amplitude (second harmonic = 2*F0)
                h2_amp = call(spectrum, "Get real value in bin", 2 * mean_f0)

                if h1_amp > 0 and h2_amp > 0:
                    features['h1_h2_db'] = 20 * np.log10(h1_amp / h2_amp)
                else:
                    features['h1_h2_db'] = np.nan
            else:
                features['h1_h2_db'] = np.nan

        except Exception as e:
            logger.warning(f"Spectral tilt calculation failed: {e}")
            features['h1_h2_db'] = np.nan

        return features


class TemporalAnalyzer:
    """Analyze temporal features (VOT, duration, coarticulation)"""

    def __init__(self):
        """Initialize temporal analyzer"""
        pass

    def measure_voice_onset_time(self,
                                 audio: Union[str, 'parselmouth.Sound'],
                                 burst_time: float,
                                 voicing_time: float) -> Dict:
        """
        Measure voice onset time (VOT)

        Args:
            audio: Path to audio file or parselmouth Sound object
            burst_time: Time of burst release (seconds)
            voicing_time: Time of voicing onset (seconds)

        Returns:
            Dict with VOT measures
        """
        vot_ms = (voicing_time - burst_time) * 1000

        features = {
            'vot_ms': vot_ms,
            'vot_category': (
                'prevoiced' if vot_ms < 0 else
                'short_lag' if vot_ms < 30 else
                'long_lag'
            )
        }

        return features

    def analyze_segment_durations(self,
                                  segments: pd.DataFrame) -> Dict:
        """
        Analyze phonetic segment durations

        Args:
            segments: DataFrame with columns ['segment_type', 'duration']
                     segment_type: 'vowel', 'consonant', 'stop', 'fricative', etc.

        Returns:
            Dict with duration statistics
        """
        features = {}

        for seg_type in segments['segment_type'].unique():
            type_durations = segments[segments['segment_type'] == seg_type]['duration']

            features[f'{seg_type}_mean_duration_ms'] = type_durations.mean() * 1000
            features[f'{seg_type}_sd_duration_ms'] = type_durations.std() * 1000

        # Vowel-to-consonant ratio
        if 'vowel' in segments['segment_type'].values and 'consonant' in segments['segment_type'].values:
            vowel_dur = segments[segments['segment_type'] == 'vowel']['duration'].sum()
            consonant_dur = segments[segments['segment_type'] == 'consonant']['duration'].sum()
            features['vowel_consonant_ratio'] = vowel_dur / consonant_dur if consonant_dur > 0 else np.nan

        return features

    def measure_coarticulation(self,
                              formant_trajectories: Dict[str, np.ndarray],
                              vowel_transitions: List[Tuple[str, str]]) -> Dict:
        """
        Measure coarticulation from formant transitions

        Args:
            formant_trajectories: Dict with 'F1', 'F2' arrays over time
            vowel_transitions: List of (vowel1, vowel2) transition pairs

        Returns:
            Dict with coarticulation metrics
        """
        features = {}

        # Calculate formant transition rate (Hz/sec)
        if 'F1' in formant_trajectories and len(formant_trajectories['F1']) > 1:
            f1_trajectory = formant_trajectories['F1']
            f1_rate = np.abs(np.diff(f1_trajectory)).mean()
            features['f1_transition_rate_hz_per_frame'] = f1_rate

        if 'F2' in formant_trajectories and len(formant_trajectories['F2']) > 1:
            f2_trajectory = formant_trajectories['F2']
            f2_rate = np.abs(np.diff(f2_trajectory)).mean()
            features['f2_transition_rate_hz_per_frame'] = f2_rate

            # F2 slope (coarticulation index)
            features['f2_slope'] = np.polyfit(np.arange(len(f2_trajectory)), f2_trajectory, 1)[0]

        return features


class PragmaticAnalyzer:
    """Analyze pragmatic features (turn-taking, prosodic matching)"""

    def __init__(self):
        """Initialize pragmatic analyzer"""
        pass

    def analyze_turn_taking(self,
                           speaker_turns: pd.DataFrame) -> Dict:
        """
        Analyze turn-taking patterns in conversation

        Args:
            speaker_turns: DataFrame with columns ['speaker', 'start_time', 'end_time']

        Returns:
            Dict with turn-taking metrics
        """
        # Calculate turn durations
        speaker_turns['duration'] = speaker_turns['end_time'] - speaker_turns['start_time']

        # Calculate gaps and overlaps between turns
        gaps_overlaps = []
        for i in range(len(speaker_turns) - 1):
            gap = speaker_turns.iloc[i+1]['start_time'] - speaker_turns.iloc[i]['end_time']
            gaps_overlaps.append(gap)

        gaps_overlaps = np.array(gaps_overlaps)

        features = {
            'mean_turn_duration_sec': speaker_turns['duration'].mean(),
            'sd_turn_duration_sec': speaker_turns['duration'].std(),
            'n_turns': len(speaker_turns),
            'mean_gap_sec': gaps_overlaps[gaps_overlaps >= 0].mean() if np.any(gaps_overlaps >= 0) else 0,
            'mean_overlap_sec': -gaps_overlaps[gaps_overlaps < 0].mean() if np.any(gaps_overlaps < 0) else 0,
            'n_overlaps': np.sum(gaps_overlaps < 0),
            'overlap_rate': np.sum(gaps_overlaps < 0) / len(gaps_overlaps) if len(gaps_overlaps) > 0 else 0,
        }

        # Response latency (time to respond after other speaker finishes)
        features['mean_response_latency_sec'] = features['mean_gap_sec']

        return features

    def analyze_prosodic_matching(self,
                                  speaker1_pitch: np.ndarray,
                                  speaker2_pitch: np.ndarray) -> Dict:
        """
        Analyze prosodic synchrony/matching between speakers

        Args:
            speaker1_pitch: F0 values for speaker 1
            speaker2_pitch: F0 values for speaker 2

        Returns:
            Dict with prosodic matching metrics
        """
        # Remove unvoiced frames
        s1_voiced = speaker1_pitch[speaker1_pitch > 0]
        s2_voiced = speaker2_pitch[speaker2_pitch > 0]

        if len(s1_voiced) == 0 or len(s2_voiced) == 0:
            return {'error': 'Insufficient voiced segments'}

        # Pitch synchrony (correlation)
        # Requires aligned time series - simplified here
        min_len = min(len(s1_voiced), len(s2_voiced))
        if min_len > 1:
            pitch_sync = stats.pearsonr(s1_voiced[:min_len], s2_voiced[:min_len])[0]
        else:
            pitch_sync = np.nan

        # Pitch convergence (difference in means)
        pitch_diff = abs(s1_voiced.mean() - s2_voiced.mean())

        features = {
            'pitch_synchrony': pitch_sync,
            'pitch_difference_hz': pitch_diff,
            'pitch_convergence_category': (
                'high' if pitch_diff < 20 else
                'moderate' if pitch_diff < 50 else
                'low'
            )
        }

        return features

    def analyze_emotional_prosody(self,
                                  audio: Union[str, 'parselmouth.Sound'],
                                  emotion_label: Optional[str] = None) -> Dict:
        """
        Analyze emotional prosody

        Args:
            audio: Path to audio file or parselmouth Sound object
            emotion_label: Optional emotion label for validation

        Returns:
            Dict with emotional prosody features
        """
        if not PARSELMOUTH_AVAILABLE:
            raise ImportError("parselmouth required for prosody analysis")

        if isinstance(audio, str):
            sound = parselmouth.Sound(audio)
        else:
            sound = audio

        # Extract prosodic features relevant to emotion
        pitch = sound.to_pitch()
        intensity = sound.to_intensity()

        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]

        intensity_values = intensity.values[0]
        intensity_values = intensity_values[intensity_values > 0]

        features = {
            'pitch_variability': np.std(f0_values) if len(f0_values) > 0 else np.nan,
            'pitch_range': np.ptp(f0_values) if len(f0_values) > 0 else np.nan,
            'intensity_variability': np.std(intensity_values) if len(intensity_values) > 0 else np.nan,
            'speech_rate_proxy': len(f0_values) / sound.duration,  # Simplified
        }

        if emotion_label:
            features['emotion_label'] = emotion_label

        return features


class MFCCExtractor:
    """Extract MFCC features for machine learning"""

    def __init__(self, n_mfcc: int = 13, sample_rate: int = 16000):
        """
        Initialize MFCC extractor

        Args:
            n_mfcc: Number of MFCCs to extract
            sample_rate: Audio sample rate
        """
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate

    def extract_mfcc_features(self,
                             audio: Union[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract MFCC features with deltas and delta-deltas

        Args:
            audio: Path to audio file or audio array

        Returns:
            Dict with MFCC features
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for MFCC extraction")

        # Load audio if path provided
        if isinstance(audio, str):
            y, sr = librosa.load(audio, sr=self.sample_rate)
        else:
            y = audio
            sr = self.sample_rate

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

        # Delta (first derivative)
        delta_mfcc = librosa.feature.delta(mfccs)

        # Delta-delta (second derivative)
        delta2_mfcc = librosa.feature.delta(mfccs, order=2)

        features = {
            'mfcc': mfccs,  # Shape: (n_mfcc, time_frames)
            'delta_mfcc': delta_mfcc,
            'delta2_mfcc': delta2_mfcc,
        }

        # Statistical summaries for each coefficient
        summary = {}
        for coef_num in range(self.n_mfcc):
            summary[f'mfcc_{coef_num}_mean'] = np.mean(mfccs[coef_num])
            summary[f'mfcc_{coef_num}_std'] = np.std(mfccs[coef_num])
            summary[f'mfcc_{coef_num}_min'] = np.min(mfccs[coef_num])
            summary[f'mfcc_{coef_num}_max'] = np.max(mfccs[coef_num])

        features['mfcc_summary'] = summary

        # Full feature vector (concatenated)
        features['feature_vector'] = np.concatenate([mfccs, delta_mfcc, delta2_mfcc], axis=0)

        return features


class VoiceAnalyzer:
    """Comprehensive voice and speech acoustics analysis"""

    def __init__(self):
        """Initialize voice analyzer"""
        self.prosodic = ProsodicAnalyzer()
        self.spectral = SpectralAnalyzer()
        self.temporal = TemporalAnalyzer()
        self.pragmatic = PragmaticAnalyzer()
        self.mfcc = MFCCExtractor()

    def analyze_voice_sample(self,
                            audio: Union[str, 'parselmouth.Sound'],
                            include_mfcc: bool = False) -> Dict:
        """
        Comprehensive voice analysis of a single audio sample

        Args:
            audio: Path to audio file or parselmouth Sound object
            include_mfcc: Whether to include MFCC features

        Returns:
            Dict with all voice features
        """
        results = {}

        # Prosodic features
        try:
            pitch_features = self.prosodic.extract_pitch_features(audio)
            results.update(pitch_features)
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")

        try:
            intensity_features = self.prosodic.extract_intensity_features(audio)
            results.update(intensity_features)
        except Exception as e:
            logger.warning(f"Intensity extraction failed: {e}")

        try:
            rhythm_features = self.prosodic.extract_rhythm_features(audio)
            results.update(rhythm_features)
        except Exception as e:
            logger.warning(f"Rhythm extraction failed: {e}")

        # Spectral features
        try:
            formant_features = self.spectral.extract_formant_features(audio)
            results.update(formant_features)
        except Exception as e:
            logger.warning(f"Formant extraction failed: {e}")

        try:
            voice_quality = self.spectral.extract_voice_quality_features(audio)
            results.update(voice_quality)
        except Exception as e:
            logger.warning(f"Voice quality extraction failed: {e}")

        # MFCC features (optional - can be large)
        if include_mfcc:
            try:
                mfcc_features = self.mfcc.extract_mfcc_features(audio)
                results['mfcc_data'] = mfcc_features
            except Exception as e:
                logger.warning(f"MFCC extraction failed: {e}")

        return results


if __name__ == '__main__':
    # Example usage
    logger.info("Voice Analyzer initialized")

    if not PARSELMOUTH_AVAILABLE:
        logger.error("parselmouth not installed. Install with: pip install praat-parselmouth")
        logger.info("Example requires parselmouth. Exiting.")
    else:
        # Create synthetic audio for testing
        duration = 1.0  # seconds
        sample_rate = 22050
        frequency = 200  # Hz (F0)

        # Generate harmonic complex (vowel-like sound)
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal_wave = np.zeros_like(t)

        # Add harmonics
        for harmonic in range(1, 6):
            amplitude = 1 / harmonic  # Decrease amplitude with harmonic number
            signal_wave += amplitude * np.sin(2 * np.pi * frequency * harmonic * t)

        # Add some amplitude modulation
        signal_wave *= (1 + 0.3 * np.sin(2 * np.pi * 3 * t))

        # Normalize
        signal_wave = signal_wave / np.max(np.abs(signal_wave)) * 0.5

        # Create parselmouth Sound object
        sound = parselmouth.Sound(signal_wave, sampling_frequency=sample_rate)

        # Analyze voice
        analyzer = VoiceAnalyzer()
        results = analyzer.analyze_voice_sample(sound, include_mfcc=False)

        print("\nVoice Analysis Results:")
        print("="*60)

        if 'pitch_mean_hz' in results:
            print("\nProsodic Features:")
            print(f"  Mean pitch: {results['pitch_mean_hz']:.1f} Hz")
            print(f"  Pitch range: {results.get('pitch_range_hz', 'N/A'):.1f} Hz")
            print(f"  Pitch variability (CV): {results.get('pitch_cv', 'N/A'):.3f}")

        if 'intensity_mean_db' in results:
            print("\n  Mean intensity: {:.1f} dB".format(results['intensity_mean_db']))

        if 'speech_rate_syll_per_sec' in results:
            print(f"\nRhythm Features:")
            print(f"  Speech rate: {results['speech_rate_syll_per_sec']:.2f} syll/sec")

        if 'f1_mean_hz' in results:
            print(f"\nSpectral Features:")
            print(f"  F1: {results['f1_mean_hz']:.0f} Hz")
            print(f"  F2: {results.get('f2_mean_hz', 'N/A'):.0f} Hz")
            print(f"  F3: {results.get('f3_mean_hz', 'N/A'):.0f} Hz")

        if 'hnr_db' in results:
            print(f"\nVoice Quality:")
            print(f"  HNR: {results['hnr_db']:.1f} dB")
            print(f"  Jitter: {results.get('jitter_local', 'N/A')}")
            print(f"  Shimmer: {results.get('shimmer_local', 'N/A')}")

        print("\nNote: Install dependencies:")
        print("  pip install praat-parselmouth librosa")
