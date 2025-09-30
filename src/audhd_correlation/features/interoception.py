#!/usr/bin/env python3
"""
Interoception Assessment Pipeline

Processes interoceptive accuracy, sensibility, and awareness measures
Implements heartbeat detection tasks and body perception questionnaires
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteroceptionProcessor:
    """Process interoceptive measures"""

    def __init__(self):
        """Initialize interoception processor"""

        # MAIA-2 subscales (Multidimensional Assessment of Interoceptive Awareness)
        self.maia2_subscales = {
            'noticing': [1, 2, 3, 4],  # Item numbers
            'not_distracting': [5, 6, 7],
            'not_worrying': [8, 9, 10],
            'attention_regulation': [11, 12, 13, 14, 15, 16, 17],
            'emotional_awareness': [18, 19, 20, 21, 22],
            'self_regulation': [23, 24, 25, 26],
            'body_listening': [27, 28, 29],
            'trusting': [30, 31, 32]
        }

        # Body Perception Questionnaire dimensions
        self.bpq_categories = {
            'autonomic': ['sweating', 'racing_heart', 'nausea', 'faintness'],
            'visceral': ['stomach_pain', 'bloating', 'hunger'],
            'musculoskeletal': ['muscle_tension', 'pain', 'fatigue'],
            'cardiovascular': ['palpitations', 'chest_pain']
        }

    def calculate_heartbeat_counting_accuracy(self,
                                             recorded_beats: np.ndarray,
                                             counted_beats: np.ndarray,
                                             confidence: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate interoceptive accuracy from heartbeat counting task

        IAcc = 1 - (1/ntrials * Σ|recorded - counted| / recorded)

        Args:
            recorded_beats: Actual heartbeats recorded (per trial)
            counted_beats: Participant's count (per trial)
            confidence: Optional confidence ratings (1-10)

        Returns:
            Dict with accuracy metrics
        """
        # Interoceptive accuracy (Schandry 1981)
        errors = np.abs(recorded_beats - counted_beats) / recorded_beats
        accuracy = 1 - np.mean(errors)

        # Alternative: average absolute accuracy per trial
        trial_accuracies = 1 - errors

        # Interoceptive sensibility (self-rated confidence)
        if confidence is not None:
            mean_confidence = np.mean(confidence)

            # Interoceptive awareness (metacognitive)
            # Correlation between confidence and accuracy
            awareness = stats.pearsonr(confidence, trial_accuracies)[0]
        else:
            mean_confidence = np.nan
            awareness = np.nan

        metrics = {
            'interoceptive_accuracy': accuracy,
            'mean_trial_accuracy': np.mean(trial_accuracies),
            'sd_trial_accuracy': np.std(trial_accuracies),
            'mean_confidence': mean_confidence,
            'interoceptive_awareness': awareness,
            'n_trials': len(recorded_beats)
        }

        return metrics

    def calculate_heartbeat_discrimination_accuracy(self,
                                                   trials: pd.DataFrame) -> Dict:
        """
        Calculate accuracy from heartbeat discrimination task

        Participants judge if tone is synchronous or asynchronous with heartbeat

        Args:
            trials: DataFrame with columns ['synchronous', 'response', 'confidence']
                   synchronous: True if tone matched heartbeat
                   response: Participant's judgment (True/False)

        Returns:
            Dict with discrimination metrics
        """
        # Overall accuracy
        correct = trials['synchronous'] == trials['response']
        accuracy = np.mean(correct)

        # Signal detection theory metrics
        hits = np.sum((trials['synchronous'] == True) & (trials['response'] == True))
        false_alarms = np.sum((trials['synchronous'] == False) & (trials['response'] == True))
        misses = np.sum((trials['synchronous'] == True) & (trials['response'] == False))
        correct_rejections = np.sum((trials['synchronous'] == False) & (trials['response'] == False))

        # Calculate d' (sensitivity) and criterion c
        n_signal = hits + misses
        n_noise = false_alarms + correct_rejections

        hit_rate = hits / n_signal if n_signal > 0 else 0.5
        fa_rate = false_alarms / n_noise if n_noise > 0 else 0.5

        # Adjust for extreme values
        hit_rate = np.clip(hit_rate, 0.01, 0.99)
        fa_rate = np.clip(fa_rate, 0.01, 0.99)

        d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
        criterion = -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(fa_rate))

        # Confidence-accuracy correlation
        if 'confidence' in trials.columns:
            awareness = stats.pearsonr(trials['confidence'], correct.astype(int))[0]
        else:
            awareness = np.nan

        metrics = {
            'discrimination_accuracy': accuracy,
            'd_prime': d_prime,
            'criterion': criterion,
            'hit_rate': hit_rate,
            'false_alarm_rate': fa_rate,
            'interoceptive_awareness': awareness
        }

        return metrics

    def score_maia2(self, responses: np.ndarray) -> Dict:
        """
        Score MAIA-2 questionnaire (32 items, 0-5 Likert scale)

        Args:
            responses: Array of 32 responses (0-5)

        Returns:
            Dict with subscale scores
        """
        if len(responses) != 32:
            raise ValueError(f"MAIA-2 requires 32 responses, got {len(responses)}")

        scores = {}

        for subscale, items in self.maia2_subscales.items():
            # Items are 1-indexed, convert to 0-indexed
            item_indices = [i - 1 for i in items]
            subscale_responses = responses[item_indices]

            # Mean score for subscale
            scores[f'maia2_{subscale}'] = np.mean(subscale_responses)

        # Overall interoceptive awareness
        scores['maia2_total'] = np.mean(responses)

        return scores

    def score_bpq(self, responses: Dict[str, int]) -> Dict:
        """
        Score Body Perception Questionnaire

        Args:
            responses: Dict mapping symptom to frequency (1-5)

        Returns:
            Dict with category scores
        """
        scores = {}

        for category, symptoms in self.bpq_categories.items():
            category_scores = [responses.get(symptom, 0) for symptom in symptoms]
            scores[f'bpq_{category}'] = np.mean(category_scores)

        # Overall body awareness
        all_scores = list(responses.values())
        scores['bpq_total'] = np.mean(all_scores)

        return scores

    def calculate_interoceptive_profile(self,
                                       heartbeat_accuracy: Optional[float] = None,
                                       maia2_scores: Optional[Dict] = None,
                                       confidence: Optional[float] = None) -> Dict:
        """
        Create multi-dimensional interoceptive profile

        Three dimensions (Garfinkel et al. 2015):
        1. Interoceptive accuracy (objective performance)
        2. Interoceptive sensibility (subjective report)
        3. Interoceptive awareness (metacognitive)

        Args:
            heartbeat_accuracy: Objective accuracy score
            maia2_scores: MAIA-2 subscale scores
            confidence: Mean confidence rating

        Returns:
            Comprehensive interoceptive profile
        """
        profile = {}

        # Dimension 1: Accuracy (objective)
        if heartbeat_accuracy is not None:
            profile['interoceptive_accuracy'] = heartbeat_accuracy
            profile['accuracy_category'] = (
                'high' if heartbeat_accuracy > 0.85 else
                'medium' if heartbeat_accuracy > 0.70 else
                'low'
            )

        # Dimension 2: Sensibility (subjective)
        if maia2_scores is not None:
            profile['interoceptive_sensibility'] = maia2_scores.get('maia2_total', np.nan)
            profile['body_listening'] = maia2_scores.get('maia2_body_listening', np.nan)
            profile['noticing'] = maia2_scores.get('maia2_noticing', np.nan)

        if confidence is not None:
            profile['mean_confidence'] = confidence

        # Dimension 3: Awareness (metacognitive)
        # This requires confidence-accuracy correlation, calculated in other methods

        return profile

    def process_interoception_battery(self, data: Dict) -> Dict:
        """
        Complete interoception assessment pipeline

        Args:
            data: Dict with keys:
                - 'heartbeat_counting': {'recorded': [], 'counted': [], 'confidence': []}
                - 'heartbeat_discrimination': DataFrame with trial data
                - 'maia2': array of 32 responses
                - 'bpq': dict of symptom frequencies

        Returns:
            Complete interoceptive profile
        """
        results = {}

        # Heartbeat counting
        if 'heartbeat_counting' in data:
            hbc = data['heartbeat_counting']
            counting_metrics = self.calculate_heartbeat_counting_accuracy(
                recorded_beats=np.array(hbc['recorded']),
                counted_beats=np.array(hbc['counted']),
                confidence=np.array(hbc.get('confidence'))
            )
            results.update(counting_metrics)

        # Heartbeat discrimination
        if 'heartbeat_discrimination' in data:
            disc_metrics = self.calculate_heartbeat_discrimination_accuracy(
                trials=data['heartbeat_discrimination']
            )
            results.update(disc_metrics)

        # MAIA-2
        if 'maia2' in data:
            maia2_scores = self.score_maia2(np.array(data['maia2']))
            results.update(maia2_scores)

        # BPQ
        if 'bpq' in data:
            bpq_scores = self.score_bpq(data['bpq'])
            results.update(bpq_scores)

        # Create comprehensive profile
        profile = self.calculate_interoceptive_profile(
            heartbeat_accuracy=results.get('interoceptive_accuracy'),
            maia2_scores={k: v for k, v in results.items() if k.startswith('maia2')},
            confidence=results.get('mean_confidence')
        )
        results.update(profile)

        return results


if __name__ == '__main__':
    # Example usage
    logger.info("Interoception Processor initialized")

    processor = InteroceptionProcessor()

    # Simulate heartbeat counting task
    np.random.seed(42)
    recorded_beats = np.array([50, 60, 70, 80])  # 30s, 45s, 60s, 90s intervals
    counted_beats = recorded_beats + np.random.randint(-10, 10, size=4)
    confidence = np.array([7, 8, 6, 7])

    counting_metrics = processor.calculate_heartbeat_counting_accuracy(
        recorded_beats, counted_beats, confidence
    )

    print("\nHeartbeat Counting Metrics:")
    print("="*50)
    for k, v in counting_metrics.items():
        print(f"  {k}: {v:.3f}" if not np.isnan(v) else f"  {k}: N/A")

    # Simulate MAIA-2 responses
    maia2_responses = np.random.randint(0, 6, size=32)
    maia2_scores = processor.score_maia2(maia2_responses)

    print("\nMAIA-2 Scores:")
    print("="*50)
    for subscale, score in list(maia2_scores.items())[:5]:
        print(f"  {subscale}: {score:.2f}")
