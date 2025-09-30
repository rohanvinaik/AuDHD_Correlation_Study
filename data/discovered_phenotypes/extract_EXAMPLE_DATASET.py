#!/usr/bin/env python3
# Auto-generated extraction script for EXAMPLE_DATASET

import pandas as pd

def extract_discovered_features(data_file):
    """Extract discovered phenotypes from EXAMPLE_DATASET"""
    df = pd.read_csv(data_file)  # or appropriate reader

    # AUTONOMIC features (1 variables)
    autonomic_vars = ['HEART_RATE_BASELINE']
    autonomic_data = df[autonomic_vars].copy()

    # CIRCADIAN features (3 variables)
    circadian_vars = ['CORTISOL_PM', 'CORTISOL_AM', 'MERCURY_HAIR']
    circadian_data = df[circadian_vars].copy()

    # SENSORY features (1 variables)
    sensory_vars = ['SENSORY_PROFILE_TACTILE']
    sensory_data = df[sensory_vars].copy()

    # INTEROCEPTION features (3 variables)
    interoception_vars = ['BP_SYSTOLIC', 'SENSORY_PROFILE_TACTILE', 'ABR_THRESHOLD']
    interoception_data = df[interoception_vars].copy()

    # AUDITORY_PROCESSING features (1 variables)
    auditory_processing_vars = ['ABR_THRESHOLD']
    auditory_processing_data = df[auditory_processing_vars].copy()

    # ENVIRONMENTAL_EXPOSURE features (2 variables)
    environmental_exposure_vars = ['LEAD_BLOOD', 'MERCURY_HAIR']
    environmental_exposure_data = df[environmental_exposure_vars].copy()

    # TRACE_MINERALS features (5 variables)
    trace_minerals_vars = ['ZINC_SERUM', 'HEART_RATE_BASELINE', 'SENSORY_PROFILE_TACTILE', 'MERCURY_HAIR', 'VISUAL_ACUITY']
    trace_minerals_data = df[trace_minerals_vars].copy()

    return {
        'autonomic': autonomic_data,
        'circadian': circadian_data,
        'sensory': sensory_data,
        'interoception': interoception_data,
        'auditory_processing': auditory_processing_data,
        'visual_processing': visual_processing_data,
        'environmental_exposure': environmental_exposure_data,
        'trace_minerals': trace_minerals_data,
        'inflammatory_markers': inflammatory_markers_data,
        'metabolic_calculated': metabolic_calculated_data,
    }
