#!/usr/bin/env python3
"""
Deep Phenotype Mining System
Discovers hidden relevant features in existing datasets using NLP and pattern matching
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime
from fuzzywuzzy import fuzz
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepPhenotypeMiner:
    """Mine data dictionaries for hidden relevant phenotypes"""

    def __init__(self, output_dir='data/discovered_phenotypes'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Comprehensive search patterns for hidden features
        self.search_patterns = {
            'autonomic': {
                'search_terms': [
                    'heart rate', 'HRV', 'heart rate variability', 'pulse', 'pulse rate',
                    'RR interval', 'heart period', 'cardiac', 'vagal tone',
                    'orthostatic', 'tilt table', 'autonomic', 'sympathetic', 'parasympathetic',
                    'blood pressure variability', 'BP variability', 'baroreflex',
                    'electrocardiogram', 'ECG', 'EKG', 'R-R', 'QT interval',
                    'respiratory sinus arrhythmia', 'RSA'
                ],
                'regex_patterns': [
                    r'HR_?\d+', r'heart_?rate', r'pulse_?rate',
                    r'RR_?interval', r'[Hh]eart[_ ]?[Pp]eriod',
                    r'BP_?variability', r'blood_?pressure_?var',
                    r'ECG_?\w+', r'EKG_?\w+',
                    r'vagal', r'autonomic'
                ],
                'context_terms': ['vital signs', 'physical exam', 'cardiovascular'],
                'exclude_terms': ['heart disease', 'heart failure', 'cardiac arrest']
            },
            'circadian': {
                'search_terms': [
                    'cortisol', 'salivary cortisol', 'morning cortisol', 'evening cortisol',
                    'cortisol awakening', 'CAR', 'diurnal cortisol',
                    'melatonin', 'DLMO', 'dim light melatonin', 'circadian rhythm',
                    'circadian phase', 'chronotype', 'morningness', 'eveningness',
                    'body temperature', 'core temperature', 'temperature rhythm',
                    'sleep diary', 'sleep log', 'bedtime', 'wake time',
                    'time of day', 'morning sample', 'evening sample', 'AM/PM'
                ],
                'regex_patterns': [
                    r'CORT_?\d{4}', r'cortisol_?\w+', r'MEL_?[AP]M',
                    r'melatonin_?\w+', r'[Tt]emp_?\d+', r'body_?temp',
                    r'circadian', r'chronotype', r'morning', r'evening',
                    r'AM_?\w+', r'PM_?\w+', r'time_?of_?day'
                ],
                'context_terms': ['endocrine', 'hormone', 'sleep', 'biological rhythm'],
                'exclude_terms': ['infection', 'stress test']
            },
            'sensory': {
                'search_terms': [
                    'sensory processing', 'sensory profile', 'sensory integration',
                    'tactile', 'touch sensitivity', 'auditory sensitivity',
                    'visual sensitivity', 'proprioception', 'vestibular',
                    'oral sensitivity', 'smell sensitivity', 'taste sensitivity',
                    'sensory seeking', 'sensory avoiding', 'SPM', 'SP2',
                    'occupational therapy', 'OT evaluation', 'sensory assessment',
                    'sound sensitivity', 'noise sensitivity', 'light sensitivity',
                    'texture sensitivity', 'clothing sensitivity'
                ],
                'regex_patterns': [
                    r'sensory_?\w+', r'SPM_?\w+', r'SP2_?\w+',
                    r'tactile', r'propriocept', r'vestibular',
                    r'OT_?\w+', r'occupational_?therapy'
                ],
                'context_terms': ['developmental', 'behavioral assessment', 'therapy'],
                'exclude_terms': ['sensory loss', 'hearing loss', 'vision loss']
            },
            'interoception': {
                'search_terms': [
                    'interoceptive accuracy', 'heartbeat detection', 'interoception',
                    'body awareness', 'internal sensation', 'visceral awareness',
                    'pain threshold', 'pain sensitivity', 'thermal pain', 'pressure pain',
                    'nociception', 'pain tolerance', 'pain rating',
                    'hunger awareness', 'fullness awareness', 'bathroom awareness'
                ],
                'regex_patterns': [
                    r'interocept', r'pain_?threshold', r'pain_?sensit',
                    r'heartbeat_?detect', r'body_?awareness'
                ],
                'context_terms': ['perception', 'awareness', 'sensation'],
                'exclude_terms': ['chronic pain', 'pain disorder']
            },
            'auditory_processing': {
                'search_terms': [
                    'ABR', 'auditory brainstem response', 'BAER', 'brainstem auditory',
                    'otoacoustic emissions', 'OAE', 'DPOAE', 'TEOAE',
                    'hearing test', 'audiometry', 'pure tone', 'tympanometry',
                    'speech in noise', 'auditory processing', 'central auditory',
                    'hearing threshold', 'acoustic reflex'
                ],
                'regex_patterns': [
                    r'ABR_?\w+', r'BAER', r'OAE', r'[DT][PE]OAE',
                    r'audio\w*', r'hearing_?test', r'tymp\w*'
                ],
                'context_terms': ['audiology', 'hearing', 'ear exam'],
                'exclude_terms': ['hearing aid', 'deaf']
            },
            'visual_processing': {
                'search_terms': [
                    'visual processing', 'visual acuity', 'contrast sensitivity',
                    'motion detection', 'visual threshold', 'reaction time visual',
                    'color vision', 'depth perception', 'visual field',
                    'eye exam', 'ophthalmology', 'optometry',
                    'retinal imaging', 'OCT', 'optical coherence tomography',
                    'ERG', 'electroretinogram', 'VEP', 'visual evoked potential'
                ],
                'regex_patterns': [
                    r'visual_?\w+', r'vision_?\w+', r'eye_?\w+',
                    r'OCT', r'ERG', r'VEP', r'retinal', r'acuity'
                ],
                'context_terms': ['ophthalmology', 'vision', 'eye'],
                'exclude_terms': ['glasses', 'corrected vision']
            },
            'environmental_exposure': {
                'search_terms': [
                    # Heavy metals
                    'lead', 'mercury', 'arsenic', 'cadmium', 'manganese',
                    'lead exposure', 'heavy metal', 'Pb blood', 'blood lead',
                    # Pesticides/chemicals
                    'pesticide', 'herbicide', 'insecticide', 'organophosphate',
                    'glyphosate', 'chlorpyrifos', 'malathion',
                    # Plastics
                    'phthalate', 'DEHP', 'DBP', 'BPA', 'bisphenol',
                    'plasticizer', 'BPS', 'BPF',
                    # POPs
                    'PCB', 'PBDE', 'dioxin', 'POP', 'persistent organic',
                    # Location-based
                    'residential history', 'address history', 'zip code',
                    'water source', 'well water', 'city water',
                    'proximity to highway', 'traffic exposure', 'air pollution',
                    'industrial area', 'agricultural area',
                    # Parental occupation
                    'parental occupation', 'mother occupation', 'father occupation',
                    'work exposure', 'occupational exposure'
                ],
                'regex_patterns': [
                    r'[Pp][Bb]_?blood', r'lead_?level', r'mercury_?\w+',
                    r'pesticide', r'phthalate', r'BPA', r'PCB', r'PBDE',
                    r'zip_?code', r'address', r'residential',
                    r'occupation', r'work_?history'
                ],
                'context_terms': ['environmental', 'exposure', 'toxicology', 'biomonitoring'],
                'exclude_terms': []
            },
            'trace_minerals': {
                'search_terms': [
                    'zinc', 'copper', 'selenium', 'magnesium', 'iron',
                    'ferritin', 'folate', 'vitamin D', 'vitamin B12', 'B12',
                    'calcium', 'phosphorus', 'trace mineral', 'micronutrient',
                    'CBC', 'complete blood count', 'CMP', 'comprehensive metabolic',
                    'serum zinc', 'plasma zinc', 'RBC magnesium'
                ],
                'regex_patterns': [
                    r'zinc', r'copper', r'selenium', r'magnesium',
                    r'Zn_?\w+', r'Cu_?\w+', r'Se_?\w+', r'Mg_?\w+',
                    r'ferritin', r'folate', r'B12', r'vitamin_?[DdBb]'
                ],
                'context_terms': ['laboratory', 'blood test', 'nutritional'],
                'exclude_terms': ['supplement', 'vitamin supplement']
            },
            'inflammatory_markers': {
                'search_terms': [
                    'CRP', 'C-reactive protein', 'high sensitivity CRP', 'hs-CRP',
                    'ESR', 'erythrocyte sedimentation', 'sed rate',
                    'interleukin', 'IL-6', 'IL-1', 'IL-8', 'TNF-alpha',
                    'cytokine', 'inflammatory marker',
                    'white blood cell', 'WBC', 'neutrophil', 'lymphocyte'
                ],
                'regex_patterns': [
                    r'CRP', r'ESR', r'IL-?\d+', r'TNF', r'cytokine',
                    r'WBC', r'white_?blood'
                ],
                'context_terms': ['inflammation', 'immune', 'laboratory'],
                'exclude_terms': ['infection']
            },
            'metabolic_calculated': {
                'search_terms': [
                    'insulin resistance', 'HOMA-IR', 'glucose insulin ratio',
                    'triglyceride HDL ratio', 'TG/HDL', 'Trig/HDL',
                    'LDL/HDL ratio', 'cholesterol ratio',
                    'hemoglobin A1c', 'HbA1c', 'A1C',
                    'metabolic syndrome', 'waist hip ratio', 'BMI'
                ],
                'regex_patterns': [
                    r'HOMA', r'insulin_?resistance', r'[Tt]rig.*[Hh][Dd][Ll]',
                    r'A1[Cc]', r'HbA1c', r'waist_?hip', r'BMI'
                ],
                'context_terms': ['metabolic', 'endocrine', 'diabetes'],
                'exclude_terms': []
            }
        }

        # Results storage
        self.discovered_features = defaultdict(lambda: defaultdict(list))

    def fuzzy_match(self, text: str, search_term: str, threshold: int = 80) -> bool:
        """Fuzzy string matching"""
        return fuzz.partial_ratio(text.lower(), search_term.lower()) >= threshold

    def regex_search(self, text: str, patterns: List[str]) -> List[str]:
        """Search text using regex patterns"""
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            matches.extend(found)
        return matches

    def mine_data_dictionary(self,
                           dataset_name: str,
                           data_dict: pd.DataFrame,
                           var_name_col: str = 'variable_name',
                           var_desc_col: str = 'description',
                           var_label_col: str = 'label') -> Dict:
        """
        Mine a data dictionary for hidden features

        Args:
            dataset_name: Name of dataset
            data_dict: DataFrame with variable names, descriptions, labels
            var_name_col: Column name for variable names
            var_desc_col: Column name for descriptions
            var_label_col: Column name for labels
        """
        logger.info(f"Mining {dataset_name} data dictionary...")

        results = {
            'dataset': dataset_name,
            'mined_date': datetime.now().isoformat(),
            'total_variables': len(data_dict),
            'categories': {}
        }

        # Search each category
        for category, search_info in self.search_patterns.items():
            logger.info(f"  Searching for {category}...")

            found_vars = []

            for idx, row in data_dict.iterrows():
                var_name = str(row.get(var_name_col, ''))
                var_desc = str(row.get(var_desc_col, ''))
                var_label = str(row.get(var_label_col, ''))

                # Combine all text for searching
                full_text = f"{var_name} {var_desc} {var_label}"

                # Check if this variable should be excluded
                if any(self.fuzzy_match(full_text, exclude, 70)
                      for exclude in search_info.get('exclude_terms', [])):
                    continue

                # Search using exact terms
                term_matches = [term for term in search_info['search_terms']
                              if self.fuzzy_match(full_text, term)]

                # Search using regex
                regex_matches = self.regex_search(full_text, search_info['regex_patterns'])

                # Check context terms for relevance boost
                context_score = sum(1 for ctx in search_info.get('context_terms', [])
                                  if self.fuzzy_match(full_text, ctx))

                if term_matches or regex_matches:
                    match_info = {
                        'variable_name': var_name,
                        'description': var_desc[:200],  # Truncate long descriptions
                        'label': var_label[:100],
                        'matched_terms': term_matches,
                        'matched_patterns': regex_matches,
                        'context_score': context_score,
                        'relevance_score': len(term_matches) * 2 + len(regex_matches) + context_score
                    }
                    found_vars.append(match_info)

            # Sort by relevance
            found_vars = sorted(found_vars, key=lambda x: x['relevance_score'], reverse=True)

            results['categories'][category] = {
                'count': len(found_vars),
                'variables': found_vars
            }

            logger.info(f"    Found {len(found_vars)} {category} variables")

            # Store in main results
            self.discovered_features[dataset_name][category] = found_vars

        return results

    def mine_variable_names_only(self, dataset_name: str, variable_list: List[str]) -> Dict:
        """Mine when only variable names are available (no descriptions)"""
        logger.info(f"Mining {dataset_name} variable names...")

        # Create simple dataframe
        data_dict = pd.DataFrame({
            'variable_name': variable_list,
            'description': [''] * len(variable_list),
            'label': [''] * len(variable_list)
        })

        return self.mine_data_dictionary(dataset_name, data_dict)

    def generate_extraction_scripts(self, discovered_features: Dict, output_format: str = 'python'):
        """Generate scripts to extract discovered variables"""
        logger.info("Generating extraction scripts...")

        if output_format == 'python':
            script_file = self.output_dir / f"extract_{discovered_features['dataset']}.py"

            with open(script_file, 'w') as f:
                f.write(f"#!/usr/bin/env python3\n")
                f.write(f"# Auto-generated extraction script for {discovered_features['dataset']}\n\n")
                f.write("import pandas as pd\n\n")

                f.write(f"def extract_discovered_features(data_file):\n")
                f.write(f"    \"\"\"Extract discovered phenotypes from {discovered_features['dataset']}\"\"\"\n")
                f.write(f"    df = pd.read_csv(data_file)  # or appropriate reader\n\n")

                for category, info in discovered_features['categories'].items():
                    if info['count'] > 0:
                        vars_list = [v['variable_name'] for v in info['variables']]
                        f.write(f"    # {category.upper()} features ({info['count']} variables)\n")
                        f.write(f"    {category}_vars = {vars_list}\n")
                        f.write(f"    {category}_data = df[{category}_vars].copy()\n\n")

                f.write(f"    return {{\n")
                for category in discovered_features['categories'].keys():
                    f.write(f"        '{category}': {category}_data,\n")
                f.write(f"    }}\n")

            logger.info(f"  Generated Python script: {script_file}")

        elif output_format == 'R':
            script_file = self.output_dir / f"extract_{discovered_features['dataset']}.R"

            with open(script_file, 'w') as f:
                f.write(f"# Auto-generated extraction script for {discovered_features['dataset']}\n\n")
                f.write(f"extract_discovered_features <- function(data_file) {{\n")
                f.write(f"  data <- read.csv(data_file)\n\n")

                for category, info in discovered_features['categories'].items():
                    if info['count'] > 0:
                        vars_list = [f'"{v["variable_name"]}"' for v in info['variables']]
                        f.write(f"  # {category.upper()} features\n")
                        f.write(f"  {category}_vars <- c({', '.join(vars_list)})\n")
                        f.write(f"  {category}_data <- data[, {category}_vars]\n\n")

                f.write(f"  return(list(\n")
                for i, category in enumerate(discovered_features['categories'].keys()):
                    comma = "," if i < len(discovered_features['categories']) - 1 else ""
                    f.write(f"    {category} = {category}_data{comma}\n")
                f.write(f"  ))\n}}\n")

            logger.info(f"  Generated R script: {script_file}")

    def save_results(self, results: Dict, dataset_name: str):
        """Save discovery results"""
        output_file = self.output_dir / f"{dataset_name}_discovered_features.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")

        # Also create summary CSV
        summary_data = []
        for category, info in results['categories'].items():
            for var in info['variables']:
                summary_data.append({
                    'dataset': dataset_name,
                    'category': category,
                    'variable_name': var['variable_name'],
                    'description': var['description'],
                    'relevance_score': var['relevance_score'],
                    'matched_terms': '; '.join(var['matched_terms'][:3])
                })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / f"{dataset_name}_summary.csv"
        summary_df.to_csv(summary_file, index=False)

        logger.info(f"Summary saved to {summary_file}")

    def generate_quality_report(self, results: Dict) -> str:
        """Generate quality assessment of discovered features"""
        report = []
        report.append(f"\n{'='*60}")
        report.append(f"QUALITY ASSESSMENT: {results['dataset']}")
        report.append(f"{'='*60}\n")

        report.append(f"Total variables in dataset: {results['total_variables']}")
        report.append(f"Total features discovered: {sum(info['count'] for info in results['categories'].values())}\n")

        for category, info in results['categories'].items():
            report.append(f"\n{category.upper()}:")
            report.append(f"  Found: {info['count']} variables")

            if info['count'] > 0:
                # Show top 3 by relevance
                report.append(f"  Top matches:")
                for var in info['variables'][:3]:
                    report.append(f"    - {var['variable_name']} (score: {var['relevance_score']})")
                    report.append(f"      {var['description'][:80]}...")

        report.append(f"\n{'='*60}\n")

        return '\n'.join(report)


def main():
    """Example usage"""
    miner = DeepPhenotypeMiner()

    # Example: Create mock data dictionary
    example_dict = pd.DataFrame({
        'variable_name': [
            'HEART_RATE_BASELINE', 'BP_SYSTOLIC', 'CORTISOL_AM', 'CORTISOL_PM',
            'LEAD_BLOOD', 'MERCURY_HAIR', 'SENSORY_PROFILE_TACTILE',
            'ABR_THRESHOLD', 'VISUAL_ACUITY', 'ZINC_SERUM'
        ],
        'description': [
            'Resting heart rate measured at baseline visit in bpm',
            'Systolic blood pressure standing position',
            'Salivary cortisol collected in morning (30 min after waking)',
            'Salivary cortisol collected in evening (before bedtime)',
            'Blood lead concentration in ug/dL',
            'Mercury concentration in hair sample in ppm',
            'Tactile sensitivity subscale from Sensory Processing Measure',
            'Auditory brainstem response hearing threshold at 2000 Hz',
            'Visual acuity Snellen chart distance corrected',
            'Serum zinc concentration in ug/dL'
        ],
        'label': [
            'HR baseline', 'BP sys', 'Cort AM', 'Cort PM',
            'Pb blood', 'Hg hair', 'SPM tactile',
            'ABR 2kHz', 'VA distance', 'Zn serum'
        ]
    })

    # Mine the dictionary
    results = miner.mine_data_dictionary('EXAMPLE_DATASET', example_dict)

    # Generate quality report
    report = miner.generate_quality_report(results)
    print(report)

    # Save results
    miner.save_results(results, 'EXAMPLE_DATASET')

    # Generate extraction scripts
    miner.generate_extraction_scripts(results, 'python')
    miner.generate_extraction_scripts(results, 'R')

    logger.info("\n✓ Mining complete!")


if __name__ == '__main__':
    main()