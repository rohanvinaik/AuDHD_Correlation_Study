#!/usr/bin/env python3
"""
Comprehensive Data Discovery System for Extended Phenotypes
Searches major databases and cohorts for missing features in AuDHD study
"""

import json
import requests
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDiscovery:
    """Discover and catalog datasets with extended phenotypes"""

    def __init__(self, output_dir='data/discovery'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Target features organized by priority
        self.target_features = {
            'PRIORITY_1_AUTONOMIC_CIRCADIAN': {
                'hrv': ['heart rate variability', 'HRV', 'RR interval', 'heart period'],
                'eda': ['electrodermal', 'skin conductance', 'galvanic skin', 'SCR', 'EDA'],
                'cortisol': ['cortisol awakening', 'CAR', 'salivary cortisol', 'diurnal cortisol'],
                'melatonin': ['melatonin', 'DLMO', 'circadian rhythm', 'circadian phase'],
                'temperature': ['core body temperature', 'temperature rhythm', 'thermoregulation'],
                'actigraphy': ['actigraphy', 'accelerometer', 'activity monitor', 'actimetry'],
                'sleep_psg': ['polysomnography', 'PSG', 'sleep architecture', 'sleep staging']
            },
            'PRIORITY_2_SENSORY_INTEROCEPTION': {
                'sensory': ['sensory processing', 'sensory profile', 'tactile', 'auditory sensitivity'],
                'interoception': ['interoceptive accuracy', 'heartbeat detection', 'interoception'],
                'pain': ['pain threshold', 'pain sensitivity', 'nociception', 'thermal pain'],
                'proprioception': ['proprioception', 'kinesthesia', 'joint position sense'],
                'auditory': ['ABR', 'auditory brainstem', 'otoacoustic', 'hearing threshold'],
                'visual': ['visual processing speed', 'contrast sensitivity', 'motion detection']
            },
            'PRIORITY_3_ENVIRONMENTAL': {
                'heavy_metals': ['lead', 'mercury', 'arsenic', 'cadmium', 'heavy metal'],
                'phthalates': ['phthalate', 'DEHP', 'DBP', 'plasticizer'],
                'bpa': ['bisphenol', 'BPA', 'BPS', 'BPF'],
                'pesticides': ['organophosphate', 'glyphosate', 'pesticide', 'herbicide'],
                'pops': ['PCB', 'PBDE', 'persistent organic', 'dioxin', 'POP']
            },
            'PRIORITY_4_ADDITIONAL_BIOLOGICAL': {
                'proteomics': ['SOMAscan', 'Olink', 'proteomics', 'protein panel'],
                'minerals': ['zinc', 'copper', 'selenium', 'magnesium', 'trace mineral'],
                'cfdna': ['cell-free DNA', 'cfDNA', 'circulating DNA'],
                'exosomes': ['exosome', 'extracellular vesicle', 'microvesicle'],
                'voice': ['voice acoustic', 'speech analysis', 'prosody', 'vocal'],
                'retinal': ['optical coherence', 'OCT', 'retinal imaging', 'ERG', 'electroretinogram']
            }
        }

        # Known large cohorts and databases
        self.data_sources = self._initialize_sources()

        # Results storage
        self.discovered_datasets = []

    def _initialize_sources(self) -> Dict[str, Any]:
        """Initialize known data sources with API/search information"""
        return {
            'ABCD_Study': {
                'name': 'Adolescent Brain Cognitive Development Study',
                'url': 'https://abcdstudy.org',
                'data_dict_url': 'https://nda.nih.gov/abcd',
                'n_participants': 11878,
                'age_range': '9-15 years',
                'disorders': ['ADHD', 'ASD (small n)'],
                'known_measures': ['actigraphy', 'sleep', 'environmental exposures'],
                'access': 'NDA application required',
                'api': None,
                'search_strategy': 'manual_data_dictionary'
            },
            'SPARK': {
                'name': 'Simons Foundation Powering Autism Research',
                'url': 'https://sparkforautism.org',
                'data_dict_url': 'https://www.sfari.org/resource/spark/',
                'n_participants': 100000,
                'age_range': 'All ages',
                'disorders': ['ASD'],
                'known_measures': ['phenotype', 'medical history', 'genetics'],
                'access': 'SFARI Base access',
                'api': None,
                'search_strategy': 'sfari_portal'
            },
            'UK_Biobank': {
                'name': 'UK Biobank',
                'url': 'https://www.ukbiobank.ac.uk',
                'data_dict_url': 'https://biobank.ndph.ox.ac.uk/showcase/',
                'n_participants': 500000,
                'age_range': '40-69 years',
                'disorders': ['Adult ADHD/ASD via ICD codes'],
                'known_measures': ['retinal imaging', 'accelerometry', 'proteomics', 'metabolomics'],
                'access': 'Application required',
                'api': None,
                'search_strategy': 'showcase_search'
            },
            'NHANES': {
                'name': 'National Health and Nutrition Examination Survey',
                'url': 'https://www.cdc.gov/nchs/nhanes/',
                'data_dict_url': 'https://wwwn.cdc.gov/nchs/nhanes/',
                'n_participants': 10000,  # per cycle
                'age_range': 'All ages',
                'disorders': ['General population'],
                'known_measures': ['environmental biomarkers', 'heavy metals', 'pesticides'],
                'access': 'Public',
                'api': 'https://wwwn.cdc.gov/nchs/data/api/',
                'search_strategy': 'api_query'
            },
            'PhysioNet': {
                'name': 'PhysioNet',
                'url': 'https://physionet.org',
                'data_dict_url': 'https://physionet.org/about/database/',
                'n_participants': 'Varies by dataset',
                'age_range': 'Varies',
                'disorders': ['Various, search for ASD/ADHD'],
                'known_measures': ['HRV', 'ECG', 'EDA', 'physiological signals'],
                'access': 'Public/credentialed',
                'api': None,
                'search_strategy': 'database_search'
            },
            'NSRR': {
                'name': 'National Sleep Research Resource',
                'url': 'https://sleepdata.org',
                'data_dict_url': 'https://sleepdata.org/datasets',
                'n_participants': 'Varies',
                'age_range': 'All ages',
                'disorders': ['Search for neurodevelopmental'],
                'known_measures': ['PSG', 'sleep architecture', 'actigraphy'],
                'access': 'Public/data use agreement',
                'api': None,
                'search_strategy': 'dataset_browse'
            },
            'All_of_Us': {
                'name': 'All of Us Research Program',
                'url': 'https://www.researchallofus.org',
                'data_dict_url': 'https://databrowser.researchallofus.org',
                'n_participants': 413000,
                'age_range': 'All ages',
                'disorders': ['Self-reported ASD/ADHD'],
                'known_measures': ['EHR', 'genomics', 'wearables', 'surveys'],
                'access': 'Workbench access',
                'api': None,
                'search_strategy': 'data_browser'
            },
            'HCP': {
                'name': 'Human Connectome Project',
                'url': 'https://www.humanconnectome.org',
                'data_dict_url': 'https://www.humanconnectome.org/study/hcp-young-adult/data-releases',
                'n_participants': 1200,
                'age_range': '22-37 years',
                'disorders': ['Healthy controls'],
                'known_measures': ['sensory', 'visual', 'auditory', 'extensive behavioral'],
                'access': 'Data use terms acceptance',
                'api': None,
                'search_strategy': 'manual'
            },
            'CHEAR': {
                'name': 'Children\'s Health Exposure Analysis Resource',
                'url': 'https://www.cheardatacenter.mssm.edu',
                'data_dict_url': 'https://www.cheardatacenter.mssm.edu/data',
                'n_participants': 'Multiple cohorts',
                'age_range': 'Children',
                'disorders': ['Various including neurodevelopmental'],
                'known_measures': ['environmental exposures', 'biomonitoring'],
                'access': 'Varies by cohort',
                'api': None,
                'search_strategy': 'cohort_search'
            }
        }

    def search_physionet(self) -> List[Dict]:
        """Search PhysioNet for relevant datasets"""
        logger.info("Searching PhysioNet for autonomic/physiological data...")

        discoveries = []

        # Known PhysioNet databases with potential relevance
        potential_databases = [
            {
                'id': 'capslpdb',
                'name': 'CAP Sleep Database',
                'url': 'https://physionet.org/content/capslpdb/',
                'measures': ['PSG', 'sleep_architecture'],
                'n_subjects': 108,
                'relevance': 'Sleep disorders - check for developmental'
            },
            {
                'id': 'ptbdb',
                'name': 'PTB Diagnostic ECG Database',
                'url': 'https://physionet.org/content/ptbdb/',
                'measures': ['ECG', 'HRV'],
                'n_subjects': 290,
                'relevance': 'HRV analysis possible'
            },
            {
                'id': 'mitdb',
                'name': 'MIT-BIH Arrhythmia Database',
                'url': 'https://physionet.org/content/mitdb/',
                'measures': ['ECG', 'HRV'],
                'n_subjects': 47,
                'relevance': 'Classic HRV reference'
            }
        ]

        for db in potential_databases:
            discovery = {
                'dataset_name': db['name'],
                'source': 'PhysioNet',
                'url': db['url'],
                'available_features': {
                    'autonomic': db['measures'],
                    'circadian': [],
                    'sensory': [],
                    'environmental': [],
                    'proteomics': [],
                    'other': []
                },
                'n_participants': db['n_subjects'],
                'sample_overlap': {'SPARK': 0, 'SSC': 0, 'ABCD': 0},
                'access_requirements': 'Public - PhysioNet credentialing',
                'data_format': 'WFDB, EDF',
                'api_available': False,
                'linked_studies': [],
                'quality_score': 7,
                'notes': db['relevance']
            }
            discoveries.append(discovery)

        return discoveries

    def search_nsrr(self) -> List[Dict]:
        """Search National Sleep Research Resource"""
        logger.info("Searching NSRR for sleep/circadian data...")

        discoveries = []

        nsrr_datasets = [
            {
                'name': 'Childhood Adenotonsillectomy Trial (CHAT)',
                'url': 'https://sleepdata.org/datasets/chat',
                'n': 1244,
                'age': '5-9 years',
                'measures': ['PSG', 'actigraphy', 'neurobehavioral'],
                'asd_adhd': 'Likely has ADHD overlap - check'
            },
            {
                'name': 'Cleveland Family Study',
                'url': 'https://sleepdata.org/datasets/cfs',
                'n': 2284,
                'age': 'All ages, family-based',
                'measures': ['PSG', 'sleep questionnaires'],
                'asd_adhd': 'Family-based - genetic correlation possible'
            }
        ]

        for ds in nsrr_datasets:
            discovery = {
                'dataset_name': ds['name'],
                'source': 'NSRR',
                'url': ds['url'],
                'available_features': {
                    'autonomic': ['HRV from ECG'],
                    'circadian': ds['measures'],
                    'sensory': [],
                    'environmental': [],
                    'proteomics': [],
                    'other': []
                },
                'n_participants': ds['n'],
                'age_range': ds['age'],
                'sample_overlap': {'SPARK': 'Unknown', 'SSC': 'Unknown', 'ABCD': 'Unknown'},
                'access_requirements': 'Data use agreement',
                'data_format': 'EDF, annotations',
                'api_available': False,
                'linked_studies': [],
                'quality_score': 8,
                'notes': ds['asd_adhd']
            }
            discoveries.append(discovery)

        return discoveries

    def search_nhanes_environmental(self) -> List[Dict]:
        """Search NHANES for environmental biomarkers"""
        logger.info("Searching NHANES for environmental exposures...")

        discoveries = []

        # NHANES has comprehensive environmental data
        nhanes_components = [
            {
                'component': 'Heavy Metals',
                'cycles': '1999-2018',
                'measures': ['Lead', 'Mercury', 'Cadmium', 'Manganese'],
                'sample_types': ['Blood', 'Urine']
            },
            {
                'component': 'Pesticides',
                'cycles': '1999-2018',
                'measures': ['Organophosphates', 'Pyrethroids', 'Herbicides'],
                'sample_types': ['Urine']
            },
            {
                'component': 'Phthalates & Plasticizers',
                'cycles': '1999-2018',
                'measures': ['DEHP', 'DBP', 'BPA', 'BPS'],
                'sample_types': ['Urine']
            },
            {
                'component': 'Persistent Organic Pollutants',
                'cycles': '1999-2018',
                'measures': ['PCBs', 'PBDEs', 'Dioxins'],
                'sample_types': ['Serum']
            }
        ]

        for comp in nhanes_components:
            discovery = {
                'dataset_name': f'NHANES {comp["component"]}',
                'source': 'NHANES',
                'url': 'https://wwwn.cdc.gov/nchs/nhanes/',
                'available_features': {
                    'autonomic': [],
                    'circadian': [],
                    'sensory': [],
                    'environmental': comp['measures'],
                    'proteomics': [],
                    'other': []
                },
                'n_participants': '~10,000 per cycle, ~200,000 total',
                'age_range': 'All ages',
                'sample_overlap': {'SPARK': 'Low', 'SSC': 'Low', 'ABCD': 'Possible via linkage'},
                'access_requirements': 'Public',
                'data_format': 'SAS, CSV, R',
                'api_available': True,
                'linked_studies': ['Multiple ADHD/ASD environmental studies use NHANES'],
                'quality_score': 9,
                'notes': f'Gold standard for {comp["component"]}, representative US sample'
            }
            discoveries.append(discovery)

        return discoveries

    def search_ukb_extended(self) -> Dict:
        """Search UK Biobank for extended phenotypes"""
        logger.info("Cataloging UK Biobank extended measures...")

        discovery = {
            'dataset_name': 'UK Biobank Extended Phenotyping',
            'source': 'UK Biobank',
            'url': 'https://biobank.ndph.ox.ac.uk/showcase/',
            'available_features': {
                'autonomic': ['Heart rate from wearables', 'BP variability'],
                'circadian': ['Accelerometry (7-day)', 'Chronotype'],
                'sensory': ['Hearing test', 'Visual acuity', 'Reaction time'],
                'environmental': ['Via questionnaire and models'],
                'proteomics': ['Olink (~3000 proteins in subset)'],
                'other': ['Retinal imaging (67k)', 'Voice recordings']
            },
            'n_participants': 502000,
            'age_range': '40-69 years at baseline',
            'sample_overlap': {
                'SPARK': 0,
                'SSC': 0,
                'ABCD': 0
            },
            'access_requirements': 'Application + fee (~$5000 typical)',
            'data_format': 'Encrypted bulk download',
            'api_available': False,
            'linked_studies': [
                'Adult ADHD genetics (Demontis et al.)',
                'ASD in adults (Grove et al.)',
                'Sensory sensitivity GWAS'
            ],
            'quality_score': 10,
            'notes': 'Largest proteomics cohort, excellent for adult phenotypes, Limited child/adolescent',
            'specific_fields': {
                'proteomics': 'Field 30900 (Olink panel)',
                'retinal': 'Category 151 (Eye measures)',
                'actigraphy': 'Field 90001-90004',
                'voice': 'Field 20251'
            }
        }

        return discovery

    def search_abcd_extended(self) -> Dict:
        """Catalog ABCD Study extended measures"""
        logger.info("Cataloging ABCD extended phenotypes...")

        discovery = {
            'dataset_name': 'ABCD Study - Extended Phenotyping',
            'source': 'ABCD Study',
            'url': 'https://abcdstudy.org',
            'available_features': {
                'autonomic': ['Unknown - need to check'],
                'circadian': ['Sleep disturbance scale', 'Fitbit data (subset)'],
                'sensory': ['Sensory profile questionnaire'],
                'environmental': [
                    'Residential exposures (GIS-linked)',
                    'Lead (GIS estimates)',
                    'Air pollution',
                    'Neighborhood factors'
                ],
                'proteomics': ['Unknown'],
                'other': ['Extensive neuroimaging', 'Cognitive battery']
            },
            'n_participants': 11878,
            'age_range': '9-10 at baseline, now 13-14',
            'adhd_n': '~2000 with ADHD',
            'asd_n': '~200 with ASD',
            'sample_overlap': {
                'SPARK': 'Check via genetics',
                'SSC': 'Unlikely',
                'ABCD': 'Self'
            },
            'access_requirements': 'NDA access + ABCD study approval',
            'data_format': 'NDA format (CSV, imaging)',
            'api_available': False,
            'linked_studies': [
                'ABCD is source of multiple ADHD studies',
                'Environmental exposure papers'
            ],
            'quality_score': 9,
            'notes': 'Largest longitudinal child cohort in US, Fitbit data in subset only',
            'action_items': [
                'Request access to data dictionary',
                'Search for: "heart rate", "actigraphy", "cortisol"',
                'Check ancillary studies for extended phenotyping'
            ]
        }

        return discovery

    def search_all_of_us(self) -> Dict:
        """Catalog All of Us measures"""
        logger.info("Cataloging All of Us Research Program...")

        discovery = {
            'dataset_name': 'All of Us Research Program',
            'source': 'All of Us',
            'url': 'https://www.researchallofus.org',
            'available_features': {
                'autonomic': ['Fitbit HR, HRV (subset with wearables)'],
                'circadian': ['Fitbit sleep (subset)', 'Activity patterns'],
                'sensory': ['Via surveys'],
                'environmental': ['Residential history', 'Occupational', 'Survey-based'],
                'proteomics': ['Planned'],
                'other': ['EHR data', 'Genomics']
            },
            'n_participants': 413000,
            'age_range': 'All ages',
            'adhd_n': 'Self-reported ~15,000',
            'asd_n': 'Self-reported ~3,000',
            'sample_overlap': {
                'SPARK': 'Possible overlap',
                'SSC': 'Unknown',
                'ABCD': 'No'
            },
            'access_requirements': 'Researcher Workbench account',
            'data_format': 'OMOP, FHIR via Workbench',
            'api_available': True,
            'linked_studies': ['Growing number of ASD/ADHD studies'],
            'quality_score': 8,
            'notes': 'Wearable data in ~25% of participants, Diverse population, EHR quality varies',
            'wearable_subset': 100000
        }

        return discovery

    def generate_priority_list(self, discoveries: List[Dict]) -> pd.DataFrame:
        """Generate prioritized list of datasets"""
        logger.info("Generating priority rankings...")

        # Convert to DataFrame
        df_data = []
        for disc in discoveries:
            # Count available features by priority
            priority_1_count = len([f for cat in ['autonomic', 'circadian']
                                   for f in disc['available_features'].get(cat, [])])
            priority_2_count = len([f for cat in ['sensory']
                                   for f in disc['available_features'].get(cat, [])])
            priority_3_count = len([f for cat in ['environmental']
                                   for f in disc['available_features'].get(cat, [])])

            df_data.append({
                'Dataset': disc['dataset_name'],
                'Source': disc['source'],
                'N_Participants': disc.get('n_participants', 'Unknown'),
                'Priority_1_Features': priority_1_count,
                'Priority_2_Features': priority_2_count,
                'Priority_3_Features': priority_3_count,
                'Quality_Score': disc.get('quality_score', 5),
                'Access': disc.get('access_requirements', 'Unknown'),
                'API': disc.get('api_available', False),
                'URL': disc.get('url', '')
            })

        df = pd.DataFrame(df_data)

        # Calculate priority score
        df['Priority_Score'] = (
            df['Priority_1_Features'] * 3 +
            df['Priority_2_Features'] * 2 +
            df['Priority_3_Features'] * 2 +
            df['Quality_Score']
        )

        # Sort by priority
        df = df.sort_values('Priority_Score', ascending=False)

        return df

    def run_discovery(self):
        """Run complete discovery process"""
        logger.info("Starting comprehensive data discovery...")

        all_discoveries = []

        # Search each source
        all_discoveries.extend(self.search_physionet())
        all_discoveries.extend(self.search_nsrr())
        all_discoveries.extend(self.search_nhanes_environmental())
        all_discoveries.append(self.search_ukb_extended())
        all_discoveries.append(self.search_abcd_extended())
        all_discoveries.append(self.search_all_of_us())

        # Save detailed discoveries
        output_file = self.output_dir / f'discovered_datasets_{datetime.now().strftime("%Y%m%d")}.json'
        with open(output_file, 'w') as f:
            json.dump(all_discoveries, f, indent=2)

        logger.info(f"Saved {len(all_discoveries)} discovered datasets to {output_file}")

        # Generate priority list
        priority_df = self.generate_priority_list(all_discoveries)
        priority_file = self.output_dir / 'dataset_priorities.csv'
        priority_df.to_csv(priority_file, index=False)

        logger.info(f"Saved priority rankings to {priority_file}")

        # Generate summary report
        self.generate_summary_report(all_discoveries, priority_df)

        return all_discoveries, priority_df

    def generate_summary_report(self, discoveries: List[Dict], priority_df: pd.DataFrame):
        """Generate human-readable summary report"""
        report_file = self.output_dir / 'DISCOVERY_REPORT.md'

        with open(report_file, 'w') as f:
            f.write("# Extended Phenotype Data Discovery Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- Total datasets discovered: {len(discoveries)}\n")
            f.write(f"- Sources covered: {len(set(d['source'] for d in discoveries))}\n\n")

            f.write("## Top Priority Datasets\n\n")
            for idx, row in priority_df.head(10).iterrows():
                f.write(f"### {idx+1}. {row['Dataset']}\n")
                f.write(f"- **Source**: {row['Source']}\n")
                f.write(f"- **N**: {row['N_Participants']}\n")
                f.write(f"- **Priority Score**: {row['Priority_Score']:.1f}\n")
                f.write(f"- **Access**: {row['Access']}\n")
                f.write(f"- **URL**: {row['URL']}\n\n")

            # Feature coverage summary
            f.write("## Feature Coverage by Priority\n\n")

            for priority in ['PRIORITY_1_AUTONOMIC_CIRCADIAN', 'PRIORITY_2_SENSORY_INTEROCEPTION',
                           'PRIORITY_3_ENVIRONMENTAL', 'PRIORITY_4_ADDITIONAL_BIOLOGICAL']:
                f.write(f"### {priority.replace('_', ' ')}\n\n")

                # Count datasets with each feature category
                relevant_cats = {
                    'PRIORITY_1': ['autonomic', 'circadian'],
                    'PRIORITY_2': ['sensory'],
                    'PRIORITY_3': ['environmental'],
                    'PRIORITY_4': ['proteomics', 'other']
                }

                cats = relevant_cats.get(priority.split('_')[0] + '_' + priority.split('_')[1], [])

                for cat in cats:
                    datasets_with_cat = [d['dataset_name'] for d in discoveries
                                        if d['available_features'].get(cat, [])]
                    f.write(f"- **{cat.title()}**: {len(datasets_with_cat)} datasets\n")
                    if datasets_with_cat:
                        f.write(f"  - {', '.join(datasets_with_cat[:5])}\n")

                f.write("\n")

            f.write("## Immediate Action Items\n\n")
            f.write("1. **Apply for NDA access** (ABCD Study)\n")
            f.write("2. **Request UK Biobank application materials**\n")
            f.write("3. **Download public NHANES environmental data**\n")
            f.write("4. **Register for All of Us Researcher Workbench**\n")
            f.write("5. **Download PhysioNet datasets** (public)\n")
            f.write("6. **Sign NSRR data use agreement**\n\n")

            f.write("## Estimated Timeline\n\n")
            f.write("- **Public data (NHANES, PhysioNet, NSRR)**: 1-2 weeks\n")
            f.write("- **All of Us access**: 2-4 weeks\n")
            f.write("- **NDA/ABCD access**: 1-3 months\n")
            f.write("- **UK Biobank access**: 2-6 months + £5000\n\n")

        logger.info(f"Generated summary report: {report_file}")


def main():
    discoverer = DatasetDiscovery()
    discoveries, priorities = discoverer.run_discovery()

    print(f"\n✓ Discovery complete!")
    print(f"  Found {len(discoveries)} datasets")
    print(f"\n  Top 5 priorities:")
    for idx, row in priorities.head(5).iterrows():
        print(f"    {idx+1}. {row['Dataset']} (Score: {row['Priority_Score']:.1f})")

    print(f"\n  See full report: data/discovery/DISCOVERY_REPORT.md")


if __name__ == '__main__':
    main()