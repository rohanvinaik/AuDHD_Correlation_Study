#!/usr/bin/env python3
"""
Biobank & Biospecimen Discovery System
Identifies stored biological samples from ASD/ADHD cohorts
that could be analyzed for missing biomarkers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BiobankDiscovery:
    """Discover and catalog ASD/ADHD biospecimens"""

    def __init__(self, output_dir='data/biobanks'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Major biobanks with ASD/ADHD samples
        self.biobanks = {
            'NIMH_Repository': {
                'full_name': 'NIMH Repository and Genomics Resource (RGR)',
                'url': 'https://www.nimhgenetics.org/',
                'focus': 'Genetic studies of mental health disorders',
                'asd_adhd_samples': {
                    'dna': {'estimated_samples': 10000, 'source': 'Transformed cell lines and DNA'},
                    'lymphoblastoid_cell_lines': {'estimated_samples': 5000, 'renewable': True},
                    'plasma': {'estimated_samples': 'Variable', 'depends_on': 'Contributing study'},
                    'serum': {'estimated_samples': 'Variable', 'depends_on': 'Contributing study'}
                },
                'key_collections': [
                    'Autism Genetic Resource Exchange (AGRE)',
                    'Autism Sequencing Consortium (ASC)',
                    'ADHD molecular genetics studies'
                ],
                'access_process': {
                    'application_required': True,
                    'irb_approval': True,
                    'cost': 'Variable, typically $50-500 per sample',
                    'turnaround': '4-8 weeks',
                    'url': 'https://www.nimhgenetics.org/available_data/request_biomaterials/'
                },
                'data_sharing': {
                    'paired_clinical_data': True,
                    'phenotype_depth': 'Basic demographics, diagnosis, sometimes ADI-R/ADOS',
                    'genomics': 'Often includes WGS or array data'
                },
                'sample_types_available': ['DNA', 'Cell lines', 'Limited plasma/serum'],
                'feasibility': {
                    'cortisol_rhythm': 'LOW - Fresh samples needed',
                    'heavy_metals': 'NONE - No hair/nail samples',
                    'proteomics': 'MEDIUM - Limited plasma/serum',
                    'metabolomics': 'LOW - Sample age/storage concerns',
                    'genomics': 'HIGH - Primary resource',
                    'transcriptomics': 'MEDIUM - Cell lines available'
                }
            },

            'Autism_BrainNet': {
                'full_name': 'Autism BrainNet (Brain Tissue Repository)',
                'url': 'https://www.autismbrainnet.org/',
                'focus': 'Postmortem brain tissue for autism research',
                'asd_adhd_samples': {
                    'brain_tissue': {'estimated_samples': 200, 'regions': 'Multiple cortical/subcortical'},
                    'csf': {'estimated_samples': 100, 'volume_ul': 'Variable'},
                    'blood': {'estimated_samples': 150, 'type': 'Postmortem'},
                    'urine': {'estimated_samples': 50, 'volume_ml': 'Variable'}
                },
                'key_collections': [
                    'Autism Tissue Program (ATP)',
                    'Developmental Brain Collection',
                    'Control tissue'
                ],
                'access_process': {
                    'application_required': True,
                    'irb_approval': True,
                    'scientific_review': True,
                    'cost': 'Tissue processing and shipping fees (~$500-2000)',
                    'turnaround': '8-16 weeks',
                    'url': 'https://www.autismbrainnet.org/for-researchers/'
                },
                'data_sharing': {
                    'paired_clinical_data': True,
                    'phenotype_depth': 'Detailed clinical history, medication, cause of death',
                    'neuropathology': 'Available for brain samples'
                },
                'sample_types_available': ['Brain tissue', 'CSF', 'Postmortem blood/urine'],
                'feasibility': {
                    'cortisol_rhythm': 'NONE - Postmortem samples',
                    'heavy_metals': 'LOW - Postmortem, but hair may be available',
                    'proteomics': 'MEDIUM - CSF is valuable',
                    'metabolomics': 'LOW - Postmortem changes',
                    'brain_proteomics': 'HIGH - Primary resource',
                    'neuropathology': 'HIGH - Unique resource'
                }
            },

            'NICHD_DASH': {
                'full_name': 'NICHD Data and Specimen Hub (DASH)',
                'url': 'https://dash.nichd.nih.gov/',
                'focus': 'Repository for NICHD-funded studies',
                'asd_adhd_samples': {
                    'biospecimens': {'status': 'Study-dependent', 'note': 'Varies by contributing study'},
                    'dna': {'availability': 'Select studies'},
                    'serum_plasma': {'availability': 'Select studies'}
                },
                'key_collections': [
                    'Early Autism Risk Longitudinal Investigation (EARLI)',
                    'Study to Explore Early Development (SEED)',
                    'Children\'s Environmental Health studies'
                ],
                'access_process': {
                    'application_required': True,
                    'irb_approval': True,
                    'data_use_agreement': True,
                    'cost': 'Variable by study',
                    'turnaround': '4-12 weeks',
                    'url': 'https://dash.nichd.nih.gov/study'
                },
                'data_sharing': {
                    'paired_clinical_data': True,
                    'phenotype_depth': 'Study-dependent, often extensive',
                    'environmental_data': 'Some studies include exposure data'
                },
                'sample_types_available': ['Study-dependent', 'Check individual studies'],
                'feasibility': {
                    'note': 'Must check individual studies - EARLI and SEED most promising',
                    'environmental_biomarkers': 'HIGH - Environmental health focus',
                    'maternal_samples': 'HIGH - Prenatal/perinatal focus'
                }
            },

            'NeuroBioBank': {
                'full_name': 'NIH NeuroBioBank',
                'url': 'https://neurobiobank.nih.gov/',
                'focus': 'Brain tissue and associated biospecimens',
                'asd_adhd_samples': {
                    'brain_tissue': {'estimated_samples': 100, 'fixation': 'Fresh frozen and fixed'},
                    'csf': {'estimated_samples': 50, 'volume_ul': 'Variable'},
                    'blood_derivatives': {'estimated_samples': 80, 'types': 'Serum, plasma, DNA'}
                },
                'consortium_sites': [
                    'University of Miami',
                    'Mount Sinai NIH Brain and Tissue Repository',
                    'University of Maryland',
                    'University of Pittsburgh'
                ],
                'access_process': {
                    'application_required': True,
                    'irb_approval': True,
                    'cost': 'Tissue processing fees (~$500-1500)',
                    'turnaround': '6-12 weeks',
                    'url': 'https://neurobiobank.nih.gov/researchers/'
                },
                'data_sharing': {
                    'paired_clinical_data': True,
                    'phenotype_depth': 'Medical history, medications, neuropathology',
                    'imaging': 'Sometimes available'
                },
                'sample_types_available': ['Brain tissue', 'CSF', 'DNA', 'Serum/plasma'],
                'feasibility': {
                    'cortisol_rhythm': 'NONE - Postmortem',
                    'brain_markers': 'HIGH - Primary resource',
                    'proteomics_csf': 'HIGH - CSF available',
                    'genomics': 'HIGH - DNA available'
                }
            },

            'SPARK_Biobank': {
                'full_name': 'SPARK (Simons Foundation Powering Autism Research)',
                'url': 'https://sparkforautism.org/',
                'focus': 'Large-scale autism genetics and phenotyping',
                'asd_adhd_samples': {
                    'saliva_dna': {'estimated_samples': 100000, 'type': 'Saliva kits'},
                    'dna': {'estimated_samples': 100000, 'extraction': 'Complete'},
                    'potential_recall': {'note': 'Can contact participants for additional samples'}
                },
                'key_features': [
                    'Largest autism genetic cohort',
                    'Ongoing recruitment',
                    'Rich phenotyping via online surveys'
                ],
                'access_process': {
                    'application_required': True,
                    'irb_approval': True,
                    'scientific_review': True,
                    'cost': 'Free for approved projects',
                    'turnaround': '8-12 weeks',
                    'url': 'https://base.sfari.org/spark'
                },
                'data_sharing': {
                    'paired_clinical_data': True,
                    'phenotype_depth': 'Detailed questionnaires, some ADOS/ADI-R',
                    'genomics': 'WGS for many participants'
                },
                'sample_types_available': ['Saliva', 'DNA'],
                'feasibility': {
                    'genomics': 'HIGH - Primary resource',
                    'saliva_biomarkers': 'MEDIUM - Saliva samples stored',
                    'recall_for_new_samples': 'HIGH - Active cohort',
                    'cortisol_saliva': 'POTENTIAL - Could propose recall study'
                }
            },

            'ABCD_Biospecimens': {
                'full_name': 'ABCD Study Biospecimen Repository',
                'url': 'https://abcdstudy.org/',
                'focus': 'Adolescent Brain Cognitive Development (includes ADHD)',
                'asd_adhd_samples': {
                    'saliva': {'estimated_samples': 11000, 'collection': 'Baseline + followups'},
                    'hair': {'estimated_samples': 5000, 'length_cm': 3, 'note': 'Subset of participants'},
                    'baby_teeth': {'estimated_samples': 2000, 'note': 'For environmental exposure'}
                },
                'key_features': [
                    'Longitudinal design',
                    'Environmental exposure focus',
                    'Rich neuroimaging',
                    'Includes typically developing for comparison'
                ],
                'access_process': {
                    'application_required': True,
                    'irb_approval': True,
                    'nda_access': 'Required through NIMH Data Archive',
                    'cost': 'Free data access; biospecimen costs TBD',
                    'turnaround': '8-16 weeks',
                    'url': 'https://nda.nih.gov/abcd'
                },
                'data_sharing': {
                    'paired_clinical_data': True,
                    'phenotype_depth': 'Extensive - imaging, cognitive, environmental',
                    'environmental_data': 'Residential history, exposures'
                },
                'sample_types_available': ['Saliva', 'Hair', 'Baby teeth'],
                'feasibility': {
                    'genomics': 'HIGH - Saliva DNA',
                    'heavy_metals_hair': 'HIGH - Hair samples available',
                    'developmental_exposures': 'HIGH - Baby teeth',
                    'salivary_biomarkers': 'MEDIUM - Depends on storage',
                    'cortisol_retrospective': 'HIGH - Hair cortisol possible'
                }
            },

            'All_of_Us_Biobank': {
                'full_name': 'All of Us Research Program Biobank',
                'url': 'https://allofus.nih.gov/',
                'focus': 'Precision medicine cohort (1M+ participants)',
                'asd_adhd_samples': {
                    'blood': {'estimated_samples': 5000, 'note': 'Self-reported ASD/ADHD'},
                    'urine': {'estimated_samples': 5000, 'volume_ml': 10},
                    'saliva': {'estimated_samples': 2000, 'backup': 'If blood unavailable'}
                },
                'key_features': [
                    'Diverse population',
                    'EHR linkage',
                    'Ongoing collection',
                    'Broad consent for future research'
                ],
                'access_process': {
                    'application_required': True,
                    'data_use_agreement': True,
                    'cost': 'Free for approved researchers',
                    'turnaround': '4-8 weeks',
                    'url': 'https://www.researchallofus.org/'
                },
                'data_sharing': {
                    'paired_clinical_data': True,
                    'phenotype_depth': 'EHR, surveys, genomics, wearables',
                    'ehr_linkage': 'Longitudinal EHR data'
                },
                'sample_types_available': ['Blood', 'Urine', 'Saliva'],
                'feasibility': {
                    'proteomics': 'HIGH - Fresh plasma/serum',
                    'metabolomics': 'HIGH - Urine and blood',
                    'genomics': 'HIGH - WGS available',
                    'inflammatory_markers': 'HIGH - Plasma available',
                    'note': 'Smaller ASD/ADHD N than specialized cohorts'
                }
            }
        }

        # Assay requirements (minimum sample needed)
        self.assay_requirements = {
            'cortisol_rhythm_saliva': {
                'sample_type': 'saliva',
                'volume_ul': 100,
                'samples_needed': 4,  # Morning, afternoon, evening, bedtime
                'storage': 'Frozen -80C',
                'cost_per_sample': 25,
                'vendor': 'Salimetrics'
            },
            'cortisol_hair': {
                'sample_type': 'hair',
                'length_cm': 3,
                'weight_mg': 10,
                'storage': 'Room temperature',
                'cost_per_sample': 50,
                'vendor': 'Multiple labs'
            },
            'heavy_metals_ICP-MS': {
                'sample_types': ['hair', 'nail', 'blood', 'urine'],
                'hair': {'length_cm': 3, 'weight_mg': 50},
                'nail': {'weight_mg': 100},
                'blood': {'volume_ul': 500},
                'urine': {'volume_ml': 10},
                'cost_per_sample': 150,
                'vendor': 'Mayo Clinic Labs, LabCorp'
            },
            'proteomics_somascan': {
                'sample_types': ['serum', 'plasma', 'csf'],
                'serum': {'volume_ul': 150},
                'plasma': {'volume_ul': 150},
                'csf': {'volume_ul': 100},
                'cost_per_sample': 750,
                'vendor': 'SomaLogic',
                'proteins_measured': 7000
            },
            'metabolomics_broad': {
                'sample_types': ['serum', 'plasma', 'urine'],
                'serum': {'volume_ul': 100},
                'plasma': {'volume_ul': 100},
                'urine': {'volume_ml': 5},
                'cost_per_sample': 400,
                'vendor': 'Broad Institute, Metabolon',
                'metabolites_measured': 1000
            },
            'inflammatory_markers': {
                'sample_types': ['serum', 'plasma'],
                'volume_ul': 200,
                'markers': ['CRP', 'IL-6', 'IL-1β', 'TNF-α', 'IL-10'],
                'cost_per_sample': 150,
                'vendor': 'R&D Systems, Meso Scale Discovery'
            },
            'microbiome_16S': {
                'sample_types': ['stool', 'saliva'],
                'stool': {'weight_mg': 200},
                'saliva': {'volume_ml': 2},
                'cost_per_sample': 100,
                'vendor': 'Microbiome centers'
            },
            'exosomes': {
                'sample_types': ['plasma', 'urine', 'csf'],
                'plasma': {'volume_ul': 500},
                'urine': {'volume_ml': 10},
                'csf': {'volume_ul': 250},
                'cost_per_sample': 300,
                'vendor': 'Exosome isolation + proteomics'
            }
        }

    def calculate_assay_feasibility(self, biobank_name: str,
                                   desired_assays: List[str],
                                   n_participants: int = 100) -> Dict:
        """
        Calculate feasibility of running desired assays with available samples
        """
        biobank = self.biobanks.get(biobank_name, {})
        if not biobank:
            return {'error': f'Biobank {biobank_name} not found'}

        feasibility = {
            'biobank': biobank_name,
            'n_participants': n_participants,
            'assays': {}
        }

        for assay in desired_assays:
            if assay not in self.assay_requirements:
                feasibility['assays'][assay] = {
                    'feasible': False,
                    'reason': 'Assay not in database'
                }
                continue

            requirements = self.assay_requirements[assay]
            available_samples = biobank.get('asd_adhd_samples', {})

            # Check if required sample type is available
            required_types = requirements.get('sample_types', [requirements.get('sample_type')])

            matches = []
            for req_type in required_types:
                if req_type in available_samples:
                    matches.append(req_type)

            if matches:
                # Calculate cost
                total_cost = requirements.get('cost_per_sample', 0) * n_participants

                feasibility['assays'][assay] = {
                    'feasible': True,
                    'sample_types_available': matches,
                    'cost_total': total_cost,
                    'cost_per_sample': requirements.get('cost_per_sample', 0),
                    'vendor': requirements.get('vendor', 'Various'),
                    'considerations': biobank.get('feasibility', {}).get(assay, 'Check sample quality')
                }
            else:
                feasibility['assays'][assay] = {
                    'feasible': False,
                    'reason': f"Required sample types {required_types} not available",
                    'alternatives': 'Consider different biobank or sample type'
                }

        return feasibility

    def generate_sample_request_template(self, biobank_name: str) -> str:
        """Generate a sample request template"""
        biobank = self.biobanks.get(biobank_name, {})

        template = f"""
# SAMPLE REQUEST TEMPLATE: {biobank.get('full_name', biobank_name)}

## Contact Information
- Website: {biobank.get('url', 'N/A')}
- Application URL: {biobank.get('access_process', {}).get('url', 'N/A')}

## PROJECT INFORMATION

**Title:** [Your project title]

**Principal Investigator:** [Name, Institution]

**Funding:** [NIH grant number or other funding source]

**IRB Approval:** [Your institution IRB approval number and date]

## SAMPLE REQUEST

**Diagnosis Groups:**
- [ ] Autism Spectrum Disorder (ASD): n = ___
- [ ] ADHD: n = ___
- [ ] ASD + ADHD comorbid: n = ___
- [ ] Typically developing controls: n = ___

**Age Range:** ___ to ___ years

**Sex:**
- [ ] Male: n = ___
- [ ] Female: n = ___
- [ ] No preference

**Sample Types Requested:**
{self._format_available_samples(biobank)}

**Clinical Data Requested:**
- [ ] Demographics (age, sex, race/ethnicity)
- [ ] Diagnosis confirmation (ADI-R, ADOS, DSM criteria)
- [ ] Cognitive assessments (IQ, adaptive function)
- [ ] Medical history
- [ ] Medication history
- [ ] Family history
- [ ] [Other: specify]

## SCIENTIFIC JUSTIFICATION

**Research Question:**
[1-2 sentences on what you're investigating]

**Specific Aims:**
1. [Aim 1]
2. [Aim 2]
3. [Aim 3]

**Assays Planned:**
[List specific assays, platforms, and laboratories]

**Expected Outcomes:**
[How results will advance the field]

## SAMPLE HANDLING

**Shipping Requirements:**
{self._format_shipping_requirements(biobank)}

**Storage Upon Receipt:**
[Your lab's storage conditions]

**Sample Tracking:**
[Your sample tracking and chain-of-custody procedures]

## BUDGET

**Estimated Costs:**
- Sample acquisition: ${biobank.get('access_process', {}).get('cost', 'TBD')}
- Shipping: $___
- Assay costs: $___
- Total: $___

**Funding Status:**
- [ ] Funded (grant number: ___)
- [ ] Pending (grant submission date: ___)
- [ ] Institutional support

## COMPLIANCE

**IRB Status:** [Approved / Pending - Include approval letter]

**Data Use Agreements:** [Will sign as required]

**Data Sharing Plan:**
[Your plan for sharing results, including deposition in public databases]

**Authorship:**
[Acknowledgment of biobank in publications]

## TIMELINE

- Application submission: [Date]
- Expected approval: [Date]
- Sample shipment: [Date]
- Assay completion: [Date]
- Results analysis: [Date]

## ADDITIONAL INFORMATION

[Any additional information specific to your project]

---

**Signature:** ___________________  **Date:** __________

**PI Name:** ___________________

**Institution:** ___________________
"""
        return template

    def _format_available_samples(self, biobank: Dict) -> str:
        """Format available sample types"""
        samples = biobank.get('sample_types_available', [])
        if isinstance(samples, list):
            return '\n'.join([f"- [ ] {s}" for s in samples])
        return "- [ ] [See biobank for available types]"

    def _format_shipping_requirements(self, biobank: Dict) -> str:
        """Format shipping requirements"""
        return """
- Dry ice shipment required: Yes/No
- Temperature monitoring: Yes/No
- Chain of custody documentation: Yes/No
- Receiving hours: [Specify]
"""

    def generate_multi_site_coordination_plan(self, biobanks: List[str],
                                              assays: List[str]) -> Dict:
        """Generate plan for coordinating samples from multiple biobanks"""

        plan = {
            'biobanks': biobanks,
            'assays': assays,
            'coordination': {},
            'total_budget': 0,
            'timeline': {}
        }

        for biobank_name in biobanks:
            feasibility = self.calculate_assay_feasibility(
                biobank_name, assays, n_participants=100
            )

            plan['coordination'][biobank_name] = feasibility

            # Calculate total cost
            for assay_name, assay_info in feasibility.get('assays', {}).items():
                if assay_info.get('feasible'):
                    plan['total_budget'] += assay_info.get('cost_total', 0)

        # Timeline
        plan['timeline'] = {
            'Month_0-2': 'Prepare and submit applications to all biobanks',
            'Month_2-4': 'IRB approvals and data use agreements',
            'Month_4-6': 'Sample requests and shipments',
            'Month_6-12': 'Assay processing',
            'Month_12-18': 'Data analysis and integration'
        }

        # Harmonization strategy
        plan['harmonization'] = {
            'vendor_selection': 'Use same vendor/platform for all sites to minimize batch effects',
            'sample_processing': 'Standardize processing protocols across biobanks',
            'quality_control': 'Include cross-site reference samples',
            'data_integration': 'Harmonize phenotype definitions across cohorts',
            'statistical_adjustment': 'Include biobank as random effect in models'
        }

        return plan

    def generate_report(self):
        """Generate comprehensive biobank discovery report"""
        logger.info("Generating biobank discovery report...")

        report_path = self.output_dir / 'BIOBANK_DISCOVERY_REPORT.md'

        with open(report_path, 'w') as f:
            f.write("# ASD/ADHD Biobank Discovery Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d')}\n\n")

            f.write("## Summary\n\n")
            f.write(f"**Total biobanks identified:** {len(self.biobanks)}\n\n")

            # Count sample types
            genetic = sum(1 for b in self.biobanks.values()
                         if 'dna' in b.get('asd_adhd_samples', {}))
            fluid = sum(1 for b in self.biobanks.values()
                       if any(k in b.get('asd_adhd_samples', {})
                             for k in ['serum', 'plasma', 'csf', 'urine']))
            tissue = sum(1 for b in self.biobanks.values()
                        if 'brain_tissue' in b.get('asd_adhd_samples', {}))

            f.write(f"- **With genetic samples:** {genetic}\n")
            f.write(f"- **With fluid samples:** {fluid}\n")
            f.write(f"- **With tissue samples:** {tissue}\n\n")

            # Detailed biobank descriptions
            f.write("## Biobank Catalog\n\n")

            for biobank_name, biobank_info in self.biobanks.items():
                f.write(f"### {biobank_info.get('full_name', biobank_name)}\n\n")
                f.write(f"**URL:** {biobank_info.get('url', 'N/A')}\n\n")
                f.write(f"**Focus:** {biobank_info.get('focus', 'N/A')}\n\n")

                f.write("**Available Sample Types:**\n")
                samples = biobank_info.get('asd_adhd_samples', {})
                for sample_type, details in samples.items():
                    if isinstance(details, dict):
                        f.write(f"- **{sample_type.replace('_', ' ').title()}:** ")
                        if 'estimated_samples' in details:
                            f.write(f"~{details['estimated_samples']} samples")
                        if 'volume_ul' in details:
                            f.write(f", {details['volume_ul']} µL")
                        if 'note' in details:
                            f.write(f" ({details['note']})")
                        f.write("\n")
                f.write("\n")

                f.write("**Access Process:**\n")
                access = biobank_info.get('access_process', {})
                f.write(f"- Application required: {access.get('application_required', 'Unknown')}\n")
                f.write(f"- IRB approval: {access.get('irb_approval', 'Unknown')}\n")
                f.write(f"- Cost: {access.get('cost', 'Unknown')}\n")
                f.write(f"- Turnaround: {access.get('turnaround', 'Unknown')}\n")
                f.write(f"- Apply: {access.get('url', 'Contact biobank')}\n\n")

                f.write("**Feasibility for Key Assays:**\n")
                feasibility = biobank_info.get('feasibility', {})
                for assay, rating in feasibility.items():
                    f.write(f"- {assay.replace('_', ' ').title()}: {rating}\n")
                f.write("\n---\n\n")

            # Assay requirements
            f.write("## Assay Requirements Reference\n\n")
            for assay_name, requirements in self.assay_requirements.items():
                f.write(f"### {assay_name.replace('_', ' ').title()}\n\n")
                if 'sample_types' in requirements:
                    f.write(f"**Compatible sample types:** {', '.join(requirements['sample_types'])}\n\n")
                elif 'sample_type' in requirements:
                    f.write(f"**Required sample type:** {requirements['sample_type']}\n\n")

                f.write(f"**Cost per sample:** ${requirements.get('cost_per_sample', 'N/A')}\n\n")
                f.write(f"**Vendor:** {requirements.get('vendor', 'Various')}\n\n")

                # Sample requirements
                for key, value in requirements.items():
                    if key not in ['sample_types', 'sample_type', 'cost_per_sample', 'vendor'] and isinstance(value, dict):
                        f.write(f"**{key.title()} requirements:**\n")
                        for req_key, req_val in value.items():
                            f.write(f"  - {req_key}: {req_val}\n")
                        f.write("\n")
                f.write("\n")

        logger.info(f"✓ Report saved to {report_path}")
        return report_path

    def save_catalog(self):
        """Save biobank catalog to JSON"""
        catalog_path = self.output_dir / 'biobank_catalog.json'

        with open(catalog_path, 'w') as f:
            json.dump({
                'generation_date': datetime.now().isoformat(),
                'biobanks': self.biobanks,
                'assay_requirements': self.assay_requirements
            }, f, indent=2)

        logger.info(f"✓ Catalog saved to {catalog_path}")
        return catalog_path


def main():
    """Generate biobank discovery deliverables"""
    logger.info("\n" + "="*60)
    logger.info("BIOBANK & BIOSPECIMEN DISCOVERY")
    logger.info("="*60 + "\n")

    discovery = BiobankDiscovery()

    # Generate main report
    discovery.generate_report()

    # Save catalog
    discovery.save_catalog()

    # Example: Calculate feasibility for key assays at each biobank
    logger.info("\nCalculating assay feasibility...")

    key_assays = [
        'cortisol_hair',
        'heavy_metals_ICP-MS',
        'proteomics_somascan',
        'metabolomics_broad',
        'inflammatory_markers'
    ]

    feasibility_results = {}
    for biobank_name in discovery.biobanks.keys():
        feasibility = discovery.calculate_assay_feasibility(
            biobank_name,
            key_assays,
            n_participants=100
        )
        feasibility_results[biobank_name] = feasibility

    # Save feasibility analysis
    feasibility_path = discovery.output_dir / 'assay_feasibility_analysis.json'
    with open(feasibility_path, 'w') as f:
        json.dump(feasibility_results, f, indent=2)

    logger.info(f"✓ Feasibility analysis saved to {feasibility_path}")

    # Generate sample request templates
    logger.info("\nGenerating sample request templates...")
    templates_dir = discovery.output_dir / 'sample_request_templates'
    templates_dir.mkdir(exist_ok=True)

    for biobank_name in discovery.biobanks.keys():
        template = discovery.generate_sample_request_template(biobank_name)
        template_path = templates_dir / f"{biobank_name}_REQUEST_TEMPLATE.md"
        with open(template_path, 'w') as f:
            f.write(template)
        logger.info(f"  ✓ {biobank_name} template")

    # Generate multi-site coordination plan
    logger.info("\nGenerating multi-site coordination plan...")

    # Scenario: Use SPARK for genomics, ABCD for environmental, All of Us for proteomics
    multi_site_plan = discovery.generate_multi_site_coordination_plan(
        biobanks=['SPARK_Biobank', 'ABCD_Biospecimens', 'All_of_Us_Biobank'],
        assays=key_assays
    )

    plan_path = discovery.output_dir / 'multi_site_coordination_plan.json'
    with open(plan_path, 'w') as f:
        json.dump(multi_site_plan, f, indent=2)

    logger.info(f"✓ Multi-site plan saved to {plan_path}")

    # Budget summary
    logger.info("\n" + "="*60)
    logger.info("BUDGET SUMMARY (Example: 100 participants per biobank)")
    logger.info("="*60)
    logger.info(f"\nTotal estimated cost: ${multi_site_plan['total_budget']:,.2f}")

    for biobank, data in multi_site_plan['coordination'].items():
        logger.info(f"\n{biobank}:")
        for assay, info in data.get('assays', {}).items():
            if info.get('feasible'):
                cost = info.get('cost_total', 0)
                logger.info(f"  {assay}: ${cost:,.2f}")

    logger.info("\n" + "="*60)
    logger.info("BIOBANK DISCOVERY COMPLETE")
    logger.info("="*60)
    logger.info("\nGenerated files:")
    logger.info(f"  1. {discovery.output_dir}/BIOBANK_DISCOVERY_REPORT.md")
    logger.info(f"  2. {discovery.output_dir}/biobank_catalog.json")
    logger.info(f"  3. {discovery.output_dir}/assay_feasibility_analysis.json")
    logger.info(f"  4. {discovery.output_dir}/sample_request_templates/")
    logger.info(f"  5. {discovery.output_dir}/multi_site_coordination_plan.json")
    logger.info("\n")


if __name__ == '__main__':
    main()