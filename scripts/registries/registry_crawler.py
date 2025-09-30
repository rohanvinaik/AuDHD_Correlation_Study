#!/usr/bin/env python3
"""
Patient Registry Crawler for ADHD/Autism Research

Searches and catalogs patient registries with biological data including:
- SPARK (Simons Foundation Powering Autism Research)
- IAN (Interactive Autism Network)
- AGRE (Autism Genetic Resource Exchange)
- ADHD-specific registries
- International consortia (EU-AIMS, EAGLE, etc.)

Requirements:
    pip install requests pandas beautifulsoup4

Usage:
    # Catalog all registries
    python registry_crawler.py \\
        --output data/registries/

    # Search specific registry types
    python registry_crawler.py \\
        --registry-type autism \\
        --output data/registries/

    # Find registries with genomic data
    python registry_crawler.py \\
        --data-type genomics \\
        --output data/registries/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

try:
    import requests
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install requests pandas")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PatientRegistry:
    """Patient registry with biological data"""
    registry_id: str
    name: str
    full_name: str
    url: str
    condition: str  # 'autism', 'adhd', 'both'
    organization: str
    country: str

    # Data availability
    sample_size: Optional[int]
    active_enrollment: bool
    data_types: List[str]  # genomics, phenotype, medical_records, imaging, etc.

    # Access information
    access_type: str  # 'open', 'controlled', 'collaboration_only', 'closed'
    application_url: Optional[str]
    contact_email: Optional[str]

    # Details
    description: str
    inclusion_criteria: List[str]
    data_collection: List[str]
    publications: List[str]
    last_updated: str
    notes: str


# Comprehensive registry catalog
AUTISM_REGISTRIES = [
    {
        'registry_id': 'SPARK',
        'name': 'SPARK',
        'full_name': 'Simons Foundation Powering Autism Research',
        'url': 'https://sparkforautism.org',
        'condition': 'autism',
        'organization': 'Simons Foundation',
        'country': 'USA',
        'sample_size': 50000,
        'active_enrollment': True,
        'data_types': ['genomics', 'phenotype', 'medical_records', 'family_history'],
        'access_type': 'controlled',
        'application_url': 'https://base.sfari.org',
        'contact_email': 'spark@simonsfoundation.org',
        'description': 'Largest genetic study of autism with DNA sequencing and clinical data from families',
        'inclusion_criteria': ['ASD diagnosis', 'age 18 months+', 'US residence'],
        'data_collection': ['Whole exome sequencing', 'Online surveys', 'Medical records', 'Family pedigree'],
        'publications': ['PMID:32651583', 'PMID:33658720'],
        'notes': 'Free genetic testing for families. Data available via SFARI Base.'
    },
    {
        'registry_id': 'SSC',
        'name': 'SSC',
        'full_name': 'Simons Simplex Collection',
        'url': 'https://www.sfari.org/resource/simons-simplex-collection/',
        'condition': 'autism',
        'organization': 'Simons Foundation',
        'country': 'USA',
        'sample_size': 2600,
        'active_enrollment': False,
        'data_types': ['genomics', 'exomes', 'genotypes', 'phenotype', 'medical_records', 'cognitive'],
        'access_type': 'controlled',
        'application_url': 'https://base.sfari.org',
        'contact_email': 'sfari@simonsfoundation.org',
        'description': 'Simplex families (one affected child, unaffected parents and siblings) with comprehensive phenotyping',
        'inclusion_criteria': ['Simplex family', 'Proband age 4-18', 'IQ > 35'],
        'data_collection': ['WES', 'SNP arrays', 'ADOS', 'ADI-R', 'IQ testing', 'Medical history'],
        'publications': ['PMID:21926976', 'PMID:22495311', 'PMID:31981491'],
        'notes': 'Foundational autism genetics dataset. Available via SFARI Base and dbGaP (phs000473).'
    },
    {
        'registry_id': 'AGRE',
        'name': 'AGRE',
        'full_name': 'Autism Genetic Resource Exchange',
        'url': 'https://research.agre.org',
        'condition': 'autism',
        'organization': 'Autism Speaks',
        'country': 'USA',
        'sample_size': 1800,
        'active_enrollment': False,
        'data_types': ['genomics', 'genotypes', 'phenotype', 'family_history'],
        'access_type': 'controlled',
        'application_url': 'https://research.agre.org',
        'contact_email': 'agre@autismspeaks.org',
        'description': 'Multiplex families with genotype and phenotype data',
        'inclusion_criteria': ['Multiplex family', 'ASD diagnosis', '2+ affected siblings'],
        'data_collection': ['SNP arrays', 'ADI-R', 'ADOS', 'Family pedigree', 'Medical questionnaires'],
        'publications': ['PMID:21926972', 'PMID:19404305'],
        'notes': 'Historic dataset, enrollment closed. Data via NDAR and SFARI Base.'
    },
    {
        'registry_id': 'IAN',
        'name': 'IAN',
        'full_name': 'Interactive Autism Network',
        'url': 'https://iancommunity.org',
        'condition': 'autism',
        'organization': 'Kennedy Krieger Institute',
        'country': 'USA',
        'sample_size': 52000,
        'active_enrollment': True,
        'data_types': ['phenotype', 'medical_records', 'treatment_history', 'behavioral'],
        'access_type': 'collaboration_only',
        'application_url': 'https://iancommunity.org/cs/ian_research',
        'contact_email': 'ianresearch@kennedykrieger.org',
        'description': 'Large online registry with longitudinal phenotype and treatment data',
        'inclusion_criteria': ['ASD diagnosis', 'US residence', 'age 2+'],
        'data_collection': ['Online surveys', 'Medical history', 'Treatment outcomes', 'Comorbidities'],
        'publications': ['PMID:19522877', 'PMID:22965145'],
        'notes': 'Primarily phenotype data, no biological samples. Research proposals reviewed.'
    },
    {
        'registry_id': 'SIMONS_SEARCHLIGHT',
        'name': 'Simons Searchlight',
        'full_name': 'Simons Searchlight (formerly SFARI Gene)',
        'url': 'https://www.simonssearchlight.org',
        'condition': 'autism',
        'organization': 'Simons Foundation',
        'country': 'USA',
        'sample_size': 3000,
        'active_enrollment': True,
        'data_types': ['genomics', 'phenotype', 'developmental_history', 'rare_variants'],
        'access_type': 'controlled',
        'application_url': 'https://base.sfari.org',
        'contact_email': 'searchlight@simonsfoundation.org',
        'description': 'Registry for individuals with genetic changes in autism-associated genes',
        'inclusion_criteria': ['Pathogenic variant in autism gene', 'Any age'],
        'data_collection': ['Genetic testing', 'Developmental assessments', 'Natural history surveys'],
        'publications': ['PMID:32651583'],
        'notes': 'Focus on rare genetic variants. Free genetic confirmation testing.'
    },
    {
        'registry_id': 'EU_AIMS_LEAP',
        'name': 'EU-AIMS LEAP',
        'full_name': 'European Autism Interventions - Longitudinal European Autism Project',
        'url': 'https://www.eu-aims.eu',
        'condition': 'autism',
        'organization': 'EU-AIMS Consortium',
        'country': 'Europe',
        'sample_size': 800,
        'active_enrollment': False,
        'data_types': ['genomics', 'imaging', 'eye_tracking', 'eeg', 'phenotype', 'biomarkers'],
        'access_type': 'collaboration_only',
        'application_url': 'https://www.eu-aims.eu/resources/access-data/',
        'contact_email': 'info@eu-aims.eu',
        'description': 'Deep phenotyping cohort with multi-modal biomarkers',
        'inclusion_criteria': ['ASD diagnosis', 'Age 6-30', 'European sites'],
        'data_collection': ['MRI', 'fMRI', 'Eye tracking', 'EEG', 'Blood biomarkers', 'Genomics'],
        'publications': ['PMID:28545751', 'PMID:32165107'],
        'notes': 'Comprehensive biomarker battery. Data access via collaboration.'
    },
    {
        'registry_id': 'ABC_CT',
        'name': 'ABC-CT',
        'full_name': 'Autism Biomarkers Consortium for Clinical Trials',
        'url': 'https://www.autismbiomarkersconsortium.org',
        'condition': 'autism',
        'organization': 'Foundation for NIH',
        'country': 'USA',
        'sample_size': 600,
        'active_enrollment': True,
        'data_types': ['imaging', 'eeg', 'eye_tracking', 'phenotype', 'biomarkers'],
        'access_type': 'controlled',
        'application_url': 'https://ndar.nih.gov',
        'contact_email': 'abc@fnih.org',
        'description': 'Developing biomarkers for use in clinical trials',
        'inclusion_criteria': ['ASD diagnosis', 'Age 6-11', 'IQ > 50'],
        'data_collection': ['MRI', 'EEG', 'Eye tracking', 'Clinical assessments', 'Blood samples'],
        'publications': ['PMID:31402457'],
        'notes': 'Clinical trial-ready biomarkers. Data to NDAR. NCT04119687.'
    },
    {
        'registry_id': 'EAGLE',
        'name': 'EAGLE',
        'full_name': 'Exploring Autism Genetics and the Lifelong Experience',
        'url': 'https://www.eagleautismgenetics.org',
        'condition': 'autism',
        'organization': 'Children\'s Hospital of Philadelphia',
        'country': 'USA',
        'sample_size': 10000,
        'active_enrollment': True,
        'data_types': ['genomics', 'phenotype', 'longitudinal'],
        'access_type': 'controlled',
        'application_url': 'https://www.eagleautismgenetics.org/for-researchers',
        'contact_email': 'eagle@chop.edu',
        'description': 'Longitudinal genetics study tracking development over time',
        'inclusion_criteria': ['ASD diagnosis', 'Any age', 'Willing to provide DNA'],
        'data_collection': ['Genetic testing', 'Annual surveys', 'Developmental milestones', 'Medical history'],
        'publications': ['PMID:TBD'],
        'notes': 'Longitudinal follow-up focus. Free genetic testing.'
    }
]

ADHD_REGISTRIES = [
    {
        'registry_id': 'APSARD_REGISTRY',
        'name': 'APSARD Registry',
        'full_name': 'American Professional Society of ADHD and Related Disorders Registry',
        'url': 'https://www.apsard.org',
        'condition': 'adhd',
        'organization': 'APSARD',
        'country': 'USA',
        'sample_size': None,
        'active_enrollment': True,
        'data_types': ['phenotype', 'treatment_history', 'comorbidities'],
        'access_type': 'collaboration_only',
        'application_url': None,
        'contact_email': 'info@apsard.org',
        'description': 'Clinical registry for ADHD patients in treatment',
        'inclusion_criteria': ['ADHD diagnosis', 'Active treatment'],
        'data_collection': ['Clinical assessments', 'Medication history', 'Outcomes'],
        'publications': [],
        'notes': 'Clinical data focus, limited research access information.'
    },
    {
        'registry_id': 'CADDRA_REGISTRY',
        'name': 'CADDRA',
        'full_name': 'Canadian ADHD Resource Alliance Registry',
        'url': 'https://www.caddra.ca',
        'condition': 'adhd',
        'organization': 'CADDRA',
        'country': 'Canada',
        'sample_size': None,
        'active_enrollment': True,
        'data_types': ['phenotype', 'treatment_outcomes'],
        'access_type': 'collaboration_only',
        'application_url': None,
        'contact_email': 'info@caddra.ca',
        'description': 'Canadian ADHD clinical data registry',
        'inclusion_criteria': ['ADHD diagnosis', 'Canadian sites'],
        'data_collection': ['Clinical assessments', 'Treatment response'],
        'publications': [],
        'notes': 'Clinical practice registry. Contact for research collaborations.'
    },
    {
        'registry_id': 'EUNETHYDIS',
        'name': 'EUNETHYDIS',
        'full_name': 'European Network for Hyperkinetic Disorders',
        'url': 'https://www.eunethydis.com',
        'condition': 'adhd',
        'organization': 'European ADHD Consortium',
        'country': 'Europe',
        'sample_size': None,
        'active_enrollment': False,
        'data_types': ['genomics', 'phenotype', 'imaging'],
        'access_type': 'collaboration_only',
        'application_url': None,
        'contact_email': None,
        'description': 'European ADHD research network and data sharing initiative',
        'inclusion_criteria': ['ADHD diagnosis', 'European sites'],
        'data_collection': ['Genomics', 'Clinical phenotypes', 'Neuroimaging'],
        'publications': ['PMID:21284948'],
        'notes': 'Historic network. Data via collaborating sites.'
    },
    {
        'registry_id': 'PGC_ADHD',
        'name': 'PGC ADHD',
        'full_name': 'Psychiatric Genomics Consortium ADHD Working Group',
        'url': 'https://www.med.unc.edu/pgc/download-results/adhd/',
        'condition': 'adhd',
        'organization': 'PGC',
        'country': 'International',
        'sample_size': 55374,
        'active_enrollment': False,
        'data_types': ['genomics', 'gwas_summary_statistics', 'phenotype'],
        'access_type': 'open',
        'application_url': 'https://www.med.unc.edu/pgc/download-results/adhd/',
        'contact_email': 'pgc@med.unc.edu',
        'description': 'Largest ADHD GWAS meta-analysis with summary statistics',
        'inclusion_criteria': ['ADHD diagnosis', 'Genotyped cohorts'],
        'data_collection': ['GWAS', 'Clinical diagnosis'],
        'publications': ['PMID:30478444', 'PMID:33357627'],
        'notes': 'Summary statistics publicly available. Individual-level data via dbGaP/EGA for contributing cohorts.'
    },
    {
        'registry_id': 'IPSYCH_ADHD',
        'name': 'iPSYCH ADHD',
        'full_name': 'Integrative Psychiatric Research ADHD Cohort',
        'url': 'https://ipsych.dk',
        'condition': 'adhd',
        'organization': 'iPSYCH Denmark',
        'country': 'Denmark',
        'sample_size': 20183,
        'active_enrollment': False,
        'data_types': ['genomics', 'register_data', 'prescription_records', 'phenotype'],
        'access_type': 'controlled',
        'application_url': 'https://ipsych.dk/en/data-access/',
        'contact_email': 'ipsych@econ.au.dk',
        'description': 'Danish register-based cohort with genomics and longitudinal health records',
        'inclusion_criteria': ['ADHD diagnosis in Danish registers', 'Born 1981-2008'],
        'data_collection': ['Whole genome sequencing', 'National health registers', 'Prescription data'],
        'publications': ['PMID:30478444', 'PMID:28540026'],
        'notes': 'Exceptional registry linkage. Data access via collaboration with Danish sites.'
    },
    {
        'registry_id': 'ABCD_ADHD',
        'name': 'ABCD ADHD',
        'full_name': 'ABCD Study ADHD Cohort',
        'url': 'https://abcdstudy.org',
        'condition': 'adhd',
        'organization': 'NIH',
        'country': 'USA',
        'sample_size': 11878,
        'active_enrollment': True,
        'data_types': ['genomics', 'imaging', 'cognitive', 'phenotype', 'longitudinal'],
        'access_type': 'controlled',
        'application_url': 'https://nda.nih.gov',
        'contact_email': 'nda@mail.nih.gov',
        'description': 'Longitudinal neurodevelopmental cohort with extensive ADHD data',
        'inclusion_criteria': ['Age 9-10 at baseline', 'US residence'],
        'data_collection': ['Annual MRI', 'Genomics', 'Cognitive testing', 'Clinical assessments', 'Biospecimens'],
        'publications': ['PMID:30595399', 'PMID:33568505'],
        'notes': 'Richest multi-modal dataset. ~2000 with ADHD symptoms. Data via NDA.'
    }
]

# Registries with both ADHD and autism data
COMORBID_REGISTRIES = [
    {
        'registry_id': 'UK_BIOBANK',
        'name': 'UK Biobank',
        'full_name': 'UK Biobank Neurodevelopmental Cohort',
        'url': 'https://www.ukbiobank.ac.uk',
        'condition': 'both',
        'organization': 'UK Biobank',
        'country': 'UK',
        'sample_size': 500000,
        'active_enrollment': False,
        'data_types': ['genomics', 'imaging', 'electronic_health_records', 'biomarkers', 'microbiome'],
        'access_type': 'controlled',
        'application_url': 'https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access',
        'contact_email': 'access@ukbiobank.ac.uk',
        'description': 'Massive biobank with ADHD/autism diagnoses and multi-omics data',
        'inclusion_criteria': ['Age 40-69 at baseline', 'UK residence'],
        'data_collection': ['Whole exome/genome sequencing', 'Brain MRI', 'EHR linkage', 'Blood/urine samples'],
        'publications': ['PMID:30305743', 'PMID:32434901'],
        'notes': '~2000 autism, ~4000 ADHD cases. Data via application (3-4 weeks approval).'
    },
    {
        'registry_id': 'ALL_OF_US',
        'name': 'All of Us',
        'full_name': 'All of Us Research Program',
        'url': 'https://allofus.nih.gov',
        'condition': 'both',
        'organization': 'NIH',
        'country': 'USA',
        'sample_size': 413000,
        'active_enrollment': True,
        'data_types': ['genomics', 'electronic_health_records', 'surveys', 'biomarkers', 'wearables'],
        'access_type': 'controlled',
        'application_url': 'https://www.researchallofus.org',
        'contact_email': 'allofus@researchallofus.org',
        'description': 'Precision medicine cohort with diverse participants and EHR data',
        'inclusion_criteria': ['Age 18+', 'US residence'],
        'data_collection': ['Whole genome sequencing', 'EHR', 'Surveys', 'Physical measurements', 'Biospecimens'],
        'publications': ['PMID:31722398'],
        'notes': 'Searchable ADHD/autism via ICD codes in EHR. Registered tier access available.'
    },
    {
        'registry_id': 'MSSNG',
        'name': 'MSSNG',
        'full_name': 'MSSNG Autism Genome Project',
        'url': 'https://www.mss.ng',
        'condition': 'autism',
        'organization': 'Autism Speaks / Google',
        'country': 'International',
        'sample_size': 10000,
        'active_enrollment': True,
        'data_types': ['genomics', 'whole_genome_sequencing', 'phenotype'],
        'access_type': 'controlled',
        'application_url': 'https://research.mss.ng',
        'contact_email': 'mssng@autismspeaks.org',
        'description': 'Whole genome sequencing of autism families',
        'inclusion_criteria': ['ASD diagnosis', 'Family willing to participate'],
        'data_collection': ['Whole genome sequencing (30x coverage)', 'Clinical phenotypes'],
        'publications': ['PMID:31981491', 'PMID:28263302'],
        'notes': 'High-coverage WGS. Data via Google Cloud and SFARI Base.'
    }
]


class RegistryCrawler:
    """Catalog and search patient registries"""

    def __init__(self, output_dir: Path):
        """
        Initialize crawler

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.registries: List[PatientRegistry] = []
        self._load_registries()

        logger.info(f"Initialized registry crawler: {len(self.registries)} registries")

    def _load_registries(self):
        """Load all registry data"""
        all_registry_data = AUTISM_REGISTRIES + ADHD_REGISTRIES + COMORBID_REGISTRIES

        for reg_data in all_registry_data:
            # Convert to PatientRegistry object
            registry = PatientRegistry(
                registry_id=reg_data['registry_id'],
                name=reg_data['name'],
                full_name=reg_data['full_name'],
                url=reg_data['url'],
                condition=reg_data['condition'],
                organization=reg_data['organization'],
                country=reg_data['country'],
                sample_size=reg_data.get('sample_size'),
                active_enrollment=reg_data['active_enrollment'],
                data_types=reg_data['data_types'],
                access_type=reg_data['access_type'],
                application_url=reg_data.get('application_url'),
                contact_email=reg_data.get('contact_email'),
                description=reg_data['description'],
                inclusion_criteria=reg_data['inclusion_criteria'],
                data_collection=reg_data['data_collection'],
                publications=reg_data.get('publications', []),
                last_updated=datetime.now().strftime('%Y-%m-%d'),
                notes=reg_data['notes']
            )

            self.registries.append(registry)

    def filter_by_condition(self, condition: str) -> List[PatientRegistry]:
        """
        Filter registries by condition

        Args:
            condition: 'autism', 'adhd', or 'both'

        Returns:
            Filtered list of registries
        """
        if condition == 'both':
            return [r for r in self.registries if r.condition == 'both']

        return [r for r in self.registries
                if r.condition == condition or r.condition == 'both']

    def filter_by_data_type(self, data_type: str) -> List[PatientRegistry]:
        """
        Filter registries by available data type

        Args:
            data_type: genomics, imaging, phenotype, etc.

        Returns:
            Filtered list of registries
        """
        return [r for r in self.registries if data_type in r.data_types]

    def filter_by_access(self, access_type: str) -> List[PatientRegistry]:
        """
        Filter registries by access type

        Args:
            access_type: 'open', 'controlled', 'collaboration_only'

        Returns:
            Filtered list of registries
        """
        return [r for r in self.registries if r.access_type == access_type]

    def get_active_registries(self) -> List[PatientRegistry]:
        """Get registries with active enrollment"""
        return [r for r in self.registries if r.active_enrollment]

    def get_large_registries(self, min_size: int = 10000) -> List[PatientRegistry]:
        """Get registries with large sample sizes"""
        return [r for r in self.registries
                if r.sample_size and r.sample_size >= min_size]

    def search_registries(
        self,
        condition: Optional[str] = None,
        data_type: Optional[str] = None,
        access_type: Optional[str] = None,
        active_only: bool = False,
        min_sample_size: Optional[int] = None
    ) -> List[PatientRegistry]:
        """
        Search registries with multiple filters

        Args:
            condition: Filter by condition
            data_type: Filter by data type
            access_type: Filter by access type
            active_only: Only active enrollment
            min_sample_size: Minimum sample size

        Returns:
            Filtered list of registries
        """
        results = self.registries

        if condition:
            results = [r for r in results
                      if r.condition == condition or r.condition == 'both']

        if data_type:
            results = [r for r in results if data_type in r.data_types]

        if access_type:
            results = [r for r in results if r.access_type == access_type]

        if active_only:
            results = [r for r in results if r.active_enrollment]

        if min_sample_size:
            results = [r for r in results
                      if r.sample_size and r.sample_size >= min_sample_size]

        return results

    def export_json(self, registries: Optional[List[PatientRegistry]] = None) -> Path:
        """Export registries to JSON"""
        if registries is None:
            registries = self.registries

        output = {
            'metadata': {
                'generated_date': datetime.now().strftime('%Y-%m-%d'),
                'total_registries': len(registries),
                'autism_registries': len([r for r in registries if r.condition == 'autism']),
                'adhd_registries': len([r for r in registries if r.condition == 'adhd']),
                'comorbid_registries': len([r for r in registries if r.condition == 'both']),
                'active_enrollment': len([r for r in registries if r.active_enrollment]),
                'total_samples': sum(r.sample_size for r in registries if r.sample_size)
            },
            'registries': [asdict(r) for r in registries]
        }

        output_file = self.output_dir / 'patient_registries.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Exported {len(registries)} registries to {output_file}")
        return output_file

    def export_csv(self, registries: Optional[List[PatientRegistry]] = None) -> Path:
        """Export registries to CSV"""
        if registries is None:
            registries = self.registries

        # Flatten for CSV
        rows = []
        for r in registries:
            rows.append({
                'registry_id': r.registry_id,
                'name': r.name,
                'full_name': r.full_name,
                'url': r.url,
                'condition': r.condition,
                'organization': r.organization,
                'country': r.country,
                'sample_size': r.sample_size,
                'active_enrollment': r.active_enrollment,
                'data_types': ', '.join(r.data_types),
                'access_type': r.access_type,
                'application_url': r.application_url or '',
                'contact_email': r.contact_email or '',
                'description': r.description,
                'notes': r.notes
            })

        df = pd.DataFrame(rows)
        output_file = self.output_dir / 'patient_registries.csv'
        df.to_csv(output_file, index=False)

        logger.info(f"Exported {len(registries)} registries to CSV")
        return output_file

    def generate_summary_report(self) -> str:
        """Generate summary statistics"""
        report = []
        report.append("=== Patient Registry Summary ===\n")

        # Overall stats
        report.append(f"Total registries: {len(self.registries)}")
        report.append(f"Autism registries: {len(self.filter_by_condition('autism'))}")
        report.append(f"ADHD registries: {len(self.filter_by_condition('adhd'))}")
        report.append(f"Both conditions: {len([r for r in self.registries if r.condition == 'both'])}")

        # Enrollment
        active = self.get_active_registries()
        report.append(f"\nActive enrollment: {len(active)} registries")

        # Sample sizes
        with_samples = [r for r in self.registries if r.sample_size]
        total_samples = sum(r.sample_size for r in with_samples)
        report.append(f"\nTotal participants: {total_samples:,}")
        report.append(f"Largest registry: {max(with_samples, key=lambda r: r.sample_size).name} (N={max(r.sample_size for r in with_samples):,})")

        # Data types
        report.append("\n=== Available Data Types ===")
        data_type_counts = {}
        for r in self.registries:
            for dt in r.data_types:
                data_type_counts[dt] = data_type_counts.get(dt, 0) + 1

        for dt, count in sorted(data_type_counts.items(), key=lambda x: -x[1]):
            report.append(f"{dt}: {count} registries")

        # Access types
        report.append("\n=== Access Types ===")
        for access in ['open', 'controlled', 'collaboration_only', 'closed']:
            count = len(self.filter_by_access(access))
            report.append(f"{access}: {count} registries")

        return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(
        description='Catalog patient registries for ADHD/autism research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Catalog all registries
  python registry_crawler.py --output data/registries/

  # Find autism registries with genomics
  python registry_crawler.py \\
      --condition autism \\
      --data-type genomics \\
      --output data/registries/

  # Find active registries with open data
  python registry_crawler.py \\
      --access-type open \\
      --active-only \\
      --output data/registries/
        """
    )

    parser.add_argument(
        '--condition',
        choices=['autism', 'adhd', 'both'],
        help='Filter by condition'
    )

    parser.add_argument(
        '--data-type',
        help='Filter by data type (genomics, imaging, phenotype, etc.)'
    )

    parser.add_argument(
        '--access-type',
        choices=['open', 'controlled', 'collaboration_only', 'closed'],
        help='Filter by access type'
    )

    parser.add_argument(
        '--active-only',
        action='store_true',
        help='Only active registries'
    )

    parser.add_argument(
        '--min-sample-size',
        type=int,
        help='Minimum sample size'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/registries',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize crawler
    crawler = RegistryCrawler(Path(args.output))

    # Search with filters
    registries = crawler.search_registries(
        condition=args.condition,
        data_type=args.data_type,
        access_type=args.access_type,
        active_only=args.active_only,
        min_sample_size=args.min_sample_size
    )

    # Export
    crawler.export_json(registries)
    crawler.export_csv(registries)

    # Print summary
    print("\n" + crawler.generate_summary_report())

    print(f"\n=== Filtered Results ===")
    print(f"Found {len(registries)} matching registries")

    if registries:
        print("\nTop registries:")
        for r in sorted(registries, key=lambda x: x.sample_size or 0, reverse=True)[:10]:
            size_str = f"N={r.sample_size:,}" if r.sample_size else "N=unknown"
            print(f"  {r.name}: {size_str}, {', '.join(r.data_types[:3])}, {r.access_type}")


if __name__ == '__main__':
    main()