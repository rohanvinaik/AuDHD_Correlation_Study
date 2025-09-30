#!/usr/bin/env python3
"""
Biobank Finder for ADHD/Autism Brain Tissue and Biological Samples

Catalogs biobanks with:
- Brain tissue repositories
- DNA/blood sample collections
- Cell lines and iPSC resources
- Biomaterials from autism/ADHD cohorts

Requirements:
    pip install requests pandas

Usage:
    # Catalog all biobanks
    python biobank_finder.py \\
        --output data/registries/

    # Find brain tissue biobanks
    python biobank_finder.py \\
        --sample-type brain_tissue \\
        --output data/registries/

    # Find biobanks with DNA samples
    python biobank_finder.py \\
        --sample-type dna \\
        --condition autism \\
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
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install pandas")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Biobank:
    """Biobank with biological samples"""
    biobank_id: str
    name: str
    full_name: str
    url: str
    organization: str
    country: str

    # Sample types
    sample_types: List[str]  # brain_tissue, dna, blood, cell_lines, ipsc, plasma, serum
    conditions: List[str]  # autism, adhd, control, other

    # Inventory
    total_samples: Optional[int]
    autism_samples: Optional[int]
    adhd_samples: Optional[int]
    control_samples: Optional[int]

    # Brain tissue specifics
    brain_regions: Optional[List[str]]
    tissue_preparation: Optional[List[str]]  # frozen, fixed, RNA_preserved

    # Access
    access_type: str  # open, controlled, collaboration_only
    application_url: Optional[str]
    application_process: str
    typical_turnaround: Optional[str]
    contact_email: Optional[str]

    # Details
    description: str
    data_available: List[str]  # clinical_data, genomics, neuropathology, imaging
    quality_control: List[str]
    publications: List[str]
    notes: str
    last_updated: str


# Brain tissue biobanks
BRAIN_BIOBANKS = [
    {
        'biobank_id': 'AUTISM_BRAINNET',
        'name': 'Autism BrainNet',
        'full_name': 'Autism BrainNet Brain Tissue Repository',
        'url': 'https://www.autismbrainnet.org',
        'organization': 'Autism Speaks',
        'country': 'USA',
        'sample_types': ['brain_tissue', 'frozen_tissue', 'fixed_tissue', 'rna'],
        'conditions': ['autism', 'control'],
        'total_samples': 3000,
        'autism_samples': 1800,
        'adhd_samples': None,
        'control_samples': 1200,
        'brain_regions': [
            'prefrontal cortex', 'anterior cingulate', 'temporal cortex',
            'cerebellum', 'hippocampus', 'amygdala', 'striatum'
        ],
        'tissue_preparation': ['frozen', 'fixed (formalin)', 'RNAlater'],
        'access_type': 'controlled',
        'application_url': 'https://www.autismbrainnet.org/resources/request-tissue/',
        'application_process': 'Submit research proposal, IRB approval required, reviewed by Scientific Advisory Committee',
        'typical_turnaround': '8-12 weeks',
        'contact_email': 'abn@psych.uic.edu',
        'description': 'Premier autism brain tissue repository with comprehensive clinical characterization',
        'data_available': ['clinical_history', 'medical_records', 'neuropathology', 'MRI (some)', 'genomics (some)'],
        'quality_control': ['PMI < 36h preferred', 'RIN scoring', 'Neuropathology review', 'Brain pH measurement'],
        'publications': ['PMID:29174023', 'PMID:31209380'],
        'notes': 'Gold standard for autism brain tissue. Excellent clinical documentation. Prioritizes innovative research.'
    },
    {
        'biobank_id': 'NIH_NEUROBIOBANK',
        'name': 'NIH NeuroBioBank',
        'full_name': 'NIH NeuroBioBank Brain and Tissue Repository',
        'url': 'https://neurobiobank.nih.gov',
        'organization': 'NIH',
        'country': 'USA',
        'sample_types': ['brain_tissue', 'frozen_tissue', 'fixed_tissue', 'spinal_cord', 'csf'],
        'conditions': ['autism', 'adhd', 'control', 'psychiatric', 'neurological'],
        'total_samples': 12000,
        'autism_samples': 450,
        'adhd_samples': 280,
        'control_samples': 5000,
        'brain_regions': [
            'All major regions', 'prefrontal cortex', 'striatum', 'cerebellum',
            'hippocampus', 'substantia nigra', 'multiple regions'
        ],
        'tissue_preparation': ['frozen', 'fixed', 'RNAlater', 'OCT-embedded'],
        'access_type': 'controlled',
        'application_url': 'https://neurobiobank.nih.gov/researchers/',
        'application_process': 'Register account, search catalog, submit request, IRB approval required',
        'typical_turnaround': '6-8 weeks',
        'contact_email': 'nbbinfo@nih.gov',
        'description': 'Large multi-disorder brain repository with autism and ADHD samples',
        'data_available': ['clinical_data', 'neuropathology', 'demographics', 'cause_of_death'],
        'quality_control': ['PMI < 24h preferred', 'Neuropathology assessment', 'Quality metrics'],
        'publications': ['PMID:29359783'],
        'notes': 'Broader than autism-specific. Free tissue for NIH-funded researchers. Searchable online catalog.'
    },
    {
        'biobank_id': 'UK_BRAIN_BANKS',
        'name': 'UK Brain Banks Network',
        'full_name': 'UK Brain Banks Network for Mental Health and Neurological Disorders',
        'url': 'https://www.mrc.ac.uk/research/facilities-and-resources-for-researchers/brain-banks/',
        'organization': 'MRC UK',
        'country': 'UK',
        'sample_types': ['brain_tissue', 'frozen_tissue', 'fixed_tissue'],
        'conditions': ['autism', 'adhd', 'psychiatric', 'neurological', 'control'],
        'total_samples': 8000,
        'autism_samples': 200,
        'adhd_samples': 50,
        'control_samples': 3000,
        'brain_regions': ['Multiple regions', 'prefrontal cortex', 'temporal lobe', 'cerebellum'],
        'tissue_preparation': ['frozen', 'fixed'],
        'access_type': 'collaboration_only',
        'application_url': 'https://www.mrc.ac.uk/research/facilities-and-resources-for-researchers/brain-banks/',
        'application_process': 'Contact individual brain banks, proposal review, collaboration agreement',
        'typical_turnaround': '12-16 weeks',
        'contact_email': 'brainbanks@mrc.ac.uk',
        'description': 'Network of UK brain banks with psychiatric and neurological samples',
        'data_available': ['clinical_data', 'neuropathology', 'some_genomics'],
        'quality_control': ['Standard neuropathology protocols', 'PMI tracking'],
        'publications': ['PMID:23847115'],
        'notes': 'Requires UK collaboration. Multiple sites (Edinburgh, London, Manchester, Oxford).'
    },
    {
        'biobank_id': 'HARVARD_BRAIN_BANK',
        'name': 'Harvard Brain Bank',
        'full_name': 'Harvard Brain Tissue Resource Center',
        'url': 'https://hbtrc.mclean.harvard.edu',
        'organization': 'Harvard Medical School',
        'country': 'USA',
        'sample_types': ['brain_tissue', 'frozen_tissue', 'fixed_tissue'],
        'conditions': ['psychiatric', 'neurological', 'autism', 'control'],
        'total_samples': 6000,
        'autism_samples': 120,
        'adhd_samples': None,
        'control_samples': 2500,
        'brain_regions': ['prefrontal cortex', 'temporal cortex', 'cerebellum', 'multiple regions'],
        'tissue_preparation': ['frozen', 'fixed'],
        'access_type': 'controlled',
        'application_url': 'https://hbtrc.mclean.harvard.edu/request.html',
        'application_process': 'Online request form, project review, IRB approval',
        'typical_turnaround': '8-10 weeks',
        'contact_email': 'btrc@mclean.harvard.edu',
        'description': 'Long-established brain bank with psychiatric disorders focus',
        'data_available': ['clinical_data', 'neuropathology', 'toxicology'],
        'quality_control': ['PMI < 30h typical', 'Neuropathology review', 'Toxicology screening'],
        'publications': ['PMID:11063850'],
        'notes': 'Nominal fee for tissue. Excellent clinical documentation.'
    },
    {
        'biobank_id': 'NIMH_BRAIN_COLLECTION',
        'name': 'NIMH Brain Collection',
        'full_name': 'NIMH Human Brain Collection Core',
        'url': 'https://www.nimh.nih.gov/research/research-conducted-at-nimh/research-areas/research-support-services/hbcc',
        'organization': 'NIMH',
        'country': 'USA',
        'sample_types': ['brain_tissue', 'frozen_tissue'],
        'conditions': ['psychiatric', 'autism', 'control'],
        'total_samples': 1500,
        'autism_samples': 80,
        'adhd_samples': None,
        'control_samples': 600,
        'brain_regions': ['prefrontal cortex', 'anterior cingulate', 'hippocampus', 'amygdala'],
        'tissue_preparation': ['frozen', 'high quality RNA'],
        'access_type': 'controlled',
        'application_url': 'https://www.nimh.nih.gov/research/research-conducted-at-nimh/research-areas/research-support-services/hbcc',
        'application_process': 'NIH-funded researchers preferred, proposal submission',
        'typical_turnaround': '10-12 weeks',
        'contact_email': 'hbcc@mail.nih.gov',
        'description': 'NIMH psychiatric brain collection with high-quality RNA preservation',
        'data_available': ['clinical_data', 'toxicology', 'RNA_quality'],
        'quality_control': ['PMI < 24h', 'RIN > 6.0', 'RNA integrity focus'],
        'publications': ['PMID:24814961'],
        'notes': 'Focus on psychiatric disorders. High RNA quality standards.'
    }
]

# DNA/blood/cell line biobanks
BIOMATERIAL_BIOBANKS = [
    {
        'biobank_id': 'AGRE_BIOBANK',
        'name': 'AGRE Biobank',
        'full_name': 'Autism Genetic Resource Exchange Biobank',
        'url': 'https://research.agre.org',
        'organization': 'Autism Speaks',
        'country': 'USA',
        'sample_types': ['dna', 'cell_lines', 'lymphoblastoid_cell_lines'],
        'conditions': ['autism', 'family_members'],
        'total_samples': 9000,
        'autism_samples': 3500,
        'adhd_samples': None,
        'control_samples': 5500,
        'brain_regions': None,
        'tissue_preparation': None,
        'access_type': 'controlled',
        'application_url': 'https://research.agre.org/program/biorepository/',
        'application_process': 'Submit research application, IRB approval, material transfer agreement',
        'typical_turnaround': '6-8 weeks',
        'contact_email': 'agre@autismspeaks.org',
        'description': 'DNA and cell lines from multiplex autism families',
        'data_available': ['genotypes', 'phenotype_data', 'pedigrees'],
        'quality_control': ['DNA concentration', 'Purity metrics', 'Cell viability'],
        'publications': ['PMID:21926972'],
        'notes': 'Historic collection. Linked to phenotype data in NDAR. Nominal fee.'
    },
    {
        'biobank_id': 'SPARK_BIOBANK',
        'name': 'SPARK Biobank',
        'full_name': 'SPARK Simons Foundation Biobank',
        'url': 'https://sparkforautism.org',
        'organization': 'Simons Foundation',
        'country': 'USA',
        'sample_types': ['dna', 'saliva'],
        'conditions': ['autism', 'family_members'],
        'total_samples': 100000,
        'autism_samples': 50000,
        'adhd_samples': None,
        'control_samples': 50000,
        'brain_regions': None,
        'tissue_preparation': None,
        'access_type': 'controlled',
        'application_url': 'https://base.sfari.org',
        'application_url': 'https://www.sfari.org/resource/sfari-base/',
        'application_process': 'Submit research proposal via SFARI Base, review by committee',
        'typical_turnaround': '8-12 weeks',
        'contact_email': 'spark@simonsfoundation.org',
        'description': 'Largest autism genetic biobank with WES data',
        'data_available': ['whole_exome_sequencing', 'phenotype', 'medical_records'],
        'quality_control': ['Clinical lab standards', 'Sequencing QC'],
        'publications': ['PMID:32651583'],
        'notes': 'Rapidly growing. WES for all samples. Data via SFARI Base.'
    },
    {
        'biobank_id': 'NIMH_REPOSITORY',
        'name': 'NIMH Repository',
        'full_name': 'NIMH Repository and Genomics Resource',
        'url': 'https://www.nimhgenetics.org',
        'organization': 'NIMH / Rutgers',
        'country': 'USA',
        'sample_types': ['dna', 'cell_lines', 'plasma', 'serum'],
        'conditions': ['autism', 'adhd', 'psychiatric', 'control'],
        'total_samples': 250000,
        'autism_samples': 12000,
        'adhd_samples': 8000,
        'control_samples': 50000,
        'brain_regions': None,
        'tissue_preparation': None,
        'access_type': 'controlled',
        'application_url': 'https://www.nimhgenetics.org/resources/request-biosamples',
        'application_process': 'Register, submit request with research plan, IRB approval',
        'typical_turnaround': '4-6 weeks',
        'contact_email': 'nimhgenetics@rutgers.edu',
        'description': 'Major psychiatric biomaterial repository with autism and ADHD samples',
        'data_available': ['genotypes', 'clinical_data', 'family_structure'],
        'quality_control': ['DNA quality metrics', 'Sample tracking', 'QC reports'],
        'publications': ['PMID:15520816'],
        'notes': 'Free for NIH-funded research. Fast turnaround. Searchable catalog.'
    },
    {
        'biobank_id': 'IPSYCH_BIOBANK',
        'name': 'iPSYCH Biobank',
        'full_name': 'Integrative Psychiatric Research Biobank',
        'url': 'https://ipsych.dk',
        'organization': 'iPSYCH Denmark',
        'country': 'Denmark',
        'sample_types': ['dna', 'plasma', 'dried_blood_spots'],
        'conditions': ['autism', 'adhd', 'psychiatric', 'control'],
        'total_samples': 150000,
        'autism_samples': 18000,
        'adhd_samples': 20000,
        'control_samples': 50000,
        'brain_regions': None,
        'tissue_preparation': None,
        'access_type': 'collaboration_only',
        'application_url': 'https://ipsych.dk/en/data-access/',
        'application_process': 'Collaboration with Danish sites required, formal application',
        'typical_turnaround': '16-24 weeks',
        'contact_email': 'ipsych@econ.au.dk',
        'description': 'Danish neonatal screening biobank with WGS and registry linkage',
        'data_available': ['whole_genome_sequencing', 'register_data', 'prescription_records'],
        'quality_control': ['Clinical lab standards', 'WGS quality control'],
        'publications': ['PMID:28540026'],
        'notes': 'Exceptional register linkage. WGS available. Requires Danish collaboration.'
    },
    {
        'biobank_id': 'CORIEL',
        'name': 'Coriell',
        'full_name': 'Coriell Institute Biorepository',
        'url': 'https://www.coriell.org',
        'organization': 'Coriell Institute',
        'country': 'USA',
        'sample_types': ['cell_lines', 'dna', 'lymphoblastoid_cell_lines', 'ipsc'],
        'conditions': ['autism', 'control', 'various'],
        'total_samples': 500000,
        'autism_samples': 2500,
        'adhd_samples': 200,
        'control_samples': 100000,
        'brain_regions': None,
        'tissue_preparation': None,
        'access_type': 'open',
        'application_url': 'https://www.coriell.org/1/Online-Ordering',
        'application_process': 'Online ordering, immediate purchase, MTA for some collections',
        'typical_turnaround': '1-2 weeks',
        'contact_email': 'ccr@coriell.org',
        'description': 'Commercial biorepository with autism cell lines from research studies',
        'data_available': ['genotypes (some)', 'phenotype (limited)', 'cell_line_data'],
        'quality_control': ['Cell viability', 'Contamination testing', 'Karyotyping'],
        'publications': ['Multiple'],
        'notes': 'Fast commercial access. Includes AGRE, SSC samples. Fee-based.'
    },
    {
        'biobank_id': 'IPSYC_LINES',
        'name': 'iPSC Lines',
        'full_name': 'Autism iPSC Repository',
        'url': 'Various',
        'organization': 'Multiple',
        'country': 'USA',
        'sample_types': ['ipsc', 'induced_pluripotent_stem_cells'],
        'conditions': ['autism', 'syndromic_autism', 'control'],
        'total_samples': 500,
        'autism_samples': 300,
        'adhd_samples': 20,
        'control_samples': 180,
        'brain_regions': None,
        'tissue_preparation': None,
        'access_type': 'controlled',
        'application_url': 'Various',
        'application_process': 'Contact originating labs, MTA required',
        'typical_turnaround': '8-12 weeks',
        'contact_email': None,
        'description': 'iPSC lines from autism patients for neuronal differentiation',
        'data_available': ['genotypes', 'clinical_phenotype', 'differentiation_protocols'],
        'quality_control': ['Pluripotency markers', 'Karyotyping', 'Mycoplasma testing'],
        'publications': ['PMID:31209380', 'PMID:30545854'],
        'notes': 'Growing resource. From labs of Geschwind, Pasca, others. Check StemBANCC, CIRM.'
    },
    {
        'biobank_id': 'UK_BIOBANK_SAMPLES',
        'name': 'UK Biobank Samples',
        'full_name': 'UK Biobank Biological Samples',
        'url': 'https://www.ukbiobank.ac.uk',
        'organization': 'UK Biobank',
        'country': 'UK',
        'sample_types': ['dna', 'plasma', 'serum', 'urine', 'saliva'],
        'conditions': ['autism', 'adhd', 'various', 'control'],
        'total_samples': 1000000,
        'autism_samples': 2000,
        'adhd_samples': 4000,
        'control_samples': 490000,
        'brain_regions': None,
        'tissue_preparation': None,
        'access_type': 'controlled',
        'application_url': 'https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access',
        'application_process': 'Submit application, institutional approval, data access agreement',
        'typical_turnaround': '3-4 weeks for data, 12-16 weeks for samples',
        'contact_email': 'access@ukbiobank.ac.uk',
        'description': 'Massive biobank with autism/ADHD cases and multi-omics',
        'data_available': ['genomics', 'metabolomics', 'proteomics', 'imaging', 'ehr'],
        'quality_control': ['Clinical lab standards', 'Standardized protocols'],
        'publications': ['PMID:30305743'],
        'notes': 'Sample access more restricted than data access. Fee for samples.'
    },
    {
        'biobank_id': 'ABCD_BIOBANK',
        'name': 'ABCD Biobank',
        'full_name': 'ABCD Study Biorepository',
        'url': 'https://abcdstudy.org',
        'organization': 'NIH',
        'country': 'USA',
        'sample_types': ['dna', 'saliva', 'blood', 'hair'],
        'conditions': ['adhd', 'neurodevelopmental', 'control'],
        'total_samples': 24000,
        'autism_samples': 300,
        'adhd_samples': 2000,
        'control_samples': 20000,
        'brain_regions': None,
        'tissue_preparation': None,
        'access_type': 'controlled',
        'application_url': 'https://nda.nih.gov',
        'application_process': 'NDA data access approval, then biospecimen request',
        'typical_turnaround': '12-20 weeks',
        'contact_email': 'ndahelp@mail.nih.gov',
        'description': 'Longitudinal neurodevelopmental cohort with ADHD biospecimens',
        'data_available': ['genomics', 'imaging', 'cognitive', 'longitudinal'],
        'quality_control': ['Standardized collection protocols', 'Central lab processing'],
        'publications': ['PMID:30595399'],
        'notes': 'Rich longitudinal data. Biospecimen access via NDA. Limited quantities.'
    }
]


class BiobankFinder:
    """Find and catalog biobanks with biological samples"""

    def __init__(self, output_dir: Path):
        """
        Initialize finder

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.biobanks: List[Biobank] = []
        self._load_biobanks()

        logger.info(f"Initialized biobank finder: {len(self.biobanks)} biobanks")

    def _load_biobanks(self):
        """Load all biobank data"""
        all_biobank_data = BRAIN_BIOBANKS + BIOMATERIAL_BIOBANKS

        for bb_data in all_biobank_data:
            biobank = Biobank(
                biobank_id=bb_data['biobank_id'],
                name=bb_data['name'],
                full_name=bb_data['full_name'],
                url=bb_data['url'],
                organization=bb_data['organization'],
                country=bb_data['country'],
                sample_types=bb_data['sample_types'],
                conditions=bb_data['conditions'],
                total_samples=bb_data.get('total_samples'),
                autism_samples=bb_data.get('autism_samples'),
                adhd_samples=bb_data.get('adhd_samples'),
                control_samples=bb_data.get('control_samples'),
                brain_regions=bb_data.get('brain_regions'),
                tissue_preparation=bb_data.get('tissue_preparation'),
                access_type=bb_data['access_type'],
                application_url=bb_data.get('application_url'),
                application_process=bb_data['application_process'],
                typical_turnaround=bb_data.get('typical_turnaround'),
                contact_email=bb_data.get('contact_email'),
                description=bb_data['description'],
                data_available=bb_data['data_available'],
                quality_control=bb_data['quality_control'],
                publications=bb_data.get('publications', []),
                notes=bb_data['notes'],
                last_updated=datetime.now().strftime('%Y-%m-%d')
            )

            self.biobanks.append(biobank)

    def filter_by_sample_type(self, sample_type: str) -> List[Biobank]:
        """Filter biobanks by sample type"""
        return [bb for bb in self.biobanks if sample_type in bb.sample_types]

    def filter_by_condition(self, condition: str) -> List[Biobank]:
        """Filter biobanks by condition"""
        return [bb for bb in self.biobanks if condition in bb.conditions]

    def get_brain_tissue_biobanks(self) -> List[Biobank]:
        """Get biobanks with brain tissue"""
        return self.filter_by_sample_type('brain_tissue')

    def get_dna_biobanks(self) -> List[Biobank]:
        """Get biobanks with DNA samples"""
        return self.filter_by_sample_type('dna')

    def get_ipsc_biobanks(self) -> List[Biobank]:
        """Get biobanks with iPSC lines"""
        return self.filter_by_sample_type('ipsc')

    def search_biobanks(
        self,
        sample_type: Optional[str] = None,
        condition: Optional[str] = None,
        access_type: Optional[str] = None,
        min_samples: Optional[int] = None
    ) -> List[Biobank]:
        """Search biobanks with multiple filters"""
        results = self.biobanks

        if sample_type:
            results = [bb for bb in results if sample_type in bb.sample_types]

        if condition:
            results = [bb for bb in results if condition in bb.conditions]

        if access_type:
            results = [bb for bb in results if bb.access_type == access_type]

        if min_samples:
            results = [bb for bb in results
                      if bb.total_samples and bb.total_samples >= min_samples]

        return results

    def export_json(self, biobanks: Optional[List[Biobank]] = None) -> Path:
        """Export biobanks to JSON"""
        if biobanks is None:
            biobanks = self.biobanks

        # Count samples by type
        brain_tissue_count = sum(1 for bb in biobanks if 'brain_tissue' in bb.sample_types)
        dna_count = sum(1 for bb in biobanks if 'dna' in bb.sample_types)
        ipsc_count = sum(1 for bb in biobanks if 'ipsc' in bb.sample_types)

        total_autism = sum(bb.autism_samples for bb in biobanks if bb.autism_samples)
        total_adhd = sum(bb.adhd_samples for bb in biobanks if bb.adhd_samples)

        output = {
            'metadata': {
                'generated_date': datetime.now().strftime('%Y-%m-%d'),
                'total_biobanks': len(biobanks),
                'brain_tissue_biobanks': brain_tissue_count,
                'dna_biobanks': dna_count,
                'ipsc_biobanks': ipsc_count,
                'total_autism_samples': total_autism,
                'total_adhd_samples': total_adhd
            },
            'biobanks': [asdict(bb) for bb in biobanks]
        }

        output_file = self.output_dir / 'biobank_inventory.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Exported {len(biobanks)} biobanks to {output_file}")
        return output_file

    def export_csv(self, biobanks: Optional[List[Biobank]] = None) -> Path:
        """Export biobanks to CSV"""
        if biobanks is None:
            biobanks = self.biobanks

        rows = []
        for bb in biobanks:
            rows.append({
                'biobank_id': bb.biobank_id,
                'name': bb.name,
                'full_name': bb.full_name,
                'url': bb.url,
                'organization': bb.organization,
                'country': bb.country,
                'sample_types': ', '.join(bb.sample_types),
                'conditions': ', '.join(bb.conditions),
                'total_samples': bb.total_samples or 0,
                'autism_samples': bb.autism_samples or 0,
                'adhd_samples': bb.adhd_samples or 0,
                'brain_regions': ', '.join(bb.brain_regions) if bb.brain_regions else '',
                'access_type': bb.access_type,
                'application_url': bb.application_url or '',
                'typical_turnaround': bb.typical_turnaround or '',
                'contact_email': bb.contact_email or '',
                'description': bb.description,
                'notes': bb.notes
            })

        df = pd.DataFrame(rows)
        output_file = self.output_dir / 'biobank_inventory.csv'
        df.to_csv(output_file, index=False)

        logger.info(f"Exported {len(biobanks)} biobanks to CSV")
        return output_file

    def generate_summary_report(self) -> str:
        """Generate summary statistics"""
        report = []
        report.append("=== Biobank Summary ===\n")

        report.append(f"Total biobanks: {len(self.biobanks)}")

        # Sample types
        brain_tissue = self.get_brain_tissue_biobanks()
        dna_banks = self.get_dna_biobanks()
        ipsc_banks = self.get_ipsc_biobanks()

        report.append(f"\nBrain tissue biobanks: {len(brain_tissue)}")
        report.append(f"DNA biobanks: {len(dna_banks)}")
        report.append(f"iPSC biobanks: {len(ipsc_banks)}")

        # Sample counts
        autism_with_counts = [bb for bb in self.biobanks if bb.autism_samples]
        adhd_with_counts = [bb for bb in self.biobanks if bb.adhd_samples]

        total_autism = sum(bb.autism_samples for bb in autism_with_counts)
        total_adhd = sum(bb.adhd_samples for bb in adhd_with_counts)

        report.append(f"\nTotal autism samples: {total_autism:,}")
        report.append(f"Total ADHD samples: {total_adhd:,}")

        # Brain tissue details
        if brain_tissue:
            report.append("\n=== Brain Tissue Biobanks ===")
            for bb in brain_tissue:
                autism_str = f"{bb.autism_samples} autism" if bb.autism_samples else "No autism count"
                report.append(f"{bb.name}: {autism_str}, {bb.access_type}")

        # Access types
        report.append("\n=== Access Types ===")
        for access in ['open', 'controlled', 'collaboration_only']:
            count = len([bb for bb in self.biobanks if bb.access_type == access])
            report.append(f"{access}: {count} biobanks")

        return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(
        description='Find biobanks with ADHD/autism biological samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Catalog all biobanks
  python biobank_finder.py --output data/registries/

  # Find brain tissue biobanks
  python biobank_finder.py \\
      --sample-type brain_tissue \\
      --output data/registries/

  # Find DNA biobanks with autism samples
  python biobank_finder.py \\
      --sample-type dna \\
      --condition autism \\
      --output data/registries/
        """
    )

    parser.add_argument(
        '--sample-type',
        help='Filter by sample type (brain_tissue, dna, ipsc, etc.)'
    )

    parser.add_argument(
        '--condition',
        choices=['autism', 'adhd', 'control'],
        help='Filter by condition'
    )

    parser.add_argument(
        '--access-type',
        choices=['open', 'controlled', 'collaboration_only'],
        help='Filter by access type'
    )

    parser.add_argument(
        '--min-samples',
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

    # Initialize finder
    finder = BiobankFinder(Path(args.output))

    # Search with filters
    biobanks = finder.search_biobanks(
        sample_type=args.sample_type,
        condition=args.condition,
        access_type=args.access_type,
        min_samples=args.min_samples
    )

    # Export
    finder.export_json(biobanks)
    finder.export_csv(biobanks)

    # Print summary
    print("\n" + finder.generate_summary_report())

    print(f"\n=== Filtered Results ===")
    print(f"Found {len(biobanks)} matching biobanks")

    if biobanks:
        print("\nTop biobanks:")
        for bb in sorted(biobanks, key=lambda x: x.autism_samples or 0, reverse=True)[:10]:
            autism_str = f"{bb.autism_samples} autism" if bb.autism_samples else "N/A"
            print(f"  {bb.name}: {autism_str}, {', '.join(bb.sample_types[:2])}, {bb.access_type}")


if __name__ == '__main__':
    main()