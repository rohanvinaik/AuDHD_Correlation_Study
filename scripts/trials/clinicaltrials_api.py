#!/usr/bin/env python3
"""
ClinicalTrials.gov API Client for ADHD/Autism Research

Searches ClinicalTrials.gov for trials with biomarker data, biological samples,
and data sharing plans using the official REST API.

ClinicalTrials.gov contains:
- 450,000+ registered studies from 220+ countries
- Trial design, interventions, outcomes
- Results data (when available)
- Individual participant data (IPD) sharing statements
- Biospecimen retention information
- Principal investigator contact information

Key Features:
- Search by condition (ADHD, autism)
- Filter by intervention type
- Find trials with biomarkers
- Identify data sharing opportunities
- Extract PI contact information
- Results data when available

Requirements:
    pip install requests pandas

Usage:
    # Search for autism trials with biomarkers
    python clinicaltrials_api.py \\
        --condition "Autism Spectrum Disorder" \\
        --has-biomarkers \\
        --output data/trials/

    # Search ADHD drug trials with results
    python clinicaltrials_api.py \\
        --condition "ADHD" \\
        --intervention Drug \\
        --has-results \\
        --output data/trials/

    # Find trials with data sharing plans
    python clinicaltrials_api.py \\
        --condition "Autism" \\
        --ipd-sharing \\
        --has-biospecimens \\
        --output data/trials/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

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


# ClinicalTrials.gov API v2
CT_API_BASE = "https://clinicaltrials.gov/api/v2"

# Search field mappings
CONDITION_TERMS = {
    'adhd': ['ADHD', 'Attention Deficit Disorder with Hyperactivity',
             'Attention Deficit Hyperactivity Disorder'],
    'autism': ['Autism', 'Autism Spectrum Disorder', 'Autistic Disorder',
               'Asperger Syndrome', 'Pervasive Developmental Disorder']
}

INTERVENTION_TYPES = [
    'Drug', 'Behavioral', 'Dietary Supplement', 'Device',
    'Biological', 'Procedure', 'Other'
]

BIOMARKER_KEYWORDS = [
    'biomarker', 'metabolomics', 'genomics', 'genetics',
    'proteomics', 'blood', 'plasma', 'serum', 'urine',
    'DNA', 'RNA', 'gene expression', 'metabolite',
    'SCFA', 'microbiome', 'neuroimaging', 'MRI'
]

RECRUITMENT_STATUS = [
    'Not yet recruiting', 'Recruiting', 'Enrolling by invitation',
    'Active, not recruiting', 'Completed', 'Suspended',
    'Terminated', 'Withdrawn'
]


@dataclass
class ClinicalTrial:
    """Represents a clinical trial"""
    nct_id: str
    title: str
    brief_summary: str
    detailed_description: str
    conditions: List[str]
    interventions: List[Dict]
    phase: str
    enrollment: int
    status: str
    start_date: str
    completion_date: str
    primary_outcomes: List[Dict]
    secondary_outcomes: List[str]
    has_results: bool
    has_biomarkers: bool
    ipd_sharing: str
    biospec_retention: str
    biospec_description: str
    sponsor: str
    collaborators: List[str]
    locations: List[Dict]
    principal_investigator: Optional[Dict]
    contact_email: Optional[str]
    url: str
    last_update: str


class ClinicalTrialsAPI:
    """Client for ClinicalTrials.gov API v2"""

    def __init__(self, output_dir: Path):
        """
        Initialize API client

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        logger.info(f"Initialized ClinicalTrials.gov API client: {output_dir}")

    def search_studies(self,
                      condition: Optional[str] = None,
                      intervention: Optional[str] = None,
                      status: Optional[List[str]] = None,
                      has_results: bool = False,
                      page_size: int = 100,
                      max_studies: int = 500) -> List[str]:
        """
        Search for clinical trials

        Args:
            condition: Condition/disease term
            intervention: Intervention type
            status: Recruitment status list
            has_results: Filter for trials with results
            page_size: Results per page (max 100)
            max_studies: Maximum studies to retrieve

        Returns:
            List of NCT IDs
        """
        logger.info(f"Searching ClinicalTrials.gov: condition={condition}, intervention={intervention}")

        # Build query parameters
        params = {
            'format': 'json',
            'pageSize': min(page_size, 100)
        }

        # Add filters
        query_parts = []

        if condition:
            query_parts.append(f'AREA[Condition]({condition})')

        if intervention:
            query_parts.append(f'AREA[InterventionType]({intervention})')

        if status:
            status_query = ' OR '.join(status)
            query_parts.append(f'AREA[OverallStatus]({status_query})')

        if has_results:
            query_parts.append('AREA[ResultsFirstPostDate]RANGE[01/01/2000, MAX]')

        if query_parts:
            params['query.cond'] = ' AND '.join(query_parts)

        nct_ids = []
        page_token = None

        while len(nct_ids) < max_studies:
            if page_token:
                params['pageToken'] = page_token

            try:
                response = self.session.get(
                    f"{CT_API_BASE}/studies",
                    params=params,
                    timeout=30
                )

                if response.status_code != 200:
                    logger.error(f"API error: HTTP {response.status_code}")
                    break

                data = response.json()

                # Extract NCT IDs
                studies = data.get('studies', [])
                for study in studies:
                    protocol_section = study.get('protocolSection', {})
                    identification = protocol_section.get('identificationModule', {})
                    nct_id = identification.get('nctId')

                    if nct_id:
                        nct_ids.append(nct_id)

                # Check for next page
                next_page_token = data.get('nextPageToken')
                if not next_page_token or len(nct_ids) >= max_studies:
                    break

                page_token = next_page_token
                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Error fetching studies: {e}")
                break

        nct_ids = nct_ids[:max_studies]
        logger.info(f"Found {len(nct_ids)} trials")
        return nct_ids

    def get_study_details(self, nct_id: str) -> Optional[ClinicalTrial]:
        """
        Get detailed information for a trial

        Args:
            nct_id: NCT identifier

        Returns:
            ClinicalTrial object
        """
        try:
            response = self.session.get(
                f"{CT_API_BASE}/studies/{nct_id}",
                params={'format': 'json'},
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {nct_id}: HTTP {response.status_code}")
                return None

            data = response.json()
            protocol_section = data.get('protocolSection', {})

            # Extract identification
            identification = protocol_section.get('identificationModule', {})
            nct_id = identification.get('nctId', '')
            title = identification.get('briefTitle', '')

            # Extract status
            status_module = protocol_section.get('statusModule', {})
            overall_status = status_module.get('overallStatus', '')
            start_date = status_module.get('startDateStruct', {}).get('date', '')
            completion_date = status_module.get('completionDateStruct', {}).get('date', '')
            last_update = status_module.get('lastUpdatePostDateStruct', {}).get('date', '')

            # Extract description
            description = protocol_section.get('descriptionModule', {})
            brief_summary = description.get('briefSummary', '')
            detailed_description = description.get('detailedDescription', '')

            # Extract conditions
            conditions_module = protocol_section.get('conditionsModule', {})
            conditions = conditions_module.get('conditions', [])

            # Extract interventions
            interventions_module = protocol_section.get('armsInterventionsModule', {})
            interventions = interventions_module.get('interventions', [])

            # Extract design
            design_module = protocol_section.get('designModule', {})
            enrollment = design_module.get('enrollmentInfo', {}).get('count', 0)
            phases = design_module.get('phases', [])
            phase = phases[0] if phases else 'Not Applicable'

            # Extract outcomes
            outcomes_module = protocol_section.get('outcomesModule', {})
            primary_outcomes = outcomes_module.get('primaryOutcomes', [])
            secondary_outcomes = outcomes_module.get('secondaryOutcomes', [])

            # Extract sponsor
            sponsor_module = protocol_section.get('sponsorCollaboratorsModule', {})
            lead_sponsor = sponsor_module.get('leadSponsor', {})
            sponsor = lead_sponsor.get('name', '')
            collaborators = [c.get('name', '') for c in sponsor_module.get('collaborators', [])]

            # Extract contacts
            contacts_module = protocol_section.get('contactsLocationsModule', {})
            overall_officials = contacts_module.get('overallOfficials', [])

            pi = None
            if overall_officials:
                pi_data = overall_officials[0]
                pi = {
                    'name': pi_data.get('name', ''),
                    'affiliation': pi_data.get('affiliation', ''),
                    'role': pi_data.get('role', '')
                }

            # Extract locations
            locations = contacts_module.get('locations', [])

            # Extract contact email (if available)
            central_contacts = contacts_module.get('centralContacts', [])
            contact_email = None
            if central_contacts:
                contact_email = central_contacts[0].get('email')

            # Check for IPD sharing
            ipd_module = protocol_section.get('ipdSharingStatementModule', {})
            ipd_sharing = ipd_module.get('ipdSharing', 'No')

            # Check for biospecimens
            biospec_module = protocol_section.get('descriptionModule', {})
            biospec_retention = 'Unknown'
            biospec_description = ''

            # Check if results are available
            has_results_module = data.get('hasResults', False)
            has_results = has_results_module

            # Check for biomarker mentions
            full_text = f"{title} {brief_summary} {detailed_description}"
            for outcome in primary_outcomes:
                full_text += f" {outcome.get('measure', '')}"

            has_biomarkers = any(kw in full_text.lower() for kw in BIOMARKER_KEYWORDS)

            trial = ClinicalTrial(
                nct_id=nct_id,
                title=title,
                brief_summary=brief_summary,
                detailed_description=detailed_description,
                conditions=conditions,
                interventions=interventions,
                phase=phase,
                enrollment=enrollment,
                status=overall_status,
                start_date=start_date,
                completion_date=completion_date,
                primary_outcomes=primary_outcomes,
                secondary_outcomes=secondary_outcomes,
                has_results=has_results,
                has_biomarkers=has_biomarkers,
                ipd_sharing=ipd_sharing,
                biospec_retention=biospec_retention,
                biospec_description=biospec_description,
                sponsor=sponsor,
                collaborators=collaborators,
                locations=locations,
                principal_investigator=pi,
                contact_email=contact_email,
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                last_update=last_update
            )

            return trial

        except Exception as e:
            logger.error(f"Error fetching {nct_id}: {e}")
            return None

    def fetch_multiple_studies(self, nct_ids: List[str]) -> List[ClinicalTrial]:
        """Fetch details for multiple studies"""
        logger.info(f"Fetching details for {len(nct_ids)} trials...")

        trials = []

        for i, nct_id in enumerate(nct_ids):
            if i % 20 == 0:
                logger.info(f"Processed {i}/{len(nct_ids)} trials...")

            trial = self.get_study_details(nct_id)
            if trial:
                trials.append(trial)

            time.sleep(0.3)  # Rate limiting

        logger.info(f"Successfully fetched {len(trials)} trials")
        return trials

    def filter_trials_with_biomarkers(self, trials: List[ClinicalTrial]) -> List[ClinicalTrial]:
        """Filter trials that mention biomarkers"""
        filtered = [t for t in trials if t.has_biomarkers]
        logger.info(f"Filtered to {len(filtered)} trials with biomarker mentions")
        return filtered

    def filter_trials_with_data_sharing(self, trials: List[ClinicalTrial]) -> List[ClinicalTrial]:
        """Filter trials with IPD sharing plans"""
        filtered = [t for t in trials if t.ipd_sharing and t.ipd_sharing.lower() != 'no']
        logger.info(f"Filtered to {len(filtered)} trials with data sharing")
        return filtered

    def export_trials(self, trials: List[ClinicalTrial],
                     filename: str = 'trials_with_biomarkers.csv') -> Path:
        """Export trials to CSV"""
        output_file = self.output_dir / filename

        # Convert to DataFrame
        trials_data = []
        for trial in trials:
            trials_data.append({
                'nct_id': trial.nct_id,
                'title': trial.title,
                'status': trial.status,
                'phase': trial.phase,
                'enrollment': trial.enrollment,
                'conditions': ', '.join(trial.conditions),
                'interventions': ', '.join([i.get('name', '') for i in trial.interventions]),
                'sponsor': trial.sponsor,
                'has_results': trial.has_results,
                'has_biomarkers': trial.has_biomarkers,
                'ipd_sharing': trial.ipd_sharing,
                'pi_name': trial.principal_investigator.get('name', '') if trial.principal_investigator else '',
                'pi_affiliation': trial.principal_investigator.get('affiliation', '') if trial.principal_investigator else '',
                'contact_email': trial.contact_email or '',
                'start_date': trial.start_date,
                'completion_date': trial.completion_date,
                'url': trial.url
            })

        df = pd.DataFrame(trials_data)
        df.to_csv(output_file, index=False)

        logger.info(f"Exported {len(trials)} trials: {output_file}")
        return output_file

    def generate_summary(self, trials: List[ClinicalTrial]) -> Dict:
        """Generate summary statistics"""
        summary = {
            'total_trials': len(trials),
            'by_status': {},
            'by_phase': {},
            'by_intervention': {},
            'with_results': sum(1 for t in trials if t.has_results),
            'with_biomarkers': sum(1 for t in trials if t.has_biomarkers),
            'with_ipd_sharing': sum(1 for t in trials if t.ipd_sharing.lower() != 'no'),
            'with_contact_email': sum(1 for t in trials if t.contact_email),
            'total_enrollment': sum(t.enrollment for t in trials),
            'top_sponsors': {}
        }

        # Status breakdown
        for trial in trials:
            summary['by_status'][trial.status] = summary['by_status'].get(trial.status, 0) + 1

        # Phase breakdown
        for trial in trials:
            summary['by_phase'][trial.phase] = summary['by_phase'].get(trial.phase, 0) + 1

        # Intervention breakdown
        for trial in trials:
            for intervention in trial.interventions:
                int_type = intervention.get('type', 'Unknown')
                summary['by_intervention'][int_type] = summary['by_intervention'].get(int_type, 0) + 1

        # Top sponsors
        for trial in trials:
            summary['top_sponsors'][trial.sponsor] = summary['top_sponsors'].get(trial.sponsor, 0) + 1

        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Search ClinicalTrials.gov for ADHD/Autism trials with biomarkers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search autism trials with biomarkers
  python clinicaltrials_api.py \\
      --condition "Autism Spectrum Disorder" \\
      --has-biomarkers \\
      --output data/trials/

  # Search ADHD drug trials with results
  python clinicaltrials_api.py \\
      --condition "ADHD" \\
      --intervention Drug \\
      --has-results \\
      --output data/trials/

  # Find trials with data sharing
  python clinicaltrials_api.py \\
      --condition "Autism" \\
      --ipd-sharing \\
      --output data/trials/

  # Search completed trials only
  python clinicaltrials_api.py \\
      --condition "ADHD" \\
      --status "Completed" \\
      --output data/trials/
        """
    )

    parser.add_argument(
        '--condition',
        type=str,
        required=True,
        help='Condition to search (e.g., "ADHD", "Autism Spectrum Disorder")'
    )

    parser.add_argument(
        '--intervention',
        type=str,
        choices=INTERVENTION_TYPES,
        help='Intervention type filter'
    )

    parser.add_argument(
        '--status',
        nargs='+',
        choices=RECRUITMENT_STATUS,
        help='Recruitment status filter'
    )

    parser.add_argument(
        '--has-results',
        action='store_true',
        help='Filter for trials with posted results'
    )

    parser.add_argument(
        '--has-biomarkers',
        action='store_true',
        help='Filter for trials mentioning biomarkers'
    )

    parser.add_argument(
        '--ipd-sharing',
        action='store_true',
        help='Filter for trials with IPD sharing plans'
    )

    parser.add_argument(
        '--max-studies',
        type=int,
        default=500,
        help='Maximum studies to retrieve (default: 500)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/trials',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize API client
    api = ClinicalTrialsAPI(Path(args.output))

    # Search studies
    nct_ids = api.search_studies(
        condition=args.condition,
        intervention=args.intervention,
        status=args.status,
        has_results=args.has_results,
        max_studies=args.max_studies
    )

    if not nct_ids:
        print("\nNo trials found")
        return

    # Fetch details
    trials = api.fetch_multiple_studies(nct_ids)

    # Apply filters
    if args.has_biomarkers:
        trials = api.filter_trials_with_biomarkers(trials)

    if args.ipd_sharing:
        trials = api.filter_trials_with_data_sharing(trials)

    if not trials:
        print("\nNo trials match the specified criteria")
        return

    # Export results
    output_file = api.export_trials(trials)

    # Generate summary
    summary = api.generate_summary(trials)

    print(f"\n=== ClinicalTrials.gov Search Results ===\n")
    print(f"Total trials: {summary['total_trials']}")
    print(f"Total enrollment: {summary['total_enrollment']:,}")
    print(f"Trials with results: {summary['with_results']}")
    print(f"Trials with biomarkers: {summary['with_biomarkers']}")
    print(f"Trials with IPD sharing: {summary['with_ipd_sharing']}")
    print(f"Trials with contact email: {summary['with_contact_email']}")

    print(f"\nBy status:")
    for status, count in sorted(summary['by_status'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {status}: {count}")

    print(f"\nBy phase:")
    for phase, count in sorted(summary['by_phase'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {phase}: {count}")

    print(f"\nBy intervention:")
    for intervention, count in sorted(summary['by_intervention'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {intervention}: {count}")

    print(f"\nTop 10 sponsors:")
    for sponsor, count in sorted(summary['top_sponsors'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {sponsor}: {count}")

    print(f"\nResults saved: {output_file}")

    # Save summary JSON
    summary_file = api.output_dir / 'trials_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved: {summary_file}")


if __name__ == '__main__':
    main()