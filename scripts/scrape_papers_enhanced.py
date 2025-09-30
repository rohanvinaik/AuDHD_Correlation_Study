#!/usr/bin/env python3
"""
Enhanced Paper Scraper with Comprehensive Data Lead Extraction

NO DATA THEFT - Proper citation tracking for all extracted information!

Given a paper (PDF/HTML), extracts and validates every actionable lead to external
artifacts (datasets, code, models) - even if only mentioned in prose.

Features:
- Text structuring with section detection
- Comprehensive pattern library for 20+ repository types
- Normalization and validation
- API verification for public repositories
- Classification and confidence scoring
- Full provenance and citation tracking
- Post-publication DataCite discovery

Architecture:
1. Ingest → 2. Structure → 3. Extract → 4. Normalize → 5. Validate →
6. Classify → 7. Score → 8. Log Citations → 9. Output
"""

import argparse
import requests
import json
import time
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote, parse_qs
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Evidence:
    """Evidence for a data lead"""
    text_snippet: str
    section: str
    page: Optional[int] = None
    line_offset: Optional[int] = None
    context_words: List[str] = field(default_factory=list)
    url: Optional[str] = None

@dataclass
class DataLead:
    """A discovered data/code/model lead"""
    identifier: str  # Normalized ID
    lead_type: str  # dataset, code, model
    repository: str  # GEO, SRA, Zenodo, GitHub, etc.
    access_level: str  # verified_public, restricted, request_only, dead_link, ambiguous
    validation_status: str  # verified, mentioned_resolvable, failed
    confidence: float  # 0-1 score
    evidence: List[Evidence]

    # Enrichment from APIs
    title: Optional[str] = None
    submitter: Optional[str] = None
    size_mb: Optional[float] = None
    file_count: Optional[int] = None
    api_metadata: Optional[Dict] = None

    # Provenance
    original_mention: str = ""  # Raw text as found
    normalized_from: Optional[str] = None  # If repaired
    redirects: List[str] = field(default_factory=list)
    paper_variant: str = "VoR"  # VoR, preprint, accepted
    discovered_date: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Citation:
    """Citation information for proper attribution"""
    paper_doi: Optional[str]
    paper_pmid: Optional[str]
    paper_pmc: Optional[str]
    paper_title: str
    authors: List[str]
    journal: str
    year: str
    extracted_leads: List[str]  # List of identifiers extracted
    citation_text: str  # Formatted citation
    data_statement: Optional[str] = None  # Direct quote from Data Availability
    extraction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    license: Optional[str] = None

@dataclass
class PaperAnalysis:
    """Complete analysis of a paper"""
    paper_id: str
    citation: Citation
    leads: List[DataLead]
    sections_analyzed: List[str]
    scrape_date: str
    version: str = "1.0"


# ============================================================================
# Pattern Library - Comprehensive Repository Coverage
# ============================================================================

class RepositoryPatterns:
    """Patterns for recognizing data/code repository identifiers"""

    # Genomics
    GEO = r'\b(GSE|GSM|GPL)\d{3,7}\b'
    SRA = r'\b(SRR|SRX|SRP|SRS|DRR|DRX|DRP|ERR|ERX|ERP)\d{6,9}\b'
    BIOPROJECT = r'\b(PRJNA|PRJEB|PRJDB)\d{4,9}\b'
    DBGAP = r'\bphs\d{6}\b'
    EGA = r'\b(EGAS|EGAD|EGAC)\d{11}\b'
    ARRAYEXPRESS = r'\bE-[A-Z]{4}-\d+\b'
    BIOSAMPLE = r'\b(SAMN|SAMEA|SAMD)\d{7,9}\b'

    # Proteomics
    PRIDE = r'\b(PXD|PRD)\d{6,9}\b'
    MASSIVE = r'\bMSV\d{9}\b'

    # Metabolomics
    METABOLOMICS_WB = r'\bST\d{6}\b'
    METABOLIGHTS = r'\bMTBLS\d+\b'

    # Imaging/Neuro
    OPENNEURO = r'\bds\d{6}\b'
    NEUROVAULT = r'\bneurovault\.org/collections/\d+\b'

    # General repositories
    ZENODO = r'\bzenodo\.org/record/\d+\b'
    FIGSHARE = r'\bfigshare\.com/articles/[^/]+/\d+\b'
    DRYAD = r'\bdatadryad\.org/stash/dataset/doi:[^\s]+\b'
    OSF = r'\bosf\.io/[a-z0-9]{5}\b'
    MENDELEY_DATA = r'\bdata\.mendeley\.com/datasets/[^/]+/\d+\b'

    # Code
    GITHUB_REPO = r'github\.com/([^/\s]+)/([^/\s]+?)(?:\.git|/|$)'
    GITLAB_REPO = r'gitlab\.com/([^/\s]+)/([^/\s]+?)(?:\.git|/|$)'
    BITBUCKET_REPO = r'bitbucket\.org/([^/\s]+)/([^/\s]+?)(?:\.git|/|$)'

    # DOIs (DataCite, Crossref)
    DOI = r'\b10\.\d{4,9}/[^\s]+'

    # Other
    ADDGENE = r'\bAddgene\s+(?:plasmid\s+)?#?(\d{4,6})\b'
    PDB = r'\b[0-9][A-Z0-9]{3}\b'  # Protein Data Bank (needs context)
    UNIPROT = r'\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b'
    BIOSTUDIES = r'\bS-[A-Z]{4}\d+\b'
    EMPIAR = r'\bEMPIAR-\d{5}\b'

    # Restricted/Special
    UKBIOBANK = r'\bUK\s+Biobank\s+application\s+(\d+)\b'
    NDA = r'\b(NDA|NDAR)\s+(?:collection\s+)?(\d+)\b'
    HCP = r'\bHuman\s+Connectome\s+Project\b'

    @classmethod
    def get_all_patterns(cls) -> Dict[str, str]:
        """Return all patterns as dict"""
        patterns = {}
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                val = getattr(cls, attr)
                if isinstance(val, str) and val.startswith('\\b') or '(' in val:
                    patterns[attr] = val
        return patterns


class LexicalTriggers:
    """Context words that indicate data availability"""

    DEPOSIT_TRIGGERS = [
        'deposited in', 'submitted to', 'available at', 'available from',
        'accessible at', 'accessed at', 'downloaded from', 'obtained from',
        'retrieved from', 'hosted at', 'archived at', 'stored in'
    ]

    ACCESSION_TRIGGERS = [
        'accession', 'accession number', 'accession code', 'under accession',
        'with accession', 'ID:', 'identifier:', 'study ID', 'project ID'
    ]

    SECTION_TRIGGERS = [
        'Data Availability', 'Data and Code Availability', 'Code Availability',
        'Accession Codes', 'Resource Availability', 'Data Deposition',
        'Source Data', 'Extended Data', 'Supplementary Data', 'Supplementary Information'
    ]

    REQUEST_TRIGGERS = [
        'upon reasonable request', 'available upon request', 'on request',
        'by request', 'from the corresponding author', 'from the authors',
        'available from the author'
    ]

    @classmethod
    def has_trigger(cls, text: str, trigger_list: List[str]) -> bool:
        """Check if text contains any trigger"""
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in trigger_list)


# ============================================================================
# Text Structuring and Extraction
# ============================================================================

class TextStructurer:
    """Structures paper text into sections"""

    SECTION_PATTERNS = {
        'title': r'(?i)^(?:Title:?\s*)?(.+)$',
        'abstract': r'(?i)(?:^|\n)(?:Abstract|Summary)[\s:]+(.*?)(?=\n(?:Introduction|Background|Methods)|\Z)',
        'methods': r'(?i)(?:^|\n)(Methods?|Materials?\s+and\s+Methods?|Experimental\s+Procedures?)[\s:]+(.*?)(?=\n(?:Results|Discussion|Data\s+Availability)|\Z)',
        'results': r'(?i)(?:^|\n)(Results?)[\s:]+(.*?)(?=\n(?:Discussion|Conclusion|References)|\Z)',
        'data_availability': r'(?i)(?:^|\n)(Data\s+(?:and\s+Code\s+)?Availability|Accession\s+Codes?|Data\s+Deposition|Resource\s+Availability)[\s:]+(.*?)(?=\n[A-Z][a-z]+\s|References|\Z)',
        'code_availability': r'(?i)(?:^|\n)(Code\s+Availability|Software\s+Availability)[\s:]+(.*?)(?=\n[A-Z][a-z]+\s|References|\Z)',
        'acknowledgements': r'(?i)(?:^|\n)(Acknowledgements?|Acknowledgments?)[\s:]+(.*?)(?=\n(?:References|Funding)|\Z)',
        'references': r'(?i)(?:^|\n)(References|Bibliography)[\s:]+(.*)',
        'supplementary': r'(?i)(Supplement(?:ary)?\s+(?:Materials?|Information|Data|Table|Figure)[^\n]*)',
        'figure_captions': r'(?i)(Figure\s+\d+[^\n]*)',
        'extended_data': r'(?i)(Extended\s+Data\s+(?:Figure|Table)\s+\d+[^\n]*)'
    }

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize typography and fix common PDF extraction issues"""
        # Fix soft hyphens and line-break hyphenation
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # Fix Unicode issues
        text = text.replace('\u2010', '-')  # hyphen
        text = text.replace('\u2011', '-')  # non-breaking hyphen
        text = text.replace('\u2012', '-')  # figure dash
        text = text.replace('\u2013', '-')  # en dash
        text = text.replace('\u2014', '--')  # em dash
        text = text.replace('\ufb01', 'fi')  # fi ligature
        text = text.replace('\ufb02', 'fl')  # fl ligature

        # Fix common ID breaks
        text = re.sub(r'GSE\s+(\d+)', r'GSE\1', text)
        text = re.sub(r'PRJ\s*([NE])\s*([AB])\s*(\d+)', r'PRJ\1\2\3', text)
        text = re.sub(r'SRR\s+(\d+)', r'SRR\1', text)

        return text

    @classmethod
    def extract_sections(cls, text: str) -> Dict[str, str]:
        """Extract structured sections from paper text"""
        text = cls.normalize_text(text)

        sections = {}
        for section_name, pattern in cls.SECTION_PATTERNS.items():
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            if matches:
                # Take the first match (or all for repeating sections like figures)
                if section_name in ['figure_captions', 'extended_data', 'supplementary']:
                    sections[section_name] = matches  # List
                else:
                    sections[section_name] = matches[0] if isinstance(matches[0], str) else matches[0][1]

        return sections


# ============================================================================
# Lead Extraction Engine
# ============================================================================

class LeadExtractor:
    """Extracts data/code leads from structured text"""

    def __init__(self):
        self.patterns = RepositoryPatterns.get_all_patterns()

    def extract_candidates(self, sections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract candidate leads from all sections"""
        candidates = []

        # Section weights for scoring
        section_weights = {
            'data_availability': 1.0,
            'code_availability': 1.0,
            'methods': 0.8,
            'results': 0.6,
            'supplementary': 0.7,
            'figure_captions': 0.5,
            'extended_data': 0.5,
            'acknowledgements': 0.3,
            'references': 0.2
        }

        for section_name, content in sections.items():
            weight = section_weights.get(section_name, 0.4)

            # Handle list-type sections
            if isinstance(content, list):
                for item in content:
                    candidates.extend(self._extract_from_text(item, section_name, weight))
            else:
                candidates.extend(self._extract_from_text(content, section_name, weight))

        return candidates

    def _extract_from_text(self, text: str, section: str, weight: float) -> List[Dict]:
        """Extract candidates from a text snippet"""
        candidates = []

        for pattern_name, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                # Extract context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                # Check for triggers
                has_deposit = LexicalTriggers.has_trigger(context, LexicalTriggers.DEPOSIT_TRIGGERS)
                has_accession = LexicalTriggers.has_trigger(context, LexicalTriggers.ACCESSION_TRIGGERS)
                has_request = LexicalTriggers.has_trigger(context, LexicalTriggers.REQUEST_TRIGGERS)

                # Calculate pattern confidence
                pattern_conf = 1.0
                if pattern_name in ['PDB', 'UNIPROT']:  # Ambiguous patterns
                    pattern_conf = 0.5 if (has_deposit or has_accession) else 0.2

                # Context confidence
                context_conf = 0.5
                if has_deposit or has_accession:
                    context_conf = 0.9
                if has_request:
                    context_conf = 0.3

                identifier = match.group(0)

                # Determine repository
                repo = self._pattern_to_repository(pattern_name)

                candidates.append({
                    'identifier': identifier,
                    'original': identifier,
                    'repository': repo,
                    'section': section,
                    'section_weight': weight,
                    'pattern_confidence': pattern_conf,
                    'context_confidence': context_conf,
                    'context_snippet': context.strip(),
                    'has_deposit_trigger': has_deposit,
                    'has_accession_trigger': has_accession,
                    'has_request_trigger': has_request,
                    'pattern_name': pattern_name
                })

        return candidates

    @staticmethod
    def _pattern_to_repository(pattern_name: str) -> str:
        """Map pattern name to repository name"""
        mapping = {
            'GEO': 'GEO',
            'SRA': 'SRA',
            'BIOPROJECT': 'BioProject',
            'DBGAP': 'dbGaP',
            'EGA': 'EGA',
            'ARRAYEXPRESS': 'ArrayExpress',
            'BIOSAMPLE': 'BioSample',
            'PRIDE': 'PRIDE',
            'MASSIVE': 'MassIVE',
            'METABOLOMICS_WB': 'Metabolomics Workbench',
            'METABOLIGHTS': 'MetaboLights',
            'OPENNEURO': 'OpenNeuro',
            'NEUROVAULT': 'NeuroVault',
            'ZENODO': 'Zenodo',
            'FIGSHARE': 'Figshare',
            'DRYAD': 'Dryad',
            'OSF': 'OSF',
            'MENDELEY_DATA': 'Mendeley Data',
            'GITHUB_REPO': 'GitHub',
            'GITLAB_REPO': 'GitLab',
            'BITBUCKET_REPO': 'Bitbucket',
            'DOI': 'DOI',
            'ADDGENE': 'Addgene',
            'PDB': 'PDB',
            'UNIPROT': 'UniProt',
            'BIOSTUDIES': 'BioStudies',
            'EMPIAR': 'EMPIAR',
            'UKBIOBANK': 'UK Biobank',
            'NDA': 'NDA',
            'HCP': 'HCP'
        }
        return mapping.get(pattern_name, 'Unknown')


# ===========================================================================
# Validation and Enrichment
# ============================================================================

class LeadValidator:
    """Validates and enriches leads using repository APIs"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research-Data-Pipeline/1.0 (mailto:research@example.com)'
        })

    def validate_lead(self, candidate: Dict) -> Tuple[str, Dict]:
        """
        Validate a candidate lead

        Returns:
            (validation_status, enrichment_data)
            validation_status: verified_public, mentioned_resolvable, restricted, request_only, failed
        """
        repo = candidate['repository']
        identifier = candidate['identifier']

        # Check for request-only
        if candidate.get('has_request_trigger'):
            return 'request_only', {}

        # Validate based on repository
        if repo == 'GEO':
            return self._validate_geo(identifier)
        elif repo == 'SRA':
            return self._validate_sra(identifier)
        elif repo == 'BioProject':
            return self._validate_bioproject(identifier)
        elif repo in ['dbGaP', 'EGA', 'UK Biobank', 'NDA']:
            return 'restricted', {'note': f'Requires access application to {repo}'}
        elif repo == 'DOI':
            return self._validate_doi(identifier)
        elif repo in ['Zenodo', 'Figshare', 'Dryad', 'OSF']:
            return self._validate_url_based(identifier)
        elif repo in ['GitHub', 'GitLab', 'Bitbucket']:
            return self._validate_code_repo(candidate)
        else:
            # Try to verify it's at least mentioned_resolvable
            return 'mentioned_resolvable', {}

    def _validate_geo(self, geo_id: str) -> Tuple[str, Dict]:
        """Validate GEO accession"""
        try:
            url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_id}&targ=self&form=text&view=brief"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200 and 'Series' in response.text or 'Sample' in response.text:
                # Parse some metadata
                title_match = re.search(r'!Series_title = (.+)', response.text)
                title = title_match.group(1) if title_match else None

                return 'verified_public', {
                    'title': title,
                    'api_response': 'GEO API confirmed'
                }
            else:
                return 'failed', {'error': 'GEO ID not found'}

        except Exception as e:
            logger.warning(f"GEO validation failed for {geo_id}: {e}")
            return 'failed', {'error': str(e)}

    def _validate_sra(self, sra_id: str) -> Tuple[str, Dict]:
        """Validate SRA accession"""
        try:
            # Use NCBI E-utilities
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {'db': 'sra', 'term': sra_id, 'retmode': 'json'}
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('esearchresult', {}).get('count', '0') != '0':
                    return 'verified_public', {'api_response': 'SRA API confirmed'}

            return 'failed', {'error': 'SRA ID not found'}

        except Exception as e:
            logger.warning(f"SRA validation failed for {sra_id}: {e}")
            return 'failed', {'error': str(e)}

    def _validate_bioproject(self, project_id: str) -> Tuple[str, Dict]:
        """Validate BioProject accession"""
        try:
            url = f"https://www.ncbi.nlm.nih.gov/bioproject/{project_id}"
            response = self.session.get(url, timeout=10, allow_redirects=True)

            if response.status_code == 200 and project_id in response.text:
                return 'verified_public', {'api_response': 'BioProject confirmed'}

            return 'failed', {'error': 'BioProject not found'}

        except Exception as e:
            return 'failed', {'error': str(e)}

    def _validate_doi(self, doi: str) -> Tuple[str, Dict]:
        """Validate DOI via DataCite/Crossref"""
        doi = doi.lower().replace('https://doi.org/', '').replace('http://doi.org/', '')

        try:
            # Try DataCite API (for data DOIs)
            url = f"https://api.datacite.org/dois/{doi}"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                attrs = data.get('data', {}).get('attributes', {})

                return 'verified_public', {
                    'title': attrs.get('titles', [{}])[0].get('title'),
                    'publisher': attrs.get('publisher'),
                    'resource_type': attrs.get('types', {}).get('resourceType'),
                    'api_response': 'DataCite confirmed'
                }

            # Fallback: just check if DOI resolves
            resolve_url = f"https://doi.org/{doi}"
            response = self.session.head(resolve_url, timeout=10, allow_redirects=True)

            if response.status_code == 200:
                return 'mentioned_resolvable', {'resolved_url': response.url}

            return 'failed', {'error': 'DOI not found'}

        except Exception as e:
            return 'failed', {'error': str(e)}

    def _validate_url_based(self, identifier: str) -> Tuple[str, Dict]:
        """Validate URL-based repository entry"""
        try:
            # Construct URL if not already a URL
            if not identifier.startswith('http'):
                if 'zenodo.org/record/' in identifier:
                    url = f"https://{identifier}"
                else:
                    return 'mentioned_resolvable', {}
            else:
                url = identifier

            response = self.session.head(url, timeout=10, allow_redirects=True)

            if response.status_code == 200:
                return 'verified_public', {'resolved_url': response.url}
            else:
                return 'failed', {'error': f'HTTP {response.status_code}'}

        except Exception as e:
            return 'failed', {'error': str(e)}

    def _validate_code_repo(self, candidate: Dict) -> Tuple[str, Dict]:
        """Validate code repository"""
        identifier = candidate['identifier']

        try:
            # GitHub API
            if 'github.com' in identifier:
                match = re.search(r'github\.com/([^/]+)/([^/\s]+)', identifier)
                if match:
                    owner, repo = match.groups()
                    repo = repo.rstrip('.git')

                    url = f"https://api.github.com/repos/{owner}/{repo}"
                    response = self.session.get(url, timeout=10)

                    if response.status_code == 200:
                        data = response.json()
                        return 'verified_public', {
                            'title': data.get('full_name'),
                            'description': data.get('description'),
                            'stars': data.get('stargazers_count'),
                            'api_response': 'GitHub API confirmed'
                        }

            # Otherwise just check if URL resolves
            url = identifier if identifier.startswith('http') else f"https://{identifier}"
            response = self.session.head(url, timeout=10, allow_redirects=True)

            if response.status_code == 200:
                return 'mentioned_resolvable', {'resolved_url': response.url}

            return 'failed', {'error': f'HTTP {response.status_code}'}

        except Exception as e:
            return 'failed', {'error': str(e)}


# ============================================================================
# Citation Tracker
# ============================================================================

class CitationTracker:
    """
    Tracks citations for proper attribution
    NO DATA THEFT - All extracted data properly attributed!
    """

    @staticmethod
    def create_citation(paper_metadata: Dict, extracted_ids: List[str],
                        data_statement: Optional[str] = None) -> Citation:
        """Create citation object for a paper"""

        # Format author list
        authors = paper_metadata.get('authors', [])
        if len(authors) > 10:
            author_str = f"{authors[0]} et al."
        elif len(authors) > 0:
            author_str = ', '.join(authors)
        else:
            author_str = "Unknown authors"

        # Format citation text (APA style)
        year = paper_metadata.get('year', 'n.d.')
        title = paper_metadata.get('title', 'Untitled')
        journal = paper_metadata.get('journal', 'Unknown journal')
        doi = paper_metadata.get('doi')

        citation_text = f"{author_str} ({year}). {title}. {journal}."
        if doi:
            citation_text += f" https://doi.org/{doi}"

        return Citation(
            paper_doi=doi,
            paper_pmid=paper_metadata.get('pmid'),
            paper_pmc=paper_metadata.get('pmc_id'),
            paper_title=title,
            authors=authors,
            journal=journal,
            year=str(year),
            extracted_leads=extracted_ids,
            citation_text=citation_text,
            data_statement=data_statement,
            license=paper_metadata.get('license')
        )

    @staticmethod
    def generate_attribution_file(citations: List[Citation], output_path: Path):
        """Generate human-readable attribution file"""
        with open(output_path, 'w') as f:
            f.write("# Data Source Attribution\n\n")
            f.write("All data leads were extracted from published scientific literature.\n")
            f.write("Proper attribution is provided below.\n\n")
            f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
            f.write("---\n\n")

            for i, citation in enumerate(citations, 1):
                f.write(f"## Source {i}\n\n")
                f.write(f"**Citation:** {citation.citation_text}\n\n")

                if citation.paper_pmid:
                    f.write(f"**PMID:** {citation.paper_pmid}\n\n")

                if citation.data_statement:
                    f.write(f"**Data Availability Statement:**\n")
                    f.write(f"> {citation.data_statement}\n\n")

                f.write(f"**Data Leads Extracted:**\n")
                for lead_id in citation.extracted_leads:
                    f.write(f"- `{lead_id}`\n")

                f.write("\n---\n\n")

        logger.info(f"Attribution file generated: {output_path}")


# ============================================================================
# Main Enhanced Scraper
# ============================================================================

class EnhancedPaperScraper:
    """
    Enhanced paper scraper with comprehensive lead extraction
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.output_dir / "cache"
        self.leads_dir = self.output_dir / "leads"
        self.citations_dir = self.output_dir / "citations"

        for d in [self.cache_dir, self.leads_dir, self.citations_dir]:
            d.mkdir(exist_ok=True)

        self.extractor = LeadExtractor()
        self.validator = LeadValidator(self.cache_dir)
        self.structurer = TextStructurer()

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 Research-Pipeline/1.0'
        })

    def analyze_paper(self, paper_text: str, paper_metadata: Dict,
                     paper_variant: str = "VoR") -> PaperAnalysis:
        """
        Complete analysis pipeline for a single paper

        Args:
            paper_text: Full text of paper
            paper_metadata: Metadata (DOI, PMID, title, etc.)
            paper_variant: "VoR", "preprint", or "accepted"

        Returns:
            PaperAnalysis object
        """
        logger.info(f"Analyzing paper: {paper_metadata.get('title', 'Unknown')[:80]}...")

        # 1. Structure text
        sections = self.structurer.extract_sections(paper_text)
        logger.info(f"  Extracted {len(sections)} sections")

        # 2. Extract candidates
        candidates = self.extractor.extract_candidates(sections)
        logger.info(f"  Found {len(candidates)} candidate leads")

        # 3. Normalize and deduplicate
        normalized_candidates = self._normalize_candidates(candidates)
        logger.info(f"  Normalized to {len(normalized_candidates)} unique leads")

        # 4. Validate and enrich
        validated_leads = []
        for candidate in normalized_candidates:
            validation_status, enrichment = self.validator.validate_lead(candidate)

            # Create DataLead object
            lead = self._create_data_lead(candidate, validation_status, enrichment, paper_variant)
            validated_leads.append(lead)

        logger.info(f"  Validated {len(validated_leads)} leads")

        # 5. Score and classify
        scored_leads = self._score_leads(validated_leads)

        # 6. Extract data availability statement for citation
        data_statement = sections.get('data_availability', sections.get('code_availability'))
        if isinstance(data_statement, list):
            data_statement = ' '.join(data_statement)

        # 7. Create citation
        extracted_ids = [lead.identifier for lead in scored_leads]
        citation = CitationTracker.create_citation(paper_metadata, extracted_ids, data_statement)

        # 8. Create analysis object
        analysis = PaperAnalysis(
            paper_id=paper_metadata.get('pmid', paper_metadata.get('doi', 'unknown')),
            citation=citation,
            leads=scored_leads,
            sections_analyzed=list(sections.keys()),
            scrape_date=datetime.now().isoformat(),
            version="1.0"
        )

        return analysis

    def _normalize_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Normalize and deduplicate candidates"""
        normalized = {}

        for cand in candidates:
            identifier = cand['identifier']

            # Normalize ID
            normalized_id = identifier.upper().strip()
            normalized_id = re.sub(r'\s+', '', normalized_id)  # Remove spaces

            # Deduplicate - keep highest confidence
            if normalized_id in normalized:
                existing = normalized[normalized_id]
                if cand['section_weight'] > existing['section_weight']:
                    normalized[normalized_id] = cand
            else:
                normalized[normalized_id] = cand

            cand['normalized_id'] = normalized_id

        return list(normalized.values())

    def _create_data_lead(self, candidate: Dict, validation_status: str,
                         enrichment: Dict, paper_variant: str) -> DataLead:
        """Create DataLead object from candidate"""

        # Determine lead type
        repo = candidate['repository']
        if repo in ['GitHub', 'GitLab', 'Bitbucket']:
            lead_type = 'code'
        elif repo in ['Addgene']:
            lead_type = 'model'
        else:
            lead_type = 'dataset'

        # Determine access level
        if validation_status == 'verified_public':
            access_level = 'verified_public'
        elif validation_status == 'restricted':
            access_level = 'restricted'
        elif validation_status == 'request_only':
            access_level = 'request_only'
        elif validation_status == 'failed':
            access_level = 'dead_link'
        else:
            access_level = 'mentioned_resolvable'

        # Create evidence
        evidence = [Evidence(
            text_snippet=candidate['context_snippet'],
            section=candidate['section'],
            context_words=[k for k, v in candidate.items() if 'trigger' in k and v]
        )]

        return DataLead(
            identifier=candidate['normalized_id'],
            lead_type=lead_type,
            repository=repo,
            access_level=access_level,
            validation_status=validation_status,
            confidence=0.0,  # Will be calculated in scoring
            evidence=evidence,
            title=enrichment.get('title'),
            submitter=enrichment.get('submitter'),
            api_metadata=enrichment if enrichment else None,
            original_mention=candidate['original'],
            normalized_from=candidate['identifier'] if candidate['identifier'] != candidate['normalized_id'] else None,
            paper_variant=paper_variant
        )

    def _score_leads(self, leads: List[DataLead]) -> List[DataLead]:
        """Calculate confidence scores for leads"""
        for lead in leads:
            # Start with base scores from evidence
            ev = lead.evidence[0] if lead.evidence else None
            if not ev:
                lead.confidence = 0.1
                continue

            # Section weight (from candidate extraction)
            section_weight = 0.5  # Default
            if 'availability' in ev.section.lower():
                section_weight = 1.0
            elif ev.section.lower() in ['methods', 'supplementary']:
                section_weight = 0.7

            # Context weight
            context_weight = 0.7 if ev.context_words else 0.4

            # Validation weight
            if lead.validation_status == 'verified_public':
                validation_weight = 1.0
            elif lead.validation_status in ['mentioned_resolvable', 'restricted']:
                validation_weight = 0.7
            elif lead.validation_status == 'request_only':
                validation_weight = 0.5
            else:
                validation_weight = 0.2

            # Combined confidence
            confidence = (section_weight * 0.3 + context_weight * 0.3 + validation_weight * 0.4)
            lead.confidence = min(1.0, max(0.0, confidence))

        # Sort by confidence
        leads.sort(key=lambda x: x.confidence, reverse=True)

        return leads

    def save_analysis(self, analysis: PaperAnalysis):
        """Save analysis results"""
        # Save JSON
        output_file = self.leads_dir / f"{analysis.paper_id}_leads.json"
        with open(output_file, 'w') as f:
            # Convert dataclasses to dict
            data = {
                'paper_id': analysis.paper_id,
                'citation': asdict(analysis.citation),
                'leads': [asdict(lead) for lead in analysis.leads],
                'sections_analyzed': analysis.sections_analyzed,
                'scrape_date': analysis.scrape_date,
                'version': analysis.version
            }
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Analysis saved: {output_file}")

        # Save citation
        citation_file = self.citations_dir / f"{analysis.paper_id}_citation.txt"
        with open(citation_file, 'w') as f:
            f.write(f"Paper: {analysis.citation.paper_title}\n")
            f.write(f"Citation: {analysis.citation.citation_text}\n\n")

            if analysis.citation.data_statement:
                f.write(f"Data Availability Statement:\n{analysis.citation.data_statement}\n\n")

            f.write(f"Data Leads Extracted ({len(analysis.leads)}):\n")
            for lead in analysis.leads:
                f.write(f"- {lead.identifier} ({lead.repository}) - {lead.access_level}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced paper scraper with comprehensive data lead extraction'
    )
    parser.add_argument('--pmid', help='PubMed ID to analyze')
    parser.add_argument('--doi', help='DOI to analyze')
    parser.add_argument('--text-file', help='Text file containing paper content')
    parser.add_argument('--output', default='data/papers_enhanced',
                       help='Output directory')
    parser.add_argument('--audit-only', action='store_true',
                       help='Extract and validate only, no downloading')

    args = parser.parse_args()

    scraper = EnhancedPaperScraper(Path(args.output))

    # For demonstration, using simple text input
    if args.text_file:
        with open(args.text_file) as f:
            paper_text = f.read()

        paper_metadata = {
            'title': 'Test Paper',
            'authors': ['Author A', 'Author B'],
            'year': '2024',
            'journal': 'Test Journal'
        }

        analysis = scraper.analyze_paper(paper_text, paper_metadata)
        scraper.save_analysis(analysis)

        print(f"\nAnalysis complete!")
        print(f"Found {len(analysis.leads)} data leads")
        print(f"Verified public: {sum(1 for l in analysis.leads if l.access_level == 'verified_public')}")
        print(f"Restricted: {sum(1 for l in analysis.leads if l.access_level == 'restricted')}")
        print(f"\nResults saved to: {args.output}")
    else:
        print("Please provide --text-file, --pmid, or --doi")
        print("Full PubMed/PMC integration coming soon!")


if __name__ == '__main__':
    main()
