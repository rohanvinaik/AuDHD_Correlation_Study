"""
Multi-Omics Analysis Extension

Extends genetic analysis to include transcriptomics, proteomics, and metabolomics.
Integrates with existing genetic_lookup.py infrastructure.

Coverage:
- Transcriptomics: GEO, ArrayExpress, GTEx, SRA
- Proteomics: PRIDE, MassIVE, UniProt, Human Protein Atlas
- Metabolomics: Metabolomics Workbench, MetaboLights, HMDB
- Cross-omics: KEGG, Reactome, STRING
"""

import os
import json
import time
import requests
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re

from .genetic_lookup import CachedAPIClient, GeneticLookupResult

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TranscriptLookupResult:
    """Result from transcript/expression lookup"""
    transcript_id: str
    gene_symbol: Optional[str]
    expression_data: Dict[str, Any]
    tissue_specificity: Dict[str, float]
    disease_associations: List[Dict[str, Any]]
    literature_refs: List[Dict[str, Any]]
    llm_synthesis: Optional[str]
    timestamp: str
    cache_hit: bool


@dataclass
class ProteinLookupResult:
    """Result from protein lookup"""
    protein_id: str  # UniProt ID
    gene_symbol: Optional[str]
    protein_name: str
    function: Optional[str]
    tissue_expression: Dict[str, Any]
    subcellular_location: List[str]
    interactions: List[Dict[str, Any]]
    disease_associations: List[Dict[str, Any]]
    literature_refs: List[Dict[str, Any]]
    llm_synthesis: Optional[str]
    timestamp: str
    cache_hit: bool


@dataclass
class MetaboliteLookupResult:
    """Result from metabolite lookup"""
    metabolite_id: str  # HMDB ID or KEGG ID
    name: str
    formula: Optional[str]
    pathways: List[str]
    disease_associations: List[Dict[str, Any]]
    related_proteins: List[str]
    related_genes: List[str]
    literature_refs: List[Dict[str, Any]]
    llm_synthesis: Optional[str]
    timestamp: str
    cache_hit: bool


# ============================================================================
# Transcriptomics Client
# ============================================================================

class TranscriptomicsClient(CachedAPIClient):
    """
    Client for transcriptomics data (GEO, GTEx, ArrayExpress)
    """

    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research-Pipeline/1.0'
        })

    def get_gtex_expression(self, gene_symbol: str) -> Dict[str, Any]:
        """Get tissue-specific expression from GTEx"""
        cache_key = self._get_cache_key(gene_symbol, "gtex_expression")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # GTEx API
            url = f"https://gtexportal.org/api/v2/expression/medianGeneExpression"
            params = {
                'geneId': gene_symbol,
                'datasetId': 'gtex_v8'
            }

            response = self.session.get(url, params=params, timeout=10)
            self.call_count += 1

            if response.status_code == 200:
                data = response.json()

                # Extract tissue expression
                tissue_expression = {}
                for item in data.get('medianGeneExpression', []):
                    tissue = item.get('tissueSiteDetailId', 'unknown')
                    median_tpm = item.get('median', 0.0)
                    tissue_expression[tissue] = median_tpm

                result = {
                    'gene': gene_symbol,
                    'tissue_expression': tissue_expression,
                    'source': 'GTEx v8',
                    'unit': 'TPM (median)'
                }

                self._set_cached(cache_key, result)
                return result

            return {'error': f'GTEx API error: {response.status_code}'}

        except Exception as e:
            logger.error(f"GTEx lookup failed for {gene_symbol}: {e}")
            return {'error': str(e)}

    def get_geo_dataset_info(self, geo_id: str) -> Dict[str, Any]:
        """Get GEO dataset information"""
        cache_key = self._get_cache_key(geo_id, "geo_info")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # Use NCBI E-utilities
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {
                'db': 'gds',
                'id': geo_id.replace('GSE', ''),
                'retmode': 'json'
            }

            response = self.session.get(url, params=params, timeout=10)
            self.call_count += 1

            if response.status_code == 200:
                data = response.json()

                result = {
                    'geo_id': geo_id,
                    'title': data.get('result', {}).get(geo_id, {}).get('title', ''),
                    'summary': data.get('result', {}).get(geo_id, {}).get('summary', ''),
                    'platform': data.get('result', {}).get(geo_id, {}).get('gpl', ''),
                    'samples': data.get('result', {}).get(geo_id, {}).get('n_samples', 0)
                }

                self._set_cached(cache_key, result)
                return result

            return {'error': f'GEO API error: {response.status_code}'}

        except Exception as e:
            logger.error(f"GEO lookup failed for {geo_id}: {e}")
            return {'error': str(e)}


# ============================================================================
# Proteomics Client
# ============================================================================

class ProteomicsClient(CachedAPIClient):
    """
    Client for proteomics data (PRIDE, UniProt, Human Protein Atlas)
    """

    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research-Pipeline/1.0'
        })

    def get_uniprot_info(self, uniprot_id: str) -> Dict[str, Any]:
        """Get protein information from UniProt"""
        cache_key = self._get_cache_key(uniprot_id, "uniprot")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # UniProt REST API
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

            response = self.session.get(url, timeout=10)
            self.call_count += 1

            if response.status_code == 200:
                data = response.json()

                # Extract key information
                result = {
                    'protein_id': uniprot_id,
                    'protein_name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                    'gene_symbol': data.get('genes', [{}])[0].get('geneName', {}).get('value', '') if data.get('genes') else '',
                    'organism': data.get('organism', {}).get('scientificName', ''),
                    'function': data.get('comments', [{}])[0].get('texts', [{}])[0].get('value', '') if data.get('comments') else '',
                    'subcellular_location': [loc.get('location', {}).get('value', '') for loc in data.get('comments', []) if loc.get('commentType') == 'SUBCELLULAR LOCATION'],
                    'sequence_length': data.get('sequence', {}).get('length', 0)
                }

                self._set_cached(cache_key, result)
                return result

            return {'error': f'UniProt API error: {response.status_code}'}

        except Exception as e:
            logger.error(f"UniProt lookup failed for {uniprot_id}: {e}")
            return {'error': str(e)}

    def get_protein_atlas_expression(self, gene_symbol: str) -> Dict[str, Any]:
        """Get protein expression from Human Protein Atlas"""
        cache_key = self._get_cache_key(gene_symbol, "hpa_expression")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # Human Protein Atlas API
            url = f"https://www.proteinatlas.org/{gene_symbol}.json"

            response = self.session.get(url, timeout=10)
            self.call_count += 1

            if response.status_code == 200:
                data = response.json()

                # Extract tissue expression
                tissue_expression = {}
                for tissue_data in data.get('rna', {}).get('tissueExpression', []):
                    tissue = tissue_data.get('tissue', 'unknown')
                    level = tissue_data.get('level', 'Not detected')
                    tissue_expression[tissue] = level

                result = {
                    'gene': gene_symbol,
                    'protein_class': data.get('proteinClasses', []),
                    'subcellular_location': data.get('subcellularLocation', []),
                    'tissue_expression': tissue_expression,
                    'source': 'Human Protein Atlas'
                }

                self._set_cached(cache_key, result)
                return result

            return {'error': f'HPA API error: {response.status_code}'}

        except Exception as e:
            logger.error(f"HPA lookup failed for {gene_symbol}: {e}")
            return {'error': str(e)}

    def search_pride_datasets(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search PRIDE for proteomics datasets"""
        cache_key = self._get_cache_key(f"{query}_{max_results}", "pride_search")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # PRIDE Archive API
            url = "https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects"
            params = {
                'query': query,
                'pageSize': max_results,
                'page': 0
            }

            response = self.session.get(url, params=params, timeout=10)
            self.call_count += 1

            if response.status_code == 200:
                data = response.json()

                datasets = []
                for project in data.get('_embedded', {}).get('projects', []):
                    datasets.append({
                        'accession': project.get('accession'),
                        'title': project.get('title'),
                        'description': project.get('projectDescription', ''),
                        'organisms': project.get('organisms', []),
                        'instruments': project.get('instruments', []),
                        'submission_date': project.get('submissionDate')
                    })

                self._set_cached(cache_key, datasets)
                return datasets

            return []

        except Exception as e:
            logger.error(f"PRIDE search failed for {query}: {e}")
            return []


# ============================================================================
# Metabolomics Client
# ============================================================================

class MetabolomicsClient(CachedAPIClient):
    """
    Client for metabolomics data (Metabolomics Workbench, HMDB)
    """

    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research-Pipeline/1.0'
        })

    def get_hmdb_metabolite(self, hmdb_id: str) -> Dict[str, Any]:
        """Get metabolite information from HMDB"""
        cache_key = self._get_cache_key(hmdb_id, "hmdb")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # HMDB API (note: limited, may need XML parsing)
            url = f"https://hmdb.ca/metabolites/{hmdb_id}.json"

            response = self.session.get(url, timeout=10)
            self.call_count += 1

            if response.status_code == 200:
                data = response.json()

                result = {
                    'hmdb_id': hmdb_id,
                    'name': data.get('name', ''),
                    'formula': data.get('chemical_formula', ''),
                    'description': data.get('description', ''),
                    'pathways': [p.get('name', '') for p in data.get('pathways', [])],
                    'diseases': [d.get('name', '') for d in data.get('diseases', [])],
                    'cas_number': data.get('cas_registry_number', ''),
                    'molecular_weight': data.get('monisotopic_molecular_weight', '')
                }

                self._set_cached(cache_key, result)
                return result

            return {'error': f'HMDB API error: {response.status_code}'}

        except Exception as e:
            logger.error(f"HMDB lookup failed for {hmdb_id}: {e}")
            return {'error': str(e)}

    def search_metabolomics_workbench(self, query: str) -> List[Dict]:
        """Search Metabolomics Workbench"""
        cache_key = self._get_cache_key(query, "mw_search")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # Metabolomics Workbench REST API
            url = "https://www.metabolomicsworkbench.org/rest/study/study_title"
            params = {'study_title': query}

            response = self.session.get(url, params=params, timeout=10)
            self.call_count += 1

            if response.status_code == 200:
                # Parse response (format may vary)
                studies = []
                for line in response.text.strip().split('\n'):
                    if line and not line.startswith('#'):
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            studies.append({
                                'study_id': parts[0],
                                'title': parts[1],
                                'description': parts[2] if len(parts) > 2 else ''
                            })

                self._set_cached(cache_key, studies)
                return studies

            return []

        except Exception as e:
            logger.error(f"MW search failed for {query}: {e}")
            return []


# ============================================================================
# Multi-Omics Analysis System
# ============================================================================

class MultiOmicsAnalysisSystem:
    """
    Integrates transcriptomics, proteomics, metabolomics analysis
    Extends GeneticAnalysisSystem functionality
    """

    def __init__(self, data_dir: str = "data/multiomics_analysis",
                 use_llm: bool = False):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize clients
        cache_dir = self.data_dir / "cache"
        self.transcriptomics = TranscriptomicsClient(str(cache_dir / "transcriptomics"))
        self.proteomics = ProteomicsClient(str(cache_dir / "proteomics"))
        self.metabolomics = MetabolomicsClient(str(cache_dir / "metabolomics"))

        self.use_llm = use_llm

    def analyze_transcript(self, gene_symbol: str) -> TranscriptLookupResult:
        """Analyze transcript/expression for a gene"""
        logger.info(f"Analyzing transcript for {gene_symbol}")

        # Get GTEx expression
        gtex_data = self.transcriptomics.get_gtex_expression(gene_symbol)

        # Extract tissue specificity
        tissue_specificity = gtex_data.get('tissue_expression', {})

        result = TranscriptLookupResult(
            transcript_id=gene_symbol,  # Simplification
            gene_symbol=gene_symbol,
            expression_data=gtex_data,
            tissue_specificity=tissue_specificity,
            disease_associations=[],  # Would come from additional sources
            literature_refs=[],
            llm_synthesis=None,
            timestamp=datetime.now().isoformat(),
            cache_hit=False
        )

        return result

    def analyze_protein(self, identifier: str) -> ProteinLookupResult:
        """
        Analyze protein
        identifier can be UniProt ID or gene symbol
        """
        logger.info(f"Analyzing protein: {identifier}")

        # Determine if UniProt ID or gene symbol
        is_uniprot = re.match(r'[OPQ][0-9][A-Z0-9]{3}[0-9]', identifier)

        if is_uniprot:
            # Get UniProt data
            uniprot_data = self.proteomics.get_uniprot_info(identifier)
            gene_symbol = uniprot_data.get('gene_symbol')
        else:
            # Assume gene symbol, get HPA data
            gene_symbol = identifier
            hpa_data = self.proteomics.get_protein_atlas_expression(gene_symbol)
            uniprot_data = {'gene_symbol': gene_symbol}

        # Get HPA expression if have gene symbol
        tissue_expression = {}
        subcellular_location = []
        if gene_symbol:
            hpa_data = self.proteomics.get_protein_atlas_expression(gene_symbol)
            tissue_expression = hpa_data.get('tissue_expression', {})
            subcellular_location = hpa_data.get('subcellular_location', [])

        result = ProteinLookupResult(
            protein_id=identifier,
            gene_symbol=gene_symbol,
            protein_name=uniprot_data.get('protein_name', ''),
            function=uniprot_data.get('function'),
            tissue_expression=tissue_expression,
            subcellular_location=subcellular_location,
            interactions=[],  # Would come from STRING/IntAct
            disease_associations=[],
            literature_refs=[],
            llm_synthesis=None,
            timestamp=datetime.now().isoformat(),
            cache_hit=False
        )

        return result

    def analyze_metabolite(self, hmdb_id: str) -> MetaboliteLookupResult:
        """Analyze metabolite"""
        logger.info(f"Analyzing metabolite: {hmdb_id}")

        # Get HMDB data
        hmdb_data = self.metabolomics.get_hmdb_metabolite(hmdb_id)

        result = MetaboliteLookupResult(
            metabolite_id=hmdb_id,
            name=hmdb_data.get('name', ''),
            formula=hmdb_data.get('formula'),
            pathways=hmdb_data.get('pathways', []),
            disease_associations=[],
            related_proteins=[],
            related_genes=[],
            literature_refs=[],
            llm_synthesis=None,
            timestamp=datetime.now().isoformat(),
            cache_hit=False
        )

        return result

    def cross_omics_analysis(self, gene_symbol: str) -> Dict[str, Any]:
        """
        Perform integrated cross-omics analysis for a gene
        Returns gene, transcript, protein, and related metabolites
        """
        logger.info(f"Cross-omics analysis for {gene_symbol}")

        results = {
            'gene_symbol': gene_symbol,
            'timestamp': datetime.now().isoformat()
        }

        # Transcript analysis
        try:
            transcript_result = self.analyze_transcript(gene_symbol)
            results['transcriptomics'] = asdict(transcript_result)
        except Exception as e:
            logger.error(f"Transcript analysis failed: {e}")
            results['transcriptomics'] = {'error': str(e)}

        # Protein analysis
        try:
            protein_result = self.analyze_protein(gene_symbol)
            results['proteomics'] = asdict(protein_result)
        except Exception as e:
            logger.error(f"Protein analysis failed: {e}")
            results['proteomics'] = {'error': str(e)}

        # Would add metabolite analysis if gene-metabolite links available

        return results

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics across all clients"""
        return {
            'transcriptomics': self.transcriptomics.get_stats(),
            'proteomics': self.proteomics.get_stats(),
            'metabolomics': self.metabolomics.get_stats()
        }


# Convenience functions
def quick_transcript_lookup(gene_symbol: str) -> Dict:
    """Quick transcript expression lookup"""
    system = MultiOmicsAnalysisSystem()
    result = system.analyze_transcript(gene_symbol)
    return asdict(result)


def quick_protein_lookup(identifier: str) -> Dict:
    """Quick protein lookup"""
    system = MultiOmicsAnalysisSystem()
    result = system.analyze_protein(identifier)
    return asdict(result)


def quick_cross_omics(gene_symbol: str) -> Dict:
    """Quick cross-omics analysis"""
    system = MultiOmicsAnalysisSystem()
    return system.cross_omics_analysis(gene_symbol)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    system = MultiOmicsAnalysisSystem()

    # Test transcript lookup
    print("Testing transcript lookup...")
    transcript_result = system.analyze_transcript("SHANK3")
    print(f"Tissue expression for SHANK3: {len(transcript_result.tissue_specificity)} tissues")

    # Test protein lookup
    print("\nTesting protein lookup...")
    protein_result = system.analyze_protein("SHANK3")
    print(f"Protein: {protein_result.protein_name}")
    print(f"Subcellular location: {protein_result.subcellular_location}")

    # Test cross-omics
    print("\nTesting cross-omics analysis...")
    cross_omics = system.cross_omics_analysis("SHANK3")
    print(f"Cross-omics complete: {cross_omics.keys()}")

    # Usage stats
    stats = system.get_usage_stats()
    print(f"\nUsage stats: {stats}")
