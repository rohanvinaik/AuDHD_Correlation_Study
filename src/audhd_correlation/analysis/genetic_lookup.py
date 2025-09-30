"""
Genetic Lookup and Analysis System

Lightweight system for:
- BLAST and database queries for genetic variants/genes
- Literature mining from PubMed/PMC
- Functional annotation aggregation
- Causal pathway inference
- Optional LLM-based synthesis (cost-optimized)

Cost target: <$1/month with caching and optimization
"""

import os
import json
import time
import hashlib
import requests
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeneticLookupResult:
    """Result from genetic database lookup"""
    gene_symbol: str
    variant_id: Optional[str]
    functional_annotations: Dict[str, Any]
    disease_associations: List[Dict[str, Any]]
    literature_refs: List[Dict[str, Any]]
    pathways: List[str]
    blast_results: Optional[Dict[str, Any]]
    causal_connections: List[Dict[str, Any]]
    llm_synthesis: Optional[str]
    timestamp: str
    cache_hit: bool


class CachedAPIClient:
    """
    Base class for API clients with aggressive caching
    Minimizes API calls and tracks costs
    """

    def __init__(self, cache_dir: str, cache_ttl_days: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(days=cache_ttl_days)
        self.call_count = 0
        self.cache_hits = 0

    def _get_cache_key(self, query: str, endpoint: str) -> str:
        """Generate cache key from query"""
        key_string = f"{endpoint}:{query}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        """Retrieve from cache if fresh"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)

            # Check TTL
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cached_time < self.cache_ttl:
                self.cache_hits += 1
                logger.debug(f"Cache hit for {cache_key}")
                return cached['data']
            else:
                logger.debug(f"Cache expired for {cache_key}")
                return None
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def _set_cached(self, cache_key: str, data: Dict):
        """Store in cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics"""
        return {
            'api_calls': self.call_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.call_count + self.cache_hits)
        }


class NCBIClient(CachedAPIClient):
    """
    Client for NCBI APIs (E-utilities, dbSNP, ClinVar, Gene, BLAST)
    All free, no API key required (but rate-limited)
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

    def __init__(self, cache_dir: str, email: str = "research@example.com"):
        super().__init__(cache_dir)
        self.email = email
        self.rate_limit_delay = 0.34  # 3 requests/second for non-API-key
        self.last_request = time.time()

    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request = time.time()

    def search_gene(self, gene_symbol: str) -> Dict[str, Any]:
        """Search for gene information"""
        cache_key = self._get_cache_key(gene_symbol, "gene_search")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        self._rate_limit()
        self.call_count += 1

        try:
            # Search for gene ID
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            params = {
                'db': 'gene',
                'term': f"{gene_symbol}[Gene Name] AND human[Organism]",
                'retmode': 'json',
                'email': self.email
            }
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            search_result = response.json()

            if not search_result.get('esearchresult', {}).get('idlist'):
                return {'error': 'Gene not found'}

            gene_id = search_result['esearchresult']['idlist'][0]

            # Fetch gene details
            self._rate_limit()
            fetch_url = f"{self.BASE_URL}/esummary.fcgi"
            params = {
                'db': 'gene',
                'id': gene_id,
                'retmode': 'json',
                'email': self.email
            }
            response = requests.get(fetch_url, params=params, timeout=10)
            response.raise_for_status()
            gene_data = response.json()

            result = {
                'gene_id': gene_id,
                'symbol': gene_data['result'][gene_id].get('name', gene_symbol),
                'description': gene_data['result'][gene_id].get('description', ''),
                'chromosome': gene_data['result'][gene_id].get('chromosome', ''),
                'summary': gene_data['result'][gene_id].get('summary', ''),
                'gene_type': gene_data['result'][gene_id].get('geneticsource', '')
            }

            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"NCBI gene search error: {e}")
            return {'error': str(e)}

    def search_variant(self, variant_id: str) -> Dict[str, Any]:
        """Search dbSNP/ClinVar for variant information"""
        cache_key = self._get_cache_key(variant_id, "variant_search")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        self._rate_limit()
        self.call_count += 1

        try:
            # Search in ClinVar first (more clinical info)
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            params = {
                'db': 'clinvar',
                'term': variant_id,
                'retmode': 'json',
                'email': self.email
            }
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            search_result = response.json()

            result = {'variant_id': variant_id}

            if search_result.get('esearchresult', {}).get('idlist'):
                clinvar_id = search_result['esearchresult']['idlist'][0]

                # Fetch ClinVar details
                self._rate_limit()
                fetch_url = f"{self.BASE_URL}/esummary.fcgi"
                params = {
                    'db': 'clinvar',
                    'id': clinvar_id,
                    'retmode': 'json',
                    'email': self.email
                }
                response = requests.get(fetch_url, params=params, timeout=10)
                response.raise_for_status()
                clinvar_data = response.json()

                variant_info = clinvar_data['result'][clinvar_id]
                result.update({
                    'clinvar_id': clinvar_id,
                    'clinical_significance': variant_info.get('clinical_significance', {}).get('description', ''),
                    'condition': variant_info.get('trait_set', [{}])[0].get('trait_name', '') if variant_info.get('trait_set') else '',
                    'review_status': variant_info.get('clinical_significance', {}).get('review_status', ''),
                    'gene_symbol': variant_info.get('genes', [{}])[0].get('symbol', '') if variant_info.get('genes') else ''
                })

            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"NCBI variant search error: {e}")
            return {'variant_id': variant_id, 'error': str(e)}

    def blast_sequence(self, sequence: str, database: str = "nr") -> Optional[Dict[str, Any]]:
        """
        Run BLAST search (use sparingly - slower and heavier)
        Returns only top 5 hits to minimize data/cost
        """
        cache_key = self._get_cache_key(sequence[:50], f"blast_{database}")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        self._rate_limit()
        self.call_count += 1

        try:
            # Submit BLAST job
            params = {
                'CMD': 'Put',
                'PROGRAM': 'blastn',
                'DATABASE': database,
                'QUERY': sequence,
                'FORMAT_TYPE': 'XML'
            }
            response = requests.post(self.BLAST_URL, data=params, timeout=30)
            response.raise_for_status()

            # Extract RID (request ID)
            rid = None
            for line in response.text.split('\n'):
                if 'RID' in line:
                    rid = line.split('=')[1].strip()
                    break

            if not rid:
                return {'error': 'Failed to submit BLAST job'}

            # Poll for results (with timeout)
            max_wait = 60  # seconds
            waited = 0
            while waited < max_wait:
                time.sleep(5)
                waited += 5

                check_params = {
                    'CMD': 'Get',
                    'RID': rid,
                    'FORMAT_TYPE': 'XML'
                }
                result_response = requests.get(self.BLAST_URL, params=check_params, timeout=10)

                if 'Status=READY' in result_response.text or '<BlastOutput>' in result_response.text:
                    # Parse XML results (top 5 only)
                    try:
                        root = ET.fromstring(result_response.text)
                        hits = []
                        for hit in root.findall('.//Hit')[:5]:  # Top 5 only
                            hits.append({
                                'id': hit.findtext('Hit_id'),
                                'definition': hit.findtext('Hit_def'),
                                'accession': hit.findtext('Hit_accession'),
                                'evalue': hit.findtext('.//Hsp_evalue'),
                                'identity': hit.findtext('.//Hsp_identity')
                            })

                        result = {'rid': rid, 'hits': hits}
                        self._set_cached(cache_key, result)
                        return result
                    except Exception as e:
                        logger.error(f"BLAST XML parse error: {e}")
                        return {'error': 'Failed to parse BLAST results'}

            return {'error': 'BLAST timeout'}

        except Exception as e:
            logger.error(f"BLAST error: {e}")
            return {'error': str(e)}


class PubMedClient(CachedAPIClient):
    """
    Client for PubMed literature mining
    Free API, rate-limited
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, cache_dir: str, email: str = "research@example.com"):
        super().__init__(cache_dir)
        self.email = email
        self.rate_limit_delay = 0.34
        self.last_request = time.time()

    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request = time.time()

    def search_gene_literature(self, gene_symbol: str, keywords: List[str] = None,
                               max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search PubMed for papers related to gene and keywords
        Returns limited results to minimize parsing
        """
        if keywords is None:
            keywords = ['autism', 'ADHD', 'neurodevelopmental']

        query = f"{gene_symbol}[Gene] AND ({' OR '.join(keywords)})"
        cache_key = self._get_cache_key(query, "pubmed_search")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        self._rate_limit()
        self.call_count += 1

        try:
            # Search for PMIDs
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance',
                'email': self.email
            }
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            search_result = response.json()

            pmids = search_result.get('esearchresult', {}).get('idlist', [])
            if not pmids:
                self._set_cached(cache_key, [])
                return []

            # Fetch paper details
            self._rate_limit()
            fetch_url = f"{self.BASE_URL}/esummary.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'json',
                'email': self.email
            }
            response = requests.get(fetch_url, params=params, timeout=10)
            response.raise_for_status()
            papers_data = response.json()

            papers = []
            for pmid in pmids:
                if pmid in papers_data.get('result', {}):
                    paper = papers_data['result'][pmid]
                    papers.append({
                        'pmid': pmid,
                        'title': paper.get('title', ''),
                        'authors': [author.get('name', '') for author in paper.get('authors', [])[:3]],  # First 3 authors
                        'journal': paper.get('source', ''),
                        'pub_date': paper.get('pubdate', ''),
                        'doi': paper.get('elocationid', '').replace('doi: ', '') if 'doi' in paper.get('elocationid', '') else None
                    })

            self._set_cached(cache_key, papers)
            return papers

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []


class LLMSynthesizer:
    """
    Lightweight LLM integration for synthesis
    Cost-optimized: ~$0.001 per gene with caching
    Supports Claude Haiku (cheapest) or other cheap models
    """

    def __init__(self, cache_dir: str, provider: str = "anthropic", model: str = "claude-3-5-haiku-20241022"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.model = model
        self.api_key = os.getenv(f"{provider.upper()}_API_KEY")
        self.call_count = 0
        self.total_tokens = 0
        self.estimated_cost = 0.0

        # Cost per 1M tokens (as of 2024)
        self.costs = {
            "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},  # $1/$5 per MTok
            "gpt-4o-mini": {"input": 0.15, "output": 0.60}  # $0.15/$0.60 per MTok
        }

    def _get_cache_key(self, data_summary: str) -> str:
        """Generate cache key from data summary"""
        return hashlib.md5(data_summary.encode()).hexdigest()

    def synthesize_findings(self, gene_data: Dict[str, Any],
                           literature: List[Dict[str, Any]],
                           functional_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate LLM synthesis of genetic findings
        Designed to be <500 tokens output per call
        """
        if not self.api_key:
            logger.warning(f"No API key found for {self.provider}, skipping LLM synthesis")
            return None

        # Create summary for caching
        data_summary = json.dumps({
            'gene': gene_data.get('symbol', ''),
            'variant': functional_data.get('variant_id'),
            'papers': len(literature)
        }, sort_keys=True)

        cache_key = self._get_cache_key(data_summary)
        cache_file = self.cache_dir / f"{cache_key}.txt"

        # Check cache
        if cache_file.exists():
            logger.debug(f"Cache hit for synthesis: {gene_data.get('symbol', 'unknown')}")
            return cache_file.read_text()

        # Prepare prompt (keep it short to minimize input tokens)
        prompt = self._create_synthesis_prompt(gene_data, literature, functional_data)

        try:
            if self.provider == "anthropic":
                synthesis = self._call_anthropic(prompt)
            elif self.provider == "openai":
                synthesis = self._call_openai(prompt)
            else:
                logger.error(f"Unsupported provider: {self.provider}")
                return None

            # Cache result
            cache_file.write_text(synthesis)

            return synthesis

        except Exception as e:
            logger.error(f"LLM synthesis error: {e}")
            return None

    def _create_synthesis_prompt(self, gene_data: Dict, literature: List[Dict],
                                 functional_data: Dict) -> str:
        """Create concise prompt for synthesis"""
        # Limit literature to top 5 papers
        lit_summary = "\n".join([
            f"- {p['title']} ({p['pub_date']})"
            for p in literature[:5]
        ])

        return f"""Synthesize genetic findings for {gene_data.get('symbol', 'unknown gene')} in 2-3 concise paragraphs:

Gene: {gene_data.get('symbol', '')} - {gene_data.get('description', '')}
Function: {gene_data.get('summary', '')[:200]}...
Variant: {functional_data.get('variant_id', 'N/A')} - {functional_data.get('clinical_significance', 'N/A')}

Key Literature:
{lit_summary}

Synthesize:
1. Gene function and biological role
2. Relevance to autism/ADHD/neurodevelopmental conditions
3. Causal implications and mechanisms
4. Clinical significance

Keep response under 300 words."""

    def _call_anthropic(self, prompt: str) -> str:
        """Call Claude API"""
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        response = client.messages.create(
            model=self.model,
            max_tokens=500,  # Limit output tokens
            messages=[{"role": "user", "content": prompt}]
        )

        # Track usage
        self.call_count += 1
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        self.total_tokens += input_tokens + output_tokens

        # Calculate cost
        cost = (
            input_tokens / 1_000_000 * self.costs[self.model]["input"] +
            output_tokens / 1_000_000 * self.costs[self.model]["output"]
        )
        self.estimated_cost += cost

        logger.info(f"LLM call: {input_tokens} in + {output_tokens} out = ${cost:.6f}")

        return response.content[0].text

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        import openai

        client = openai.OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        # Track usage
        self.call_count += 1
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        self.total_tokens += input_tokens + output_tokens

        cost = (
            input_tokens / 1_000_000 * self.costs[self.model]["input"] +
            output_tokens / 1_000_000 * self.costs[self.model]["output"]
        )
        self.estimated_cost += cost

        logger.info(f"LLM call: {input_tokens} in + {output_tokens} out = ${cost:.6f}")

        return response.choices[0].message.content

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage and cost statistics"""
        return {
            'calls': self.call_count,
            'total_tokens': self.total_tokens,
            'estimated_cost_usd': round(self.estimated_cost, 6)
        }


class GeneticAnalysisSystem:
    """
    Main system orchestrating genetic lookups and analysis
    Designed for cost-effectiveness and caching
    """

    def __init__(self, data_dir: str = "data/genetic_analysis",
                 use_llm: bool = False, llm_provider: str = "anthropic"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize clients with caching
        self.ncbi = NCBIClient(str(self.data_dir / "cache" / "ncbi"))
        self.pubmed = PubMedClient(str(self.data_dir / "cache" / "pubmed"))

        # Optional LLM
        self.use_llm = use_llm
        self.llm = None
        if use_llm:
            self.llm = LLMSynthesizer(
                str(self.data_dir / "cache" / "llm"),
                provider=llm_provider
            )

    def analyze_gene(self, gene_symbol: str,
                    include_blast: bool = False,
                    keywords: List[str] = None) -> GeneticLookupResult:
        """
        Complete genetic analysis for a gene

        Args:
            gene_symbol: Gene symbol (e.g., 'SHANK3')
            include_blast: Whether to run BLAST (slower, use sparingly)
            keywords: Keywords for literature search

        Returns:
            GeneticLookupResult with all findings
        """
        logger.info(f"Analyzing gene: {gene_symbol}")

        # 1. Gene information from NCBI
        gene_data = self.ncbi.search_gene(gene_symbol)

        # 2. Literature search
        literature = self.pubmed.search_gene_literature(gene_symbol, keywords=keywords)

        # 3. BLAST (optional, rarely needed)
        blast_results = None
        if include_blast and 'error' not in gene_data:
            # Would need gene sequence - skip for now
            pass

        # 4. Functional annotations (from gene data)
        functional_annotations = {
            'description': gene_data.get('description', ''),
            'summary': gene_data.get('summary', ''),
            'chromosome': gene_data.get('chromosome', ''),
            'gene_type': gene_data.get('gene_type', '')
        }

        # 5. Disease associations (from literature)
        disease_associations = [
            {'source': 'literature', 'condition': 'autism/ADHD', 'papers': len(literature)}
        ]

        # 6. Pathways (placeholder - could integrate with KEGG/Reactome)
        pathways = []

        # 7. Causal connections (extracted from literature titles)
        causal_connections = self._extract_causal_connections(literature)

        # 8. LLM synthesis (optional)
        llm_synthesis = None
        if self.use_llm and self.llm and literature:
            llm_synthesis = self.llm.synthesize_findings(
                gene_data, literature, functional_annotations
            )

        result = GeneticLookupResult(
            gene_symbol=gene_symbol,
            variant_id=None,
            functional_annotations=functional_annotations,
            disease_associations=disease_associations,
            literature_refs=literature,
            pathways=pathways,
            blast_results=blast_results,
            causal_connections=causal_connections,
            llm_synthesis=llm_synthesis,
            timestamp=datetime.now().isoformat(),
            cache_hit=self.ncbi.cache_hits > 0
        )

        # Save result
        self._save_result(result)

        return result

    def analyze_variant(self, variant_id: str,
                       keywords: List[str] = None) -> GeneticLookupResult:
        """
        Complete analysis for a genetic variant (e.g., rs ID)

        Args:
            variant_id: Variant identifier (e.g., 'rs6265')
            keywords: Keywords for literature search

        Returns:
            GeneticLookupResult with all findings
        """
        logger.info(f"Analyzing variant: {variant_id}")

        # 1. Variant information from ClinVar/dbSNP
        variant_data = self.ncbi.search_variant(variant_id)

        # 2. Gene information
        gene_symbol = variant_data.get('gene_symbol', '')
        gene_data = {}
        if gene_symbol:
            gene_data = self.ncbi.search_gene(gene_symbol)

        # 3. Literature search
        search_term = f"{variant_id} OR {gene_symbol}"
        literature = self.pubmed.search_gene_literature(search_term, keywords=keywords)

        # 4. Functional annotations
        functional_annotations = {
            'variant_id': variant_id,
            'clinical_significance': variant_data.get('clinical_significance', ''),
            'condition': variant_data.get('condition', ''),
            'review_status': variant_data.get('review_status', ''),
            'gene': gene_symbol
        }

        # 5. Disease associations
        disease_associations = []
        if variant_data.get('condition'):
            disease_associations.append({
                'source': 'clinvar',
                'condition': variant_data['condition'],
                'significance': variant_data.get('clinical_significance', '')
            })

        # 6. Causal connections
        causal_connections = self._extract_causal_connections(literature)

        # 7. LLM synthesis (optional)
        llm_synthesis = None
        if self.use_llm and self.llm and literature:
            llm_synthesis = self.llm.synthesize_findings(
                gene_data, literature, functional_annotations
            )

        result = GeneticLookupResult(
            gene_symbol=gene_symbol,
            variant_id=variant_id,
            functional_annotations=functional_annotations,
            disease_associations=disease_associations,
            literature_refs=literature,
            pathways=[],
            blast_results=None,
            causal_connections=causal_connections,
            llm_synthesis=llm_synthesis,
            timestamp=datetime.now().isoformat(),
            cache_hit=self.ncbi.cache_hits > 0
        )

        self._save_result(result)

        return result

    def _extract_causal_connections(self, literature: List[Dict]) -> List[Dict[str, Any]]:
        """Extract causal connections from paper titles"""
        causal_keywords = ['cause', 'role', 'mechanism', 'pathway', 'regulates',
                          'mediates', 'contributes', 'associated with']

        connections = []
        for paper in literature:
            title = paper.get('title', '').lower()
            for keyword in causal_keywords:
                if keyword in title:
                    connections.append({
                        'pmid': paper['pmid'],
                        'title': paper['title'],
                        'causal_keyword': keyword,
                        'strength': 'suggested'  # Would need full text for stronger claims
                    })
                    break

        return connections

    def _save_result(self, result: GeneticLookupResult):
        """Save analysis result to file"""
        output_dir = self.data_dir / "results"
        output_dir.mkdir(exist_ok=True)

        filename = f"{result.gene_symbol}_{result.variant_id or 'gene'}_{datetime.now().strftime('%Y%m%d')}.json"
        output_file = output_dir / filename

        with open(output_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        logger.info(f"Saved result to {output_file}")

    def batch_analyze_genes(self, gene_list: List[str],
                           use_llm: bool = None) -> List[GeneticLookupResult]:
        """
        Batch analyze multiple genes with rate limiting

        Args:
            gene_list: List of gene symbols
            use_llm: Override instance LLM setting for this batch

        Returns:
            List of GeneticLookupResult
        """
        if use_llm is not None:
            original_llm_setting = self.use_llm
            self.use_llm = use_llm

        results = []
        for i, gene in enumerate(gene_list):
            logger.info(f"Processing gene {i+1}/{len(gene_list)}: {gene}")
            try:
                result = self.analyze_gene(gene)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {gene}: {e}")

            # Rate limiting between genes
            if i < len(gene_list) - 1:
                time.sleep(1)

        if use_llm is not None:
            self.use_llm = original_llm_setting

        return results

    def get_cost_report(self) -> Dict[str, Any]:
        """Generate cost and usage report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'ncbi_stats': self.ncbi.get_stats(),
            'pubmed_stats': self.pubmed.get_stats()
        }

        if self.llm:
            report['llm_stats'] = self.llm.get_usage_stats()

        # Calculate estimated monthly cost
        if self.llm:
            daily_cost = report['llm_stats']['estimated_cost_usd']
            report['estimated_monthly_cost'] = round(daily_cost * 30, 2)
        else:
            report['estimated_monthly_cost'] = 0.0

        return report


# Convenience functions
def quick_gene_lookup(gene_symbol: str, use_llm: bool = False) -> Dict[str, Any]:
    """Quick gene lookup with optional LLM synthesis"""
    system = GeneticAnalysisSystem(use_llm=use_llm)
    result = system.analyze_gene(gene_symbol)
    return asdict(result)


def quick_variant_lookup(variant_id: str, use_llm: bool = False) -> Dict[str, Any]:
    """Quick variant lookup with optional LLM synthesis"""
    system = GeneticAnalysisSystem(use_llm=use_llm)
    result = system.analyze_variant(variant_id)
    return asdict(result)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test gene lookup
    print("Testing gene lookup...")
    system = GeneticAnalysisSystem(use_llm=False)  # Set to True if you have API key

    result = system.analyze_gene("SHANK3", keywords=['autism', 'neurodevelopmental'])

    print(f"\nGene: {result.gene_symbol}")
    print(f"Description: {result.functional_annotations.get('description', '')}")
    print(f"Literature refs: {len(result.literature_refs)}")
    print(f"Causal connections: {len(result.causal_connections)}")

    if result.causal_connections:
        print("\nTop causal connections:")
        for conn in result.causal_connections[:3]:
            print(f"  - {conn['title'][:80]}...")

    # Cost report
    print("\nCost report:")
    report = system.get_cost_report()
    print(json.dumps(report, indent=2))
