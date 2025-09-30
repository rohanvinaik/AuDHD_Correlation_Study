#!/usr/bin/env python3
"""
Dataset Mention Finder

Uses natural language processing to identify dataset mentions in abstracts
and full text, including:
- Sample size statements ("N = 142 participants")
- Cohort descriptions ("UK Biobank", "ABCD Study")
- Unpublished dataset mentions
- Data collection descriptions
- Collaboration opportunities

Requirements:
    pip install requests pandas spacy
    python -m spacy download en_core_web_sm

Usage:
    # Find datasets in papers JSON
    python dataset_mention_finder.py \\
        --input data/literature/papers_with_data.json \\
        --output data/literature/

    # Extract sample sizes
    python dataset_mention_finder.py \\
        --input data/literature/papers_with_data.json \\
        --extract-sample-sizes \\
        --output data/literature/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging

try:
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install pandas")
    sys.exit(1)

# Try to import spaCy (optional)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Known cohort/dataset patterns
KNOWN_COHORTS = [
    'UK Biobank', 'ABCD', 'Adolescent Brain Cognitive Development',
    'All of Us', 'Framingham Heart Study',
    'Generation R', 'Norwegian Mother and Child Cohort',
    'ALSPAC', 'Avon Longitudinal Study',
    'IMAGEN', 'PGC', 'Psychiatric Genomics Consortium',
    'SPARK', 'Simons Foundation Autism Research Initiative',
    'SSC', 'Simons Simplex Collection',
    'AGRE', 'Autism Genetic Resource Exchange',
    'iPSYCH', 'American Gut Project',
    'Human Microbiome Project', 'MetaHIT',
    '1000 Genomes', 'HapMap', 'gnomAD',
    'GTEx', 'ENCODE', 'Roadmap Epigenomics',
    'TOPMed', 'PAGE Study', 'Million Veteran Program'
]

# Sample size patterns
SAMPLE_SIZE_PATTERNS = [
    r'(?:N|n)\s*=\s*(\d+(?:,\d{3})*)',
    r'sample\s+(?:of|size|included?)\s+(\d+(?:,\d{3})*)',
    r'(\d+(?:,\d{3})*)\s+(?:participants?|subjects?|individuals?|patients?|cases?|controls?)',
    r'cohort\s+(?:of|including|with)\s+(\d+(?:,\d{3})*)',
    r'total\s+(?:of\s+)?(\d+(?:,\d{3})*)\s+(?:participants?|subjects?)',
    r'(\d+(?:,\d{3})*)\s+(?:children|adults|males?|females?)'
]

# Dataset mention patterns
DATASET_MENTION_PATTERNS = [
    r'(?:we|our)\s+(?:collected|recruited|enrolled|obtained)\s+(?:data|samples?|participants?)',
    r'(?:newly|previously)\s+(?:collected|recruited|generated)\s+(?:data|cohort|samples?)',
    r'data\s+(?:were|was)\s+(?:collected|obtained|acquired)\s+(?:from|at|between)',
    r'participants?\s+(?:were|was)\s+recruited\s+(?:from|at|between)',
    r'retrospective\s+(?:analysis|study|review)\s+of',
    r'prospective\s+(?:cohort|study)\s+of',
    r'longitudinal\s+(?:study|cohort|data)\s+(?:of|from|including)',
    r'cross-sectional\s+(?:study|analysis|data)\s+(?:of|from)'
]

# Unpublished data indicators
UNPUBLISHED_DATA_PATTERNS = [
    r'unpublished\s+data',
    r'data\s+not\s+(?:yet\s+)?published',
    r'manuscript\s+in\s+preparation',
    r'data\s+available\s+(?:upon|on)\s+request',
    r'(?:please\s+)?contact\s+(?:the\s+)?(?:corresponding\s+)?author',
    r'collaboration\s+(?:is\s+)?welcome',
    r'interested\s+(?:in\s+)?collaboration'
]

# Data type mentions
DATA_TYPE_PATTERNS = {
    'genomics': r'(?:genomic|genetic|genotyping|sequencing|WGS|WES|GWAS)\s+data',
    'transcriptomics': r'(?:transcriptomic|RNA-seq|gene expression|microarray)\s+data',
    'metabolomics': r'(?:metabolomic|metabolite|NMR|LC-MS|GC-MS)\s+(?:data|profiling)',
    'proteomics': r'(?:proteomic|protein)\s+(?:data|profiling)',
    'imaging': r'(?:MRI|fMRI|DTI|PET|CT|neuroimaging)\s+data',
    'microbiome': r'(?:microbiome|microbiota|16S|metagenomic)\s+data',
    'clinical': r'(?:clinical|phenotypic|behavioral)\s+(?:data|assessment)',
    'longitudinal': r'longitudinal\s+(?:data|measurements|assessments)',
    'multiomics': r'(?:multi-omics|multiomics|integrative)\s+(?:data|analysis)'
}

# Contact/collaboration patterns
CONTACT_PATTERNS = [
    r'(?:corresponding\s+author|contact|email):\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
    r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
    r'for\s+(?:further\s+)?information[,\s]+(?:please\s+)?contact\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
]


class DatasetMentionFinder:
    """Find dataset mentions in scientific papers"""

    def __init__(self, output_dir: Path):
        """
        Initialize finder

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load spaCy if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model")
            except:
                logger.warning("Could not load spaCy model. Install with: python -m spacy download en_core_web_sm")

        logger.info(f"Initialized dataset mention finder: {output_dir}")

    def find_sample_sizes(self, text: str) -> List[Dict]:
        """
        Extract sample size mentions

        Args:
            text: Input text

        Returns:
            List of sample size mentions
        """
        mentions = []

        for pattern in SAMPLE_SIZE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                sample_size_str = match.group(1).replace(',', '')
                try:
                    sample_size = int(sample_size_str)

                    # Get context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]

                    mentions.append({
                        'sample_size': sample_size,
                        'matched_text': match.group(0),
                        'context': context
                    })
                except ValueError:
                    continue

        # Remove duplicates and sort by sample size
        unique_mentions = {}
        for mention in mentions:
            size = mention['sample_size']
            if size not in unique_mentions or len(mention['context']) > len(unique_mentions[size]['context']):
                unique_mentions[size] = mention

        return sorted(unique_mentions.values(), key=lambda x: x['sample_size'], reverse=True)

    def find_known_cohorts(self, text: str) -> List[str]:
        """Find mentions of known cohorts/datasets"""
        found_cohorts = []

        for cohort in KNOWN_COHORTS:
            if re.search(rf'\b{re.escape(cohort)}\b', text, re.IGNORECASE):
                found_cohorts.append(cohort)

        return found_cohorts

    def find_dataset_mentions(self, text: str) -> List[Dict]:
        """Find general dataset collection mentions"""
        mentions = []

        for pattern in DATASET_MENTION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                # Get context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]

                mentions.append({
                    'matched_text': match.group(0),
                    'context': context,
                    'pattern': pattern
                })

        return mentions

    def find_unpublished_data(self, text: str) -> List[str]:
        """Find mentions of unpublished data"""
        mentions = []

        for pattern in UNPUBLISHED_DATA_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                mentions.append(pattern.replace(r'\s+', ' ').replace(r'\?', ''))

        return mentions

    def find_data_types(self, text: str) -> Dict[str, bool]:
        """Identify data types mentioned"""
        data_types = {}

        for data_type, pattern in DATA_TYPE_PATTERNS.items():
            data_types[data_type] = bool(re.search(pattern, text, re.IGNORECASE))

        return data_types

    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses"""
        emails = []

        for pattern in CONTACT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if '@' in match:
                    emails.append(match)

        return list(set(emails))

    def extract_entities_with_spacy(self, text: str) -> Dict:
        """Extract named entities using spaCy"""
        if not self.nlp:
            return {}

        doc = self.nlp(text[:1000000])  # Limit text length

        entities = {
            'organizations': [],
            'persons': [],
            'locations': [],
            'dates': []
        }

        for ent in doc.ents:
            if ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def analyze_paper(self, paper: Dict) -> Dict:
        """
        Analyze single paper for dataset mentions

        Args:
            paper: Paper dictionary from pubmed_miner

        Returns:
            Analysis results
        """
        # Combine title and abstract
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

        result = {
            'pmid': paper.get('pmid'),
            'pmcid': paper.get('pmcid'),
            'title': paper.get('title', ''),
            'sample_sizes': self.find_sample_sizes(text),
            'known_cohorts': self.find_known_cohorts(text),
            'dataset_mentions': self.find_dataset_mentions(text),
            'unpublished_data_indicators': self.find_unpublished_data(text),
            'data_types': self.find_data_types(text),
            'contact_emails': self.extract_emails(text),
            'has_data_collection': bool(self.find_dataset_mentions(text)),
            'collaboration_opportunity': bool(self.find_unpublished_data(text))
        }

        # Add spaCy entities if available
        if self.nlp:
            result['entities'] = self.extract_entities_with_spacy(text)

        return result

    def process_papers(self, papers: List[Dict]) -> List[Dict]:
        """Process multiple papers"""
        logger.info(f"Analyzing {len(papers)} papers for dataset mentions...")

        results = []

        for i, paper in enumerate(papers):
            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(papers)} papers...")

            result = self.analyze_paper(paper)
            results.append(result)

        logger.info(f"Completed analysis of {len(results)} papers")
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Find dataset mentions in scientific papers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze papers from JSON
  python dataset_mention_finder.py \\
      --input data/literature/papers_with_data.json \\
      --output data/literature/

  # Extract sample sizes only
  python dataset_mention_finder.py \\
      --input data/literature/papers_with_data.json \\
      --extract-sample-sizes \\
      --output data/literature/

  # Find collaboration opportunities
  python dataset_mention_finder.py \\
      --input data/literature/papers_with_data.json \\
      --find-collaborations \\
      --output data/literature/
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input papers JSON file'
    )

    parser.add_argument(
        '--extract-sample-sizes',
        action='store_true',
        help='Focus on sample size extraction'
    )

    parser.add_argument(
        '--find-collaborations',
        action='store_true',
        help='Find collaboration opportunities'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/literature',
        help='Output directory'
    )

    args = parser.parse_args()

    # Load papers
    with open(args.input) as f:
        papers = json.load(f)

    logger.info(f"Loaded {len(papers)} papers")

    # Initialize finder
    finder = DatasetMentionFinder(Path(args.output))

    # Process papers
    results = finder.process_papers(papers)

    # Export results
    output_file = finder.output_dir / 'dataset_mentions.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate summary
    print(f"\n=== Dataset Mention Analysis Summary ===\n")
    print(f"Total papers analyzed: {len(results)}")

    # Sample sizes
    papers_with_sample_sizes = [r for r in results if r['sample_sizes']]
    print(f"Papers with sample sizes: {len(papers_with_sample_sizes)}")

    if papers_with_sample_sizes:
        all_sizes = [s['sample_size'] for r in papers_with_sample_sizes for s in r['sample_sizes']]
        print(f"  Largest sample size: {max(all_sizes):,}")
        print(f"  Median sample size: {sorted(all_sizes)[len(all_sizes)//2]:,}")

    # Known cohorts
    papers_with_cohorts = [r for r in results if r['known_cohorts']]
    print(f"\nPapers mentioning known cohorts: {len(papers_with_cohorts)}")

    cohort_counts = {}
    for result in results:
        for cohort in result['known_cohorts']:
            cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1

    if cohort_counts:
        print(f"Top cohorts mentioned:")
        for cohort, count in sorted(cohort_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cohort}: {count}")

    # Data types
    data_type_counts = {dt: 0 for dt in DATA_TYPE_PATTERNS.keys()}
    for result in results:
        for dt, present in result['data_types'].items():
            if present:
                data_type_counts[dt] += 1

    print(f"\nData types mentioned:")
    for dt, count in sorted(data_type_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {dt}: {count}")

    # Collaboration opportunities
    collab_papers = [r for r in results if r['collaboration_opportunity']]
    print(f"\nPapers with collaboration opportunities: {len(collab_papers)}")

    # Contact information
    papers_with_emails = [r for r in results if r['contact_emails']]
    print(f"Papers with contact emails: {len(papers_with_emails)}")

    print(f"\nResults saved: {output_file}")

    # Generate CSV for easy review
    if args.extract_sample_sizes:
        sample_size_data = []
        for result in results:
            for sample in result['sample_sizes']:
                sample_size_data.append({
                    'pmid': result['pmid'],
                    'title': result['title'][:100],
                    'sample_size': sample['sample_size'],
                    'context': sample['context']
                })

        if sample_size_data:
            df = pd.DataFrame(sample_size_data)
            csv_file = finder.output_dir / 'sample_sizes.csv'
            df.to_csv(csv_file, index=False)
            print(f"Sample sizes exported: {csv_file}")

    if args.find_collaborations:
        collab_data = []
        for result in collab_papers:
            collab_data.append({
                'pmid': result['pmid'],
                'pmcid': result['pmcid'],
                'title': result['title'],
                'emails': ', '.join(result['contact_emails']),
                'unpublished_indicators': ', '.join(result['unpublished_data_indicators'])
            })

        if collab_data:
            df = pd.DataFrame(collab_data)
            csv_file = finder.output_dir / 'collaboration_opportunities.csv'
            df.to_csv(csv_file, index=False)
            print(f"Collaboration opportunities exported: {csv_file}")


if __name__ == '__main__':
    main()