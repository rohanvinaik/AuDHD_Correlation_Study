#!/usr/bin/env python3
"""
Convenient script for genetic analysis

Usage:
    # Single gene lookup
    python scripts/analyze_genes.py SHANK3

    # Multiple genes
    python scripts/analyze_genes.py SHANK3 NRXN1 CNTNAP2

    # With LLM synthesis (requires API key)
    python scripts/analyze_genes.py SHANK3 --llm

    # Variant lookup
    python scripts/analyze_genes.py --variant rs6265

    # Batch from file
    python scripts/analyze_genes.py --batch data/gene_list.txt

    # Cost report
    python scripts/analyze_genes.py --cost-report
"""

import sys
import argparse
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audhd_correlation.analysis import GeneticAnalysisSystem, quick_gene_lookup, quick_variant_lookup


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def analyze_genes_cli(args):
    """Run gene analysis from CLI"""
    system = GeneticAnalysisSystem(use_llm=args.llm, llm_provider=args.llm_provider)

    if args.variant:
        # Variant lookup
        print(f"\nAnalyzing variant: {args.variant}")
        result = system.analyze_variant(args.variant, keywords=args.keywords)
        print_result(result)

    elif args.batch:
        # Batch analysis from file
        gene_file = Path(args.batch)
        if not gene_file.exists():
            print(f"Error: File not found: {gene_file}")
            return

        genes = [line.strip() for line in gene_file.read_text().split('\n') if line.strip()]
        print(f"\nBatch analyzing {len(genes)} genes from {gene_file}")

        results = system.batch_analyze_genes(genes, use_llm=args.llm)

        print(f"\nCompleted {len(results)}/{len(genes)} analyses")
        print("\nCost Report:")
        print(json.dumps(system.get_cost_report(), indent=2))

    elif args.cost_report:
        # Just show cost report
        system = GeneticAnalysisSystem()
        report = system.get_cost_report()
        print("\nCost Report:")
        print(json.dumps(report, indent=2))

    elif args.genes:
        # Single or multiple gene analysis
        for gene in args.genes:
            print(f"\n{'='*80}")
            print(f"Analyzing gene: {gene}")
            print('='*80)

            result = system.analyze_gene(gene, keywords=args.keywords)
            print_result(result)

        # Cost report
        if args.llm:
            print("\n" + "="*80)
            print("Cost Report:")
            print("="*80)
            print(json.dumps(system.get_cost_report(), indent=2))

    else:
        print("Error: Specify genes, --variant, --batch, or --cost-report")
        sys.exit(1)


def print_result(result):
    """Pretty print result"""
    print(f"\n{'Gene':<20} {result.gene_symbol}")
    if result.variant_id:
        print(f"{'Variant':<20} {result.variant_id}")

    print(f"\n{'-'*80}")
    print("FUNCTIONAL ANNOTATIONS")
    print('-'*80)
    for key, value in result.functional_annotations.items():
        if value:
            print(f"{key:<20} {str(value)[:100]}...")

    print(f"\n{'-'*80}")
    print(f"LITERATURE ({len(result.literature_refs)} papers)")
    print('-'*80)
    for i, paper in enumerate(result.literature_refs[:5], 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   {', '.join(paper.get('authors', [])[:3])} et al.")
        print(f"   {paper.get('journal', '')} ({paper.get('pub_date', '')})")
        print(f"   PMID: {paper['pmid']}")

    if len(result.literature_refs) > 5:
        print(f"\n   ... and {len(result.literature_refs) - 5} more papers")

    print(f"\n{'-'*80}")
    print(f"DISEASE ASSOCIATIONS ({len(result.disease_associations)})")
    print('-'*80)
    for assoc in result.disease_associations:
        print(f"  - {assoc.get('condition', 'N/A')} ({assoc.get('source', 'N/A')})")
        if 'significance' in assoc:
            print(f"    Significance: {assoc['significance']}")

    print(f"\n{'-'*80}")
    print(f"CAUSAL CONNECTIONS ({len(result.causal_connections)})")
    print('-'*80)
    for conn in result.causal_connections[:5]:
        print(f"  - [{conn['causal_keyword']}] {conn['title'][:70]}...")
        print(f"    PMID: {conn['pmid']}")

    if len(result.causal_connections) > 5:
        print(f"\n  ... and {len(result.causal_connections) - 5} more connections")

    if result.llm_synthesis:
        print(f"\n{'-'*80}")
        print("LLM SYNTHESIS")
        print('-'*80)
        print(result.llm_synthesis)

    print(f"\n{'-'*80}")
    print(f"Analysis cached: {result.cache_hit}")
    print(f"Timestamp: {result.timestamp}")
    print(f"Results saved to: data/genetic_analysis/results/")
    print('-'*80)


def main():
    parser = argparse.ArgumentParser(
        description="Genetic analysis with BLAST, literature mining, and optional LLM synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single gene
  python scripts/analyze_genes.py SHANK3

  # Multiple genes with LLM synthesis
  python scripts/analyze_genes.py SHANK3 NRXN1 CNTNAP2 --llm

  # Variant lookup
  python scripts/analyze_genes.py --variant rs6265

  # Batch analysis
  python scripts/analyze_genes.py --batch data/candidate_genes.txt --llm

  # Check costs
  python scripts/analyze_genes.py --cost-report

Environment Variables:
  ANTHROPIC_API_KEY    API key for Claude (Haiku recommended)
  OPENAI_API_KEY       API key for OpenAI (gpt-4o-mini recommended)
        """
    )

    parser.add_argument('genes', nargs='*', help='Gene symbols to analyze')
    parser.add_argument('--variant', help='Variant ID to analyze (e.g., rs6265)')
    parser.add_argument('--batch', help='Batch analyze genes from file (one per line)')
    parser.add_argument('--llm', action='store_true', help='Enable LLM synthesis')
    parser.add_argument('--llm-provider', default='anthropic', choices=['anthropic', 'openai'],
                       help='LLM provider (default: anthropic)')
    parser.add_argument('--keywords', nargs='+', default=['autism', 'ADHD', 'neurodevelopmental'],
                       help='Keywords for literature search')
    parser.add_argument('--cost-report', action='store_true', help='Show cost report only')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    setup_logging(args.verbose)

    analyze_genes_cli(args)


if __name__ == "__main__":
    main()
