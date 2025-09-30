"""
Pipeline Integration for Genetic Analysis

Automatically looks up significant SNPs/genes identified in analysis pipeline:
- GWAS significant variants
- Differentially expressed genes
- Copy number variations
- Pathway enrichment hits

Generates researcher-friendly reports with:
- Disease associations
- Phenotype correlations
- Literature evidence
- Causal connections
"""

import os
import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import time

from .genetic_lookup import GeneticAnalysisSystem, GeneticLookupResult

logger = logging.getLogger(__name__)


class PipelineGeneticAnalysis:
    """
    Integrates genetic analysis into main pipeline
    Automatically analyzes significant findings
    """

    def __init__(self,
                 output_dir: str = "data/processed/genetic_analysis",
                 use_llm: bool = False,
                 llm_provider: str = "anthropic",
                 significance_threshold: float = 5e-8,
                 max_variants: int = 100):
        """
        Args:
            output_dir: Directory for analysis results
            use_llm: Enable LLM synthesis
            llm_provider: LLM provider (anthropic/openai)
            significance_threshold: P-value threshold for GWAS
            max_variants: Maximum variants to analyze (cost control)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.significance_threshold = significance_threshold
        self.max_variants = max_variants

        # Initialize genetic analysis system
        self.analysis_system = GeneticAnalysisSystem(
            data_dir="data/genetic_analysis",
            use_llm=use_llm,
            llm_provider=llm_provider
        )

        # Report storage
        self.reports = {
            'gwas_variants': [],
            'deg_genes': [],
            'pathway_genes': [],
            'cnv_genes': []
        }

    def analyze_gwas_results(self, gwas_file: str,
                            variant_col: str = 'variant_id',
                            pval_col: str = 'pvalue',
                            gene_col: Optional[str] = 'nearest_gene') -> Dict[str, Any]:
        """
        Analyze significant GWAS hits

        Args:
            gwas_file: Path to GWAS results CSV
            variant_col: Column name for variant IDs
            pval_col: Column name for p-values
            gene_col: Column name for gene symbols (if available)

        Returns:
            Summary of analysis
        """
        logger.info(f"Analyzing GWAS results from {gwas_file}")

        # Load GWAS results
        gwas_df = pd.read_csv(gwas_file)

        # Filter significant variants
        significant = gwas_df[gwas_df[pval_col] < self.significance_threshold].copy()
        significant = significant.sort_values(pval_col).head(self.max_variants)

        logger.info(f"Found {len(significant)} significant variants (p < {self.significance_threshold})")

        if len(significant) == 0:
            logger.warning("No significant variants found")
            return {'variants_analyzed': 0, 'genes_found': 0}

        # Analyze each variant
        results = []
        for idx, row in significant.iterrows():
            variant_id = row[variant_col]
            pvalue = row[pval_col]

            logger.info(f"Analyzing variant {variant_id} (p={pvalue:.2e})")

            try:
                # Lookup variant
                result = self.analysis_system.analyze_variant(
                    variant_id,
                    keywords=['autism', 'ADHD', 'neurodevelopmental', 'psychiatric']
                )

                # Store result with GWAS metadata
                variant_report = {
                    'variant_id': variant_id,
                    'pvalue': pvalue,
                    'gene_symbol': result.gene_symbol,
                    'clinical_significance': result.functional_annotations.get('clinical_significance', 'Unknown'),
                    'condition': result.functional_annotations.get('condition', 'Unknown'),
                    'literature_count': len(result.literature_refs),
                    'causal_connections': len(result.causal_connections),
                    'disease_associations': result.disease_associations,
                    'top_papers': result.literature_refs[:5],  # Top 5 papers
                    'llm_summary': result.llm_synthesis,
                    'timestamp': result.timestamp
                }

                results.append(variant_report)
                self.reports['gwas_variants'].append(variant_report)

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error analyzing variant {variant_id}: {e}")
                continue

        # Generate summary report
        summary = self._generate_gwas_summary(results, significant)

        # Save results
        self._save_gwas_report(results, summary)

        return summary

    def analyze_deg_results(self, deg_file: str,
                           gene_col: str = 'gene_symbol',
                           padj_col: str = 'padj',
                           logfc_col: str = 'log2FoldChange',
                           padj_threshold: float = 0.05,
                           logfc_threshold: float = 1.0) -> Dict[str, Any]:
        """
        Analyze differentially expressed genes

        Args:
            deg_file: Path to DEG results CSV
            gene_col: Column name for gene symbols
            padj_col: Column name for adjusted p-values
            logfc_col: Column name for log fold change
            padj_threshold: Adjusted p-value threshold
            logfc_threshold: Absolute log fold change threshold

        Returns:
            Summary of analysis
        """
        logger.info(f"Analyzing DEG results from {deg_file}")

        # Load DEG results
        deg_df = pd.read_csv(deg_file)

        # Filter significant genes
        significant = deg_df[
            (deg_df[padj_col] < padj_threshold) &
            (abs(deg_df[logfc_col]) > logfc_threshold)
        ].copy()
        significant = significant.sort_values(padj_col).head(self.max_variants)

        logger.info(f"Found {len(significant)} significant DEGs")

        if len(significant) == 0:
            logger.warning("No significant DEGs found")
            return {'genes_analyzed': 0}

        # Analyze each gene
        results = []
        for idx, row in significant.iterrows():
            gene_symbol = row[gene_col]
            padj = row[padj_col]
            logfc = row[logfc_col]
            direction = 'up' if logfc > 0 else 'down'

            logger.info(f"Analyzing gene {gene_symbol} (padj={padj:.2e}, logFC={logfc:.2f})")

            try:
                # Lookup gene
                result = self.analysis_system.analyze_gene(
                    gene_symbol,
                    keywords=['autism', 'ADHD', 'neurodevelopmental', 'brain', 'expression']
                )

                # Store result with DEG metadata
                gene_report = {
                    'gene_symbol': gene_symbol,
                    'padj': padj,
                    'log2_fold_change': logfc,
                    'direction': direction,
                    'description': result.functional_annotations.get('description', 'Unknown'),
                    'chromosome': result.functional_annotations.get('chromosome', 'Unknown'),
                    'literature_count': len(result.literature_refs),
                    'causal_connections': len(result.causal_connections),
                    'disease_associations': result.disease_associations,
                    'top_papers': result.literature_refs[:5],
                    'llm_summary': result.llm_synthesis,
                    'timestamp': result.timestamp
                }

                results.append(gene_report)
                self.reports['deg_genes'].append(gene_report)

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error analyzing gene {gene_symbol}: {e}")
                continue

        # Generate summary report
        summary = self._generate_deg_summary(results, significant)

        # Save results
        self._save_deg_report(results, summary)

        return summary

    def analyze_pathway_genes(self, pathway_file: str,
                             pathway_col: str = 'pathway',
                             genes_col: str = 'genes',
                             pval_col: str = 'pvalue',
                             top_pathways: int = 10) -> Dict[str, Any]:
        """
        Analyze genes from enriched pathways

        Args:
            pathway_file: Path to pathway enrichment results
            pathway_col: Column name for pathway names
            genes_col: Column name for gene lists
            pval_col: Column name for p-values
            top_pathways: Number of top pathways to analyze

        Returns:
            Summary of analysis
        """
        logger.info(f"Analyzing pathway enrichment from {pathway_file}")

        # Load pathway results
        pathway_df = pd.read_csv(pathway_file)
        pathway_df = pathway_df.sort_values(pval_col).head(top_pathways)

        logger.info(f"Analyzing top {len(pathway_df)} pathways")

        # Collect unique genes from top pathways
        all_genes = set()
        for idx, row in pathway_df.iterrows():
            genes = row[genes_col]
            if isinstance(genes, str):
                gene_list = [g.strip() for g in genes.split(',')]
                all_genes.update(gene_list)

        all_genes = list(all_genes)[:self.max_variants]  # Limit total

        logger.info(f"Analyzing {len(all_genes)} unique genes from top pathways")

        # Analyze genes
        results = []
        for gene_symbol in all_genes:
            logger.info(f"Analyzing pathway gene {gene_symbol}")

            try:
                result = self.analysis_system.analyze_gene(
                    gene_symbol,
                    keywords=['autism', 'ADHD', 'neurodevelopmental', 'pathway', 'signaling']
                )

                gene_report = {
                    'gene_symbol': gene_symbol,
                    'description': result.functional_annotations.get('description', 'Unknown'),
                    'literature_count': len(result.literature_refs),
                    'causal_connections': len(result.causal_connections),
                    'disease_associations': result.disease_associations,
                    'top_papers': result.literature_refs[:3],  # Top 3
                    'llm_summary': result.llm_synthesis,
                    'timestamp': result.timestamp
                }

                results.append(gene_report)
                self.reports['pathway_genes'].append(gene_report)

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error analyzing gene {gene_symbol}: {e}")
                continue

        # Generate summary
        summary = self._generate_pathway_summary(results, pathway_df)

        # Save results
        self._save_pathway_report(results, summary)

        return summary

    def _generate_gwas_summary(self, results: List[Dict], significant_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary for GWAS analysis"""
        genes_found = [r for r in results if r['gene_symbol']]
        with_literature = [r for r in results if r['literature_count'] > 0]
        with_clinical = [r for r in results if r['clinical_significance'] != 'Unknown']

        return {
            'total_significant_variants': len(significant_df),
            'variants_analyzed': len(results),
            'genes_identified': len(genes_found),
            'variants_with_literature': len(with_literature),
            'variants_with_clinical_significance': len(with_clinical),
            'avg_papers_per_variant': sum(r['literature_count'] for r in results) / max(1, len(results)),
            'total_causal_connections': sum(r['causal_connections'] for r in results),
            'timestamp': datetime.now().isoformat()
        }

    def _generate_deg_summary(self, results: List[Dict], significant_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary for DEG analysis"""
        with_literature = [r for r in results if r['literature_count'] > 0]
        upregulated = [r for r in results if r['direction'] == 'up']
        downregulated = [r for r in results if r['direction'] == 'down']

        return {
            'total_significant_genes': len(significant_df),
            'genes_analyzed': len(results),
            'genes_with_literature': len(with_literature),
            'upregulated': len(upregulated),
            'downregulated': len(downregulated),
            'avg_papers_per_gene': sum(r['literature_count'] for r in results) / max(1, len(results)),
            'total_causal_connections': sum(r['causal_connections'] for r in results),
            'timestamp': datetime.now().isoformat()
        }

    def _generate_pathway_summary(self, results: List[Dict], pathway_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary for pathway analysis"""
        with_literature = [r for r in results if r['literature_count'] > 0]

        return {
            'pathways_analyzed': len(pathway_df),
            'genes_analyzed': len(results),
            'genes_with_literature': len(with_literature),
            'avg_papers_per_gene': sum(r['literature_count'] for r in results) / max(1, len(results)),
            'total_causal_connections': sum(r['causal_connections'] for r in results),
            'timestamp': datetime.now().isoformat()
        }

    def _save_gwas_report(self, results: List[Dict], summary: Dict):
        """Save GWAS analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed results
        results_file = self.output_dir / f"gwas_genetic_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'results': results
            }, f, indent=2)

        # Save summary table
        summary_df = pd.DataFrame(results)
        summary_csv = self.output_dir / f"gwas_genetic_summary_{timestamp}.csv"
        summary_df.to_csv(summary_csv, index=False)

        # Generate researcher report
        self._generate_researcher_report('gwas', results, summary, timestamp)

        logger.info(f"GWAS analysis saved to {results_file}")

    def _save_deg_report(self, results: List[Dict], summary: Dict):
        """Save DEG analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_file = self.output_dir / f"deg_genetic_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'results': results
            }, f, indent=2)

        summary_df = pd.DataFrame(results)
        summary_csv = self.output_dir / f"deg_genetic_summary_{timestamp}.csv"
        summary_df.to_csv(summary_csv, index=False)

        self._generate_researcher_report('deg', results, summary, timestamp)

        logger.info(f"DEG analysis saved to {results_file}")

    def _save_pathway_report(self, results: List[Dict], summary: Dict):
        """Save pathway analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_file = self.output_dir / f"pathway_genetic_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'results': results
            }, f, indent=2)

        summary_df = pd.DataFrame(results)
        summary_csv = self.output_dir / f"pathway_genetic_summary_{timestamp}.csv"
        summary_df.to_csv(summary_csv, index=False)

        self._generate_researcher_report('pathway', results, summary, timestamp)

        logger.info(f"Pathway analysis saved to {results_file}")

    def _generate_researcher_report(self, analysis_type: str, results: List[Dict],
                                   summary: Dict, timestamp: str):
        """
        Generate human-readable report for researcher review
        """
        report_file = self.output_dir / f"{analysis_type}_researcher_report_{timestamp}.md"

        with open(report_file, 'w') as f:
            f.write(f"# Genetic Analysis Report: {analysis_type.upper()}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary section
            f.write("## Summary\n\n")
            for key, value in summary.items():
                if key != 'timestamp':
                    f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")

            # Key findings section
            f.write("\n## Key Findings\n\n")

            # High-impact variants/genes
            if analysis_type == 'gwas':
                high_impact = [r for r in results
                             if r['clinical_significance'] not in ['Unknown', 'Benign', 'Likely benign']]
                if high_impact:
                    f.write("### High-Impact Variants\n\n")
                    for r in high_impact[:10]:
                        f.write(f"**{r['variant_id']}** → {r['gene_symbol']}\n")
                        f.write(f"- P-value: {r['pvalue']:.2e}\n")
                        f.write(f"- Clinical Significance: {r['clinical_significance']}\n")
                        f.write(f"- Condition: {r['condition']}\n")
                        f.write(f"- Literature: {r['literature_count']} papers\n")
                        if r['llm_summary']:
                            f.write(f"\n*Summary:* {r['llm_summary'][:200]}...\n")
                        f.write("\n")

            # Top genes with literature support
            with_lit = sorted([r for r in results if r['literature_count'] > 0],
                            key=lambda x: x['literature_count'], reverse=True)

            if with_lit:
                f.write("### Top Genes with Literature Support\n\n")
                for r in with_lit[:20]:
                    gene = r['gene_symbol'] if 'gene_symbol' in r else r.get('variant_id', 'Unknown')
                    f.write(f"**{gene}**\n")

                    if 'description' in r:
                        f.write(f"- Description: {r['description'][:100]}...\n")

                    f.write(f"- Literature: {r['literature_count']} papers\n")
                    f.write(f"- Causal connections: {r['causal_connections']}\n")

                    # Disease associations
                    if r.get('disease_associations'):
                        f.write(f"- Disease associations:\n")
                        for assoc in r['disease_associations'][:3]:
                            f.write(f"  - {assoc.get('condition', 'N/A')} ({assoc.get('source', 'N/A')})\n")

                    # Top papers
                    if r.get('top_papers'):
                        f.write(f"- Key papers:\n")
                        for paper in r['top_papers'][:3]:
                            f.write(f"  - {paper['title']} (PMID: {paper['pmid']})\n")

                    if r.get('llm_summary'):
                        f.write(f"\n*Summary:* {r['llm_summary'][:300]}...\n")

                    f.write("\n")

            # Novel/unexpected findings
            f.write("### Novel Findings\n\n")
            novel = [r for r in results if r['literature_count'] == 0]
            if novel:
                f.write(f"Found {len(novel)} genes/variants with no direct literature on autism/ADHD:\n\n")
                for r in novel[:10]:
                    gene = r.get('gene_symbol', r.get('variant_id', 'Unknown'))
                    f.write(f"- **{gene}**")
                    if 'description' in r:
                        f.write(f": {r['description'][:80]}...")
                    f.write("\n")
            else:
                f.write("All identified genes/variants have existing literature support.\n")

            # Cost report
            f.write("\n## Analysis Metadata\n\n")
            cost_report = self.analysis_system.get_cost_report()
            f.write(f"- API calls: {cost_report['ncbi_stats']['api_calls']} (NCBI), "
                   f"{cost_report['pubmed_stats']['api_calls']} (PubMed)\n")
            f.write(f"- Cache hit rate: {cost_report['ncbi_stats']['cache_hit_rate']:.1%}\n")
            if 'llm_stats' in cost_report:
                f.write(f"- LLM calls: {cost_report['llm_stats']['calls']}\n")
                f.write(f"- Estimated cost: ${cost_report['llm_stats']['estimated_cost_usd']:.4f}\n")

            f.write("\n---\n")
            f.write("\n*This report was generated automatically by the AuDHD genetic analysis pipeline.*\n")
            f.write("*Review all findings and verify key associations with primary literature.*\n")

        logger.info(f"Researcher report saved to {report_file}")

    def generate_master_report(self) -> str:
        """
        Generate master report combining all analyses
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"master_genetic_report_{timestamp}.md"

        with open(report_file, 'w') as f:
            f.write("# Master Genetic Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overall statistics
            f.write("## Overall Statistics\n\n")
            total_gwas = len(self.reports['gwas_variants'])
            total_deg = len(self.reports['deg_genes'])
            total_pathway = len(self.reports['pathway_genes'])

            f.write(f"- **GWAS variants analyzed:** {total_gwas}\n")
            f.write(f"- **DEG genes analyzed:** {total_deg}\n")
            f.write(f"- **Pathway genes analyzed:** {total_pathway}\n")
            f.write(f"- **Total analyses:** {total_gwas + total_deg + total_pathway}\n\n")

            # Cross-analysis findings
            f.write("## Cross-Analysis Findings\n\n")

            # Find genes appearing in multiple analyses
            gwas_genes = {r['gene_symbol'] for r in self.reports['gwas_variants'] if r.get('gene_symbol')}
            deg_genes = {r['gene_symbol'] for r in self.reports['deg_genes']}
            pathway_genes = {r['gene_symbol'] for r in self.reports['pathway_genes']}

            # Overlaps
            gwas_deg_overlap = gwas_genes & deg_genes
            gwas_pathway_overlap = gwas_genes & pathway_genes
            deg_pathway_overlap = deg_genes & pathway_genes
            all_three = gwas_genes & deg_genes & pathway_genes

            if all_three:
                f.write(f"### High-Confidence Genes (found in all analyses)\n\n")
                for gene in sorted(all_three):
                    f.write(f"- **{gene}** - Appears in GWAS, DEG, and pathway analyses\n")
                f.write("\n")

            if gwas_deg_overlap - all_three:
                f.write(f"### GWAS + DEG Overlap ({len(gwas_deg_overlap - all_three)} genes)\n\n")
                for gene in sorted(list(gwas_deg_overlap - all_three)[:10]):
                    f.write(f"- {gene}\n")
                f.write("\n")

            # Cost summary
            f.write("## Cost Summary\n\n")
            cost_report = self.analysis_system.get_cost_report()
            f.write(f"```json\n{json.dumps(cost_report, indent=2)}\n```\n\n")

            f.write("---\n")
            f.write("\n*See individual analysis reports for detailed findings.*\n")

        logger.info(f"Master report saved to {report_file}")
        return str(report_file)


def run_integrated_pipeline(gwas_file: Optional[str] = None,
                           deg_file: Optional[str] = None,
                           pathway_file: Optional[str] = None,
                           use_llm: bool = False) -> Dict[str, Any]:
    """
    Convenience function to run integrated genetic analysis pipeline

    Args:
        gwas_file: Path to GWAS results
        deg_file: Path to DEG results
        pathway_file: Path to pathway enrichment results
        use_llm: Enable LLM synthesis

    Returns:
        Summary of all analyses
    """
    pipeline = PipelineGeneticAnalysis(use_llm=use_llm)

    results = {}

    if gwas_file and Path(gwas_file).exists():
        logger.info("Running GWAS analysis...")
        results['gwas'] = pipeline.analyze_gwas_results(gwas_file)

    if deg_file and Path(deg_file).exists():
        logger.info("Running DEG analysis...")
        results['deg'] = pipeline.analyze_deg_results(deg_file)

    if pathway_file and Path(pathway_file).exists():
        logger.info("Running pathway analysis...")
        results['pathway'] = pipeline.analyze_pathway_genes(pathway_file)

    # Generate master report
    if results:
        master_report = pipeline.generate_master_report()
        results['master_report'] = master_report

    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test with mock data
    print("Pipeline integration module loaded successfully")
    print("Use run_integrated_pipeline() to analyze results")
