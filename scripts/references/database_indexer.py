#!/usr/bin/env python3
"""
Pathway and Interaction Database Indexer

Downloads and indexes biological pathway databases, protein-protein interactions,
drug-gene interactions, and metabolic pathways for ADHD/Autism research.

Supported Databases:
- KEGG: Metabolic and signaling pathways
- Reactome: Curated biological pathways
- WikiPathways: Community-curated pathways
- STRING: Protein-protein interactions
- BioGRID: Genetic and protein interactions
- DGIdb: Drug-gene interactions
- STITCH: Chemical-protein interactions

Creates SQLite database with:
- Pathway membership (genes in pathways)
- Protein interactions (network edges)
- Drug-gene interactions
- Pathway hierarchies
- Annotations and evidence codes

Requirements:
    pip install requests pandas sqlite3 networkx

Usage:
    # Download and index all databases
    python database_indexer.py --download-all --output data/references/

    # Index specific database
    python database_indexer.py --database STRING --output data/references/

    # Query indexed database
    python database_indexer.py --query-gene DRD4 --database data/references/pathway_database.db

    # Get pathways for gene list
    python database_indexer.py --query-pathways genes.txt --database data/references/pathway_database.db

Author: AuDHD Correlation Study Team
"""

import argparse
import gzip
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging

try:
    import requests
    import pandas as pd
    import networkx as nx
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install requests pandas networkx")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Database download URLs
DATABASE_SOURCES = {
    'KEGG': {
        'name': 'KEGG Pathways',
        'url_base': 'https://rest.kegg.jp',
        'license': 'Academic use only',
        'description': 'Metabolic and signaling pathways',
        'relevance': 'Dopamine, serotonin, GABA pathways',
        'api_methods': ['list pathway', 'get pathway', 'link genes pathway']
    },
    'REACTOME': {
        'name': 'Reactome',
        'url': 'https://reactome.org/download/current/UniProt2Reactome.txt',
        'pathways_url': 'https://reactome.org/download/current/ReactomePathways.txt',
        'relations_url': 'https://reactome.org/download/current/ReactomePathwaysRelation.txt',
        'license': 'Open (CC BY 4.0)',
        'description': 'Curated biological pathways',
        'pathways_count': 2500
    },
    'WIKIPATHWAYS': {
        'name': 'WikiPathways',
        'url': 'http://data.wikipathways.org/current/gmt/wikipathways-20240810-gmt-Homo_sapiens.gmt',
        'license': 'Open (CC BY 3.0)',
        'description': 'Community-curated pathways',
        'format': 'GMT'
    },
    'GO_PATHWAYS': {
        'name': 'Gene Ontology Biological Process',
        'url': 'http://geneontology.org/gene-associations/goa_human.gaf.gz',
        'license': 'Open',
        'description': 'GO biological process annotations',
        'format': 'GAF'
    },
    'STRING': {
        'name': 'STRING Protein Interactions',
        'url': 'https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz',
        'info_url': 'https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz',
        'license': 'Open (CC BY 4.0)',
        'description': 'Protein-protein interaction network',
        'interactions_count': 11000000,
        'confidence_threshold': 400
    },
    'BIOGRID': {
        'name': 'BioGRID',
        'url': 'https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.228/BIOGRID-ORGANISM-4.4.228.tab3.zip',
        'license': 'Open (MIT)',
        'description': 'Genetic and protein interactions',
        'note': 'Update version number in URL'
    },
    'DGIDB': {
        'name': 'DGIdb Drug-Gene Interactions',
        'url': 'https://www.dgidb.org/data/monthly_tsvs/2024-Jan/interactions.tsv',
        'categories_url': 'https://www.dgidb.org/data/monthly_tsvs/2024-Jan/categories.tsv',
        'license': 'Open (CC0)',
        'description': 'Drug-gene interaction database',
        'relevance': 'ADHD medications, SSRIs',
        'note': 'Update month in URL'
    },
    'STITCH': {
        'name': 'STITCH Chemical-Protein Interactions',
        'url': 'http://stitch.embl.de/download/protein_chemical.links.v5.0/9606.protein_chemical.links.v5.0.txt.gz',
        'license': 'Open (CC BY 4.0)',
        'description': 'Chemical-protein interactions',
        'relevance': 'Drug mechanisms, metabolites'
    },
    'MSIGDB': {
        'name': 'MSigDB Gene Sets',
        'url': 'https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp',
        'license': 'Open (with registration)',
        'description': 'Curated gene sets for enrichment analysis',
        'note': 'Requires registration at GSEA website'
    }
}

# ADHD/Autism relevant pathways
RELEVANT_PATHWAYS = {
    'neurotransmitter': [
        'Dopaminergic synapse', 'Serotonergic synapse', 'GABAergic synapse',
        'Glutamatergic synapse', 'Cholinergic synapse',
        'Dopamine synthesis', 'Serotonin synthesis', 'GABA synthesis'
    ],
    'synaptic': [
        'Synaptic vesicle cycle', 'Long-term potentiation', 'Long-term depression',
        'Neurotransmitter release cycle', 'Synaptic transmission'
    ],
    'neurodevelopment': [
        'Axon guidance', 'Neurotrophin signaling', 'Neuron development',
        'Synaptogenesis', 'Neuronal migration'
    ],
    'signaling': [
        'MAPK signaling', 'PI3K-Akt signaling', 'Calcium signaling',
        'cAMP signaling', 'Wnt signaling'
    ],
    'inflammation': [
        'Cytokine signaling', 'TNF signaling', 'NF-kappa B signaling',
        'Inflammatory response'
    ],
    'metabolism': [
        'Tryptophan metabolism', 'Tyrosine metabolism', 'Phenylalanine metabolism',
        'Folate metabolism', 'One-carbon metabolism'
    ]
}


class DatabaseIndexer:
    """Download and index pathway/interaction databases"""

    def __init__(self, output_dir: Path, db_path: Optional[Path] = None):
        """
        Initialize indexer

        Args:
            output_dir: Output directory for downloads
            db_path: Path to SQLite database (default: output_dir/pathway_database.db)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if db_path is None:
            db_path = self.output_dir / 'pathway_database.db'

        self.db_path = db_path
        self.conn = None

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        logger.info(f"Initialized database indexer: {output_dir}")
        logger.info(f"Database: {db_path}")

    def initialize_database(self):
        """Create database schema"""
        logger.info("Initializing database schema...")

        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Pathways table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pathways (
                pathway_id TEXT PRIMARY KEY,
                pathway_name TEXT,
                database_source TEXT,
                category TEXT,
                description TEXT,
                organism TEXT DEFAULT 'Homo sapiens',
                gene_count INTEGER,
                url TEXT
            )
        """)

        # Pathway membership table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pathway_genes (
                pathway_id TEXT,
                gene_symbol TEXT,
                gene_id TEXT,
                evidence_code TEXT,
                FOREIGN KEY (pathway_id) REFERENCES pathways(pathway_id)
            )
        """)

        # Protein interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS protein_interactions (
                protein_a TEXT,
                protein_b TEXT,
                database_source TEXT,
                interaction_type TEXT,
                confidence_score REAL,
                evidence TEXT,
                PRIMARY KEY (protein_a, protein_b, database_source)
            )
        """)

        # Drug-gene interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drug_gene_interactions (
                drug_name TEXT,
                drug_chembl_id TEXT,
                gene_symbol TEXT,
                gene_entrez_id TEXT,
                interaction_type TEXT,
                database_source TEXT,
                pmid TEXT
            )
        """)

        # Pathway hierarchy table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pathway_hierarchy (
                parent_pathway_id TEXT,
                child_pathway_id TEXT,
                FOREIGN KEY (parent_pathway_id) REFERENCES pathways(pathway_id),
                FOREIGN KEY (child_pathway_id) REFERENCES pathways(pathway_id)
            )
        """)

        # Download metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS database_metadata (
                database_name TEXT PRIMARY KEY,
                download_date TEXT,
                version TEXT,
                source_url TEXT,
                record_count INTEGER
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pathway_genes ON pathway_genes(gene_symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pathway_id ON pathway_genes(pathway_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_protein_a ON protein_interactions(protein_a)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_protein_b ON protein_interactions(protein_b)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drug_gene ON drug_gene_interactions(gene_symbol)")

        self.conn.commit()
        logger.info("Database schema created")

    def download_reactome(self) -> bool:
        """Download and index Reactome pathways"""
        logger.info("Downloading Reactome...")

        try:
            source = DATABASE_SOURCES['REACTOME']

            # Download gene-pathway mappings
            logger.info(f"Downloading: {source['url']}")
            response = self.session.get(source['url'], timeout=60)
            gene_pathway_file = self.output_dir / 'UniProt2Reactome.txt'
            gene_pathway_file.write_text(response.text)

            # Download pathway names
            logger.info(f"Downloading: {source['pathways_url']}")
            response = self.session.get(source['pathways_url'], timeout=60)
            pathways_file = self.output_dir / 'ReactomePathways.txt'
            pathways_file.write_text(response.text)

            # Download pathway relations
            logger.info(f"Downloading: {source['relations_url']}")
            response = self.session.get(source['relations_url'], timeout=60)
            relations_file = self.output_dir / 'ReactomePathwaysRelation.txt'
            relations_file.write_text(response.text)

            # Index pathways
            logger.info("Indexing Reactome pathways...")
            pathway_names = {}
            with open(pathways_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        pathway_id, name = parts[0], parts[1]
                        if 'Homo sapiens' in line:
                            pathway_names[pathway_id] = name

            # Insert pathways
            cursor = self.conn.cursor()
            for pathway_id, name in pathway_names.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO pathways
                    (pathway_id, pathway_name, database_source, organism, url)
                    VALUES (?, ?, ?, ?, ?)
                """, (pathway_id, name, 'Reactome', 'Homo sapiens',
                     f'https://reactome.org/content/detail/{pathway_id}'))

            # Insert gene-pathway associations
            gene_count = 0
            with open(gene_pathway_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        uniprot, pathway_id, url, name, evidence, species = parts[:6]
                        if species == 'Homo sapiens' and pathway_id in pathway_names:
                            cursor.execute("""
                                INSERT INTO pathway_genes
                                (pathway_id, gene_symbol, gene_id, evidence_code)
                                VALUES (?, ?, ?, ?)
                            """, (pathway_id, uniprot, uniprot, evidence))
                            gene_count += 1

            # Insert pathway hierarchy
            with open(relations_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        parent, child = parts[0], parts[1]
                        if parent in pathway_names and child in pathway_names:
                            cursor.execute("""
                                INSERT INTO pathway_hierarchy (parent_pathway_id, child_pathway_id)
                                VALUES (?, ?)
                            """, (parent, child))

            # Update metadata
            cursor.execute("""
                INSERT OR REPLACE INTO database_metadata
                (database_name, download_date, version, source_url, record_count)
                VALUES (?, ?, ?, ?, ?)
            """, ('Reactome', datetime.now().isoformat(), 'current',
                 source['url'], len(pathway_names)))

            self.conn.commit()
            logger.info(f"Indexed {len(pathway_names)} Reactome pathways with {gene_count} associations")
            return True

        except Exception as e:
            logger.error(f"Error downloading Reactome: {e}")
            return False

    def download_string(self) -> bool:
        """Download and index STRING protein interactions"""
        logger.info("Downloading STRING...")

        try:
            source = DATABASE_SOURCES['STRING']

            # Download protein info
            logger.info(f"Downloading protein info...")
            info_file = self.output_dir / '9606.protein.info.v12.0.txt.gz'
            if not info_file.exists():
                response = self.session.get(source['info_url'], stream=True, timeout=300)
                with open(info_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Parse protein names
            logger.info("Parsing protein names...")
            protein_names = {}
            with gzip.open(info_file, 'rt') as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        string_id, gene_name = parts[0], parts[1]
                        protein_names[string_id] = gene_name

            # Download interactions
            logger.info(f"Downloading interactions...")
            links_file = self.output_dir / '9606.protein.links.v12.0.txt.gz'
            if not links_file.exists():
                response = self.session.get(source['url'], stream=True, timeout=600)
                with open(links_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Index interactions
            logger.info("Indexing STRING interactions...")
            cursor = self.conn.cursor()
            interaction_count = 0
            confidence_threshold = source['confidence_threshold']

            with gzip.open(links_file, 'rt') as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        protein1, protein2, score = parts[0], parts[1], int(parts[2])

                        if score >= confidence_threshold:
                            gene1 = protein_names.get(protein1, protein1.split('.')[-1])
                            gene2 = protein_names.get(protein2, protein2.split('.')[-1])

                            cursor.execute("""
                                INSERT OR REPLACE INTO protein_interactions
                                (protein_a, protein_b, database_source, confidence_score)
                                VALUES (?, ?, ?, ?)
                            """, (gene1, gene2, 'STRING', score / 1000.0))

                            interaction_count += 1

                            if interaction_count % 100000 == 0:
                                logger.info(f"Indexed {interaction_count} interactions...")
                                self.conn.commit()

            # Update metadata
            cursor.execute("""
                INSERT OR REPLACE INTO database_metadata
                (database_name, download_date, version, source_url, record_count)
                VALUES (?, ?, ?, ?, ?)
            """, ('STRING', datetime.now().isoformat(), 'v12.0',
                 source['url'], interaction_count))

            self.conn.commit()
            logger.info(f"Indexed {interaction_count} STRING interactions (confidence >= {confidence_threshold})")
            return True

        except Exception as e:
            logger.error(f"Error downloading STRING: {e}")
            return False

    def download_dgidb(self) -> bool:
        """Download and index DGIdb drug-gene interactions"""
        logger.info("Downloading DGIdb...")

        try:
            source = DATABASE_SOURCES['DGIDB']

            # Download interactions
            logger.info(f"Downloading: {source['url']}")
            response = self.session.get(source['url'], timeout=60)
            interactions_file = self.output_dir / 'dgidb_interactions.tsv'
            interactions_file.write_text(response.text)

            # Index interactions
            logger.info("Indexing drug-gene interactions...")
            cursor = self.conn.cursor()
            interaction_count = 0

            with open(interactions_file) as f:
                header = next(f).strip().split('\t')
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        gene, drug, interaction_type, source_db = parts[:4]

                        cursor.execute("""
                            INSERT INTO drug_gene_interactions
                            (drug_name, gene_symbol, interaction_type, database_source)
                            VALUES (?, ?, ?, ?)
                        """, (drug, gene, interaction_type, source_db))

                        interaction_count += 1

            # Update metadata
            cursor.execute("""
                INSERT OR REPLACE INTO database_metadata
                (database_name, download_date, version, source_url, record_count)
                VALUES (?, ?, ?, ?, ?)
            """, ('DGIdb', datetime.now().isoformat(), '2024-Jan',
                 source['url'], interaction_count))

            self.conn.commit()
            logger.info(f"Indexed {interaction_count} drug-gene interactions")
            return True

        except Exception as e:
            logger.error(f"Error downloading DGIdb: {e}")
            return False

    def query_gene_pathways(self, gene_symbol: str) -> List[Dict]:
        """Query pathways for a gene"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT p.pathway_id, p.pathway_name, p.database_source, p.category
            FROM pathways p
            JOIN pathway_genes pg ON p.pathway_id = pg.pathway_id
            WHERE pg.gene_symbol = ? OR pg.gene_id = ?
        """, (gene_symbol, gene_symbol))

        results = []
        for row in cursor.fetchall():
            results.append({
                'pathway_id': row[0],
                'pathway_name': row[1],
                'database': row[2],
                'category': row[3]
            })

        return results

    def query_gene_interactions(self, gene_symbol: str) -> List[Dict]:
        """Query protein interactions for a gene"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT protein_b, database_source, confidence_score
            FROM protein_interactions
            WHERE protein_a = ?
            UNION
            SELECT protein_a, database_source, confidence_score
            FROM protein_interactions
            WHERE protein_b = ?
        """, (gene_symbol, gene_symbol))

        results = []
        for row in cursor.fetchall():
            results.append({
                'partner': row[0],
                'database': row[1],
                'confidence': row[2]
            })

        return results

    def query_drug_targets(self, drug_name: str) -> List[Dict]:
        """Query gene targets for a drug"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT gene_symbol, interaction_type, database_source
            FROM drug_gene_interactions
            WHERE drug_name LIKE ?
        """, (f'%{drug_name}%',))

        results = []
        for row in cursor.fetchall():
            results.append({
                'gene': row[0],
                'interaction_type': row[1],
                'database': row[2]
            })

        return results

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    parser = argparse.ArgumentParser(
        description='Download and index pathway and interaction databases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize database and download all
  python database_indexer.py --download-all --output data/references/

  # Download specific database
  python database_indexer.py --database STRING --output data/references/

  # Query gene pathways
  python database_indexer.py --query-gene DRD4 --database data/references/pathway_database.db

  # Query drug targets
  python database_indexer.py --query-drug methylphenidate --database data/references/pathway_database.db

  # List database statistics
  python database_indexer.py --stats --database data/references/pathway_database.db
        """
    )

    parser.add_argument(
        '--download-all',
        action='store_true',
        help='Download and index all databases'
    )

    parser.add_argument(
        '--database',
        type=str,
        choices=['REACTOME', 'STRING', 'DGIDB'],
        help='Download specific database'
    )

    parser.add_argument(
        '--query-gene',
        type=str,
        help='Query pathways and interactions for gene'
    )

    parser.add_argument(
        '--query-drug',
        type=str,
        help='Query gene targets for drug'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/references',
        help='Output directory'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        help='Path to SQLite database'
    )

    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else None
    indexer = DatabaseIndexer(Path(args.output), db_path=db_path)

    # Handle stats
    if args.stats:
        indexer.conn = sqlite3.connect(indexer.db_path)
        cursor = indexer.conn.cursor()

        print("\n=== Database Statistics ===\n")

        cursor.execute("SELECT COUNT(*) FROM pathways")
        print(f"Pathways: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM pathway_genes")
        print(f"Pathway-gene associations: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM protein_interactions")
        print(f"Protein interactions: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM drug_gene_interactions")
        print(f"Drug-gene interactions: {cursor.fetchone()[0]}")

        print("\nDownload history:")
        cursor.execute("SELECT * FROM database_metadata")
        for row in cursor.fetchall():
            print(f"\n{row[0]}:")
            print(f"  Downloaded: {row[1]}")
            print(f"  Version: {row[2]}")
            print(f"  Records: {row[4]}")

        indexer.close()
        return

    # Handle queries
    if args.query_gene or args.query_drug:
        indexer.conn = sqlite3.connect(indexer.db_path)

        if args.query_gene:
            print(f"\n=== Pathways for {args.query_gene} ===\n")
            pathways = indexer.query_gene_pathways(args.query_gene)
            if pathways:
                for pw in pathways:
                    print(f"{pw['pathway_name']} ({pw['database']})")
            else:
                print("No pathways found")

            print(f"\n=== Interactions for {args.query_gene} ===\n")
            interactions = indexer.query_gene_interactions(args.query_gene)
            if interactions:
                for i in interactions[:20]:
                    print(f"{i['partner']} (confidence: {i['confidence']:.2f})")
                if len(interactions) > 20:
                    print(f"\n... and {len(interactions) - 20} more")
            else:
                print("No interactions found")

        if args.query_drug:
            print(f"\n=== Targets for {args.query_drug} ===\n")
            targets = indexer.query_drug_targets(args.query_drug)
            if targets:
                for t in targets:
                    print(f"{t['gene']}: {t['interaction_type']}")
            else:
                print("No targets found")

        indexer.close()
        return

    # Handle downloads
    indexer.initialize_database()

    if args.download_all:
        logger.info("Downloading all databases...")
        indexer.download_reactome()
        time.sleep(2)
        indexer.download_string()
        time.sleep(2)
        indexer.download_dgidb()

    elif args.database:
        if args.database == 'REACTOME':
            indexer.download_reactome()
        elif args.database == 'STRING':
            indexer.download_string()
        elif args.database == 'DGIDB':
            indexer.download_dgidb()

    else:
        parser.print_help()

    indexer.close()


if __name__ == '__main__':
    main()