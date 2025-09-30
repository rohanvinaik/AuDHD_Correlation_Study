#!/usr/bin/env python3
"""
Sort papers by citation count and extract data in batches
"""
import json
import requests
import time
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

def get_citation_counts(pmids, batch_size=200):
    """Fetch citation counts from NCBI E-utilities"""
    citations = {}

    # Process in batches
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        pmid_str = ','.join(batch)

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        params = {
            'dbfrom': 'pubmed',
            'id': pmid_str,
            'linkname': 'pubmed_pubmed_citedin',
            'retmode': 'json'
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            # Parse citation counts
            for linkset in data.get('linksets', []):
                pmid = linkset.get('ids', [''])[0]
                linksetdbs = linkset.get('linksetdbs', [])
                if linksetdbs:
                    citations[pmid] = len(linksetdbs[0].get('links', []))
                else:
                    citations[pmid] = 0

        except Exception as e:
            print(f"Error fetching citations for batch: {e}")
            # Default to 0 for this batch
            for pmid in batch:
                citations[pmid] = 0

        time.sleep(0.5)  # Be nice to NCBI

    return citations

def main():
    project_root = Path("/Users/rohanvinaik/AuDHD_Correlation_Study")
    papers_file = project_root / "data/papers/scraped_papers.json"

    print("Loading papers...")
    with open(papers_file) as f:
        papers = json.load(f)

    print(f"Found {len(papers)} papers")

    # Extract PMIDs
    pmids = [p['pmid'] for p in papers if p.get('pmid')]
    print(f"Fetching citation counts for {len(pmids)} papers...")

    # Get citation counts
    citations = get_citation_counts(pmids)

    # Add citations to papers and calculate impact score
    for paper in papers:
        pmid = paper.get('pmid')
        if pmid:
            paper['citations'] = citations.get(pmid, 0)
            # Impact score = citations / age (normalized)
            age = 2025 - int(paper.get('year', 2024))
            age = max(age, 1)  # Avoid division by zero
            paper['impact_score'] = paper['citations'] / age
        else:
            paper['citations'] = 0
            paper['impact_score'] = 0

    # Sort by impact score
    papers_sorted = sorted(papers, key=lambda x: x['impact_score'], reverse=True)

    # Save sorted papers
    sorted_file = project_root / "data/papers/scraped_papers_sorted.json"
    with open(sorted_file, 'w') as f:
        json.dump(papers_sorted, f, indent=2)

    print(f"\n✓ Papers sorted by impact and saved to {sorted_file}")

    # Show top 10
    print("\nTop 10 highest impact papers:")
    for i, paper in enumerate(papers_sorted[:10], 1):
        print(f"{i}. {paper['title'][:80]}...")
        print(f"   Citations: {paper['citations']}, Year: {paper['year']}, Impact: {paper['impact_score']:.2f}")

    # Get PMC IDs that haven't been extracted yet
    extracted_file = project_root / "data/papers/repositories/summary.json"
    if extracted_file.exists():
        with open(extracted_file) as f:
            extracted = json.load(f)
        extracted_pmcs = set(e['pmc_id'] for e in extracted)
    else:
        extracted_pmcs = set()

    # Get next batch of 50 high-impact papers not yet extracted
    to_extract = []
    for paper in papers_sorted:
        pmc_id = paper.get('pmc_id', '').replace('PMC', '')
        if pmc_id and pmc_id not in extracted_pmcs:
            to_extract.append(pmc_id)
        if len(to_extract) >= 50:
            break

    print(f"\n✓ Identified {len(to_extract)} high-impact papers for extraction")

    if to_extract:
        print("\nStarting data extraction for next batch...")

        # Run extraction
        cmd = [
            sys.executable,
            str(project_root / "scripts/extract_data_repos.py"),
            "--pmc-ids", *to_extract,
            "--output", str(project_root / "data/papers")
        ]

        log_file = project_root / "logs/downloads/data_extraction_batch2.log"
        with open(log_file, 'w') as f:
            subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

        print(f"✓ Extraction started in background")
        print(f"   Monitor: tail -f {log_file}")
    else:
        print("\n✓ All high-impact papers already extracted!")

if __name__ == '__main__':
    main()
