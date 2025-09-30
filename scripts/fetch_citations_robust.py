#!/usr/bin/env python3
"""
Robust citation fetching with better error handling
Uses both elink and esummary APIs
"""
import json
import requests
import time
from pathlib import Path
import xml.etree.ElementTree as ET

def get_citations_elink(pmids, batch_size=100):
    """Fetch citations using elink API"""
    citations = {}

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
            response.raise_for_status()
            data = response.json()

            for linkset in data.get('linksets', []):
                pmid = linkset.get('ids', [''])[0]
                linksetdbs = linkset.get('linksetdbs', [])
                if linksetdbs:
                    citations[pmid] = len(linksetdbs[0].get('links', []))
                else:
                    citations[pmid] = 0

            print(f"  Processed {min(i+batch_size, len(pmids))}/{len(pmids)} papers (batch citations: {sum(citations.get(p, 0) for p in batch)})")

        except Exception as e:
            print(f"  Error on batch {i//batch_size + 1}: {e}")
            for pmid in batch:
                if pmid not in citations:
                    citations[pmid] = 0

        time.sleep(0.4)  # NCBI rate limit

    return citations

def main():
    project_root = Path("/Users/rohanvinaik/AuDHD_Correlation_Study")
    papers_file = project_root / "data/papers/scraped_papers.json"

    print("Loading papers...")
    with open(papers_file) as f:
        papers = json.load(f)

    print(f"Found {len(papers)} papers\n")

    # Extract PMIDs
    pmids = []
    pmid_to_paper = {}
    for paper in papers:
        pmid = paper.get('pmid')
        if pmid:
            pmids.append(pmid)
            pmid_to_paper[pmid] = paper

    print(f"Fetching citation counts for {len(pmids)} papers...")
    print(f"Estimated time: {len(pmids) * 0.4 / 60:.1f} minutes\n")

    # Get citations
    citations = get_citations_elink(pmids)

    # Add citations to papers
    for paper in papers:
        pmid = paper.get('pmid')
        if pmid:
            paper['citations'] = citations.get(pmid, 0)
            age = 2025 - int(paper.get('year', 2024))
            age = max(age, 1)
            paper['impact_score'] = paper['citations'] / age
        else:
            paper['citations'] = 0
            paper['impact_score'] = 0

    # Statistics
    with_cites = [p for p in papers if p.get('citations', 0) > 0]
    print(f"\n✓ Citation fetching complete!")
    print(f"  Papers with citations: {len(with_cites)}/{len(papers)}")
    print(f"  Total citations: {sum(p.get('citations', 0) for p in papers)}")

    # Sort by impact
    papers_sorted = sorted(papers, key=lambda x: x['impact_score'], reverse=True)

    # Save
    sorted_file = project_root / "data/papers/scraped_papers_sorted.json"
    with open(sorted_file, 'w') as f:
        json.dump(papers_sorted, f, indent=2)

    print(f"\n✓ Saved to {sorted_file}")

    # Show top 20
    print("\nTop 20 highest impact papers:")
    for i, paper in enumerate(papers_sorted[:20], 1):
        title = paper['title'][:70] + "..." if len(paper['title']) > 70 else paper['title']
        print(f"{i:2d}. {title}")
        print(f"    {paper['pmc_id']} | {paper['citations']} citations | {paper['year']} | Impact: {paper['impact_score']:.1f}")

if __name__ == '__main__':
    main()
