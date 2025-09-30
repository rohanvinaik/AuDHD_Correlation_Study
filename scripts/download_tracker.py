#!/usr/bin/env python3
"""
Real-time Download Tracker
Monitors download progress and displays status
"""

import json
import time
from pathlib import Path
from datetime import datetime
import subprocess
import sys

project_root = Path("/Users/rohanvinaik/AuDHD_Correlation_Study")

# ANSI color codes for terminal
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def get_file_size_mb(filepath):
    """Get file size in MB"""
    try:
        return filepath.stat().st_size / (1024 * 1024)
    except:
        return 0

def parse_log_for_progress(log_file):
    """Parse log file for progress information"""
    if not log_file.exists():
        return None

    try:
        with open(log_file) as f:
            lines = f.readlines()

        if not lines:
            return None

        # Get last few lines for recent activity
        recent = lines[-10:]
        last_line = lines[-1].strip() if lines else ""

        return {
            'last_activity': last_line,
            'total_lines': len(lines)
        }
    except:
        return None

def check_data_status():
    """Check status of all data sources"""

    status = {
        'timestamp': datetime.now().isoformat(),
        'sources': {}
    }

    # 1. GEO Gene Expression Downloads
    geo_dir = project_root / "data/raw/geo"
    if geo_dir.exists():
        n_files = len(list(geo_dir.glob("*.txt*"))) + len(list(geo_dir.glob("*.tar")))
        total_size = sum(get_file_size_mb(f) for f in geo_dir.glob("*") if f.is_file())

        # Check individual dataset logs
        datasets_complete = []
        for dataset in ['GSE28521', 'GSE28475', 'GSE64018', 'GSE98793', 'GSE18123',
                       'GSE80655', 'GSE119605', 'GSE42133']:
            log_file = project_root / f"logs/downloads/{dataset}.log"
            if log_file.exists():
                log_info = parse_log_for_progress(log_file)
                if log_info and 'complete' in log_info['last_activity'].lower():
                    datasets_complete.append(dataset)

        status['sources']['GEO_Expression'] = {
            'status': f'✓ Complete ({len(datasets_complete)}/8)' if len(datasets_complete) == 8 else f'⟳ Downloading ({len(datasets_complete)}/8)',
            'datasets': ', '.join(datasets_complete[:3]) + ('...' if len(datasets_complete) > 3 else ''),
            'files': n_files,
            'size_mb': total_size
        }
    else:
        status['sources']['GEO_Expression'] = {
            'status': '○ Not started',
            'files': 0,
            'size_mb': 0
        }

    # 2. SRA Microbiome Downloads
    microbiome_dir = project_root / "data/raw/sra"
    if microbiome_dir.exists():
        n_samples = len([d for d in microbiome_dir.iterdir() if d.is_dir() and d.name.startswith('SRR')])
        total_size = sum(get_file_size_mb(f) for f in microbiome_dir.rglob("*") if f.is_file())

        # Parse SRA log
        sra_log = project_root / "logs/downloads/sra_download.log"
        sra_info = parse_log_for_progress(sra_log)
        eta = ""
        if sra_info and 'ETA' in sra_info['last_activity']:
            import re
            eta_match = re.search(r'ETA: ([\d.]+)', sra_info['last_activity'])
            if eta_match:
                eta = f" ETA: {eta_match.group(1)} min"

        status['sources']['SRA_Microbiome'] = {
            'status': f'⟳ Downloading{eta}',
            'samples': n_samples,
            'size_mb': total_size
        }
    else:
        status['sources']['SRA_Microbiome'] = {
            'status': '○ Not started',
            'samples': 0,
            'size_mb': 0
        }

    # 3. Paper Scraping
    paper_log = project_root / "logs/downloads/paper_scraping.log"
    paper_info = parse_log_for_progress(paper_log)

    if paper_info:
        # Count papers processed
        with open(paper_log) as f:
            content = f.read()
            processing_count = content.count('Processing:')
            downloaded_count = content.count('Downloading PMC')
            supplement_count = content.count('Downloaded supplement')

        status['sources']['Paper_Scraping'] = {
            'status': '⟳ Active' if 'Processing' in paper_info['last_activity'] else '✓ Complete',
            'papers_processed': processing_count,
            'papers_downloaded': downloaded_count,
            'supplements': supplement_count,
            'last': paper_info['last_activity'][:80] + '...' if len(paper_info['last_activity']) > 80 else paper_info['last_activity']
        }
    else:
        status['sources']['Paper_Scraping'] = {
            'status': '○ Not started'
        }

    # 4. Data Repository Extraction
    extraction_log = project_root / "logs/downloads/data_extraction.log"
    repos_dir = project_root / "data/papers/repositories"

    if extraction_log.exists():
        extract_info = parse_log_for_progress(extraction_log)

        # Count repositories found and downloaded
        with open(extraction_log) as f:
            content = f.read()
            repos_found = content.count('Found repositories')
            github_downloads = content.count('Downloaded ') - content.count('Downloaded supplement')

        # Count actual files
        repo_files = []
        total_repo_size = 0
        if repos_dir.exists():
            repo_files = list(repos_dir.glob("*.zip"))
            total_repo_size = sum(get_file_size_mb(f) for f in repo_files)

        status['sources']['Data_Repositories'] = {
            'status': '⟳ Extracting' if 'Processing' in extract_info['last_activity'] else '✓ Complete',
            'repos_found': repos_found,
            'files_downloaded': len(repo_files),
            'size_mb': total_repo_size,
            'last': extract_info['last_activity'][:80] + '...' if len(extract_info['last_activity']) > 80 else extract_info['last_activity']
        }
    else:
        status['sources']['Data_Repositories'] = {
            'status': '○ Not started'
        }

    # 5. GWAS Data (for reference)
    adhd_gwas = project_root / "data/raw/genetics/adhd_eur_jun2017.gz"
    status['sources']['GWAS_Data'] = {
        'status': '✓ Complete' if adhd_gwas.exists() else '✗ Missing',
        'size_mb': get_file_size_mb(adhd_gwas) if adhd_gwas.exists() else 0,
        'significant_snps': '317 variants'
    }

    return status

def display_status(status):
    """Display status in terminal"""

    # Clear screen
    print("\033[2J\033[H")

    # Header
    print(f"{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}AuDHD Multi-Omics Data Download - LIVE TRACKER{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Data sources with better formatting
    for source, info in status['sources'].items():
        source_name = source.replace('_', ' ')
        print(f"{BOLD}{source_name}:{RESET}")

        # Status with color
        status_text = info.get('status', 'Unknown')
        if '✓' in status_text:
            color = GREEN
        elif '⟳' in status_text:
            color = YELLOW
        elif '✗' in status_text:
            color = RED
        else:
            color = RESET

        print(f"  Status: {color}{status_text}{RESET}")

        # Show last activity if available
        if 'last' in info:
            # Extract just the message part after timestamp
            last_msg = info['last']
            if ' - INFO - ' in last_msg:
                last_msg = last_msg.split(' - INFO - ', 1)[1]
            elif ' - WARNING - ' in last_msg:
                last_msg = last_msg.split(' - WARNING - ', 1)[1]
            print(f"  {BLUE}→{RESET} {last_msg}")

        # Show key metrics
        for key, value in info.items():
            if key not in ['status', 'path', 'last', 'size_mb']:
                key_display = key.replace('_', ' ').title()
                print(f"  {key_display}: {value}")

        # File size if available
        if 'size_mb' in info and info['size_mb'] > 0:
            size_gb = info['size_mb'] / 1024
            if size_gb > 1:
                print(f"  {BOLD}Size: {size_gb:.2f} GB{RESET}")
            else:
                print(f"  Size: {info['size_mb']:.1f} MB")

        print()

    # Summary with totals
    ready_count = sum(1 for s in status['sources'].values() if '✓' in s.get('status', ''))
    downloading_count = sum(1 for s in status['sources'].values() if '⟳' in s.get('status', ''))
    total_count = len(status['sources'])

    # Calculate total downloaded size
    total_size_gb = sum(s.get('size_mb', 0) for s in status['sources'].values()) / 1024

    print(f"{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}SUMMARY:{RESET}")
    print(f"  {GREEN}✓ Complete: {ready_count}/{total_count}{RESET}")
    print(f"  {YELLOW}⟳ Active: {downloading_count}/{total_count}{RESET}")
    print(f"  {BOLD}Total Downloaded: {total_size_gb:.2f} GB{RESET}")

    # Show active processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        active_downloads = len([line for line in result.stdout.split('\n')
                               if 'python' in line and any(x in line for x in ['download', 'scrape', 'extract'])])
        print(f"  Active Download Processes: {active_downloads}")
    except:
        pass

    print(f"\n{BOLD}Next Steps:{RESET}")
    if ready_count == total_count:
        print(f"  {GREEN}✓ All downloads complete! Ready for analysis.{RESET}")
        print(f"  → Run: python scripts/run_full_analysis.py")
    else:
        print(f"  Downloads in progress... {downloading_count} active")
        print(f"  Monitor individual logs: tail -f logs/downloads/*.log")

    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"Press Ctrl+C to exit | Refreshes every 5 seconds")

def save_status_json(status):
    """Save status to JSON file"""
    status_file = project_root / "data/catalogs/download_status.json"
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

def main():
    """Main tracker loop"""

    print(f"{BOLD}Starting download tracker...{RESET}")
    print("This will update every 5 seconds. Press Ctrl+C to stop.\n")

    try:
        while True:
            status = check_data_status()
            display_status(status)
            save_status_json(status)
            time.sleep(5)

    except KeyboardInterrupt:
        print(f"\n\n{GREEN}Tracker stopped.{RESET}")
        print(f"Status saved to: data/catalogs/download_status.json")
        sys.exit(0)

if __name__ == '__main__':
    main()