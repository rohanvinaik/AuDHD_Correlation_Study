#!/usr/bin/env python3
"""
Update Scanner for Database Monitoring

Monitors key databases for new data releases and version updates:
- RSS feeds for dataset announcements
- API endpoints for version checks
- Web scraping for release notes
- DOI tracking for new publications

Usage:
    python update_scanner.py --check-all
    python update_scanner.py --source SFARI --alert
    python update_scanner.py --daemon --interval 3600

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import re

try:
    import requests
    import feedparser
    from bs4 import BeautifulSoup
    import yaml
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install requests feedparser beautifulsoup4 pyyaml")
    import sys
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataUpdate:
    """Detected data update"""
    source: str
    dataset_name: str
    update_type: str  # new_release, version_update, new_data, announcement
    detected_date: str
    release_date: Optional[str]
    version: Optional[str]
    description: str
    url: str
    priority: str  # high, medium, low
    action_required: bool
    metadata: Optional[Dict]


@dataclass
class MonitoringState:
    """State tracking for monitoring"""
    source: str
    last_checked: str
    last_update_found: Optional[str]
    check_count: int
    updates_found: int
    content_hash: Optional[str]  # Hash of content to detect changes


class UpdateScanner:
    """Scan multiple sources for data updates"""

    def __init__(self, config_path: Path = Path('configs/monitoring_config.yaml')):
        self.config = self._load_config(config_path)
        self.state_file = Path('data/temp/monitoring_state.json')
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Study-Monitor/1.0 (Research; mailto:contact@example.com)'
        })

    def _load_config(self, path: Path) -> Dict:
        """Load monitoring configuration"""
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def _load_state(self) -> Dict:
        """Load monitoring state"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def _save_state(self):
        """Save monitoring state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _update_state(self, source: str, updates_found: int = 0, content_hash: Optional[str] = None):
        """Update monitoring state for source"""
        if source not in self.state:
            self.state[source] = {
                'last_checked': datetime.now().isoformat(),
                'last_update_found': None,
                'check_count': 0,
                'updates_found': 0,
                'content_hash': None
            }

        self.state[source]['last_checked'] = datetime.now().isoformat()
        self.state[source]['check_count'] += 1
        self.state[source]['updates_found'] += updates_found

        if updates_found > 0:
            self.state[source]['last_update_found'] = datetime.now().isoformat()

        if content_hash:
            self.state[source]['content_hash'] = content_hash

        self._save_state()

    def check_rss_feed(self, source: str, feed_url: str, keywords: List[str]) -> List[DataUpdate]:
        """Check RSS feed for updates"""
        updates = []

        try:
            logger.info(f"Checking RSS feed: {source}")
            feed = feedparser.parse(feed_url)

            # Get last check time
            last_checked = self.state.get(source, {}).get('last_checked')
            if last_checked:
                last_checked_dt = datetime.fromisoformat(last_checked)
            else:
                last_checked_dt = datetime.now() - timedelta(days=30)

            for entry in feed.entries:
                # Parse entry date
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    entry_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    entry_date = datetime(*entry.updated_parsed[:6])
                else:
                    entry_date = datetime.now()

                # Check if entry is new since last check
                if entry_date > last_checked_dt:
                    # Check if entry matches keywords
                    title = entry.get('title', '').lower()
                    summary = entry.get('summary', '').lower()
                    content = f"{title} {summary}"

                    if any(keyword.lower() in content for keyword in keywords):
                        update = DataUpdate(
                            source=source,
                            dataset_name=self._extract_dataset_name(entry.get('title', '')),
                            update_type='announcement',
                            detected_date=datetime.now().isoformat(),
                            release_date=entry_date.isoformat() if entry_date else None,
                            version=self._extract_version(entry.get('title', '')),
                            description=entry.get('summary', '')[:500],
                            url=entry.get('link', ''),
                            priority=self._determine_priority(content, keywords),
                            action_required=True,
                            metadata={
                                'title': entry.get('title', ''),
                                'author': entry.get('author', '')
                            }
                        )
                        updates.append(update)
                        logger.info(f"Found update: {update.dataset_name} from {source}")

            self._update_state(source, len(updates))

        except Exception as e:
            logger.error(f"Error checking RSS feed {source}: {e}")

        return updates

    def check_api_endpoint(self, source: str, api_url: str, params: Dict) -> List[DataUpdate]:
        """Check API endpoint for version updates"""
        updates = []

        try:
            logger.info(f"Checking API: {source}")
            response = self.session.get(api_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Calculate content hash
                content_str = json.dumps(data, sort_keys=True)
                current_hash = self._calculate_content_hash(content_str)
                previous_hash = self.state.get(source, {}).get('content_hash')

                # Check if content changed
                if previous_hash and current_hash != previous_hash:
                    update = DataUpdate(
                        source=source,
                        dataset_name=source,
                        update_type='version_update',
                        detected_date=datetime.now().isoformat(),
                        release_date=data.get('release_date') or data.get('updated'),
                        version=data.get('version'),
                        description='API endpoint data has changed',
                        url=api_url,
                        priority='medium',
                        action_required=True,
                        metadata=data
                    )
                    updates.append(update)
                    logger.info(f"Detected change in {source} API")

                self._update_state(source, len(updates), current_hash)

        except Exception as e:
            logger.error(f"Error checking API {source}: {e}")

        return updates

    def check_webpage(self, source: str, url: str, selectors: Dict) -> List[DataUpdate]:
        """Check webpage for updates using CSS selectors"""
        updates = []

        try:
            logger.info(f"Checking webpage: {source}")
            response = self.session.get(url, timeout=30)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract version/date info using selectors
                version_elem = soup.select_one(selectors.get('version', ''))
                date_elem = soup.select_one(selectors.get('date', ''))
                description_elem = soup.select_one(selectors.get('description', ''))

                version = version_elem.get_text(strip=True) if version_elem else None
                release_date = date_elem.get_text(strip=True) if date_elem else None
                description = description_elem.get_text(strip=True) if description_elem else ''

                # Calculate page content hash
                page_content = str(soup)
                current_hash = self._calculate_content_hash(page_content)
                previous_hash = self.state.get(source, {}).get('content_hash')

                # Check if page changed
                if previous_hash and current_hash != previous_hash:
                    update = DataUpdate(
                        source=source,
                        dataset_name=source,
                        update_type='new_release',
                        detected_date=datetime.now().isoformat(),
                        release_date=release_date,
                        version=version,
                        description=description[:500],
                        url=url,
                        priority='high',
                        action_required=True,
                        metadata={
                            'version': version,
                            'release_date': release_date
                        }
                    )
                    updates.append(update)
                    logger.info(f"Detected webpage change for {source}")

                self._update_state(source, len(updates), current_hash)

        except Exception as e:
            logger.error(f"Error checking webpage {source}: {e}")

        return updates

    def check_dbgap_studies(self, search_terms: List[str]) -> List[DataUpdate]:
        """Check dbGaP for new studies"""
        updates = []

        try:
            logger.info("Checking dbGaP for new studies")

            for term in search_terms:
                # dbGaP search API
                api_url = "https://www.ncbi.nlm.nih.gov/gap/"
                params = {
                    'term': term,
                    'report': 'brief'
                }

                response = self.session.get(api_url, params=params, timeout=30)

                if response.status_code == 200:
                    # Parse HTML for study listings
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Look for study accessions (phs numbers)
                    phs_pattern = r'phs\d{6}\.\w+\.\w+'
                    study_links = soup.find_all('a', href=re.compile(phs_pattern))

                    for link in study_links:
                        accession = re.search(phs_pattern, link.get('href', '')).group()

                        # Check if this is a new study
                        state_key = f"dbGaP_{accession}"
                        if state_key not in self.state:
                            update = DataUpdate(
                                source='dbGaP',
                                dataset_name=link.get_text(strip=True),
                                update_type='new_data',
                                detected_date=datetime.now().isoformat(),
                                release_date=None,
                                version=accession,
                                description=f'New dbGaP study matching "{term}"',
                                url=f"https://www.ncbi.nlm.nih.gov/gap/{accession}",
                                priority='high',
                                action_required=True,
                                metadata={'accession': accession, 'search_term': term}
                            )
                            updates.append(update)
                            self._update_state(state_key)
                            logger.info(f"Found new dbGaP study: {accession}")

        except Exception as e:
            logger.error(f"Error checking dbGaP: {e}")

        return updates

    def check_clinical_trials(self, conditions: List[str]) -> List[DataUpdate]:
        """Check ClinicalTrials.gov for new results postings"""
        updates = []

        try:
            logger.info("Checking ClinicalTrials.gov for new results")

            for condition in conditions:
                # ClinicalTrials.gov API v2
                api_url = "https://clinicaltrials.gov/api/v2/studies"
                params = {
                    'query.cond': condition,
                    'filter.overallStatus': 'COMPLETED',
                    'filter.resultsFirstPostDate': f'RANGE[{(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")},MAX]',
                    'pageSize': 100
                }

                response = self.session.get(api_url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    studies = data.get('studies', [])

                    for study in studies:
                        nct_id = study.get('protocolSection', {}).get('identificationModule', {}).get('nctId')
                        title = study.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle')

                        state_key = f"ClinicalTrials_{nct_id}"
                        if state_key not in self.state:
                            update = DataUpdate(
                                source='ClinicalTrials.gov',
                                dataset_name=title,
                                update_type='new_data',
                                detected_date=datetime.now().isoformat(),
                                release_date=None,
                                version=nct_id,
                                description=f'New results posted for {condition} trial',
                                url=f"https://clinicaltrials.gov/study/{nct_id}",
                                priority='medium',
                                action_required=False,
                                metadata={'nct_id': nct_id, 'condition': condition}
                            )
                            updates.append(update)
                            self._update_state(state_key)
                            logger.info(f"Found new trial results: {nct_id}")

        except Exception as e:
            logger.error(f"Error checking ClinicalTrials.gov: {e}")

        return updates

    def _extract_dataset_name(self, text: str) -> str:
        """Extract dataset name from text"""
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'(Release|Version|Update|Announcement|v\d+\.?\d*)', '', text, flags=re.IGNORECASE)
        return cleaned.strip()

    def _extract_version(self, text: str) -> Optional[str]:
        """Extract version number from text"""
        # Look for version patterns like v1.0, version 2.1, etc.
        version_match = re.search(r'v?(\d+\.?\d*\.?\d*)', text, re.IGNORECASE)
        if version_match:
            return version_match.group(1)
        return None

    def _determine_priority(self, content: str, keywords: List[str]) -> str:
        """Determine priority based on content and keywords"""
        # High priority keywords
        high_priority = ['release', 'available', 'published', 'new data']
        # Medium priority keywords
        medium_priority = ['update', 'revised', 'amended']

        content_lower = content.lower()

        if any(kw in content_lower for kw in high_priority):
            return 'high'
        elif any(kw in content_lower for kw in medium_priority):
            return 'medium'
        else:
            return 'low'

    def scan_all_sources(self) -> List[DataUpdate]:
        """Scan all configured sources"""
        all_updates = []

        sources = self.config.get('sources', {})

        # Check RSS feeds
        for source_name, source_config in sources.items():
            if source_config.get('type') == 'rss':
                updates = self.check_rss_feed(
                    source_name,
                    source_config['url'],
                    source_config.get('keywords', [])
                )
                all_updates.extend(updates)

            elif source_config.get('type') == 'api':
                updates = self.check_api_endpoint(
                    source_name,
                    source_config['url'],
                    source_config.get('params', {})
                )
                all_updates.extend(updates)

            elif source_config.get('type') == 'webpage':
                updates = self.check_webpage(
                    source_name,
                    source_config['url'],
                    source_config.get('selectors', {})
                )
                all_updates.extend(updates)

        # Check special sources
        if self.config.get('check_dbgap', True):
            updates = self.check_dbgap_studies(
                self.config.get('dbgap_search_terms', ['autism', 'ADHD'])
            )
            all_updates.extend(updates)

        if self.config.get('check_clinical_trials', True):
            updates = self.check_clinical_trials(
                self.config.get('clinical_trial_conditions', ['Autism', 'ADHD'])
            )
            all_updates.extend(updates)

        return all_updates

    def save_updates(self, updates: List[DataUpdate], output_file: Path):
        """Save detected updates to file"""
        # Load existing updates
        if output_file.exists():
            with open(output_file) as f:
                existing = json.load(f)
        else:
            existing = {'generated_date': datetime.now().isoformat(), 'updates': []}

        # Add new updates
        for update in updates:
            existing['updates'].append(asdict(update))

        # Save
        existing['generated_date'] = datetime.now().isoformat()
        existing['total_updates'] = len(existing['updates'])

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Saved {len(updates)} updates to {output_file}")

    def run_daemon(self, interval_seconds: int = 3600):
        """Run scanner as daemon"""
        logger.info(f"Starting update scanner daemon (interval: {interval_seconds}s)")

        while True:
            try:
                logger.info("Starting scan cycle...")
                updates = self.scan_all_sources()

                if updates:
                    logger.info(f"Found {len(updates)} updates")
                    self.save_updates(updates, Path('data/monitoring/detected_updates.json'))
                else:
                    logger.info("No new updates found")

                logger.info(f"Sleeping for {interval_seconds} seconds...")
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Daemon stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in daemon loop: {e}")
                time.sleep(60)  # Sleep 1 minute before retrying


def main():
    parser = argparse.ArgumentParser(description='Monitor databases for updates')
    parser.add_argument('--config', default='configs/monitoring_config.yaml',
                       help='Configuration file')
    parser.add_argument('--check-all', action='store_true',
                       help='Check all sources once')
    parser.add_argument('--source', help='Check specific source')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon')
    parser.add_argument('--interval', type=int, default=3600,
                       help='Daemon check interval in seconds')
    parser.add_argument('--output', default='data/monitoring/detected_updates.json',
                       help='Output file for updates')

    args = parser.parse_args()

    scanner = UpdateScanner(Path(args.config))

    if args.daemon:
        scanner.run_daemon(args.interval)
    elif args.check_all:
        updates = scanner.scan_all_sources()
        print(f"\n=== Found {len(updates)} updates ===\n")
        for update in updates:
            print(f"[{update.priority.upper()}] {update.source}: {update.dataset_name}")
            print(f"  Type: {update.update_type}")
            print(f"  URL: {update.url}")
            print(f"  Description: {update.description[:100]}...")
            print()
        scanner.save_updates(updates, Path(args.output))
    elif args.source:
        # Check specific source (implement source-specific logic)
        print(f"Checking {args.source}...")
    else:
        parser.print_help()


if __name__ == '__main__':
    import sys
    main()