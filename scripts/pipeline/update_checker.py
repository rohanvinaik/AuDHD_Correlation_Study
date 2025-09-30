#!/usr/bin/env python3
"""
Update Checker for Incremental Downloads

Checks for new data and updates using:
- Last-Modified headers
- ETag comparison
- Version tracking
- Incremental updates

Usage:
    python update_checker.py --check-all --config configs/download_config.yaml

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

try:
    import requests
    import yaml
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install requests pyyaml")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UpdateChecker:
    """Check for dataset updates"""

    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)
        self.state_file = Path('data/temp/update_state.json')
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_config(self, path: Path) -> Dict:
        """Load configuration"""
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def _load_state(self) -> Dict:
        """Load update state"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def _save_state(self):
        """Save update state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def check_for_updates(
        self,
        url: str,
        name: str
    ) -> Optional[Dict]:
        """
        Check if URL has updates

        Returns dict with update info if available
        """
        try:
            # HEAD request to check metadata
            response = requests.head(url, timeout=10, allow_redirects=True)

            if response.status_code != 200:
                logger.warning(f"Failed to check {name}: HTTP {response.status_code}")
                return None

            # Get metadata
            last_modified = response.headers.get('Last-Modified')
            etag = response.headers.get('ETag')
            content_length = response.headers.get('Content-Length')

            # Compare with stored state
            stored = self.state.get(name, {})

            has_update = False
            reasons = []

            if last_modified and last_modified != stored.get('last_modified'):
                has_update = True
                reasons.append('last_modified changed')

            if etag and etag != stored.get('etag'):
                has_update = True
                reasons.append('etag changed')

            if content_length and content_length != stored.get('content_length'):
                has_update = True
                reasons.append('size changed')

            if has_update or name not in self.state:
                update_info = {
                    'name': name,
                    'url': url,
                    'has_update': has_update,
                    'reasons': reasons,
                    'last_modified': last_modified,
                    'etag': etag,
                    'content_length': content_length,
                    'checked_date': datetime.now().isoformat()
                }

                # Update state
                self.state[name] = {
                    'last_modified': last_modified,
                    'etag': etag,
                    'content_length': content_length,
                    'last_downloaded': stored.get('last_downloaded'),
                    'checked_date': datetime.now().isoformat()
                }
                self._save_state()

                return update_info

            logger.info(f"No updates for {name}")
            return None

        except Exception as e:
            logger.error(f"Error checking {name}: {e}")
            return None

    def mark_downloaded(self, name: str):
        """Mark dataset as downloaded"""
        if name in self.state:
            self.state[name]['last_downloaded'] = datetime.now().isoformat()
            self._save_state()


def main():
    parser = argparse.ArgumentParser(description='Check for data updates')
    parser.add_argument('--config', default='configs/download_config.yaml')
    parser.add_argument('--check-all', action='store_true', help='Check all configured datasets')
    parser.add_argument('--url', help='URL to check')
    parser.add_argument('--name', help='Dataset name')

    args = parser.parse_args()

    checker = UpdateChecker(Path(args.config))

    if args.check_all:
        # Check all datasets from config
        datasets = checker.config.get('datasets', [])
        updates = []

        for dataset in datasets:
            update = checker.check_for_updates(dataset['url'], dataset['name'])
            if update and update['has_update']:
                updates.append(update)

        print(f"\n=== Update Check ===")
        print(f"Checked {len(datasets)} datasets")
        print(f"Updates available: {len(updates)}")

        if updates:
            print("\nDatasets with updates:")
            for update in updates:
                print(f"  - {update['name']}: {', '.join(update['reasons'])}")

    elif args.url and args.name:
        update = checker.check_for_updates(args.url, args.name)
        if update and update['has_update']:
            print(f"Update available for {args.name}")
            print(f"Reasons: {', '.join(update['reasons'])}")
        else:
            print(f"No updates for {args.name}")


if __name__ == '__main__':
    import sys
    main()