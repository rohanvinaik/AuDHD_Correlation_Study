#!/usr/bin/env python3
"""
Queue Processor for Download Pipeline

Manages download queue with priority levels and scheduling.

Usage:
    python queue_processor.py --config configs/download_config.yaml

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueueProcessor:
    """Process download queue with priorities"""

    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)
        self.queue_file = Path('data/temp/download_queue.json')
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self, path: Path) -> Dict:
        """Load configuration"""
        with open(path) as f:
            return yaml.safe_load(f)

    def add_to_queue(
        self,
        url: str,
        name: str,
        priority: str = 'normal',
        data_type: str = 'general',
        checksum: str = None
    ):
        """Add item to download queue"""
        queue = self._load_queue()

        item = {
            'url': url,
            'name': name,
            'priority': priority,
            'data_type': data_type,
            'checksum': checksum,
            'added_date': datetime.now().isoformat(),
            'status': 'pending'
        }

        queue.append(item)
        self._save_queue(queue)

        logger.info(f"Added to queue: {name} (priority: {priority})")

    def _load_queue(self) -> List[Dict]:
        """Load queue from file"""
        if self.queue_file.exists():
            with open(self.queue_file) as f:
                return json.load(f)
        return []

    def _save_queue(self, queue: List[Dict]):
        """Save queue to file"""
        with open(self.queue_file, 'w') as f:
            json.dump(queue, f, indent=2)

    def get_next_batch(self, batch_size: int = 5) -> List[Dict]:
        """Get next batch of items to download"""
        queue = self._load_queue()

        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'normal': 2, 'low': 3}
        queue_sorted = sorted(
            [q for q in queue if q['status'] == 'pending'],
            key=lambda x: priority_order.get(x['priority'], 2)
        )

        return queue_sorted[:batch_size]


def main():
    parser = argparse.ArgumentParser(description='Queue processor')
    parser.add_argument('--config', default='configs/download_config.yaml')
    parser.add_argument('--add-url', help='Add URL to queue')
    parser.add_argument('--name', help='Name for item')
    parser.add_argument('--priority', default='normal', choices=['critical', 'high', 'normal', 'low'])

    args = parser.parse_args()

    processor = QueueProcessor(Path(args.config))

    if args.add_url:
        processor.add_to_queue(args.add_url, args.name, args.priority)
        print(f"Added to queue: {args.name}")
    else:
        batch = processor.get_next_batch()
        print(f"Next batch: {len(batch)} items")
        for item in batch:
            print(f"  - {item['name']} ({item['priority']})")


if __name__ == '__main__':
    main()