#!/usr/bin/env python3
"""
Download Manager with Parallel Processing and Retry Logic

Manages automated downloads with:
- Parallel download processing
- Exponential backoff retry logic
- Progress tracking
- Checksum verification
- Resumable downloads

Requirements:
    pip install requests tqdm pyyaml

Usage:
    # Download from queue
    python download_manager.py \\
        --config configs/download_config.yaml \\
        --parallel 5

    # Add item to queue
    python download_manager.py \\
        --add-url https://example.com/data.zip \\
        --name example_dataset \\
        --priority high

Author: AuDHD Correlation Study Team
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    import requests
    from tqdm import tqdm
    import yaml
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install requests tqdm pyyaml")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/download_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DownloadTask:
    """Download task with metadata"""

    def __init__(
        self,
        url: str,
        name: str,
        output_dir: Path,
        priority: str = 'normal',
        checksum: Optional[str] = None,
        checksum_type: str = 'sha256',
        headers: Optional[Dict] = None
    ):
        self.url = url
        self.name = name
        self.output_dir = Path(output_dir)
        self.priority = priority
        self.checksum = checksum
        self.checksum_type = checksum_type
        self.headers = headers or {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Status tracking
        self.status = 'pending'  # pending, downloading, completed, failed
        self.attempts = 0
        self.max_attempts = 3
        self.file_path = None
        self.error_message = None
        self.start_time = None
        self.end_time = None
        self.bytes_downloaded = 0
        self.total_bytes = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'url': self.url,
            'name': self.name,
            'output_dir': str(self.output_dir),
            'priority': self.priority,
            'checksum': self.checksum,
            'checksum_type': self.checksum_type,
            'status': self.status,
            'attempts': self.attempts,
            'file_path': str(self.file_path) if self.file_path else None,
            'error_message': self.error_message,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'bytes_downloaded': self.bytes_downloaded,
            'total_bytes': self.total_bytes
        }


class DownloadManager:
    """Manage parallel downloads with retry logic"""

    PRIORITY_ORDER = {'critical': 0, 'high': 1, 'normal': 2, 'low': 3}

    def __init__(
        self,
        max_workers: int = 5,
        retry_backoff: float = 2.0,
        chunk_size: int = 8192
    ):
        """
        Initialize download manager

        Args:
            max_workers: Maximum parallel downloads
            retry_backoff: Exponential backoff multiplier
            chunk_size: Download chunk size in bytes
        """
        self.max_workers = max_workers
        self.retry_backoff = retry_backoff
        self.chunk_size = chunk_size

        self.queue: List[DownloadTask] = []
        self.completed: List[DownloadTask] = []
        self.failed: List[DownloadTask] = []

        self.lock = threading.Lock()

        logger.info(f"Initialized download manager: {max_workers} workers")

    def add_task(self, task: DownloadTask):
        """Add task to queue"""
        with self.lock:
            self.queue.append(task)
            self._sort_queue()
        logger.info(f"Added task: {task.name} (priority: {task.priority})")

    def _sort_queue(self):
        """Sort queue by priority"""
        self.queue.sort(key=lambda t: self.PRIORITY_ORDER.get(t.priority, 2))

    def download_file(
        self,
        task: DownloadTask,
        resume: bool = True
    ) -> bool:
        """
        Download file with resume support

        Args:
            task: Download task
            resume: Enable resumable downloads

        Returns:
            True if successful
        """
        task.status = 'downloading'
        task.start_time = datetime.now().isoformat()
        task.attempts += 1

        # Determine output filename
        filename = task.name or task.url.split('/')[-1]
        output_path = task.output_dir / filename

        try:
            # Check if file exists (resume)
            initial_pos = 0
            if resume and output_path.exists():
                initial_pos = output_path.stat().st_size
                logger.info(f"Resuming download from {initial_pos} bytes")

            # Prepare headers for resume
            headers = task.headers.copy()
            if initial_pos > 0:
                headers['Range'] = f'bytes={initial_pos}-'

            # Start download
            response = requests.get(
                task.url,
                headers=headers,
                stream=True,
                timeout=30
            )

            # Check status
            if response.status_code not in [200, 206]:
                raise Exception(f"HTTP {response.status_code}: {response.reason}")

            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            task.total_bytes = total_size + initial_pos

            # Download with progress bar
            mode = 'ab' if initial_pos > 0 else 'wb'
            with open(output_path, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=0,
                    unit='B',
                    unit_scale=True,
                    desc=task.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            task.bytes_downloaded += len(chunk)
                            pbar.update(len(chunk))

            task.file_path = output_path
            task.status = 'completed'
            task.end_time = datetime.now().isoformat()

            logger.info(f"Download completed: {task.name}")
            return True

        except Exception as e:
            task.error_message = str(e)
            logger.error(f"Download failed: {task.name} - {e}")

            # Retry logic
            if task.attempts < task.max_attempts:
                wait_time = self.retry_backoff ** task.attempts
                logger.info(f"Retrying in {wait_time:.1f}s (attempt {task.attempts}/{task.max_attempts})")
                time.sleep(wait_time)
                return self.download_file(task, resume=resume)
            else:
                task.status = 'failed'
                task.end_time = datetime.now().isoformat()
                return False

    def verify_checksum(
        self,
        task: DownloadTask
    ) -> bool:
        """
        Verify file checksum

        Args:
            task: Download task with checksum

        Returns:
            True if checksum matches
        """
        if not task.checksum or not task.file_path:
            return True

        logger.info(f"Verifying {task.checksum_type} checksum for {task.name}")

        try:
            # Calculate checksum
            hash_func = hashlib.new(task.checksum_type)

            with open(task.file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(self.chunk_size), b''):
                    hash_func.update(chunk)

            calculated = hash_func.hexdigest()

            if calculated.lower() == task.checksum.lower():
                logger.info(f"Checksum verified: {task.name}")
                return True
            else:
                logger.error(f"Checksum mismatch: {task.name}")
                logger.error(f"Expected: {task.checksum}")
                logger.error(f"Got: {calculated}")
                return False

        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False

    def process_queue(self):
        """Process download queue with parallel workers"""
        logger.info(f"Processing {len(self.queue)} tasks with {self.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {}
            for task in self.queue[:]:
                future = executor.submit(self._process_task, task)
                futures[future] = task

            # Process completions
            for future in as_completed(futures):
                task = futures[future]
                try:
                    success = future.result()
                    with self.lock:
                        self.queue.remove(task)
                        if success:
                            self.completed.append(task)
                        else:
                            self.failed.append(task)
                except Exception as e:
                    logger.error(f"Task execution failed: {task.name} - {e}")
                    with self.lock:
                        self.queue.remove(task)
                        task.status = 'failed'
                        task.error_message = str(e)
                        self.failed.append(task)

        logger.info(f"Queue processed: {len(self.completed)} completed, {len(self.failed)} failed")

    def _process_task(self, task: DownloadTask) -> bool:
        """Process a single task"""
        # Download
        success = self.download_file(task)

        if not success:
            return False

        # Verify checksum
        if task.checksum:
            if not self.verify_checksum(task):
                task.status = 'failed'
                task.error_message = 'Checksum verification failed'
                return False

        return True

    def save_history(self, output_path: Path):
        """Save download history to JSON"""
        history = {
            'generated_date': datetime.now().isoformat(),
            'completed': [t.to_dict() for t in self.completed],
            'failed': [t.to_dict() for t in self.failed],
            'pending': [t.to_dict() for t in self.queue]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved download history to {output_path}")

    def get_summary(self) -> Dict:
        """Get download summary statistics"""
        total_bytes = sum(t.bytes_downloaded for t in self.completed)

        return {
            'total_tasks': len(self.completed) + len(self.failed) + len(self.queue),
            'completed': len(self.completed),
            'failed': len(self.failed),
            'pending': len(self.queue),
            'total_bytes_downloaded': total_bytes,
            'total_bytes_human': f"{total_bytes / (1024**3):.2f} GB"
        }


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Download manager with parallel processing and retry logic'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/download_config.yaml',
        help='Configuration file'
    )

    parser.add_argument(
        '--parallel',
        type=int,
        default=5,
        help='Number of parallel downloads'
    )

    parser.add_argument(
        '--add-url',
        help='Add URL to queue'
    )

    parser.add_argument(
        '--name',
        help='Name for download'
    )

    parser.add_argument(
        '--priority',
        choices=['critical', 'high', 'normal', 'low'],
        default='normal',
        help='Download priority'
    )

    parser.add_argument(
        '--checksum',
        help='Expected checksum'
    )

    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        config = load_config(Path(args.config))
    else:
        logger.warning(f"Config file not found: {args.config}")
        config = {}

    # Initialize manager
    manager = DownloadManager(max_workers=args.parallel)

    # Add task if specified
    if args.add_url:
        task = DownloadTask(
            url=args.add_url,
            name=args.name or args.add_url.split('/')[-1],
            output_dir=Path('data/raw'),
            priority=args.priority,
            checksum=args.checksum
        )
        manager.add_task(task)

    # Process queue
    if manager.queue:
        manager.process_queue()

        # Save history
        manager.save_history(Path('logs/download_history.json'))

        # Print summary
        summary = manager.get_summary()
        print("\n=== Download Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
    else:
        print("No tasks in queue")


if __name__ == '__main__':
    main()