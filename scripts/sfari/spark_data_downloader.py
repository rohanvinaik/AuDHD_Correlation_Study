#!/usr/bin/env python3
"""
SPARK Data Downloader

Robust downloader for SPARK dataset files with:
- Resume capability for interrupted downloads
- Checksum verification (MD5/SHA256)
- Parallel downloads for multiple files
- Progress tracking and logging
- AWS S3 integration for iSEC data

Usage:
    # Download from manifest
    python spark_data_downloader.py --manifest spark_manifest.csv --output data/raw/spark/

    # Download specific file types
    python spark_data_downloader.py --manifest spark_manifest.csv --types genomic,phenotype

    # Resume interrupted download
    python spark_data_downloader.py --manifest spark_manifest.csv --resume

    # Download from AWS S3 (iSEC)
    python spark_data_downloader.py --s3 --bucket spark-isec --prefix genomics/
"""

import argparse
import hashlib
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import urlparse

try:
    import pandas as pd
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required packages not installed")
    print("Install with: pip install pandas tqdm")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spark_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DownloadJob:
    """Represents a file download job"""
    url: str
    output_path: Path
    file_type: str
    size_bytes: int = 0
    checksum: Optional[str] = None
    checksum_type: str = "md5"
    priority: int = 0
    attempts: int = 0
    max_attempts: int = 3
    status: str = "pending"  # pending, downloading, completed, failed, verified
    error: Optional[str] = None


class SPARKDataDownloader:
    """Download SPARK dataset files with resume and verification"""

    def __init__(
        self,
        output_dir: Path,
        max_workers: int = 4,
        chunk_size: int = 8192,
        verify_checksums: bool = True,
        use_aria2c: bool = True
    ):
        """
        Initialize SPARK data downloader

        Args:
            output_dir: Base output directory
            max_workers: Number of parallel download threads
            chunk_size: Download chunk size in bytes
            verify_checksums: Verify file checksums after download
            use_aria2c: Use aria2c for faster downloads (if available)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.verify_checksums = verify_checksums
        self.use_aria2c = use_aria2c and self._check_aria2c()

        self.progress_file = self.output_dir / ".download_progress.json"
        self.jobs: List[DownloadJob] = []

    def _check_aria2c(self) -> bool:
        """Check if aria2c is available"""
        try:
            subprocess.run(
                ["aria2c", "--version"],
                capture_output=True,
                check=True
            )
            logger.info("✓ aria2c available for faster downloads")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info("aria2c not found, using standard download")
            logger.info("Install aria2c for faster downloads: brew install aria2")
            return False

    def load_manifest(self, manifest_path: Path, file_types: Optional[List[str]] = None) -> int:
        """
        Load download manifest from CSV

        Args:
            manifest_path: Path to manifest CSV file
            file_types: Optional list of file types to download

        Returns:
            Number of jobs loaded
        """
        logger.info(f"Loading manifest: {manifest_path}")

        try:
            df = pd.read_csv(manifest_path)

            required_cols = ["name", "download_url", "type"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Manifest must contain columns: {required_cols}")

            # Filter by file type if specified
            if file_types:
                df = df[df["type"].isin(file_types)]

            # Create download jobs
            for _, row in df.iterrows():
                # Parse filename from URL or use provided name
                filename = row["name"]
                output_path = self.output_dir / row["type"] / filename

                job = DownloadJob(
                    url=row["download_url"],
                    output_path=output_path,
                    file_type=row["type"],
                    size_bytes=int(row.get("size_bytes", 0)) if pd.notna(row.get("size_bytes")) else 0,
                    checksum=row.get("checksum") if pd.notna(row.get("checksum")) else None,
                    checksum_type=row.get("checksum_type", "md5")
                )

                self.jobs.append(job)

            logger.info(f"Loaded {len(self.jobs)} download jobs")
            return len(self.jobs)

        except Exception as e:
            logger.error(f"Error loading manifest: {e}")
            return 0

    def resume_downloads(self) -> int:
        """
        Resume incomplete downloads from progress file

        Returns:
            Number of jobs resumed
        """
        if not self.progress_file.exists():
            logger.info("No previous download progress found")
            return 0

        logger.info("Resuming incomplete downloads...")

        try:
            import json
            with open(self.progress_file) as f:
                progress_data = json.load(f)

            resumed = 0
            for job_data in progress_data.get("jobs", []):
                if job_data["status"] not in ["completed", "verified"]:
                    job = DownloadJob(**job_data)
                    self.jobs.append(job)
                    resumed += 1

            logger.info(f"Resumed {resumed} incomplete downloads")
            return resumed

        except Exception as e:
            logger.error(f"Error resuming downloads: {e}")
            return 0

    def _save_progress(self):
        """Save download progress to file"""
        try:
            import json
            progress_data = {
                "timestamp": time.time(),
                "jobs": [job.__dict__ for job in self.jobs]
            }

            # Convert Path objects to strings
            for job in progress_data["jobs"]:
                job["output_path"] = str(job["output_path"])

            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not save progress: {e}")

    def _download_file_aria2c(self, job: DownloadJob) -> bool:
        """Download file using aria2c (faster, resumable)"""
        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "aria2c",
            "--console-log-level=warn",
            "--file-allocation=none",
            "--continue=true",  # Resume capability
            "--max-tries=5",
            "--retry-wait=3",
            "--max-connection-per-server=8",
            "--split=8",  # Multi-connection download
            "--min-split-size=1M",
            "--dir", str(job.output_path.parent),
            "--out", job.output_path.name,
            job.url
        ]

        try:
            logger.info(f"Downloading (aria2c): {job.output_path.name}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                job.status = "completed"
                return True
            else:
                job.error = result.stderr
                logger.error(f"aria2c download failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            job.error = "Download timeout"
            logger.error(f"Download timeout: {job.url}")
            return False
        except Exception as e:
            job.error = str(e)
            logger.error(f"Download error: {e}")
            return False

    def _download_file_urllib(self, job: DownloadJob) -> bool:
        """Download file using urllib (fallback)"""
        import urllib.request

        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file partially downloaded
        resume_byte = 0
        if job.output_path.exists():
            resume_byte = job.output_path.stat().st_size
            logger.info(f"Resuming from byte {resume_byte}")

        try:
            # Setup request with resume header
            req = urllib.request.Request(job.url)
            if resume_byte > 0:
                req.add_header("Range", f"bytes={resume_byte}-")

            mode = "ab" if resume_byte > 0 else "wb"

            with urllib.request.urlopen(req, timeout=300) as response:
                total_size = int(response.getheader('Content-Length', 0))

                with open(job.output_path, mode) as f:
                    with tqdm(
                        total=total_size,
                        initial=resume_byte,
                        unit='B',
                        unit_scale=True,
                        desc=job.output_path.name
                    ) as pbar:
                        while True:
                            chunk = response.read(self.chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))

            job.status = "completed"
            return True

        except Exception as e:
            job.error = str(e)
            logger.error(f"Download error: {e}")
            return False

    def _verify_checksum(self, job: DownloadJob) -> bool:
        """Verify file checksum"""
        if not job.checksum:
            logger.debug(f"No checksum provided for {job.output_path.name}")
            return True

        if not job.output_path.exists():
            return False

        logger.info(f"Verifying checksum: {job.output_path.name}")

        try:
            # Calculate file checksum
            if job.checksum_type.lower() == "md5":
                hasher = hashlib.md5()
            elif job.checksum_type.lower() == "sha256":
                hasher = hashlib.sha256()
            else:
                logger.warning(f"Unsupported checksum type: {job.checksum_type}")
                return True

            with open(job.output_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)

            calculated = hasher.hexdigest()

            if calculated.lower() == job.checksum.lower():
                logger.info(f"✓ Checksum verified: {job.output_path.name}")
                job.status = "verified"
                return True
            else:
                logger.error(f"✗ Checksum mismatch: {job.output_path.name}")
                logger.error(f"  Expected: {job.checksum}")
                logger.error(f"  Got: {calculated}")
                job.status = "failed"
                job.error = "Checksum mismatch"
                return False

        except Exception as e:
            logger.error(f"Checksum verification error: {e}")
            return False

    def _download_job(self, job: DownloadJob) -> DownloadJob:
        """Download and verify a single file"""
        # Skip if already completed
        if job.output_path.exists() and job.status == "verified":
            logger.info(f"Skipping (already downloaded): {job.output_path.name}")
            return job

        # Retry logic
        while job.attempts < job.max_attempts:
            job.attempts += 1
            job.status = "downloading"

            logger.info(f"Attempt {job.attempts}/{job.max_attempts}: {job.url}")

            # Use aria2c if available, otherwise urllib
            if self.use_aria2c:
                success = self._download_file_aria2c(job)
            else:
                success = self._download_file_urllib(job)

            if success:
                # Verify checksum
                if self.verify_checksums:
                    if self._verify_checksum(job):
                        logger.info(f"✓ Download complete: {job.output_path.name}")
                        return job
                else:
                    job.status = "completed"
                    logger.info(f"✓ Download complete (no verification): {job.output_path.name}")
                    return job

            # Wait before retry
            if job.attempts < job.max_attempts:
                wait_time = 2 ** job.attempts  # Exponential backoff
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        # All attempts failed
        job.status = "failed"
        logger.error(f"✗ Download failed after {job.max_attempts} attempts: {job.url}")
        return job

    def download_all(self) -> Dict[str, int]:
        """
        Download all files in parallel

        Returns:
            Dictionary with download statistics
        """
        if not self.jobs:
            logger.warning("No download jobs found")
            return {"total": 0, "completed": 0, "failed": 0}

        logger.info(f"Starting download of {len(self.jobs)} files...")
        logger.info(f"Max workers: {self.max_workers}")
        logger.info(f"Output directory: {self.output_dir}")

        stats = {"total": len(self.jobs), "completed": 0, "failed": 0}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._download_job, job): job
                for job in self.jobs
            }

            # Process completed downloads
            for future in tqdm(
                as_completed(future_to_job),
                total=len(self.jobs),
                desc="Overall progress"
            ):
                job = future.result()

                if job.status in ["completed", "verified"]:
                    stats["completed"] += 1
                else:
                    stats["failed"] += 1

                # Save progress periodically
                self._save_progress()

        logger.info("\n" + "="*50)
        logger.info("Download Summary")
        logger.info("="*50)
        logger.info(f"Total files: {stats['total']}")
        logger.info(f"Completed: {stats['completed']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info("="*50)

        return stats

    def download_s3_bucket(
        self,
        bucket: str,
        prefix: str = "",
        aws_profile: Optional[str] = None
    ) -> bool:
        """
        Download files from AWS S3 bucket (for iSEC data)

        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix
            aws_profile: AWS profile name (optional)

        Returns:
            True if successful
        """
        logger.info(f"Downloading from S3: s3://{bucket}/{prefix}")

        # Check if AWS CLI available
        try:
            subprocess.run(["aws", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("AWS CLI not found. Install with: pip install awscli")
            return False

        # Build AWS command
        cmd = ["aws", "s3", "sync", f"s3://{bucket}/{prefix}", str(self.output_dir)]

        if aws_profile:
            cmd.extend(["--profile", aws_profile])

        # Add options for robust download
        cmd.extend([
            "--no-progress",  # Disable progress (we'll use our own)
            "--region", "us-east-1",  # Default region
        ])

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            if result.returncode == 0:
                logger.info("✓ S3 sync complete")
                return True
            else:
                logger.error(f"S3 sync failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("S3 sync timeout")
            return False
        except Exception as e:
            logger.error(f"S3 sync error: {e}")
            return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="SPARK Data Downloader with resume capability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from manifest
  python spark_data_downloader.py --manifest spark_manifest.csv --output data/raw/spark/

  # Download specific file types
  python spark_data_downloader.py --manifest spark_manifest.csv --types genomic,phenotype

  # Resume interrupted download
  python spark_data_downloader.py --resume --output data/raw/spark/

  # Download from AWS S3 (iSEC)
  python spark_data_downloader.py --s3 --bucket spark-isec --prefix genomics/ --output data/raw/spark/

  # Parallel downloads (faster)
  python spark_data_downloader.py --manifest spark_manifest.csv --workers 8
        """
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to download manifest CSV"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--types",
        help="Comma-separated list of file types to download (e.g., 'genomic,phenotype')"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download threads (default: 4)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete downloads"
    )

    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip checksum verification"
    )

    parser.add_argument(
        "--no-aria2c",
        action="store_true",
        help="Don't use aria2c even if available"
    )

    # S3 options
    parser.add_argument(
        "--s3",
        action="store_true",
        help="Download from AWS S3 bucket"
    )

    parser.add_argument(
        "--bucket",
        help="S3 bucket name"
    )

    parser.add_argument(
        "--prefix",
        default="",
        help="S3 key prefix"
    )

    parser.add_argument(
        "--aws-profile",
        help="AWS profile name"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = SPARKDataDownloader(
        output_dir=args.output,
        max_workers=args.workers,
        verify_checksums=not args.no_verify,
        use_aria2c=not args.no_aria2c
    )

    # S3 download
    if args.s3:
        if not args.bucket:
            parser.error("--bucket required for S3 download")

        success = downloader.download_s3_bucket(
            bucket=args.bucket,
            prefix=args.prefix,
            aws_profile=args.aws_profile
        )

        sys.exit(0 if success else 1)

    # Resume previous downloads
    if args.resume:
        downloader.resume_downloads()

    # Load manifest
    if args.manifest:
        file_types = args.types.split(",") if args.types else None
        n_jobs = downloader.load_manifest(args.manifest, file_types)

        if n_jobs == 0:
            logger.error("No download jobs loaded")
            sys.exit(1)

    # Start downloads
    stats = downloader.download_all()

    # Exit with error if any downloads failed
    sys.exit(0 if stats["failed"] == 0 else 1)


if __name__ == "__main__":
    main()