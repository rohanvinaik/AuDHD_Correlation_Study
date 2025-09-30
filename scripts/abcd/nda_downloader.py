#!/usr/bin/env python3
"""
NDA (NIMH Data Archive) Downloader for ABCD Study

This script automates downloads from the NDA using their Python client and API.
Specifically designed for ABCD Study data packages relevant to ADHD/Autism research.

Requirements:
    pip install nda-tools pandas requests tqdm pyyaml

NDA Access Prerequisites:
    1. NDA account at https://nda.nih.gov/
    2. ABCD study access approved (submit Data Use Certification)
    3. NDA tools configured: nda-tools configure

Usage:
    # Configure NDA credentials (one-time)
    python nda_downloader.py --configure

    # List available ABCD packages
    python nda_downloader.py --list-packages

    # Download specific packages
    python nda_downloader.py --packages abcd_cbcls01,abcd_ksad01 --output data/abcd/

    # Download all ADHD/autism-relevant packages
    python nda_downloader.py --adhd-autism-packages --output data/abcd/

    # Resume interrupted download
    python nda_downloader.py --resume --output data/abcd/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set
import time
import logging

try:
    import pandas as pd
    import requests
    from tqdm import tqdm
    import yaml
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install pandas requests tqdm pyyaml")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ABCD Data Packages for ADHD/Autism Research
ABCD_ADHD_AUTISM_PACKAGES = {
    # Clinical Assessments
    'abcd_cbcls01': {
        'name': 'Child Behavior Checklist (CBCL)',
        'description': 'Parent-reported behavioral/emotional problems',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'clinical',
        'size_mb': 5
    },
    'abcd_ksad01': {
        'name': 'Kiddie Schedule for Affective Disorders (KSADS)',
        'description': 'Psychiatric diagnoses including ADHD/ASD',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'clinical',
        'size_mb': 15
    },
    'abcd_medhy01': {
        'name': 'Medication History',
        'description': 'Current and past medications',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'clinical',
        'size_mb': 3
    },
    'abcd_sscey01': {
        'name': 'Social Communication Questionnaire (SCQ)',
        'description': 'Autism screening questionnaire',
        'adhd_relevant': False,
        'autism_relevant': True,
        'category': 'clinical',
        'size_mb': 2
    },

    # Neuroimaging
    'abcd_betnet02': {
        'name': 'Brain Connectivity (Resting-State fMRI)',
        'description': 'Network connectivity matrices',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'neuroimaging',
        'size_mb': 200
    },
    'abcd_mrfindings01': {
        'name': 'MRI Findings',
        'description': 'Incidental findings from structural MRI',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'neuroimaging',
        'size_mb': 1
    },
    'abcd_smrip10201': {
        'name': 'Structural MRI (FreeSurfer)',
        'description': 'Cortical thickness, surface area, volumes',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'neuroimaging',
        'size_mb': 50
    },
    'abcd_dmdtifp101': {
        'name': 'Diffusion MRI (DTI)',
        'description': 'White matter microstructure (FA, MD)',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'neuroimaging',
        'size_mb': 30
    },

    # Biospecimens / Metabolomics
    'abcd_biospec01': {
        'name': 'Biospecimen Inventory',
        'description': 'Available biological samples',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'biospecimen',
        'size_mb': 1
    },
    'abcd_hsss01': {
        'name': 'Saliva Hormone Samples',
        'description': 'DHEA, testosterone (subset)',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'biospecimen',
        'size_mb': 2
    },

    # Sleep and Circadian
    'abcd_sds01': {
        'name': 'Sleep Disturbance Scale',
        'description': 'Sleep problems and disorders',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'sleep',
        'size_mb': 2
    },
    'abcd_midacsss01': {
        'name': 'Munich Chronotype (Sleep timing)',
        'description': 'Circadian preference/sleep schedule',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'sleep',
        'size_mb': 1
    },

    # Diet and Metabolism
    'abcd_eatqp01': {
        'name': 'Eating Questionnaire',
        'description': 'Dietary patterns and eating behaviors',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'diet',
        'size_mb': 3
    },
    'abcd_ssbpm01': {
        'name': 'Sugar-Sweetened Beverages',
        'description': 'Beverage consumption patterns',
        'adhd_relevant': True,
        'autism_relevant': False,
        'category': 'diet',
        'size_mb': 1
    },

    # Cognitive/Executive Function
    'abcd_tbss01': {
        'name': 'NIH Toolbox Cognition',
        'description': 'Executive function, working memory, attention',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'cognitive',
        'size_mb': 10
    },

    # Environmental Exposures
    'abcd_airsleep01': {
        'name': 'Air Quality and Sleep Environment',
        'description': 'Environmental exposures at residence',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'environmental',
        'size_mb': 2
    },
    'abcd_rhds01': {
        'name': 'Residential History',
        'description': 'Address history for environmental linkage',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'environmental',
        'size_mb': 1
    },

    # Family History and Genetics
    'abcd_fhxssp01': {
        'name': 'Family History Summary Scores',
        'description': 'Parental psychiatric history',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'family',
        'size_mb': 2
    },
    'abcd_lpds01': {
        'name': 'Longitudinal Parent Demographics',
        'description': 'Parent education, income, employment',
        'adhd_relevant': True,
        'autism_relevant': True,
        'category': 'demographics',
        'size_mb': 3
    }
}


@dataclass
class NDADownloadJob:
    """Represents a single NDA package download job"""
    package_id: str
    package_name: str
    category: str
    size_mb: int
    output_path: Path
    status: str = 'pending'  # pending, downloading, completed, failed
    download_time: Optional[float] = None
    error_message: Optional[str] = None
    file_count: int = 0
    downloaded_files: List[str] = field(default_factory=list)


class NDADownloader:
    """ABCD Study data downloader using NDA API"""

    def __init__(self, output_dir: Path, config_file: Optional[Path] = None):
        """
        Initialize NDA downloader

        Args:
            output_dir: Output directory for downloaded data
            config_file: Optional config file for credentials
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress_file = self.output_dir / '.nda_download_progress.json'
        self.jobs: Dict[str, NDADownloadJob] = {}
        self.load_progress()

        # NDA configuration
        self.nda_config_dir = Path.home() / '.NDATools'
        self.config_file = config_file or self.nda_config_dir / 'settings.cfg'

        logger.info(f"Initialized NDA downloader: {output_dir}")

    def check_nda_tools(self) -> bool:
        """Check if NDA tools are installed and configured"""
        try:
            result = subprocess.run(
                ['downloadcmd', '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.error("NDA tools not properly installed")
                return False

            # Check configuration
            if not self.config_file.exists():
                logger.warning(f"NDA configuration not found: {self.config_file}")
                logger.warning("Run: python nda_downloader.py --configure")
                return False

            logger.info("NDA tools properly configured")
            return True

        except FileNotFoundError:
            logger.error("NDA tools not installed. Install with: pip install nda-tools")
            return False
        except subprocess.TimeoutExpired:
            logger.error("NDA tools check timed out")
            return False

    def configure_nda(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Configure NDA credentials

        Args:
            username: NDA username (will prompt if not provided)
            password: NDA password (will prompt if not provided)
        """
        try:
            if username and password:
                # Use provided credentials
                cmd = ['downloadcmd', '-u', username, '-p', password, '-l']
            else:
                # Interactive configuration
                logger.info("Starting interactive NDA configuration...")
                subprocess.run(['downloadcmd', '-c'], check=True)
                return

            # Test credentials
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info("NDA credentials configured successfully")
            else:
                logger.error(f"Failed to configure NDA credentials: {result.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error configuring NDA: {e}")
        except subprocess.TimeoutExpired:
            logger.error("NDA configuration timed out")

    def list_available_packages(self, category: Optional[str] = None) -> pd.DataFrame:
        """
        List available ABCD packages

        Args:
            category: Filter by category (clinical, neuroimaging, biospecimen, etc.)

        Returns:
            DataFrame with package information
        """
        packages = []
        for pkg_id, info in ABCD_ADHD_AUTISM_PACKAGES.items():
            if category and info['category'] != category:
                continue

            packages.append({
                'package_id': pkg_id,
                'name': info['name'],
                'description': info['description'],
                'category': info['category'],
                'adhd_relevant': info['adhd_relevant'],
                'autism_relevant': info['autism_relevant'],
                'size_mb': info['size_mb']
            })

        df = pd.DataFrame(packages)
        return df.sort_values(['category', 'package_id'])

    def create_download_manifest(self, package_ids: List[str]) -> Path:
        """
        Create NDA download manifest (package list file)

        Args:
            package_ids: List of package IDs to download

        Returns:
            Path to manifest file
        """
        manifest_file = self.output_dir / 'nda_download_manifest.txt'

        # NDA downloadcmd expects one package per line
        with open(manifest_file, 'w') as f:
            for pkg_id in package_ids:
                f.write(f"{pkg_id}\n")

        logger.info(f"Created download manifest: {manifest_file}")
        logger.info(f"Packages to download: {len(package_ids)}")

        return manifest_file

    def download_package(self, package_id: str, use_s3: bool = True) -> bool:
        """
        Download a single ABCD package from NDA

        Args:
            package_id: Package ID (e.g., 'abcd_cbcls01')
            use_s3: Use S3 links if available (faster)

        Returns:
            True if successful, False otherwise
        """
        if package_id not in ABCD_ADHD_AUTISM_PACKAGES:
            logger.error(f"Unknown package: {package_id}")
            return False

        pkg_info = ABCD_ADHD_AUTISM_PACKAGES[package_id]

        # Create job
        job = NDADownloadJob(
            package_id=package_id,
            package_name=pkg_info['name'],
            category=pkg_info['category'],
            size_mb=pkg_info['size_mb'],
            output_path=self.output_dir / package_id
        )

        # Check if already downloaded
        if job.output_path.exists() and any(job.output_path.iterdir()):
            logger.info(f"Package already exists: {package_id}")
            job.status = 'completed'
            self.jobs[package_id] = job
            self.save_progress()
            return True

        job.output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading package: {package_id} ({pkg_info['name']})")
        logger.info(f"Estimated size: {pkg_info['size_mb']} MB")

        job.status = 'downloading'
        self.jobs[package_id] = job
        self.save_progress()

        start_time = time.time()

        try:
            # Use NDA downloadcmd
            cmd = [
                'downloadcmd',
                '-dp', package_id,
                '-d', str(job.output_path),
                '-t', 'datastructure',
                '-wt', '600'  # 10-minute timeout per file
            ]

            if use_s3:
                cmd.append('-s')  # Use S3 links

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2-hour timeout for entire package
            )

            if result.returncode == 0:
                # Count downloaded files
                files = list(job.output_path.glob('*.txt')) + list(job.output_path.glob('*.csv'))
                job.file_count = len(files)
                job.downloaded_files = [f.name for f in files]
                job.status = 'completed'
                job.download_time = time.time() - start_time

                logger.info(f"Successfully downloaded {package_id} ({job.file_count} files)")
                logger.info(f"Time: {job.download_time:.1f}s")

                self.jobs[package_id] = job
                self.save_progress()
                return True
            else:
                job.status = 'failed'
                job.error_message = result.stderr
                logger.error(f"Failed to download {package_id}: {result.stderr}")

                self.jobs[package_id] = job
                self.save_progress()
                return False

        except subprocess.TimeoutExpired:
            job.status = 'failed'
            job.error_message = 'Download timeout (2 hours)'
            logger.error(f"Download timeout for {package_id}")

            self.jobs[package_id] = job
            self.save_progress()
            return False

        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            logger.error(f"Error downloading {package_id}: {e}")

            self.jobs[package_id] = job
            self.save_progress()
            return False

    def download_packages(self, package_ids: List[str], use_s3: bool = True) -> Dict[str, bool]:
        """
        Download multiple packages

        Args:
            package_ids: List of package IDs
            use_s3: Use S3 links if available

        Returns:
            Dict mapping package_id -> success status
        """
        results = {}

        logger.info(f"Downloading {len(package_ids)} packages...")

        for pkg_id in tqdm(package_ids, desc="Downloading packages"):
            success = self.download_package(pkg_id, use_s3=use_s3)
            results[pkg_id] = success

        # Summary
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Download complete: {successful}/{len(package_ids)} successful")

        return results

    def download_adhd_autism_packages(self, adhd_only: bool = False,
                                     autism_only: bool = False,
                                     use_s3: bool = True) -> Dict[str, bool]:
        """
        Download all ADHD/Autism relevant packages

        Args:
            adhd_only: Download only ADHD-relevant packages
            autism_only: Download only autism-relevant packages
            use_s3: Use S3 links if available

        Returns:
            Dict mapping package_id -> success status
        """
        package_ids = []

        for pkg_id, info in ABCD_ADHD_AUTISM_PACKAGES.items():
            if adhd_only and not info['adhd_relevant']:
                continue
            if autism_only and not info['autism_relevant']:
                continue
            package_ids.append(pkg_id)

        logger.info(f"Selected {len(package_ids)} ADHD/Autism-relevant packages")

        return self.download_packages(package_ids, use_s3=use_s3)

    def resume_downloads(self) -> Dict[str, bool]:
        """
        Resume any failed or incomplete downloads

        Returns:
            Dict mapping package_id -> success status
        """
        incomplete = [
            pkg_id for pkg_id, job in self.jobs.items()
            if job.status in ['pending', 'failed', 'downloading']
        ]

        if not incomplete:
            logger.info("No incomplete downloads to resume")
            return {}

        logger.info(f"Resuming {len(incomplete)} incomplete downloads...")
        return self.download_packages(incomplete)

    def generate_download_summary(self) -> pd.DataFrame:
        """
        Generate summary of download status

        Returns:
            DataFrame with download statistics
        """
        if not self.jobs:
            return pd.DataFrame()

        summary = []
        for pkg_id, job in self.jobs.items():
            summary.append({
                'package_id': pkg_id,
                'package_name': job.package_name,
                'category': job.category,
                'status': job.status,
                'file_count': job.file_count,
                'download_time_sec': job.download_time,
                'error': job.error_message
            })

        df = pd.DataFrame(summary)
        return df.sort_values(['status', 'category'])

    def save_progress(self):
        """Save download progress to JSON"""
        progress = {
            pkg_id: asdict(job)
            for pkg_id, job in self.jobs.items()
        }

        # Convert Path objects to strings
        for job_data in progress.values():
            if 'output_path' in job_data:
                job_data['output_path'] = str(job_data['output_path'])

        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def load_progress(self):
        """Load download progress from JSON"""
        if not self.progress_file.exists():
            return

        try:
            with open(self.progress_file) as f:
                progress = json.load(f)

            self.jobs = {}
            for pkg_id, job_data in progress.items():
                # Convert string back to Path
                if 'output_path' in job_data:
                    job_data['output_path'] = Path(job_data['output_path'])

                self.jobs[pkg_id] = NDADownloadJob(**job_data)

            logger.info(f"Loaded progress: {len(self.jobs)} jobs")

        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Download ABCD Study data from NIMH Data Archive (NDA)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Configure NDA credentials (first time)
  python nda_downloader.py --configure

  # List available packages
  python nda_downloader.py --list-packages

  # List packages by category
  python nda_downloader.py --list-packages --category clinical

  # Download specific packages
  python nda_downloader.py --packages abcd_cbcls01,abcd_ksad01 --output data/abcd/

  # Download all ADHD/Autism-relevant packages
  python nda_downloader.py --adhd-autism-packages --output data/abcd/

  # Download only ADHD-relevant packages
  python nda_downloader.py --adhd-only --output data/abcd/

  # Resume interrupted downloads
  python nda_downloader.py --resume --output data/abcd/

  # Check download status
  python nda_downloader.py --status --output data/abcd/

Prerequisites:
  1. Install NDA tools: pip install nda-tools
  2. Create NDA account: https://nda.nih.gov/
  3. Request ABCD study access (submit Data Use Certification)
  4. Configure credentials: python nda_downloader.py --configure
        """
    )

    parser.add_argument(
        '--configure',
        action='store_true',
        help='Configure NDA credentials'
    )

    parser.add_argument(
        '--list-packages',
        action='store_true',
        help='List available ABCD packages'
    )

    parser.add_argument(
        '--category',
        choices=['clinical', 'neuroimaging', 'biospecimen', 'sleep', 'diet',
                'cognitive', 'environmental', 'family', 'demographics'],
        help='Filter packages by category'
    )

    parser.add_argument(
        '--packages',
        type=str,
        help='Comma-separated list of package IDs to download'
    )

    parser.add_argument(
        '--adhd-autism-packages',
        action='store_true',
        help='Download all ADHD/Autism-relevant packages'
    )

    parser.add_argument(
        '--adhd-only',
        action='store_true',
        help='Download only ADHD-relevant packages'
    )

    parser.add_argument(
        '--autism-only',
        action='store_true',
        help='Download only autism-relevant packages'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume incomplete downloads'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show download status'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/abcd',
        help='Output directory (default: data/abcd)'
    )

    parser.add_argument(
        '--no-s3',
        action='store_true',
        help='Disable S3 links (slower but more reliable)'
    )

    args = parser.parse_args()

    # Handle configuration
    if args.configure:
        downloader = NDADownloader(Path(args.output))
        downloader.configure_nda()
        return

    # Handle list packages
    if args.list_packages:
        downloader = NDADownloader(Path(args.output))
        df = downloader.list_available_packages(category=args.category)

        print("\n=== Available ABCD Packages ===\n")
        print(df.to_string(index=False))
        print(f"\nTotal packages: {len(df)}")
        print(f"Total size: {df['size_mb'].sum():.0f} MB")
        return

    # Initialize downloader
    downloader = NDADownloader(Path(args.output))

    # Check NDA tools
    if not downloader.check_nda_tools():
        print("\nError: NDA tools not properly configured")
        print("Run: python nda_downloader.py --configure")
        sys.exit(1)

    # Handle status
    if args.status:
        summary = downloader.generate_download_summary()
        if summary.empty:
            print("\nNo downloads yet")
        else:
            print("\n=== Download Status ===\n")
            print(summary.to_string(index=False))

            # Statistics
            completed = len(summary[summary['status'] == 'completed'])
            failed = len(summary[summary['status'] == 'failed'])
            pending = len(summary[summary['status'] == 'pending'])

            print(f"\nCompleted: {completed}")
            print(f"Failed: {failed}")
            print(f"Pending: {pending}")
        return

    # Handle resume
    if args.resume:
        results = downloader.resume_downloads()
        summary = downloader.generate_download_summary()
        print("\n=== Download Summary ===\n")
        print(summary.to_string(index=False))
        return

    # Handle downloads
    use_s3 = not args.no_s3

    if args.packages:
        package_ids = [p.strip() for p in args.packages.split(',')]
        results = downloader.download_packages(package_ids, use_s3=use_s3)
    elif args.adhd_autism_packages:
        results = downloader.download_adhd_autism_packages(use_s3=use_s3)
    elif args.adhd_only:
        results = downloader.download_adhd_autism_packages(adhd_only=True, use_s3=use_s3)
    elif args.autism_only:
        results = downloader.download_adhd_autism_packages(autism_only=True, use_s3=use_s3)
    else:
        parser.print_help()
        return

    # Print summary
    summary = downloader.generate_download_summary()
    print("\n=== Download Summary ===\n")
    print(summary.to_string(index=False))

    # Save summary
    summary_file = downloader.output_dir / 'download_summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()