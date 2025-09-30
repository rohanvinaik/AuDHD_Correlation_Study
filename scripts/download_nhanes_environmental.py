#!/usr/bin/env python3
"""
Download NHANES Environmental Biomarker Data
Targets: Heavy metals, pesticides, phthalates, POPs
"""

import requests
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import time
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NHANESDownloader:
    """Download NHANES environmental exposure data"""

    def __init__(self, output_dir='data/raw/nhanes'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = "https://wwwn.cdc.gov/Nchs/Nhanes"

        # Environmental biomarker datasets by cycle
        self.datasets = {
            'heavy_metals': {
                'Lead': [
                    ('2017-2018', 'PbCd_J'),  # Lead and Cadmium
                    ('2015-2016', 'PbCd_I'),
                    ('2013-2014', 'PbCd_H'),
                    ('2011-2012', 'PbCd_G'),
                ],
                'Mercury': [
                    ('2017-2018', 'HDL_J'),  # Mercury in blood
                    ('2015-2016', 'HDL_I'),
                ],
                'Cadmium': [  # Also in PbCd files
                    ('2017-2018', 'PbCd_J'),
                    ('2015-2016', 'PbCd_I'),
                ],
                'Manganese': [
                    ('2017-2018', 'PBCD_J'),
                    ('2015-2016', 'PBCD_I'),
                ]
            },
            'pesticides': {
                'Organophosphates': [
                    ('2017-2018', 'OPD_J'),
                    ('2015-2016', 'OPD_I'),
                    ('2013-2014', 'OPD_H'),
                ],
                'Pyrethroids': [
                    ('2015-2016', 'UPHOPM_I'),
                    ('2013-2014', 'UPHOPM_H'),
                ],
                'Herbicides': [
                    ('2013-2014', 'UHG_H'),
                    ('2011-2012', 'UHG_G'),
                ]
            },
            'phthalates_plasticizers': {
                'Phthalates': [
                    ('2017-2018', 'PHTHTE_J'),
                    ('2015-2016', 'PHTHTE_I'),
                    ('2013-2014', 'PHTHTE_H'),
                    ('2011-2012', 'PHTHTE_G'),
                ],
                'BPA_BPS': [
                    ('2015-2016', 'EPH_I'),
                    ('2013-2014', 'EPH_H'),
                    ('2011-2012', 'EPH_G'),
                ]
            },
            'persistent_pollutants': {
                'PCBs': [
                    ('2015-2016', 'PCBPOL_I'),
                    ('2013-2014', 'PCBPOL_H'),
                    ('2011-2012', 'PCBPOL_G'),
                ],
                'PBDEs': [
                    ('2015-2016', 'BFRPOL_I'),
                    ('2013-2014', 'BFRPOL_H'),
                ],
                'Dioxins': [
                    ('2013-2014', 'DOXPOL_H'),
                    ('2011-2012', 'DOXPOL_G'),
                ]
            }
        }

    def download_dataset(self, cycle: str, dataset_code: str, category: str) -> Path:
        """Download a single NHANES dataset"""
        # NHANES uses cycle format like "2017-2018" -> "2017-2018"
        cycle_formatted = cycle.replace('-', '_')

        # Build URL - NHANES provides both XPT (SAS) and CSV formats
        # We'll download XPT (more reliable) and convert
        url = f"{self.base_url}/{cycle}/{dataset_code}.XPT"

        # Create category directory
        category_dir = self.output_dir / category
        category_dir.mkdir(exist_ok=True, parents=True)

        output_file = category_dir / f"{dataset_code}_{cycle_formatted}.xpt"

        logger.info(f"Downloading {dataset_code} ({cycle})...")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            with open(output_file, 'wb') as f:
                f.write(response.content)

            size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Downloaded {dataset_code} ({size_mb:.1f} MB)")

            # Convert XPT to CSV for easier analysis
            try:
                df = pd.read_sas(output_file)
                csv_file = output_file.with_suffix('.csv')
                df.to_csv(csv_file, index=False)
                logger.info(f"  Converted to CSV ({len(df)} records)")
                return csv_file
            except Exception as e:
                logger.warning(f"  Could not convert to CSV: {e}")
                return output_file

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Dataset {dataset_code} not found for {cycle}")
            else:
                logger.error(f"Failed to download {dataset_code}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {dataset_code}: {e}")
            return None

    def download_demographics(self, cycles: List[str]):
        """Download demographics files (needed for linkage)"""
        logger.info("Downloading demographics data...")

        demo_dir = self.output_dir / 'demographics'
        demo_dir.mkdir(exist_ok=True, parents=True)

        for cycle in cycles:
            # Demographics file code varies by cycle
            if cycle >= '2017-2018':
                demo_code = 'DEMO_' + chr(ord('J') + (int(cycle[:4]) - 2017) // 2)
            else:
                # Map cycles to letter codes
                cycle_map = {
                    '2015-2016': 'DEMO_I',
                    '2013-2014': 'DEMO_H',
                    '2011-2012': 'DEMO_G',
                }
                demo_code = cycle_map.get(cycle, 'DEMO_J')

            self.download_dataset(cycle, demo_code, 'demographics')
            time.sleep(1)

    def download_all_environmental(self):
        """Download all environmental biomarker datasets"""
        logger.info("Starting NHANES environmental biomarker download...")

        all_cycles = set()
        downloaded_files = []

        # Download each category
        for category, compounds in self.datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Category: {category.upper().replace('_', ' ')}")
            logger.info(f"{'='*60}\n")

            for compound, dataset_list in compounds.items():
                logger.info(f"\n{compound}:")

                for cycle, code in dataset_list:
                    all_cycles.add(cycle)
                    file = self.download_dataset(cycle, code, category)
                    if file:
                        downloaded_files.append(file)
                    time.sleep(1)  # Be nice to CDC servers

        # Download demographics for linkage
        logger.info(f"\n{'='*60}")
        self.download_demographics(sorted(all_cycles))

        return downloaded_files

    def generate_data_dictionary(self, downloaded_files: List[Path]):
        """Generate summary of downloaded data"""
        logger.info("\nGenerating data dictionary...")

        summary = {
            'download_date': datetime.now().isoformat(),
            'total_files': len(downloaded_files),
            'categories': {},
            'cycles': set(),
            'total_size_mb': 0
        }

        for file in downloaded_files:
            if file and file.exists():
                category = file.parent.name
                size_mb = file.stat().st_size / (1024 * 1024)
                summary['total_size_mb'] += size_mb

                if category not in summary['categories']:
                    summary['categories'][category] = []

                # Extract cycle from filename
                for part in file.stem.split('_'):
                    if len(part) == 4 and part.isdigit():  # year
                        summary['cycles'].add(part[:4])

                summary['categories'][category].append({
                    'file': file.name,
                    'size_mb': round(size_mb, 2)
                })

        summary['cycles'] = sorted(list(summary['cycles']))

        # Save summary
        summary_file = self.output_dir / 'download_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"✓ Summary saved to {summary_file}")
        logger.info(f"\nTotal downloaded: {summary['total_size_mb']:.1f} MB")
        logger.info(f"Files: {summary['total_files']}")
        logger.info(f"Cycles: {', '.join(summary['cycles'])}")


def main():
    downloader = NHANESDownloader()
    downloaded_files = downloader.download_all_environmental()

    logger.info(f"\n{'='*60}")
    logger.info("NHANES DOWNLOAD COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Downloaded {len([f for f in downloaded_files if f])} files")
    logger.info(f"Location: {downloader.output_dir}")

    downloader.generate_data_dictionary(downloaded_files)


if __name__ == '__main__':
    main()