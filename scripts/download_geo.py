#!/usr/bin/env python3
"""
GEO Dataset Downloader

Downloads gene expression data from GEO (Gene Expression Omnibus)
"""

import argparse
import requests
import gzip
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_geo_dataset(gse_id, output_dir):
    """Download GEO dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {gse_id}...")

    # GEO FTP URLs
    # Series matrix file (processed data)
    ftp_base = "https://ftp.ncbi.nlm.nih.gov/geo/series"

    # Determine series directory (e.g., GSE28521 -> GSE28nnn)
    gse_num = int(gse_id.replace('GSE', ''))
    series_dir = f"GSE{gse_num // 1000}nnn"

    # Try to download series matrix
    matrix_url = f"{ftp_base}/{series_dir}/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"

    try:
        logger.info(f"Downloading series matrix...")
        response = requests.get(matrix_url, timeout=300)

        if response.status_code == 200:
            matrix_file = output_dir / f"{gse_id}_series_matrix.txt.gz"
            with open(matrix_file, 'wb') as f:
                f.write(response.content)

            logger.info(f"✓ Downloaded series matrix ({len(response.content)/1e6:.1f} MB)")

            # Decompress
            with gzip.open(matrix_file, 'rb') as f_in:
                with open(matrix_file.with_suffix('').with_suffix('.txt'), 'wb') as f_out:
                    f_out.write(f_in.read())

            logger.info(f"✓ Decompressed")

        else:
            logger.warning(f"Series matrix not found (status {response.status_code})")

    except Exception as e:
        logger.error(f"Failed to download series matrix: {e}")

    # Try to download supplementary files
    supp_url = f"{ftp_base}/{series_dir}/{gse_id}/suppl/"

    try:
        logger.info(f"Checking for supplementary files...")
        response = requests.get(supp_url, timeout=30)

        if response.status_code == 200:
            # Parse FTP listing
            import re
            files = re.findall(r'href="([^"]+)"', response.text)

            for file in files:
                if file.endswith(('.txt.gz', '.csv.gz', '.tsv.gz', '.tar')):
                    file_url = f"{supp_url}{file}"
                    logger.info(f"Downloading {file}...")

                    try:
                        file_response = requests.get(file_url, timeout=600)
                        out_file = output_dir / file
                        with open(out_file, 'wb') as f:
                            f.write(file_response.content)
                        logger.info(f"✓ Downloaded {file} ({len(file_response.content)/1e6:.1f} MB)")
                        time.sleep(2)  # Rate limiting
                    except Exception as e:
                        logger.warning(f"Failed to download {file}: {e}")

    except Exception as e:
        logger.warning(f"Failed to check supplementary files: {e}")

    logger.info(f"✓ {gse_id} download complete")


def main():
    parser = argparse.ArgumentParser(description='Download GEO dataset')
    parser.add_argument('--dataset', required=True, help='GEO accession (e.g., GSE28521)')
    parser.add_argument('--output', required=True, help='Output directory')

    args = parser.parse_args()

    download_geo_dataset(args.dataset, args.output)


if __name__ == '__main__':
    main()