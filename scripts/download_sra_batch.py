#!/usr/bin/env python3
"""
SRA Batch Downloader

Downloads microbiome sequencing data from SRA using SRA Toolkit
Uses the catalog we created to prioritize downloads
"""

import argparse
import pandas as pd
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_sra_toolkit():
    """Check if SRA toolkit is installed"""
    try:
        result = subprocess.run(['prefetch', '--version'],
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def download_run(run_id, output_dir):
    """Download a single SRA run"""
    logger.info(f"Downloading {run_id}...")

    try:
        # Use prefetch (faster than fastq-dump)
        result = subprocess.run(
            ['prefetch', run_id, '--output-directory', str(output_dir)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per file
        )

        if result.returncode == 0:
            logger.info(f"✓ Downloaded {run_id}")
            return {'run_id': run_id, 'status': 'success'}
        else:
            logger.warning(f"✗ Failed {run_id}: {result.stderr[:200]}")
            return {'run_id': run_id, 'status': 'failed', 'error': result.stderr[:200]}

    except subprocess.TimeoutExpired:
        logger.warning(f"✗ Timeout {run_id}")
        return {'run_id': run_id, 'status': 'timeout'}
    except Exception as e:
        logger.error(f"✗ Error {run_id}: {e}")
        return {'run_id': run_id, 'status': 'error', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Download SRA microbiome data in batch')
    parser.add_argument('--catalog', required=True, help='SRA catalog CSV file')
    parser.add_argument('--top', type=int, default=50, help='Download top N studies')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--threads', type=int, default=4, help='Parallel downloads')

    args = parser.parse_args()

    # Check SRA toolkit
    if not check_sra_toolkit():
        logger.error("SRA Toolkit not found!")
        logger.error("Install with:")
        logger.error("  brew install sratoolkit  # Mac")
        logger.error("  sudo apt install sra-toolkit  # Linux")
        logger.error("Or download from: https://github.com/ncbi/sra-tools/wiki/Downloads")
        return 1

    logger.info("✓ SRA Toolkit found")

    # Load catalog
    catalog = pd.read_csv(args.catalog)
    logger.info(f"Loaded catalog: {len(catalog)} studies")

    # Sort by relevance and take top N
    catalog = catalog.sort_values('relevance_score', ascending=False).head(args.top)
    logger.info(f"Selected top {len(catalog)} studies by relevance")

    # Get unique run IDs
    # Assuming catalog has 'run_id' column (from SRA search)
    run_ids = catalog['bioproject_id'].unique()[:args.top]  # Use bioproject as proxy

    logger.info(f"Will download {len(run_ids)} runs")
    logger.info(f"Estimated time: {len(run_ids) * 5 // args.threads} minutes (rough estimate)")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download in parallel
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(download_run, run_id, output_dir): run_id
            for run_id in run_ids
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # Progress
            completed = len(results)
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (len(run_ids) - completed) / rate if rate > 0 else 0

            logger.info(f"Progress: {completed}/{len(run_ids)} "
                       f"({completed/len(run_ids)*100:.1f}%) "
                       f"ETA: {remaining/60:.1f} min")

    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - success

    logger.info(f"\n{'='*80}")
    logger.info("DOWNLOAD COMPLETE")
    logger.info('='*80)
    logger.info(f"Successful: {success}/{len(results)}")
    logger.info(f"Failed: {failed}/{len(results)}")
    logger.info(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    logger.info(f"Output directory: {output_dir}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'download_results.csv', index=False)
    logger.info(f"Results saved: {output_dir / 'download_results.csv'}")


if __name__ == '__main__':
    main()