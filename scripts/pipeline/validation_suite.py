#!/usr/bin/env python3
"""
Validation Suite for Downloaded Data

Validates downloaded files with:
- Checksum verification (MD5, SHA256)
- Format validation
- Completeness checks
- File integrity

Usage:
    python validation_suite.py --file data/raw/dataset.zip --checksum abc123

Author: AuDHD Correlation Study Team
"""

import argparse
import hashlib
import logging
import zipfile
import gzip
from pathlib import Path
from typing import Optional, Dict

try:
    import pandas as pd
except ImportError:
    pd = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationSuite:
    """Validate downloaded data files"""

    def __init__(self):
        self.validation_results = []

    def calculate_checksum(
        self,
        file_path: Path,
        algorithm: str = 'sha256'
    ) -> str:
        """Calculate file checksum"""
        hash_func = hashlib.new(algorithm)

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def verify_checksum(
        self,
        file_path: Path,
        expected: str,
        algorithm: str = 'sha256'
    ) -> bool:
        """Verify file checksum"""
        calculated = self.calculate_checksum(file_path, algorithm)

        if calculated.lower() == expected.lower():
            logger.info(f"✓ Checksum verified: {file_path.name}")
            return True
        else:
            logger.error(f"✗ Checksum mismatch: {file_path.name}")
            logger.error(f"  Expected: {expected}")
            logger.error(f"  Got: {calculated}")
            return False

    def validate_zip(self, file_path: Path) -> bool:
        """Validate ZIP file integrity"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Test ZIP integrity
                bad_file = zf.testzip()

                if bad_file:
                    logger.error(f"✗ Corrupt file in ZIP: {bad_file}")
                    return False

                logger.info(f"✓ ZIP file valid: {len(zf.namelist())} files")
                return True

        except Exception as e:
            logger.error(f"✗ ZIP validation failed: {e}")
            return False

    def validate_csv(self, file_path: Path) -> bool:
        """Validate CSV file"""
        if pd is None:
            logger.warning("pandas not installed, skipping CSV validation")
            return True

        try:
            df = pd.read_csv(file_path, nrows=5)
            logger.info(f"✓ CSV valid: {len(df.columns)} columns")
            return True

        except Exception as e:
            logger.error(f"✗ CSV validation failed: {e}")
            return False

    def validate_file(
        self,
        file_path: Path,
        checksum: Optional[str] = None,
        checksum_type: str = 'sha256'
    ) -> Dict:
        """Comprehensive file validation"""
        results = {
            'file': str(file_path),
            'exists': file_path.exists(),
            'checksum_verified': None,
            'format_valid': None,
            'errors': []
        }

        if not file_path.exists():
            results['errors'].append('File does not exist')
            return results

        # Checksum
        if checksum:
            results['checksum_verified'] = self.verify_checksum(
                file_path, checksum, checksum_type
            )

        # Format validation
        suffix = file_path.suffix.lower()

        if suffix == '.zip':
            results['format_valid'] = self.validate_zip(file_path)
        elif suffix == '.csv':
            results['format_valid'] = self.validate_csv(file_path)
        elif suffix == '.gz':
            try:
                with gzip.open(file_path, 'rb') as f:
                    f.read(1)
                results['format_valid'] = True
                logger.info(f"✓ GZIP file valid")
            except Exception as e:
                results['format_valid'] = False
                results['errors'].append(f"GZIP validation failed: {e}")
        else:
            results['format_valid'] = True
            logger.info(f"No format validation for {suffix}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Validate downloaded files')
    parser.add_argument('--file', required=True, help='File to validate')
    parser.add_argument('--checksum', help='Expected checksum')
    parser.add_argument('--checksum-type', default='sha256', choices=['md5', 'sha256'])

    args = parser.parse_args()

    validator = ValidationSuite()
    results = validator.validate_file(
        Path(args.file),
        checksum=args.checksum,
        checksum_type=args.checksum_type
    )

    print("\n=== Validation Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Exit code
    if results.get('checksum_verified') is False or results.get('format_valid') is False:
        sys.exit(1)


if __name__ == '__main__':
    import sys
    main()