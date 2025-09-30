#!/usr/bin/env python3
"""
SFARI Base Portal Navigator

Automates navigation and data access for SFARI Base portal (base.sfari.org).
Handles authentication, session management, and data discovery.

IMPORTANT: This script requires approved SFARI Base access. You must:
1. Have completed SFARI Base registration
2. Have executed Data Use Agreement (DUA)
3. Have received approval from SFARI Data Access Committee

Usage:
    python sfari_portal_navigator.py --username YOUR_EMAIL --list-datasets
    python sfari_portal_navigator.py --username YOUR_EMAIL --dataset SPARK --explore
"""

import argparse
import getpass
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.common.exceptions import (
        TimeoutException,
        NoSuchElementException,
        WebDriverException
    )
except ImportError:
    print("ERROR: Selenium not installed. Install with: pip install selenium")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Install with: pip install pandas")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SFARI_BASE_URL = "https://base.sfari.org"
LOGIN_URL = f"{SFARI_BASE_URL}/login"
TIMEOUT = 30  # seconds
MAX_RETRIES = 3


class SFARIPortalNavigator:
    """Navigate and interact with SFARI Base portal"""

    def __init__(self, username: str, password: Optional[str] = None, headless: bool = True):
        """
        Initialize SFARI portal navigator

        Args:
            username: SFARI Base email/username
            password: SFARI Base password (will prompt if not provided)
            headless: Run browser in headless mode
        """
        self.username = username
        self.password = password or getpass.getpass("SFARI Base password: ")
        self.headless = headless
        self.driver = None
        self.logged_in = False
        self.session_file = Path.home() / ".sfari_session.json"

    def _setup_driver(self) -> webdriver.Chrome:
        """Setup Chrome WebDriver with appropriate options"""
        options = webdriver.ChromeOptions()

        if self.headless:
            options.add_argument('--headless')

        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')

        # User agent to avoid bot detection
        options.add_argument(
            'user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )

        try:
            driver = webdriver.Chrome(options=options)
            logger.info("Chrome WebDriver initialized successfully")
            return driver
        except WebDriverException as e:
            logger.error(f"Failed to initialize Chrome WebDriver: {e}")
            logger.info("Install ChromeDriver: brew install chromedriver")
            logger.info("Or download from: https://chromedriver.chromium.org/")
            raise

    def login(self) -> bool:
        """
        Login to SFARI Base portal

        Returns:
            True if login successful, False otherwise
        """
        if self.logged_in:
            logger.info("Already logged in")
            return True

        logger.info("Initializing browser...")
        self.driver = self._setup_driver()

        try:
            logger.info(f"Navigating to SFARI Base: {LOGIN_URL}")
            self.driver.get(LOGIN_URL)

            # Wait for login form
            wait = WebDriverWait(self.driver, TIMEOUT)

            # Enter username
            logger.info("Entering credentials...")
            username_field = wait.until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            username_field.clear()
            username_field.send_keys(self.username)

            # Enter password
            password_field = self.driver.find_element(By.ID, "password")
            password_field.clear()
            password_field.send_keys(self.password)

            # Submit form
            password_field.send_keys(Keys.RETURN)

            # Wait for successful login (check for dashboard or profile element)
            try:
                wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "dashboard"))
                )
                logger.info("✓ Login successful")
                self.logged_in = True

                # Save session cookies
                self._save_session()

                return True

            except TimeoutException:
                # Check for error message
                try:
                    error_msg = self.driver.find_element(
                        By.CLASS_NAME, "error-message"
                    ).text
                    logger.error(f"Login failed: {error_msg}")
                except NoSuchElementException:
                    logger.error("Login failed: Unknown error")

                return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def _save_session(self):
        """Save session cookies for reuse"""
        try:
            cookies = self.driver.get_cookies()
            session_data = {
                "username": self.username,
                "cookies": cookies,
                "timestamp": datetime.now().isoformat()
            }
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f)
            logger.debug(f"Session saved to {self.session_file}")
        except Exception as e:
            logger.warning(f"Could not save session: {e}")

    def _load_session(self) -> bool:
        """Load saved session cookies"""
        if not self.session_file.exists():
            return False

        try:
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)

            # Check if session is recent (< 24 hours old)
            timestamp = datetime.fromisoformat(session_data["timestamp"])
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600

            if age_hours > 24:
                logger.info("Session expired (> 24 hours old)")
                return False

            # Load cookies
            self.driver = self._setup_driver()
            self.driver.get(SFARI_BASE_URL)

            for cookie in session_data["cookies"]:
                self.driver.add_cookie(cookie)

            self.driver.refresh()

            # Verify session is valid
            try:
                WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "dashboard"))
                )
                logger.info("✓ Session restored successfully")
                self.logged_in = True
                return True
            except TimeoutException:
                logger.info("Saved session is invalid")
                return False

        except Exception as e:
            logger.warning(f"Could not load session: {e}")
            return False

    def list_datasets(self) -> List[Dict]:
        """
        List available datasets for the logged-in user

        Returns:
            List of dataset information dictionaries
        """
        if not self.logged_in:
            logger.error("Must login first")
            return []

        logger.info("Fetching available datasets...")

        try:
            # Navigate to datasets page
            datasets_url = f"{SFARI_BASE_URL}/data-catalog"
            self.driver.get(datasets_url)

            wait = WebDriverWait(self.driver, TIMEOUT)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "dataset-list")))

            # Parse dataset information
            datasets = []
            dataset_elements = self.driver.find_elements(By.CLASS_NAME, "dataset-item")

            for elem in dataset_elements:
                try:
                    dataset = {
                        "name": elem.find_element(By.CLASS_NAME, "dataset-name").text,
                        "description": elem.find_element(By.CLASS_NAME, "dataset-description").text,
                        "sample_size": elem.find_element(By.CLASS_NAME, "sample-count").text,
                        "access_status": elem.find_element(By.CLASS_NAME, "access-status").text,
                        "last_updated": elem.find_element(By.CLASS_NAME, "last-updated").text,
                    }
                    datasets.append(dataset)
                except NoSuchElementException:
                    continue

            logger.info(f"Found {len(datasets)} datasets")
            return datasets

        except Exception as e:
            logger.error(f"Error fetching datasets: {e}")
            return []

    def explore_dataset(self, dataset_name: str) -> Dict:
        """
        Explore dataset structure and available files

        Args:
            dataset_name: Name of dataset (e.g., "SPARK", "SSC")

        Returns:
            Dictionary with dataset structure information
        """
        if not self.logged_in:
            logger.error("Must login first")
            return {}

        logger.info(f"Exploring dataset: {dataset_name}")

        try:
            # Navigate to dataset page
            dataset_url = f"{SFARI_BASE_URL}/data-catalog/{dataset_name.lower()}"
            self.driver.get(dataset_url)

            wait = WebDriverWait(self.driver, TIMEOUT)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "dataset-details")))

            # Extract dataset metadata
            metadata = {
                "name": dataset_name,
                "data_types": [],
                "sample_count": 0,
                "files": [],
                "phenotypes": [],
            }

            # Parse data types
            data_type_elements = self.driver.find_elements(By.CLASS_NAME, "data-type")
            metadata["data_types"] = [elem.text for elem in data_type_elements]

            # Parse sample count
            try:
                sample_elem = self.driver.find_element(By.CLASS_NAME, "sample-count")
                metadata["sample_count"] = int(sample_elem.text.replace(",", ""))
            except (NoSuchElementException, ValueError):
                pass

            # Parse available files
            file_elements = self.driver.find_elements(By.CLASS_NAME, "data-file")
            for elem in file_elements:
                try:
                    file_info = {
                        "name": elem.find_element(By.CLASS_NAME, "file-name").text,
                        "size": elem.find_element(By.CLASS_NAME, "file-size").text,
                        "type": elem.find_element(By.CLASS_NAME, "file-type").text,
                        "download_url": elem.find_element(By.TAG_NAME, "a").get_attribute("href"),
                    }
                    metadata["files"].append(file_info)
                except NoSuchElementException:
                    continue

            logger.info(f"Dataset exploration complete: {len(metadata['files'])} files found")
            return metadata

        except Exception as e:
            logger.error(f"Error exploring dataset: {e}")
            return {}

    def get_phenotype_browser(self, dataset_name: str) -> pd.DataFrame:
        """
        Access phenotype browser and extract variable information

        Args:
            dataset_name: Name of dataset

        Returns:
            DataFrame with phenotype variable information
        """
        if not self.logged_in:
            logger.error("Must login first")
            return pd.DataFrame()

        logger.info(f"Accessing phenotype browser for {dataset_name}...")

        try:
            # Navigate to phenotype browser
            pheno_url = f"{SFARI_BASE_URL}/data-catalog/{dataset_name.lower()}/phenotypes"
            self.driver.get(pheno_url)

            wait = WebDriverWait(self.driver, TIMEOUT)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "phenotype-table")))

            # Extract phenotype table
            table_elem = self.driver.find_element(By.CLASS_NAME, "phenotype-table")
            html = table_elem.get_attribute('outerHTML')

            # Parse with pandas
            df = pd.read_html(html)[0]

            logger.info(f"Extracted {len(df)} phenotype variables")
            return df

        except Exception as e:
            logger.error(f"Error accessing phenotype browser: {e}")
            return pd.DataFrame()

    def download_file(self, file_url: str, output_path: Path, verify_checksum: bool = True) -> bool:
        """
        Download file from SFARI Base

        Args:
            file_url: URL of file to download
            output_path: Local path to save file
            verify_checksum: Verify MD5 checksum after download

        Returns:
            True if download successful
        """
        if not self.logged_in:
            logger.error("Must login first")
            return False

        logger.info(f"Downloading: {file_url}")
        logger.info(f"Output: {output_path}")

        try:
            # Navigate to download URL
            self.driver.get(file_url)

            # Wait for download to start (implementation depends on SFARI Base UI)
            # This is a placeholder - actual implementation may vary
            time.sleep(5)

            # Check if file was downloaded
            if output_path.exists():
                logger.info(f"✓ Download complete: {output_path}")

                if verify_checksum:
                    # Verify checksum (if provided by SFARI Base)
                    # Implementation depends on SFARI Base checksums
                    pass

                return True
            else:
                logger.error("Download failed")
                return False

        except Exception as e:
            logger.error(f"Download error: {e}")
            return False

    def get_download_manifest(self, dataset_name: str, output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Generate manifest of all downloadable files for a dataset

        Args:
            dataset_name: Name of dataset
            output_path: Optional path to save manifest CSV

        Returns:
            DataFrame with file manifest
        """
        logger.info(f"Generating download manifest for {dataset_name}...")

        metadata = self.explore_dataset(dataset_name)

        if not metadata.get("files"):
            logger.warning("No files found")
            return pd.DataFrame()

        # Create manifest DataFrame
        manifest = pd.DataFrame(metadata["files"])

        # Add metadata
        manifest["dataset"] = dataset_name
        manifest["generated_date"] = datetime.now().isoformat()

        if output_path:
            manifest.to_csv(output_path, index=False)
            logger.info(f"Manifest saved to: {output_path}")

        return manifest

    def logout(self):
        """Logout and close browser"""
        if self.driver:
            try:
                # Navigate to logout URL
                logout_url = f"{SFARI_BASE_URL}/logout"
                self.driver.get(logout_url)
                time.sleep(2)

                logger.info("Logged out successfully")
            except Exception as e:
                logger.warning(f"Logout error: {e}")
            finally:
                self.driver.quit()
                self.logged_in = False

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.logout()


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="SFARI Base Portal Navigator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python sfari_portal_navigator.py --username your@email.com --list-datasets

  # Explore SPARK dataset
  python sfari_portal_navigator.py --username your@email.com --dataset SPARK --explore

  # Get phenotype browser data
  python sfari_portal_navigator.py --username your@email.com --dataset SPARK --phenotypes

  # Generate download manifest
  python sfari_portal_navigator.py --username your@email.com --dataset SPARK --manifest

Note: You must have approved SFARI Base access before using this tool.
        """
    )

    parser.add_argument(
        "--username",
        required=True,
        help="SFARI Base username/email"
    )

    parser.add_argument(
        "--password",
        help="SFARI Base password (will prompt if not provided)"
    )

    parser.add_argument(
        "--dataset",
        choices=["SPARK", "SSC", "AGRE", "Simons_Searchlight"],
        help="Dataset to access"
    )

    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets"
    )

    parser.add_argument(
        "--explore",
        action="store_true",
        help="Explore dataset structure"
    )

    parser.add_argument(
        "--phenotypes",
        action="store_true",
        help="Get phenotype browser data"
    )

    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Generate download manifest"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize navigator
    with SFARIPortalNavigator(
        username=args.username,
        password=args.password,
        headless=args.headless
    ) as navigator:

        # Login
        if not navigator.login():
            logger.error("Login failed. Please check credentials and DUA approval status.")
            sys.exit(1)

        # Execute requested action
        if args.list_datasets:
            datasets = navigator.list_datasets()
            for ds in datasets:
                print(f"\n{ds['name']}")
                print(f"  Description: {ds['description']}")
                print(f"  Samples: {ds['sample_size']}")
                print(f"  Access: {ds['access_status']}")

        elif args.dataset:
            if args.explore:
                metadata = navigator.explore_dataset(args.dataset)
                print(json.dumps(metadata, indent=2))

            elif args.phenotypes:
                df = navigator.get_phenotype_browser(args.dataset)
                if args.output:
                    df.to_csv(args.output, index=False)
                    print(f"Phenotypes saved to: {args.output}")
                else:
                    print(df.to_string())

            elif args.manifest:
                output_path = args.output or Path(f"data/manifests/{args.dataset.lower()}_manifest.csv")
                manifest = navigator.get_download_manifest(args.dataset, output_path)
                print(f"Manifest generated: {len(manifest)} files")

        else:
            parser.print_help()


if __name__ == "__main__":
    main()