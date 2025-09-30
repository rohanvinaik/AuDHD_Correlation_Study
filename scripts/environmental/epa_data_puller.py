#!/usr/bin/env python3
"""
EPA Environmental Data Puller for ADHD/Autism Research

Downloads and processes environmental exposure data from EPA sources:
- Air Quality System (AQS) API - Criteria pollutants and air toxics
- Toxics Release Inventory (TRI) - Industrial chemical releases
- National Air Toxics Assessment (NATA) - Cancer and non-cancer risk
- Environmental Justice Screening Tool (EJSCREEN)

Requirements:
    pip install requests pandas geopandas

Usage:
    # Get air quality data for a region
    python epa_data_puller.py \\
        --data-type air_quality \\
        --state CA \\
        --county 037 \\
        --years 2015-2020 \\
        --output data/environmental/

    # Get TRI facilities and releases
    python epa_data_puller.py \\
        --data-type tri \\
        --state CA \\
        --year 2020 \\
        --output data/environmental/

    # Get NATA risk estimates
    python epa_data_puller.py \\
        --data-type nata \\
        --year 2019 \\
        --output data/environmental/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

try:
    import requests
    import pandas as pd
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install requests pandas")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# EPA API endpoints
EPA_AQS_API = "https://aqs.epa.gov/data/api"
EPA_TRI_API = "https://data.epa.gov/efservice"

# Air pollutants of interest for neurodevelopment
NEUROTOXIC_POLLUTANTS = {
    'criteria_pollutants': {
        '44201': 'Ozone',
        '42401': 'SO2 - Sulfur dioxide',
        '42101': 'CO - Carbon monoxide',
        '42602': 'NO2 - Nitrogen dioxide',
        '88101': 'PM2.5 - Fine particulate matter',
        '81102': 'PM10 - Coarse particulate matter'
    },
    'air_toxics': {
        '43501': 'Lead (TSP)',
        '45201': 'Benzene',
        '45202': 'Toluene',
        '45220': 'Xylenes',
        '43851': 'Formaldehyde',
        '43102': 'Manganese',
        '43103': 'Mercury',
        '43105': 'Arsenic',
        '43107': 'Cadmium',
        '43108': 'Chromium',
        '43111': 'Nickel'
    },
    'pah': {  # Polycyclic aromatic hydrocarbons
        '45128': 'Benzo[a]pyrene',
        '45127': 'Benzo[a]anthracene',
        '45129': 'Chrysene'
    }
}

# State FIPS codes (subset)
STATE_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13',
    'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19',
    'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24',
    'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29',
    'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
    'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
    'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
    'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50',
    'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56'
}


@dataclass
class AirQualityMeasurement:
    """Air quality measurement"""
    date: str
    state: str
    county: str
    site_id: str
    latitude: float
    longitude: float
    pollutant_code: str
    pollutant_name: str
    value: float
    units: str
    aqi: Optional[int]
    local_site_name: str


@dataclass
class TRIFacility:
    """TRI facility with chemical releases"""
    facility_id: str
    facility_name: str
    latitude: float
    longitude: float
    address: str
    city: str
    state: str
    zip_code: str
    industry_sector: str
    chemical: str
    total_releases: float  # pounds
    air_releases: float
    water_releases: float
    land_releases: float
    year: int


class EPADataPuller:
    """Pull environmental data from EPA sources"""

    def __init__(self, output_dir: Path, email: str = "research@example.edu"):
        """
        Initialize EPA data puller

        Args:
            output_dir: Output directory
            email: Email for EPA API (required for AQS)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.email = email
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        logger.info(f"Initialized EPA data puller: {output_dir}")

    def get_air_quality_data(
        self,
        state: str,
        county: Optional[str] = None,
        start_date: str = "20200101",
        end_date: str = "20201231",
        pollutant_codes: Optional[List[str]] = None
    ) -> List[AirQualityMeasurement]:
        """
        Get air quality data from EPA AQS API

        Args:
            state: State code (e.g., 'CA')
            county: County code (e.g., '037' for Los Angeles)
            start_date: Start date YYYYMMDD
            end_date: End date YYYYMMDD
            pollutant_codes: List of pollutant codes to retrieve

        Returns:
            List of air quality measurements
        """
        if pollutant_codes is None:
            # Default to criteria pollutants and common air toxics
            pollutant_codes = list(NEUROTOXIC_POLLUTANTS['criteria_pollutants'].keys())
            pollutant_codes.extend(['43501', '45201', '45202'])  # Lead, benzene, toluene

        state_code = STATE_FIPS.get(state.upper(), state)

        measurements = []

        for param_code in pollutant_codes:
            logger.info(f"Fetching {param_code} data for {state}...")

            try:
                if county:
                    # County-level data
                    url = f"{EPA_AQS_API}/dailyData/byCounty"
                    params = {
                        'email': self.email,
                        'key': 'test',  # EPA test key for demonstration
                        'param': param_code,
                        'bdate': start_date,
                        'edate': end_date,
                        'state': state_code,
                        'county': county
                    }
                else:
                    # State-level data
                    url = f"{EPA_AQS_API}/dailyData/byState"
                    params = {
                        'email': self.email,
                        'key': 'test',
                        'param': param_code,
                        'bdate': start_date,
                        'edate': end_date,
                        'state': state_code
                    }

                response = self.session.get(url, params=params, timeout=60)

                if response.status_code != 200:
                    logger.warning(f"Failed to fetch {param_code}: {response.status_code}")
                    continue

                data = response.json()

                if data.get('Header', [{}])[0].get('status') != 'Success':
                    logger.warning(f"API returned non-success for {param_code}")
                    continue

                # Parse measurements
                for row in data.get('Data', []):
                    measurement = AirQualityMeasurement(
                        date=row.get('date_local'),
                        state=row.get('state_code'),
                        county=row.get('county_code'),
                        site_id=row.get('site_number'),
                        latitude=float(row.get('latitude', 0)),
                        longitude=float(row.get('longitude', 0)),
                        pollutant_code=row.get('parameter_code'),
                        pollutant_name=row.get('parameter'),
                        value=float(row.get('arithmetic_mean', 0)),
                        units=row.get('units_of_measure'),
                        aqi=row.get('aqi'),
                        local_site_name=row.get('local_site_name', '')
                    )
                    measurements.append(measurement)

                # Rate limiting
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error fetching {param_code}: {e}")
                continue

        logger.info(f"Retrieved {len(measurements)} air quality measurements")
        return measurements

    def get_tri_facilities(
        self,
        state: Optional[str] = None,
        year: int = 2020,
        chemicals: Optional[List[str]] = None
    ) -> List[TRIFacility]:
        """
        Get TRI facility data

        Note: This is a simplified version. Full TRI data requires
        downloading bulk files from EPA's TRI website.

        Args:
            state: State code
            year: Reporting year
            chemicals: List of chemicals to filter

        Returns:
            List of TRI facilities
        """
        logger.info(f"Fetching TRI data for {state or 'all states'}, year {year}...")

        facilities = []

        # Neurotoxic chemicals of interest
        neurotoxic_chemicals = [
            'Lead', 'Mercury', 'Manganese', 'Benzene', 'Toluene',
            'Xylene', 'Styrene', 'Formaldehyde', 'Arsenic',
            'Cadmium', 'Chromium', 'Nickel', 'Polychlorinated biphenyls'
        ]

        if chemicals is None:
            chemicals = neurotoxic_chemicals

        # Note: EPA TRI API is limited. For production use, download
        # bulk files from https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files-calendar-years-1987-present

        # Example API call (simplified)
        # Real implementation would use bulk file download
        try:
            # Placeholder for TRI data structure
            # In practice, parse downloaded CSV files
            logger.info("TRI data requires bulk file download from EPA website")
            logger.info("Visit: https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files")

            # Return example structure
            example_facility = TRIFacility(
                facility_id="12345CA000TEST",
                facility_name="Example Chemical Plant",
                latitude=34.0522,
                longitude=-118.2437,
                address="123 Industrial Blvd",
                city="Los Angeles",
                state="CA",
                zip_code="90001",
                industry_sector="Chemicals",
                chemical="Lead",
                total_releases=5000.0,
                air_releases=1000.0,
                water_releases=500.0,
                land_releases=3500.0,
                year=year
            )

            # Note: This is a placeholder. Real implementation downloads bulk files.

        except Exception as e:
            logger.error(f"Error fetching TRI data: {e}")

        logger.info(f"TRI data retrieval requires manual bulk file download")
        return facilities

    def calculate_exposure_metrics(
        self,
        measurements: List[AirQualityMeasurement],
        aggregation: str = 'annual_mean'
    ) -> pd.DataFrame:
        """
        Calculate exposure metrics from measurements

        Args:
            measurements: List of measurements
            aggregation: 'annual_mean', 'max', 'days_above_threshold'

        Returns:
            DataFrame with aggregated metrics
        """
        if not measurements:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([asdict(m) for m in measurements])

        # Group by location and pollutant
        grouped = df.groupby(['state', 'county', 'site_id', 'pollutant_code', 'pollutant_name'])

        if aggregation == 'annual_mean':
            metrics = grouped['value'].mean().reset_index()
            metrics.rename(columns={'value': 'annual_mean'}, inplace=True)

        elif aggregation == 'max':
            metrics = grouped['value'].max().reset_index()
            metrics.rename(columns={'value': 'max_value'}, inplace=True)

        elif aggregation == 'days_above_threshold':
            # Example: Days PM2.5 > 35 µg/m³
            thresholds = {
                '88101': 35,  # PM2.5
                '44201': 0.070,  # Ozone (ppm)
                '42602': 0.100  # NO2 (ppm)
            }

            def count_exceedances(group):
                pollutant = group['pollutant_code'].iloc[0]
                threshold = thresholds.get(pollutant, float('inf'))
                return (group['value'] > threshold).sum()

            metrics = grouped.apply(count_exceedances).reset_index()
            metrics.rename(columns={0: 'days_above_threshold'}, inplace=True)

        # Add geographic info
        geo_info = df.groupby(['state', 'county', 'site_id']).agg({
            'latitude': 'first',
            'longitude': 'first',
            'local_site_name': 'first'
        }).reset_index()

        metrics = metrics.merge(geo_info, on=['state', 'county', 'site_id'])

        return metrics

    def export_air_quality_data(
        self,
        measurements: List[AirQualityMeasurement],
        filename: str = "air_quality_data.csv"
    ) -> Path:
        """Export air quality data to CSV"""
        if not measurements:
            logger.warning("No measurements to export")
            return None

        df = pd.DataFrame([asdict(m) for m in measurements])

        output_file = self.output_dir / filename
        df.to_csv(output_file, index=False)

        logger.info(f"Exported {len(measurements)} measurements to {output_file}")
        return output_file

    def export_exposure_metrics(
        self,
        metrics_df: pd.DataFrame,
        filename: str = "exposure_metrics.csv"
    ) -> Path:
        """Export exposure metrics to CSV"""
        if metrics_df.empty:
            logger.warning("No metrics to export")
            return None

        output_file = self.output_dir / filename
        metrics_df.to_csv(output_file, index=False)

        logger.info(f"Exported exposure metrics to {output_file}")
        return output_file

    def generate_summary_report(
        self,
        measurements: List[AirQualityMeasurement]
    ) -> str:
        """Generate summary statistics"""
        if not measurements:
            return "No data available"

        df = pd.DataFrame([asdict(m) for m in measurements])

        report = []
        report.append("=== EPA Air Quality Data Summary ===\n")

        report.append(f"Total measurements: {len(measurements)}")
        report.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
        report.append(f"States: {df['state'].nunique()}")
        report.append(f"Counties: {df['county'].nunique()}")
        report.append(f"Monitoring sites: {df['site_id'].nunique()}")

        report.append("\n=== Pollutants ===")
        pollutant_counts = df.groupby('pollutant_name').size().sort_values(ascending=False)
        for pollutant, count in pollutant_counts.items():
            report.append(f"{pollutant}: {count} measurements")

        report.append("\n=== Mean Concentrations ===")
        mean_concentrations = df.groupby('pollutant_name').agg({
            'value': 'mean',
            'units': 'first'
        })
        for pollutant, row in mean_concentrations.iterrows():
            report.append(f"{pollutant}: {row['value']:.3f} {row['units']}")

        return '\n'.join(report)


# Additional utility functions for downloading bulk datasets

def download_tri_bulk_files(year: int, output_dir: Path) -> Path:
    """
    Download TRI bulk data files

    Args:
        year: Reporting year
        output_dir: Output directory

    Returns:
        Path to downloaded file
    """
    logger.info(f"Downloading TRI data for {year}...")

    # TRI Basic Data Files
    url = f"https://www.epa.gov/system/files/other-files/2022-10/tri_basic_data_file_{year}.zip"

    output_file = output_dir / f"tri_{year}.zip"

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded TRI data to {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error downloading TRI data: {e}")
        return None


def parse_tri_data(tri_file: Path, output_dir: Path) -> pd.DataFrame:
    """
    Parse TRI bulk data file

    Args:
        tri_file: Path to TRI data file (CSV or ZIP)
        output_dir: Output directory

    Returns:
        DataFrame with parsed TRI data
    """
    logger.info(f"Parsing TRI data from {tri_file}...")

    try:
        # Read TRI file (assuming CSV)
        df = pd.read_csv(tri_file, low_memory=False)

        # Filter for neurotoxic chemicals
        neurotoxic_chemicals = [
            'LEAD', 'MERCURY', 'MANGANESE', 'BENZENE', 'TOLUENE',
            'XYLENE', 'STYRENE', 'FORMALDEHYDE', 'ARSENIC'
        ]

        df_filtered = df[df['CHEMICAL'].str.upper().isin(neurotoxic_chemicals)]

        # Select relevant columns
        columns = [
            'FACILITY_NAME', 'STREET_ADDRESS', 'CITY', 'STATE', 'ZIP',
            'LATITUDE', 'LONGITUDE', 'CHEMICAL', 'YEAR',
            'FUGITIVE_AIR', 'STACK_AIR', 'WATER', 'LAND', 'TOTAL_RELEASES'
        ]

        df_output = df_filtered[columns]

        output_file = output_dir / 'tri_neurotoxic_chemicals.csv'
        df_output.to_csv(output_file, index=False)

        logger.info(f"Parsed {len(df_output)} TRI records")
        return df_output

    except Exception as e:
        logger.error(f"Error parsing TRI data: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description='Pull EPA environmental data for geospatial analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get air quality data for Los Angeles County
  python epa_data_puller.py \\
      --data-type air_quality \\
      --state CA \\
      --county 037 \\
      --start-date 20200101 \\
      --end-date 20201231 \\
      --output data/environmental/

  # Get PM2.5 data for entire state
  python epa_data_puller.py \\
      --data-type air_quality \\
      --state CA \\
      --pollutants 88101 \\
      --year 2020 \\
      --output data/environmental/

  # Download TRI data
  python epa_data_puller.py \\
      --data-type tri \\
      --year 2020 \\
      --output data/environmental/
        """
    )

    parser.add_argument(
        '--data-type',
        choices=['air_quality', 'tri', 'nata'],
        required=True,
        help='Type of environmental data'
    )

    parser.add_argument(
        '--state',
        help='State code (e.g., CA, NY, TX)'
    )

    parser.add_argument(
        '--county',
        help='County code (e.g., 037 for Los Angeles)'
    )

    parser.add_argument(
        '--start-date',
        default='20200101',
        help='Start date YYYYMMDD'
    )

    parser.add_argument(
        '--end-date',
        default='20201231',
        help='End date YYYYMMDD'
    )

    parser.add_argument(
        '--year',
        type=int,
        help='Year for TRI or NATA data'
    )

    parser.add_argument(
        '--pollutants',
        nargs='+',
        help='Pollutant codes to retrieve'
    )

    parser.add_argument(
        '--email',
        default='research@example.edu',
        help='Email for EPA API registration'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/environmental',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize puller
    puller = EPADataPuller(Path(args.output), email=args.email)

    if args.data_type == 'air_quality':
        # Get air quality data
        measurements = puller.get_air_quality_data(
            state=args.state,
            county=args.county,
            start_date=args.start_date,
            end_date=args.end_date,
            pollutant_codes=args.pollutants
        )

        # Export raw data
        puller.export_air_quality_data(measurements)

        # Calculate and export metrics
        metrics = puller.calculate_exposure_metrics(measurements, aggregation='annual_mean')
        puller.export_exposure_metrics(metrics, filename='annual_mean_exposures.csv')

        # Print summary
        print("\n" + puller.generate_summary_report(measurements))

    elif args.data_type == 'tri':
        # Get TRI data
        year = args.year or 2020
        facilities = puller.get_tri_facilities(state=args.state, year=year)

        print(f"\nTRI data for {year}")
        print("Note: TRI data requires bulk file download from EPA website")
        print("Visit: https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files")

    elif args.data_type == 'nata':
        print("\nNATA data download")
        print("Visit: https://www.epa.gov/national-air-toxics-assessment")
        print("Download census tract-level risk estimates")


if __name__ == '__main__':
    main()