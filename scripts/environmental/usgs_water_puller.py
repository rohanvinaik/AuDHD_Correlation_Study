#!/usr/bin/env python3
"""
USGS Water Quality and Pesticide Data Puller

Downloads water quality and pesticide data from USGS sources:
- NWIS (National Water Information System) - Water quality measurements
- Pesticide National Synthesis Project - Pesticide concentrations
- Water-Quality Watch - Real-time data

Requirements:
    pip install requests pandas

Usage:
    # Get water quality data for a region
    python usgs_water_puller.py \\
        --data-type water_quality \\
        --state CA \\
        --start-date 2020-01-01 \\
        --end-date 2020-12-31 \\
        --output data/environmental/

    # Get pesticide data
    python usgs_water_puller.py \\
        --data-type pesticides \\
        --state CA \\
        --year 2020 \\
        --output data/environmental/

Author: AuDHD Correlation Study Team
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
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


# USGS NWIS API
NWIS_BASE_URL = "https://waterdata.usgs.gov/nwis"
NWIS_API_URL = "https://waterservices.usgs.gov/nwis"

# Contaminants of interest for neurodevelopment
NEUROTOXIC_CONTAMINANTS = {
    'heavy_metals': {
        '01051': 'Lead, water, dissolved',
        '01049': 'Lead, water, total',
        '71900': 'Mercury, water, total',
        '50287': 'Mercury, water, dissolved',
        '01067': 'Manganese, water, dissolved',
        '01055': 'Manganese, water, total',
        '01002': 'Arsenic, water, dissolved',
        '01000': 'Arsenic, water, total',
        '01027': 'Cadmium, water, dissolved',
        '01025': 'Cadmium, water, total'
    },
    'pesticides': {
        '04041': 'Atrazine',
        '04040': 'Chlorpyrifos',
        '82663': 'Malathion',
        '82670': 'Glyphosate',
        '04095': 'Organophosphates, total'
    },
    'nutrients': {
        '00618': 'Nitrate',
        '00631': 'Nitrite plus nitrate',
        '00665': 'Phosphorus, total'
    },
    'general': {
        '00010': 'Temperature, water',
        '00300': 'Dissolved oxygen',
        '00400': 'pH'
    }
}

# State abbreviations to FIPS
STATE_FIPS_USGS = {
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
class WaterQualityMeasurement:
    """Water quality measurement"""
    site_no: str
    site_name: str
    latitude: float
    longitude: float
    datetime: str
    parameter_code: str
    parameter_name: str
    value: float
    units: str
    qualifier: str
    site_type: str  # Stream, Lake, Well, etc.


@dataclass
class PesticideDetection:
    """Pesticide detection"""
    site_no: str
    site_name: str
    latitude: float
    longitude: float
    sample_date: str
    pesticide_code: str
    pesticide_name: str
    concentration: float
    units: str
    detection_limit: float
    method: str


class USGSWaterPuller:
    """Pull water quality data from USGS"""

    def __init__(self, output_dir: Path):
        """
        Initialize USGS water puller

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        logger.info(f"Initialized USGS water puller: {output_dir}")

    def get_sites_in_state(
        self,
        state_code: str,
        site_type: str = 'ST'  # ST=Stream, LK=Lake, GW=Groundwater
    ) -> pd.DataFrame:
        """
        Get monitoring sites in a state

        Args:
            state_code: State abbreviation
            site_type: Site type code

        Returns:
            DataFrame with site information
        """
        logger.info(f"Fetching monitoring sites in {state_code}...")

        url = f"{NWIS_API_URL}/site/"

        params = {
            'format': 'rdb',
            'stateCd': state_code,
            'siteType': site_type,
            'siteStatus': 'all',
            'hasDataTypeCd': 'qw'  # Water quality data
        }

        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()

            # Parse RDB format (tab-delimited with header comments)
            lines = response.text.split('\n')
            data_lines = [line for line in lines if not line.startswith('#')]

            # Read into DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO('\n'.join(data_lines)), sep='\t', comment='#', skiprows=1)

            logger.info(f"Found {len(df)} monitoring sites")
            return df

        except Exception as e:
            logger.error(f"Error fetching sites: {e}")
            return pd.DataFrame()

    def get_water_quality_data(
        self,
        site_numbers: List[str],
        parameter_codes: List[str],
        start_date: str,
        end_date: str
    ) -> List[WaterQualityMeasurement]:
        """
        Get water quality data for sites

        Args:
            site_numbers: List of USGS site numbers
            parameter_codes: List of parameter codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of water quality measurements
        """
        logger.info(f"Fetching water quality data for {len(site_numbers)} sites...")

        measurements = []

        # USGS API limits batch sizes, process in chunks
        chunk_size = 20
        for i in range(0, len(site_numbers), chunk_size):
            site_chunk = site_numbers[i:i+chunk_size]

            for param_code in parameter_codes:
                try:
                    url = f"{NWIS_API_URL}/qw/"

                    params = {
                        'format': 'json',
                        'sites': ','.join(site_chunk),
                        'startDT': start_date,
                        'endDT': end_date,
                        'parameterCd': param_code
                    }

                    response = self.session.get(url, params=params, timeout=60)

                    if response.status_code != 200:
                        logger.warning(f"Failed to fetch data for {param_code}: {response.status_code}")
                        continue

                    data = response.json()

                    # Parse measurements
                    for result in data.get('value', {}).get('timeSeries', []):
                        site_info = result.get('sourceInfo', {})
                        variable_info = result.get('variable', {})
                        values = result.get('values', [{}])[0].get('value', [])

                        for value_obj in values:
                            measurement = WaterQualityMeasurement(
                                site_no=site_info.get('siteCode', [{}])[0].get('value', ''),
                                site_name=site_info.get('siteName', ''),
                                latitude=float(site_info.get('geoLocation', {}).get('geogLocation', {}).get('latitude', 0)),
                                longitude=float(site_info.get('geoLocation', {}).get('geogLocation', {}).get('longitude', 0)),
                                datetime=value_obj.get('dateTime', ''),
                                parameter_code=variable_info.get('variableCode', [{}])[0].get('value', ''),
                                parameter_name=variable_info.get('variableDescription', ''),
                                value=float(value_obj.get('value', 0)),
                                units=variable_info.get('unit', {}).get('unitCode', ''),
                                qualifier=value_obj.get('qualifiers', [''])[0],
                                site_type=site_info.get('siteTypeCd', '')
                            )
                            measurements.append(measurement)

                    # Rate limiting
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Error fetching parameter {param_code}: {e}")
                    continue

            logger.info(f"Processed {i+len(site_chunk)}/{len(site_numbers)} sites...")

        logger.info(f"Retrieved {len(measurements)} water quality measurements")
        return measurements

    def get_pesticide_data(
        self,
        state_code: str,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """
        Get pesticide data from USGS Pesticide National Synthesis Project

        Note: This requires downloading bulk data files from USGS

        Args:
            state_code: State abbreviation
            start_year: Start year
            end_year: End year

        Returns:
            DataFrame with pesticide detections
        """
        logger.info(f"Fetching pesticide data for {state_code}, {start_year}-{end_year}...")

        # Note: USGS pesticide data is typically in downloadable datasets
        # https://water.usgs.gov/nawqa/pnsp/usage/maps/

        logger.info("Pesticide data requires bulk file download from USGS")
        logger.info("Visit: https://water.usgs.gov/nawqa/pnsp/")

        # Placeholder for data structure
        return pd.DataFrame()

    def calculate_exposure_metrics(
        self,
        measurements: List[WaterQualityMeasurement]
    ) -> pd.DataFrame:
        """
        Calculate exposure metrics from water quality measurements

        Args:
            measurements: List of measurements

        Returns:
            DataFrame with aggregated metrics
        """
        if not measurements:
            return pd.DataFrame()

        df = pd.DataFrame([asdict(m) for m in measurements])

        # Group by site and parameter
        grouped = df.groupby(['site_no', 'site_name', 'parameter_code', 'parameter_name'])

        metrics = grouped.agg({
            'value': ['mean', 'median', 'max', 'count'],
            'latitude': 'first',
            'longitude': 'first',
            'site_type': 'first'
        }).reset_index()

        # Flatten column names
        metrics.columns = ['_'.join(col).strip('_') for col in metrics.columns.values]

        # Check exceedances for key contaminants
        # EPA MCLs (Maximum Contaminant Levels)
        mcls = {
            '01051': 0.015,  # Lead (mg/L) - action level
            '71900': 0.002,  # Mercury (mg/L)
            '01002': 0.010,  # Arsenic (mg/L)
            '00618': 10.0,   # Nitrate (mg/L as N)
        }

        # Add exceedance flags
        for param_code, mcl in mcls.items():
            param_mask = metrics['parameter_code'] == param_code
            if param_mask.any():
                metrics.loc[param_mask, 'exceeds_mcl'] = metrics.loc[param_mask, 'value_max'] > mcl

        return metrics

    def export_water_quality_data(
        self,
        measurements: List[WaterQualityMeasurement],
        filename: str = "water_quality_data.csv"
    ) -> Path:
        """Export water quality data to CSV"""
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
        filename: str = "water_exposure_metrics.csv"
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
        measurements: List[WaterQualityMeasurement]
    ) -> str:
        """Generate summary statistics"""
        if not measurements:
            return "No data available"

        df = pd.DataFrame([asdict(m) for m in measurements])

        report = []
        report.append("=== USGS Water Quality Data Summary ===\n")

        report.append(f"Total measurements: {len(measurements)}")
        report.append(f"Monitoring sites: {df['site_no'].nunique()}")
        report.append(f"Site types: {df['site_type'].unique().tolist()}")

        report.append("\n=== Parameters ===")
        param_counts = df.groupby('parameter_name').size().sort_values(ascending=False)
        for param, count in param_counts.items():
            report.append(f"{param}: {count} measurements")

        report.append("\n=== Contaminant Levels ===")

        # Heavy metals
        heavy_metals = ['Lead', 'Mercury', 'Arsenic', 'Manganese']
        for metal in heavy_metals:
            metal_data = df[df['parameter_name'].str.contains(metal, case=False, na=False)]
            if not metal_data.empty:
                mean_val = metal_data['value'].mean()
                max_val = metal_data['value'].max()
                units = metal_data['units'].iloc[0]
                report.append(f"{metal}: mean={mean_val:.4f} {units}, max={max_val:.4f} {units}")

        return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(
        description='Pull USGS water quality and pesticide data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get water quality data for California streams
  python usgs_water_puller.py \\
      --data-type water_quality \\
      --state CA \\
      --start-date 2020-01-01 \\
      --end-date 2020-12-31 \\
      --output data/environmental/

  # Get specific contaminants
  python usgs_water_puller.py \\
      --data-type water_quality \\
      --state CA \\
      --parameters 01051 71900 01002 \\
      --start-date 2020-01-01 \\
      --end-date 2020-12-31 \\
      --output data/environmental/
        """
    )

    parser.add_argument(
        '--data-type',
        choices=['water_quality', 'pesticides', 'sites'],
        required=True,
        help='Type of data to retrieve'
    )

    parser.add_argument(
        '--state',
        required=True,
        help='State code (e.g., CA, NY, TX)'
    )

    parser.add_argument(
        '--site-type',
        default='ST',
        choices=['ST', 'LK', 'GW', 'ES'],
        help='Site type (ST=Stream, LK=Lake, GW=Groundwater, ES=Estuary)'
    )

    parser.add_argument(
        '--start-date',
        default='2020-01-01',
        help='Start date YYYY-MM-DD'
    )

    parser.add_argument(
        '--end-date',
        default='2020-12-31',
        help='End date YYYY-MM-DD'
    )

    parser.add_argument(
        '--parameters',
        nargs='+',
        help='Parameter codes to retrieve'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/environmental',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize puller
    puller = USGSWaterPuller(Path(args.output))

    if args.data_type == 'sites':
        # Get monitoring sites
        sites_df = puller.get_sites_in_state(args.state, args.site_type)

        output_file = puller.output_dir / f'usgs_sites_{args.state}.csv'
        sites_df.to_csv(output_file, index=False)

        print(f"\n=== USGS Monitoring Sites ===")
        print(f"Found {len(sites_df)} sites in {args.state}")
        print(f"Saved to {output_file}")

    elif args.data_type == 'water_quality':
        # Get monitoring sites first
        sites_df = puller.get_sites_in_state(args.state, args.site_type)

        if sites_df.empty:
            print("No monitoring sites found")
            return

        # Limit to first 50 sites for demo
        site_numbers = sites_df['site_no'].head(50).tolist()

        # Default parameters: heavy metals and nutrients
        if args.parameters is None:
            parameters = ['01051', '71900', '01002', '00618']  # Lead, Mercury, Arsenic, Nitrate
        else:
            parameters = args.parameters

        # Get water quality data
        measurements = puller.get_water_quality_data(
            site_numbers=site_numbers,
            parameter_codes=parameters,
            start_date=args.start_date,
            end_date=args.end_date
        )

        # Export raw data
        puller.export_water_quality_data(measurements)

        # Calculate and export metrics
        metrics = puller.calculate_exposure_metrics(measurements)
        puller.export_exposure_metrics(metrics)

        # Print summary
        print("\n" + puller.generate_summary_report(measurements))

    elif args.data_type == 'pesticides':
        print("\nPesticide data download")
        print("Visit: https://water.usgs.gov/nawqa/pnsp/")
        print("Download state-level pesticide use and detection data")


if __name__ == '__main__':
    main()