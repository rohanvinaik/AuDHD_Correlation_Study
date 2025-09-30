#!/usr/bin/env python3
"""
Environmental Data Linkage System
Links participant data to external environmental databases using
zip codes, dates, and GPS coordinates
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional
import requests
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnvironmentalDataLinker:
    """Link participant data to environmental exposures"""

    def __init__(self, output_dir='data/environmental_linked'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # External data sources
        self.data_sources = {
            'air_quality': {
                'name': 'EPA AQI (Air Quality Index)',
                'temporal_resolution': 'daily',
                'spatial_resolution': 'zip code / county',
                'api': 'https://www.airnowapi.org',
                'requires_key': True,
                'free': True,
                'variables': ['PM2.5', 'PM10', 'Ozone', 'NO2', 'SO2', 'CO', 'AQI']
            },
            'epa_toxics': {
                'name': 'EPA TRI (Toxics Release Inventory)',
                'temporal_resolution': 'annual',
                'spatial_resolution': 'facility-level',
                'api': 'https://data.epa.gov/efservice/',
                'requires_key': False,
                'free': True,
                'variables': ['Chemical releases', 'Distance to facility']
            },
            'traffic': {
                'name': 'DOT Traffic Volume',
                'temporal_resolution': 'annual',
                'spatial_resolution': 'road segment',
                'api': None,  # State-specific
                'requires_key': False,
                'free': True,
                'variables': ['AADT (Annual Average Daily Traffic)', 'Truck percentage']
            },
            'green_space': {
                'name': 'NDVI (Normalized Difference Vegetation Index)',
                'temporal_resolution': '16-day',
                'spatial_resolution': '250m-1km',
                'api': 'NASA MODIS/Landsat',
                'requires_key': True,
                'free': True,
                'variables': ['NDVI', 'Green space percentage']
            },
            'noise': {
                'name': 'DOT/FAA Noise Maps',
                'temporal_resolution': 'static',
                'spatial_resolution': 'address-level',
                'api': None,
                'requires_key': False,
                'free': True,
                'variables': ['Aviation noise (dBA)', 'Highway noise (dBA)']
            },
            'water_quality': {
                'name': 'USGS Water Quality',
                'temporal_resolution': 'varies',
                'spatial_resolution': 'monitoring site',
                'api': 'https://waterdata.usgs.gov/nwis',
                'requires_key': False,
                'free': True,
                'variables': ['Contaminants', 'Water source type']
            },
            'walkability': {
                'name': 'EPA Walkability Index',
                'temporal_resolution': 'static (decennial)',
                'spatial_resolution': 'block group',
                'api': None,
                'requires_key': False,
                'free': True,
                'variables': ['Walkability score', 'Transit access']
            }
        }

    def geocode_zipcode(self, zipcode: str) -> Optional[Dict]:
        """
        Convert zip code to lat/long using Census API (free, no key needed)
        """
        if not zipcode or pd.isna(zipcode):
            return None

        # Clean zip code
        zipcode = str(zipcode).strip().split('-')[0].zfill(5)

        # Use free Census geocoder
        url = f"https://geocoding.geo.census.gov/geocoder/locations/address"
        params = {
            'zip': zipcode,
            'benchmark': 'Public_AR_Current',
            'format': 'json'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('result', {}).get('addressMatches'):
                    coords = data['result']['addressMatches'][0]['coordinates']
                    return {
                        'latitude': coords['y'],
                        'longitude': coords['x'],
                        'zipcode': zipcode
                    }
        except Exception as e:
            logger.warning(f"Geocoding failed for {zipcode}: {e}")

        return None

    def link_epa_air_quality(self,
                           participant_data: pd.DataFrame,
                           zip_col: str = 'zipcode',
                           date_col: str = 'visit_date',
                           api_key: Optional[str] = None) -> pd.DataFrame:
        """
        Link to EPA AirNow API for air quality data

        Note: Requires free API key from https://docs.airnowapi.org/
        """
        logger.info("Linking EPA air quality data...")

        if not api_key:
            logger.warning("EPA AirNow API key not provided. "
                         "Get free key at: https://docs.airnowapi.org/")
            logger.warning("Generating placeholder linkage instructions...")

            # Create linkage template
            linkage_template = participant_data[[zip_col, date_col]].copy()
            linkage_template['AQI'] = np.nan
            linkage_template['PM2.5'] = np.nan
            linkage_template['Ozone'] = np.nan
            linkage_template['linkage_status'] = 'NEEDS_API_KEY'

            return linkage_template

        # If API key provided, do actual linking
        linked_data = []

        for idx, row in participant_data.iterrows():
            zipcode = row.get(zip_col)
            date = row.get(date_col)

            if pd.notna(zipcode) and pd.notna(date):
                # API call
                url = "https://www.airnowapi.org/aq/observation/zipCode/historical/"
                params = {
                    'zipCode': zipcode,
                    'date': date,
                    'API_KEY': api_key
                }

                try:
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        # Process air quality data
                        linked_data.append({
                            'participant_id': row.get('participant_id'),
                            'zipcode': zipcode,
                            'date': date,
                            'aqi_data': data
                        })
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Failed to get AQI for {zipcode}, {date}: {e}")

        return pd.DataFrame(linked_data)

    def link_epa_toxics_release(self,
                               participant_data: pd.DataFrame,
                               zip_col: str = 'zipcode',
                               radius_miles: float = 5.0) -> pd.DataFrame:
        """
        Link to EPA Toxics Release Inventory
        Identifies nearby industrial chemical releases
        """
        logger.info(f"Linking EPA TRI data (radius: {radius_miles} miles)...")

        linked_data = []

        for idx, row in participant_data.iterrows():
            zipcode = row.get(zip_col)

            if pd.notna(zipcode):
                # Geocode participant location
                coords = self.geocode_zipcode(str(zipcode))

                if coords:
                    # Query EPA TRI API (no key needed!)
                    url = "https://data.epa.gov/efservice/tri_facility/zip/beginning/"
                    url += f"{str(zipcode)[:3]}/JSON"

                    try:
                        response = requests.get(url, timeout=15)
                        if response.status_code == 200:
                            facilities = response.json()

                            # Calculate distance to each facility
                            # (simplified - should use proper haversine)
                            nearby = [f for f in facilities
                                    if self._rough_distance(coords, f) < radius_miles]

                            linked_data.append({
                                'participant_id': row.get('participant_id'),
                                'zipcode': zipcode,
                                'num_facilities_nearby': len(nearby),
                                'total_releases_lbs': sum(f.get('total_releases', 0)
                                                        for f in nearby),
                                'facilities': nearby[:5]  # Top 5
                            })

                        time.sleep(0.5)
                    except Exception as e:
                        logger.warning(f"TRI query failed for {zipcode}: {e}")

        return pd.DataFrame(linked_data)

    def _rough_distance(self, coords1: Dict, coords2: Dict) -> float:
        """Rough distance calculation (replace with haversine for accuracy)"""
        lat1, lon1 = coords1['latitude'], coords1['longitude']
        lat2 = coords2.get('latitude', coords2.get('lat', 0))
        lon2 = coords2.get('longitude', coords2.get('lon', 0))

        # Very rough approximation (1 degree ≈ 69 miles)
        return np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2) * 69

    def generate_linkage_plan(self,
                            participant_data: pd.DataFrame,
                            available_fields: List[str]) -> Dict:
        """
        Generate a comprehensive linkage plan based on available data
        """
        logger.info("Generating environmental linkage plan...")

        plan = {
            'generation_date': datetime.now().isoformat(),
            'n_participants': len(participant_data),
            'available_fields': available_fields,
            'linkage_strategies': []
        }

        # Check what we can link
        has_zipcode = any('zip' in f.lower() for f in available_fields)
        has_address = any('address' in f.lower() for f in available_fields)
        has_coords = any('lat' in f.lower() or 'lon' in f.lower()
                        for f in available_fields)
        has_date = any('date' in f.lower() or 'time' in f.lower()
                      for f in available_fields)

        if has_zipcode or has_address:
            plan['linkage_strategies'].append({
                'data_source': 'EPA Air Quality (AirNow)',
                'method': 'Zip code + date matching',
                'feasibility': 'HIGH' if has_date else 'MEDIUM',
                'api_key_needed': True,
                'cost': 'Free',
                'steps': [
                    '1. Get free API key: https://docs.airnowapi.org/',
                    '2. Use link_epa_air_quality() function',
                    '3. Will provide daily PM2.5, Ozone, AQI'
                ]
            })

            plan['linkage_strategies'].append({
                'data_source': 'EPA Toxics Release Inventory',
                'method': 'Zip code geocoding + proximity',
                'feasibility': 'HIGH',
                'api_key_needed': False,
                'cost': 'Free',
                'steps': [
                    '1. Geocode zip codes (automatic)',
                    '2. Use link_epa_toxics_release() function',
                    '3. Will provide nearby industrial chemical releases'
                ]
            })

        if has_coords:
            plan['linkage_strategies'].append({
                'data_source': 'NDVI (Green Space)',
                'method': 'GPS coordinates + satellite imagery',
                'feasibility': 'MEDIUM',
                'api_key_needed': True,
                'cost': 'Free (NASA EarthData account)',
                'steps': [
                    '1. Register at https://urs.earthdata.nasa.gov/',
                    '2. Download MODIS NDVI data',
                    '3. Extract values at participant coordinates'
                ]
            })

        # Add more strategies...

        return plan

    def save_linkage_scripts(self, plan: Dict):
        """Generate ready-to-use linkage scripts"""
        script_file = self.output_dir / 'LINKAGE_INSTRUCTIONS.md'

        with open(script_file, 'w') as f:
            f.write("# Environmental Data Linkage Instructions\n\n")
            f.write(f"Generated: {plan['generation_date']}\n\n")

            f.write(f"## Available Data\n\n")
            f.write(f"- Participants: {plan['n_participants']}\n")
            f.write(f"- Available fields: {', '.join(plan['available_fields'])}\n\n")

            f.write(f"## Recommended Linkage Strategies\n\n")

            for i, strategy in enumerate(plan['linkage_strategies'], 1):
                f.write(f"### {i}. {strategy['data_source']}\n\n")
                f.write(f"- **Method**: {strategy['method']}\n")
                f.write(f"- **Feasibility**: {strategy['feasibility']}\n")
                f.write(f"- **API Key Needed**: {strategy['api_key_needed']}\n")
                f.write(f"- **Cost**: {strategy['cost']}\n\n")
                f.write(f"**Steps:**\n")
                for step in strategy['steps']:
                    f.write(f"{step}\n")
                f.write("\n")

        logger.info(f"Linkage instructions saved to {script_file}")


def main():
    """Example usage"""
    linker = EnvironmentalDataLinker()

    # Example participant data
    example_data = pd.DataFrame({
        'participant_id': [f'P{i:04d}' for i in range(1, 101)],
        'zipcode': np.random.choice(['10001', '94105', '60601', '90001'], 100),
        'visit_date': pd.date_range('2020-01-01', periods=100),
        'latitude': np.random.uniform(30, 45, 100),
        'longitude': np.random.uniform(-120, -70, 100)
    })

    # Generate linkage plan
    plan = linker.generate_linkage_plan(
        example_data,
        available_fields=list(example_data.columns)
    )

    logger.info("\nLinkage Plan Generated:")
    logger.info(f"  {len(plan['linkage_strategies'])} strategies identified")

    # Save instructions
    linker.save_linkage_scripts(plan)

    # Demo: Try EPA TRI linkage (works without API key)
    logger.info("\nDemo: EPA Toxics Release Inventory linkage...")
    tri_data = linker.link_epa_toxics_release(
        example_data.head(5),  # Just first 5 for demo
        radius_miles=5.0
    )

    logger.info(f"Linked {len(tri_data)} participants to TRI data")

    logger.info("\n✓ Environmental linkage system ready!")


if __name__ == '__main__':
    main()