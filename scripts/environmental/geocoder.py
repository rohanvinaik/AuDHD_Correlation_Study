#!/usr/bin/env python3
"""
Geocoder for Address to Coordinate Conversion

Converts addresses, ZIP codes, and place names to geographic coordinates
for linking environmental exposure data to participant locations.

Features:
- US Census Geocoding API (free, no key required)
- Batch geocoding support
- Coordinate to census tract mapping
- Distance calculations between points
- Privacy-preserving coordinate fuzzing

Requirements:
    pip install requests pandas geopy

Usage:
    # Geocode addresses from CSV
    python geocoder.py \\
        --input participants.csv \\
        --address-column address \\
        --output data/environmental/geocoded.csv

    # Geocode ZIP codes
    python geocoder.py \\
        --input zipcodes.csv \\
        --zip-column zip \\
        --output data/environmental/geocoded.csv

    # Add census tract info
    python geocoder.py \\
        --input geocoded.csv \\
        --add-census-tract \\
        --output data/environmental/geocoded_with_tract.csv

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
import random
import math

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


# US Census Geocoding API (free, no key required)
CENSUS_GEOCODE_URL = "https://geocoding.geo.census.gov/geocoder"

# ZIP code centroid database (ZCTA - ZIP Code Tabulation Area)
# Note: For production, download full ZCTA shapefile from Census


@dataclass
class GeocodedAddress:
    """Geocoded address result"""
    original_address: str
    matched_address: str
    latitude: float
    longitude: float
    match_quality: str  # Exact, Non_Exact, Tie, No_Match
    census_tract: Optional[str] = None
    census_block: Optional[str] = None
    state_fips: Optional[str] = None
    county_fips: Optional[str] = None


@dataclass
class CensusTract:
    """Census tract information"""
    geoid: str  # 11-digit GEOID (state+county+tract)
    state_fips: str
    county_fips: str
    tract_code: str
    name: str
    latitude: float
    longitude: float


class Geocoder:
    """Geocode addresses and link to geographic data"""

    def __init__(self, output_dir: Path):
        """
        Initialize geocoder

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AuDHD-Research-Pipeline/1.0'
        })

        # Cache for geocoded addresses
        self.geocode_cache = {}

        logger.info(f"Initialized geocoder: {output_dir}")

    def geocode_address(
        self,
        address: str,
        benchmark: str = 'Public_AR_Current',
        vintage: str = 'Current_Current'
    ) -> Optional[GeocodedAddress]:
        """
        Geocode a single address using US Census API

        Args:
            address: Full address string
            benchmark: Census benchmark dataset
            vintage: Vintage year

        Returns:
            GeocodedAddress or None
        """
        # Check cache
        if address in self.geocode_cache:
            return self.geocode_cache[address]

        try:
            url = f"{CENSUS_GEOCODE_URL}/locations/onelineaddress"

            params = {
                'address': address,
                'benchmark': benchmark,
                'vintage': vintage,
                'format': 'json'
            }

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse response
            result = data.get('result', {})
            address_matches = result.get('addressMatches', [])

            if not address_matches:
                logger.warning(f"No match found for: {address}")
                return None

            # Take first match
            match = address_matches[0]
            coords = match.get('coordinates', {})
            geo_data = match.get('geographies', {}).get('Census Tracts', [{}])[0]

            geocoded = GeocodedAddress(
                original_address=address,
                matched_address=match.get('matchedAddress', ''),
                latitude=float(coords.get('y', 0)),
                longitude=float(coords.get('x', 0)),
                match_quality=match.get('matchType', 'Unknown'),
                census_tract=geo_data.get('GEOID'),
                census_block=geo_data.get('BLKGRP'),
                state_fips=geo_data.get('STATE'),
                county_fips=geo_data.get('COUNTY')
            )

            # Cache result
            self.geocode_cache[address] = geocoded

            # Rate limiting
            time.sleep(0.5)

            return geocoded

        except Exception as e:
            logger.error(f"Error geocoding {address}: {e}")
            return None

    def geocode_zip(
        self,
        zip_code: str
    ) -> Optional[Tuple[float, float]]:
        """
        Get approximate coordinates for ZIP code

        Args:
            zip_code: 5-digit ZIP code

        Returns:
            (latitude, longitude) or None
        """
        # Note: This is simplified. For production, use ZCTA shapefile
        # or commercial ZIP code database

        # Check cache
        cache_key = f"ZIP_{zip_code}"
        if cache_key in self.geocode_cache:
            cached = self.geocode_cache[cache_key]
            return (cached.latitude, cached.longitude)

        # Use Census ZIP Code Tabulation Area API or approximation
        logger.info(f"ZIP code geocoding: {zip_code}")
        logger.info("Note: For production, use full ZCTA shapefile from Census")

        # Placeholder: Return None for now
        # In production, implement ZIP code centroid lookup
        return None

    def batch_geocode_addresses(
        self,
        addresses: List[str],
        max_batch: int = 100
    ) -> List[GeocodedAddress]:
        """
        Geocode multiple addresses

        Args:
            addresses: List of address strings
            max_batch: Maximum batch size (Census API limits)

        Returns:
            List of GeocodedAddress results
        """
        logger.info(f"Batch geocoding {len(addresses)} addresses...")

        results = []

        for i, address in enumerate(addresses):
            if i % 10 == 0:
                logger.info(f"Geocoded {i}/{len(addresses)}...")

            result = self.geocode_address(address)
            if result:
                results.append(result)

            # Rate limiting
            time.sleep(0.5)

        logger.info(f"Successfully geocoded {len(results)}/{len(addresses)} addresses")
        return results

    def get_census_tract(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[CensusTract]:
        """
        Get census tract for coordinates

        Args:
            latitude: Latitude
            longitude: Longitude

        Returns:
            CensusTract or None
        """
        try:
            url = f"{CENSUS_GEOCODE_URL}/geographies/coordinates"

            params = {
                'x': longitude,
                'y': latitude,
                'benchmark': 'Public_AR_Current',
                'vintage': 'Current_Current',
                'format': 'json'
            }

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse census tract info
            result = data.get('result', {})
            tracts = result.get('geographies', {}).get('Census Tracts', [])

            if not tracts:
                return None

            tract_data = tracts[0]

            tract = CensusTract(
                geoid=tract_data.get('GEOID'),
                state_fips=tract_data.get('STATE'),
                county_fips=tract_data.get('COUNTY'),
                tract_code=tract_data.get('TRACT'),
                name=tract_data.get('NAME'),
                latitude=float(tract_data.get('CENTLAT', latitude)),
                longitude=float(tract_data.get('CENTLON', longitude))
            )

            return tract

        except Exception as e:
            logger.error(f"Error getting census tract for {latitude},{longitude}: {e}")
            return None

    def calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two points (Haversine formula)

        Args:
            lat1, lon1: First point
            lat2, lon2: Second point

        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth radius in km

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat/2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon/2) ** 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        distance = R * c

        return distance

    def fuzz_coordinates(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 1.0
    ) -> Tuple[float, float]:
        """
        Add random noise to coordinates for privacy

        Args:
            latitude: Original latitude
            longitude: Original longitude
            radius_km: Maximum fuzzing radius in km

        Returns:
            (fuzzed_latitude, fuzzed_longitude)
        """
        # Convert km to degrees (approximate)
        lat_offset = (random.random() - 0.5) * 2 * (radius_km / 111.0)
        lon_offset = (random.random() - 0.5) * 2 * (radius_km / (111.0 * math.cos(math.radians(latitude))))

        fuzzed_lat = latitude + lat_offset
        fuzzed_lon = longitude + lon_offset

        return (fuzzed_lat, fuzzed_lon)

    def geocode_dataframe(
        self,
        df: pd.DataFrame,
        address_column: Optional[str] = None,
        zip_column: Optional[str] = None,
        fuzz_km: float = 0.0
    ) -> pd.DataFrame:
        """
        Geocode addresses in DataFrame

        Args:
            df: Input DataFrame
            address_column: Column with addresses
            zip_column: Column with ZIP codes
            fuzz_km: Fuzzing radius for privacy (0 = no fuzzing)

        Returns:
            DataFrame with added latitude/longitude columns
        """
        logger.info(f"Geocoding {len(df)} records...")

        results = []

        for idx, row in df.iterrows():
            if idx % 10 == 0:
                logger.info(f"Processing {idx}/{len(df)}...")

            if address_column and pd.notna(row.get(address_column)):
                address = row[address_column]
                geocoded = self.geocode_address(address)

                if geocoded:
                    lat, lon = geocoded.latitude, geocoded.longitude

                    # Apply fuzzing if requested
                    if fuzz_km > 0:
                        lat, lon = self.fuzz_coordinates(lat, lon, fuzz_km)

                    results.append({
                        'index': idx,
                        'latitude': lat,
                        'longitude': lon,
                        'matched_address': geocoded.matched_address,
                        'match_quality': geocoded.match_quality,
                        'census_tract': geocoded.census_tract,
                        'state_fips': geocoded.state_fips,
                        'county_fips': geocoded.county_fips
                    })
                else:
                    results.append({
                        'index': idx,
                        'latitude': None,
                        'longitude': None,
                        'matched_address': None,
                        'match_quality': 'No_Match',
                        'census_tract': None,
                        'state_fips': None,
                        'county_fips': None
                    })

            elif zip_column and pd.notna(row.get(zip_column)):
                zip_code = str(row[zip_column]).zfill(5)
                coords = self.geocode_zip(zip_code)

                if coords:
                    lat, lon = coords

                    if fuzz_km > 0:
                        lat, lon = self.fuzz_coordinates(lat, lon, fuzz_km)

                    results.append({
                        'index': idx,
                        'latitude': lat,
                        'longitude': lon,
                        'matched_address': f"ZIP {zip_code}",
                        'match_quality': 'ZIP_Centroid',
                        'census_tract': None,
                        'state_fips': None,
                        'county_fips': None
                    })
                else:
                    results.append({
                        'index': idx,
                        'latitude': None,
                        'longitude': None,
                        'matched_address': None,
                        'match_quality': 'No_Match',
                        'census_tract': None,
                        'state_fips': None,
                        'county_fips': None
                    })

        # Merge results back to DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.set_index('index')

        output_df = df.copy()
        for col in ['latitude', 'longitude', 'matched_address', 'match_quality',
                    'census_tract', 'state_fips', 'county_fips']:
            output_df[col] = results_df[col]

        return output_df

    def export_geocoded_data(
        self,
        df: pd.DataFrame,
        filename: str = "geocoded_data.csv"
    ) -> Path:
        """Export geocoded data to CSV"""
        output_file = self.output_dir / filename
        df.to_csv(output_file, index=False)

        logger.info(f"Exported geocoded data to {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Geocode addresses and link to census geography',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Geocode addresses from CSV
  python geocoder.py \\
      --input participants.csv \\
      --address-column address \\
      --output data/environmental/geocoded.csv

  # Geocode with privacy fuzzing (1km radius)
  python geocoder.py \\
      --input participants.csv \\
      --address-column address \\
      --fuzz-km 1.0 \\
      --output data/environmental/geocoded.csv

  # Geocode ZIP codes
  python geocoder.py \\
      --input data.csv \\
      --zip-column zip_code \\
      --output data/environmental/geocoded.csv
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file'
    )

    parser.add_argument(
        '--address-column',
        help='Column name with addresses'
    )

    parser.add_argument(
        '--zip-column',
        help='Column name with ZIP codes'
    )

    parser.add_argument(
        '--fuzz-km',
        type=float,
        default=0.0,
        help='Fuzzing radius in km for privacy (0 = no fuzzing)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/environmental/geocoded.csv',
        help='Output file path'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.address_column and not args.zip_column:
        print("Error: Must specify either --address-column or --zip-column")
        sys.exit(1)

    # Load input data
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} records from {args.input}")

    # Initialize geocoder
    output_dir = Path(args.output).parent
    geocoder = Geocoder(output_dir)

    # Geocode
    geocoded_df = geocoder.geocode_dataframe(
        df,
        address_column=args.address_column,
        zip_column=args.zip_column,
        fuzz_km=args.fuzz_km
    )

    # Export
    geocoder.export_geocoded_data(geocoded_df, Path(args.output).name)

    # Print summary
    print(f"\n=== Geocoding Summary ===")
    print(f"Total records: {len(geocoded_df)}")
    print(f"Successfully geocoded: {geocoded_df['latitude'].notna().sum()}")
    print(f"Failed to geocode: {geocoded_df['latitude'].isna().sum()}")

    if args.fuzz_km > 0:
        print(f"Applied privacy fuzzing: {args.fuzz_km} km radius")

    print(f"\nOutput saved to: {args.output}")


if __name__ == '__main__':
    main()