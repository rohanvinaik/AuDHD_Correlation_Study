#!/usr/bin/env python3
"""
Environmental Exposure Geospatial Linking Pipeline

Links participant residential data to environmental exposure databases:
- Air quality (EPA AirNow)
- Water quality (USGS NWIS)
- Traffic exposure (DOT, FAA)
- Green space (Landsat/MODIS NDVI)
- Social environment (Census/ACS)

Calculates cumulative exposure burdens for prenatal, early life, and lifetime windows.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import requests
import logging
import json
from scipy.spatial import distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentalExposureLinker:
    """
    Link participant residential histories to environmental exposure databases

    Integrates with EPA, USGS, DOT, NASA, and Census data sources
    """

    def __init__(self, cache_dir: str = 'data/environmental_cache'):
        """Initialize environmental linker"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Data source configurations
        self.data_sources = {
            'air_quality': {
                'api': 'https://www.airnowapi.org/aq/observation/zipCode/historical/',
                'metrics': ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO', 'AQI'],
                'temporal_resolution': 'daily',
                'spatial_resolution': 'zip_code',
                'requires_api_key': True
            },
            'epa_tri': {
                'api': 'https://data.epa.gov/efservice/tri_facility',
                'metrics': ['total_releases', 'air_releases', 'water_releases',
                           'land_releases', 'fugitive_air'],
                'temporal_resolution': 'annual',
                'spatial_resolution': 'facility_level',
                'requires_api_key': False
            },
            'usgs_water': {
                'api': 'https://waterdata.usgs.gov/nwis',
                'contaminants': ['lead', 'arsenic', 'nitrates', 'fluoride'],
                'temporal_resolution': 'varies',
                'requires_api_key': False
            },
            'census': {
                'api': 'https://api.census.gov/data',
                'metrics': ['poverty_rate', 'median_income', 'unemployment',
                           'education_level', 'crowding'],
                'temporal_resolution': 'annual',
                'spatial_resolution': 'block_group',
                'requires_api_key': False
            }
        }

    def geocode_address(self, address: str, city: str = None,
                       state: str = None, zipcode: str = None) -> Optional[Dict]:
        """
        Geocode address to latitude/longitude using Census API

        Args:
            address: Street address
            city: City name
            state: State abbreviation
            zipcode: ZIP code

        Returns:
            dict with lat, lon, and full address components
        """
        # Try full address first
        if address and city and state:
            url = "https://geocoding.geo.census.gov/geocoder/locations/address"
            params = {
                'street': address,
                'city': city,
                'state': state,
                'benchmark': 'Public_AR_Current',
                'format': 'json'
            }

            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('result', {}).get('addressMatches'):
                        match = data['result']['addressMatches'][0]
                        coords = match['coordinates']
                        return {
                            'latitude': coords['y'],
                            'longitude': coords['x'],
                            'matched_address': match['matchedAddress'],
                            'geocode_quality': 'address_match'
                        }
            except Exception as e:
                logger.warning(f"Address geocoding failed: {e}")

        # Fall back to ZIP code centroid
        if zipcode:
            return self.geocode_zipcode(zipcode)

        return None

    def geocode_zipcode(self, zipcode: str) -> Optional[Dict]:
        """Geocode ZIP code to centroid"""
        zipcode = str(zipcode).strip().split('-')[0].zfill(5)

        url = "https://geocoding.geo.census.gov/geocoder/locations/address"
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
                        'zipcode': zipcode,
                        'geocode_quality': 'zipcode_centroid'
                    }
        except Exception as e:
            logger.warning(f"ZIP geocoding failed for {zipcode}: {e}")

        return None

    def link_air_quality(self, locations: pd.DataFrame,
                        date_range: Tuple[str, str],
                        api_key: Optional[str] = None) -> pd.DataFrame:
        """
        Link locations to EPA AirNow air quality data

        Args:
            locations: DataFrame with columns ['participant_id', 'zipcode', 'start_date', 'end_date']
            date_range: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
            api_key: EPA AirNow API key (get from https://docs.airnowapi.org/)

        Returns:
            DataFrame with daily air quality metrics
        """
        if not api_key:
            logger.warning("EPA AirNow API key required. Returning template.")
            return pd.DataFrame()

        results = []

        for _, row in locations.iterrows():
            zipcode = str(row['zipcode']).zfill(5)
            start = pd.to_datetime(row.get('start_date', date_range[0]))
            end = pd.to_datetime(row.get('end_date', date_range[1]))

            # Query daily data
            current = start
            while current <= end:
                date_str = current.strftime('%Y-%m-%d')

                url = f"https://www.airnowapi.org/aq/observation/zipCode/historical/"
                params = {
                    'format': 'application/json',
                    'zipCode': zipcode,
                    'date': date_str,
                    'API_KEY': api_key
                }

                try:
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()

                        # Parse pollutant data
                        daily_data = {
                            'participant_id': row['participant_id'],
                            'date': date_str,
                            'zipcode': zipcode
                        }

                        for obs in data:
                            param = obs.get('ParameterName', '')
                            value = obs.get('AQI', np.nan)
                            daily_data[f"{param}_AQI"] = value

                        results.append(daily_data)

                except Exception as e:
                    logger.warning(f"Failed to get air quality for {zipcode}, {date_str}: {e}")

                current += timedelta(days=1)

        return pd.DataFrame(results)

    def link_epa_toxics_release(self, locations: pd.DataFrame,
                                radius_miles: float = 5.0,
                                year: int = 2020) -> pd.DataFrame:
        """
        Link locations to EPA Toxics Release Inventory (TRI)

        Identifies nearby industrial facilities and their chemical releases

        Args:
            locations: DataFrame with geocoded locations
            radius_miles: Search radius in miles
            year: TRI reporting year

        Returns:
            DataFrame with facility counts and release totals
        """
        results = []

        for _, row in locations.iterrows():
            if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
                continue

            # Query TRI facilities by ZIP prefix (faster than lat/lon)
            zipcode = str(row.get('zipcode', '')).zfill(5)
            zip_prefix = zipcode[:3]

            url = f"https://data.epa.gov/efservice/tri_facility/zip/beginning/{zip_prefix}/JSON"

            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    facilities = response.json()

                    # Filter by distance
                    nearby = []
                    for facility in facilities:
                        if 'latitude' in facility and 'longitude' in facility:
                            fac_lat = float(facility['latitude'])
                            fac_lon = float(facility['longitude'])

                            dist = self._haversine_distance(
                                row['latitude'], row['longitude'],
                                fac_lat, fac_lon
                            )

                            if dist <= radius_miles:
                                nearby.append({
                                    'distance_miles': dist,
                                    'facility_name': facility.get('facility_name', ''),
                                    'industry_sector': facility.get('industry_sector', ''),
                                    'releases': facility.get('total_releases', 0)
                                })

                    # Summarize
                    results.append({
                        'participant_id': row['participant_id'],
                        'n_facilities': len(nearby),
                        'total_releases_lbs': sum(f['releases'] for f in nearby),
                        'mean_distance_miles': np.mean([f['distance_miles'] for f in nearby]) if nearby else np.nan,
                        'closest_facility_miles': min([f['distance_miles'] for f in nearby]) if nearby else np.nan
                    })

            except Exception as e:
                logger.warning(f"TRI query failed for {row['participant_id']}: {e}")

        return pd.DataFrame(results)

    def link_traffic_exposure(self, locations: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate traffic exposure based on proximity to major roads

        Uses OpenStreetMap or local DOT data
        Calculates distance to highways and traffic density in buffer zones

        Args:
            locations: DataFrame with geocoded locations

        Returns:
            DataFrame with traffic exposure metrics
        """
        logger.warning("Traffic exposure linking requires OSM or DOT data - not yet implemented")

        # Placeholder for traffic metrics
        results = []
        for _, row in locations.iterrows():
            results.append({
                'participant_id': row['participant_id'],
                'distance_to_highway_m': np.nan,
                'traffic_density_500m': np.nan,
                'major_road_length_300m': np.nan
            })

        return pd.DataFrame(results)

    def link_green_space(self, locations: pd.DataFrame,
                        buffer_m: int = 500) -> pd.DataFrame:
        """
        Calculate green space exposure using NDVI

        Would use NASA MODIS or Landsat data
        Calculates mean NDVI in buffer around residence

        Args:
            locations: DataFrame with geocoded locations
            buffer_m: Buffer radius in meters

        Returns:
            DataFrame with NDVI metrics
        """
        logger.warning("Green space (NDVI) linking requires satellite data - not yet implemented")

        # Placeholder
        results = []
        for _, row in locations.iterrows():
            results.append({
                'participant_id': row['participant_id'],
                'ndvi_mean_500m': np.nan,
                'park_access_1km': np.nan,
                'tree_canopy_pct': np.nan
            })

        return pd.DataFrame(results)

    def link_census_demographics(self, locations: pd.DataFrame,
                                 year: int = 2020) -> pd.DataFrame:
        """
        Link to Census/ACS neighborhood demographics

        Provides socioeconomic context (poverty, income, education, etc.)

        Args:
            locations: DataFrame with geocoded locations
            year: Census year (2010, 2020) or ACS year

        Returns:
            DataFrame with neighborhood SES metrics
        """
        logger.warning("Census linking requires Census API setup - returning placeholders")

        # Would use Census API to get block group data
        results = []
        for _, row in locations.iterrows():
            results.append({
                'participant_id': row['participant_id'],
                'poverty_rate': np.nan,
                'median_income': np.nan,
                'pct_college': np.nan,
                'unemployment_rate': np.nan,
                'crowding_index': np.nan
            })

        return pd.DataFrame(results)

    def _haversine_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points on Earth in miles

        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)

        Returns:
            distance: Distance in miles
        """
        R = 3959  # Earth radius in miles

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def calculate_exposure_windows(self, exposures: pd.DataFrame,
                                  birth_dates: pd.DataFrame,
                                  windows: Dict[str, Tuple] = None) -> pd.DataFrame:
        """
        Calculate cumulative exposures for specific time windows

        Args:
            exposures: DataFrame with daily/annual exposure data
            birth_dates: DataFrame with participant_id and birth_date
            windows: Dict of window names and (start_days, end_days) relative to birth
                    Default: prenatal (-280, 0), early_life (0, 730), childhood (730, 4380)

        Returns:
            DataFrame with windowed exposure summaries
        """
        if windows is None:
            windows = {
                'prenatal': (-280, 0),
                'early_life': (0, 730),  # 0-2 years
                'childhood': (730, 4380),  # 2-12 years
                'cumulative': (0, None)  # Birth to present
            }

        # Merge birth dates
        exposures = exposures.merge(birth_dates, on='participant_id', how='left')
        exposures['date'] = pd.to_datetime(exposures['date'])
        exposures['birth_date'] = pd.to_datetime(exposures['birth_date'])
        exposures['days_from_birth'] = (exposures['date'] - exposures['birth_date']).dt.days

        # Calculate for each window
        windowed = []

        for participant in exposures['participant_id'].unique():
            p_data = exposures[exposures['participant_id'] == participant]

            window_metrics = {'participant_id': participant}

            for window_name, (start, end) in windows.items():
                if end is None:
                    end = p_data['days_from_birth'].max()

                window_data = p_data[
                    (p_data['days_from_birth'] >= start) &
                    (p_data['days_from_birth'] <= end)
                ]

                # Aggregate metrics
                for col in window_data.columns:
                    if col.endswith('_AQI') or 'release' in col or 'PM' in col:
                        window_metrics[f"{col}_{window_name}_mean"] = window_data[col].mean()
                        window_metrics[f"{col}_{window_name}_max"] = window_data[col].max()

            windowed.append(window_metrics)

        return pd.DataFrame(windowed)

    def calculate_cumulative_burden(self, exposures: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative environmental burden score

        Combines multiple exposure types into single burden index

        Args:
            exposures: DataFrame with all exposure metrics

        Returns:
            DataFrame with burden scores
        """
        burden_scores = []

        for participant in exposures['participant_id'].unique():
            p_data = exposures[exposures['participant_id'] == participant]

            # Weight different exposures (example weights)
            weights = {
                'air_quality': 0.3,
                'toxics_release': 0.25,
                'traffic': 0.15,
                'water_quality': 0.15,
                'low_ses': 0.15
            }

            # Calculate z-scores and weighted sum
            burden = 0

            # Air quality burden (higher AQI = worse)
            if 'PM2.5_AQI' in p_data.columns:
                pm25 = p_data['PM2.5_AQI'].mean()
                burden += weights['air_quality'] * (pm25 / 100)  # Normalize to 0-1

            # TRI burden
            if 'total_releases_lbs' in p_data.columns:
                releases = p_data['total_releases_lbs'].mean()
                burden += weights['toxics_release'] * min(releases / 10000, 1)  # Cap at 1

            burden_scores.append({
                'participant_id': participant,
                'environmental_burden_score': burden,
                'burden_category': 'high' if burden > 0.7 else 'medium' if burden > 0.4 else 'low'
            })

        return pd.DataFrame(burden_scores)

    def create_exposure_matrix(self, locations: pd.DataFrame,
                              birth_dates: pd.DataFrame,
                              date_range: Tuple[str, str],
                              api_keys: Dict[str, str] = None) -> pd.DataFrame:
        """
        Complete pipeline: link all environmental exposures

        Args:
            locations: Participant residential data
            birth_dates: Participant birth dates for windowing
            date_range: Overall date range to query
            api_keys: Dict of API keys (airnow, etc.)

        Returns:
            Complete exposure matrix
        """
        logger.info("Starting comprehensive environmental linkage...")

        if api_keys is None:
            api_keys = {}

        # Geocode if needed
        if 'latitude' not in locations.columns:
            logger.info("Geocoding addresses...")
            geocoded = []
            for _, row in locations.iterrows():
                coords = self.geocode_zipcode(row.get('zipcode', ''))
                if coords:
                    geocoded.append({
                        'participant_id': row['participant_id'],
                        'latitude': coords['latitude'],
                        'longitude': coords['longitude'],
                        'zipcode': row.get('zipcode')
                    })
            locations = pd.DataFrame(geocoded)

        # Link each exposure type
        exposure_data = locations[['participant_id', 'zipcode', 'latitude', 'longitude']].copy()

        # Air quality
        if 'airnow' in api_keys:
            logger.info("Linking air quality data...")
            air_quality = self.link_air_quality(locations, date_range, api_keys['airnow'])
            if not air_quality.empty:
                exposure_data = exposure_data.merge(air_quality, on='participant_id', how='left')

        # EPA TRI
        logger.info("Linking EPA Toxics Release...")
        tri_data = self.link_epa_toxics_release(locations, radius_miles=5.0)
        if not tri_data.empty:
            exposure_data = exposure_data.merge(tri_data, on='participant_id', how='left')

        # Traffic (placeholder)
        traffic_data = self.link_traffic_exposure(locations)
        if not traffic_data.empty:
            exposure_data = exposure_data.merge(traffic_data, on='participant_id', how='left')

        # Green space (placeholder)
        green_data = self.link_green_space(locations)
        if not green_data.empty:
            exposure_data = exposure_data.merge(green_data, on='participant_id', how='left')

        # Census demographics (placeholder)
        census_data = self.link_census_demographics(locations)
        if not census_data.empty:
            exposure_data = exposure_data.merge(census_data, on='participant_id', how='left')

        # Calculate burden scores
        burden = self.calculate_cumulative_burden(exposure_data)
        exposure_data = exposure_data.merge(burden, on='participant_id', how='left')

        logger.info(f"✓ Environmental linkage complete: {len(exposure_data)} participants")

        return exposure_data


if __name__ == '__main__':
    # Example usage
    logger.info("Environmental Exposure Linker initialized")

    # Simulate participant data
    participants = pd.DataFrame({
        'participant_id': ['P001', 'P002', 'P003'],
        'zipcode': ['10001', '94105', '60601'],
        'address': ['123 Main St', '456 Market St', '789 Michigan Ave'],
        'city': ['New York', 'San Francisco', 'Chicago'],
        'state': ['NY', 'CA', 'IL']
    })

    # Initialize linker
    linker = EnvironmentalExposureLinker()

    # Geocode
    logger.info("\nGeocoding example addresses...")
    for _, row in participants.iterrows():
        coords = linker.geocode_zipcode(row['zipcode'])
        if coords:
            print(f"{row['participant_id']}: {coords['latitude']:.4f}, {coords['longitude']:.4f}")

    # Demo TRI linkage (works without API key)
    logger.info("\nDemo: EPA Toxics Release Inventory linkage...")
    participants['latitude'] = [40.7589, 37.7902, 41.8821]
    participants['longitude'] = [-73.9851, -122.4013, -87.6278]

    tri_data = linker.link_epa_toxics_release(participants, radius_miles=5.0)
    if not tri_data.empty:
        print("\nTRI Results:")
        print(tri_data)