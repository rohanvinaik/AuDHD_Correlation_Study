# Environmental Exposure Data for Geospatial Linkage

Comprehensive system for downloading and linking environmental exposure data to participant locations for ADHD/autism research. Includes air quality, water quality, industrial toxics, and socioeconomic indicators.

## Overview

This system enables:
1. **Air Quality Data**: EPA AQS criteria pollutants and air toxics
2. **Water Quality Data**: USGS NWIS contaminants and nutrients
3. **Industrial Releases**: EPA TRI facility emissions
4. **Geocoding**: Address to coordinate conversion with privacy protection
5. **Exposure Matrices**: Census tract-level integrated exposure estimates

### Key Components

1. **EPA Data Puller** (`epa_data_puller.py`)
   - Downloads air quality data from EPA AQS API
   - Retrieves TRI facility information
   - Calculates exposure metrics
   - Tracks neurotoxic pollutants (PM2.5, benzene, lead, etc.)

2. **USGS Water Puller** (`usgs_water_puller.py`)
   - Downloads water quality data from USGS NWIS
   - Tracks heavy metals, pesticides, nutrients
   - Identifies MCL exceedances
   - Maps monitoring sites

3. **Geocoder** (`geocoder.py`)
   - Converts addresses to coordinates using Census API
   - Maps coordinates to census tracts
   - Implements privacy-preserving coordinate fuzzing
   - Calculates distances between points

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install requests pandas geopy
```

No API keys required for EPA AQS or Census Geocoding APIs (rate limits apply).

## Usage

### 1. Geocode Participant Addresses

```bash
# Geocode addresses from CSV
python scripts/environmental/geocoder.py \
    --input participants.csv \
    --address-column address \
    --output data/environmental/geocoded.csv

# Geocode with privacy fuzzing (1km radius)
python scripts/environmental/geocoder.py \
    --input participants.csv \
    --address-column address \
    --fuzz-km 1.0 \
    --output data/environmental/geocoded_private.csv

# Geocode ZIP codes only
python scripts/environmental/geocoder.py \
    --input participants.csv \
    --zip-column zip_code \
    --output data/environmental/geocoded_zip.csv
```

**Output:** CSV with added columns:
- `latitude`, `longitude`
- `census_tract` (11-digit GEOID)
- `state_fips`, `county_fips`
- `match_quality`

### 2. Download Air Quality Data

```bash
# Get air quality for Los Angeles County
python scripts/environmental/epa_data_puller.py \
    --data-type air_quality \
    --state CA \
    --county 037 \
    --start-date 20200101 \
    --end-date 20201231 \
    --output data/environmental/

# Get specific pollutants statewide
python scripts/environmental/epa_data_puller.py \
    --data-type air_quality \
    --state CA \
    --pollutants 88101 44201 42602 \
    --start-date 20200101 \
    --end-date 20201231 \
    --output data/environmental/

# Pollutant codes:
# 88101 = PM2.5
# 44201 = Ozone
# 42602 = NO2
# 43501 = Lead
# 45201 = Benzene
```

**Output:**
- `air_quality_data.csv` - Raw measurements
- `annual_mean_exposures.csv` - Aggregated metrics by site

### 3. Download Water Quality Data

```bash
# Get water quality for California
python scripts/environmental/usgs_water_puller.py \
    --data-type water_quality \
    --state CA \
    --start-date 2020-01-01 \
    --end-date 2020-12-31 \
    --output data/environmental/

# Get specific contaminants
python scripts/environmental/usgs_water_puller.py \
    --data-type water_quality \
    --state CA \
    --parameters 01051 71900 01002 00618 \
    --start-date 2020-01-01 \
    --end-date 2020-12-31 \
    --output data/environmental/

# Parameter codes:
# 01051 = Lead (dissolved)
# 71900 = Mercury (total)
# 01002 = Arsenic (dissolved)
# 01067 = Manganese (dissolved)
# 00618 = Nitrate
```

**Output:**
- `water_quality_data.csv` - Raw measurements
- `water_exposure_metrics.csv` - Aggregated metrics by site

### 4. Download TRI Industrial Releases

```bash
# Download TRI bulk data (requires manual step)
python scripts/environmental/epa_data_puller.py \
    --data-type tri \
    --year 2020 \
    --output data/environmental/

# Note: TRI data requires bulk file download from:
# https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files
```

## Complete Workflow

### End-to-End Exposure Assessment

```bash
# Step 1: Geocode participant addresses with privacy protection
python scripts/environmental/geocoder.py \
    --input data/participants.csv \
    --address-column address \
    --fuzz-km 1.0 \
    --output data/environmental/participants_geocoded.csv

# Step 2: Download air quality data for relevant regions
# (Identify states/counties from geocoded data first)
python scripts/environmental/epa_data_puller.py \
    --data-type air_quality \
    --state CA \
    --county 037 \
    --start-date 20200101 \
    --end-date 20201231 \
    --output data/environmental/

# Step 3: Download water quality data
python scripts/environmental/usgs_water_puller.py \
    --data-type water_quality \
    --state CA \
    --start-date 2020-01-01 \
    --end-date 2020-12-31 \
    --output data/environmental/

# Step 4: Link exposure data to participants using census tracts
python -c "
import pandas as pd

# Load geocoded participants
participants = pd.read_csv('data/environmental/participants_geocoded.csv')

# Load exposure matrix
exposures = pd.read_csv('data/environmental/exposure_matrices.csv')

# Join on census tract
merged = participants.merge(
    exposures,
    left_on='census_tract',
    right_on='census_tract_geoid',
    how='left'
)

merged.to_csv('data/environmental/participants_with_exposures.csv', index=False)
print(f'Linked exposure data for {len(merged)} participants')
"
```

## Output Files

### exposure_matrices.csv

Comprehensive census tract-level exposure estimates:

| Column | Description |
|--------|-------------|
| `census_tract_geoid` | 11-digit census tract identifier |
| `tract_centroid_lat/lon` | Tract center coordinates |
| `pm25_annual_mean` | PM2.5 annual mean (µg/m³) |
| `ozone_4th_max` | Ozone 4th highest daily max (ppm) |
| `no2_annual_mean` | NO2 annual mean (ppm) |
| `benzene_annual_mean` | Benzene annual mean (µg/m³) |
| `water_lead_mgl` | Lead in water (mg/L) |
| `water_nitrate_mgl` | Nitrate in water (mg/L) |
| `tri_facilities_1km` | TRI facilities within 1km |
| `tri_total_releases_5km_lbs` | Total TRI releases within 5km (lbs/year) |
| `major_roads_500m` | Major roads within 500m |
| `highway_distance_km` | Distance to nearest highway |
| `median_income` | Census tract median income |
| `poverty_rate` | Poverty rate (0-1) |
| `ejscreen_*_percentile` | EPA EJSCREEN environmental justice indicators |

### air_quality_exposure_matrix.csv

Census tract-level air quality metrics:
- PM2.5, PM10, Ozone, NO2 (criteria pollutants)
- Benzene, Toluene, Lead (air toxics)
- AQI exceedance days
- Standard exceedances

### water_quality_exposure_matrix.csv

USGS monitoring site water quality:
- Heavy metals (Lead, Mercury, Arsenic, Manganese)
- Nutrients (Nitrate, Phosphorus)
- MCL exceedances
- Sample counts and quality

### tri_facilities_exposure_matrix.csv

EPA TRI facility emissions:
- Chemical releases by type (air, water, land)
- Neurotoxic chemicals (Lead, Mercury, Benzene, Toluene)
- Distance to residential areas
- Industry sectors

## Neurotoxic Pollutants Monitored

### Air Pollutants

**Criteria Pollutants:**
- PM2.5 (Fine particulate matter) - Oxidative stress, neuroinflammation
- PM10 (Coarse particulate matter)
- Ozone - Respiratory effects, systemic inflammation
- NO2 (Nitrogen dioxide) - Traffic-related, respiratory/neurological
- SO2 (Sulfur dioxide)
- CO (Carbon monoxide)

**Air Toxics:**
- Lead - Neurodevelopmental toxicant
- Benzene - Hematotoxic, neurotoxic
- Toluene - Neurotoxic solvent
- Xylenes - Neurotoxic
- Formaldehyde - Respiratory irritant
- Mercury - Neurodevelopmental toxicant
- Manganese - Neurotoxic metal
- Arsenic - Neurotoxic metalloid
- Cadmium - Neurotoxic metal
- Chromium - Oxidative stress
- Nickel - Respiratory/neurological

**Polycyclic Aromatic Hydrocarbons (PAHs):**
- Benzo[a]pyrene
- Benzo[a]anthracene
- Chrysene

### Water Contaminants

**Heavy Metals:**
- Lead - Neurodevelopmental toxicant (MCL: 0.015 mg/L)
- Mercury - Neurotoxic (MCL: 0.002 mg/L)
- Arsenic - Neurotoxic (MCL: 0.010 mg/L)
- Manganese - Neurotoxic (no federal MCL, state varies)
- Cadmium - Neurotoxic (MCL: 0.005 mg/L)

**Pesticides:**
- Atrazine - Endocrine disruptor
- Chlorpyrifos - Organophosphate, neurotoxic
- Malathion - Organophosphate
- Glyphosate - Herbicide

**Nutrients:**
- Nitrate - Endocrine disruption (MCL: 10 mg/L as N)
- Phosphorus - Eutrophication indicator

### Industrial Chemicals (TRI)

- Lead compounds
- Mercury compounds
- Benzene
- Toluene
- Xylenes
- Styrene
- Formaldehyde
- Polychlorinated biphenyls (PCBs)

## Geospatial Linkage Methods

### Method 1: Census Tract Linkage

**Best for:** Population-level exposure estimates

```python
import pandas as pd

# Geocode participants to census tracts
participants = pd.read_csv('participants_geocoded.csv')

# Load exposure matrix
exposures = pd.read_csv('exposure_matrices.csv')

# Join on census tract
merged = participants.merge(
    exposures,
    left_on='census_tract',
    right_on='census_tract_geoid',
    how='left'
)
```

**Advantages:**
- Links to socioeconomic data
- Links to EPA EJSCREEN indicators
- Protects participant privacy
- Widely used in environmental epidemiology

**Limitations:**
- Assumes homogeneous exposure within tract
- May not capture local gradients

### Method 2: Nearest Monitor

**Best for:** Individual-level exposure estimates

```python
from scipy.spatial import cKDTree

# Participant coordinates
part_coords = participants[['latitude', 'longitude']].values

# Monitor coordinates
monitor_coords = air_quality[['latitude', 'longitude']].values

# Find nearest monitor for each participant
tree = cKDTree(monitor_coords)
distances, indices = tree.query(part_coords, k=1)

# Assign nearest monitor's exposure
participants['pm25'] = air_quality.iloc[indices]['pm25_ugm3'].values
participants['distance_to_monitor_km'] = distances * 111  # Convert degrees to km
```

**Advantages:**
- More precise individual estimates
- Captures local variation

**Limitations:**
- Requires sufficient monitor density
- Distance decay not accounted for

### Method 3: Distance-Weighted Average

**Best for:** Multiple nearby monitors

```python
def inverse_distance_weighting(part_lat, part_lon, monitors_df, k=3, power=2):
    \"\"\"
    Calculate IDW exposure from k nearest monitors

    Args:
        part_lat, part_lon: Participant coordinates
        monitors_df: DataFrame with monitor data
        k: Number of nearest monitors
        power: IDW power parameter
    \"\"\"
    from scipy.spatial import cKDTree
    import numpy as np

    # Find k nearest monitors
    coords = monitors_df[['latitude', 'longitude']].values
    tree = cKDTree(coords)
    distances, indices = tree.query([part_lat, part_lon], k=k)

    # Calculate weights
    weights = 1 / (distances ** power)
    weights = weights / weights.sum()

    # Weighted average
    exposure = (monitors_df.iloc[indices]['pm25'].values * weights).sum()

    return exposure
```

### Method 4: Buffer Analysis

**Best for:** TRI facility proximity

```python
def count_facilities_in_buffer(part_lat, part_lon, facilities_df, radius_km):
    \"\"\"Count TRI facilities within buffer radius\"\"\"
    from scipy.spatial import cKDTree

    coords = facilities_df[['latitude', 'longitude']].values
    tree = cKDTree(coords)

    # Query all facilities within radius
    indices = tree.query_ball_point([part_lat, part_lon], r=radius_km/111.0)

    # Sum releases
    total_releases = facilities_df.iloc[indices]['total_releases_lbs'].sum()

    return len(indices), total_releases
```

## Privacy Protection

### Coordinate Fuzzing

```bash
# Add 1km random noise to coordinates
python scripts/environmental/geocoder.py \
    --input participants.csv \
    --address-column address \
    --fuzz-km 1.0 \
    --output data/environmental/participants_fuzzed.csv
```

**Fuzzing preserves:**
- Census tract assignment (1km << tract size)
- Approximate exposure estimates
- Distance-based analyses

**Fuzzing protects:**
- Exact home locations
- Re-identification risk
- Address privacy

### Census Tract Only

```python
# Export only census tract, not coordinates
participants_private = participants[['participant_id', 'census_tract']]
participants_private.to_csv('participants_tract_only.csv', index=False)
```

**Advantages:**
- Maximum privacy protection
- Still enables exposure linkage
- Complies with HIPAA Safe Harbor (>20k population)

**Limitations:**
- Cannot use nearest-monitor or IDW methods
- Coarser exposure estimates

## Integration with Other Data

### Link to Clinical Data

```python
import pandas as pd

# Load geocoded participants with exposures
exposures = pd.read_csv('data/environmental/participants_with_exposures.csv')

# Load clinical phenotype data
clinical = pd.read_csv('data/phenotypes/clinical_assessments.csv')

# Merge
merged = clinical.merge(exposures, on='participant_id')

# Analyze exposure-phenotype associations
import scipy.stats as stats

# Example: PM2.5 and ADHD symptom severity
adhd_high_pm = merged[merged['pm25_annual_mean'] > merged['pm25_annual_mean'].median()]['adhd_score']
adhd_low_pm = merged[merged['pm25_annual_mean'] <= merged['pm25_annual_mean'].median()]['adhd_score']

t_stat, p_value = stats.ttest_ind(adhd_high_pm, adhd_low_pm)
print(f'High vs low PM2.5 exposure: t={t_stat:.3f}, p={p_value:.4f}')
```

### Link to Genetic Data

```python
# Load genetic data
genetics = pd.read_csv('data/genetics/participant_genotypes.csv')

# Merge with exposures
gxe_data = genetics.merge(exposures, on='participant_id')

# Gene-environment interaction analysis
# Example: Dopamine transporter gene × lead exposure
high_lead = gxe_data['water_lead_mgl'] > gxe_data['water_lead_mgl'].median()
dat1_risk = gxe_data['DAT1_genotype'] == 'risk_allele'

# 2x2 contingency table
from scipy.stats import chi2_contingency

contingency = pd.crosstab(high_lead, dat1_risk)
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f'GxE interaction: χ²={chi2:.3f}, p={p_value:.4f}')
```

### Link to Microbiome Data

```python
# Load microbiome data
microbiome = pd.read_csv('data/microbiome/alpha_diversity.csv')

# Merge with exposures
merged = microbiome.merge(exposures, on='participant_id')

# Examine exposure effects on microbiome
import seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(data=merged, x='pm25_annual_mean', y='shannon_diversity')
plt.xlabel('PM2.5 Exposure (µg/m³)')
plt.ylabel('Gut Microbiome Shannon Diversity')
plt.title('Air Pollution and Microbiome Diversity')
plt.savefig('pm25_microbiome.png')
```

## Data Sources and Documentation

### EPA Air Quality System (AQS)
- **URL**: https://aqs.epa.gov/api
- **Documentation**: https://aqs.epa.gov/aqsweb/documents/data_api.html
- **Coverage**: 4,000+ monitoring sites nationwide
- **Data**: Hourly/daily measurements since 1980s
- **Access**: Free, no API key required (rate limits apply)

### EPA Toxics Release Inventory (TRI)
- **URL**: https://www.epa.gov/toxics-release-inventory-tri-program
- **Documentation**: https://www.epa.gov/tri/tri-basic-data-files-guides
- **Coverage**: 21,000+ facilities, 770+ chemicals
- **Data**: Annual facility releases since 1987
- **Access**: Free bulk file download

### EPA EJSCREEN
- **URL**: https://www.epa.gov/ejscreen
- **Data**: Census tract-level environmental justice indicators
- **Metrics**: Pollution + demographics percentiles
- **Access**: Web tool + downloadable data

### USGS National Water Information System (NWIS)
- **URL**: https://waterdata.usgs.gov/nwis
- **Documentation**: https://waterservices.usgs.gov/
- **Coverage**: 1.9M+ monitoring sites
- **Data**: Real-time and historical water quality
- **Access**: Free API

### US Census Geocoding API
- **URL**: https://geocoding.geo.census.gov/geocoder
- **Documentation**: https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html
- **Access**: Free, no API key required
- **Features**: Address to coordinate, coordinate to census tract

### Census American Community Survey (ACS)
- **URL**: https://www.census.gov/programs-surveys/acs
- **Data**: Socioeconomic indicators by census tract
- **Variables**: Income, poverty, education, demographics
- **Access**: Free via Census API or data.census.gov

## Environmental Health References

### ADHD and Environmental Exposures

**Air Pollution:**
- PM2.5 exposure and ADHD incidence (Costa et al. 2020, Environ Health Perspect)
- Traffic-related air pollution and ADHD symptoms (Newman et al. 2013, Environ Health)
- Prenatal air pollution and childhood ADHD (Morales et al. 2022, Environ Int)

**Lead:**
- Blood lead levels and ADHD (Nigg et al. 2008, Biol Psychiatry)
- Low-level lead exposure and executive function (Surkan et al. 2007, Environ Health Perspect)

**Pesticides:**
- Organophosphate pesticides and ADHD (Bouchard et al. 2010, Pediatrics)
- Pyrethroid pesticides and behavioral problems (Wagner-Schuman et al. 2015, Environ Health)

### Autism and Environmental Exposures

**Air Pollution:**
- Prenatal air pollution and ASD risk (Volk et al. 2013, JAMA Psychiatry)
- PM2.5 and autism prevalence (Kalkbrenner et al. 2015, Epidemiology)
- Traffic proximity and ASD (Volk et al. 2011, Environ Health Perspect)

**Heavy Metals:**
- Mercury exposure and ASD (Desoto & Hitlan 2007, Health Place)
- Manganese and ASD (Dickerson et al. 2021, Environ Res)

**Pesticides:**
- Proximity to agricultural pesticides and ASD (Shelton et al. 2014, Environ Health Perspect)
- Maternal pesticide exposure and ASD (Roberts et al. 2007, Environ Health Perspect)

### Gene-Environment Interactions

- Dopaminergic genes and lead exposure in ADHD (Roy et al. 2009, Neurotoxicology)
- Paraoxonase 1 (PON1) and pesticides in ASD (D'Amelio et al. 2005, Pediatrics)
- Glutathione-S-transferase and air pollution (Cardenas et al. 2013, Environ Health Perspect)

## Troubleshooting

### EPA AQS API Rate Limits

```python
# Add delays between requests
import time

for pollutant in pollutants:
    data = get_air_quality_data(pollutant)
    time.sleep(5)  # 5 second delay
```

EPA recommends:
- Max 10 requests per minute
- Max 500 requests per hour
- Use `email` parameter for identification

### USGS NWIS Large Requests

```python
# Break into smaller date ranges
from datetime import datetime, timedelta

start = datetime(2020, 1, 1)
end = datetime(2020, 12, 31)

# Monthly chunks
current = start
while current < end:
    next_date = current + timedelta(days=30)
    data = get_water_quality(start_date=current, end_date=next_date)
    current = next_date
```

### Geocoding Failures

```bash
# Try with ZIP code fallback
python geocoder.py \
    --input participants.csv \
    --address-column address \
    --zip-column zip_code \
    --output geocoded.csv

# Manual review of failed matches
grep "No_Match" geocoded.csv > failed_geocodes.csv
```

### Missing Data Imputation

```python
import pandas as pd
import numpy as np

# Load exposure matrix
exposures = pd.read_csv('exposure_matrices.csv')

# Identify missing data
missing_summary = exposures.isnull().sum()
print(missing_summary[missing_summary > 0])

# Options:
# 1. Use county-level means
exposures['pm25_imputed'] = exposures.groupby('county')['pm25_annual_mean'].transform(
    lambda x: x.fillna(x.mean())
)

# 2. Use state-level means
exposures['pm25_imputed'] = exposures.groupby('state')['pm25_annual_mean'].transform(
    lambda x: x.fillna(x.mean())
)

# 3. Flag and exclude
exposures['pm25_missing'] = exposures['pm25_annual_mean'].isnull()
complete_cases = exposures[~exposures['pm25_missing']]
```

## Future Enhancements

Potential additions:
- **NATA (National Air Toxics Assessment)** - Census tract cancer/non-cancer risk
- **Pesticide Use Data** - USGS/USDA county-level application rates
- **Traffic Density** - Road network analysis
- **Noise Pollution** - DOT noise exposure models
- **Proximity to Greenspace** - NDVI from satellite imagery
- **Land Use Regression** - Predictive exposure models
- **Temporal Analysis** - Time-varying exposure windows
- **Machine Learning** - Exposure prediction from satellite/GIS data

## Support

For questions or issues:
1. Check EPA/USGS API documentation
2. Verify coordinate formats and census tract IDs
3. Review geocoding match quality
4. Test with example data first
5. Open GitHub issue with detailed description

---

**Last updated**: 2025-09-30
**Version**: 1.0
**Maintained by**: AuDHD Correlation Study Team