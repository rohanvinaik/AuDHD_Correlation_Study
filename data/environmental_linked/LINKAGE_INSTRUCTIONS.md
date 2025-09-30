# Environmental Data Linkage Instructions

Generated: 2025-09-30T11:55:37.863711

## Available Data

- Participants: 100
- Available fields: participant_id, zipcode, visit_date, latitude, longitude

## Recommended Linkage Strategies

### 1. EPA Air Quality (AirNow)

- **Method**: Zip code + date matching
- **Feasibility**: HIGH
- **API Key Needed**: True
- **Cost**: Free

**Steps:**
1. Get free API key: https://docs.airnowapi.org/
2. Use link_epa_air_quality() function
3. Will provide daily PM2.5, Ozone, AQI

### 2. EPA Toxics Release Inventory

- **Method**: Zip code geocoding + proximity
- **Feasibility**: HIGH
- **API Key Needed**: False
- **Cost**: Free

**Steps:**
1. Geocode zip codes (automatic)
2. Use link_epa_toxics_release() function
3. Will provide nearby industrial chemical releases

### 3. NDVI (Green Space)

- **Method**: GPS coordinates + satellite imagery
- **Feasibility**: MEDIUM
- **API Key Needed**: True
- **Cost**: Free (NASA EarthData account)

**Steps:**
1. Register at https://urs.earthdata.nasa.gov/
2. Download MODIS NDVI data
3. Extract values at participant coordinates

