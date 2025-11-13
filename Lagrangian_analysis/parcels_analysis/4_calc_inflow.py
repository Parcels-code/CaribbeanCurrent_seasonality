'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- calculates inflow strength and sacves as cache file

Author: vesnaber
Kernel: parcels-dev-local
'''

# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import os
import pickle
from pathlib import Path

# Configuration
parcels_config = 'GRENAVENE_coastrep050ED'
data_path = '../parcels_run/GRENAVENE_output/'
years_to_process = list(range(1993, 2025))

# Cross-section definition
lon_start = -62.5
lat_start = 10.72
lon_end = -61.7
lat_end = 12.04

# Cache configuration
cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)

# %% Calculate distance using haversine formula

R = 6371.0  # Earth radius in km
lat1, lon1 = radians(lat_start), radians(lon_start)
lat2, lon2 = radians(lat_end), radians(lon_end)
dlat = lat2 - lat1
dlon = lon2 - lon1
a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
c = 2 * atan2(sqrt(a), sqrt(1 - a))
distance_km = R * c

# Number of particles (10 per km)
n_particles = int(distance_km * 10)
particle_spacing_km = distance_km / n_particles

# Calculate normal vector to cross-section
dx = lon_end - lon_start
dy = lat_end - lat_start

# Normal vector (perpendicular to line)
normal_x = -dy
normal_y = dx

# Normalize
norm_magnitude = sqrt(normal_x**2 + normal_y**2)
normal_x /= norm_magnitude
normal_y /= norm_magnitude

print(f"Cross-section distance: {distance_km:.2f} km")
print(f"Number of particles: {n_particles}")
print(f"Particle spacing: {particle_spacing_km:.3f} km")
print(f"Normal vector: ({normal_x:.3f}, {normal_y:.3f})")

# %%  3. Load Particle Data from Zarr Files

# Check if cached data exists
cache_key = f"{parcels_config}_{min(years_to_process)}-{max(years_to_process)}"
cache_file = cache_dir / f"particle_data_{cache_key}.pkl"

if cache_file.exists():
    print(f"Loading data from cache: {cache_file}")
    with open(cache_file, 'rb') as f:
        all_initial_data = pickle.load(f)
    print("Using cached particle data")
else:
    print("Loading particle trajectory data from zarr files...")
    
    all_initial_data = []
    
    for year in years_to_process:
        zarr_file = f'{data_path}{parcels_config}_{year}.zarr'
        
        if not os.path.exists(zarr_file):
            print(f"Warning: File not found - {zarr_file}")
            continue
            
        print(f"Loading {year}...")
        ds = xr.open_zarr(zarr_file)
        
        # Extract initial velocities and positions
        if ds.U.values.ndim == 1:
            initial_u = ds.U.values.copy()
            initial_v = ds.V.values.copy()
            print(f"  Year {year}: {len(initial_u)} particles, velocities stored once per trajectory")
        else:
            initial_u = ds.U.values[:, 0].copy()
            initial_v = ds.V.values[:, 0].copy()
            print(f"  Year {year}: {len(initial_u)} particles, taking first timestep")
        
        # Get initial latitudes for conversion
        if ds.lat.values.ndim == 1:
            initial_lat = ds.lat.values.copy()
        else:
            initial_lat = ds.lat.values[:, 0].copy()
            
        # Store data with year info
        year_data = {
            'year': year,
            'initial_u': initial_u,
            'initial_v': initial_v,
            'initial_lat': initial_lat,
            'n_particles': len(initial_u)
        }
        all_initial_data.append(year_data)
        
        del ds
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(all_initial_data, f)
    print(f"Saved data to cache: {cache_file}")

# %% 4. Calculate Daily Inflow

# Check if cached inflow data exists
cache_file_inflow = cache_dir / f"particle_data_{cache_key}_inflow_m2s.pkl"

if cache_file_inflow.exists():
    print(f"Loading inflow data from cache: {cache_file_inflow}")
    with open(cache_file_inflow, 'rb') as f:
        df_daily = pickle.load(f)
    print("Using cached daily inflow data")
else:
    print("Processing particle data for daily inflow...")
    
    # Earth radius for velocity conversion
    R_earth = 6371000.0  # meters
    
    # Assumed depth for transport calculation
    width_per_particle_m = (particle_spacing_km * 1000) 
    
    daily_inflows = []
    
    for year_data in all_initial_data:
        year = year_data['year']
        initial_u = year_data['initial_u']
        initial_v = year_data['initial_v']
        initial_lat = year_data['initial_lat']
        n_particles_year = year_data['n_particles']
        
        print(f"\nProcessing year {year}: {n_particles_year} particles")
        
        # Convert velocities from deg/s to m/s
        lat_rad = np.radians(initial_lat)
        m_per_deg_lon = R_earth * np.cos(lat_rad) * np.pi / 180.0
        m_per_deg_lat = R_earth * np.pi / 180.0
        
        u_m_s = initial_u * m_per_deg_lon
        v_m_s = initial_v * m_per_deg_lat
        
        v_normal = u_m_s * normal_x + v_m_s * normal_y  # m/s

        # Calculate transport per unit depth (m²/s)
        transport_per_particle_m2s = v_normal * width_per_particle_m  # m²/s
        
        # Group by release batches (1707 particles per day)
        n_particles_per_release = 1707
        n_releases = n_particles_year // n_particles_per_release
        
        print(f"  Expected releases: {n_releases}")
        print(f"  Particles per release: {n_particles_per_release}")
        
        for release_idx in range(n_releases):
            start_idx = release_idx * n_particles_per_release
            end_idx = start_idx + n_particles_per_release
            
            # Sum transport for this release day
            daily_transport = np.sum(transport_per_particle_m2s[start_idx:end_idx])
            
            # Calculate release date (assuming daily releases starting Jan 1)
            release_date = pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=release_idx)
            
            daily_inflows.append({
                'release_date': release_date,
                'transport_m2s': daily_transport,
                'year': year,
                'release_idx': release_idx
            })
    
    # Convert to DataFrame
    df_daily = pd.DataFrame(daily_inflows)
    
    # Save to cache
    with open(cache_file_inflow, 'wb') as f:
        pickle.dump(df_daily, f)
    print(f"Saved inflow data to cache: {cache_file_inflow}")


# %%
