'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- calculates meridional crossings of particles at a specified longitude
- save results in netcdf file

Author: vesnaber
Kernel: parcels-dev-local
'''

#%%
import xarray as xr
import numpy as np
from datetime import datetime
import os

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================
part_config = 'GRENAVENE_coastrep050ED'
target_longitude = -68.9
max_days = 90
ANALYSIS_NAME = part_config

# Years to process
years_to_process = list(range(1993, 2024))

# Latitudinal zones for grouping crossings
lat_zones = [(11.25, 12.25), (12.25, 13.25), (13.25, 14.25), (14.25, 15.25), 
             (15.25, 16.25), (16.25, 17.25), (17.25, 18.4)]
zone_labels = ['11.25-12.25°N', '12.25-13.25°N', '13.25-14.25°N', '14.25-15.25°N', 
               '15.25-16.25°N', '16.25-17.25°N', '17.25-18.4°N']

# =============================================================================

def load_particle_data(year):
    """Load particle trajectory data for a specific year"""
    output_file = f'../parcels_run/GRENAVENE_output/{part_config}_{year}.zarr'
    
    if not os.path.exists(output_file):
        print(f"Warning: File not found - {output_file}")
        return None
        
    print(f"Loading particle data for {year}...")
    ds = xr.open_zarr(output_file)
    print(f"Data shape for {year}: {ds.dims}")
    return ds

def find_crossings_numpy(ds, target_lon, max_days):

    print(f"Converting to NumPy and detecting crossings at {target_lon}°...")
    
    # Load all data into NumPy arrays immediately
    print("Loading data into memory...")
    lon = ds.lon.values.copy()  # shape: (trajectory, obs)
    lat = ds.lat.values.copy()
    time_ns = ds.time.values.copy()  # nanoseconds
    
    # Check if U,V are 1D (stored once) or 2D (stored multiple times)
    if ds.U.values.ndim == 1:
        initial_u = ds.U.values.copy()  # shape: (trajectory,)
        initial_v = ds.V.values.copy()
        print(f"Initial velocities shape: {initial_u.shape} (stored once per trajectory)")
    else:
        initial_u = ds.U.values[:, 0].copy()  # shape: (trajectory,)
        initial_v = ds.V.values[:, 0].copy()
        print(f"Initial velocities shape: {initial_u.shape} (taking first timestep)")
    
    trajectory_ids = ds.trajectory.values.copy()
    
    n_traj, n_obs = lon.shape
    print(f"Processing {n_traj} trajectories, {n_obs} observations each")
    
    # Create time mask for max_days filter
    start_times = time_ns[:, 0:1]  # shape: (trajectory, 1) for broadcasting
    time_diff_days = (time_ns - start_times) / np.timedelta64(1, 'D')
    time_mask = time_diff_days <= max_days
    
    # Apply time mask
    lon = np.where(time_mask, lon, np.nan)
    lat = np.where(time_mask, lat, np.nan)
    
    # Get consecutive pairs (remove last observation)
    lon1 = lon[:, :-1]  # positions at t
    lon2 = lon[:, 1:]   # positions at t+1
    lat1 = lat[:, :-1]
    lat2 = lat[:, 1:]
    time1 = time_ns[:, :-1]
    time2 = time_ns[:, 1:]
    
    print("Detecting crossings...")
    # Vectorized crossing detection (westward: lon1 > target > lon2)
    crosses_westward = (lon1 > target_lon) & (lon2 <= target_lon)
    
    # Remove longitude wrap-around and NaN values
    lon_jump = np.abs(lon2 - lon1)
    valid_crossings = (crosses_westward & 
                      (lon_jump < 180.0) & 
                      ~np.isnan(lon1) & 
                      ~np.isnan(lon2) & 
                      ~np.isnan(lat1) & 
                      ~np.isnan(lat2))
    
    # Find trajectories that cross and their first crossing index
    has_crossing = np.any(valid_crossings, axis=1)  # shape: (trajectory,)
    num_crossings = np.sum(has_crossing)
    
    if num_crossings == 0:
        print("No crossings found!")
        return None
    
    print(f"Found {num_crossings} trajectories with crossings")
    
    # Get first crossing index for each trajectory (vectorized with argmax)
    first_crossing_idx = np.argmax(valid_crossings, axis=1)  # shape: (trajectory,)
    
    # Filter to only crossing trajectories
    crossing_mask = has_crossing
    crossing_traj_ids = trajectory_ids[crossing_mask]
    crossing_obs_idx = first_crossing_idx[crossing_mask]
    crossing_traj_idx = np.where(crossing_mask)[0]
    
    print("Extracting crossing data...")
    # Vectorized extraction using advanced indexing
    lon1_cross = lon1[crossing_traj_idx, crossing_obs_idx]
    lon2_cross = lon2[crossing_traj_idx, crossing_obs_idx]
    lat1_cross = lat1[crossing_traj_idx, crossing_obs_idx]
    lat2_cross = lat2[crossing_traj_idx, crossing_obs_idx]
    time1_cross = time1[crossing_traj_idx, crossing_obs_idx]
    time2_cross = time2[crossing_traj_idx, crossing_obs_idx]
    
    # Vectorized interpolation to exact crossing point
    alpha = (target_lon - lon1_cross) / (lon2_cross - lon1_cross)
    crossing_lats = lat1_cross + alpha * (lat2_cross - lat1_cross)
    crossing_times = time1_cross + alpha * (time2_cross - time1_cross)
    
    # Calculate days to crossing
    start_times_cross = time_ns[crossing_traj_idx, 0]
    days_to_crossing = (crossing_times - start_times_cross) / np.timedelta64(1, 'D')
    
    # Get initial velocities for crossing trajectories
    u_initial = initial_u[crossing_traj_idx]
    v_initial = initial_v[crossing_traj_idx]
    
    print("Assigning zones...")
    # Vectorized zone assignment
    zone_indices = np.full(len(crossing_lats), -1, dtype=int)
    zone_labels_assigned = np.full(len(crossing_lats), 'outside_zones', dtype=object)
    
    for i, (lat_min, lat_max) in enumerate(lat_zones):
        in_zone = (crossing_lats >= lat_min) & (crossing_lats < lat_max)
        zone_indices[in_zone] = i
        zone_labels_assigned[in_zone] = zone_labels[i]
    
    # Extract years from start times
    years = np.array([np.datetime64(t, 'Y').astype(int) + 1970 
                     for t in start_times_cross], dtype=int)
    
    print(f"Processing complete: {len(crossing_traj_ids)} crossings found")
    
    # Return all crossing data as numpy arrays
    return {
        'trajectory_id': crossing_traj_ids,
        'year': years,
        'start_time': start_times_cross,
        'crossing_time': crossing_times,
        'crossing_lat': crossing_lats,
        'crossing_lon': np.full(len(crossing_lats), target_lon),
        'days_to_crossing': days_to_crossing,
        'initial_u': u_initial,
        'initial_v': v_initial,
        'zone_index': zone_indices,
        'zone_label': zone_labels_assigned
    }

def process_all_years_numpy():
    """Process all years and combine results"""
    print(f"Target longitude: {target_longitude}°W")
    print(f"Maximum trajectory age: {max_days} days")
    
    all_crossings = []
    
    for year in years_to_process:
        print(f"\n{'='*60}")
        print(f"Processing year {year}")
        print(f"{'='*60}")
        
        # Load data
        ds = load_particle_data(year)
        if ds is None:
            continue
            
        try:
            # Find crossings using pure NumPy
            crossing_data = find_crossings_numpy(ds, target_longitude, max_days)
            
            if crossing_data is not None:
                all_crossings.append(crossing_data)
                
                # Print summary statistics
                n_crossings = len(crossing_data['trajectory_id'])
                print(f"\nYear {year} summary:")
                print(f"  Total crossings: {n_crossings}")
                print(f"  Mean crossing time: {crossing_data['days_to_crossing'].mean():.2f} days")
                print(f"  Mean initial speed: {np.sqrt(crossing_data['initial_u']**2 + crossing_data['initial_v']**2).mean():.3f} m/s")
                
                # Zone statistics
                unique_zones, zone_counts = np.unique(crossing_data['zone_index'], return_counts=True)
                for zi, count in zip(unique_zones, zone_counts):
                    if zi >= 0:
                        print(f"    {zone_labels[zi]}: {count} crossings")
                    else:
                        print(f"    Outside zones: {count} crossings")
                        
        except Exception as e:
            print(f"Error processing year {year}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Clean up memory
        del ds
    
    if not all_crossings:
        print("No crossing data found!")
        return None
        
    print(f"\n{'='*60}")
    print("Combining all years...")
    
    # Concatenate all years using NumPy
    combined_data = {}
    for key in all_crossings[0].keys():
        if key == 'zone_label':
            # Handle string arrays specially
            combined_data[key] = np.concatenate([data[key] for data in all_crossings])
        else:
            combined_data[key] = np.concatenate([data[key] for data in all_crossings])
    
    # Sort by crossing time
    sort_indices = np.argsort(combined_data['crossing_time'])
    for key in combined_data.keys():
        combined_data[key] = combined_data[key][sort_indices]
    
    print(f"Total crossings across all years: {len(combined_data['trajectory_id'])}")
    
    return combined_data

def save_to_netcdf(crossing_data, filename):
    """Convert NumPy arrays to xarray Dataset and save as NetCDF"""
    print(f"Saving to NetCDF: {filename}")
    
    # Convert numpy arrays to xarray Dataset
    ds = xr.Dataset({
        'trajectory_id': (['crossing'], crossing_data['trajectory_id']),
        'year': (['crossing'], crossing_data['year']),
        'start_time': (['crossing'], crossing_data['start_time']),
        'crossing_time': (['crossing'], crossing_data['crossing_time']),
        'crossing_lat': (['crossing'], crossing_data['crossing_lat']),
        'crossing_lon': (['crossing'], crossing_data['crossing_lon']),
        'days_to_crossing': (['crossing'], crossing_data['days_to_crossing']),
        'initial_u': (['crossing'], crossing_data['initial_u']),
        'initial_v': (['crossing'], crossing_data['initial_v']),
        'zone_index': (['crossing'], crossing_data['zone_index']),
        'zone_label': (['crossing'], crossing_data['zone_label'])
    })
    
    # Add metadata
    ds.attrs.update({
        'title': 'Particle Meridional Crossings Analysis - NumPy Vectorized',
        'target_longitude': target_longitude,
        'max_trajectory_days': max_days,
        'analysis_name': ANALYSIS_NAME,
        'creation_date': datetime.now().isoformat(),
        'processing_method': 'Pure NumPy vectorized operations',
        'total_crossings': int(len(crossing_data['trajectory_id'])),
        'years_processed': list(np.unique(crossing_data['year'])),
        'lat_zones': str(lat_zones),
        'zone_labels': str(zone_labels)
    })
    
    # Add variable attributes
    ds['trajectory_id'].attrs = {'long_name': 'Particle trajectory ID'}
    ds['year'].attrs = {'long_name': 'Year of particle release'}
    ds['start_time'].attrs = {'long_name': 'Particle release time'}
    ds['crossing_time'].attrs = {'long_name': 'Time of meridional crossing'}
    ds['crossing_lat'].attrs = {'long_name': 'Latitude of crossing', 'units': 'degrees_north'}
    ds['crossing_lon'].attrs = {'long_name': 'Longitude of crossing', 'units': 'degrees_east'}
    ds['days_to_crossing'].attrs = {'long_name': 'Days from release to crossing', 'units': 'days'}
    ds['initial_u'].attrs = {'long_name': 'Initial eastward velocity', 'units': 'deg/s'}
    ds['initial_v'].attrs = {'long_name': 'Initial northward velocity', 'units': 'deg/s'}
    ds['zone_index'].attrs = {'long_name': 'Latitudinal zone index (-1 if outside zones)'}
    ds['zone_label'].attrs = {'long_name': 'Latitudinal zone label'}
    
    # Save to NetCDF
    ds.to_netcdf(filename)
    print(f"Successfully saved {len(crossing_data['trajectory_id'])} crossings to {filename}")
    
    return ds

if __name__ == "__main__":
    # Main execution
    print("="*80)
    print("PARTICLE CROSSING ANALYSIS")
    print("="*80)
    
    # Process all years with NumPy
    combined_crossings = process_all_years_numpy()
    
    if combined_crossings is not None:
        # Create output directory
        output_dir = 'crossings_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to NetCDF
        output_file = f'{output_dir}/{ANALYSIS_NAME}_ALLY_all_crossings_numpy.nc'
        final_ds = save_to_netcdf(combined_crossings, output_file)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"Output file: {output_file}")
        print(f"Total crossings: {len(combined_crossings['trajectory_id'])}")
        print(f"Years covered: {sorted(np.unique(combined_crossings['year']))}")
        print(f"Mean crossing time: {combined_crossings['days_to_crossing'].mean():.2f} ± {combined_crossings['days_to_crossing'].std():.2f} days")
        print(f"Mean initial velocity: U={combined_crossings['initial_u'].mean():.3f} m/s, V={combined_crossings['initial_v'].mean():.3f} m/s")
        
        # Final zone statistics
        print(f"\nCrossings by zone:")
        unique_zones, zone_counts = np.unique(combined_crossings['zone_index'], return_counts=True)
        for zi, count in zip(unique_zones, zone_counts):
            if zi >= 0:
                print(f"  {zone_labels[zi]}: {count} crossings")
            else:
                print(f"  Outside zones: {count} crossings")
                
        print(f"\nTo load the data:")
        print(f"import xarray as xr")
        print(f"ds = xr.open_dataset('{output_file}')")
        
    else:
        print("No crossing data found across all years!")
# %%
