'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- calculates geostrophic velocities from SSH data downloaded from CMEMS (product GLORYS)
- unifies the NaN fields of U and V velocities to create a consistent land mask
- adds a displacement field to push particles off the coast (to avoid beaching) - this is done in the form of repellent velocities near the shore on the coastal grid cells

Data is divided in two subsets (my and myint, as is available in the product).

Author: vesnaber
kernel: parcels-dev-local

'''


# %%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os
import cmocean

# --- Configuration & Constants ---
GEO_VEL_FILE_1 = "GLORYS_input/cmems_mod_glo_phy_my_0.083deg_P1D-m_zos_70.00W-58.92W_9.50N-18.83N_1993-01-01-2021-06-30.nc"
GEO_VEL_FILE_2 = "GLORYS_input/cmems_mod_glo_phy_myint_0.083deg_P1D-m_zos_70.00W-58.92W_9.50N-18.83N_2021-07-01-2024-12-31.nc"
OUTPUT_DIR = 'geostrophic_flow_input'
DISPLACEMENT_SPEED = 0.5  # m/s

g = 9.81
omega = 7.2921e-5
R_earth = 6.371e6

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("="*70)
print("STARTING FULL VELOCITY FIELD GENERATION SCRIPT")
print("="*70)

# ============================================================================
# STEP 1: CALCULATE GEOSTROPHIC VELOCITIES (CENTRAL DIFFERENCES)
# ============================================================================

print("--- STEP 1: Calculating Geostrophic Velocities ---")

# Load and combine SSH
ds1 = xr.open_dataset(GEO_VEL_FILE_1)
ds2 = xr.open_dataset(GEO_VEL_FILE_2)
ds_combined = xr.concat([ds1, ds2], dim='time')

ssh = ds_combined.zos
time = ssh.time.values
lat_f = ssh.latitude.values
lon_f = ssh.longitude.values
ssh_values = ssh.values
ntime, nlat, nlon = ssh_values.shape

# --- U-velocity Calculation ---
dlat = np.abs(np.mean(np.diff(lat_f)))
dy = np.deg2rad(dlat) * R_earth
deta_dy = (ssh_values[:, 2:, :] - ssh_values[:, :-2, :]) / (2 * dy)
lon_u_2d, lat_u_2d = np.meshgrid(lon_f, lat_f[1:-1])
f_u = 2 * omega * np.sin(np.deg2rad(lat_u_2d))
u_geo_calc = -(g / f_u[np.newaxis, :, :]) * deta_dy

# --- V-velocity Calculation ---
dlon = np.abs(np.mean(np.diff(lon_f)))
lat_f_2d = np.repeat(lat_f[:, np.newaxis], nlon, axis=1)
dx_f = np.deg2rad(dlon) * R_earth * np.cos(np.deg2rad(lat_f_2d))
# Note: dx_f is 2D (lat, lon), deta_dx is (time, lat, lon-2)
deta_dx = (ssh_values[:, :, 2:] - ssh_values[:, :, :-2]) / (2 * dx_f[:, 1:-1][np.newaxis, :, :])
lon_v_2d, lat_v_2d = np.meshgrid(lon_f[1:-1], lat_f)
f_v = 2 * omega * np.sin(np.deg2rad(lat_v_2d))
v_geo_calc = (g / f_v[np.newaxis, :, :]) * deta_dx

# --- Create full grids with NaNs and unify masks ---
u_geo_full = np.full((ntime, nlat, nlon), np.nan)
v_geo_full = np.full((ntime, nlat, nlon), np.nan)
u_geo_full[:, 1:-1, :] = u_geo_calc
v_geo_full[:, :, 1:-1] = v_geo_calc

# Unify NaN mask (required for accurate land mask)
for t in range(ntime):

    unified_mask = np.isnan(ssh_values[t]) | np.isnan(u_geo_full[t]) | np.isnan(v_geo_full[t])

    u_geo_full[t][unified_mask] = np.nan
    v_geo_full[t][unified_mask] = np.nan

# Define Land Mask (0=ocean, 1=land)
landmask = np.isnan(u_geo_full[0, :, :]).astype(int)
print(f"Geostrophic calculation complete. Land cells: {landmask.sum()}")


# ============================================================================
# STEP 2: CREATE DISPLACEMENT FIELD
# ============================================================================

print("--- STEP 2: Creating Displacement Field ---")

# --- Helper Functions (Copied from original script) ---
def get_shore_nodes(landmask):
    """ Detects shore nodes (land nodes directly next to ocean) using Laplacian. """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4 * landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0).mask.astype(int)
    return shore

def create_displacement_field(landmask, shore_nodes):
    """ Creates displacement field vectors pointing away from shore. """
    nlat, nlon = landmask.shape
    dispU = np.zeros((nlat, nlon))
    dispV = np.zeros((nlat, nlon))
    
    for i in range(nlat):
        for j in range(nlon):
            if shore_nodes[i, j] == 1:
                dlat = 0
                dlon = 0
                
                # North-South gradient
                if i > 0:
                    dlat += landmask[i, j] - landmask[i-1, j]
                if i < nlat-1:
                    dlat += landmask[i+1, j] - landmask[i, j]
                
                # East-West gradient  
                if j > 0:
                    dlon += landmask[i, j] - landmask[i, j-1]
                if j < nlon-1:
                    dlon += landmask[i, j+1] - landmask[i, j]
                
                # Displacement vector points FROM land TO ocean (negative gradient)
                u_disp = -dlon
                v_disp = -dlat
                
                # Normalize and scale
                magnitude = np.sqrt(u_disp**2 + v_disp**2)
                if magnitude > 0:
                    dispU[i, j] = (u_disp / magnitude) * DISPLACEMENT_SPEED
                    dispV[i, j] = (v_disp / magnitude) * DISPLACEMENT_SPEED
    return dispU, dispV

# --- Calculate Displacement ---
shore_nodes = get_shore_nodes(landmask)
dispU, dispV = create_displacement_field(landmask, shore_nodes)

# Expand to all timesteps
dispU_3d = np.tile(dispU[np.newaxis, :, :], (ntime, 1, 1))
dispV_3d = np.tile(dispV[np.newaxis, :, :], (ntime, 1, 1))
print(f"Displacement field created (Speed: {DISPLACEMENT_SPEED} m/s).")


# ============================================================================
# STEP 3: COMBINE VELOCITY FIELDS AND SAVE
# ============================================================================

print("--- STEP 3: Combining and Saving Fields ---")

# Replace NaNs with zeros in the geostrophic field for combination
u_geo_zeros = np.nan_to_num(u_geo_full, nan=0.0)
v_geo_zeros = np.nan_to_num(v_geo_full, nan=0.0)

# Combine fields
u_combined = u_geo_zeros + dispU_3d
v_combined = v_geo_zeros + dispV_3d

# Create xarray dataset for combined velocities
ds_combined = xr.Dataset(
    {
        'U_combined': (['time', 'latitude', 'longitude'], u_combined),
        'V_combined': (['time', 'latitude', 'longitude'], v_combined),
    },
    coords={
        'time': time,
        'latitude': lat_f,
        'longitude': lon_f,
    }
)

# Add attributes
ds_combined['U_combined'].attrs = {'long_name': 'Combined eastward velocity (geostrophic + displacement)', 'units': 'm/s'}
ds_combined['V_combined'].attrs = {'long_name': 'Combined northward velocity (geostrophic + displacement)', 'units': 'm/s'}

# Save combined dataset  
output_file = f'{OUTPUT_DIR}/geostrophic_velocity_field.nc'
ds_combined.to_netcdf(output_file)
print(f"Successfully saved combined data to: {output_file}")

#%%
# ============================================================================
# STEP 4: VISUALIZATION OF COMBINED FIELD
# ============================================================================

print("--- STEP 4: Visualizing combined field ---")

# Visualize a single timestep
tidx = 0
u2d = u_combined[tidx, :, :]
v2d = v_combined[tidx, :, :]
speed = np.sqrt(u2d**2 + v2d**2)
lon2d, lat2d = np.meshgrid(lon_f, lat_f)

fig = plt.figure(figsize=(12, 9))
ax = plt.axes(projection=ccrs.PlateCarree())
pcm = ax.pcolormesh(lon2d, lat2d, speed, cmap=cmocean.cm.ice_r, shading='auto', transform=ccrs.PlateCarree())
cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Total Speed (m/s)')

nx = max(1, int(len(lon_f) / 80))
ny = max(1, int(len(lat_f) / 80))
skip = (slice(None, None, ny), slice(None, None, nx))
ax.quiver(lon2d[skip], lat2d[skip], u2d[skip], v2d[skip],
          scale=3, scale_units='inches', width=0.001,
          transform=ccrs.PlateCarree(), color='k', 
          label=f'Velocity Vectors (Subsampled by {ny}x{nx})')
ax.hlines(9.8, -70, -60, colors='orange', linestyles='--', label='Southern Release Line')
ax.hlines(18.6, -70, -60, colors='orange', linestyles='--', label='Northern Release Line')
ax.vlines(-69.7, 9.8, 18.6, colors='orange', linestyles='--')
ax.vlines(-60.5, 9.8, 18.6, colors='orange', linestyles='--')
ax.coastlines(resolution='10m', linewidth=0.5)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
tstr = f"Time: {np.datetime_as_string(time[tidx], unit='D')}"
ax.set_title(f'Combined Velocity Field (Geostrophic + Displacement)\n{tstr}', fontsize=14)
ax.legend(loc='upper right')

plt.tight_layout()
viz_output_file = f'{OUTPUT_DIR}/geostrophic_velocity_visualization.png'
plt.savefig(viz_output_file, dpi=200, bbox_inches='tight')

print(f"Saved visualization to: {viz_output_file}")

# %%
