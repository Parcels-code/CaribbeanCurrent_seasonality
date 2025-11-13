'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
    - Defines particle release locations for Parcels run GRENAVENE
    - Saves the locations as numpy arrays

Author: vesnaber
Kernel: parcels-dev-local
'''

#%%
# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import xarray as xr
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import cmocean.cm as cmo

#%%
# =============================================================================
# LOAD DATA
# =============================================================================

ds = xr.open_dataset(
    "GLORYS_input/cmems_mod_glo_phy_myint_0.083deg_P1D-m_zos_70.00W-58.92W_9.50N-18.83N_2021-07-01-2024-12-31.nc"
)

#%%
# =============================================================================
# DEFINE PARTICLE RELEASE LOCATIONS
# =============================================================================

# Define start and end points of release line
lon_start = -62.5
lat_start = 10.72
lon_end = -61.7
lat_end = 12.04

# Calculate distance in km using Haversine formula
R = 6371.0  # Earth radius in km
lat1, lon1 = radians(lat_start), radians(lon_start)
lat2, lon2 = radians(lat_end), radians(lon_end)
dlat = lat2 - lat1
dlon = lon2 - lon1

a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
c = 2 * atan2(sqrt(a), sqrt(1 - a))
distance_km = R * c
print(f"Distance between points: {distance_km:.2f} km")

# Create particle locations (10 particles per km)
n_particles = int(distance_km * 10)
lon = np.linspace(lon_start, lon_end, n_particles)
lat = np.linspace(lat_start, lat_end, n_particles)

#%%
# =============================================================================
# VISUALIZE PARTICLE LOCATIONS
# =============================================================================

plt.figure(figsize=(10, 6))

# Plot ocean current at t=0
ds.zos.isel(time=0).plot(cmap=cmo.ice_r, vmin=-0.5, vmax=0.5)

# Plot particle release locations
plt.scatter(lon, lat, color='red', s=1, label='Particle Release Locations')

plt.title('Ocean Current at t=0')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-64, -60)
plt.ylim(10, 13)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()

#%%
# =============================================================================
# SAVE PARTICLE LOCATIONS
# =============================================================================

np.save('parcels_input/particles_GRENAVENE_lon.npy', lon)
np.save('parcels_input/particles_GRENAVENE_lat.npy', lat)
print(f"Saved {n_particles} particle locations to parcels_input/")

#%%