'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- Calculats geostrophic flow from SSH
- Defines EDDY-flow regime based on speed and direction criteria
- Computes annual statistics of EDDY-flow regime occurrence
- Plots annual EDDY-flow regime days with seasonal breakdown (Figure 3 in manuscript)

Author: vesnaber
kernel: parcels-dev-local
'''

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cmocean
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress
from matplotlib.lines import Line2D

#%%
# =============================================================================
# LOAD DATA AND CALCULATE GEOSTROPHIC VELOCITIES
# =============================================================================

print("Loading SSH datasets...")
ds1 = xr.open_dataset("data/cmems_mod_glo_phy_my_0.083deg_P1D-m_1759843361350.nc")
ds2 = xr.open_dataset("data/cmems_mod_glo_phy_myint_0.083deg_P1D-m_1759843398751.nc")
ds = xr.concat([ds1, ds2], dim='time')

# Standardize time
ds['time'] = pd.to_datetime(ds['time'].values)
full_time = pd.date_range("1993-01-01", "2024-12-31", freq='D')
ds = ds.reindex(time=full_time)

# Define constants
g = 9.81  # m/s^2
Omega = 7.2921e-5  # rad/s
deg2rad = np.pi / 180
Re = 6.371e6  # Earth radius in m

# Extract coordinates
lat = ds['latitude']
lon = ds['longitude']
ssh = ds['zos']

#%%
# =============================================================================
# COMPUTE CORIOLIS PARAMETER AND GRID SPACING 
# =============================================================================

# Coriolis parameter: f = 2Ωsin(φ)
f = 2 * Omega * np.sin(np.deg2rad(lat))
f = xr.DataArray(f, coords=[lat], dims=['latitude'])

# Grid spacing in radians
dlat = np.gradient(lat.values) * deg2rad
dlon = np.gradient(lon.values) * deg2rad

# Meridional distance (constant with longitude)
dy = Re * dlat
dy_da = xr.DataArray(dy, coords=[lat], dims=['latitude'])

# Zonal distance (varies with latitude due to spherical Earth)
lat_2d = lat.values[:, np.newaxis]
dlon_2d = dlon[np.newaxis, :]
dx = Re * np.cos(np.deg2rad(lat_2d)) * dlon_2d
dx_da = xr.DataArray(dx, coords=[lat, lon], dims=['latitude', 'longitude'])

#%%
# =============================================================================
# CALCULATE SSH GRADIENTS USING CENTERED DIFFERENCES 
# =============================================================================

print("Calculating SSH gradients using centeral differences...")

# Latitude gradient (meridional)
ssh_y_numerator = ssh.shift(latitude=-1) - ssh.shift(latitude=1)
ssh_y = ssh_y_numerator / (2 * dy_da)
ssh_y = ssh_y.isel(latitude=slice(1, -1))

# Longitude gradient (zonal)
ssh_x_numerator = ssh.shift(longitude=-1) - ssh.shift(longitude=1)
ssh_x = ssh_x_numerator / (2 * dx_da)
ssh_x = ssh_x.isel(longitude=slice(1, -1))

# Adjust Coriolis parameter to match reduced latitude grid
f_interp = f.isel(latitude=slice(1, -1))

#%%
# =============================================================================
# CALCULATE GEOSTROPHIC VELOCITIES
# =============================================================================

u = -g / f_interp * ssh_y  # Zonal velocity (eastward)
v = g / f_interp * ssh_x   # Meridional velocity (northward)

#%%
# =============================================================================
# SELECT CROSS-SECTION AND REGION OF INTEREST 
# =============================================================================

# Select longitude closest to target (-68.99999°W)
target_lon = -68.99999
lon_idx = np.argmin(np.abs(u.longitude.values - target_lon))
selected_lon = float(u.longitude.isel(longitude=lon_idx))

uo = u.isel(longitude=lon_idx)
vo = v.isel(longitude=lon_idx)

# Select latitude range between Curaçao and Venezuela
lat_range = slice(11.4, 12.2)
uo_region = uo.sel(latitude=lat_range)
vo_region = vo.sel(latitude=lat_range)

uo_eastward_region = uo_region
vo_eastward_region = vo_region

#%%
# =============================================================================
# CALCULATE SPATIAL STATISTICS 
# =============================================================================

# Calculate flow speed
speed = np.sqrt(uo_eastward_region**2 + vo_eastward_region**2)

# Spatial mean across latitude band for each time step
uo_spatial_mean = uo_eastward_region.mean(dim='latitude')
vo_spatial_mean = vo_eastward_region.mean(dim='latitude')
speed_spatial_mean = speed.mean(dim='latitude')

#%%
# =============================================================================
# DEFINE EDDY REGIME CRITERIA
# =============================================================================

# Speed threshold: bottom 25% of speeds
speed_threshold = float(speed_spatial_mean.quantile(0.25))
weak_flow = speed_spatial_mean < speed_threshold

# Direction calculation (oceanographic convention: 0° = North, 90° = East)
direction_spatial_mean = np.arctan2(uo_spatial_mean, vo_spatial_mean) * 180 / np.pi

# Define normal flow direction range: -90° to -25° (NW direction)
nw_min = -90
nw_max = -25
not_nw_direction = (direction_spatial_mean < nw_min) | (direction_spatial_mean > nw_max)

# EDDY REGIME: Weak flow OR not in NW direction
eddy_regime_complex = weak_flow | not_nw_direction

#%%
# =============================================================================
# PREPARE DATA FOR PLOTS A & B
# =============================================================================

years = range(1993, 2025)

# 1. Annual counts (total and by season)
eddy_days_per_year = []
eddy_days_per_year_mam = []  # March, April, May
eddy_days_per_year_son = []  # September, October, November
eddy_days_per_year_other = []  # All other months

for year in years:
    # Total
    year_mask = (uo_region.time.dt.year == year)
    eddy_days = eddy_regime_complex[year_mask].sum()
    eddy_days_per_year.append(int(eddy_days))
    
    # MAM
    year_mam_mask = (uo_region.time.dt.year == year) & (uo_region.time.dt.month.isin([3, 4, 5]))
    eddy_days_mam = eddy_regime_complex[year_mam_mask].sum()
    eddy_days_per_year_mam.append(int(eddy_days_mam))
    
    # SON
    year_son_mask = (uo_region.time.dt.year == year) & (uo_region.time.dt.month.isin([9, 10, 11]))
    eddy_days_son = eddy_regime_complex[year_son_mask].sum()
    eddy_days_per_year_son.append(int(eddy_days_son))
    
    # Other months
    year_other_mask = (uo_region.time.dt.year == year) & (~uo_region.time.dt.month.isin([3, 4, 5, 9, 10, 11]))
    eddy_days_other = eddy_regime_complex[year_other_mask].sum()
    eddy_days_per_year_other.append(int(eddy_days_other))

# 2. Seasonal contribution percentages
mam_pct_by_year = []
son_pct_by_year = []
other_pct_by_year = []

for year in years:
    total_eddy_days = eddy_days_per_year[year - 1993]
    
    if total_eddy_days > 0:
        mam_pct = (eddy_days_per_year_mam[year - 1993] / total_eddy_days) * 100
        son_pct = (eddy_days_per_year_son[year - 1993] / total_eddy_days) * 100
        other_pct = 100.0 - mam_pct - son_pct
    else:
        mam_pct = son_pct = other_pct = 0
        other_pct = 0
    
    mam_pct_by_year.append(mam_pct)
    son_pct_by_year.append(son_pct)
    other_pct_by_year.append(other_pct) # Retained for calculation clarity

#%%
# =============================================================================
# CREATE FIGURE (PLOTS A & B ONLY)
# =============================================================================

# Use a figure size appropriate for two plots
fig = plt.figure(figsize=(13, 5))

gs = plt.GridSpec(1, 8, figure=fig)


# =============================================================================
# LEFT: ANNUAL BAR CHART WITH SEASONAL BREAKDOWN (PLOT A)
# =============================================================================

# Place Plot A to the left, spanning 4 columns
ax1 = fig.add_subplot(gs[0, :5])

bars1 = ax1.bar(years, eddy_days_per_year_mam, color='darkgoldenrod', 
               edgecolor=None, alpha=0.8, label='MAM: Mar-Apr-May')
bars2 = ax1.bar(years, eddy_days_per_year_son, bottom=eddy_days_per_year_mam,
               color='khaki', edgecolor=None, alpha=0.8, label='SON: Sep-Oct-Nov')
bars3 = ax1.bar(years, eddy_days_per_year_other,
               bottom=np.array(eddy_days_per_year_mam) + np.array(eddy_days_per_year_son),
               color='gray', edgecolor=None, alpha=0.7, label='Other months')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Number of EDDY-flow days', fontsize=12)
ax1.set_title('(a) Annual occurrence of EDDY-flow regime days (1993-2024)', fontsize=13, pad=10)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xlim(1992.5, 2024.5)
ax1.set_xticks(years, minor=True)
ax1.tick_params(axis='x', which='minor', length=4, width=1)


# =============================================================================
# RIGHT: SEASONAL CONTRIBUTION PERCENTAGE WITH TRENDS (PLOT B)
# =============================================================================

# Place Plot B to the right
ax6 = fig.add_subplot(gs[0, 5:])

# Calculate trends
slope_mam, intercept_mam, r_mam, p_mam, _ = linregress(years, mam_pct_by_year)
slope_son, intercept_son, r_son, p_son, _ = linregress(years, son_pct_by_year)

trend_mam = [slope_mam * year + intercept_mam for year in years]
trend_son = [slope_son * year + intercept_son for year in years]

# Plot data and trends
ax6.plot(years, mam_pct_by_year, 'o-', color='goldenrod', linewidth=1, markersize=4, alpha=1)
ax6.plot(years, son_pct_by_year, 's-', color='khaki', linewidth=1, markersize=4, alpha=1)

ax6.plot(years, trend_mam, '--', color='goldenrod', linewidth=2, alpha=1, 
        label=f'MAM: {slope_mam:.2f}%/yr (p={p_mam:.3f})')
ax6.plot(years, trend_son, '--', color='khaki', linewidth=2, alpha=1,
        label=f'SON: {slope_son:.2f}%/yr (p={p_son:.3f})')

ax6.set_xlabel('Year', fontsize=12)
ax6.set_ylabel('Seasonal contribution [%]', fontsize=12)
ax6.set_title('(b) Seasonal shift in EDDY regime occurrence', fontsize=13, pad=10)
ax6.legend(fontsize=10, loc='best', framealpha=1, facecolor='white')
ax6.grid(True, alpha=0.3, axis='both')
ax6.set_xlim(1992.5, 2024.5)
ax6.set_ylim(0, 100)
ax6.set_xticks(years, minor=True)
ax6.tick_params(axis='x', which='minor', length=4, width=1)
ax6.set_facecolor('gray') # Kept 'gray' from original, but lightened slightly for background


# =============================================================================
# SAVE FIGURE
# =============================================================================

plt.tight_layout()
plt.savefig('figures/Fig03_EDDY-flow_annual_statistics.png',
           dpi=300, bbox_inches='tight')

# %%
