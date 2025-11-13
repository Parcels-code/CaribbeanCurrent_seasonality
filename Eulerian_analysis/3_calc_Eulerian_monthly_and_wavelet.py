'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- Calculats geostrophic flow from SSH
- Defines EDDY-flow regime based on speed and direction criteria
- Computes monthly statistics of EDDY-flow regime occurrence and wavelet analysis
- Plots monthly EDDY-flow regime days with seasonal breakdown (Figure 4 in manuscript)
- saves integrated spectrum for Figure 9 of the manuscript

Author: vesnaber
kernel: parcels-dev-local
'''

#%%

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pycwt as wavelet
from scipy.stats import linregress
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
# Fix for numpy compatibility (for wavelet)
import numpy
if not hasattr(numpy, 'int'):
    numpy.int = int

# for changing perspective, change lines:
FLOW_NAME = 'EDDY-flow'
# you also need to change this: eddy_regime_complex = nw_flow_regime !

#%%
# =============================================================================
# LOAD DATA AND CALCULATE GEOSTROPHIC VELOCITIES (UNCHANGED)
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
# COMPUTE CORIOLIS PARAMETER AND GRID SPACING (UNCHANGED)
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
# CALCULATE SSH GRADIENTS USING CENTERED DIFFERENCES (UNCHANGED)
# =============================================================================

ssh_y_numerator = ssh.shift(latitude=-1) - ssh.shift(latitude=1)
ssh_y = ssh_y_numerator / (2 * dy_da)
ssh_y = ssh_y.isel(latitude=slice(1, -1))

ssh_x_numerator = ssh.shift(longitude=-1) - ssh.shift(longitude=1)
ssh_x = ssh_x_numerator / (2 * dx_da)
ssh_x = ssh_x.isel(longitude=slice(1, -1))

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

target_lon = -68.99999
lon_idx = np.argmin(np.abs(u.longitude.values - target_lon))
selected_lon = float(u.longitude.isel(longitude=lon_idx))

uo = u.isel(longitude=lon_idx)
vo = v.isel(longitude=lon_idx)

lat_range = slice(11.4, 12.2)
uo_region = uo.sel(latitude=lat_range)
vo_region = vo.sel(latitude=lat_range)

uo_eastward_region = uo_region
vo_eastward_region = vo_region

#%%
# =============================================================================
# CALCULATE SPATIAL STATISTICS 
# =============================================================================

speed = np.sqrt(uo_eastward_region**2 + vo_eastward_region**2)
uo_spatial_mean = uo_eastward_region.mean(dim='latitude')
vo_spatial_mean = vo_eastward_region.mean(dim='latitude')
speed_spatial_mean = speed.mean(dim='latitude')

#%%
# =============================================================================
# DEFINE FLOW REGIMES (NW-FLOW)
# =============================================================================

speed_threshold = float(speed_spatial_mean.quantile(0.25))
weak_flow = speed_spatial_mean < speed_threshold

direction_spatial_mean = np.arctan2(uo_spatial_mean, vo_spatial_mean) * 180 / np.pi
nw_min = -90
nw_max = -25
not_nw_direction = (direction_spatial_mean < nw_min) | (direction_spatial_mean > nw_max)

eddy_regime_complex = weak_flow | not_nw_direction
nw_flow_regime = ~eddy_regime_complex

# Use nw_flow_regime data for subsequent calculations
# eddy_regime_complex = nw_flow_regime

#%%
# =============================================================================
# PREPARE DATA FOR PLOTS
# =============================================================================

years = range(1993, 2025)


# 1. Monthly distribution (for boxplot)
monthly_eddy_days_by_year = np.zeros((len(years), 12))
for i, year in enumerate(years):
    for month in range(1, 13):
        year_month_mask = (uo_region.time.dt.year == year) & (uo_region.time.dt.month == month)
        eddy_days = eddy_regime_complex[year_month_mask].sum()
        monthly_eddy_days_by_year[i, month-1] = int(eddy_days)

# 2. Monthly percentage by year (for heatmap)
monthly_eddy_pct = np.zeros((len(years), 12))
for i, year in enumerate(years):
    for month in range(1, 13):
        year_month_mask = (uo_region.time.dt.year == year) & (uo_region.time.dt.month == month)
        total_days = year_month_mask.sum()
        if total_days > 0:
            eddy_days = eddy_regime_complex[year_month_mask].sum()
            monthly_eddy_pct[i, month-1] = float(eddy_days / total_days * 100)

# 3. Wavelet analysis
eddy_time_series = eddy_regime_complex.astype(float).values
time_np = pd.to_datetime(uo_region.time.values)
valid_mask = ~np.isnan(eddy_time_series)
eddy_clean = eddy_time_series[valid_mask]
time_clean = time_np[valid_mask]

eddy_norm = (eddy_clean - np.mean(eddy_clean)) / np.std(eddy_clean)
dt = 1
mother = wavelet.Morlet(6)
s0 = 2 * dt
dj = 1/12
J = 11 / dj
alpha, _, _ = wavelet.ar1(eddy_norm)

wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(eddy_norm, dt, dj, s0, J, mother)
power = (np.abs(wave)) ** 2
period = 1 / freqs

signif, _ = wavelet.significance(eddy_norm, dt, scales, 0, alpha, 
                                 significance_level=0.95, wavelet=mother)
sig95 = signif[:, np.newaxis] * np.ones(wave.shape)

global_ws = power.mean(axis=1)
dof = len(eddy_norm) - scales
global_signif, _ = wavelet.significance(eddy_norm, dt, scales, 1, alpha, 
                                       significance_level=0.95, dof=dof, wavelet=mother)

period_months = period / 30.44
coi_months = coi / 30.44

max_power_idx = np.nanargmax(global_ws)
max_period_months = period_months[max_power_idx]

#%%
# =============================================================================
# CREATE FIGURE (A, B, C)
# =============================================================================

fig = plt.figure(figsize=(13, 10)) 

# Grid layout: 2 rows, 3 columns to maintain 2:1 width ratio on the top row
gs = plt.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)


# =============================================================================
# TOP LEFT: MONTHLY BOXPLOT (PLOT A)
# =============================================================================

ax1 = fig.add_subplot(gs[0, :2]) 

monthly_data_for_box = [monthly_eddy_days_by_year[:, month-1] for month in range(1, 13)]
bp = ax1.boxplot(monthly_data_for_box, labels=['J','F','M','A','M','J','J','A','S','O','N','D'],
                patch_artist=True, showmeans=True, widths=0.4,
                meanprops=dict(marker='^', markerfacecolor='lightgrey', markeredgecolor='lightgrey', markersize=7),
                flierprops=dict(marker='o', markerfacecolor='black', markersize=4, 
                              linestyle='none', markeredgecolor='black'),
                medianprops=dict(color='gold', linewidth=2))

for patch in bp['boxes']:
    patch.set_facecolor('black')
    patch.set_alpha(0.6)
    patch.set_edgecolor('black')
    patch.set_linewidth(0)

legend_elements = [Line2D([0], [0], color='gold', linewidth=2, label='Median'),
                  Line2D([0], [0], marker='^', color='w', markerfacecolor='lightgrey', 
                        markersize=7, linestyle='None', label='Mean')]
ax1.legend(handles=legend_elements, fontsize=10, loc='upper left')

ax1.set_xlabel('Month', fontsize=11)
ax1.set_ylabel(f'{FLOW_NAME} days per month', fontsize=11)
ax1.set_title(f'(a) Monthly distribution of {FLOW_NAME} days', fontsize=12, pad=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 32)


# =============================================================================
# TOP RIGHT: HEATMAP (PLOT B)
# =============================================================================

ax2 = fig.add_subplot(gs[0, 2]) 

im = ax2.imshow(monthly_eddy_pct, aspect='auto', cmap=cmocean.cm.rain_r,
               origin='lower', vmin=0, vmax=100)
ax2.set_xlabel('Month', fontsize=11)
ax2.set_ylabel('Year', fontsize=11)
ax2.set_title(f'(b) Monthly {FLOW_NAME} regime\noccurrence [%]', fontsize=12, pad=10)
ax2.set_xticks(range(12))
ax2.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=9)

# Y-axis ticks
year_indices = [2, 7, 12, 17, 22, 27]
year_labels = [1995, 2000, 2005, 2010, 2015, 2020]
ax2.set_yticks(year_indices)
ax2.set_yticklabels(year_labels, fontsize=9)
ax2.set_yticks(range(len(years)), minor=True)
ax2.tick_params(axis='y', which='minor', length=3, width=1)

# Colorbar for Plot B
cbar2 = fig.add_axes([0.915, 0.55, 0.015, 0.325])
cbar = plt.colorbar(im, cax=cbar2, orientation='vertical')
cbar.set_label('% of days', fontsize=11)


# =============================================================================
# BOTTOM: WAVELET POWER SPECTRUM (PLOT C) - FULL WIDTH
# =============================================================================

# Position (1, 0) spanning all 3 columns
ax3 = fig.add_subplot(gs[1, :])

# Determine colorbar range
max_power = np.percentile(power, 95)
if max_power < 10:
    max_power_rounded = np.ceil(max_power)
elif max_power < 50:
    max_power_rounded = np.ceil(max_power / 5) * 5
else:
    max_power_rounded = np.ceil(max_power / 10) * 10

levels = np.linspace(0, max_power_rounded, 100)

contourf = ax3.contourf(time_clean, period_months, power,
                       levels=levels, cmap=cmocean.cm.amp_r, extend='max')
ax3.contour(time_clean, period_months, power / sig95, levels=[1],
           colors='blue', linewidths=1.5, alpha=0.8, linestyles=':')
ax3.fill_between(time_clean, coi_months, period_months.max(),
                color='white', alpha=0.5, hatch='///',
                edgecolor='white', linewidth=0.8, label='Cone of influence')

ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Period [months]', fontsize=11)
ax3.set_title(f'(c) Wavelet power spectrum of {FLOW_NAME} occurrence\n(95% significance in blue contour dots)',
             fontsize=12, pad=10)
ax3.set_yscale('log')
ax3.set_ylim([period_months.min(), period_months.max()])

yticks = [0.1, 0.5, 1, 6, 12, 24, 60]
ax3.set_yticks(yticks)
ax3.set_yticklabels(yticks)

major_years = pd.date_range(start='1995-01-01', end='2024-01-01', freq='5YS')
ax3.set_xticks(major_years)
ax3.set_xticklabels([str(yr.year) for yr in major_years])
minor_years = pd.date_range(start=time_clean[0], end=time_clean[-1], freq='YS')
ax3.set_xticks(minor_years, minor=True)
ax3.tick_params(axis='x', which='minor', length=4, width=1)

ax3.axhline(1, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
ax3.axhline(6, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
ax3.axhline(12, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
ax3.invert_yaxis()

# Colorbar for Plot C (placed below the full width)
cbar_ax = fig.add_axes([0.13, 0.04, 0.765, 0.014]) # Adjusted position and width to span the full figure
cbar = plt.colorbar(contourf, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Normalized power [ ]', fontsize=11)
num_ticks = 5
cbar_ticks = np.linspace(0, max_power_rounded, num_ticks)
cbar.set_ticks(cbar_ticks)

ax3.text(0.005, 0.98, f'Max: {max_period_months:.1f} months', # Adjusted x-position slightly
        transform=ax3.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='square,pad=0.3', facecolor='yellow', alpha=0.9))


# =============================================================================
# SAVE FIGURE
# =============================================================================

# Adjusted rect for tight_layout to account for the single colorbar and general size
fig.tight_layout(rect=[0, 0.05, 1, 1]) 
plt.savefig(f'figures/Fig04_{FLOW_NAME}_monthly_and_wavelet.png', 
           dpi=300, bbox_inches='tight')

# %%
# =============================================================================
# DATA EXTRACTION FOR INTEGRATED SPECTRA FIGURE
# =============================================================================

# necessary arrays
plot_d_data = {
    'period_months': period_months,        # Y-axis / Period scale
    'global_ws': global_ws,                # Global Spectrum (GWS) values
    'global_signif': global_signif         # 95% Significance values
}

# filename and directory
output_dir = '../integrated_spectra_combined/data_wavelet'
filename = os.path.join(output_dir, 'eulerian_flow_integrated_spectrum.pkl')

# save data using pickle
with open(filename, 'wb') as f:
    pickle.dump(plot_d_data, f)

# %%
