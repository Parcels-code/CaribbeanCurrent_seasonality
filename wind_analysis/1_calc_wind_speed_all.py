'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- calculates the wind speed time series from CMEMS wind data
- computes monthly statistics and wavelet spectra for two locations (Grenada and Curaçao)
- creates a combined figure with climatology and wavelet spectrum (Figure 8 in the manuscript)
- saves integrated spectrum for Figure 9 of the manuscript

Author: vesnaber
Kernel: parcels-dev-local
'''

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cmocean
import pycwt as wavelet
from scipy.stats import linregress
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
import os

# Fix for numpy compatibility
if not hasattr(np, 'int'):
    np.int = int

#%%
# =============================================================================
# LOAD WIND DATA
# =============================================================================

print("Loading wind datasets...")

ds_gre1 = xr.open_dataset("data/cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H_1755519522043.nc")
ds_gre2 = xr.open_dataset("data/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_1755519644238.nc")
ds_gre1 = ds_gre1.isel(longitude=0)
ds_gre2 = ds_gre2.isel(longitude=0, latitude=1)

ds_cur1 = xr.open_dataset("data/cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H_1755519716959.nc")
ds_cur2 = xr.open_dataset("data/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_1755519662689.nc")
ds_cur2 = ds_cur2.isel(longitude=1, latitude=-1)

# Compute wind speeds
wind_speed_gre1 = np.sqrt(ds_gre1['eastward_wind']**2 + ds_gre1['northward_wind']**2)
wind_speed_gre2 = np.sqrt(ds_gre2['eastward_wind']**2 + ds_gre2['northward_wind']**2)
wind_speed_cur1 = np.sqrt(ds_cur1['eastward_wind']**2 + ds_cur1['northward_wind']**2)
wind_speed_cur2 = np.sqrt(ds_cur2['eastward_wind']**2 + ds_cur2['northward_wind']**2)

# Convert to daily averages
daily_gre1 = wind_speed_gre1.resample(time='1D').mean()
daily_gre2 = wind_speed_gre2.resample(time='1D').mean()
daily_cur1 = wind_speed_cur1.resample(time='1D').mean()
daily_cur2 = wind_speed_cur2.resample(time='1D').mean()

# Combine data for each location
df_gre1 = daily_gre1.to_dataframe(name='wind_speed').reset_index()
df_gre2 = daily_gre2.to_dataframe(name='wind_speed').reset_index()
df_cur1 = daily_cur1.to_dataframe(name='wind_speed').reset_index()
df_cur2 = daily_cur2.to_dataframe(name='wind_speed').reset_index()

# Merge GRE data
gre_combined = pd.concat([df_gre1, df_gre2], ignore_index=True)
gre_df = gre_combined.groupby('time')['wind_speed'].mean().reset_index()
gre_df = gre_df.dropna()

# Merge CUR data
cur_combined = pd.concat([df_cur1, df_cur2], ignore_index=True)
cur_df = cur_combined.groupby('time')['wind_speed'].mean().reset_index()
cur_df = cur_df.dropna()

print(f"GRE data: {len(gre_df)} daily records")
print(f"CUR data: {len(cur_df)} daily records")

#%%
# =============================================================================
# PREPARE DATA FOR WAVELET ANALYSIS
# =============================================================================

# Create xarray DataArrays
gre_data = xr.DataArray(
    gre_df['wind_speed'].values,
    coords=[gre_df['time'].values],
    dims=['time'],
    name='wind_speed'
)

cur_data = xr.DataArray(
    cur_df['wind_speed'].values,
    coords=[cur_df['time'].values],
    dims=['time'],
    name='wind_speed'
)

#%%
# =============================================================================
# CALCULATE MONTHLY STATISTICS FOR BOXPLOT
# =============================================================================

# Add month and year columns
gre_df['month'] = pd.to_datetime(gre_df['time']).dt.month
gre_df['year'] = pd.to_datetime(gre_df['time']).dt.year
cur_df['month'] = pd.to_datetime(cur_df['time']).dt.month
cur_df['year'] = pd.to_datetime(cur_df['time']).dt.year

# Calculate monthly averages for each year
years = range(max(gre_df['year'].min(), cur_df['year'].min()), 
              min(gre_df['year'].max(), cur_df['year'].max()) + 1)

gre_monthly_by_year = np.zeros((len(years), 12))
cur_monthly_by_year = np.zeros((len(years), 12))

for i, year in enumerate(years):
    for month in range(1, 13):
        # GRE
        gre_month_data = gre_df[(gre_df['year'] == year) & (gre_df['month'] == month)]
        if len(gre_month_data) > 0:
            gre_monthly_by_year[i, month-1] = gre_month_data['wind_speed'].mean()
        else:
            gre_monthly_by_year[i, month-1] = np.nan
        
        # CUR
        cur_month_data = cur_df[(cur_df['year'] == year) & (cur_df['month'] == month)]
        if len(cur_month_data) > 0:
            cur_monthly_by_year[i, month-1] = cur_month_data['wind_speed'].mean()
        else:
            cur_monthly_by_year[i, month-1] = np.nan

# Prepare data for combined boxplot
gre_monthly_data = [gre_monthly_by_year[:, month-1][~np.isnan(gre_monthly_by_year[:, month-1])] 
                    for month in range(1, 13)]
cur_monthly_data = [cur_monthly_by_year[:, month-1][~np.isnan(cur_monthly_by_year[:, month-1])] 
                    for month in range(1, 13)]

#%%
# =============================================================================
# WAVELET ANALYSIS FUNCTION
# =============================================================================

def perform_wavelet_analysis(data, dt_days=1):
    """
    Perform wavelet analysis on wind speed data
    """
    wind_values = data.values
    time_values = pd.to_datetime(data.time.values)
    
    # Remove NaN values
    valid_mask = ~np.isnan(wind_values)
    wind_clean = wind_values[valid_mask]
    time_clean = time_values[valid_mask]
    
    if len(wind_clean) < 100:
        print("Not enough clean data for wavelet analysis.")
        return None 

    # Normalize the signal
    wind_norm = (wind_clean - np.mean(wind_clean)) / np.std(wind_clean)
    
    # Wavelet parameters
    mother = wavelet.Morlet(6)
    dt = dt_days
    s0 = 2 * dt
    dj = 1/12
    J = 11 / dj
    alpha, _, _ = wavelet.ar1(wind_norm)
    
    # Compute wavelet transform
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(wind_norm, dt, dj, s0, J, mother)
    power = (np.abs(wave)) ** 2
    period = 1 / freqs
    
    # Significance level
    signif, _ = wavelet.significance(wind_norm, dt, scales, 0, alpha, 
                                   significance_level=0.95, wavelet=mother)
    sig95 = signif[:, np.newaxis] * np.ones(wave.shape)
    
    # Global wavelet spectrum
    global_ws = power.mean(axis=1)
    dof = len(wind_clean) - scales
    global_signif, _ = wavelet.significance(wind_norm, dt, scales, 1, alpha, 
                                          significance_level=0.95, dof=dof, wavelet=mother)
    
    # Convert periods to months
    period_months = period / 30.44
    coi_months = coi / 30.44
    
    # Find maximum power period
    max_power_idx = np.nanargmax(global_ws)
    max_period_months = period_months[max_power_idx]
    
    return {
        'time_clean': time_clean,
        'power': power,
        'period_months': period_months,
        'sig95': sig95,
        'coi_months': coi_months,
        'global_ws': global_ws,
        'global_signif': global_signif,
        'max_period_months': max_period_months
    }

# Perform wavelet analysis
print("Performing wavelet analysis...")
gre_wavelet = perform_wavelet_analysis(gre_data)
cur_wavelet = perform_wavelet_analysis(cur_data)
if gre_wavelet is None or cur_wavelet is None:
    raise RuntimeError("Wavelet analysis failed due to insufficient data.")

#%%
# =============================================================================
# CREATE FINAL FIGURE
# =============================================================================

# Define locations for the map
grenada_wind_lon, grenada_wind_lat = -61.94, 11.69
curacao_wind_lon, curacao_wind_lat = -68.94, 11.94

fig = plt.figure(figsize=(16, 14))
gs = plt.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25, height_ratios=[1, 1, 1.2])

# Color scheme for locations
gre_color = 'olivedrab'
cur_color = 'firebrick'

# =============================================================================
# TOP LEFT: MAP (A)
# =============================================================================

ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())

# Set extent to cover the two locations
ax1.set_extent([-71, -60, 9, 14.5], crs=ccrs.PlateCarree()) 

# Add map features
ax1.add_feature(cfeature.LAND, facecolor='saddlebrown', alpha=0.5, edgecolor='saddlebrown')
ax1.add_feature(cfeature.OCEAN, facecolor='#f7f7f7')
ax1.coastlines(resolution='50m', linewidth=0.8, color='saddlebrown')

# Add gridlines
gl_a = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl_a.top_labels = False
gl_a.right_labels = False
gl_a.xlabel_style = {'size': 10}
gl_a.ylabel_style = {'size': 10}

# Wind icon plotting function
def plot_wind_icon(ax, lon, lat, color, edge_color):
    """Plot a wind icon with three curved lines"""
    # Background circle
    ax.scatter(lon, lat, c=color, s=200, marker='o',
              edgecolor=edge_color, linewidth=2, transform=ccrs.PlateCarree(), zorder=29)
    
    # Three horizontal curved lines (wind streaks)
    line_offsets = [-0.04, 0, 0.04]
    for offset in line_offsets:
        x_line = [lon - 0.08, lon - 0.03, lon + 0.02, lon + 0.08]
        y_line = [lat + offset - 0.01, lat + offset, lat + offset, lat + offset + 0.01]
        ax.plot(x_line, y_line, color='white', linewidth=1,
               transform=ccrs.PlateCarree(), zorder=30, solid_capstyle='round')

# Plot Grenada wind station and label
plot_wind_icon(ax1, grenada_wind_lon, grenada_wind_lat, gre_color, 'darkgreen')
ax1.text(grenada_wind_lon - 0.8, grenada_wind_lat + 0.5, 'Grenada', 
         fontsize=10, color=gre_color, fontweight='bold', transform=ccrs.PlateCarree())

# Plot Curaçao wind station and label
plot_wind_icon(ax1, curacao_wind_lon, curacao_wind_lat, cur_color, 'darkred')
ax1.text(curacao_wind_lon - 0.8, curacao_wind_lat + 0.5, 'Curaçao', 
         fontsize=10, color=cur_color, fontweight='bold', transform=ccrs.PlateCarree())

ax1.set_title('(a) Wind data locations', fontsize=12, pad=10)

# =============================================================================
# TOP RIGHT: MONTHLY BOXPLOTS (B)
# =============================================================================

ax2 = fig.add_subplot(gs[0, 1])

positions = np.arange(1, 13)
x_ticks = positions
x_labels = ['J','F','M','A','M','J','J','A','S','O','N','D']

# Adjust positions for side-by-side plots
gre_positions = positions - 0.2
cur_positions = positions + 0.2
widths = 0.35

# Boxplot GRE
bp_gre = ax2.boxplot(gre_monthly_data, labels=None, positions=gre_positions,
                  patch_artist=True, showmeans=True, widths=widths, manage_ticks=False,
                  meanprops=dict(marker='^', markerfacecolor='lightgrey', 
                                markeredgecolor='lightgrey', markersize=7),
                  flierprops=dict(marker='o', markerfacecolor=gre_color, markersize=4, 
                                linestyle='none', markeredgecolor=gre_color),
                  medianprops=dict(color='gold', linewidth=2))

for patch in bp_gre['boxes']:
    patch.set_facecolor(gre_color)
    patch.set_alpha(0.6)
    patch.set_edgecolor(gre_color)
    patch.set_linewidth(1)

# Boxplot CUR
bp_cur = ax2.boxplot(cur_monthly_data, labels=None, positions=cur_positions,
                  patch_artist=True, showmeans=True, widths=widths, manage_ticks=False,
                  meanprops=dict(marker='^', markerfacecolor='lightgrey', 
                                markeredgecolor='lightgrey', markersize=7),
                  flierprops=dict(marker='s', markerfacecolor=cur_color, markersize=4, 
                                linestyle='none', markeredgecolor=cur_color),
                  medianprops=dict(color='gold', linewidth=2))

for patch in bp_cur['boxes']:
    patch.set_facecolor(cur_color)
    patch.set_alpha(0.6)
    patch.set_edgecolor(cur_color)
    patch.set_linewidth(1)

ax2.set_xticks(x_ticks)
ax2.set_xticklabels(x_labels)
ax2.set_xlabel('Month', fontsize=11)
ax2.set_ylabel('Wind speed [m/s]', fontsize=11)
ax2.set_title('(b) Monthly wind speed distribution', fontsize=12, pad=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(3, ax2.get_ylim()[1])

# Add legend
legend_elements = [
    Rectangle((0, 0), 1, 1, facecolor=gre_color, alpha=0.6, edgecolor=gre_color, label='Grenada'),
    Rectangle((0, 0), 1, 1, facecolor=cur_color, alpha=0.6, edgecolor=cur_color, label='Curaçao'),
    Line2D([0], [0], color='gold', linewidth=2, label='Median'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='lightgrey', 
           markersize=7, linestyle='None', label='Mean')
]
ax2.legend(handles=legend_elements, fontsize=9, loc='lower left', ncol=2)

# =============================================================================
# MIDDLE ROW: TIME SERIES (C)
# =============================================================================

ax3 = fig.add_subplot(gs[1, :])

# Plot daily values as thin background
ax3.plot(gre_df['time'], gre_df['wind_speed'], color=gre_color, linewidth=0.3, 
         alpha=0.15, zorder=1)
ax3.plot(cur_df['time'], cur_df['wind_speed'], color=cur_color, linewidth=0.3, 
         alpha=0.15, zorder=1)

# Plot 30-day rolling mean
gre_rolling = gre_df.set_index('time')['wind_speed'].rolling('30D', center=True).mean()
cur_rolling = cur_df.set_index('time')['wind_speed'].rolling('30D', center=True).mean()

ax3.plot(gre_rolling.index, gre_rolling.values, color=gre_color, linewidth=1.5, 
         alpha=0.9, zorder=5)
ax3.plot(cur_rolling.index, cur_rolling.values, color=cur_color, linewidth=1.5, 
         alpha=0.9, zorder=5)

ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Wind Speed [m/s]', fontsize=11)
ax3.set_title('(c) Wind Speed Time Series (Daily and 30-day rolling mean)', fontsize=12, pad=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(pd.Timestamp('1994-01-01'), pd.Timestamp('2025-01-01'))
ax3.set_ylim(1, 14)

legend_elements_c = [
    Line2D([0], [0], color=gre_color, linewidth=1.8, alpha=0.9, label='Grenada (30-day rolling mean)'),
    Line2D([0], [0], color=cur_color, linewidth=1.8, alpha=0.9, label='Curaçao (30-day rolling mean)'),
    Line2D([0], [0], color='k', linewidth=0.5, alpha=0.3, label='Daily values'),
]
ax3.legend(handles=legend_elements_c, loc='upper right', fontsize=10)

# Format x-axis
ax3.xaxis.set_major_locator(mdates.YearLocator(5))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.xaxis.set_minor_locator(mdates.YearLocator())

# =============================================================================
# BOTTOM ROW: WAVELET POWER SPECTRA (D and E)
# =============================================================================

# GRE Wavelet (D)
ax4 = fig.add_subplot(gs[2, 0])

# Determine colorbar range for GRE
max_power_gre = np.percentile(gre_wavelet['power'], 95)
if max_power_gre < 10:
    max_power_rounded_gre = np.ceil(max_power_gre)
elif max_power_gre < 50:
    max_power_rounded_gre = np.ceil(max_power_gre / 5) * 5
else:
    max_power_rounded_gre = np.ceil(max_power_gre / 10) * 10

levels_gre = np.linspace(0, max_power_rounded_gre, 100)

contourf_gre = ax4.contourf(gre_wavelet['time_clean'], gre_wavelet['period_months'], 
                            gre_wavelet['power'], levels=levels_gre, 
                            cmap=cmocean.cm.amp_r, extend='max')
ax4.contour(gre_wavelet['time_clean'], gre_wavelet['period_months'], 
           gre_wavelet['power'] / gre_wavelet['sig95'], levels=[1],
           colors='blue', linewidths=1.5, alpha=0.8, linestyles=':')
ax4.fill_between(gre_wavelet['time_clean'], gre_wavelet['coi_months'], 
                 gre_wavelet['period_months'].max(),
                 color='white', alpha=0.5, hatch='///',
                 edgecolor='white', linewidth=0.8)

ax4.set_xlabel('Year', fontsize=11)
ax4.set_ylabel('Period [months]', fontsize=11)
ax4.set_title('(d) Grenada: Wavelet power spectrum\n(95% significance in blue contour)',
             fontsize=12, pad=10)
ax4.set_yscale('log')
ax4.set_ylim([gre_wavelet['period_months'].min(), gre_wavelet['period_months'].max()])

# Y-axis ticks
yticks = [0.1, 0.5, 1, 6, 12, 24, 60]
ax4.set_yticks(yticks)
ax4.set_yticklabels(yticks)

# X-axis formatting
major_years = pd.date_range(start='1995-01-01', end='2025-01-01', freq='5YS')
ax4.set_xticks(major_years)
ax4.set_xticklabels([str(yr.year) for yr in major_years])
minor_years = pd.date_range(start='1995-01-01', end='2025-01-01', freq='YS')
ax4.set_xticks(minor_years, minor=True)
ax4.tick_params(axis='x', which='minor', length=4, width=1)

# Reference lines
ax4.axhline(1, color='grey', linestyle=':', linewidth=1.5, alpha=0.8)
ax4.axhline(6, color='grey', linestyle=':', linewidth=1.5, alpha=0.8)
ax4.axhline(12, color='grey', linestyle=':', linewidth=1.5, alpha=0.8)
ax4.invert_yaxis()

# Max period annotation
ax4.text(0.02, 0.98, f'Max: {gre_wavelet["max_period_months"]:.1f} months',
        transform=ax4.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='square,pad=0.3', facecolor='yellow', alpha=0.9))

# CUR Wavelet (E)
ax5 = fig.add_subplot(gs[2, 1])

# Determine colorbar range for CUR
max_power_cur = np.percentile(cur_wavelet['power'], 95)
if max_power_cur < 10:
    max_power_rounded_cur = np.ceil(max_power_cur)
elif max_power_cur < 50:
    max_power_rounded_cur = np.ceil(max_power_cur / 5) * 5
else:
    max_power_rounded_cur = np.ceil(max_power_cur / 10) * 10

levels_cur = np.linspace(0, max_power_rounded_cur, 100)

contourf_cur = ax5.contourf(cur_wavelet['time_clean'], cur_wavelet['period_months'], 
                            cur_wavelet['power'], levels=levels_cur, 
                            cmap=cmocean.cm.amp_r, extend='max')
ax5.contour(cur_wavelet['time_clean'], cur_wavelet['period_months'], 
           cur_wavelet['power'] / cur_wavelet['sig95'], levels=[1],
           colors='blue', linewidths=1.5, alpha=0.8, linestyles=':')
ax5.fill_between(cur_wavelet['time_clean'], cur_wavelet['coi_months'], 
                 cur_wavelet['period_months'].max(),
                 color='white', alpha=0.5, hatch='///',
                 edgecolor='white', linewidth=0.8)

ax5.set_xlabel('Year', fontsize=11)
ax5.set_ylabel('Period [months]', fontsize=11)
ax5.set_title('(e) Curaçao: Wavelet power spectrum\n(95% significance in blue contour)',
             fontsize=12, pad=10)
ax5.set_yscale('log')
ax5.set_ylim([cur_wavelet['period_months'].min(), cur_wavelet['period_months'].max()])

# Y-axis ticks
ax5.set_yticks(yticks)
ax5.set_yticklabels(yticks)

# X-axis formatting
ax5.set_xticks(major_years)
ax5.set_xticklabels([str(yr.year) for yr in major_years])
ax5.set_xticks(minor_years, minor=True)
ax5.tick_params(axis='x', which='minor', length=4, width=1)

# Reference lines
ax5.axhline(1, color='grey', linestyle=':', linewidth=1.5, alpha=0.8)
ax5.axhline(6, color='grey', linestyle=':', linewidth=1.5, alpha=0.8)
ax5.axhline(12, color='grey', linestyle=':', linewidth=1.5, alpha=0.8)
ax5.invert_yaxis()

# Max period annotation
ax5.text(0.02, 0.98, f'Max: {cur_wavelet["max_period_months"]:.1f} months',
        transform=ax5.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='square,pad=0.3', facecolor='yellow', alpha=0.9))

# =============================================================================
# COLORBARS
# =============================================================================

# Colorbar for GRE wavelet
cbar_ax_gre = fig.add_axes([0.125, 0.055, 0.35, 0.012])
cbar_gre = plt.colorbar(contourf_gre, cax=cbar_ax_gre, orientation='horizontal')
cbar_gre.set_label('Normalized power [ ]', fontsize=10)
cbar_gre.set_ticks(np.linspace(0, max_power_rounded_gre, 5))

# Colorbar for CUR wavelet
cbar_ax_cur = fig.add_axes([0.555, 0.055, 0.35, 0.012])
cbar_cur = plt.colorbar(contourf_cur, cax=cbar_ax_cur, orientation='horizontal')
cbar_cur.set_label('Normalized power [ ]', fontsize=10)
cbar_cur.set_ticks(np.linspace(0, max_power_rounded_cur, 5))

# =============================================================================
# FINAL ADJUSTMENTS AND SAVE
# =============================================================================

fig.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig('figures/Fig08_wind_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# SAVE GLOBAL WAVELET SPECTRUM DATA
# =============================================================================

# Extract data for Grenada
gre_ps_data = {
    'period_months': gre_wavelet['period_months'],
    'global_ws': gre_wavelet['global_ws'],
    'global_signif': gre_wavelet['global_signif']
}

# Extract data for Curaçao
cur_ps_data = {
    'period_months': cur_wavelet['period_months'],
    'global_ws': cur_wavelet['global_ws'],
    'global_signif': cur_wavelet['global_signif']
}

output_dir = '../integrated_spectra_combined/data_wavelet'

# Create directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save GRE data
gre_filename = os.path.join(output_dir, 'wind_GRE_integrated_spectrum.pkl')
with open(gre_filename, 'wb') as f:
    pickle.dump(gre_ps_data, f)
print(f"\nGRE integrated spectrum data saved to: {gre_filename}")

# Save CUR data
cur_filename = os.path.join(output_dir, 'wind_CUR_integrated_spectrum.pkl')
with open(cur_filename, 'wb') as f:
    pickle.dump(cur_ps_data, f)
print(f"CUR integrated spectrum data saved to: {cur_filename}")

#%%
# =============================================================================
# PLOT GLOBAL WAVELET SPECTRUM COMPARISON
# =============================================================================

def load_spectrum_data(filename):
    """Load spectrum data from pickle file"""
    if not os.path.exists(filename):
        print(f"Error: Data file not found at {filename}.")
        return None
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load data
gre_filename = 'POWER_SPECTRUM_DATA/GRE_integrated_spectrum.pkl'
cur_filename = 'POWER_SPECTRUM_DATA/CUR_integrated_spectrum.pkl'

gre_data = load_spectrum_data(gre_filename)
cur_data = load_spectrum_data(cur_filename)

if not gre_data or not cur_data:
    raise FileNotFoundError("Could not load both GRE and CUR data files.")

# Create figure
fig, ax = plt.subplots(figsize=(6, 8))

# Define colors
GRE_FILL = 'firebrick'
GRE_LINE = 'darkred'
CUR_FILL = 'olivedrab'
CUR_LINE = 'darkgreen'
SIG_LINESTYLE = ':'

# Plot GRE Spectrum
ax.fill_betweenx(gre_data['period_months'], 0, gre_data['global_ws'], 
                 color=GRE_FILL, alpha=0.3)
ax.plot(gre_data['global_ws'], gre_data['period_months'], color=GRE_LINE, 
        linewidth=2, label='GRE Global Spectrum')
ax.plot(gre_data['global_signif'], gre_data['period_months'], color=GRE_LINE, 
        linestyle=SIG_LINESTYLE, linewidth=1.5, label='GRE 95% Sig.')

# Plot CUR Spectrum
ax.fill_betweenx(cur_data['period_months'], 0, cur_data['global_ws'], 
                 color=CUR_FILL, alpha=0.3)
ax.plot(cur_data['global_ws'], cur_data['period_months'], color=CUR_LINE, 
        linewidth=2, label='CUR Global Spectrum')
ax.plot(cur_data['global_signif'], cur_data['period_months'], color=CUR_LINE, 
        linestyle=SIG_LINESTYLE, linewidth=1.5, label='CUR 95% Sig.')

# Axis formatting
ax.set_xlabel('Normalized Power [ ]', fontsize=11)
ax.set_ylabel('Period [months]', fontsize=11)
ax.set_title('Global Wavelet Spectrum Comparison', fontsize=13, pad=10) 

ax.set_yscale('log')
ax.set_ylim([0.1, 60])

# Y-axis ticks
yticks = [0.1, 0.5, 1, 6, 12, 24, 60]
ax.set_yticks(yticks)
ax.set_yticklabels(yticks)

ax.set_xlim(0, max(ax.get_xlim()[1], 10))

# Reference lines
ax.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(6, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(12, color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax.grid(True, alpha=0.3, axis='both')
ax.legend(fontsize=9, loc='best')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('figures_FINALIZED/wind_global_wavelet_spectrum_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.show()

#%%