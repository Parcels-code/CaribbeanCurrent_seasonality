'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- calculates and plots the monthly climatology of particle arrivals per segment
- performs wavelet analysis on daily time series of Segment 1
- creates a combined figure with climatology and wavelet spectrum (Figure 6 in the manuscript)
- saves integrated spectrum for Figure 9 of the manuscript

Author: vesnaber
Kernel: parcels-dev-local
'''
#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import pandas as pd
import pycwt as wavelet
from matplotlib.gridspec import GridSpec
import pickle
import os

# Fix numpy compatibility
if not hasattr(np, 'int'):
    np.int = int

# =============================================================================
# CONFIGURATION
# =============================================================================
parcels_config = 'GRENAVENE_coastrep050ED'
nc_file = f'crossings_output/{parcels_config}_ALLY_all_crossings_numpy.nc'

# Original zone labels
zone_labels_original = ['11.25-12.25°N', '12.25-13.25°N', '13.25-14.25°N', '14.25-15.25°N', 
                        '15.25-16.25°N', '16.25-17.25°N', '17.25-18.4°N']

zone_labels_short = ['Segment 1', 'Segment 2', 'Segment 3', 'Segment 4',
                     'Segment 5', 'Segment 6', 'Segment 7']

# Helper function for velocity conversion
def deg_per_s_to_m_per_s(u_deg_s, v_deg_s, lat):
    """Convert velocities from degrees/second to m/s"""
    R_earth = 6371000.0
    lat_rad = np.radians(lat)
    m_per_deg_lon = R_earth * np.cos(lat_rad) * np.pi / 180.0
    m_per_deg_lat = R_earth * np.pi / 180.0
    u_m_s = u_deg_s * m_per_deg_lon
    v_m_s = v_deg_s * v_deg_s
    return u_m_s, v_m_s

# =============================================================================
# LOAD DATA
# =============================================================================
print(f"Loading NetCDF: {nc_file}")
ds = xr.open_dataset(nc_file)

# Convert to pandas
df = ds.to_dataframe().reset_index()
df = df[df['zone_label'].isin(zone_labels_original)]

# Convert crossing_time to datetime
df['crossing_time'] = pd.to_datetime(df['crossing_time'])

# Convert velocities to m/s and calculate speed weights
print("Converting initial velocities from degrees/s to speed...")
u_m_s, v_m_s = deg_per_s_to_m_per_s(df['initial_u'], df['initial_v'], df['crossing_lat'])
speed_m_s = np.sqrt(u_m_s**2 + v_m_s**2)
particle_spacing_m = 100  # meters (from 10 particles/km over 170.73 km transect)

# Calculate speed magnitude in m/s
speed_m_s = np.sqrt(u_m_s**2 + v_m_s**2)
df['speed_weight'] = speed_m_s * particle_spacing_m

# =============================================================================
# MONTHLY CLIMATOLOGY FOR ALL SEGMENTS
# =============================================================================
print("Calculating monthly climatology...")
df_monthly = df.groupby([df['crossing_time'].dt.to_period('M'), 'zone_label'])['speed_weight'].sum().reset_index()
df_monthly['crossing_time'] = df_monthly['crossing_time'].dt.to_timestamp()
df_monthly['month'] = df_monthly['crossing_time'].dt.month

# Climatology (average by month across all years)
climatology = df_monthly.groupby(['month', 'zone_label'])['speed_weight'].mean().unstack(fill_value=0.0)
climatology = climatology.reindex(columns=zone_labels_original, fill_value=0.0)

# Standard deviation
climatology_std = df_monthly.groupby(['month', 'zone_label'])['speed_weight'].std().unstack(fill_value=0.0)
climatology_std = climatology_std.reindex(columns=zone_labels_original, fill_value=0.0)

# =============================================================================
# PREPARE SEGMENT 1 DAILY TIME SERIES FOR WAVELET
# =============================================================================
print("\nPreparing Segment 1 daily time series...")
daily_weights = df.groupby([df['crossing_time'].dt.date, 'zone_label'])['speed_weight'].sum().unstack(fill_value=0.0)
daily_weights = daily_weights.reindex(columns=zone_labels_original, fill_value=0.0)
daily_weights.index = pd.to_datetime(daily_weights.index)

# Extract Segment 1 time series
segment1_label = zone_labels_original[0]
segment1_daily = daily_weights[[segment1_label]].copy()
segment1_daily = segment1_daily.sort_index()

# Create complete date range
date_range = pd.date_range(start=segment1_daily.index.min(), 
                           end=segment1_daily.index.max(), 
                           freq='D')
segment1_daily = segment1_daily.reindex(date_range, fill_value=0.0)

# =============================================================================
# WAVELET ANALYSIS ON DAILY DATA
# =============================================================================
print("\nPerforming wavelet analysis on daily data...")

time_series = segment1_daily[segment1_label].values
time_np = segment1_daily.index

# Remove any NaNs
valid_mask = ~np.isnan(time_series)
data_clean = time_series[valid_mask]
time_clean = time_np[valid_mask]

# Normalize
data_norm = (data_clean - np.mean(data_clean)) / np.std(data_clean)

# Wavelet parameters for DAILY data
dt = 1
mother = wavelet.Morlet(6)
s0 = 2 * dt
dj = 1/12
J = 11 / dj
alpha, _, _ = wavelet.ar1(data_norm)

# Compute wavelet transform
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data_norm, dt, dj, s0, J, mother)
power = (np.abs(wave)) ** 2
period = 1 / freqs

# Significance testing
signif, _ = wavelet.significance(data_norm, dt, scales, 0, alpha,
                                significance_level=0.95, wavelet=mother)
sig95 = signif[:, np.newaxis] * np.ones(wave.shape)

# Global wavelet spectrum (DATA TO BE SAVED)
global_ws = power.mean(axis=1)
dof = len(data_norm) - scales
global_signif, _ = wavelet.significance(data_norm, dt, scales, 1, alpha,
                                       significance_level=0.95, dof=dof, wavelet=mother)

# Convert period to months
period_months = period / 30.44
coi_months = coi / 30.44

# Find max power period
max_power_idx = np.nanargmax(global_ws)
max_period_months = period_months[max_power_idx]

#%%
# =============================================================================
# CREATE FIGURE
# =============================================================================
print("\nCreating integrated figure...")

# Use 2 rows and 1 column, forcing the bottom plot to be full width
fig = plt.figure(figsize=(12, 10))
# GridSpec with 2 rows, 1 implied column (or 3 columns and span all 3 for width)
gs = GridSpec(2, 3, figure=fig, 
              width_ratios=[1, 1, 1], # Used to define 3 equal columns
              height_ratios=[1, 1],
              hspace=0.35, wspace=0.1)

# =============================================================================
# TOP: MONTHLY CLIMATOLOGY (SPANS ALL 3 COLUMNS)
# =============================================================================
ax_clim = fig.add_subplot(gs[0, :])

colors_palette = ['darkred'] + list(cmo.gray(np.linspace(0.01, 0.99, 7)))
colors_palette = colors_palette[:-1]
linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
linewidths = [2.5, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
alphas = [1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
markers = ['o', 's', '^', 'D', 'v', 'p', '*']
markersizes = [6, 4, 4, 4, 4, 4, 4]

for i, (zone_orig, zone_short) in enumerate(zip(zone_labels_original, zone_labels_short)):
    months = climatology.index
    mean_vals = climatology[zone_orig]
    std_vals = climatology_std[zone_orig]
    zorder = 10 if i == 0 else 5
    ax_clim.plot(months, mean_vals, 
                 marker=markers[i], markersize=markersizes[i],
                 linewidth=linewidths[i], linestyle=linestyles[i],
                 label=zone_short, color=colors_palette[i], 
                 alpha=alphas[i], zorder=zorder)
    fill_alpha = 0.25 if i == 0 else 0.15
    ax_clim.fill_between(months, mean_vals - std_vals, mean_vals + std_vals,
                         alpha=fill_alpha, color=colors_palette[i], zorder=zorder-1)

# Custom y-axis formatting with scientific notation in label
ax_clim.set_ylabel('Transport per unit depth [×10⁶ m²/s]', fontsize=11)

ymin, ymax = ax_clim.get_ylim()
n_ticks = 6
tick_values = np.linspace(ymin, ymax, n_ticks)
ax_clim.set_yticks(tick_values)
ax_clim.set_yticklabels([f'{v/1e6:.1f}' for v in tick_values], fontsize=10)

ax_clim.set_title('(a) Monthly climatology of particle arrivals by segment (1993-2024)', 
                  fontsize=13, pad=15)
ax_clim.set_xticks(range(1, 13))
ax_clim.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 
                         'J', 'A', 'S', 'O', 'N', 'D'], fontsize=11)
ax_clim.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.95)
ax_clim.grid(True, alpha=0.3)
ax_clim.set_xlim(1, 12)
# =============================================================================
# BOTTOM: WAVELET POWER SPECTRUM (FULL WIDTH)
# =============================================================================

ax1 = fig.add_subplot(gs[1, :])
cmap_wavelet = cmo.amp_r
t_plot = time_clean

# Round the maximum power value to a nice number
max_power = np.percentile(power, 95)
if max_power < 10:
    max_power_rounded = np.ceil(max_power)
elif max_power < 50:
    max_power_rounded = np.floor(max_power / 5) * 5
else:
    max_power_rounded = np.ceil(max_power / 10) * 10

levels = np.linspace(0, max_power_rounded, 100)

contourf = ax1.contourf(t_plot, period_months, power,
                    levels=levels, cmap=cmap_wavelet, extend='max')
ax1.contour(t_plot, period_months, power / sig95, levels=[1],
        colors='blue', linewidths=1.5, alpha=0.8, linestyles=':')

# Cone of influence
ax1.fill_between(t_plot, coi_months, period_months.max(),
                color='white', alpha=0.5, hatch='///',
                edgecolor='white', linewidth=0.8,
                label='Cone of influence')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Period [months]', fontsize=12)
ax1.set_title('(b) Wavelet power spectrum of Segment 1 (daily transport)\n(95% significance in blue contour)',
            fontsize=13, pad=10)
ax1.set_yscale('log')

ax1.set_ylim([0.08, period_months.max()])

yticks = [0.1, 0.5, 1, 3, 6, 12, 24, 60]
ax1.set_yticks(yticks)
ax1.set_yticklabels(yticks)

# X-axis formatting
major_years = pd.date_range(start='1995-01-01', end='2024-01-01', freq='5YS')
ax1.set_xticks(major_years)
ax1.set_xticklabels([str(yr.year) for yr in major_years], fontsize=11)
minor_years = pd.date_range(start=time_clean[0], end=time_clean[-1], freq='YS')
ax1.set_xticks(minor_years, minor=True)
ax1.tick_params(axis='x', which='minor', length=4, width=1)
ax1.tick_params(axis='both', which='major', labelsize=11)

# Reference lines
ax1.axhline(1, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.axhline(6, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.axhline(12, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)

# Colorbar for Plot B 
cbar_ax = fig.add_axes([0.13, 0.022, 0.77, 0.02]) # Adjusted position and width to span the full figure
cbar1 = plt.colorbar(contourf, cax=cbar_ax, orientation='horizontal')
cbar1.set_label('Normalized power [ ]', fontsize=11)
cbar1.ax.tick_params(labelsize=10)

num_ticks = 5
cbar_ticks = np.linspace(0, max_power_rounded, num_ticks)
cbar1.set_ticks(cbar_ticks)

# Max power annotation
ax1.text(0.005, 0.98, f'Max: {max_period_months:.1f} months',
        transform=ax1.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='square,pad=0.3', facecolor='yellow', alpha=0.9))

ax1.invert_yaxis()

# Save figure
output_path = f"figures/Fig06_{parcels_config}_climatology_wavelet_m2s.png"
plt.tight_layout(rect=[0, 0.08, 1, 1]) 
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to {output_path}")
plt.show()


#%%
# =============================================================================
# DATA EXTRACTION FOR STANDALONE PLOTTING (Integrated Spectrum)
# =============================================================================

plot_d_data = {
    'period_months': period_months,        # Y-axis / Period scale
    'global_ws': global_ws,                # Global Spectrum (GWS) values
    'global_signif': global_signif         # 95% Significance values
}

output_dir = '../../integrated_spectra_combined/data_wavelet'
filename = os.path.join(output_dir, 'lagrangian_integrated_spectrum.pkl')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(filename, 'wb') as f:
    pickle.dump(plot_d_data, f)

print(f"\nData for Integrated Spectrum saved to: {filename}")

#%%