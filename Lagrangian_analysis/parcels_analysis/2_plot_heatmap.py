'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- plots heatmap of particle crossings and duration of travelling

Author: vesnaber
Kernel: parcels-dev-local
'''

#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from matplotlib.colors import ListedColormap, Normalize
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# =============================================================================
# CONFIGURATION
# =============================================================================
parcels_config = 'GRENAVENE_coastrep050ED'
nc_file = f'crossings_output/{parcels_config}_ALLY_all_crossings_numpy.nc'

# Original zone labels (as they appear in the NetCDF file)
zone_labels_original = ['11.25-12.25°N', '12.25-13.25°N', '13.25-14.25°N', '14.25-15.25°N', 
                        '15.25-16.25°N', '16.25-17.25°N', '17.25-18.4°N']

# Full labels for the map 
zone_labels_map = ['Segment 1:\n11.25-12.25°N', 'Segment 2:\n12.25-13.25°N', 
                   'Segment 3:\n13.25-14.25°N', 'Segment 4:\n14.25-15.25°N',
                   'Segment 5:\n15.25-16.25°N', 'Segment 6:\n16.25-17.25°N', 
                   'Segment 7:\n17.25-18.4°N']

# Short labels for heatmaps 
zone_labels_short = ['Segment 1', 'Segment 2', 'Segment 3', 'Segment 4',
                     'Segment 5', 'Segment 6', 'Segment 7']
zone_labels_shortest = ['1', '2', '3', '4', '5', '6', '7']

# Segment boundaries for the map
segment_boundaries = [11.25, 12.25, 13.25, 14.25, 15.25, 16.25, 17.25, 18.4]
segment_lon_start = -69.75
segment_lon_end = -68.25

# velocity conversion
def deg_per_s_to_m_per_s(u_deg_s, v_deg_s, lat):
    """Convert velocities from degrees/second to m/s"""
    R_earth = 6371000.0
    lat_rad = np.radians(lat)
    m_per_deg_lon = R_earth * np.cos(lat_rad) * np.pi / 180.0
    m_per_deg_lat = R_earth * np.pi / 180.0
    u_m_s = u_deg_s * m_per_deg_lon
    v_m_s = v_deg_s * m_per_deg_lat
    return u_m_s, v_m_s

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
print(f"Loading NetCDF: {nc_file}")
ds = xr.open_dataset(nc_file)
print(f"Total crossings: {len(ds.crossing)}")

# Convert to pandas for easier manipulation
df = ds.to_dataframe().reset_index()

# Filter valid zones using ORIGINAL labels
df = df[df['zone_label'].isin(zone_labels_original)]
print(f"Valid crossings in zones: {len(df)}")

# Convert times
df['start_time'] = pd.to_datetime(df['start_time'])
df['crossing_time'] = pd.to_datetime(df['crossing_time'])
df['year'] = df['crossing_time'].dt.year
df['month'] = df['crossing_time'].dt.month
df['doy'] = df['crossing_time'].dt.dayofyear
df['release_month'] = df['start_time'].dt.month

# Ensure days_to_crossing is float
if df['days_to_crossing'].dtype.kind == 'm':  # timedelta
    df['days_to_crossing'] = df['days_to_crossing'].dt.total_seconds() / (24 * 3600)
else:
    df['days_to_crossing'] = df['days_to_crossing'].astype(float)

# =============================================================================
# CALCULATE SPEED WEIGHTS
# =============================================================================
print("Converting initial velocities from degrees/s to speed...")

# Convert initial velocities from deg/s to m/s
u_m_s, v_m_s = deg_per_s_to_m_per_s(df['initial_u'], df['initial_v'], df['crossing_lat'])

particle_spacing_m = 100  # meters (from 10 particles/km over 170.73 km transect)

# Calculate speed magnitude in m/s
speed_m_s = np.sqrt(u_m_s**2 + v_m_s**2)
df['speed_weight'] = speed_m_s * particle_spacing_m

print(f"Speed statistics:")
print(f"  Mean speed: {speed_m_s.mean():.4f} m/s")
print(f"  Max speed: {speed_m_s.max():.4f} m/s")
print(f"  Min speed: {speed_m_s.min():.4f} m/s")
print(f"  Std speed: {speed_m_s.std():.4f} m/s")

# =============================================================================
# CREATE DAILY WEIGHTED HEATMAP DATA
# =============================================================================
print("\nCreating daily speed-weighted data...")

# Create daily weighted counts using ORIGINAL labels
daily_weights = df.groupby([df['crossing_time'].dt.date, 'zone_label'])['speed_weight'].sum().unstack(fill_value=0.0)
daily_weights = daily_weights.reindex(columns=zone_labels_original, fill_value=0.0)
daily_weights.index = pd.to_datetime(daily_weights.index)
daily_weights['year'] = daily_weights.index.year
daily_weights['doy'] = daily_weights.index.dayofyear

years = sorted(daily_weights['year'].unique())
print(f"Years in data: {years[0]} to {years[-1]}")

# =============================================================================
# PREPARE DURATION DATA
# =============================================================================
print("\nPreparing duration distribution data...")

# Duration bins for heatmap
duration_bins = np.arange(0, 91, 5)  # 0-90 days in 2-day bins
months = np.arange(1, 13)

# Reverse zones for display (top to bottom)
reversed_zones_original = zone_labels_original[::-1]
reversed_zones_short = zone_labels_shortest[::-1]
reversed_zones_map = zone_labels_map[::-1]

#%%
# =============================================================================
# CREATE FIGURE WITH MAP, HEATMAPS, AND DURATION DISTRIBUTIONS
# =============================================================================
print("\nCreating figure with map, heatmaps, and duration distributions...")

fig = plt.figure(figsize=(16, 9))
# Three columns: Map | Daily Heatmaps | Duration Distributions
gs = GridSpec(len(zone_labels_original), 3, figure=fig, 
              width_ratios=[0.7, 3, 0.9], 
              wspace=0.15, hspace=0.12)

# =============================================================================
# LEFT PANEL: MAP SUBPLOT
# =============================================================================
ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
ax_map.set_extent([segment_lon_start, segment_lon_end, 11.15, 18.5], crs=ccrs.PlateCarree())

# Add map features
ax_map.add_feature(cfeature.LAND, facecolor='saddlebrown', alpha=0.4, edgecolor='saddlebrown')
ax_map.add_feature(cfeature.OCEAN, facecolor='#f7f7f7')
ax_map.coastlines(resolution='10m', linewidth=0.5)

# Add gridlines
gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10, 'rotation': 45}
gl.ylabel_style = {'size': 10}
gl.xlocator = mticker.FixedLocator([-69.5, -69, -68.5])
# Draw segment boundaries
for i, lat_boundary in enumerate(segment_boundaries):
    ax_map.plot([segment_lon_start+0.4, segment_lon_end-0.4], [lat_boundary, lat_boundary], 
               color='blue', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=10)

# Add meridional line at -69°W
ax_map.plot([-69, -69], [11.25, 18.4], color='blue', linestyle='--', linewidth=1.5, 
           transform=ccrs.PlateCarree(), zorder=10)

# Add zone labels (FULL labels on map)
for i in range(len(zone_labels_original)):
    lat_mid = (segment_boundaries[i] + segment_boundaries[i+1]) / 2
    ax_map.text(-69, lat_mid, zone_labels_map[i], 
               fontsize=8, va='center', ha='center', color='white', fontweight='bold',
               transform=ccrs.PlateCarree(),
               bbox=dict(boxstyle='square', facecolor='royalblue', alpha=0.7), zorder=11)

ax_map.set_title('(A) Segment\n       boundaries', fontsize=12, pad=10)

# =============================================================================
# MIDDLE PANELS: DAILY HEATMAP SUBPLOTS
# =============================================================================
axes_heatmap = []
for i in range(len(zone_labels_original)):
    ax = fig.add_subplot(gs[i, 1])
    axes_heatmap.append(ax)

# Create custom colormap with grey for zeros (for daily heatmaps)
base_cmap_daily = cmo.rain
colors = base_cmap_daily(np.linspace(0, 1, 256))
grey = np.array([1.0, 1.0, 1.0, 1.0]) #np.array([0.2, 0.2, 0.2, 1.0])
new_colors = np.vstack([grey, colors])
custom_cmap_daily = ListedColormap(new_colors)

# Set maximum value for color scale
vmax_daily = 45000

# Create heatmaps for each segment
for ax, zone_orig, zone_short in zip(axes_heatmap, reversed_zones_original, reversed_zones_short):
    # Create heatmap matrix
    heatmap_data = np.zeros((len(years), 366))
    
    for i, year in enumerate(years):
        year_data = daily_weights[daily_weights['year'] == year]
        for _, row in year_data.iterrows():
            doy = int(row['doy']) - 1
            if doy < 366:
                heatmap_data[i, doy] = row[zone_orig]
    
    # Prepare display data (grey for zeros)
    display_data = heatmap_data.copy()
    display_data[heatmap_data == 0] = -1
    display_data[heatmap_data > 0] = np.clip(heatmap_data[heatmap_data > 0], 0, vmax_daily)
    
    # Plot heatmap
    im = ax.imshow(display_data, aspect='auto', cmap=custom_cmap_daily,
                  interpolation='nearest', origin='lower', vmin=-1, vmax=vmax_daily)
    
    # Y-axis (years)
    ax.set_ylabel('Year', fontsize=10)
    yticks_years = [y for y in [1995, 2000, 2005, 2010, 2015, 2020] if y in years]
    yticks_idx = [years.index(y) for y in yticks_years]
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels(yticks_years, fontsize=10)
    
    # Add segment label
    ax.text(0.02, 0.92, zone_short, transform=ax.transAxes, fontsize=10, color='white',
            fontweight='bold', va='top',
            bbox=dict(boxstyle='square', facecolor='royalblue', alpha=0.7))

# Configure x-axis for daily heatmaps
for ax in axes_heatmap[:-1]:
    ax.set_xticklabels([])

axes_heatmap[-1].set_xlabel('Day of year', fontsize=10)
axes_heatmap[-1].set_xlim(0, 365)
month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
axes_heatmap[-1].set_xticks(month_starts)
axes_heatmap[-1].set_xticklabels(month_names, fontsize=10)

# =============================================================================
# RIGHT PANELS: DURATION DISTRIBUTION HEATMAPS (TRANSPOSED)
# =============================================================================
axes_duration = []
for i in range(len(zone_labels_original)):
    ax = fig.add_subplot(gs[i, 2])
    axes_duration.append(ax)

# Custom colormap with white for zeros (for duration heatmaps)
base_cmap_duration = cmo.rain
colors_dur = base_cmap_duration(np.linspace(0, 1, 256))
white = np.array([1.0, 1.0, 1.0, 1.0])
new_colors_dur = np.vstack([white, colors_dur])
custom_cmap_duration = ListedColormap(new_colors_dur)

# Set consistent vmax for duration
vmax_duration = 2000000

# Duration bins limited to 0-60 days
duration_bins_60 = np.arange(0, 61, 2)  # 0-60 days in 2-day bins

for ax, zone_orig, zone_short in zip(axes_duration, reversed_zones_original, reversed_zones_short):
    # Filter data for this zone
    zone_data = df[df['zone_label'] == zone_orig]
    
    if len(zone_data) == 0:
        ax.text(0.5, 0.5, 'No crossings', ha='center', va='center', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_yticks([])
        continue
    
    # Create 2D histogram data for this zone (TRANSPOSED: days x months)
    hist_data = np.zeros((len(duration_bins_60)-1, len(months)))
    
    for i, month in enumerate(months):
        month_data = zone_data[zone_data['release_month'] == month]
        if len(month_data) > 0:
            # Weighted histogram with 60-day limit
            hist, _ = np.histogram(month_data['days_to_crossing'], 
                                 bins=duration_bins_60, 
                                 weights=month_data['speed_weight'])
            hist_data[:, i] = hist
    
    # Prepare data for plotting (white for zeros)
    display_data = hist_data.copy()
    display_data[hist_data == 0] = -1  # Will map to white
    display_data[hist_data > 0] = np.clip(hist_data[hist_data > 0], 0, vmax_duration)
    
    # Plot heatmap (transposed: days on y-axis, months on x-axis)
    im = ax.imshow(display_data, aspect='auto', cmap=custom_cmap_duration, 
                  origin='lower', interpolation='nearest', vmin=-1, vmax=vmax_duration,
                  extent=[0.5, 12.5, 0, 60])
    
    # Set y-axis (days to crossing)
    ax.set_ylabel('Days', fontsize=10)
    ax.set_ylim(0, 60)
    ax.set_yticks(np.arange(0, 65, 10))
    
    # Set x-axis (months)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(months)

    ax.text(0.05, 0.92, zone_short, transform=ax.transAxes, fontsize=10, color='white',
        fontweight='bold', va='top',
        bbox=dict(boxstyle='square', facecolor='royalblue', alpha=0.7))

# Configure x-axis for duration heatmaps
for ax in axes_duration[:-1]:
    ax.set_xticklabels([])

axes_duration[-1].set_xlabel('Release month', fontsize=10)
axes_duration[-1].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], 
                                  fontsize=9)

# =============================================================================
# ADD TITLES
# =============================================================================
# Title for daily heatmaps (middle column)
fig.text(0.485, 0.908, '(B) Daily speed-weighted particle crossings (1993-2024)', 
         fontsize=12, ha='center')

# Title for duration distributions (right column)
fig.text(0.82, 0.9, '(C) Crossing timescales\n   by release month', 
         fontsize=12, ha='center')

# =============================================================================
# ADD COLORBARS AT BOTTOM
# =============================================================================
# Get positions of the bottom plots
pos_heatmap = axes_heatmap[-1].get_position()
pos_duration = axes_duration[-1].get_position()

# Colorbar for daily heatmaps (below middle column)
norm_daily = Normalize(vmin=0, vmax=vmax_daily)
cbar_ax_daily = fig.add_axes([pos_heatmap.x0, 0.02, pos_heatmap.width, 0.015])
cbar_daily = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm_daily, cmap=base_cmap_daily),
    cax=cbar_ax_daily,
    orientation='horizontal',
    extend='max'
)
cbar_daily.set_label('Daily transport per unit depth [m²/s]', fontsize=10)
cbar_daily.ax.tick_params(labelsize=9)

# Colorbar for duration heatmaps (below right column)
norm_duration = Normalize(vmin=0, vmax=vmax_duration)
cbar_ax_duration = fig.add_axes([pos_duration.x0, 0.02, pos_duration.width, 0.015])
cbar_duration = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm_duration, cmap=base_cmap_duration),
    cax=cbar_ax_duration,
    orientation='horizontal',
    extend='max'
)

cbar_duration.ax.tick_params(labelsize=9)
tick_values = np.linspace(0, vmax_duration, 5)
cbar_duration.set_ticks(tick_values)
cbar_duration.set_ticklabels([f'{int(v/1e6)}' for v in tick_values])
cbar_duration.set_label('Monthly transport \nper unit depth [×10⁶ m²/s]', fontsize=10)

# =============================================================================
# SAVE FIGURE
# =============================================================================
plt.tight_layout(rect=[0, 0.06, 1, 0.95]) 
output_path = f"figures/Fig05_{parcels_config}_heatmap_with_duration_m2s.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to {output_path}")

# %%
