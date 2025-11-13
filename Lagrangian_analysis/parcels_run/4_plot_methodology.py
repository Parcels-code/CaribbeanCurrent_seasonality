"""
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script plots:
(a) Eulerian cross-section location
(b) Lagrangian seeding location, example trajectories, and cross-sections
This is Figure 1 in manuscript.

Author: vesnaber
Kernel: parcels-dev-local
"""

#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import FancyArrowPatch
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# =============================================================================
# CONFIGURATION
# =============================================================================
parcels_config = 'GRENAVENE_coastrep050ED'
year = 2004

# Release dates to plot 
release1_month = 1   # January
release1_day = 10
release2_month = 9   # September
release2_day = 12

# Segment configuration
zone_labels_map = ['Segment 1:\n11.25-12.25°N', 'Segment 2:\n12.25-13.25°N', 
                   'Segment 3:\n13.25-14.25°N', 'Segment 4:\n14.25-15.25°N',
                   'Segment 5:\n15.25-16.25°N', 'Segment 6:\n16.25-17.25°N', 
                   'Segment 7:\n17.25-18.4°N']

segment_boundaries = [11.25, 12.25, 13.25, 14.25, 15.25, 16.25, 17.25, 18.4]
segment_lon_start = -70
segment_lon_end = -68

# Eulerian cross-section configuration
eulerian_lon = -69.0
eulerian_lat_start = 11.4
eulerian_lat_end = 12.2

#%%
# =============================================================================
# LOAD SEEDING LOCATIONS
# =============================================================================
try:
    lon_seed = np.load('parcels_input/particles_GRENAVENE_lon.npy')
    lat_seed = np.load('parcels_input/particles_GRENAVENE_lat.npy')
    print(f"Loaded {len(lon_seed)} seeding locations")
except Exception as e:
    print(f"Could not load seeding locations: {e}")
    lon_seed, lat_seed = None, None

#%%
# =============================================================================
# LOAD TRAJECTORY DATA
# =============================================================================
try:
    filename = f'GRENAVENE_output/{parcels_config}_{year}.zarr'
    ds = xr.open_zarr(filename)
    print(f"Loaded zarr file with {len(ds.trajectory)} trajectories")
    
    # Calculate number of releases
    if lon_seed is not None:
        n_releases = len(ds.trajectory) // len(lon_seed)
        print(f"Number of releases: {n_releases}")
        print(f"Particles per release: {len(lon_seed)}")
    
except Exception as e:
    print(f"Error loading trajectory data: {e}")
    ds = None

#%%
# =============================================================================
# EXTRACT TRAJECTORIES FOR SPECIFIC RELEASE DATES
# =============================================================================
max_days = 200
max_obs = max_days * 2  # Output every 12 hours

if ds is not None and lon_seed is not None:
    n_particles = len(lon_seed)
    
    # DAILY RELEASES - Calculate day of year for each date
    from datetime import datetime
    
    # Release 1
    release1_date = datetime(year, release1_month, release1_day)
    release1_day_of_year = release1_date.timetuple().tm_yday
    release1_idx = release1_day_of_year - 1  # 0-indexed
    release1_start = release1_idx * n_particles
    release1_end = release1_start + n_particles
    release1_indices = slice(release1_start, release1_end)
    
    lons_release1 = ds.lon.isel(trajectory=release1_indices, obs=slice(0, max_obs)).values
    lats_release1 = ds.lat.isel(trajectory=release1_indices, obs=slice(0, max_obs)).values
    
    print(f"{release1_date.strftime('%B %d')} trajectories: indices {release1_start} to {release1_end-1}")
    print(f"Release 1 trajectories shape: {lons_release1.shape}")
    
    # Release 2
    release2_date = datetime(year, release2_month, release2_day)
    release2_day_of_year = release2_date.timetuple().tm_yday
    release2_idx = release2_day_of_year - 1  # 0-indexed
    release2_start = release2_idx * n_particles
    release2_end = release2_start + n_particles
    release2_indices = slice(release2_start, release2_end)
    
    lons_release2 = ds.lon.isel(trajectory=release2_indices, obs=slice(0, max_obs)).values
    lats_release2 = ds.lat.isel(trajectory=release2_indices, obs=slice(0, max_obs)).values
    
    print(f"{release2_date.strftime('%B %d')} trajectories: indices {release2_start} to {release2_end-1}")
    print(f"Release 2 trajectories shape: {lons_release2.shape}")
else:
    lons_release1, lats_release1 = None, None
    lons_release2, lats_release2 = None, None



#%%
# =============================================================================
# CREATE FIGURE WITH TWO PANELS (HEIGHT RATIO 1:3)
# =============================================================================
fig = plt.figure(figsize=(14, 13))

# Create GridSpec for custom height ratios
gs = GridSpec(2, 2, height_ratios=[1, 2.6], width_ratios=[3.7, 1], hspace=0.25)

# =============================================================================
# PANEL A: EULERIAN CROSS-SECTION LOCATION 
# =============================================================================
ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())

# Set wider extent with more longitudes
ax_a.set_extent([-70, -61.3, 10, 13.5], crs=ccrs.PlateCarree())

# Add map features
ax_a.add_feature(cfeature.LAND, facecolor='saddlebrown', alpha=0.4, edgecolor='saddlebrown')
ax_a.add_feature(cfeature.OCEAN, facecolor='#f7f7f7')
ax_a.coastlines(resolution='10m', linewidth=0.8, color='saddlebrown')

# Add gridlines
gl_a = ax_a.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl_a.top_labels = False
gl_a.right_labels = False
gl_a.xlabel_style = {'size': 10}
gl_a.ylabel_style = {'size': 10}

# Plot the Eulerian cross-section
ax_a.plot([eulerian_lon, eulerian_lon], [eulerian_lat_start+0.2, eulerian_lat_end-0.1],
         color='blue', linewidth=3, transform=ccrs.PlateCarree(), 
         label=f'Cross-section for \nEulerian analysis ({-eulerian_lon}°W)', zorder=10)


# Add location labels
ax_a.text(-69.6, 11, 'Venezuela', fontsize=11, fontweight='bold', transform=ccrs.PlateCarree(), zorder=12,
       bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.8))
ax_a.text(-69.5, 12.7, 'Curaçao', fontsize=11, fontweight='bold', transform=ccrs.PlateCarree(), zorder=12,
       bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.8))

# Plot Caribbean Current inflow cross-section
if lon_seed is not None and lat_seed is not None:
    min_lon, max_lon = np.min(lon_seed), np.max(lon_seed)
    min_lat, max_lat = np.min(lat_seed), np.max(lat_seed)
    ax_a.plot([min_lon, max_lon], [min_lat, max_lat], color='k', linewidth=3, 
              label='Cross-section for Caribbean\nCurrent inflow analysis', 
              transform=ccrs.PlateCarree(), zorder=10)

# Plot wind statistics locations
grenada_wind_lon, grenada_wind_lat = -61.94, 11.69
curacao_wind_lon, curacao_wind_lat = -68.94, 11.94

# Plot wind statistics locations with wind flow lines (PowerPoint-style)
def plot_wind_icon(ax, lon, lat, color, edge_color):
    """Plot a wind icon with three curved lines"""
    # Background circle
    ax.scatter(lon, lat, c=color, s=200, marker='o',
              edgecolor=edge_color, linewidth=2, transform=ccrs.PlateCarree(), zorder=29)
    
    # Three horizontal curved lines (wind streaks)
    line_offsets = [-0.04, 0, 0.04]  # Vertical offsets for three lines
    for offset in line_offsets:
        # Create curved line effect with multiple short segments
        x_line = [lon - 0.08, lon - 0.03, lon + 0.02, lon + 0.08]
        y_line = [lat + offset - 0.01, lat + offset, lat + offset, lat + offset + 0.01]
        ax.plot(x_line, y_line, color='white', linewidth=1,
               transform=ccrs.PlateCarree(), zorder=30, solid_capstyle='round')

# Grenada wind station
plot_wind_icon(ax_a, grenada_wind_lon, grenada_wind_lat, 'olivedrab', 'darkgreen')
# Curaçao wind station
plot_wind_icon(ax_a, curacao_wind_lon, curacao_wind_lat, 'firebrick', 'darkred')

# Create custom legend handles for wind stations

grenada_legend = Circle((0, 0), 0.03, facecolor='olivedrab', edgecolor='darkgreen', linewidth=2)
curacao_legend = Circle((0, 0), 0.03, facecolor='firebrick', edgecolor='darkred', linewidth=2)



# Create custom legend handles as circles
grenada_legend = Line2D([0], [0], marker='o', color='w', markerfacecolor='olivedrab', 
                        markeredgecolor='darkgreen', markersize=10, linewidth=0,
                        markeredgewidth=2)
curacao_legend = Line2D([0], [0], marker='o', color='w', markerfacecolor='firebrick',
                        markeredgecolor='darkred', markersize=10, linewidth=0,
                        markeredgewidth=2)

# Get existing legend handles and labels
handles, labels = ax_a.get_legend_handles_labels()

# Add wind station legend entries
handles.extend([grenada_legend, curacao_legend])
labels.extend(['Wind statistics: Grenada', 'Wind statistics: Curaçao'])
# Add legend outside plot on the right side
ax_a.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1.0, 0.72), 
           fontsize=11, framealpha=0.1, edgecolor='black', fancybox=True)

ax_a.set_title('(a) Static analysis \n(Eulerian analysis, Caribbean Current inflow analysis, wind analysis)', 
              fontsize=13, pad=10, x = 0.75)


# =============================================================================
# PANEL B: LAGRANGIAN SETUP 
# =============================================================================

ax_b = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree())

# Set extent to show full region
ax_b.set_extent([-72.2, -58.5, 10, 19.5], crs=ccrs.PlateCarree())

# Add map features
ax_b.add_feature(cfeature.LAND, facecolor='saddlebrown', alpha=0.4, edgecolor='saddlebrown')
ax_b.add_feature(cfeature.OCEAN, facecolor='#f7f7f7')
ax_b.coastlines(resolution='10m', linewidth=0.8, color='saddlebrown')

# Add gridlines
gl_b = ax_b.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl_b.top_labels = False
gl_b.right_labels = False
gl_b.xlabel_style = {'size': 10}
gl_b.ylabel_style = {'size': 10}

# Plot seeding location
if lon_seed is not None:
    ax_b.scatter(lon_seed, lat_seed, c='k', s=10, marker='o', 
              edgecolor=None, linewidth=1, alpha=1,
              label='Seeding location', transform=ccrs.PlateCarree(), zorder=10)

# Plot segment boundaries
for i, lat_boundary in enumerate(segment_boundaries):
    ax_b.plot([segment_lon_start+0.5, segment_lon_end-0.5], [lat_boundary, lat_boundary], 
           color='blue', linewidth=2, transform=ccrs.PlateCarree(), zorder=8)

# Add vertical line at -69°
ax_b.plot([-69, -69], [11.25, 18.4], color='blue', linestyle='--', linewidth=2, 
       transform=ccrs.PlateCarree(), zorder=48, label='Cross-section boundary')

# Add zone labels
for i in range(len(zone_labels_map)):
    lat_mid = (segment_boundaries[i] + segment_boundaries[i+1]) / 2
    ax_b.text(-70, lat_mid, zone_labels_map[i], 
           fontsize=9, va='center', ha='center', color='white', fontweight='bold',
           transform=ccrs.PlateCarree(),
           bbox=dict(boxstyle='square', facecolor='royalblue', alpha=0.8), zorder=11)

# Plot Release 1 trajectories (all thin lines with low alpha)
if lons_release1 is not None:
    print(f"Plotting {lons_release1.shape[0]} Release 1 trajectories...")
    
    for idx in range(lons_release1.shape[0]):
        lons = lons_release1[idx, :]
        lats = lats_release1[idx, :]
        
        valid = ~(np.isnan(lons) | np.isnan(lats))
        lons_valid = lons[valid]
        lats_valid = lats[valid]
        
        if len(lons_valid) > 1:
            ax_b.plot(lons_valid, lats_valid, 'coral', linewidth=0.5, 
                   alpha=0.08, transform=ccrs.PlateCarree(), zorder=5)

# Plot Release 2 trajectories (all thin lines with low alpha)
if lons_release2 is not None:
    print(f"Plotting {lons_release2.shape[0]} Release 2 trajectories...")
    
    for idx in range(lons_release2.shape[0]):
        lons = lons_release2[idx, :]
        lats = lats_release2[idx, :]
        
        valid = ~(np.isnan(lons) | np.isnan(lats))
        lons_valid = lons[valid]
        lats_valid = lats[valid]
        
        if len(lons_valid) > 1:
            ax_b.plot(lons_valid, lats_valid, 'steelblue', linewidth=0.5, 
                   alpha=0.08, transform=ccrs.PlateCarree(), zorder=5)

# Create manual legend entries with higher alpha for visibility
legend_line1 = Line2D([0], [0], color='coral', linewidth=2, alpha=0.9,
                      label=f'Trajectories of all particles released on {release1_date.strftime("%d %B %Y")}')
legend_line2 = Line2D([0], [0], color='steelblue', linewidth=2, alpha=0.9,
                      label=f'Trajectories of all particles released on {release2_date.strftime("%d %B %Y")}')

# Add location labels for panel B
grenada_lon, grenada_lat = -61.68, 12.12
ax_b.text(grenada_lon+0.25, grenada_lat+0.1, 'Grenada', fontsize=11, 
       fontweight='bold', transform=ccrs.PlateCarree(), zorder=12,
       bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.8))

venezuela_lon, venezuela_lat = -64, 10.5
ax_b.text(venezuela_lon+1.6, venezuela_lat-0.1, 'Venezuela', fontsize=11, 
       fontweight='bold', transform=ccrs.PlateCarree(), zorder=12,
       bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.8))

# Combine all legend elements
handles, labels = ax_b.get_legend_handles_labels()
handles.extend([legend_line1, legend_line2])
ax_b.legend(handles=handles, loc='upper right', fontsize=11, framealpha=0.9)

ax_b.set_title('(b) Lagrangian analysis\n (Particle seeding locations, trajectories and segments)', 
              fontsize=13, pad=10)


# Save figure
output_path = f"figures/Fig01_methodology_{parcels_config}.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Combined methodology figure saved to {output_path}")


# %%