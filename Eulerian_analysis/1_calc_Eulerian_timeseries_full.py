'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- Calculats geostrophic flow from SSH
- defines EDDY/NW-flow regimes
- plot timeseries and critera distribution (Figure 2 in manuscript)

Author: vesnaber
kernel: parcels-dev-local
'''

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons
import matplotlib.dates as mdates
import cmocean
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap, Normalize


# =============================================================================
# LOAD DATA AND CALCULATE GEOSTROPHIC VELOCITIES
# =============================================================================

print("Loading SSH datasets...")
ds1 = xr.open_dataset("data/cmems_mod_glo_phy_my_0.083deg_P1D-m_1759843361350.nc")
ds2 = xr.open_dataset("data/cmems_mod_glo_phy_myint_0.083deg_P1D-m_1759843398751.nc")
ds = xr.concat([ds1, ds2], dim='time')

# Standardize time and reindex to full daily range
ds['time'] = pd.to_datetime(ds['time'].values)
full_time = pd.date_range("1993-01-01", "2024-12-31", freq='D')
ds = ds.reindex(time=full_time)

# Constants
g = 9.81  # m/s^2
Omega = 7.2921e-5  # rad/s
deg2rad = np.pi / 180
Re = 6.371e6  # Earth radius in m

# Extract coords and SSH
lat = ds['latitude']
lon = ds['longitude']
ssh = ds['zos']
#%%
# =============================================================================
# COMPUTE CORIOLIS PARAMETER AND GRID SPACING
# =============================================================================

# Coriolis parameter
f = 2 * Omega * np.sin(np.deg2rad(lat))
f = xr.DataArray(f, coords=[lat], dims=['latitude'])

# Grid spacing
dlat = np.gradient(lat.values) * deg2rad
dlon = np.gradient(lon.values) * deg2rad
dy = Re * dlat
dy_da = xr.DataArray(dy, coords=[lat], dims=['latitude'])

dx = Re * np.cos(np.deg2rad(lat.values)[:, None]) * dlon[None, :]
dx_da = xr.DataArray(dx, coords=[lat, lon], dims=['latitude', 'longitude'])

# =============================================================================
# CALCULATE SSH GRADIENTS (CENTERED DIFFERENCES)
# =============================================================================

print("Calculating SSH gradients using centered differences...")

# Meridional (latitude) gradient
ssh_y_numerator = ssh.shift(latitude=-1) - ssh.shift(latitude=1)
ssh_y = ssh_y_numerator / (2 * dy_da)
ssh_y = ssh_y.isel(latitude=slice(1, -1))

# Zonal (longitude) gradient
ssh_x_numerator = ssh.shift(longitude=-1) - ssh.shift(longitude=1)
ssh_x = ssh_x_numerator / (2 * dx_da)
ssh_x = ssh_x.isel(longitude=slice(1, -1))

# Adjust Coriolis to reduced latitude grid
f_interp = f.isel(latitude=slice(1, -1))

# =============================================================================
# GEOSTROPHIC VELOCITIES
# =============================================================================

u = -g / f_interp * ssh_y  # zonal (eastward)
v = g / f_interp * ssh_x   # meridional (northward)

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

# Spatial means across the latitude band per time step
speed = np.sqrt(uo_region**2 + vo_region**2)
uo_spatial_mean = uo_region.mean(dim='latitude')
vo_spatial_mean = vo_region.mean(dim='latitude')
speed_spatial_mean = speed.mean(dim='latitude')

# =============================================================================
# DEFINE EDDY REGIME CRITERIA
# =============================================================================

# Criterion 2: weak flow (below 25th percentile)
speed_threshold = float(speed_spatial_mean.quantile(0.25))
weak_flow = speed_spatial_mean < speed_threshold

# Criterion 1: not NW direction
direction_spatial_mean = np.arctan2(uo_spatial_mean, vo_spatial_mean) * 180 / np.pi
nw_min = -90
nw_max = -25
not_nw_direction = (direction_spatial_mean < nw_min) | (direction_spatial_mean > nw_max)

# EDDY regime = weak OR not NW direction
eddy_regime_complex = weak_flow | not_nw_direction

# =============================================================================
# RESAMPLE CRITERIA TO MONTHLY FRACTIONS (FOR STACKED BARS)
# =============================================================================

# Colors (final choices)
COLOR_C1_COMBINED = "#E5A607"  # Direction-driven / anomaly
COLOR_C2_ONLY = "#9A7347"      # Pure weak flow
COLOR_NORMAL = "#404040"       # Strong NW-flow

# Daily mutually exclusive categories
c1_combined_daily = not_nw_direction
c2_only_daily = (~not_nw_direction) & weak_flow
normal_flow_daily = (~not_nw_direction) & (~weak_flow)

# Monthly fractions (month start)
monthly_c1_combined_fraction = c1_combined_daily.resample(time='MS').mean()
monthly_c2_only_fraction = c2_only_daily.resample(time='MS').mean()
monthly_normal_flow_fraction = normal_flow_daily.resample(time='MS').mean()

# Plot shading parameters
MAX_ALPHA = 0.8
MAX_SHADING_HEIGHT_FRACTION = (11.6 - lat_range.start) / (lat_range.stop - lat_range.start)
#%%
# =============================================================================
# FIGURE SETUP
# =============================================================================

decades = [
    ('1993-2000', '1993-01-01', '1999-12-31', 0.3, 1.0),
    ('2000-2010', '2000-01-01', '2009-12-31', 0.0, 1.0),
    ('2010-2020', '2010-01-01', '2019-12-31', 0.0, 1.0),
    ('2020-2024', '2020-01-01', '2024-12-31', 0.0, 0.5)
]

fig = plt.figure(figsize=(13, 15))
fig.suptitle(f"(b) Meridional cross-section of geostrophic zonal velocity at {-selected_lon:.1f}°W\n"
             f"with flow regime stacked bar by monthly fraction based on criteria",
             fontsize=12, x=0.63, y=0.78)

vmax = 0.6
vmin = -0.6
cmap = cmocean.cm.balance

# =============================================================================
# PLOT PANELS (B)
# =============================================================================

for i, (title, start, end, start_frac, end_frac) in enumerate(decades):
    row_height = 0.13
    row_bottom = 0.62 - i * 0.16
    subplot_width = 0.8 * (end_frac - start_frac)
    subplot_left = 0.1 + 0.8 * start_frac
    ax = fig.add_axes([subplot_left, row_bottom, subplot_width, row_height])

    v = uo.sel(time=slice(start, end), latitude=lat_range)
    time_numeric = mdates.date2num(v.time.values)
    T, Y = np.meshgrid(time_numeric, v.latitude.values, indexing='ij')

    levels = np.linspace(vmin, vmax, 19)
    cf = ax.contourf(T, Y, v.values.T, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, alpha=1, extend='both')

    # Filter monthly data for decade
    start_dt = pd.to_datetime(start) + pd.offsets.MonthBegin(0)
    end_dt = pd.to_datetime(end) + pd.offsets.MonthBegin(0)

    c1_combined_data = monthly_c1_combined_fraction.sel(time=slice(start_dt, end_dt))
    c2_only_data = monthly_c2_only_fraction.sel(time=slice(start_dt, end_dt))
    normal_flow_data = monthly_normal_flow_fraction.sel(time=slice(start_dt, end_dt))

    first_label_c1 = (i == 0)
    first_label_c2 = (i == 0)
    first_label_normal = (i == 0)

    for month_start_date in c1_combined_data.time.values:
        frac_c1_combined = c1_combined_data.sel(time=month_start_date).item()
        frac_c2_only = c2_only_data.sel(time=month_start_date).item()
        frac_normal = normal_flow_data.sel(time=month_start_date).item()
        total_fraction = frac_c1_combined + frac_c2_only + frac_normal

        if total_fraction > 0.001:
            month_start_date_ts = pd.to_datetime(month_start_date)
            month_end_date = month_start_date_ts + pd.offsets.MonthEnd(0)
            clip_start = max(month_start_date_ts, pd.to_datetime(start))
            clip_end = min(month_end_date + pd.Timedelta(days=1), pd.to_datetime(end))

            h_c2 = frac_c2_only * MAX_SHADING_HEIGHT_FRACTION
            h_c1 = frac_c1_combined * MAX_SHADING_HEIGHT_FRACTION
            h_normal = frac_normal * MAX_SHADING_HEIGHT_FRACTION

            # C2 only (bottom)
            if h_c2 > 0:
                ax.axvspan(clip_start, clip_end, ymin=0, ymax=h_c2, alpha=MAX_ALPHA,
                           color=COLOR_C2_ONLY,
                           label="C2 Only (Weak Flow)" if first_label_c2 else "_nolegend_")
                first_label_c2 = False

            # C1 combined (middle)
            ymin_c1 = h_c2
            ymax_c1 = h_c2 + h_c1
            if h_c1 > 0:
                ax.axvspan(clip_start, clip_end, ymin=ymin_c1, ymax=ymax_c1, alpha=MAX_ALPHA,
                           color=COLOR_C1_COMBINED,
                           label="C1/C1+2 (direction-driven)" if first_label_c1 else "_nolegend_")
                first_label_c1 = False

            # Normal flow (top)
            ymin_normal = ymax_c1
            ymax_normal = ymax_c1 + h_normal
            if h_normal > 0:
                ax.axvspan(clip_start, clip_end, ymin=ymin_normal, ymax=ymax_normal, alpha=0.8,
                           color=COLOR_NORMAL,
                           label="Normal Flow" if first_label_normal else "_nolegend_")
                first_label_normal = False

    # Legend proxies for the last decade
    if i == len(decades) - 1:
        proxy_c1 = Rectangle((0, 0), 1, 1, fc=COLOR_C1_COMBINED, alpha=MAX_ALPHA)
        proxy_c2 = Rectangle((0, 0), 1, 1, fc=COLOR_C2_ONLY, alpha=MAX_ALPHA)
        proxy_normal = Rectangle((0, 0), 1, 1, fc=COLOR_NORMAL, alpha=0.7)
        handles = [proxy_c1, proxy_c2, proxy_normal]
        labels = ["C1/C1+2\n(direction-driven)", "C2 only\n(weak flow)", "Strong\nNW-flow"]
        ax.legend(handles, labels, loc='lower right', bbox_to_anchor=(1.96, -0.05), ncol=3,
                  frameon=True, fontsize=10, edgecolor='black',
                  title=r'Total bar height $\propto$ all days in a month')

    ax.set_xlim(mdates.date2num(pd.to_datetime(start)), mdates.date2num(pd.to_datetime(end)))
    ax.set_ylim(lat_range.start, lat_range.stop - 0.1)
    ax.set_yticks(np.arange(11.6, 12.1 + 0.01, 0.2))

    # Vertical year lines and labels
    start_yr = int(start[:4])
    end_yr = int(end[:4])
    for year in range(start_yr, end_yr + 1):
        jan_1 = pd.Timestamp(f'{year}-01-01')
        if jan_1 >= pd.to_datetime(start) and jan_1 <= pd.to_datetime(end):
            ax.axvline(x=jan_1, color='k', linestyle='--', alpha=1, linewidth=1)
    yearly_ticks = pd.date_range(start=f'{start_yr}-01-01', end=f'{end_yr}-01-01', freq='YS')
    ax.set_xticks(yearly_ticks)
    ax.set_xticklabels([str(year) for year in range(start_yr, end_yr + 1)])
    ax.set_ylabel('Latitude [°N]', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

    # Map inset in first panel
    if i == 0:
        map_ax = inset_axes(ax, width="24%", height="60%", loc='upper left',
                            bbox_to_anchor=(-0.5, -0.65, 2, 1.7), bbox_transform=ax.transAxes,
                            axes_class=cartopy.mpl.geoaxes.GeoAxes,
                            axes_kwargs=dict(projection=ccrs.PlateCarree()))
        map_ax.plot([selected_lon, selected_lon], [lat_range.start + 0.2, lat_range.stop - 0.1],
                    color='blue', linewidth=3, transform=ccrs.PlateCarree())
        mean_u = uo_spatial_mean.mean().item()
        mean_v = vo_spatial_mean.mean().item()
        map_ax.set_aspect('equal')
        map_ax.add_feature(cfeature.LAND, facecolor='saddlebrown', alpha=0.4, edgecolor='saddlebrown')
        map_ax.add_feature(cfeature.OCEAN, facecolor='#f7f7f7')
        map_ax.set_xlim(-70.5, -67.5)
        map_ax.set_ylim(10.5, 13.5)
        map_ax.set_xticks(np.arange(-70, -67, 1))
        map_ax.set_yticks(np.arange(11, 14, 1))
        map_ax.set_xticklabels([f"{abs(lon):.0f}°W" for lon in map_ax.get_xticks()], fontsize=10)
        map_ax.set_yticklabels([f"{lat:.0f}°N" for lat in map_ax.get_yticks()], fontsize=10)
        map_ax.set_title('(a) Cross-section location', fontsize=12, y=1.09)
        map_ax.text(-69.6, 11, 'Venezuela', fontsize=10, color='saddlebrown')
        map_ax.text(-69.5, 12.5, 'Curaçao', fontsize=10, color='saddlebrown')

    if i == len(decades) - 1:
        ax.set_xlabel('Year', fontsize=11)

    for spine in ax.spines.values():
        spine.set_visible(False)

# =============================================================================
# COLORBAR
# =============================================================================

cbar_ax_ssh = fig.add_axes([0.53, 0.237, 0.35, 0.01])
cbar_ssh = fig.colorbar(cf, cax=cbar_ax_ssh, orientation='horizontal',
                        label="Geostrophic zonal velocity [m/s]")
cbar_ssh.set_ticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
cbar_ssh.ax.tick_params(labelsize=11)
cbar_ssh.ax.xaxis.label.set_size(11)

plt.annotate('',
             xy=(0.452, 0.32), xycoords='figure fraction',
             xytext=(0.495, 0.32), textcoords='figure fraction',
             fontsize=12, ha='left', va='center',
             arrowprops=dict(arrowstyle='->', lw=2, color='k'))

# =============================================================================
# BOTTOM LEFT: DIRECTION ROSE (C)
# =============================================================================
rose_ax = fig.add_axes([0.12, -0.13, 0.32, 0.18], projection='polar')
directions_rad = direction_spatial_mean.values * np.pi / 180
bins = np.linspace(-np.pi, np.pi, 37)
hist, bin_edges = np.histogram(directions_rad, bins=bins)
hist_months = hist / 30.44
hist_transformed = np.sqrt(hist_months)
theta = (bin_edges[:-1] + bin_edges[1:]) / 2
width = np.diff(bin_edges)
bars = rose_ax.bar(theta, hist_transformed, width=width, bottom=0, edgecolor='black', linewidth=0.5)

for bar, angle in zip(bars, theta):
    angle_deg = angle * 180 / np.pi
    if -90 <= angle_deg <= -25:
        bar.set_facecolor(COLOR_NORMAL)
        bar.set_alpha(0.8)
    else:
        bar.set_facecolor(COLOR_C1_COMBINED)
        bar.set_alpha(0.8)

theta_start = -90 * np.pi / 180
theta_end = -25 * np.pi / 180
rose_ax.fill_between(np.linspace(theta_start, theta_end, 100),
                      0, rose_ax.get_ylim()[1],
                      color='lightgrey', alpha=0.3, zorder=0)

rose_ax.set_theta_zero_location('N')
rose_ax.set_theta_direction(-1)
rose_ax.set_title('(c) Criterion 1: Flow direction distribution\n ', fontsize=12, pad=15)
rose_ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
rose_ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10)

# Set the radial limit to cap at 100 months (sqrt(100) = 10)
rose_ax.set_ylim(0, 10)

yticks = rose_ax.get_yticks()
yticklabels = [f'{y**2:.0f}' if y > 0 else '0' for y in yticks]
rose_ax.set_yticks(yticks)
rose_ax.set_yticklabels(yticklabels, fontsize=9)
rose_ax.set_ylabel('Frequency [months] in log scale', fontsize=10, labelpad=25)

# =============================================================================
# BOTTOM RIGHT: SPEED DISTRIBUTION (D) - FULL 100% STACK
# =============================================================================

vel_ax = fig.add_axes([0.55, -0.13, 0.32, 0.203])

all_speed_data = speed_spatial_mean.values
weights_factor = 1 / 30.44

c1_combined_data_arr = all_speed_data[c1_combined_daily.values]
c2_only_data_arr = all_speed_data[c2_only_daily.values]
normal_flow_data_arr = all_speed_data[normal_flow_daily.values]

bins_vel = np.linspace(np.min(all_speed_data), np.max(all_speed_data), 50)

counts_c2_only, _, _ = vel_ax.hist(c2_only_data_arr, bins=bins_vel,
                                   edgecolor='black', linewidth=0.5, alpha=MAX_ALPHA, color=COLOR_C2_ONLY,
                                   weights=np.ones_like(c2_only_data_arr) * weights_factor,
                                   label='C2 only (weak flow)')

counts_normal, _, _ = vel_ax.hist(normal_flow_data_arr, bins=bins_vel,
                                  edgecolor='black', linewidth=0.5, alpha=0.7, color=COLOR_NORMAL,
                                  weights=np.ones_like(normal_flow_data_arr) * weights_factor,
                                  bottom=counts_c2_only,
                                  label='Strong NW-flow')

counts_c1_combined, _, _ = vel_ax.hist(c1_combined_data_arr, bins=bins_vel,
                                       edgecolor='black', linewidth=0.5, alpha=MAX_ALPHA, color=COLOR_C1_COMBINED,
                                       weights=np.ones_like(c1_combined_data_arr) * weights_factor,
                                       bottom=counts_c2_only + counts_normal,
                                       label='C1/C1+2 (direction-driven)')

vel_ax.axvline(speed_threshold, color='red', linestyle='--', linewidth=2.5,
               label=f'25th percentile: {speed_threshold:.3f} m/s')

handles, labels = vel_ax.get_legend_handles_labels()
# Reorder legend to C1, C2, Normal, Q25
handles_final = [handles[2], handles[0], handles[1], handles[3]]
labels_final = [labels[2], labels[0], labels[1], labels[3]]

vel_ax.set_xlabel('Flow Speed [m/s]', fontsize=11)
vel_ax.set_ylabel('Frequency [months]', fontsize=11)
vel_ax.set_title('(d) Criterion 2: Flow speed distribution', fontsize=12, pad=10)
vel_ax.legend(handles_final, labels_final, fontsize=9, loc='upper right')
vel_ax.grid(True, alpha=0.3, axis='y')
vel_ax.tick_params(labelsize=10)

# =============================================================================
# SAVE & SHOW
# =============================================================================

plt.savefig("figures/Fig02_EDDY-flow_periods.png",
            dpi=300, bbox_inches='tight')
plt.show()

# %%
