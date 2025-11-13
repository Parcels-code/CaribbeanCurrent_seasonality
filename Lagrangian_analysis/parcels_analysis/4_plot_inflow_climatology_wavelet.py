'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- loads inflow strength (calculated in 4_calc_inflow.py) and its climatology
- creates a combined figure with climatology and wavelet spectrum (Figure 7 in the manuscript)
- saves integrated spectrum for Figure 9 of the manuscript

Author: vesnaber
Kernel: parcels-dev-local
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pycwt as wavelet
from pathlib import Path
import pickle
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D 
import cmocean.cm as cmo
import os
# Fix numpy compatibility
if not hasattr(np, 'int'):
    np.int = int

#%%
# =============================================================================
# CONFIGURATION - use cache for faster plotting
# =============================================================================
parcels_config = 'GRENAVENE_coastrep050ED'
years_to_process = list(range(1993, 2025))
cache_dir = Path("cache")
cache_key = f"{parcels_config}_{min(years_to_process)}-{max(years_to_process)}_inflow_m2s"

# =============================================================================
# LOAD CACHED DAILY DATA
# =============================================================================
print("Loading cached daily inflow data...")
cache_file = cache_dir / f"particle_data_{cache_key}.pkl"

if not cache_file.exists():
    daily_inflow = pd.DataFrame({
        'release_date': pd.date_range('1993-01-01', periods=1000, freq='D'),
        'transport_m2s': np.random.rand(1000) * 10
    })
    print(f"Warning: Cache file not found. Using dummy data for demonstration.")
else:
    with open(cache_file, 'rb') as f:
        daily_inflow = pickle.load(f)

daily_inflow = daily_inflow.sort_values('release_date').reset_index(drop=True)

# =============================================================================
# PROCESS DAILY DATA
# =============================================================================
print("\nProcessing daily data for time series...")

daily_inflow['ma_30d'] = daily_inflow['transport_m2s'].rolling(window=30, center=True, min_periods=1).mean()
daily_inflow['ma_180d'] = daily_inflow['transport_m2s'].rolling(window=180, center=True, min_periods=1).mean()

x_days = np.arange(len(daily_inflow))
z_daily = np.polyfit(x_days, daily_inflow['transport_m2s'].values, 1)
p_daily = np.poly1d(z_daily)
daily_inflow['trend'] = p_daily(x_days)
trend_per_year_daily = z_daily[0] * 365.25

# =============================================================================
# MONTHLY CLIMATOLOGY
# =============================================================================
print("\nComputing monthly climatology...")

daily_inflow['month'] = daily_inflow['release_date'].dt.month

# =============================================================================
# AGGREGATE TO MONTHLY MEANS (for wavelet analysis)
# =============================================================================
print("\nAggregating to monthly means...")

daily_inflow['year_month'] = daily_inflow['release_date'].dt.to_period('M')

monthly_means = daily_inflow.groupby('year_month')['transport_m2s'].mean().reset_index()
monthly_means['date'] = monthly_means['year_month'].dt.to_timestamp() + pd.Timedelta(days=15)

# =============================================================================
# WAVELET ANALYSIS ON MONTHLY DATA
# =============================================================================
print("\nPerforming wavelet analysis on monthly data...")

# Prepare time series
data_monthly = monthly_means['transport_m2s'].values
time_monthly = monthly_means['date'].values

# Normalize
data_mean = np.mean(data_monthly)
data_std = np.std(data_monthly, ddof=1)
data_norm = (data_monthly - data_mean) / data_std

# Wavelet parameters
dt = 1.0
mother = wavelet.Morlet(6)
s0 = 2 * dt
dj = 1/12
J = 7 / dj
alpha, _, _ = wavelet.ar1(data_norm)

# CWT
wave, scales, freqs, coi, _, _ = wavelet.cwt(data_norm, dt, dj, s0, J, mother)
power = (np.abs(wave)) ** 2
period = 1 / freqs

# Significance testing
signif, _ = wavelet.significance(1.0, dt, scales, 0, alpha,
                                 significance_level=0.95, wavelet=mother)
sig95 = np.ones([1, len(data_norm)]) * signif[:, None]
sig95 = power / sig95

# Global spectrum
global_power = power.mean(axis=1)
dof = len(data_norm) - scales
global_signif, _ = wavelet.significance(1.0, dt, scales, 1, alpha,
                                        significance_level=0.95, dof=dof, wavelet=mother)

# Find dominant period
max_idx = np.nanargmax(global_power)
dominant_period = period[max_idx]


# =============================================================================
# CREATE FIGURE (FINAL 2x2 BALANCED LAYOUT)
# =============================================================================

fig = plt.figure(figsize=(14, 10)) 

gs = GridSpec(2, 2, figure=fig, 
              width_ratios=[1.8, 1], # 1.8 for Plot A (Time Series), 1 for Plot B (Boxplot)
              height_ratios=[1, 1],
              hspace=0.35, wspace=0.25)

# =============================================================================
# TOP LEFT: DAILY TIME SERIES (PLOT A)
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0]) 

ax1.plot(daily_inflow['release_date'], daily_inflow['transport_m2s'], 
         color='grey', linewidth=0.4, alpha=0.7, label='Daily inflow')
ax1.plot(daily_inflow['release_date'], daily_inflow['trend'], 
         'k--', linewidth=0.5, label=f'Trend: {trend_per_year_daily:.3f} m²/s per year', zorder=40)
ax1.plot(daily_inflow['release_date'], daily_inflow['ma_30d'], 
         color='royalblue', linewidth=1, alpha=0.8, label='30-day MA')
ax1.plot(daily_inflow['release_date'], daily_inflow['ma_180d'], 
         color='k', linewidth=1.5, alpha=0.8, label='180-day MA')

# Create legend with intensity gradients
legend_handles, legend_labels = ax1.get_legend_handles_labels()

ax1.set_xlim(daily_inflow['release_date'].min(), daily_inflow['release_date'].max())
ax1.set_ylim(daily_inflow['transport_m2s'].min() - 0.5, daily_inflow['transport_m2s'].max() + 1)
ax1.set_ylabel('Daily inflow transport per unit depth [m²/s]', fontsize=11)
ax1.set_title('(a) Daily inflow transport with moving averages and ENSO periods', 
              fontsize=12, pad=15)
ax1.grid(True, alpha=0.3)
ax1.legend(handles=legend_handles, loc='upper left', fontsize=7, ncol=4, framealpha=0.8)

major_years = pd.date_range(start='1995-01-01', end='2024-01-01', freq='5YS')
ax1.set_xticks(major_years)
ax1.set_xticklabels([str(yr.year) for yr in major_years], fontsize=10)
minor_years = pd.date_range(start='1993-01-01', end='2024-12-31', freq='YS')
ax1.set_xticks(minor_years, minor=True)
ax1.tick_params(axis='x', which='minor', length=4, width=1)
ax1.tick_params(axis='both', which='major', labelsize=10)


# =============================================================================
# TOP RIGHT: MONTHLY CLIMATOLOGY (PLOT B)
# =============================================================================
ax2 = fig.add_subplot(gs[0, 1])

monthly_data = []
for month in range(1, 13):
    month_values = daily_inflow[daily_inflow['month'] == month]['transport_m2s'].values
    monthly_data.append(month_values)

bp = ax2.boxplot(monthly_data, labels=['J','F','M','A','M','J','J','A','S','O','N','D'],
                 patch_artist=True, showmeans=True,
                 widths=0.5,
                 meanprops=dict(marker='^', markerfacecolor='lightgrey', 
                               markeredgecolor='lightgrey', markersize=6),
                 flierprops=dict(marker='o', markerfacecolor='black', 
                                markersize=3, linestyle='none', markeredgecolor='black'),
                 medianprops=dict(color='gold', linewidth=2))

for patch in bp['boxes']:
    patch.set_facecolor('k')
    patch.set_alpha(0.6)
    patch.set_edgecolor('k')
    patch.set_linewidth(0)

legend_elements = [
    Line2D([0], [0], color='gold', linewidth=2, label='Median'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='lightgrey',
           markersize=7, linestyle='None', label='Mean')
]
ax2.legend(handles=legend_elements, fontsize=9, loc='upper right')

ax2.set_xlabel('Month', fontsize=11)
ax2.set_ylabel('Daily inflow transport per unit depth [m²/s]', fontsize=11)
ax2.set_title('(b) Monthly distribution of daily inflow', 
              fontsize=12, pad=15)
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='both', labelsize=10)
ax2.set_ylim(daily_inflow['transport_m2s'].min() - 0.5, daily_inflow['transport_m2s'].max() + 0.5)


# =============================================================================
# BOTTOM: WAVELET POWER SPECTRUM (PLOT C) - Spans both columns
# =============================================================================
ax3 = fig.add_subplot(gs[1, :])
time_plot = pd.to_datetime(time_monthly)

# Round the maximum power value to a nice number
max_power = np.percentile(power, 95)
if max_power < 10:
    max_power_rounded = np.ceil(max_power)
elif max_power < 50:
    max_power_rounded = np.ceil(max_power / 5) * 5
else:
    max_power_rounded = np.ceil(max_power / 10) * 10

levels = np.linspace(0, max_power_rounded, 100)

contourf = ax3.contourf(time_plot, period, power,
                       levels=levels, cmap=cmo.amp_r, extend='max')
ax3.contour(time_plot, period, sig95,
           levels=[1], colors='blue', linewidths=2, linestyles=':', alpha=0.9)
ax3.fill_between(time_plot, coi, period.max(),
                color='white', alpha=0.5, hatch='xxx', edgecolor='white', linewidth=0.8)

ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Period [months]', fontsize=11)
ax3.set_title('(c) Wavelet power spectrum (based on monthly data)\n(95% significance in blue contour dots)',
             fontsize=12, pad=10)
ax3.set_yscale('log')
ax3.set_ylim([period.min(), 140])
ax3.set_xlim(time_plot.min(), time_plot.max())

yticks = [1, 3, 6, 12, 24, 48, 96]
yticks = [y for y in yticks if period.min() <= y <= period.max()]
ax3.set_yticks(yticks)
ax3.set_yticklabels([str(int(y)) for y in yticks])
major_years = pd.date_range(start='1995-01-01', end='2024-01-01', freq='5YS')
ax3.set_xticks(major_years)
ax3.set_xticklabels([str(yr.year) for yr in major_years], fontsize=10)
minor_years = pd.date_range(start='1993-01-01', end='2024-12-31', freq='YS')
ax3.set_xticks(minor_years, minor=True)
ax3.tick_params(axis='x', which='minor', length=4, width=1)
ax3.tick_params(axis='both', which='major', labelsize=10)
ax3.axhline(6, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
ax3.axhline(12, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
ax3.axhline(24, color='grey', linestyle='--', linewidth=1.5, alpha=0.8)
ax3.invert_yaxis()

# Colorbar for Plot C (placed below the full width)
cbar_ax = fig.add_axes([0.13, 0.03, 0.76, 0.02]) 
cbar = plt.colorbar(contourf, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Normalized power [ ]', fontsize=11)

num_ticks = 5
cbar_ticks = np.linspace(0, max_power_rounded, num_ticks)
cbar.set_ticks(cbar_ticks)

# Max power annotation
ax3.text(0.005, 0.98, f'Max: {dominant_period:.1f} months',
        transform=ax3.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='square,pad=0.3', facecolor='yellow', alpha=0.9))

# =============================================================================
# SAVE FIGURE
# =============================================================================
output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"Fig07_{parcels_config}_inflow_wavelet_m2s.png"

# Adjusted rect for tight_layout for the bottom colorbar and plot titles
plt.tight_layout(rect=[0, 0.08, 1, 1]) 
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved: {output_path}")
plt.show()

# %%
# =============================================================================
# EXTRACT JULY AND NOVEMBER STATISTICS FOR MANUSCRIPT
# =============================================================================
print("\n" + "="*80)
print("STATISTICS:")
print("="*80)

july_values = daily_inflow[daily_inflow['month'] == 7]['transport_m2s'].values
november_values = daily_inflow[daily_inflow['month'] == 11]['transport_m2s'].values

print(f"July median: {np.median(july_values):.3f} m²/s")
print(f"November median: {np.median(november_values):.3f} m²/s")
print("="*80)

# %%
# =============================================================================
# TREND STATISTICS FOR MANUSCRIPT
# =============================================================================
from scipy import stats

print("\n" + "="*80)
print("STATISTICS")
print("="*80)

# Calculate p-value for the linear trend
x_days = np.arange(len(daily_inflow))
slope, intercept, r_value, p_value, std_err = stats.linregress(x_days, daily_inflow['transport_m2s'].values)

# Time span
start_date = daily_inflow['release_date'].min()
end_date = daily_inflow['release_date'].max()
time_span_years = (end_date - start_date).days / 365.25

# Trend statistics
trend_per_year = slope * 365.25
total_decline = trend_per_year * time_span_years
mean_inflow = daily_inflow['transport_m2s'].mean()
decline_percentage = (total_decline / mean_inflow) * 100

print(f"\nTime period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"Time span: {time_span_years:.2f} years")
print(f"\nMean inflow: {mean_inflow:.3f} m²/s")
print(f"\nTrend per year: {trend_per_year:.4f} m²/s/year")
print(f"P-value: {p_value:.6f}")
print(f"R²: {r_value**2:.4f}")
print(f"\nTotal decline over {time_span_years:.1f} years: {total_decline:.3f} m²/s")
print(f"Decline as % of mean: {decline_percentage:.2f}%")
print("="*80)

# %%
# =============================================================================
# DATA EXTRACTION for Integrated Spectrum plot (Figure 9)
# =============================================================================

plot_d_data = {
    'period': period,              # Y-axis / Period scale
    'global_power': global_power,  # Global Spectrum (GWS) values
    'global_signif': global_signif # 95% Significance values
}

output_dir = Path('../../integrated_spectra_combined/data_wavelet')
filename = output_dir / 'CC_inflow_integrated_spectrum.pkl'

if not output_dir.exists():
    output_dir.mkdir(parents=True)

# Save the data using pickle
with open(filename, 'wb') as f:
    pickle.dump(plot_d_data, f)

print(f"\nData for Integrated Spectrum saved to: {filename}")

# %%
