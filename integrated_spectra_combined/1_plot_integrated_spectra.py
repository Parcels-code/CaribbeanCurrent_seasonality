'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- plots integrated spectrum for Figure 9 of the manuscript

Author: vesnaber
Kernel: parcels-dev-local
'''


#%%
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle 

# =============================================================================
# 1. DATA LOADING
# =============================================================================

# Define filenames for the integrated spectrum data
filenames = {
    'eulerian_NW': 'data_wavelet/eulerian_flow_integrated_spectrum.pkl',
    'lagrangean_NW': 'data_wavelet/lagrangian_integrated_spectrum.pkl',
    'CC_inflow': 'data_wavelet/CC_inflow_integrated_spectrum.pkl',
    'CUR_wind': 'data_wavelet/wind_CUR_integrated_spectrum.pkl', # Corrected name if needed based on previous output
    'GRE_wind': 'data_wavelet/wind_GRE_integrated_spectrum.pkl'  # Corrected name if needed based on previous output
}

loaded_data = {}

def load_spectrum_data(filename_path):
    if not os.path.exists(filename_path):
        print(f"Error: Data file not found at {filename_path}.")
        return None
    with open(filename_path, 'rb') as f:
        return pickle.load(f)

for key, path in filenames.items():
    data = load_spectrum_data(path)
    if data:
        loaded_data[key] = data
    else:
        print(f"Skipping plot for {key} due to missing data.")

# =============================================================================
# 2. PLOT GENERATION
# =============================================================================

fig, axes = plt.subplots(1, 4, figsize=(12, 6), sharey=True) # 1 row, 4 columns, sharey=True

fig.suptitle('Comparison of integrated wavelet spectra', fontsize=14, y=0.94)

# Common Y-axis settings
COMMON_Y_TICKS = [0.1, 0.5, 1, 4,  6, 12, 24, 60] # Period [months]
COMMON_Y_LIM = [0.1, 60] # Common y-limit (Min must be > 0 for log scale)

# Plotting styles
FILL_ALPHA = 0.3
LINE_WIDTH = 2
SIG_LINESTYLE = ':'
SIG_LINEWIDTH = 1.5
REF_LINE_COLOR = 'gray'
REF_LINE_STYLE = '--'
REF_LINE_WIDTH = 1
REF_LINE_ALPHA = 0.7

# --- UNIFIED COLOR FOR NON-WIND PLOTS (Eulerian, Lagrangian, CC Inflow) ---
UNIFIED_COLOR_LINE = 'darkred'
UNIFIED_COLOR_FILL = 'darkred'

# Specific colors for each spectrum
colors = {
    'eulerian_NW': {'fill': UNIFIED_COLOR_FILL, 'line': UNIFIED_COLOR_LINE},
    'lagrangean_NW': {'fill': UNIFIED_COLOR_FILL, 'line': UNIFIED_COLOR_LINE},
    'CC_inflow': {'fill': UNIFIED_COLOR_FILL, 'line': UNIFIED_COLOR_LINE},
    'GRE_wind': {'fill': 'olivedrab', 'line': 'darkgreen'},
    'CUR_wind': {'fill': 'firebrick', 'line': 'darkred'},
}
# --------------------------------------------------------------------------

# Mapping titles for the subplots
plot_titles = {
    'eulerian_NW': '(a) Eulerian analysis:\nEDDY-flow occurance',
    'lagrangean_NW': '(b) Lagrangian analysis: daily\ntransport through Segment 1',
    'CC_inflow': '(c) Caribbean Current inflow:\n monthly distribution',
    'wind_combined': '(d) Wind speed\n(Curaçao & Grenada)',
}

# Order of plots in the figure
plot_keys = ['eulerian_NW', 'lagrangean_NW', 'CC_inflow']
current_ax_idx = 0

def get_period_data(data):
    """Tries to retrieve period data using 'period_months' then 'period'."""
    if 'period_months' in data:
        return data['period_months']
    elif 'period' in data:
        return data['period']
    else:
        raise KeyError("Data dictionary missing both 'period_months' and 'period' keys.")

def get_global_ws_data(data):
    """Tries to retrieve GWS data using 'global_ws', 'global_power', then 'power'."""
    if 'global_ws' in data:
        return data['global_ws']
    elif 'global_power' in data: 
        return data['global_power']
    elif 'power' in data:
        return data['power']
    else:
        raise KeyError("Data dictionary missing suitable keys for the spectrum ('global_ws', 'global_power', or 'power').")

def get_global_signif_data(data):
    """Tries to retrieve Significance data using 'global_signif' then 'signif'."""
    if 'global_signif' in data:
        return data['global_signif']
    elif 'signif' in data:
        return data['signif']
    else:
        raise KeyError("Data dictionary missing both 'global_signif' and 'signif' keys for significance.")


for key in plot_keys:
    ax = axes[current_ax_idx]
    if key in loaded_data:
        data = loaded_data[key]
        fill_color = colors[key]['fill']
        line_color = colors[key]['line']
        
        # Robust data retrieval for all three necessary arrays
        try:
            plot_period = get_period_data(data)
            plot_ws = get_global_ws_data(data)
            plot_signif = get_global_signif_data(data)
        except KeyError as e:
            print(f"Critical Error: {e} in data for {key}. Skipping plot.")
            ax.set_title(f"Data Error: {key}", fontsize=14, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            current_ax_idx += 1
            continue


        ax.fill_betweenx(plot_period, 0, plot_ws, 
                         color=fill_color, alpha=FILL_ALPHA)
        ax.plot(plot_ws, plot_period, color=line_color, 
                linewidth=LINE_WIDTH, label=f'{key.replace("_", " ").title()} Spectrum')
        ax.plot(plot_signif, plot_period, color=line_color, 
                linestyle=SIG_LINESTYLE, linewidth=SIG_LINEWIDTH, label='95% Sig.')
    
        ax.set_title(plot_titles[key], fontsize=12, pad=10)
        ax.grid(True, alpha=0.3, axis='both')
        ax.set_xlabel('Normalized power [ ]', fontsize=12)
        
    else:
        ax.set_title(f"Missing Data: {key}", fontsize=14, color='red')
        ax.set_xticks([])
        ax.set_yticks([])

    current_ax_idx += 1

# --- Combined CUR and GRE Wind Spectrum Plot (Last subplot) ---
ax_wind = axes[current_ax_idx]
if 'CUR_wind' in loaded_data and 'GRE_wind' in loaded_data:
    cur_data = loaded_data['CUR_wind']
    gre_data = loaded_data['GRE_wind']

    cur_period = get_period_data(cur_data)
    gre_period = get_period_data(gre_data)

    cur_ws = get_global_ws_data(cur_data)
    cur_signif = get_global_signif_data(cur_data)
    gre_ws = get_global_ws_data(gre_data)
    gre_signif = get_global_signif_data(gre_data)

    # Plot GRE Wind Spectrum
    ax_wind.fill_betweenx(gre_period, 0, gre_ws, 
                     color=colors['GRE_wind']['fill'], alpha=FILL_ALPHA)
    ax_wind.plot(gre_ws, gre_period, color=colors['GRE_wind']['line'], 
            linewidth=LINE_WIDTH, label='GRE Global Spectrum')
    ax_wind.plot(gre_signif, gre_period, color=colors['GRE_wind']['line'], 
            linestyle=SIG_LINESTYLE, linewidth=SIG_LINEWIDTH, label='GRE 95% Sig.')
    
    # Plot CUR Wind Spectrum
    ax_wind.fill_betweenx(cur_period, 0, cur_ws, 
                     color=colors['CUR_wind']['fill'], alpha=FILL_ALPHA)
    ax_wind.plot(cur_ws, cur_period, color=colors['CUR_wind']['line'], 
            linewidth=LINE_WIDTH, label='CUR Global Spectrum')
    ax_wind.plot(cur_signif, cur_period, color=colors['CUR_wind']['line'], 
            linestyle=SIG_LINESTYLE, linewidth=SIG_LINEWIDTH, label='CUR 95% Sig.')

    ax_wind.set_title(plot_titles['wind_combined'], fontsize=12, pad=10)
    ax_wind.grid(True, alpha=0.3, axis='both')
    ax_wind.set_xlabel('Normalized power [ ]', fontsize=12)

    wind_legend_elements = [
        Line2D([0], [0], color=colors['GRE_wind']['line'], linewidth=LINE_WIDTH, label='Grenada'),
        Line2D([0], [0], color=colors['CUR_wind']['line'], linewidth=LINE_WIDTH, label='Curaçao'),
        Line2D([0], [0], color='k', linestyle=SIG_LINESTYLE, linewidth=SIG_LINEWIDTH, label='95% significance'),
    ]
    ax_wind.legend(handles=wind_legend_elements, fontsize=10, loc='best')

else:
    ax_wind.set_title(f"Missing Wind Data", fontsize=14, color='red')
    ax_wind.set_xticks([])
    ax_wind.set_yticks([])


# --- Apply common Y-axis settings to all plots ---
for ax in axes:
    ax.set_yscale('log')
    ax.set_yticks(COMMON_Y_TICKS)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_yticklabels([str(t) if t >= 1 else str(t) for t in COMMON_Y_TICKS]) # Format minor ticks nicely
    ax.set_ylim(COMMON_Y_LIM)
    ax.set_xlim(0,None)
    ax.invert_yaxis()
    for y_val in [4, 6, 12]:
        ax.axhline(y_val, color=REF_LINE_COLOR, linestyle=REF_LINE_STYLE, 
                   linewidth=REF_LINE_WIDTH, alpha=REF_LINE_ALPHA)
axes[0].set_ylabel('Period [months]', fontsize=12)


plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
plt.savefig('figures/Fig09_integrated_wavelet_spectra.png', dpi=300)

# %%
