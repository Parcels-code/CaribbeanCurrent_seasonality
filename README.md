# CaribbeanCurrent_seasonality

Repository for Vesna's project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This project contains the following folders:

## Project Structure:

- [**Eulerian_analysis**](./Eulerian_analysis/): Contains the scripts to analyse geostrophic flow at 69.0 E meridional cross-section for the 32-year period
- [**Lagrangian_analysis**](./Lagrangian_analysis/): Contains scripts to run and analyse Parcels experiments to track Caribbean Current across the Caribbean Sea and the stength of the inflow
- [**wind_analysis**](./wind_analysis/): Contains scripts to analyse wind data
- [**integrated_spectra_combined**](./integrated_spectra_combined/): Contains scripts to plot all 4 wavelet spectra (integrated in time) for comparison of spectra

## Eulerian analysis: scripts

- `1_calc_Eulerian_timeseries_full.py`: calculate geostrophic flow from SSH, define EDDY/NW-flow regimes and create **Figure 2** from the manuscript
- `2_calc_Eulerian_yearly_statistics.py`: calculate geostrophic flow from SSH, define EDDY/NW-flow regimes, compute annual statistics and create **Figure 3** from the manuscript
- `3_calc_Eulerian_monthly_and_wavelet.py`: calculate geostrophic flow from SSH, define EDDY/NW-flow regimes, compute monthly statistics and wavelt analhysis, and create **Figure 4** from the manuscript

## Lagrangian particle tracking using Parcels

Lagrangian particle tracking simulations were conducted using Parcels v3.1.2 to track Caribbean Current pathways through the eastern Caribbean Sea. Particles were released every 24 hours from 1 January 1993 to 31 December 2024 along a meridional transect between Venezuela (10.72°N, 62.5°W) and Grenada (12.04°N, 61.7°W), spanning 171 km through the Grenada Passage.

The simulated particles represent surface geostrophic transport derived from GLORYS12V1 SSH gradients at 1/12° resolution. A total of 1707 particles were spaced at 0.1 km intervals along the release transect, providing uniform sampling of the inflow. Particles were advected using a fourth-order Runge-Kutta integration scheme with a 1-hour internal timestep, with positions recorded every 12 hours until particles exited the domain.

Name of the Parcels simualtion is: **GRENAVENE** (based on release location of the particles at Grenada-Venezuela cross-section).

**Hydrodynamic input: GLORYS model (CMEMS)**

For this project the hydrodynamic input used is the 32 years (1993–2024) of geostrophic flow patterns derived from sea surface height data with the EU Copernicus Marine Service Information product reanalysis GLORYS12V1 (Lellouche et al., 2021). 

Lellouche, J.-M., Greiner, E., Bourdallé-Badie, R., Garric, G., Melet, A., Drévillon, M., Bricaud, C., Hamon, M., Le Galloudec, O., Regnier, C., Candela, T., Testut, C.-E., Gasparin, F., Ruggiero, G., Benkiran, M., Drillet, Y., & Le Traon, P.-Y. (2021). The Copernicus global 1/12° oceanic and sea ice GLORYS12 reanalysis. Frontiers in Earth Science, 9, 698876. https://doi.org/10.3389/feart.2021.698876

Geostrophic flow is calculated from the sea surface height data output (SSH) using central differences approach. 

**Simulations and analysis:**

In folder [**Lagrangian_analysis/parcels_run**](./Lagrangian_analysis/parcels_run/): 
- `0_download_GLORYS_SSH.py`: script to download GLORYS dataset
- `1_calc_geostrophic_flow.py`: calculate geostrophic flow from SSH, unify the NaN fields of U and V velocities to create a consistent land mask, add a displacement field to push particles off the coast and save calculated geostrophic flow as .nc file
- `2_particle_release_locations.py`: define particle release locations at the cross-section between Grenada and Venezuela and save locations as numpy array
- `3_run_GRENAVENE.py`: run Parcels simulaiton called GRENAVENE in parallel (using slurm)
- `4_plot_methodology.py`: create figure for Methodology section (**Figure 1** in the manuscript)
- `submit_run_GRENAVENE.sh`: script to submit multiple parallel runs for Parcels simulaitons (Lorenz IMAU cluster computer)

In folder [**Lagrangian_analysis/parcels_analysis**](./Lagrangian_analysis/parcels_analysis/): 

- `1_calc_trajectory_crossings.py`: calculate meridional crossings of particles at a specified longitude and save results as .nc file
- `2_plot_heatmap`: plot heatmap of particle crossings and duration of travelling (**Figure 5** in manuscript)
- `3_plot_climatology_wavelet`: calculate and plot the monthly climatology of particle arrivals per segment, wavelet analysis on Segment 1 and create a combined figure with climatology and wavelet spectrum (**Figure 6** in the manuscript)
- `4_calc_inflow`: calculate inflow strength and save as cache file
- `4_plot_inflow_climatology_wavelet`: load inflow strength, calculate climatology and wavelet analysis, and create a combined figure (**Figure 7** in the manuscript)

## Wind analysis

Wind speed data were obtained from the product Global Ocean Hourly Reprocessed Sea Surface Wind and Stress from Scatterometer and Model, provided by the E.U. Copernicus Marine Service, covering the period from 1 June 1994 to 31 December 2024. The data was downlaoded at https://doi.org/10.48670/moi-00185

Two locations were used for the analysis: Curaçao and Grenada.

- `1_calc_wind_speed_all`: calculates the sea surface wind speed time series, compute monthly statistics and wavelet spectra for two locations (Grenada and Curaçao) and create a combined figure with climatology and wavelet spectra (**Figure 8** in the manuscript)

## Combaning integrated spectra across analyses

All integrated wavelet spectra for each analysis are already calculated during the each individual wavelet analysis. They are saved as **.pkl** files. This script only combines them all and create a figure of all of them with the same scale of y-axis.

- `1_plot_integrated_spectra`: loads and plots integrated spectra for **Figure 9** of the manuscript. 
