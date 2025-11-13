# CaribbeanCurrent_seasonality

Repository for Vesna's project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This project contains the following folders:

## Project Structure:

- [**Eulerian_analysis**](./Eulerian_analysis/): Contains the scripts to analyse geostrophic flow at 69.0 E meridional cross-section for the 32-year period
- [**Lagrangian_analysis**](./Lagrangian_analysis/): Contains scripts to run and analyse Parcels experiments to track Caribbean Current across the Caribbean Sea and the stength of the inflow
- [**wind_analysis**](./wind_analysis/): Contains scripts to analyse wind data
- [**integrated_spectra_combined**](./integrated_spectra_combined/): Contains scripts to plot all 4 wavelet spectra (integrated in time) for comparison of spectra

## Hydrodynamic input: GLORYS model (CMEMS)

For this project the hydrodynamic input used is the 32 years (1993–2024) of geostrophic flow patterns derived from sea surface height data with the EU Copernicus Marine Service Information product reanalysis GLORYS12V1 (Lellouche et al., 2021). 

Lellouche, J.-M., Greiner, E., Bourdallé-Badie, R., Garric, G., Melet, A., Drévillon, M., Bricaud, C., Hamon, M., Le Galloudec, O., Regnier, C., Candela, T., Testut, C.-E., Gasparin, F., Ruggiero, G., Benkiran, M., Drillet, Y., & Le Traon, P.-Y. (2021). The Copernicus global 1/12° oceanic and sea ice GLORYS12 reanalysis. Frontiers in Earth Science, 9, 698876. https://doi.org/10.3389/feart.2021.698876

Geostrophic flow is calculated from the sea surface height data output (SSH) using central differences approach. 

## Eulerian analysis: scripts

- `1_calc_Eulerian_timeseries_full.py`: calculate geostrophic flow from SSH, define EDDY/NW-flow regimes and create **Figure 2** from the manuscript
- `2_calc_Eulerian_yearly_statistics.py`: calculate geostrophic flow from SSH, define EDDY/NW-flow regimes, compute annual statistics and create **Figure 3** from the manuscript
- `3_calc_Eulerian_monthly_and_wavelet.py`: calculate geostrophic flow from SSH, define EDDY/NW-flow regimes, compute monthly statistics and wavelt analhysis, and create **Figure 4** from the manuscript

## Lagrangian particle tracking using Parcels: simulations and diagnostics

Lagrangian particle tracking simulations were conducted using Parcels v3.1.2 to track Caribbean Current pathways through the eastern Caribbean Sea. Particles were released every 24 hours from 1 January 1993 to 31 December 2024 along a meridional transect between Venezuela (10.72°N, 62.5°W) and Grenada (12.04°N, 61.7°W), spanning 171 km through the Grenada Passage.

The simulated particles represent surface geostrophic transport derived from GLORYS12V1 SSH gradients at 1/12° resolution. A total of 1707 particles were spaced at 0.1 km intervals along the release transect, providing uniform sampling of the inflow. Particles were advected using a fourth-order Runge-Kutta integration scheme with a 1-hour internal timestep, with positions recorded every 12 hours until particles exited the domain.

Name of the Parcels simualtion is: **GRENAVENE** (based on release location of the particles at Grenada-Venezuela cross-section).

In folder [**Lagrangian_analysis/parcels_run**](./Lagrangian_analysis/parcels_run/): 
- `0_download_GLORYS_SSH.py`: script to download GLORYS dataset
- `1_calc_geostrophic_flow.py`: calculate geostrophic flow from SSH, unify the NaN fields of U and V velocities to create a consistent land mask, add a displacement field to push particles off the coast and save calculated geostrophic flow as .nc file
- `2_particle_release_locations.py`: define particle release locations at the cross-section between Grenada and Venezuela and save locations as numpy array
- `3_run_GRENAVENE.py`: run Parcels simulaiton called GRENAVENE in parallel (using slurm)
- `4_plot_methodology.py`: create figure for Methodology section (**Figure 1** in the manuscript)
- `submit_run_GRENAVENE.sh`: script to submit multiple parallel runs for Parcels simulaitons (Lorenz IMAU cluster computer)

In folder [**Lagrangian_analysis/parcels_analysis**](./Lagrangian_analysis/parcels_analysis/): 

- `1_calc_trajectory_crossings.py`:
- `2_plot_heatmap`:
- `3_plot_climatology_wavelet`:
- `4_calc_inflow`:
- `4_plot_inflow_climatology_wavelet`:

## Wind analysis

- `1_calc_wind_speed_all`:

## Combaning integrated spectra across analyses

- `1_plot_integrated_spectra`:
