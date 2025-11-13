'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
- downloads SSH directly from CMEMS (product GLORYS)

Data is divided in two subsets (my and myint, as is available in the product).

Author: vesnaber
kernel: parcels-dev-local
Requirement: kernel needs to have the module copernicusmarine installed
'''

# %%
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
  variables=["zos"],
  minimum_longitude=-70.04293289633253,
  maximum_longitude=-58.9031948952658,
  minimum_latitude=9.45121100898802,
  maximum_latitude=18.887625156680638,
  start_datetime="1993-01-01T00:00:00",
  end_datetime="2021-06-30T00:00:00",
  minimum_depth=0.49402499198913574,
  maximum_depth=0.49402499198913574,
  output_directory="/nethome/berto006/CaribbeanCurrent_seasonality/Lagrangian_analysis/parcels_run/GLORYS_input"
)

copernicusmarine.subset(
  dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
  variables=["zos"],
  minimum_longitude=-70.04293289633253,
  maximum_longitude=-58.9031948952658,
  minimum_latitude=9.45121100898802,
  maximum_latitude=18.887625156680638,
  start_datetime="2021-07-01T00:00:00",
  end_datetime="2024-12-31T00:00:00",
  minimum_depth=0.49402499198913574,
  maximum_depth=0.49402499198913574,
  output_directory="/nethome/berto006/CaribbeanCurrent_seasonality/Lagrangian_analysis/parcels_run/GLORYS_input"
)
# %%
