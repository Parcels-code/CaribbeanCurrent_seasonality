'''
Project: Seasonal patterns of the Caribbean Current and its influence on the small island of Curaçao 

This script:
    - Runs Parcels simulation for particle releases along the Grenada-Venezuela line
    - Uses geostrophic velocities with coastal repellent field added

part_config = 'GRENAVENE_coastrep050ED' 
- GRENAVENE stands for Grenada-Venezuela release locations of particles
- ED stands for edge deleted (due to code in 1_calc_geostrophic_flow.py that creates displacement at the domain edges too)
- coastrep050 stands for repellent velocity of 0.5 m/s

Author: vesnaber
Kernel: parcels-dev-local
'''

import sys
import numpy as np
from datetime import datetime, timedelta
from parcels import JITParticle, FieldSet, Variable, ParticleSet, AdvectionRK4
import parcels


part_config = 'GRENAVENE_coastrep050ED' 
print(f"Parcels configuration name: {part_config}")

# # --- Parse batch number from command-line ---
if len(sys.argv) < 2:
    raise ValueError("Please provide a batch number (1-16) as argument.")
batch = int(sys.argv[1])
if batch not in range(1, 17):
    raise ValueError("Batch must be an integer between 1 and 16.")

# running on 16 cores for fast computation
batch_ranges = {
    1: (1993, 1994),
    2: (1995, 1996),
    3: (1997, 1998),
    4: (1999, 2000),
    5: (2001, 2002),
    6: (2003, 2004),
    7: (2005, 2006),
    8: (2007, 2008),
    9: (2009, 2010),
    10: (2011, 2012),
    11: (2013, 2014),
    12: (2015, 2016),
    13: (2017, 2018),
    14: (2019, 2020),
    15: (2021, 2022),
    16: (2023, 2024)
}


start_year, end_year = (batch_ranges[batch])

# start_year, end_year = batch_ranges[batch]
start_date = datetime(start_year, 1, 1)
end_date = datetime(end_year, 12, 31)

# If batch 16, adjust end date to available data
if batch == 16:
    end_date = datetime(2024, 12, 31)

print(f"Particles released along the line between Grenada and Venezuela at every 1 degree")

# --- Particle Configuration ---
lon_part = np.load('parcels_input/particles_GRENAVENE_lon.npy')
lat_part = np.load('parcels_input/particles_GRENAVENE_lat.npy')
print(f"Loaded {len(lon_part)} particles per release")


fieldset = FieldSet.from_netcdf(
    filenames= f'geostrophic_flow_input/geostrophic_velocity_field.nc', #'GLORYS_input/geostrophic_velocities_f_points.nc',  #f'../parcels_run_CORRECT/geostrophic_uninans.nc',
    variables={'U': 'U_combined', 'V': 'V_combined'},
    dimensions={'lon': 'longitude', 'lat': 'latitude', 'time': 'time'},
    interp_method={
        "U": "freeslip",
        "V": "freeslip",
    }, 
)

print("Loaded fieldset with U and V velocities at f-points")

# --- Particle Class ---
SampleParticle = parcels.JITParticle.add_variables(
    [
        parcels.Variable("U", dtype=np.float32, initial=np.nan, to_write="once"), 
        parcels.Variable("V", dtype=np.float32, initial=np.nan, to_write="once"),
        parcels.Variable('particle_age', dtype=np.float32, initial=0.)
    ]
)

# --- Process by year to reduce memory usage ---
for year in range(start_year, end_year + 1):
    year_start = datetime(year, 1, 1)
    year_end = datetime(year, 12, 31)
    
    # Adjust for batch 6 final year
    if year == 2024:
        year_end = datetime(2024, 12, 31)
    
    print(f"\nProcessing year {year}...")
    
    # --- Time Release Setup ---
    # Calculate days since GLORYS reference time (1993-01-01)
    glorys_ref_date = datetime(1993, 1, 1)
    
    # Create daily releases for this year
    current_date = year_start
    time_releases = []
    
    while current_date <= year_end:
        # Calculate seconds since GLORYS reference date
        seconds_since_ref = (current_date - glorys_ref_date).total_seconds()
        time_releases.append(seconds_since_ref)
        current_date += timedelta(days=1)
    
    # Create particle arrays
    lons = np.tile(lon_part, len(time_releases))
    lats = np.tile(lat_part, len(time_releases))
    times = np.repeat(time_releases, len(lon_part))
    
    # --- Create Particle Set ---
    pset = ParticleSet(
        fieldset=fieldset,
        pclass=SampleParticle,
        lon=lons,
        lat=lats,
        time=times
    )
    
    print(f"Created {len(pset)} particles for year {year}")
    
    # --- Output ---
    output_file = f'GRENAVENE_output/{part_config}_{year}.zarr'
    outputfile = pset.ParticleFile(name=output_file, outputdt=timedelta(hours=12))
    
    # --- Kernels ---
    def DeleteParticle(particle, fieldset, time):
        if particle.state >= 50:
            particle.delete()
    
    def UpdateAge(particle, fieldset, time):
        if particle.time > 0:
            particle.particle_age += particle.dt
    
    def DeleteFarParticles(particle, fieldset, time): # this is used because we don't want to use the repellent velocities at the edges of the domain
        if particle.lon < -69.7 or particle.lon > -60.5 or particle.lat < 9.8 or particle.lat > 18.6:
            particle.delete()

    def SampleVel_correct(particle, fieldset, time):
        particle.U, particle.V = fieldset.UV[time, particle.depth, particle.lat, particle.lon]


    kernels = [AdvectionRK4, SampleVel_correct, DeleteParticle, UpdateAge, DeleteFarParticles]

    # --- Run Simulation ---
    # Calculate runtime: from start of year to end of year + 178 days (1/2 year) for particle aging
    runtime_days = (year_end - year_start).days + 1 + 178
    
    print(f"Starting simulation for year {year} (runtime: {runtime_days} days)...")
    
    pset.execute(
        kernels,
        runtime=timedelta(days=runtime_days),
        dt=timedelta(hours=1),
        output_file=outputfile
    )
    
    print(f"Completed year {year}")

print(f"\nBatch {batch} completed successfully!")
