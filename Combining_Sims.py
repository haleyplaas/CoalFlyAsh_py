#%% COMPILING PD RUNS TO FIX DUST ISSUES -- h2 files
import numpy as np
import xarray as xr
from netCDF4 import Dataset

# LOAD IN THE NETCDF FILE -----------------------------------------
# FIRE x1 -- DERECHO -- FIRE COMES FROM HERE
sim_directory_1 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL0.2-WOOD10-OIL38.cam.h1.2009-2011_PD.nc"
ds_h1_x1 = xr.open_dataset(sim_directory_1)
#ds_h1_x1 = ds_h1_x1.assign_coords(lat=ds_h1_x1["lat"].round(3))

# FIRE x2 -- CHEYENNE -- DUST AND ANTHRO COME FROM HERE 
sim_directory_2 = "D:\\CoalFlyAsh\\Cheyenne_runs_with_seasonality\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL0.2-WOOD10-OIL38.cam.h1.2009-2011_Cheyenne.nc"
ds_h1_x2 = xr.open_dataset(sim_directory_2)
#ds_h1_x2 = ds_h1_x2.assign_coords(lat=ds_h1_x2["lat"].round(3))

ds_h1_x1 = ds_h1_x1.assign_coords(lat=ds_h1_x2["lat"]) # -- this resulted in aligned dimensions for lat

# Compare original latitudes
lat_diff = ds_h1_x1["lat"].values != ds_h1_x2["lat"].values

if lat_diff.any():
    print("Differences found in latitude values:")
    for i, diff in enumerate(lat_diff):
        if diff:
            print(f"Index {i}: ds_h1_x1 = {ds_h1_x1['lat'].values[i]}, ds_h1_x2 = {ds_h1_x2['lat'].values[i]}")
else:
    print("No differences found in latitude values.")

# Compare original latitudes
lon_diff = ds_h1_x1["lon"].values != ds_h1_x2["lon"].values

if lon_diff.any():
    print("Differences found in longitude values:")
    for i, diff in enumerate(lon_diff):
        if diff:
            print(f"Index {i}: ds_h1_x1 = {ds_h1_x1['lon'].values[i]}, ds_h1_x2 = {ds_h1_x2['lon'].values[i]}")
else:
    print("No differences found in longitude values.")

# EXTRACT VARIABLES ---------------------------------------------------------------
#FEANSOLSRF_var = ds_h1_x2["FEANSOLSRF"]
#FEANSOLSRF_mean = FEANSOLSRF_var.squeeze(dim="time")

#FEDUSOLSRF_var = ds_h1_x2["FEDUSOLSRF"]
#FEDUSOLSRF_mean = FEDUSOLSRF_var.squeeze(dim="time")

#FEBBSOLSRF_var = ds_h1_x1["FEBBSOLSRF"]
#FEBBSOLSRF_mean = FEBBSOLSRF_var.mean(dim="time")

#FESOLSRF = FEANSOLSRF_var + FEDUSOLSRF_var + FEBBSOLSRF_var

#FEANTOTSRF_var = ds_h1_x2["FEANTOTSRF"]
#FEANTOTSRF_mean = FEANTOTSRF_var.squeeze(dim="time")

#FEDUTOTSRF_var = ds_h1_x2["FEDUTOTSRF"]
#FEDUTOTSRF_mean = FEDUTOTSRF_var.squeeze(dim="time")

#FEBBTOTSRF_var = ds_h1_x1["FEBBTOTSRF"]
#FEBBTOTSRF_mean = FEBBTOTSRF_var.mean(dim="time")

#FETOTSRF = FEANTOTSRF_var + FEDUTOTSRF_var + FEBBTOTSRF_var

#print(FESOLSRF.dims, FESOLSRF.shape)
#print(FETOTSRF.dims, FETOTSRF.shape)

# ADD ALL VARIABLES TO SINGLE DATASET --------------------------------------------
#all_FE_DEP = xr.Dataset({
 #  "FESOLSRF": FESOLSRF,
 #   "FEANSOLSRF": FEANSOLSRF_var,
 #   "FEBBSOLSRF": FEBBSOLSRF_var,
 #   "FEDUSOLSRF": FEDUSOLSRF_var,

 #   "FETOTSRF": FETOTSRF,
 #  "FEANTOTSRF": FEANTOTSRF_var,
 #   "FEBBTOTSRF": FEBBTOTSRF_var,
 #   "FEDUTOTSRF": FEDUTOTSRF_var,
 #   })

#print(all_FE_DEP)

FEANSOLDRY_var = ds_h1_x2["FEANSOLDRY"]
#FEANSOLDRY_mean = FEANSOLDRY_var.squeeze(dim="time")
FEANSOLWET_var = ds_h1_x2["FEANSOLWET"]
#FEANSOLWET_mean = FEANSOLWET_var.squeeze(dim="time")

FEDUSOLDRY_var = ds_h1_x2["FEDUSOLDRY"] 
#FEDUSOLDRY_mean = FEDUSOLDRY_var.squeeze(dim="time")
FEDUSOLWET_var = ds_h1_x2["FEDUSOLWET"] 
#FEDUSOLWET_mean = FEDUSOLWET_var.squeeze(dim="time")

FEBBSOLDRY_var = ds_h1_x1["FEBBSOLDRY"]
#FEBBSOLDRY_mean = FEBBSOLDRY_var.mean(dim="time")
FEBBSOLWET_var = ds_h1_x1["FEBBSOLWET"]
#FEBBSOLWET_mean = FEBBSOLWET_var.mean(dim="time")

FESOLDRY = FEANSOLDRY_var + FEDUSOLDRY_var + FEBBSOLDRY_var
FESOLWET = FEANSOLWET_var + FEDUSOLWET_var + FEBBSOLWET_var

FEANTOTDRY_var = ds_h1_x2["FEANTOTDRY"]
#FEANTOTDRY_mean = FEANTOTDRY_var.squeeze(dim="time")
FEANTOTWET_var = ds_h1_x2["FEANTOTWET"]
#FEANTOTWET_mean = FEANTOTWET_var.squeeze(dim="time")

FEDUTOTDRY_var = ds_h1_x2["FEDUTOTDRY"] 
#FEDUTOTDRY_mean = FEDUTOTDRY_var.squeeze(dim="time")
FEDUTOTWET_var = ds_h1_x2["FEDUTOTWET"] 
#FEDUTOTWET_mean = FEDUTOTWET_var.squeeze(dim="time")

FEBBTOTDRY_var = ds_h1_x1["FEBBTOTDRY"]
#FEBBTOTDRY_mean = FEBBTOTDRY_var.mean(dim="time")
FEBBTOTWET_var = ds_h1_x1["FEBBTOTWET"]
#FEBBTOTWET_mean = FEBBTOTWET_var.mean(dim="time")

FETOTDRY = FEANTOTDRY_var + FEDUTOTDRY_var + FEBBTOTDRY_var
FETOTWET = FEANTOTWET_var + FEDUTOTWET_var + FEBBTOTWET_var

# ADD ALL VARIABLES TO SINGLE DATASET --------------------------------------------
all_FE_DEP = xr.Dataset({
    "FESOLDRY": FESOLDRY,
    "FEANSOLDRY": FEANSOLDRY_var,
    "FEBBSOLDRY": FEBBSOLDRY_var,
    "FEDUSOLDRY": FEDUSOLDRY_var,
    
    "FESOLWET": FESOLWET,
    "FEANSOLWET": FEANSOLWET_var,
    "FEBBSOLWET": FEBBSOLWET_var,
    "FEDUSOLWET": FEDUSOLWET_var,

    "FETOTDRY": FETOTDRY,
    "FEANTOTDRY": FEANTOTDRY_var,
    "FEBBTOTDRY": FEBBTOTDRY_var,
    "FEDUTOTDRY": FEDUTOTDRY_var,

    "FETOTWET": FETOTWET,
    "FEANTOTWET": FEANTOTWET_var,
    "FEBBTOTWET": FEBBTOTWET_var,
    "FEDUTOTWET": FEDUTOTWET_var,
    })

# Save the updated dataset combining fire x1 and correct dust outputs to a new NetCDF file 
output_path = "D:\\CoalFlyAsh\\no_soil_state_firex1_runs\\no_soil_state_an+du_cheyenne_bb_derecho.h1_v1.nc"
all_FE_DEP.to_netcdf(output_path)

#%% COMPILING PD RUNS TO FIX DUST ISSUES 
import numpy as np
import xarray as xr
from netCDF4 import Dataset

# LOAD IN THE NETCDF FILE -----------------------------------------
# As of 2-20-25 this is being modified so that I can take PD dust from the PD files and add it to the FU files with mask multiplied through
# - need to update both the .h1 and .h2 files  
# PD 
sim_directory_1 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL0.2-WOOD10-OIL38.cam.h1.2009-2011_PD.nc"
ds_h1_x1 = xr.open_dataset(sim_directory_1)
#ds_h1_x1 = ds_h1_x1.assign_coords(lat=ds_h1_x1["lat"].round(3))

# FU
sim_directory_2 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI-SSP370-INDCOAL0.2-RESICOAL0.2-WOOD10-OIL38-FIRE33_midcentury.cam.h1.2009-2011.nc"
ds_h1_x2 = xr.open_dataset(sim_directory_2)
#ds_h1_x2 = ds_h1_x2.assign_coords(lat=ds_h1_x2["lat"].round(3))

# DUST MASK -- APPLY DIFFERENT YEARS FOR DIFFERENT FU SCENARIOS 
dust_mask_dir = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\dst_tunedeppastfuture_0.9x1.25_1850-2100.nc"
ds_h1_x3 = xr.open_dataset(dust_mask_dir)
#ds_h1_x3 = ds_h1_x3.sel(date=2090) # for end-century runs
ds_h1_x3 = ds_h1_x3.sel(date=2040) # for mid-century runs
#ds_h1_x3 = ds_h1_x3.assign_coords(lat=ds_h1_x3["lat"].round(3))

# Replace latitude in file with latitudes from actual model outputs per rounding differences
lat2 = ds_h1_x2['lat'].values  # Assuming 'lat' is the latitude variable
lat1 = ds_h1_x3['lat'].values
nearest_lat = np.array([lat2[np.abs(lat2 - l).argmin()] for l in lat1])
ds_h1_x3 = ds_h1_x3.assign_coords(lat=("lat", nearest_lat))

time_coord = ds_h1_x2["time"]
ds_h1_x3 = ds_h1_x3.rename({"date": "time"})
ds_h1_x3 = ds_h1_x3.expand_dims(time=time_coord)

# Compare original latitudes
lat_diff = ds_h1_x1["lat"].values != ds_h1_x3["lat"].values

if lat_diff.any():
    print("Differences found in latitude values:")
    for i, diff in enumerate(lat_diff):
        if diff:
            print(f"Index {i}: ds_h1_x1 = {ds_h1_x1['lat'].values[i]}, ds_h1_x2 = {ds_h1_x2['lat'].values[i]}")
else:
    print("No differences found in latitude values.")

# Compare original latitudes
lon_diff = ds_h1_x1["lon"].values != ds_h1_x2["lon"].values

if lon_diff.any():
    print("Differences found in longitude values:")
    for i, diff in enumerate(lon_diff):
        if diff:
            print(f"Index {i}: ds_h1_x1 = {ds_h1_x1['lon'].values[i]}, ds_h1_x2 = {ds_h1_x2['lon'].values[i]}")
else:
    print("No differences found in longitude values.")

# Replace the latitudes in ds_h1_x1 with those from ds_h1_x2
# attributed to some point float error with adding variables 
#ds_h1_x1 = ds_h1_x1.assign_coords(lat=ds_h1_x2["lat"]) # -- this resulted in aligned dimensions for lat

# EXTRACT VARIABLES ---------------------------------------------------------------
FEANSOLDRY_var = ds_h1_x2["FEANSOLDRY"]
#FEANSOLDRY_mean = FEANSOLDRY_var.squeeze(dim="time")
FEANSOLWET_var = ds_h1_x2["FEANSOLWET"]
#FEANSOLWET_mean = FEANSOLWET_var.squeeze(dim="time")

FEDUSOLDRY_var = ds_h1_x1["FEDUSOLDRY"] * ds_h1_x3["DUST"] # apply scaling factor for future dust deposition
#FEDUSOLDRY_mean = FEDUSOLDRY_var.squeeze(dim="time")
FEDUSOLWET_var = ds_h1_x1["FEDUSOLWET"] * ds_h1_x3["DUST"] # apply scaling factor for future dust deposition
#FEDUSOLWET_mean = FEDUSOLWET_var.squeeze(dim="time")

FEBBSOLDRY_var = ds_h1_x2["FEBBSOLDRY"]
#FEBBSOLDRY_mean = FEBBSOLDRY_var.mean(dim="time")
FEBBSOLWET_var = ds_h1_x2["FEBBSOLWET"]
#FEBBSOLWET_mean = FEBBSOLWET_var.mean(dim="time")

FESOLDRY = FEANSOLDRY_var + FEDUSOLDRY_var + FEBBSOLDRY_var
FESOLWET = FEANSOLWET_var + FEDUSOLWET_var + FEBBSOLWET_var

FEANTOTDRY_var = ds_h1_x2["FEANTOTDRY"]
#FEANTOTDRY_mean = FEANTOTDRY_var.squeeze(dim="time")
FEANTOTWET_var = ds_h1_x2["FEANTOTWET"]
#FEANTOTWET_mean = FEANTOTWET_var.squeeze(dim="time")

FEDUTOTDRY_var = ds_h1_x1["FEDUTOTDRY"] * ds_h1_x3["DUST"] # apply scaling factor for future dust deposition
#FEDUTOTDRY_mean = FEDUTOTDRY_var.squeeze(dim="time")
FEDUTOTWET_var = ds_h1_x1["FEDUTOTWET"] * ds_h1_x3["DUST"] # apply scaling factor for future dust deposition
#FEDUTOTWET_mean = FEDUTOTWET_var.squeeze(dim="time")

FEBBTOTDRY_var = ds_h1_x2["FEBBTOTDRY"]
#FEBBTOTDRY_mean = FEBBTOTDRY_var.mean(dim="time")
FEBBTOTWET_var = ds_h1_x2["FEBBTOTWET"]
#FEBBTOTWET_mean = FEBBTOTWET_var.mean(dim="time")

FETOTDRY = FEANTOTDRY_var + FEDUTOTDRY_var + FEBBTOTDRY_var
FETOTWET = FEANTOTWET_var + FEDUTOTWET_var + FEBBTOTWET_var

# ADD ALL VARIABLES TO SINGLE DATASET --------------------------------------------
all_FE_DEP = xr.Dataset({
    "FESOLDRY": FESOLDRY,
    "FEANSOLDRY": FEANSOLDRY_var,
    "FEBBSOLDRY": FEBBSOLDRY_var,
    "FEDUSOLDRY": FEDUSOLDRY_var,
    
    "FESOLWET": FESOLWET,
    "FEANSOLWET": FEANSOLWET_var,
    "FEBBSOLWET": FEBBSOLWET_var,
    "FEDUSOLWET": FEDUSOLWET_var,

    "FETOTDRY": FETOTDRY,
    "FEANTOTDRY": FEANTOTDRY_var,
    "FEBBTOTDRY": FEBBTOTDRY_var,
    "FEDUTOTDRY": FEDUTOTDRY_var,

    "FETOTWET": FETOTWET,
    "FEANTOTWET": FEANTOTWET_var,
    "FEBBTOTWET": FEBBTOTWET_var,
    "FEDUTOTWET": FEDUTOTWET_var,
    })

# Save the updated dataset combining fire x1 and correct dust outputs to a new NetCDF file 
output_path = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI-SSP370-INDCOAL0.2-RESICOAL0.2-WOOD10-OIL38-FIRE33_midcentury.cam.h1.2009-2011_scaledDUST.nc"
all_FE_DEP.to_netcdf(output_path)

# FUTURE DUST FLUX ADJUSTMENTS -- TEST PASSED WITH MANUAL CHECK IN PANOPLY