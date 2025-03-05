#%%
# Making Fire Emissions dataset file for Transient Run (converting surface emissions to vertical)

"""
Modified from
Vertical profile calculation SIMFIRE-BLAZE
Jessica Wan (jsw352)
13 August 2019
"""

# Import modules
# import netCDF4
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import cftime
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

# Import data files
# FILES ARE TOO BIG DO ONE CHEMICAL SPECIES AT A TIME 
# regridded all using cdo to match lat and lon using remapbil on hpc 
# surface emissions for each species 
# these are in months since 1750-01-01 starting with 1800 (representing 1850-01-01)
# BC
#sfc_in_BC = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Fire_Emissions\\LPJ-GUESS-BLAZE_BC_1850_onward_remapped.nc", decode_times=False)
# convert time to days since 1850-01-01
#time_var = sfc_in_BC["time"].values 
#base_time = cftime.DatetimeGregorian(1700, 1, 1)
#new_base_time = cftime.DatetimeGregorian(1850, 1, 1)
#time_dates = np.array([cftime.date2num(base_time.replace(year=1700 + int(t) // 12, month=(int(t) % 12) + 1, day=1),
                                      # units="days since 1850-01-01", calendar="gregorian") for t in time_var])
#sfc_in_BC["time"] = ("time", time_dates)
#sfc_in_BC["time"].attrs["units"] = "days since 1850-01-01"
#sfc_in_BC["time"].attrs["calendar"] = "gregorian"

# SO2
sfc_in_SO2 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Fire_Emissions\\LPJ-GUESS-BLAZE_SO2_1850_onward_remapped.nc", decode_times=False)
# convert time to days since 1850-01-01
time_var2 = sfc_in_SO2["time"].values
base_time2 = cftime.DatetimeGregorian(1700, 1, 1)
new_base_time2 = cftime.DatetimeGregorian(1850, 1, 1)
time_dates2 = np.array([cftime.date2num(base_time2.replace(year=1700 + int(t) // 12, month=(int(t) % 12) + 1, day=1),
                                       units="days since 1850-01-01", calendar="gregorian") for t in time_var2])
sfc_in_SO2["time"] = ("time", time_dates2)
sfc_in_SO2["time"].attrs["units"] = "days since 1850-01-01"
sfc_in_SO2["time"].attrs["calendar"] = "gregorian"
sfc_in_SO2["emissions.monthly"] = sfc_in_SO2["emissions.monthly"].astype("float32")

# OC
#sfc_in_OC = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Fire_Emissions\\LPJ-GUESS-BLAZE_OC_1850_onward_remapped.nc", decode_times=False)
# convert time to days since 1850-01-01
#time_var3 = sfc_in_OC["time"].values
#base_time3 = cftime.DatetimeGregorian(1700, 1, 1)
#new_base_time3 = cftime.DatetimeGregorian(1850, 1, 1)
#time_dates3 = np.array([cftime.date2num(base_time3.replace(year=1700 + int(t) // 12, month=(int(t) % 12) + 1, day=1),
                                      # units="days since 1850-01-01", calendar="gregorian") for t in time_var3])
#sfc_in_OC["time"] = ("time", time_dates3)
#sfc_in_OC["time"].attrs["units"] = "days since 1850-01-01"
#sfc_in_OC["time"].attrs["calendar"] = "gregorian"

# these were originally in days since 1750-01-01
# VERTICAL EMISSION LAYERS FILE
vert_in = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Fire_Emissions\\emissions-cmip6_bc_a4_bb_vertical_1750-2015_0.9x1.25_c20170322_remapped.nc", decode_times=False)

# convert to days since 1850-01-01
time_var4 = vert_in["time"].values
base_time_1750 = cftime.DatetimeGregorian(1750, 1, 1)
new_base_time_1850 = cftime.DatetimeGregorian(1850, 1, 1)
days_since_1850 = (new_base_time_1850 - base_time_1750).days
valid_time_indices = np.where(time_var4 >= days_since_1850)[0]
vert_in = vert_in.isel(time=valid_time_indices)
new_time_values = vert_in["time"].values - days_since_1850 - 15 # minus fifteen is so that monthly value is the first day of the month
vert_in["time"] = ("time", new_time_values)
vert_in["time"].attrs["units"] = "days since 1850-01-01"
vert_in["time"].attrs["calendar"] = "gregorian"
vert_in = vert_in.isel(time=slice(None, -24)) # removing 2014-2015 to match chemical species infile which ends in 2013
vert_in['time'] = vert_in['time'].astype(np.int64)

# TIMES ARE ALL ALIGNING NOW
# LAT AND LON ALIGNING AFTER REMAP
# SFC is in degrees east (-180 through 180) but CMIP6 files are in 0-360
# Latitudes are slightly different 
#print(sfc_in_BC["time"])
#print(sfc_in_OC["time"])
#print(sfc_in_SO2["time"])
#print(vert_in["time"])

# Retrieve variables from data files
sfc_lat_array = sfc_in_SO2.variables['lat'][:].copy()
sfc_lon_array = sfc_in_SO2.variables['lon'][:].copy()

# encoded as emissions.monthly and chemical species is denoted in name
# I have OC, BC, and SO2 currently in g m-2 s-1

#sfc_emiss_BC  = sfc_in_BC.variables['emissions.monthly'][:].copy()
sfc_emiss_SO2  = sfc_in_SO2.variables['emissions.monthly'][:].copy()
#sfc_emiss_OC  = sfc_in_OC.variables['emissions.monthly'][:].copy()

vert_altitude_array = vert_in.variables['altitude'][:].copy()
vert_emiss_array = vert_in.variables['emiss_bb'][:].copy()
vert_lat_array = vert_in.variables['lat'][:].copy()
vert_lon_array = vert_in.variables['lon'][:].copy()

# Set fractional distribution of emissions heights for wild fires by biome (Dentener et al., 2006) -----------------------------------------------------
# make empty dataframe with correct dimensions filled with 1.0 -- # Correct for any emissions outside of defined biomes to be surface emissions
vert_emiss_frac_array = vert_in.copy()
dims = vert_emiss_frac_array.dims  # Get dataset dimensions
shape = tuple(vert_emiss_frac_array.sizes[dim] for dim in dims)  # Get size of each dimension
vert_emiss_frac_array["vert_emiss_array"] = xr.DataArray(np.ones(shape), dims=dims)

# now to replace filler with fractional contribution of emissions at altitude by biome
tropical_alt = [0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #30S-30N
temperate_alt = [0.4, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #30N-45N; 30S-60S
boreal_eur_alt = [0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0] #45N-90N; 0-180E
boreal_can_alt = [0.2, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] #45N-90N; 180W-0

# Tropical (30S-30N)
tropical_lat = (sfc_lat_array >= -30.0) & (sfc_lat_array <= 30.0)
tropical_lon = (sfc_lon_array >= 0.0) & (sfc_lon_array <= 360.0)

# Temperate (30N-45N; 30S-60S)
temperate_lat = ((sfc_lat_array > 30.0) & (sfc_lat_array <= 45.0)) | \
                          ((sfc_lat_array >= -60.0) & (sfc_lat_array < -30.0))
temperate_lon = (sfc_lon_array >= 0.0) & (sfc_lon_array <= 360.0)

# Boreal Europe (45N-90N; 0-180E)
boreal_eur_lat = (sfc_lat_array > 45.0) & (sfc_lat_array <= 90.0)
boreal_eur_lon = (sfc_lon_array >= 0.0) & (sfc_lon_array <= 180.0)

# Boreal Canada (45N-90N; 180W-0)
boreal_can_lat = (sfc_lat_array > 45.0) & (sfc_lat_array <= 90.0)
boreal_can_lon = (sfc_lon_array > 180.0) & (sfc_lon_array <= 360.0)

# Convert altitude lists to NumPy arrays
tropical_alt = np.array(tropical_alt)
temperate_alt = np.array(temperate_alt)
boreal_eur_alt = np.array(boreal_eur_alt)
boreal_can_alt = np.array(boreal_can_alt)

# Convert the emission fraction array to a NumPy array
vert_emiss_array = vert_emiss_frac_array["vert_emiss_array"].values  # Shape: (time, lon, lat, alt)

# Apply altitude fractions along the correct axis
for i in range(10):  # 10 altitude levels
    vert_emiss_array[:, :, tropical_lat, i] = tropical_alt[i]
    vert_emiss_array[:, :, temperate_lat, i] = temperate_alt[i]
    vert_emiss_array[:, :180, boreal_eur_lat, i] = boreal_eur_alt[i]  # 0-180E
    vert_emiss_array[:, 180:, boreal_can_lat, i] = boreal_can_alt[i]  # 180-360E

# Store back into xarray Dataset
vert_emiss_frac_array["vert_emiss_array"].values = vert_emiss_array

# passed test, vertical layers are as assigned by biome -- now to multiply through emissions
#vert_emiss_frac_array.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Fire_Emissions\\vert_emiss_frac_array.nc")
# file is stored in vert_emiss_frac_array

# convert emissions from BLAZE in g/m2/s to molecules/km2/s --------------------------------------------------------------------------
vert_dist = 0.5 #km
avogadro = 6.022e23 #molecules
bc_mw = 12.011 #g
so2_mw = 64.066 #g
oc_mw = 12.011 #g # ASK DOUGLAS ABOUT THE MW OF OC TO USE 

#sfc_BC_emiss_mol_km2_s =  sfc_in_BC['emissions.monthly']  * (1.0e6) * avogadro / bc_mw  #convert g/m2/s to molecules/km2/s
#sfc_BC_emiss_mol_km2_s = sfc_BC_emiss_mol_km2_s.expand_dims({"altitude": vert_emiss_frac_array.coords["altitude"]})
#sfc_BC_emiss_mol_km2_s = sfc_BC_emiss_mol_km2_s.T.transpose('time','lon','lat','altitude')

sfc_SO2_emiss_mol_km2_s = sfc_in_SO2['emissions.monthly'] * (1.0e6) * avogadro / so2_mw #convert g/m2/s to molecules/km2/s
sfc_SO2_emiss_mol_km2_s = sfc_SO2_emiss_mol_km2_s.expand_dims({"altitude": vert_emiss_frac_array.coords["altitude"]})
sfc_SO2_emiss_mol_km2_s = sfc_SO2_emiss_mol_km2_s.T.transpose('time','lon','lat','altitude')

#sfc_OC_emiss_mol_km2_s = sfc_in_OC['emissions.monthly'] * (1.0e6) * avogadro / oc_mw #convert g/m2/s to molecules/km2/s
#sfc_OC_emiss_mol_km2_s = sfc_OC_emiss_mol_km2_s.expand_dims({"altitude": vert_emiss_frac_array.coords["altitude"]})
#sfc_OC_emiss_mol_km2_s = sfc_OC_emiss_mol_km2_s.T.transpose('time','lon','lat','altitude')

# passed test data looks as it should -- just a lot of NaNs over ocean where there are no emissions
# and current emissions are the SAME in each altitude layer 
# sfc_BC_emiss_mol_km2_s.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Fire_Emissions\\sfc_BC_emiss_mol_km2_s.nc")

# Apply vertical emission mask to the emissions for each fire species
#vert_BC_dentener = vert_emiss_frac_array.copy()
#vert_BC_dentener["emiss_bb"] = vert_BC_dentener["vert_emiss_array"] * sfc_BC_emiss_mol_km2_s
#vert_BC_dentener["emiss_bb"] = vert_BC_dentener["emiss_bb"] / vert_dist # convert from km2 to km3
#vert_BC_dentener["emiss_bb"] = vert_BC_dentener["emiss_bb"] / (1.0e15) # convert from km3 to cm3

vert_SO2_dentener = vert_emiss_frac_array.copy()
vert_SO2_dentener["emiss_bb"] = vert_SO2_dentener["vert_emiss_array"] * sfc_SO2_emiss_mol_km2_s
vert_SO2_dentener["emiss_bb"] = vert_SO2_dentener["emiss_bb"] / vert_dist # convert from km2 to km3
vert_SO2_dentener["emiss_bb"] = vert_SO2_dentener["emiss_bb"] / (1.0e15) # convert from km3 to cm3

#vert_OC_dentener = vert_emiss_frac_array.copy()
#vert_OC_dentener["emiss_bb"] = vert_OC_dentener["vert_emiss_array"] * sfc_OC_emiss_mol_km2_s
#vert_OC_dentener["emiss_bb"] = vert_OC_dentener["emiss_bb"] / vert_dist # convert from km2 to km3
#vert_OC_dentener["emiss_bb"] = vert_OC_dentener["emiss_bb"] / (1.0e15) # convert from km3 to cm3

# altitude interfaces got removed during remap so add those back -------------------------------------------------------------------------------------
vert_in_alt_in = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Fire_Emissions\\emissions-cmip6_bc_a4_bb_vertical_1750-2015_0.9x1.25_c20170322.nc", decode_times=False)
#vert_BC_dentener['altitude_int'] = vert_in_alt_in['altitude_int']
#vert_BC_dentener['altitude_int'].attrs = vert_in_alt_in['altitude_int'].attrs

vert_SO2_dentener['altitude_int'] = vert_in_alt_in['altitude_int']
vert_SO2_dentener['altitude_int'].attrs = vert_in_alt_in['altitude_int'].attrs

#vert_OC_dentener['altitude_int'] = vert_in_alt_in['altitude_int']
#vert_OC_dentener['altitude_int'].attrs = vert_in_alt_in['altitude_int'].attrs

# pass my new file through original vertical file so that the dimensions line up accurately --------------------------------------------------
vertical_file_with_alt_int = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Fire_Emissions\\emissions-cmip6_bc_a4_bb_vertical_1750-2015_0.9x1.25_c20170322.nc", decode_times=False)
vertical_file = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Fire_Emissions\\emissions-cmip6_bc_a4_bb_vertical_1750-2015_0.9x1.25_c20170322_remapped.nc", decode_times=False)

vertical_file['altitude_int'] = vertical_file_with_alt_int['altitude_int']
vertical_file['altitude_int'].attrs = vertical_file_with_alt_int['altitude_int'].attrs

# convert to days since 1850-01-01
time_var4 = vertical_file["time"].values
base_time_1750 = cftime.DatetimeGregorian(1750, 1, 1)
new_base_time_1850 = cftime.DatetimeGregorian(1850, 1, 1)
days_since_1850 = (new_base_time_1850 - base_time_1750).days
valid_time_indices = np.where(time_var4 >= days_since_1850)[0]
vertical_file = vertical_file.isel(time=valid_time_indices)
new_time_values = vertical_file["time"].values - days_since_1850 - 15 # minus fifteen is so that monthly value is the first day of the month
vertical_file["time"] = ("time", new_time_values)
vertical_file["time"].attrs["units"] = "days since 1850-01-01"
vertical_file["time"].attrs["calendar"] = "gregorian"
vertical_file = vertical_file.isel(time=slice(None, -24)) # removing 2014-2015 to match chemical species infile which ends in 2013
vertical_file['time'] = vertical_file['time'].astype(np.int64)

vertical_file = vertical_file.drop_vars("emiss_bb")

#BC_ds = vert_BC_dentener.copy()
#vertical_file["emiss_bb"] = BC_ds["emiss_bb"]

SO2_ds = vert_SO2_dentener.copy()
vertical_file["emiss_bb"] = SO2_ds["emiss_bb"]

#%%
vertical_file["emiss_bb"].attrs["units"] = "molecules/cm3/s" 
vertical_file["emiss_bb"].attrs["long_name"] = 'SO2 biomass burning emissions distributed vertically' 
vertical_file["emiss_bb"].attrs["molecular_weight"] = float(64) # float(12) # CHANGE FOR EACH CHEMICAL SPECIES
vertical_file["emiss_bb"].attrs["cell_methods"] = "time: mean"
vertical_file["time"].attrs["units"] = "days since 1850-01-01 00:00:00"

vertical_file.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Fire_Emissions\\LPJ-GUESS-BLAZE_SO2_bb_vertical_1850-2013_0.9x1.25_heplaas_03052025.nc")

#%% ---------------------------------------------------------------------------------------------------------------------------------
import numpy as np

# Convert emiss to number 
vertical_num = vertical_file.copy()

# Define values for calculations
diameter = 0.134e-6 #m
rho = 1700. #g/m3 # What is RHO for SO2
mw = 64.066 # 12.011 #g
pi = np.pi

# Calculate number emissions
mass_per_particle = rho * (pi/6) * (diameter**3) #g/particle
vertical_num["num_so2_bb"] = (vertical_num["emiss_bb"] * mw) / mass_per_particle

vertical_num = vertical_num.drop_vars("emiss_bb")
vertical_file["emiss_bb"].attrs["units"] = '(particles/cm2/s)(molecules/mole)(g/kg)'
vertical_file["emiss_bb"].attrs["long_name"] = 'particle number emissions of num_so2_bb'
vertical_file["emiss_bb"].attrs["molecular_weight"] = float(64)

vertical_file.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Fire_Emissions\\LPJ-GUESS-BLAZE_SO2_num)bb_vertical_1850-2013_0.9x1.25_heplaas_03052025.nc")

