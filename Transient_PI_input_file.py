#%% CALCULATING SECTOR SPECIFIC Fe EMISSIONS FROM CMIP BLACK CARBON EMISSION DATASETS  ----------------------
import xarray as xr
import pandas as pd
import numpy as np

# Load the NetCDF file -- Fe emissions PI 
ds_Fe_PI = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\ClaquinMineralsCAM6-LL_SPEW_BGC_Feb2020-SDR_PI_bf_remapcon_0.9x1.25.nc")

time = pd.date_range("1850-01-01", "1850-12-31", freq="MS")  # Monthly start date
time = time + pd.Timedelta(days=14) # to match middle of month as PD files are 

# Convert datetime to days since 1850-01-01 00:00:00
time_days = (time - pd.Timestamp("1850-01-01")).days

# Assign the time dimension with the required attributes
ds_Fe_PI = ds_Fe_PI.assign_coords(
    time=("time", time_days))

# alternate time options --------------
# simplest way to conserve timestamps for middle of each month as seen in Douglas' files
#ds_Fe_PI = ds_Fe_PI.assign_coords(time=("time", time))
# If converting to 'days since 1850-01-01' format (number of days from start date)
#time_days = (time - pd.Timestamp("1850-01-01")).days
#ds_Fe_PI = ds_Fe_PI.assign_coords(time=("time", time_days))
# -------------------------------------

# Add the time attributes
ds_Fe_PI["time"].attrs = {
    "standard_name": "time",
    "long_name": "time",
    "units": "days since 1850-01-01 00:00:00",
    "calendar": "gregorian",
    "axis": "T"
}

# a1 = Accumulation, a2 = Aitken, a3 = Coarse -- split by fine / coarse and in soil_erod file, Aitken will be separated
ds_Fe_PI = ds_Fe_PI.assign(
    # FOTHER indicates a combination of fossil fuel, wood, oil, and smelting emissions (do not exist in PI) -- need dummy variable for these with values of 0.0 with float point precision of 32 
    # instead of FOTHER -- we are going through with separating by fuel type per application of regional emission mask
    sol_FeComb_FF_fine =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    sol_FeComb_FF_coa =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    insol_FeComb_FF_fine =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    insol_FeComb_FF_coa =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),

    sol_FeComb_OIL_fine =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    sol_FeComb_OIL_coa =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    insol_FeComb_OIL_fine =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    insol_FeComb_OIL_coa =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),

    sol_FeComb_WOOD_fine =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    sol_FeComb_WOOD_coa =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    insol_FeComb_WOOD_fine =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    insol_FeComb_WOOD_coa =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),

    sol_FeComb_SMELT_fine =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    sol_FeComb_SMELT_coa =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    insol_FeComb_SMELT_fine =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
    insol_FeComb_SMELT_coa =(["time", "lat", "lon"], np.full((12, len(ds_Fe_PI["lat"]), len(ds_Fe_PI["lon"])), np.float32(0.0))),
)
 
# Split BF emissions into soluble and insoluble fractions based on mingin paper
ds_Fe_PI["sol_FeComb_BF_fine"] = ds_Fe_PI["FFBFf"]*.33
ds_Fe_PI["sol_FeComb_BF_coa"] = ds_Fe_PI["FFBFc"]*.33

ds_Fe_PI["insol_FeComb_BF_fine"] = ds_Fe_PI["FFBFf"]*.67
ds_Fe_PI["insol_FeComb_BF_coa"] = ds_Fe_PI["FFBFc"]*.67

# passed test to ensure the split conserved all of the mass --------- 
#BF_a1 = ds_Fe_PI["sol_FeComb_BF_coa"].mean()
#BF_a2 = ds_Fe_PI["insol_FeComb_BF_coa"].mean()
#tot_fine = BF_a1 + BF_a2
#BFf = ds_Fe_PI["FFBFc"].mean()
#print("added", tot_fine)
#print("BF", BFf)
# -------------------------------------------------------------------
# adding biofuels to the PI file 
ds_Fe_PI = ds_Fe_PI.assign(
    sol_FeComb_BF_fine =(["time", "lat", "lon"], np.tile(ds_Fe_PI["sol_FeComb_BF_fine"].values, (12, 1, 1))),
    sol_FeComb_BF_coa =(["time", "lat", "lon"], np.tile(ds_Fe_PI["sol_FeComb_BF_coa"].values, (12, 1, 1))),
    insol_FeComb_BF_fine =(["time", "lat", "lon"], np.tile(ds_Fe_PI["insol_FeComb_BF_fine"].values, (12, 1, 1))),
    insol_FeComb_BF_coa =(["time", "lat", "lon"], np.tile(ds_Fe_PI["insol_FeComb_BF_coa"].values, (12, 1, 1)))
)

variables = ["time", 
             "lat", 
             "lon", 
             "sol_FeComb_BF_fine", 
             "sol_FeComb_BF_coa", 
             "sol_FeComb_FF_fine", 
             "sol_FeComb_FF_coa",
             "sol_FeComb_OIL_fine", 
             "sol_FeComb_OIL_coa",
             "sol_FeComb_WOOD_fine", 
             "sol_FeComb_WOOD_coa",
             "sol_FeComb_SMELT_fine", 
             "sol_FeComb_SMELT_coa",
             "insol_FeComb_BF_fine", 
             "insol_FeComb_BF_coa", 
             "insol_FeComb_FF_fine", 
             "insol_FeComb_FF_coa",
             "insol_FeComb_OIL_fine", 
             "insol_FeComb_OIL_coa",
             "insol_FeComb_WOOD_fine", 
             "insol_FeComb_WOOD_coa",
             "insol_FeComb_SMELT_fine", 
             "insol_FeComb_SMELT_coa"
             ]

ds_Fe_PI = ds_Fe_PI[variables]

#print(ds_Fe_PI)
#ds_Fe_PI.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\test_PI_file.nc")

# Reading in the PD SPEWFUELS and splitting into 12 time points (months) ---------------------------------------------------------
ds_Fe_PD2 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\Collaborations\Coal Fly Ash\\data\\ClaquinMineralsCAM6_SPEWFUELS-Dec2023_OCNFRAC_remapcon_regridded.nc")

new_times = np.array(["2010-" + str(i).zfill(2) for i in range(1, 13)], dtype="datetime64[ns]")
ds_Fe_PD_expanded = ds_Fe_PD2.sel(time=ds_Fe_PD2.time.values[0]).expand_dims(time=new_times, axis=0)

ds_Fe_PD_expanded = ds_Fe_PD_expanded[[var for var in ds_Fe_PD_expanded.data_vars if "Med" in var]]

time = pd.date_range("2010-01-01", "2010-12-31", freq="MS")  # Monthly start date
time = time + pd.Timedelta(days=14) # to match middle of month as PD files are 

# Convert datetime to days since 1850-01-01 00:00:00 -- so that I can eventually merge with the PI file 
time_days = (time - pd.Timestamp("1850-01-01")).days

# Assign the time dimension with the required attributes
ds_Fe_PD_expanded = ds_Fe_PD_expanded.assign_coords(
    time=("time", time_days))

# Add the time attributes
ds_Fe_PD_expanded["time"].attrs = {
    "standard_name": "time",
    "long_name": "time",
    "units": "days since 1850-01-01 00:00:00",
    "calendar": "gregorian",
    "axis": "T"
}

#print(ds_Fe_PD_expanded["time"])

# splitting fuel types into soluble and insoluble fractions for each, based on mingjin best practice run
ds_Fe_PD_expanded["sol_FeComb_BF_fine"] = ds_Fe_PD_expanded["FineMed_FCOALBF"]*.33
ds_Fe_PD_expanded["sol_FeComb_BF_coa"] = ds_Fe_PD_expanded["CoaMed_FCOALBF"]*.33
ds_Fe_PD_expanded["insol_FeComb_BF_fine"] = ds_Fe_PD_expanded["FineMed_FCOALBF"]*.67
ds_Fe_PD_expanded["insol_FeComb_BF_coa"] = ds_Fe_PD_expanded["CoaMed_FCOALBF"]*.67

ds_Fe_PD_expanded["sol_FeComb_FF_fine"] = ds_Fe_PD_expanded["FineMed_FCOALFF"]*.002
ds_Fe_PD_expanded["sol_FeComb_FF_coa"] = ds_Fe_PD_expanded["CoaMed_FCOALFF"]*.002
ds_Fe_PD_expanded["insol_FeComb_FF_fine"] = ds_Fe_PD_expanded["FineMed_FCOALFF"]*.998
ds_Fe_PD_expanded["insol_FeComb_FF_coa"] = ds_Fe_PD_expanded["CoaMed_FCOALFF"]*.998

ds_Fe_PD_expanded["sol_FeComb_OIL_fine"] = ds_Fe_PD_expanded["FineMed_FOIL"]*.38
ds_Fe_PD_expanded["sol_FeComb_OIL_coa"] = ds_Fe_PD_expanded["CoaMed_FOIL"]*.38
ds_Fe_PD_expanded["insol_FeComb_OIL_fine"] = ds_Fe_PD_expanded["FineMed_FOIL"]*.62
ds_Fe_PD_expanded["insol_FeComb_OIL_coa"] = ds_Fe_PD_expanded["CoaMed_FOIL"]*.62

ds_Fe_PD_expanded["sol_FeComb_WOOD_fine"] = ds_Fe_PD_expanded["FineMed_FWOOD"]*.56
ds_Fe_PD_expanded["sol_FeComb_WOOD_coa"] = ds_Fe_PD_expanded["CoaMed_FWOOD"]*.56
ds_Fe_PD_expanded["insol_FeComb_WOOD_fine"] = ds_Fe_PD_expanded["FineMed_FWOOD"]*.44
ds_Fe_PD_expanded["insol_FeComb_WOOD_coa"] = ds_Fe_PD_expanded["CoaMed_FWOOD"]*.44

ds_Fe_PD_expanded["sol_FeComb_SMELT_fine"] = ds_Fe_PD_expanded["FineMed_FSMELT"]*.00003
ds_Fe_PD_expanded["sol_FeComb_SMELT_coa"] = ds_Fe_PD_expanded["CoaMed_FSMELT"]*.00003
ds_Fe_PD_expanded["insol_FeComb_SMELT_fine"] = ds_Fe_PD_expanded["FineMed_FSMELT"]*.99997
ds_Fe_PD_expanded["insol_FeComb_SMELT_coa"] = ds_Fe_PD_expanded["CoaMed_FSMELT"]*.99997

variables2 = ["time", 
             "lat", 
             "lon", 
             "sol_FeComb_BF_fine", 
             "sol_FeComb_BF_coa", 
             "sol_FeComb_FF_fine", 
             "sol_FeComb_FF_coa",
             "sol_FeComb_OIL_fine", 
             "sol_FeComb_OIL_coa",
             "sol_FeComb_WOOD_fine", 
             "sol_FeComb_WOOD_coa",
             "sol_FeComb_SMELT_fine", 
             "sol_FeComb_SMELT_coa",
             "insol_FeComb_BF_fine", 
             "insol_FeComb_BF_coa", 
             "insol_FeComb_FF_fine", 
             "insol_FeComb_FF_coa",
             "insol_FeComb_OIL_fine", 
             "insol_FeComb_OIL_coa",
             "insol_FeComb_WOOD_fine", 
             "insol_FeComb_WOOD_coa",
             "insol_FeComb_SMELT_fine", 
             "insol_FeComb_SMELT_coa"
             ]

ds_Fe_PD = ds_Fe_PD_expanded[variables2]

#print(ds_Fe_PD)
# Reading in Douglas' PD transient run input emission files -- use to determine seasonality to apply to SPEWFUELS data PI and PD ----------------------------------------
ds_emiss_fesca1 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\emiss_fesca1_SPEW_1979-2015_0.9x1.25.nc")
num_time_steps = ds_emiss_fesca1.dims["time"]  # Number of time steps in the dataset
new_time_values = pd.date_range(start="1979-01-01", periods=num_time_steps, freq="MS") + pd.Timedelta(days=14) 
ref_date = pd.Timestamp("1979-01-01")
time_int64 = (new_time_values - ref_date).days.astype("int64")
ds_emiss_fesca1 = ds_emiss_fesca1.assign_coords(time=("time", time_int64))
ds_emiss_fesca1["time"].attrs = {
    "standard_name": "time",
   "long_name": "time",
    "units": "days since 1979-01-01 00:00:00",
    "calendar": "gregorian",
    "axis": "T"}
ds_emiss_fesca1 = ds_emiss_fesca1.drop_vars("date")

# passed manual check for changes to time 
#ds_emiss_fesca1.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\test_emiss_fesca1_file.nc")

ds_emiss_fesca2 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\emiss_fesca2_SPEW_1979-2015_0.9x1.25.nc")
num_time_steps = ds_emiss_fesca2.dims["time"]  # Number of time steps in the dataset
new_time_values = pd.date_range(start="1979-01-01", periods=num_time_steps, freq="MS") + pd.Timedelta(days=14) 
ref_date = pd.Timestamp("1979-01-01")
time_int64 = (new_time_values - ref_date).days.astype("int64")
ds_emiss_fesca2 = ds_emiss_fesca2.assign_coords(time=("time", time_int64))
ds_emiss_fesca2["time"].attrs = {
    "standard_name": "time",
    "long_name": "time",
    "units": "days since 1979-01-01 00:00:00",
    "calendar": "gregorian",
    "axis": "T"}
ds_emiss_fesca2 = ds_emiss_fesca2.drop_vars("date")

ds_emiss_fesca3 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\emiss_fesca3_SPEW_1979-2015_0.9x1.25.nc")
num_time_steps = ds_emiss_fesca3.dims["time"]  # Number of time steps in the dataset
new_time_values = pd.date_range(start="1979-01-01", periods=num_time_steps, freq="MS") + pd.Timedelta(days=14) 
ref_date = pd.Timestamp("1979-01-01")
time_int64 = (new_time_values - ref_date).days.astype("int64")
ds_emiss_fesca3 = ds_emiss_fesca3.assign_coords(time=("time", time_int64))
ds_emiss_fesca3["time"].attrs = {
    "standard_name": "time",
    "long_name": "time",
    "units": "days since 1979-01-01 00:00:00",
    "calendar": "gregorian",
    "axis": "T"}
ds_emiss_fesca3 = ds_emiss_fesca3.drop_vars("date")

# now for insoluble iron -- which is labeled t but long description says insoluble
ds_emiss_fetca1 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\emiss_fetca1_SPEW_1979-2015_0.9x1.25.nc")
num_time_steps = ds_emiss_fetca1.dims["time"]  # Number of time steps in the dataset
new_time_values = pd.date_range(start="1979-01-01", periods=num_time_steps, freq="MS") + pd.Timedelta(days=14) 
ref_date = pd.Timestamp("1979-01-01")
time_int64 = (new_time_values - ref_date).days.astype("int64")
ds_emiss_fetca1 = ds_emiss_fetca1.assign_coords(time=("time", time_int64))
ds_emiss_fetca1["time"].attrs = {
    "standard_name": "time",
    "long_name": "time",
    "units": "days since 1979-01-01 00:00:00",
    "calendar": "gregorian",
    "axis": "T"}
ds_emiss_fetca1 = ds_emiss_fetca1.drop_vars("date")

ds_emiss_fetca2 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\emiss_fetca2_SPEW_1979-2015_0.9x1.25.nc")
num_time_steps = ds_emiss_fetca2.dims["time"]  # Number of time steps in the dataset
new_time_values = pd.date_range(start="1979-01-01", periods=num_time_steps, freq="MS") + pd.Timedelta(days=14) 
ref_date = pd.Timestamp("1979-01-01")
time_int64 = (new_time_values - ref_date).days.astype("int64")
ds_emiss_fetca2 = ds_emiss_fetca2.assign_coords(time=("time", time_int64))
ds_emiss_fetca2["time"].attrs = {
    "standard_name": "time",
    "long_name": "time",
    "units": "days since 1979-01-01 00:00:00",
    "calendar": "gregorian",
    "axis": "T"}
ds_emiss_fetca2 = ds_emiss_fetca2.drop_vars("date")

ds_emiss_fetca3 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\emiss_fetca3_SPEW_1979-2015_0.9x1.25.nc")
num_time_steps = ds_emiss_fetca3.dims["time"]  # Number of time steps in the dataset
new_time_values = pd.date_range(start="1979-01-01", periods=num_time_steps, freq="MS") + pd.Timedelta(days=14) 
ref_date = pd.Timestamp("1979-01-01")
time_int64 = (new_time_values - ref_date).days.astype("int64")
ds_emiss_fetca3 = ds_emiss_fetca3.assign_coords(time=("time", time_int64))
ds_emiss_fetca3["time"].attrs = {
    "standard_name": "time",
    "long_name": "time",
    "units": "days since 1979-01-01 00:00:00",
    "calendar": "gregorian",
    "axis": "T"}
ds_emiss_fetca3 = ds_emiss_fetca3.drop_vars("date")

# combining data to make variables to apply through 
sol_FeComb_fine = ds_emiss_fesca2['emiss_fesca2'] + ds_emiss_fesca1['emiss_fesca1'] # Add Aitken and Accumulation modes to make fine aerosol 
sol_FeComb_coa = ds_emiss_fesca3['emiss_fesca3'] # - ds_emiss_fesca3['sol_FeComb_BF_a3'] (adjust if t is actually total and not insoluble)
insol_FeComb_fine = ds_emiss_fetca2['emiss_fetca2'] + ds_emiss_fetca1['emiss_fetca1'] # Add Aitken and Accumulation modes to make fine aerosol 
insol_FeComb_coa = ds_emiss_fetca3['emiss_fetca3'] # - ds_emiss_fesca3['sol_FeComb_BF_a3'] (adjust if t is actually total and not insoluble)

#print("sol fine", sol_FeComb_fine)
#print("insol fine", insol_FeComb_fine)

# repeating seasonality but stored as names of fuel type variables 
sol_FeComb_BF_fine = sol_FeComb_fine.copy()
sol_FeComb_BF_coa = sol_FeComb_coa.copy()
sol_FeComb_FF_fine = sol_FeComb_fine.copy()
sol_FeComb_FF_coa = sol_FeComb_coa.copy()
sol_FeComb_OIL_fine = sol_FeComb_fine.copy()
sol_FeComb_OIL_coa = sol_FeComb_coa.copy()
sol_FeComb_WOOD_fine = sol_FeComb_fine.copy()
sol_FeComb_WOOD_coa = sol_FeComb_coa.copy()
sol_FeComb_SMELT_fine = sol_FeComb_fine.copy()
sol_FeComb_SMELT_coa = sol_FeComb_coa.copy()

insol_FeComb_BF_fine = insol_FeComb_fine.copy()
insol_FeComb_BF_coa = insol_FeComb_coa.copy()
insol_FeComb_FF_fine = insol_FeComb_fine.copy()
insol_FeComb_FF_coa = insol_FeComb_coa.copy()
insol_FeComb_OIL_fine = insol_FeComb_fine.copy()
insol_FeComb_OIL_coa = insol_FeComb_coa.copy()
insol_FeComb_WOOD_fine = insol_FeComb_fine.copy()
insol_FeComb_WOOD_coa = insol_FeComb_coa.copy()
insol_FeComb_SMELT_fine = insol_FeComb_fine.copy()
insol_FeComb_SMELT_coa = insol_FeComb_coa.copy()

# passed test to ensure the split conserved all of the mass, only small differences <1 --------- 
#BF_a1 = sol_FeComb_BF_coa.mean()
#BF_a2 = sol_FeComb_OIL_coa.mean()
#BF_a3 = sol_FeComb_FF_coa.mean()
#BF_a4 = sol_FeComb_WOOD_coa.mean()
#BF_a5 = sol_FeComb_SMELT_coa.mean()
#tot_fine = BF_a1 + BF_a2 + BF_a3 + BF_a4 + BF_a5
#BFf = sol_FeComb_coa.mean()
#print("added", tot_fine)
#print("BF", BFf)

# -------------------------------------------------------------------------------------------
# Now we are ready to find monthly deviation from averages in the PD files and apply this to PI emissions (only biofuel will be affected because all others will start at 0)
# compiling PD variables into one xarray
all_emiss_PD = xr.Dataset({
    "sol_FeComb_BF_fine": sol_FeComb_BF_fine, 
    "sol_FeComb_BF_coa": sol_FeComb_BF_coa, 
    "sol_FeComb_FF_fine":sol_FeComb_FF_fine, 
    "sol_FeComb_FF_coa":sol_FeComb_FF_coa,
    "sol_FeComb_OIL_fine":sol_FeComb_OIL_fine, 
    "sol_FeComb_OIL_coa":sol_FeComb_OIL_coa,
    "sol_FeComb_WOOD_fine":sol_FeComb_WOOD_fine, 
    "sol_FeComb_WOOD_coa":sol_FeComb_WOOD_coa,
    "sol_FeComb_SMELT_fine":sol_FeComb_SMELT_fine, 
    "sol_FeComb_SMELT_coa":sol_FeComb_SMELT_coa,
    "insol_FeComb_BF_fine":insol_FeComb_BF_fine, 
    "insol_FeComb_BF_coa":insol_FeComb_BF_coa, 
    "insol_FeComb_FF_fine":insol_FeComb_FF_fine, 
    "insol_FeComb_FF_coa":insol_FeComb_FF_coa,
    "insol_FeComb_OIL_fine":insol_FeComb_OIL_fine,
    "insol_FeComb_OIL_coa":insol_FeComb_OIL_coa,
    "insol_FeComb_WOOD_fine":insol_FeComb_WOOD_fine, 
    "insol_FeComb_WOOD_coa":insol_FeComb_WOOD_coa,
    "insol_FeComb_SMELT_fine":insol_FeComb_SMELT_fine, 
    "insol_FeComb_SMELT_coa":insol_FeComb_SMELT_coa
    })

# Convert the time coordinate (in days) back to a pd.DatetimeIndex
ref_date = pd.Timestamp("1979-01-01")  # Adjust if your reference date differs
all_emiss_PD = all_emiss_PD.assign_coords(time=ref_date + pd.to_timedelta(all_emiss_PD["time"].values, unit="D"))
# Compute monthly averages for each variable
all_emiss_PD_monthly_avgs = all_emiss_PD.groupby("time.month").mean(dim="time")

# passed manual check for changes to time 
#monthly_avg_fesc_BF_fine.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\test_emiss_monthly.nc")

# determine monthly deviations from average (seasonality) by grid cell and create mask to apply to PI data 
monthly_avgs = {month: all_emiss_PD_monthly_avgs.sel(month=month) for month in range(1, 13)}
annual_avg = all_emiss_PD_monthly_avgs.mean(dim="month")

#print("annual", annual_avg)
#print("december", monthly_avgs[12])

# ratio of monthly average to annual average (individual grid cells)
seasonality_mask = {}
for month in range(1, 13):  # Loop over months from 1 to 12
    seasonality_mask[f"month_{month}"] = all_emiss_PD_monthly_avgs.sel(month=month) / annual_avg
    seasonality_mask[f"month_{month}"] = seasonality_mask[f"month_{month}"].expand_dims(month=1)  # Add 'month' dimension

# filling NaNs with 1's so if data was missing in transient input dataset but not missing in SPEW it will be conserved
for key in seasonality_mask:
    seasonality_mask[key] = seasonality_mask[key].fillna(1.0).where(np.isfinite(seasonality_mask[key]), 1.0)

#print(seasonality_mask)

# APPLY SEASONALITY MASK TO PI SPEW FUELS ---------------------------------------------------------
# calculating monthly deviation from mean for PI files by applying seasonality mask
ds_Fe_PI_monthly = ds_Fe_PI.copy()
ref_date = pd.Timestamp("1850-01-01")  # Adjust if your reference date differs
ds_Fe_PI_monthly = ds_Fe_PI_monthly.assign_coords(time=ref_date + pd.to_timedelta(ds_Fe_PI_monthly["time"].values, unit="D"))
ds_Fe_PI_monthly = ds_Fe_PI_monthly.groupby("time.month").mean(dim="time")

ds_Fe_PI_jan = ds_Fe_PI_monthly.sel(month=1).copy()
ds_Fe_PI_feb = ds_Fe_PI_monthly.sel(month=2).copy()
ds_Fe_PI_mar = ds_Fe_PI_monthly.sel(month=3).copy()
ds_Fe_PI_apr = ds_Fe_PI_monthly.sel(month=4).copy()
ds_Fe_PI_may = ds_Fe_PI_monthly.sel(month=5).copy()
ds_Fe_PI_jun = ds_Fe_PI_monthly.sel(month=6).copy()
ds_Fe_PI_jul = ds_Fe_PI_monthly.sel(month=7).copy()
ds_Fe_PI_aug = ds_Fe_PI_monthly.sel(month=8).copy()
ds_Fe_PI_sep = ds_Fe_PI_monthly.sel(month=9).copy()
ds_Fe_PI_oct = ds_Fe_PI_monthly.sel(month=10).copy()
ds_Fe_PI_nov = ds_Fe_PI_monthly.sel(month=11).copy()
ds_Fe_PI_dec = ds_Fe_PI_monthly.sel(month=12).copy()

ds_Fe_PI_jan_szn = ds_Fe_PI_jan * seasonality_mask["month_1"]
ds_Fe_PI_feb_szn = ds_Fe_PI_feb * seasonality_mask["month_2"]
ds_Fe_PI_mar_szn = ds_Fe_PI_mar * seasonality_mask["month_3"]
ds_Fe_PI_apr_szn = ds_Fe_PI_apr * seasonality_mask["month_4"]
ds_Fe_PI_may_szn = ds_Fe_PI_may * seasonality_mask["month_5"]
ds_Fe_PI_jun_szn = ds_Fe_PI_jun * seasonality_mask["month_6"]
ds_Fe_PI_jul_szn = ds_Fe_PI_jul * seasonality_mask["month_7"]
ds_Fe_PI_aug_szn = ds_Fe_PI_aug * seasonality_mask["month_8"]
ds_Fe_PI_sep_szn = ds_Fe_PI_sep * seasonality_mask["month_9"]
ds_Fe_PI_oct_szn = ds_Fe_PI_oct * seasonality_mask["month_10"]
ds_Fe_PI_nov_szn = ds_Fe_PI_nov * seasonality_mask["month_11"]
ds_Fe_PI_dec_szn = ds_Fe_PI_dec * seasonality_mask["month_12"]

#print("before", ds_Fe_PI_jan)
#print("seasonal", ds_Fe_PI_jan_szn)

# MANUAL CHECK THAT SEASONALITY MASK WAS APPLIED CORRECTLY 
# manual checks in Panoply revealed that the mask was applied correctly
#ds_Fe_PI_jan.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\ds_Fe_PI_jan.nc")
#ds_Fe_PI_jan_szn.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\ds_Fe_PI_jan_szn.nc")
#jan_mask = seasonality_mask["month_1"]
#jan_mask.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\jan_mask.nc")

# SEASONALITY MASK APPLIED TO PD SPEWFUELS FILE --------------------------------------------------------
ds_Fe_PD_monthly = ds_Fe_PD.copy()
ref_date = pd.Timestamp("1850-01-01")  # Adjust if your reference date differs
ds_Fe_PD_monthly = ds_Fe_PD_monthly.assign_coords(time=ref_date + pd.to_timedelta(ds_Fe_PD_monthly["time"].values, unit="D"))
ds_Fe_PD_monthly = ds_Fe_PD_monthly.groupby("time.month").mean(dim="time")

ds_Fe_PD_jan = ds_Fe_PD_monthly.sel(month=1).copy()
ds_Fe_PD_feb = ds_Fe_PD_monthly.sel(month=2).copy()
ds_Fe_PD_mar = ds_Fe_PD_monthly.sel(month=3).copy()
ds_Fe_PD_apr = ds_Fe_PD_monthly.sel(month=4).copy()
ds_Fe_PD_may = ds_Fe_PD_monthly.sel(month=5).copy()
ds_Fe_PD_jun = ds_Fe_PD_monthly.sel(month=6).copy()
ds_Fe_PD_jul = ds_Fe_PD_monthly.sel(month=7).copy()
ds_Fe_PD_aug = ds_Fe_PD_monthly.sel(month=8).copy()
ds_Fe_PD_sep = ds_Fe_PD_monthly.sel(month=9).copy()
ds_Fe_PD_oct = ds_Fe_PD_monthly.sel(month=10).copy()
ds_Fe_PD_nov = ds_Fe_PD_monthly.sel(month=11).copy()
ds_Fe_PD_dec = ds_Fe_PD_monthly.sel(month=12).copy()

ds_Fe_PD_jan_szn = ds_Fe_PD_jan * seasonality_mask["month_1"]
ds_Fe_PD_feb_szn = ds_Fe_PD_feb * seasonality_mask["month_2"]
ds_Fe_PD_mar_szn = ds_Fe_PD_mar * seasonality_mask["month_3"]
ds_Fe_PD_apr_szn = ds_Fe_PD_apr * seasonality_mask["month_4"]
ds_Fe_PD_may_szn = ds_Fe_PD_may * seasonality_mask["month_5"]
ds_Fe_PD_jun_szn = ds_Fe_PD_jun * seasonality_mask["month_6"]
ds_Fe_PD_jul_szn = ds_Fe_PD_jul * seasonality_mask["month_7"]
ds_Fe_PD_aug_szn = ds_Fe_PD_aug * seasonality_mask["month_8"]
ds_Fe_PD_sep_szn = ds_Fe_PD_sep * seasonality_mask["month_9"]
ds_Fe_PD_oct_szn = ds_Fe_PD_oct * seasonality_mask["month_10"]
ds_Fe_PD_nov_szn = ds_Fe_PD_nov * seasonality_mask["month_11"]
ds_Fe_PD_dec_szn = ds_Fe_PD_dec * seasonality_mask["month_12"]

#print("before", ds_Fe_PD_jan)
#print("seasonal", ds_Fe_PD_jan_szn)
# # COMBINING ALL IRON EMISSIONS BACK INTO ONE FILE WITH SEASONALITY APPLIED --------------------------------------
# use xrconcat to combine the months back into a single array with month as a coordinate
# List of datasets for each month
PI_Fe_monthly_averages_list = [
    ds_Fe_PI_jan_szn, ds_Fe_PI_feb_szn, ds_Fe_PI_mar_szn, ds_Fe_PI_apr_szn,
    ds_Fe_PI_may_szn, ds_Fe_PI_jun_szn, ds_Fe_PI_jul_szn, ds_Fe_PI_aug_szn,
    ds_Fe_PI_sep_szn, ds_Fe_PI_oct_szn, ds_Fe_PI_nov_szn, ds_Fe_PI_dec_szn
]

# Concatenate along a new 'month' dimension
PI_Fe_monthly_averages = xr.concat(PI_Fe_monthly_averages_list, dim="month")
# Assign month values (1-12)
PI_Fe_monthly_averages["month"] = range(1, 13)
# and replace NaNs with 0.0 --  I dont think I need this anymore since I replaced them with 1.0 in seasonality mask 
# PI_Fe_monthly_averages = PI_Fe_monthly_averages.fillna(0.0).where(np.isfinite(PI_Fe_monthly_averages), 0.0)

# now go back to date time that matches the input file to combine PI and PD files 
# two file names are currently all_emiss_PD and PI_Fe_monthly_averages
#print("PD", all_emiss_PD)
#print("PI", PI_Fe_monthly_averages)

#print("PD", all_emiss_PD.time)
#print(all_emiss_PD.time.values)

# PD is currently at 444 time points -- the 15th of each month starting at 1979-01-15
# PI is at 12 time points listed as months, need this to be time again starting at 1850-01-01

time = pd.date_range("1850-01-01", "1850-12-31", freq="MS") + pd.Timedelta(days=14)
# Ensure time remains as datetime64[ns] and assign it as coordinates
PI_Fe_monthly_averages = PI_Fe_monthly_averages.assign_coords(time=("month", time))

#print("PD", all_emiss_PD.time)
#print("PI", PI_Fe_monthly_averages.time)

# Swap 'month' dimension with 'time' -- then drop month as a coordinate
PI_Fe_monthly_averages = PI_Fe_monthly_averages.swap_dims({"month": "time"})
PI_Fe_monthly_averages = PI_Fe_monthly_averages.drop_vars("month", errors="ignore")

#print(PI_Fe_monthly_averages)
#print(PI_Fe_monthly_averages["time"])

PD_Fe_monthly_averages_list = [
    ds_Fe_PD_jan_szn, ds_Fe_PD_feb_szn, ds_Fe_PD_mar_szn, ds_Fe_PD_apr_szn,
    ds_Fe_PD_may_szn, ds_Fe_PD_jun_szn, ds_Fe_PD_jul_szn, ds_Fe_PD_aug_szn,
    ds_Fe_PD_sep_szn, ds_Fe_PD_oct_szn, ds_Fe_PD_nov_szn, ds_Fe_PD_dec_szn
]

PD_Fe_monthly_averages = xr.concat(PD_Fe_monthly_averages_list, dim="month")
PD_Fe_monthly_averages["month"] = range(1, 13)
time = pd.date_range("2010-01-01", "2010-12-31", freq="MS") + pd.Timedelta(days=14)
PD_Fe_monthly_averages = PD_Fe_monthly_averages.assign_coords(time=("month", time))
PD_Fe_monthly_averages = PD_Fe_monthly_averages.swap_dims({"month": "time"})
PD_Fe_monthly_averages = PD_Fe_monthly_averages.drop_vars("month", errors="ignore")

#print(PD_Fe_monthly_averages)
#print(PD_Fe_monthly_averages["time"])
#PI_Fe_monthly_averages.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\PI_Fe_monthly_averages.nc")
#PD_Fe_monthly_averages.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\PD_Fe_monthly_averages.nc")
# these look good, no emissions for ocean other than PD oil
# combining PD and PI Fe emissions datasets -------------------------------------------------------------------------------
combined_PI_PD_Fe_Comb_emiss = xr.concat([PI_Fe_monthly_averages, PD_Fe_monthly_averages], dim="time")

#print(combined_PI_PD_Fe_Comb_emiss["time"])
#print(combined_PI_PD_Fe_Comb_emiss)

#combined_PI_PD_Fe_Comb_emiss.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\combined_PIPD_test.nc")
# looks right when read out

# Building Regional Emission Correction Factor Mask based on Rathod et al., 2024 ------------------------------------------
# Ocean mask will be necessary so that over open shipping emissions are not altered 
ocn_frac_ds = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\Collaborations\Coal Fly Ash\\data\\ClaquinMineralsCAM6_SPEWFUELS-Dec2023_OCNFRAC_remapcon_regridded.nc") 
ocn_frac_ds = ocn_frac_ds.drop_vars("time")
ocn_frac_ds = ocn_frac_ds.squeeze(dim='time') 
ocnfrac = ocn_frac_ds['OCNFRAC'] 
shipping_mask = ocn_frac_ds['OCNFRAC'] > 0.9
lon_indices = ocn_frac_ds['lon'].where(shipping_mask)
lat_indices = ocn_frac_ds['lat'].where(shipping_mask)
shipping_emission_indices = list(zip(lon_indices.values.flatten(), lat_indices.values.flatten()))
shipping_emission_indices = [(lon, lat) for lon, lat in shipping_emission_indices if not (np.isnan(lon) or np.isnan(lat))]

# Defining Regional Boundaries (mostly separated by continents)
# South America
SAM_condition = (ocn_frac_ds['lon'] >= 279.0) & (ocn_frac_ds['lon'] <= 326.0) & \
                (ocn_frac_ds['lat'] >= -56.0) & (ocn_frac_ds['lat'] <= 12.0)
SAM_conditions_land_mask = SAM_condition.where((SAM_condition == True) & (ocnfrac < 0.5), False)

# North America other than USA
NAM_condition = ( # MEXICO / CENTRAL AM
                ((ocn_frac_ds['lon'] >= 190.0) & (ocn_frac_ds['lon'] <= 310.0) & \
                (ocn_frac_ds['lat'] > 12.0) & (ocn_frac_ds['lat'] < 30.0))
                | # CANADA
                ((ocn_frac_ds['lon'] >= 190.0) & (ocn_frac_ds['lon'] <= 310.0) & \
                (ocn_frac_ds['lat'] > 49.0) & (ocn_frac_ds['lat'] < 60.0))
                )       
NAM_conditions_land_mask = NAM_condition.where((NAM_condition == True) & (ocnfrac < 0.5), False)

USA_condition = (ocn_frac_ds['lon'] >= 190.0) & (ocn_frac_ds['lon'] <= 310.0) & \
                (ocn_frac_ds['lat'] >= 30.0) & (ocn_frac_ds['lat'] <= 49.0)
USA_conditions_land_mask = USA_condition.where((USA_condition == True) & (ocnfrac < 0.5), False)

# Africa -- when straddling prime meridian must code this way 
AFR_condition = (
                ((ocn_frac_ds['lon'] >= 340) & (ocn_frac_ds['lat'] < 37.0) & (ocn_frac_ds['lat'] >= 0.0) | \
                (ocn_frac_ds['lon'] <= 35) & (ocn_frac_ds['lat'] < 37.0) & (ocn_frac_ds['lat'] >= 0.0))
                |
                ((ocn_frac_ds['lon'] >= 340) & (ocn_frac_ds['lat'] < 15.0) & (ocn_frac_ds['lat'] >= 0.0) | \
                (ocn_frac_ds['lon'] <= 50) & (ocn_frac_ds['lat'] < 15.0) & (ocn_frac_ds['lat'] >= 0.0))
                )
AFR_conditions_land_mask = AFR_condition.where((AFR_condition == True) & (ocnfrac < 0.5), False) 

# Southern portions of Africa < Equator
SAFR_condition = (ocn_frac_ds['lon'] >= 340) & (ocn_frac_ds['lat'] < 0.0) & (ocn_frac_ds['lat'] >= -35.0) | \
                (ocn_frac_ds['lon'] <= 50) & (ocn_frac_ds['lat'] < 0.0) & (ocn_frac_ds['lat'] >= -35.0)
SAFR_conditions_land_mask = SAFR_condition.where((SAFR_condition == True) & (ocnfrac < 0.5), False) 

# Europe 
EUR_condition = (ocn_frac_ds['lon'] >= 335) & (ocn_frac_ds['lat'] < 60.0) & (ocn_frac_ds['lat'] >= 37.0) | \
                (ocn_frac_ds['lon'] <= 35) & (ocn_frac_ds['lat'] < 60.0) & (ocn_frac_ds['lat'] >= 37.0)
EUR_conditions_land_mask = EUR_condition.where((EUR_condition == True) & (ocnfrac < 0.5), False) 

# Northern/ Eastern Asia/China, Himalayas separate the air mass movement so I need to separate finer around them
CHINA_condition = (
                  ((ocn_frac_ds['lon'] < 180) & (ocn_frac_ds['lon'] > 70.0) & \
                  (ocn_frac_ds['lat'] < 55.0) & (ocn_frac_ds['lat'] >= 30))
                  | 
                  ((ocn_frac_ds['lon'] < 180) & (ocn_frac_ds['lon'] > 85.0) & \
                  (ocn_frac_ds['lat'] < 30.0) & (ocn_frac_ds['lat'] >= 22.0))
                  )   
CHINA_conditions_land_mask = CHINA_condition.where((CHINA_condition == True) & (ocnfrac < 0.5), False)

# Southeastern Asia and India
SEAS_condition =  (
                  ((ocn_frac_ds['lon'] < 180) & (ocn_frac_ds['lon'] > 55.0) & \
                  (ocn_frac_ds['lat'] < 30) & (ocn_frac_ds['lat'] >= 22.0))
                  | 
                  ((ocn_frac_ds['lon'] < 180) & (ocn_frac_ds['lon'] > 55.0) & \
                  (ocn_frac_ds['lat'] < 22.0) & (ocn_frac_ds['lat'] >= -10.0))
                  )
SEAS_conditions_land_mask = SEAS_condition.where((SEAS_condition == True) & (ocnfrac < 0.5), False)

# Australia/South Pacific
AUS_condition = (ocn_frac_ds['lon'] <= 180) & (ocn_frac_ds['lon'] >= 110.0) & \
               (ocn_frac_ds['lat'] < -10.0) & (ocn_frac_ds['lat'] > -50.0)
AUS_conditions_land_mask = AUS_condition.where((AUS_condition == True) & (ocnfrac < 0.5), False)

# Southern Hemisphere
SH_condition = (ocn_frac_ds['lon'] >= 0.0) & (ocn_frac_ds['lon'] <= 360.0) & \
               (ocn_frac_ds['lat'] < 0.0) & (ocn_frac_ds['lat'] >= -90.0)
SH_conditions_land_mask = SH_condition.where((SH_condition == True) & (ocnfrac < 0.5), False)

# checked what the coverage was for each and it looks good -- entries are currently just zero 
#AUS_conditions_land_mask.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\AUS_conditions_land_mask.nc")

# creating blank dataset with scaling factors of 1 before supplying regional conditions
regional_scaling_factor = xr.DataArray(
    1.0,  # The value to assign to each cell
    dims=combined_PI_PD_Fe_Comb_emiss.dims,
    coords=combined_PI_PD_Fe_Comb_emiss.coords,  
    name='regional_scaling_factor'  
)

# THIS APPROACH WORKED TO APPLY REGIONAL SCALING FACTORS 
# FOR AUSTRALIA
AUS_cf_mask = AUS_conditions_land_mask.where(AUS_conditions_land_mask != 1)
stacked = AUS_cf_mask.stack(cells=("lat", "lon")) # flatten dimension
nonzero_coords = stacked.where(stacked, drop=True)
AUS_indices = list(zip(nonzero_coords["lat"].values, nonzero_coords["lon"].values)) # extract indices
for lat, lon in AUS_indices:
    regional_scaling_factor.loc[{"lat": lat, "lon": lon}] = 2.81 # CF from Rathod et al., 2024

SEAS_cf_mask = SEAS_conditions_land_mask.where(SEAS_conditions_land_mask != 1)
stacked2 = SEAS_cf_mask.stack(cells=("lat", "lon")) # flatten dimension
nonzero_coords2 = stacked2.where(stacked2, drop=True)
ASIA_indices = list(zip(nonzero_coords2["lat"].values, nonzero_coords2["lon"].values)) # extract indices
for lat, lon in ASIA_indices:
    regional_scaling_factor.loc[{"lat": lat, "lon": lon}] = 1.02 # CF from Rathod et al., 2024

CHINA_cf_mask = CHINA_conditions_land_mask.where(CHINA_conditions_land_mask != 1)
stacked3 = CHINA_cf_mask.stack(cells=("lat", "lon")) # flatten dimension
nonzero_coords3 = stacked3.where(stacked3, drop=True)
CHINA_indices = list(zip(nonzero_coords3["lat"].values, nonzero_coords3["lon"].values)) # extract indices
for lat, lon in CHINA_indices:
    regional_scaling_factor.loc[{"lat": lat, "lon": lon}] = 0.5 # CF from Bunnell et al., 2025

EUR_cf_mask = EUR_conditions_land_mask.where(EUR_conditions_land_mask != 1)
stacked4 = EUR_cf_mask.stack(cells=("lat", "lon")) # flatten dimension
nonzero_coords4 = stacked4.where(stacked4, drop=True)
EUR_indices = list(zip(nonzero_coords4["lat"].values, nonzero_coords4["lon"].values)) # extract indices
for lat, lon in EUR_indices:
    regional_scaling_factor.loc[{"lat": lat, "lon": lon}] = 0.87 # CF from Rathod et al., 2024

AFR_cf_mask = AFR_conditions_land_mask.where(AFR_conditions_land_mask != 1)
stacked5 = AFR_cf_mask.stack(cells=("lat", "lon")) # flatten dimension
nonzero_coords5 = stacked5.where(stacked5, drop=True)
AFR_indices = list(zip(nonzero_coords5["lat"].values, nonzero_coords5["lon"].values)) # extract indices
for lat, lon in AFR_indices:
    regional_scaling_factor.loc[{"lat": lat, "lon": lon}] = 0.63 # CF from Rathod et al., 2024

SAFR_cf_mask = SAFR_conditions_land_mask.where(SAFR_conditions_land_mask != 1)
stacked6 = SAFR_cf_mask.stack(cells=("lat", "lon")) # flatten dimension
nonzero_coords6 = stacked6.where(stacked6, drop=True)
SAFR_indices = list(zip(nonzero_coords6["lat"].values, nonzero_coords6["lon"].values)) # extract indices
for lat, lon in SAFR_indices:
    regional_scaling_factor.loc[{"lat": lat, "lon": lon}] = 5.0 # CF from Liu et al., 2022
# this factor of 5 needs to be applied for only South Africa 
# Ito paper (2023?) Southern Ocean paper -- didnt catch a correction but also found some increased input from northern countries in South America

NAM_cf_mask = NAM_conditions_land_mask.where(NAM_conditions_land_mask != 1)
stacked7 = NAM_cf_mask.stack(cells=("lat", "lon")) # flatten dimension
nonzero_coords7 = stacked7.where(stacked7, drop=True)
NAM_indices = list(zip(nonzero_coords7["lat"].values, nonzero_coords7["lon"].values)) # extract indices
for lat, lon in NAM_indices:
    regional_scaling_factor.loc[{"lat": lat, "lon": lon}] = 1.3 #1.22 # CF from Rathod et al., 2024 -- is reported as 1.3 and 1.22
# Using PMF 0.45 for USA -- double check in the text for USA vs N. Am. Read this paper in full today 
# Douglas will check with Sagar 

USA_cf_mask = USA_conditions_land_mask.where(USA_conditions_land_mask != 1)
stacked8 = USA_cf_mask.stack(cells=("lat", "lon")) # flatten dimension
nonzero_coords8 = stacked8.where(stacked8, drop=True)
USA_indices = list(zip(nonzero_coords8["lat"].values, nonzero_coords8["lon"].values)) # extract indices
for lat, lon in USA_indices:
    regional_scaling_factor.loc[{"lat": lat, "lon": lon}] = 0.45 # CF from Rathod et al., 2024 for USA specific using PMF (positive matrix factorization to source apportion)

SAM_cf_mask = SAM_conditions_land_mask.where(SAM_conditions_land_mask != 1)
stacked9 = SAM_cf_mask.stack(cells=("lat", "lon")) # flatten dimension
nonzero_coords9 = stacked9.where(stacked9, drop=True)
SAM_indices = list(zip(nonzero_coords9["lat"].values, nonzero_coords9["lon"].values)) # extract indices
for lat, lon in SAM_indices:
    regional_scaling_factor.loc[{"lat": lat, "lon": lon}] = 0.65 # CF from Rathod et al., 2024

#regional_scaling_factor.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\regional_scaling_factor.nc")

# PLOTTING MASK FOR VISUAL REFERENCE
# Plotting the regional emission mask for anthropogenic iron 
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# Load the NetCDF file
#ds2 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\regional_scaling_factor.nc") # Replace with your NetCDF file path
regional_scaling_factor2 = regional_scaling_factor.isel(time=0)

# Replace 'variable_name', 'lat', and 'lon' with actual names from your dataset
variable = regional_scaling_factor2
lat = regional_scaling_factor2['lat'].copy()
lon = regional_scaling_factor2['lon'].copy()

# Create a custom colormap (blue → white → red)
cmap = LinearSegmentedColormap.from_list('custom_cmap', [
    (0.0, 'blue'),   # Value 0 → Blue = negative correction
    (0.16, '#CDE4FA'),   #
    (0.2, 'white'),  # 1 → White
    (0.25, '#FED2D2'), # Red = positive correction 
    (1.0, '#950606')     # Max (5) 
])

# Set up the plot with Robinson projection
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Handle longitude range (if needed)
if np.any(lon > 180):
    lon = ((lon + 180) % 360) - 180  # Convert 0-360 to -180 to 180

# Plot the data
# Adjust transform based on your data's coordinate system
plot = ax.pcolormesh(
    lon, lat, variable, 
    transform=ccrs.PlateCarree(), 
    cmap=cmap, 
    vmin=0, vmax=5  # Scale from 0 to 5
  #  norm=LogNorm(vmin=0.01, vmax=5) 
)

# Add colorbar
cbar = plt.colorbar(plot, orientation='horizontal', pad=0.05, shrink=0.8)
cbar.set_label('Fe Emission Scaling Factor')

# Display the plot
plt.title('Regional Anthropogenic Iron Emission Mask')
plt.show()

# APPLYING REGIONAL MASK TO FE EMISSIONS 
#print("scaling factor", regional_scaling_factor)
#print("emissions", combined_PI_PD_Fe_Comb_emiss)

scaling_factors_notime = regional_scaling_factor.isel(time=0)

SAFR_indices2 = list(zip(nonzero_coords6["lon"].values, nonzero_coords6["lat"].values)) # extract indices
AUS_indices2 = list(zip(nonzero_coords["lon"].values, nonzero_coords["lat"].values)) # extract indices
SAM_indices2 = list(zip(nonzero_coords9["lon"].values, nonzero_coords9["lat"].values)) # extract indices

# Loop through each variable in the emission dataset and multiply through by regional scaling factor 
#for var in combined_PI_PD_Fe_Comb_emiss.data_vars:
  #  if "sol" in var:  # Check if 'sol' is in the variable name
  #      combined_PI_PD_Fe_Comb_emiss[var] = combined_PI_PD_Fe_Comb_emiss[var] * scaling_factors_notime

# Adjusted loop with conditions (skip regional adjustments over ocean and for wood/biofuels in southern half Africa)
# Loop through each variable in the dataset
for var in combined_PI_PD_Fe_Comb_emiss.data_vars:
    if "sol" in var:
        # Start with a mask that applies scaling everywhere
        mask = xr.ones_like(combined_PI_PD_Fe_Comb_emiss[var], dtype=bool)

        # Skip scaling where 'BF OR WOOD' is in the variable name and matches southern_africa_indices
       # if "BF" in var or "WOOD" in var:
       #     bf_wood_mask = xr.zeros_like(mask, dtype=bool)
       #     for lon_idx, lat_idx in SAFR_indices2:
       #         bf_wood_mask.loc[dict(lon=lon_idx, lat=lat_idx)] = True
       #     mask = mask & ~bf_wood_mask  # Exclude these indices

        # Skip scaling where 'BF OR WOOD' is in the variable name (Globally)
        #if "BF" in var or "WOOD" in var:
           # continue  # Skip scaling for these variables

        # Skip scaling where 'BF OR WOOD' is in the variable name and matches southern_hemisphere_indices
        if "BF" in var or "WOOD" in var:
            bf_wood_mask = xr.zeros_like(mask, dtype=bool)
            for indices in [SAFR_indices2, SAM_indices2, AUS_indices2]:
                for lon_idx, lat_idx in indices:
                    bf_wood_mask.loc[dict(lon=lon_idx, lat=lat_idx)] = True
            mask = mask & ~bf_wood_mask  # Exclude these indices

        # Apply scaling only where the mask is True
        combined_PI_PD_Fe_Comb_emiss[var] = combined_PI_PD_Fe_Comb_emiss[var].where(~mask, combined_PI_PD_Fe_Comb_emiss[var] * scaling_factors_notime)

# passed original manual test in Panoply of applying correction factors so far -- with no exceptions applied yet 
# this finally passed the after NOT changing the WOOD and BF emissions in the southern part of Africa 

# CONVERTING IRON EMISSIONS TO UNITS THEY NEED TO BE IN FOR INPUT DIRECTLY INTO THE MODEL 
# Option 1: convert PD emiss_file to kg/m2/s
# -- eventually will also need to convert to num file units 
# ------------ where ----- cm2 to m2 - Avogadro# - MW Fe - g to kg -----------------
# all_emiss_PD = all_emiss_PD * 10E4 * (1/6.023E23) * 55.854 * 10E-3

# recombining fuel types into single anthropogenic combustion emission values for Fe and splitting fine into accumulation and aitken modes 
combined_PI_PD_Fe_Comb_emiss["sol_FeComb_a1"] = (combined_PI_PD_Fe_Comb_emiss['sol_FeComb_BF_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_FF_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_OIL_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_WOOD_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_SMELT_fine'])*.9

combined_PI_PD_Fe_Comb_emiss["sol_FeComb_a2"] = (combined_PI_PD_Fe_Comb_emiss['sol_FeComb_BF_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_FF_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_OIL_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_WOOD_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_SMELT_fine'])*.1

combined_PI_PD_Fe_Comb_emiss["sol_FeComb_a3"] = (combined_PI_PD_Fe_Comb_emiss['sol_FeComb_BF_coa'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_FF_coa'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_OIL_coa'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_WOOD_coa'] + 
                                                combined_PI_PD_Fe_Comb_emiss['sol_FeComb_SMELT_coa'])

combined_PI_PD_Fe_Comb_emiss["insol_FeComb_a1"] = (combined_PI_PD_Fe_Comb_emiss['insol_FeComb_BF_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_FF_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_OIL_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_WOOD_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_SMELT_fine'])*.9

combined_PI_PD_Fe_Comb_emiss["insol_FeComb_a2"] = (combined_PI_PD_Fe_Comb_emiss['insol_FeComb_BF_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_FF_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_OIL_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_WOOD_fine'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_SMELT_fine'])*.1

combined_PI_PD_Fe_Comb_emiss["insol_FeComb_a3"] = (combined_PI_PD_Fe_Comb_emiss['insol_FeComb_BF_coa'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_FF_coa'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_OIL_coa'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_WOOD_coa'] + 
                                                combined_PI_PD_Fe_Comb_emiss['insol_FeComb_SMELT_coa'])

nonfueltype = [
    "lat",
    "lon",
    "time",
    "sol_FeComb_a1",
    "sol_FeComb_a2",
    "sol_FeComb_a3",
    "insol_FeComb_a1",
    "insol_FeComb_a2",
    "insol_FeComb_a3"
]

combined_PI_PD_Fe_Comb_emiss_ax = combined_PI_PD_Fe_Comb_emiss[nonfueltype]

print(combined_PI_PD_Fe_Comb_emiss_ax)

# seems like I also need a comprehensive SPEWFUELS file with this data in it according to user_nl_cam from Douglas' most recent transient run
# add the variables with the names that match both alternative combustion emission variable names in older SPEWFUELS files in case they are needed to be read into the soil_erod_model

soil_erod_ds = combined_PI_PD_Fe_Comb_emiss.copy()

soil_erod_ds["Fine_InsolFe_Comb"] = combined_PI_PD_Fe_Comb_emiss["insol_FeComb_a1"] + combined_PI_PD_Fe_Comb_emiss["insol_FeComb_a2"]
soil_erod_ds["Coar_InsolFe_Comb"] = combined_PI_PD_Fe_Comb_emiss["insol_FeComb_a3"] 
soil_erod_ds["Fine_SolFe_Comb"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_a1"] + combined_PI_PD_Fe_Comb_emiss["sol_FeComb_a2"]
soil_erod_ds["Coar_SolFe_Comb"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_a3"] 

soil_erod_ds["CoaMed_FCOALBF"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_BF_coa"] +combined_PI_PD_Fe_Comb_emiss["insol_FeComb_BF_coa"]
soil_erod_ds["FineMed_FCOALBF"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_BF_fine"] +combined_PI_PD_Fe_Comb_emiss["insol_FeComb_BF_fine"]
soil_erod_ds["CoaMed_FCOALFF"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_FF_coa"] +combined_PI_PD_Fe_Comb_emiss["insol_FeComb_FF_coa"]
soil_erod_ds["FineMed_FCOALFF"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_FF_fine"] +combined_PI_PD_Fe_Comb_emiss["insol_FeComb_FF_fine"]
soil_erod_ds["CoaMed_FOIL"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_OIL_coa"] +combined_PI_PD_Fe_Comb_emiss["insol_FeComb_OIL_coa"]
soil_erod_ds["FineMed_FOIL"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_OIL_fine"] +combined_PI_PD_Fe_Comb_emiss["insol_FeComb_OIL_fine"]
soil_erod_ds["CoaMed_FWOOD"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_WOOD_coa"] +combined_PI_PD_Fe_Comb_emiss["insol_FeComb_WOOD_coa"]
soil_erod_ds["FineMed_FWOOD"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_WOOD_fine"] +combined_PI_PD_Fe_Comb_emiss["insol_FeComb_WOOD_fine"]
soil_erod_ds["CoaMed_FSMELT"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_SMELT_coa"] +combined_PI_PD_Fe_Comb_emiss["insol_FeComb_SMELT_coa"]
soil_erod_ds["FineMed_FSMELT"] = combined_PI_PD_Fe_Comb_emiss["sol_FeComb_SMELT_fine"] +combined_PI_PD_Fe_Comb_emiss["insol_FeComb_SMELT_fine"]

print(soil_erod_ds)
#soil_erod_ds.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\SPEWFUEL_emissions_1850-2010_CF_no_BF_in_SouthernHemi_5.nc")

# convert SPEWFUELS emissions in (kg/m2/s) to (molecules/cm2/s) to make emiss file
# ------------ where ------------------------- m2 to cm2 - Avogadro# - MW Fe - kg to g -----------------
combined_PI_PD_Fe_Comb_emiss_molecules = combined_PI_PD_Fe_Comb_emiss_ax * 1E-4 * 6.023E23 * (1/55.845) * 1E3

combined_PI_PD_Fe_Comb_emiss_molecules["emiss_fesca1"] = combined_PI_PD_Fe_Comb_emiss_molecules["sol_FeComb_a1"]
combined_PI_PD_Fe_Comb_emiss_molecules["emiss_fesca2"] = combined_PI_PD_Fe_Comb_emiss_molecules["sol_FeComb_a2"]
combined_PI_PD_Fe_Comb_emiss_molecules["emiss_fesca3"] = combined_PI_PD_Fe_Comb_emiss_molecules["sol_FeComb_a3"]
combined_PI_PD_Fe_Comb_emiss_molecules["emiss_fetca1"] = combined_PI_PD_Fe_Comb_emiss_molecules["insol_FeComb_a1"]
combined_PI_PD_Fe_Comb_emiss_molecules["emiss_fetca2"] = combined_PI_PD_Fe_Comb_emiss_molecules["insol_FeComb_a2"]
combined_PI_PD_Fe_Comb_emiss_molecules["emiss_fetca3"] = combined_PI_PD_Fe_Comb_emiss_molecules["insol_FeComb_a3"]

combined_PI_PD_Fe_Comb_emiss_all = combined_PI_PD_Fe_Comb_emiss_molecules.drop_vars(["sol_FeComb_a1", "sol_FeComb_a2", "sol_FeComb_a3", "insol_FeComb_a1", "insol_FeComb_a2", "insol_FeComb_a3"])

combined_PI_PD_Fe_Comb_emiss_all = combined_PI_PD_Fe_Comb_emiss_all.fillna(0.0).where(np.isfinite(combined_PI_PD_Fe_Comb_emiss_all), 0.0)

# combined_PI_PD_Fe_Comb_emiss_molecules.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\combined_PI_PD_Fe_Comb_emiss_molecules.nc")
# order of magnitude looks right for coarse -- nearly identical
# convert SPEWFUELS emissions to number emissions in (particles/cm2/s) (molecules/mol) (g/kg) to make num file
import math

rho_a1 = 1500.0 # (kg m-3)
rho_a2 = 1500.0 # (kg m-3)
rho_a3 = 2600.0 # (kg m-3)

dp_emiss_a1 = 0.1340*1E-6 # (m)
dp_emiss_a2 = 0.0504*1E-6 # (m)
dp_emiss_a3 = 2.0600*1E-6 # (m)

combined_PI_PD_Fe_Comb_emiss_num_a1 = (combined_PI_PD_Fe_Comb_emiss_molecules * 55.845) / (rho_a1 *(math.pi/6) * (dp_emiss_a1**3))
combined_PI_PD_Fe_Comb_emiss_num_a2 = (combined_PI_PD_Fe_Comb_emiss_molecules * 55.845) / (rho_a2 *(math.pi/6) * (dp_emiss_a2**3))
combined_PI_PD_Fe_Comb_emiss_num_a3 = (combined_PI_PD_Fe_Comb_emiss_molecules * 55.845) / (rho_a3 *(math.pi/6) * (dp_emiss_a3**3))

#combined_PI_PD_Fe_Comb_emiss_num_a1.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\combined_PI_PD_Fe_Comb_num_emiss.nc")
# order of magnitude looks right

#Now changing names of final variables to match model needs
a1_num_vars = ["insol_FeComb_a1", "sol_FeComb_a1"]
a2_num_vars = ["insol_FeComb_a2", "sol_FeComb_a2"]
a3_num_vars = ["insol_FeComb_a3", "sol_FeComb_a3"]

# Align all datasets to ensure they share the same dimensions
combined_PI_PD_Fe_Comb_emiss_num_a1, combined_PI_PD_Fe_Comb_emiss_num_a2, combined_PI_PD_Fe_Comb_emiss_num_a3 = xr.align(combined_PI_PD_Fe_Comb_emiss_num_a1, combined_PI_PD_Fe_Comb_emiss_num_a2, combined_PI_PD_Fe_Comb_emiss_num_a3 , join="outer")  # "outer" keeps all values

# Select the two variables from each dataset
selected_ds1 = combined_PI_PD_Fe_Comb_emiss_num_a1[a1_num_vars]
selected_ds2 = combined_PI_PD_Fe_Comb_emiss_num_a2[a2_num_vars]
selected_ds3 = combined_PI_PD_Fe_Comb_emiss_num_a3[a3_num_vars]

# Merge the selected variables from all three datasets
combined_PI_PD_Fe_Comb_emiss_num_all = xr.merge([selected_ds1, selected_ds2, selected_ds3])

# rename to variables recognized by the model 
combined_PI_PD_Fe_Comb_emiss_num_all["num_fesca1"] = combined_PI_PD_Fe_Comb_emiss_num_all["sol_FeComb_a1"]
combined_PI_PD_Fe_Comb_emiss_num_all["num_fesca2"] = combined_PI_PD_Fe_Comb_emiss_num_all["sol_FeComb_a2"]
combined_PI_PD_Fe_Comb_emiss_num_all["num_fesca3"] = combined_PI_PD_Fe_Comb_emiss_num_all["sol_FeComb_a3"]
combined_PI_PD_Fe_Comb_emiss_num_all["num_fetca1"] = combined_PI_PD_Fe_Comb_emiss_num_all["insol_FeComb_a1"]
combined_PI_PD_Fe_Comb_emiss_num_all["num_fetca2"] = combined_PI_PD_Fe_Comb_emiss_num_all["insol_FeComb_a2"]
combined_PI_PD_Fe_Comb_emiss_num_all["num_fetca3"] = combined_PI_PD_Fe_Comb_emiss_num_all["insol_FeComb_a3"]

combined_PI_PD_Fe_Comb_emiss_num_all = combined_PI_PD_Fe_Comb_emiss_num_all.drop_vars(["sol_FeComb_a1", "sol_FeComb_a2", "sol_FeComb_a3", "insol_FeComb_a1", "insol_FeComb_a2", "insol_FeComb_a3"])

# summing number emission files to compare orders of magnitude ------------------------
ds_num_fesca2 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\num_fesca2_SPEW_1979-2015_0.9x1.25.nc")
num_time_steps = ds_num_fesca2.dims["time"]
new_time_values = pd.date_range(start="1979-01-01", periods=num_time_steps, freq="MS")
ds_num_fesca2 = ds_num_fesca2.assign_coords(time=("time", new_time_values))


ds_num_fesca2 = ds_num_fesca2.sel(time=slice("2010-01-01", "2010-12-31"))
new_ds_num_fesca2 = combined_PI_PD_Fe_Comb_emiss_num_all.sel(time=slice("1850-01-01", "1850-12-31"))

num_sum_new = combined_PI_PD_Fe_Comb_emiss_num_all['num_fesca2'].sum()
num_sum_old = ds_num_fesca2['num_fesca2'].sum()

print("new", num_sum_new)
print("old", num_sum_old)

ds_num_fesca2 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\num_fesca2_SPEW_1979-2015_0.9x1.25.nc")
num_time_steps = ds_num_fesca2.dims["time"]
new_time_values = pd.date_range(start="1979-01-01", periods=num_time_steps, freq="MS")
ds_num_fesca2 = ds_num_fesca2.assign_coords(time=("time", new_time_values))
# Extract the existing time variable
existing_time = ds_num_fesca2['time']

# Create a new time array for 1850 with values on the 15th of each month
new_time = pd.date_range(start="1850-01-01", end="1850-12-31", freq='MS') + pd.Timedelta(days=14)
new_time_da = xr.DataArray(new_time, dims=["time"], coords={"time": new_time})

new_ds = xr.Dataset()
for var in ds_num_fesca2.data_vars:
    if "time" in ds_num_fesca2[var].dims:
        new_shape = list(ds_num_fesca2[var].shape)
        new_shape[ds_num_fesca2[var].dims.index("time")] = len(new_time)
        new_ds[var] = (ds_num_fesca2[var].dims, np.full(new_shape, np.nan, dtype=ds_num_fesca2[var].dtype))

new_ds["time"] = new_time_da

# Merge the new dataset (1850-2010) with the original
ds_expanded = xr.concat([new_ds, ds_num_fesca2], dim="time")

test_ds = combined_PI_PD_Fe_Comb_emiss_num_all.copy()
test_ds["num_fesca2_b"] = test_ds["num_fesca2"]

new_order = ('time', 'lat', 'lon')  # Change this as needed

# Apply reordering to all variables
ds_reordered = test_ds.map(lambda var: var.transpose(*new_order), keep_attrs=True)

ds_expanded = ds_expanded.assign(num_fesca2_b=ds_reordered['num_fesca2_b'])

#ds_expanded.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\num_fe_SPEWFUELS_1850-2010_test.nc")

#%% ---- PARSING NEW EMISSIONS FILES THROUGH OLD FILES TO ENSURE MODEL CAN READ FORMATTING OF DS -----------------------
ds_num_fesca3 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\num_fesca3_SPEW_1979-2015_0.9x1.25.nc")
num_time_steps = ds_num_fesca3.dims["time"]
new_time_values = pd.date_range(start="1979-01-01", periods=num_time_steps, freq="MS")
ds_num_fesca3 = ds_num_fesca3.assign_coords(time=("time", new_time_values))
ds_num_fesca3 = ds_num_fesca3.sel(time=slice(None, "2009-12-31"))

# for emissions in molecules
# combined_PI_PD_Fe_Comb_emiss_all
# for number emissions
# combined_PI_PD_Fe_Comb_emiss_num_all
test_ds = combined_PI_PD_Fe_Comb_emiss_num_all.copy()
test_ds["num_fesca3_b"] = test_ds["num_fesca3"]
new_order = ('time', 'lat', 'lon')  # Change this as needed
ds_reordered = test_ds.map(lambda var: var.transpose(*new_order), keep_attrs=True)
ds_expanded = test_ds.assign(num_fesca3_b=ds_reordered['num_fesca3_b'])

ds_expanded['num_fesca3_b'] = ds_expanded['num_fesca3_b'].astype(np.float32)

# Extract time variables
time_orig = ds_num_fesca3['time']
time_new = ds_expanded['time']

# Combine time coordinates, ensuring unique sorted values
combined_time = xr.concat([time_orig, time_new], dim="time").sortby("time").drop_duplicates("time")

# Reindex both datasets to the combined time axis
ds_orig_expanded = ds_num_fesca3.reindex(time=combined_time, fill_value=np.nan)
ds_new_expanded = ds_expanded.reindex(time=combined_time, fill_value=np.nan)

# Merge datasets
ds_final = ds_orig_expanded.assign(num_fesca3_b=ds_new_expanded['num_fesca3_b'])

# this worked to add the data but now let's examine how date is being treated
indices_to_replace = [0,1,2,3,4,5,6,7,8,9,10,11,384,385,386,387,388,389,390,391,392,393,394,395]  # Indices to be replaced
new_values = [18500116.0, 18500216.0, 18500316.0, 18500416.0,
              18500516.0, 18500616.0, 18500716.0, 18500816.0,
              18500916.0, 18501016.0, 18501116.0, 18501216.0,
              20100116.0, 20100216.0, 20100316.0, 20100416.0,
              20100516.0, 20100616.0, 20100716.0, 20100816.0,
              20100916.0, 20101016.0, 20101116.0, 20101216.0]  # New values to insert

# Replace date by index
ds_final["date"].values[indices_to_replace] = new_values

ds_final["num_fesca3_b"].attrs = ds_final["num_fesca3"].attrs.copy()

start_time = "1851-01-01"
end_time = "2009-12-31"
ds_final = ds_final.sel(time=slice(None, start_time)).combine_first(ds_final.sel(time=slice(end_time, None)))

ds_final = ds_final.drop_vars("num_fesca3")
ds_final["num_fesca3"] = ds_final["num_fesca3_b"]
ds_final = ds_final.drop_vars("num_fesca3_b")

ds_final.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\input data files\\num_fesca3_SPEW_1850-2010_0.9x1.25_updated_CombFe_scaling.nc")


