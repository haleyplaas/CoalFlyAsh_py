#%%
# FINDING FOSSIL FUEL EMISSION BUDGET FOR FE SSP370
import xarray as xr
import pandas as pd
import numpy as np

# Load the NetCDF file -- BC EMISSIONS SSP370
ds_SSP370 = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_2090_2100MEAN_remapcon_regridded.nc")
ds_SSP370 = ds_SSP370.drop_vars(["time_bnds", "sector_bnds", "time"])
ds_SSP370 = ds_SSP370.squeeze(dim='time') 

# Load the NetCDF file -- BC EMISSIONS PD CMIP6
ds_PD = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_2000-2015MEAN_remapcon_regridded.nc") 
ds_PD = ds_PD.drop_vars(["time_bnds", "sector_bnds", "time"])
ds_PD = ds_PD.squeeze(dim='time') 

# Load the NetCDF file -- BC EMISSIONS PD CMIP6 (use as OCNFRAC too)
ds_PD_Fe = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\ClaquinMineralsCAM6_SPEWFUELS-Dec2023_OCNFRAC_remapcon_regridded.nc")
ds_PD_Fe['OCNFRAC'] = ds_PD_Fe['OCNFRAC'].squeeze(dim='time')
ds_PD_Fe = ds_PD_Fe.drop_dims(["time"]) 

# Extract Black Carbon (BC) emission values SSP370 
BC_em_anthro_SSP370 = ds_SSP370['BC_em_anthro'] # Emission data
lon = ds_SSP370['lon'].values  # Longitude values -- the .values extracts as a numPy array, removes metadata
lat = ds_SSP370['lat'].values  # Latitude values
sector = ds_SSP370['sector'].values # Sector identifiers

BC_emiss_1_SSP370 = BC_em_anthro_SSP370.isel(sector=1) # Energy
BC_emiss_2_SSP370 = BC_em_anthro_SSP370.isel(sector=2) # Industrial Coal

# Extract Black Carbon (BC) emission values PD
BC_em_anthro_PD = ds_PD['BC_em_anthro'] # Emission data
lon = ds_PD['lon'].values  # Longitude values -- the .values extracts as a numPy array, removes metadata
lat = ds_PD['lat'].values  # Latitude values
sector = ds_PD['sector'].values # Sector identifiers

BC_emiss_1_PD = BC_em_anthro_PD.isel(sector=1) # Energy
BC_emiss_2_PD = BC_em_anthro_PD.isel(sector=2) # Industrial Coal

# Extract Iron (Fe) emission values PD 
CoaMed_FCOALFF_PD = ds_PD_Fe['CoaMed_FCOALFF']  
FineMed_FCOALFF_PD = ds_PD_Fe['FineMed_FCOALFF']  

# Adding aerosol size fractions together 
BothMode_FCOALFF_PD = (CoaMed_FCOALFF_PD + FineMed_FCOALFF_PD).copy()

# Assigning Sector to Fuel Type
BC_COALFF_SSP370 = (BC_emiss_1_SSP370 + BC_emiss_2_SSP370).copy()
BC_COALFF_PD = (BC_emiss_1_PD + BC_emiss_2_PD).copy()

BC_COALFF_SSP370 = BC_COALFF_SSP370.drop_vars(['time','OCNFRAC','sector'], errors= 'ignore')
BC_COALFF_PD = BC_COALFF_PD.drop_vars(['time','OCNFRAC','sector'], errors= 'ignore')

# multiply through by average Fe_Frac
# 0.08237672525125478 Avg % Iron
# 0.08464804545797079 Avg % Iron Terrestrial
# 0.06243864500442786 Avg % Iron Ocean

avg_terr_FeBC_frac = 0.08464804545797079

# single grid cell way 
Fe_COALFF_SSP370_single_gridcell = (BothMode_FCOALFF_PD/BC_COALFF_PD) * BC_COALFF_SSP370
print(Fe_COALFF_SSP370_single_gridcell) 
# global average way
Fe_COALFF_SSP370_global_mean = BC_COALFF_SSP370 * avg_terr_FeBC_frac
print(Fe_COALFF_SSP370_global_mean) 

# CHOOSE ONE OR THE OTHER FOR SIMPLICITY
# single grid cell way 
Fe_COALFF_SSP370 = (BothMode_FCOALFF_PD/BC_COALFF_PD) * BC_COALFF_SSP370
# global average way
#Fe_COALFF_SSP370 = BC_COALFF_SSP370 * avg_terr_FeBC_frac

# Separate into Coarse and Fine Modes
CoaFrac_FCOALFF = CoaMed_FCOALFF_PD/(CoaMed_FCOALFF_PD + FineMed_FCOALFF_PD).copy()
FineFrac_FCOALFF = FineMed_FCOALFF_PD/(CoaMed_FCOALFF_PD + FineMed_FCOALFF_PD).copy()

CoaMed_FCOALFF_SSP370 = CoaFrac_FCOALFF * Fe_COALFF_SSP370
FineMed_FCOALFF_SSP370 = FineFrac_FCOALFF * Fe_COALFF_SSP370
TOT_FCOALFF_SSP370 = CoaMed_FCOALFF_SSP370 + FineMed_FCOALFF_SSP370 

Fe_COALFF_SSP370_sum = Fe_COALFF_SSP370.sum().item()
print(Fe_COALFF_SSP370_sum)
TOT_FCOALFF_SSP370_sum = TOT_FCOALFF_SSP370.sum().item()
print(TOT_FCOALFF_SSP370_sum)

# Find Emission budgets for FF Fine and Coarse 
# first must separate BC into Fine and Coarse in order to calculate
CoaMed_BCCOALFF_SSP370 = CoaFrac_FCOALFF * BC_COALFF_SSP370
FineMed_BCCOALFF_SSP370 = FineFrac_FCOALFF * BC_COALFF_SSP370

CoaMed_BCCOALFF_PD = CoaFrac_FCOALFF * BC_COALFF_PD
FineMed_BCCOALFF_PD = FineFrac_FCOALFF * BC_COALFF_PD

all_variables = xr.Dataset({
    "CoaMed_FCOALFF_SSP370": CoaMed_FCOALFF_SSP370,
    "FineMed_FCOALFF_SSP370": FineMed_FCOALFF_SSP370,

    "CoaMed_FCOALFF_PD": CoaMed_FCOALFF_PD,
    "FineMed_FCOALFF_PD": FineMed_FCOALFF_PD,

    "CoaMed_BCCOALFF_SSP370": CoaMed_BCCOALFF_SSP370,
    "FineMed_BCCOALFF_SSP370":FineMed_BCCOALFF_SSP370,

    "CoaMed_BCCOALFF_PD": CoaMed_BCCOALFF_PD,
    "FineMed_BCCOALFF_PD": FineMed_BCCOALFF_PD,})

print(all_variables)

ds_file = all_variables
lon = ds_file['lon'].values  # Longitude in degrees
lat = ds_file['lat'].values  # Latitude in degrees
lon_rads = np.radians(lon)
lat_rads = np.radians(lat)
d_lat = np.abs(lat[1] - lat[0])  # Latitude grid spacing in degrees
d_lon = np.abs(lon[1] - lon[0])  # Longitude grid spacing in degrees
g_lat = np.radians(d_lat / 2)  # Latitude half-spacing in radians
g_lon = np.radians(d_lon / 2)  # Longitude half-spacing in radians

R = 6.3781E6 
cell_areas_staggered = []
# specifies the coordinates with [j,i] -- lat being i and lon being j 
for i in range(len(lat)):
    for j in range(len(lon)):
        # Convert latitude and longitude to radians
        lat_center = lat_rads[i]
        lon_center = lon_rads[j]
        # Compute staggered latitudes per Arakawa Lamb - C gridding for CESM
        lat_north = lat_center + g_lat
        lat_south = lat_center - g_lat
        # Ensure staggered latitudes are within valid range (-π/2 to π/2)
        lat_north = np.clip(lat_north, -np.pi / 2, np.pi / 2)
        lat_south = np.clip(lat_south, -np.pi / 2, np.pi / 2)
        # Compute area of the cell
        area = R**2 * (np.sin(lat_north) - np.sin(lat_south)) * (2 * g_lon)
        cell_areas_staggered.append(area)

cell_areas_staggered = np.array(cell_areas_staggered).reshape(len(lat), len(lon))
# Verify to see if areas add to 5.1E14 in m^2
sum_sa_earth = cell_areas_staggered.sum()
print(f"surface area, {sum_sa_earth:3e}") 

ds_file['cell_area'] = xr.DataArray(
    cell_areas_staggered,
    dims=["lat", "lon"],  # Same dimensions as in the original dataset
    coords={"lat": ds_file['lat'], "lon": ds_file['lon']},  # Use original coordinates
    attrs={
        "units": "m^2",  # Specify units for the cell area
        "description": "Calculated grid cell area using staggered grid approach",
    },
)

global_budgets = {}
individual_cell_emissions = {}
for var_name in ds_file.data_vars:
    if "Med" in var_name:  
        # Calculate individual emissions (for each cell) and convert from sec to annual and from kg to Tg
        # Assuming input variable is the flux in kg/m²/s and 'cell_area' is the area of each grid cell in m²
        individual_cell_emissions[var_name] = (ds_file[var_name] * ds_file['cell_area'] * 3600 * 24 * 365 *1E-9)
        
        # Calculate the total budget by summing individual emissions
        total_budget = individual_cell_emissions[var_name].sum()
        
        # Store the total budget in the dictionary
        global_budgets[var_name] = total_budget
        
        # Print the budget for the variable
        print(f"FU Total Fe budget of {var_name} (Tg/year): {total_budget:.3e}")

#COARSE
CoaMed_FCOALFF_SSP370_Tg = ds_file['CoaMed_FCOALFF_SSP370']*ds_file['cell_area']*3600*24*365*1E-9
total_Fe_SSP370_emiss = CoaMed_FCOALFF_SSP370_Tg.sum()

CoaMed_FCOALFF_PD_Tg = ds_file['CoaMed_FCOALFF_PD']*ds_file['cell_area']*3600*24*365*1E-9
total_Fe_PD_emiss = CoaMed_FCOALFF_PD_Tg.sum()

CoaMed_BCCOALFF_SSP370_Tg = ds_file['CoaMed_BCCOALFF_SSP370']*ds_file['cell_area']*3600*24*365*1E-9
total_BC_SSP370_emiss = CoaMed_BCCOALFF_SSP370_Tg.sum()

CoaMed_BCCOALFF_PD_Tg = ds_file['CoaMed_BCCOALFF_PD']*ds_file['cell_area']*3600*24*365*1E-9
total_BC_PD_emiss = CoaMed_BCCOALFF_PD_Tg.sum()

quick_total_ratio_PD = total_Fe_PD_emiss/total_BC_PD_emiss
print("PD_ratio Coa", quick_total_ratio_PD) # 18.6%
quick_total_ratio_SSP370 = total_Fe_SSP370_emiss/total_BC_SSP370_emiss
print("SSP370_ratio Coa", quick_total_ratio_SSP370) # 19.3%

#FINE
FineMed_FCOALFF_SSP370_Tg = ds_file['FineMed_FCOALFF_SSP370']*ds_file['cell_area']*3600*24*365*1E-9
total_Fe_SSP370_emiss = FineMed_FCOALFF_SSP370_Tg.sum()

FineMed_FCOALFF_PD_Tg = ds_file['FineMed_FCOALFF_PD']*ds_file['cell_area']*3600*24*365*1E-9
total_Fe_PD_emiss = FineMed_FCOALFF_PD_Tg.sum()

FineMed_BCCOALFF_SSP370_Tg = ds_file['FineMed_BCCOALFF_SSP370']*ds_file['cell_area']*3600*24*365*1E-9
total_BC_SSP370_emiss = FineMed_BCCOALFF_SSP370_Tg.sum() 

FineMed_BCCOALFF_PD_Tg = ds_file['FineMed_BCCOALFF_PD']*ds_file['cell_area']*3600*24*365*1E-9
total_BC_PD_emiss = FineMed_BCCOALFF_PD_Tg.sum()

quick_total_ratio_PD = total_Fe_PD_emiss/total_BC_PD_emiss
print("PD_ratio Fine", quick_total_ratio_PD) # 18.5%
quick_total_ratio_SSP370 = total_Fe_SSP370_emiss/total_BC_SSP370_emiss
print("SSP370_ratio Fine", quick_total_ratio_SSP370) # 18.9%

print("Fe_SSP370_Tg", total_Fe_SSP370_emiss)
print("Fe_PD_Tg", total_Fe_PD_emiss)
print("BC_SSP370_Tg", total_BC_SSP370_emiss)
print("BC_PD_Tg", total_BC_PD_emiss)

print("Fe_SSP370_Tg", total_Fe_SSP370_emiss)
print("Fe_PD_Tg", total_Fe_PD_emiss)
print("BC_SSP370_Tg", total_BC_SSP370_emiss)
print("BC_PD_Tg", total_BC_PD_emiss)