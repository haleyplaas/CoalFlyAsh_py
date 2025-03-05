#%% CALCULATING SECTOR SPECIFIC Fe EMISSIONS FROM CMIP BLACK CARBON EMISSION DATASETS  ----------------------
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
#ds_PD = ds_PD.drop_dims(["time"])
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

# Specifying emission sector for each dataset
BC_emiss_4_SSP370 = BC_em_anthro_SSP370.isel(sector=4) # Residential Coal

# ---------- Grouping emission sources from different sectors for COALFLYASH variables ------------------------
# need to find ratio of BF to WOOD before assigning Residential coal (4) 
ds_PD_Fe = ds_PD_Fe.drop_vars(['time'], errors= 'ignore')

CoaMed_FCOALBF_PD = ds_PD_Fe['CoaMed_FCOALBF'] 
FineMed_FCOALBF_PD = ds_PD_Fe['FineMed_FCOALBF']   
CoaMed_FWOOD_PD = ds_PD_Fe['CoaMed_FWOOD']  
FineMed_FWOOD_PD = ds_PD_Fe['FineMed_FWOOD']  

# Adding fractions of Fe together
BothMode_FCOALBF_PD = (CoaMed_FCOALBF_PD + FineMed_FCOALBF_PD)
BothMode_FWOOD_PD = (CoaMed_FWOOD_PD + FineMed_FWOOD_PD)

BC_COALBF_SSP370 = BC_emiss_4_SSP370.copy()
BC_WOOD_SSP370 = BC_emiss_4_SSP370.copy()

# ----------------------------------------------------------------------------------------------------------
# The same now but for the PD BC file 
# Extract Black Carbon (BC) emission values PD 
BC_em_anthro_PD = ds_PD['BC_em_anthro']  # Emission data
BC_emiss_4_PD = BC_em_anthro_PD.isel(sector=4) # Residential Coal

BC_COALBF_PD = BC_emiss_4_PD.copy()
BC_WOOD_PD = BC_emiss_4_PD.copy()

#---------------------------------------------------------------------------------------------------------------
# Finding average ratio of Fe to BC 
avg_iron_frac_valid = .12

# removing variables I no longer need (OCNFRAC and time from emission xarrays)
BC_COALBF_SSP370 = BC_COALBF_SSP370.drop_vars(['time','OCNFRAC','sector'], errors= 'ignore')
BC_WOOD_SSP370 = BC_WOOD_SSP370.drop_vars(['time','OCNFRAC','sector'], errors= 'ignore')

BC_COALBF_PD = BC_COALBF_PD.drop_vars(['time','OCNFRAC','sector'], errors= 'ignore')
BC_WOOD_PD = BC_WOOD_PD.drop_vars(['time','OCNFRAC','sector'], errors= 'ignore')

# Masks for indexing cells where a BC value exists from SSP370 dataset, but no data is available for PD iron, resulting in an extra NA value
zero_mask_COALBF = (BC_COALBF_SSP370 != 0) & ((BothMode_FCOALBF_PD == 0) | (BC_COALBF_PD == 0))
BC_COALBF_SSP370_masked = BC_COALBF_SSP370.where(zero_mask_COALBF)
masked_indices_COALBF = list(zip(*np.where(zero_mask_COALBF)))
print(len(masked_indices_COALBF))
print("indices", masked_indices_COALBF)

zero_mask_WOOD = (BC_WOOD_SSP370 != 0) & ((BothMode_FWOOD_PD == 0) | (BC_WOOD_PD == 0))
BC_WOOD_SSP370_masked = BC_WOOD_SSP370.where(zero_mask_WOOD)
masked_indices_WOOD = list(zip(*np.where(zero_mask_WOOD)))
print(len(masked_indices_WOOD))
print("indices", masked_indices_WOOD)

# Calculating Fe_SSP370 emission values from ratio of FePD/BCPD to BCFU ----------------------------
# FCOALBF --------------------------------------------------------------------
BothMode_FCOALBF_SSP370 = xr.zeros_like(BC_COALBF_SSP370)
BC_COALBF_SSP370_copy = BC_COALBF_SSP370.copy()
for index in np.ndindex(BC_COALBF_SSP370.shape):
    if index in masked_indices_COALBF:
        # For masked indices (where no PD data), multiply the cell by avg_iron_frac_valid 
        BothMode_FCOALBF_SSP370[index] = BC_COALBF_SSP370_copy[index] * avg_iron_frac_valid
    else:
        # For all other indices, use cell by cell ratios of Fe:BC to determine regional Fe emission frac 
        BothMode_FCOALBF_SSP370[index] = (BothMode_FCOALBF_PD[index] / BC_COALBF_PD[index]) * BC_COALBF_SSP370_copy[index]

# FWOOD --------------------------------------------------------------------
BothMode_FWOOD_SSP370 = xr.zeros_like(BC_WOOD_SSP370)
BC_WOOD_SSP370_copy = BC_WOOD_SSP370.copy()
for index in np.ndindex(BC_WOOD_SSP370.shape):
    if index in masked_indices_WOOD:
        BothMode_FWOOD_SSP370[index] = BC_WOOD_SSP370_copy[index] * avg_iron_frac_valid
    else:
            BothMode_FWOOD_SSP370[index] = (BothMode_FWOOD_PD[index] / BC_WOOD_PD[index]) * BC_WOOD_SSP370_copy[index]
# -------------------------------------------------------------------------------------------------------

# counting BC_SSP370 0's prior to calculating Fe concs. 
zero_counts = np.count_nonzero(BC_COALBF_SSP370_copy == 0)
print("Number of Zeros:", zero_counts)
# counting NaNs after multiplying by PD Fe/ratio and supplementing missing values with 11.3%
nan_counts = BothMode_FCOALBF_SSP370.isnull().sum()
print("Number of NaNs per variable:", nan_counts)
# only slight value differences between each for each sector (<10)

# ------- Size distribution (ratio of Coa to Fine mode Fe) is distinct for all sectors -----------------------
# -------- calculating the fine and coarse fractionation from the PD run to apply percentage each mode to the future data -----
# -------- emissions are 0 over the ocean so the data over the ocean becomes NULL without changing to 0's ---------------------
CoaFrac_FCOALBF = CoaMed_FCOALBF_PD/(CoaMed_FCOALBF_PD + FineMed_FCOALBF_PD)
FineFrac_FCOALBF = FineMed_FCOALBF_PD/(CoaMed_FCOALBF_PD + FineMed_FCOALBF_PD)

CoaFrac_FWOOD = CoaMed_FWOOD_PD/(CoaMed_FWOOD_PD + FineMed_FWOOD_PD)
FineFrac_FWOOD = FineMed_FWOOD_PD/(CoaMed_FWOOD_PD + FineMed_FWOOD_PD)

average_coafrac_FCOALBF = CoaFrac_FCOALBF.mean()
print("avg coafrac_FCOALBF", average_coafrac_FCOALBF)
average_Finefrac_FCOALBF = FineFrac_FCOALBF.mean()
print("avg finefrac_FCOALBF", average_Finefrac_FCOALBF)
total_frac_FCOALBF = average_coafrac_FCOALBF + average_Finefrac_FCOALBF
print("avg_total_FCOALBF", total_frac_FCOALBF)

average_coafrac_FWOOD = CoaFrac_FWOOD.mean()
print("avg coafrac_FWOOD", average_coafrac_FWOOD)
average_Finefrac_FWOOD = FineFrac_FWOOD.mean()
print("avg finefrac_FWOOD", average_Finefrac_FWOOD)
total_frac_FWOOD = average_coafrac_FWOOD + average_Finefrac_FWOOD
print("avg_total_FWOOD", total_frac_FWOOD)
# totals all add to one, passed this test

# now to size fractionate the FU Fe, using individual grid cells where possible, but average size fractionation by sector where data is missing
# BIOFUELS ------------------------------------------------------------------------------------------------
CoaMed_FCOALBF_SSP370 = xr.zeros_like(BothMode_FCOALBF_SSP370)
for index in np.ndindex(BothMode_FCOALBF_SSP370.shape):
    if index in masked_indices_COALBF:
        # For masked indices where data does not exist for PD files, multiply the cell by the avg aerosol mode fraction for that sector
        CoaMed_FCOALBF_SSP370[index] = BothMode_FCOALBF_SSP370[index] * average_coafrac_FCOALBF
    else:
        # For all other indices, use the Fe SPEWFUELS data to determine regional Fe emission frac 
        CoaMed_FCOALBF_SSP370[index] = BothMode_FCOALBF_SSP370[index] * CoaFrac_FCOALBF[index]

FineMed_FCOALBF_SSP370 = xr.zeros_like(BothMode_FCOALBF_SSP370)
for index in np.ndindex(BothMode_FCOALBF_SSP370.shape):
    if index in masked_indices_COALBF:
        # For masked indices where data does not exist for PD files, multiply the cell by the avg aerosol mode fraction for that sector
        FineMed_FCOALBF_SSP370[index] = BothMode_FCOALBF_SSP370[index] * average_Finefrac_FCOALBF
    else:
        # For all other indices, use the Fe SPEWFUELS data to determine regional Fe emission frac 
        FineMed_FCOALBF_SSP370[index] = BothMode_FCOALBF_SSP370[index] * FineFrac_FCOALBF[index]

# WOOD ------------------------------------------------------------------------------------------------
CoaMed_FWOOD_SSP370 = xr.zeros_like(BothMode_FWOOD_SSP370)
for index in np.ndindex(BothMode_FWOOD_SSP370.shape):
    if index in masked_indices_COALBF:
        # For masked indices where data does not exist for PD files, multiply the cell by the avg aerosol mode fraction for that sector
        CoaMed_FWOOD_SSP370[index] = BothMode_FWOOD_SSP370[index] * average_coafrac_FWOOD
    else:
        # For all other indices, use the Fe SPEWFUELS data to determine regional Fe emission frac 
        CoaMed_FWOOD_SSP370[index] = BothMode_FWOOD_SSP370[index] * CoaFrac_FWOOD[index]

FineMed_FWOOD_SSP370 = xr.zeros_like(BothMode_FWOOD_SSP370)
for index in np.ndindex(BothMode_FWOOD_SSP370.shape):
    if index in masked_indices_COALBF:
        # For masked indices where data does not exist for PD files, multiply the cell by the avg aerosol mode fraction for that sector
        FineMed_FWOOD_SSP370[index] = BothMode_FWOOD_SSP370[index] * average_Finefrac_FWOOD
    else:
        # For all other indices, use the Fe SPEWFUELS data to determine regional Fe emission frac 
        FineMed_FWOOD_SSP370[index] = BothMode_FWOOD_SSP370[index] * FineFrac_FWOOD[index]

# REPEATING FRACTIONATION FOR BLACK CARBON SSP370
# BIOFUELS ------------------------------------------------------------------------------------------------
CoaMed_BCCOALBF_SSP370 = xr.zeros_like(BC_COALBF_PD)
for index in np.ndindex(BC_COALBF_PD.shape):
    if index in masked_indices_COALBF:
# # For masked indices where data does not exist for PD files, multiply the cell by the avg aerosol mode fraction for that sector
        CoaMed_BCCOALBF_SSP370[index] = BC_COALBF_PD[index] * average_coafrac_FCOALBF
    else:
#        # For all other indices, use the Fe SPEWFUELS data to determine regional Fe emission frac 
        CoaMed_BCCOALBF_SSP370[index] = BC_COALBF_PD[index] * CoaFrac_FCOALBF[index]

FineMed_BCCOALBF_SSP370 = xr.zeros_like(BothMode_FCOALBF_SSP370)
for index in np.ndindex(BothMode_FCOALBF_SSP370.shape):
    if index in masked_indices_COALBF:
 #       # For masked indices where data does not exist for PD files, multiply the cell by the avg aerosol mode fraction for that sector
        FineMed_BCCOALBF_SSP370[index] = BC_COALBF_PD[index] * average_Finefrac_FCOALBF
    else:
 #       # For all other indices, use the Fe SPEWFUELS data to determine regional Fe emission frac 
        FineMed_BCCOALBF_SSP370[index] = BC_COALBF_PD[index] * FineFrac_FCOALBF[index]

# WOOD ------------------------------------------------------------------------------------------------
CoaMed_BCWOOD_SSP370 = xr.zeros_like(BC_WOOD_PD)
for index in np.ndindex(BC_WOOD_PD.shape):
    if index in masked_indices_COALBF:
        # For masked indices where data does not exist for PD files, multiply the cell by the avg aerosol mode fraction for that sector
        CoaMed_BCWOOD_SSP370[index] = BC_WOOD_PD[index] * average_coafrac_FWOOD
    else:
        # For all other indices, use the Fe SPEWFUELS data to determine regional Fe emission frac 
        CoaMed_BCWOOD_SSP370[index] = BC_WOOD_PD[index] * CoaFrac_FWOOD[index]

FineMed_BCWOOD_SSP370 = xr.zeros_like(BC_WOOD_PD)
for index in np.ndindex(BC_WOOD_PD.shape):
    if index in masked_indices_COALBF:
        # For masked indices where data does not exist for PD files, multiply the cell by the avg aerosol mode fraction for that sector
        FineMed_BCWOOD_SSP370[index] = BC_WOOD_PD[index] * average_Finefrac_FWOOD
    else:
        # For all other indices, use the Fe SPEWFUELS data to determine regional Fe emission frac 
        FineMed_BCWOOD_SSP370[index] = BC_WOOD_PD[index] * FineFrac_FWOOD[index]

CoaMed_BCCOALBF_PD = CoaFrac_FCOALBF * BC_COALBF_PD
CoaMed_BCCOALBF_PD = CoaMed_BCCOALBF_PD.fillna(0.0)
CoaMed_BCCOALBF_PD = CoaMed_BCCOALBF_PD.where(np.isfinite(CoaMed_BCCOALBF_PD), 0.0)

FineMed_BCCOALBF_PD = FineFrac_FCOALBF * BC_COALBF_PD
FineMed_BCCOALBF_PD = FineMed_BCCOALBF_PD.fillna(0.0)
FineMed_BCCOALBF_PD = FineMed_BCCOALBF_PD.where(np.isfinite(FineMed_BCCOALBF_PD), 0.0)

CoaMed_BCWOOD_PD = CoaFrac_FWOOD * BC_WOOD_PD
CoaMed_BCWOOD_PD = CoaMed_BCWOOD_PD.fillna(0.0)
CoaMed_BCWOOD_PD = CoaMed_BCWOOD_PD.where(np.isfinite(CoaMed_BCWOOD_PD), 0.0)

FineMed_BCWOOD_PD = FineFrac_FWOOD * BC_WOOD_PD
FineMed_BCWOOD_PD = FineMed_BCWOOD_PD.fillna(0.0)
FineMed_BCWOOD_PD = FineMed_BCWOOD_PD.where(np.isfinite(FineMed_BCWOOD_PD), 0.0)

# Zeroing out emissions at very end 
CoaMed_FCOALBF_SSP370 = CoaMed_FCOALBF_SSP370.fillna(0.0).where(np.isfinite(CoaMed_FCOALBF_SSP370), 0.0)
CoaMed_FWOOD_SSP370 = CoaMed_FWOOD_SSP370.fillna(0.0).where(np.isfinite(CoaMed_FWOOD_SSP370), 0.0)

FineMed_FCOALBF_SSP370 = FineMed_FCOALBF_SSP370.fillna(0.0).where(np.isfinite(FineMed_FCOALBF_SSP370), 0.0)
FineMed_FWOOD_SSP370 = FineMed_FWOOD_SSP370.fillna(0.0).where(np.isfinite(FineMed_FWOOD_SSP370), 0.0)

CoaMed_BCCOALBF_SSP370 = CoaMed_BCCOALBF_SSP370.fillna(0.0).where(np.isfinite(CoaMed_BCCOALBF_SSP370), 0.0)
CoaMed_BCWOOD_SSP370 = CoaMed_BCWOOD_SSP370.fillna(0.0).where(np.isfinite(CoaMed_BCWOOD_SSP370), 0.0)

FineMed_BCCOALBF_SSP370 = FineMed_BCCOALBF_SSP370.fillna(0.0).where(np.isfinite(FineMed_BCCOALBF_SSP370), 0.0)
FineMed_BCWOOD_SSP370 = FineMed_BCWOOD_SSP370.fillna(0.0).where(np.isfinite(FineMed_BCWOOD_SSP370), 0.0)

# CALCULATING EMISSION BUDGETS -------------------------------
import numpy as np
import xarray as xr

all_variables = xr.Dataset({
    "CoaMed_FCOALBF_SSP370": CoaMed_FCOALBF_SSP370,
    "FineMed_FCOALBF_SSP370": FineMed_FCOALBF_SSP370,
    "CoaMed_FWOOD_SSP370": CoaMed_FWOOD_SSP370,
    "FineMed_FWOOD_SSP370": FineMed_FWOOD_SSP370,
    "CoaMed_FCOALBF_PD": CoaMed_FCOALBF_PD,
    "FineMed_FCOALBF_PD": FineMed_FCOALBF_PD,
    "CoaMed_FWOOD_PD": CoaMed_FWOOD_PD,
    "FineMed_FWOOD_PD": FineMed_FWOOD_PD,
    "CoaMed_BCCOALBF_SSP370": CoaMed_BCCOALBF_SSP370,
    "FineMed_BCCOALBF_SSP370": FineMed_BCCOALBF_SSP370,
    "CoaMed_BCWOOD_SSP370": CoaMed_BCWOOD_SSP370,
    "FineMed_BCWOOD_SSP370": FineMed_BCWOOD_SSP370,
    "CoaMed_BCCOALBF_PD": CoaMed_BCCOALBF_PD,
    "FineMed_BCCOALBF_PD": FineMed_BCCOALBF_PD,
    "CoaMed_BCWOOD_PD": CoaMed_BCWOOD_PD,
    "FineMed_BCWOOD_PD": FineMed_BCWOOD_PD,
})

ds_file = all_variables

lon = ds_file['lon'].values  # Longitude in degrees
lat = ds_file['lat'].values  # Latitude in degrees

# Convert to radians
lon_rads = np.radians(lon)
lat_rads = np.radians(lat)

# Compute grid spacings (assuming equidistant grid)
d_lat = np.abs(lat[1] - lat[0])  # Latitude grid spacing in degrees
d_lon = np.abs(lon[1] - lon[0])  # Longitude grid spacing in degrees

# Half-grid spacing for staggered grid
g_lat = np.radians(d_lat / 2)  # Latitude half-spacing in radians
g_lon = np.radians(d_lon / 2)  # Longitude half-spacing in radians

# Radius of Earth in m
R = 6.3781E6 

# Empty list to store results
cell_areas_staggered = []

# Loop through each grid cell
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

# Convert to numpy array and reshape to match grid dimensions
cell_areas_staggered = np.array(cell_areas_staggered).reshape(len(lat), len(lon))

# Verify to see if areas add to 5.1E14 in m^2
sum_sa_earth = cell_areas_staggered.sum()
print(f"surface area, {sum_sa_earth:3e}") 

# add cell area to original xarray for easy calling 
ds_file['cell_area'] = xr.DataArray(
    cell_areas_staggered,
    dims=["lat", "lon"],  # Same dimensions as in the original dataset
    coords={"lat": ds_file['lat'], "lon": ds_file['lon']},  # Use original coordinates
    attrs={
        "units": "m^2",  # Specify units for the cell area
        "description": "Calculated grid cell area using staggered grid approach",
    },
)

print(ds_file)

# Calculating budgets-- started with OG Fe emissions to check against MIMI, got the same results as the for loop 
import pandas as pd
# Dictionaries to store both individual and total budgets
global_budgets = {}
individual_cell_emissions = {}
# Not sure if this loop is running properly, getting different results when I do it manually
# Loop over all variables in the dataset
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

# Create a DataFrame from the budgets dictionary (Total Budgets)
budget_df = pd.DataFrame(list(global_budgets.items()), columns=["Variable", "Total_Budget_Tg_per_year"])
print(budget_df)

## Create a DataFrame for the individual emissions, flattening the xarray values
#individual_df = pd.DataFrame()
#for var_name in individual_cell_emissions:
    # Flatten the DataArray into a pandas DataFrame for the individual emissions
    #individual_df[var_name] = individual_cell_emissions[var_name].values.flatten()

# Save the total budgets and individual emissions to separate sheets in an Excel file
output_file = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\Budgets_01072025_4.xlsx"
with pd.ExcelWriter(output_file) as writer:
   budget_df.to_excel(writer, sheet_name="Total_Budgets_zeroed_throughout", index=False)  
   print(f"Total budgets and individual emissions saved to {output_file}")
