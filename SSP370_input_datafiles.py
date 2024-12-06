#%% DONT RE-RUN THIS REGRIDDER
#from netCDF4 import Dataset

# Quick view of some of the details within the netcdf file
# Open the dataset 
#dataset = Dataset('C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_2000-2015MEAN.nc', 'r')

# List all variables and attributes
#print(dataset.variables.keys())
#print(dataset.variables['BC_em_anthro'])
#print(dataset.variables['sector'])
#print(dataset.variables['sector_bnds'])

# 0: Agriculture; 1: Energy;  2: Industrial; 3: Transportation; 4: Residential, Commercial, Other; 5: Solvents production and application; 6: Waste; 7: International Shipping

# FCOAL = 1 + 2 (FF) + 4 (BF)
# FOIL = 3 (only from terrestrial cells) + 7 (only from ocean cells)
# FWOOD = 4 (BF?)
# FSMELT = bring over from Fe_Emissions_Fuel
# I think I actually need to split coal into the following though, based on model variable pointers
# FCOALFF = 1 + 2
# FCOALBF = 4

# F370[i,j,s] = BC370[i,j,s] / BCPD[i,j,s] * FePD[i,j,s]
# where i is lon, j is lat, and s is sector 
# calculate for each sector, and then add to create new consolidated variables
# need to double check with Douglas on wood and FCOAL

# visualizing the BC_emiss data for each sector in separate pandas dataframes
#import xarray as xr
#import pandas as pd
#import numpy as np

# Load the NetCDF file -- for Future simulation
#file_path = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_2000-2015MEAN.nc"
#ds = xr.open_dataset(file_path)

# grid dimensions are currently 360 lat x 720 lon
# re-gridding to 192 lat x 288 lon
#import xesmf as xe

# Create target grid (192 x 288)
#lat_target = xr.DataArray(
    #data=np.linspace(-90, 90, 192),
   # dims='lat',
    #name='lat')
#lon_target = xr.DataArray(
  #  data=np.linspace(0, 360, 288, endpoint=False),  # 0 to 360 for 288 points
   # dims='lon',
    #name='lon')
#target_grid = xr.Dataset({'lat': lat_target, 'lon': lon_target})

# Apply regridding
#regridder = xe.Regridder(ds, target_grid, method='bilinear')
#ds_regridded = regridder(ds)

# Save to a new NetCDF file
#ds_regridded.to_netcdf('C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_2000-2015MEAN_regridded_288x192.nc') 

# now for SSP370 File
#from netCDF4 import Dataset
# Open the dataset 
#dataset = Dataset('C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_2090-2100MEAN.nc', 'r')

# List all variables and attributes
#print(dataset.variables.keys())
#print(dataset.variables['BC_em_anthro'])
#print(dataset.variables['sector'])
#print(dataset.variables['sector_bnds'])

# the time variables needs to be condensed into a mean first -- time is 120 options, and need to regrid
#import xarray as xr

# Load the NetCDF file -- for Future simulation
#file_path = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_2090_2100MEAN.nc"
#ds = xr.open_dataset(file_path)
#BC_em_anthro = ds['BC_em_anthro']

# Calculate the mean across the `time` dimension
#mean_BC_em_anthro = BC_em_anthro.mean(dim="time", keep_attrs=True)

#mean_BC_em_anthro.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_MEAN.nc")

# ----------------------------------------------------------------------------------------------------------------
# grid dimensions are currently 360 lat x 720 lon
# re-gridding to 192 lat x 288 lon
#import xarray as xr
#import xesmf as xe
#import numpy as np

#file_path = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_2090_2100MEAN.nc"
#ds = xr.open_dataset(file_path)


# Create target grid (192 x 288)
#lat_target = xr.DataArray(
   # data=np.linspace(-90, 90, 192),
   # dims='lat',
    #name='lat')
#lon_target = xr.DataArray(
   # data=np.linspace(0, 360, 288, endpoint=False),  # 0 to 360 for 288 points
   ## dims='lon',
   # name='lon')
#target_grid = xr.Dataset({'lat': lat_target, 'lon': lon_target})

# Apply regridding
#regridder = xe.Regridder(ds, target_grid, method='bilinear')
#ds_regridded = regridder(ds)

# Save to a new NetCDF file
#ds_regridded.to_netcdf('C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_2090-2100MEAN_regridded_288x192.nc')

# this whole regridder didnt work, 
# instead it is necessary to use cdo remapbil 

# %% DATAFRAMES FOR QUICK NUMBERS VIZ IN TABLE --------------------------------------------------------------------------
# visualizing the BC_emiss data for each sector in separate pandas dataframes
import xarray as xr
import pandas as pd
import numpy as np

# Load the NetCDF file -- using newly regridded data
file_path = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_2000-2015MEAN_regridded_288x192.nc"
ds = xr.open_dataset(file_path)

# Extract Black Carbon (BC) emission values SSP370 
BC_em_anthro = ds['BC_em_anthro']  # Emission data
lon = ds['lon'].values           # Longitude values -- the .values extracts as a numPy array automatically, removes metadata
lat = ds['lat'].values            # Latitude values
sector = ds['sector'].values      # Sector identifiers

# Check
print("lon", lon)
print("lat", lat)
print("sector", sector) # looks right 

# dataframes are nice for quick viz but not great for manipulation and plotting, xarray is best 
# Create a dictionary to store dataframes for each sector
sector_dataframes = {}

# For loop to loop through individual sectors
for sec_idx, sec_value in enumerate(sector):
    # Filter data for the current sector index
    filtered_data = BC_em_anthro.isel(sector=sec_idx).squeeze()
    
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(filtered_data.values, index=lat, columns=lon)
    df.index.name = 'Latitude'
    df.columns.name = 'Longitude'
    
    # Store the DataFrame in the dictionary
    sector_dataframes[sec_value] = df

# Access the dataframe for a specific sector, e.g., sector at midpoint `sector_value`
BC_emiss_0 = sector[0]  # Adjust as needed
BC_emiss_0_df = sector_dataframes[BC_emiss_0]
# BC_emiss_0_df # pop out to jupyter notebook 

# Sector 1: Energy
BC_emiss_1 = sector[1]  
BC_emiss_1_df = sector_dataframes[BC_emiss_1]
# Sector 2: Industrial Coal
BC_emiss_2 = sector[2] 
BC_emiss_2_df = sector_dataframes[BC_emiss_2]
# Sector 3: Transportation
BC_emiss_3 = sector[3]  
BC_emiss_3_df = sector_dataframes[BC_emiss_3]
# Sector 4: Residential Coal
BC_emiss_4 = sector[4]  # Adjust as needed
BC_emiss_4_df = sector_dataframes[BC_emiss_4]
# Sector 5: Solvents, etc. 
BC_emiss_5 = sector[5]  # Adjust as needed
BC_emiss_5_df = sector_dataframes[BC_emiss_5]
# Sector 6: Waste
BC_emiss_6 = sector[6]  # Adjust as needed
BC_emiss_6_df = sector_dataframes[BC_emiss_6]
# Sector 7: International Shipping
BC_emiss_7 = sector[7]  # Adjust as needed
BC_emiss_7_df = sector_dataframes[BC_emiss_7]

# used this to do a quick check to make sure different values were coming up uniquely for each sector
from itables import show
# Show the DataFrame interactively
show(BC_emiss_0_df, scrollY=True, scrollX=True, maxRows=10, maxColumns=200) # looks good, data is dif between sectors
print(lon.shape, "lon dim")
print(lat.shape, "lat values")




# %% USING XARRAY TO MANIPULATE DATA WITHIN NETCDF FORMAT ----------------------
import xarray as xr
import pandas as pd
import numpy as np

# Load the NetCDF file -- using newly regridded data
file_path_SSP370 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_2090_2100MEAN_regridded_288x192_test.nc"
# DATASET THAT WAS REGRIDDED USING CDO WAS WHAT FIXED THE REGRIDDING ISSUE
ds_SSP370 = xr.open_dataset(file_path_SSP370)
#combined_Fe_emiss_PD = ds_SSP370.drop_vars(["OCNFRAC"])
ds_SSP370 = ds_SSP370.squeeze(dim='time') # only needed for PD

# Extract Black Carbon (BC) emission values SSP370 
BC_em_anthro_SSP370 = ds_SSP370['BC_em_anthro']  # Emission data
lon = ds_SSP370['lon'].values           # Longitude values -- the .values extracts as a numPy array automatically, removes metadata
lat = ds_SSP370['lat'].values            # Latitude values
sector = ds_SSP370['sector'].values      # Sector identifiers

values_at_lon_180 = BC_em_anthro_SSP370.sel(lon=180, method="nearest")
non_zero_values = values_at_lon_180.values[values_at_lon_180.values != 0]
print("Non-zero values at longitude 180:", non_zero_values)

# Adding OCNFRAC as variable to array
ds_ocnfrac = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\ClaquinMineralsCAM6_SPEWFUELS-Dec2023_OCNFRAC.nc")
ds_ocnfrac = ds_ocnfrac.squeeze(dim='time')
ocnfrac = ds_ocnfrac['OCNFRAC']  # Emission data

# trying using a for loop instead of directly indexing ocnfrac as a dimension -- either would have worked, the issue was the regridder 
# Assuming 'ocnfrac' is a DataArray from xarray and contains the data to filter
ocnfrac_values = ocnfrac.values  # Extract the raw values as a numpy array
# Create a DataFrame with latitudes as rows and longitudes as columns
ocnfrac_df = pd.DataFrame(ocnfrac_values, columns=ocnfrac['lon'].values, index=ocnfrac['lat'].values)

# Generate lists of indices where ocnfrac >= 0.5 and < 0.5
indices_gte_05 = []
indices_lt_05 = []

for lat_idx in range(ocnfrac_df.shape[0]):  
    for lon_idx in range(ocnfrac_df.shape[1]):  
        if ocnfrac_df.iloc[lat_idx, lon_idx] >= 0.5:
            indices_gte_05.append((lat_idx, lon_idx))
        else:
            indices_lt_05.append((lat_idx, lon_idx))

count_gte_05 = len(indices_gte_05) # indices where ocnfrac was >= 0.5
count_lt_05 = len(indices_lt_05) # indices where ocnfrac was < 0.5

# sum_ocnfrac_indices = count_gte_05 + count_lt_05 # quality assurance to make sure they added to original # of entries in array
#print(f"Indices where ocnfrac >= 0.5: {indices_gte_05[:5]}")
#print(f"Indices where ocnfrac < 0.5: {indices_lt_05[:5]}")
#print(f"Number of indices where ocnfrac >= 0.5: {count_gte_05}")
#print(f"Number of indices where ocnfrac < 0.5: {count_lt_05}")
#print(f"sum: {sum_ocnfrac_indices}") # adds to 55296 as expected, good sign
# passed all of the above tests 

BC_emiss_3_SSP370 = BC_em_anthro_SSP370.isel(sector=3) # Transportation

values_at_lon_180 = BC_em_anthro_SSP370.sel(lon=180, method="nearest")

# Filter for non-zero values
non_zero_values = values_at_lon_180.values[values_at_lon_180.values != 0]

# Print the non-zero values
print("Non-zero values at longitude 180:")
print(non_zero_values)

filtered_BC_emiss_3_SSP370 = np.full_like(BC_emiss_3_SSP370.values, np.nan)
for lat_idx, lon_idx in indices_lt_05:
    filtered_BC_emiss_3_SSP370[lat_idx, lon_idx] = BC_emiss_3_SSP370.values[lat_idx, lon_idx]
# Convert filtered array back to an xarray DataArray 
filtered_BC_emiss_3_SSP370 = xr.DataArray(filtered_BC_emiss_3_SSP370, 
                                           coords=[BC_emiss_3_SSP370['lat'], BC_emiss_3_SSP370['lon']], 
                                           dims=['lat', 'lon'])

# ensure NaNs add to number of true values in ocn fraction df
# nan_count = np.sum(np.isnan(filtered_BC_emiss_3_SSP370))
# print(f"Number of NaNs in the array: {nan_count}") # PASSED THIS TEST 

BC_emiss_7_SSP370 = BC_em_anthro_SSP370.isel(sector=7) # Shipping
filtered_BC_emiss_7_SSP370 = np.full_like(BC_emiss_7_SSP370.values, np.nan)
for lat_idx, lon_idx in indices_gte_05:
    filtered_BC_emiss_7_SSP370[lat_idx, lon_idx] = BC_emiss_7_SSP370.values[lat_idx, lon_idx]
# Convert filtered array back to an xarray DataArray 
filtered_BC_emiss_7_SSP370 = xr.DataArray(filtered_BC_emiss_7_SSP370, 
                                           coords=[BC_emiss_7_SSP370['lat'], BC_emiss_7_SSP370['lon']], 
                                           dims=['lat', 'lon'])


#nan_count = np.sum(np.isnan(filtered_BC_emiss_7_SSP370))
#print(f"Number of NaNs in the array: {nan_count}") # PASSED THIS TEST 

# Expand OCNFRAC to include the sector dimension
# ocnfrac_expanded_SSP370 = ocnfrac.expand_dims(dim={'sector': BC_em_anthro_SSP370['sector']}, axis=0)

# Add OCNFRAC to the BC_emiss dataset
#BC_em_anthro_SSP370['OCNFRAC'] = ocnfrac_expanded_SSP370
#BC_em_anthro_SSP370 = BC_em_anthro_SSP370.drop_vars(["time"])

# Specifying emission sector for each dataset
BC_emiss_1_SSP370 = BC_em_anthro_SSP370.isel(sector=1) # Energy
BC_emiss_2_SSP370 = BC_em_anthro_SSP370.isel(sector=2) # Industrial Coal
#BC_emiss_3_SSP370 = BC_em_anthro_SSP370.isel(sector=3) # Transportation
#filtered_BC_emiss_3_SSP370 = BC_emiss_3_SSP370.where(ocnfrac < 0.5)
BC_emiss_4_SSP370 = BC_em_anthro_SSP370.isel(sector=4) # Residential Coal
#BC_emiss_7_SSP370 = BC_em_anthro_SSP370.isel(sector=7) # Shipping
#filtered_BC_emiss_7_SSP370 = BC_emiss_7_SSP370.where(ocnfrac >= 0.5)

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Plot using xarray's built-in function
BC_emiss_3_SSP370.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
plt.title("NOT FILTERED")
plt.gca().coastlines()
plt.show() 

filtered_BC_emiss_3_SSP370.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
plt.title("TRANSPORT")
plt.gca().coastlines()
plt.show() # YAY! The land-ocean masking worked this way, checked both transport and shipping

filtered_BC_emiss_7_SSP370.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
plt.title("SHIPPING")
plt.gca().coastlines()
plt.show() # YAY! The land-ocean masking worked this way, checked both transport and shipping

#BC_emiss_1_SSP370.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
#plt.title("1: Energy")
#plt.gca().coastlines()
#plt.show() # plot looks right for each sector

#BC_emiss_2_SSP370.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
#plt.title("2: Industrial Coal")
#plt.gca().coastlines()
#plt.show() # plot looks right for each sector

#BC_emiss_3_SSP370.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
#plt.title("3: Transportation")
#plt.gca().coastlines()
#plt.show() # plot looks right for each sector

#BC_emiss_4_SSP370.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
#plt.title("4: Residential Coal")
#plt.gca().coastlines()
#plt.show() # plot looks right for each sector

#BC_emiss_7_SSP370.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
#plt.title("7: Shipping")
#plt.gca().coastlines()
#plt.show() # plot looks right for each sector

# creating composite of emission sources from different sectors for COALFLYASH variables
BC_COALFF_SSP370 = BC_emiss_1_SSP370 + BC_emiss_2_SSP370
BC_COALBF_SSP370 = BC_emiss_4_SSP370
BC_WOOD_SSP370 = BC_emiss_4_SSP370
BC_OIL_SSP370 = filtered_BC_emiss_3_SSP370.fillna(filtered_BC_emiss_7_SSP370) # replacing all NAs for transportation with values from shipping
#FSMELT WILL COME DIRECTLY WHEN I ADD IN THE PD DATA

#Check combination 
BC_OIL_SSP370.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
plt.title("BC_OIL_SSP370")
plt.gca().coastlines()
plt.show() # plot looks right for each sector

# removing variables I no longer need 
BC_COALFF_SSP370 = BC_COALFF_SSP370.drop_vars(['time','OCNFRAC'], errors= 'ignore')
BC_COALBF_SSP370 = BC_COALBF_SSP370.drop_vars(['time','OCNFRAC'], errors= 'ignore')
BC_WOOD_SSP370 = BC_WOOD_SSP370.drop_vars(['time','OCNFRAC'], errors= 'ignore')
BC_OIL_SSP370 = BC_OIL_SSP370.drop_vars(['time','OCNFRAC'], errors= 'ignore')

# --------------------------------------------------------------------------------------------------------------
# The same now but for the PD BC file 
file_path_PD = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_2000-2015MEAN_regridded_288x192_test.nc" #FOR PD
ds_PD = xr.open_dataset(file_path_PD)
ds_PD = ds_PD.squeeze(dim='time') # only needed for PD (SQUEEZE VS REMOVE MAY BE AN ISSUE?)

ds_PD['lon'] = ds_SSP370['lon'] # ensuring lats and lons are identical
ds_PD['lat'] = ds_SSP370['lat']

# Extract Black Carbon (BC) emission values PD 
BC_em_anthro_PD = ds_PD['BC_em_anthro']  # Emission data

# Expand OCNFRAC to include the sector dimension
ocnfrac_expanded_PD = ocnfrac.expand_dims(dim={'sector': BC_em_anthro_PD['sector']}, axis=0)
# Add OCNFRAC to the BC_emiss dataset
BC_em_anthro_PD['OCNFRAC'] = ocnfrac_expanded_PD

print("Longitude range of BC_em_anthro_PD:", BC_em_anthro_PD['lon'].values[0], BC_em_anthro_PD['lon'].values[-1])
print("Longitude range of ocnfrac:", ocnfrac['lon'].values[0], ocnfrac['lon'].values[-1])
print("Latitude range of BC_em_anthro_PD:", BC_em_anthro_PD['lat'].values[0], BC_em_anthro_PD['lat'].values[-1])
print("Latitude range of ocnfrac:", ocnfrac['lat'].values[0], ocnfrac['lat'].values[-1])
# now they are identical, let's see if this fixes the issue

BC_emiss_3_PD = BC_em_anthro_PD.isel(sector=3) # Transportation
filtered_BC_emiss_3_PD = np.full_like(BC_emiss_3_PD.values, np.nan)
for lat_idx, lon_idx in indices_lt_05:
    filtered_BC_emiss_3_PD[lat_idx, lon_idx] = BC_emiss_3_PD.values[lat_idx, lon_idx]
# Convert filtered array back to an xarray DataArray 
filtered_BC_emiss_3_PD = xr.DataArray(filtered_BC_emiss_3_PD, 
                                           coords=[BC_emiss_3_PD['lat'], BC_emiss_3_PD['lon']], 
                                           dims=['lat', 'lon'])

BC_emiss_7_PD = BC_em_anthro_PD.isel(sector=7) # Shipping
filtered_BC_emiss_7_PD = np.full_like(BC_emiss_7_PD.values, np.nan)
for lat_idx, lon_idx in indices_gte_05:
    filtered_BC_emiss_7_PD[lat_idx, lon_idx] = BC_emiss_7_PD.values[lat_idx, lon_idx]
# Convert filtered array back to an xarray DataArray 
filtered_BC_emiss_7_PD = xr.DataArray(filtered_BC_emiss_7_PD, 
                                           coords=[BC_emiss_7_PD['lat'], BC_emiss_7_PD['lon']], 
                                           dims=['lat', 'lon'])

# Specifying emission sector for each dataset
BC_emiss_1_PD = BC_em_anthro_PD.isel(sector=1) # Energy
BC_emiss_2_PD = BC_em_anthro_PD.isel(sector=2) # Industrial Coal
#BC_emiss_3_PD = BC_em_anthro_PD.isel(sector=3) # Transportation
#filtered_BC_emiss_3_PD = BC_emiss_3_PD.where(ocnfrac < 0.5)
BC_emiss_4_PD = BC_em_anthro_PD.isel(sector=4) # Residential Coal
#BC_emiss_7_PD = BC_em_anthro_PD.isel(sector=7) # Shipping
#filtered_BC_emiss_7_PD = BC_emiss_7_PD.where(ocnfrac >= 0.5)

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

filtered_BC_emiss_3_PD.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
plt.title("TRANSPORT")
plt.gca().coastlines()
plt.show() 

filtered_BC_emiss_7_PD.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
plt.title("SHIPPING")
plt.gca().coastlines()
plt.show() 

BC_COALFF_PD = BC_emiss_1_PD + BC_emiss_2_PD
BC_COALBF_PD = BC_emiss_4_PD
BC_WOOD_PD = BC_emiss_4_PD
BC_OIL_PD = filtered_BC_emiss_3_PD.fillna(filtered_BC_emiss_7_PD) 

#Check combination 
BC_OIL_PD.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
plt.title("BC_OIL_PD")
plt.gca().coastlines()
plt.show() # plot looks right for each sector

#print("before", BC_COALFF_PD) #test 
BC_COALFF_PD = BC_COALFF_PD.drop_vars(['time','OCNFRAC'], errors= 'ignore')
#print("after", BC_COALFF_PD) # above worked so only lat and lon are being considered 
BC_COALBF_PD = BC_COALBF_PD.drop_vars(['time','OCNFRAC'], errors= 'ignore')
BC_WOOD_PD = BC_WOOD_PD.drop_vars(['time','OCNFRAC'], errors= 'ignore')
BC_OIL_PD = BC_OIL_PD.drop_vars(['time','OCNFRAC'], errors= 'ignore')

#----------------------------------------------------------------------------------------------------------------
# now to read in present day simulations to determine future Fe from Fe:BC ratio
import xarray as xr
import numpy as np
import pandas as pd 

# load the netCDF file for Douglas' previous present day run
file_path_1 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\ClaquinMineralsCAM6_SPEWFUELS-Dec2023.nc"
ds_PD_Fe = xr.open_dataset(file_path_1)

ds_PD_Fe['lon'] = ds_SSP370['lon'] # ensuring lats and lons are identical
ds_PD_Fe['lat'] = ds_SSP370['lat']  

CoaMed_FCOALBF_PD = ds_PD_Fe['CoaMed_FCOALBF']  
FineMed_FCOALBF_PD = ds_PD_Fe['FineMed_FCOALBF']  
CoaMed_FCOALFF_PD = ds_PD_Fe['CoaMed_FCOALFF']  
FineMed_FCOALFF_PD = ds_PD_Fe['FineMed_FCOALFF']    
CoaMed_FOIL_PD= ds_PD_Fe['CoaMed_FOIL']  
FineMed_FOIL_PD = ds_PD_Fe['FineMed_FOIL']   
CoaMed_FSMELT_PD= ds_PD_Fe['CoaMed_FSMELT']  
FineMed_FSMELT_PD = ds_PD_Fe['FineMed_FSMELT'] 
CoaMed_FWOOD_PD= ds_PD_Fe['CoaMed_FWOOD']  
FineMed_FWOOD_PD = ds_PD_Fe['FineMed_FWOOD']  

# making sure size distribution is the same for all sectors -- it is in fact, not the same, so I need to find Coarse_Frac and Fine Frac for all emission types to apply to future sim
# calculating the fine and coarse fractionation from the PD run to apply to the future data
CoaFrac_FCOALBF = CoaMed_FCOALBF_PD/(CoaMed_FCOALBF_PD + FineMed_FCOALBF_PD)
CoaFrac_FCOALBF = CoaFrac_FCOALBF.fillna(0.0) # NaNs needed to be converted to 0.0
FineFrac_FCOALBF = FineMed_FCOALBF_PD/(CoaMed_FCOALBF_PD + FineMed_FCOALBF_PD)
FineFrac_FCOALBF = FineFrac_FCOALBF.fillna(0.0)

CoaFrac_FCOALFF = CoaMed_FCOALFF_PD/(CoaMed_FCOALFF_PD + FineMed_FCOALFF_PD)
CoaFrac_FCOALFF = CoaFrac_FCOALFF.fillna(0.0)
FineFrac_FCOALFF = FineMed_FCOALFF_PD/(CoaMed_FCOALFF_PD + FineMed_FCOALFF_PD)
FineFrac_FCOALFF = FineFrac_FCOALFF .fillna(0.0)

CoaFrac_FOIL = CoaMed_FOIL_PD/(CoaMed_FOIL_PD + FineMed_FOIL_PD)
CoaFrac_FOIL = CoaFrac_FOIL.fillna(0.0)
FineFrac_FOIL = FineMed_FOIL_PD/(CoaMed_FOIL_PD + FineMed_FOIL_PD)
FineFrac_FOIL = FineFrac_FOIL.fillna(0.0)

CoaFrac_FWOOD = CoaMed_FWOOD_PD/(CoaMed_FWOOD_PD + FineMed_FWOOD_PD)
CoaFrac_FWOOD = CoaFrac_FWOOD.fillna(0.0)
FineFrac_FWOOD = FineMed_FWOOD_PD/(CoaMed_FWOOD_PD + FineMed_FWOOD_PD)
FineFrac_FWOOD = FineFrac_FWOOD.fillna(0.0)

# get rid of time and OCNFRAC in BC arrays before I can directly manipulate with Fe data
# check to ensure dimensions are the same -- forced them all to be
#print("PD", BC_OIL_PD)
#print("SSP370", BC_OIL_SSP370)
#print("FINE", FineFrac_FOIL)
#print("COA", CoaFrac_FOIL)

#FineFrac_FOIL.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Fraction'})
#plt.title("FineFrac_FCOALBF")
#plt.gca().coastlines()
#plt.show() 

#CoaFrac_FOIL.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Fraction'})
#plt.title("CoaFrac_FCOALBF")
#plt.gca().coastlines()
#plt.show() 
# why is the latitude resolution wrong now?? 95 when it should be 192 - was due to floating point dif fixed now
# appears to be a lot of NaNs in the FineFrac -- should investigate why this is, due to emissions only being located on terrestrial cells, so replace NaNs (over ocean I assume, with zeros for multiplying through with BC)
# confirmed, much less NaNs for shipping because there are emissions over the sea! 

# separating BC into fine and coarse fraction based on specific grid cell ratios from Fe file 
CoaMed_BCCOALBF_SSP370 = CoaFrac_FCOALBF * BC_COALBF_SSP370 
FineMed_BCCOALBF_SSP370 = FineFrac_FCOALBF * BC_COALBF_SSP370 
total_BCCOALBF_SSP370 = CoaMed_BCCOALBF_SSP370 + FineMed_BCCOALBF_SSP370

CoaMed_BCCOALBF_PD = CoaFrac_FCOALBF * BC_COALBF_PD
FineMed_BCCOALBF_PD = FineFrac_FCOALBF * BC_COALBF_PD
total_BCCOALBF_PD = CoaMed_BCCOALBF_PD + FineMed_BCCOALBF_PD

CoaMed_BCCOALFF_SSP370 = CoaFrac_FCOALFF * BC_COALFF_SSP370 
FineMed_BCCOALFF_SSP370 = FineFrac_FCOALFF * BC_COALFF_SSP370 
total_BCCOALFF_SSP370 = CoaMed_BCCOALFF_SSP370  + FineMed_BCCOALFF_SSP370 

CoaMed_BCCOALFF_PD = CoaFrac_FCOALFF * BC_COALFF_PD 
FineMed_BCCOALFF_PD = FineFrac_FCOALFF * BC_COALFF_PD 
total_BCCOALFF_PD = CoaMed_BCCOALFF_PD + FineMed_BCCOALFF_PD

CoaMed_BCOIL_SSP370 = CoaFrac_FOIL * BC_OIL_SSP370
FineMed_BCOIL_SSP370 = FineFrac_FOIL* BC_OIL_SSP370
total_BCOIL_SSP370 = CoaMed_BCOIL_SSP370 + FineMed_BCOIL_SSP370

CoaMed_BCOIL_PD = CoaFrac_FOIL * BC_OIL_PD
FineMed_BCOIL_PD = FineFrac_FOIL * BC_OIL_PD
total_BCOIL_PD = CoaMed_BCOIL_PD + FineMed_BCOIL_PD

CoaMed_BCWOOD_SSP370 = CoaFrac_FWOOD * BC_WOOD_SSP370
FineMed_BCWOOD_SSP370 = FineFrac_FWOOD * BC_WOOD_SSP370
total_BCWOOD_SSP370 = CoaMed_BCWOOD_SSP370 + FineMed_BCWOOD_SSP370

CoaMed_BCWOOD_PD = CoaFrac_FWOOD * BC_WOOD_PD
FineMed_BCWOOD_PD = FineFrac_FWOOD * BC_WOOD_PD
total_BCWOOD_PD = CoaMed_BCWOOD_PD + FineMed_BCWOOD_PD

# ran this for every combination to check, if mean error was <e-14 assume rounding diff
print("OG", BC_WOOD_SSP370.mean().item())
print("Coa", CoaMed_BCWOOD_SSP370.mean().item())
print("Fine", FineMed_BCWOOD_SSP370.mean().item())
print("added", total_BCWOOD_SSP370.mean().item()) 
difference = BC_WOOD_SSP370 - total_BCWOOD_SSP370
print("Mean difference:", difference.mean().item()) 
# results for differences below
# COALBF_SSP370 = e-15
# COALBF_PD = e-15
# COALFF_SSP370 = e-15
# COALFF_PD = e-15
# OIL_SSP370 = e-16
# OIL_PD = e-16
# WOOD_SSP370 = e-15
# WOOD_PD = e-15

# Finally calculating Fe_SSP370 emission values 
CoaMed_FCOALBF_SSP370 = (CoaMed_FCOALBF_PD/CoaMed_BCCOALBF_PD) * CoaMed_BCCOALBF_SSP370
CoaMed_FCOALBF_SSP370 = CoaMed_FCOALBF_SSP370.fillna(0.0)
FineMed_FCOALBF_SSP370 = (FineMed_FCOALBF_PD/FineMed_BCCOALBF_PD) * FineMed_BCCOALBF_SSP370
FineMed_FCOALBF_SSP370 = FineMed_FCOALBF_SSP370.fillna(0.0)

CoaMed_FCOALFF_SSP370 = (CoaMed_FCOALFF_PD/CoaMed_BCCOALFF_PD) * CoaMed_BCCOALFF_SSP370
CoaMed_FCOALFF_SSP370 = CoaMed_FCOALFF_SSP370.fillna(0.0)
FineMed_FCOALFF_SSP370 = (FineMed_FCOALFF_PD/FineMed_BCCOALFF_PD) * FineMed_BCCOALFF_SSP370
FineMed_FCOALFF_SSP370 = FineMed_FCOALFF_SSP370.fillna(0.0)

CoaMed_FOIL_SSP370 = (CoaMed_FOIL_PD/CoaMed_BCOIL_PD) * CoaMed_BCOIL_SSP370
CoaMed_FOIL_SSP370 = CoaMed_FOIL_SSP370.fillna(0.0)
FineMed_FOIL_SSP370 = (FineMed_FOIL_PD/FineMed_BCOIL_PD) * FineMed_BCOIL_SSP370
FineMed_FOIL_SSP370 = FineMed_FOIL_SSP370.fillna(0.0)

CoaMed_FWOOD_SSP370 = (CoaMed_FWOOD_PD/CoaMed_BCWOOD_PD) * CoaMed_BCWOOD_SSP370
CoaMed_FWOOD_SSP370 = CoaMed_FWOOD_SSP370.fillna(0.0)
FineMed_FWOOD_SSP370 = (FineMed_FWOOD_PD/FineMed_BCWOOD_PD) * FineMed_BCWOOD_SSP370
FineMed_FWOOD_SSP370 = FineMed_FWOOD_SSP370.fillna(0.0)

CoaMed_FSMELT_SSP370 = CoaMed_FSMELT_PD
CoaMed_FSMELT_SSP370 = CoaMed_FSMELT_SSP370.fillna(0.0)
FineMed_FSMELT_SSP370 = FineMed_FSMELT_PD
FineMed_FSMELT_SSP370 = FineMed_FSMELT_SSP370.fillna(0.0)

combined_Fe_emiss_SSP370 = xr.Dataset({
    "CoaMed_FCOALBF": CoaMed_FCOALBF_SSP370,
    "FineMed_FCOALBF": FineMed_FCOALBF_SSP370,
    "CoaMed_FCOALFF": CoaMed_FCOALFF_SSP370,
    "FineMed_FCOALFF": FineMed_FCOALFF_SSP370,
    "CoaMed_FOIL": CoaMed_FOIL_SSP370,
    "FineMed_FOIL": FineMed_FOIL_SSP370,
    "CoaMed_FWOOD": CoaMed_FWOOD_SSP370,
    "FineMed_FWOOD": FineMed_FWOOD_SSP370,
    "CoaMed_FSMELT": CoaMed_FSMELT_SSP370,
    "FineMed_FSMELT": FineMed_FSMELT_SSP370,
})

combined_BC_emiss_SSP370 = xr.Dataset({
    "CoaMed_FCOALBF": CoaMed_BCCOALBF_SSP370,
    "FineMed_FCOALBF": FineMed_BCCOALBF_SSP370,
    "CoaMed_FCOALFF": CoaMed_BCCOALFF_SSP370,
    "FineMed_FCOALFF": FineMed_BCCOALFF_SSP370,
    "CoaMed_FOIL": CoaMed_BCOIL_SSP370,
    "FineMed_FOIL": FineMed_BCOIL_SSP370,
    "CoaMed_FWOOD": CoaMed_BCWOOD_SSP370,
    "FineMed_FWOOD": FineMed_BCWOOD_SSP370
})

combined_BC_emiss_PD = xr.Dataset({
    "CoaMed_FCOALBF": CoaMed_BCCOALBF_PD,
    "FineMed_FCOALBF": FineMed_BCCOALBF_PD,
    "CoaMed_FCOALFF": CoaMed_BCCOALFF_PD,
    "FineMed_FCOALFF": FineMed_BCCOALFF_PD,
    "CoaMed_FOIL": CoaMed_BCOIL_PD,
    "FineMed_FOIL": FineMed_BCOIL_PD,
    "CoaMed_FWOOD": CoaMed_BCWOOD_PD,
    "FineMed_FWOOD": FineMed_BCWOOD_PD
})

combined_Fe_emiss_SSP370 = combined_Fe_emiss_SSP370.drop_vars(["sector"])
print(combined_Fe_emiss_SSP370)
combined_Fe_emiss_SSP370.to_netcdf("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\Fe_Emissions_SSP370_coalflyash_HEP_7.nc")
# I think this looks great! Now to calculate budgets and compare future to PD emissions

combined_Fe_emiss_PD = ds_PD_Fe.drop_vars(["FracI","FracK","FracC1", "FracC2", "FracF", "FracG", "FracH1", "FracH2", "FracQ1", "FracQ2", "FracS", "Coar_InsolFe_Comb", "Coar_SolFe_Comb", "Fine_InsolFe_Comb", "Fine_SolFe_Comb", "Coar_InsolFe_Comb_UPDATED", "Coar_SolFe_Comb_UPDATED", "Fine_InsolFe_Comb_UPDATED", "Fine_SolFe_Comb_UPDATED"])

summed_variables_Fe_SSP370 = combined_Fe_emiss_SSP370.sum(dim=["lat", "lon"])
print("SSP370 Fe", summed_variables_Fe_SSP370)

summed_variables_Fe_PD = combined_Fe_emiss_PD.sum(dim=["lat", "lon"])
print("PD Fe", summed_variables_Fe_PD)

summed_variables_BC_SSP370 = combined_BC_emiss_SSP370.sum(dim=["lat", "lon"])
print("SSP370 BC", summed_variables_BC_SSP370)

summed_variables_BC_PD = combined_BC_emiss_PD.sum(dim=["lat", "lon"])
print("PD BC", summed_variables_BC_PD)

# COMPARING PD vs SSP370 PLOT
CoaMed_FOIL_SSP370.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'emission'})
plt.title("CoaFrac_FOIL_SSP370")
plt.gca().coastlines()
plt.show() 

CoaMed_FOIL_PD.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'emission'})
plt.title("CoaFrac_FOIL_PD")
plt.gca().coastlines()
plt.show() 

# next step is calculating emissions by grid cell -- this will allow me to determine if values I calculated make sense 
