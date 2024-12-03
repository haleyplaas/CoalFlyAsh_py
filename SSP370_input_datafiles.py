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
file_path = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_2000-2015MEAN_regridded_288x192.nc"
ds = xr.open_dataset(file_path)
ds = ds.squeeze(dim='time')

# Extract Black Carbon (BC) emission values SSP370 
BC_em_anthro = ds['BC_em_anthro']  # Emission data
lon = ds['lon'].values           # Longitude values -- the .values extracts as a numPy array automatically, removes metadata
lat = ds['lat'].values            # Latitude values
sector = ds['sector'].values      # Sector identifiers

# Adding OCNFRAC as variable to array
ds_ocnfrac = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI-PI-RESICOAL33-FIRE33.cam.h0.2010-05.nc")
ds_ocnfrac = ds_ocnfrac.squeeze(dim='time')
ocnfrac = ds_ocnfrac['OCNFRAC']  # Emission data

# Expand OCNFRAC to include the sector dimension
ocnfrac_expanded = ocnfrac.expand_dims(dim={'sector': BC_em_anthro['sector']}, axis=0)
# Add OCNFRAC to the BC_emiss dataset
BC_em_anthro['OCNFRAC'] = ocnfrac_expanded

print(BC_em_anthro) # confirmed OCNFRAC has been added

# Specifying emission sector for each dataset
BC_emiss_1 = BC_em_anthro.isel(sector=1) # Energy
BC_emiss_2 = BC_em_anthro.isel(sector=2) # Industrial Coal
BC_emiss_3 = BC_em_anthro.isel(sector=3) # Transportation
filtered_BC_emiss_3 = BC_emiss_3.where(ocnfrac < 0.5)
BC_emiss_4 = BC_em_anthro.isel(sector=4) # Residential Coal
BC_emiss_7 = BC_em_anthro.isel(sector=7) # Shipping
filtered_BC_emiss_7 = BC_emiss_7.where(ocnfrac >= 0.5)

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Plot using xarray's built-in function
#BC_emiss_3.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
#plt.title("NOT FILTERED")
#plt.gca().coastlines()
#plt.show() 

filtered_BC_emiss_3.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
plt.title("TRANSPORT")
plt.gca().coastlines()
plt.show() # YAY! The land-ocean masking worked this way, checked both transport and shipping

filtered_BC_emiss_7.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
plt.title("SHIPPING")
plt.gca().coastlines()
plt.show() # YAY! The land-ocean masking worked this way, checked both transport and shipping

#BC_emiss_1.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
#plt.title("1: Energy")
#plt.gca().coastlines()
#plt.show() # plot looks right for each sector

#BC_emiss_2.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
#plt.title("2: Industrial Coal")
#plt.gca().coastlines()
#plt.show() # plot looks right for each sector

#BC_emiss_3.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
#plt.title("3: Transportation")
#plt.gca().coastlines()
#plt.show() # plot looks right for each sector

#BC_emiss_4.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
#plt.title("4: Residential Coal")
#plt.gca().coastlines()
#plt.show() # plot looks right for each sector

#BC_emiss_7.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
#plt.title("7: Shipping")
#plt.gca().coastlines()
#plt.show() # plot looks right for each sector

# ------------------------------------------------------------------------------------------------------------------
# creating composite of emission sources from different sectors for COALFLYASH variables
BC_COALFF_SSP370 = BC_emiss_1 + BC_emiss_2
BC_COALBF_SSP370 = BC_emiss_4
BC_WOOD_SSP370 = BC_emiss_4
BC_OIL_SSP370 = filtered_BC_emiss_3.fillna(filtered_BC_emiss_7) # replacing all NAs for transportation with values from shipping
#FSMELT WILL COME DIRECTLY WHEN I ADD IN THE PD DATA

#Check combination 
BC_OIL_SSP370.plot(subplot_kws={'projection': ccrs.PlateCarree()},transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Emissions'})
plt.title("BC_OIL_SSP370")
plt.gca().coastlines()
plt.show() # plot looks right for each sector


# -----------------------------------------------------------------------------------------------------------------
# now to read in present day simulations to determine future Fe from Fe:BC ratio
import xarray as xr
import numpy as np
import pandas as pd 

# load the netCDF file for Douglas' previous present day run
file_path_1 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\ClaquinMineralsCAM6_SPEWFUELS-Dec2023.nc"
ds_PD = xr.open_dataset(file_path_1)

# should be identical
# lon = ds_PD['lon']        
# lat = ds_PD['lat']   

CoaMed_FCOALBF_PD = ds_1['CoaMed_FCOALBF']  
FineMed_FCOALBF_PD = ds_1['FineMed_FCOALBF']  
CoaMed_FCOALFF_PD = ds_1['CoaMed_FCOALFF']  
FineMed_FCOALFF_PD = ds_1['FineMed_FCOALFF']    
CoaMed_FOIL_PD= ds_1['CoaMed_FOIL']  
FineMed_FOIL_PD = ds_1['FineMed_FOIL']   
CoaMed_FSMELT_PD= ds_1['CoaMed_FSMELT']  
FineMed_FSMELT_PD = ds_1['FineMed_FSMELT'] 
CoaMed_FWOOD_PD= ds_1['CoaMed_FWOOD']  
FineMed_FWOOD_PD = ds_1['FineMed_FWOOD']  

# making sure size distribution is the same for all sectors -- it is in fact, not the same, so I need to find Coarse_Frac and Fine Frac for all emission types to apply to future sim
# calculating the fine and coarse fractionation from the PD run to apply to the future data
CoaFrac_FCOALBF = CoaMed_FCOALBF_PD/(CoaMed_FCOALBF_PD + FineMed_FCOALBF_PD)
print("Coa_frac", CoaFrac_FCOALBF)
avg_CoaFrac_COALBF = CoaFrac_FCOALBF.mean()
print("Coa_frac_mean", avg_CoaFrac_COALBF)

FineFrac_FCOALBF = FineMed_FCOALBF_PD/(CoaMed_FCOALBF_PD + FineMed_FCOALBF_PD)

CoaFrac_FCOALFF = CoaMed_FCOALFF_PD/(CoaMed_FCOALFF_PD + FineMed_FCOALFF_PD)
FineFrac_FCOALFF = FineMed_FCOALFF_PD/(CoaMed_FCOALFF_PD + FineMed_FCOALFF_PD)

CoaFrac_FOIL = CoaMed_FOIL_PD/(CoaMed_FOIL_PD + FineMed_FOIL_PD)
FineFrac_FOIL = FineMed_FOIL_PD/(CoaMed_FOIL_PD + FineMed_FOIL_PD)

CoaFrac_FWOOD = CoaMed_FWOOD_PD/(CoaMed_FWOOD_PD + FineMed_FWOOD_PD)
FineFrac_FWOOD = FineMed_FWOOD_PD/(CoaMed_FWOOD_PD + FineMed_FWOOD_PD)

# Multiplying through by coarse:fine fraction and Fe
CoaMed_FCOALBF_SSP370 = CoaFrac_FCOALBF * BC_COALBF_SSP370 * (CoaMed_FCOALBF_PD/CoaMed_BCCOALBF_PD)
FineMed_FCOALBF_SSP370 = FineFrac_FCOALBF * BC_COALBF_SSP370 

CoaMed_FCOALFF_SSP370 = CoaFrac_FCOALFF * BC_COALFF_SSP370 
FineMed_FCOALFF_SSP370 = FineFrac_FCOALFF * BC_COALFF_SSP370 

CoaMed_FOIL_SSP370 = CoaFrac_FCOALFF * BC_COALFF_SSP370 
FineMed_FOIL_SSP370 = FineFrac_FCOALFF * BC_COALFF_SSP370 


# %% 

# converting to df may not be what I ultimately want to do but for now it is nice to be able to visualize everything
# save this code if helpful
FCOALBF_PD_df = pd.DataFrame(FCOALBF_PD, index=lat, columns=lon)
from itables import show
show(FCOALBF_PD_df, scrollY=True, scrollX=True, maxRows=200, maxColumns=10) # looks good 

FCOALFF_PD_df = pd.DataFrame(FCOALFF_PD, index=lat, columns=lon)
FOIL_PD_df = pd.DataFrame(FOIL_PD, index=lat, columns=lon)
FSMELT_PD_df = pd.DataFrame(FSMELT_PD, index=lat, columns=lon)
FWOOD_PD_df = pd.DataFrame(FWOOD_PD, index=lat, columns=lon)

