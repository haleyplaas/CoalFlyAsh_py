#%%
#  SPATIAL REPRESENTATION OF CHANGES TO SOLUBLE IRON DEPOSITION FLUXES
import xarray as xr
import numpy as np 
from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

#sim_directory ="C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL0.2-WOOD10-OIL38_2009-2011.h1.BBx1_ANx2_DUx2.nc"

# Dust version Cheyenne (NO SOIL STATE)
#sim_directory ="D:\\CoalFlyAsh\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL0.2-WOOD10-OIL38_2009-2011_DEPMEAN.nc"
#MIMI_ds = xr.open_dataset(sim_directory)
#MIMI_ds = MIMI_ds.squeeze("time")

# Dust version Derecho (SOIL STATE)
#sim_directory = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL0.2-WOOD10-OIL38.cam.h1.2009-2011_PD.nc"
sim_directory = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Transient Run\\Test outputs\\CAM6-MIMI-PD-test-20250129_noSoilState.cam.h1.2008-JAN.nc"
MIMI_ds = xr.open_dataset(sim_directory)
MIMI_ds = MIMI_ds.median(dim="time")

lon = MIMI_ds['lon'].values  # Longitude in degrees
lat = MIMI_ds['lat'].values  # Latitude in degrees

FESOLDRY_var_MIMI = MIMI_ds["FETOTDRY"] # Dust specific 
FESOLWET_var_MIMI = MIMI_ds["FETOTWET"] # Dust specific
FESOLDEP_MIMI = (FESOLDRY_var_MIMI + FESOLWET_var_MIMI)*4.45

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

# Create a figure and axis
fig = plt.figure(figsize=(10, 8), edgecolor='w')
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

# Set the global map view
ax.set_global()

# Set the background color and add coastlines
ax.set_facecolor('white')
ax.coastlines(resolution='110m', color='black')

# Add gridlines with labels
ax.gridlines(draw_labels=True, linewidth=0.5, color='black', alpha=0.5)

# Define your custom color map
color1 = '#f7e2bf'
color2 = '#cc9f3e'
color3 = '#c4532f'
positions = [0.0, 0.5, 1.0]
colors = [color1, color2, color3]
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

# Assuming FESOLDEP_v3 is a 2D array or pandas DataFrame
# Plot the data with specified vmin and vmax
vmin = 1e-20
vmax = 5e-11

# If FESOLDEP_v3 is a pandas DataFrame with lat/lon, make sure to transform it correctly:
# Plot the data with pcolormesh
c = ax.pcolormesh(lon, lat, FESOLDEP_MIMI.values, cmap=custom_cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

# Add a colorbar
plt.colorbar(c, ax=ax, orientation='horizontal', pad=0.05, aspect=50)

# Display the plot
plt.show()


#%% MAPS COMPARING BASELINE TO V2-V3, BY SHOWING REGIONAL INCREASES
import xarray as xr
import numpy as np 
from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

# V1
sim_directory = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL0.2-WOOD10-OIL38.cam.h1.2009-2011_PD.nc"
MIMI_ds = xr.open_dataset(sim_directory)
MIMI_ds = MIMI_ds.mean(dim="time")

lon = MIMI_ds['lon'].values  # Longitude in degrees
lat = MIMI_ds['lat'].values  # Latitude in degrees

DRY_MIMI = MIMI_ds["FEANSOLDRY"] 
WET_MIMI = MIMI_ds["FEANSOLWET"] 
FEANSOLDEP_MIMI = (DRY_MIMI + WET_MIMI)

# V2
sim_directory = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL33-WOOD10-OIL38.cam.h1.2009-2011_PD.nc"
v2_ds = xr.open_dataset(sim_directory)
v2_ds = v2_ds.mean(dim="time")

lon = v2_ds['lon'].values  # Longitude in degrees
lat = v2_ds['lat'].values  # Latitude in degrees

DRY_v2 = v2_ds["FEANSOLDRY"] 
WET_v2 = v2_ds["FEANSOLWET"] 
FEANSOLDEP_v2 = (DRY_v2 + WET_v2)

# V3
sim_directory = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL33-WOOD56-OIL38.cam.h1.2009-2011_PD.nc"
v3_ds = xr.open_dataset(sim_directory)
v3_ds = v3_ds.mean(dim="time")

lon = v3_ds['lon'].values  # Longitude in degrees
lat = v3_ds['lat'].values  # Latitude in degrees

DRY_v3 = v3_ds["FEANSOLDRY"] 
WET_v3 = v3_ds["FEANSOLWET"] 
FEANSOLDEP_v3 = (DRY_v3 + WET_v3)

# V4
sim_directory = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL33-WOOD56-OIL38.cam.h1.2009-2011_PD.nc"
v4_ds = xr.open_dataset(sim_directory)
v4_ds = v4_ds.mean(dim="time")

lon = v4_ds['lon'].values  # Longitude in degrees
lat = v4_ds['lat'].values  # Latitude in degrees

DRY_v4 = v4_ds["FEANSOLDRY"] 
WET_v4 = v4_ds["FEANSOLWET"] 
FEANSOLDEP_v4 = (DRY_v4 + WET_v4)

diff_1 = FEANSOLDEP_v2 - FEANSOLDEP_MIMI 
diff_2 = FEANSOLDEP_v3 - FEANSOLDEP_MIMI 
diff_3= FEANSOLDEP_v4 - FEANSOLDEP_MIMI 

print("Min:", diff_3.min().values) # e-19
print("Max:", diff_3.max().values) # e-12

#%% Using a t-test to compare means for each lat/lon pairing
import xarray as xr
import numpy as np
from scipy import stats

# Example: Assume these are your two xarray DataArrays with dims: ('time', 'lat', 'lon')
# Let's call them da_A and da_B. They must share the same dimensions.
# For demonstration, assume they are already loaded.
ds_A = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL33-WOOD56-OIL38.cam.h1.2009-2011_PD.nc")
ds_A = ds_A.mean(dim="time")
#ds_A = ds_A[[var for var, da in ds_A.data_vars.items()
            #    if set(['time', 'lat', 'lon']).issubset(da.dims)]]

ds_B = xr.open_dataset("C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.05-RESICOAL33-WOOD56-OIL25.cam.h1.2009-2011_PD.nc")
ds_B = ds_B.mean(dim="time")
#ds_B = ds_B[[var for var, da in ds_B.data_vars.items()
        #        if set(['time', 'lat', 'lon']).issubset(da.dims)]]

lon = ds_A['lon'].values  # Longitude in degrees
lat = ds_A['lat'].values  # Latitude in degrees

DRY_MIMI_A = ds_A["FEANSOLDRY"] 
WET_MIMI_A = ds_A["FEANSOLWET"] 
FEANSOLDEP_MIMI_A = (DRY_MIMI_A + WET_MIMI_A)

DRY_MIMI_B = ds_B["FEANSOLDRY"] 
WET_MIMI_B = ds_B["FEANSOLWET"] 
FEANSOLDEP_MIMI_B = (DRY_MIMI_B + WET_MIMI_B)


diff_sims = ((FEANSOLDEP_MIMI_B - FEANSOLDEP_MIMI_A) / FEANSOLDEP_MIMI_A)*100

# Define a function that takes two 1D arrays (time series) and performs a paired t-test.
#def paired_ttest(series_A, series_B):
    # Remove any potential NaNs
  #  mask = np.isfinite(series_A) & np.isfinite(series_B)
 #   if np.sum(mask) < 2:
  #      return np.nan  # Not enough data
  #  t_stat, p_val = stats.ttest_rel(series_A[mask], series_B[mask])
   # return p_val

# Use xarray's apply_ufunc to apply the function over lat/lon:
#p_values = xr.apply_ufunc(
   # paired_ttest,
   # ds_A, ds_B,
    #input_core_dims=[['time'], ['time']],  # these are the dimensions to apply over
    #vectorize=True,
   # dask="parallelized",  # if you are using dask arrays
    #output_dtypes=[float]
#)

# p_values is a DataArray with dims ('lat', 'lon') containing the p-value at each grid cell.
# Now extract the indices where p < 0.05. For example, you might create a mask:
#significant_mask = p_values < 0.05

#print(significant_mask)
#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

# Create a figure and axis
fig = plt.figure(figsize=(10, 8), edgecolor='w')
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

# Set the global map view
ax.set_global()

# Set the background color and add coastlines
ax.set_facecolor('white')
ax.coastlines(resolution='110m', color='black')

# Add gridlines with labels
ax.gridlines(draw_labels=True, linewidth=0.5, color='black', alpha=0.5)

# Define your custom color map
#color1 = '#f7e2bf'
#color2 = '#cc9f3e'
#color3 = '#c4532f'
#positions = [0.0, 0.5, 1.0]
#colors = [color1, color2, color3]

color_neg10 = '#0c2340'  # Deep navy
color_neg9 = '#142d56'  # Darker navy blue
color_neg8 = '#1d3b6f'  # Dark blue
color_neg7 = '#26508b'  # Muted royal blue
color_neg6 = '#2f599b'  # Medium blue
color_neg5 = '#3a6fb0'  # Slightly lighter blue
color_neg4 = '#4882c2'  # Lighter blue
color_neg3 = '#5c9ad3'  # Soft sky blue
color_neg2 = '#6aa7d4'  # Softer blue
color_neg1 = '#94c3df'  # Muted sky blue

color0 = '#fdf3e6'  # Very light warm beige

color1 = '#f7e2bf'  # Light warm beige
color2 = '#cc9f3e'  # Warm mustard
color3 = '#b97832'  # Deep golden brown
color4 = '#c4532f'  # Burnt orange-red
color5 = '#a03d24'  # Earthy brick red
color6 = '#7d2a1a'  # Deep rust red
color7 = '#681f14'  # Dark rust red
color8 = '#5a1c11'  # Dark red-brown
color9 = '#4d170e'  # Very dark reddish-brown
color10 = '#3e120b'  # Near-black warm brown
color11 = '#2e0d08'  # Deep blackened red
color12 = 'black'  # Deep blackened red

# Updated palette and positions:
#positions = [0.0, 0.07, 0.14, 0.21, 0.28, 0.35, 0.42, 0.49, 0.56, 0.63, 0.70, 0.77, 0.90, 0.98, 1.0]
#colors = [color_neg7, color_neg6, color_neg5, color_neg4, color_neg3, color_neg2, color_neg1, 
       #   color1, color2, color3, color4, color5, color6, color7, color8]

positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]  # Now 10 positions
colors = [color_neg10, color_neg9, color_neg8, color_neg7, color_neg6, color_neg5, color_neg4, color_neg3, color_neg2, color1]
  # 10 colors
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

# Assuming FESOLDEP_v3 is a 2D array or pandas DataFrame
# Plot the data with specified vmin and vmax
vmin = -2e1
vmax = 1

# Define discrete boundaries for the color bar
boundaries = np.linspace(vmin, vmax, num=len(colors) + 1)
norm = BoundaryNorm(boundaries, custom_cmap.N)  # Assign boundaries to the colormap

# If FESOLDEP_v3 is a pandas DataFrame with lat/lon, make sure to transform it correctly:
# Plot the data with pcolormesh
c = ax.pcolormesh(lon, lat, diff_sims.values, cmap=custom_cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

# Add a colorbar
#plt.colorbar(c, ax=ax, orientation='horizontal', pad=0.05, aspect=50)

# Display the plot
#plt.show()

#ax.set_extent([100, 150, 10, 50], crs=ccrs.PlateCarree()) # zooming in on China

# Add a colorbar
cax = ax.imshow(diff_sims, cmap=custom_cmap, norm=norm)
cbar = plt.colorbar(cax, orientation='horizontal', ticks=boundaries, spacing='proportional')

# Display the plot
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator

# Define positions for custom colormap
positions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.5, 0.75, 1.0]
colors = [color1, color2, color3, color4, color5, color6, color7, color8, color8]
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

# Define discrete boundaries for the color bar
boundaries = np.linspace(vmin, vmax, num=len(colors) + 1)  # Equal spacing
norm = BoundaryNorm(boundaries, custom_cmap.N)  # Assign boundaries to the colormap

# Plot
fig, ax = plt.subplots()
cax = ax.imshow(diff_sims, cmap=custom_cmap, norm=norm)

# Add color bar with segmentation
cbar = plt.colorbar(cax, ticks=boundaries, spacing='proportional')
cbar.ax.set_yticklabels([f'{b:.1e}' for b in boundaries])  # Format labels in scientific notation

plt.show()
