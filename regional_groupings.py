# %%
# REGIONS V2.5
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon
import pandas as pd

# Read in observational data (excel file) with specified parameters
data_file = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\FeObs_Hamilton2022.xlsx"

# Define NaN values
na_values = ["-99", "-9999", "-99.0", "-0.99"]

# Read the Excel file
obs_data = pd.read_excel(data_file, sheet_name="FeObs_Hamilton2021", header=0, na_values=na_values)

# Define regions with corresponding colors
regions = {
    "Arabian Sea": {"lon_range": (30, 78), "lat_range": (-15, 30), "color": "#2d519280"},
    "ARCT": {"lon_range": (0, 360), "lat_range": (59, 90), "color": "#1B9E7780"},
    "AUSP": {"lon_range": (135, 295), "lat_range": (-45, -15), "color": "#66666680"},
    "Bay of Bengal": {"lon_range": (78, 100), "lat_range": (-15, 30), "color": "#D95F0280"},
    "CPAO": {"lon_range": (-210, 30), "lat_range": (-15, 30), "color": "#66A61E80"},
    "NATL": {"lon_range": (-95, 100), "lat_range": (30, 60), "color": "#7570B380"},
    "NPAC": {"lon_range": (150, 265), "lat_range": (30, 60), "color": "#E6AB0280"},
    "SATL": {"lon_range": (-65, 45), "lat_range": (-45, -15), "color": "#E7298A80"},
    "SEAS": {"lon_range": (100, 150), "lat_range": (-15, 60), "color": "#A6761D80"},
    "SIND": {"lon_range": (45, 135), "lat_range": (-45, -15), "color": "#FF7F0080"},
    "SO": {"lon_range": (0, 360), "lat_range": (-90, -45), "color": "#1F78B480"}
}

# not sure why these values do not match the values in R for the histograms... just plot the histograms in R 

def draw_map(ax, scale=0.2):
    ax.set_facecolor('white')

    # Fill continents with black color
    ax.add_feature(cfeature.LAND, facecolor='black', zorder=2)
    
    ax.coastlines(resolution='110m', color='black')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='black', alpha=0.5)

    # Fill each region with its respective color
    for region, bounds in regions.items():
        lon_range = bounds["lon_range"]
        lat_range = bounds["lat_range"]
        color = bounds["color"]
            
        ax.add_patch(Polygon([(lon_range[0], lat_range[0]),
                              (lon_range[1], lat_range[0]),
                              (lon_range[1], lat_range[1]),
                              (lon_range[0], lat_range[1])],
                             edgecolor='none', facecolor=color, transform=ccrs.PlateCarree()))

# Create a figure and axis
fig = plt.figure(figsize=(8, 6), edgecolor='w')
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

# Set the central latitude and longitude
ax.set_global()

# Draw the map
draw_map(ax)

# Dictionary to store the count of points plotted in each region
point_counts = {}

# Plot each latitude and longitude point from obs_data
for region, bounds in regions.items():
    lon_range = bounds["lon_range"]
    lat_range = bounds["lat_range"]
    color = bounds["color"]
    
    # Filter the data points within the region
    region_data = obs_data[(obs_data["Longitude"] >= lon_range[0]) & (obs_data["Longitude"] <= lon_range[1]) &
                           (obs_data["Latitude"] >= lat_range[0]) & (obs_data["Latitude"] <= lat_range[1])]
    
    # Plot the points
    ax.scatter(region_data["Longitude"], region_data["Latitude"], color='white', marker='.', s=6, transform=ccrs.Geodetic())
    
    # Store the count of points plotted for this region
    point_counts[region] = len(region_data)

# Display the count of points plotted in each region
print("Number of points plotted in each region:")
for region, count in point_counts.items():
    print(f"{region}: {count}")

plt.show()

import matplotlib.pyplot as plt

# Define colors for each region
region_colors = [regions[region]["color"] for region in point_counts.keys()]

# Plot histogram horizontally
plt.figure(figsize=(10, 6))
plt.barh(list(point_counts.keys()), list(point_counts.values()), color=region_colors)
plt.xlabel('Number of Observations', fontsize=14)
plt.ylabel('', fontsize=14)
plt.gca().invert_yaxis()  # Invert y-axis to display regions from top to bottom
plt.grid(axis='x')  # Add gridlines only along the x-axis

# Increase font size of tick labels
plt.tick_params(axis='both', which='major', labelsize=20)

# Remove spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Show histogram
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib.colors import LinearSegmentedColormap

# Path to NetCDF file
sim = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL0.2-WOOD10-OIL38.cam.h1.2009-2011_PD.nc"
# Open the NetCDF file
model_data_x = Dataset(sim)

# Extract the required variables
all_long_options = model_data_x.variables['lon'][:]
all_lat_options = model_data_x.variables['lat'][:]
wet_dep = model_data_x.variables['FESOLWET'][:]
dry_dep = model_data_x.variables['FESOLDRY'][:]
tot_dep = dry_dep + wet_dep

model_data_x.close()

# Create a figure and axis
fig = plt.figure(figsize=(10, 8), edgecolor='w')
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

# Set the central latitude and longitude
ax.set_global()

# Set face color and add features
ax.set_facecolor('white')
#ax.add_feature(cfeature.LAND, facecolor='black', zorder=2)
ax.coastlines(resolution='110m', color='black')
ax.gridlines(draw_labels=True, linewidth=0.5, color='black', alpha=0.5)

# Plot the variable_sol data
# Assuming variable_sol is a 2D array with dimensions corresponding to latitude and longitude
lon, lat = np.meshgrid(all_long_options, all_lat_options)

# Define your custom color map
color1 = '#f7e2bf'
color2 = '#cc9f3e'
color3 = '#c4532f'
positions = [0.0, 0.5, 1.0]
colors = [color1, color2, color3]
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

# Plot the data with specified vmin and vmax
vmin = 1e-20
vmax = 1e-13
c = ax.pcolormesh(lon, lat, tot_dep[0, :, :], cmap=custom_cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

# Add a colorbar
plt.colorbar(c, ax=ax, orientation='horizontal', pad=0.05, aspect=50)

# Display the map
plt.show()

# Now you can print the head of the arrays
print("Longitude head:", all_long_options[:5])
print("Latitude head:", all_lat_options[:5])


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib.colors import LinearSegmentedColormap

# Present Day Simulations
# Path to NetCDF file
sim_v1 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL0.2-WOOD10-OIL38.cam.h1.2009-2011_PD.nc"
# Open the NetCDF file
model_data = Dataset(sim_v1)

# Extract the required variables
all_long_options = model_data.variables['lon'][:]
all_lat_options = model_data.variables['lat'][:]
wet_dep_v1 = model_data.variables['FESOLWET'][:]
dry_dep_v1 = model_data.variables['FESOLDRY'][:]
tot_dep_median_v1 = np.median(tot_dep_v1, axis=0)
tot_dep_v1 = dry_dep_v1 + wet_dep_v1

model_data.close()

# v2
sim_v2 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL33-WOOD10-OIL38.cam.h1.2009-2011_PD.nc"
model_data_v2 = Dataset(sim_v2)
wet_dep_v2 = model_data_v2.variables['FESOLWET'][:]
dry_dep_v2 = model_data_v2.variables['FESOLDRY'][:]
tot_dep_v2 = dry_dep_v2 + wet_dep_v2
tot_dep_median_v2 = np.median(tot_dep_v2, axis=0)
model_data_v2.close()

# v3
sim_v3 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.2-RESICOAL33-WOOD56-OIL38.cam.h1.2009-2011_PD.nc"
model_data_v3 = Dataset(sim_v3)
wet_dep_v3 = model_data_v3.variables['FESOLWET'][:]
dry_dep_v3 = model_data_v3.variables['FESOLDRY'][:]
tot_dep_v3 = dry_dep_v3 + wet_dep_v3
tot_dep_median_v3 = np.median(tot_dep_v3, axis=0)
model_data_v3.close()

# v4
sim_v4 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.05-RESICOAL33-WOOD56-OIL25.cam.h1.2009-2011_PD.nc"
model_data_v4 = Dataset(sim_v4)
wet_dep_v4 = model_data_v4.variables['FESOLWET'][:]
dry_dep_v4 = model_data_v4.variables['FESOLDRY'][:]
tot_dep_v4 = dry_dep_v4 + wet_dep_v4
tot_dep_median_v4 = np.median(tot_dep_v4, axis=0)
model_data_v4.close()

# v5
sim_v5 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI_2010CLIMO_INDCOAL0.05-RESICOAL33-WOOD56-OIL25-FIREFINE56.cam.h1.2009-2011_PD.nc"
model_data_v5 = Dataset(sim_v5)
wet_dep_v5 = model_data_v5.variables['FESOLWET'][:]
dry_dep_v5 = model_data_v5.variables['FESOLDRY'][:]
tot_dep_v5 = dry_dep_v5 + wet_dep_v5
tot_dep_median_v5 = np.median(tot_dep_v5, axis=0)
model_data_v5.close()

# Preindustrial simulations
# v6
sim_v6 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI-PI-RESICOAL0.2-FIRE33.cam.h1.2009-2011_PI.nc"
model_data_v6 = Dataset(sim_v6)
wet_dep_v6 = model_data_v6.variables['FESOLWET'][:]
dry_dep_v6 = model_data_v6.variables['FESOLDRY'][:]
tot_dep_v6 = dry_dep_v6 + wet_dep_v6
tot_dep_median_v6 = np.median(tot_dep_v6, axis=0)
model_data_v6.close()

# v7
sim_v7 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI-PI-RESICOAL0.2-FIRE56.cam.h1.2009-2011_PI.nc"
model_data_v7 = Dataset(sim_v7)
wet_dep_v7 = model_data_v7.variables['FESOLWET'][:]
dry_dep_v7 = model_data_v7.variables['FESOLDRY'][:]
tot_dep_v7 = dry_dep_v7 + wet_dep_v7
tot_dep_median_v7 = np.median(tot_dep_v7, axis=0)
model_data_v7.close()

# v8
sim_v8 = "C:\\Users\\heplaas\\OneDrive - North Carolina State University\\Collaborations\\Coal Fly Ash\\data\\CAM6-MIMI-PI-RESICOAL33-FIRE33.cam.h1.2009-2011_PI.nc"
model_data_v8 = Dataset(sim_v8)
wet_dep_v8 = model_data_v8.variables['FESOLWET'][:]
dry_dep_v8 = model_data_v8.variables['FESOLDRY'][:]
tot_dep_v8 = dry_dep_v8 + wet_dep_v8
tot_dep_median_v8 = np.median(tot_dep_v8, axis=0)
model_data_v8.close()

# determining percent changes for each simulation from base
a_percent_change_MIMI_v2 = (tot_dep_median_v2-tot_dep_median_v1)/tot_dep_median_v1 # see change in resi coal
b_percent_change_v2_v3 = (tot_dep_median_v3-tot_dep_median_v2)/tot_dep_median_v2 # see change in wood
c_percent_change_MIMI_v3 = (tot_dep_median_v3-tot_dep_median_v1)/tot_dep_median_v1 # see change in wood + resicoal
d_percent_change_v3_v4 = (tot_dep_median_v4-tot_dep_median_v3)/tot_dep_median_v3 # see change in oil/indcoal
e_percent_change_MIMI_v4 = (tot_dep_median_v4-tot_dep_median_v1)/tot_dep_median_v1 # see change in wood + resicoal + oil/indcoal
f_percent_change_v4_v5 = (tot_dep_median_v5-tot_dep_median_v4)/tot_dep_median_v4 # see change in finefire
g_percent_change_MIMI_v5 = (tot_dep_median_v5-tot_dep_median_v1)/tot_dep_median_v1 # see change in wood + resicoal + oil/indcoal + finefire

# Define your custom color map
color1 = '#3E5DC6'
color2 = '#5B99E6'
color3 = '#f7e2bf'
color4 = '#cc9f3e'
color5 = '#c4532f'
positions = [0.0, 0.25, 0.5, 0.75, 1.0]
colors = [color1, color2, color3, color4, color5]
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

# Define vmin and vmax for the color map
vmin = -1.5
vmax = 1.5

# Create a function to plot each figure
def plot_figure(data, title):
    fig = plt.figure(figsize=(10, 8), edgecolor='w')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # Set the central latitude and longitude
    ax.set_global()

    # Set face color and add features
    ax.set_facecolor('white')
    ax.coastlines(resolution='110m', color='black')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='black', alpha=0.5)

    # Plot the data
    lon, lat = np.meshgrid(all_long_options, all_lat_options)
    plot = ax.pcolormesh(lon, lat, data, cmap=custom_cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

    # Add a colorbar
    plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.05, aspect=50)

    # Set title
    ax.set_title(title)

    # Display the figure
    plt.show()

# Plot each figure
plot_figure(a_percent_change_MIMI_v2, "a_percent_change_MIMI_v2")
plot_figure(b_percent_change_v2_v3, "b_percent_change_v2_v3")
plot_figure(c_percent_change_MIMI_v3, "c_percent_change_MIMI_v3")
plot_figure(d_percent_change_v3_v4, "d_percent_change_v3_v4")
plot_figure(e_percent_change_MIMI_v4, "e_percent_change_MIMI_v4")
plot_figure(f_percent_change_v4_v5, "f_percent_change_v4_v5")
plot_figure(g_percent_change_MIMI_v5, "g_percent_change_MIMI_v5")
