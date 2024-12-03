# %% 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def draw_map(ax, scale=0.2):
    ax.stock_img()
    ax.coastlines()
    # Draw parallels and meridians
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

# Create a figure and axis
fig = plt.figure(figsize=(8, 6), edgecolor='w')
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())

# Set the central latitude and longitude
ax.set_global()

# Draw the map
draw_map(ax)

plt.show()
# %%
