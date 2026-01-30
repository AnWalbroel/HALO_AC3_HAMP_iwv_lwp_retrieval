import gc
import datetime as dt
import pdb
import glob
import os
import sys
wdir = os.getcwd() + "/"
remote = ((("/net/blanc/" in wdir) | ("/work/awalbroe/" in wdir)) and ("/mnt/f/" not in wdir))      # identify if the code is executed on the blanc computer or at home

import numpy as np
import xarray as xr
import matplotlib as mpl
if not remote: mpl.use("WebAgg")
mpl.rcParams.update({"font.family": "monospace"})
import matplotlib.pyplot as plt


"""
    This script visualizes the atmos part of the training data with scatterplots,
    histograms...
    - import training data
    - modify quantities if required
    - visualize
"""


# paths:
if remote:
    path_data = {'era5': "/net/blanc/awalbroe/Data/METRS_SS23/merged_add/"}
    path_plots = "/net/blanc/awalbroe/Plots/HALO_AC3/lwp_retrieval/training_data_atmos_add/"
else:
    path_data = {'era5': "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/training_data/merged/"}
    path_plots = "/mnt/f/Studium_NIM/work/Plots/HALO_AC3/lwp_retrieval/training_data_atmos/"

# additional settings:
set_dict = {'save_figures': True,
            'iwv_hist': True,
            'lwp_hist': True,
            'overview_map_plot': True,
            '1D_aligned': True}

path_plots_dir = os.path.dirname(path_plots)
if not os.path.exists(path_plots_dir):
    os.makedirs(path_plots_dir)

if set_dict['1D_aligned']:
    dims_1d_list = ['x']
    dims_2d_list = ['x', 'z']

# import data:
file = path_data['era5'] + "HALO-AC3_ERA5_PAMTRA_training_data_outlevel_10000m.nc"
era5_atmos = xr.open_dataset(file)


# visualize:

fs = 16
fs_small = fs - 2
fs_dwarf = fs_small - 2

# IWV histogram:
if set_dict['iwv_hist']:
    # compute weights for histogram:
    data_plot = xr.DataArray(era5_atmos.iwv, dims='x').values
    data_plot = data_plot[np.where(~np.isnan(data_plot))[0]]
    n_data = float(len(data_plot))
    weights_data = np.ones((int(n_data),)) / n_data

    f1 = plt.figure(figsize=(16,7))
    a1 = plt.axes()

    x_lim = [0, 20]     # kg m-2

    # plotting:
    le_hist = a1.hist(data_plot, bins=np.arange(x_lim[0], x_lim[1]+0.00001, 0.25),
                        weights=weights_data, color=(0.8,0.8,0.8), ec=(0,0,0))
    # add auxiliary info:
    a1.text(0.98, 0.98, f"Min = {np.min(data_plot):.1f}\nMax = {np.max(data_plot):.1f}\nMean = {data_plot.mean():.1f}\n" +
            f"Median = {np.median(data_plot):.1f}\nN = {len(data_plot)}", fontsize=fs_dwarf, ha='right', va='top',
            transform=a1.transAxes)

    # set axis limits:
    a1.set_xlim(x_lim)

    # set ticks and tick labels and parameters:
    a1.tick_params(axis='both', labelsize=fs_small)

    # grid:
    a1.minorticks_on()
    a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

    # set labels:
    a1.set_xlabel("IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
    a1.set_ylabel("Freq. occurrence", fontsize=fs)

    if set_dict['save_figures']:
        plotname = f"ERA5_HALO-AC3_lwp_ret_training_iwv_histogram"
        f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
        pdb.set_trace()

    f1.clf()
    plt.close()
    gc.collect()


if set_dict['lwp_hist']:
    # compute weights for histogram:
    data_plot = xr.DataArray(era5_atmos.lwp, dims='x').values
    data_plot = data_plot[np.where(~np.isnan(data_plot))[0]]*1000.0
    n_data = float(len(data_plot))
    weights_data = np.ones((int(n_data),)) / n_data

    f1 = plt.figure(figsize=(16,7))
    a1 = plt.axes()

    x_lim = [0, 600]        # g m-2

    # plotting:
    le_hist = a1.hist(data_plot, bins=np.arange(x_lim[0], x_lim[1]+0.00001, 5.0),
                        weights=weights_data, color=(0.8,0.8,0.8), ec=(0,0,0))
    # add auxiliary info: 
    a1.text(0.98, 0.98, f"Min = {np.min(data_plot):.1f}\nMax = {np.max(data_plot):.1f}\nMean = {data_plot.mean():.1f}\n" +
            f"Median = {np.median(data_plot):.1f}\nN = {len(data_plot)}", fontsize=fs_dwarf, ha='right', va='top',
            transform=a1.transAxes)

    # set axis limits:
    a1.set_xlim(x_lim)

    # set ticks and tick labels and parameters:
    a1.tick_params(axis='both', labelsize=fs_small)

    # grid:
    a1.minorticks_on()
    a1.grid(axis='both', which='both', color=(0.5,0.5,0.5), alpha=0.5)

    # set labels:
    a1.set_xlabel("LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
    a1.set_ylabel("Freq. occurrence", fontsize=fs)

    if set_dict['save_figures']:
        plotname = f"ERA5_HALO-AC3_lwp_ret_training_lwp_histogram"
        f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
        pdb.set_trace()

    f1.clf()
    plt.close()
    gc.collect()


if set_dict['overview_map_plot']:

    data_plot = era5_atmos.sfc_sif*100.


    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    marker_size = 9.0

    # map_settings:
    # lon_centre = 0.0
    # lat_centre = 75.0
    # lon_lat_extent = [-60.0, 60.0, 60.0, 90.0]        # (zoomed in)
    # sel_projection = ccrs.Orthographic(central_longitude=lon_centre, central_latitude=lat_centre)
    lon_centre = 8.5
    lat_centre = 73.5
    lon_lat_extent = [-70.0, 70.0, 68.5, 81.0]
    sel_projection = ccrs.NearsidePerspective(central_longitude=lon_centre, central_latitude=lat_centre, satellite_height=1000000.)


    # some extra info for the plot:
    station_coords = {'Longyearbyen': [15.632, 78.222]}


    f1 = plt.figure(figsize=(10,7.5))
    a1 = plt.axes(projection=sel_projection)
    a1.set_extent(lon_lat_extent, crs=ccrs.PlateCarree())
    # a1.add_image(cimgt.Stamen('terrain-background'), 4)

    # add some land marks:
    a1.coastlines(resolution="50m", zorder=9999.0, linewidth=0.5)
    a1.add_feature(cartopy.feature.BORDERS, zorder=9999.0)
    a1.add_feature(cartopy.feature.OCEAN, zorder=-1.0)
    a1.add_feature(cartopy.feature.LAND, color=(0.9,0.85,0.85), zorder=-1.0)
    a1.gridlines(draw_labels=True, color=(0.8,0.8,0.8), zorder=9999.0)

    PlateCarree_mpl_transformer = ccrs.PlateCarree()._as_mpl_transform(a1)
    text_transform = mpl.transforms.offset_copy(PlateCarree_mpl_transformer, units='dots', 
                                                x=marker_size*2.50, y=marker_size*2.50)


    # plot the grid points:
    idx_time_0 = np.where(era5_atmos.time.values == era5_atmos.time.values[0])[0]
    sel_lats = era5_atmos.lat.isel(x=idx_time_0).values
    sel_lons = era5_atmos.lon.isel(x=idx_time_0).values
    for slat, slon in zip(sel_lats, sel_lons):
        a1.plot(slon, slat, linestyle='none', color=(0,1,1), marker='o', markersize=marker_size, 
                    markeredgecolor=(0,0,0), transform=ccrs.PlateCarree(), zorder=10000.0, 
                    label=f"{slat:.2f}N, {slon:.2f}E")



    # place markers and labels:
    a1.plot(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], color=(1,0,0),
            marker='.', markersize=marker_size, markeredgecolor=(0,0,0),
            transform=ccrs.PlateCarree(), zorder=10000.0)

    a1.text(station_coords['Longyearbyen'][0], station_coords['Longyearbyen'][1], "LYR",
            ha='left', va='bottom',
            color=(1,0,0), fontsize=fs_dwarf, transform=text_transform, 
            bbox={'facecolor': (211.0/255.0,211.0/255.0,211.0/255.0), 'edgecolor': (0,0,0), 'boxstyle': 'square'},
            zorder=10000.0)


    if set_dict['save_figures']:
        plotname = f"ERA5_HALO-AC3_lwp_ret_training_map_plot_example"
        f1.savefig(path_plots + plotname + ".png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
        pdb.set_trace()

    f1.clf()
    plt.close()
    gc.collect()