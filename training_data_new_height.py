import numpy as np
import xarray as xr
import os
import glob
import pdb
import sys

wdir = os.getcwd() + "/"
remote = ((("/net/blanc/" in wdir) | ("/work/awalbroe/" in wdir)) and ("/mnt/f/" not in wdir))      # identify if the code is executed on the blanc computer or at home


def interp_to_new_hgt_grd(
    data, 
    new_height, 
    height_vars,
    set_dict):

    """
    Interpolate variables listed in height_vars in the data to the new height grid 
    specified by new_height.

    Parameters:
    -----------
    data : xarray dataset
        Data of which the height depended variables will be interpolated to a new grid.
        The height variable of data must be called 'height'.
    height_vars : list
        Contains variable names as keys and their expected dimension number as values.
    set_dict : dict
        Dictionary containing additional information.
    """

    # lopp through all variables:
    for height_var in height_vars:
        if height_var in data.data_vars:
            for kk in range(data[height_var].shape[0]):
                data[height_var][kk,:set_dict['n_height']] = np.interp(new_height, 
                                                                        data['height'][kk,:],
                                                                        data[height_var][kk,:])

    # truncate above new_height:
    data = data.isel(z=slice(0,set_dict['n_height']))
    data['height'] = xr.DataArray(np.repeat(np.reshape(new_height, (1, set_dict['n_height'])), 
                                    len(data.x), axis=0), dims=['x', 'z'],
                                    attrs={'long_name': "Height grid", 'units': "m"})

    return data


"""
    This script interpolates the ERA5 training data output onto a new height grid, after converting units. 
    The new height grid is equal to the one used during MWR_PRO processing of HATPRO data for profiles.
    - load data (year by year)
    - interpolate to new height
    - export
"""


# paths:
if remote:
    path_data = "/net/blanc/awalbroe/Data/METRS_SS23/merged_add/"
    path_output = "/net/blanc/awalbroe/Data/METRS_SS23/merged_add/new_z_grid/"
else:
    path_data = "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/training_data/merged/"
    path_output = "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/training_data/merged/new_z_grid/"


# settings:
set_dict = {'height_vars': ['temp', 'rh', 'pres', 'q']}
new_height = np.array([0, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000,
                        6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000]).astype(np.float64) # new height grid in m
set_dict['n_height'] = len(new_height)


# create output path if not existing:
outpath_dir = os.path.dirname(path_output)
if not os.path.exists(outpath_dir):
    os.makedirs(outpath_dir)


# locate files and loop over them:
files = sorted(glob.glob(path_data + "HALO-AC3_ERA5_PAMTRA_training_data_outlevel_*.nc"))
for file in files:
    print(f"Processing {file.replace(path_data,'')}....")

    # import data:
    era5_inst = xr.open_dataset(file)

    # convert units in DS: rh
    era5_inst['rh'] = era5_inst['rh']*0.01      # conversion to [0,1]
    era5_inst['rh'].attrs['units'] = "[0,1]"
    era5_inst['r2m'] = era5_inst['r2m']*0.01        # conversion to [0,1]
    era5_inst['r2m'].attrs['units'] = "[0,1]"

    # rename variables:
    for ori_var, rhen_var in zip(['groundtemp', 'hgt', 't', 'p'], ['temp_sfc', 'height', 'temp', 'pres']):
        era5_inst = era5_inst.rename({ori_var: rhen_var})


    # interpolate to new height grid
    era5_inst = interp_to_new_hgt_grd(era5_inst, new_height, set_dict['height_vars'], set_dict)


    # export dataset:
    # era5_inst['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
    # era5_inst['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    # era5_inst['time'].encoding['dtype'] = 'double'
    era5_inst.to_netcdf(path_output + os.path.basename(file), mode='w', format="NETCDF4")
    era5_inst = era5_inst.close()

    # Clear memory:
    del era5_inst