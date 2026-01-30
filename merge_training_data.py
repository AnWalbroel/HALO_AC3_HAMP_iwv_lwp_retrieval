import xarray as xr
import pdb
import numpy as np
import glob
import os
import sys


def concat_time(DS):
    """
    Concatenate a dataset consisting of multiple files along a newly formed time 
    dimension.

    Parameters:
    -----------
    DS : xarray dataset
        Dataset where a new time dimension must be created for concatenation.
    """

    # inquire time:
    DS = DS.expand_dims({'time': 1}, axis=0)

    if DS.unixtime.dtype != np.dtype("<M8[ns]"):
        DS['unixtime'] = DS.unixtime.astype("datetime64[s]").astype("datetime64[ns]")

    return DS


def simplify_DS(DS):
    """
    Simplify dataset by reducing dimensions where not needed.

    Parameters:
    -----------
    DS : xarray dataset
        Dataset that can be simplified.
    """

    # exclude time axis where not needed (like lat, lon)
    exclude_time = ['lat', 'lon', 'obs_height', 'ang', 'freq', 'pol']
    for et in exclude_time: 
        if et in DS.data_vars:
            # pdb.set_trace()
            DS[et] = DS[et].isel(time=0)

    if len(DS.obs_height) > 1:      # nout=0 must be chosen because it contains total atm. integral
        DS = DS.sel(nout=0)

    # also remove unnecessary dimensions for other variables (i.e., x and y for time):
    exclude_x_y = ['unixtime']
    for exy in exclude_x_y:
        if exy in DS.data_vars:
            DS[exy] = DS[exy].isel(x=0,y=0)

    return DS


###################################################################################################
###################################################################################################


"""
    This script merges the ERA5 training data output from LEVANTE into one
    file containing all simulated TBs and the atmospheric counterpart. The files 
    will be separated into yearly files (i.e., one for 2019, one for 2020, ...).
    - locate the correct files
    - load data
    - split by years (loop through years)
    - merge into one dataset per year
    - export
"""


# paths
path_data = {'input': "/net/blanc/awalbroe/Data/synergetic_ret/training_data_01/",
            'output': "/net/blanc/awalbroe/Data/synergetic_ret/training_data_01/merged/"}

# create output folder if not existing:
outpath_dir = os.path.dirname(path_data['output'])
if not os.path.exists(outpath_dir):
    os.makedirs(outpath_dir)

# additional settings:
set_dict = {'skip_existing': True}


# identify files:
set_dict['1D_aligned'] = True           # if True, data in path_data['input'] have been aligned along one axis only
set_dict['file_types'] = ['atmos', 'hatpro', 'mirac-p']


# load data, separated by years:
years = np.arange(2000,2019, dtype=np.int32)                                                                                                                                ###################
for year in years:
    print(f"Processing year {year}....")

    # check if merged file already exists:
    if set_dict['skip_existing']:
        if os.path.exists(path_data['output'] + f"ERA5_syn_ret_hatpro_mirac-p_training_{year}.nc"):
            print(f"Skipping {year} as the merged file already exists.")
            continue
    

    files = dict()
    DS_dict = dict()
    for file_type in set_dict['file_types']:
        if file_type in ['hatpro', 'mirac-p']:
            files[file_type] = sorted(glob.glob(path_data['input'] + f"*{file_type}*{year}[0-9][0-9][0-9][0-9]_[0-2][0-9].nc"))
        else:
            files[file_type] = sorted(glob.glob(path_data['input'] + f"*{year}[0-9][0-9][0-9][0-9]_[0-2][0-9]_{file_type}.nc"))

        # at first, concatenate all files of one type of one year:
        if len(files[file_type]) > 0:
            DS_dict[file_type] = xr.open_mfdataset(files[file_type], concat_dim='time', combine='nested', preprocess=concat_time)

            # assign variable coordinates and remove unnecessary dimensions:
            DS_dict[file_type] = simplify_DS(DS_dict[file_type])
            if not set_dict['1D_aligned']:
                DS_dict[file_type] = DS_dict[file_type].assign_coords({'x': DS_dict[file_type].lon.values[:,0]})
                DS_dict[file_type] = DS_dict[file_type].assign_coords({'y': DS_dict[file_type].lat.values[0,:]})

            else:   # then, all data has been aligned along x
                DS_dict[file_type] = DS_dict[file_type].isel(y=0)

            DS_dict[file_type] = DS_dict[file_type].assign_coords({'time': DS_dict[file_type].unixtime})
            DS_dict[file_type]['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
            DS_dict[file_type] = DS_dict[file_type].drop('unixtime')

            # reduce DS to current year:
            DS_dict[file_type] = DS_dict[file_type].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))


    if DS_dict:
        # Merge into one dataset per year:
        DS_merged = xr.Dataset(coords={'time': (['time'], DS_dict['atmos'].time.values)})

        for file_type in set_dict['file_types']:
            for var in DS_dict[file_type].data_vars:
                if var not in DS_merged.data_vars and var not in ["time", 'freq', 'tb']:
                    DS_merged[var] = DS_dict[file_type][var]
                elif var not in DS_merged.data_vars and var in ['freq', 'tb']:
                    DS_merged[var] = xr.concat([DS_dict['hatpro'][var], DS_dict['mirac-p'][var]],
                                                dim='nfreq')
            DS_dict[file_type].close()


        # Export dataset:
        DS_merged['time'] = DS_merged.time.values.astype("datetime64[s]").astype(np.float64)
        DS_merged['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
        DS_merged['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
        DS_merged['time'].encoding['dtype'] = 'double'
        DS_merged.to_netcdf(path_data['output'] + f"ERA5_syn_ret_hatpro_mirac-p_training_{year}.nc", mode='w', format="NETCDF4")
        DS_merged.close()

        # Clear memory:
        del DS_merged, DS_dict, files

print("No errors... gg")
