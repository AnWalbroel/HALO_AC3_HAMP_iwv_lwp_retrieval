import sys
import os
import glob
import pdb
import datetime as dt

import numpy as np
import xarray as xr

wdir = os.getcwd() + "/"
remote = ((("/net/blanc/" in wdir) | ("/work/awalbroe/" in wdir)) and ("/mnt/f/" not in wdir))


def concat_obs_height(DS):

    """
    Concatentate datasets along observation heights (obs_heights). This dimension must be created 
    at first.
    """

    DS = DS.expand_dims(dim=['o_h'], axis=-1)

    return DS


"""
    Post process the output of NN_retrieval.py with exec_type='op_ret' to merge the retrieved
    quantity estimated for different observation heights (obs_heights). The correct obs_height
    is selected for each part of the research flight based on the aircraft altitude. Finally,
    one file for each research flight will be created.
    - identify obs_heights
    - for each research flight, import retrieved quantity, merge along obs_heights
    - select correct obs_height for flight sections
    - export to netCDF
"""


predictand_id = "lwp"       # to manually set the predictand
if len(sys.argv) == 2:      # can also be inquired via given input
    predictand_id = sys.argv[1]
    assert predictand_id in ['iwv', 'lwp']


# Paths:
if remote:
    path_data = {'input': "/net/blanc/awalbroe/Data/HALO_AC3/lwp_retrieval/output/l2/",
                'output': "/net/blanc/awalbroe/Data/HALO_AC3/lwp_retrieval/output/l2/"}
else:
    path_data = {'input': "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/output/l2/",
                'output': "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/output/l2/"}

# settings:
set_dict = {'obs_heights': np.arange(8000., 13500.001, 250.).astype(np.int32),
            'date_0': "2022-03-11",     # campaign start
            'date_1': "2022-04-12"}     # campaign end
set_dict['n_oh'] = len(set_dict['obs_heights'])
set_dict['predictand_id'] = predictand_id


# identify obs_heights (saved to folders within path_data['input']:
obs_height_dirs = list()
for item in os.listdir(path_data['input']):
    sub_item = os.path.join(path_data['input'], item)
    if os.path.isdir(sub_item): obs_height_dirs.append(sub_item + "/")

all_files = dict()
for ohd in obs_height_dirs:
    all_files[ohd] = sorted(glob.glob(ohd + f"HALO-AC3_HALO_HAMP_radiometer_l2_{set_dict['predictand_id']}_v00*.nc"))


# loop over days within the campaign period and find research flights:
date_0_dt = dt.datetime.strptime(set_dict['date_0'], "%Y-%m-%d")
date_1_dt = dt.datetime.strptime(set_dict['date_1'], "%Y-%m-%d")
n_days = (date_1_dt - date_0_dt).days + 1
for c_date in (date_0_dt + n*dt.timedelta(days=1) for n in range(n_days)):

    c_date_str = c_date.strftime("%Y-%m-%d")
    c_date_file = c_date.strftime("%Y%m%d")

    # find files in subdirectories:
    rf_files = list()
    for ohd in all_files.keys():
        rf_file = [file for file in all_files[ohd] if c_date_file in file]
        if len(rf_file) == 1:
            rf_files.append(rf_file[0])


    # if the correct number of files was found, they can be processed
    if len(rf_files) == 0:
        continue
    elif len(rf_files) == set_dict['n_oh']:

        print(f"Post processing files for {c_date_str}....")

        # import and concatenate along a new dimension 'obs_height':
        DS = xr.open_mfdataset(rf_files, concat_dim='o_h', combine='nested', preprocess=concat_obs_height)

        # for each obs height, find time indices where flight altitude is close to that obs height:
        idx_time = dict()
        for i_o, o_h in enumerate(DS.obs_height.values):
            idx_time[str(o_h)] = np.where((DS.alt.values[:,0] >= o_h - 125.) & (DS.alt.values[:,0] < o_h + 125.))[0]


        # construct the new data array:
        PROC_DS = xr.Dataset(coords={'time': DS.time})
        excl_var = ['obs_height']
        for var in DS.data_vars:
            if var not in excl_var:
                PROC_DS[var] = xr.DataArray(np.full((DS.time.shape), np.nan), dims=['time'])
                PROC_DS[var].attrs = DS[var].attrs      # set attributes

        # set values for correct altitudes:
        for ii, key in enumerate(idx_time.keys()):
            if len(idx_time[key]) > 0:
                for var in DS.data_vars:
                    if var not in excl_var:
                        PROC_DS[var][idx_time[key]] = DS[var].isel(o_h=ii).values[idx_time[key]]


        # set global attributes:
        PROC_DS.attrs = DS.attrs
        
        # encode time:
        PROC_DS['time'] = DS.time.values.astype("datetime64[s]").astype(np.float64)
        PROC_DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
        PROC_DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
        PROC_DS['time'].encoding['dtype'] = 'double'

        # export:
        nc_output_name = os.path.basename(rf_files[0])
        PROC_DS.to_netcdf(path_data['output'] + nc_output_name, mode='w', format="NETCDF4")
        PROC_DS.close()


    else:
        pdb.set_trace()
        raise RuntimeError(f"Not all obs_heights are available on {c_date_str}. Please check.")