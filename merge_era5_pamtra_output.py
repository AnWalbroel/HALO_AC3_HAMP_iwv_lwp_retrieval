import numpy as np
import xarray as xr

import glob

import sys
import os
wdir = os.getcwd() + "/"
remote = ((("/net/blanc/" in wdir) | ("/work/awalbroe/" in wdir)) and ("/mnt/f/" not in wdir))      # identify if the code is executed on the blanc computer or at home

from met_tools import e_sat, convert_relhum_to_mix_rat, convert_spechum_to_mix_rat
from data_tools import Gband_double_side_band_average, Fband_double_side_band_average
import pdb



R_d = 287.04    # gas constant of dry air, in J kg-1 K-1
R_v = 461.5     # gas constant of water vapour, in J kg-1 K-1
g = 9.80665     # gravitation acceleration, in m s^-2 (from https://doi.org/10.6028/NIST.SP.330-2019 )


def post_process_pamtra(DS, outlevel):

    """
    Post process PAMTRA output by removing obsolete dimensions, selecting the 
    needed angle, average over polarizations, perform double side band averaging.

    Parameters:
    -----------
    DS : xarray dataset
        Dataset that will be post processed.
    outlevel : int
        Index selecting the simulated observation height from the PAMTRA dataset.
    """

    # remove obsolete dimensions and select the right frequencies:
    DS = DS.isel(grid_y=0, angles=0, outlevel=outlevel) # angle index 0 == nadir (angle==180 deg); index -1 == zenith (angle==0 deg)
    DS['tb'] = DS.tb.mean(axis=-1)      # average over polarisation

    # double side band averaging:
    tb, freq_sb = Gband_double_side_band_average(DS.tb.values, DS.tb.frequency.values)
    tb, freq_sb = Fband_double_side_band_average(tb, freq_sb)
    DS = DS.sel(frequency=freq_sb)
    DS['tb'] = xr.DataArray(tb, dims=['grid_x', 'frequency'], coords=DS.tb.coords)

    return DS


"""
    Script to merge ERA5 data and PAMTRA simulations where certain grid points were selected.
    - import ERA5 data
    - import PAMTRA data
    - select grid points
    - align into a new array structure
"""


# paths:
if remote:
    path_data = {'era5': "/net/blanc/awalbroe/Data/METRS_SS23/ERA5_add/",
                'pamtra': "/net/blanc/awalbroe/Data/METRS_SS23/pamtra_output_add/"}
    path_output = "/net/blanc/awalbroe/Data/METRS_SS23/merged_add/"
else:
    path_data = {'era5': "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/training_data/ERA5/",
                'pamtra': "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/training_data/pamtra_output/"}
    path_output = "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/training_data/merged/"


# settings:
set_dict = {'obs_hgt': np.arange(8000.0, 13500.1, 250.0),       # obs height in m
            'grid_lat': np.arange(74.5, 79.5001, 1.0),          # sel grid point latitudes
            'grid_lon': [-2.5, 2.5, 7.5, 12.5],                 # sel grid point longitudes
            }


# Check if the PAMTRA output path exists; if not, create it.
path_out_dir = os.path.dirname(path_output)
if not os.path.exists(path_out_dir):
    os.makedirs(path_out_dir)


# identify ERA5 files and loop over them (monthly files). There should be the same number of
# single level and multi level files!
files_sl = path_data['era5'] + "ERA5_single_level_daily_for_pamtra_metrs_2000-2020.nc"
files_ml = sorted(glob.glob(path_data['era5'] + "ERA5_pressure_levels_daily_for_pamtra_metrs*.nc"))
files_pam = sorted(glob.glob(path_data['pamtra'] + "ERA5_PAMTRA_simulation_HALO_HAMP_freq_fram_strait_*.nc"))


# already import the smaller single level file:
ERA5_SL_ALL_DS = xr.open_dataset(files_sl)

# loop over files and over simulated aircraft altitudes to generate retrieval training/test data
# files for each aircraft altitude:

for outlevel in set_dict['obs_hgt']:
    year_idx = 0            # index which year is currently processed
    ds_list = list()        # list that saves xarray datasets of each year

    for file_ml, file_pam in zip(files_ml, files_pam):

        # import data:
        ERA5_ML_DS = xr.open_dataset(file_ml)
        ERA5_SL_DS = ERA5_SL_ALL_DS.sel(time=ERA5_ML_DS.time)       # reduce the single level dataset to the time of ERA5_ML_DS
        PAMTRA_DS = xr.open_dataset(file_pam)

        # make sure that ML and SL have got the same time, lon dimensions:
        assert len(ERA5_SL_DS.time) == len(ERA5_ML_DS.time)
        assert len(ERA5_SL_DS.longitude) == len(ERA5_ML_DS.longitude)

        # select grid points:
        ERA5_ML_DS = ERA5_ML_DS.sel(latitude=set_dict['grid_lat'], method='nearest')
        ERA5_SL_DS = ERA5_SL_DS.sel(latitude=set_dict['grid_lat'], method='nearest')
        ERA5_ML_DS = ERA5_ML_DS.sel(longitude=set_dict['grid_lon'], method='nearest')
        ERA5_SL_DS = ERA5_SL_DS.sel(longitude=set_dict['grid_lon'], method='nearest')


        # convert some units:
        ERA5_ML_DS['level'] = ERA5_ML_DS['level']*100.0     # pressure from hPa to Pa
        ERA5_ML_DS['z'] = ERA5_ML_DS['z'] / g       # geopotential to geopotential height
        ERA5_SL_DS['z_sfc'] = ERA5_SL_DS['z'] / g       # geopotential to geopotential height (at surface)
        ERA5_SL_DS['r2m'] = 100*e_sat(ERA5_SL_DS.d2m) / e_sat(ERA5_SL_DS.t2m)   # rel humidity (in %) from sat. water vapour press.
        ERA5_SL_DS['q2m'] = convert_relhum_to_mix_rat(ERA5_SL_DS.r2m*0.01, ERA5_SL_DS.t2m, ERA5_SL_DS.sp)


        # form into a new array:
        n_time = len(ERA5_ML_DS.time)
        n_lon = len(ERA5_ML_DS.longitude)
        n_lat = len(ERA5_ML_DS.latitude)
        n_lev = len(ERA5_ML_DS.level)
        n_x = n_time*n_lon*n_lat

        # PAM_DS will contain all relevant atmospheric data; first, set pressure level vars:
        pres_vars = ['z', 'r', 't', 'q']
        PAM_DS = xr.Dataset(coords={'x': np.arange(n_x),
                                    'level': ERA5_ML_DS.level})
        for var in pres_vars: PAM_DS[var] = xr.DataArray(np.full((n_x, n_lev), np.nan), dims=['x', 'level'])

        for k in range(n_lev):
            for var in pres_vars: PAM_DS[var][:,k] = ERA5_ML_DS[var].values[:,k,:,:].ravel()

        # add time + lat-lon information:
        PAM_DS['longitude'] = xr.DataArray(np.tile(np.reshape(ERA5_ML_DS.longitude.values, (1,1,n_lon)), (n_time, n_lat, 1)).ravel(), dims=['x'])
        PAM_DS['time'] = xr.DataArray(np.tile(np.reshape(ERA5_ML_DS.time.values, (n_time,1,1)), (1, n_lat, n_lon)).ravel(), dims=['x'])
        PAM_DS['latitude'] = xr.DataArray(np.tile(np.reshape(ERA5_ML_DS.latitude.values, (1,n_lat,1)), (n_time, 1, n_lon)).ravel(), dims=['x'])


        # flip the vertical axis so that the surface is at index 0:
        PAM_DS = PAM_DS.reindex({'level': PAM_DS.level[::-1]})

        # also put the single level data into PAM_DS
        sing_vars = ['t2m', 'r2m', 'q2m', 'z_sfc', 'sp', 'lsm', 'siconc', 'sst', 'skt', 'tcwv', 'tciw', 'tclw', 'tcrw', 'tcsw']
        for var in sing_vars: PAM_DS[var] = xr.DataArray(ERA5_SL_DS[var].values.ravel(), dims=['x'])


        # filter for open ocean only cases: (remove land and ice grid points)
        idx_land = PAM_DS.lsm > 0
        idx_ice = PAM_DS.siconc > 0
        idx_open_ocean = (~idx_land) & (~idx_ice)
        PAM_DS = PAM_DS.isel(x=idx_open_ocean)

        n_x = len(PAM_DS.x)


        # to avoid negative geopot heights due to the pressure level convention of ERA5 (i.e., 
        # 1000 hPa level can be < 0 m geopot. height because of cyclones), take the surface pressure
        # into account:
        # first, create arrays with the extended pressure grid with dummy values at the surface:
        n_lev_p = n_lev + 1
        PAM_DS['pres'] = xr.DataArray(np.repeat(np.reshape(np.concatenate((np.array([1013.25]), PAM_DS.level.values)), 
                                        (1, n_lev_p)), n_x, axis=0), dims=['x', 'levels'],
                                        coords={'levels': np.concatenate((np.array([1013.25]), PAM_DS.level.values))})
        PAM_DS['temp'] = xr.DataArray(np.zeros((n_x, n_lev_p)), dims=['x', 'levels'])
        PAM_DS['height'] = xr.DataArray(np.zeros((n_x, n_lev_p)), dims=['x', 'levels'])
        PAM_DS['relhum'] = xr.DataArray(np.zeros((n_x, n_lev_p)), dims=['x', 'levels'])
        PAM_DS['q_spec'] = xr.DataArray(np.zeros((n_x, n_lev_p)), dims=['x', 'levels'])

        # fill the initialised arrays with data: first, at the surface; then aloft:
        PAM_DS['pres'][:,0] = PAM_DS.sp
        PAM_DS['temp'][:,0] = PAM_DS.t2m
        PAM_DS['height'][:,0] = PAM_DS.z_sfc
        PAM_DS['relhum'][:,0] = PAM_DS.r2m
        PAM_DS['q_spec'][:,0] = PAM_DS.q2m

        # aloft: 
        PAM_DS['temp'][:,1:] = PAM_DS.t
        PAM_DS['height'][:,1:] = PAM_DS.z
        PAM_DS['relhum'][:,1:] = PAM_DS.r
        PAM_DS['q_spec'][:,1:] = PAM_DS.q

        # loop over all cases to interpolate between the surface and the next pressure level
        # ABOVE the surface:
        for ix in range(n_x):
            if PAM_DS['sp'][ix] <= PAM_DS['level'][0]:

                # identify where data is useless:
                idx = np.where(PAM_DS.pres[ix,:] >= PAM_DS.sp[ix])[0]

                # interpolate to new pressure that bridges the sfc pressure and the next higher
                # pressure level; then replace meteo values by interpolated ones:
                p_bridge = np.linspace(PAM_DS.sp[ix], PAM_DS.pres[ix,idx[-1]+1], len(idx)+1)
                p_bridge_pillars = np.array([p_bridge[0], p_bridge[-1]])[::-1]  # coords had to be in asc. order
                PAM_DS['pres'][ix,idx] = p_bridge[:-1]  # don't take the last one as it already exists
                PAM_DS['temp'][ix,idx] = np.interp(p_bridge, p_bridge_pillars,
                                                    np.array([PAM_DS.t2m[ix], PAM_DS.temp[ix,idx[-1]+1]])[::-1])[:-1]
                PAM_DS['height'][ix,idx] = np.interp(p_bridge, p_bridge_pillars,
                                                    np.array([PAM_DS.z_sfc[ix], PAM_DS.height[ix,idx[-1]+1]])[::-1])[:-1]
                PAM_DS['relhum'][ix,idx] = np.interp(p_bridge, p_bridge_pillars,
                                                    np.array([PAM_DS.r2m[ix], PAM_DS.relhum[ix,idx[-1]+1]])[::-1])[:-1]
                PAM_DS['q_spec'][ix,idx] = np.interp(p_bridge, p_bridge_pillars,
                                                    np.array([PAM_DS.q2m[ix], PAM_DS.q_spec[ix,idx[-1]+1]])[::-1])[:-1]



        # replace old leveled data and remove unneeded variables to save memory:
        PAM_DS['t'] = PAM_DS['temp']
        PAM_DS['z'] = PAM_DS['height']
        PAM_DS['r'] = PAM_DS['relhum']
        PAM_DS['q'] = PAM_DS['q_spec']
        PAM_DS = PAM_DS.drop(['temp', 'relhum', 'height', 'q_spec'])
        PAM_DS = PAM_DS.drop_dims('level')


        # Now, merge PAMTRA simulations with the selected ERA5 grid points from above:
        outlevel_idx = np.where(PAMTRA_DS.outlevels[0,0,:] == outlevel)[0]
        PAMTRA_DS = post_process_pamtra(PAMTRA_DS, outlevel=outlevel_idx[0])
        PAM_DS['tb'] = xr.DataArray(PAMTRA_DS.tb.values, dims=['x', 'frequency'], coords={'frequency': PAMTRA_DS.frequency})
        PAM_DS['outlevel'] = PAMTRA_DS.outlevels.values[0]

        # save PAM_DS to the dataset ist and increment year index
        ds_list.append(PAM_DS)
        year_idx += 1

        # close datasets:
        PAMTRA_DS = PAMTRA_DS.close()
        PAM_DS = PAM_DS.close()


    # merge all years into one dataset:
    MERGED_DS = xr.combine_nested(ds_list, concat_dim='x')
    MERGED_DS['outlevel'] = MERGED_DS.outlevel[0]
    MERGED_DS = MERGED_DS.assign_coords({'x': np.arange(len(MERGED_DS.x))})

    # save some attributes:
    vars_for_attrs = ['r', 't', 'q', 'longitude', 'latitude', 'time']
    vars_sl_for_attrs = ['t2m', 'sp', 'lsm', 'siconc', 'sst', 'skt', 'tcwv', 'tciw', 'tclw', 'tcrw', 'tcsw']
    for var in vars_for_attrs:
        MERGED_DS[var].attrs = ERA5_ML_DS[var].attrs
    for var in vars_sl_for_attrs:
        MERGED_DS[var].attrs = ERA5_SL_DS[var].attrs
    MERGED_DS['z'].attrs = {'long_name': "Geopotential height", 'units': "m"}
    MERGED_DS['z_sfc'].attrs = {'long_name': "Surface geopotential height", 'units': "m"}
    MERGED_DS['r2m'].attrs = {'long_name': "2 metre relative humidity", 'units': "%"}
    MERGED_DS['q2m'].attrs = {'long_name': "2 metre specific humidity", 'units': "kg kg-1"}
    MERGED_DS['pres'].attrs = {'long_name': "Air pressure", 'units': "Pa"}
    MERGED_DS['tb'].attrs = {'long_name': "Simulated brightness temperatures", 'units': "K"}
    MERGED_DS['frequency'].attrs = {'long_name': "Frequencies of the simulated brightness temperatures", 
                                    'units': "GHz"}
    MERGED_DS['outlevel'].attrs = {'long_name': "Height of the simulated observations", 'units': "m"}


    # rename variables:
    rename_dict = {'z': 'hgt', 'r': 'rh', 'pres': 'p', 'latitude': 'lat', 'longitude': 'lon',
                    'outlevel': 'obs_height', 'lsm': 'sfc_slf', 'siconc': 'sfc_sif', 'sst': 'groundtemp',
                    'tcwv': 'iwv', 'tciw': 'iwp', 'tclw': 'cwp', 'tcrw': 'rwp', 'tcsw': 'swp', 
                    'frequency': 'freq', 'levels': 'z'}
    for name_key in rename_dict.keys():
        MERGED_DS = MERGED_DS.rename({name_key: rename_dict[name_key]})

    # compute LWP:
    MERGED_DS['lwp'] = xr.DataArray(MERGED_DS['cwp'].values + MERGED_DS['rwp'].values,
                                        dims=['x'], attrs={'long_name': "Liquid water path",
                                        'units': "kg m**-2"})

    # remove _FillValue from attributes:
    for varr in MERGED_DS.variables:
        MERGED_DS[varr].encoding["_FillValue"] = None


    # GLOBAL ATTRIBUTES:
    MERGED_DS.attrs['title'] = "Training data set for open-ocean HALO-based atmospheric retrievals from microwave radiometers designed for HALO-(AC)3"
    MERGED_DS.attrs['title_short_name'] = "HALO-(AC)3 training data"
    MERGED_DS.attrs['comments'] = ("ERA5 grid points have been selected and simulated brightness temperatures (tb) have been " +
                                    "created with PAMTRA for icefree ocean-only grid points.")
    MERGED_DS.attrs['python_version'] = f"python version: {sys.version}"
    MERGED_DS.attrs['python_packages'] = (f"numpy: {np.__version__}, xarray: {xr.__version__}, ")


    # time encoding
    MERGED_DS['time_sec'] = MERGED_DS.time.values.astype("datetime64[s]").astype(np.float64)
    MERGED_DS['time_sec'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
    MERGED_DS['time_sec'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    MERGED_DS['time_sec'].encoding['dtype'] = 'double'


    outfile = path_output + f"HALO-AC3_ERA5_PAMTRA_training_data_outlevel_{int(outlevel)}m.nc"
    MERGED_DS.to_netcdf(outfile, mode='w', format='NETCDF4')
    MERGED_DS = MERGED_DS.close()

    print("Merged dataset saved to " + outfile)