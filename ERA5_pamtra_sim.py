import numpy as np
import xarray as xr
from copy import deepcopy

import datetime as dt
import pandas as pd
import glob

import multiprocessing
import sys
import os
import warnings

from met_tools import e_sat, convert_relhum_to_mix_rat, convert_spechum_to_mix_rat
import pdb

os.environ['OPENBLAS_NUM_THREADS'] = "1"
if 'PAMTRA_DATADIR' not in os.environ:
    os.environ['PAMTRA_DATADIR'] = "" # actual path is not required, but the variable has to be defined.
import pyPamtra


R_d = 287.04    # gas constant of dry air, in J kg-1 K-1
R_v = 461.5     # gas constant of water vapour, in J kg-1 K-1
g = 9.80665     # gravitation acceleration, in m s^-2 (from https://doi.org/10.6028/NIST.SP.330-2019 )


"""
    Script to simulate HALO based HAMP TBs for the Fram Strait with PAMTRA based on ERA5 data. 
    Certain grid points will be selected, arranged in a 3D array (locations, time, height) 
    or 2D array (locations*time, height). Because of ERA5 download limitations, yearly
    files are taken as input and will be given as output.
    - import ERA5 data
    - select grid points
    - align into a new array structure
    - PAMTRA settings
    - set PAMTRA data
    - simulate
"""


# paths:
path_data = "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/training_data/ERA5/"
path_putput = "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/training_data/pamtra_output/"


# settings:
set_dict = {'obs_hgt': np.arange(8000.0, 13500.1, 250.0),       # obs height in m
            'grid_lat': np.arange(74.5, 79.5001, 1.0),          # sel grid point latitudes
            'grid_lon': [-2.5, 2.5, 7.5, 12.5],                 # sel grid point longitudes
            'frq': np.sort([22.2400, 23.0400, 23.8400, 25.4400, 26.2400, 27.8400, 31.4000,
                    50.3000, 51.7600, 52.8000, 53.7500, 54.9400, 56.6600, 58.0000, 
                    90.0000, 110.250, 114.550, 116.450, 117.350, 120.150, 121.050, 122.950, 127.250,
                    175.810, 178.310, 179.810, 180.810, 181.810, 182.710, 183.910, 184.810, 185.810, 186.810, 188.310, 190.810])
            }


# Check if the PAMTRA output path exists:
path_out_dir = os.path.dirname(path_putput)
if not os.path.exists(path_out_dir):
    os.makedirs(path_out_dir)


# identify ERA5 files and loop over them (monthly files). There should be the same number of
# single level and multi level files!
files_sl = path_data + "ERA5_single_level_daily_for_pamtra_metrs_2000-2020.nc"
files_ml = sorted(glob.glob(path_data + "ERA5_pressure_levels_daily_for_pamtra_metrs*.nc"))


# already import the smaller single level file:
ERA5_SL_ALL_DS = xr.open_dataset(files_sl)


# loop over files:
for file_ml in files_ml:

    # import data:
    ERA5_ML_DS = xr.open_dataset(file_ml)
    ERA5_SL_DS = ERA5_SL_ALL_DS.sel(time=ERA5_ML_DS.time)

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

    # PAM_DS will contain all data necessary for PAMTRA; first, set pressure level vars:
    pres_vars = ['z', 'r', 't', 'q', 'ciwc', 'clwc', 'crwc', 'cswc']
    PAM_DS = xr.Dataset(coords={'x': np.arange(n_x),
                                'level': ERA5_ML_DS.level})
    for var in pres_vars: PAM_DS[var] = xr.DataArray(np.full((n_x, n_lev), np.nan), dims=['x', 'level'])

    for k in range(n_lev):
        for var in pres_vars: PAM_DS[var][:,k] = ERA5_ML_DS[var].values[:,k,:,:].ravel()

    # add time + lat-lon information:
    PAM_DS['longitude'] = xr.DataArray(np.tile(np.reshape(ERA5_ML_DS.longitude.values, (1,1,n_lon)), (n_time, n_lat, 1)).ravel(), dims=['x'])
    PAM_DS['time'] = xr.DataArray(np.tile(np.reshape(ERA5_ML_DS.time.values, (n_time,1,1)), (1, n_lat, n_lon)).ravel(), dims=['x'])
    PAM_DS['latitude'] = xr.DataArray(np.tile(np.reshape(ERA5_ML_DS.latitude.values, (1,n_lat,1)), (n_time, 1, n_lon)).ravel(), dims=['x'])


    # flip the vertical axis:
    PAM_DS = PAM_DS.reindex({'level': PAM_DS.level[::-1]})


    # also put the single level data into PAM_DS
    sing_vars = ['u10', 'v10', 't2m', 'r2m', 'q2m', 'z_sfc', 'sp', 'lsm', 'siconc', 'sst', 'skt']
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
    hyd_mets = ['ciwc', 'clwc', 'crwc', 'cswc']
    PAM_DS['pres'] = xr.DataArray(np.repeat(np.reshape(np.concatenate((np.array([1013.25]), PAM_DS.level.values)), 
                                    (1, n_lev_p)), n_x, axis=0), dims=['x', 'levels'],
                                    coords={'levels': np.concatenate((np.array([1013.25]), PAM_DS.level.values))})
    PAM_DS['temp'] = xr.DataArray(np.zeros((n_x, n_lev_p)), dims=['x', 'levels'])
    PAM_DS['height'] = xr.DataArray(np.zeros((n_x, n_lev_p)), dims=['x', 'levels'])
    PAM_DS['relhum'] = xr.DataArray(np.zeros((n_x, n_lev_p)), dims=['x', 'levels'])
    PAM_DS['q_spec'] = xr.DataArray(np.zeros((n_x, n_lev_p)), dims=['x', 'levels'])
    for hm in hyd_mets: 
        PAM_DS[hm + "_"] = xr.DataArray(np.zeros((n_x, n_lev_p)), dims=['x', 'levels'])

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
    for hm in hyd_mets:
        PAM_DS[hm + "_"][:,1:] = PAM_DS[hm]

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

            # hydrometeors: interpolate from one press. level below sfc to the next pressure level:
            for hm in hyd_mets:
                PAM_DS[hm][ix,idx] = np.interp(p_bridge, np.array([PAM_DS.level[idx[-1]-1], p_bridge[-1]])[::-1],
                                                np.array([PAM_DS[hm + "_"][ix,idx[-1]], PAM_DS[hm + "_"][ix,idx[-1]+1]])[::-1])[:-1]


    # replace old leveled data and remove unneeded variables to save memory:
    PAM_DS['t'] = PAM_DS['temp']
    PAM_DS['z'] = PAM_DS['height']
    PAM_DS['r'] = PAM_DS['relhum']
    PAM_DS['q'] = PAM_DS['q_spec']
    for hm in hyd_mets: PAM_DS[hm] = PAM_DS[hm + "_"]
    PAM_DS = PAM_DS.drop(['temp', 'relhum', 'height', 'q_spec', 'ciwc_', 'clwc_', 'crwc_', 'cswc_'])


    # create pamtra object; change settings:
    pam = pyPamtra.pyPamtra()

    pam.nmlSet['passive'] = True                        # passive simulation
    pam.nmlSet['active'] = False                        # False: no radar simulation

    pamData = dict()
    shape2d = [n_x, 1]


    # set time and location info:
    pamData['lon'] = np.reshape(PAM_DS.longitude.values, shape2d)
    pamData['lat'] = np.reshape(PAM_DS.latitude.values, shape2d)
    pamData['timestamp'] = np.reshape(PAM_DS.time.values.astype('datetime64[s]').astype('float64'), shape2d)
    pamData['obs_height'] = np.broadcast_to(set_dict['obs_hgt'], shape2d + [len(set_dict['obs_hgt']), ])


    # surface type & reflectivity:
    pamData['sfc_slf'] = np.reshape(PAM_DS.lsm.values, shape2d)             # surface land fraction
    pamData['sfc_sif'] = np.reshape(PAM_DS.siconc.values, shape2d)          # surface ice fraction
    pamData['sfc_type'] = np.around(pamData['sfc_slf']).astype('int32')     # 0: ocean, 1: land
    pamData['sfc_model'] = np.zeros(shape2d, dtype='int32')
    pamData['sfc_refl'] = np.chararray(shape2d, unicode=True)
    pamData['sfc_refl'][:] = "F"                            # ocean: Fresnel
    pamData['sfc_refl'][pamData['sfc_type'] > 0] = "S"      # land: Specular

    # sea ice surface: taken from telsem2, defined to be Lambertian
    ice_idx = pamData['sfc_sif'] > 0
    pamData['sfc_type'][ice_idx] = 1
    pamData['sfc_model'][ice_idx] = 0
    pamData['sfc_refl'][ice_idx] = "L"


    # save data to pamData dict:
    pam_open_ocean = (pamData['sfc_slf'] == 0.0) & (pamData['sfc_sif'] == 0.0)
    pamData['groundtemp'] = np.reshape(PAM_DS.skt.values, shape2d)      # also applicable for sea ice and land; but I prefer SST for ocean
    pamData['groundtemp'][pam_open_ocean] = np.reshape(PAM_DS.sst.values, shape2d)[pam_open_ocean]      # sea surface temperature
    pamData['wind10u'] = np.reshape(PAM_DS.u10.values, shape2d)
    pamData['wind10v'] = np.reshape(PAM_DS.v10.values, shape2d)

    # 3d variables:
    shape3d = shape2d + [n_lev_p]
    pamData['hgt_lev'] = np.reshape(PAM_DS.z.values, shape3d)
    pamData['temp_lev'] = np.reshape(PAM_DS.t.values, shape3d)      # in K
    pamData['press_lev'] = np.reshape(PAM_DS.pres.values, shape3d)  # in Pa
    pamData['relhum_lev'] = np.reshape(PAM_DS.r.values, shape3d)    # in %

    # pamData on layers:
    shape3d_lay = shape2d + [n_lev_p-1]
    pamData['hgt'] = (pamData['hgt_lev'][...,1:] + pamData['hgt_lev'][...,:-1])*0.5
    pamData['temp'] = (pamData['temp_lev'][...,1:] + pamData['temp_lev'][...,:-1])*0.5
    pamData['press'] = (pamData['press_lev'][...,1:] + pamData['press_lev'][...,:-1])*0.5
    pamData['relhum'] = (pamData['relhum_lev'][...,1:] + pamData['relhum_lev'][...,:-1])*0.5

    # make sure that relhum stays within 0, 100:
    # pamData['relhum_lev'][pamData['relhum_lev'] > 100.0] = 100.0
    pamData['relhum_lev'][pamData['relhum_lev'] < 0.0] = 0.0
    pamData['relhum'][pamData['relhum'] < 0.0] = 0.0


    # remove unrealistic values
    PAM_DS.clwc.values[PAM_DS.clwc.values < 0.0] = 0.0
    PAM_DS.ciwc.values[PAM_DS.ciwc.values < 0.0] = 0.0
    PAM_DS.crwc.values[PAM_DS.crwc.values < 0.0] = 0.0
    PAM_DS.cswc.values[PAM_DS.cswc.values < 0.0] = 0.0

    # 4d variables: hydrometeors: convert to LAYERS
    shape4d = shape3d_lay + [4]         # potentially 4 hydrometeor classes with this setting
    pamData['hydro_q'] = np.zeros(shape4d)
    pamData['hydro_q'][...,0] = np.reshape((PAM_DS.clwc.values[:,:-1] + PAM_DS.clwc.values[:,1:])*0.5, shape3d_lay)
    pamData['hydro_q'][...,1] = np.reshape((PAM_DS.ciwc.values[:,:-1] + PAM_DS.ciwc.values[:,1:])*0.5, shape3d_lay)
    pamData['hydro_q'][...,2] = np.reshape((PAM_DS.crwc.values[:,:-1] + PAM_DS.crwc.values[:,1:])*0.5, shape3d_lay)
    pamData['hydro_q'][...,3] = np.reshape((PAM_DS.cswc.values[:,:-1] + PAM_DS.cswc.values[:,1:])*0.5, shape3d_lay)

    descriptorFile = "/home/tenweg/pamtra/descriptorfiles/descriptor_file_ecmwf.txt"
    pam.df.readFile(descriptorFile)


    # Create pamtra profile and go:
    pam.createProfile(**pamData)

    sss = dt.datetime.utcnow()
    n_cpus = int(multiprocessing.cpu_count()*0.75)      # taking a fraction of available CPUs
    pam.runParallelPamtra(set_dict['frq'], pp_deltaX=1, pp_deltaY=0, pp_deltaF=0, pp_local_workers=n_cpus)

    ERA5_ML_DS.close()
    ERA5_SL_DS.close()
    PAM_DS.close()

    # save output:
    date_for_filename = file_ml[-7:-3]
    filename_out = os.path.join(path_putput, f"ERA5_PAMTRA_simulation_HALO_HAMP_freq_fram_strait_{date_for_filename}.nc")
    pam.writeResultsToNetCDF(filename_out, xarrayCompatibleOutput=True, ncCompression=True)

    print(f"Saved PAMTRA simulations to {filename_out}.")
    print(f"Simulation time: ", dt.datetime.utcnow() - sss)