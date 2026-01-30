import pdb
import glob
import copy
from copy import deepcopy
import datetime as dt
import gc
import os
import sys
import subprocess

wdir = os.getcwd() + "/"
remote = ((("/net/blanc/" in wdir) | ("/work/awalbroe/" in wdir)) and ("/mnt/f/" not in wdir))      # identify if the code is executed on the blanc computer or at home

import numpy as np
import matplotlib as mpl
if not remote: mpl.use("WebAgg")
import yaml
mpl.rcParams.update({'font.family': 'monospace'})

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from matplotlib.ticker import PercentFormatter


from nn_classes import predictor_class, predictand_class
from data_tools import *

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow

from sklearn.model_selection import KFold

ssstart = dt.datetime.utcnow()


def reduce_dimensions(
    data, 
    check_dims_vars):

    """
    This function reduces dimensions of variables in data defined in check_dims_vars.keys()
    that exceed the dimension number given as value in check_dims_vars. It is expected to
    have the time axis == first axis, height axis == last axis. Furthermore, it's assumed
    that each variable apart for some separately handled ones have the same dimension.

    Parameters:
    -----------
    data : class radiometers, era5
        Data whose attributes are checked for number of dimensions. If exceeding the dimension
        number given in check_dims_vars, the dimensions will be concatenated.
    check_dims_vars : dict
        Contains variable names as keys and their expected dimension number as values.
    """

    # find out how the shape of the other dimensions are that will be concatenated into one:
    other_dims = ()
    for dv in check_dims_vars.keys():
        if (dv in data.__dict__.keys() and check_dims_vars[dv] == 1 
            and data.__dict__[dv].ndim > 1):

            other_dim_new = data.__dict__[dv].shape
            if other_dims and other_dim_new != other_dims:
                raise ValueError("Some other dimensions seem to exist")
            elif not other_dims:
                other_dims = other_dim_new

    # loop over variables and change (flatten) dimensions
    for dim_var in check_dims_vars.keys():

        if (dim_var in data.__dict__.keys() and 
            data.__dict__[dim_var].ndim > check_dims_vars[dim_var]):
            pdb.set_trace()
            # just flatten arrays if dim should be 1; else, handle other dimensions:
            if check_dims_vars[dim_var] == 1:
                # .flatten() reduces a [time,x,y] array to [y*x*time] --> reverses order
                # so that former first dimension changes slowest
                data.__dict__[dim_var] = data.__dict__[dim_var].flatten()

            elif check_dims_vars[dim_var] == 2:
                # flatten the array for each index of the last dimension; 
                # last dimension could be i.e., height or frequency!
                old_shape = data.__dict__[dim_var].shape
                new = np.full((np.prod(old_shape[:-1]), old_shape[-1]), np.nan)
                for i_z in range(old_shape[-1]):
                    new[:,i_z] = data.__dict__[dim_var][...,i_z].flatten()

                data.__dict__[dim_var] = new
                del new, old_shape
            
    return data


def apply_sea_mask(
    data, 
    sfc_mask, 
    check_dims_vars):

    """
    Reduce variables given in check_dims_vars of data to sea only grid cells as specified in
    sfc_mask.

    Parameters:
    -----------
    data : class radiometers, era5
        Data whose variables will be reduced to sea only regions given by sfc_mask.
    sfc_mask : array
        Surface mask depicting the land fraction of data. Must have the same shape as non-height
        dependend (or other secondary dimensions) variables of data. 
    check_dims_vars : dict
        Contains variable names as keys and their expected dimension number as values.
    """

    # loop through variables and reduce them to indices where sfc_mask is True.
    for dim_var in check_dims_vars:
        if dim_var in data.__dict__.keys():
            if check_dims_vars[dim_var] == 1 and dim_var not in ['freq']:
                data.__dict__[dim_var] = data.__dict__[dim_var][sfc_mask]

            elif check_dims_vars[dim_var] == 2:
                data.__dict__[dim_var] = data.__dict__[dim_var][sfc_mask,:]

    return data


def hamp_tb_offset_correction(
    DS,
    path_offsets):

    """
    Corrects TB offsets for microwave radiometers in HALO's microwave package HAMP. 

    Parameters:
    -----------
    DS : xarray dataset
        Dataset containing TB data as variable 'tb' on a (time,frequency) 2D-array. Time (frequency) 
        dimension and variable name must be called 'time' ('freq').
    path_offsets : str
        Path where the netCDF containing the offsets, slopes and biases are located. The files are
        the output of CSSC_v2.
    """

    cur_date = str(DS.time.values[0].astype('datetime64[D]')).replace("-","")       # current date in yyyymmdd

    try:
        OFF_DS = xr.open_dataset(path_offsets + f"HALO-AC3_HALO_HAMP_TB_offset_correction_{cur_date}.nc")
    except FileNotFoundError:
        print("WARNING! TB offset correction has been attempted, but file for offset correction has not been found!")
        return DS


    # compute offset corrected TBs for the current research flight and each freq
    DS['tb_cor'] = deepcopy(DS['tb'])
    DS['tb_cor'][:, :] = (OFF_DS.slope.squeeze().values*DS['tb_cor'] + OFF_DS.offset.squeeze().values)

    # replace old TBs by corrected ones:
    OFF_DS = OFF_DS.close()
    del DS['tb'], OFF_DS
    DS = DS.rename({'tb_cor': 'tb'})

    return DS


def halo_clear_sky_detection(
    MWR_DS,
    aux_i,
    date_now):

    """
    Detect clear sky periods based on radar reflectivity thresholds and MWR TB standar deviations.
    The returned array will be a mask indicating clear sky as True, cloudy as False on the MWR
    time axis. Also BAHAMAS data is needed for flight altitude information.

    Parameters:
    -----------
    MWR_DS : xarray dataset
        Dataset containing TB data on a (time,frequency) grid.
    aux_i : dict
        Dictionary containing additional information.
    date_now : str
        String indicating the current date to load the respective BAHAMAS and radar data. Must be
        given in "yyyy-mm-dd".
    """

    def cut_vars(DS):

        # remove unnecessary variables:
        remove_vars = ["tpow", "npw1", "npw2", "cpw1", "cpw2", "grst", 
                        "aziv", "LO_Frequency", "DetuneFine", "SNRgc", "VELgc", "RMSgc", 
                        "LDRgc", "NPKgc", "SNRg", "VELg", "RMSg", "LDRg", 
                        "NPKg", "SNRcx", "RHO", "DPS", "RHOwav", "LDRnormal", "HSDco", 
                        "HSDcx", "ISDRco", "ISDRcx", "MRMco", "MRMcx", "RadarConst",
                        "SNRCorFaCo", "SNRCorFaCx", "SKWg"]
        DS = DS.drop_vars(remove_vars, errors='ignore')

        # compute radar refl in dBZ:
        # DS['dBZ'] = 10*np.log10(DS.Zg)                # equivalent reflectivity factor in dBZ
        if 'dBZg' in DS.variables: DS['dBZ'] = DS['dBZg']

        return DS

    # import radar data:
    test_date_file = date_now.replace('-','')
    # file_radar = sorted(glob.glob(aux_i['path_radar_obs'] + f"HALO-AC3_HALO_hamp_mira_{test_date_file}*.nc"))     # old files
    file_radar = sorted(glob.glob(aux_i['path_radar_obs'] + f"HALO_HALO_AC3_radar_unified_*.nc"))
    file_radar = [file for file in file_radar if test_date_file in os.path.basename(file)]
    radar_cloudy_flag = np.full((len(MWR_DS.time),), False)
    if len(file_radar) > 0:
        RADAR_DS = xr.open_dataset(file_radar[0])
        RADAR_DS = cut_vars(RADAR_DS)

        # import BAHAMAS data:
        file_bah = sorted(glob.glob(aux_i['path_bahamas_obs'] + f"bahamas_{test_date_file}_*.nc"))[0]
        BAH_DS = xr.open_dataset(file_bah)


        # interpolate flight altitude on radar time axis:
        if not "alt" in RADAR_DS.variables:
            RADAR_DS['alt'] = xr.DataArray(np.interp(RADAR_DS.time.values.astype('datetime64[s]').astype(np.float64), 
                                            BAH_DS.time.values.astype('datetime64[s]').astype(np.float64), BAH_DS.alt.values),
                                            dims=['time'])

        # compute height to avoid ground clutter:
        radar_height = None
        if not "height" in RADAR_DS.variables:
            radar_height = np.full((len(RADAR_DS.time), len(RADAR_DS.range)), np.nan)
            for kk in range(len(RADAR_DS.time)):
                radar_height[kk,:] = RADAR_DS['alt'].values[kk] - RADAR_DS.range.values
            RADAR_DS['height'] = xr.DataArray(radar_height, dims=['time', 'range'])

        
        # count bins with radar reflectivity > -40 dBZ for a time step (watch for ground clutter):
        # also limit height to avoid pure-ice clouds:
        RADAR_DS['cloudy_flag'] = xr.where((RADAR_DS.dBZ > -40.0) & (RADAR_DS.height > 300.0) & (RADAR_DS.height < 4000.0), True, False)
        try:
            RADAR_DS['refl_bins_count'] = RADAR_DS['cloudy_flag'].sum("range")
        except ValueError:
            RADAR_DS['refl_bins_count'] = RADAR_DS['cloudy_flag'].sum("height")

        radar_cloudy_flag = (RADAR_DS['refl_bins_count'].values > 5).astype('int')      # 1 = cloudy, 0 = clear or not detectable

        # interpolate radar_cludy_flag to MWR time axis:
        radar_cloudy_flag = np.interp(MWR_DS.time.values.astype('datetime64[s]').astype(np.float64),
                                        RADAR_DS.time.values.astype('datetime64[s]').astype(np.float64),
                                        radar_cloudy_flag, left=0, right=0) + 0.0000001 # the addition is needed so that rounding works correctly
        radar_cloudy_flag = np.round(radar_cloudy_flag).astype('bool')

        # clear memory:     
        del RADAR_DS, radar_height


    # std dev of MWR TBs over 10 seconds: based on Marek Jacob's dissertation, p. 40
    dtime = 30      # in seconds
    tb_std_threshold = 0.5      # in K
    tb_std = np.ones(MWR_DS.tb.shape)*(-1.0)
    tb_std_max = np.ones(MWR_DS.tb.shape)*(-1.0)
    for i_f in range(len(MWR_DS.freq)):
        tb_std_roll = MWR_DS.tb[:,i_f].to_dataframe(name='tb').rolling(f"{int(dtime)}S", center=True).std()
        tb_std_max[:,i_f] = tb_std_roll.rolling(f"{int(dtime)}S", center=True).max().to_xarray().tb
        tb_std[:,i_f] = tb_std_roll.to_xarray().tb


    # proxy for cloudy scenes: find if std dev threshold is surpassed:
    # The channels (K,V,W,F) are assigned clear sky if the std. in ALL (K,V,W,F) channels 
    # is less than the threshold. In case ANY non G band channel shows a greater std. dev.,
    # cloudy sky scene is assumed.
    clear_sky_mask = np.full((len(MWR_DS.time),), False)

    # find indices of the different bands:
    idx_KVWF = select_MWR_channels(MWR_DS.tb.values, MWR_DS.freq.values, band='K+V+W+F', return_idx=2)
    if len(idx_KVWF) > 0:
        clear_rest = np.all(tb_std_max[:,idx_KVWF] <= tb_std_threshold, axis=1)
    
    # set clear sky to false when radar detects some cloudy looking scenes:
    clear_sky_mask[(clear_rest) & (~radar_cloudy_flag)] = True
    # clear_sky_mask[clear_rest] = True         # if only the tb_std flag is to be used


    # adding another criterion: clear sky conditions must persist for at least 10 seconds to avoid
    # many jumps in the offset correction:
    clear_sky_mask_int = clear_sky_mask.astype(np.int32)
    idx_clear_sky = np.where(clear_sky_mask_int == 1)[0]
    idx_clear_start = np.where(np.diff(clear_sky_mask_int) == 1)[0] + 1     # beginning of clear sky period
    idx_clear_end = np.where(np.diff(clear_sky_mask_int) == -1)[0] + 1      # end of clear sky period

    # manage clear sky at time axis borders:
    if (idx_clear_start.shape != idx_clear_end.shape) and clear_sky_mask[-1]:
        # then, time axis ends with clear sky:
        idx_clear_end = np.concatenate((idx_clear_end, np.array([-1])))
    elif (idx_clear_start.shape != idx_clear_end.shape) and clear_sky_mask[0]:
        # then, time axis starts with clear sky:
        idx_clear_start = np.concatenate((np.array([0]), idx_clear_start))

    # temporal duration of each clear sky period:
    t_diff_clear = MWR_DS.time.values[idx_clear_end] - MWR_DS.time.values[idx_clear_start]

    # set those clear sky periods to cloudy (=False) that are too short:
    min_clear_time = dtime      # minimum clear sky period time in seconds
    for ics, ice, t_clear in zip(idx_clear_start, idx_clear_end, t_diff_clear): 
        if t_clear < np.timedelta64(min_clear_time, "s"):
            if ice != -1:
                clear_sky_mask[ics:ice] = False
            else:   # because of python indexing
                clear_sky_mask[ics:] = False


    # code block to visualize tb_std and masks:
    # f1 = plt.figure(figsize=(15,6))
    # a1 = plt.axes()
    # for f in range(len(MWR_DS.freq)):
        # a1.plot(MWR_DS.time, tb_std_max[:,f], label=f"{MWR_DS.freq[f].values:.2f}")
    # a1.plot(MWR_DS.time, (~radar_cloudy_flag).astype(np.int32)*(-0.25) - 0.25, linewidth=1.2, color=(1,0.12,0.8,0.5))
    # a1.plot(MWR_DS.time, clear_sky_mask.astype(np.int32)*(-1), linewidth=1.2, color=(0,0,0,0.5))
    # a1.set_ylim(-1.1,2)
    # lh,ll = a1.get_legend_handles_labels()
    # a1.legend(lh, ll, loc='upper right')
    # a1.grid(which='both', axis='both')
    # # f1.savefig(f"/mnt/f/Studium_NIM/work/Plots/HALO_AC3/lwp_retrieval/tb_std_{str(MWR_DS.time.values[0].astype('datetime64[D]'))}.png", dpi=250)
    # plt.show()
    # plt.close()
    # pdb.set_trace()

    return clear_sky_mask


def halo_offset_lwp(
    time,
    LWP,
    clear_sky_mask):

    """
    For LWP, clear sky offsets are corrected. 

    Parameters:
    time : array of floats
        Array containing the time steps (in seconds since 1970-01-01 00:00:00 UTC) of the LWP.
    LWP : array of floats
        Array of Liquid Water Path (LWP, in kg m-2), which is supposed to be corrected.
    clear_sky_mask : array of bool
        Output of halo_clear_sky_detection to identify clear sky scenes (=True) for the offset
        correction.
    """

    time_npdt = time.astype('datetime64[s]')
    dtime = 30      # time over which rolling mean for offset correction and clear sky detection 
                    # is used in seconds

    LWP_cor = np.full_like(LWP, np.nan)     # will contain the corrected LWP for all days
    LWP_off = np.full_like(LWP, np.nan)     # will contain 20-min mean of LWP in clear sky tiem steps
    cloudy_flag = (~clear_sky_mask).astype('int')   # == 0 for clear sky, == 1 for cloudy scenes


    # Compute rolling mean over entire time axis and then use the clear sky flag to identify
    # correct LWP values for offset correction:
    # Need xarray DataArrays for functionalities:
    LWP_DA = xr.DataArray(LWP, dims=['time'], coords={'time': time_npdt})
    LWP_DF = LWP_DA.to_dataframe(name='LWP')    # PANDAS DF to be used to have rolling window width in time units
    LWP_off = LWP_DF.rolling(f"{int(dtime)}S", center=True).mean().to_xarray().LWP  # mean LWP over dtime s <-> serves as offset in clear sky

    idx_cloudy = np.where(cloudy_flag == 1)[0]
    idx_clear_sky = np.where(cloudy_flag == 0)[0]
    LWP_off[idx_cloudy] = np.nan


    # LWP offsets for cloudy time steps are computed by interpolating from adjacent clear sky periods:
    max_gap_val = np.timedelta64(6, "h")
    LWP_off = LWP_off.interpolate_na(dim='time', method='linear', max_gap=max_gap_val, fill_value=0.0)

    # handle potential nans that still exist because max_gap and fill_value don't work together:
    # Causes of nans: cloudy scenes, measurement gaps. The following cases still need to be covered:
    # 1. Cloudy after measurement gap: take LWP_off of first clear sky scene after measurement gap
    # 2. Cloudy for longer period than max_gap: just expand interpolation
    # 3. Cloudy before measurement gap: take the LWP_off of the latest clear sky scene before the gap
    still_nan = np.isnan(LWP_off.values)
    if np.all(still_nan): 
        LWP_off[:] = 0.0    # otherwise, LWP_cor is also nan

    # handle measurement gaps: identify via time differences larger than max_gap_val. If shorter
    # measurement gaps are seen, it's considered that the interpolation does a decent job.
    meas_gaps = np.where(np.diff(time_npdt) > max_gap_val)[0]

    # check sky state around measurement gaps:
    for meas_gap in meas_gaps:
        # meas_gap + 1 == first value after measurement gap
        if meas_gap > 0:
            cloudy_before = np.isnan(LWP_off[meas_gap].values)
            cloudy_after = np.isnan(LWP_off[meas_gap+1].values)

            if cloudy_before:   # take latest LWP_off
                idx_last_clear_sky = np.where(cloudy_flag[:meas_gap+1] == 0)[0]
                if len(idx_last_clear_sky) > 0:
                    idx_last_clear_sky = idx_last_clear_sky[-1]

                    # check if temporal distance is okay:
                    if (time_npdt[meas_gap] - time_npdt[idx_last_clear_sky]) <= np.timedelta64(72, "h"):
                        LWP_off[idx_last_clear_sky+1:meas_gap+1] = LWP_off[idx_last_clear_sky]

            if cloudy_after:    # take first LWP_off
                idx_first_clear_sky = np.where(cloudy_flag[meas_gap+1:] == 0)[0] + meas_gap + 1
                if len(idx_first_clear_sky) > 0:
                    idx_first_clear_sky = idx_first_clear_sky[0]

                    # check if temporal distance is okay:
                    if (time_npdt[idx_first_clear_sky] - time_npdt[meas_gap+1]) <= np.timedelta64(72, "h"):
                        LWP_off[meas_gap+1:idx_first_clear_sky] = LWP_off[idx_first_clear_sky]


    # take care of the array boundaries similar as the measurement gaps if the temporal distance
    # to the clear sky scene isn't too far away:
    if np.isnan(LWP_off[0].values) and ((time_npdt[idx_clear_sky[0]] - time_npdt[0]) < np.timedelta64(72, "h")):
        LWP_off[:idx_clear_sky[0]] = LWP_off[idx_clear_sky[0]]
    if np.isnan(LWP_off[-1].values) and((time_npdt[-1] - time_npdt[idx_clear_sky[-1]]) < np.timedelta64(72, "h")):
        LWP_off[idx_clear_sky[-1]+1:] = LWP_off[idx_clear_sky[-1]]

    # # avoid correcting 'smooth clouds' where tb_std was low but LWP has a significant signal:
    # # just don't correct offsets > 40 g m-2:
    # idx_smooth_cloud = np.where(np.abs(LWP_off) > 100.)[0]
    # LWP_off[idx_smooth_cloud] = 0.0

    # interpolate over longer cloudy periods and set remaining nans to 0:
    LWP_off = LWP_off.interpolate_na(dim='time', method='linear', max_gap=np.timedelta64(72, "h"))
    LWP_off[np.isnan(LWP_off)] = 0.0


    # correct LWP and save it to LWP_cor:
    LWP_cor = LWP - LWP_off

    return LWP_cor


def specify_output(
    predictand_obj,
    predictand_list,
    n_samples):

    """
    Specify output based on the predictand_list. Each output will be concatenated, respecting its
    feature dimension (e.g., LWP has got only one feature while temperature profile may have more
    features, depending on height grid).

    Parameters:
    -----------
    predictand_obj : class object
        Object of the class predictand_class in nn_classes.py. Here, the predictands will be 
        concatenated along axis=1, respecting each predictand's size per sample.
    predictand_list : list of str
        List indicating which predictands have been chosen (what is to be retrieved).
    n_samples : int
        Number of samples in the predictand data. Used to ensure the correct shape while reshaping.
    """

    # specify output:
    predictand_obj.output = None
    aux_i['n_ax1'] = dict()         # will contain information about the length of the 2nd dimension (i.e., height)
                                    # and should be identical among all predictand_obj
    for k, predictand in enumerate(predictand_list):

        aux_i['n_ax1'][predictand] = 0      # dimension of axis=1 of predictand (needed for concatenation)
        if predictand_obj.__dict__[predictand].ndim == 1: 
            aux_i['n_ax1'][predictand] = 1
        elif predictand_obj.__dict__[predictand].ndim == 2:
            aux_i['n_ax1'][predictand] = predictand_obj.__dict__[predictand].shape[1]
        else:
            raise ValueError(f"Unexpected shape of {predictand} while building the output vector.")

        # concatenate predictands into an output vector:
        if k == 0:
            predictand_obj.output = np.reshape(predictand_obj.__dict__[predictand], (n_samples, aux_i['n_ax1'][predictand]))
        else:
            predictand_obj.output = np.concatenate((predictand_obj.output, 
                                                        np.reshape(predictand_obj.__dict__[predictand], (n_samples, aux_i['n_ax1'][predictand]))),
                                                        axis=1)

    return predictand_obj


def compute_error_stats(
    prediction, 
    predictand, 
    predictand_id,
    height=np.array([])):

    """
    Compute error statistics (Root Mean Squared Error (rmse), bias, Standard Deviation (stddev))
    between prediction and (test data) predictand. Height must be provided if prediction or respective
    predictand is a profile. The prediction_id describes the prediction and predictand and must also
    be forwarded to the function.

    Parameters:
    -----------
    prediction : array of floats
        Predicted variables also available in predictand, predicted by the Neural Network.
    predictand : array of floats
        Predictand data as array, used as evaluation reference. Likely equals the attribute 
        'output' of the predictand class object.
    predictand_id : str
        String indicating which output variable is forwarded to the function.
    height : array of floats
        Height array for respective predictand or predictand profiles (of i.e., temperature or 
        humidity). Can be a 1D or 2D array (latter must be of shape (n_training,n_height)).
    """

    error_dict = dict()

    # on x axis: reference; y axis: prediction
    x_stuff = predictand
    y_stuff = prediction

    # Compute statistics:
    if predictand_id in ['iwv', 'lwp']:
        # remove redundant dimension:
        x_stuff = x_stuff.squeeze()
        y_stuff = y_stuff.squeeze()
        stats_dict = compute_retrieval_statistics(x_stuff.squeeze(), y_stuff.squeeze(), compute_stddev=True)

        # For entire range:
        error_dict['rmse_tot'] = stats_dict['rmse']
        error_dict['stddev'] = stats_dict['stddev']
        error_dict['bias_tot'] = stats_dict['bias']

        # also compute rmse and bias for specific ranges only:
        # 'bias': np.nanmean(y_stuff - x_stuff),
        # 'rmse': np.sqrt(np.nanmean((x_stuff - y_stuff)**2)),
        range_dict = dict()
        if predictand_id == 'iwv':  # in mm
            range_dict['bot'] = [0,5]
            range_dict['mid'] = [5,10]
            range_dict['top'] = [10,100]
        elif predictand_id == 'lwp': # in g m^-2
            range_dict['bot'] = [0,25]
            range_dict['mid'] = [25,100]
            range_dict['top'] = [100, 1e+06]
        # elif predictand_id == 'lwp': # in kg m^-2
            # range_dict['bot'] = [0,0.025]
            # range_dict['mid'] = [0.025,0.100]
            # range_dict['top'] = [0.100, 1e+06]

        mask_range = dict()
        x_stuff_range = dict()
        y_stuff_range = dict()
        stats_dict_range = dict()
        for range_id in range_dict.keys():
            mask_range[range_id] = ((x_stuff >= range_dict[range_id][0]) & (x_stuff < range_dict[range_id][1]))
            x_stuff_range[range_id] = x_stuff[mask_range[range_id]]
            y_stuff_range[range_id] = y_stuff[mask_range[range_id]]

            # compute retrieval stats for the respective ranges:
            stats_dict_range[range_id] = compute_retrieval_statistics(x_stuff_range[range_id], y_stuff_range[range_id], compute_stddev=True)
            error_dict[f"rmse_{range_id}"] = stats_dict_range[range_id]['rmse']
            error_dict[f"stddev_{range_id}"] = stats_dict_range[range_id]['stddev']
            error_dict[f"bias_{range_id}"] = stats_dict_range[range_id]['bias']

    return error_dict


def visualize_evaluation(
    prediction, 
    predictand, 
    predictand_id,
    ret_stats_dict,
    aux_i,
    height=np.array([])):

    """
    Visualize the evaluation of the Neural Network prediction against a predictand (i.e., test data).
    Depending on the predicted variable (specified by predictand_id), different plots will be created:
    IWV: scatter plot, LWP: scatter plot, temperature profile: standard deviation and bias profile,
    specific humidity profile: standard deviation and bias profile

    Parameters:
    -----------
    prediction : array of floats
        Predicted variables also available in predictand_class.output, predicted by the Neural Network.
    predictand : array of floats
        Predictand (i.e., of test data) data as array, used as evaluation reference. Likely equals the attribute 
        'output' of the predictand class object.
    predictand_id : str
        String indicating which output variable is forwarded to the function.
    ret_stats_dict : dict
        Dictionary which has got several retrieval statistics as values and their names as keys. Output of function
        compute_error_stats.
    aux_i : dict
        Dictionary containing additional information.
    height : array of floats
        Height array for respective predictand or predictand profiles (of i.e., temperature or 
        humidity). Can be a 1D or 2D array (latter must be of shape (n_training,n_height)).
    """

    if predictand_id in ['temp', 'q'] and len(height) == 0:
            raise ValueError("Please specify a height variable to estimate error statistics for profiles.")


    # create output path if not existing:
    plotpath_dir = os.path.dirname(aux_i['path_plots'] + f"{predictand_id}/")
    if not os.path.exists(plotpath_dir):
        os.makedirs(plotpath_dir)

    # visualize:
    fs = 26
    fs_small = fs - 2
    fs_dwarf = fs_small - 2
    fs_micro = fs_dwarf - 2
    msize = 7.0

    c_H = (0.7,0,0)


    # IWV scatter plot:
    if predictand_id == 'iwv':

        predictand = predictand[:,0]
        prediction = prediction[:,0]

        # again have to compute retrieval stats for N and R:
        ret_stats_temp = compute_retrieval_statistics(predictand, prediction)

        f1 = plt.figure(figsize=(9,9))
        a1 = plt.axes()

        ax_lims = np.asarray([0.0, 35.0])

        # plotting:
        a1.plot(predictand, prediction, linestyle='none', color=c_H, marker='.', markersize=msize,
                markeredgecolor=(0,0,0), label='Prediction')

        # generate a linear fit with least squares approach: notes, p.2:
        # filter nan values:
        nonnan_idx = np.argwhere(~np.isnan(prediction) & ~np.isnan(predictand)).flatten()
        y_fit = prediction[nonnan_idx]
        x_fit = predictand[nonnan_idx]

        # there must be at least 2 measurements to create a linear fit:
        if (len(y_fit) > 1) and (len(x_fit) > 1):
            G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T     # must be transposed because of python's strange conventions
            m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)  # least squares solution
            a = m_fit[0]
            b = m_fit[1]

            ds_fit = a1.plot(ax_lims, a*ax_lims + b, color=c_H, linewidth=1.2, label="Best fit")

        # plot a line for orientation which would represent a perfect fit:
        a1.plot(ax_lims, ax_lims, color=(0,0,0), linewidth=1.2, alpha=0.5, label="Theoretical perfect fit")


        # add statistics:
        a1.text(0.99, 0.01, f"N = {ret_stats_temp['N']}\nMean = {np.mean(np.concatenate((x_fit, y_fit), axis=0)):.2f}\n" +
                f"bias = {ret_stats_dict['bias_tot']:.2f}\nrmse = {ret_stats_dict['rmse_tot']:.2f}\n" +
                f"std. = {ret_stats_dict['stddev']:.2f}\nR = {ret_stats_temp['R']:.3f}", 
                ha='right', va='bottom', transform=a1.transAxes, fontsize=fs_dwarf)


        # Legends:
        lh, ll = a1.get_legend_handles_labels()
        a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.05, 1.00), fontsize=fs_micro-4,
                    framealpha=0.5)

        # set axis limits:
        a1.set_ylim(bottom=ax_lims[0], top=ax_lims[1])
        a1.set_xlim(left=ax_lims[0], right=ax_lims[1])

        # set axis ticks, ticklabels and tick parameters:
        a1.minorticks_on()
        a1.tick_params(axis='both', labelsize=fs_micro-4)

        # aspect ratio:
        a1.set_aspect('equal')

        # grid:
        a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

        # labels:
        a1.set_ylabel("IWV$_{\mathrm{prediction}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
        a1.set_xlabel("IWV$_{\mathrm{reference}}$ ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs)
        a1.set_title(f"{aux_i['file_descr']}", fontsize=fs)

        if aux_i['save_figures']:
            plotname = f"HALO-AC3_NN_ret_eval_{predictand_id}_scatterplot_{aux_i['file_descr']}"
            f1.savefig(aux_i['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


        # error diff composit: Generate bins and compute RMSE, Bias for each bin:
        val_max = 25.0
        val_bins = np.array([np.arange(0., val_max-2.+0.001, 2.), np.arange(2., val_max+0.001, 2.)]).T

        # compute errors for each bin
        RMSE_bins = np.full((val_bins.shape[0],), np.nan)
        BIAS_bins = np.full((val_bins.shape[0],), np.nan)
        N_bins = np.zeros((val_bins.shape[0],))     # number of matches for each bin
        for ibi, val_bin in enumerate(val_bins):
            # find indices for the respective bin (based on the reference (==truth)):
            idx_bin = np.where((predictand >= val_bin[0]) & (predictand < val_bin[1]))[0]
            N_bins[ibi] = len(idx_bin)

            # compute errors:
            RMSE_bins[ibi] = np.sqrt(np.nanmean((prediction[idx_bin] - predictand[idx_bin])**2))
            BIAS_bins[ibi] = np.nanmean(prediction[idx_bin] - predictand[idx_bin])


        # visualize:
        f1 = plt.figure(figsize=(11,7))
        a1 = plt.axes()

        # deactivate some spines:
        a1.spines[['right', 'top']].set_visible(False)

        ax_lims = np.asarray([0.0, val_max])
        er_lims = np.asarray([-1.5, 1.5])

        # plotting:
        # thin lines indicating RELATIVE errors:
        rel_err_contours = np.array([1.0,2.0,5.0,10.0,20.0])
        rel_err_range = np.arange(0.0, val_max+0.0001, 0.01)
        rel_err_curves = np.zeros((len(rel_err_contours), len(rel_err_range)))
        for i_r, r_e_c in enumerate(rel_err_contours):
            rel_err_curves[i_r,:] = rel_err_range*r_e_c / 100.0
            a1.plot(rel_err_range, rel_err_curves[i_r,:], color=(0,0,0,0.5), linewidth=0.75, linestyle='dotted')
            a1.plot(rel_err_range, -1.0*rel_err_curves[i_r,:], color=(0,0,0,0.5), linewidth=0.75, linestyle='dotted')

            # add annotation (label) to rel error curve:
            rel_err_label_pos_x = er_lims[1] * 100. / r_e_c
            if rel_err_label_pos_x > val_max:
                a1.text(ax_lims[1], ax_lims[1]*r_e_c / 100., f"{int(r_e_c)} %",
                    color=(0,0,0,0.5), ha='left', va='center', transform=a1.transData, fontsize=fs_micro-6)
            else:
                a1.text(rel_err_label_pos_x, er_lims[1], f"{int(r_e_c)} %", 
                    color=(0,0,0,0.5), ha='left', va='bottom', transform=a1.transData, fontsize=fs_micro-6)

        val_bins_plot = (val_bins[:,1] - val_bins[:,0])*0.5 + val_bins[:,0]
        a1.plot(ax_lims, [0,0], color=(0,0,0))
        a1.plot(val_bins_plot, RMSE_bins, color=(0.11,0.46,0.70), linewidth=1.2, label='RMSE')
        a1.plot(val_bins_plot, BIAS_bins, color=(0.11,0.46,0.70), linewidth=1.2, linestyle='dashed', label='Bias')

        
        # Legends:
        lh, ll = a1.get_legend_handles_labels()
        a1.legend(handles=lh, labels=ll, loc='lower left', bbox_to_anchor=(0.02, 0.00), fontsize=fs_micro-4,
                    framealpha=0.5)

        # set axis limits:
        a1.set_ylim(bottom=er_lims[0], top=er_lims[1])
        a1.set_xlim(left=ax_lims[0], right=ax_lims[1])

        # set axis ticks, ticklabels and tick parameters:
        a1.minorticks_on()
        a1.tick_params(axis='both', labelsize=fs_micro-4)

        # grid:
        a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

        # labels:
        a1.set_ylabel("Error: Predicted - reference IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs_micro-2)
        a1.set_xlabel("Reference IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs_micro-2)
        a1.set_title(f"{aux_i['file_descr']}", fontsize=fs_micro)

        if aux_i['save_figures']:
            plotname = f"HALO-AC3_NN_ret_eval_{predictand_id}_err_diff_comp_{aux_i['file_descr']}"
            f1.savefig(aux_i['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


    if predictand_id == 'lwp':

        predictand = predictand[:,0]
        prediction = prediction[:,0]

        # again have to compute retrieval stats for N and R:
        ret_stats_temp = compute_retrieval_statistics(predictand, prediction)

        f1 = plt.figure(figsize=(18,9))
        a1 = plt.subplot2grid((1,2), (0,0), fig=f1)
        a2 = plt.subplot2grid((1,2), (0,1), fig=f1)

        ax_lims = np.asarray([-50.0, 1000.0])       # g m-2
        ax_lims2 = np.asarray([0.0, 250.0])     # g m-2

        # plotting:
        a1.plot(predictand, prediction, linestyle='none', color=c_H, marker='.', markersize=msize,
                markeredgecolor=(0,0,0), label='Prediction')
        a2.plot(predictand, prediction, linestyle='none', color=c_H, marker='.', markersize=msize,
                markeredgecolor=(0,0,0))

        # generate a linear fit with least squares approach: notes, p.2:
        # filter nan values:
        nonnan_idx = np.argwhere(~np.isnan(prediction) & ~np.isnan(predictand)).flatten()
        y_fit = prediction[nonnan_idx]
        x_fit = predictand[nonnan_idx]

        # there must be at least 2 measurements to create a linear fit:
        if (len(y_fit) > 1) and (len(x_fit) > 1):
            G_fit = np.array([x_fit, np.ones((len(x_fit),))]).T     # must be transposed because of python's strange conventions
            m_fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_fit.T, G_fit)), G_fit.T), y_fit)  # least squares solution
            a = m_fit[0]
            b = m_fit[1]

            ds_fit = a1.plot(ax_lims, a*ax_lims + b, color=c_H, linewidth=1.2, label="Best fit")
            ds_fit = a2.plot(ax_lims2, a*ax_lims2 + b, color=c_H, linewidth=1.2)

        # plot a line for orientation which would represent a perfect fit:
        a1.plot(ax_lims, ax_lims, color=(0,0,0), linewidth=1.2, alpha=0.5, label="Theoretical perfect fit")
        a2.plot(ax_lims2, ax_lims2, color=(0,0,0), linewidth=1.2, alpha=0.5)


        # add statistics:
        a1.text(0.99, 0.01, f"N = {ret_stats_temp['N']}\nMean = {np.mean(np.concatenate((x_fit, y_fit), axis=0)):.2f}\n" +
                f"bias = {ret_stats_dict['bias_tot']:.2f}\nrmse = {ret_stats_dict['rmse_tot']:.2f}\n" +
                f"std. = {ret_stats_dict['stddev']:.2f}\nR = {ret_stats_temp['R']:.3f}", 
                ha='right', va='bottom', transform=a1.transAxes, fontsize=fs_dwarf)


        # Legends:
        lh, ll = a1.get_legend_handles_labels()
        a1.legend(handles=lh, labels=ll, loc='upper left', bbox_to_anchor=(0.05, 1.00), fontsize=fs_dwarf,
                    framealpha=0.5)

        # set axis limits:
        a1.set_ylim(bottom=ax_lims[0], top=ax_lims[1])
        a1.set_xlim(left=ax_lims[0], right=ax_lims[1])
        a2.set_ylim(bottom=ax_lims2[0], top=ax_lims2[1])
        a2.set_xlim(left=ax_lims2[0], right=ax_lims2[1])

        # set axis ticks, ticklabels and tick parameters:
        a1.minorticks_on()
        a1.tick_params(axis='both', labelsize=fs_dwarf)
        a2.minorticks_on()
        a2.tick_params(axis='both', labelsize=fs_dwarf)

        # aspect ratio:
        a1.set_aspect('equal')
        a2.set_aspect('equal')

        # grid:
        a1.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)
        a2.grid(which='both', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

        # labels:
        a1.set_ylabel("LWP$_{\mathrm{prediction}}$ ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
        a2.set_ylabel("LWP$_{\mathrm{prediction}}$ ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
        a1.set_xlabel("LWP$_{\mathrm{reference}}$ ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
        a2.set_xlabel("LWP$_{\mathrm{reference}}$ ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs)
        a1.set_title(f"{aux_i['file_descr']}", fontsize=fs)

        if aux_i['save_figures']:
            plotname = f"HALO-AC3_NN_ret_eval_{predictand_id}_scatterplot_{aux_i['file_descr']}"
            f1.savefig(aux_i['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


        # error diff composit: Generate bins and compute RMSE, Bias for each bin:
        # # # predictand *= 1000.0  # convert to g m-2
        # # # prediction *= 1000.0  # convert to g m-2
        val_max = 1000.0        # in g m-2
        val_bins_log = np.array([np.arange(0.0, 3.001, 0.1), np.arange(0.1, 3.101, 0.1)]).T     # in log10 of g m-2 scale
        val_bins = 10**val_bins_log     # in g m-2


        # compute errors for each bin
        RMSE_bins = np.full((val_bins.shape[0],), np.nan)       # in g m-2
        BIAS_bins = np.full((val_bins.shape[0],), np.nan)       # in g m-2
        N_bins = np.zeros((val_bins.shape[0],))     # number of matches for each bin
        for ibi, val_bin in enumerate(val_bins):
            # find indices for the respective bin (based on the reference (==truth)):
            idx_bin = np.where((predictand >= val_bin[0]) & (predictand < val_bin[1]))[0]
            N_bins[ibi] = len(idx_bin)

            # compute errors:
            RMSE_bins[ibi] = np.sqrt(np.nanmean((prediction[idx_bin] - predictand[idx_bin])**2))
            BIAS_bins[ibi] = np.nanmean(prediction[idx_bin] - predictand[idx_bin])


        # visualize:
        f1 = plt.figure(figsize=(11,7))
        a1 = plt.axes()

        # deactivate some spines:
        a1.spines[['right', 'top']].set_visible(False)

        ax_lims = np.asarray([1.0, val_max])    # in g m-2
        er_lims = np.asarray([-80, 80])         # in g m-2

        # plotting:
        # thin lines indicating RELATIVE errors:
        rel_err_contours = np.array([10.,20.,50.,100.])
        rel_err_range = np.arange(0.0, val_max+0.0001, 0.01)
        rel_err_curves = np.zeros((len(rel_err_contours), len(rel_err_range)))
        for i_r, r_e_c in enumerate(rel_err_contours):
            rel_err_curves[i_r,:] = rel_err_range*r_e_c / 100.0
            a1.plot(rel_err_range, rel_err_curves[i_r,:], color=(0,0,0,0.5), linewidth=0.75, linestyle='dotted')
            a1.plot(rel_err_range, -1.0*rel_err_curves[i_r,:], color=(0,0,0,0.5), linewidth=0.75, linestyle='dotted')

            # add annotation (label) to rel error curve:
            rel_err_label_pos_x = er_lims[1] * 100. / r_e_c
            if rel_err_label_pos_x > val_max:
                a1.text(ax_lims[1], ax_lims[1]*r_e_c / 100., f"{int(r_e_c)} %",
                    color=(0,0,0,0.5), ha='left', va='center', transform=a1.transData, fontsize=fs_micro-6)
            else:
                a1.text(rel_err_label_pos_x, er_lims[1], f"{int(r_e_c)} %", 
                    color=(0,0,0,0.5), ha='left', va='bottom', transform=a1.transData, fontsize=fs_micro-6)

        val_bins_plot = 10**((val_bins_log[:,1] - val_bins_log[:,0])*0.5 + val_bins_log[:,0])
        a1.plot(ax_lims, [0,0], color=(0,0,0))
        a1.plot(val_bins_plot, RMSE_bins, color=(0.11,0.46,0.70), linewidth=1.2, label='RMSE')
        a1.plot(val_bins_plot, BIAS_bins, color=(0.11,0.46,0.70), linewidth=1.2, linestyle='dashed', label='Bias')

        
        # Legends:
        lh, ll = a1.get_legend_handles_labels()
        a1.legend(handles=lh, labels=ll, loc='lower left', bbox_to_anchor=(0.02, 0.00), fontsize=fs_micro-4,
                    framealpha=0.5)

        # set axis limits:
        a1.set_ylim(bottom=er_lims[0], top=er_lims[1])
        a1.set_xlim(left=ax_lims[0], right=ax_lims[1])
        a1.set_xscale('log')

        # set axis ticks, ticklabels and tick parameters:
        a1.minorticks_on()
        a1.tick_params(axis='both', labelsize=fs_micro-4)

        # grid:
        a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

        # labels:
        a1.set_ylabel("Error: Predicted - reference LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs_micro-2)
        a1.set_xlabel("Reference LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs_micro-2)
        a1.set_title(f"{aux_i['file_descr']}", fontsize=fs_micro)

        if aux_i['save_figures']:
            plotname = f"HALO-AC3_NN_ret_eval_{predictand_id}_err_diff_comp_{aux_i['file_descr']}"
            f1.savefig(aux_i['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


    # plt.clf()
    gc.collect()


def save_obs_predictions(
    path_output,
    prediction_ds,
    predictand_id,
    now_date,
    aux_i):

    """
    Save the Neural Network prediction to a netCDF file. Variables to be included:
    time, flag, output variable (prediction), standard error (std. dev. (bias corrected!),
    lat, lon, zsl (altitude above mean sea level),

    Parameters:
    -----------
    path_output : str
        Path where output is saved to.
    prediction_ds : xarray dataset
        Variables predicted by the Neural Network and additional information (flags, input tbs).
    predictand_id : str
        String indicating which output variable is forwarded to the function.
    now_date : str
        String idicating the currently processed date (in yyyy-mm-dd).
    aux_i : dict
        Dictionary containing additional information.
    """

    prediction = prediction_ds['output']

    path_output_l1 = path_output + "l1/"
    path_output_l2 = path_output + "l2/"


    # Save the data on a daily basis:
    # Also set flag bits to 1024 (which means adding 1024) when retrieved quantity is beyond thresholds:
    import sklearn
    import netCDF4 as nc
    l1_var = 'tb'
    l1_var_units = "K"
    l1_version = "v00"
    l2_version = "v00"
    if predictand_id == 'iwv':
        output_var = 'iwv'
        output_units = "kg m-2"

        prediction_thresh = [0, 100]        # kg m-2
        idx_beyond = np.where((prediction < prediction_thresh[0]) | (prediction > prediction_thresh[1]))[0]

    elif predictand_id == 'lwp':
        output_var = 'lwp'
        output_units = "g m-2"

        prediction_thresh = [-200., 3000.]      # g m-2
        idx_beyond = np.where((prediction < prediction_thresh[0]) | (prediction > prediction_thresh[1]))[0]



    now_date = dt.datetime.strptime(now_date, "%Y-%m-%d")
    path_addition = f"{str(aux_i['obs_height'])}m/"

    # check if path exists:
    path_output_dir = os.path.dirname(path_output_l2 + path_addition)
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)


    # dictionary to find research flight number for the current day:
    RF_dict = {
                '20220225': "RF00",
                "20220311": "RF01",
                "20220312": "RF02",
                "20220313": "RF03",
                "20220314": "RF04",
                "20220315": "RF05",
                "20220316": "RF06",
                "20220320": "RF07",
                "20220321": "RF08",
                "20220328": "RF09",
                "20220329": "RF10",
                "20220330": "RF11",
                "20220401": "RF12",
                "20220404": "RF13",
                "20220407": "RF14",
                "20220408": "RF15",
                "20220410": "RF16",
                "20220411": "RF17",
                "20220412": "RF18",
                }
    RF_now = RF_dict[now_date.strftime("%Y%m%d")]



    # Save predictions (level 2) to xarray dataset, then to netcdf:
    nc_output_name = f"HALO-AC3_HALO_HAMP_radiometer_l2_{output_var}_{l2_version}_{RF_now}_{now_date.strftime('%Y%m%d')}"

    # create Dataset:
    DS = xr.Dataset({'lat':         (['time'], prediction_ds['lat'].values.astype(np.float32),
                                    {'units': "degree_north",
                                    'standard_name': "latitude",
                                    'long_name': "latitude of HALO"}),
                    'lon':          (['time'], prediction_ds['lon'].values.astype(np.float32),
                                    {'units': "degree_east",
                                    'standard_name': "longitude",
                                    'long_name': "longitude of HALO"}),
                    'alt':          (['time'], prediction_ds['alt'].values.astype(np.float32),
                                    {'units': "m",
                                    'standard_name': "altitude",
                                    'long_name': "altitude above mean sea level"}),
                    'obs_height':   ([], aux_i['obs_height'],
                                    {'units': "m",
                                    'standard_name': "observation_height",
                                    'long_name': "observation height above mean sea level"}),
                    },
                    coords=         {'time': (['time'], prediction_ds.time.values.astype("datetime64[s]").astype(np.float64),
                                                {'units': "seconds since 1970-01-01 00:00:00 UTC",
                                                'standard_name': "time"})})

    if predictand_id == 'iwv':
        DS[output_var] = xr.DataArray(prediction.values.flatten().astype(np.float32), dims=['time'],
                                        attrs={
                                        'units': output_units,
                                        'standard_name': "atmosphere_mass_content_of_water_vapor",
                                        'comment': ("These values denote the vertically integrated amount of water vapor from the surface to HALO's altitude.")})

    if predictand_id == 'lwp':
        DS[output_var] = xr.DataArray(prediction.values.flatten().astype(np.float32), dims=['time'],
                                        attrs={
                                        'units': output_units,
                                        'standard_name': "atmosphere_mass_content_of_cloud_liquid_water_content",
                                        'comment': ("These values denote the vertically integrated amount of condensed water from the surface to HALO's altitude.")})



    # adapt fill values:
    # Make sure that _FillValue is not added to certain variables:
    exclude_vars_fill_value = ['time', 'lat', 'lon', 'zsl']
    for kk in exclude_vars_fill_value:
        if kk in DS.variables:
            DS[kk].encoding["_FillValue"] = None

    # add fill values to remaining variables:
    vars_fill_value = ['iwv', 'lwp']
    for kk in vars_fill_value:
        if kk in DS.variables:
            DS[kk].encoding["_FillValue"] = float(-999.)

    DS.attrs['Title'] = f"Microwave radiometer retrieved {output_var}"
    DS.attrs['Institution'] = "Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
    DS.attrs['Contact_person'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
    DS.attrs['Source'] = "HALO HAMP microwave radiometers"
    datetime_utc = dt.datetime.now(dt.timezone.utc)
    DS.attrs['History'] = f"{datetime_utc.strftime('%Y-%m-%d %H:%M:%S')}, created with {sys.argv[0]}"
    DS.attrs['Author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de)"
    DS.attrs['Comments'] = ""
    DS.attrs['License'] = "For non-commercial use only."
    DS.attrs['Measurement_site'] = "HALO-AC3"

    DS.attrs['retrieval_type'] = "Neural Network"
    DS.attrs['python_packages'] = (f"python version: {sys.version}, tensorflow: {tensorflow.__version__}, keras: {keras.__version__}, " +
                                    f"numpy: {np.__version__}, sklearn: {sklearn.__version__}, netCDF4: {nc.__version__}, " +
                                    f"matplotlib: {mpl.__version__}, xarray: {xr.__version__}, pandas: {pd.__version__}")

    DS.attrs['retrieval_net_architecture'] = f"n_hidden_layers: {str(aux_i['n_layers'])}; nodes_for_hidden_layers: {str(aux_i['n_nodes'])}"
    DS.attrs['retrieval_batch_size'] = f"{str(aux_i['batch_size'])}"
    DS.attrs['retrieval_epochs'] = f"{str(aux_i['epochs'])}"
    DS.attrs['retrieval_learning_rate'] = f"{str(aux_i['learning_rate'])}"
    DS.attrs['retrieval_activation_function'] = f"{aux_i['activation']} (from input to hidden layer and subsequent hidden layers)"
    DS.attrs['retrieval_feature_range'] = f"feature range of sklearn.preprocessing.MinMaxScaler: {str(aux_i['feature_range'])}"
    DS.attrs['retrieval_rng_seed'] = str(aux_i['seed'])
    DS.attrs['retrieval_kernel_initializer'] = f"{aux_i['kernel_init']}"
    DS.attrs['retrieval_optimizer'] = "keras.optimizers.Adam"
    DS.attrs['retrieval_callbacks'] = (f"EarlyStopping(monitor=val_loss, patience={str(aux_i['callback_patience'])}, " +
                                        f"min_delta={str(aux_i['min_delta'])}, restore_best_weights=True)")

    DS.attrs['training_data'] = "ERA5"
    tdy_str = ""
    for year_str in aux_i['yrs_training']: tdy_str += f"{str(year_str)}, "
    DS.attrs['training_data_years'] = tdy_str[:-2]

    DS.attrs['test_data'] = "ERA5"
    tdy_str = ""
    for year_str in aux_i['yrs_testing']: tdy_str += f"{str(year_str)}, "
    DS.attrs['test_data_years'] = tdy_str[:-2]
        

    DS.attrs['n_training_samples'] = aux_i['n_training']
    DS.attrs['n_test_samples'] = aux_i['n_test']
    DS.attrs['training_test_TB_noise_std_dev'] = (f"22-60 GHz: {cat[test_id]['noise_kv']:.2g}, " +
                                                    f"90 GHz: {cat[test_id]['noise_w']:.2g}, " +
                                                    f"110-130 GHz: {cat[test_id]['noise_f']:.2g}, " +
                                                    f"170-195 GHz: {cat[test_id]['noise_g']:.2g}, ")

    # input vector information: First, TBs, then remaining predictors
    DS.attrs['input_vector'] = "("
    for ff in prediction_ds.freq:
        DS.attrs['input_vector'] += f"TB_{ff.values:.2f}GHz, "
    DS.attrs['input_vector'] = DS.attrs['input_vector'][:-2]
    DS.attrs['input_vector'] = DS.input_vector + ")"
    DS.attrs['output_vector'] = f"{output_var}"


    # encode time:
    DS['time'] = prediction_ds.time.values.astype("datetime64[s]").astype(np.float64)
    DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
    DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    DS['time'].encoding['dtype'] = 'double'

    DS.to_netcdf(path_output_l2 + path_addition + nc_output_name + ".nc", mode='w', format="NETCDF4")
    DS = DS.close()


def save_halo_test_obs(
    prediction, 
    predictand_id,
    aux_i,
    height=np.array([])):

    """
    Saves the prediction of the Neural Network based on a small test set of real MOSAiC 
    observations.

    Parameters:
    -----------
    prediction : xarray DataArray of floats
        Variables predicted by the Neural Network.
    predictand_id : str
        String indicating which output variable is forwarded to the function.
    aux_i : dict
        Dictionary containing additional information.
    height : array of floats
        Height array for respective predictand or predictand profiles (of i.e., temperature or 
        humidity). Can be a 1D or 2D array (latter must be of shape (n_training,n_height)).
    """

    # check if output path exists: if it doesn't, create it:
    path_output_dir = os.path.dirname(aux_i['path_output_pred_ref'])
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)


    # create xarray Dataset:
    DS = xr.Dataset(coords={'time': (prediction.time)})

    # save data into it:
    if predictand_id in ['iwv', 'lwp']:
        DS[predictand_id] = prediction
    DS[predictand_id].attrs = {'long_name': f"Predicted {predictand_id}", 'units': "SI units"}


    # GLOBAL ATTRIBUTES:
    DS.attrs['title'] = f"HALO-(AC)3 HALO HAMP test observations; predicted {predictand_id}"
    DS.attrs['author'] = "Andreas Walbroel (a.walbroel@uni-koeln.de), Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
    DS.attrs['predictor_TBs'] = aux_i['predictor_TBs']
    DS.attrs['predictors'] = aux_i['predictors']
    DS.attrs['setup_id'] = aux_i['file_descr']
    datetime_utc = dt.datetime.utcnow()
    DS.attrs['processing_date'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")
    DS.attrs['python_version'] = f"python version: {sys.version}"
    DS.attrs['python_packages'] = (f"numpy: {np.__version__}, matplotlib: {mpl.__version__}, " +
                                    f"xarray: {xr.__version__}, yaml: {yaml.__version__}, " +
                                    f"tensorflow: {tensorflow.__version__}, pandas: {pd.__version__}")

    # time encoding:
    DS['time'] = DS.time.values.astype("datetime64[s]").astype(np.float64)
    DS['time'].attrs['units'] = "seconds since 1970-01-01 00:00:00"
    DS['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'
    DS['time'].encoding['dtype'] = 'double'


    # export to netCDF:
    save_filename = aux_i['path_output_pred_ref'] + f"HALO-AC3_HALO_HAMP_test_obs_NN_ret_prediction_{predictand_id}_{aux_i['file_descr']}.nc"
    DS.to_netcdf(save_filename, mode='w', format='NETCDF4')
    DS = DS.close()
    print(f"Saved {save_filename}")


def save_retrieval_stats_eval(
    retrieval_stats_add: dict,
    retrieval_stats_syn: dict,
    aux_i: dict,
    test_id: str):
    
    # Save retrieval stats to xarray dataset, then to netcdf:
    nc_output_name = f"HALO-AC3_NN_retrieval_eval_test_id_{aux_i['file_descr']}"

    # start forming the data set, inserting retrieval setup information:
    RETRIEVAL_STAT_DS = xr.Dataset({'epochs':       (['test_id'], np.asarray(retrieval_stats_syn['epochs']),
                                                     {'description': "Neural Network training epoch number"}),
                                    'elapsed_epochs': (['test_id'], np.asarray(retrieval_stats_syn['elapsed_epochs']),
                                                       {'description': "Number of epochs elapsed during training"}),
                                    'seed':         (['test_id'], np.asarray(retrieval_stats_syn['seed']),
                                                     {'description': "RNG seed for numpy.random.seed and tensorflow.random.set_seed"}),
                                    'learning_rate':(['test_id'], np.asarray(retrieval_stats_syn['learning_rate']),
                                                     {'description': "Learning rate of NN optimizer"})
                                    },
                                   coords=          {'test_id': (['test_id'], 
                                                                 np.arange(len(retrieval_stats_syn['test_loss'])),
                                                                 {'description': "Test number"}),
                                                    })

    # add the retrieval metrics of test data vs. prediction:
    ret_met_units = {'iwv': "mm", 'lwp': "kg m-2"}
    ret_met_range = {   'iwv': {'bot': "[0,5) mm", 'mid': "[5,10) mm", 'top': "[10,100) mm"},
                        'lwp': {'bot': "[0,0.025) kg m-2", 'mid': "[0.025,0.100) kg m-2", 'top': "[0.100, 1e+06) kg m-2"},
                    }

    for predictand in aux_i['predictand']:

        # description attributes:
        ret_met_descr = {'rmse_tot': f"Eval data Root Mean Square Error (RMSE) of target and predicted {predictand}",
                        'rmse_bot': f"Like rmse_tot but confined to {predictand} range {ret_met_range[predictand]['bot']}",
                        'rmse_mid': f"Like rmse_tot but confined to {predictand} range {ret_met_range[predictand]['mid']}",
                        'rmse_top': f"Like rmse_tot but confined to {predictand} range {ret_met_range[predictand]['top']}",
                        'bias_tot': f"Bias of test data predicted - target {predictand}",
                        'bias_bot': f"Like bias_tot but confined to {predictand} range {ret_met_range[predictand]['bot']}",
                        'bias_mid': f"Like bias_tot but confined to {predictand} range {ret_met_range[predictand]['mid']}",
                        'bias_top': f"Like bias_tot but confined to {predictand} range {ret_met_range[predictand]['top']}",
                        'stddev': f"Eval data standard deviation (bias corrected RMSE) of target and predicted {predictand}",
                        'stddev_bot': f"Like stddev but confined to {predictand} range {ret_met_range[predictand]['bot']}",
                        'stddev_mid': f"Like stddev but confined to {predictand} range {ret_met_range[predictand]['mid']}",
                        'stddev_top': f"Like stddev but confined to {predictand} range {ret_met_range[predictand]['top']}"}

        # save retrieval metrics to dataset and forward the variable attributes:
        for ret_met in ret_metrics:

            RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"] = xr.DataArray(np.asarray(retrieval_stats_add[f"{predictand}_metrics"][ret_met]),
                                                                        dims=['test_id'])
            RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['description'] = ret_met_descr[ret_met]
            RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['units'] = ret_met_units[predictand]
            if "bot" in ret_met:
                RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['range'] = ret_met_range[predictand]['bot']
            elif "mid" in ret_met:
                RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['range'] = ret_met_range[predictand]['mid']
            elif "top" in ret_met:
                RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['range'] = ret_met_range[predictand]['top']


    # Provide some global attributes
    RETRIEVAL_STAT_DS.attrs['test_purpose'] = test_id
    RETRIEVAL_STAT_DS.attrs['author'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"
    RETRIEVAL_STAT_DS.attrs['predictands'] = ""
    for predictand in aux_i['predictand']: RETRIEVAL_STAT_DS.attrs['predictands'] += predictand + ", "
    RETRIEVAL_STAT_DS.attrs['predictands'] = RETRIEVAL_STAT_DS.attrs['predictands'][:-2]


    if aux_i['site'] == 'era5':
        RETRIEVAL_STAT_DS.attrs['training_data'] = "ERA5, PAMTRA simulations"
        RETRIEVAL_STAT_DS.attrs['test_data'] = "ERA5, PAMTRA simulations"

    datetime_utc = dt.datetime.now(dt.timezone.utc)
    RETRIEVAL_STAT_DS.attrs['datetime_of_creation'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")


    # create output path if not existing:
    outpath_dir = os.path.dirname(aux_i['path_output'] + "ret_stat/")
    if not os.path.exists(outpath_dir):
        os.makedirs(outpath_dir)
    RETRIEVAL_STAT_DS.to_netcdf(aux_i['path_output'] + "ret_stat/" + nc_output_name + ".nc", mode='w', format="NETCDF4")
    RETRIEVAL_STAT_DS = RETRIEVAL_STAT_DS.close()


def NN_retrieval(predictor_training, predictand_training, predictor_test,
                    predictand_test, aux_i, return_test_loss=True):

    print("(batch_size, epochs, seed)=", aux_i['batch_size'], aux_i['epochs'], aux_i['seed'])
    print("learning_rate=", aux_i['learning_rate'])

    # Initialize and define the NN model
    input_shape = predictor_training.input.shape
    output_shape = predictand_training.output.shape
    model = Sequential()

    model.add(Dense(aux_i['n_nodes'][0], input_dim=input_shape[1], activation=aux_i['activation'], kernel_initializer=aux_i['kernel_init']))
    if aux_i['batch_normalization']: model.add(BatchNormalization())
    if aux_i['dropout'] > 0.0: model.add(Dropout(aux_i['dropout']))

    # space for more layers:
    if (aux_i['n_layers'] > 1) and (aux_i['dropout'] > 0.0):
        for n_l in range(1,aux_i['n_layers']):
            model.add(Dense(aux_i['n_nodes'][n_l], activation=aux_i['activation'], kernel_initializer=aux_i['kernel_init']))
            if aux_i['batch_normalization']: model.add(BatchNormalization())
            model.add(Dropout(aux_i['dropout']))
    elif aux_i['n_layers'] > 1:
        for n_l in range(1,aux_i['n_layers']):
            model.add(Dense(aux_i['n_nodes'][n_l], activation=aux_i['activation'], kernel_initializer=aux_i['kernel_init']))
            if aux_i['batch_normalization']: model.add(BatchNormalization())

    model.add(Dense(output_shape[1], activation='linear'))      # output layer shape must be equal to retrieved variables

    # compile and train the NN model
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=aux_i['learning_rate']))
    history = model.fit(predictor_training.input_scaled, predictand_training.output, batch_size=aux_i['batch_size'],
                epochs=aux_i['epochs'], verbose=1,
                validation_data=(predictor_test.input_scaled, predictand_test.output),
                callbacks=[EarlyStopping(monitor='val_loss', patience=aux_i['callback_patience'], min_delta=aux_i['min_delta'],
                restore_best_weights=True)],
                )

    test_loss = np.asarray(history.history['val_loss']).min()           # test data MSE
    print("n_epochs executed: ", len(history.history['loss']))
    print("Test loss: ", test_loss)

    if return_test_loss:
        return model, test_loss
    else:
        return model


###################################################################################################
###################################################################################################


"""
    In this script, Tensorflow.Keras will be used to retrieve LWP from airborne microwave radiometer 
    (MWR) TB measurements. The following steps are executed:
    - Importing training and test data; split into training and test data sets
    - define, rescale and build input vector (predictors)
    - define predictands
    - define and build Neural Network model
    - compile model: choose loss function and optimizer
    - fit model (training): try various subsets of the entire data as training; try
        different batch sizes and learning rates; validate with test data
    - evaluate model (with test data)
    - predict unknown output from new data
"""


# determine test_id
test_id = "000" # specify the intention of a test (used for the retrieval statistics output .nc file)
if len(sys.argv) == 2:
    test_id = sys.argv[1]
elif len(sys.argv) > 2:
    raise ValueError("Sorry, I didn't get that. Just type 'python3 NN_retrieval.py' or " +
                    "'python3 NN_retrieval.py " + '"003"' + "' (as example for test run 003)....")

# exec_type determines if 20 random numbers shall be cycled through ("20_runs") or whether only one random
# number is to be used ("op_ret")
exec_type = '20_runs'




# open test_purpose.YAML file to manage settings:
with open(wdir + "test_purpose.yaml", 'r') as f:
    cat = yaml.safe_load(f)


aux_i = dict()  # dictionary that collects additional information
aux_i['file_descr'] = test_id.replace(" ", "_").lower() # file name addition (of some plots and netCDF output)
aux_i['site'] = 'era5'          # options of training and test data: 'era5': ERA5 training and test data
aux_i['predictors'] = cat[test_id]['predictors']    # specify input vector (predictors): options: TBs
                                                    # TBs: up to all HAMP channels
aux_i['predictor_TBs'] = cat[test_id]['predictor_TBs']  # string to identify which bands of TBs are used as predictors
                                                        # syntax as in data_tools.select_MWR_channels

# NN settings:
aux_i['n_layers'] = cat[test_id]['n_layers']        # number of hidden layers (integer)
aux_i['n_nodes'] = cat[test_id]['n_nodes']          # number of nodes for each hidden layer as list: 
                                                    # [n_node_layer0, n_node_layer1, n_node_layer2, ...]
aux_i['dropout'] = cat[test_id]['dropout']          # dropout chance in [0.0, 1.0]; if 0.0: no dropout layers
aux_i['batch_normalization'] = cat[test_id]['batch_normalization']  # bool if BatchNormalization layer is used in hidden layers
aux_i['activation'] = cat[test_id]['activ_f']       # default or best estimate for i.e., iwv: exponential
aux_i['feature_range'] = tuple(cat[test_id]['feature_range'])   # best est. with exponential (-3.0, 1.0)
aux_i['epochs'] = cat[test_id]['epochs']
aux_i['batch_size'] = cat[test_id]['batch_size']
aux_i['learning_rate'] = cat[test_id]['learning_rate']      # default: 0.001
aux_i['kernel_init'] = cat[test_id]['kernel_init']          # default: 'glorot_uniform'
aux_i['callback_patience'] = cat[test_id]['callback_patience']  # patience of the callback
aux_i['min_delta'] = cat[test_id]['min_delta']              # min val_loss improvement needed for earlystopping

aux_i['predictor_instrument'] = {'era5': "era5_pam"}        # argument to load predictor data
aux_i['predictand'] = cat[test_id]['predictand']            # output variable / predictand: options: 
                                                            # list with elements in ["lwp"]


aux_i['yrs'] = {'era5': ["2001", "2002", "2003", "2004", "2006", "2007", "2008", "2009", "2011", 
                "2012", "2013", "2014", "2016", "2017", "2018", "2019"]}        # available years of data
aux_i['yrs'] = aux_i['yrs'][aux_i['site']]
n_yrs = len(aux_i['yrs'])
n_training = round(0.75*n_yrs)          # number of training years
n_test = n_yrs - n_training


aux_i['add_TB_noise'] = True                # if True, random noise will be added to training and test data. 
                                            # Remember to define a noise dictionary if True
aux_i['vis_eval'] = True                    # if True: visualize retrieval evaluation (test predictand vs prediction) (only true if aux_i['op_ret'] == False)
aux_i['save_figures'] = True                # if True: figures created will be saved to file
aux_i['lwp_offset_cor'] = True              # if True: LWP offset correction will be applied on the halo_test_subset
aux_i['tb_offset_cor'] = False              # if True: HAMP TB obs will be corrected for offsets
aux_i['op_ret'] = False                     # if True: some NN output of one spec. random number will be generated
aux_i['save_obs_predictions'] = False   # if True, predictions made from MWR observations will be saved
                                        # to a netCDF file (i.e., for op_ret retrieval)
aux_i['halo_test_subset'] = False       # used to decide if also HAMP obs will be tested and saved for one RNG seed
aux_i['test_on_all_rngs'] = True        # if True, visualize_evaluation and halo_test_subset (if active) are exec. on all RNG seeds; should be False mostly
aux_i['add_val'] = True                 # if True, additional data for evaluation is loaded separately
aux_i['1D_aligned'] = False                 # indicator if training/test data is aligned on a 1D or 2D spatial grid
if aux_i['site'] == "era5":
    aux_i['1D_aligned'] = True

aux_i['all_obs_heights'] = False            # if True (only for exec_type='op_ret'), the NN will be trained for all obs_heights


if exec_type == 'op_ret':
    aux_i['op_ret'] = True
    aux_i['vis_eval'] = False
    aux_i['save_obs_predictions'] = True
    aux_i['test_on_all_rngs'] = False
    aux_i['add_val'] = False
    aux_i['all_obs_heights'] = True

if aux_i['all_obs_heights']:
    aux_i['obs_heights'] = np.arange(8000., 13500.01, 250.).astype(np.int32)
else:
    aux_i['obs_heights'] = np.array([11000])


# activate offset correction for certain test_id:
if test_id in ["060", "061", "063", "065"]:
    aux_i['tb_offset_cor'] = True



# paths:
if remote:
    aux_i['path_output'] = "/net/blanc/awalbroe/Data/HALO_AC3/lwp_retrieval/output/"                # path where output is saved to
    aux_i['path_output_pred_ref'] = "/net/blanc/awalbroe/Data/HALO_AC3/lwp_retrieval/prediction_and_reference/" # path where output is saved to
    aux_i['path_data'] = {'era5': "/net/blanc/awalbroe/Data/METRS_SS23/merged/new_z_grid/"}         # path of training/test data
    aux_i['path_data'] = aux_i['path_data'][aux_i['site']]
    aux_i['path_data_add'] = "/net/blanc/awalbroe/Data/METRS_SS23/merged_add/new_z_grid/"
    aux_i['path_tb_obs'] = "/net/blanc/awalbroe/Data/HALO_AC3/HALO/HAMP/unified/"       # path of HAMP tb data
    aux_i['path_radar_obs'] = "/data/obs/campaigns/ac3airborne/ac3cloud_server/halo-ac3/halo/hamp_mira/"    # RADAR data path
    aux_i['path_bahamas_obs'] = "/data/obs/campaigns/halo-ac3/halo/BAHAMAS/unified/"    # path of BAHAMAS data
    aux_i['path_tb_offsets'] = "/net/blanc/awalbroe/Data/HALO_AC3/HALO/CSSC/"           # path of the HAMP TB offset correction
    aux_i['path_plots'] = "/net/blanc/awalbroe/Plots/HALO_AC3/lwp_retrieval/"

else:
    aux_i['path_output'] = "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/output/"           # path where output is saved to
    aux_i['path_output_pred_ref'] = "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/prediction_and_reference/"            # path where output is saved to
    aux_i['path_data'] = {'era5': "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/training_data/merged/new_z_grid/"}      # path of training/test data
    aux_i['path_data'] = aux_i['path_data'][aux_i['site']]
    aux_i['path_data_add'] = "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/training_data/merged_add/new_z_grid/"
    aux_i['path_tb_obs'] = "/mnt/f/heavy_data/HALO_AC3/HALO/HAMP/"                      # path of HAMP tb data
    aux_i['path_radar_obs'] = "/mnt/f/heavy_data/HALO_AC3/HALO/HAMP_mira/"              # RADAR data path
    aux_i['path_bahamas_obs'] = "/mnt/f/heavy_data/HALO_AC3/HALO/BAHAMAS/"              # path of BAHAMAS data
    aux_i['path_tb_offsets'] = "/mnt/f/heavy_data/HALO_AC3/CSSC/"                       # path of the HAMP TB offset correction
    aux_i['path_plots'] = "/mnt/f/Studium_NIM/work/Plots/HALO_AC3/lwp_retrieval/"


# time range of tb data to be imported (LATER PART OF THE RETRIEVAL DEV)
aux_i['date_start'] = "2022-03-11"
aux_i['date_end'] = "2022-04-12"



# create output path if not existing:
outpath_dir = os.path.dirname(aux_i['path_output'])
if not os.path.exists(outpath_dir):
    os.makedirs(outpath_dir)


# if desired, import some MOSAiC radiosondes and radiometer data (HATPRO and MiRAC-P) for
# a small test data set:
sonde_dict = dict()
MWR_DS = xr.Dataset()
if aux_i['halo_test_subset'] and not aux_i['op_ret']:
    aux_i['test_date'] = "2022-03-21"           # RF08

    # find files and import radiometer data:
    print("Importing HAMP radiometer data....")
    file_mwr = sorted(glob.glob(aux_i['path_tb_obs'] + f"radiometer_{aux_i['test_date'].replace('-','')}*.nc"))[0]
    MWR_DS = xr.open_dataset(file_mwr)
    MWR_DS = MWR_DS.sortby('freq')


    # rename some variables to unify with synergetic ret NN_retrieval.py:
    MWR_DS = MWR_DS.drop('freq').rename({'TB': 'tb', 'uniRadiometer_freq': 'freq'})

    # filter icey or land surface:
    MWR_DS = MWR_DS.isel(time=np.where(MWR_DS.surface_mask==0)[0])

    # filter roll angles:
    # import BAHAMAS data, put roll angle on MWR time axis and fitler for straight line only:
    file_bah = sorted(glob.glob(aux_i['path_bahamas_obs'] + f"bahamas_{aux_i['test_date'].replace('-','')}_*.nc"))[0]
    BAH_DS = xr.open_dataset(file_bah)
    roll_mwr = np.interp(MWR_DS.time.values.astype('datetime64[s]').astype(np.float64), BAH_DS.time.values.astype('datetime64[s]').astype(np.float64), 
                        BAH_DS['roll'].values, left=np.nan, right=np.nan)
    idx_no_curve = np.where(np.abs(roll_mwr) < 1.0)[0]
    MWR_DS = MWR_DS.isel(time=idx_no_curve)


    # eventually correct TB offsets:
    if aux_i['tb_offset_cor']:
        MWR_DS = hamp_tb_offset_correction(MWR_DS, aux_i['path_tb_offsets'])


    del BAH_DS, roll_mwr
    


# 20 random numbers generated with np.random.uniform(0, 1000, 20).astype(np.int32)
if exec_type == '20_runs':
    some_seeds = [773, 994, 815, 853, 939, 695, 472, 206, 159, 307, 
                  612, 442, 405, 487, 549, 806, 45, 110, 35, 701]
elif exec_type == 'op_ret':
    if (aux_i['predictand'] == ["lwp"]) or (test_id in ["031", "065"]): 
        some_seeds = [773]
    elif (aux_i['predictand'] == ['iwv']) and (test_id == "048"):
        some_seeds = [773]
    elif (aux_i['predictand'] == ['iwv']) and (test_id == "051"):
        some_seeds = [442]
    elif (aux_i['predictand'] == ['iwv']) and (test_id == "052"):
        some_seeds = [307]
    elif (aux_i['predictand'] == ['iwv']) and (test_id in ["056", "060", "064"]):
        some_seeds = [110]


# dict which will save information about each test
ret_metrics = ['rmse_tot', 'rmse_bot', 'rmse_mid', 'rmse_top', 'stddev', 'stddev_bot', 
                'stddev_mid', 'stddev_top', 'bias_tot', 'bias_bot', 'bias_mid', 'bias_top']
aux_i_stats = ['test_loss', 'training_loss', 'val_loss_array', 'loss_array', 'batch_size', 'epochs', 
                'elapsed_epochs', 'activation', 'seed', 'learning_rate', 'feature_range']

retrieval_stats_syn = dict()
for ais in aux_i_stats:
    if ais not in ['val_loss_array', 'loss_array']:
        retrieval_stats_syn[ais] = list()
    else:
        retrieval_stats_syn[ais] = np.full((len(some_seeds), aux_i['epochs']), np.nan)
for predictand in aux_i['predictand']:
    retrieval_stats_syn[predictand + "_metrics"] = dict()

    for ret_met in ret_metrics:
        retrieval_stats_syn[predictand + "_metrics"][ret_met] = list()
if aux_i['add_val']:
    retrieval_stats_add = deepcopy(retrieval_stats_syn)
    for var in aux_i_stats: del retrieval_stats_add[var]


for obs_height in aux_i['obs_heights']:
    aux_i['obs_height'] = obs_height

    for k_s, aux_i['seed'] in enumerate(some_seeds):

        # set rng seeds
        np.random.seed(seed=aux_i['seed'])
        tensorflow.random.set_seed(aux_i['seed'])
        tensorflow.keras.utils.set_random_seed(aux_i['seed'])

        # randomly select training and test years
        yrs_idx_rng = np.random.permutation(np.arange(n_yrs))
        yrs_idx_training = sorted(yrs_idx_rng[:n_training])
        yrs_idx_test = sorted(yrs_idx_rng[n_training:])

        aux_i['yrs_training'] = np.asarray(aux_i['yrs'])[yrs_idx_training]
        aux_i['yrs_testing'] = np.asarray(aux_i['yrs'])[yrs_idx_test]

        print("Years Training: %s"%(aux_i['yrs_training']))
        print("Years Testing: %s"%(aux_i['yrs_testing']))


        # split training and test data:
        data_files_training = sorted(glob.glob(aux_i['path_data'] + "*.nc"))
        data_files_test = data_files_training
        if aux_i['add_val']: data_files_add = sorted(glob.glob(aux_i['path_data_add'] + "*.nc"))


        # Define noise strength dictionary for the function add_TB_noise in class radiometers:
        noise_dict = {  '22.24':    cat[test_id]['noise_kv'],
                        '23.04':    cat[test_id]['noise_kv'],
                        '23.84':    cat[test_id]['noise_kv'],
                        '25.44':    cat[test_id]['noise_kv'],
                        '26.24':    cat[test_id]['noise_kv'],
                        '27.84':    cat[test_id]['noise_kv'],
                        '31.40':    cat[test_id]['noise_kv'],
                        '50.30':    cat[test_id]['noise_kv'],
                        '51.76':    cat[test_id]['noise_kv'],
                        '52.80':    cat[test_id]['noise_kv'],
                        '53.75':    cat[test_id]['noise_kv'],
                        '54.94':    cat[test_id]['noise_kv'],
                        '56.66':    cat[test_id]['noise_kv'],
                        '58.00':    cat[test_id]['noise_kv'],
                        '90.00':    cat[test_id]['noise_w'],
                        '120.15':   cat[test_id]['noise_f'],
                        '121.05':   cat[test_id]['noise_f'],
                        '122.95':   cat[test_id]['noise_f'],
                        '127.25':   cat[test_id]['noise_f'],
                        '183.91':   cat[test_id]['noise_g'],
                        '184.81':   cat[test_id]['noise_g'],
                        '185.81':   cat[test_id]['noise_g'],
                        '186.81':   cat[test_id]['noise_g'],
                        '188.31':   cat[test_id]['noise_g'],
                        '190.81':   cat[test_id]['noise_g']}


        # Load data and split into test and training, and predictor and predictand. Also make sure, frequencies
        # are in ascending order:
        data_file_training = [dft for dft in data_files_training if str(obs_height) in dft]
        data_DS = xr.open_dataset(data_file_training[0])
        data_DS = data_DS.sortby(['freq'], ascending=True)


        # Select indices for the respective years in the training or test subset:
        if type(aux_i['yrs_training'][0]) not in [int, np.int8, np.int16, np.int32, np.int64]:
            subset_training = np.asarray(aux_i['yrs_training']).astype(np.int32)
            subset_testing = np.asarray(aux_i['yrs_testing']).astype(np.int32)
        else:
            subset_training = aux_i['yrs_training']
            subset_testing = aux_i['yrs_testing']
        data_DS_training = data_DS.isel(x=(data_DS.time.dt.year.isin(subset_training)))
        data_DS_test = data_DS.isel(x=(data_DS.time.dt.year.isin(subset_testing)))

        # # To filter for cloudy cases only (with at least 1 g m-2 LWP):
        # idx_cloudy = np.where(data_DS_training.lwp > 0.001)[0]
        # data_DS_training = data_DS_training.isel(x=idx_cloudy)


        # define predictors:
        predictor_training = predictor_class(data_DS_training, return_DS=True, 
                                        add_TB_noise=aux_i['add_TB_noise'], noise_dict=noise_dict)
        predictor_test = predictor_class(data_DS_test, return_DS=True,
                                        add_TB_noise=aux_i['add_TB_noise'], noise_dict=noise_dict)

        # define predictands:
        processed_b = "new_z_grid" in aux_i['path_data']    # True if training data had been processed with training_data_new_height.py
        predictand_training = predictand_class(data_DS_training, processed=processed_b, return_DS=True)
        predictand_test = predictand_class(data_DS_test, processed=processed_b, return_DS=True)

        # clear memory
        del data_DS, data_DS_training, data_DS_test


        # Eventually need to convert the predictand and predictor data to a (n_training x n_input) (and respective
        # output): Before changing: sample is FIRST dimension; height is LAST dimension
        check_dims_vars = {'temp_sfc': 1, 'height': 2, 'temp': 2, 'rh': 2, 'pres': 2, # int says how many dims it should have after reduction
                            'sfc_slf': 1, 'iwv': 1, 'cwp': 1, 'rwp': 1, 'lwp': 1, 'swp': 1, 'iwp': 1, 'q': 2,
                            'lat': 1, 'lon': 1, 'launch_time': 1, 'time': 1, 'freq': 1, 'flag': 1, 'TB': 2}

        predictand_training = reduce_dimensions(predictand_training, check_dims_vars)
        predictand_test = reduce_dimensions(predictand_test, check_dims_vars)

        predictor_training = reduce_dimensions(predictor_training, check_dims_vars)
        predictor_test = reduce_dimensions(predictor_test, check_dims_vars)


        # confine to sea grid cells only and create a new height grid (if era5):
        if aux_i['site'] == 'era5':
            predictand_training.sfc_mask = predictand_training.sfc_slf < 0.01
            predictand_test.sfc_mask = predictand_test.sfc_slf < 0.01

            predictand_training = apply_sea_mask(predictand_training, predictand_training.sfc_mask, check_dims_vars)
            predictand_test = apply_sea_mask(predictand_test, predictand_test.sfc_mask, check_dims_vars)
            predictor_training = apply_sea_mask(predictor_training, predictand_training.sfc_mask, check_dims_vars)
            predictor_test = apply_sea_mask(predictor_test, predictand_test.sfc_mask, check_dims_vars)



        aux_i['n_training'] = len(predictand_training.launch_time)
        aux_i['n_test'] = len(predictand_test.launch_time)
        print(aux_i['n_training'], aux_i['n_test'])



        # Start building input vector for training and test data: Eventually reduce TBs
        # to certain frequencies:
        predictor_training.TB, predictor_training.freq = select_MWR_channels(predictor_training.TB,
                                                                            predictor_training.freq,
                                                                            band=aux_i['predictor_TBs'],
                                                                            return_idx=0)
        predictor_test.TB, predictor_test.freq = select_MWR_channels(predictor_test.TB,
                                                                    predictor_test.freq,
                                                                    band=aux_i['predictor_TBs'],
                                                                    return_idx=0)

        predictor_training.input = predictor_training.TB
        predictor_test.input = predictor_test.TB


        """
            Define and build Neural Network model: Input_shape depends on whether or not
            DOY and surface pressure are included.
            Loss function: MSE, optimiser: adam (these options (among others) might also be changed during testing and
            build phase)
            Fit model, avoid overfitting by applying Early Stop: callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
        """

        print(aux_i['activation'], aux_i['feature_range'])

        # Rescale input: Use MinMaxScaler:
        scaler = MinMaxScaler(feature_range=aux_i['feature_range']).fit(predictor_training.input)
        predictor_training.input_scaled = scaler.transform(predictor_training.input)
        predictor_test.input_scaled = scaler.transform(predictor_test.input)


        # if additional data is used for evaluation, repeat the steps from above:
        if aux_i['add_val']:

            data_file_add = [dft for dft in data_files_add if str(obs_height) in dft]
            data_DS_add = xr.open_dataset(data_file_add[0])
            data_DS_add = data_DS_add.sortby(['freq'], ascending=True)

            # define predictors:
            predictor_add = predictor_class(data_DS_add, return_DS=True,
                                            add_TB_noise=aux_i['add_TB_noise'], noise_dict=noise_dict)

            # define predictands:
            predictand_add = predictand_class(data_DS_add, processed=True, return_DS=True)

            # clear memory
            del data_DS_add

            # Reduce dimensions and apply masks:
            predictand_add = reduce_dimensions(predictand_add, check_dims_vars)
            predictor_add = reduce_dimensions(predictor_add, check_dims_vars)
            predictand_add.sfc_mask = predictand_add.sfc_slf < 0.01
            predictand_add = apply_sea_mask(predictand_add, predictand_add.sfc_mask, check_dims_vars)
            predictor_add = apply_sea_mask(predictor_add, predictand_add.sfc_mask, check_dims_vars)


            # Build input vector and scale input:
            predictor_add.TB, predictor_add.freq = select_MWR_channels(predictor_add.TB,
                                                                        predictor_add.freq,
                                                                        band=aux_i['predictor_TBs'],
                                                                        return_idx=0)

            predictor_add.input = predictor_add.TB
            predictor_add.input_scaled = scaler.transform(predictor_add.input)


        # specify output:
        predictand_training = specify_output(predictand_training, aux_i['predictand'], aux_i['n_training'])
        predictand_test = specify_output(predictand_test, aux_i['predictand'], aux_i['n_test'])



        # Create the NN model and predict stuff from it:
        model, test_loss = NN_retrieval(predictor_training, predictand_training, predictor_test, 
                                        predictand_test, aux_i, return_test_loss=True)
        n_epochs_elapsed = len(model.history.epoch)     # number of elapsed epochs
        loss_array = np.asarray(model.history.history['loss'])
        val_loss_array = np.asarray(model.history.history['val_loss'])
        training_loss = loss_array[np.argmin(val_loss_array)]

        # make prediction:
        prediction_syn = model.predict(predictor_test.input_scaled)
        if aux_i['add_val']: 
            predictand_add = specify_output(predictand_add, aux_i['predictand'], len(predictand_add.launch_time))
            predictand_add.prediction_syn = model.predict(predictor_add.input_scaled)


        if exec_type == '20_runs':

            if (aux_i['halo_test_subset'] and (aux_i['seed'] == 773)) | (aux_i['halo_test_subset'] and aux_i['test_on_all_rngs']):

                # repeat what has been done to the predictor training data:
                MWR_TB, MWR_freq = select_MWR_channels(MWR_DS.tb.values, MWR_DS.freq.values,
                                                        band=aux_i['predictor_TBs'],
                                                        return_idx=0)

                # build input vector:
                MWR_input = MWR_TB

                # Rescale input: Use MinMaxScaler:
                halo_input_scaled = scaler.transform(MWR_input)

                # retrieve:
                halo_output_pred = model.predict(halo_input_scaled)
                MWR_DS = MWR_DS.sel(freq=MWR_freq)


            # evaluate prediction of each predictand:
            error_dict_syn = dict()
            shape_pred_0 = 0
            shape_pred_1 = 0
            for id_i, predictand in enumerate(aux_i['predictand']):
                # inquire shape of current predictand and its position in the output vector or prediction:
                shape_pred_0 = shape_pred_1
                shape_pred_1 = shape_pred_1 + aux_i['n_ax1'][predictand]

                # compute error statistics:
                if predictand in ['iwv', 'lwp']:
                    error_dict_syn = compute_error_stats(prediction_syn[:,shape_pred_0:shape_pred_1], 
                                                        predictand_test.output[:,shape_pred_0:shape_pred_1], 
                                                        predictand)
                    if aux_i['add_val']:
                        error_dict_add = compute_error_stats(predictand_add.prediction_syn[:,shape_pred_0:shape_pred_1],
                                                            predictand_add.output[:,shape_pred_0:shape_pred_1],
                                                            predictand)

                else:
                    raise ValueError("Unknown predictand.")

                # save error statistics in other dictionary:
                for ek in error_dict_syn.keys():
                    retrieval_stats_syn[f"{predictand}_metrics"][ek].append(error_dict_syn[ek])
                    
                if aux_i['add_val']:
                    for ek in error_dict_add.keys():
                        retrieval_stats_add[f'{predictand}_metrics'][ek].append(error_dict_add[ek])

                # visualize evaluation if desired: (scatter plot for 1D, profiles for...profiles)
                # if aux_i['vis_eval']:
                if (aux_i['seed'] == 773) | aux_i['test_on_all_rngs']:
                    visualize_evaluation(prediction_syn[:,shape_pred_0:shape_pred_1], 
                                        predictand_test.output[:,shape_pred_0:shape_pred_1],
                                        predictand, error_dict_syn, aux_i, predictand_test.height)

                    # if aux_i['add_val']:
                    #     pdb.set_trace()
                        # visualize_evaluation(predictand_add.prediction_syn[:,shape_pred_0:shape_pred_1], 
                        #                     predictand_add.output[:,shape_pred_0:shape_pred_1],
                        #                     predictand, error_dict_add, aux_i, predictand_add.height)


                    # save test MOSAiC obs data if desired:
                    if aux_i['halo_test_subset']:
                        if predictand in ['iwv', 'lwp']:
                            MWR_DS['output'] = xr.DataArray(halo_output_pred[:,shape_pred_0:shape_pred_1].squeeze(), dims=['time'])

                            if (predictand == 'lwp') and aux_i['lwp_offset_cor']: # then also apply clear sky LWP offset correction as in MWR_PRO
                                # identify clear sky:
                                halo_clear_sky_mask = halo_clear_sky_detection(MWR_DS, aux_i, date_now=aux_i['test_date'])      # True: clear sky, False: cloudy

                                lwp_cor = halo_offset_lwp(MWR_DS.time.values.astype('datetime64[s]').astype(np.float64),
                                                        MWR_DS['output'].values, halo_clear_sky_mask)
                                MWR_DS['output'][:] = lwp_cor

                        save_halo_test_obs(MWR_DS['output'], predictand, aux_i, predictand_test.height)


                        # Execute script mosaic_test_obs_comp.py if desired:
                        if aux_i['test_on_all_rngs']:
                            subprocess.run(["python3", f"{wdir}halo_test_obs_comp.py", aux_i['file_descr'], str(aux_i['seed'])])
                            print(f"Successfully executed {wdir}halo_test_obs_comp.py {aux_i['file_descr']}, {str(aux_i['seed'])} \n") 
                            continue


        elif exec_type == 'op_ret':

            # import radiometer data and apply the retrieval for the entire MOSAiC period day by day to save memory:
            date_0_dt = dt.datetime.strptime(aux_i['date_start'], "%Y-%m-%d")
            date_1_dt = dt.datetime.strptime(aux_i['date_end'], "%Y-%m-%d")
            n_days = (date_1_dt - date_0_dt).days + 1
            for c_date in (date_0_dt + n*dt.timedelta(days=1) for n in range(n_days)): 
                c_date_str = c_date.strftime("%Y-%m-%d")


                file_mwr = sorted(glob.glob(aux_i['path_tb_obs'] + f"HALO-AC3_HALO_hamp_radiometer_{c_date_str.replace('-','')}*.nc"))
                if len(file_mwr) == 0:
                    print(f"Skipping {c_date_str}....")
                    continue

                MWR_DS = xr.open_dataset(file_mwr[0]).load()
                MWR_DS = MWR_DS.sortby('freq')
                print(f"Processing HAMP data for {c_date_str}....")

                # rename some variables to unify with synergetic ret NN_retrieval.py:
                MWR_DS = MWR_DS.drop('freq').rename({'TB': 'tb', 'uniRadiometer_freq': 'freq'})

                # filter icey or land surface:
                MWR_DS = MWR_DS.isel(time=np.where(MWR_DS.surface_mask==0)[0])

                # filter roll angles:
                # import BAHAMAS data, put roll angle on MWR time axis and fitler for straight line only:
                file_bah = sorted(glob.glob(aux_i['path_bahamas_obs'] + f"bahamas_{c_date_str.replace('-','')}_*.nc"))[0]
                BAH_DS = xr.open_dataset(file_bah)
                roll_mwr = np.interp(MWR_DS.time.values.astype('datetime64[s]').astype(np.float64), 
                                    BAH_DS.time.values.astype('datetime64[s]').astype(np.float64), 
                                    BAH_DS['roll'].values, left=np.nan, right=np.nan)
                idx_no_curve = np.where(np.abs(roll_mwr) < 1.0)[0]
                MWR_DS = MWR_DS.isel(time=idx_no_curve)


                # eventually correct TB offsets:
                if aux_i['tb_offset_cor']:
                    MWR_DS = hamp_tb_offset_correction(MWR_DS, aux_i['path_tb_offsets'])


                # repeat what has been done to the predictor training data:
                MWR_TB, MWR_freq = select_MWR_channels(MWR_DS.tb.values, MWR_DS.freq.values,
                                                        band=aux_i['predictor_TBs'],
                                                        return_idx=0)

                # build input vector:
                MWR_input = MWR_TB

                # Rescale input: Use MinMaxScaler:
                halo_input_scaled = scaler.transform(MWR_input)

                # retrieve:
                halo_output_pred = model.predict(halo_input_scaled)
                MWR_DS = MWR_DS.sel(freq=MWR_freq)


                # separate predictands and save prediction on MOSAiC data:
                shape_pred_0 = 0
                shape_pred_1 = 0
                for id_i, predictand in enumerate(aux_i['predictand']):
                    # inquire shape of current predictand and its position in the output vector or prediction:
                    shape_pred_0 = shape_pred_1
                    shape_pred_1 = shape_pred_1 + aux_i['n_ax1'][predictand]


                    # save test MOSAiC obs data if desired:
                    if predictand in ['iwv', 'lwp']:
                        MWR_DS['output'] = xr.DataArray(halo_output_pred[:,shape_pred_0:shape_pred_1].squeeze(), dims=['time'])

                        if ((predictand == 'lwp') and aux_i['lwp_offset_cor'] and 
                        (c_date_str not in ['2022-03-15', '2022-04-04', '2022-04-07', '2022-04-08', '2022-04-12'])):
                            # then also apply clear sky LWP offset correction as in MWR_PRO
                            # identify clear sky:
                            halo_clear_sky_mask = halo_clear_sky_detection(MWR_DS, aux_i, date_now=c_date_str)      # True: clear sky, False: cloudy

                            lwp_cor = halo_offset_lwp(MWR_DS.time.values.astype('datetime64[s]').astype(np.float64),
                                                    MWR_DS['output'].values, halo_clear_sky_mask)
                            MWR_DS['output'][:] = lwp_cor


                    # Save to file for each day:
                    if aux_i['save_obs_predictions']:
                        save_obs_predictions(aux_i['path_output'], MWR_DS, predictand, c_date_str, aux_i)


                # clear memory:
                del MWR_DS, BAH_DS, roll_mwr, halo_output_pred, MWR_TB, MWR_freq, halo_input_scaled, MWR_input



            # evaluate prediction of each predictand:
            error_dict_syn = dict()
            shape_pred_0 = 0
            shape_pred_1 = 0
            for id_i, predictand in enumerate(aux_i['predictand']):
                # inquire shape of current predictand and its position in the output vector or prediction:
                shape_pred_0 = shape_pred_1
                shape_pred_1 = shape_pred_1 + aux_i['n_ax1'][predictand]

                # compute error statistics:
                if predictand in ['iwv', 'lwp']:
                    error_dict_syn = compute_error_stats(prediction_syn[:,shape_pred_0:shape_pred_1], 
                                                        predictand_test.output[:,shape_pred_0:shape_pred_1], 
                                                        predictand)

                else:
                    raise ValueError("Unknown predictand.")

                # save error statistics in other dictionary:
                for ek in error_dict_syn.keys():
                    retrieval_stats_syn[f"{predictand}_metrics"][ek].append(error_dict_syn[ek])


        # save other retrieval information (test loss and NN settings):
        retrieval_stats_syn['test_loss'].append(test_loss)      # likely equals np.nanmean((prediction_syn - predictand_test.output)**2)
        retrieval_stats_syn['training_loss'].append(training_loss)
        retrieval_stats_syn['elapsed_epochs'].append(n_epochs_elapsed)  # n elapsed epochs
        retrieval_stats_syn['val_loss_array'][k_s,:n_epochs_elapsed] = val_loss_array
        retrieval_stats_syn['loss_array'][k_s,:n_epochs_elapsed] = loss_array
        for ek in aux_i_stats:
            if ek in ['test_loss', 'elapsed_epochs', 'val_loss_array', 'loss_array', 'training_loss']:
                continue
            else:
                retrieval_stats_syn[ek].append(aux_i[ek])



    if exec_type == '20_runs':
        
        if aux_i['add_val']:
            save_retrieval_stats_eval(retrieval_stats_add, retrieval_stats_syn, aux_i, test_id)

        # Save retrieval stats to xarray dataset, then to netcdf:
        nc_output_name = f"HALO-AC3_NN_ret_retrieval_stat_test_{aux_i['file_descr']}"

        feature_range_0 = np.asarray([fr[0] for fr in retrieval_stats_syn['feature_range']])
        feature_range_1 = np.asarray([fr[1] for fr in retrieval_stats_syn['feature_range']])

        # start forming the data set, inserting retrieval setup information:
        RETRIEVAL_STAT_DS = xr.Dataset({'test_loss':    (['test_id'], np.asarray(retrieval_stats_syn['test_loss']),
                                                        {'description': "Last epoch test data loss, mean square error",
                                                        'units': "SI units"}),
                                        'training_loss':(['test_id'], np.asarray(retrieval_stats_syn['training_loss']),
                                                        {'description': "Last epoch training data loss, mean square error",
                                                        'units': "SI units"}),
                                        'val_loss':     (['test_id', 'n_epochs'], retrieval_stats_syn['val_loss_array'],
                                                        {'description': "Test loss for each elapsed epoch, mean square error"}),
                                        'loss':         (['test_id', 'n_epochs'], retrieval_stats_syn['loss_array'],
                                                        {'description': "Training loss for each elapsed epoch, mean square error"}),
                                        'batch_size':   (['test_id'], np.asarray(retrieval_stats_syn['batch_size']),
                                                        {'description': "Neural Network training batch size"}),
                                        'epochs':       (['test_id'], np.asarray(retrieval_stats_syn['epochs']),
                                                        {'description': "Neural Network training epoch number"}),
                                        'elapsed_epochs': (['test_id'], np.asarray(retrieval_stats_syn['elapsed_epochs']),
                                                        {'description': "Number of epochs elapsed during training"}),
                                        'activation':   (['test_id'], np.asarray(retrieval_stats_syn['activation']),
                                                        {'description': "Neural Network activation function from input to hidden layer"}),
                                        'seed':         (['test_id'], np.asarray(retrieval_stats_syn['seed']),
                                                        {'description': "RNG seed for numpy.random.seed and tensorflow.random.set_seed"}),
                                        'learning_rate':(['test_id'], np.asarray(retrieval_stats_syn['learning_rate']),
                                                        {'description': "Learning rate of NN optimizer"}),
                                        'feature_range0': (['test_id'], feature_range_0,
                                                        {'description': "Lower end of feature range of tensorflow's MinMaxScaler"}),
                                        'feature_range1': (['test_id'], feature_range_1,
                                                        {'description': "Upper end of feature range of tensorflow's MinMaxScaler"})},
                                        coords=         {'test_id': (['test_id'], np.arange(len(retrieval_stats_syn['test_loss'])),
                                                        {'description': "Test number"}),
                                                        'height': (['height'], predictand_test.height[0,:],
                                                        {'description': "Test data height grid",
                                                        'units': "m"})})

        # add the retrieval metrics of test data vs. prediction:
        ret_met_units = {'iwv': "mm", 'lwp': "kg m-2"}
        ret_met_range = {   'iwv': {'bot': "[0,5) mm", 'mid': "[5,10) mm", 'top': "[10,100) mm"},
                            'lwp': {'bot': "[0,0.025) kg m-2", 'mid': "[0.025,0.100) kg m-2", 'top': "[0.100, 1e+06) kg m-2"},
                        }

        for predictand in aux_i['predictand']:

            # description attributes:
            ret_met_descr = {'rmse_tot': f"Test data Root Mean Square Error (RMSE) of target and predicted {predictand}",
                            'rmse_bot': f"Like rmse_tot but confined to {predictand} range {ret_met_range[predictand]['bot']}",
                            'rmse_mid': f"Like rmse_tot but confined to {predictand} range {ret_met_range[predictand]['mid']}",
                            'rmse_top': f"Like rmse_tot but confined to {predictand} range {ret_met_range[predictand]['top']}",
                            'bias_tot': f"Bias of test data predicted - target {predictand}",
                            'bias_bot': f"Like bias_tot but confined to {predictand} range {ret_met_range[predictand]['bot']}",
                            'bias_mid': f"Like bias_tot but confined to {predictand} range {ret_met_range[predictand]['mid']}",
                            'bias_top': f"Like bias_tot but confined to {predictand} range {ret_met_range[predictand]['top']}",
                            'stddev': f"Test data standard deviation (bias corrected RMSE) of target and predicted {predictand}",
                            'stddev_bot': f"Like stddev but confined to {predictand} range {ret_met_range[predictand]['bot']}",
                            'stddev_mid': f"Like stddev but confined to {predictand} range {ret_met_range[predictand]['mid']}",
                            'stddev_top': f"Like stddev but confined to {predictand} range {ret_met_range[predictand]['top']}"}

            # save retrieval metrics to dataset and forward the variable attributes:
            for ret_met in ret_metrics:

                RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"] = xr.DataArray(np.asarray(retrieval_stats_syn[f"{predictand}_metrics"][ret_met]),
                                                                            dims=['test_id'])
                RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['description'] = ret_met_descr[ret_met]
                RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['units'] = ret_met_units[predictand]
                if "bot" in ret_met:
                    RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['range'] = ret_met_range[predictand]['bot']
                elif "mid" in ret_met:
                    RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['range'] = ret_met_range[predictand]['mid']
                elif "top" in ret_met:
                    RETRIEVAL_STAT_DS[f"{predictand}_{ret_met}"].attrs['range'] = ret_met_range[predictand]['top']


        # Provide some global attributes
        RETRIEVAL_STAT_DS.attrs['test_purpose'] = test_id
        RETRIEVAL_STAT_DS.attrs['author'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"
        RETRIEVAL_STAT_DS.attrs['predictands'] = ""
        for predictand in aux_i['predictand']: RETRIEVAL_STAT_DS.attrs['predictands'] += predictand + ", "
        RETRIEVAL_STAT_DS.attrs['predictands'] = RETRIEVAL_STAT_DS.attrs['predictands'][:-2]


        if aux_i['site'] == 'era5':
            RETRIEVAL_STAT_DS.attrs['training_data'] = "ERA5, PAMTRA simulations"
            RETRIEVAL_STAT_DS.attrs['test_data'] = "ERA5, PAMTRA simulations"

        datetime_utc = dt.datetime.utcnow()
        RETRIEVAL_STAT_DS.attrs['datetime_of_creation'] = datetime_utc.strftime("%Y-%m-%d %H:%M:%S")


        # create output path if not existing:
        outpath_dir = os.path.dirname(aux_i['path_output'] + "ret_stat/")
        if not os.path.exists(outpath_dir):
            os.makedirs(outpath_dir)
        RETRIEVAL_STAT_DS.to_netcdf(aux_i['path_output'] + "ret_stat/" + nc_output_name + ".nc", mode='w', format="NETCDF4")
        RETRIEVAL_STAT_DS = RETRIEVAL_STAT_DS.close()


print(f"Test purpose: {test_id}")
print("Done....")
datetime_utc = dt.datetime.utcnow()
print(datetime_utc - ssstart)