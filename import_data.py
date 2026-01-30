import numpy as np
import netCDF4 as nc
import datetime as dt
import pandas as pd
# import xarray as xr
import copy
import pdb
import os
import glob
import sys
import warnings
import csv
from met_tools import *
from data_tools import *


def import_mirac_IWV_RPG(
    filename,
    keys='basic',
    minute_avg=False):

    """
    Importing automatically created MiRAC-P IWV hourly retrieval files
    with the ending .IWV.NC in the level 1 folder. Time will be
    converted to seconds since 1970-01-01 00:00:00 UTC.

    Parameters:
    -----------
    filename : str
        Path and filename of MiRAC-P .IWV.NC data.
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    minute_avg : bool
        If True: averages over one minute are computed and returned instead of False when all
        data points are returned (more outliers, higher memory usage).
    """

    # IWV in kg m^-2!

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        # keys = ['time', 'RF', 'ElAng', 'AziAng', 'IWV']
        keys = ['time', 'RF', 'IWV']

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    reftime = dt.datetime(1970,1,1,0,0,0)
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in MiRAC-P .IWV.NC file." % key)

        mwr_dict[key] = np.asarray(file_nc.variables[key])

        if key == 'time':   # convert to sec since 1970-01-01 00:00:00 UTC (USE FLOAT64)
            mwr_dict['time'] = (np.float64(datetime_to_epochtime(dt.datetime(2001,1,1,0,0,0))) +
                                mwr_dict[key].astype(np.float64))



    if minute_avg and 'time' in keys:
        time_shape_old = mwr_dict['time'].shape
        # start the timer at the first time, when seconds is 00 (e.g. 09:31:00):
        time0 = mwr_dict['time'][0]     # start time in sec since 1970-01-01...
        dt_time0 = dt.datetime.utcfromtimestamp(mwr_dict['time'][0])
        dt_time0_Y = dt_time0.year
        dt_time0_M = dt_time0.month
        dt_time0_D = dt_time0.day
        dt_time0_s = dt_time0.second
        dt_time0_m = dt_time0.minute
        dt_time0_h = dt_time0.hour
        if dt_time0_s != 0:     # then the array mwr_dict['time'] does not start at second 0
            start_time = datetime_to_epochtime(dt.datetime(dt_time0_Y, dt_time0_M,
                            dt_time0_D, dt_time0_h, dt_time0_m+1, 0))
        else:
            start_time = time0

        if np.abs(start_time - time0) >= 60:
            print("Start time is far off the first time point in this file.")
            pdb.set_trace()
        # compute minute average
        n_minutes = int(np.ceil((mwr_dict['time'][-1] - start_time)/60))    # number of minutes
        min_time_idx_save = 0       # saves the last min_time_index value to speed up computation
        for min_count in range(n_minutes):
            # find time_idx when time is in the correct minute:
            # slower version:
            # # # min_time_idx = np.argwhere((mwr_dict['time'] >= (start_time + min_count*60)) & 
                            # # # (mwr_dict['time'] < (start_time + (min_count+1)*60))).flatten()
            # faster version:
            min_time_idx = np.argwhere((mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] >= (start_time + min_count*60)) & 
                            (mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] < (start_time + (min_count+1)*60))).flatten()

            # it may occur that no measurement exists in a certain minute-range. Then
            # we cannot compute the average but simply set that minute to nan.
            if len(min_time_idx) == 0:
                for key in keys:
                    if key == 'time':
                        mwr_dict['time'][min_count] = start_time + min_count*60
                    elif mwr_dict[key].shape == time_shape_old and key != 'RF':
                        mwr_dict[key][min_count] = np.nan
                    elif mwr_dict[key].shape == time_shape_old and key == 'RF':
                        mwr_dict[key][min_count] = 99       # np.nan not possible because int is required
            else:
                min_time_idx = min_time_idx + min_time_idx_save     # also belonging to the 'faster version'
                min_time_idx_save = min_time_idx[-1]                # also belonging to the 'faster version'
                for key in keys:
                    if key == 'time':
                        mwr_dict['time'][min_count] = start_time + min_count*60
                    elif mwr_dict[key].shape == time_shape_old and key != 'RF':
                        if min_time_idx[-1] < len(mwr_dict['time']):
                            mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx])
                        else:
                            mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx[0]:])
                    elif mwr_dict[key].shape == time_shape_old and key == 'RF':
                        # find out how many entries show flag > 0. Then if it exceeds a threshold
                        # the whole minute is flagged. If threshold not exceeded, minute is not
                        # flagged!
                        if min_time_idx[-1] < len(mwr_dict['time']):
                            if np.count_nonzero(mwr_dict[key][min_time_idx]) > len(min_time_idx)/10:    
                                # then there are too many flags set... so flag the whole minute:
                                mwr_dict[key][min_count] = 99
                            else:
                                mwr_dict[key][min_count] = 0
                        else:
                            if np.count_nonzero(mwr_dict[key][min_time_idx[0]:]) > len(min_time_idx)/10:
                                # then there are too many flags set... so flag the whole minute:
                                mwr_dict[key][min_count] = 99
                            else:
                                mwr_dict[key][min_count] = 0

        # truncate time arrays to reduce memory usage!
        for key in keys:
            if mwr_dict[key].shape == time_shape_old:
                mwr_dict[key] = mwr_dict[key][:n_minutes]

    else:
        if minute_avg:
            raise KeyError("'time' must be included in the list of keys that will be imported for minute averages.")

    return mwr_dict


def import_mirac_LWP_RPG(
    filename,
    keys='basic',
    minute_avg=False):

    """
    Importing automatically created MiRAC-P LWP hourly retrieval files
    with the ending .LWP.NC in the level 1 folder. Time will be
    converted to seconds since 1970-01-01 00:00:00 UTC.

    Parameters:
    -----------
    filename : str
        Path and filename of MiRAC-P .LWP.NC data.
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    minute_avg : bool
        If True: averages over one minute are computed and returned instead of False when all
        data points are returned (more outliers, higher memory usage).
    """

    # LWP in g m^-2!

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        # keys = ['time', 'RF', 'ElAng', 'AziAng', 'LWP', 'retrieval']
        keys = ['time', 'RF', 'LWP']

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    reftime = dt.datetime(1970,1,1,0,0,0)
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in MiRAC-P .LWP.NC file." % key)

        mwr_dict[key] = np.asarray(file_nc.variables[key])

        if key == 'time':   # convert to sec since 1970-01-01 00:00:00 UTC (USE FLOAT64)
            mwr_dict['time'] = (np.float64(datetime_to_epochtime(dt.datetime(2001,1,1,0,0,0))) +
                                mwr_dict[key].astype(np.float64))
        elif key == 'LWP':
            mwr_dict[key] = mwr_dict[key]/1000      # convert LWP to kg m^-2


    if minute_avg and 'time' in keys:
        time_shape_old = mwr_dict['time'].shape
        # start the timer at the first time, when seconds is 00 (e.g. 09:31:00):
        time0 = mwr_dict['time'][0]     # start time in sec since 1970-01-01...
        dt_time0 = dt.datetime.utcfromtimestamp(mwr_dict['time'][0])
        dt_time0_Y = dt_time0.year
        dt_time0_M = dt_time0.month
        dt_time0_D = dt_time0.day
        dt_time0_s = dt_time0.second
        dt_time0_m = dt_time0.minute
        dt_time0_h = dt_time0.hour
        if dt_time0_s != 0:     # then the array mwr_dict['time'] does not start at second 0
            start_time = datetime_to_epochtime(dt.datetime(dt_time0_Y, dt_time0_M, 
                                                dt_time0_D, dt_time0_h, dt_time0_m+1, 0))
        else:
            start_time = time0

        if np.abs(start_time - time0) >= 60:
            print("Start time is far off the first time point in this file.")
            pdb.set_trace()
        # compute minute average
        n_minutes = int(np.ceil((mwr_dict['time'][-1] - start_time)/60))    # number of minutes
        min_time_idx_save = 0       # saves the last min_time_index value to speed up computation
        for min_count in range(n_minutes):
            # find time_idx when time is in the correct minute:
            # slower version:
            # # # min_time_idx = np.argwhere((mwr_dict['time'] >= (start_time + min_count*60)) & 
                            # # # (mwr_dict['time'] < (start_time + (min_count+1)*60))).flatten()
            # faster version:
            min_time_idx = np.argwhere((mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] >= (start_time + min_count*60)) & 
                            (mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] < (start_time + (min_count+1)*60))).flatten()

            # it may occur that no measurement exists in a certain minute-range. Then
            # we cannot compute the average but simply set that minute to nan.
            if len(min_time_idx) == 0:
                for key in keys:
                    if key == 'time':
                        mwr_dict['time'][min_count] = start_time + min_count*60
                    elif mwr_dict[key].shape == time_shape_old and key != 'RF':
                        mwr_dict[key][min_count] = np.nan
                    elif mwr_dict[key].shape == time_shape_old and key == 'RF':
                        mwr_dict[key][min_count] = 99       # np.nan not possible because int is required
            else:
                min_time_idx = min_time_idx + min_time_idx_save     # also belonging to the 'faster version'
                min_time_idx_save = min_time_idx[-1]                # also belonging to the 'faster version'
                for key in keys:
                    if key == 'time':
                        mwr_dict['time'][min_count] = start_time + min_count*60
                    elif mwr_dict[key].shape == time_shape_old and key != 'RF':
                        if min_time_idx[-1] < len(mwr_dict['time']):
                            mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx])
                        else:
                            mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx[0]:])
                    elif mwr_dict[key].shape == time_shape_old and key == 'RF':
                        # find out how many entries show flag > 0. Then if it exceeds a threshold
                        # the whole minute is flagged. If threshold not exceeded, minute is not
                        # flagged!
                        if min_time_idx[-1] < len(mwr_dict['time']):
                            if np.count_nonzero(mwr_dict[key][min_time_idx]) > len(min_time_idx)/10:    
                                # then there are too many flags set... so flag the whole minute:
                                mwr_dict[key][min_count] = 99
                            else:
                                mwr_dict[key][min_count] = 0
                        else:
                            if np.count_nonzero(mwr_dict[key][min_time_idx[0]:]) > len(min_time_idx)/10:
                                # then there are too many flags set... so flag the whole minute:
                                mwr_dict[key][min_count] = 99
                            else:
                                mwr_dict[key][min_count] = 0

        # truncate time arrays to reduce memory usage!
        for key in keys:
            if mwr_dict[key].shape == time_shape_old:
                mwr_dict[key] = mwr_dict[key][:n_minutes]

    else:
        if minute_avg:
            raise KeyError("'time' must be included in the list of keys that will be imported for minute averages.")


    return mwr_dict


def import_mirac_IWV_LWP_RPG_daterange(
    path_data,
    date_start,
    date_end,
    which_retrieval='both',
    minute_avg=False,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the IWV or LWP time
    series of each day so that you'll have one dictionary that will contain the LWP or IWV
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Base path of MiRAC-P level 1 data. This directory contains subfolders representing the 
        year, which, in turn, contain months, which contain day subfolders. Example:
        path_data = "/data/obs/campaigns/mosaic/mirac-p/l1/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. 'both' will 
        load both. Default: 'both'
    minute_avg : bool
        If True: averages over one minute are computed and returned instead of False when all
        data points are returned (more outliers, higher memory usage).
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # check if the input of the retrieval variable is okay:
    if isinstance(which_retrieval, str) and (which_retrieval in ['prw', 'iwv', 'clwvi', 'lwp', 'both']):
        wr = {'iwv': ['IWV'],
                'prw': ['IWV'],
                'lwp': ['LWP'],
                'clwvi': ['LWP'],
                'both': ['IWV', 'LWP']} 
        which_retrieval = wr[which_retrieval]

    else:
        raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")


    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1

    # basic variables that should always be imported:
    # mwr_time_keys = ['time', 'ElAng', 'AziAng', 'RF'] # keys with time as coordinate
    mwr_time_keys = ['time', 'RF']  # keys with time as coordinate

    # mwr_master_dict (output) will contain all desired variables on time axis for entire date range:
    mwr_master_dict = dict()
    if minute_avg:  # max number of minutes: n_days*1440
        n_minutes = n_days*1440
        for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_minutes,), np.nan)
        for data_key in which_retrieval: mwr_master_dict[data_key] = np.full((n_minutes,), np.nan)
    else:           # max number of seconds: n_days*86400
        n_seconds = n_days*86400
        for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_seconds,), np.nan)
        for data_key in which_retrieval: mwr_master_dict[data_key] = np.full((n_seconds,), np.nan)


    # Load the IWV and/or LWP into mwr_master_dict:
    # cycle through all years, all months and days:
    time_index_l = 0    # this index will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly. For LWP
    time_index_i = 0    # For IWV
    day_index_l = 0 # will increase for each day. For LWP
    day_index_i = 0 # will increase for each day. For IWV
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("Working on RPG retrieval, MiRAC-P IWV or LWP, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # list of .LWP.NC and .IWV.NC files: Sorting is important as this will
        # ensure automatically that the time series of each hour will
        # be concatenated appropriately!
        if 'LWP' in which_retrieval:
            mirac_lwp_nc = sorted(glob.glob(day_path + "*.LWP.NC"))
            if len(mirac_lwp_nc) == 0:
                if verbose >= 2:
                    warnings.warn("No .LWP.NC files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
                continue

            # load one retrieved variable after another from current day and save it into the mwr_master_dict
            for lwp_file in mirac_lwp_nc: 
                mwr_dict = import_mirac_LWP_RPG(lwp_file, minute_avg=minute_avg)

                n_time = len(mwr_dict['time'])
                time_shape_l = mwr_dict['time'].shape

                # save to mwr_master_dict
                for mwr_key in mwr_dict.keys():
                    if mwr_dict[mwr_key].shape == time_shape_l:
                        mwr_master_dict[mwr_key][time_index_l:time_index_l + n_time] = mwr_dict[mwr_key]

                    elif mwr_dict[mwr_key].shape == ():
                        mwr_master_dict[mwr_key][day_index_l] = mwr_dict[mwr_key]

                    else:
                            raise ValueError("The length of one used variable ('%s') of MiRAC-P .LWP.NC data "%(mwr_key) +
                                "neither equals the length of the time axis nor equals 1.")

                time_index_l = time_index_l + n_time
            day_index_l = day_index_l + 1

        if 'IWV' in which_retrieval:
            mirac_iwv_nc = sorted(glob.glob(day_path + "*.IWV.NC"))

            if len(mirac_iwv_nc) == 0:
                if verbose >= 2:
                    warnings.warn("No .IWV.NC files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
                continue

            for iwv_file in mirac_iwv_nc: 
                mwr_dict = import_mirac_IWV_RPG(iwv_file, minute_avg=minute_avg)

                n_time = len(mwr_dict['time'])
                time_shape_i = mwr_dict['time'].shape

                # save to mwr_master_dict
                for mwr_key in mwr_dict.keys():
                    if mwr_dict[mwr_key].shape == time_shape_i:
                        mwr_master_dict[mwr_key][time_index_i:time_index_i + n_time] = mwr_dict[mwr_key]

                    elif mwr_dict[mwr_key].shape == ():
                        mwr_master_dict[mwr_key][day_index_i] = mwr_dict[mwr_key]

                    else:
                            raise ValueError("The length of one used variable ('%s') of MiRAC-P .LWP.NC data "%(mwr_key) +
                                "neither equals the length of the time axis nor equals 1.")

                time_index_i = time_index_i + n_time
            day_index_i = day_index_i + 1

    if 'LWP' in which_retrieval and 'IWV' in which_retrieval:
        if np.any(np.asarray([time_index_i, time_index_l]) == 0) and verbose >= 1:  # otherwise no data has been found
            raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                    dt.datetime.strftime(date_end, "%Y-%m-%d"))
        elif time_index_i == time_index_l:
            # truncate the mwr_master_dict to the last nonnan time index:
            last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
            time_shape_old = mwr_master_dict['time'].shape
            for mwr_key in mwr_master_dict.keys():
                if mwr_master_dict[mwr_key].shape == time_shape_old:
                    mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]
        else:
            pdb.set_trace()
            raise ValueError("LWP and IWV data seem to have a different number of data points.")

    elif which_retrieval == ['LWP'] or which_retrieval == ['IWV']:
        time_index = np.amax(np.asarray([time_index_l, time_index_i]))
    
        if time_index == 0 and verbose >= 1:    # otherwise no data has been found
            raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                    dt.datetime.strftime(date_end, "%Y-%m-%d"))
        else:
            # truncate the mwr_master_dict to the last nonnan time index:
            last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
            time_shape_old = mwr_master_dict['time'].shape
            for mwr_key in mwr_master_dict.keys():
                if mwr_master_dict[mwr_key].shape == time_shape_old:
                    mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]


    return mwr_master_dict


def import_mirac_BRT(
    filename,
    keys='basic'):

    """
    Importing automatically created MiRAC-P BRT hourly files
    with the ending .BRT.NC in the level 1 folder. Time will be
    converted to seconds since 1970-01-01 00:00:00 UTC.

    Parameters:
    -----------
    filename : str
        Path and filename of MiRAC-P .BRT.NC data.
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    """

    # BRT in K!

    file_nc = nc.Dataset(filename)

    if keys == 'basic':
        keys = ['time', 'RF', 'TBs', 'Freq']

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    reftime = dt.datetime(1970,1,1,0,0,0)
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in MiRAC-P .BRT.NC file." % key)

        mwr_dict[key] = np.asarray(file_nc.variables[key])

        if key == 'time':   # convert to sec since 1970-01-01 00:00:00 UTC (USE FLOAT64)
            mwr_dict['time'] = (np.float64(datetime_to_epochtime(dt.datetime(2001,1,1,0,0,0))) +
                                mwr_dict[key].astype(np.float64))

    return mwr_dict


def import_mirac_BRT_RPG_daterange(
    path_data,
    date_start,
    date_end,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the BRT time
    series of each day so that you'll have one dictionary that will contain the TBs
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Base path of MiRAC-P level 1 data. This directory contains subfolders representing the 
        year, which, in turn, contain months, which contain day subfolders. Example:
        path_data = "/data/obs/campaigns/mosaic/mirac-p/l1/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1

    # basic variables that should always be imported:
    mwr_time_keys = ['time', 'RF']  # keys with time as coordinate
    mwr_time_freq_keys = ['TBs']    # keys with time and frequency as coordinates
    mwr_freq_keys = ['Freq']        # keys with frequency as coordinate

    # mwr_master_dict (output) will contain all desired variables on time axis for entire date range:
    mwr_master_dict = dict()
    n_seconds = n_days*86400        # max number of seconds: n_days*86400
    n_freq = 8                      # number of frequencies (inquired from .BRT.NC file)
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_seconds,), np.nan)
    for mtfk in mwr_time_freq_keys: mwr_master_dict[mtfk] = np.full((n_seconds, n_freq), np.nan)
    for mfk in mwr_freq_keys: mwr_master_dict[mfk] = np.full((n_freq,), np.nan)


    # Load the TBs into mwr_master_dict:
    # cycle through all years, all months and days:
    time_index = 0  # this index will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    day_index = 0   # will increase for each day
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("Working on RPG retrieval, MiRAC-P BRT, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # list of .BRT.NC files: Sorting is important as this will
        # ensure automatically that the time series of each hour will
        # be concatenated appropriately!
        mirac_nc = sorted(glob.glob(day_path + "*.BRT.NC"))
        if len(mirac_nc) == 0:
            if verbose >= 2:
                warnings.warn("No .BRT.NC files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for l1_file in mirac_nc: 
            mwr_dict = import_mirac_BRT(l1_file)

            n_time = len(mwr_dict['time'])
            time_shape = mwr_dict['time'].shape

            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                if mwr_key in mwr_time_keys:
                    mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key]

                elif mwr_key in mwr_time_freq_keys:
                    mwr_master_dict[mwr_key][time_index:time_index + n_time,:] = mwr_dict[mwr_key]

                elif mwr_dict[mwr_key].shape == ():
                    mwr_master_dict[mwr_key][day_index] = mwr_dict[mwr_key]

                elif mwr_key in mwr_freq_keys:  # frequency will be handled after the for loop
                    continue

                else:
                        raise ValueError("The length of one used variable ('%s') of MiRAC-P .BRT.NC data "%(mwr_key) +
                            "neither equals the length of the time axis nor equals 1.")

            time_index = time_index + n_time
        day_index = day_index + 1


    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))

    else:
        # Save frequency array into mwr_master_dict:
        for fkey in mwr_freq_keys: mwr_master_dict[fkey] = mwr_dict[fkey]

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        time_freq_shape_old = mwr_master_dict[mwr_time_freq_keys[0]].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_freq_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1,:]

    return mwr_master_dict


def import_mirac_HPC_RPG(
    filename,
    keys='basic'):

    """
    Importing automatically created MiRAC-P HPC hourly retrieval files
    with the ending .HPC.NC in the level 1 folder. Time will be
    converted to seconds since 1970-01-01 00:00:00 UTC. Absolute humidity
    will be converted to kg m^-3.

    Parameters:
    -----------
    filename : str
        Path and filename of MiRAC-P .HPC.NC data.
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    """

    # absolute humidity in g m^-3! Must be converted

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        # keys = ['time', 'RF', 'altitude', 'AH_Prof', 'elevation', 'azimuth', 'right_ascension',
                    # 'declination']
        keys = ['time', 'RF', 'altitude', 'AH_Prof']

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    reftime = dt.datetime(1970,1,1,0,0,0)
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in MiRAC-P .LWP.NC file." % key)

        mwr_dict[key] = np.asarray(file_nc.variables[key])

        if key == 'time':   # convert to sec since 1970-01-01 00:00:00 UTC (USE FLOAT64)
            mwr_dict['time'] = (np.float64(datetime_to_epochtime(dt.datetime(2001,1,1))) +
                                mwr_dict[key].astype(np.float64))
        elif key == 'AH_Prof':
            mwr_dict[key] = mwr_dict[key]/1000      # convert absolute humidity to kg m^-2
            mwr_dict['hua'] = mwr_dict[key]         # renamed variable
        elif key == 'altitude':
            mwr_dict['height'] = mwr_dict[key]      # renamed variable

    return mwr_dict


def import_mirac_HUA_RPG_daterange(
    path_data,
    date_start,
    date_end,
    around_radiosondes=True,
    path_radiosondes="",
    s_version='level_2',
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the humidity profile data
    time steps of each day at certain sample times [default: 05, 11, 17 and 23 UTC to agree with
    radiosonde launches during MOSAiC] so that you'll have one dictionary that will contain the
    data for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Base path of MiRAC-P level 1 data. This directory contains subfolders representing the 
        year, which, in turn, contain months, which contain day subfolders. Example:
        path_data = "/data/obs/campaigns/mosaic/mirac-p/l1/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    around_radiosondes : bool, optional
        If True, data will be limited to the time around radiosonde launches. If False, something else
        (e.g. around 4 times a day) might be done. Default: True
    path_radiosondes : str, optional
        Path to radiosonde data (Level 2). Default: ""
    s_version : str
        Specifies the radiosonde version that is to be imported. Must be 'level_2' to work properly.
        Other versions have not been implemeted because they are considered to be inferior to level_2
        radiosondes.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # check if around_radiosondes is the right type:
    if not isinstance(around_radiosondes, bool):
        raise TypeError("Argument 'around_radiosondes' must be either True or False (boolean type).")

    # outlier file to filter invalid data:
    MiRAC_outlier_file = "/net/blanc/awalbroe/Codes/MOSAiC/MiRAC-P_outliers.txt"

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_hgt = 93      # inquired from .HPC.NC file

    # Create an array that includes the radiosonde launch times:
    if around_radiosondes:
        if not path_radiosondes:
            raise ValueError("If 'around_radiosondes' is True, the path to the radiosonde level 2 data ('pathradiosondes') " +
                                "must be given.")
        if s_version != 'level_2':
            raise ValueError("Radiosonde version 's_version' must be 'level_2' if around_radiosondes is True because " +
                                "for this version, the launch time is directly read from the filename. This has not " +
                                "been implemeted for other radiosonde versions ('mossonde', 'psYYMMDDwHH') because these " +
                                "are considered to be inferior.")
        else:
            add_files = sorted(glob.glob(path_radiosondes + "*.nc"))        # filenames only; filter path
            add_files = [os.path.basename(a_f) for a_f in add_files]
            
            # identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            for a_f in add_files:
                ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                # only save those that are in the considered period
                if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                    launch_times[kk] = ltt
                    kk += 1
            
            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            launch_times_sec = datetime_to_epochtime(launch_times)
            sample_times_t = launch_times_sec
            n_samp_tot = len(sample_times_t)

    else:
        # max number of samples: n_days*4
        sample_times = [5, 11, 17, 23]      # UTC on each day
        n_samp = len(sample_times)
        n_samp_tot = n_days*n_samp

    n_seconds = n_days*86400        # number of entries

    # basic variables that should always be imported:
    # mwr_time_keys = ['time', 'elevation', 'azimuth', 'right_ascension', 
                        # 'declination', 'RF']  # keys with time as coordinate
    mwr_time_keys = ['time', 'RF']  # keys with time as coordinate
    mwr_time_height_keys = ['hua']  # keys with time and height and coordinates

    # mwr_master_dict (output) will contain all desired variables on time axis for entire date range:
    mwr_master_dict = dict()
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_seconds,), np.nan)
    for mthk in mwr_time_height_keys: mwr_master_dict[mthk] = np.full((n_seconds, n_hgt), np.nan)
    mwr_master_dict['height'] = np.full((n_hgt,), np.nan)

    # import_keys contains the names of the variables that will be imported.
    # all_relevant_keys contains the variable names after the import (where renaming 
    # was performed) has been completed
    import_keys = mwr_time_keys + ['AH_Prof', 'altitude']
    all_relevant_keys = mwr_time_keys + mwr_time_height_keys + ['height']


    # Load the humidity profile data into mwr_master_dict:
    # cycle through all years, months and days:
    time_index = 0  # this index will be increased by the length of the time series of the
                        # current day (now_date) to fill the mwr_master_dict time axis accordingly.
    day_index = 0   # will increase for each day
    reftime = dt.datetime(1970, 1, 1)       # reference time: 1970-01-01 00:00:00 UTC
    if not around_radiosondes:
        sample_times_t = np.full((n_samp_tot,), np.nan)     # will contain all manual sample times
        sample_idx = 0                                      # index to increment sample_times_t
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("Working on RPG retrieval, MiRAC-P IWV or LWP, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # list of .HPC.NC files: Sorting is important as this will
        # ensure automatically that the time series of each hour will
        # be concatenated appropriately!
        mirac_hpc_nc = sorted(glob.glob(day_path + "*.HPC.NC"))
        if len(mirac_hpc_nc) == 0:
            if verbose >= 2:
                warnings.warn("No .HPC.NC files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        if not around_radiosondes:  # in this case, the sample times will be e.g. at certain times
            sample_times_tc = np.asarray([datetime_to_epochtime(dt.datetime(yyyy, mm, dd, st, 0, 0)) for st in sample_times])
            sample_times_t[sample_idx:sample_idx + n_samp] = sample_times_tc
            sample_idx += n_samp

        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for hpc_file in mirac_hpc_nc: 
            mwr_dict = import_mirac_HPC_RPG(hpc_file, keys=import_keys)

            # update the flag by taking the manually detected outliers into account:
            mwr_dict['RF'] = outliers_per_eye(mwr_dict['RF'], mwr_dict['time'], instrument='mirac', filename=MiRAC_outlier_file)

            n_time = len(mwr_dict['time'][mwr_dict['RF'] == 0])

            # save to mwr_master_dict
            for mwr_key in all_relevant_keys:
                if mwr_dict[mwr_key].shape == mwr_dict['time'].shape:
                    mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key][mwr_dict['RF'] == 0]

                elif mwr_key == 'height': # handled after the loop
                    continue

                elif mwr_key in mwr_time_height_keys:
                    # first: filter for non-flagged values
                    mwr_dict[mwr_key] = mwr_dict[mwr_key][mwr_dict['RF'] == 0, :]
                    mwr_master_dict[mwr_key][time_index:time_index + n_time,:] = mwr_dict[mwr_key]
                    
                else:
                        raise ValueError("The shape of one used variable ('%s') of MiRAC-P .HPC.NC data "%(mwr_key) +
                            " is unexpected.")

            time_index = time_index + n_time
        day_index = day_index + 1


    if time_index == 0 and verbose >= 1:    # no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # Save height key to master dict:
        mwr_master_dict['height'] = mwr_dict['height']

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        time_height_shape_old = mwr_master_dict[mwr_time_height_keys[0]].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_height_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1, :]


    # Filter for radiosonde launches or specific sample times:
    # specify sample times as time: sec since 1970-01-01 00:00:00 UTC:
    sample_time_tolerance = 900     # radiometer time must be within sonde launch time +/- sample_time_tolerance
    if around_radiosondes:
        n_time = len(mwr_master_dict['time'])
        sample_mask = np.full((n_time,), False)
        for l_t in sample_times_t:
            le_condition = np.abs(mwr_master_dict['time'] - l_t) == np.nanmin(np.abs(mwr_master_dict['time'] - l_t))
            # make sure only one minimum is found:
            narg = np.argwhere(le_condition).flatten()
            if len(narg) > 1:
                le_condition[narg[0]] = False
            if np.all(np.abs(mwr_master_dict['time'][narg] - l_t) <= sample_time_tolerance):
                sample_mask[le_condition] = True

        # truncate the mwr_master_dict:
        time_shape_old = mwr_master_dict['time'].shape
        time_height_shape_old = mwr_master_dict[mwr_time_height_keys[0]].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][sample_mask]
            elif shape_new == time_height_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][sample_mask,:]

    else:
        n_time = len(mwr_master_dict['time'])
        sample_mask = np.full((n_time,), False)
        for l_t in sample_times_t:
            sample_mask[np.abs(mwr_master_dict['time'] - l_t) == np.nanmin(np.abs(mwr_master_dict['time'] - l_t))] = True
            # make sure only one minimum is found:
            narg = np.argwhere(le_condition).flatten()
            if len(narg) > 1:
                le_condition[narg[0]] = False
            if np.abs(mwr_master_dict['time'][narg] - l_t) <= sample_time_tolerance: 
                sample_mask[le_condition] = True

        # truncate the mwr_master_dict:
        time_shape_old = mwr_master_dict['time'].shape
        time_height_shape_old = mwr_master_dict[mwr_time_height_keys[0]].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][sample_mask]
            elif shape_new == time_height_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][sample_mask,:]

    return mwr_master_dict


def import_mirac_level2a(
    filename,
    keys='basic',
    minute_avg=False):

    """
    Importing MiRAC-P level 2a (integrated quantities, e.g. IWV, LWP).

    Parameters:
    -----------
    filename : str
        Path and filename of mwr data (level2a).
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    minute_avg : bool
        If True: averages over one minute are computed and returned instead of False when all
        data points are returned (more outliers, higher memory usage).
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'lat', 'lon', 'zsl', 'azi', 'ele', 'flag']
        if 'clwvi_' in filename:
            for add_key in ['clwvi']: keys.append(add_key)
        if 'prw_' in filename:
            for add_key in ['prw']: keys.append(add_key)

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in level 2a file." % key)
        mwr_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        time_shape_old = mwr_dict['time'].shape

        if minute_avg:
            # start the timer at the first time, when seconds is 00 (e.g. 09:31:00):
            time0 = mwr_dict['time'][0]     # start time in sec since 1970-01-01...
            dt_time0 = dt.datetime.utcfromtimestamp(mwr_dict['time'][0])
            dt_time0_Y = dt_time0.year
            dt_time0_M = dt_time0.month
            dt_time0_D = dt_time0.day
            dt_time0_s = dt_time0.second
            dt_time0_m = dt_time0.minute
            dt_time0_h = dt_time0.hour
            if dt_time0_s != 0:     # then the array mwr_dict['time'] does not start at second 0
                start_time = datetime_to_epochtime(dt.datetime(dt_time0_Y, dt_time0_M,
                                                    dt_time0_D, dt_time0_h, dt_time0_m+1, 0))
            else:
                start_time = time0

            if np.abs(start_time - time0) >= 60:
                print("Start time is far off the first time point in this file.")
                pdb.set_trace()
            # compute minute average
            n_minutes = int(np.ceil((mwr_dict['time'][-1] - start_time)/60))    # number of minutes
            min_time_idx_save = 0       # saves the last min_time_index value to speed up computation
            for min_count in range(n_minutes):
                # find time_idx when time is in the correct minute:
                # slower version:
                # # # # min_time_idx = np.argwhere((mwr_dict['time'] >= (start_time + min_count*60)) & 
                                # # # # (mwr_dict['time'] < (start_time + (min_count+1)*60))).flatten()
                # faster version:
                min_time_idx = np.argwhere((mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] >= (start_time + min_count*60)) & 
                                (mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] < (start_time + (min_count+1)*60))).flatten()

                # it may occur that no measurement exists in a certain minute-range. Then
                # we cannot compute the average but simply set that minute to nan.
                if len(min_time_idx) == 0:
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            mwr_dict[key][min_count] = np.nan
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            mwr_dict[key][min_count] = 99       # np.nan not possible because int is required
                else:
                    min_time_idx = min_time_idx + min_time_idx_save     # also belonging to the 'faster version'
                    min_time_idx_save = min_time_idx[-1]                # also belonging to the 'faster version'
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx])
                            else:
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx[0]:])
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            # find out how many entries show flag > 0. Then if it exceeds a threshold
                            # the whole minute is flagged. If threshold not exceeded, minute is not
                            # flagged!
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                if np.count_nonzero(mwr_dict[key][min_time_idx]) > len(min_time_idx)/10:    
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0
                            else:
                                if np.count_nonzero(mwr_dict[key][min_time_idx[0]:]) > len(min_time_idx)/10:
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0

            # truncate time arrays to reduce memory usage!
            for key in keys:
                if mwr_dict[key].shape == time_shape_old:
                    mwr_dict[key] = mwr_dict[key][:n_minutes]

    else:
        if minute_avg:
            raise KeyError("'time' must be included in the list of keys that will be imported for minute averages.")

    return mwr_dict


def import_mirac_level1b_daterange(
    path_data,
    date_start,
    date_end,
    vers='v01',
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the level 1b TB time
    series of each day so that you'll have one dictionary, whose 'TB' will contain the IWV
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Base path of level 1 data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/mirac-p/l1/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    vers : str
        Indicates the mwr_pro output version number. Valid options: 'v00', and 'v01'. Default: 'v01'
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    if vers not in ['v00', 'v01']:
        raise ValueError("In import_hatpro_level1b_daterange, the argument 'vers' must be one of the" +
                            " following options: 'v00', 'v01'")

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")


    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_freq = 8          # inquired from level 1 data, number of frequencies

    # basic variables that should always be imported:
    mwr_time_keys = ['time', 'flag', 'ta', 'pa', 'hur']             # keys with time as coordinate
    mwr_freq_keys = ['freq_sb', 'freq_shift', 'tb_absolute_accuracy']   # keys with frequency as coordinate
    mwr_time_freq_keys = ['tb', 'tb_bias_estimate']                     # keys with frequency and time as coordinates

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    mwr_master_dict = dict()

    # max number of seconds: n_days*86400
    n_seconds = n_days*86400
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_seconds,), np.nan)
    for mtkab in mwr_time_freq_keys: mwr_master_dict[mtkab] = np.full((n_seconds, n_freq), np.nan)
    for mtkab in mwr_freq_keys: mwr_master_dict[mtkab] = np.full((n_freq,), np.nan)
    mwr_master_dict['tb_cov'] = np.full((n_freq,n_freq), np.nan)        # has got a special shape -> handled manually


    # cycle through all years, all months and days:
    time_index = 0  # this index (for lvl 1) will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    day_index = 0   # same as above, but only increases by 1 for each day
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("Working on MiRAC-P Level 1, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # list of v01 tb files with zenith scan:
        file_nc = sorted(glob.glob(day_path + f"*_l1_tb_{vers}_*.nc"))

        if len(file_nc) == 0:
            if verbose >= 2:
                warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for lvl_nc in file_nc: 
            mwr_dict = import_hatpro_level1b(lvl_nc, keys='all')

            n_time = len(mwr_dict['time'])
            cur_time_shape = mwr_dict['time'].shape

            # save to mwr_master_dict
            for mwr_key in mwr_time_keys:
                mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key]
            for mwr_key in mwr_time_freq_keys:
                mwr_master_dict[mwr_key][time_index:time_index + n_time,:] = mwr_dict[mwr_key]


        time_index = time_index + n_time
        day_index = day_index + 1

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # assign frequency to master dict:
        for mwr_key in mwr_freq_keys: mwr_master_dict[mwr_key] = mwr_dict[mwr_key]
        mwr_master_dict['tb_cov'] = mwr_dict['tb_cov']

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        time_freq_shape_old = mwr_master_dict['tb'].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_freq_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1, :]

    return mwr_master_dict


def import_mirac_level2a_daterange(
    path_data,
    date_start,
    date_end,
    which_retrieval='both',
    vers='v01',
    minute_avg=False,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the level 2a data time
    series of each day so that you'll have one dictionary, whose e.g. 'IWV' will contain the IWV
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Base path of level 2a data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/mirac-p/l2/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. 'both' will 
        load both. Default: 'both'
    vers : str
        Indicates the mwr_pro output version number. Valid options: 'v01'
    minute_avg : bool
        If True: averages over one minute are computed and returned instead of False when all
        data points are returned (more outliers, higher memory usage).
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'clwvi', 'lwp', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

        else:
                if which_retrieval == 'iwv':
                    which_retrieval = ['prw']
                elif which_retrieval == 'lwp':
                    which_retrieval = ['clwvi']
                elif which_retrieval == 'both':
                    which_retrieval = ['prw', 'clwvi']
                else:
                    raise ValueError("Argument '" + which_retrieval + "' not recognised. Please use one of the following options: " +
                        "'iwv' or 'prw' will load the " +
                        "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                        "'both' will load both. Default: 'both'")
                    

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")


    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_ret = 86          # inquired from level 2a data, number of available elevation angles in retrieval

    # basic variables that should always be imported:
    mwr_time_keys = ['time', 'azi', 'ele', 'flag', 'lat', 'lon', 'zsl']             # keys with time as coordinate

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    # e.g. level 2a and 2b have got the same time axis (according to pl_mk_nds.pro)
    # and azimuth and elevation angles.
    mwr_master_dict = dict()
    if minute_avg:  # max number of minutes: n_days*1440
        n_minutes = n_days*1440
        for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_minutes,), np.nan)

        if 'prw' in which_retrieval:
            mwr_master_dict['prw'] = np.full((n_minutes,), np.nan)

        if 'clwvi' in which_retrieval:
            mwr_master_dict['clwvi'] = np.full((n_minutes,), np.nan)

    else:           # max number of seconds: n_days*86400
        n_seconds = n_days*86400
        for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_seconds,), np.nan)

        if 'prw' in which_retrieval:
            mwr_master_dict['prw'] = np.full((n_seconds,), np.nan)

        if 'clwvi' in which_retrieval:
            mwr_master_dict['clwvi'] = np.full((n_seconds,), np.nan)


    # cycle through all years, all months and days:
    time_index = 0  # this index (for lvl 2a) will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    day_index = 0   # same as above, but only increases by 1 for each day
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("Working on MiRAC-P Level 2a, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # list of files:
        mirac_level2_nc = sorted(glob.glob(day_path + "*.nc"))
        # filter for i01 files:
        mirac_level2_nc = [lvl2_nc for lvl2_nc in mirac_level2_nc if vers in lvl2_nc]

        if len(mirac_level2_nc) == 0:
            if verbose >= 2:
                warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        # identify level 2a files:
        mirac_level2a_nc = []
        for lvl2_nc in mirac_level2_nc:
            for wr in which_retrieval:
                if wr + '_' in lvl2_nc:
                    mirac_level2a_nc.append(lvl2_nc)

        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for lvl2_nc in mirac_level2a_nc: 
            mwr_dict = import_mirac_level2a(lvl2_nc, minute_avg=minute_avg)

            n_time = len(mwr_dict['time'])
            cur_time_shape = mwr_dict['time'].shape

            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                mwr_key_shape = mwr_dict[mwr_key].shape
                if mwr_key_shape == cur_time_shape: # then the variable is on time axis:
                    mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key]

                elif len(mwr_dict[mwr_key]) == 1:
                    mwr_master_dict[mwr_key][day_index:day_index + 1] = mwr_dict[mwr_key]

                else:
                    raise RuntimeError("Something went wrong in the " +
                        "import_mirac_level2a_daterange routine. Unexpected MWR variable dimension. " + 
                        "The length of one used variable ('%s') of level 2a data "%(mwr_key) +
                            "neither equals the length of the time axis nor equals 1.")


        time_index = time_index + n_time
        day_index = day_index + 1

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        for mwr_key in mwr_master_dict.keys():
            if mwr_master_dict[mwr_key].shape == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]

    return mwr_master_dict


def import_hatpro_level1b(
    filename,
    keys='basic'):

    """
    Importing HATPRO level 1b (zenith TBs in K). Can also be used to import
    MiRAC-P level 1b (zenith TBs in K) data if it was processed with mwr_pro.

    Parameters:
    -----------
    filename : str
        Path and filename of mwr data (level2b).
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (specify keys='all') or import basic keys
        that the author considers most important (specify keys='basic')
        or leave this argument out.).
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'freq_sb', 'flag', 'tb']

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key '%s'. Key not found in level 1 file." % key)
        mwr_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)

    return mwr_dict


def import_hatpro_level2a(
    filename,
    keys='basic',
    minute_avg=False):

    """
    Importing HATPRO level 2a (integrated quantities, e.g. IWV, LWP).

    Parameters:
    -----------
    filename : str
        Path and filename of mwr data (level2a).
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    minute_avg : bool
        If True: averages over one minute are computed and returned instead of False when all
        data points are returned (more outliers, higher memory usage).
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'lat', 'lon', 'zsl', 'flag']
        if 'clwvi_' in filename:
            for add_key in ['clwvi', 'clwvi_err', 'clwvi_offset']: keys.append(add_key)
        if 'prw_' in filename:
            for add_key in ['prw', 'prw_err', 'prw_offset']: keys.append(add_key)

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in level 2a file." % key)
        mwr_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        time_shape_old = mwr_dict['time'].shape

        if minute_avg:
            # start the timer at the first time, when seconds is 00 (e.g. 09:31:00):
            time0 = mwr_dict['time'][0]     # start time in sec since 1970-01-01...
            dt_time0 = dt.datetime.utcfromtimestamp(mwr_dict['time'][0])
            dt_time0_Y = dt_time0.year
            dt_time0_M = dt_time0.month
            dt_time0_D = dt_time0.day
            dt_time0_s = dt_time0.second
            dt_time0_m = dt_time0.minute
            dt_time0_h = dt_time0.hour
            if dt_time0_s != 0:     # then the array mwr_dict['time'] does not start at second 0
                start_time = datetime_to_epochtime(dt.datetime(dt_time0_Y, dt_time0_M, dt_time0_D,
                                                    dt_time0_h, dt_time0_m+1, 0))
            else:
                start_time = time0

            if np.abs(start_time - time0) >= 60:
                print("Start time is far off the first time point in this file.")
                pdb.set_trace()
            # compute minute average
            n_minutes = int(np.ceil((mwr_dict['time'][-1] - start_time)/60))    # number of minutes
            min_time_idx_save = 0       # saves the last min_time_index value to speed up computation
            for min_count in range(n_minutes):
                # find time_idx when time is in the correct minute:
                # slower version:
                # # # # min_time_idx = np.argwhere((mwr_dict['time'] >= (start_time + min_count*60)) & 
                                # # # # (mwr_dict['time'] < (start_time + (min_count+1)*60))).flatten()
                # faster version:
                min_time_idx = np.argwhere((mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] >= (start_time + min_count*60)) & 
                                (mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] < (start_time + (min_count+1)*60))).flatten()

                # it may occur that no measurement exists in a certain minute-range. Then
                # we cannot compute the average but simply set that minute to nan.
                if len(min_time_idx) == 0:
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            mwr_dict[key][min_count] = np.nan
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            mwr_dict[key][min_count] = 99       # np.nan not possible because int is required
                else:
                    min_time_idx = min_time_idx + min_time_idx_save     # also belonging to the 'faster version'
                    min_time_idx_save = min_time_idx[-1]                # also belonging to the 'faster version'
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx])
                            else:
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx[0]:])
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            # find out how many entries show flag > 0. Then if it exceeds a threshold
                            # the whole minute is flagged. If threshold not exceeded, minute is not
                            # flagged!
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                if np.count_nonzero(mwr_dict[key][min_time_idx]) > len(min_time_idx)/10:    
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0
                            else:
                                if np.count_nonzero(mwr_dict[key][min_time_idx[0]:]) > len(min_time_idx)/10:
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0

            # truncate time arrays to reduce memory usage!
            for key in keys:
                if mwr_dict[key].shape == time_shape_old:
                    mwr_dict[key] = mwr_dict[key][:n_minutes]

    else:
        if minute_avg:
            raise KeyError("'time' must be included in the list of keys that will be imported for minute averages.")

    return mwr_dict


def import_hatpro_level2b(
    filename,
    keys='basic',
    minute_avg=False):

    """
    Importing HATPRO level 2b (zenith profiles, temperature or humidity 
    (in K or kg m^-3, respectively).

    Parameters:
    -----------
    filename : str
        Path and filename of mwr data (level2b).
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (specify keys='all') or import basic keys
        that the author considers most important (specify keys='basic')
        or leave this argument out.
    minute_avg : bool
        If True: averages over one minute are computed and returned. False: all
        data points are returned (more outliers, higher memory usage but may result in
        long computation time).
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'lat', 'lon', 'zsl', 'height', 'flag']
        if 'hua_' in filename:
            for add_key in ['hua']: keys.append(add_key)
        if 'ta_' in filename:
            for add_key in ['ta']: keys.append(add_key)

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key '%s'. Key not found in level 2b file." % key)
        mwr_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        time_shape_old = mwr_dict['time'].shape

        if minute_avg:
            # start the timer at the first time, when seconds is 00 (e.g. 09:31:00):
            time0 = mwr_dict['time'][0]     # start time in sec since 1970-01-01...
            dt_time0 = dt.datetime.utcfromtimestamp(mwr_dict['time'][0])
            dt_time0_Y = dt_time0.year
            dt_time0_M = dt_time0.month
            dt_time0_D = dt_time0.day
            dt_time0_s = dt_time0.second
            dt_time0_m = dt_time0.minute
            dt_time0_h = dt_time0.hour
            if dt_time0_s != 0:     # then the array mwr_dict['time'] does not start at second 0
                start_time = datetime_to_epochtime(dt.datetime(dt_time0_Y, dt_time0_M,
                                dt_time0_D, dt_time0_h, dt_time0_m+1, 0))
            else:
                start_time = time0

            if np.abs(start_time - time0) >= 60:
                print("Start time is far off the first time point in this file.")
                pdb.set_trace()
            # compute minute average
            n_minutes = int(np.ceil((mwr_dict['time'][-1] - start_time)/60))    # number of minutes
            min_time_idx_save = 0       # saves the last min_time_index value to speed up computation
            for min_count in range(n_minutes):
                # find time_idx when time is in the correct minute:
                # slower version:
                # # # # min_time_idx = np.argwhere((mwr_dict['time'] >= (start_time + min_count*60)) & 
                                # # # # (mwr_dict['time'] < (start_time + (min_count+1)*60))).flatten()
                # faster version:
                min_time_idx = np.argwhere((mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] >= (start_time + min_count*60)) & 
                                (mwr_dict['time'][min_time_idx_save:min_time_idx_save+180] < (start_time + (min_count+1)*60))).flatten()

                # it may occur that no measurement exists in a certain minute-range. Then
                # we cannot compute the average but simply set that minute to nan.
                if len(min_time_idx) == 0:
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            mwr_dict[key][min_count] = np.nan
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            mwr_dict[key][min_count] = 99       # np.nan not possible because int is required
                else:
                    min_time_idx = min_time_idx + min_time_idx_save     # also belonging to the 'faster version'
                    min_time_idx_save = min_time_idx[-1]                # also belonging to the 'faster version'
                    for key in keys:
                        if key == 'time':
                            mwr_dict['time'][min_count] = start_time + min_count*60
                        elif mwr_dict[key].shape == time_shape_old and key != 'flag':
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx])
                            else:
                                mwr_dict[key][min_count] = np.nanmean(mwr_dict[key][min_time_idx[0]:])
                        elif mwr_dict[key].shape == time_shape_old and key == 'flag':
                            # find out how many entries show flag > 0. Then if it exceeds a threshold
                            # the whole minute is flagged. If threshold not exceeded, minute is not
                            # flagged!
                            if min_time_idx[-1] < len(mwr_dict['time']):
                                if np.count_nonzero(mwr_dict[key][min_time_idx]) > len(min_time_idx)/10:    
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0
                            else:
                                if np.count_nonzero(mwr_dict[key][min_time_idx[0]:]) > len(min_time_idx)/10:
                                    # then there are too many flags set... so flag the whole minute:
                                    mwr_dict[key][min_count] = 99
                                else:
                                    mwr_dict[key][min_count] = 0

            # truncate time arrays to reduce memory usage!
            for key in keys:
                if mwr_dict[key].shape == time_shape_old:
                    mwr_dict[key] = mwr_dict[key][:n_minutes]

    else:
        if minute_avg:
            raise KeyError("'time' must be included in the list of keys that will be imported for minute averages.")


    return mwr_dict


def import_hatpro_level2c(
    filename,
    keys='basic'):

    """
    Importing HATPRO level 2c (boundary layer profiles, temperature (or humidity)
    (in K or kg m^-3, respectively).

    Parameters:
    -----------
    filename : str
        Path and filename of mwr data (level2c).
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (specify keys='all') or import basic keys
        that the author considers most important (specify keys='basic')
        or leave this argument out.
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'lat', 'lon', 'zsl', 'height', 'flag']
        if 'hua_' in filename:
            for add_key in ['hua']: keys.append(add_key)
        if 'ta_' in filename:
            for add_key in ['ta']: keys.append(add_key)

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key '%s'. Key not found in level 2c file." % key)
        mwr_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        time_shape_old = mwr_dict['time'].shape

    return mwr_dict


def import_hatpro_level1b_daterange(
    path_data,
    date_start,
    date_end,
    vers='v01',
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the level 1b TB time
    series of each day so that you'll have one dictionary, whose 'TB' will contain the IWV
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Base path of level 1 data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/hatpro/l1/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    vers : str
        Indicates the mwr_pro output version number. Valid options: 'i01', 'v00', and 'v01'. Default: 'v01'
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    if vers not in ['i01', 'v00', 'v01']:
        raise ValueError("In import_hatpro_level1b_daterange, the argument 'vers' must be one of the" +
                            " following options: 'i01', 'v00', 'v01'")

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")


    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_freq = 14         # inquired from level 1 data, number of frequencies

    # basic variables that should always be imported:
    mwr_time_keys = ['time', 'flag']                # keys with time as coordinate
    mwr_freq_keys = ['freq_sb']                     # keys with frequency as coordinate
    mwr_time_freq_keys = ['tb']                     # keys with frequency and time as coordinates

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    mwr_master_dict = dict()

    # max number of seconds: n_days*86400
    n_seconds = n_days*86400
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_seconds,), np.nan)
    for mtkab in mwr_time_freq_keys: mwr_master_dict[mtkab] = np.full((n_seconds, n_freq), np.nan)
    for mtkab in mwr_freq_keys: mwr_master_dict[mtkab] = np.full((n_freq,), np.nan)


    # cycle through all years, all months and days:
    time_index = 0  # this index (for lvl 1) will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    day_index = 0   # same as above, but only increases by 1 for each day
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("Working on HATPRO Level 1, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # list of v01 tb files with zenith scan:
        hatpro_nc = sorted(glob.glob(day_path + "*_mwr00_*_%s_*.nc"%vers))

        if len(hatpro_nc) == 0:
            if verbose >= 2:
                warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for lvl_nc in hatpro_nc: 
            mwr_dict = import_hatpro_level1b(lvl_nc)

            n_time = len(mwr_dict['time'])
            cur_time_shape = mwr_dict['time'].shape

            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                mwr_key_shape = mwr_dict[mwr_key].shape
                if mwr_key_shape == cur_time_shape: # then the variable is on time axis:
                    mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key]

                elif mwr_key_shape == (n_time, n_freq):
                    mwr_master_dict[mwr_key][time_index:time_index + n_time,:] = mwr_dict[mwr_key]

                elif mwr_key in mwr_freq_keys:  # will be handled after the loop
                    continue

                else:
                    raise RuntimeError("Something went wrong in the " +
                        "import_hatpro_level1b_daterange routine. Unexpected MWR variable dimension of '%s'. "%(mwr_key))


        time_index = time_index + n_time
        day_index = day_index + 1

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # assign frequency to master dict:
        for mwr_key in mwr_freq_keys: mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        time_freq_shape_old = mwr_master_dict['tb'].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_freq_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1, :]

    return mwr_master_dict


def import_hatpro_level1b_daterange_pangaea(
    path_data,
    date_start,
    date_end=None):

    """
    Runs through all days between a start and an end date. It concats the level 1b TB time
    series of each day so that you'll have one dictionary, whose 'TB' will contain the TB
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Path of level 1 (brightness temperature, TB) data. This directory contains daily files
        as netCDF.
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str or None
        If date_start is str: Marks the last day of the desired period. To be specified in 
        yyyy-mm-dd (e.g. 2021-01-14)!
    """

    def cut_vars(DS):
        DS = DS.drop_vars(['freq_sb', 'freq_shift', 'tb_absolute_accuracy', 'tb_cov'])
        return DS


    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # Identify files in the date range: First, load all into a list, then check which ones 
    # suit the daterange:
    mwr_dict = dict()
    sub_str = "_v01_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + "ioppol_tro_mwr00_l1_tb_v01_*.nc"))


    if type(date_start) == type(""):
        # extract day, month and year from start date:
        date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    else:
        # extract day, month and year from date_start:
        date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)


    # load data:
    DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested', preprocess=cut_vars)
    interesting_vars = ['time', 'flag', 'ta', 'pa', 'hur', 'tb', 'tb_bias_estimate', 'freq_sb', 'freq_shift',
                        'tb_absolute_accuracy', 'tb_cov']
    for vava in interesting_vars:
        if vava not in ['freq_sb', 'freq_shift', 'tb_absolute_accuracy', 'tb_cov']:
            mwr_dict[vava] = DS[vava].values.astype(np.float64)

    mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
    mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
    DS.close()

    DS = xr.open_dataset(files_filtered[0], decode_times=False)
    mwr_dict['freq_sb'] = DS.freq_sb.values.astype(np.float32)
    mwr_dict['freq_shift'] = DS.freq_shift.values.astype(np.float32)
    mwr_dict['tb_absolute_accuracy'] = DS.tb_absolute_accuracy.values.astype(np.float32)
    mwr_dict['tb_cov'] = DS.tb_cov.values.astype(np.float32)

    DS.close()
    del DS

    return mwr_dict


def import_hatpro_level1c_daterange_pangaea(
    path_data,
    date_start,
    date_end=None):

    """
    Runs through all days between a start and an end date. It concats the level 1c TB time
    series of each day so that you'll have one dictionary, whose 'TB' will contain the TB
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Path of level 1 (brightness temperature, TB) data. This directory contains daily files
        as netCDF.
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str or None
        If date_start is str: Marks the last day of the desired period. To be specified in 
        yyyy-mm-dd (e.g. 2021-01-14)!
    """

    def cut_vars(DS):
        DS = DS.drop_vars(['freq_sb', 'freq_shift', 'tb_absolute_accuracy', 'tb_cov', 'ele'])
        return DS


    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # Identify files in the date range: First, load all into a list, then check which ones 
    # suit the daterange:
    mwr_dict = dict()
    sub_str = "_v01_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + "ioppol_tro_mwrBL00_l1_tb_v01_*.nc"))


    if type(date_start) == type(""):
        # extract day, month and year from start date:
        date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    else:
        # extract day, month and year from date_start:
        date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)


    # load data:
    DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested', preprocess=cut_vars)
    interesting_vars = ['time', 'flag', 'tb', 'freq_sb', 'ele']
    for vava in interesting_vars:
        if vava in DS.variables:
            mwr_dict[vava] = DS[vava].values.astype(np.float64)

    mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
    mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
    DS.close()

    DS = xr.open_dataset(files_filtered[0], decode_times=False)
    mwr_dict['freq_sb'] = DS.freq_sb.values.astype(np.float32)
    mwr_dict['ele'] = DS.ele.values.astype(np.float32)

    DS.close()
    del DS

    return mwr_dict


def import_hatpro_level2a_daterange(
    path_data,
    date_start,
    date_end,
    which_retrieval='both',
    minute_avg=False,
    vers='v01',
    campaign='mosaic',
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the level 2a data time
    series of each day so that you'll have one dictionary, whose e.g. 'IWV' will contain the IWV
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Base path of level 2a data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/hatpro/l2/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. 'both' will 
        load both. Default: 'both'
    minute_avg : bool
        If True: averages over one minute are computed and returned instead of False when all
        data points are returned (more outliers, higher memory usage).
    vers : str
        Indicates the mwr_pro output version number. Valid options: 'i01', 'v00', and 'v01'. Default: 'v01'
    campaign : str
        Indicates which campaign is addressed. Options: 'mosaic', 'walsema'. Default: 'mosaic'
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    if vers not in ['i01', 'v00', 'v01']:
        raise ValueError("In import_hatpro_level2a_daterange, the argument 'vers' must be one of the" +
                            " following options: 'i01', 'v00', 'v01'")

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'clwvi', 'lwp', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

        else:
                if which_retrieval == 'iwv':
                    which_retrieval = ['prw']
                elif which_retrieval == 'lwp':
                    which_retrieval = ['clwvi']
                elif which_retrieval == 'both':
                    which_retrieval = ['prw', 'clwvi']
                else:
                    raise ValueError("Argument '" + which_retrieval + "' not recognised. Please use one of the following options: " +
                        "'iwv' or 'prw' will load the " +
                        "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                        "'both' will load both. Default: 'both'")
                    

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")


    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    if campaign == 'mosaic':
        n_ret = 86          # inquired from level 2a data, number of available elevation angles in retrieval
    elif campaign == 'walsema':
        n_ret = 2           # inquired from level 2a data, number of available elevation angles in retrieval

    # basic variables that should always be imported:
    mwr_time_keys = ['time', 'flag', 'lat', 'lon', 'zsl']               # keys with time as coordinate

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    # e.g. level 2a and 2b have got the same time axis (according to pl_mk_nds.pro)
    # and azimuth and elevation angles.
    mwr_master_dict = dict()
    if minute_avg:  # max number of minutes: n_days*1440
        n_minutes = n_days*1440
        for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_minutes,), np.nan)

        if 'prw' in which_retrieval:
            mwr_master_dict['prw'] = np.full((n_minutes,), np.nan)
            mwr_master_dict['prw_offset'] = np.full((n_minutes,), np.nan)
            mwr_master_dict['prw_err'] = np.full((n_ret,), np.nan)

        if 'clwvi' in which_retrieval:
            mwr_master_dict['clwvi'] = np.full((n_minutes,), np.nan)
            mwr_master_dict['clwvi_offset'] = np.full((n_minutes,), np.nan)
            mwr_master_dict['clwvi_err'] = np.full((n_ret,), np.nan)

    else:           # max number of seconds: n_days*86400
        n_seconds = n_days*86400
        for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_seconds,), np.nan)

        if 'prw' in which_retrieval:
            mwr_master_dict['prw'] = np.full((n_seconds,), np.nan)
            mwr_master_dict['prw_offset'] = np.full((n_seconds,), np.nan)
            mwr_master_dict['prw_err'] = np.full((n_ret,), np.nan)

        if 'clwvi' in which_retrieval:
            mwr_master_dict['clwvi'] = np.full((n_seconds,), np.nan)
            mwr_master_dict['clwvi_offset'] = np.full((n_seconds,), np.nan)
            mwr_master_dict['clwvi_err'] = np.full((n_ret,), np.nan)


    # cycle through all years, all months and days:
    time_index = 0  # this index (for lvl 2a) will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    day_index = 0   # same as above, but only increases by 1 for each day
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("Working on HATPRO Level 2a, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # list of files:
        hatpro_level2_nc = sorted(glob.glob(day_path + "*.nc"))
        # filter for v01 files:
        hatpro_level2_nc = [lvl2_nc for lvl2_nc in hatpro_level2_nc if vers in lvl2_nc]

        if len(hatpro_level2_nc) == 0:
            if verbose >= 2:
                warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        # identify level 2a files:
        hatpro_level2a_nc = []
        for lvl2_nc in hatpro_level2_nc:
            for wr in which_retrieval:
                if wr + '_' in lvl2_nc:
                    hatpro_level2a_nc.append(lvl2_nc)

        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for lvl2_nc in hatpro_level2a_nc: 
            mwr_dict = import_hatpro_level2a(lvl2_nc, minute_avg=minute_avg)

            n_time = len(mwr_dict['time'])
            cur_time_shape = mwr_dict['time'].shape

            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                mwr_key_shape = mwr_dict[mwr_key].shape
                if mwr_key_shape == cur_time_shape: # then the variable is on time axis:
                    mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key]

                elif mwr_key == 'prw_err' or mwr_key == 'clwvi_err':    # these variables are nret x 1 arrays
                    # mwr_master_dict[mwr_key][day_index:day_index + 1, :] = mwr_dict[mwr_key]          ## for the case that we leave _err a daily value
                    mwr_master_dict[mwr_key][:] = mwr_dict[mwr_key]

                elif mwr_dict[mwr_key].size == 1:
                    mwr_master_dict[mwr_key][day_index:day_index + 1] = mwr_dict[mwr_key]

                else:
                    raise RuntimeError("Something went wrong in the " +
                        "import_hatpro_level2a_daterange routine. Unexpected MWR variable dimension. " + 
                        "The length of one used variable ('%s') of level 2a data "%(mwr_key) +
                            "neither equals the length of the time axis nor equals 1.")


        time_index = time_index + n_time
        day_index = day_index + 1

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        for mwr_key in mwr_master_dict.keys():
            if mwr_master_dict[mwr_key].shape == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]

    return mwr_master_dict


def import_hatpro_level2b_daterange(
    path_data,
    date_start,
    date_end,
    which_retrieval='both',
    vers='v01',
    campaign='mosaic',
    around_radiosondes=True,
    path_radiosondes="",
    s_version='level_2',
    mwr_avg=0,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the level 2b data time
    series of each day so that you'll have one dictionary, whose e.g. 'ta' will contain the
    temperature profile for the entire date range period with samples around the radiosonde
    launch times or alternatively 4 samples per day at fixed times: 05, 11, 17 and 23 UTC.

    Parameters:
    -----------
    path_data : str
        Base path of level 2b data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/hatpro/l2/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'ta' or 'hus' will load either the
        temperature or the specific humidity profile. 'both' will load both. Default: 'both'
    vers : str, optional
        Indicates the mwr_pro output version number. Valid options: 'i01', 'v00', and 'v01'. Default: 'v01'
    campaign : str
        Indicates which campaign is addressed. Options: 'mosaic', 'walsema'. Default: 'mosaic'
    around_radiosondes : bool, optional
        If True, data will be limited to the time around radiosonde launches. If False, something else
        (e.g. around 4 times a day) might be done. Default: True
    path_radiosondes : str, optional
        Path to radiosonde data (Level 2). Default: ""
    s_version : str, optional
        Specifies the radiosonde version that is to be imported. Must be 'level_2' to work properly.
        Other versions have not been implemeted because they are considered to be inferior to level_2
        radiosondes.
    mwr_avg : int, optional
        If > 0, an average over mwr_avg seconds will be performed from sample_time to sample_time + 
        mwr_avg seconds. If == 0, no averaging will be performed.
    verbose : int, optional
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    if vers not in ['i01', 'v00', 'v01']:
        raise ValueError("In import_hatpro_level2b_daterange, the argument 'vers' must be one of the" +
                            " following options: 'i01', 'v00', 'v01'")

    if mwr_avg < 0:
        raise ValueError("mwr_avg must be an int >= 0.")
    elif type(mwr_avg) != type(1):
        raise TypeError("mwr_avg must be int.")

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
        raise TypeError("Argument 'which_retrieval' must be a string. Options: 'ta' or 'hus' will load either the " +
            "temperature or the specific humidity profile. 'both' will load both. Default: 'both'")

    elif which_retrieval not in ['ta', 'hus', 'both']:
        raise ValueError("Argument 'which_retrieval' must be one of the following options: 'ta' or 'hus' will load either the " +
            "temperature or the specific humidity profile. 'both' will load both. Default: 'both'")

    else:
        which_retrieval_dict = {'ta': ['ta'],
                                'hus': ['hus'],
                                'both': ['ta', 'hus']}
        level2b_dataID_dict = {'ta': ['ta'],
                                'hus': ['hua'],
                                'both': ['ta', 'hua']}
        level2b_dataID = level2b_dataID_dict[which_retrieval]           # to find correct file names
        which_retrieval = which_retrieval_dict[which_retrieval]


    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_ret = 1           # inquired from level 2b data, number of available elevation angles in retrieval
    n_hgt = 43          # inquired from level 2b data, number of vertical retrieval levels (height levels)

    # basic variables that should always be imported:
    if campaign == 'mosaic':
        mwr_time_keys = ['time', 'flag', 'lat', 'lon', 'zsl']               # keys with time as coordinate
    else:
        mwr_time_keys = ['time', 'flag']
    mwr_height_keys = ['height']                            # keys with height as coordinate

    # Create an array that includes the radiosonde launch times:
    if around_radiosondes:
        if not path_radiosondes:
            raise ValueError("If 'around_radiosondes' is True, the path to the radiosonde level 2 data ('pathradiosondes') " +
                                "must be given.")

        if campaign == 'mosaic' and s_version != 'level_2':
            raise ValueError("Radiosonde version 's_version' must be 'level_2' if around_radiosondes is True because " +
                                "for this version, the launch time is directly read from the filename. This has not " +
                                "been implemeted for other radiosonde versions ('mossonde', 'psYYMMDDwHH') because these " +
                                "are considered to be inferior.")

        elif campaign == 'walsema':
            add_files = sorted(glob.glob(path_radiosondes + "*.txt"))       # filenames only; filter path
            
            # load radiosonde data and identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            sonde_dict_temp = import_radiosondes_PS131_txt(add_files)
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            for ii, a_f in enumerate(add_files):
                ltt = dt.datetime.utcfromtimestamp(sonde_dict_temp[str(ii)]['launch_time'])
                # only save those that are in the considered period
                if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                    launch_times[kk] = ltt
                    kk += 1
            
            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

        else:
            add_files = sorted(glob.glob(path_radiosondes + "*.nc"))        # filenames only; filter path
            add_files = [os.path.basename(a_f) for a_f in add_files]
            
            # identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            for a_f in add_files:
                ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                # only save those that are in the considered period
                if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                    launch_times[kk] = ltt
                    kk += 1
            
            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

    else:
        # max number of samples: n_days*4
        sample_times = [5, 11, 17, 23]      # UTC on each day
        n_samp = len(sample_times)
        n_samp_tot = n_days*n_samp

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    # e.g. level 2b has got a time axis (according to pl_mk_nds.pro) for flag,
    # azimuth, elevation angles and the data.
    mwr_master_dict = dict()

    # save import keys for each retrieval option in a dict:
    import_keys = dict()
    mwr_time_height_keys = []
    for l2b_ID in level2b_dataID: mwr_time_height_keys.append(l2b_ID)

    if 'ta' in which_retrieval:
        mwr_master_dict['ta_err'] = np.full((n_hgt, n_ret), np.nan)

        # define the keys that will be imported via import_hatpro_level2b:
        import_keys['ta'] = (mwr_time_keys + mwr_height_keys +
                        ['ta', 'ta_err'])

    if 'hus' in which_retrieval:
        # here, we can only import and concat absolute humidity (hua) because
        # the conversion requires temperature and pressure
        mwr_master_dict['hua_err'] = np.full((n_hgt, n_ret), np.nan)

        # define the keys that will be imported via import_hatpro_level2b:
        import_keys['hua'] = (mwr_time_keys + mwr_height_keys +
                        ['hua', 'hua_err'])

    for mthk in mwr_time_height_keys: mwr_master_dict[mthk] = np.full((n_samp_tot, n_hgt), np.nan)
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_samp_tot,), np.nan)
    for mhk in mwr_height_keys: mwr_master_dict[mhk] = np.full((n_hgt,), np.nan)

    # cycle through all years, all months and days:
    time_index = 0  # this index (for lvl 2b) will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    sample_time_tolerance = 900     # sample time tolerance in seconds: mwr time must be within this
                                    # +/- tolerance of a sample_time to be accepted
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("Working on HATPRO Level 2b, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # specify sample times as time: sec since 1970-01-01 00:00:00 UTC:
        if around_radiosondes:
            now_date_date = now_date.date()
            sample_mask = np.full((n_samp_tot,), False)
            for kk, l_t in enumerate(launch_times):
                sample_mask[kk] = l_t.date() == now_date_date

            sample_times_t = sample_times[sample_mask]

        else:
            sample_times_t = np.asarray([datetime_to_epochtime(dt.datetime(yyyy, mm, dd, st, 0, 0)) for st in sample_times])

        # list of v01 files:
        hatpro_level2_nc = sorted(glob.glob(day_path + "*_%s_*.nc"%vers))
        if len(hatpro_level2_nc) == 0:
            if verbose >= 2:
                warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        # identify level 2b files:
        # also save the dataID into the list to access the correct keys to be imported (import_keys)
        # later on.
        hatpro_level2b_nc = []
        for lvl2_nc in hatpro_level2_nc:
            for dataID in level2b_dataID:
                # must avoid including the boundary layer scan
                if (dataID + '_' in lvl2_nc) and ('BL00_' not in lvl2_nc):
                    hatpro_level2b_nc.append([lvl2_nc, dataID])


        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for lvl2_nc in hatpro_level2b_nc:
            mwr_dict = import_hatpro_level2b(lvl2_nc[0], import_keys[lvl2_nc[1]])

            # it may occur that the whole day is flagged. If so, skip this file:
            if campaign == 'mosaic':

                if not np.any(mwr_dict['flag'] == 0):
                    n_samp_real = 0
                    continue

                    # remove values where flag > 0:
                    for mthk in mwr_time_height_keys:
                        if mthk in lvl2_nc[1]:
                            mwr_dict[mthk] = mwr_dict[mthk][mwr_dict['flag'] == 0,:]
                    for mtkab in mwr_time_keys:
                        if mtkab != 'flag':
                            mwr_dict[mtkab] = mwr_dict[mtkab][mwr_dict['flag'] == 0]
                    mwr_dict['flag'] = mwr_dict['flag'][mwr_dict['flag'] == 0]

                    # # # update the flag by taking the manually detected outliers into account:
                    # # # (not needed if v01 or later is used)
                    # mwr_dict['flag'] = outliers_per_eye(mwr_dict['flag'], mwr_dict['time'], instrument='hatpro')


            # find the time slice where the mwr time is closest to the sample_times.
            # The identified index must be within 15 minutes, otherwise it will be discarded
            # Furthermore, it needs to be respected, that the flag value must be 0 for that case.
            if mwr_avg == 0:
                sample_idx = []
                for st in sample_times_t:
                    idx = np.argmin(np.abs(mwr_dict['time'] - st))
                    if np.abs(mwr_dict['time'][idx] - st) < sample_time_tolerance:
                        sample_idx.append(idx)
                sample_idx = np.asarray(sample_idx)
                n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

            else:
                sample_idx = []
                for st in sample_times_t:
                    idx = np.where((mwr_dict['time'] >= st) & (mwr_dict['time'] <= st + mwr_avg))[0]
                    if len(idx) > 0:    # then an overlap has been found
                        sample_idx.append(idx)
                sample_idx = np.asarray(sample_idx)
                n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

            if n_samp_real == 0: continue

            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                mwr_key_shape = mwr_dict[mwr_key].shape

                if (mwr_key_shape == mwr_dict['time'].shape) and (mwr_key in mwr_time_keys):    # then the variable is on time axis:
                    if mwr_avg > 0:             # these values won't be averaged because they don't contain "data"
                        sample_idx_idx = [sii[0] for sii in sample_idx]
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real] = mwr_dict[mwr_key][sample_idx_idx]
                    
                    else:
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real] = mwr_dict[mwr_key][sample_idx]

                elif mwr_key == 'hua_err' or mwr_key == 'ta_err':   # these variables are n_hgt x n_ret arrays
                    mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

                elif mwr_key in mwr_height_keys:    # handled after the for loop
                    continue

                elif mwr_key in mwr_time_height_keys:
                    if mwr_avg > 0:
                        for k, sii in enumerate(sample_idx):
                            mwr_master_dict[mwr_key][time_index+k:time_index+k + 1,:] = np.nanmean(mwr_dict[mwr_key][sii,:], axis=0)
                    else:
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real,:] = mwr_dict[mwr_key][sample_idx,:]

                else:
                    raise RuntimeError("Something went wrong in the " +
                        "import_hatpro_level2b_daterange routine. Unexpected MWR variable dimension for " + mwr_key + ".")


        time_index = time_index + n_samp_real

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # save non height dependent variables to master dict:
        for mwr_key in mwr_height_keys: mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        time_height_shape_old = mwr_master_dict[mwr_time_height_keys[0]].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_height_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1, :]

    return mwr_master_dict


def import_hatpro_level2c_daterange(
    path_data,
    date_start,
    date_end,
    which_retrieval='both',
    vers='v01',
    campaign='mosaic',
    around_radiosondes=True,
    path_radiosondes="",
    s_version='level_2',
    sample_time_tolerance=1800,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the level 2c data time
    series of each day so that you'll have one dictionary, whose e.g. 'ta' will contain the
    temperature profile for the entire date range period with samples around the radiosonde
    launch times or alternatively 4 samples per day at fixed times: 05, 11, 17 and 23 UTC.

    Parameters:
    -----------
    path_data : str
        Base path of level 2c data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/hatpro/l2/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'ta' will load the temperature 
        profile (boundary layer scan). 'both' will also load temperature only because humidity profile
        boundary layer scan does not exist. Default: 'both'
    vers : str
        Indicates the mwr_pro output version number. Valid options: 'i01', 'v00', and 'v01'. Default: 'v01'
    campaign : str
        Indicates which campaign is addressed. Options: 'mosaic', 'walsema'. Default: 'mosaic'
    around_radiosondes : bool, optional
        If True, data will be limited to the time around radiosonde launches. If False, something else
        (e.g. around 4 times a day) might be done. Default: True
    path_radiosondes : str, optional
        Path to radiosonde data (Level 2). Default: ""
    s_version : str
        Specifies the radiosonde version that is to be imported. Must be 'level_2' to work properly.
        Other versions have not been implemeted because they are considered to be inferior to level_2
        radiosondes.
    sample_time_tolerance : int
        Integer indicating the sample time tolerance in seconds. This means that MWR time must be within
        this +/- tolerance of a sample time (i.e., radiosonde launch time) to be accepted.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    if vers not in ['i01', 'v00', 'v01']:
        raise ValueError("In import_hatpro_level2c_daterange, the argument 'vers' must be one of the" +
                            " following options: 'i01', 'v00', 'v01'")

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
        raise TypeError("Argument 'which_retrieval' must be a string. Options: 'ta' will load the temperature " +
        "profile (boundary layer scan). 'both' will also load temperature only because humidity profile" +
        "boundary layer scan does not exist. Default: 'both'")

    elif which_retrieval not in ['ta', 'hus', 'both']:
        raise ValueError("Argument 'which_retrieval' must be one of the following options: 'ta' will load the temperature " +
        "profile (boundary layer scan). 'both' will also load temperature only because humidity profile" +
        "boundary layer scan does not exist. Default: 'both'")

    else:
        which_retrieval_dict = {'ta': ['ta'],
                                'both': ['ta']}
        level2c_dataID_dict = {'ta': ['ta'],
                                'both': ['ta']}
        level2c_dataID = level2c_dataID_dict[which_retrieval]
        which_retrieval = which_retrieval_dict[which_retrieval]

    # check if around_radiosondes is the right type:
    if not isinstance(around_radiosondes, bool):
        raise TypeError("Argument 'around_radiosondes' must be either True or False (boolean type).")

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_ret = 1           # inquired from level 2c data, number of available elevation angles in retrieval
    n_hgt = 43          # inquired from level 2c data, number of vertical retrieval levels (height levels)

    # basic variables that should always be imported:
    if campaign == 'mosaic':
        mwr_time_keys = ['time', 'flag', 'lat', 'lon', 'zsl']               # keys with time as coordinate
    elif campaign == 'walsema':
        mwr_time_keys = ['time', 'flag']
    mwr_height_keys = ['height']                        # keys with height as coordinate

    # Create an array that includes the radiosonde launch times:
    if around_radiosondes:
        if not path_radiosondes:
            raise ValueError("If 'around_radiosondes' is True, the path to the radiosonde level 2 data ('pathradiosondes') " +
                                "must be given.")

        if campaign == 'mosaic' and s_version != 'level_2':
            raise ValueError("Radiosonde version 's_version' must be 'level_2' if around_radiosondes is True because " +
                                "for this version, the launch time is directly read from the filename. This has not " +
                                "been implemeted for other radiosonde versions ('mossonde', 'psYYMMDDwHH') because these " +
                                "are considered to be inferior.")

        elif campaign == 'walsema':
            add_files = sorted(glob.glob(path_radiosondes + "*.txt"))       # filenames only; filter path
            
            # load radiosonde data and identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            sonde_dict_temp = import_radiosondes_PS131_txt(add_files)
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            for ii, a_f in enumerate(add_files):
                ltt = dt.datetime.utcfromtimestamp(sonde_dict_temp[str(ii)]['launch_time'])
                # only save those that are in the considered period
                if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                    launch_times[kk] = ltt
                    kk += 1
            
            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

        else:
            add_files = sorted(glob.glob(path_radiosondes + "*.nc"))        # filenames only; filter path
            add_files = [os.path.basename(a_f) for a_f in add_files]
            
            # identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            for a_f in add_files:
                ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                # only save those that are in the considered period
                if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                    launch_times[kk] = ltt
                    kk += 1
            
            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

    else:
        # max number of samples: n_days*4
        sample_times = [5, 11, 17, 23]      # UTC on each day
        n_samp = len(sample_times)
        n_samp_tot = n_days*n_samp

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    # e.g. level 2c has got a time axis (according to pl_mk_nds.pro) for flag,
    # and the data.
    mwr_master_dict = dict()

    # save import keys for each retrieval option in a dict:
    import_keys = dict()
    mwr_time_height_keys = []
    for l2b_ID in level2c_dataID: mwr_time_height_keys.append(l2b_ID)

    if 'ta' in which_retrieval:
        mwr_master_dict['ta_err'] = np.full((n_hgt,), np.nan)

        # define the keys that will be imported via import_hatpro_level2b:
        import_keys['ta'] = (mwr_time_keys + mwr_height_keys +
                        ['ta', 'ta_err'])

    for mthk in mwr_time_height_keys: mwr_master_dict[mthk] = np.full((n_samp_tot, n_hgt), np.nan)
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_samp_tot,), np.nan)
    for mhk in mwr_height_keys: mwr_master_dict[mhk] = np.full((n_hgt,), np.nan)

    # cycle through all years, all months and days:
    time_index = 0  # this index (for lvl 2c) will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("Working on HATPRO Level 2c, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # specify sample times as time: sec since 1970-01-01 00:00:00 UTC:
        if around_radiosondes:
            now_date_date = now_date.date()
            sample_mask = np.full((n_samp_tot,), False)
            for kk, l_t in enumerate(launch_times):
                sample_mask[kk] = l_t.date() == now_date_date

            sample_times_t = sample_times[sample_mask]

        else:
            sample_times_t = np.asarray([datetime_to_epochtime(dt.datetime(yyyy, mm, dd, st, 0, 0)) for st in sample_times])

        # list of v01 files:
        hatpro_level2_nc = sorted(glob.glob(day_path + "*_%s_*.nc"%vers))

        if len(hatpro_level2_nc) == 0:
            if verbose >= 2:
                warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        # identify level 2c files:
        # also save the dataID into the list to access the correct keys to be imported (import_keys)
        # later on.
        hatpro_level2c_nc = []
        for lvl2_nc in hatpro_level2_nc:
            for dataID in level2c_dataID:
                # must include the boundary layer scan
                if (dataID + '_' in lvl2_nc) and ('BL00_' in lvl2_nc):
                    hatpro_level2c_nc.append([lvl2_nc, dataID])

        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for lvl2_nc in hatpro_level2c_nc:
            mwr_dict = import_hatpro_level2c(lvl2_nc[0], import_keys[lvl2_nc[1]])

            if campaign == 'mosaic':
                # it may occur that the whole day is flagged. If so, skip this file:
                if not np.any(mwr_dict['flag'] == 0):
                    n_samp_real = 0
                    continue

                # remove values where flag > 0:
                for mthk in mwr_time_height_keys: mwr_dict[mthk] = mwr_dict[mthk][mwr_dict['flag'] == 0,:]
                for mtkab in mwr_time_keys:
                    if mtkab != 'flag':
                        mwr_dict[mtkab] = mwr_dict[mtkab][mwr_dict['flag'] == 0]
                mwr_dict['flag'] = mwr_dict['flag'][mwr_dict['flag'] == 0]

                # # # update the flag by taking the manually detected outliers into account:
                # # # (not needed if v01 or later is used)
                # mwr_dict['flag'] = outliers_per_eye(mwr_dict['flag'], mwr_dict['time'], instrument='hatpro')


            # find the time slice where the mwr time is closest to the sample_times.
            # The identified index must be within 30 minutes, otherwise it will be discarded.
            # Furthermore, it needs to be respected, that the flag value must be 0 for that case.
            sample_idx = []
            for st in sample_times_t:
                idx = np.argmin(np.abs(mwr_dict['time'] - st))
                if np.abs(mwr_dict['time'][idx] - st) < sample_time_tolerance:
                    sample_idx.append(idx)
            sample_idx = np.asarray(sample_idx)
            n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

            if n_samp_real == 0: continue

            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                mwr_key_shape = mwr_dict[mwr_key].shape

                if (mwr_key_shape == mwr_dict['time'].shape) and (mwr_key in mwr_time_keys):    # then the variable is on time axis:
                    mwr_master_dict[mwr_key][time_index:time_index + n_samp_real] = mwr_dict[mwr_key][sample_idx]

                elif mwr_key == 'ta_err':   # these variables are n_hgt x n_ret arrays
                    mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

                elif mwr_key in mwr_height_keys: # handled after the for loop
                    continue

                elif mwr_key in mwr_time_height_keys:
                    mwr_master_dict[mwr_key][time_index:time_index + n_samp_real,:] = mwr_dict[mwr_key][sample_idx,:]

                else:
                    raise RuntimeError("Something went wrong in the " +
                        "import_hatpro_level2c_daterange routine. Unexpected MWR variable dimension for %s."%mwr_key)


        time_index = time_index + n_samp_real

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # save non time dependent variables in master dict
        for mwr_key in mwr_height_keys: mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        time_height_shape_old = mwr_master_dict[mwr_time_height_keys[0]].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_height_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1, :]

    return mwr_master_dict


def import_hatpro_level2a_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='both'):

    """
    Runs through all days between a start and an end date. It concats the level 2a data time
    series of each day so that you'll have one dictionary, whose e.g. 'IWV' will contain the IWV
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Path of level 2a data. 
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. 'both' will 
        load both. Default: 'both'
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'clwvi', 'lwp', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

        else:
                if which_retrieval == 'iwv':
                    which_retrieval = ['prw']
                elif which_retrieval == 'lwp':
                    which_retrieval = ['clwvi']
                elif which_retrieval == 'both':
                    which_retrieval = ['prw', 'clwvi']
                else:
                    raise ValueError("Argument '" + which_retrieval + "' not recognised. Please use one of the following options: " +
                        "'iwv' or 'prw' will load the " +
                        "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                        "'both' will load both. Default: 'both'")
                    

    # Identify files in the date range: First, load all into a list, then check
    # which ones suit the daterange:
    mwr_dict = dict()
    sub_str = "_v01_"
    l_sub_str = len(sub_str)
    if 'prw' in which_retrieval:
        files = sorted(glob.glob(path_data + "ioppol_tro_mwr00_l2_prw_v01_*.nc"))

        if type(date_start) == type(""):
            # extract day, month and year from start date:
            date_start_dt = dt.datetime.strptime(date_start, "%Y-%m-%d")
            date_end_dt = dt.datetime.strptime(date_end, "%Y-%m-%d")

            # run through list: identify where date is written and check if within date range:
            files_filtered = list()
            for file in files:
                ww = file.find(sub_str) + l_sub_str
                if file.find(sub_str) == -1: continue
                file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
                if file_date >= date_start_dt and file_date <= date_end_dt:
                    files_filtered.append(file)
        else:
            # extract day, month and year from date_start:
            date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

            # run through list: identify where date is written and check if within date range:
            files_filtered = list()
            for file in files:
                ww = file.find(sub_str) + l_sub_str
                if file.find(sub_str) == -1: continue
                file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
                if file_date in date_list:
                    files_filtered.append(file)


        # laod data:
        DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested')
        interesting_vars = ['time', 'flag', 'lat', 'lon', 'zsl', 'prw', 'prw_offset', 'prw_err']
        for vava in interesting_vars: 
            if vava != 'prw_err':
                mwr_dict[vava] = DS[vava].values.astype(np.float64)
            else:
                mwr_dict[vava] = DS[vava][0,:].values.astype(np.float64)
        mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
        mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        DS.close()

    if 'clwvi' in which_retrieval:
        files = sorted(glob.glob(path_data + "ioppol_tro_mwr00_l2_clwvi_v01_*.nc"))

        if type(date_start) == type(""):
            # extract day, month and year from start date:
            date_start_dt = dt.datetime.strptime(date_start, "%Y-%m-%d")
            date_end_dt = dt.datetime.strptime(date_end, "%Y-%m-%d")

            # run through list: identify where date is written and check if within date range:
            files_filtered = list()
            for file in files:
                ww = file.find(sub_str) + l_sub_str
                if file.find(sub_str) == -1: continue
                file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
                if file_date >= date_start_dt and file_date <= date_end_dt:
                    files_filtered.append(file)
        else:
            # extract day, month and year from date_start:
            date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

            # run through list: identify where date is written and check if within date range:
            files_filtered = list()
            for file in files:
                ww = file.find(sub_str) + l_sub_str
                if file.find(sub_str) == -1: continue
                file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
                if file_date in date_list:
                    files_filtered.append(file)


        # load data:
        DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested')
        if mwr_dict:
            interesting_vars = ['flag', 'clwvi', 'clwvi_offset_zeroing', 'clwvi_err']
            for vava in interesting_vars:
                if vava != 'clwvi_err':
                    mwr_dict[vava] = DS[vava].values.astype(np.float64)
                else:
                    mwr_dict[vava] = DS[vava][0,:].values.astype(np.float64)
            mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.

        else:
            interesting_vars = ['time', 'flag', 'lat', 'lon', 'zsl', 'clwvi', 'clwvi_offset_zeroing', 'clwvi_err']
            for vava in interesting_vars:
                if vava != 'clwvi_err':
                    mwr_dict[vava] = DS[vava].values.astype(np.float64)
                else:
                    mwr_dict[vava] = DS[vava][0,:].values.astype(np.float64)
            mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
            mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
        DS.close()

    return mwr_dict


def import_hatpro_level2b_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='both',
    around_radiosondes=True,
    path_radiosondes="",
    s_version='level_2',
    mwr_avg=0,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the level 2b data time
    series of each day so that you'll have one dictionary, whose e.g. 'ta' will contain the
    temperature profile for the entire date range period with samples around the radiosonde
    launch times or alternatively 4 samples per day at fixed times: 05, 11, 17 and 23 UTC.

    Parameters:
    -----------
    path_data : str
        Base path of level 2b data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/hatpro/l2/"
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case. The
        date list must be sorted in ascending order!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'ta' or 'hus' will load either the
        temperature or the specific humidity profile. 'both' will load both. Default: 'both'
    around_radiosondes : bool, optional
        If True, data will be limited to the time around radiosonde launches. If False, something else
        (e.g. around 4 times a day) might be done. Default: True
    path_radiosondes : str, optional
        Path to radiosonde data (Level 2). Default: ""
    s_version : str, optional
        Specifies the radiosonde version that is to be imported. Must be 'level_2' or 'level_3' to 
        work properly. Other versions have not been implemeted because they are considered to be 
        inferior to level_2 or level_3 radiosondes.
    mwr_avg : int, optional
        If > 0, an average over mwr_avg seconds will be performed from sample_time to sample_time + 
        mwr_avg seconds. If == 0, no averaging will be performed.
    verbose : int, optional
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")


    if mwr_avg < 0:
        raise ValueError("mwr_avg must be an int >= 0.")
    elif type(mwr_avg) != type(1):
        raise TypeError("mwr_avg must be int.")

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
        raise TypeError("Argument 'which_retrieval' must be a string. Options: 'ta' or 'hus' will load either the " +
            "temperature or the absolute humidity profile. 'both' will load both. Default: 'both'")

    elif which_retrieval not in ['ta', 'hus', 'both']:
        raise ValueError("Argument 'which_retrieval' must be one of the following options: 'ta' or 'hus' will load either the " +
            "temperature or the absolute humidity profile. 'both' will load both. Default: 'both'")

    else:
        which_retrieval_dict = {'ta': ['ta'],
                                'hus': ['hus'],
                                'both': ['ta', 'hus']}
        level2b_dataID_dict = {'ta': ['ta'],
                                'hus': ['hua'],
                                'both': ['ta', 'hua']}
        level2b_dataID = level2b_dataID_dict[which_retrieval]           # to find correct file names
        which_retrieval = which_retrieval_dict[which_retrieval]


    # extract day, month and year from start date:
    date_list = []
    if type(date_start) == type([]): 
        date_list = copy.deepcopy(date_start)
        date_start = date_start[0]
        date_list = [dt.datetime.strptime(dl, "%Y-%m-%d").date() for dl in date_list]
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_ret = 1           # inquired from level 2b data, number of available elevation angles in retrieval
    n_hgt = 43          # inquired from level 2b data, number of vertical retrieval levels (height levels)

    # basic variables that should always be imported:
    mwr_time_keys = ['time', 'flag', 'lat', 'lon', 'zsl']               # keys with time as coordinate
    mwr_height_keys = ['height']                            # keys with height as coordinate

    # Create an array that includes the radiosonde launch times:
    if around_radiosondes:
        if not path_radiosondes:
            raise ValueError("If 'around_radiosondes' is True, the path to the radiosonde level 2 data ('path_radiosondes') " +
                                "must be given.")

        if s_version not in ['level_2', 'level_3']:
            raise ValueError("Radiosonde version 's_version' must be 'level_2' if around_radiosondes is True because " +
                                "for this version, the launch time is directly read from the filename. This has not " +
                                "been implemeted for other radiosonde versions ('mossonde', 'psYYMMDDwHH') because these " +
                                "are considered to be inferior.")
        else:
            add_files = sorted(glob.glob(path_radiosondes + "*.nc"))        # filenames only; filter path
            add_files = [os.path.basename(a_f) for a_f in add_files]
            
            # identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            if date_list:   # then only consider dates within date_list
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt.date() in date_list:
                        launch_times[kk] = ltt
                        kk += 1
            else:
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                        launch_times[kk] = ltt
                        kk += 1
            
            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

    else:
        # max number of samples: n_days*4
        sample_times = [5, 11, 17, 23]      # UTC on each day
        n_samp = len(sample_times)
        n_samp_tot = n_days*n_samp

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    # e.g. level 2b has got a time axis (according to pl_mk_nds.pro) for flag,
    # azimuth, elevation angles and the data.
    mwr_master_dict = dict()

    # save import keys for each retrieval option in a dict:
    import_keys = dict()
    mwr_time_height_keys = []
    for l2b_ID in level2b_dataID: mwr_time_height_keys.append(l2b_ID)

    if 'ta' in which_retrieval:
        mwr_master_dict['ta_err'] = np.full((n_hgt, n_ret), np.nan)

        # define the keys that will be imported via import_hatpro_level2b:
        import_keys['ta'] = (mwr_time_keys + mwr_height_keys +
                        ['ta', 'ta_err'])

    if 'hus' in which_retrieval:
        # here, we can only import and concat absolute humidity (hua) because
        # the conversion requires temperature and pressure
        mwr_master_dict['hua_err'] = np.full((n_hgt, n_ret), np.nan)

        # define the keys that will be imported via import_hatpro_level2b:
        import_keys['hua'] = (mwr_time_keys + mwr_height_keys +
                        ['hua', 'hua_err'])

    for mthk in mwr_time_height_keys: mwr_master_dict[mthk] = np.full((n_samp_tot, n_hgt), np.nan)
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_samp_tot,), np.nan)
    for mhk in mwr_height_keys: mwr_master_dict[mhk] = np.full((n_hgt,), np.nan)


    # first list all available files and then reduce them to the specific date range and sampling:
    # list of v01 files:
    hatpro_level2_nc = sorted(glob.glob(path_data + "*_v01_*.nc"))
    if len(hatpro_level2_nc) == 0:
        if verbose >= 2:
            raise RuntimeError("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))


    # cycle through all years, all months and days:
    time_index = 0  # this index (for lvl 2b) will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    sample_time_tolerance = 900     # sample time tolerance in seconds: mwr time must be within this
                                    # +/- tolerance of a sample_time to be accepted


    if not date_list:
        date_list = (date_start + dt.timedelta(days=n) for n in range(n_days))
    else:
        date_list = [dt.datetime(dl_i.year, dl_i.month, dl_i.day) for dl_i in date_list]
    for now_date in date_list:

        if verbose >= 1: print("Working on HATPRO Level 2b, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day
        now_date_str = now_date.strftime("%Y%m%d")


        # specify sample times as time: sec since 1970-01-01 00:00:00 UTC:
        if around_radiosondes:
            now_date_date = now_date.date()
            sample_mask = np.full((n_samp_tot,), False)
            for kk, l_t in enumerate(launch_times):
                sample_mask[kk] = l_t.date() == now_date_date

            sample_times_t = sample_times[sample_mask]

        else:
            sample_times_t = np.asarray([datetime_to_epochtime(dt.datetime(yyyy, mm, dd, st, 0, 0)) for st in sample_times])


        # identify level 2b files:
        # also save the dataID into the list to access the correct keys to be imported (import_keys)
        # later on.
        hatpro_level2b_nc = []
        for lvl2_nc in hatpro_level2_nc:
            for dataID in level2b_dataID:
                # must avoid including the boundary layer scan
                if (dataID + '_' in lvl2_nc) and ('BL00_' not in lvl2_nc) and (now_date_str in lvl2_nc):
                    hatpro_level2b_nc.append([lvl2_nc, dataID])

        if len(hatpro_level2b_nc) == 0: continue


        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for lvl2_nc in hatpro_level2b_nc:
            mwr_dict = import_hatpro_level2b(lvl2_nc[0], import_keys[lvl2_nc[1]])

            # it may occur that the whole day is flagged. If so, skip this file:
            if not np.any(mwr_dict['flag'] == 0):
                n_samp_real = 0
                continue

            # remove values where flag > 0:
            for mthk in mwr_time_height_keys:
                if mthk in lvl2_nc[1]:
                    mwr_dict[mthk] = mwr_dict[mthk][(mwr_dict['flag'] == 0) | (mwr_dict['flag'] == 32),:]
            for mtkab in mwr_time_keys:
                if mtkab != 'flag':
                    mwr_dict[mtkab] = mwr_dict[mtkab][(mwr_dict['flag'] == 0) | (mwr_dict['flag'] == 32)]
            mwr_dict['flag'] = mwr_dict['flag'][(mwr_dict['flag'] == 0) | (mwr_dict['flag'] == 32)]


            # find the time slice where the mwr time is closest to the sample_times.
            # The identified index must be within 15 minutes, otherwise it will be discarded
            # Furthermore, it needs to be respected, that the flag value must be 0 for that case.
            if mwr_avg == 0:
                sample_idx = []
                for st in sample_times_t:
                    idx = np.argmin(np.abs(mwr_dict['time'] - st))
                    if np.abs(mwr_dict['time'][idx] - st) < sample_time_tolerance:
                        sample_idx.append(idx)
                sample_idx = np.asarray(sample_idx)
                n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

            else:
                sample_idx = []
                for st in sample_times_t:
                    idx = np.where((mwr_dict['time'] >= st) & (mwr_dict['time'] <= st + mwr_avg))[0]
                    if len(idx) > 0:    # then an overlap has been found
                        sample_idx.append(idx)
                n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

            if n_samp_real == 0: continue

            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                mwr_key_shape = mwr_dict[mwr_key].shape

                if (mwr_key_shape == mwr_dict['time'].shape) and (mwr_key in mwr_time_keys):    # then the variable is on time axis:
                    if mwr_avg > 0:             # these values won't be averaged because they don't contain "data"
                        sample_idx_idx = [sii[0] for sii in sample_idx]
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real] = mwr_dict[mwr_key][sample_idx_idx]
                    
                    else:
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real] = mwr_dict[mwr_key][sample_idx]

                elif mwr_key == 'hua_err' or mwr_key == 'ta_err':   # these variables are n_hgt x n_ret arrays
                    mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

                elif mwr_key in mwr_height_keys:    # handled after the for loop
                    continue

                elif mwr_key in mwr_time_height_keys:
                    if mwr_avg > 0:
                        for k, sii in enumerate(sample_idx):
                            mwr_master_dict[mwr_key][time_index+k:time_index+k + 1,:] = np.nanmean(mwr_dict[mwr_key][sii,:], axis=0)
                    else:
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real,:] = mwr_dict[mwr_key][sample_idx,:]

                else:
                    raise RuntimeError("Something went wrong in the " +
                        "import_hatpro_level2b_daterange routine. Unexpected MWR variable dimension for " + mwr_key + ".")


        time_index = time_index + n_samp_real

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # save non height dependent variables to master dict:
        for mwr_key in mwr_height_keys: mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        time_height_shape_old = mwr_master_dict[mwr_time_height_keys[0]].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_height_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1, :]

    return mwr_master_dict


def import_hatpro_level2c_daterange_pangaea(
    path_data,
    date_start,
    date_end,
    which_retrieval='both',
    around_radiosondes=True,
    path_radiosondes="",
    s_version='level_2',
    mwr_avg=0,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the level 2c data time
    series of each day so that you'll have one dictionary, whose e.g. 'ta' will contain the
    temperature profile for the entire date range period with samples around the radiosonde
    launch times or alternatively 4 samples per day at fixed times: 05, 11, 17 and 23 UTC.

    Parameters:
    -----------
    path_data : str
        Base path of level 2c data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/hatpro/l2/"
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case. The
        date list must be sorted in ascending order!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'ta' will load the temperature 
        profile (boundary layer scan). 'both' will also load temperature only because humidity profile
        boundary layer scan does not exist. Default: 'both'
    around_radiosondes : bool, optional
        If True, data will be limited to the time around radiosonde launches. If False, something else
        (e.g. around 4 times a day) might be done. Default: True
    path_radiosondes : str, optional
        Path to radiosonde data (Level 2). Default: ""
    s_version : str
        Specifies the radiosonde version that is to be imported. Must be 'level_2' to work properly.
        Other versions have not been implemeted because they are considered to be inferior to level_2
        radiosondes.
    mwr_avg : int, optional
        If > 0, an average over mwr_avg seconds will be performed from sample_time - mwr_avg to 
        sample_time + mwr_avg seconds. If == 0, no averaging will be performed.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
        raise TypeError("Argument 'which_retrieval' must be a string. Options: 'ta' will load the temperature " +
        "profile (boundary layer scan). 'both' will also load temperature only because humidity profile" +
        "boundary layer scan does not exist. Default: 'both'")

    elif which_retrieval not in ['ta', 'hus', 'both']:
        raise ValueError("Argument 'which_retrieval' must be one of the following options: 'ta' will load the temperature " +
        "profile (boundary layer scan). 'both' will also load temperature only because humidity profile" +
        "boundary layer scan does not exist. Default: 'both'")

    else:
        which_retrieval_dict = {'ta': ['ta'],
                                'both': ['ta']}
        level2c_dataID_dict = {'ta': ['ta'],
                                'both': ['ta']}
        level2c_dataID = level2c_dataID_dict[which_retrieval]
        which_retrieval = which_retrieval_dict[which_retrieval]

    if mwr_avg < 0:
        raise ValueError("mwr_avg must be an int >= 0.")
    elif type(mwr_avg) != type(1):
        raise TypeError("mwr_avg must be int.")

    # check if around_radiosondes is the right type:
    if not isinstance(around_radiosondes, bool):
        raise TypeError("Argument 'around_radiosondes' must be either True or False (boolean type).")

    # extract day, month and year from start date:
    date_list = []
    if type(date_start) == type([]): 
        date_list = copy.deepcopy(date_start)
        date_start = date_start[0]
        date_list = [dt.datetime.strptime(dl, "%Y-%m-%d").date() for dl in date_list]
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_ret = 1           # inquired from level 2c data, number of available elevation angles in retrieval
    n_hgt = 43          # inquired from level 2c data, number of vertical retrieval levels (height levels)

    # basic variables that should always be imported:
    mwr_time_keys = ['time', 'flag', 'lat', 'lon', 'zsl']               # keys with time as coordinate
    mwr_height_keys = ['height']                        # keys with height as coordinate

    # Create an array that includes the radiosonde launch times:
    if around_radiosondes:
        if not path_radiosondes:
            raise ValueError("If 'around_radiosondes' is True, the path to the radiosonde level 2 data ('pathradiosondes') " +
                                "must be given.")

        if s_version != 'level_2':
            raise ValueError("Radiosonde version 's_version' must be 'level_2' if around_radiosondes is True because " +
                                "for this version, the launch time is directly read from the filename. This has not " +
                                "been implemeted for other radiosonde versions ('mossonde', 'psYYMMDDwHH') because these " +
                                "are considered to be inferior.")
        else:
            add_files = sorted(glob.glob(path_radiosondes + "*.nc"))        # filenames only; filter path
            add_files = [os.path.basename(a_f) for a_f in add_files]
            
            # identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            for a_f in add_files:
                ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                # only save those that are in the considered period
                if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                    launch_times[kk] = ltt
                    kk += 1
            
            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

    else:
        # max number of samples: n_days*4
        sample_times = [5, 11, 17, 23]      # UTC on each day
        n_samp = len(sample_times)
        n_samp_tot = n_days*n_samp

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    # e.g. level 2c has got a time axis (according to pl_mk_nds.pro) for flag,
    # and the data.
    mwr_master_dict = dict()

    # save import keys for each retrieval option in a dict:
    import_keys = dict()
    mwr_time_height_keys = []
    for l2b_ID in level2c_dataID: mwr_time_height_keys.append(l2b_ID)

    if 'ta' in which_retrieval:
        mwr_master_dict['ta_err'] = np.full((n_hgt,), np.nan)

        # define the keys that will be imported via import_hatpro_level2b:
        import_keys['ta'] = (mwr_time_keys + mwr_height_keys +
                        ['ta', 'ta_err'])

    for mthk in mwr_time_height_keys: mwr_master_dict[mthk] = np.full((n_samp_tot, n_hgt), np.nan)
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_samp_tot,), np.nan)
    for mhk in mwr_height_keys: mwr_master_dict[mhk] = np.full((n_hgt,), np.nan)


    # first list all available files and then reduce them to the specific date range and sampling:
    # list of v01 files:
    hatpro_level2_nc = sorted(glob.glob(path_data + "*_v01_*.nc"))
    if len(hatpro_level2_nc) == 0:
        if verbose >= 2:
            raise RuntimeError("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))


    # cycle through all years, all months and days:
    time_index = 0  # this index (for lvl 2c) will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    sample_time_tolerance = 1800        # sample time tolerance in seconds: mwr time must be within this
                                        # +/- tolerance of a sample_time to be accepted


    if not date_list:
        date_list = (date_start + dt.timedelta(days=n) for n in range(n_days))
    else:
        date_list = [dt.datetime(dl_i.year, dl_i.month, dl_i.day) for dl_i in date_list]
    for now_date in date_list:

        if verbose >= 1: print("Working on HATPRO Level 2c, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day
        now_date_str = now_date.strftime("%Y%m%d")

        # specify sample times as time: sec since 1970-01-01 00:00:00 UTC:
        if around_radiosondes:
            now_date_date = now_date.date()
            sample_mask = np.full((n_samp_tot,), False)
            for kk, l_t in enumerate(launch_times):
                sample_mask[kk] = l_t.date() == now_date_date

            sample_times_t = sample_times[sample_mask]

        else:
            sample_times_t = np.asarray([datetime_to_epochtime(dt.datetime(yyyy, mm, dd, st, 0, 0)) for st in sample_times])


        # identify level 2c files:
        # also save the dataID into the list to access the correct keys to be imported (import_keys)
        # later on.
        hatpro_level2c_nc = []
        for lvl2_nc in hatpro_level2_nc:
            for dataID in level2c_dataID:
                # must include the boundary layer scan
                if (dataID + '_' in lvl2_nc) and ('BL00_' in lvl2_nc) and (now_date_str in lvl2_nc):
                    hatpro_level2c_nc.append([lvl2_nc, dataID])

        if len(hatpro_level2c_nc) == 0: continue


        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for lvl2_nc in hatpro_level2c_nc:
            mwr_dict = import_hatpro_level2c(lvl2_nc[0], import_keys[lvl2_nc[1]])

            # it may occur that the whole day is flagged. If so, skip this file:
            if not np.any(mwr_dict['flag'] == 0):
                n_samp_real = 0
                continue

            # remove values where flag > 0:
            for mthk in mwr_time_height_keys: mwr_dict[mthk] = mwr_dict[mthk][mwr_dict['flag'] == 0,:]
            for mtkab in mwr_time_keys:
                if mtkab != 'flag':
                    mwr_dict[mtkab] = mwr_dict[mtkab][mwr_dict['flag'] == 0]
            mwr_dict['flag'] = mwr_dict['flag'][mwr_dict['flag'] == 0]


            # # # update the flag by taking the manually detected outliers into account:
            # # # (not needed if v01 or later is used)
            # mwr_dict['flag'] = outliers_per_eye(mwr_dict['flag'], mwr_dict['time'], instrument='hatpro')


            # find the time slice where the mwr time is closest to the sample_times.
            # The identified index must be within 15 minutes, otherwise it will be discarded
            # Furthermore, it needs to be respected, that the flag value must be 0 for that case.
            if mwr_avg == 0:
                sample_idx = []
                for st in sample_times_t:
                    idx = np.argmin(np.abs(mwr_dict['time'] - st))
                    if np.abs(mwr_dict['time'][idx] - st) < sample_time_tolerance:
                        sample_idx.append(idx)
                sample_idx = np.asarray(sample_idx)
                n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

            else:
                sample_idx = []
                for st in sample_times_t:
                    idx = np.where((mwr_dict['time'] >= st - mwr_avg) & (mwr_dict['time'] <= st + mwr_avg))[0]
                    if len(idx) > 0:    # then an overlap has been found
                        sample_idx.append(idx)
                n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

            if n_samp_real == 0: continue


            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                mwr_key_shape = mwr_dict[mwr_key].shape

                if (mwr_key_shape == mwr_dict['time'].shape) and (mwr_key in mwr_time_keys):    # then the variable is on time axis:
                    if mwr_avg > 0:             # these values won't be averaged because they don't contain "data"
                        sample_idx_idx = [sii[0] for sii in sample_idx]
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real] = mwr_dict[mwr_key][sample_idx_idx]
                    
                    else:
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real] = mwr_dict[mwr_key][sample_idx]

                elif mwr_key == 'ta_err':   # these variables are n_hgt x n_ret arrays
                    mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

                elif mwr_key in mwr_height_keys: # handled after the for loop
                    continue

                elif mwr_key in mwr_time_height_keys:
                    if mwr_avg > 0:
                        for k, sii in enumerate(sample_idx):
                            mwr_master_dict[mwr_key][time_index+k:time_index+k + 1,:] = np.nanmean(mwr_dict[mwr_key][sii,:], axis=0)
                    else:
                        mwr_master_dict[mwr_key][time_index:time_index + n_samp_real,:] = mwr_dict[mwr_key][sample_idx,:]

                else:
                    raise RuntimeError("Something went wrong in the " +
                        "import_hatpro_level2c_daterange routine. Unexpected MWR variable dimension for %s."%mwr_key)


        time_index = time_index + n_samp_real

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # save non time dependent variables in master dict
        for mwr_key in mwr_height_keys: mwr_master_dict[mwr_key] = mwr_dict[mwr_key]

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        time_height_shape_old = mwr_master_dict[mwr_time_height_keys[0]].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_height_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1, :]

    return mwr_master_dict


def import_hatpro_level2a_daterange_xarray(
    path_data,
    date_start,
    date_end,
    which_retrieval='both'):

    """
    Imports all HATPRO level 2a data (IWV, LWP) between a start and an end date. The data will be returned
    in an xarray dataset.

    Parameters:
    -----------
    path_data : str
        Path of level 2a data. All files must lie in this folder.
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. 'both' will 
        load both. Default: 'both'
    """

    def remove_vars(DS):

        """
        Remove unnecessary variables from the dataset.
        """

        unwanted_vars = ['time_bnds', 'azi', 'ele', 'ele_ret', 'prw_offset', 'prw_off_zenith', 'prw_off_zenith_offset',
                        'clwvi_offset', 'clwvi_off_zenith', 'clwvi_off_zenith_offset']

        for var in unwanted_vars:
            if var in DS.data_vars:
                DS = DS.drop_vars(var)

        return DS

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'clwvi', 'lwp', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

        else:
                if which_retrieval == 'iwv':
                    which_retrieval = ['prw']
                elif which_retrieval == 'lwp':
                    which_retrieval = ['clwvi']
                elif which_retrieval == 'both':
                    which_retrieval = ['prw', 'clwvi']
                else:
                    raise ValueError("Argument '" + which_retrieval + "' not recognised. Please use one of the following options: " +
                        "'iwv' or 'prw' will load the " +
                        "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                        "'both' will load both. Default: 'both'")
                    

    # extract day, month and year from start date:
    date_start_dt = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end_dt = dt.datetime.strptime(date_end, "%Y-%m-%d")

    interesting_vars = ['time', 'flag', 'lat', 'lon', 'zsl']    # base of interesting vars to be kept in the dataset
    sub_str = "_v01_"               # substring to identify the date printed in the filenames
    l_sub_str = len(sub_str)

    # Import all files for the respective variable in the date range:
    if 'prw' in which_retrieval:
        files = sorted(glob.glob(path_data + "ioppol_tro_mwr00_l2_prw_v01_*.nc"))

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start_dt and file_date <= date_end_dt:
                files_filtered.append(file)

        DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested', preprocess=remove_vars)

        # 'repair' some variables:
        DS['flag'][np.isnan(DS['flag']).load()] = 0.
        DS['time'] = np.rint(DS['time']).astype(float).astype('datetime64[s]')


    if 'clwvi' in which_retrieval:
        raise RuntimeError("You have entered a construction site. Please leave immediately. Eltern haften für ihre Kinder.")

            # files = sorted(glob.glob(path_data + "ioppol_tro_mwr00_l2_clwvi_v01_*.nc"))

            # # run through list: identify where date is written and check if within date range:
            # files_filtered = list()
            # for file in files:
                # ww = file.find(sub_str) + l_sub_str
                # if file.find(sub_str) == -1: continue
                # file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
                # if file_date >= date_start_dt and file_date <= date_end_dt:
                    # files_filtered.append(file)


            # # load data:
            # DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested')
            # if mwr_dict:
                # interesting_vars = ['flag', 'clwvi', 'clwvi_offset', 'clwvi_err']
                # for vava in interesting_vars:
                    # if vava != 'clwvi_err':
                        # mwr_dict[vava] = DS[vava].values.astype(np.float64)
                    # else:
                        # mwr_dict[vava] = DS[vava][0,:].values.astype(np.float64)
                # mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.

            # else:
                # interesting_vars = ['time', 'flag', 'lat', 'lon', 'zsl', 'clwvi', 'clwvi_offset', 'clwvi_err']
                # for vava in interesting_vars:
                    # if vava != 'clwvi_err':
                        # mwr_dict[vava] = DS[vava].values.astype(np.float64)
                    # else:
                        # mwr_dict[vava] = DS[vava][0,:].values.astype(np.float64)
                # mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
                # mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
            # DS.close()

    return DS


def import_mirac_level1b_daterange_pangaea(
    path_data,
    date_start,
    date_end=None):

    """
    Runs through all days between a start and an end date. It concats the level 1b TB time
    series of each day so that you'll have one dictionary, whose 'TB' will contain the TB
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Path of level 1 (brightness temperature, TB) data. This directory contains daily files
        as netCDF.
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str or None
        If date_start is str: Marks the last day of the desired period. To be specified in 
        yyyy-mm-dd (e.g. 2021-01-14)!
    """

    def cut_vars(DS):
        DS = DS.drop_vars(['freq_sb', 'freq_shift', 'tb_absolute_accuracy', 'tb_cov'])
        return DS


    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # Identify files in the date range: First, load all into a list, then check which ones 
    # suit the daterange:
    mwr_dict = dict()
    sub_str = "_v01_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + "MOSAiC_uoc_lhumpro-243-340_l1_tb_v01_*.nc"))


    # extract day, month and year from start date:
    if type(date_start) == type(""):
        date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    else:
        # extract day, month and year from date_start:
        date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)


    # load data:
    DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested', preprocess=cut_vars)
    interesting_vars = ['time', 'flag', 'ta', 'pa', 'hur', 'tb', 'tb_bias_estimate', 'freq_sb', 'freq_shift',
                        'tb_absolute_accuracy', 'tb_cov']
    for vava in interesting_vars:
        if vava not in ['freq_sb', 'freq_shift', 'tb_absolute_accuracy', 'tb_cov']:
            mwr_dict[vava] = DS[vava].values.astype(np.float64)

    mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
    mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
    DS = DS.close()

    DS = xr.open_dataset(files_filtered[0], decode_times=False)
    mwr_dict['freq_sb'] = DS.freq_sb.values.astype(np.float32)
    mwr_dict['freq_shift'] = DS.freq_shift.values.astype(np.float32)
    mwr_dict['tb_absolute_accuracy'] = DS.tb_absolute_accuracy.values.astype(np.float32)
    mwr_dict['tb_cov'] = DS.tb_cov.values.astype(np.float32)

    DS = DS.close()
    del DS

    return mwr_dict


def import_mirac_level2a_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='both'):

    """
    Runs through all days between a start and an end date. It concats the level 2a data time
    series of each day so that you'll have one dictionary, whose e.g. 'IWV' will contain the IWV
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Path of level 2a data. 
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'both' will also load integrated water vapour only. Default: 'both'
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'both' will also load integrated water vapour only. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'both' will also load integrated water vapour only. Default: 'both'")

        else:
                if which_retrieval == 'iwv':
                    which_retrieval = ['prw']
                elif which_retrieval == 'both':
                    which_retrieval = ['prw']
                    

    # Identify files in the date range: First, load all into a list, then check
    # which ones suit the daterange:
    mwr_dict = dict()
    sub_str = "_v01_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + "MOSAiC_uoc_lhumpro-243-340_l2_prw_v01_*.nc"))

    if type(date_start) == type(""):
        # extract day, month and year from start date:
        date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    else:
        # extract day, month and year from date_start:
        date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)


    # laod data:
    DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested')
    interesting_vars = ['time', 'flag', 'lat', 'lon', 'zsl', 'prw']
    for vava in interesting_vars: mwr_dict[vava] = DS[vava].values.astype(np.float64)
    mwr_dict['flag'][np.isnan(mwr_dict['flag'])] = 0.
    mwr_dict['time'] = np.rint(mwr_dict['time']).astype(float)
    DS.close()

    return mwr_dict


def import_mirac_level2a_daterange_xarray(
    path_data,
    date_start,
    date_end,
    which_retrieval='both'):

    """
    Imports all MiRAC-P level 2a data (IWV) between a start and an end date. The data will be returned
    in an xarray dataset.

    Parameters:
    -----------
    path_data : str
        Path of level 2a data. All files must be in this folder.
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'both' will also load integrated water vapour only. Default: 'both'
    """

    def remove_vars(DS):

        """
        Remove unnecessary variables from the dataset.
        """

        unwanted_vars = ['time_bnds', 'azi', 'ele', 'ele_ret', 'prw_offset', 'prw_off_zenith', 'prw_off_zenith_offset']

        for var in unwanted_vars:
            if var in DS.data_vars:
                DS = DS.drop_vars(var)

        return DS

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'both' will also load integrated water vapour only. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'both' will also load integrated water vapour only. Default: 'both'")

        else:
                if which_retrieval == 'iwv':
                    which_retrieval = ['prw']
                elif which_retrieval == 'both':
                    which_retrieval = ['prw']
                    

    # extract day, month and year from start date:
    date_start_dt = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end_dt = dt.datetime.strptime(date_end, "%Y-%m-%d")

    interesting_vars = ['time', 'flag', 'lat', 'lon', 'zsl']    # base of interesting vars to be kept in the dataset
    sub_str = "_v01_"               # substring to identify the date printed in the filenames
    l_sub_str = len(sub_str)

    # Import all files for the respective variable in the date range:
    files = sorted(glob.glob(path_data + "MOSAiC_uoc_lhumpro-243-340_l2_prw_v01_*.nc"))

    # run through list: identify where date is written and check if within date range:
    files_filtered = list()
    for file in files:
        ww = file.find(sub_str) + l_sub_str
        if file.find(sub_str) == -1: continue
        file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
        if file_date >= date_start_dt and file_date <= date_end_dt:
            files_filtered.append(file)

    DS = xr.open_mfdataset(files_filtered, decode_times=False, concat_dim='time', combine='nested', preprocess=remove_vars)

    # 'repair' some variables:
    DS['flag'][np.isnan(DS['flag']).load()] = 0.
    DS['time'] = np.rint(DS['time']).astype(float).astype('datetime64[s]')

    return DS


def import_single_mossonde_curM1(
    filename,
    keys='all',
    height_grid=np.array([]),
    verbose=0):

    """
    Imports radiosonde data 'mossonde-curM1' of a single file. Converts 'time' to seconds
    since 1970-01-01 00:00:00 UTC and interpolates to a height grid with 5 m resolution
    from 0 to 15000 m. Furthermore, absolute and specific humidity is computed and saved
    into the dictionary that will be returned.

    Parameters:
    -----------
    filename : str
        Name (including path) of radiosonde data file.
    keys : list of str or str, optional
        This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
        Default: 'all'
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    """
        Loaded values are imported in the following units:
        temp: in K
        pres: in hPa
        rh: in [1]
        dewp: in K
        wdir: in degree
        wspeed: in m s^-1
        geopheight: in m
        time: will be converted to sec since 1970-01-01 00:00:00 UTC here!
    """

    file_nc = nc.Dataset(filename)

    if (not isinstance(keys, str)) and (not isinstance(keys, list)):
        raise TypeError("Argument 'key' must be a list of strings or 'all'.")

    if keys == 'all':
        keys = file_nc.variables.keys()

    sonde_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in radiosonde file." % key)

        sonde_dict[key] = np.asarray(file_nc.variables[key])

        if key == 'time':   # convert to sec since 1970-01-01 00:00:00 UTC (USE FLOAT64)
            sonde_dict['time'] = (np.float64(datetime_to_epochtime(dt.datetime.strptime(
                                    file_nc.variables[key].units[14:], "%Y-%m-%dT%H:%M"))) +
                                    sonde_dict[key].astype(np.float64))
        if key == 'pres': sonde_dict[key] = 100*sonde_dict[key]

    sonde_dict['rho_v'] = convert_rh_to_abshum(sonde_dict['temp'], sonde_dict['rh'])
    sonde_dict['q'] = convert_rh_to_spechum(sonde_dict['temp'], sonde_dict['pres'], sonde_dict['rh'])

    if 'geopheight' in keys:
        sonde_dict['height'] = sonde_dict['geopheight']

    # interpolate to new height grid:
    if len(height_grid) == 0:
        height_grid = np.arange(0,15001,5)      # in m
    keys = [*keys]      # converts dict_keys to a list
    for addkey in ['rho_v', 'q', 'height']: keys.append(addkey)
    for key in keys:
        if sonde_dict[key].shape == sonde_dict['time'].shape:
            if key not in ['time', 'geopheight']:
                sonde_dict[key + "_ip"] = np.interp(height_grid, sonde_dict['geopheight'], sonde_dict[key])
            elif key == 'geopheight':
                sonde_dict[key + "_ip"] = height_grid

    sonde_dict['launch_time'] = sonde_dict['time'][0]
                
    return sonde_dict


def import_single_psYYMMDD_wHH_sonde(
    filename,
    keys='all',
    height_grid=np.array([]),
    verbose=0):

    """
    Imports radiosonde data 'psYYMMDD.wHH.nc' of a single file. Converts 'time' to seconds
    since 1970-01-01 00:00:00 UTC and interpolates to a height grid with 5 m resolution
    from 0 to 15000 m. Furthermore, absolute and specific humidity is computed and saved
    into the dictionary that will be returned.

    Parameters:
    -----------
    filename : str
        Name (including path) of radiosonde data file.
    keys : list of str or str, optional
        This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
        Specifying 'basic' will load the variables the author consideres most useful for his current
        analysis.
        Default: 'all'
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    """
        Loaded values are imported in the following units:
        T: in K
        p: in Pa
        RH: in %, will be converted to [0-1]
        DD: in degree
        FF: in m s^-1
        GeopHgt: in m
        GPSHgt : in m
        qv : in kg kg^-1 (water vapor specific humidity)
        time: will be converted to sec since 1970-01-01 00:00:00 UTC here!
    """

    file_nc = nc.Dataset(filename)

    if (not isinstance(keys, str)) and (not isinstance(keys, list)):
        raise TypeError("Argument 'key' must be a list of strings or 'all'.")

    if keys == 'all':
        keys = file_nc.variables.keys()
    elif keys == 'basic':
        keys = ['time', 'T', 'p', 'RH', 'qv', 'GeopHgt', 'GPSHgt']

    sonde_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in radiosonde file." % key)

        sonde_dict[key] = np.asarray(file_nc.variables[key])
        
        if len(sonde_dict[key]) == 0: return None

        if key == 'time':   # convert to sec since 1970-01-01 00:00:00 UTC (USE FLOAT64)
            # find launch time based on filename:
            time_base_fn = filename[-13:-3]
            yyyy = 2000 + int(time_base_fn[:2])
            mm = int(time_base_fn[2:4])
            dd = int(time_base_fn[4:6])
            hh = int(time_base_fn[8:10])
            rs_date_dt = dt.datetime(yyyy,mm,dd,hh)
            sonde_dict['time'] = np.float64(datetime_to_epochtime(rs_date_dt)) + sonde_dict[key].astype(np.float64)
        if key == 'RH':
            sonde_dict[key] = 0.01*sonde_dict[key]
        if key in ['Lat', 'Lon']:   # only interested in the first lat, lon position
            sonde_dict[key] = sonde_dict[key][0]

    # if len(sonde_dict['time']) > 0:
    sonde_dict['rho_v'] = convert_rh_to_abshum(sonde_dict['T'], sonde_dict['RH'])


    # interpolate to new height grid:
    if len(height_grid) == 0:
        height_grid = np.arange(0,15001,5)      # in m
    keys = [*keys]      # converts dict_keys to a list
    for addkey in ['rho_v']: keys.append(addkey)
    for key in keys:
        if sonde_dict[key].shape == sonde_dict['time'].shape:
            if key not in ['time', 'GeopHgt']:
                sonde_dict[key + "_ip"] = np.interp(height_grid, sonde_dict['GeopHgt'], sonde_dict[key])
            elif key == 'GeopHgt':
                sonde_dict[key + "_ip"] = height_grid

    sonde_dict['launch_time'] = sonde_dict['time'][0]

    # else: # then the file is empty
        # for geokey in ['Lat', 'Lon']: sonde_dict[geokey] = -99
        # sonde_dict['rho_v'] = np.asarray([])
        # for addkey in ['rho_v']: keys.append(addkey)
        # for key in keys:
            # if sonde_dict[key].shape == sonde_dict['time'].shape:
                # if key != 'time':
                    # sonde_dict[key + "_ip"] = np.array([])

        # sonde_dict['launch_time'] = np.nan

    # Renaming variables: ['Lat', 'Lon', 'p', 'T', 'RH', 'GeopHgt', 'qv', 'time']
    renaming = {'T': 'temp',    'p': 'pres',    'RH': 'rh',
                'DD': 'wdir',   'FF': 'wspeed', 'GeopHgt': 'height',
                'Lat': 'lat',   'Lon': 'lon',   'qv': 'q',
                'T_ip': 'temp_ip', 'p_ip': 'pres_ip', 'RH_ip': 'rh_ip',
                'GeopHgt_ip': 'height_ip', 'qv_ip': 'q_ip'}
    for ren_key in renaming.keys():
        if ren_key in sonde_dict.keys():
            sonde_dict[renaming[ren_key]] = sonde_dict[ren_key]
        
    return sonde_dict


def import_single_PS122_mosaic_radiosonde_level2(
    filename,
    keys='all',
    height_grid=np.array([]),
    ip_type='lin',
    verbose=0):

    """
    Imports single level 2 radiosonde data created with PANGAEA_tab_to_nc.py 
    ('PS122_mosaic_radiosonde_level2_yyyymmdd_hhmmssZ.nc'). Converts to SI units
    and interpolates to a height grid with 5 m resolution from 0 to 15000 m. 

    Parameters:
    -----------
    filename : str
        Name (including path) of radiosonde data file.
    keys : list of str or str, optional
        This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
        Specifying 'basic' will load the variables the author consideres most useful for his current
        analysis.
        Default: 'all'
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    ip_type : str
        String indicating the interpolation type. Option: 'lin' for linear interpolation using 
        np.interp; 'avg' for using interp_w_avg, which averages data over layers centered on the
        height_grid.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    """
        Loaded values are imported in the following units:
        T: in deg C, will be converted to K
        P: in hPa, will be converted to Pa
        RH: in %, will be converted to [0-1]
        Altitude: in m
        q: in kg kg^-1 (water vapor specific humidity)
        time: in sec since 1970-01-01 00:00:00 UTC
    """

    file_nc = nc.Dataset(filename)

    if (not isinstance(keys, str)) and (not isinstance(keys, list)):
        raise TypeError("Argument 'key' must be a list of strings or 'all'.")

    if keys == 'all':
        keys = file_nc.variables.keys()
    elif keys == 'basic':
        keys = ['time', 'T', 'P', 'RH', 'q', 'Altitude']

    sonde_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in radiosonde file." % key)

        sonde_dict[key] = np.asarray(file_nc.variables[key])
        if key != "IWV" and len(sonde_dict[key]) == 0: # 'and': second condition only evaluated if first condition True
            return None

        if key in ['Latitude', 'Longitude']:    # only interested in the first lat, lon position
            sonde_dict[key] = sonde_dict[key][0]
        if key == 'IWV':
            sonde_dict[key] = np.float64(sonde_dict[key])

    # convert units:
    if 'RH' in keys:    # from percent to [0, 1]
        sonde_dict['RH'] = sonde_dict['RH']*0.01
    if 'T' in keys:     # from deg C to K
        sonde_dict['T'] = sonde_dict['T'] + 273.15
    if 'P' in keys:     # from hPa to Pa
        sonde_dict['P'] = sonde_dict['P']*100
    if 'time' in keys:  # from int64 to float64
        sonde_dict['time'] = np.float64(sonde_dict['time'])
        sonde_dict['launch_time'] = sonde_dict['time'][0]

    # interpolate to new height grid:
    if len(height_grid) == 0:
        height_grid = np.arange(0,15001,5)      # in m
    keys = [*keys]      # converts dict_keys to a list
    for key in keys:
        if sonde_dict[key].shape == sonde_dict['time'].shape:
            if key not in ['time', 'Latitude', 'Longitude', 'ETIM', 'Altitude']:
                sonde_dict[key + "_ip"] = np.interp(height_grid, sonde_dict['Altitude'], sonde_dict[key], right=np.nan)
            elif key == 'Altitude':
                sonde_dict[key + "_ip"] = height_grid


    # Renaming variables: ['Lat', 'Lon', 'p', 'T', 'RH', 'GeopHgt', 'qv', 'time', ...]
    renaming = {'T': 'temp',    'P': 'pres',    'RH': 'rh',
                'Altitude': 'height', 'h_geom': 'height_geom',
                'Latitude': 'lat',  'Longitude': 'lon',
                'T_ip': 'temp_ip', 'P_ip': 'pres_ip', 'RH_ip': 'rh_ip',
                'Altitude_ip': 'height_ip', 'h_geom_ip': 'height_geom_ip',
                'IWV': 'iwv'}
    for ren_key in renaming.keys():
        if ren_key in sonde_dict.keys():
            sonde_dict[renaming[ren_key]] = sonde_dict[ren_key]

    # height check: how high does the data reach:
    sonde_dict['height_check'] = sonde_dict['height'][-1]

    return sonde_dict


def import_single_PS122_mosaic_radiosonde_GRUAN(
    filename,
    keys='all',
    height_grid=np.array([]),
    ip_type='lin',
    verbose=0):

    """
    Imports single level 3 GRUAN radiosonde data created with PANGAEA_tab_to_nc.py 
    ('PS122_mosaic_radiosonde_level3_GRUAN_yyyymmdd_hhmmssZ.nc'). Converts to SI units
    and interpolates to a height grid with 5 m resolution from 0 to 15000 m. 

    Parameters:
    -----------
    filename : str
        Name (including path) of radiosonde data file.
    keys : list of str or str, optional
        This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
        Specifying 'basic' will load the variables the author consideres most useful for his current
        analysis.
        Default: 'all'
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    ip_type : str
        String indicating the interpolation type. Option: 'lin' for linear interpolation using 
        np.interp; 'avg' for using interp_w_avg, which averages data over layers centered on the
        height_grid.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    """
        Loaded values are imported in the following units:
        T: in K
        P: in hPa, will be converted to Pa
        RH: in %, will be converted to [0-1]
        h_geop: in m
        q: in kg kg^-1 (water vapor specific humidity)
        time: in sec since 1970-01-01 00:00:00 UTC
    """

    DS = xr.open_dataset(filename)

    if (not isinstance(keys, str)) and (not isinstance(keys, list)):
        raise TypeError("Argument 'key' must be a list of strings or 'all'.")

    if keys == 'all':
        keys = [*DS.variables]
    elif keys == 'basic':
        keys = ['time', 'T', 'P', 'RH', 'q', 'h_geop']

    sonde_dict = dict()
    for key in keys:
        if not key in DS.variables:
            raise KeyError(f"I have no memory of this key: {key}. Key not found in radiosonde file.")

        sonde_dict[key] = DS[key].values
        if key != "IWV" and len(sonde_dict[key]) == 0: # 'and': second condition only evaluated if first condition True
            pdb.set_trace()
            return None

        if key in ['Latitude', 'Longitude']:    # only interested in the first lat, lon position
            sonde_dict[key] = sonde_dict[key][0]
        if key == 'IWV':
            sonde_dict[key] = np.float64(sonde_dict[key])

    # convert units:
    if 'RH' in keys:    # from percent to [0, 1]
        sonde_dict['RH'] = sonde_dict['RH']*0.01
    if 'P' in keys:     # from hPa to Pa
        sonde_dict['P'] = sonde_dict['P']*100
    if 'time' in keys:  # from int64 to float64
        sonde_dict['time'] = sonde_dict['time'].astype('datetime64[s]').astype('float64')
        sonde_dict['launch_time'] = sonde_dict['time'][0]


    # interpolate to new height grid:
    if len(height_grid) == 0:
        height_grid = np.arange(0,15001,5)      # in m
    keys = [*keys]      # converts dict_keys to a list
    for key in keys:
        if sonde_dict[key].shape == sonde_dict['time'].shape:
            if key not in ['time', 'Latitude', 'Longitude', 'ETIM', 'h_geop']:
                sonde_dict[key + "_ip"] = np.interp(height_grid, sonde_dict['h_geop'], sonde_dict[key], right=np.nan)
            elif key == 'h_geop':
                sonde_dict[key + "_ip"] = height_grid


    # Renaming variables: ['Lat', 'Lon', 'p', 'T', 'RH', 'GeopHgt', 'qv', 'time', ...]
    renaming = {'T': 'temp', 'P': 'pres', 'RH': 'rh',
                'h_geop': 'height',
                'Latitude': 'lat',  'Longitude': 'lon',
                'T_ip': 'temp_ip', 'P_ip': 'pres_ip', 'RH_ip': 'rh_ip',
                'h_geop_ip': 'height_ip',
                'IWV': 'iwv'}
    for ren_key in renaming.keys():
        if ren_key in sonde_dict.keys():
            sonde_dict[renaming[ren_key]] = sonde_dict[ren_key]

    # height check: how high does the data reach:
    sonde_dict['height_check'] = sonde_dict['height'][-1]

    DS = DS.close()

    return sonde_dict


def import_single_PS122_mosaic_radiosonde_level3(
    filename,
    keys='all',
    height_grid=np.array([]),
    ip_type='lin',
    verbose=0):

    """
    Imports single level 3 radiosonde data (merged with MetTower) created with PANGAEA_tab_to_nc.py
    ('PS122_mosaic_radiosonde_level3_yyyymmdd_hhmmssZ.nc'). Converts to SI units and interpolates 
    to a height grid with 5 m resolution from 0 to 15000 m. 

    Parameters:
    -----------
    filename : str
        Name (including path) of radiosonde data file.
    keys : list of str or str, optional
        This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
        Specifying 'basic' will load the variables the author consideres most useful for his current
        analysis.
        Default: 'all'
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    ip_type : str
        String indicating the interpolation type. Option: 'lin' for linear interpolation using 
        np.interp; 'avg' for using interp_w_avg, which averages data over layers centered on the
        height_grid.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    """
        Loaded values are imported in the following units:
        T: in deg C, will be converted to K
        P: in hPa, will be converted to Pa
        RH: in %, will be converted to [0-1]
        Altitude: in m
        q: in kg kg^-1 (water vapor specific humidity)
        time: in sec since 1970-01-01 00:00:00 UTC
    """

    DS = xr.open_dataset(filename)

    if (not isinstance(keys, str)) and (not isinstance(keys, list)):
        raise TypeError("Argument 'key' must be a list of strings or 'all'.")

    if keys == 'all':
        keys = [*DS.variables]
    elif keys == 'basic':
        keys = ['time', 'T', 'P', 'RH', 'q', 'Altitude']

    sonde_dict = dict()
    for key in keys:
        if not key in DS.variables:
            raise KeyError(f"I have no memory of this key: {key}. Key not found in radiosonde file.")

        sonde_dict[key] = DS[key].values
        if key != "IWV" and len(sonde_dict[key]) == 0: # 'and': second condition only evaluated if first condition True
            return None

        if key in ['Latitude', 'Longitude']:    # only interested in the first lat, lon position
            sonde_dict[key] = sonde_dict[key][np.where(~np.isnan(sonde_dict[key]))[0][0]]
        if key == 'IWV':
            sonde_dict[key] = np.float64(sonde_dict[key])


    # convert units:
    if 'T' in keys:     # from deg C to K
        sonde_dict['T'] = sonde_dict['T'] + 273.15
    if 'RH' in keys:    # from percent to [0, 1]
        sonde_dict['RH'] = sonde_dict['RH']*0.01
    if 'P' in keys:     # from hPa to Pa
        sonde_dict['P'] = sonde_dict['P']*100
    if 'time' in keys:  # from int64 to float64
        sonde_dict['time'] = sonde_dict['time'].astype('datetime64[s]').astype('float64')
        sonde_dict['launch_time'] = np.datetime64(DS.launch_time).astype("datetime64[s]").astype('float64')


    # interpolate to new height grid:
    if len(height_grid) == 0:
        height_grid = np.arange(0,15001,5)      # in m
    keys = [*keys]      # converts dict_keys to a list
    if ip_type == 'lin':
        for key in keys:
            if sonde_dict[key].shape == sonde_dict['time'].shape:
                if key not in ['time', 'Latitude', 'Longitude', 'Altitude']:
                    sonde_dict[key + "_ip"] = np.interp(height_grid, sonde_dict['Altitude'], sonde_dict[key], right=np.nan)
                elif key == 'Altitude':
                    sonde_dict[key + "_ip"] = height_grid
    elif ip_type == 'avg':
        for key in keys:
            if sonde_dict[key].shape == sonde_dict['time'].shape:
                if key not in ['time', 'Latitude', 'Longitude', 'Altitude']:
                    sonde_dict[key + "_ip"] = interp_w_avg(sonde_dict['Altitude'], sonde_dict[key], height_grid, respect_weights=True)
                elif key == 'Altitude':
                    sonde_dict[key + "_ip"] = height_grid


    # Renaming variables: ['Lat', 'Lon', 'p', 'T', 'RH', 'GeopHgt', 'qv', 'time', ...]
    renaming = {'T': 'temp', 'P': 'pres', 'RH': 'rh',
                'Altitude': 'height',
                'Latitude': 'lat', 'Longitude': 'lon',
                'T_ip': 'temp_ip', 'P_ip': 'pres_ip', 'RH_ip': 'rh_ip',
                'Altitude_ip': 'height_ip',
                'IWV': 'iwv'}
    for ren_key in renaming.keys():
        if ren_key in sonde_dict.keys():
            sonde_dict[renaming[ren_key]] = sonde_dict[ren_key]

    # height check: how high does the data reach:
    sonde_dict['height_check'] = sonde_dict['height'][-1]

    DS = DS.close()

    return sonde_dict


def import_single_NYA_RS_radiosonde(
    filename,
    keys='all',
    height_grid=np.array([]),
    verbose=0):

    """
    Imports single NYA-RS radiosonde data for Ny Alesund. Converts to SI units
    and interpolates to a height grid with 5 m resolution from 0 to 15000 m. 

    Parameters:
    -----------
    filename : str
        Name (including path) of radiosonde data file.
    keys : list of str or str, optional
        This describes which variable(s) will be loaded. Specifying 'all' will import all variables.
        Specifying 'basic' will load the variables the author consideres most useful for his current
        analysis.
        Default: 'all'
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    """
        Loaded values are imported in the following units:
        T: in K
        P: in hPa, will be converted to Pa
        RH: in [0-1]
        Altitude: in m
        time: will be converted to sec since 1970-01-01 00:00:00 UTC
    """

    file_nc = nc.Dataset(filename)

    if (not isinstance(keys, str)) and (not isinstance(keys, list)):
        raise TypeError("Argument 'key' must be a list of strings or 'all'.")

    if keys == 'all':
        keys = file_nc.variables.keys()
    elif keys == 'basic':
        keys = ['time', 'temp', 'press', 'rh', 'alt']

    sonde_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in radiosonde file." % key)

        sonde_dict[key] = np.asarray(file_nc.variables[key])
        if key != "IWV" and len(sonde_dict[key]) == 0: # 'and': second condition only evaluated if first condition True
            return None

        if key in ['lat', 'lon']:   # only interested in the first lat, lon position
            sonde_dict[key] = sonde_dict[key][0]

    # convert units:
    if 'P' in keys:     # from hPa to Pa
        sonde_dict['P'] = sonde_dict['P']*100
    if 'time' in keys:  # from int64 to float64
        time_unit = file_nc.variables['time'].units
        time_offset = (dt.datetime.strptime(time_unit[-19:], "%Y-%m-%dT%H:%M:%S") - dt.datetime(1970,1,1)).total_seconds()
        sonde_dict['time'] = np.float64(sonde_dict['time']) + time_offset
        sonde_dict['launch_time'] = sonde_dict['time'][0]

    # interpolate to new height grid:
    if len(height_grid) == 0:
        height_grid = np.arange(0,15001,5)      # in m
    keys = [*keys]      # converts dict_keys to a list
    for key in keys:
        if sonde_dict[key].shape == sonde_dict['time'].shape:
            if key not in ['time', 'lat', 'lon', 'alt']:
                sonde_dict[key + "_ip"] = np.interp(height_grid, sonde_dict['alt'], sonde_dict[key])
            elif key == 'alt':
                sonde_dict[key + "_ip"] = height_grid


    # Renaming variables to a standard convention
    renaming = {'press': 'pres', 'alt': 'height', 'press_ip': 'pres_ip', 'alt_ip': 'height_ip'}
    for ren_key in renaming.keys():
        if ren_key in sonde_dict.keys():
            sonde_dict[renaming[ren_key]] = sonde_dict[ren_key]

    return sonde_dict


def import_radiosonde_daterange(
    path_data,
    date_start,
    date_end,
    s_version='level_2',
    with_wind=False,
    remove_failed=False,
    extend_height_grid=False,
    height_grid=np.array([]),
    ip_type='lin',
    verbose=0):

    """
    Imports radiosonde data (several versions supported, see below) and concatenates the files 
    into time series x height. E.g. temperature profile will have the dimension: n_sondes x n_height

    Parameters:
    -----------
    path_data : str
        Path of radiosonde data.
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    s_version : str, optional
        Specifies the radiosonde version that is to be imported. Possible options: 'mossonde',
        'psYYMMDDwHH', 'level_2', 'nya-rs', 'GRUAN', 'level_3'. Default: 'level_2' (published by 
        Marion Maturilli)
    with_wind : bool, optional
        This describes if wind measurements are included (True) or not (False). Does not work with
        s_version='psYYMMDDwHH'. Default: False
    remove_failed : bool, optional
        If True, failed sondes with unrealistic IWV values will be removed (currently only implmented
        for s_version in ['level_2', 'GRUAN', 'level_3']). It also includes "height_check" to avoid 
        sondes that burst before reaching > 10000 m.
    extend_height_grid : bool
        If True, the new height grid, to which the radiosonde data is interpolated to is 0, 10, 20, ...
        25000 m. If False, it's 0, 5, 10, 15, ..., 15000 m.
    height_grid : array of floats or None
        If not None, height_grid contains a 1D array of floats indicating the new height grid to 
        which the radiosonde data is interpolated to.
    ip_type : str
        String indicating the interpolation type. Option: 'lin' for linear interpolation using 
        np.interp; 'avg' for using interp_w_avg, which averages data over layers centered on the
        height_grid. 'avg' is only available for s_version in ['level_2', 'level_3', 'GRUAN'].
    verbose : int, optional
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    def time_prematurely_bursted_sondes():

        """
        This little function merely returns time stamps of MOSAiC radiosondes, whose
        burst altitude was <= 10000 m. (Or other errors occurred.)
        """

        failed_sondes_dt = np.array([dt.datetime(2019, 10, 7, 11, 0),
                            dt.datetime(2019, 10, 15, 23, 0),
                            dt.datetime(2019, 11, 4, 11, 0),
                            dt.datetime(2019, 11, 17, 17, 0),
                            dt.datetime(2019, 12, 17, 5, 0),
                            dt.datetime(2019, 12, 24, 11, 0),
                            dt.datetime(2020, 1, 13, 11, 0),
                            dt.datetime(2020, 2, 1, 11, 0),
                            dt.datetime(2020, 2, 6, 23, 0),
                            dt.datetime(2020, 3, 9, 23, 0),
                            dt.datetime(2020, 3, 9, 11, 0), # unrealistic temperature and humidity values at the surface
                            dt.datetime(2020, 3, 11, 17, 0),
                            dt.datetime(2020, 3, 29, 5, 0),
                            dt.datetime(2020, 5, 14, 17, 0),
                            dt.datetime(2020, 6, 14, 17, 0),
                            dt.datetime(2020, 6, 19, 11, 0),
                            dt.datetime(2020, 9, 27, 9, 0)])

        reftime = dt.datetime(1970,1,1)
        failed_sondes_t = np.asarray([datetime_to_epochtime(fst) for fst in failed_sondes_dt])
        failed_sondes_t = np.asarray([(fst - reftime).total_seconds() for fst in failed_sondes_dt])
        
        return failed_sondes_t, failed_sondes_dt

    if not isinstance(s_version, str): raise TypeError("s_version in import_radiosonde_daterange must be a string.")

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    if s_version == 'mossonde':
        all_radiosondes_nc = sorted(glob.glob(path_data + "mossonde-curM1" + "*.nc"))

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-16:-8]     # date of radiosonde from filename
            yyyy = int(rs_date[:4])
            mm = int(rs_date[4:6])
            dd = int(rs_date[6:])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)

    elif s_version == 'psYYMMDDwHH':
        all_radiosondes_nc = sorted(glob.glob(path_data + "ps*.w*.nc"))[:-1]    # exclude last file because it's something about Ozone

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-13:-3]     # date of radiosonde from filename
            yyyy = 2000 + int(rs_date[:2])
            mm = int(rs_date[2:4])
            dd = int(rs_date[4:6])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)

    elif s_version == 'nya-rs':
        all_radiosondes_nc = sorted(glob.glob(path_data + "NYA-RS_*.nc"))

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-15:-3]     # date of radiosonde from filename
            yyyy = int(rs_date[:4])
            mm = int(rs_date[4:6])
            dd = int(rs_date[6:8])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)

    elif s_version == 'level_2':
        all_radiosondes_nc = sorted(glob.glob(path_data + "PS122_mosaic_radiosonde_level2*.nc"))

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-19:-3]     # date of radiosonde from filename
            yyyy = int(rs_date[:4])
            mm = int(rs_date[4:6])
            dd = int(rs_date[6:8])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)

    elif s_version == 'GRUAN':
        all_radiosondes_nc = sorted(glob.glob(path_data + "PS122_mosaic_radiosonde_level3_GRUAN*.nc"))

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-19:-3]     # date of radiosonde from filename
            yyyy = int(rs_date[:4])
            mm = int(rs_date[4:6])
            dd = int(rs_date[6:8])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)

    elif s_version == 'level_3':
        all_radiosondes_nc = sorted(glob.glob(path_data + "PS122_mosaic_radiosonde_level3*.nc"))

        # inquire the number of radiosonde files (date and time of launch is in filename):
        # And fill a list which will include the relevant radiosonde files.
        radiosondes_nc = []
        for rs_nc in all_radiosondes_nc:
            rs_date = rs_nc[-19:-3]     # date of radiosonde from filename
            yyyy = int(rs_date[:4])
            mm = int(rs_date[4:6])
            dd = int(rs_date[6:8])
            rs_date_dt = dt.datetime(yyyy,mm,dd)
            if rs_date_dt >= date_start and rs_date_dt <= date_end:
                radiosondes_nc.append(rs_nc)


    # number of sondes:
    n_sondes = len(radiosondes_nc)

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days

    # basic variables that should always be imported:
    if s_version == 'mossonde':
        geoinfo_keys = ['lat', 'lon', 'alt', 'launch_time']
        time_height_keys = ['pres', 'temp', 'rh', 'height', 'rho_v', 'q']       # keys with time and height as coordinate
        if with_wind: time_height_keys = time_height_keys + ['wspeed', 'wdir']
    elif s_version == 'psYYMMDDwHH':
        geoinfo_keys = ['lat', 'lon', 'launch_time']
        time_height_keys = ['pres', 'temp', 'rh', 'height', 'rho_v', 'q']
        if with_wind:
            print("No direct wind calculation for s_version='%s'."%s_version)
    elif s_version == 'nya-rs':
        geoinfo_keys = ['lat', 'lon', 'launch_time']
        time_height_keys = ['pres', 'temp', 'rh', 'height']
        if with_wind: time_height_keys = time_height_keys + ['wspeed', 'wdir']
    elif s_version in ['level_2', 'GRUAN', 'level_3']:
        geoinfo_keys = ['lat', 'lon', 'launch_time', 'iwv']
        time_height_keys = ['pres', 'temp', 'rh', 'height', 'rho_v', 'q']
        if with_wind: time_height_keys = time_height_keys + ['wspeed', 'wdir']
    else:
        raise ValueError("s_version in import_radiosonde_daterange must be 'mossonde', 'psYYMMDDwHH', 'nya-rs', " +
                            "'level_2', 'GRUAN' or 'level_3'.")
    all_keys = geoinfo_keys + time_height_keys


    # sonde_master_dict (output) will contain all desired variables on specific axes:
    # Time axis (one sonde = 1 timestamp) = axis 0; height axis = axis 1
    if len(height_grid) == 0:
        if extend_height_grid:
            new_height_grid = np.arange(0,25001,10)
        else:
            new_height_grid = np.arange(0,15001,5)
    else:
        new_height_grid = height_grid
    n_height = len(new_height_grid) # length of the interpolated height grid
    sonde_master_dict = dict()
    for gk in geoinfo_keys: sonde_master_dict[gk] = np.full((n_sondes,), np.nan)
    for thk in time_height_keys: sonde_master_dict[thk] = np.full((n_sondes, n_height), np.nan)


    if s_version == 'mossonde':
        all_keys_import = geoinfo_keys + time_height_keys + ['time', 'geopheight']  # 'time' required to create 'launch_time'
        all_keys_import.remove('launch_time')       # because this key is not saved in the radiosonde files
        all_keys_import.remove('rho_v')             # because this key is not saved in the radiosonde files
        all_keys_import.remove('q')                 # because this key is not saved in the radiosonde files
        all_keys_import.remove('height')                    # because this key is not saved in the radiosonde files
        if with_wind: all_keys_import = all_keys_import + ['wspeed', 'wdir']

        # cycle through all relevant sonde files:
        for rs_idx, rs_nc in enumerate(radiosondes_nc):

            if verbose >= 1:
                # rs_date = rs_nc[-16:-8]
                # print("Working on Radiosonde, ", 
                    # dt.datetime(int(rs_date[:4]), int(rs_date[4:6]), int(rs_date[6:])))
                print("Working on Radiosonde, " + rs_nc)

            sonde_dict = import_single_mossonde_curM1(rs_nc, keys=all_keys_import)

            # save to sonde_master_dict:
            for key in all_keys:
                if key in geoinfo_keys:
                    sonde_master_dict[key][rs_idx] = sonde_dict[key]

                elif key in time_height_keys:
                    sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]     # must use the interpolated versions!

                else:
                    raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with import_single_mossonde_curM1")

    elif s_version == 'psYYMMDDwHH':
        all_keys_import = ['Lat', 'Lon', 'p', 'T', 'RH', 'GeopHgt', 'qv', 'time']   # 'time' required to create 'launch_time'


        # cycle through all relevant sonde files:
        for rs_idx, rs_nc in enumerate(radiosondes_nc):

            if verbose >= 1: 
                # rs_date = rs_nc[-16:-8]
                print("Working on Radiosonde, " + rs_nc)

            sonde_dict = import_single_psYYMMDD_wHH_sonde(rs_nc, keys=all_keys_import)
            if not sonde_dict:  # then the imported sonde file appears to be empty
                continue

            else:
                # save to sonde_master_dict:
                for key in all_keys:
                    if key in geoinfo_keys:
                        sonde_master_dict[key][rs_idx] = sonde_dict[key]

                    elif key in time_height_keys:
                        sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]     # must use the interpolated versions!

                    else:
                        raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with import_single_mossonde_curM1")

        # As there are empty files among the current psYYMMDD.wHH sondes, they have to be filtered out:
        not_corrupted_sondes = ~np.isnan(sonde_master_dict['launch_time'])
        # not_corrupted_sondes_idx = np.where(~np.isnan(sonde_master_dict['launch_time']))[0]
        for key in sonde_master_dict.keys():
            if key in geoinfo_keys:
                sonde_master_dict[key] = sonde_master_dict[key][not_corrupted_sondes]
            else:
                sonde_master_dict[key] = sonde_master_dict[key][not_corrupted_sondes,:]

    elif s_version == 'nya-rs':
        all_keys_import = ['lat', 'lon', 'press', 'temp', 'rh', 'alt', 'time']
        if with_wind: all_keys_import = all_keys_import + ['wdir', 'wspeed']


        # cycle through all relevant sonde files:
        for rs_idx, rs_nc in enumerate(radiosondes_nc):
            
            if verbose >= 1:
                # rs_date = rs_nc[-19:-3]
                print("\rWorking on Radiosonde, " + rs_nc, end="")

            sonde_dict = import_single_NYA_RS_radiosonde(rs_nc, keys=all_keys_import)
            
            # save to sonde_master_dict:
            for key in all_keys:
                if key in geoinfo_keys:
                    sonde_master_dict[key][rs_idx] = sonde_dict[key]

                elif key in time_height_keys:
                    sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]     # must use the interpolated versions!

                else:
                    raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with " +
                                    "import_single_NYA_RS_radiosonde")

    elif s_version == 'level_2':
        all_keys_import = ['Latitude', 'Longitude', 'P', 'T', 'RH', 'Altitude', 'rho_v', 'q', 'time', 'IWV']
        if with_wind: all_keys_import = all_keys_import + ['wdir', 'wspeed']

        if remove_failed:
            failed_sondes_t, failed_sondes_dt = time_prematurely_bursted_sondes()       # load times of failed sondes


        # cycle through all relevant sonde files:
        rs_idx = 0
        for rs_nc in radiosondes_nc:
            
            if verbose >= 1:
                # rs_date = rs_nc[-19:-3]
                print("Working on Radiosonde, " + rs_nc, end='\r')

            sonde_dict = import_single_PS122_mosaic_radiosonde_level2(rs_nc, keys=all_keys_import, height_grid=new_height_grid,
                                                                        ip_type=ip_type)
            if (remove_failed and ((sonde_dict['iwv'] == 0.0) or (np.isnan(sonde_dict['iwv'])) or
                (sonde_dict['height_check'] < 10000) or (np.any(np.abs(sonde_dict['launch_time'] - failed_sondes_t) < 7200)))):
                continue
            
            # save to sonde_master_dict:
            for key in all_keys:
                if key in geoinfo_keys:
                    sonde_master_dict[key][rs_idx] = sonde_dict[key]

                elif key in time_height_keys:
                    sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]     # must use the interpolated versions!

                else:
                    raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with " +
                                    "import_single_PS122_mosaic_radiosonde_level2")

            rs_idx += 1

        # Truncate number of sondes:
        if remove_failed and (rs_idx < n_sondes):
            for key in geoinfo_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx]
            for key in time_height_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx,:]

    elif s_version == 'GRUAN':
        all_keys_import = ['Latitude', 'Longitude', 'P', 'T', 'RH', 'h_geop', 'q', 'rho_v', 'time', 'IWV']
        if with_wind: all_keys_import = all_keys_import + ['wdir', 'wspeed']

        if remove_failed:
            failed_sondes_t, failed_sondes_dt = time_prematurely_bursted_sondes()       # load times of failed sondes


        # cycle through all relevant sonde files:
        rs_idx = 0
        for rs_nc in radiosondes_nc:
            
            if verbose >= 1:
                # rs_date = rs_nc[-19:-3]
                print("Working on Radiosonde, " + rs_nc, end='\r')

            sonde_dict = import_single_PS122_mosaic_radiosonde_GRUAN(rs_nc, keys=all_keys_import, height_grid=new_height_grid,
                                                                    ip_type=ip_type)
            if (remove_failed and ((sonde_dict['iwv'] == 0.0) or (np.isnan(sonde_dict['iwv'])) or
                (sonde_dict['height_check'] < 10000) or (np.any(np.abs(sonde_dict['launch_time'] - failed_sondes_t) < 7200)))):
                continue
            
            # save to sonde_master_dict:
            for key in all_keys:
                if key in geoinfo_keys:
                    sonde_master_dict[key][rs_idx] = sonde_dict[key]

                elif key in time_height_keys:
                    sonde_master_dict[key][rs_idx,:] = sonde_dict[key + "_ip"]      # must use the interpolated versions!

                else:
                    raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with " +
                                    "import_single_PS122_mosaic_radiosonde_GRUAN")

            rs_idx += 1     # only counts non-failed sondes


        # Truncate number of sondes:
        if remove_failed and (rs_idx < n_sondes):
            for key in geoinfo_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx]
            for key in time_height_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx,:]

    elif s_version == 'level_3':
        all_keys_import = ['Latitude', 'Longitude', 'P', 'T', 'RH', 'Altitude', 'rho_v', 'q', 'time', 'IWV']
        if with_wind: all_keys_import += ['wdir', 'wspeed']

        if remove_failed:
            failed_sondes_t, failed_sondes_dt = time_prematurely_bursted_sondes()       # load times of failed sondes


        # cycle through all relevant sonde files:
        rs_idx = 0
        for rs_nc in radiosondes_nc:
            
            if verbose >= 1:
                # rs_date = rs_nc[-19:-3]
                print("Working on Radiosonde, " + rs_nc, end='\r')

            sonde_dict = import_single_PS122_mosaic_radiosonde_level3(rs_nc, keys=all_keys_import, height_grid=new_height_grid,
                                                                    ip_type=ip_type)

            if (remove_failed and ((sonde_dict['iwv'] == 0.0) or (np.isnan(sonde_dict['iwv'])) or
                (sonde_dict['height_check'] < 10000) or (np.any(np.abs(sonde_dict['launch_time'] - failed_sondes_t) < 7200)))):
                continue
            
            # save to sonde_master_dict:
            for key in all_keys:
                if key in geoinfo_keys:
                    sonde_master_dict[key][rs_idx] = sonde_dict[key]

                elif key in time_height_keys:
                    sonde_master_dict[key][rs_idx, :] = sonde_dict[key + "_ip"]     # must use the interpolated versions!

                else:
                    raise KeyError("Key '" + key + "' not found in radiosonde dictionary after importing it with " +
                                    "import_single_PS122_mosaic_radiosonde_level3")

            rs_idx += 1

        # Truncate number of sondes:
        if remove_failed and (rs_idx < n_sondes):
            for key in geoinfo_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx]
            for key in time_height_keys: sonde_master_dict[key] = sonde_master_dict[key][:rs_idx,:]


    if verbose >= 1: print("")

    return sonde_master_dict


def import_concat_IWV_LWP_mwr_master_time(
    filename,
    date_start,
    date_end):

    """
    Simple importer to get the IWV or LWP data, that was stored on a 'master time axis',
    of a radiometer (HATPRO, MiRAC-P, ARM).

    Parameters:
    -----------
    filename : str
        Filename and path of the IWV or LWP data of radiometers on the master time axis.
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    """

    date_start = datetime_to_epochtime(dt.datetime.strptime(date_start, "%Y-%m-%d"))
    date_end = (dt.datetime.strptime(date_end, "%Y-%m-%d") - dt.datetime(1969,12,31)).total_seconds()
    # 1969,12,31 because then the date_end day is INCLUDED

    file_nc = nc.Dataset(filename)

    mwr_dict = dict()
    for key in file_nc.variables.keys():
        if key == 'time':
            master_time = np.asarray(file_nc.variables[key])
        elif key in ['IWV', 'LWP']:
            data_mt = np.asarray(file_nc.variables[key])
        else:
            raise KeyError("Key %s was not found in file containing the concatenated IWV/LWP data on master time axis"%key)

    # Trim the data and time arrays:
    # case: time[0] < date_start: trim lower end
    # case: time[0] >= date_start: dont trim lower end
    # case: time[-1] > date_end: trim upper end
    # case: time[-1] <= date_end: dont trim upper end
    if master_time[0] < date_start:
        trim_low_idx = master_time >= date_start
        master_time = master_time[trim_low_idx]
        data_mt = data_mt[trim_low_idx]
    if master_time[-1] > date_end:
        trim_high_idx = master_time < date_end
        master_time = master_time[trim_high_idx]
        data_mt = data_mt[trim_high_idx]

    return master_time, data_mt


def import_concat_IWV_LWP_mwr_running_mean(
    filename,
    date_start,
    date_end,
    instrument):

    """
    Simple importer to get the IWV or LWP data, to which a moving average over a certain time
    span was performed, of a radiometer (HATPRO, MiRAC-P, ARM). The loaded data will be trimmed
    according to the specified date_start and date_end.

    Parameters:
    -----------
    filename : str
        Filename and path of the IWV or LWP data of radiometers with moving average (running mean).
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    instrument : str
        Specifies which instrument is mwr_dict. Can only be 'hatpro', 'mirac' or 'arm'.
    """

    if instrument not in ['hatpro', 'mirac', 'arm']:
        raise ValueError("'instrument' must be either 'hatpro', 'mirac' or 'arm'.")

    date_start = datetime_to_epochtime(dt.datetime.strptime(date_start, "%Y-%m-%d"))
    date_end = (dt.datetime.strptime(date_end, "%Y-%m-%d") - dt.datetime(1969,12,31)).total_seconds()
    # 1969,12,31 because then the date_end day is INCLUDED

    file_nc = nc.Dataset(filename)

    mwr_dict = dict()
    for key in file_nc.variables.keys():
        if key == 'rm_window':
            rm_window = int(np.asarray(file_nc.variables[key]))
        elif key == 'time':
            time_rm = np.asarray(file_nc.variables[key])
        elif key in ['IWV', 'LWP']:
            data_rm = np.asarray(file_nc.variables[key])

    # Trim the data and time arrays:
    # case: time_rm[0] < date_start: trim lower end
    # case: time_rm[0] >= date_start: dont trim lower end
    # case: time_rm[-1] > date_end: trim upper end
    # case: time_rm[-1] <= date_end: dont trim upper end
    if instrument in ['hatpro', 'arm', 'mirac']:
        if time_rm[0] < date_start:
            trim_low_idx = time_rm >= date_start
            time_rm = time_rm[trim_low_idx]
            data_rm = data_rm[trim_low_idx]
        if time_rm[-1] > date_end:
            trim_high_idx = time_rm < date_end
            time_rm = time_rm[trim_high_idx]
            data_rm = data_rm[trim_high_idx]
        

    return rm_window, data_rm, time_rm


def import_arm_def_daterange(
    path_data,
    date_start,
    date_end,
    which_retrieval='both',
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the default products as time
    series of each day so that you'll have one dictionary, whose e.g. 'IWV' will contain the IWV
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Base path of ARM data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/arm/mwrret/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. 'both' will 
        load both. Default: 'both'
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'clwvi', 'lwp', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

        else:
                if which_retrieval == 'iwv':
                    which_retrieval = ['prw']
                elif which_retrieval == 'lwp':
                    which_retrieval = ['clwvi']
                elif which_retrieval == 'both':
                    which_retrieval = ['prw', 'clwvi']
                else:
                    raise ValueError("Argument '" + which_retrieval + "' not recognised. Please use one of the following options: " +
                        "'iwv' or 'prw' will load the " +
                        "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                        "'both' will load both. Default: 'both'")
                    

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")


    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1

    # basic variables that should always be imported:
    mwr_geoinfo_keys = ['lat', 'lon', 'alt']
    mwr_time_keys = ['time', 'time_offset']             # keys with time as coordinate

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    mwr_master_dict = dict()
    for mgk in mwr_geoinfo_keys: mwr_master_dict[mgk] = np.full((n_days,), np.nan)                  ## could be reduced to one value for the entire period
    # for mgk in mwr_geoinfo_keys: mwr_master_dict[mgk] = np.nan

    # max number of seconds: n_days*86400
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_days*86400,), np.nan)

    if 'prw' in which_retrieval:
        mwr_master_dict['prw'] = np.full((n_days*86400,), np.nan)
        mwr_master_dict['prw_flag'] = np.full((n_days*86400,), np.nan)

    if 'clwvi' in which_retrieval:
        mwr_master_dict['lwp'] = np.full((n_days*86400,), np.nan)
        mwr_master_dict['lwp_flag'] = np.full((n_days*86400,), np.nan)


    # cycle through all years, all months and days:
    time_index = 0  # this index will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    day_index = 0   # same as above, but only increases by 1 for each day
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("Working on ARM default products, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # list of files:
        arm_nc = sorted(glob.glob(day_path + "*.nc"))
        if len(arm_nc) > 1: pdb.set_trace()

        if len(arm_nc) == 0:
            if verbose >= 2:
                warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        # load retrieved variable(s) from current day and save it into the mwr_master_dict
        for a_nc in arm_nc:
            if 'prw' in which_retrieval and 'clwvi' in which_retrieval:
                mwr_dict = import_arm_default(a_nc, 'basic', 'both')
            else:
                mwr_dict = import_arm_default(a_nc, 'basic', which_retrieval[0])

            n_time = len(mwr_dict['time'])
            cur_time_shape = mwr_dict['time'].shape

            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                if mwr_key in ['be_lwp', 'qc_be_lwp', 'be_pwv', 'qc_be_pwv']: continue  # other variable names exist                                    ############# could be made nicer ?
                mwr_key_shape = mwr_dict[mwr_key].shape

                if mwr_key_shape == cur_time_shape: # then the variable is on time axis:
                    mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key]

                elif mwr_key_shape == () and mwr_key in mwr_geoinfo_keys:
                    mwr_master_dict[mwr_key][day_index:day_index + 1] = mwr_dict[mwr_key]               ## for the case that we use daily values
                    # mwr_master_dict[mwr_key] = mwr_dict[mwr_key]      # up to now, geoinfo value is always the same

                elif mwr_dict[mwr_key].size == 1:
                    mwr_master_dict[mwr_key][day_index:day_index + 1] = mwr_dict[mwr_key]

                else:
                    raise RuntimeError("Something went wrong in the " +
                        "import_arm_def_daterange routine. Unexpected MWR variable dimension. " + 
                        "The length of one used variable ('%s') of arm data "%(mwr_key) +
                            "neither equals the length of the time axis nor equals 1.")


        time_index = time_index + n_time
        day_index = day_index + 1

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        for mwr_key in mwr_master_dict.keys():
            if mwr_master_dict[mwr_key].shape == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]

    return mwr_master_dict


def import_arm_default(
    filename,
    keys='basic',
    which_retrieval='both'):

    """
    Importing ARM default data (integrated quantities, e.g. IWV, LWP).

    Parameters:
    -----------
    filename : str
        Path and filename of ARM default products.
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. 'both' will 
        load both. Default: 'both'
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'time_offset', 'alt', 'lat', 'lon']
        if which_retrieval in ['lwp', 'clwvi', 'both']:
            for add_key in ['be_lwp', 'qc_be_lwp']: keys.append(add_key)
        if which_retrieval in ['iwv', 'prw', 'both']:
            for add_key in ['be_pwv', 'qc_be_pwv']: keys.append(add_key)

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in level 2a file." % key)
        mwr_dict[key] = np.asarray(file_nc.variables[key])

        # rename some variables:
        if key == 'be_lwp': mwr_dict['lwp'] = mwr_dict[key]/1000        # convert to kg/m^2
        if key == 'be_pwv': mwr_dict['prw'] = mwr_dict[key]*10          # convert to kg/m^2
        if key == 'qc_be_lwp': mwr_dict['lwp_flag'] = mwr_dict[key]
        if key == 'qc_be_pwv': mwr_dict['prw_flag'] = mwr_dict[key]


    if file_nc.variables['time_offset'].units[14:-5] != file_nc.variables['time'].units[14:-5]: pdb.set_trace()

    # convert time to unixtime
    if 'time' in keys:
        base_time = np.asarray(file_nc.variables['base_time'])  # shall not be saved to mwr_dict
        mwr_dict['time'] = base_time + mwr_dict['time']

    return mwr_dict


def import_IWV_sonde_txt(filename):

    """
    Importing a .txt file of IWV computed from radiosondes by Sandro Dahlke (sandro.dahlke@awi.de).
    Furthermore includes the time and balloon burst altitude.

    Parameters:
    -----------
    filename : str
        Includes the filename and path of the all_iwv.txt.
    """

    reftime = dt.datetime(1970,1,1)

    headersize = 8
    file_handler = open(filename, 'r')
    list_of_lines = list()

    sonde_iwv_dict = {'datetime': np.full((1555,), reftime),
                    'time': np.full((1555,), np.nan),
                    'iwv': np.full((1555,), np.nan),
                    'balloon_burst_alt': np.full((1555,), np.nan)}
    for k, line in enumerate(file_handler):
        m = k - 8       # used as index
        if k >= headersize: # ignore header
            current_line = line.strip().split(' ')  # split by 3 spaces

            # convert time stamp to datetime and seconds since 1970-01-01 00:00:00 UTC:
            yyyy = int(float(current_line[0]))
            mm = int(float(current_line[1]))
            dd = int(float(current_line[2]))
            HH = int(float(current_line[3]))
            MM = int(float(current_line[4]))
            SS = int(float(current_line[5]))
            
            sonde_iwv_dict['datetime'][m] = dt.datetime(yyyy,mm,dd,HH,MM,SS)
            sonde_iwv_dict['time'][m] = datetime_to_epochtime(sonde_iwv_dict['datetime'][m])
            sonde_iwv_dict['iwv'][m] = float(current_line[6])
            sonde_iwv_dict['balloon_burst_alt'][m] = float(current_line[7])

    return sonde_iwv_dict


def import_IWV_OE_JR(
    filename,
    date_start,
    date_end,
    verbose=0):

    """
    Imports the OE retrieval results from satellite-based IWV retrieval done by Janna Rückert.
    Time will be converted to seconds since 1970-01-01 00:00:00 UTC and to datetime objects.

    Parameters:
    -----------
    filename : str
        Path and filename of the .csv table that includes the IWV data.
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # open the csv and read it:
    with open(filename, newline='') as csvfile:

        csv_reader = csv.reader(csvfile, dialect='excel', delimiter=',')
        list_of_lines = list()
        for row in csv_reader:
            list_of_lines.append(row)
        n_lines = len(list_of_lines) - 1        # -1 to ignore the first line

    reftime = dt.datetime(1970,1,1)

    IWV_JR_dict = {'IWV': np.full((n_lines,), np.nan),
                    'IWV_std': np.full((n_lines,), np.nan),
                    'time': np.full((n_lines,), np.nan),
                    'datetime': np.full((n_lines,), reftime)}
    # save data to dict:
    for k, c_line in enumerate(list_of_lines[1:]):      # skip first line
        IWV_JR_dict['IWV'][k] = float(c_line[8])
        IWV_JR_dict['IWV_std'][k] = float(c_line[9])
        IWV_JR_dict['datetime'][k] = dt.datetime.strptime(c_line[1], "%Y-%m-%d %H:%M:%S")
        IWV_JR_dict['time'][k] = datetime_to_epochtime(IWV_JR_dict['datetime'][k])

    # trim redundant time steps that are out of [date_start, date_end]:
    # # # # case: time[0] < date_start: trim lower end
    # # # # case: time[0] >= date_start: dont trim lower end
    # # # # case: time[-1] > date_end: trim upper end
    # # # # case: time[-1] <= date_end: dont trim upper end
    # # # if master_time[0] < date_start:
        # # # trim_low_idx = master_time >= date_start
        # # # master_time = master_time[trim_low_idx]
        # # # data_mt = data_mt[trim_low_idx]
    # # # if master_time[-1] > date_end:
        # # # trim_high_idx = master_time < date_end
        # # # master_time = master_time[trim_high_idx]
        # # # data_mt = data_mt[trim_high_idx]

    return IWV_JR_dict


def import_OEM_JR_v5(
    filename,
    return_DS=False):

    """
    Imports the OE results from satellite-based IWV retrieval (v5) performed by Janna Rückert.
    Time will be converted to seconds since 1970-01-01 00:00:00 UTC and np.datetime64.

    Parameters:
    -----------
    filename : str
        Path and filename of the .csv table that includes the IWV data.
    """

    # open the csv and read it:
    with open(filename, newline='') as csvfile:

        csv_reader = csv.reader(csvfile, delimiter=',')
        list_of_lines = list()
        for row in csv_reader:
            list_of_lines.append(row)
        n_lines = len(list_of_lines) - 1        # -1 to ignore the first line


    IWV_JR_dict = {'IWV': np.full((n_lines,), np.nan),
                    'IWV_std': np.full((n_lines,), np.nan),
                    'time': np.full((n_lines,), np.nan),
                    'time_npdt': np.full((n_lines,), np.datetime64("1970-01-01T00:00:00"))}
    # save data to dict:
    for k, c_line in enumerate(list_of_lines[1:]):      # skip first line
        IWV_JR_dict['IWV'][k] = float(c_line[1])
        IWV_JR_dict['IWV_std'][k] = float(c_line[2])
        IWV_JR_dict['time_npdt'][k] = np.datetime64(c_line[0]).astype("datetime64[s]")
        IWV_JR_dict['time'][k] = IWV_JR_dict['time_npdt'][k].astype("float64")


    # create xarray dataset if desired:
    if return_DS:
        DS = xr.Dataset({'IWV':         (['time'], IWV_JR_dict['IWV'],
                                        {'units': "kg m-2"}),
                        'IWV_std':      (['time'], IWV_JR_dict['IWV_std'],
                                        {'units': "kg m-2"})},
                        coords={'time': (['time'], IWV_JR_dict['time_npdt'])})
        return DS
    else:
        return IWV_JR_dict


def import_OEM_JR_colocated(
    filename,
    return_DS=True):

    """
    Imports the OE results from satellite-based IWV retrieval colocated with Polarstern during
    MOSAiC created by Janna Rückert. Time will be converted to seconds since 1970-01-01 00:00:00 
    UTC and np.datetime64.

    Parameters:
    -----------
    filename : str
        Path and filename of the .csv table that includes the IWV data.
    return_DS : bool
        If True, an xarray dataset that contains the imported data will be created and returned.
    """

    # open the csv and read it:
    headersize = 1
    with open(filename, newline='') as csvfile:

        csv_reader = csv.reader(csvfile, delimiter=',')
        list_of_lines = list()
        for row in csv_reader:
            list_of_lines.append(row)
        n_data = len(list_of_lines) - headersize        # ignore the header


    IWV_JR_dict = {'IWV': np.full((n_data,), np.nan),
                    'IWV_std': np.full((n_data,), np.nan),
                    'lat': np.full((n_data,), np.nan),
                    'lon': np.full((n_data,), np.nan),
                    'count': np.full((n_data,), np.nan),
                    'time': np.full((n_data,), np.nan),
                    'time_npdt': np.full((n_data,), np.datetime64("1970-01-01T00:00:00").astype('datetime64[ns]'))}

    # save data to dict:
    for k, c_line in enumerate(list_of_lines[headersize:]):     # skip header
        IWV_JR_dict['lat'][k] = float(c_line[1])        # in deg N
        IWV_JR_dict['lon'][k] = float(c_line[2])        # in deg E
        IWV_JR_dict['IWV'][k] = float(c_line[3])        # in kg m**-2
        IWV_JR_dict['IWV_std'][k] = float(c_line[4])    # in kg m**-2
        IWV_JR_dict['count'][k] = float(c_line[5])
        IWV_JR_dict['time_npdt'][k] = np.datetime64(c_line[0]).astype("datetime64[ns]")
        IWV_JR_dict['time'][k] = IWV_JR_dict['time_npdt'][k].astype('datetime64[s]').astype(np.float64)


    # create xarray dataset if desired:
    if return_DS:
        DS = xr.Dataset({'IWV':         (['time'], IWV_JR_dict['IWV'],
                                        {'units': "kg m**-2"}),
                        'IWV_std':      (['time'], IWV_JR_dict['IWV_std'],
                                        {'units': "kg m**-2"}),
                        'lat':          (['time'], IWV_JR_dict['lat'],
                                        {'units': "deg N"}),
                        'lon':          (['time'], IWV_JR_dict['lon'],
                                        {'units': "deg E"}),
                        'count':        (['time'], IWV_JR_dict['count'],
                                        {'comment': "number of pixels close to Polarstern"}),
                        },
                        coords={'time': (['time'], IWV_JR_dict['time_npdt'])})
        return DS
    else:
        return IWV_JR_dict


def import_PS_mastertrack_tab(filename):

    """
    Imports Polarstern master track data during MOSAiC published on PANGAEA. Time
    will be given in seconds since 1970-01-01 00:00:00 UTC and datetime. It also
    returns global attributes in the .tab file so that the information can be
    forwarded to the netcdf version of the master tracks.

    Leg 1, Version 2:
    Rex, Markus (2020): Links to master tracks in different resolutions of POLARSTERN
    cruise PS122/1, Tromsø - Arctic Ocean, 2019-09-20 - 2019-12-13 (Version 2). Alfred
    Wegener Institute, Helmholtz Centre for Polar and Marine Research, Bremerhaven, 
    PANGAEA, https://doi.org/10.1594/PANGAEA.924668

    Leg 2, Version 2:
    Haas, Christian (2020): Links to master tracks in different resolutions of POLARSTERN
    cruise PS122/2, Arctic Ocean - Arctic Ocean, 2019-12-13 - 2020-02-24 (Version 2).
    Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research,
    Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924674

    Leg 3, Version 2:
    Kanzow, Torsten (2020): Links to master tracks in different resolutions of POLARSTERN
    cruise PS122/3, Arctic Ocean - Longyearbyen, 2020-02-24 - 2020-06-04 (Version 2).
    Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, 
    Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924681

    Leg 4:
    Rex, Markus (2021): Master tracks in different resolutions of POLARSTERN cruise
    PS122/4, Longyearbyen - Arctic Ocean, 2020-06-04 - 2020-08-12. Alfred Wegener 
    Institute, Helmholtz Centre for Polar and Marine Research, Bremerhaven, PANGAEA,
    https://doi.org/10.1594/PANGAEA.926829

    Leg 5:
    Rex, Markus (2021): Master tracks in different resolutions of POLARSTERN cruise
    PS122/5, Arctic Ocean - Bremerhaven, 2020-08-12 - 2020-10-12. Alfred Wegener
    Institute, Helmholtz Centre for Polar and Marine Research, Bremerhaven, PANGAEA,
    https://doi.org/10.1594/PANGAEA.926910

    Parameters:
    -----------
    filename : str
        Filename + path of the Polarstern Track data (.tab) downloaded from the DOI
        given above.
    """

    n_prel = 20000      # just a preliminary assumption of the amount of data entries
    reftime = dt.datetime(1970,1,1)
    pstrack_dict = {'time_sec': np.full((n_prel,), np.nan),     # in seconds since 1970-01-01 00:00:00 UTC
                    'time': np.full((n_prel,), reftime),        # datetime object
                    'Latitude': np.full((n_prel,), np.nan),     # in deg N
                    'Longitude': np.full((n_prel,), np.nan),    # in deg E
                    'Speed': np.full((n_prel,), np.nan),        # in knots
                    'Course': np.full((n_prel,), np.nan)}       # in deg

    f_handler = open(filename, 'r')
    list_of_lines = list()

    # identify header size and save global attributes:
    attribute_info = list()
    for k, line in enumerate(f_handler):
        attribute_info.append(line.strip().split("\t")) # split by tabs
        if line.strip() == "*/":
            break
    attribute_info = attribute_info[1:-1]   # first and last entry are "*/"

    m = 0       # used as index to save the entries into pstrack_dict
    for k, line in enumerate(f_handler):
        if k > 0:       # skip header
            current_line = line.strip().split()     # split by tabs

            # convert time stamp to seconds since 1970-01-01 00:00:00 UTC:
            pstrack_dict['time_sec'][m] = datetime_to_epochtime(dt.datetime.strptime(current_line[0], "%Y-%m-%dT%H:%M"))

            # extract other info:
            pstrack_dict['Latitude'][m] = float(current_line[1])
            pstrack_dict['Longitude'][m] = float(current_line[2])
            pstrack_dict['Speed'][m] = float(current_line[3])
            pstrack_dict['Course'][m] = float(current_line[4])

            m = m + 1

    # truncate redundant lines:
    last_nonnan = np.where(~np.isnan(pstrack_dict['time_sec']))[0][-1] + 1      # + 1 because of python indexing
    for key in pstrack_dict.keys(): pstrack_dict[key] = pstrack_dict[key][:last_nonnan]

    # time to datetime:
    pstrack_dict['time'] = np.asarray([dt.datetime.utcfromtimestamp(tt) for tt in pstrack_dict['time_sec']])

    return pstrack_dict, attribute_info


def import_MOSAiC_Radiosondes_PS122_Level2_tab(filename):

    """
    Imports level 2 radiosonde data launched from Polarstern
    during the MOSAiC campaign. Time will be given in seconds since 1970-01-01 00:00:00 UTC
    and datetime. Furthermore, the Integrated Water Vapour will be computed
    using the saturation water vapour pressure according to Hyland and Wexler 1983.

    Maturilli, Marion; Holdridge, Donna J; Dahlke, Sandro; Graeser, Jürgen;
    Sommerfeld, Anja; Jaiser, Ralf; Deckelmann, Holger; Schulz, Alexander 
    (2021): Initial radiosonde data from 2019-10 to 2020-09 during project 
    MOSAiC. Alfred Wegener Institute, Helmholtz Centre for Polar and Marine 
    Research, Bremerhaven, PANGAEA, https://doi.pangaea.de/10.1594/PANGAEA.928656 
    (DOI registration in progress)

    Parameters:
    -----------
    filename : str
        Filename + path of the Level 2 radiosonde data (.tab) downloaded from the DOI
        given above.
    """

    n_sonde_prel = 3000     # just a preliminary assumption of the amount of radiosondes
    n_data_per_sonde = 12000    # assumption of max. time points per sonde
    reftime = dt.datetime(1970,1,1)
    # the radiosonde dict will be structured as follows:
    # rs_dict['0'] contains all data from the first radiosonde: rs_dict['0']['T'] contains temperature
    # rs_dict['1'] : second radiosonde, ...
    # this structure allows to have different time dimensions for each radiosonde
    rs_dict = dict()
    for k in range(n_sonde_prel):
        rs_dict[str(k)] = {'time': np.full((n_data_per_sonde,), reftime),       # datetime object
                            'time_sec': np.full((n_data_per_sonde,), np.nan),   # in seconds since 1970-01-01 00:00:00 UTC
                            'Latitude': np.full((n_data_per_sonde,), np.nan),   # in deg N
                            'Longitude': np.full((n_data_per_sonde,), np.nan),  # in deg E
                            'Altitude': np.full((n_data_per_sonde,), np.nan),   # in m
                            'h_geom': np.full((n_data_per_sonde,), np.nan),     # geometric height in m
                            'ETIM': np.full((n_data_per_sonde,), np.nan),       # elapsed time in seconds since sonde start
                            'P': np.full((n_data_per_sonde,), np.nan),          # in hPa
                            'T': np.full((n_data_per_sonde,), np.nan),          # in deg C
                            'RH': np.full((n_data_per_sonde,), np.nan),         # in percent
                            'wdir': np.full((n_data_per_sonde,), np.nan),       # in deg
                            'wspeed': np.full((n_data_per_sonde,), np.nan)}     # in m s^-1


    f_handler = open(filename, 'r')

    # identify header size and save global attributes:
    attribute_info = list()
    for k, line in enumerate(f_handler):
        if line.strip().split("\t")[0] in ['Citation:', 'In:', 'Abstract:', 'Keyword(s):']:
            attribute_info.append(line.strip().split("\t")) # split by tabs
        if line.strip() == "*/":
            break


    m = -1      # used as index to save the entries into rs_dict; will increase for each new radiosonde
    mm = 0      # runs though all time points of one radiosonde and is reset to 0 for each new radiosonde
    precursor_event = ''
    for k, line in enumerate(f_handler):
        if k > 0:       # skip header
            current_line = line.strip().split("\t")     # split by tabs
            current_event = current_line[0]         # marks the radiosonde launch

            if current_event != precursor_event:    # then a new sonde is found in the current_line
                m = m + 1
                mm = 0

            # convert time stamp to seconds since 1970-01-01 00:00:00 UTC:
            rs_dict[str(m)]['time'][mm] = dt.datetime.strptime(current_line[1], "%Y-%m-%dT%H:%M:%S")
            rs_dict[str(m)]['time_sec'][mm] = datetime_to_epochtime(rs_dict[str(m)]['time'][mm])

            # extract other info:
            try:
                rs_dict[str(m)]['Latitude'][mm] = float(current_line[2])
                rs_dict[str(m)]['Longitude'][mm] = float(current_line[3])
                rs_dict[str(m)]['Altitude'][mm] = float(current_line[4])
                rs_dict[str(m)]['h_geom'][mm] = float(current_line[5])
                rs_dict[str(m)]['ETIM'][mm] = float(current_line[6])
                rs_dict[str(m)]['P'][mm] = float(current_line[7])
                rs_dict[str(m)]['T'][mm] = float(current_line[8])
                rs_dict[str(m)]['RH'][mm] = float(current_line[9])
                rs_dict[str(m)]['wdir'][mm] = float(current_line[10])
                rs_dict[str(m)]['wspeed'][mm] = float(current_line[11])

            except ValueError:      # then at least one measurement is missing:
                for ix, cr in enumerate(current_line):
                    if cr == '':
                        current_line[ix] = 'nan'
                try:
                    rs_dict[str(m)]['Latitude'][mm] = float(current_line[2])
                    rs_dict[str(m)]['Longitude'][mm] = float(current_line[3])
                    rs_dict[str(m)]['Altitude'][mm] = float(current_line[4])
                    rs_dict[str(m)]['h_geom'][mm] = float(current_line[5])
                    rs_dict[str(m)]['ETIM'][mm] = float(current_line[6])
                    rs_dict[str(m)]['P'][mm] = float(current_line[7])
                    rs_dict[str(m)]['T'][mm] = float(current_line[8])
                    rs_dict[str(m)]['RH'][mm] = float(current_line[9])
                    rs_dict[str(m)]['wdir'][mm] = float(current_line[10])
                    rs_dict[str(m)]['wspeed'][mm] = float(current_line[11])

                except IndexError:      # GPS connection lost
                    rs_dict[str(m)]['Latitude'][mm] = float('nan')
                    rs_dict[str(m)]['Longitude'][mm] = float('nan')
                    rs_dict[str(m)]['Altitude'][mm] = float(current_line[4])
                    rs_dict[str(m)]['h_geom'][mm] = float('nan')
                    rs_dict[str(m)]['ETIM'][mm] = float(current_line[6])
                    rs_dict[str(m)]['P'][mm] = float(current_line[7])
                    rs_dict[str(m)]['T'][mm] = float(current_line[8])
                    rs_dict[str(m)]['RH'][mm] = float(current_line[9])
                    rs_dict[str(m)]['wdir'][mm] = float('nan')
                    rs_dict[str(m)]['wspeed'][mm] = float('nan')

            mm = mm + 1
            precursor_event = current_event

    # truncate redundantly initialised sondes:
    for k in range(m+1, n_sonde_prel): del rs_dict[str(k)]
    
    # finally truncate unneccessary time dimension for each sonde and compute IWV:
    for k in range(m+1):
        last_nonnan = np.where(~np.isnan(rs_dict[str(k)]['time_sec']))[0][-1] + 1       # + 1 because of python indexing
        for key in rs_dict[str(k)].keys(): rs_dict[str(k)][key] = rs_dict[str(k)][key][:last_nonnan]
        rs_dict[str(k)]['q'] = np.asarray([convert_rh_to_spechum(t+273.15, p*100, rh/100) 
                                for t, p, rh in zip(rs_dict[str(k)]['T'], rs_dict[str(k)]['P'], rs_dict[str(k)]['RH'])])
        rs_dict[str(k)]['rho_v'] = np.asarray([convert_rh_to_abshum(t+273.15, rh/100) 
                                for t, rh in zip(rs_dict[str(k)]['T'], rs_dict[str(k)]['RH'])])
        rs_dict[str(k)]['IWV'] = compute_IWV_q(rs_dict[str(k)]['q'], rs_dict[str(k)]['P']*100)
    
    return rs_dict, attribute_info


def import_radiosondes_PS131_txt(
    files,
    add_info=False,
    add_loc_info=False):

    """
    Imports radiosonde data gathered during PS131 from Polarstern during the ATWAICE
    campaign. The Integrated Water Vapour will be computed using the saturation 
    water vapour pressure according to Hyland and Wexler 1983. Measurements will be
    given in SI units.
    The radiosonde data will be stored in a dict with keys being the sonde index and
    the values are 1D arrays with shape (n_data_per_sonde,).

    Parameters:
    -----------
    files : str
        List of filename + path of the Polarstern PS131 radiosonde data (.tab) taken 
        directly from the ship.
    add_info : bool
        If True, auxiliary information is added to the returned dictionary 
        (i.e., operator of radiosonde, max altitude before burst).
    add_loc_info : bool
        If True, latitude and longitude position of the radiosonde at launch will be provided in
        the returned dictionary as well. The data is extracted from the radiosonde .txt file 
        header.
    """

    n_sondes = len(files)       # just a preliminary assumption of the amount of radiosondes
    n_data_per_sonde = 12000    # assumption of max. time (data) points per sonde
    reftime = dt.datetime(1970,1,1)

    # the radiosonde dict will be structured as follows:
    # rs_dict['0'] contains all data from the first radiosonde: rs_dict['0']['temp'] contains temperature
    # rs_dict['1'] : second radiosonde, ...
    # this structure allows to have different time dimensions for each radiosonde
    rs_dict = dict()
    for k in range(n_sondes):
        rs_dict[str(k)] = {'height': np.full((n_data_per_sonde,), np.nan),      # in m
                            'pres': np.full((n_data_per_sonde,), np.nan),       # in Pa
                            'temp': np.full((n_data_per_sonde,), np.nan),       # in K
                            'relhum': np.full((n_data_per_sonde,), np.nan),     # in [0,1]
                            'wdir': np.full((n_data_per_sonde,), np.nan),       # in deg
                            'wspeed': np.full((n_data_per_sonde,), np.nan)}     # in m s^-1

    # loop over files:
    for kk, file in enumerate(files):
        f_handler = open(file, 'r')
        s_idx = str(kk)

        mm = 0      # runs though all data points of one radiosonde and is reset to 0 for each new radiosonde
        headersize = 16
        for k, line in enumerate(f_handler):
            if k < headersize:      # to extract launch time
                lt_idx = line.find("DATE/TIME START: ")
                if lt_idx != -1:
                    rs_dict[s_idx]['launch_time_npdt'] = np.datetime64(line[lt_idx+len("DATE_TIME START: "):lt_idx+len("DATE_TIME START: ")+19])
                    rs_dict[s_idx]['launch_time'] = rs_dict[s_idx]['launch_time_npdt'].astype(np.float64)   # in sec since 1970-01-01 00:00:00

                if add_info:
                    op_idx = line.find("Operator: ")            # person who launched the sonde
                    ma_idx = line.find("MAXIMUM ALTITUDE: ")    # altitude where the balloon burst
                    if op_idx != -1:
                        rs_dict[s_idx]['op'] = line[op_idx+len("Operator: "):line.find("\n")]

                    if ma_idx != -1:
                        rs_dict[s_idx]['max_alt'] = int(line[ma_idx+len("MAXIMUM ALTITUDE: "):line.find("\n")-1])

                if add_loc_info:
                    lat_idx = line.find("LATITUDE: ")
                    lon_idx = line.find("LONGITUDE: ")
                    if lat_idx != -1:
                        rs_dict[s_idx]['ref_lat'] = float(line[lat_idx+len("LATITUDE: "):line.find("*")-1])

                    if lon_idx != -1:
                        rs_dict[s_idx]['ref_lon'] = float(line[lon_idx+len("LONGITUDE: "):line.find("\n")])


            else:       # skip header
                current_line = line.strip().split("\t")     # split by tabs

                # extract data:
                try:
                    rs_dict[s_idx]['height'][mm] = float(current_line[0])
                    rs_dict[s_idx]['pres'][mm] = float(current_line[1])*100.0
                    rs_dict[s_idx]['temp'][mm] = float(current_line[2]) + 273.15
                    rs_dict[s_idx]['relhum'][mm] = float(current_line[3])*0.01
                    rs_dict[s_idx]['wdir'][mm] = float(current_line[4])
                    rs_dict[s_idx]['wspeed'][mm] = float(current_line[5])

                except ValueError:      # then at least one measurement is missing:
                    for ix, cr in enumerate(current_line):
                        if cr == '':
                            current_line[ix] = 'nan'
                    rs_dict[s_idx]['height'][mm] = float(current_line[0])
                    rs_dict[s_idx]['pres'][mm] = float(current_line[1])*100.0
                    rs_dict[s_idx]['temp'][mm] = float(current_line[2]) + 273.15
                    rs_dict[s_idx]['relhum'][mm] = float(current_line[3])*0.01
                    rs_dict[s_idx]['wdir'][mm] = float(current_line[4])
                    rs_dict[s_idx]['wspeed'][mm] = float(current_line[5])

                mm += 1

        # finally truncate unneeded data lines and compute specific humidity and IWV:
        last_nonnan = np.array([np.where(~np.isnan(rs_dict[s_idx][key]))[0][-1] + 1 for key in rs_dict[s_idx].keys() if key not in ['launch_time', 'launch_time_npdt', 'max_alt', 'op', 'ref_lat', 'ref_lon']]).min()
        for key in rs_dict[s_idx].keys(): 
            if key not in ['launch_time', 'launch_time_npdt', 'op', 'max_alt', 'ref_lat', 'ref_lon']:
                rs_dict[s_idx][key] = rs_dict[s_idx][key][:last_nonnan]
        rs_dict[s_idx]['q'] = convert_rh_to_spechum(rs_dict[s_idx]['temp'], rs_dict[s_idx]['pres'], 
                                                    rs_dict[s_idx]['relhum'])

        rs_dict[s_idx]['IWV'] = compute_IWV_q(rs_dict[s_idx]['q'], rs_dict[s_idx]['pres'])
    
    return rs_dict


def import_PS_mastertrack(
    filename,
    keys='all',
    return_DS=False):

    """
    Imports Polarstern master track data during MOSAiC published on PANGAEA. Time
    will be given in seconds since 1970-01-01 00:00:00 UTC and datetime. It also
    returns global attributes in the .tab file so that the information can be
    forwarded to the netcdf version of the master tracks.

    Parameters:
    -----------
    filename : str or list of str
        Filename + path of the Polarstern Track data (.nc).
    keys : list of str
        List of names of variables to be imported. 'all' will import all keys.
        Default: 'all'
    return_DS : bool
        If True, an xarray dataset will be returned (only if type(filename) == list).
    """

    if type(filename) == list:
        DS = xr.open_mfdataset(filename, combine='nested', concat_dim='time')
        if return_DS: return DS

    else:

        file_nc = nc.Dataset(filename)

        if keys == 'all':
            keys = file_nc.variables.keys()

        elif isinstance(keys, str) and (keys != 'all'):
            raise ValueError("Argument 'keys' must either be a string ('all') or a list of variable names.")

        ps_track_dict = dict()
        for key in keys:
            if not key in file_nc.variables.keys():
                raise KeyError("I have no memory of this key: '%s'. Key not found in file '%s'." %(key, filename))
            ps_track_dict[key] = np.asarray(file_nc.variables[key])

        return ps_track_dict


def import_PS_track_txt(filename: str, resample: int):

    """
    Imports Polarstern track data saved as .txt file and returns an xarray.Dataset.

    Parameters:
    -----------
    filename : str
        Filename + path of the Polarstern track data (.txt).
    resample : int
        Whether of not to resample the data to the time resolution in seconds given by resample.
        If 0, no resampling is performed.
    """

    DS = pd.read_csv(filename, sep='\t', header=0, index_col=0)
    DS = DS.to_xarray()
    DS = DS.rename({'Date/Time (UTC)': 'time', 'Flag for data source': 'flag'})
    DS['time'] = DS.time.astype('datetime64[ns]')
    DS = DS.drop_vars('flag')

    if resample:
        DS = DS.resample(time='600s').nearest()

    if "PS144_" in filename:
        DS.attrs['Citation'] = ("Rabe, Benjamin; Geibert, Walter (2024): Master track of POLARSTERN cruise " +
                                "PS144 in 1 sec resolution (zipped, 243 MB) [dataset]. Alfred Wegener " +
                                "Institute, Helmholtz Centre for Polar and Marine Research, Bremerhaven, " +
                                "PANGAEA, https://doi.org/10.1594/PANGAEA.974028")
    elif "PS149_" in filename:
        DS.attrs['Citation'] = ("Nicolaus, Marcel (2025): Master track of POLARSTERN cruise PS149 in 1 " +
                                "sec resolution (zipped, 221 MB) [dataset]. Alfred Wegener Institute, Helmholtz " +
                                "Centre for Polar and Marine Research, Bremerhaven, PANGAEA, " +
                                "https://doi.pangaea.de/10.1594/PANGAEA.987132")

    return DS


def import_cloudnet_product(
    filename,
    keys='basic'):

    """
    Importing CLOUDNET data (classification).

    Parameters:
    -----------
    filename : str
        Path and filename of data.
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    """

    file_nc = nc.Dataset(filename)

    if keys == 'basic': 
        keys = ['time', 'height', 'target_classification']

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    data_dict = dict()
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in cloudnet file." % key)
        data_dict[key] = np.asarray(file_nc.variables[key])


    if 'time' in keys:  # avoid nasty digita after decimal point and convert to seconds since 1970-01-01 00:00:00 UTC
        time_units = file_nc.variables['time'].units
        reftime = dt.datetime.strptime(time_units[12:-6], '%Y-%m-%d %H:%M:%S')
        reftime_epoch = datetime_to_epochtime(reftime)
        data_dict['time'] = np.float64(data_dict['time'])   # for the conversion, I need greater precision
        data_dict['time'] = data_dict['time']*3600 + reftime_epoch

        data_dict['time'] = np.rint(data_dict['time']).astype(float)

    return data_dict


def import_cloudnet_product_daterange(
    path_data,
    date_start,
    date_end,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the cloudnet product data time
    series of each day so that you'll have one dictionary for all data.

    Parameters:
    -----------
    path_data : str
        Base path of level 2a data. This directory contains subfolders representing the year which,
        in turn, contains the daily files. Example path_data:
        "/data/obs/campaigns/mosaic/cloudnet/products/classification/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_hgt = 595         # inquired from cloudnet data, number of model height levels

    # basic variables that should always be imported:
    time_keys = ['time']                # keys with time as coordinate
    height_keys = ['height']                            # keys with height as coordinate
    time_height_keys = ['target_classification']


    # master_dict (output) will contain all desired variables on specific axes:
    master_dict = dict()

    # save import keys in a list:
    import_keys = time_keys + height_keys + time_height_keys

    n_samp_tot = n_days*3000        # number of expected time dimension entries
    for mthk in time_height_keys: master_dict[mthk] = np.full((n_samp_tot, n_hgt), -99)
    for mtkab in time_keys: master_dict[mtkab] = np.full((n_samp_tot,), np.nan)
    for mhk in height_keys: master_dict[mhk] = np.full((n_hgt,), np.nan)

    # cycle through all years, all months and days:
    time_index = 0  # this index will be increased by the length of the time series of the 
                    # current day (now_date) to fill the master_dict time axis accordingly.
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("\rWorking on Cloudnet Product, ", now_date, end="")

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + str(yyyy) + "/"

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # cloudnet file for the current day:
        file_nc = sorted(glob.glob(day_path + "%4i%02i%02i*_classification.nc"%(yyyy,mm,dd)))

        if len(file_nc) == 0:
            if verbose >= 2:
                warnings.warn("No netcdf files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        # select current day and import the data:
        file_nc = file_nc[0]
        data_dict = import_cloudnet_product(file_nc, import_keys)

        # save to master_dict
        for key in data_dict.keys():
            mwr_key_shape = data_dict[key].shape
            n_time = len(data_dict['time'])

            if mwr_key_shape == data_dict['time'].shape:    # then the variable is on time axis:
                master_dict[key][time_index:time_index + n_time] = data_dict[key]

            elif key in height_keys:    # will be handled after the loop
                continue

            elif key in time_height_keys:
                master_dict[key][time_index:time_index + n_time,:] = data_dict[key]

            else:
                raise RuntimeError("Something went wrong in the " +
                    "import_cloudnet_product_daterange routine. Unexpected variable dimension for " + key + ".")

        time_index = time_index + n_time


    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:
        # save height keys to master dict:
        for hkey in height_keys: master_dict[hkey] = data_dict[hkey]

        # truncate the master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(master_dict['time']))[-1][0]
        time_shape_old = master_dict['time'].shape
        time_height_shape_old = master_dict[time_height_keys[0]].shape
        for mwr_key in master_dict.keys():
            shape_new = master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                master_dict[mwr_key] = master_dict[mwr_key][:last_time_step+1]
            elif shape_new == time_height_shape_old:
                master_dict[mwr_key] = master_dict[mwr_key][:last_time_step+1, :]

    if verbose >= 1: print("")

    return master_dict


def import_IDL_TBs_txt(
    filename):

    """
    Imports TBs of radiosondes simulated with the IDL STP tool saved to a .txt file.

    Parameters:
    -----------
    filename : str
        Path and filename of the .txt file.
    """

    # Initialise arrays:
    n_sondes = 1500         # assumed number of sondes for initialisation
    n_freq = 28             # number of frequencies; inquired via inspection of .txt file
    mwr_dict = {'tb': np.full((n_sondes, n_freq), np.nan),
                'time': np.zeros((n_sondes,)),
                'lwp': np.zeros((n_sondes,))}

    # Open file:
    f_handler = open(filename, 'r')
    list_of_lines = list()

    header = 13     # linenumber header+1 is the first data line

    m = 0       # used as index to save the entries into rs_dict; will increase for each new radiosonde
    # precursor_event = ''
    for k, line in enumerate(f_handler):

        current_line = line.strip().split()     # split by removing all spaces

        if k == 12:     # frequency def.
            mwr_dict['freq'] = np.asarray([float(freq) for freq in current_line])

        if k > header:      # skip header
            mwr_dict['time'][m] = datetime_to_epochtime(dt.datetime.strptime(current_line[0], "%Y%m%d%H"))
                                                    # marks the radiosonde launch; in yyyymmddhh; must be converted to
                                                    # seconds since 1970-01-01 00:00:00 UTC
            mwr_dict['lwp'][m] = current_line[1]    # LWP in kg m^-2
            mwr_dict['tb'][m,:] = current_line[3:]
            m = m + 1

    
    # finally truncate unneccessary time dimension for each sonde and compute IWV:
    mwr_dict['tb'] = mwr_dict['tb'][:m,:]
    mwr_dict['time'] = mwr_dict['time'][:m]
    mwr_dict['lwp'] = mwr_dict['lwp'][:m]

    return mwr_dict


def import_synthetic_TBs(
    files):

    """
    Imports simulated TBs (must be .nc) from a certain site (e.g. Ny Alesund radiosondes)
    saved in a format readable by the mwr_pro retrieval.
    Also converts time to sec since 1970-01-01 00:00:00 UTC.

    Parameters:
    -----------
    files : str or list of str
        Path in which all files are selected to be imported or list of files
            to be imported.
    """

    if type(files) == str:
        files = sorted(glob.glob(files + "*.nc"))

    DS = xr.open_mfdataset(files, concat_dim='n_date', combine='nested',
                                preprocess=syn_MWR_cut_useless_variables_TB)

    # convert time:
    time_dt = np.asarray([dt.datetime.strptime(str(tttt), "%Y%m%d%H") for tttt in DS.date.values])
    time = datetime_to_epochtime(time_dt)
    DS['time'] = xr.DataArray(time, dims=['n_date'])

    # Cut unwanted dimensions in variables 'frequency' and 'elevation_angle':
    DS['frequency'] = DS.frequency[0,:]
    DS['elevation_angle'] = DS.elevation_angle[0]

    return DS


def import_mwr_pro_radiosondes(
    files,
    **kwargs):

    """
    Imports radiosonde (or radiosonde-like) data (must be .nc) from a certain site 
    (e.g. Ny Alesund radiosondes) saved in a format readable by the mwr_pro retrieval. 
    Also converts time to sec since 1970-01-01 00:00:00 UTC and computes specific 
    and relative humidity from absolute humidity. 

    Parameters:
    -----------
    files : str or list of str
        Path in which all files are selected to be imported or list of files
            to be imported.

    **kwargs:
    with_lwp : bool
        If True, the sonde dict will also include liquid water path.
    """

    if type(files) == str:
        files = sorted(glob.glob(files + "*.nc"))

    DS = xr.open_mfdataset(files, concat_dim='n_date', combine='nested',
                                preprocess=syn_MWR_cut_useless_variables_RS)

    # convert time:
    time_dt = np.asarray([dt.datetime.strptime(str(tttt), "%Y%m%d%H") for tttt in DS.date.values])
    time = datetime_to_epochtime(time_dt)
    n_time = len(time)
    DS['time'] = xr.DataArray(time, dims=['n_date'])

    # compute specific and relative humidity:
    spec_hum = convert_abshum_to_spechum(DS.atmosphere_temperature.values, DS.atmosphere_pressure.values, 
                                            DS.atmosphere_humidity.values)
    spec_hum_sfc = convert_abshum_to_spechum(DS.atmosphere_temperature_sfc.values, DS.atmosphere_pressure_sfc.values, 
                                            DS.atmosphere_humidity_sfc.values)
    DS['atmosphere_spec_humidity'] = xr.DataArray(spec_hum, dims=['n_date', 'n_height'])
    DS['atmosphere_spec_humidity_sfc'] = xr.DataArray(spec_hum_sfc, dims=['n_date'])

    rel_hum = convert_abshum_to_relhum(DS.atmosphere_temperature.values, DS.atmosphere_humidity.values)
    rel_hum_sfc = convert_abshum_to_relhum(DS.atmosphere_temperature_sfc.values, DS.atmosphere_humidity_sfc.values)
    DS['atmosphere_rel_humidity'] = xr.DataArray(rel_hum, dims=['n_date', 'n_height'])
    DS['atmosphere_rel_humidity_sfc'] = xr.DataArray(rel_hum_sfc, dims=['n_date'])


    # convert to dict:
    sonde_dict = {  'pres': DS.atmosphere_pressure.values,      # in Pa, time x height
                    'temp': DS.atmosphere_temperature.values,   # in K
                    'rh': DS.atmosphere_rel_humidity.values,    # between 0 and 1
                    'height': DS.height_grid.values,            # in m; also time x height
                    'rho_v': DS.atmosphere_humidity.values,     # in kg m^-3
                    'q': DS.atmosphere_spec_humidity.values,    # in kg kg^-1
                    'wspeed': np.zeros(DS.atmosphere_temperature.shape),    # in m s^-1; is 0 because unknown
                    'wdir': np.zeros(DS.atmosphere_temperature.shape),  # in deg; is 0 because unknown
                    'lat': np.repeat(DS.latitude, n_time),      # in deg N
                    'lon': np.repeat(DS.longitude, n_time),     # in deg E
                    'launch_time': time,                        # in sec since 1970-01-01 00:00:00 UTC
                    'iwv': DS.integrated_water_vapor.values}    # in kg m^-2

    if kwargs['with_lwp']:
        sonde_dict['lwp'] = DS.liquid_water_path.values         # in kg m^-2

    return sonde_dict


def import_MiRAC_outliers(filename):

    """
    Import and convert manually (per eye) detected outliers of MiRAC-P 
    to an array filled with datetime.

    Parameters:
    -----------
    filename : str
        Filename (including path) of the text file (.txt) that contains
        the outliers.
    """

    headersize = 1
    file_handler = open(filename, 'r')
    list_of_lines = list()

    for line in file_handler:
        current_line = line.strip().split('   ')    # split by 3 spaces
        list_of_lines.append(current_line)

    # delete header:
    list_of_lines = list_of_lines[headersize:]
    n_outliers = len(list_of_lines)         # number of outliers

    # read start and end time of an outlier from a line:
    list_outliers = []
    for ix, line in enumerate(list_of_lines):
        list_outliers.append([dt.datetime.strptime(line[0], "%Y %m %d %H %M"),
                            dt.datetime.strptime(line[1], "%Y %m %d %H %M")])

    return list_outliers


def import_mirac_MET(
    filename,
    keys='basic'):

    """
    Importing automatically created MiRAC-P MET hourly files
    with the ending .MET.NC in the level 1 folder. Time will be
    converted to seconds since 1970-01-01 00:00:00 UTC.
    For now, only surface pressure (in Pa) will be loaded.

    Parameters:
    -----------
    filename : str
        Path and filename of MiRAC-P .MET.NC data.
    keys : list of str or str, optional
        Specify which variables are to be imported. Another option is
        to import all keys (keys='all') or import basic keys
        that the author considers most important (keys='basic')
        or leave this argument out.
    """


    file_nc = nc.Dataset(filename)

    if keys == 'basic':
        keys = ['time', 'RF', 'Surf_P']

    elif keys == 'all':
        keys = file_nc.variables.keys()

    elif isinstance(keys, str) and ((keys != 'all') and (keys != 'basic')):
        raise ValueError("Argument 'keys' must either be a string ('all' or 'basic') or a list of variable names.")

    mwr_dict = dict()
    reftime = dt.datetime(1970,1,1,0,0,0)
    for key in keys:
        if not key in file_nc.variables.keys():
            raise KeyError("I have no memory of this key: '%s'. Key not found in MiRAC-P .MET.NC file." % key)

        mwr_dict[key] = np.asarray(file_nc.variables[key])

        if key == 'time':   # convert to sec since 1970-01-01 00:00:00 UTC (USE FLOAT64)
            mwr_dict['time'] = (np.float64(datetime_to_epochtime(dt.datetime(2001,1,1,0,0,0))) +
                                mwr_dict[key].astype(np.float64))

    if "Surf_P" in mwr_dict.keys():
        mwr_dict['pres'] = mwr_dict['Surf_P']*100       # convert to Pa

    return mwr_dict


def import_mirac_MET_RPG_daterange(
    path_data,
    date_start,
    date_end,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the MET time
    series of each day so that you'll have one dictionary that will contain the pressure data
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Base path of MiRAC-P level 1 data. This directory contains subfolders representing the 
        year, which, in turn, contain months, which contain day subfolders. Example:
        path_data = "/data/obs/campaigns/mosaic/mirac-p/l1/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1

    # basic variables that should always be imported:
    mwr_time_keys = ['time', 'RF', 'pres']  # keys with time as coordinate

    # mwr_master_dict (output) will contain all desired variables on time axis for entire date range:
    mwr_master_dict = dict()
    n_seconds = n_days*86400        # max number of seconds: n_days*86400
    for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_seconds,), np.nan)


    # Load the data into mwr_master_dict:
    # cycle through all years, all months and days:
    time_index = 0  # this index will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    day_index = 0   # will increase for each day
    for now_date in (date_start + dt.timedelta(days=n) for n in range(n_days)):

        if verbose >= 1: print("Working on RPG retrieval, MiRAC-P MET, ", now_date)

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day

        day_path = path_data + "%04i/%02i/%02i/"%(yyyy,mm,dd)

        if not os.path.exists(os.path.dirname(day_path)):
            continue

        # list of .MET.NC files: Sorting is important as this will
        # ensure automatically that the time series of each hour will
        # be concatenated appropriately!
        mirac_nc = sorted(glob.glob(day_path + "*.MET.NC"))
        if len(mirac_nc) == 0:
            if verbose >= 2:
                warnings.warn("No .MET.NC files found for date %04i-%02i-%02i."%(yyyy,mm,dd))
            continue

        # load one retrieved variable after another from current day and save it into the mwr_master_dict
        for l1_file in mirac_nc: 
            mwr_dict = import_mirac_MET(l1_file)

            n_time = len(mwr_dict['time'])
            time_shape = mwr_dict['time'].shape

            # save to mwr_master_dict
            for mwr_key in mwr_dict.keys():
                if mwr_key in mwr_time_keys:
                    mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key]


            time_index = time_index + n_time
        day_index = day_index + 1


    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))

    else:

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        for mwr_key in mwr_master_dict.keys():
            shape_new = mwr_master_dict[mwr_key].shape
            if shape_new == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]

    return mwr_master_dict


def import_mvr_pro_rt(
    files,
    keys='all',
    instrument='hatpro',
    ele_angles='all'):

    """
    Import input files of mvr_pro program that creates retrieval files readable
    for MWR_PRO. Usually, multiple years are supposed to be imported which is
    why the input 'files' cannot just be one file. Variables will be converted 
    to SI units. Time will be given in seconds since 1970-01-01 00:00:00 UTC.

    Parameters:
    -----------
    files : str or list of str
        Path in which all files are selected to be imported or list of files
            to be imported.
    keys : list of str or str
        List of variable names as str that are supposed to be imported. Can also
        be a str for special cases. By default, all variables will be imported. 
        Default: 'all'
    instrument : str
        Specifies which instrument is considered. 'hatpro' and 'mirac-p' are the only
        valid options. They differ in the utilized elevation angles. Default: 'hatpro'
    """

    if type(files) == str:
        files = sorted(glob.glob(files + "*.nc"))

    DS = xr.open_mfdataset(files, concat_dim='n_date', combine='nested')

    # check which variables can be dropped:
    if (isinstance(keys, str)) and (keys != 'all'):
        raise ValueError("Argument 'keys' in import_mvr_pro_rt must be 'all' or a list of str.")

    elif isinstance(keys, list):
        useless_vars = list()
        for var in DS.data_vars:
            if not var in keys:
                useless_vars.append(var)

        DS = DS.drop_vars(useless_vars)

    # convert time:
    time_dt = np.asarray([dt.datetime.strptime(str(tttt), "%Y%m%d%H") for tttt in DS.date.values])
    time = datetime_to_epochtime(time_dt)
    DS['time'] = xr.DataArray(time, dims=['n_date'])

    # select the right elevation angles: don't interpolate (even if the don't fit because that would
    # have decreased the retrieval accuracy in MVR_PRO. So interpolated was also not performed there.
    if instrument == 'hatpro':
        if ele_angles != 'all':
            ele = np.array([90.0, 30.0, 19.2, 14.4, 11.4, 8.4, 6.6, 5.4])   # inquired from par_mvr_pro.pro
        else:
            ele = np.arange(90., 4.9, -1.0)
    elif instrument == 'mirac-p':
        ele = np.array([90.0])                                          # inquired from par_mvr_pro.pro
    else:
        raise ValueError("Argument 'instrument' in import_mvr_pro_rt must be either 'hatpro' or 'mirac-p'.")


    idx_ele = np.zeros((len(ele),), dtype=np.short)
    for ii, ee in enumerate(ele):
        ix = np.where((DS.elevation_angle.values[0,:] > ee - 0.6) & (DS.elevation_angle.values[0,:] < ee + 0.6))[0]
        
        if len(ix) > 0:
            idx_ele[ii] = np.argmin(np.abs(DS.elevation_angle.values[0,:] - ee))
    
    # reduce dataset to selected elev angles, the only cloud model, and reduce some redundant data array dims:
    DS = DS.isel(n_angle=idx_ele, n_cloud_model=0)
    DS['frequency'] = DS.frequency[0,:]     # doesn't change over time
    DS['elevation_angle'] = DS.elevation_angle[0,:]     # invariant over time
    DS['height_grid'] = DS.height_grid[0,:]             # invariant over time
    # Convert to SI units: (not required because all vars are in SI units
    
    return DS


def import_mirac_temp(
    path_data,
    date_start,
    date_end,
    which_retrieval='both',
    vers='v01',
    minute_avg=False,
    verbose=0):

    """
    Runs through all days between a start and an end date. It concats the level 2a data time
    series of each day so that you'll have one dictionary, whose e.g. 'IWV' will contain the IWV
    for the entire date range period.

    Parameters:
    -----------
    path_data : str
        Base path of level 2a data. This directory contains subfolders representing the year, which,
        in turn, contain months, which contain day subfolders. Example path_data:
        "/data/obs/campaigns/mosaic/mirac-p/l2/"
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. 'both' will 
        load both. Default: 'both'
    vers : str
        Indicates the mwr_pro output version number. Valid options: 'v01'
    minute_avg : bool
        If True: averages over one minute are computed and returned instead of False when all
        data points are returned (more outliers, higher memory usage).
    verbose : int
        If 0, output is suppressed. If 1, basic output is printed. If 2, more output (more warnings,...)
        is printed.
    """

    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'clwvi', 'lwp', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                "'both' will load both. Default: 'both'")

        else:
                if which_retrieval == 'iwv':
                    which_retrieval = ['prw']
                elif which_retrieval == 'lwp':
                    which_retrieval = ['clwvi']
                elif which_retrieval == 'both':
                    which_retrieval = ['prw', 'clwvi']
                else:
                    raise ValueError("Argument '" + which_retrieval + "' not recognised. Please use one of the following options: " +
                        "'iwv' or 'prw' will load the " +
                        "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                        "'both' will load both. Default: 'both'")
                    

    # extract day, month and year from start date:
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")


    # count the number of days between start and end date as max. array size:
    n_days = (date_end - date_start).days + 1
    n_ret = 86          # inquired from level 2a data, number of available elevation angles in retrieval

    # basic variables that should always be imported:
    mwr_time_keys = ['time', 'azi', 'ele', 'flag', 'lat', 'lon', 'zsl']             # keys with time as coordinate

    # mwr_master_dict (output) will contain all desired variables on specific axes:
    # e.g. level 2a and 2b have got the same time axis (according to pl_mk_nds.pro)
    # and azimuth and elevation angles.
    mwr_master_dict = dict()
    if minute_avg:  # max number of minutes: n_days*1440
        n_minutes = n_days*1440
        for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_minutes,), np.nan)

        if 'prw' in which_retrieval:
            mwr_master_dict['prw'] = np.full((n_minutes,), np.nan)

        if 'clwvi' in which_retrieval:
            mwr_master_dict['clwvi'] = np.full((n_minutes,), np.nan)

    else:           # max number of seconds: n_days*86400
        n_seconds = n_days*86400
        for mtkab in mwr_time_keys: mwr_master_dict[mtkab] = np.full((n_seconds,), np.nan)

        if 'prw' in which_retrieval:
            mwr_master_dict['prw'] = np.full((n_seconds,), np.nan)

        if 'clwvi' in which_retrieval:
            mwr_master_dict['clwvi'] = np.full((n_seconds,), np.nan)


    # cycle through all years, all months and days:
    time_index = 0  # this index (for lvl 2a) will be increased by the length of the time
                        # series of the current day (now_date) to fill the mwr_master_dict time axis
                        # accordingly.
    day_index = 0   # same as above, but only increases by 1 for each day

    # list of files:
    mirac_level2a_nc = sorted(glob.glob(path_data + "*.nc"))

    # load one retrieved variable after another from current day and save it into the mwr_master_dict
    for lvl2_nc in mirac_level2a_nc: 
        mwr_dict = import_mirac_level2a(lvl2_nc, minute_avg=minute_avg)

        n_time = len(mwr_dict['time'])
        cur_time_shape = mwr_dict['time'].shape

        # save to mwr_master_dict
        for mwr_key in mwr_dict.keys():
            mwr_key_shape = mwr_dict[mwr_key].shape
            if mwr_key_shape == cur_time_shape: # then the variable is on time axis:
                mwr_master_dict[mwr_key][time_index:time_index + n_time] = mwr_dict[mwr_key]

            elif len(mwr_dict[mwr_key]) == 1:
                mwr_master_dict[mwr_key][day_index:day_index + 1] = mwr_dict[mwr_key]

            else:
                raise RuntimeError("Something went wrong in the " +
                    "import_mirac_level2a_daterange routine. Unexpected MWR variable dimension. " + 
                    "The length of one used variable ('%s') of level 2a data "%(mwr_key) +
                        "neither equals the length of the time axis nor equals 1.")


        time_index = time_index + n_time
        day_index = day_index + 1

    if time_index == 0 and verbose >= 1:    # otherwise no data has been found
        raise ValueError("No data found in date range: " + dt.datetime.strftime(date_start, "%Y-%m-%d") + " - " + 
                dt.datetime.strftime(date_end, "%Y-%m-%d"))
    else:

        # truncate the mwr_master_dict to the last nonnan time index:
        last_time_step = np.argwhere(~np.isnan(mwr_master_dict['time']))[-1][0]
        time_shape_old = mwr_master_dict['time'].shape
        for mwr_key in mwr_master_dict.keys():
            if mwr_master_dict[mwr_key].shape == time_shape_old:
                mwr_master_dict[mwr_key] = mwr_master_dict[mwr_key][:last_time_step+1]

    return mwr_master_dict


def import_DSHIP(
    file,
    vars=['date time', 'SYS.STR.Course', 'SYS.STR.HDG', 'SYS.STR.PosAlt', 
            'SYS.STR.PosLat', 'SYS.STR.PosLon', 'SYS.STR.Speed', 
            'NACOS.RAOSD.system_heading']):

    """
    Imports DSHIP data (from .dat file) into a dictionary where the key indicates the variable name
    and the value equals the time series (1D array).

    Parameters:
    -----------
    file : str
        Path and filename of the DSHIP file.
     vars : list
        List of variables to import. Please check the DSHIP file header manually.
        Importing all variables consumes about 10 GB of memory.
    """

    data_dict = dict()              ######### maybe as xarray?
    for var_name in vars: data_dict[var_name] = []      # empty strings

    # open and read first line to extract var names:
    with open(file, 'r', encoding='utf-8', errors='ignore') as f_handler:

        for k, line in enumerate(f_handler):

            if k == 0:  # first line: variable names:
                var_names = line.split("\t")

            elif k == 1: 
                continue

            elif k == 2:
                units = line.split("\t")        # but corrupted due to utf-8 decoding errors ignored

            else:
                data_list = line.split("\t")

                # leave it all as string for now and convert later:
                for var_name, data in zip(var_names, data_list):
                    if var_name in vars:
                        data_dict[var_name].append(data)


    # convert data variables to something useful:
    for var_name in var_names:
        if var_name in vars:
            if var_name in ['date time']:
                data_dict[var_name] = [ddd.replace("/", "-") for ddd in data_dict[var_name]]
                data_dict[var_name] = np.asarray(data_dict[var_name], dtype="datetime64[s]")

            elif var_name in ['SYS.STR.Course', 'SYS.STR.HDG', 'SYS.STR.Speed', 'NACOS.RAOSD.system_heading']:
                data_dict[var_name] = np.asarray(data_dict[var_name], dtype=np.float32)

            elif var_name in ['SYS.STR.PosAlt']:
                data_dict[var_name] = np.asarray(data_dict[var_name], dtype=np.int16)

            elif var_name in ['SYS.STR.PosLat', 'SYS.STR.PosLon']:
                for k, dat in enumerate(data_dict[var_name]):
                    dat = dat.split(" ")

                    if len(dat) != 3:       # missing data
                        dec_deg = np.nan

                    else:
                        deg, dec_min, sign = float(dat[0]), float(dat[1][:-1]), dat[2]

                        # convert from degrees decimal minutes to decimal degrees:
                        dec_deg = deg + dec_min / 60.0
                        if sign in ["S", "W"]:
                            dec_deg = -1.0*dec_deg

                    data_dict[var_name][k] = str(dec_deg)

                data_dict[var_name] = np.asarray(data_dict[var_name], dtype=np.float32)

    return data_dict


def import_DSHIP_csv(
    file,
    vars=['date time', 'SYS.STR.Course', 'SYS.STR.HDG', 'SYS.STR.PosLat', 
            'SYS.STR.PosLon', 'SYS.STR.Speed']):

    """
    Imports DSHIP data (from .csv file) into a dictionary where the key indicates the variable name
    and the value equals the time series (1D array).

    Parameters:
    -----------
    file : str
        Path and filename of the DSHIP file.
     vars : list
        List of variables to import. Please check the DSHIP file header manually.
        Importing all variables consumes about 10 GB of memory.
    """

    data_dict = dict()              ######### maybe as xarray?
    for var_name in vars: data_dict[var_name] = []      # empty strings

    # open the csv and read it:
    with open(file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, dialect='excel', delimiter=',')
        list_of_lines = list()
        for row in csv_reader:
            list_of_lines.append(row)

    # loop through all lines to extract data and meta data:
    var_names = list()
    units = list()
    for k, line in enumerate(list_of_lines):

        if k == 0:      # first line: variable names:
            var_names = line

        elif k == 1:
            continue

        elif k == 2:    # units
            units = line

        else:
            data_list = line

            # leave data as string for now and convert later:
            for var_name, data in zip(var_names, data_list):
                if var_name in vars:
                    data_dict[var_name].append(data)


    # convert data variables to something useful:
    for var_name in var_names:
        if var_name in vars:
            if var_name in ['date time']:
                data_dict[var_name] = [ddd.replace("/", "-") for ddd in data_dict[var_name]]
                data_dict[var_name] = np.asarray(data_dict[var_name], dtype="datetime64[s]")

            elif var_name in ['SYS.STR.Course', 'SYS.STR.HDG', 'SYS.STR.Speed', 'NACOS.RAOSD.system_heading',
                'WEATHER.PAWIBWW.air_temperature', 'WEATHER.PAWIBWW.dewpoint', 'WEATHER.PAWIBWW.rel_humidity']:
                data_dict[var_name] = np.asarray(data_dict[var_name], dtype=np.float32)

            elif var_name in ['SYS.STR.PosAlt']:
                data_dict[var_name] = np.asarray(data_dict[var_name], dtype=np.int16)

            elif var_name in ['SYS.STR.PosLat', 'SYS.STR.PosLon']:
                for k, dat in enumerate(data_dict[var_name]):
                    dat = dat.split(" ")

                    if len(dat) != 3:       # missing data
                        dec_deg = np.nan

                    else:
                        # eventually, dat[0] contains a degree symbol: replace it
                        deg_utf8 = b"\xc2\xb0"
                        if deg_utf8 in dat[0].encode('utf-8'): 
                            dat[0] = dat[0].replace("°", "")
                        deg, dec_min, sign = float(dat[0]), float(dat[1][:-1]), dat[2]

                        # convert from degrees decimal minutes to decimal degrees:
                        dec_deg = deg + dec_min / 60.0
                        if sign in ["S", "W"]:
                            dec_deg = -1.0*dec_deg

                    data_dict[var_name][k] = str(dec_deg)

                data_dict[var_name] = np.asarray(data_dict[var_name], dtype=np.float32)

    return data_dict


def import_iasi_nc(
    files):

    """
    Import IASI file(s) and return as xarray dataset. Also cuts irrelevant variables and
    dimensions. Also the time needs to be decoded manually.

    Parameters:
    -----------
    files : list of str
        Path + filenames of the netCDF files to be imported. 
    """

    def cut_vars(DS):

        unwanted_vars = ['forli_layer_heights_co', 'forli_layer_heights_hno3', 'forli_layer_heights_o3', 'brescia_altitudes_so2', 'solar_zenith', 
                    'satellite_zenith', 'solar_azimuth', 'satellite_azimuth', 'fg_atmospheric_ozone', 'fg_qi_atmospheric_ozone', 'atmospheric_ozone', 
                    'integrated_ozone', 'integrated_n2o', 'integrated_co', 'integrated_ch4', 'integrated_co2', 'surface_emissivity', 
                    'number_cloud_formations', 'fractional_cloud_cover', 'cloud_top_temperature', 'cloud_top_pressure', 'cloud_phase', 
                    'instrument_mode', 'spacecraft_altitude', 'flag_cdlfrm', 'flag_cdltst', 'flag_daynit', 'flag_dustcld', 
                    'flag_numit', 'flag_nwpbad', 'flag_physcheck', 'flag_satman', 'flag_sunglnt', 'flag_thicir', 'nerr_values', 'error_data_index', 
                    'temperature_error', 'water_vapour_error', 'ozone_error', 'co_qflag', 'co_bdiv', 'co_npca', 'co_nfitlayers', 'co_nbr_values', 
                    'co_cp_air', 'co_cp_co_a', 'co_x_co', 'co_h_eigenvalues', 'co_h_eigenvectors', 'hno3_qflag', 'hno3_bdiv', 'hno3_npca', 
                    'hno3_nfitlayers', 'hno3_nbr_values', 'hno3_cp_air', 'hno3_cp_hno3_a', 'hno3_x_hno3', 'hno3_h_eigenvalues', 'hno3_h_eigenvectors',
                    'o3_qflag', 'o3_bdiv', 'o3_npca', 'o3_nfitlayers', 'o3_nbr_values', 'o3_cp_air', 'o3_cp_o3_a', 'o3_x_o3', 'o3_h_eigenvalues', 
                    'o3_h_eigenvectors', 'so2_qflag', 'so2_col_at_altitudes', 'so2_altitudes', 'so2_col', 'so2_bt_difference', 'fg_surface_temperature',
                    'fg_qi_surface_temperature', 'surface_temperature']
        remaining_unwanted_dims = ['cloud_formations', 'nlo', 'new']

        # check if unwanted_vars exist in the dataset (it may happen that not all variables 
        # exist in all IASI files):
        uv_exist = np.full((len(unwanted_vars),), False)
        for i_u, u_v in enumerate(unwanted_vars):
            if u_v in DS.variables:
                uv_exist[i_u] = True

        DS = DS.drop_vars(np.asarray(unwanted_vars)[uv_exist])
        DS = DS.drop_dims(remaining_unwanted_dims)
        return DS

    DS = xr.open_mfdataset(files, combine='nested', concat_dim='along_track', preprocess=cut_vars, decode_times=False)

    # decode time:
    reftime = np.datetime64("2000-01-01T00:00:00").astype('float32')    # in sec since 1970-01-01 00:00:00 UTC
    record_start_time_npdt = (DS.record_start_time.values + reftime).astype('datetime64[s]')
    record_stop_time_npdt = (DS.record_stop_time.values + reftime).astype('datetime64[s]')
    DS['record_start_time'] = xr.DataArray(record_start_time_npdt, dims=['along_track'], attrs={'long_name': "Record start time", 
                                            'units': "seconds since 1970-01-01 00:00:00 UTC"})
    DS['record_stop_time'] = xr.DataArray(record_stop_time_npdt, dims=['along_track'], attrs={'long_name': "Record stop time", 
                                            'units': "seconds since 1970-01-01 00:00:00 UTC"})

    # also rename integrated water vapour variable:
    DS = DS.rename({'integrated_water_vapor': 'iwv'})

    return DS


def import_iasi_step1(
    path_data,
    date_0="",
    date_1=""):

    """
    Imports IASI step 1 data processed with manage_iasi.py that lie within the path given. The data will be 
    returned as xarray dataset sorted by time. The second dimension 'n_points' will be truncated as far as possible. 
    The initial size of this dimension was only a proxy to cover as many identified IASI pixels for one 
    Polarstern track time step as possible. Also a date range can be specified to improve importing performance
    by loading less data.

    Parameters:
    -----------
    path_data : str
        Data path as string indicating the location of the processed IASI files. No subfolders will be searched.
    date_0 : str
        String indicating the start date of the IASI step 1 data to be imported in "yyyy-mm-dd", e.g., 
        "2019-12-25".
    date_1 : str
        String indicating the end date of the IASI step 1 data to be imported in "yyyy-mm-dd", e.g., 
        "2019-12-31". The end date is included in the data.
    """

    # identify files within daterange:
    files = sorted(glob.glob(path_data + "*.nc"))
    if date_0 and date_1:
        date_range = np.arange(np.datetime64(f"{date_0}"), np.datetime64(f"{date_1}") + np.timedelta64(1, "D"), 
                                np.timedelta64(1, "D"))

        # loop over date range and identify files:
        files_list = list()
        for c_date in date_range:
            c_d_str = str(c_date)
            c_files = [file for file in files if c_d_str.replace("-","") in file]
            files_list += c_files
        files = files_list
        
    DS = xr.open_mfdataset(files, concat_dim='time', combine='nested')


    # adjustment of time; and constrain to the specified date range:
    DS = DS.sortby(DS.time)

    # check if time duplicates exist:
    if np.any(np.diff(DS.time.values) == np.timedelta64(0, "ns")):
        idx_duplo = np.where(np.diff(DS.time.values) == np.timedelta64(0, "ns"))[0]
        n_time = len(DS.time)
        if len(idx_duplo) / n_time > 0.05: pdb.set_trace() # too many duplicates
        okay_mask = np.full((len(DS.time),), True)      # if True, index is to be used, if False, index will be discarded


        # take the one with more hits:
        for i_d in idx_duplo:
            i_d_p = i_d + 1     # got the same time as index i_d
            # set the index of okay_mask False which has less hits
            if np.count_nonzero(~np.isnan(DS.lat[i_d,:])) < np.count_nonzero(~np.isnan(DS.lat[i_d_p,:])):
                okay_mask[i_d] = False
            else:
                okay_mask[i_d_p] = False        


    # check how much of the dimension 'n_hits' is really needed: sufficient to check only one 
    # of the variables with this second dimension because all others use this dimension in the same way:
    max_n_points = -1
    n_hits_max = len(DS.n_hits)

    # loop through all columns (n_hits dimension) and check if data still exists:
    kk = 0
    is_something_here = True        # will check if data is present at index kk of 
                                    # dimension n_hits at any time step
    while is_something_here and (kk < n_hits_max):
        is_something_here = np.any(~np.isnan(DS['lat'].values[:,kk]))
        if is_something_here: kk += 1

    max_n_points = kk

    # truncate the second dimension and removee time indices without data:
    DS = DS.sel(n_hits=slice(None,max_n_points-1))

    DS = DS.load()


    # check if pressure_levels_temp or pressure_levels_humidity have more than 1 dimension:
    # if more than 1 dim exists, it's probably a time dimension that can be eradicated.
    def check_pressure_levels_dim(DS, level_coords):
        max_std_idx = DS[level_coords].std('time').argmax()
        max_diff_coords = DS[level_coords].max('time')[max_std_idx] - DS[level_coords].min('time')[max_std_idx]

        # if the variation of the pressure level coord is small (i.e., less than 1.0 Pa), the
        # time dependence can be ignored by selecting the median value:
        attr_dict = DS[level_coords].attrs              # to avoid losing the attributes
        if max_diff_coords < 1.0:
            DS[level_coords] = DS[level_coords].median('time')
            DS[level_coords].attrs = attr_dict

        return DS


    if DS.pressure_levels_temp.values.ndim > 1:
        DS = check_pressure_levels_dim(DS, 'pressure_levels_temp')
    if DS.pressure_levels_humidity.values.ndim > 1:
        DS = check_pressure_levels_dim(DS, 'pressure_levels_humidity')

    return DS


def import_iasi_processed(
    path_data, 
    date_start,
    date_end):

    """
    Imports IASI data processed with manage_iasi.py AND manage_iasi_step2.py for a specific date range 
    (date_start - date_end) as xarray dataset. The dimension n_hits indicates how many pixels were 
    found in a specific spatio-temporal constraint. 

    Parameters:
    -----------
    path_data : str
        Data path as string indicating the location of the processed IASI files. No subfolders will be searched.
    date_start : str
        Marks the first day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    """

    # identify files in daterange:
    files = sorted(glob.glob(path_data + "*.nc"))
    date_start_dt = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end_dt = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # run through list: identify where date is written and check if within date range:
    files_filtered = list()
    for file in files:
        file_date = dt.datetime.strptime(file[-11:-3], "%Y%m%d")
        if (file_date >= date_start_dt) and (file_date <= date_end_dt):
            files_filtered.append(file)
        
    
    # import:
    DS = xr.open_mfdataset(files_filtered, concat_dim='time', combine='nested')

    return DS


def read_iasi_step2(path: str, file_pattern: str, date_range: np.ndarray, pres_sfc=None, RS_DS=None):
    
    """
    Import IASI data for a specific date range that has been processed with manage_iasi.py AND 
    manage_iasi_step2.py. If radiosonde data (RS_DS) is provided, it will be used to compute a  
    height grid for IASI profile data, which is available on pressure levels only. The 
    radiosonde data must have xr.DataArrays of pressurve ("pres") on ("launch_time","height")
    and height ("height") on ("height",) dimensions. Otherwise, please provide the surface
    pressure (pres_sfc), which allows to mask pressure levels of the IASI data located below
    the surface.
    
    Parameters:
    -----------
    path : str
        Full path of the IASI step2 data.
    file_pattern : str
        File pattern to apply a glob.glob search for files.
    date_range : np.ndarray
        Array of np.datetime64 objects covering all days of the date range.
    pres_sfc : float or np.ndarray
        Surface air pressure in Pa that can be used to improve the calculated height axis from the 
        pressure levels.
    RS_DS : xr.Dataset
        Radiosonde dataset. Must contain xr.DataArrays of pressurve ("pres") on 
        ("launch_time","height") and height ("height") on ("height",) dimensions.
    """
    
    def first_guess_and_OE_retrieval_flags(DS: xr.Dataset):
        
        """
        Check some more quality flags regarding the first guess (piecewise linear regression) retrieval
        and the physical (optimal estimation) retrieval using the flags flag_fgcheck and flag_retcheck.
        Because several IASI pixels in the vicinity of Polarstern were considered (nhits dimension), 
        the quality flag check must be performed for all pixels. Here, the quality flag of all pixels
        must be good. The flags need to be converted to bit strings to check for the flagged value.

        Parameters:
        -----------
        DS : xarray Dataset
            Dataset containing IASI data (processed with manage_iasi_step2.py) and quality flags.
        """
        
        flag_okay = {'flag_fgcheck': np.full((len(DS.time),), False),
                        'flag_retcheck': np.full((len(DS.time),), False)}
        for flag_value in ['flag_fgcheck', 'flag_retcheck']:
            for k in range(len(DS.time)):
                nonnan_flag = ~np.isnan(DS[flag_value][k,:].values)
                flag_uint = DS[flag_value][k,nonnan_flag].values.astype(np.uint16)
                flag_str = np.asarray([np.binary_repr(val, width=16)[::-1] for val in flag_uint]) # inverting string to have bit 1 at the front
                
                # set flag to okay if water vapour flag is not active:
                flag_okay[flag_value][k] = np.all(np.asarray([flagcheck[1] for flagcheck in flag_str]) == "0")
        
        flag_okay_fg_DA = xr.DataArray(flag_okay['flag_fgcheck'], dims=['time'], coords={'time': (['time'], DS.time.values)})
        flag_okay_ret_DA = xr.DataArray(flag_okay['flag_retcheck'], dims=['time'], coords={'time': (['time'], DS.time.values)})
        
        return flag_okay_fg_DA, flag_okay_ret_DA


    def get_iasi_quality_flags(DS: xr.Dataset):
        
        flag_okay_fg_DA, flag_okay_ret_DA = first_guess_and_OE_retrieval_flags(DS)
        flag_itconv = ~np.any(DS.flag_itconv < 5, axis=1)
        
        return flag_okay_fg_DA, flag_okay_ret_DA, flag_itconv


    def apply_flags_and_split_first_guess_and_OE(
        DS: xr.Dataset, 
        var: str,
        reasonable_value_limits: tuple,
        flag_okay_fg_DA: xr.DataArray,
        flag_okay_ret_DA: xr.DataArray, 
        flag_itconv: xr.DataArray):
        
        if var == 'temp':
            var = "atmospheric_temperature_mean"
        elif var == 'q':
            var = "atmospheric_water_vapor_mean"

        data_in_limits_mask = (DS[var] > reasonable_value_limits[0]) & (DS[var] < reasonable_value_limits[1])
        fg_data_in_limits_mask = (DS["fg_"+var] > reasonable_value_limits[0]) & (DS["fg_"+var] < reasonable_value_limits[1])
        iasi_data_okay = xr.where(data_in_limits_mask & flag_okay_ret_DA & flag_itconv,
                                    x=DS[var], y=np.nan)
        iasi_data_fg_okay = xr.where(fg_data_in_limits_mask & flag_okay_fg_DA,
                                        x=DS["fg_"+var], y=np.nan)
        
        return iasi_data_okay, iasi_data_fg_okay


    def iasi_pressure_to_height_levels(
        iasi_data: xr.DataArray,
        iasi_data_fg: xr.DataArray,
        iasi_pres: np.ndarray,
        sonde_pres: xr.DataArray,
        sonde_height: xr.DataArray,
        pres_sfc=None,
        ):
        
        sonde_pres_ip = sonde_pres.interp(launch_time=iasi_data.time)
        sonde_pres_ip = sonde_pres_ip.bfill(dim='time')
        sonde_pres_ip = sonde_pres_ip.ffill(dim='time')
        
        if pres_sfc is None:
            pres_sfc = np.nanmax(sonde_pres_ip, axis=-1)
        
        iasi_hgt = np.full(iasi_data.shape, np.nan)
        iasi_data_temp = np.full(iasi_data.shape, np.nan)
        for k in range(iasi_data.shape[0]):
            iasi_hgt[k,:] = np.interp(iasi_pres, sonde_pres_ip[k,::-1], sonde_height.values[::-1],
                                        left=np.nan, right=np.nan)
            
            idx_okay = np.where(iasi_pres <= pres_sfc[k])[0]
            iasi_data_temp[k,idx_okay] = iasi_data.values[k,idx_okay]
            
            if np.all(np.isnan(iasi_data_temp[k,idx_okay])):
                iasi_data_temp[k,idx_okay] = iasi_data_fg[k,idx_okay]
        
        iasi_data[:] = iasi_data_temp
        
        return iasi_hgt, iasi_data


    def create_geopot_height_axis(DS: xr.Dataset):
        
        """
        Provides the geopotential height based on the IASI temperature and humidity profiles. However, 
        it's recommended to use the height data computed from radiosonde pressure levels instead.
        """
        
        iasi_z = Z_from_pres(DS.pressure_levels_temp.values, DS.temp.values, DS.q.values)
        DS['Z'] = xr.DataArray(iasi_z, dims=DS.temp.dims, attrs={'long_name': "Geopotential height", 'units': 'm'})
        
        return DS


    def generate_fake_radiosonde_data(time: np.ndarray):
        
        std_pres, std_hgt, std_temp, std_q = create_ICAO_std_atmosphere()
        std_data = {'pres': std_pres, 'temp': std_temp, 'q': std_q}
        RS_DS = xr.Dataset(coords={'launch_time': (['launch_time'], time),
                                'height': (['height'], std_hgt)})
        
        n_time = len(time)
        for var in ['pres', 'temp', 'q']:
            RS_DS[var] = xr.DataArray(np.repeat(np.expand_dims(std_data[var], axis=0), n_time, axis=0), 
                                    dims=['launch_time', 'height'])
        
        return RS_DS
    
    
    files = identify_files_daterange(path, daterange=date_range, file_pattern=file_pattern)
    DS = xr.open_mfdataset(files, concat_dim='time', combine='nested').load()
    DS = DS.isel(time=(np.where(
        (~np.any(DS.flag_fgcheck > 511, axis=1)) & 
        (~np.any(DS.fg_qi_atmospheric_water_vapour > 3.95, axis=1))
        )[0]))
    
    IASI_pres = {'temp': DS.pressure_levels_temp.values,      # height axis for temperature
                   'q': DS.pressure_levels_humidity.values}     # height axis for q
    reasonable_value_limits = {'temp': (153.15, 333.15),
                               'q': (-0.1,0.1)}
    flag_okay_fg_DA, flag_okay_ret_DA, flag_itconv = get_iasi_quality_flags(DS)
    
    if RS_DS is None:
        RS_DS = generate_fake_radiosonde_data(DS.time.values)
    
    for var in ['temp', 'q']:
        iasi_data, iasi_data_fg = apply_flags_and_split_first_guess_and_OE(DS, var, reasonable_value_limits[var],
                                                                           flag_okay_fg_DA, flag_okay_ret_DA, flag_itconv)
        
        iasi_hgt, iasi_data = iasi_pressure_to_height_levels(iasi_data, iasi_data_fg, IASI_pres[var], 
                                                             RS_DS.pres, RS_DS.height, pres_sfc=pres_sfc)
        
        DS[var] = iasi_data
        DS['height_'+var] = xr.DataArray(iasi_hgt, dims=['time', 'nl'+var[0]])
    

    DS = DS.sel(nlt=DS.nlt[::-1], nlq=DS.nlq[::-1])     # surface shall be index 0
    DS = create_geopot_height_axis(DS)
    
    return DS


def import_ny_alesund_radiosondes_pangaea_tab(
    files: str,
    homogenized=False):

    """
    Imports radiosonde data from Ny-Alesund published to PANGAEA, i.e., 
    https://doi.org/10.1594/PANGAEA.845373 , https://doi.org/10.1594/PANGAEA.875196 , 
    https://doi.org/10.1594/PANGAEA.914973 . The continued homogenized data set is available on
    https://doi.pangaea.de/10.1594/PANGAEA.961203 . The Integrated Water Vapour will be computed using 
    the saturation water vapour pressure according to Hyland and Wexler 1983. Measurements will be
    given in SI units. 
    The radiosonde data will be stored in a dict with keys being the sonde index and
    the values are 1D arrays with shape (n_data_per_sonde,). Since we have more than one sonde per
    .tab file, single sondes must be identified via time difference (i.e., 900 seconds) or the
    provided sonde ID (latter is only available for sondes before 2017 or when using the 
    homogenized data set).

    Parameters:
    -----------
    files : str
        List of filename + path of the Ny-Alesund radiosonde data (.tab) published on PANGAEA.
    homogenized : bool
        If True, the homogenized radiosonde data set is used beyond 2017-04-01. If False, the 
        high-resolution data set is used after 2017-04-01.
    """


    # Ny-Alesund radiosonde .tab files are often composits of multiple radiosondes (i.e., one month
    # or a year). Therefore, first, just load the data and sort out single radiosondes later:
    n_data_per_file = 1000000       # assumed max. length of a file for data array initialisation

    # loop through files and load the data into a temporary dictionary:
    data_dict = dict()
    for kk, file in enumerate(files):

        f_handler = open(file, 'r')

        # # automatised inquiry of file length (is slower than just assuming a max number of lines):
        # n_data_per_file = len(f_handler.readlines())
        # f_handler.seek(0) # set position of pointer back to beginning of file

        translator_dict = {'Date/Time': "time",
                            'Altitude [m]': "height",
                            'PPPP [hPa]': "pres",
                            'TTT [°C]': "temp",
                            'RH [%]': "relhum",
                            'ff [m/s]': "wspeed",
                            'dd [deg]': "wdir",
                            'Latitude': "lat",
                            'Longitude': "lon"}     # translates naming from .tab files to the convention used here

        print(f"\rImporting {file}", end='')
        str_kk = str(kk)    # string index of file
        data_dict[str_kk] = {'time': np.full((n_data_per_file,), np.nan),       # in sec since 1970-01-01 00:00:00 UTC or numpy datetime64
                            'height': np.full((n_data_per_file,), np.nan),      # in m
                            'pres': np.full((n_data_per_file,), np.nan),        # in Pa
                            'temp': np.full((n_data_per_file,), np.nan),        # in K
                            'relhum': np.full((n_data_per_file,), np.nan),      # in [0,1]
                            'wspeed': np.full((n_data_per_file,), np.nan),      # in m s^-1
                            'wdir': np.full((n_data_per_file,), np.nan),        # in deg
                            'lat': np.full((n_data_per_file,), np.nan),         # in deg N
                            'lon': np.full((n_data_per_file), np.nan),          # in deg E
                            }
        if "NYA_UAS_" in file:
            translator_dict['ID'] = "ID"
            data_dict[str_kk]['ID'] = np.full((n_data_per_file,), 20*" ")


        mm = 0      # runs though all data points of one radiosonde and is reset to 0 for each new radiosonde
        data_line_indicator = -1        # if this is no longer -1, then the line where data begins has been identified
        for k, line in enumerate(f_handler):

            if data_line_indicator == -1:
                data_line_indicator = line.find("*/")       # string indicating the beginning of data

            else:   # data begins:
                current_line = line.strip().split("\t")     # split by tabs

                if 'Date/Time' in current_line: 
                    data_descr = current_line   # list with data description

                    # identify which column of a line represents which data type:
                    data_col_id = dict()
                    for data_key in translator_dict.keys():
                        data_col_id[translator_dict[data_key]] = data_descr.index(data_key)

                else:

                    # extract data:
                    for data_key in translator_dict.values():

                        try:
                            if data_key == 'time': 
                                data_dict[str_kk][data_key][mm] = np.datetime64(current_line[data_col_id[data_key]])
                            elif data_key == 'pres': 
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])*100.0
                            elif data_key == 'temp':
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]]) + 273.15
                            elif data_key == 'relhum':
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])*0.01
                            elif data_key == 'ID':
                                data_dict[str_kk][data_key][mm] = current_line[data_col_id[data_key]]

                            else:
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])

                        except IndexError:      # wind direction or wind speed data missing:
                            data_dict[str_kk]['wspeed'][mm] = float('nan')
                            data_dict[str_kk]['wdir'][mm] = float('nan')


                        except ValueError:      # then at least one measurement is missing:
                            current_line[current_line.index('')] = 'nan'        # 'repair' the data for import

                            if data_key == 'time': 
                                data_dict[str_kk][data_key][mm] = np.datetime64(current_line[data_col_id[data_key]])
                            elif data_key == 'pres': 
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])*100.0
                            elif data_key == 'temp':
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]]) + 273.15
                            elif data_key == 'relhum':
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])*0.01
                            elif data_key == 'ID':
                                data_dict[str_kk][data_key][mm] = current_line[data_col_id[data_key]]

                            else:
                                data_dict[str_kk][data_key][mm] = float(current_line[data_col_id[data_key]])

                    mm += 1

        # truncate data_dict of current file:
        for key in data_dict[str_kk].keys():
            data_dict[str_kk][key] = data_dict[str_kk][key][:mm]

    print("")

    # delete "ID" in translator_dict:
    if "ID" in translator_dict.keys(): del translator_dict["ID"]

    # concatenate all data_dict:
    data_dict_all = dict()
    for key in translator_dict.values():        # loop over measured and ID variables

        if key == "ID": continue

        for k, str_kk in enumerate(data_dict.keys()):       # loop over data files
            if k == 0:
                data_dict_all[key] = data_dict[str_kk][key]

                # add ID if available:
                if 'ID' in data_dict[str_kk].keys():
                    data_dict_all['ID'] = data_dict[str_kk]['ID']
            else:
                data_dict_all[key] = np.concatenate((data_dict_all[key], data_dict[str_kk][key]), axis=0)
                if 'ID' in data_dict[str_kk].keys():
                    data_dict_all['ID'] = np.concatenate((data_dict_all['ID'], data_dict[str_kk]['ID']), axis=0)

    # clear memory:
    del data_dict


    # set faulty data to nan:
    idx_broken = np.where(data_dict_all['relhum'] > 1.2)[0]
    data_dict_all['relhum'][idx_broken] = np.nan


    # identify single radiosondes based on time difference or sonde ID:
    new_sonde_idx = []
    for k in range(len(data_dict_all['ID'])-1):
        if data_dict_all['ID'][k+1] != data_dict_all['ID'][k]:
            new_sonde_idx.append(k)
    new_sonde_idx = np.asarray(new_sonde_idx)

    # identify the remaining radiosondes via time stamp differences and concatenate both identifier arrays:
    new_sonde_idx_time = np.where(np.abs(np.diff(data_dict_all['time'][new_sonde_idx[-1]+1:])) > 900.0)[0] + new_sonde_idx[-1]+1
                                                                            # indicates the last index belonging to the current sonde
                                                                            # i.e.: np.array([399,799,1194]) => 1st sonde: [0:400]
                                                                            # 2nd sonde: [400:800], 3rd sonde: [800:1195], 4th: [1195:]
                                                                            # ALSO negative time diffs must be respected because it
                                                                            # happens that one sonde may have been launched before the
                                                                            # previous one burst
    new_sonde_idx = np.concatenate((new_sonde_idx, new_sonde_idx_time), axis=0)
    # new_sonde_idx = np.where(np.abs(np.diff(data_dict_all['time'])) > 900.0)[0]   # this line would be for pure time-based sonde detect.

    n_sondes = len(new_sonde_idx) + 1
    rs_dict = dict()

    # loop over new_sonde_idx to identify single radiosondes and save their data:
    for k, nsi in enumerate(new_sonde_idx):
        k_str = str(k)

        # initialise rs_dict for each radiosonde:
        rs_dict[k_str] = dict()

        if (k > 0) & (k < n_sondes-1):
            for key in translator_dict.values():
                rs_dict[k_str][key] = data_dict_all[key][new_sonde_idx[k-1]+1:nsi+1]
        elif k == 0:
            for key in translator_dict.values():
                rs_dict[k_str][key] = data_dict_all[key][:nsi+1]

    # last sonde must be treated separately:
    rs_dict[str(n_sondes-1)] = dict()
    for key in translator_dict.values():
        rs_dict[str(n_sondes-1)][key] = data_dict_all[key][new_sonde_idx[k]+1:]


    # clear memory:
    del data_dict_all


    # finally, compute specific humidity and IWV, looping over all sondes:
    if not homogenized: 
        time_nya_uas_limit = 1491044046.0       # 2017-04-01T10:54:06
    else:
        # use current time as limit
        time_nya_uas_limit = np.datetime64(dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")).astype(np.float32)
    for s_idx in rs_dict.keys():

        # limit profiles of the NYA_UAS radiosondes to 10 km height:
        if rs_dict[s_idx]['time'][-1] < time_nya_uas_limit:
            idx_hgt = np.where(rs_dict[s_idx]['height'] <= 10000.0)[0]
            for key in rs_dict[s_idx].keys():
                rs_dict[s_idx][key] = rs_dict[s_idx][key][idx_hgt]

        # compute specific humidity and IWV:
        rs_dict[s_idx]['q'] = convert_rh_to_spechum(rs_dict[s_idx]['temp'], rs_dict[s_idx]['pres'], 
                                                    rs_dict[s_idx]['relhum'])
        rs_dict[s_idx]['IWV'] = compute_IWV_q(rs_dict[s_idx]['q'], rs_dict[s_idx]['pres'], nan_threshold=0.5, scheme='balanced')

    return rs_dict


def import_MOSAiC_Radiosondes_PS122_Level3_tab(filename):

    """
    Imports level 3 radiosonde data launched from Polarstern
    during the MOSAiC campaign. Time will be given in seconds since 1970-01-01 00:00:00 UTC
    and datetime. Furthermore, the Integrated Water Vapour will be computed
    using the saturation water vapour pressure according to Hyland and Wexler 1983.

    Maturilli, Marion; Sommer, Michael; Holdridge, Donna J; Dahlke, Sandro; 
    Graeser, Jürgen; Sommerfeld, Anja; Jaiser, Ralf; Deckelmann, Holger; 
    Schulz, Alexander (2022): MOSAiC radiosonde data (level 3). PANGAEA, 
    https://doi.org/10.1594/PANGAEA.943870

    Parameters:
    -----------
    filename : str
        Filename + path of the Level 3 radiosonde data (.tab) downloaded from the DOI
        given above.
    """

    n_sonde_prel = 3000     # just a preliminary assumption of the amount of radiosondes
    n_data_per_sonde = 12000    # assumption of max. time points per sonde
    reftime = np.datetime64("1970-01-01T00:00:00")
    # the radiosonde dict will be structured as follows:
    # rs_dict['0'] contains all data from the first radiosonde: rs_dict['0']['T'] contains temperature
    # rs_dict['1'] : second radiosonde, ...
    # this structure allows to have different time dimensions for each radiosonde
    rs_dict = dict()
    for k in range(n_sonde_prel):
        rs_dict[str(k)] = {'time': np.full((n_data_per_sonde,), reftime),       # np datetime64
                            'time_sec': np.full((n_data_per_sonde,), np.nan),   # in seconds since 1970-01-01 00:00:00 UTC
                            'Latitude': np.full((n_data_per_sonde,), np.nan),   # in deg N
                            'Longitude': np.full((n_data_per_sonde,), np.nan),  # in deg E
                            'Altitude': np.full((n_data_per_sonde,), np.nan),   # gps altitude above WGS84 in m
                            'h_geop': np.full((n_data_per_sonde,), np.nan),     # geopotential height in m
                            'h_gps': np.full((n_data_per_sonde,), np.nan),      # geometric/GPS receiver height in m
                            'P': np.full((n_data_per_sonde,), np.nan),          # in hPa
                            'T': np.full((n_data_per_sonde,), np.nan),          # in K
                            'RH': np.full((n_data_per_sonde,), np.nan),         # in percent
                            'mixrat': np.full((n_data_per_sonde,), np.nan),     # in mg kg-1
                            'wdir': np.full((n_data_per_sonde,), np.nan),       # in deg
                            'wspeed': np.full((n_data_per_sonde,), np.nan),     # in m s^-1
                            'IWV': np.full((n_data_per_sonde,), np.nan)}        # in kg m-2


    f_handler = open(filename, 'r')

    # identify header size and save global attributes:
    attribute_info = list()
    for k, line in enumerate(f_handler):
        if line.strip().split("\t")[0] in ['Citation:', 'Project(s):', 'Abstract:', 'Keyword(s):']:
            attribute_info.append(line.strip().split("\t")) # split by tabs
        if line.strip() == "*/":
            break

    m = -1      # used as index to save the entries into rs_dict; will increase for each new radiosonde
    mm = 0      # runs though all time points of one radiosonde and is reset to 0 for each new radiosonde
    precursor_event = ''
    for k, line in enumerate(f_handler):
        if k == 0:
            headerline = line.strip().split("\t")

        if k > 0:       # skip header
            current_line = line.strip().split("\t")     # split by tabs
            current_event = current_line[0]         # marks the radiosonde launch

            if current_event != precursor_event:    # then a new sonde is found in the current_line
                m = m + 1
                mm = 0

            # convert time stamp to seconds since 1970-01-01 00:00:00 UTC:
            rs_dict[str(m)]['time'][mm] = np.datetime64(current_line[1])
            rs_dict[str(m)]['time_sec'][mm] = rs_dict[str(m)]['time'][mm].astype(np.float64)

            # extract other info:
            try:
                rs_dict[str(m)]['Latitude'][mm] = float(current_line[10])
                rs_dict[str(m)]['Longitude'][mm] = float(current_line[8])
                rs_dict[str(m)]['Altitude'][mm] = float(current_line[6])
                rs_dict[str(m)]['h_geop'][mm] = float(current_line[2])
                rs_dict[str(m)]['h_gps'][mm] = float(current_line[4])
                rs_dict[str(m)]['P'][mm] = float(current_line[12])
                rs_dict[str(m)]['T'][mm] = float(current_line[16])
                rs_dict[str(m)]['RH'][mm] = float(current_line[18])
                rs_dict[str(m)]['mixrat'][mm] = float(current_line[22])
                rs_dict[str(m)]['wdir'][mm] = float(current_line[30])
                rs_dict[str(m)]['wspeed'][mm] = float(current_line[32])
                try:
                    rs_dict[str(m)]['IWV'][mm] = float(current_line[41])
                except IndexError:  # sometimes, the final two columns just don't exist...whyever
                    rs_dict[str(m)]['IWV'][mm] = float('nan')

            except ValueError:      # then at least one measurement is missing:
                for ix, cr in enumerate(current_line):
                    if cr == '':
                        current_line[ix] = 'nan'
                try:
                    rs_dict[str(m)]['Latitude'][mm] = float(current_line[10])
                    rs_dict[str(m)]['Longitude'][mm] = float(current_line[8])
                    rs_dict[str(m)]['Altitude'][mm] = float(current_line[6])
                    rs_dict[str(m)]['h_geop'][mm] = float(current_line[2])
                    rs_dict[str(m)]['h_gps'][mm] = float(current_line[4])
                    rs_dict[str(m)]['P'][mm] = float(current_line[12])
                    rs_dict[str(m)]['T'][mm] = float(current_line[16])
                    rs_dict[str(m)]['RH'][mm] = float(current_line[18])
                    rs_dict[str(m)]['mixrat'][mm] = float(current_line[22])
                    rs_dict[str(m)]['wdir'][mm] = float(current_line[30])
                    rs_dict[str(m)]['wspeed'][mm] = float(current_line[32])
                    rs_dict[str(m)]['IWV'][mm] = float(current_line[41])

                except IndexError:      # GPS connection lost
                    rs_dict[str(m)]['Latitude'][mm] = float('nan')
                    rs_dict[str(m)]['Longitude'][mm] = float('nan')
                    rs_dict[str(m)]['Altitude'][mm] = float('nan')
                    rs_dict[str(m)]['h_geop'][mm] = float(current_line[6])
                    rs_dict[str(m)]['h_gps'][mm] = float('nan')
                    rs_dict[str(m)]['P'][mm] = float(current_line[12])
                    rs_dict[str(m)]['T'][mm] = float(current_line[16])
                    rs_dict[str(m)]['RH'][mm] = float(current_line[18])
                    rs_dict[str(m)]['mixrat'][mm] = float(current_line[22])
                    rs_dict[str(m)]['wdir'][mm] = float('nan')
                    rs_dict[str(m)]['wspeed'][mm] = float('nan')
                    rs_dict[str(m)]['IWV'][mm] = float('nan')

            mm = mm + 1
            precursor_event = current_event

    # truncate redundantly initialised sondes:
    for k in range(m+1, n_sonde_prel): del rs_dict[str(k)]
    
    # finally truncate unneccessary time dimension for each sonde and compute IWV:
    for k in range(m+1):
        last_nonnan = np.where(~np.isnan(rs_dict[str(k)]['time_sec']))[0][-1] + 1       # + 1 because of python indexing
        for key in rs_dict[str(k)].keys(): rs_dict[str(k)][key] = rs_dict[str(k)][key][:last_nonnan]
        rs_dict[str(k)]['q'] = np.asarray([convert_rh_to_spechum(t, p*100.0, rh/100.0) 
                                for t, p, rh in zip(rs_dict[str(k)]['T'], rs_dict[str(k)]['P'], rs_dict[str(k)]['RH'])])
        rs_dict[str(k)]['rho_v'] = np.asarray([convert_rh_to_abshum(t, rh/100.0) 
                                for t, rh in zip(rs_dict[str(k)]['T'], rs_dict[str(k)]['RH'])])

        rs_dict[str(k)]['IWV'] = rs_dict[str(k)]['IWV'][~np.isnan(rs_dict[str(k)]['IWV'])][-1]

    return rs_dict, attribute_info


def import_PS131_radiosondes_pangaea_tab(file: str):

    """
    Imports radiosonde data from PS131 (ATWAICE) published to PANGAEA, 
    https://doi.org/10.1594/PANGAEA.974503. Measurements will be given in SI units. 
    The radiosonde data will be stored in a dict with keys being the sonde index and
    the values are 1D arrays with shape (n_data_per_sonde,). Since we have more than one sonde per
    .tab file, single sondes must be identified via time difference (i.e., 900 seconds) or the
    provided event ID.

    Parameters:
    -----------
    files : str
        List of filename + path of the radiosonde data (.tab) published on PANGAEA.
    """


    n_sonde_prel = 3000     # just a preliminary assumption of the amount of radiosondes
    n_data_per_sonde = 12000    # assumption of max. time points per sonde
    reftime = np.datetime64("1970-01-01T00:00:00")
    # the radiosonde dict will be structured as follows:
    # rs_dict['0'] contains all data from the first radiosonde: rs_dict['0']['T'] contains temperature
    # rs_dict['1'] : second radiosonde, ...
    # this structure allows to have different time dimensions for each radiosonde
    rs_dict = dict()
    for k in range(n_sonde_prel):
        rs_dict[str(k)] = {'time': np.full((n_data_per_sonde,), reftime),      # np.datetime64
                           'time_sec': np.full((n_data_per_sonde,), np.nan),   # in seconds since 1970-01-01 00:00:00 UTC
                           'Latitude': np.full((n_data_per_sonde,), np.nan),   # in deg N
                           'Longitude': np.full((n_data_per_sonde,), np.nan),  # in deg E
                           'height': np.full((n_data_per_sonde,), np.nan),     # geopotential height in m
                           'Z': np.full((n_data_per_sonde,), np.nan),          # geopotential height in m
                           'h_geom': np.full((n_data_per_sonde,), np.nan),     # geometric/GPS receiver height in m
                           'pres': np.full((n_data_per_sonde,), np.nan),       # in hPa
                           'temp': np.full((n_data_per_sonde,), np.nan),       # in K
                           'relhum': np.full((n_data_per_sonde,), np.nan),     # in percent
                           'wdir': np.full((n_data_per_sonde,), np.nan),       # in deg
                           'wspeed': np.full((n_data_per_sonde,), np.nan),     # in m s^-1
                           }


    f_handler = open(file, 'r')

    # identify header size and save global attributes:
    attribute_info = list()
    for k, line in enumerate(f_handler):
        if line.strip().split("\t")[0] in ['Citation:', 'Project(s):', 'Abstract:', 'Keyword(s):', 'Comment:']:
            attribute_info.append(line.strip().split("\t")) # split by tabs
        if line.strip() == "*/":
            break

    m = -1      # used as index to save the entries into rs_dict; will increase for each new radiosonde
    mm = 0      # runs though all time points of one radiosonde and is reset to 0 for each new radiosonde
    precursor_event = ''
    for k, line in enumerate(f_handler):
        if k == 0:
            headerline = line.strip().split("\t")

        if k > 0:       # skip header
            current_line = line.strip().split("\t")     # split by tabs
            current_event = current_line[0]         # marks the radiosonde launch

            if current_event != precursor_event:    # then a new sonde is found in the current_line
                m = m + 1
                mm = 0

            # convert time stamp to seconds since 1970-01-01 00:00:00 UTC:
            rs_dict[str(m)]['time'][mm] = np.datetime64(current_line[1])
            rs_dict[str(m)]['time_sec'][mm] = rs_dict[str(m)]['time'][mm].astype(np.float64)

            # extract other info:
            try:
                rs_dict[str(m)]['Latitude'][mm] = float(current_line[2])
                rs_dict[str(m)]['Longitude'][mm] = float(current_line[3])
                rs_dict[str(m)]['height'][mm] = float(current_line[4])
                rs_dict[str(m)]['Z'][mm] = float(current_line[5])
                rs_dict[str(m)]['h_geom'][mm] = float(current_line[6])
                rs_dict[str(m)]['pres'][mm] = float(current_line[8])*100.
                rs_dict[str(m)]['temp'][mm] = float(current_line[9]) + 273.15
                rs_dict[str(m)]['relhum'][mm] = float(current_line[10])*0.01
                rs_dict[str(m)]['wdir'][mm] = float(current_line[11])
                rs_dict[str(m)]['wspeed'][mm] = float(current_line[12])

            except ValueError:      # then at least one measurement is missing:
                for ix, cr in enumerate(current_line):
                    if cr == '':
                        current_line[ix] = 'nan'
                try:
                    rs_dict[str(m)]['Latitude'][mm] = float(current_line[2])
                    rs_dict[str(m)]['Longitude'][mm] = float(current_line[3])
                    rs_dict[str(m)]['height'][mm] = float(current_line[4])
                    rs_dict[str(m)]['Z'][mm] = float(current_line[5])
                    rs_dict[str(m)]['h_geom'][mm] = float(current_line[6])
                    rs_dict[str(m)]['pres'][mm] = float(current_line[8])*100.
                    rs_dict[str(m)]['temp'][mm] = float(current_line[9]) + 273.15
                    rs_dict[str(m)]['relhum'][mm] = float(current_line[10])*0.01
                    rs_dict[str(m)]['wdir'][mm] = float(current_line[11])
                    rs_dict[str(m)]['wspeed'][mm] = float(current_line[12])

                except IndexError:      # GPS connection lost
                    rs_dict[str(m)]['Latitude'][mm] = float('nan')
                    rs_dict[str(m)]['Longitude'][mm] = float('nan')
                    rs_dict[str(m)]['height'][mm] = float('nan')
                    rs_dict[str(m)]['Z'][mm] = float(current_line[5])
                    rs_dict[str(m)]['h_geom'][mm] = float('nan')
                    rs_dict[str(m)]['pres'][mm] = float(current_line[8])*100.
                    rs_dict[str(m)]['temp'][mm] = float(current_line[9]) + 273.15
                    rs_dict[str(m)]['relhum'][mm] = float(current_line[10])*0.01
                    rs_dict[str(m)]['wdir'][mm] = float('nan')
                    rs_dict[str(m)]['wspeed'][mm] = float('nan')

            mm = mm + 1
            precursor_event = current_event

    # truncate redundantly initialised sondes:
    for k in range(m+1, n_sonde_prel): del rs_dict[str(k)]
    
    # finally truncate unneccessary time dimension for each sonde and compute IWV:
    for k in range(m+1):
        last_nonnan = np.where(~np.isnan(rs_dict[str(k)]['time_sec']))[0][-1] + 1       # + 1 because of python indexing
        for key in rs_dict[str(k)].keys(): rs_dict[str(k)][key] = rs_dict[str(k)][key][:last_nonnan]
        rs_dict[str(k)]['q'] = np.asarray([convert_rh_to_spechum(t, p, rh) 
                                for t, p, rh in zip(rs_dict[str(k)]['temp'], rs_dict[str(k)]['pres'], rs_dict[str(k)]['relhum'])])
        rs_dict[str(k)]['rho_v'] = np.asarray([convert_rh_to_abshum(t, rh) 
                                for t, rh in zip(rs_dict[str(k)]['temp'], rs_dict[str(k)]['relhum'])])

        rs_dict[str(k)]['IWV'] = compute_IWV_q(rs_dict[str(k)]['q'], rs_dict[str(k)]['pres'])

    return rs_dict, attribute_info


def merge_PS131_radiosondes(
    path):
    
    """
    PS131 radiosonde obs are imported with import_radiosondes_PS131_txt and merged into 2D arrays
    with dimensions (launch_time,height).
    
    Parameters:
    -----------
    path : str
        Absolute path of the PS131 radiosonde .txt data.
    """
    
    files = sorted(glob.glob(path + "*.txt"))
    sonde_dict_temp = import_radiosondes_PS131_txt(files)
    
    new_height = np.arange(0.0, 20000.0001, 20.0)
    n_height = len(new_height)
    n_sondes = len(sonde_dict_temp.keys())
    sonde_dict = {'temp': np.full((n_sondes, n_height), np.nan),
                    'pres': np.full((n_sondes, n_height), np.nan),
                    'relhum': np.full((n_sondes, n_height), np.nan),
                    'height': np.full((n_sondes, n_height), np.nan),
                    'wdir': np.full((n_sondes, n_height), np.nan),
                    'wspeed': np.full((n_sondes, n_height), np.nan),
                    'q': np.full((n_sondes, n_height), np.nan),
                    'IWV': np.full((n_sondes,), np.nan),
                    'launch_time': np.zeros((n_sondes,)),           # in sec since 1970-01-01 00:00:00 UTC
                    'launch_time_npdt': np.full((n_sondes,), np.datetime64("1970-01-01T00:00:00"))}

    # interpolate to new grid:
    for idx in sonde_dict_temp.keys():
        for key in sonde_dict_temp[idx].keys():
            if key not in ["height", "IWV", 'launch_time', 'launch_time_npdt']:
                sonde_dict[key][int(idx),:] = np.interp(new_height, sonde_dict_temp[idx]['height'], sonde_dict_temp[idx][key])
            elif key == "height":
                sonde_dict[key][int(idx),:] = new_height
            elif key in ["IWV", 'launch_time', 'launch_time_npdt']:
                sonde_dict[key][int(idx)] = sonde_dict_temp[idx][key]
    del sonde_dict_temp
    
    # remove failed sonde:
    launch_time_failed = np.datetime64('2022-07-31T22:55:18')
    idx_broken_sonde = np.where(np.abs(sonde_dict['launch_time_npdt'] - launch_time_failed) < 
                                np.timedelta64(900, "s"))[0]
    sonde_mask = np.full((n_sondes,), True)
    sonde_mask[idx_broken_sonde] = False
    for key in sonde_dict.keys(): sonde_dict[key] = sonde_dict[key][sonde_mask,...]
    sonde_dict['iwv'] = sonde_dict['IWV']       # alias
    
    return sonde_dict


def import_hatpro_mirac_level2a_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='both',
    data_version='v00'):

    """
    Imports the synergetic neural network retrieval output combining data from HATPRO and MiRAC-P
    for all days between a start and an end date or imports data for a certain list of dates. 
    Each day is concatenated in ascending order.

    Parameters:
    -----------
    path_data : str
        Path of the synergetic retrieval level 2a (IWV (prw), LWP (clwvi)) data. 
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'iwv' or 'prw' will load the
        integrated water vapour. 'clwvi' or 'lwp' will load the liquid water path. 'both' will 
        load integrated water vapour and liquid water path. Default: 'both'
    data_version : str, optional
        Indicated the version of the data as string. Example: "v00", "v01, "v02".
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'both' will also load integrated water vapour only. Default: 'both'")

    else:
        if which_retrieval not in ['prw', 'iwv', 'clwvi', 'lwp', 'both']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'iwv' or 'prw' will load the " +
                "integrated water vapour. 'clwvi' or 'lwp' will load the liquid water path. 'both' will load integrated " +
                "water vapour and liquid water path. Default: 'both'")

        else:
            if which_retrieval in ['iwv', 'prw']:
                which_retrieval = ['prw']
            elif which_retrieval in ['lwp', 'clwvi']:
                which_retrieval = ['clwvi']
            elif which_retrieval == 'both':
                which_retrieval = ['prw', 'clwvi']
            else:
                raise ValueError("Argument '" + which_retrieval + "' not recognized. Please use one of the following options: " +
                        "'iwv' or 'prw' will load the " +
                        "integrated water vapour. 'lwp' or 'clwvi' will load the liquid water path. " +
                        "'both' will load both. Default: 'both'")
                    

    # Identify files in the date range: First, load all into a list, then check
    # which ones suit the daterange:
    sub_str = f"_{data_version}_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + f"MOSAiC_uoc_hatpro_lhumpro-243-340_l2_*.nc"))

    if type(date_start) == type(""):
        # extract day, month and year from start date:
        date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    else:
        # extract day, month and year from date_start:
        date_list = [dt.datetime.strptime(ddd, "%Y-%m-%d") for ddd in date_start]

        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)


    # distinguish between the retrieved products:
    DS_ret = xr.Dataset()       # dummy dataset
    if 'prw' in which_retrieval:
        files_prw = [file for file in files_filtered if "_prw_" in file]

        # load data:
        if len(files_prw) > 0:
            DS_p = xr.open_mfdataset(files_prw, concat_dim='time', combine='nested', decode_times=False)
            DS_ret = DS_p

    if 'clwvi' in which_retrieval:
        files_clwvi = [file for file in files_filtered if "_clwvi_" in file]

        # load data:
        if len(files_clwvi) > 0:
            DS_c = xr.open_mfdataset(files_clwvi, concat_dim='time', combine='nested', decode_times=False)
            DS_ret = DS_c

    # if both are requested, merge both datasets by just adding the former:
    if ('prw' in which_retrieval) and ('clwvi' in which_retrieval):
        if len(files_prw) > 0: DS_ret['prw'] = DS_p['prw']


    # 'repair' some variables:
    if DS_ret:
        DS_ret['flag_h'][np.isnan(DS_ret['flag_h']).load()] = 0.
        DS_ret['flag_m'][np.isnan(DS_ret['flag_m']).load()] = 0.

    return DS_ret


def import_hatpro_mirac_level2b_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='q',
    data_version='v00',
    around_radiosondes=True,
    path_radiosondes="",
    s_version='level_2',
    mwr_avg=0,
    flag_data=True):

    """
    Imports the synergetic neural network retrieval output combining data from HATPRO and MiRAC-P
    for all days between a start and an end date or imports data for a certain list of dates. 
    Each day is concatenated in ascending order. Because of memory usage, only times around 
    radiosonde launch times or alternatively 4 samples per day at fixed times (05, 11, 17 and 23 UTC)
    are considered.

    Parameters:
    -----------
    path_data : str
        Path of the synergetic retrieval level 2b (temperature (temp), specific humidity (q)) data.
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'temp' or 'ta' will load the
        zenith temperature profile. 'q' will load the specific humidity profile. 'rh' will load the
        generated relative humidity profile product.
    data_version : str, optional
        Indicated the version of the data as string. Example: "v00"
    around_radiosondes : bool, optional
        If True, data will be limited to the time around radiosonde launches. If False, something else
        (e.g. around 4 times a day) might be done. Default: True
    path_radiosondes : str, optional
        Path to radiosonde data (Level 2). Default: ""
    s_version : str, optional
        Specifies the radiosonde version that is to be imported. Must be 'level_2' or 'level_3' to 
        work properly. Other versions have not been implemeted because they are considered to be 
        inferior to level_2 or level_3 radiosondes.
    mwr_avg : int, optional
        If > 0, an average over mwr_avg seconds will be performed from sample_time to sample_time + 
        mwr_avg seconds. If == 0, no averaging will be performed.
    flag_data : bool, optional
        If True, data where flags indicate bad values are excluded (for HATPRO, flag = 0 and 
        flag = 32 are okay; for MiRAC-P, flag = 0 is okay). If False, no filtering will be applied.
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'temp' or 'ta' will load the " +
                "zenith temperature profile. 'q' will load humidity profiles. 'rh' will load the " +
                "generated relative humidity profile product.")

    else:
        if which_retrieval not in ['ta', 'temp', 'q', 'hus', 'rh']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'temp' or 'ta' will load the " +
                "zenith temperature profile. 'q' will load the specific humidity profile. 'rh' will load the " +
                "generated relative humidity profile product.")

        else:
            if which_retrieval in ['ta', 'temp']:
                which_retrieval = 'temp'
            elif which_retrieval in ['hus', 'q']:
                which_retrieval = 'q'
            elif which_retrieval == 'rh':
                which_retrieval = 'rh'
            elif which_retrieval not in ['ta', 'temp', 'q', 'rh']:
                raise ValueError("Argument '" + which_retrieval + "' not recognized")


    # extract dates from date_start (and date_end):
    date_list = []
    if type(date_start) == type([]):
        date_list = copy.deepcopy(date_start)
        date_start = date_start[0]
        date_end = date_list[-1]
        date_list = [dt.datetime.strptime(dl, "%Y-%m-%d").date() for dl in date_list]
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count number of days between start and end date as estimated max. array size:
    n_days = (date_end - date_start).days + 1
    n_hgt = {'q': 43, 'temp': 39, 'rh': 39}
    n_hgt = n_hgt[which_retrieval]


    # Create an array that includes the radiosonde launch times:
    if around_radiosondes:
        if not path_radiosondes:
            raise ValueError("If 'around_radiosondes' is True, the path to the radiosonde data ('path_radiosondes') " +
                                "must be given.")

        if s_version not in ['level_2', 'level_3']:
            raise ValueError("Radiosonde version 's_version' must be in ['level_2', 'level_3'] if around_radiosondes " +
                                "is True because " +
                                "for this version, the launch time is directly read from the filename. This has not " +
                                "been implemeted for other radiosonde versions ('mossonde', 'psYYMMDDwHH') because these " +
                                "are considered to be inferior.")
        else:
            add_files = sorted(glob.glob(path_radiosondes + "*.nc"))        # filenames only; filter path
            add_files = [os.path.basename(a_f) for a_f in add_files]
            
            # identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            if date_list:   # then only consider dates within date_list
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt.date() in date_list:
                        launch_times[kk] = ltt
                        kk += 1
            else:           # take all files between date_start and date_end
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                        launch_times[kk] = ltt
                        kk += 1

            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

    else:
        # max number of samples: n_days*4
        sample_times = [5, 11, 17, 23]      # UTC on each day
        n_samp = len(sample_times)
        n_samp_tot = n_days*n_samp


    # Identify files in the date range: First, load all into a list, then check
    # which ones suit the daterange:
    sub_str = f"_{data_version}_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + f"MOSAiC_uoc_hatpro_lhumpro-243-340_l2_*.nc"))


    if date_list:
        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)

    else:
        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    if which_retrieval == 'temp':
        file_substr = "_temp_v"
    elif which_retrieval == 'q':
        file_substr = "_q_"
    elif which_retrieval == 'rh':
        file_substr = "_rh_"
    files_filtered = [file for file in files_filtered if file_substr in file]


    # cycle through all years, all months and days:
    time_idx = 0        # this index (for lvl 2b) will be increased by the length of the time
                        # series of the current day (now_date) to indicate if data has already been found
    sample_time_tolerance = 900     # sample time tolerance in seconds: mwr time must be within this
                                    # +/- tolerance of a sample_time to be accepted
    if not date_list:
        date_list = (date_start + dt.timedelta(days=n) for n in range(n_days))
    else:
        date_list = [dt.datetime(dl_i.year, dl_i.month, dl_i.day) for dl_i in date_list]

    for now_date in date_list:

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day
        now_date_str = now_date.strftime("%Y%m%d")

        print(f"\rImporting level 2b data for {now_date_str}.", end="")

        # specify sample times as time: sec since 1970-01-01 00:00:00 UTC:
        if around_radiosondes:
            now_date_date = now_date.date()
            sample_mask = np.full((n_samp_tot,), False)
            for kk, l_t in enumerate(launch_times):
                sample_mask[kk] = l_t.date() == now_date_date

            sample_times_t = sample_times[sample_mask]

        else:
            sample_times_t = np.asarray([datetime_to_epochtime(dt.datetime(yyyy, mm, dd, st, 0, 0)) for st in sample_times])


        # identify files for current date:
        files_now = []
        for file in files_filtered:
            if now_date_str in file:
                files_now.append(file)

        if len(files_now) == 0: continue
        if len(files_now) > 1: 
            print("\nUnexpected number of files.")
            pdb.set_trace() # unexpected. debug


        # load data:
        DS = xr.open_dataset(files_now[0], decode_times=False)

        # repair flags:
        DS['flag_h'][np.isnan(DS['flag_h'])] = 0.
        DS['flag_m'][np.isnan(DS['flag_m'])] = 0.

        # it may occur that the whole day is flagged. If so, skip this file:
        flag_mask = (DS.flag_m == 0) & ((DS.flag_h == 0) | (DS.flag_h == 32))
        if not np.any(flag_mask):
            n_samp_real = 0
            continue
            pdb.set_trace()

        # if desired, reduce data to good flags only:
        if flag_data: DS = DS.isel(time=flag_mask)


        # Option 1: find the time slice where the mwr time is closest to the sample_times.
        # The identified index must be within 15 minutes, otherwise it will be discarded
        # Furthermore, it needs to be respected, that the flag value must be 0 for that case.
        # Option 2: Average over launch_time:launch_time+mwr_avg. For this, find the respecitve
        # indices:
        if mwr_avg == 0:
            sample_idx = []
            for st in sample_times_t:
                idx = np.argmin(np.abs(DS['time'].values - st))
                if np.abs(DS['time'].values[idx] - st) < sample_time_tolerance:
                    sample_idx.append(idx)
            sample_idx = np.asarray(sample_idx)
            n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

        else:
            sample_idx = []
            for st in sample_times_t:
                idx = np.where((DS['time'].values >= st) & (DS['time'].values < st + mwr_avg))[0]
                if len(idx) > 0:    # then an overlap has been found
                    sample_idx.append(idx)
            n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

        if n_samp_real == 0: continue


        # select times at sample_idx and concatenate data (eventually combined with averaging):
        if mwr_avg == 0:
            if time_idx == 0:
                DS_ret = DS.isel(time=sample_idx)
            else:
                DS_ret = xr.concat((DS_ret, DS.isel(time=sample_idx)), dim='time')

        else:
            # reduce time around sample times (radiosonde launches) only and average
            # over time:
            ds_list = []
            for sii in sample_idx:
                DS_app = DS.isel(time=sii).mean('time').expand_dims('time', axis=0)

                # forward the highest flag value to DS_app:
                DS_app['flag_h'] = DS.isel(time=sii).flag_h.max('time').expand_dims('time', axis=0)
                DS_app['flag_m'] = DS.isel(time=sii).flag_m.max('time').expand_dims('time', axis=0)

                DS_app = DS_app.assign_coords({'time': np.array([DS.time[sii[0]]])})
                ds_list.append(DS_app)
            ds_list = xr.concat(ds_list, dim='time')

            # concatenate:
            if time_idx == 0:
                DS_ret = ds_list
            else:
                DS_ret = xr.concat((DS_ret, ds_list), dim='time')

                
        time_idx = time_idx + n_samp_real
        DS.close()
        del DS


    print("")
    if time_idx == 0:
        print("No files have been found....")
        DS_ret = None
    return DS_ret


def import_hatpro_mirac_level2c_daterange_pangaea(
    path_data,
    date_start,
    date_end=None,
    which_retrieval='temp',
    data_version='v00',
    around_radiosondes=True,
    path_radiosondes="",
    s_version='level_2',
    mwr_avg=0,
    flag_data=True):

    """
    Imports the synergetic neural network retrieval output for boundary layer temperature
    profiles for all days between a start and an end date or imports data for a certain list 
    of dates. Each day is concatenated in ascending order. Only times around radiosonde 
    launch times or alternatively 4 samples per day at fixed times (05, 11, 17 and 23 UTC)
    are considered.

    Parameters:
    -----------
    path_data : str
        Path of the synergetic retrieval level 2b (temperature (temp), specific humidity (q)) data.
    date_start : str or list of str
        If str: Marks the first day of the desired period. To be specified in yyyy-mm-dd 
        (e.g. 2021-01-14)! Requires date_end to be specified. If list of str: Imports only the 
        dates listes in the list of dates (in yyyy-mm-dd). date_end must be None in this case.
    date_end : str
        Marks the last day of the desired period. To be specified in yyyy-mm-dd (e.g. 2021-01-14)!
    which_retrieval : str, optional
        This describes which variable(s) will be loaded. Options: 'temp' or 'ta' will both load the
        boundary layer scan based temperature profile.
    data_version : str, optional
        Indicated the version of the data as string. Example: "v00"
    around_radiosondes : bool, optional
        If True, data will be limited to the time around radiosonde launches. If False, something else
        (e.g. around 4 times a day) might be done. Default: True
    path_radiosondes : str, optional
        Path to radiosonde data (Level 2). Default: ""
    s_version : str, optional
        Specifies the radiosonde version that is to be imported. Must be 'level_2' to work properly.
        Other versions have not been implemeted because they are considered to be inferior to level_2
        radiosondes.
    mwr_avg : int, optional
        If > 0, an average over mwr_avg seconds will be performed from sample_time to sample_time + 
        mwr_avg seconds. If == 0, no averaging will be performed.
    flag_data : bool, optional
        If True, data where flags indicate bad values are excluded (for HATPRO, flag = 0 and 
        flag = 32 are okay; for MiRAC-P, flag = 0 is okay). If False, no filtering will be applied.
    """

    # identify if date_start is string or list of string:
    if type(date_start) == type("") and not date_end:
        raise ValueError("'date_end' must be specified if 'date_start' is a string.")
    elif type(date_start) == type([]) and date_end:
        raise ValueError("'date_end' should be none if 'date_start' is a list of strings.")


    # check if the input of the retrieval variable is okay:
    if not isinstance(which_retrieval, str):
            raise TypeError("Argument 'which_retrieval' must be a string. Options: 'temp' or 'ta' will load the " +
                "zenith temperature profile. 'q' will load humidity profiles.")

    else:
        if which_retrieval not in ['ta', 'temp']:
            raise ValueError("Argument 'which_retrieval' must be one of the following options: 'temp' or 'ta' will load the " +
                "zenith temperature profile.")

        else:
            if which_retrieval in ['ta', 'temp']:
                which_retrieval = 'temp'


    # extract dates from date_start (and date_end):
    date_list = []
    if type(date_start) == type([]):
        date_list = copy.deepcopy(date_start)
        date_start = date_start[0]
        date_end = date_list[-1]
        date_list = [dt.datetime.strptime(dl, "%Y-%m-%d").date() for dl in date_list]
    date_start = dt.datetime.strptime(date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d")

    # count number of days between start and end date as estimated max. array size:
    n_days = (date_end - date_start).days + 1
    n_hgt = 39      # manually inquired in the netCDF files


    # Create an array that includes the radiosonde launch times:
    if around_radiosondes:
        if not path_radiosondes:
            raise ValueError("If 'around_radiosondes' is True, the path to the radiosonde data ('path_radiosondes') " +
                                "must be given.")

        if s_version != 'level_2':
            raise ValueError("Radiosonde version 's_version' must be 'level_2' if around_radiosondes is True because " +
                                "for this version, the launch time is directly read from the filename. This has not " +
                                "been implemeted for other radiosonde versions ('mossonde', 'psYYMMDDwHH') because these " +
                                "are considered to be inferior.")
        else:
            add_files = sorted(glob.glob(path_radiosondes + "*.nc"))        # filenames only; filter path
            add_files = [os.path.basename(a_f) for a_f in add_files]
            
            # identify launch time:
            n_samp = len(add_files)     # number of radiosondes
            launch_times = np.full((n_samp,), dt.datetime(1970,1,1))
            kk = 0
            if date_list:   # then only consider dates within date_list
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt.date() in date_list:
                        launch_times[kk] = ltt
                        kk += 1
            else:           # take all files between date_start and date_end
                for a_f in add_files:
                    ltt = dt.datetime.strptime(a_f[-19:-4], "%Y%m%d_%H%M%S")
                    # only save those that are in the considered period
                    if ltt >= date_start and ltt < (date_end + dt.timedelta(days=1)):
                        launch_times[kk] = ltt
                        kk += 1

            # truncate launch_times and convert to sec since 1970-01-01 00:00:00 UTC:
            launch_times = launch_times[:kk]
            sample_times = datetime_to_epochtime(launch_times)
            n_samp_tot = len(sample_times)

    else:
        # max number of samples: n_days*4
        sample_times = [5, 11, 17, 23]      # UTC on each day
        n_samp = len(sample_times)
        n_samp_tot = n_days*n_samp
            

    # Identify files in the date range: First, load all into a list, then check
    # which ones suit the daterange:
    sub_str = f"_{data_version}_"
    l_sub_str = len(sub_str)
    files = sorted(glob.glob(path_data + f"MOSAiC_uoc_hatpro_lhumpro-243-340_l2_*.nc"))


    if date_list:
        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date in date_list:
                files_filtered.append(file)

    else:
        # run through list: identify where date is written and check if within date range:
        files_filtered = list()
        for file in files:
            ww = file.find(sub_str) + l_sub_str
            if file.find(sub_str) == -1: continue
            file_date = dt.datetime.strptime(file[ww:ww+8], "%Y%m%d")
            if file_date >= date_start and file_date <= date_end:
                files_filtered.append(file)
    files_filtered = [file for file in files_filtered if "_temp_bl_" in file]


    # cycle through all years, all months and days:
    time_idx = 0        # this index (for lvl 2c) will be increased by the length of the time
                        # series of the current day (now_date) to indicate if data has already been found
    sample_time_tolerance = 1800        # sample time tolerance in seconds: mwr time must be within this
                                    # +/- tolerance of a sample_time to be accepted
    if not date_list:
        date_list = (date_start + dt.timedelta(days=n) for n in range(n_days))
    else:
        date_list = [dt.datetime(dl_i.year, dl_i.month, dl_i.day) for dl_i in date_list]

    for now_date in date_list:

        yyyy = now_date.year
        mm = now_date.month
        dd = now_date.day
        now_date_str = now_date.strftime("%Y%m%d")

        print(f"\rImporting level 2c data for {now_date_str}.", end="")

        # specify sample times as time: sec since 1970-01-01 00:00:00 UTC:
        if around_radiosondes:
            now_date_date = now_date.date()
            sample_mask = np.full((n_samp_tot,), False)
            for kk, l_t in enumerate(launch_times):
                sample_mask[kk] = l_t.date() == now_date_date

            sample_times_t = sample_times[sample_mask]

        else:
            sample_times_t = np.asarray([datetime_to_epochtime(dt.datetime(yyyy, mm, dd, st, 0, 0)) for st in sample_times])


        # identify files for current date:
        files_now = []
        for file in files_filtered:
            if now_date_str in file:
                files_now.append(file)

        if len(files_now) == 0: continue
        if len(files_now) > 1: 
            print("\nUnexpected number of files.")
            pdb.set_trace() # unexpected. debug


        # load data:
        DS = xr.open_dataset(files_now[0], decode_times=False)

        # repair flags:
        DS['flag_h'][np.isnan(DS['flag_h'])] = 0.
        DS['flag_m'][np.isnan(DS['flag_m'])] = 0.

        # it may occur that the whole day is flagged. If so, skip this file:
        flag_mask = (DS.flag_m == 0) & ((DS.flag_h == 0) | (DS.flag_h == 32))
        if not np.any(flag_mask):
            n_samp_real = 0
            continue
            pdb.set_trace()

        # if desired, reduce data to good flags only:
        if flag_data: DS = DS.isel(time=flag_mask)


        # Option 1: find the time slice where the mwr time is closest to the sample_times.
        # The identified index must be within 15 minutes, otherwise it will be discarded
        # Furthermore, it needs to be respected, that the flag value must be 0 for that case.
        # Option 2: Average over launch_time-mwr_avg:launch_time+mwr_avg. For this, find the 
        # respecitve indices:
        if mwr_avg == 0:
            sample_idx = []
            for st in sample_times_t:
                idx = np.argmin(np.abs(DS['time'].values - st))
                if np.abs(DS['time'].values[idx] - st) < sample_time_tolerance:
                    sample_idx.append(idx)
            sample_idx = np.asarray(sample_idx)
            n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

        else:
            sample_idx = []
            for st in sample_times_t:
                idx = np.where((DS['time'].values >= st - mwr_avg) & (DS['time'].values <= st + mwr_avg))[0]
                if len(idx) > 0:    # then an overlap has been found
                    sample_idx.append(idx)
            n_samp_real = len(sample_idx)   # number of samples that are valid to use; will be equal to n_samp in most cases

        if n_samp_real == 0: continue


        # select times at sample_idx and concatenate data (eventually combined with averaging):
        if mwr_avg == 0:
            if time_idx == 0:
                DS_ret = DS.isel(time=sample_idx)
            else:
                DS_ret = xr.concat((DS_ret, DS.isel(time=sample_idx)), dim='time')

        else:
            # reduce time around sample times (radiosonde launches) only and average
            # over time:
            ds_list = []
            for sii in sample_idx:
                DS_app = DS.isel(time=sii).mean('time').expand_dims('time', axis=0)

                # forward the highest flag value to DS_app:
                DS_app['flag_h'] = DS.isel(time=sii).flag_h.max('time').expand_dims('time', axis=0)
                DS_app['flag_m'] = DS.isel(time=sii).flag_m.max('time').expand_dims('time', axis=0)

                DS_app = DS_app.assign_coords({'time': np.array([DS.time[sii[0]]])})
                ds_list.append(DS_app)
            ds_list = xr.concat(ds_list, dim='time')

            # concatenate:
            if time_idx == 0:
                DS_ret = ds_list
            else:
                DS_ret = xr.concat((DS_ret, ds_list), dim='time')

                
        time_idx = time_idx + n_samp_real
        DS.close()
        del DS


    print("")
    if time_idx == 0:
        print("No files have been found....")
        DS_ret = None
    return DS_ret


def import_MOSAiC_Radiosondes_PS122_Level3_merged_txt(filename):

    """
    Imports level 3 radiosonde data that has been merged with MetTower data by Sandro Dahlke.
    Time will be given in seconds since 1970-01-01 00:00:00 UTC. Furthermore, the Integrated Water 
    Vapour, absolute humidity and specific humidity will be computed using the saturation water 
    vapour pressure according to Hyland and Wexler 1983.

    Dahlke, Sandro; Shupe, Matthew D; Cox, Christopher J; Brooks, Ian M; Blomquist, Byron; 
    Persson, P Ola G (2023): Extended radiosonde profiles 2019/09-2020/10 during MOSAiC 
    Legs PS122/1 - PS122/5. PANGAEA, https://doi.org/10.1594/PANGAEA.961881

    Parameters:
    -----------
    filename : str
        Filename + path of the Level 3 radiosonde data (.tab) downloaded from the DOI
        given above.
    """

    import re       # to search for dates

    n_data_per_sonde = 12000    # assumption of max. time points per sonde
    reftime = np.datetime64("1970-01-01T00:00:00")

    # initialize arrays:
    # the radiosonde dict will be structured as follows:
    # rs_dict contains all data from the first radiosonde: rs_dict['0']['T'] contains temperature
    # rs_dict['1'] : second radiosonde, ...
    # this structure allows to have different time dimensions for each radiosonde
    rs_dict = {'time': np.full((n_data_per_sonde,), reftime),       # np datetime64
                'time_sec': np.full((n_data_per_sonde,), np.nan),   # in seconds since 1970-01-01 00:00:00 UTC
                'lat': np.full((n_data_per_sonde,), np.nan),        # in deg N
                'lon': np.full((n_data_per_sonde,), np.nan),        # in deg E
                'height': np.full((n_data_per_sonde,), np.nan),     # altitude in m
                'pres': np.full((n_data_per_sonde,), np.nan),       # in hPa
                'temp': np.full((n_data_per_sonde,), np.nan),       # in K
                'rh': np.full((n_data_per_sonde,), np.nan),         # in percent
                'wdir': np.full((n_data_per_sonde,), np.nan),       # in deg
                'wspeed': np.full((n_data_per_sonde,), np.nan),     # in m s^-1
                'temp_flag': np.full((n_data_per_sonde,), np.nan),
                'rh_flag': np.full((n_data_per_sonde,), np.nan)}


    with open(filename, 'r', encoding='utf-8') as f_handler:

        # identify header size and save global attributes:
        attribute_info = {'launch_time': reftime, 'data_based_on': [],
                            'data_author': "", 'flag_comments': []}
        k_data_based_on = -100000000        # to save the line number of multi-line attributes
        k_flag = -100000000
        for k, line in enumerate(f_handler):
            line_str = line.strip()
            # pdb.set_trace()
            if "Radiosonde launch: " in line_str:
                match_str = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', line_str)     # finds the date with the format yyyy-mm-ddTHH:MM:SS
                attribute_info['launch_time'] = np.datetime64(match_str.group())
            elif "Data product based on:" in line_str:
                k_data_based_on = k
            elif "data prepared by" in line_str:
                attribute_info['data_author'] = line_str[len("data prepared by") + 1:].replace("...", " ")
            elif "FLAG:" in line_str:
                k_flag = k

            # the lines after k_flag and k_data_based_on contain some attributes:
            if k in [k_data_based_on+1, k_data_based_on+2, k_data_based_on+3]:
                attribute_info['data_based_on'] += [line_str]

            if k in [k_flag+1, k_flag+2, k_flag+3, k_flag+4, k_flag+5, k_flag+6]:
                attribute_info['flag_comments'] += [line_str]

            if line_str == "########ENDHEADER########":
                break

        mm = 0      # runs though all time points of one radiosonde and is reset to 0 for each new radiosonde
        for k, line in enumerate(f_handler):

            current_line = line.strip().split("\t")     # split by tabs

            # convert time stamp to seconds since 1970-01-01 00:00:00 UTC:
            rs_dict['time'][mm] = np.datetime64(current_line[0])
            rs_dict['time_sec'][mm] = rs_dict['time'][mm].astype(np.float64)

            # extract other info:
            try:
                rs_dict['lat'][mm] = float(current_line[3])
                rs_dict['lon'][mm] = float(current_line[4])
                rs_dict['height'][mm] = float(current_line[1])
                rs_dict['pres'][mm] = float(current_line[2])
                rs_dict['temp'][mm] = float(current_line[5])
                rs_dict['rh'][mm] = float(current_line[7])
                rs_dict['temp_flag'][mm] = float(current_line[6])
                rs_dict['rh_flag'][mm] = float(current_line[8])
                rs_dict['wdir'][mm] = float(current_line[10])
                rs_dict['wspeed'][mm] = float(current_line[9])

            except:     # then at least one measurement is missing:
                pdb.set_trace()

            mm += 1


        # convert fillvalues to nan:
        for key in rs_dict.keys():
            if key not in ['time', 'time_sec', 'temp_flag', 'rh_flag']:
                rs_dict[key][rs_dict[key] == -9999] = np.nan
        
        # finally truncate unneccessary time dimension, filter fillvalues and compute IWV, rho_v and q for each sonde:
        last_nonnan = np.where(~np.isnan(rs_dict['pres']))[0][-1] + 1       # + 1 because of python indexing
        for key in rs_dict.keys(): 
            rs_dict[key] = rs_dict[key][:last_nonnan]

        rs_dict['q'] = np.asarray([convert_rh_to_spechum(t+273.15, p*100.0, rh/100.0) 
                                for t, p, rh in zip(rs_dict['temp'], rs_dict['pres'], rs_dict['rh'])])
        rs_dict['rho_v'] = np.asarray([convert_rh_to_abshum(t+273.15, rh/100.0) 
                                for t, rh in zip(rs_dict['temp'], rs_dict['rh'])])


        # limit pressure and humidity to values from first non nan to last:
        non_nan_idx = np.where(~np.isnan(rs_dict['pres']))[0]
        pres = copy.deepcopy(rs_dict['pres'][non_nan_idx[0]:non_nan_idx[-1]+1])

        # repair pressure (fill gaps): Fill gaps in between when pressure measurements around the gap exist:
        n_nans = np.count_nonzero(np.isnan(pres))
        nan_idx = np.where(np.isnan(pres))[0]

        # restrict gap filling: if too many/large gaps, don't fill:
        if (n_nans / len(pres) < 0.1) and n_nans > 0:

            # fill gaps:
            nan_mask = np.isnan(pres)
            where_diff = np.where(np.diff(nan_mask))[0] # yields position of holes
            n_holes = round(len(where_diff) / 1.9999999999)


            # indices of bottom and top boundary of each hole:
            hole_boundaries = np.asarray([[where_diff[2*jj], where_diff[2*jj+1]+1] for jj in range(n_holes)])
                        
            # use the values of the hole boundaries as interpolation targets:
            pres_temp = copy.deepcopy(pres)

            # cycle through holes:
            for hole_b, hole_t in zip(hole_boundaries[:,0], hole_boundaries[:,1]):
                rpl_idx = np.arange(hole_b, hole_t + 1) # +1 because of python indexing
                bd_idx = np.array([rpl_idx[0], rpl_idx[-1]])
                bd_val = np.array([pres_temp[hole_b], pres_temp[hole_t]])

                bridge = np.interp(rpl_idx, bd_idx, bd_val)

                # fill the whole hole:
                pres[rpl_idx] = bridge

            # update value:
            rs_dict['pres'][non_nan_idx[0]:non_nan_idx[-1]+1] = pres


        # for IWV computation, neglect lowest measurements because also the MWRs are not directly on
        # the sea ice and limit until 20 km height (because sometimes, slightly negative q and rh 
        # occur at these high altitudes and because of some sondes containing many nans above):
        height_limit = np.where((rs_dict['height'] >= 15.0) & (rs_dict['height'] <= 20000.0))[0]    # index to cut pressure and humidity

        # if attribute_info['launch_time'] == np.datetime64("2020-09-13T07:55:58"): pdb.set_trace()
        rs_dict['IWV'] = compute_IWV_q(rs_dict['q'][height_limit], rs_dict['pres'][height_limit]*100, nan_threshold=0.1)

    return rs_dict, attribute_info


def import_mosaic_lidar_optical_profiles(files):

    """
    (Unfinished) Importer for data downloaded from https://doi.org/10.1594/PANGAEA.935539.

    Parameters:
    -----------
    files : list of str
        List of strings containing the full file names (with absolute path).
    """

    # loop over files and open them:
    n_data = 10000  # assumed amount of data points per file to initialize arrays
    for file in files:

        headersize = 1      # manually inquired
        with open(file, 'r', encoding='cp1252') as f_handler:   # cp1252 for ANSI encoding

            ii = 0  # index of the current data sample (basically corresponding to the current line of the 
                    # file without header)
            for k, line in enumerate(f_handler):
                line_str = line.strip()

                if k == 0:
                    # indentify the data variable and unit; also remove some colons and replace spaces
                    # by underscores:
                    data_keys = np.array(line_str.split('\t'))
                    data_units = np.full(data_keys.shape, "", dtype="<U32")
                    for kk, key in enumerate(data_keys):
                        data_keys[kk] = data_keys[kk].replace(':', '')

                        # units:
                        if "(" in data_keys[kk]:
                            unit_loc0, unit_loc1 = data_keys[kk].find("("), data_keys[kk].find(")")
                            data_units[kk] = data_keys[kk][unit_loc0+1:unit_loc1]

                        # update variabile name and units: e.g., remove unit and replace space by underscore:
                        data_keys[kk] = data_keys[kk].replace(f"({data_units[kk]})", "").replace(" ","_").replace("ö","o")
                        data_units[kk] = data_units[kk].replace("^","") # remove exponent sign
                        if data_keys[kk][-1] == "_": data_keys[kk] = data_keys[kk][:-1] # final underscore not needed

                    # initialize arrays:
                    n_keys = len(data_keys)
                    data_dict = dict()
                    for key in data_keys:
                        data_dict[key] = np.full((n_data,), np.nan)

                else:
                    # extract data:
                    line_info = line_str.split('\t')

                    # check if length agrees with amount of data_keys: If true, put the data into the
                    # data dict:
                    if n_keys != len(line_info): pdb.set_trace() # debug
                    for kk, key in enumerate(data_keys):
                        data_dict[key][ii] = float(line_info[kk])
                    ii += 1

            # limit data dictionary to the true amount of detected data:
            for key in data_keys:
                data_dict[key] = data_dict[key][:ii]


            # put into xarray data set:
            date_str_search = re.search(r'\d{8}_\d{4}-\d{4}', file)
            time_str = date_str_search.group()
            date_str = dt.datetime.strptime(time_str[:8],"%Y%m%d").strftime("%Y-%m-%d")
            time_0_str = dt.datetime.strptime(time_str[9:13],"%H%M").strftime("%H:%M")
            time_1_str = dt.datetime.strptime(time_str[-4:],"%H%M").strftime("%H:%M")
            DS = xr.Dataset(coords={'time': (['time'], np.array([np.datetime64(date_str+"T"+time_0_str)])),
                                    'height': (['height'], data_dict['Height']*1000., 
                                                {'long_name': "Height", 'units': "m"})})

            # put data into DS:
            DS['time_end'] = xr.DataArray(np.datetime64(date_str+"T"+time_1_str), dims=[])
            for kk, key in enumerate(data_keys):
                if key != 'Height':
                    DS[key] = xr.DataArray(np.broadcast_to(data_dict[key], (1,len(data_dict[key]))), dims=['time', 'height'], 
                                            attrs={'units': data_units[kk]})

            pdb.set_trace()


def import_co2_n2o_c4_sf6_csv(files):

    """
    Imports the NOAA Trace Gas data from "Lan, X., K.W. Thoning, and E.J. Dlugokencky: 
    Trends in globally-averaged CH4, N2O, and SF6 determined from NOAA Global Monitoring 
    Laboratory measurements. Version 2024-06, https://doi.org/10.15138/P8XG-AA10" and puts it
    into an xarray data set.

    Parameters:
    -----------
    files : list of str
        List containing the full path and filename of each trace gas concentration data set in 
        a .csv table.
    """

    ds_dict = dict()    # dictionary, which is to contain data sets for each trace gas

    # loop over files:
    for file in files:

        # identify trace gas name:
        tg_name = os.path.basename(file)[:os.path.basename(file).find("_")]

        # open the csv and read it:
        with open(file, newline='') as csvfile:

            csv_reader = csv.reader(csvfile, delimiter=',')
            list_of_lines = [row for row in csv_reader]

        # header finished after rows stop starting with '#'. Extract data from 
        # lines:
        data_lines = [line for line in list_of_lines if ((len(line) > 0) and (line[0][0] != '#'))]
        unit = [line[-1][line[-1].find("abbreviated as ")+len("abbreviated as "):] 
                for line in list_of_lines if (len(line) > 0) and ("abbreviated as " in line[-1])][0]

        # initialize data set and fill it with data. data_lines[0] contains list of variable names,
        # skip it when extracting data:
        data = np.array([[d_l[0], d_l[1], d_l[2]] for d_l in data_lines[1:]])
        DS = xr.Dataset(coords={'time': (['time'], data[:,0].astype('datetime64[Y]').astype('datetime64[ns]'))})
        DS[tg_name] = xr.DataArray(data[:,1].astype(np.float64), dims=['time'], attrs={'units': unit})
        DS[tg_name+"_unc"] = xr.DataArray(data[:,2].astype(np.float64), dims=['time'],
                                            attrs={'units': unit})

        ds_dict[tg_name] = DS
        DS = DS.close()

    # merge trace gas data sets:
    DS = xr.merge(ds_dict.values(), join='outer')

    return DS


def import_trace_gas_csv(file):

    """
    Imports the NOAA Trace Gas data from "https://gml.noaa.gov/aggi/aggi.html" and puts it
    into an xarray data set.

    Parameters:
    -----------
    file : str
        Full path and filename of trace gas concentration data in a .csv table.
    """

    sfac = {'ppm': 1e-06, 'ppb': 1e-09, 'ppt': 1e-12}   # acronyms for molecule 

    # open the csv and read it:
    with open(file, newline='') as csvfile:

        csv_reader = csv.reader(csvfile, delimiter=',')
        list_of_lines = [row for row in csv_reader]

    # below the header, the trace gas name and units can be extracted:
    tg_species = [species.lower().replace('-','') for species in list_of_lines[2] if species != ""]
    units = [unit for unit in list_of_lines[3] if unit != ""]
    assert len(tg_species) == len(units)

    # extract data: multiply data with the respective unit
    data_lines = list_of_lines[4:-5]
    data = np.array([[np.nan if dd == "nd" else float(dd)*sfac[units[kk]] for kk, dd in enumerate(data_line[1:])] 
                    for data_line in data_lines])
    data_times = np.array([np.datetime64(f"{int(float(data_line[0]))}-07-01T00:00:00") for data_line in data_lines])
    

    # initialize data set and fill it with data
    DS = xr.Dataset(coords={'time': (['time'], data_times.astype('datetime64[ns]'))})
    for kk, species in enumerate(tg_species):
        DS[species] = xr.DataArray(data[:,kk], dims=['time'])

    return DS