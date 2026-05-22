import numpy as np
import copy
import datetime as dt
import xarray as xr
import pandas as pd
import os
import glob
import pdb
import warnings


def running_mean(x, N):

    """
    Moving average of a 1D array x with a window width of N

    Parameters:
    -----------
    x : array of floats
        1D data vector of which the running mean is to be taken.
    N : int
        Running mean window width.
    """
    x = x.astype(np.float64)
    x_m = copy.deepcopy(x)
    
    # run through the array:
    for k in range(len(x)):
        if k%400000 == 0: print(k/len(x))   # output required to avoid the ssh connection to
                                            # be automatically dropped

        # Identify which indices are addressed for the running
        # mean of the current index k:
        if N%2 == 0:    # even:
            rm_range = np.arange(k - int(N/2), k + int(N/2), dtype = np.int32)
        else:           # odd:
            rm_range = np.arange(k - int(N/2), k + int(N/2) + 1, dtype=np.int32)

        # remove indices that exceed array bounds:
        rm_range = rm_range[(rm_range >= 0) & (rm_range < len(x))]

        # moving average:
        x_m[k] = np.mean(x[rm_range])
    
    return x_m


def running_mean_datetime(x, N, t):

    """
    Moving average of a 1D array x with a window width of N in seconds.
    Here it is required to find out the actual window range. E.g. if
    the desired window width is 300 seconds but the measurement rate
    is one/minute, the actual window width is 5.

    Parameters:
    -----------
    x : array of floats
        1D data vector of which the running mean is to be taken.
    N : int
        Running mean window width in seconds.
    t : array of floats
        1D time vector (in seconds since a reference time) required to
        compute the actual running mean window width.
    """

    x = x.astype(np.float64)
    x_m = copy.deepcopy(x)

    n_x = len(x)
    is_even = (N%2 == 0)        # if N is even: is_even is True

    # ii = np.arange(len(x))    # array of indices, required for the 'slow version'

    # inquire mean delta time to get an idea of how broad a window
    # must be (roughly) <-> used to speed up computation time:
    mdt = np.nanmean(t[1:] - t[:-1])
    look_range = int(np.ceil(N/mdt))
    
    # run through the array:
    look_save = 0
    for k in range(n_x):    # k, t_c in enumerate(t)?
        if k%400000 == 0: print(k/n_x)  # output required to avoid ssh connection to
                                        # be automatically dropped

        # Identify the correct running mean window width from the current
        # time t_c:
        t_c = t[k]
        if is_even: # even:
            t_c_plus = t_c + int(N/2)
            t_c_minus = t_c - int(N/2)
            # t_range = t[(t >= t_c_minus) & (t <= t_c_plus)]   # not required
        else:           # odd:
            t_c_plus = t_c + int(N/2) + 1
            t_c_minus = t_c - int(N/2)
            # t_range = t[(t >= t_c_minus) & (t <= t_c_plus)]   # not required

        # rm_range_SAVE = ii[(t >= t_c_minus) & (t <= t_c_plus)]        # very slow for large time axis array but also works
        # faster:

        if (k > look_range) and (k < n_x - look_range): # in between
            look_save = k-look_range
            rm_range = np.argwhere((t[k-look_range:k+look_range] >= t_c_minus) & (t[k-look_range:k+look_range] <= t_c_plus)).flatten() + look_save
        elif k <= look_range:   # lower end of array
            look_save = 0
            rm_range = np.argwhere((t[:k+look_range] >= t_c_minus) & (t[:k+look_range] <= t_c_plus)).flatten()
        else:   # upper end of array
            look_save = k-look_range
            rm_range = np.argwhere((t[k-look_range:] >= t_c_minus) & (t[k-look_range:] <= t_c_plus)).flatten() + look_save
            

        # moving average:
        x_m[k] = np.mean(x[rm_range])
    
    return x_m


def running_mean_pdtime(x, N, t):
    """
    Running mean of a 1D array x with a window width of N seconds.

    Parameters:
    -----------
    x : array of floats
        1D data vector of which the running mean is to be taken.
    N : int
        Running mean window width in seconds.
    t : array of floats
        1D time vector (in numpy datetim64[ns]) required to
        compute the actual running mean window width.
    """

    # first, create xarray DataArray and convert it to pandas DataFrame:
    x_DA = xr.DataArray(x, dims=['time'], coords={'time': (['time'], t)})
    x_DF = x_DA.to_dataframe(name='x')

    # compute running mean (rolling mean): center=True is recommended to have a 5-min running
    # at 2020-01-01T14:00:00 from 2020-01-01T13:57:30 until 2020-01-01T14:02:30.
    x_rm = x_DF.rolling(f"{int(N)}S", center=True).mean().to_xarray().x

    return x_rm.values


def running_mean_pd(x, N, t):
    """
    Running mean of a 1D array x with a window width of N.

    Parameters:
    -----------
    x : array of floats
        1D data vector of which the running mean is to be taken.
    N : int
        Running mean window width.
    t : array of floats
        1D index array (e.g., of floats or ints) required to
        compute the actual running mean window width.
    """

    # first, create xarray DataArray and convert it to pandas DataFrame: 
    # t doesn't have to be a time array.
    x_DA = xr.DataArray(x, dims=['time'], coords={'time': (['time'], t)})
    x_DF = x_DA.to_dataframe(name='x')

    # compute running mean (rolling mean): center=True is recommended
    x_rm = x_DF.rolling(int(N), center=True).mean().to_xarray().x

    return x_rm.values


def running_mean_time_2D(x, N, t, axis=0):

    """
    Moving average of a 2D+ array x with a window width of N in seconds.
    The moving average will be taken over the specifiec axis.
    Here it is required to find out the actual window range. E.g. if
    the desired window width is 300 seconds but the measurement rate
    is one/minute, the actual window width is 5.

    Parameters:
    -----------
    x : array of floats
        Data array (multi-dim) of which the running mean is to be taken for a
        certain axis.
    N : int
        Running mean window width in seconds.
    t : array of floats
        1D time vector (in seconds since a reference time) required to
        compute the actual running mean window width.
    axis : int
        Indicates, which axis represents the time axis, over which the moving
        average will be taken. Default: 0
    """

    # check if shape of x is correct:
    n_x = x.shape[axis]
    assert n_x == len(t)

    x = x.astype(np.float64)
    x_m = copy.deepcopy(x)

    is_even = (N%2 == 0)        # if N is even: is_even is True

    # inquire mean delta time to get an idea of how broad a window
    # must be (roughly) <-> used to speed up computation time:
    mdt = np.nanmean(np.diff(t))
    look_range = int(np.ceil(N/mdt))
    
    # run through the array:
    look_save = 0
    for k in range(n_x):    # k, t_c in enumerate(t)?
        if k%400000 == 0: print(k/n_x)  # output required to avoid ssh connection to
                                        # be automatically dropped

        # Identify the correct running mean window width from the current
        # time t_c:
        t_c = t[k]
        if is_even: # even:
            t_c_plus = t_c + int(N/2)
            t_c_minus = t_c - int(N/2)
        else:           # odd:
            t_c_plus = t_c + int(N/2) + 1
            t_c_minus = t_c - int(N/2)


        if (k > look_range) and (k < n_x - look_range): # in between
            look_save = k-look_range
            rm_range = np.argwhere((t[k-look_range:k+look_range] >= t_c_minus) & (t[k-look_range:k+look_range] <= t_c_plus)).flatten() + look_save
        elif k <= look_range:   # lower end of array
            look_save = 0
            rm_range = np.argwhere((t[:k+look_range] >= t_c_minus) & (t[:k+look_range] <= t_c_plus)).flatten()
        else:   # upper end of array
            look_save = k-look_range
            rm_range = np.argwhere((t[k-look_range:] >= t_c_minus) & (t[k-look_range:] <= t_c_plus)).flatten() + look_save
            

        # moving average:
        x_m[k] = np.mean(x[rm_range], axis=axis)
    
    return x_m


def cumsum_with_reset_xr(data: xr.DataArray, dim=None, counter_reset_val=0):
    
    """
    Compute the cumulative sum over the dimension 'dim' that resets when False is encountered 
    in the data. 
    
    Parameters:
    -----------
    data : xr.DataArray
        Data for which the cumulative sum is computed over a certain dimension.
    dim : str
        Name of the dimension over which to compute the cumulative sum.
    counter_reset_val : int
        Value to which the cumulative sum is reset once False is encountered in the data.
    """
    
    axis, _ = get_axis_from_dim_or_vice_versa(da=data, dim=dim)
    
    cumsum = data.cumsum(dim)
    reset_vals = xr.where(~data, cumsum, counter_reset_val)
    reset_vals[...] = np.maximum.accumulate(reset_vals.values, axis=axis)
    cumsum = cumsum - reset_vals
    
    return cumsum


def compute_fft1d_along_dimension(
    da: xr.DataArray, 
    axis=None, 
    dim=None, 
    compute_freq_spectrum=True,
    smpl_interval=None,
    fft_freq_unit=""):
    
    """
    Apply discrete Fast Fourier Transform (fft) to a 2D data array (da) for each index of a given axis. 
    The axis can also be inferred from the dimension name of the input data array (da).
    
    Parameters:
    -----------
    da : xr.DataArray
        Data array for which to compute fft for each index of a given dimension or axis.
    axis : int or None
        Axis number of the data array (da) for whose indices ffts are computed.
    dim : str or None
        Name of the dimension for whose indices the ffts are computed.
    compute_freq_spectrum : bool
        Whether or not to compute the frequency spectrum of the Fourier Transform. Will be used as
        coordinate of the dimension over which fft is computed if True.
    smpl_interval : float or None
        Sample interval of the data stored in the data array (1/measurement_frequency or
        1/sample_rate) over which the fft is computed (note, != dim). Used to compute the frequency
        spectrum for the fft(s).
    fft_freq_unit : str
        String describing the unit of the frequency spectrum of the fft.
    """
    
    n_data = da.shape[axis]
    fft_da = xr.DataArray(np.empty(da.shape, dtype=np.complex128),
                           dims=[dim, 'fft_freq'],
                           coords={dim: da.coords[da.dims[axis]]})
    
    for k in range(n_data):
        fft_da[{dim: k}] = np.fft.fft(da.isel({dim: k}))

    if compute_freq_spectrum:
        
        if smpl_interval is None: smpl_interval = 1.0
        
        fft_axes = np.arange(fft_da.ndim)
        n_fft = fft_da.shape[fft_axes[fft_axes != axis][0]]
        frequency_spectrum = np.fft.fftfreq(n_fft, d=smpl_interval)
        fft_da = fft_da.assign_coords({'fft_freq': (['fft_freq'], frequency_spectrum)})
        fft_da['fft_freq'].attrs['units'] = fft_freq_unit
        
    return fft_da


def get_axis_from_dim_or_vice_versa(
    da: xr.DataArray,
    axis=None, 
    dim=None,
    error_mssg=""):
    
    if (dim is None) and (axis is not None):
        dim = da.dims[axis]
    elif (axis is None) and (dim is not None):
        axis = da.get_axis_num(dim)
    elif (axis is None) and (dim is None):
        raise ValueError(error_mssg)
        
    return axis, dim


def datetime_to_epochtime(dt_array):
    
    """
    This tool creates a 1D array (or of seconds since 1970-01-01 00:00:00 UTC
    (type: float) out of a datetime object or an array of datetime objects.

    Parameters:
    -----------
    dt_array : array of datetime objects or datetime object
        Array (1D) that includes datetime objects. Alternatively, dt_array is directly a
        datetime object.
    """

    reftime = dt.datetime(1970,1,1)

    try:
        sec_epochtime = np.asarray([(dtt - reftime).total_seconds() for dtt in dt_array])
    except TypeError:   # then, dt_array is no array
        sec_epochtime = (dt_array - reftime).total_seconds()

    return sec_epochtime


def numpydatetime64_to_epochtime(npdt_array):

    """
    Converts numpy datetime64 array to array in seconds since 1970-01-01 00:00:00 UTC (type:
    float).
    Alternatively, just use "some_array.astype(np.float64)" or it might be needed to first
    convert to some_array.astype("datetime64[s]").astype(np.float64).

    Parameters:
    -----------
    npdt_array : numpy array of type np.datetime64 or np.datetime64 type
        Array (1D) or directly a np.datetime64 type variable.
    """

    sec_epochtime = npdt_array.astype(np.timedelta64) / np.timedelta64(1, 's')

    return sec_epochtime


def numpydatetime64_to_reftime(
    npdt_array, 
    reftime):

    """
    Converts numpy datetime64 array to array in seconds since a reftime as type:
    float. Reftime could be for example: "2017-01-01 00:00:00" (in UTC)

    Parameters:
    -----------
    npdt_array : numpy array of type np.datetime64 or np.datetime64 type
        Array (1D) or directly a np.datetime64 type variable.
    reftime : str
        Specification of the reference time in "yyyy-mm-dd HH:MM:SS" (in UTC).
    """

    time_dt = numpydatetime64_to_datetime(npdt_array)

    reftime = dt.datetime.strptime(reftime, "%Y-%m-%d %H:%M:%S")

    try:
        sec_epochtime = np.asarray([(dtt - reftime).total_seconds() for dtt in time_dt])
    except TypeError:   # then, time_dt is no array
        sec_epochtime = (time_dt - reftime).total_seconds()

    return sec_epochtime


def numpydatetime64_to_datetime(npdt_array):

    """
    Converts numpy datetime64 array to a datetime object array.

    Parameters:
    -----------
    npdt_array : numpy array of type np.datetime64 or np.datetime64 type
        Array (1D) or directly a np.datetime64 type variable.
    """

    sec_epochtime = npdt_array.astype(np.timedelta64) / np.timedelta64(1, 's')

    # sec_epochtime can be an array or just a float
    if sec_epochtime.ndim > 0:
        time_dt = np.asarray([dt.datetime.utcfromtimestamp(tt) for tt in sec_epochtime])

    else:
        time_dt = dt.datetime.utcfromtimestamp(sec_epochtime)

    return time_dt


def encode_time(
    DS: xr.Dataset, 
    time_var='time', 
    time_dim='time', 
    reference_period=np.datetime64("1970-01-01T00:00:00"),
    calendar="proleptic_gregorian"):
    
    """
    Encode the time dimension of a Dataset with respect to a given reference period.
    
    Parameters:
    -----------
    DS : xr.Dataset
        Dataset whose time dimension should be encoded.
    time_var : str
        Name of the time variable to be encoded
    time_dim : str
        Name of the time dimension of the variable to be encoded.
    reference_period : np.datetime64
        Reference period as np.datetime64 object, given in YYYY-MM-DDThh:mm:ss (ISO 8601).
    calendar : str
        String describing the calendar used. Numpy's datetime64 uses 'proleptic_gregorian'.
        See also https://cfconventions.org/cf-conventions/cf-conventions.html#calendar .
    """
    
    reference_period_str = str(reference_period).replace("T", " ")
    
    time_values = (DS[time_var].values - reference_period).astype('timedelta64[s]').astype(np.float64)
    if time_var == time_dim:
        DS[time_var] = time_values
    else:
        DS[time_var] = xr.DataArray(time_values, dims=time_dim)
    DS[time_var].attrs['units'] = f"seconds since {reference_period_str}"
    DS[time_var].encoding['units'] = f'seconds since {reference_period_str}'
    DS[time_var].encoding['dtype'] = 'double'
    
    if time_var == 'time': DS[time_var].attrs['standard_name'] = 'time'
    DS[time_var].attrs['calendar'] = calendar
    
    return DS


def write_basic_attributes(DS: xr.Dataset):
    
    DS.attrs['institution'] = "Institute for Geophysics and Meteorology, University of Cologne, Cologne, Germany"
    DS.attrs['contact'] = "Andreas Walbroel (a.walbroel@uni-koeln.de, https://orcid.org/0000-0003-2603-2724)"
    DS.attrs['author'] = "Andreas Walbroel"
    DS.attrs['licence'] = "CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/"
    
    return DS


def update_netCDF_file_history(
    DS: xr.Dataset, 
    script_name: str, 
    summary_str="", 
    history_attr='history',
    history_attr_exists=True):

    """
    Updates the history of an xarray Dataset that shall be saved to a netCDF file.
    
    Parameters:
    -----------
    DS : xr.Dataset
        Dataset to be saved and where the attribute is added.
    script_name : str
        Name of the script that was mainly used to update/modify DS.
    summary_str : str
        String that concisely describes the changes made to DS.
    history_attr : str
        Name of the attribute where the history of DS is described.
    history_attr_exists : bool
        Boolean indicating whether the hisotry attribute already exists. If False,
        it's created.
    """
    
    if not history_attr_exists: DS.attrs[history_attr] = ""

    attr_add = ""
    if history_attr_exists and (";" not in DS.attrs[history_attr][-2:]):
        attr_add = "; "
    DS.attrs[history_attr] += (f"{attr_add}{str(np.datetime64('now')).replace('T', ' ')}" +
                               f", {summary_str} with {script_name}; ")
    
    return DS


def convert_units(data: np.ndarray, unit_conv_list: list):
    
    """
    Convert some units: first (second) element of list: must be added to the data (the data 
    must be multiplied by) to get to the desired unit. The multiplication is performed after 
    adding the unit_conv_list[0] value.
    """
    
    return (data + unit_conv_list[0])*unit_conv_list[1]


def convert_units_back(data: np.ndarray, unit_conv_list: list):
    
    """
    Inverse of 'convert_units'. Undo conversion changes.
    """
    
    return (data / unit_conv_list[1]) - unit_conv_list[0]


def compute_DOY(
    time,
    return_dt=True,
    reshape=False):

    """
    Compute the cos and sin of the day of the year for a given time.

    Parameters:
    -----------
    time : numpy array of floats or float or xarray.DataArray
        Time data (must be in seconds since 1970-01-01 00:00:00 UTC or np.datetime64) used to compute 
        the cos and sin of the day of the year.
    return_dt : bool
        If True the datetime object/array used for the computation is returned as well.
    reshape : bool
        If True an additional dimension of length 1 will be added to DOY_1 and DOY_2 via
        reshaping.
    """
    
    if type(time) == xr.DataArray:
        time = time.values
        
    if (type(time) == np.ndarray) and (type(time[0]) == np.datetime64):
        DOY = (time - time.astype('datetime64[Y]')).astype('timedelta64[D]').astype(np.float64)*2.*np.pi/365.
    else:
        time_dt = np.asarray([dt.datetime.fromtimestamp(ttt, tz=dt.timezone.utc) for ttt in time])
        DOY = np.asarray([(ttt - dt.datetime(ttt.year,1,1, tzinfo=dt.timezone.utc)).days*2*np.pi/365 for ttt in time_dt])
        
    DOY_1 = np.cos(DOY)
    DOY_2 = np.sin(DOY)

    if reshape:
        n_data = len(time)
        DOY_1 = np.reshape(DOY_1, (n_data,1))
        DOY_2 = np.reshape(DOY_2, (n_data,1))

    if return_dt:
        return DOY_1, DOY_2, time_dt
    else:
        return DOY_1, DOY_2


def break_str_into_lines(
    le_string,
    n_max,
    split_at=' ',
    keep_split_char=False):

    """
    Break a long strings into multiple lines if a certain number of chars may
    not be exceeded per line. String will be split into two lines if its length
    is > n_max but <= 2*n_max.

    Parameters:
    -----------
    le_string : str
        String that will be broken into several lines depending on n_max.
    n_max : int
        Max number of chars allowed in one line.
    split_at : str
        Character to look for where the string will be broken. Default: space ' '
    keep_split_char : bool
        If True, the split char indicated by split_at will not be removed (useful for "-" as split char).
        Default: False
    """

    n_str = len(le_string)
    if n_str > n_max:
        # if string is > 2*n_max, then it has to be split into three lines, ...:
        n_lines = (n_str-1) // n_max        # // is flooring division

        # look though the string in backwards direction to find the first space before index n_max:
        le_string_bw = le_string[::-1]
        new_line_str = "\n"

        for k in range(n_lines):
            space_place = le_string_bw.find(split_at, n_str - (k+1)*n_max)
            if keep_split_char:
                le_string_bw = le_string_bw[:space_place].replace("\n","") + new_line_str + le_string_bw[space_place:]
            else:
                le_string_bw = le_string_bw[:space_place] + new_line_str + le_string_bw[space_place+1:]

        # reverse the string again
        le_string = le_string_bw[::-1]

    return le_string


def lowercase_letter_from_number(ix: int):
    
    """
    Returns the (ix+1)-th lower case letter of the alphabet (ix=0 -> a, ix=1 ->b, ...).
    
    Parameters:
    -----------
    ix : int
        Integer for indexing the alphabet (0 == a, 1 == b, ...).
    """
    
    base = ord('a')
    
    return chr(base + ix)


def bin_to_dec(b_in):

    """
    Converts a binary number given as string to normal decimal number (as integer).

    Parameters:
    -----------
    b_in : str
        String of a binary number that may either directly start with the
        binary number or start with "0b".
    """

    d_out = 0       # output as decimal number (int or float)
    if "b" in b_in:
        b_in = b_in[b_in.find("b")+1:]  # actual bin number starts after "b"
    b_len = len(b_in)

    for ii, a in enumerate(b_in): d_out += int(a)*2**(b_len-ii-1)

    return d_out


def dec_to_binary_string(data_1d: np.ndarray, max=None):
    
    """
    Data of integer type are converted to a binary string representation.
    E.g., 15 would be '1111' or similar with zero padding '00001111'.
    """
    
    if max == None:
        max = int(np.nanmax(data_1d))
    
    N = int(np.ceil(np.log2(max))) + 2 # + 2 because of '0b'
    format_ = f"#0{N}b"
    bin_data = np.asarray([format(int(num), format_)[2:] for num in data_1d])
    
    return bin_data


def flag_values_to_2d_flag_bits(flags, n_bits=None):
    
    """
    Flag values are converted to a 2d array where the second axis are the flag bits.
    
    Parameters:
    -----------
    flags : np.ndarray or xr.DataArray
        Flag values.
    bits : int
        Number of quality bits.
    """
    
    flag_bits = [np.binary_repr(int(qf), width=n_bits) for qf in flags]
    flag_bits = np.array(list(map(list, flag_bits))).astype(int)
    
    return flag_bits


def mwr_pro_flags_to_2d_bits(
    DS,
    quality_flag_name,
    n_bits):
    
    """
    Convert the quality flag values to flag bit string and then to a 2D array with dimensions
    (time,bits).
    
    Parameters:
    -----------
    DS : xarray dataset
        Dataset containing the quality flag values on the dimension 'time'.
    quality_flag_name : str
        String to indicate the name of the quality flag in the dataset DS.
    n_bits : int
        Number of quality bits.
    """
    
    flag_bits = flag_values_to_2d_flag_bits(DS[quality_flag_name].values, n_bits=n_bits)
    n_bits = flag_bits.shape[1]
    DS[quality_flag_name+'_bits'] = xr.DataArray(flag_bits, dims=['time', 'bit'],
                                                 coords={'bit': (['bit'], n_bits-np.arange(n_bits))})
    
    return DS


def compute_retrieval_statistics(
    x_stuff,
    y_stuff,
    compute_stddev=False):

    """
    Compute bias, RMSE and Pearson correlation coefficient (and optionally the standard deviation)
    from x and y data.

    Parameters:
    x_stuff : float or array of floats
        Data that is to be plotted on the x axis.
    y_stuff : float or array of floats
        Data that is to be plotted on the y axis.
    compute_stddev : bool
        If True, the standard deviation is computed (bias corrected RMSE).
    """

    where_nonnan = np.where(~np.isnan(x_stuff+y_stuff))[0]
                    # -> must be used to ignore nans in corrcoef
    stat_dict = {   'N': np.count_nonzero(~np.isnan(x_stuff+y_stuff)),
                    'bias': np.nanmean(y_stuff - x_stuff),
                    'rmse': np.sqrt(np.nanmean((x_stuff - y_stuff)**2)),
                    'R': np.corrcoef(x_stuff[where_nonnan], y_stuff[where_nonnan])[0,1]}

    if compute_stddev:
        stat_dict['stddev'] = np.sqrt(np.nanmean((x_stuff - (y_stuff - stat_dict['bias']))**2))

    return stat_dict


def compute_RMSE_profile(
    x,
    x_o,
    which_axis=0):

    """
    Compute RMSE 'profile' of a i.e., (height x time)-matrix (e.g. temperature profile):
    RMSE(z_i) = sqrt(mean((x - x_o)^2, dims='time'))
    
    Parameters:
    -----------
    x : 2D array of numerical
        Data matrix whose deviation from a reference is desired.
    x_o : 2d array of numerical
        Data matrix of the reference.
    which_axis : int
        Indicator which axis is to be averaged over. For the RMSE profile, you would
        want to average over time!
    """

    if which_axis not in [0, 1]:
        raise ValueError("'which_axis' must be either 0 or 1!")

    return np.sqrt(np.nanmean((x - x_o)**2, axis=which_axis))


def compute_error_profiles(x, x_o, which_axis=0, height_axis=-1, compute_stddev=False):
    
    """
    Compute RMSE, bias and standard deviation profiles of data x with respect to a reference x_o.
    It is expected that the height axis is axis=-1.
    
    Parameters:
    -----------
    x : array of numerical
        Data array whose deviation from a reference is desired.
    x_o : array of numerical
        Data array of the reference.
    which_axis : int
        Indicator which axis is to be averaged over. For the RMSE profile, you would
        want to average over time!
    height_axis : int
        Indicator which axis is the height axis.
    compute_stddev : bool
        Whether or not to compute standard deviation (bias corrected RMSE) profiles.
    """
    
    n_hgt = x_o.shape[height_axis]
    no_nan_idx = np.where(np.count_nonzero(np.isnan(x_o+x), axis=height_axis) < n_hgt/2)[0]
    
    error_dict = dict()
    if x.ndim > 1:
        error_dict['fraction_nonnan'] = len(no_nan_idx) / np.prod(x.shape[:-1])
    error_dict['N'] = 1
    x_o = x_o[no_nan_idx,:]
    x = x[no_nan_idx,:]

    error_dict['rmse'] = compute_RMSE_profile(x, x_o, which_axis=0)
    error_dict['bias'] = np.nanmean(x - x_o, axis=0)
    error_dict['stddev'] = compute_RMSE_profile(x - error_dict['bias'], x_o, which_axis=0)

    x_mean = np.nanmean(x_o, axis=0)
    error_dict['rmse_rel'] = error_dict['rmse'] / x_mean
    error_dict['bias_rel'] = error_dict['bias'] / x_mean
    error_dict['stddev_rel'] = error_dict['stddev'] / x_mean
    
    return error_dict


def interp_w_avg(
    hgt,
    data,
    hgt_ip,
    respect_weights=False,
    ):

    """
    Interpolate data, which is on height grid hgt, to the target height grid hgt_ip by averaging
    over layers of the hgt_ip grid. Please make sure that the initial height grid hgt is free of
    gaps (nans) and monotonically increasing or decreasing.

    Parameters:
    hgt : array of floats
        Height grid of the data (e.g., in m, hPa or Pa). Must be a 1D array.
    data : array of floats
        Data that is on hgt and is to be interpolated onto hgt_ip. The height axis must be the
        last axis (axis -1).
    hgt_ip : array of floats
        Target height grid (e.g., in m, hPa, Pa).
    respect_weights : bool
        If True, weights for each hgt_ip layer (:= basically 0.5*(hgt_ip[:-1] + hgt_ip[1:])) are
        computed manually. The weights are then used for a weighted average, which improves 
        accuracy but reduces computational efficiency. If False, data will simply be averaged 
        over each hgt_ip layer.
    """

    # initialize array:
    data_shape = data.shape
    n_hgt_ip = len(hgt_ip)
    n_hgt = len(hgt)
    data_ip = np.full((data_shape[:-1] + (n_hgt_ip,)), np.nan)      # recycle old shape except height axis

    # create input height LAYER array:
    hgt_lay = np.zeros((n_hgt+1,))
    hgt_lay[1:-1] = 0.5*(hgt[:-1] + hgt[1:])
    hgt_lay[0] = hgt[0] - 0.5*(hgt[1] - hgt[0])
    hgt_lay[-1] = hgt[-1] + 0.5*(hgt[-1] - hgt[-2])

    # create target height LAYER array:
    hgt_ip_lay = np.zeros((n_hgt_ip+1,))
    hgt_ip_lay[1:-1] = 0.5*(hgt_ip[:-1] + hgt_ip[1:])
    hgt_ip_lay[0] = hgt_ip[0] - 0.5*(hgt_ip[1] - hgt_ip[0])
    hgt_ip_lay[-1] = hgt_ip[-1] + 0.5*(hgt_ip[-1] - hgt_ip[-2])

    # Loop over height axis and average e.g., hgt_ip_lay[0] to hgt_ip_lay[1] for data_ip[0]:
    # Differentiate between ascending (e.g., height) and descending (e.g., pressure) height axis
    # to set the idx_lay search correctly (including the boundary value closer to the surface; excluding
    # the upper):
    hgt_diff = np.diff(hgt)
    if ~np.any(hgt_diff < 0): # similar to np.all(hgt_diff > 0), but allows nans in hgt_diff

        if respect_weights:
            for k in range(n_hgt_ip):
                idx_lay = np.where((hgt >= hgt_ip_lay[k]) & (hgt < hgt_ip_lay[k+1]))[0]

                # check if top of grid is exceeded or current hgt_ip layer is nan:
                if (hgt_ip_lay[k] > hgt_lay[-1]) or (np.isnan(hgt_ip_lay[k] + hgt_ip_lay[k+1])): 
                    continue        # then, data_ip[...,k] = np.nan

                # compute weights (see notes, p. 187-192):
                # set lower boundary of current layer to hgt[0] if it's the first layer:
                if hgt_ip_lay[k] < hgt[0]:
                    ll_bound = hgt[0]
                else:
                    ll_bound = hgt_ip_lay[k]


                n_idx_lay = len(idx_lay)
                if n_idx_lay >= 1:

                    # weight at upper layer boundary:
                    if idx_lay[-1] < n_hgt-1:

                        # Extension of upper boundary of current layer evtl needed: Check if the top of the hgt grid
                        # is reached:
                        hgt_ext = hgt_lay[idx_lay[-1]+1]
                        if hgt_ext < hgt_ip_lay[k+1]:   # then include part of hgt grid of next layer and extend idx_lay
                            ww_upper = (hgt_ip_lay[k+1] - hgt_ext) / (hgt_ip_lay[k+1] - ll_bound)
                            idx_lay = np.concatenate((idx_lay, np.array([idx_lay[-1]+1])))

                        else:   # then, evtl. exclude part of hgt_lay[idx_lay[-1]:idx_lay[-1]+2]-layer
                            ww_upper = ((hgt_ip_lay[k+1] - np.nanmax(np.array([hgt[0], hgt_lay[idx_lay[-1]], hgt_ip_lay[k]])) ) / 
                                        (hgt_ip_lay[k+1] - ll_bound))

                    else: # top of hgt grid
                        ww_upper = ((hgt_ip_lay[k+1] - np.nanmax( np.array([hgt_lay[idx_lay[-1]], hgt_ip_lay[k]]) ) ) / 
                                    (hgt_ip_lay[k+1] - ll_bound))

                    # weight at lower layer boundary:
                    if idx_lay[0] > 0:
                        hgt_lower = hgt_lay[idx_lay[0]]
                        if hgt_lower > hgt_ip_lay[k]:   # include part of hgt[idx_lay[0]-1]-layer and extend idx_lay
                            ww_lower = (hgt_lower - hgt_ip_lay[k]) / (hgt_ip_lay[k+1] - hgt_ip_lay[k])
                            idx_lay = np.concatenate((np.array([idx_lay[0]-1]), idx_lay))

                        else:   # evtl. exclude part of hgt[idx_lay[0]]-layer
                            if idx_lay[0] == n_hgt-1:   # then, it's at the top of the hgt grid
                                # no hgt grid point will follow, therefore, ww_lower must be 1
                                ww_lower = 1.0
                            else:
                                ww_lower = ((np.nanmin(np.array([hgt_lay[idx_lay[0]+1], hgt_ip_lay[k+1]])) - hgt_ip_lay[k]) / 
                                            (hgt_ip_lay[k+1] - hgt_ip_lay[k]))

                    else: # bottom of hgt grid:
                        ww_lower = ((np.nanmin(np.array([hgt_lay[idx_lay[0]+1], hgt_ip_lay[k+1]])) - hgt[idx_lay[0]]) / 
                                    (hgt_ip_lay[k+1] - ll_bound))


                else:

                    # no hgt in current hgt_ip layer found.
                    # check idx_lay of prev layer concat to idx_lay:
                    if k > 0:
                        idx_lay_prev = np.where((hgt >= hgt[0]) & (hgt < hgt_ip_lay[k]))[0]
                        if len(idx_lay_prev) > 0:
                            idx_lay = np.concatenate((np.array([idx_lay_prev[-1]]), idx_lay))
                            if idx_lay[0] < n_hgt-1:
                                ww_lower = ((np.nanmin(np.array([hgt_lay[idx_lay[0]+1], hgt_ip_lay[k+1]])) - hgt_ip_lay[k]) / 
                                            (hgt_ip_lay[k+1] - hgt_ip_lay[k]))
                            else:
                                ww_lower = 1.0      # because no other hgt value contributes
                        else:
                            ww_lower = 0.0
                    else:

                        # case where no hgt values are found within first hgt_ip layer: 
                        # check layer between first hgt (bottom of hgt) and first hgt_ip_lay
                        idx_lay_prev = np.where((hgt >= hgt[0]) & (hgt < hgt_ip_lay[0]))[0]
                        if len(idx_lay_prev) > 0:
                            idx_lay = np.concatenate((np.array([idx_lay_prev[-1]]), idx_lay))
                            if hgt_lay[idx_lay[0]+1] >= hgt_ip_lay[k+1]:
                                ww_lower = 1.0      # because 'next' hgt-layer (hgt_lay[idx_lay[0]+1]) doesn't contribute
                            else:
                                ww_lower = (hgt_lay[idx_lay[0]+1] - hgt_ip_lay[k]) / (hgt_ip_lay[k+1] - ll_bound)
                        else:
                            print("data_tools.py.interp_w_avg: Skipping height level")
                            continue        # skip this height level because no data seems to be available
                            raise RuntimeError("It seems like the target height grid contains height levels at " +
                                                "its lower boundary that are not included in the base height grid. " +
                                                "Please provide a target height grid whose lower boundary is at or " +
                                                "above the lower boundary of the base height grid.")


                    # check next layer and concat to idx_lay:
                    if k < len(hgt_ip_lay)-2:
                        idx_lay_next = np.where((hgt >= hgt_ip_lay[k+1]) & (hgt < hgt_ip_lay[k+2]))[0]
                        if len(idx_lay_next) == 0: # then manually use the next one:
                            idx_lay_next = np.array([idx_lay[-1] + 1])

                        if len(idx_lay_next) > 0:
                            if idx_lay_next[0] < n_hgt:
                                idx_lay = np.concatenate((idx_lay, np.array([idx_lay_next[0]])))
                                ww_upper = ((hgt_ip_lay[k+1] - np.nanmax( np.array([hgt_lay[idx_lay[-1]], hgt_ip_lay[k]]) )) / 
                                            (hgt_ip_lay[k+1] - ll_bound))

                            else:   # top of grid. no more hgt grid value that could contribute
                                ww_upper = 0.0
                    else:
                        ww_upper = 0.0

                # set ww_upper and ww_lower to 0 if smaller than 0 and check if they are <= 1:
                if ww_upper < 0: ww_upper = 0.0
                if ww_lower < 0: ww_lower = 0.0
                if ww_upper > 1: pdb.set_trace()
                if ww_lower > 1: pdb.set_trace()


                # compute weights:
                n_idx_lay = len(idx_lay)    # updated idx_lay length
                ww = np.full((n_idx_lay,), np.nan)
                for kk, ii in enumerate(idx_lay):
                    if (kk > 0) and (kk < n_idx_lay-1):
                        ww[kk] = (hgt_lay[ii+1] - hgt_lay[ii]) / (hgt_ip_lay[k+1] - ll_bound)

                    # eventually, parts of the next or previous hgt_ip_lay must be included at the
                    # boundaries the current layer: check idx_lay[0] and [-1]:
                    if ii == idx_lay[0]:
                        ww[kk] = ww_lower
                    elif ii == idx_lay[-1]:
                        ww[kk] = ww_upper

                if np.any(ww < 0): pdb.set_trace()


                # compute weighted mean over those regions:
                if n_idx_lay > 0:
                    if (ww.sum() > 1.05) | (ww.sum() < 0.995): 
                        # can occur for several reasons: one: hgt[0] is greater than the 
                        # upper boundary of the current hgt_ip layer (hgt_ip_lay[k+1]):
                        if hgt[0] > hgt_ip_lay[k+1]: 
                            continue        # skip because original height grid doesn't have data
                                            # within this hgt_ip layer

                        else:   # other reasons??
                            pdb.set_trace()

                    # handle nans:
                    if np.all(np.isnan(data[...,idx_lay])):
                        data_ip[...,k] = np.nan

                    elif np.any(np.isnan(data[...,idx_lay])):

                        # flatten data to handle multi-dimensional data:
                        data_flat = data[...,idx_lay].ravel()
                        where_nn = np.where(~np.isnan(data_flat))[0]

                        # extend weights array to data shape:
                        new_shape = data.shape[:-1] + (ww.shape[0],)
                        ww_r = np.broadcast_to(ww, new_shape).ravel()
                        ww_nonnan = np.zeros(ww_r.shape)

                        # check for full-nan entries and set ww_r to zero in those regions where
                        # all idx_lay entries are nan:
                        only_nans = np.where((np.count_nonzero(np.isnan(data[...,idx_lay]), axis=-1) == len(idx_lay)).ravel())[0]
                        for onn in only_nans: ww_r[n_idx_lay*onn:n_idx_lay*(onn+1)] = np.nan


                        # sum up weights that would correspond to nans:
                        n_where_nn = len(where_nn)
                        for i_nn,w_nn in enumerate(where_nn):
                            # only edit weights within current data batch; therefore, identify, which ww_nonnan indices 
                            # may be accessed:
                            i_x = int(w_nn/n_idx_lay)       # current data sample batch
                            idx_batch = np.arange(i_x*n_idx_lay, (i_x+1)*n_idx_lay)     # all indices of current batch

                            # if i_nn == 0: watch out not to access index -1!
                            if i_nn > 0:
                                if w_nn > idx_batch[0]: # then add weights of nan values before w_nn to the weight at w_nn
                                    prev_nonnan = where_nn[i_nn-1]
                                    if prev_nonnan < idx_batch[0]:  # then, the first index of the current batch is nan
                                        prev_nonnan = idx_batch[0]-1            # prev nonnan is either > idx_batch[0] or w_nn is the first nonnan
                                    # prev_nonnan = np.nanmax(np.array([where_nn[i_nn-1], idx_batch[0]]))       # prev nonnan is either > idx_batch[0]
                                                                                                            # # or w_nn is the first nonnan
                                    ww_nonnan[w_nn] = np.nansum(ww_r[prev_nonnan+1:w_nn+1]) # sum over index following the prev nonnan until w_nn

                                else:
                                    ww_nonnan[w_nn] = ww_r[w_nn]        # no summation needed because w_nn=first nonnan of batch

                            else:   # then, w_nn is the first nonnan index:
                                ww_nonnan[w_nn] = np.nansum(ww_r[idx_batch[0]:w_nn+1])


                            # check if w_nn is also the last nonnan of current batch: Catch index out of bounds error
                            if i_nn == n_where_nn-1:
                                ww_nonnan[w_nn] += np.nansum(ww_r[w_nn+1:idx_batch[-1]+1])
                            else:
                                if where_nn[i_nn+1] not in idx_batch:   # then also add weights of remaining indices of the batch
                                    ww_nonnan[w_nn] += np.nansum(ww_r[w_nn+1:idx_batch[-1]+1])


                        # make sure that ww_nonnan.sum() is 1
                        n_data_samples = 1      # number of data samples of the current height layer
                        if len(new_shape) > 1: n_data_samples = new_shape[0]
                        ww_nonnan_sum = ww_nonnan.sum() / (n_data_samples - len(only_nans))
                        if (ww_nonnan_sum > 1.05) | (ww_nonnan_sum < 0.995): pdb.set_trace()    # debug


                        # return to original shape and compute weighted mean:
                        ww_nonnan = np.reshape(ww_nonnan, data[...,idx_lay].shape)
                        data_ip[...,k] = np.nansum(ww_nonnan*data[...,idx_lay], axis=-1)

                        # set those data_ip values to nan where the corresponding data[...,idx_lay] showed
                        # only nans:
                        data_ip_r = data_ip[...,k].ravel()
                        data_ip_r[only_nans] = np.nan
                        data_ip[...,k] = np.reshape(data_ip_r, data_ip[...,k].shape)

                    else:
                        data_ip[...,k] = np.nansum(ww*data[...,idx_lay], axis=-1)

        else:   # just compute weighted mean
            for k in range(n_hgt_ip):
                idx_lay = np.where((hgt >= hgt_ip_lay[k]) & (hgt < hgt_ip_lay[k+1]))[0]
                if len(idx_lay) > 0:
                    data_ip[...,k] = np.nanmean(data[...,idx_lay], axis=-1)

    elif ~np.any(hgt_diff > 0):     # similar to np.all(hgt_diff < 0), but allows nans

        if respect_weights:
            pdb.set_trace()     # invert data[...,:], hgt, hgt_ip, hgt_ip_lay?
        else:
            for k in range(n_hgt_ip):
                idx_lay = np.where((hgt <= hgt_ip_lay[k]) & (hgt > hgt_ip_lay[k+1]))[0]
                if len(idx_lay) > 0:
                    data_ip[...,k] = np.nanmean(data[...,idx_lay], axis=-1)

    else:
        print("data_tools.py.interp_w_avg: Height axis does not monotonically increase or decrease.")
        pdb.set_trace() # debug

    return data_ip


def vertical_interpolation(
    height: np.ndarray, 
    height_data: np.ndarray,
    data: np.ndarray,
    interp_type='avg'):
    
    """
    Interpolating from height_data to height (target height grid).
    """
    
    if interp_type == 'linear':             # simple and fast
        if data.ndim == 2:
            data_ip = np.full((data.shape[0], len(height)), np.nan)
            for k in range(data.shape[0]):
                data_ip[k,:] = np.interp(height, height_data, data[k,:], right=np.nan)
        else:
            data_ip = np.interp(height, height_data, data, right=np.nan)
            
    elif interp_type == 'avg':
        data_ip = interp_w_avg(height_data, data, height, respect_weights=False)
        
    elif interp_type == 'weighted_avg':     # slowest but most exact
        data_ip = interp_w_avg(height_data, data, height, respect_weights=True)
        
    return data_ip


def build_K_reg(
    y,
    order=1):

    """
    Constructs the observation matrix typically used for regression retrievals where the
    rows indicate the samples (i.e., time series). The first column usually contains "1"
    only and the remaining columns contain observations in first and higher order.

    Parameters:
    -----------
    y : array of floats
        Observation vector. Must be a numpy array with M observations and N samples. The 
        shape must be N x M. (Also if M == 1, y must be a 2D array.)
    order : int
        Defines the order of the regression equation. Options: i.e., 1, 2, 3. Default:
        1
    """

    n_obs = y.shape[1]      # == M
    n_samples = y.shape[0]  # == N

    assert y.shape == (n_samples,n_obs)

    # generate regression matrix K out of obs vector:
    K_reg = np.ones((n_samples, order*n_obs+1))
    K_reg[:,1:n_obs+1] = y

    if order > 1:
        for kk in range(order-1):
            jj = kk + 1
            K_reg[:,jj*n_obs+1:(jj+1)*n_obs+1] = y**(jj+1)

    return K_reg


def regression(
    x,
    y,
    order=1):
    
    """
    Computes regression coefficients m_est to map observations y (i.e., brightness temperatures)
    to state variable x (i.e., temperature profile at one height level, or IWV). The regression
    order can also be specified.
    
    Parameters:
    -----------
    x : array of floats
        State variable vector. Must be a numpy array with N samples (N = training data size).
    y : array of floats
        Observation vector. Must be a numpy array with M observations (i.e., M frequencies) 
        and N samples. The shape must be N x M. (Also if M == 1, y must be a 2D array.)
    order : int
        Defines the order of the regression equation. Options: i.e., 1, 2, 3. Default:
        1
    """

    # Generate matrix from observations:
    K_reg = build_K_reg(y, order)

    # compute m_est
    K_reg_T = K_reg.T
    m_est = np.linalg.inv(K_reg_T@K_reg)@K_reg_T@x

    return m_est


def Gband_double_side_band_average(
    TB,
    freqs,
    xarray_compatibility=False,
    freq_dim_name=""):

    """
    Computes the double side band average of TBs that contain both
    sides of the G band absorption line. Returns either only the TBs
    or both the TBs and frequency labels with double side band avg.
    If xarray_compatibility is True, also more dimensional TB arrays
    can be included. Then, also the frequency dimension name must be
    supplied.

    Parameters:
    -----------
    TB : array of floats
        Brightness temperature array. Must have the following shape
        (time x frequency). More dimensions and other shapes are only
        allowed if xarray_compatibility=True.
    freqs : array of floats
        1D Array containing the frequencies of the TBs. The array must be
        sorted in ascending order.
    xarray_compatibility : bool
        If True, xarray utilities can be used, also allowing TBs of other
        shapes than (time x frequency). Then, also freq_dim_name must be
        provided.
    freq_dim_name : str
        Name of the xarray frequency dimension. Must be specified if 
        xarray_compatibility=True.
    """

    if xarray_compatibility and not freq_dim_name:
        raise ValueError("Please specify 'freq_dim_name' when using the xarray compatible mode.")

    # Double side band average for G band if G band frequencies are available, which must first be clarified:
    # Determine, which frequencies are around the G band w.v. absorption line:
    g_upper_end = 183.31 + 15
    g_lower_end = 183.31 - 15
    g_freq = np.where((freqs > g_lower_end) & (freqs < g_upper_end))[0]
    non_g_freq = np.where(~((freqs > g_lower_end) & (freqs < g_upper_end)))[0]

    TB_dsba = copy.deepcopy(TB)

    if g_freq.size > 0: # G band within frequencies
        g_low = np.where((freqs <= 183.31) & (freqs >= g_lower_end))[0]
        g_high = np.where((freqs >= 183.31) & (freqs <= g_upper_end))[0]

        assert len(g_low) == len(g_high)
        if not xarray_compatibility:
            for jj in range(len(g_high)):
                TB_dsba[:,jj] = (TB[:,g_low[-1-jj]] + TB[:,g_high[jj]])/2.0

        else:
            for jj in range(len(g_high)):
                # TB_dsba[{freq_dim_name: jj}] = (TB[{freq_dim_name: g_low[-1-jj]}] + TB[{freq_dim_name: g_high[jj]}])/2.0
                TB_dsba[{freq_dim_name: g_high[jj]}] = (TB[{freq_dim_name: g_low[-1-jj]}] + TB[{freq_dim_name: g_high[jj]}])/2.0
    else:
        return TB, freqs


    # Indices for sorting:
    idx_have = np.concatenate((g_high, non_g_freq), axis=0)
    idx_sorted = np.argsort(idx_have)

    # truncate and append the unedited frequencies (e.g. 243 and 340 GHz):
    if not xarray_compatibility:
        TB_dsba = TB_dsba[:,:len(g_low)]
        TB_dsba = np.concatenate((TB_dsba, TB[:,non_g_freq]), axis=1)

        # Now, the array just needs to be sorted correctly:
        TB_dsba = TB_dsba[:,idx_sorted]

        # define freq_dsba (usually, the upper side of the G band is then used as
        # frequency label:
        freq_dsba = np.concatenate((freqs[g_high], freqs[non_g_freq]))[idx_sorted]

    else:
        # TB_dsba = TB_dsba[{freq_dim_name: slice(0,len(g_low))}]
        TB_dsba = TB_dsba[{freq_dim_name: g_high}]
        TB_dsba = xr.concat([TB_dsba, TB[{freq_dim_name: non_g_freq}]], dim=freq_dim_name)

        # Now, the array just needs to be sorted correctly:
        TB_dsba = TB_dsba[{freq_dim_name: idx_sorted}]

        # define freq_dsba (usually, the upper side of the G band is then used as
        # frequency label:
        freq_dsba = xr.concat([freqs[g_high], freqs[non_g_freq]], dim=freq_dim_name)[idx_sorted]


    return TB_dsba, freq_dsba


def Fband_double_side_band_average(
    TB,
    freqs,
    xarray_compatibility=False,
    freq_dim_name=""):

    """
    Computes the double side band average of TBs that contain both
    sides of the F band absorption line. Returns either only the TBs
    or both the TBs and frequency labels with double side band avg.

    Parameters:
    -----------
    TB : array of floats
        Brightness temperature array. Must have the following shape
        (time x frequency).
    freqs : array of floats
        1D Array containing the frequencies of the TBs. The array must be
        sorted in ascending order.
    xarray_compatibility : bool
        If True, xarray utilities can be used, also allowing TBs of other
        shapes than (time x frequency). Then, also freq_dim_name must be
        provided.
    freq_dim_name : str
        Name of the xarray frequency dimension. Must be specified if 
        xarray_compatibility=True.
    """

    if xarray_compatibility and not freq_dim_name:
        raise ValueError("Please specify 'freq_dim_name' when using the xarray compatible mode.")

    # Double side band average for F band if F band frequencies are available, which must first be clarified:
    # Determine, which frequencies are around the F band w.v. absorption line:
    upper_end = 118.75 + 10
    lower_end = 118.75 - 10
    f_freq = np.where((freqs > lower_end) & (freqs < upper_end))[0]
    non_f_freq = np.where(~((freqs > lower_end) & (freqs < upper_end)))[0]

    TB_dsba = copy.deepcopy(TB)
    
    if f_freq.size > 0: # F band within frequencies
        low = np.where((freqs <= 118.75) & (freqs >= lower_end))[0]
        high = np.where((freqs >= 118.75) & (freqs <= upper_end))[0]

        assert len(low) == len(high)
        if not xarray_compatibility:
            for jj in range(len(high)):
                TB_dsba[:,jj] = (TB[:,low[-1-jj]] + TB[:,high[jj]])/2.0

        else:
            for jj in range(len(high)):
                TB_dsba[{freq_dim_name: jj}] = (TB[{freq_dim_name: low[-1-jj]}] + TB[{freq_dim_name: high[jj]}])/2.0


    # Indices for sorting:
    idx_have = np.concatenate((high, non_f_freq), axis=0)
    idx_sorted = np.argsort(idx_have)

    # truncate and append the unedited frequencies (e.g. 243 and 340 GHz):
    if not xarray_compatibility:
        TB_dsba = TB_dsba[:,:len(low)]
        TB_dsba = np.concatenate((TB_dsba, TB[:,non_f_freq]), axis=1)

        # Now, the array just needs to be sorted correctly:
        TB_dsba = TB_dsba[:,idx_sorted]

        # define freq_dsba (usually, the upper side of the G band is then used as
        # frequency label:
        freq_dsba = np.concatenate((freqs[high], freqs[non_f_freq]))[idx_sorted]

    else:
        TB_dsba = TB_dsba[{freq_dim_name: slice(0,len(low))}]
        TB_dsba = xr.concat([TB_dsba, TB[{freq_dim_name: non_f_freq}]], dim=freq_dim_name)

        # Now, the array just needs to be sorted correctly:
        TB_dsba = TB_dsba[{freq_dim_name: idx_sorted}]

        # define freq_dsba (usually, the upper side of the G band is then used as
        # frequency label:
        freq_dsba = xr.concat([freqs[high], freqs[non_f_freq]], dim=freq_dim_name)[idx_sorted]

    return TB_dsba, freq_dsba


def select_MWR_channels(
    TB,
    freq,
    band,
    return_idx=0):

    """
    This function selects certain frequencies (channels) of brightness temperatures (TBs)
    from a given set of TBs. The output will therefore be a subset of the input TBs. Single
    frequencies cannot be selected but only 'bands' (e.g. K band, V band, ...). Combinations
    are also possible.

    Parameters:
    -----------
    TB : array of floats
        2D array (i.e., time x freq; freq must be the second dimension) or higher dimensional
        array (where freq must be on axis -1) of TBs (in K).
    freq : array of floats
        1D array of frequencies (in GHz).
    band : str
        Specify the frequencies to be selected. Valid options:
        'K': 20-40 GHz, 'V': 50-60 GHz, 'W': 85-95 GHz, 'F': 110-130 GHz, 'G': 170-200 GHz,
        '243/340': 240-350 GHz
        Combinations are also possible: e.g. 'K+V+W' = 20-95 GHz
    return_idx : int
        If 0 the frq_idx list is not returned and merely TB and freq are returned.
        If 1 TB, freq, and frq_idx are returned. If 2 only frq_idx is returned.
    """

    # define dict of band limits:
    band_lims = {   'K': [20, 40],
                    'V': [50, 60],
                    'W': [85, 95],
                    'F': [110, 130],
                    'G': [170, 200],
                    '243/340': [240, 350]}

    # split band input:
    band_split = band.split('+')

    # cycle through all bands:
    frq_idx = list()
    for k, baba in enumerate(band_split):
        # find the right indices for the appropriate frequencies:
        frq_idx_temp = np.where((freq >= band_lims[baba][0]) & (freq <= band_lims[baba][1]))[0]
        for fit in frq_idx_temp: frq_idx.append(fit)

    # sort the list and select TBs:
    frq_idx = sorted(frq_idx)
    TB = TB[..., frq_idx]
    freq = freq[frq_idx]

    if return_idx == 0:
        return TB, freq

    elif return_idx == 1:
        return TB, freq, frq_idx

    elif return_idx == 2:
        return frq_idx

    else:
        raise ValueError("'return_idx' in function 'select_MWR_channels' must be an integer. Valid options: 0, 1, 2")


def filter_time(
    time_have,
    time_wanted,
    window=0,
    around=False):

    """
    This function returns a mask (True, False) when the first argument (time_have) is in
    the range time_wanted:time_wanted+window (in seconds) (for around=False) or in the
    range time_wanted-window:time_wanted+window.

    It is important that time_have and time_wanted have the same units (e.g., seconds 
    since 1970-01-01 00:00:00 UTC). t_mask will be True when time_have and time_wanted
    overlap according to 'window' and 'around'. The overlap always includes the boundaries
    (e.g., time_have >= time_wanted & time_have <= time_wanted + window).

    Parameters:
    -----------
    time_have : 1D array of float or int
        Time array that should be masked so that you will know, when time_have overlaps
        with time_wanted.
    time_wanted : 1D array of float or int
        Time array around which 
    window : int or float
        Window in seconds around time_wanted (or merely from time_wanted until time_wanted
        + window) that will be set True in the returned mask. If window = 0, the closest
        match will be used.
    around : bool
        If True, time_wanted - window : time_wanted + window is considered. If False,
        time_wanted : time_wanted + window is considered.
    """

    if not isinstance(around, bool):
        return TypeError("Argument 'around' must be boolean.")

    # Initialise mask with False. Overlap with time_wanted will then be set True.
    have_shape = time_have.shape
    t_mask = np.full(have_shape, False)

    if window > 0:
        if around:  # search window is in both directions around time_wanted
            for tw in time_wanted:
                idx = np.where((time_have >= tw - window) & (time_have <= tw + window))[0]
                t_mask[idx] = True

        else:       # search window only in one direction
            for tw in time_wanted:
                idx = np.where((time_have >= tw) & (time_have <= tw + window))[0]
                t_mask[idx] = True

    else:   # window <= 0: use closest match; around = True or False doesn't matter here
        for tw in time_wanted:
            idx = np.argmin(np.abs(time_have - tw)).flatten()
            t_mask[idx] = True

    return t_mask


def find_files_daterange(
    all_files, 
    date_start_dt, 
    date_end_dt,
    idx,
    file_dt_fmt="%Y%m%d"):

    """
    Filter from a given set of files the correct ones within the date range
    date_start_dt - date_end_dt (including start and end date).

    Parameters:
    -----------
    all_files : list of str
        List of str that includes all the files.
    date_start_dt : datetime object
        Start date as a datetime object.
    date_end_dt : datetime object
        End date as a datetime object.
    idx : list of int
        List of int where the first entry specifies the start and the second one
        the end of the date string in any all_files item (i.e., [-17,-9]).
    file_dt_fmt : str
        String indicating the date format in the file names. E.g., "%Y%m%d" for
        "20190725".
    """

    files = list()
    for pot_file in all_files:
        # check if file is within our date range:
        file_dt = dt.datetime.strptime(pot_file[idx[0]:idx[1]], file_dt_fmt)
        if (file_dt >= date_start_dt) & (file_dt <= date_end_dt):
            files.append(pot_file)

    return files


def identify_files_daterange(path: str, daterange: np.ndarray, file_pattern: str, yyyymmdd_delim=""):
    
    """    
    Parameters:
    -----------
    path : str
        Full path where files containing the data are located.
    daterange : np.ndarray
        Array of np.datetime64 indicating the date range.
    file_pattern : str
        String indicating the file pattern of the data.
    yyyymmdd_delim : str
        Delimiter used between year, month and days in the date strings of the data files.
    """
    
    daterange = daterange.astype('datetime64[D]')

    files = list()
    for date in daterange:
        date_str = str(date).replace("-", yyyymmdd_delim)
        file = glob.glob(path + file_pattern.replace("__DATE_STRING__", date_str))
        if len(file) >= 1:
            files.extend(file)
    
    return files


def handle_daterange_or_date_start_end(
    daterange=None, 
    date0=None, 
    date1=None,
    default_daterange=np.array([np.datetime64("2000-01-01")])):
    
    if (daterange is None) and (date0 is None) and (date1 is None):
        daterange = default_daterange
    elif (daterange is None) and ((date0 is not None) and (date1 is not None)):
        daterange = np.arange(np.datetime64(date0), np.datetime64(date1) + np.timedelta64(1, "D"),
                              np.timedelta64(1, "D"))
    elif (daterange is None) and (date0 is not None) and (date1 is None):
        daterange = np.array([np.datetime64(date0)])
        
    return daterange


def find_files_in_daily_dirs(
    path: str, 
    date_range, 
    name: str):

    """
    Search for files within a certain date range (date_range), hidden in subdirectories of path.
    The name of the files to be retrieved must also be specified.
    
    Parameters:
    -----------
    path : str
        Base path of the files to be searched for. Contains subdirectories indicating dates over
        which the search is conducted.
    date_range : np array of datetime64
        Specifies the date range of the files
    name : str
        Name of the files to be searched for within the date-indicating subdirectories.
    """
    
    all_files = list()
    for date in date_range:
        date_str = str(date)
        yyyy, mm, dd = date_str[:4], date_str[5:7], date_str[8:10]
        
        path_date = path + f"{yyyy}/{mm}/{dd}/"
        files = sorted(glob.glob(path_date + name))
        if len(files) >= 1: all_files += files
            
    return all_files


def vector_intersection_2d(
    A1,
    A2,
    B1,
    B2):

    """
    Compute the intersection point between two 2D vectors (a: A1->A2 and b: B1->B2).
    a = A1 + nn*(A2 - A1)
    b = B1 + mm*(B2 - B1)
    Points A1 and A2 must not be identical. The same applies for B1 and B2.

    Parameters:
    -----------
    A1 : 2D array of float
        Origin of the first 2D vector a.
    A2 : 2D array of float
        Endpoint of the first 2D vector a.
    B1 : 2D array of float
        Origin of the second 2D vector b.
    B2 : 2D array of float
        Endpoint of the second 2D vector b.
    """

    A1x = A1[0]
    A1y = A1[1]
    A2x = A2[0]
    A2y = A2[1]
    B1x = B1[0]
    B1y = B1[1]
    B2x = B2[0]
    B2y = B2[1]

    if A1x == A2x:
        aa = (B1x - A1x - (B1y - A1y)*(A2x - A1x) / (A2y - A1y))
        bb = ((B2y - B1y)*(A2x - A1x) / (A2y - A1y) - B2x + B1x)
        if bb == 0:
            mm = np.inf
        else:
            mm = aa / bb
        nn = (B1y - A1y + mm*(B2y - B1y)) / (A2y - A1y)

    else:
        aa = (B1y - A1y - (B1x - A1x)*(A2y - A1y) / (A2x - A1x))
        bb = ((B2x - B1x)*(A2y - A1y) / (A2x - A1x) - B2y + B1y)
        if bb == 0:
            mm = np.inf
        else:
            mm = aa / bb
        nn = (B1x - A1x + mm*(B2x - B1x)) / (A2x - A1x)

    return mm, nn


def sigmoid(x):
    
    """
    Compute sigmoid of x.

    Parameters:
    -----------
    x : float or array of floats
        Input vector, array or number.
    """
    return 1 / (1 + np.exp(-x))
