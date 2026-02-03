import sys
import os
import glob
import pdb

import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt

from data_tools import (running_mean_pdtime, update_netCDF_file_history, encode_time,
                        convert_units)

script_name = os.path.basename(__file__)


drive_dir = "/mnt/d/"
test_ids = {'iwv': "060",
            'lwp': "031"}
final_seeds = {'iwv': 110,
               'lwp': 773}
standard_name = {'iwv': "atmosphere_mass_content_of_water_vapor",
                 'lwp': "atmosphere_mass_content_of_cloud_liquid_water"}
long_names = {'iwv': "integrated water vapor or precipitable water",
              'lwp': "liquid water path or total liquid cloud water"}
valid_ranges = {'iwv': np.array([0., 100.]),
                'lwp': np.array([-0.1, 3.0])}
unit_conv_dict = {'iwv': [0., 1.],
                  'lwp': [0., 0.001]}       # g m-2 to kg m-2

def main():
    
    path_data_base = f"{drive_dir}heavy_data/HALO_AC3/lwp_retrieval/"
    path_ret_output = f"{path_data_base}output/l2/"
    path_ret_stats = f"{path_data_base}output/ret_stat/"
    path_output = f"{path_data_base}for_publication/"
    
    path_plots_base = f"{drive_dir}Studium_NIM/work/Plots/HALO_AC3/lwp_retrieval/eval/"
    
    visualise_stats = False
    set_dict = {'save_figures': False}
    
    
    predictands = ['iwv', 'lwp']
    for predictand in predictands:
        
        path_plots = path_plots_base + predictand + "/"    
            
        try:
            STAT_DS = load_retrieval_stats(path_ret_stats, predictand)
        except FileNotFoundError:
            continue
        
        if visualise_stats:
            plot_stats(STAT_DS, predictand=predictand, path_plots=path_plots, **set_dict)
            
        ret_files = sorted(glob.glob(path_ret_output + f"HALO-AC3_HALO_HAMP_radiometer_l2_{predictand}_v00_RF*.nc"))
        for ret_file in ret_files:
            DS = xr.open_dataset(ret_file).load()
            
            DS = post_process_retrieval_files(DS, STAT_DS, predictand)
            export_DS(DS, path_output)



def load_retrieval_stats(path: str, predictand='iwv'):
    
    file = path + f"HALO-AC3_NN_retrieval_eval_test_id_{test_ids[predictand]}.nc"
    DS = xr.open_dataset(file).load()
    
    return DS


def post_process_retrieval_files(DS: xr.Dataset, STAT_DS: xr.Dataset, predictand='iwv'):
    
    """
    Post process the retrieval output files by adding retrieval uncertainties, perform some final
    smoothing and apply final corrections, add quality flags and improve attributes.
    
    Parameters:
    -----------
    DS : xr.Dataset
        Dataset containing the retrieved quantity.
    STAT_DS : xr.Dataset
        Dataset containing the retrieval uncertainties.
    predictand : str
        String indicating the retrieved quantity.
    """
            
    if predictand == 'iwv':
        DS[predictand][:] = running_mean_pdtime(DS[predictand].values, 10, DS.time.values)
    if predictand == 'lwp':
        DS = manual_lwp_offset_correction(DS)
    
    DS[predictand][:] = convert_units(DS[predictand], unit_conv_dict[predictand])
    
    DS = add_retrieval_uncertainties(DS, STAT_DS, predictand)
    DS = add_quality_flags(DS, predictand)
    DS = improve_attrs(DS, predictand=predictand)
    
    return DS


def plot_stats(
    DS: xr.Dataset,
    predictand='iwv',
    path_plots=f"{drive_dir}work/Plots/HALO_AC3/lwp_retrieval/eval/",
    save_figures=False):
    
    stat_ranges = ['bot', 'mid', 'top', 'tot']
    stat_types = ['rmse', 'stddev', 'bias']
    stat_varnames = get_stat_varnames(stat_ranges, stat_types, predictand=predictand)
    
    y_lims = {'iwv_rmse': [0., 0.9],     # kg m-2
              'iwv_stddev': [0., 0.9],   # kg m-2
              'iwv_bias': [-0.4, 0.4],    # kg m-2
              'lwp_rmse': [0., 25.],    # g m-2
              'lwp_stddev': [0., 25.],  # g m-2
              'lwp_bias': [-15., 15.],  # g m-2
              }
    pred_error_unit = {'iwv': "kg$\,$m$^{-2}$",
                       'lwp': "g$\,$m$^{-2}$"}
    range_tot = {'iwv': "[0,inf)",
                 'lwp': "[0,inf)"}
    x_lims = [-0.999, 19.999]
    
    f1, axs = plt.subplot_mosaic(stat_varnames, figsize=(8,5), sharex=True)
    
    plt.subplots_adjust(top=0.90, right=0.96, left=0.11, bottom=0.11, wspace=0)
    
    for key, ax in axs.items():
        
        try:
            pred, stat_type, stat_range = key.split('_')
        except ValueError:
            pred, stat_type = key.split('_')
            stat_range = 'tot'
        
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims['_'.join([pred, stat_type])])
        
        if stat_type == 'bias':
            ax.axhline(y=0, color=(0.5,0.5,0.5), linewidth=0.75)
        
        ax.plot(DS.test_id, DS[key], color='k', marker='.', markersize=4)
        ax.plot(DS.test_id, DS[key], color='k', marker='.', markersize=4)
        ax.plot(DS.test_id.sel(test_id=DS.seed==final_seeds[pred]), 
                DS[key].sel(test_id=DS.seed==final_seeds[pred]), linestyle='none',
                color='r', marker='.', markersize=10, label='Final')
        ax.axhline(y=DS[key].mean('test_id'), color='c', linestyle='dotted', linewidth=2.0,
                label='Mean')
        
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax.label_outer()
        
        if key in stat_varnames[:,-1]:
            ax.text(1.02, 0.5, stat_type.upper(), ha='left', va='center', rotation=90,
                    fontweight='bold',
                    transform=ax.transAxes)
            
        if key in stat_varnames[0,:]:
            range_str = ""
            if "range" in DS[key].attrs:
                range_str = DS[key].range
            elif stat_range == 'tot':
                range_str = (range_tot[pred] + 
                             DS[f'{pred}_{stat_type}_bot'].range.split(')')[1])
            if pred == 'iwv': range_str = range_str.replace('mm', pred_error_unit[pred])
            if pred == 'lwp': range_str = range_str.replace('kg m-2', 'kg$\,$m$^{-2}$')
            
            ax.text(0.5, 1.01, stat_range.upper() + f":\n{range_str}", 
                    ha='center', va='bottom',
                    fontweight='bold',
                    transform=ax.transAxes)
    
    lh, ll = axs[stat_varnames[0,0]].get_legend_handles_labels()
    axs[stat_varnames[0,0]].legend(lh, ll, loc='upper right', frameon=False)
    
    f1.supylabel(f"{pred.upper()} error ({pred_error_unit[pred]})")
    f1.supxlabel("RNG seed index")    
    
    
    if save_figures:
        os.makedirs(path_plots, exist_ok=True)
        
        plotname = f"HALO_AC3_HAMP_{pred}_error_stats"
        plotfile = os.path.join(path_plots, plotname + ".png")
        f1.savefig(plotfile, dpi=150)

        print(f"Saved {plotfile} ....")
    else:
        plt.show()
        pdb.set_trace()
        
    plt.close()
    

def get_stat_varnames(stat_ranges: list, stat_types: list, predictand='iwv'):
    
    stat_varnames = np.full((len(stat_types), len(stat_ranges)), " ",
                            dtype='<U30')
    
    for k, s_type in enumerate(stat_types):
        for l, s_range in enumerate(stat_ranges):
            stat_varnames[k,l] = f'{predictand}_{s_type}_{s_range}'
            if (s_type == 'stddev') and (s_range == 'tot'):
                stat_varnames[k,l] = stat_varnames[k,l].replace(f'_{s_range}', '')
            
    return stat_varnames


def manual_lwp_offset_correction(DS: xr.Dataset):
    
    """
    Add lwp offset for 2022-03-28T09:05:47 - 2022-03-28T10:00:33 of 100 g m-2.
    """
    
    time0 = np.datetime64("2022-03-28T09:05:47")
    time1 = np.datetime64("2022-03-28T10:00:33")
    DS['lwp'][(DS.time >= time0) & (DS.time <= time1)] += 100.
    if np.any((DS.time >= time0) & (DS.time <= time1)): 
        DS['lwp'].attrs['comment'] += (" Added an LWP offset of 100 g m-2 to the time steps between " +
                                       f"{str(time0)} and {str(time1)} because automatic offset correction failed. " +
                                       "This offset has been carefully chosen to match clear sky conditions seen in " +
                                       "the specMACS images.")
    
    return DS


def add_retrieval_uncertainties(DS: xr.Dataset, STAT_DS: xr.Dataset, predictand='iwv'):
        
    unc_rounding = {'iwv': 0.1,
                    'lwp': 0.002}
    unc_units = {'iwv': "kg m-2",
                 'lwp': "kg m-2"}
    fillval = -9999.
    
    DS[f'{predictand}_err'] = xr.DataArray(np.full(DS[predictand].shape, fillval, dtype=np.float32), 
                                            dims=DS[predictand].dims,
                                            attrs={'standard_name': f"{standard_name[predictand]} standard_error",
                                                   'long_name': f"Estimated uncertainty of {predictand} given as root mean squared error",
                                                   'units': f"{unc_units[predictand]}",
                                                   'comment': ("Root mean squared error has been computed using 11114 " +
                                                               "samples of ERA5 data spanning the month of May from 2000 " +
                                                               "to 2020. The uncertainty has been rounded up to the " +
                                                               f"next {str(unc_rounding[predictand])}.")})
    
    err_stat = get_base_error_stats(STAT_DS, unc_rounding, predictand)
    err_stat = refine_bins_err_stat(err_stat)
    
    for bin0, bin1, e_stat in zip(err_stat.range0.values, err_stat.range1.values, err_stat.values):
        DS[f'{predictand}_err'][:] = xr.where((DS[predictand] >= bin0) & (DS[predictand] < bin1), 
                                               e_stat,
                                               DS[f'{predictand}_err'].values)
    DS[f'{predictand}_err'][DS[predictand] < err_stat.range0.min()] = err_stat.isel(var_range=err_stat.range0.argmin()).item()
    DS[f'{predictand}_err'][DS[predictand] > err_stat.range1.max()] = err_stat.isel(var_range=err_stat.range1.argmax()).item()
    
    DS[f'{predictand}_err'].encoding["_FillValue"] = float(fillval)
    
    return DS


def get_base_error_stats(STAT_DS: xr.Dataset, uncertainty_rounding: dict, predictand='iwv'):
    
    base_ranges = ['bot', 'mid', 'top']
    n_ranges = len(base_ranges)
    err_stat = xr.DataArray(np.zeros((n_ranges,), dtype=np.float32), dims=['var_range'],
                            coords={'range0': (['var_range'], np.zeros((n_ranges,), dtype=np.float32)),
                                    'range1': (['var_range'], np.zeros((n_ranges,), dtype=np.float32))})
    for k, st_key in enumerate(base_ranges):
        range_str = STAT_DS[f'{predictand}_rmse_{st_key}'].range.split('[')[1].split(")")[0]
        range_0, range_1 = range_str.split(',')
        range_0, range_1 = float(range_0), float(range_1)
        err_stat['range0'][k] = float(range_0)
        err_stat['range1'][k] = float(range_1)
        
        err_stat[k] = np.ceil(STAT_DS[f'{predictand}_rmse_{st_key}'].mean('test_id').item() 
                              / uncertainty_rounding[predictand]) * uncertainty_rounding[predictand]
        err_stat[k] = convert_units(err_stat[k], unit_conv_dict[predictand])
        
    return err_stat


def refine_bins_err_stat(err_stat: xr.DataArray):
    
    range0_fine = np.linspace(0., err_stat.range0.max(), 100)
    range1_fine = np.concatenate((range0_fine[1:], np.array([err_stat.range1.max()])))
    err_stat_fine = np.interp(range0_fine, err_stat.range0.values, err_stat.values)
    err_stat = xr.DataArray(err_stat_fine, dims=err_stat.dims,
                            coords={'range0': (err_stat.range0.dims, range0_fine),
                                    'range1': (err_stat.range1.dims, range1_fine)})
    
    return err_stat


def add_quality_flags(DS: xr.Dataset, predictand: str):
    
    valid_ranges_str = {'iwv': (f"[{str(int(valid_ranges['iwv'][0]))}, " + 
                                f"{str(int(valid_ranges['iwv'][1]))}] in kg m-2"),
                        'lwp': f"[{valid_ranges['lwp'][0]:.1f}, {valid_ranges['lwp'][1]:.1f}] in kg m-2"}
    
    DS['flag'] = xr.DataArray(np.full((len(DS.time),), 0, dtype=np.short), dims=['time'],
                              attrs={'standard_name': 'quality_flag',
                                     'long_name': f'quality flag of retrieved {predictand}',
                                     'units': "1",
                                     'flag_masks': np.array([1, 2], dtype=np.short),
                                     'flag_meanings': ("visual_inspection retrieved_quantity_threshold"),
                                     'valid_range': np.array([0, 3], dtype=np.short),
                                     'comment': ("A value of 0 (or nan) means that the data has not been flagged. " +
                                                 "Any value > 0 should be used with care or discarded. " +
                                                 "Visual inspection includes the look at time series of the retrieved " +
                                                 "quantity, raw TBs, specMACS images and flight logs. " +
                                                 f"Retrieved quantity valid range: {valid_ranges_str[predictand]}; ")})
    
    sus_times = sus_times_visual_inspection(predictand)
    for sus_time in sus_times:
        DS['flag'][(DS.time >= sus_time[0]) & (DS.time <= sus_time[1])] += 1
        
    DS['flag'][(DS[predictand] > valid_ranges[predictand][1]) | (DS[predictand] < valid_ranges[predictand][0])] += 2
    
    return DS


def sus_times_visual_inspection(predictand='iwv'):
    
    sus_times = np.array([], dtype='datetime64[s]')
    
    if predictand == 'iwv':
        sus_times = np.array([[np.datetime64("2022-03-12T08:40:00"), np.datetime64("2022-03-12T08:44:34")],
                              [np.datetime64("2022-03-12T16:16:40"), np.datetime64("2022-03-12T16:24:00")],
                              [np.datetime64("2022-03-13T08:20:00"), np.datetime64("2022-03-13T08:27:50")],
                              [np.datetime64("2022-03-13T09:34:04"), np.datetime64("2022-03-13T09:34:54")],
                              [np.datetime64("2022-03-13T09:40:39"), np.datetime64("2022-03-13T10:16:08")],
                              [np.datetime64("2022-03-13T10:50:12"), np.datetime64("2022-03-13T10:50:40")],
                              [np.datetime64("2022-03-13T14:57:08"), np.datetime64("2022-03-13T14:57:32")],
                              [np.datetime64("2022-03-13T15:39:11"), np.datetime64("2022-03-13T15:40:28")],
                              [np.datetime64("2022-03-13T16:05:18"), np.datetime64("2022-03-13T16:05:46")],
                              [np.datetime64("2022-03-13T16:29:58"), np.datetime64("2022-03-13T16:40:00")],
                              [np.datetime64("2022-03-14T09:08:00"), np.datetime64("2022-03-14T09:22:05")],
                              [np.datetime64("2022-03-14T09:45:07"), np.datetime64("2022-03-14T09:45:23")],
                              [np.datetime64("2022-03-14T10:44:17"), np.datetime64("2022-03-14T10:45:29")],
                              [np.datetime64("2022-03-14T14:47:17"), np.datetime64("2022-03-14T14:47:29")],
                              [np.datetime64("2022-03-14T16:48:38"), np.datetime64("2022-03-14T16:55:00")],
                              [np.datetime64("2022-03-15T09:20:00"), np.datetime64("2022-03-15T09:30:00")],
                              [np.datetime64("2022-03-15T10:09:32"), np.datetime64("2022-03-15T10:10:24")],
                              [np.datetime64("2022-03-15T10:18:10"), np.datetime64("2022-03-15T10:18:35")],
                              [np.datetime64("2022-03-15T10:18:50"), np.datetime64("2022-03-15T10:19:07")],
                              [np.datetime64("2022-03-15T14:19:40"), np.datetime64("2022-03-15T14:19:47")],
                              [np.datetime64("2022-03-15T16:44:53"), np.datetime64("2022-03-15T16:45:33")],
                              [np.datetime64("2022-03-15T16:46:04"), np.datetime64("2022-03-15T16:46:15")],
                              [np.datetime64("2022-03-15T16:51:56"), np.datetime64("2022-03-15T16:52:50")],
                              [np.datetime64("2022-03-15T17:16:02"), np.datetime64("2022-03-15T17:25:00")],
                              [np.datetime64("2022-03-16T09:15:00"), np.datetime64("2022-03-16T09:18:27")],
                              [np.datetime64("2022-03-16T09:18:57"), np.datetime64("2022-03-16T09:19:00")],
                              [np.datetime64("2022-03-16T09:21:13"), np.datetime64("2022-03-16T09:22:16")],
                              [np.datetime64("2022-03-16T09:51:24"), np.datetime64("2022-03-16T09:52:48")],
                              [np.datetime64("2022-03-16T09:54:13"), np.datetime64("2022-03-16T09:54:55")],
                              [np.datetime64("2022-03-16T09:58:20"), np.datetime64("2022-03-16T09:58:56")],
                              [np.datetime64("2022-03-16T10:21:15"), np.datetime64("2022-03-16T10:21:35")],
                              [np.datetime64("2022-03-16T10:24:49"), np.datetime64("2022-03-16T10:25:12")],
                              [np.datetime64("2022-03-16T10:25:30"), np.datetime64("2022-03-16T10:25:52")],
                              [np.datetime64("2022-03-16T10:26:33"), np.datetime64("2022-03-16T10:27:31")],
                              [np.datetime64("2022-03-16T10:33:40"), np.datetime64("2022-03-16T10:34:31")],
                              [np.datetime64("2022-03-16T10:35:23"), np.datetime64("2022-03-16T10:35:49")],
                              [np.datetime64("2022-03-16T10:52:00"), np.datetime64("2022-03-16T10:52:27")],
                              [np.datetime64("2022-03-16T10:54:08"), np.datetime64("2022-03-16T10:54:51")],
                              [np.datetime64("2022-03-16T10:58:16"), np.datetime64("2022-03-16T10:58:31")],
                              [np.datetime64("2022-03-16T10:59:58"), np.datetime64("2022-03-16T11:00:27")],
                              [np.datetime64("2022-03-16T12:23:30"), np.datetime64("2022-03-16T12:24:20")],
                              [np.datetime64("2022-03-16T12:25:49"), np.datetime64("2022-03-16T12:26:40")],
                              [np.datetime64("2022-03-16T12:44:47"), np.datetime64("2022-03-16T12:45:06")],
                              [np.datetime64("2022-03-16T12:49:40"), np.datetime64("2022-03-16T12:49:49")],
                              [np.datetime64("2022-03-16T13:31:08"), np.datetime64("2022-03-16T13:31:20")],
                              [np.datetime64("2022-03-16T13:37:40"), np.datetime64("2022-03-16T13:38:07")],
                              [np.datetime64("2022-03-16T14:09:22"), np.datetime64("2022-03-16T14:09:48")],
                              [np.datetime64("2022-03-16T14:10:40"), np.datetime64("2022-03-16T14:10:57")],
                              [np.datetime64("2022-03-16T14:35:20"), np.datetime64("2022-03-16T14:35:40")],
                              [np.datetime64("2022-03-16T14:38:00"), np.datetime64("2022-03-16T14:38:45")],
                              [np.datetime64("2022-03-16T16:16:55"), np.datetime64("2022-03-16T16:17:14")],
                              [np.datetime64("2022-03-16T16:28:10"), np.datetime64("2022-03-16T16:28:40")],
                              [np.datetime64("2022-03-16T16:30:50"), np.datetime64("2022-03-16T16:31:30")],
                              [np.datetime64("2022-03-16T16:34:40"), np.datetime64("2022-03-16T16:35:10")],
                              [np.datetime64("2022-03-16T16:40:20"), np.datetime64("2022-03-16T16:40:50")],
                              [np.datetime64("2022-03-16T17:01:47"), np.datetime64("2022-03-16T17:02:16")],
                              [np.datetime64("2022-03-16T17:25:46"), np.datetime64("2022-03-16T17:31:02")],
                              [np.datetime64("2022-03-16T17:39:18"), np.datetime64("2022-03-16T17:39:52")],
                              [np.datetime64("2022-03-16T17:40:04"), np.datetime64("2022-03-16T17:40:43")],
                              [np.datetime64("2022-03-16T17:53:59"), np.datetime64("2022-03-16T17:54:58")],
                              [np.datetime64("2022-03-16T17:55:48"), np.datetime64("2022-03-16T18:02:00")],
                              [np.datetime64("2022-03-20T08:15:00"), np.datetime64("2022-03-20T08:27:48")],
                              [np.datetime64("2022-03-20T08:35:33"), np.datetime64("2022-03-20T08:36:08")],
                              [np.datetime64("2022-03-20T10:02:12"), np.datetime64("2022-03-20T10:02:40")],
                              [np.datetime64("2022-03-20T12:55:00"), np.datetime64("2022-03-20T12:55:41")],
                              [np.datetime64("2022-03-20T13:09:31"), np.datetime64("2022-03-20T13:10:30")],
                              [np.datetime64("2022-03-20T13:15:04"), np.datetime64("2022-03-20T13:15:35")],
                              [np.datetime64("2022-03-20T15:40:06"), np.datetime64("2022-03-20T15:41:38")],
                              [np.datetime64("2022-03-20T16:40:45"), np.datetime64("2022-03-20T16:42:00")],
                              [np.datetime64("2022-03-21T09:05:00"), np.datetime64("2022-03-21T09:15:51")],
                              [np.datetime64("2022-03-21T09:21:41"), np.datetime64("2022-03-21T09:21:58")],
                              [np.datetime64("2022-03-21T09:28:37"), np.datetime64("2022-03-21T09:28:57")],
                              [np.datetime64("2022-03-21T09:31:43"), np.datetime64("2022-03-21T09:32:04")],
                              [np.datetime64("2022-03-21T09:32:42"), np.datetime64("2022-03-21T09:33:13")],
                              [np.datetime64("2022-03-21T09:39:02"), np.datetime64("2022-03-21T09:39:43")],
                              [np.datetime64("2022-03-21T09:41:10"), np.datetime64("2022-03-21T09:42:22")],
                              [np.datetime64("2022-03-21T09:43:29"), np.datetime64("2022-03-21T09:44:56")],
                              [np.datetime64("2022-03-21T09:45:40"), np.datetime64("2022-03-21T09:45:58")],
                              [np.datetime64("2022-03-21T09:46:40"), np.datetime64("2022-03-21T09:47:04")],
                              [np.datetime64("2022-03-21T09:51:41"), np.datetime64("2022-03-21T09:52:00")],
                              [np.datetime64("2022-03-21T09:54:07"), np.datetime64("2022-03-21T09:54:24")],
                              [np.datetime64("2022-03-21T11:51:00"), np.datetime64("2022-03-21T11:51:37")],
                              [np.datetime64("2022-03-21T12:00:30"), np.datetime64("2022-03-21T12:04:15")],
                              [np.datetime64("2022-03-21T13:00:23"), np.datetime64("2022-03-21T13:00:47")],
                              [np.datetime64("2022-03-21T14:00:41"), np.datetime64("2022-03-21T14:01:38")],
                              [np.datetime64("2022-03-21T16:05:50"), np.datetime64("2022-03-21T16:15:00")],
                              [np.datetime64("2022-03-28T08:58:00"), np.datetime64("2022-03-28T08:59:17")],
                              [np.datetime64("2022-03-28T08:59:30"), np.datetime64("2022-03-28T09:00:05")],
                              [np.datetime64("2022-03-28T09:01:55"), np.datetime64("2022-03-28T09:02:12")],
                              [np.datetime64("2022-03-28T09:05:58"), np.datetime64("2022-03-28T09:07:04")],
                              [np.datetime64("2022-03-28T09:07:35"), np.datetime64("2022-03-28T09:09:09")],
                              [np.datetime64("2022-03-28T09:19:24"), np.datetime64("2022-03-28T09:19:36")],
                              [np.datetime64("2022-03-28T09:22:15"), np.datetime64("2022-03-28T09:22:47")],
                              [np.datetime64("2022-03-28T09:28:08"), np.datetime64("2022-03-28T09:28:39")],
                              [np.datetime64("2022-03-28T09:36:51"), np.datetime64("2022-03-28T09:37:02")],
                              [np.datetime64("2022-03-28T11:08:00"), np.datetime64("2022-03-28T11:10:12")],
                              [np.datetime64("2022-03-28T11:37:37"), np.datetime64("2022-03-28T11:37:48")],
                              [np.datetime64("2022-03-28T11:38:08"), np.datetime64("2022-03-28T11:38:19")],
                              [np.datetime64("2022-03-28T11:52:15"), np.datetime64("2022-03-28T11:52:33")],
                              [np.datetime64("2022-03-28T11:59:57"), np.datetime64("2022-03-28T12:00:40")],
                              [np.datetime64("2022-03-28T12:26:00"), np.datetime64("2022-03-28T12:28:37")],
                              [np.datetime64("2022-03-28T12:56:56"), np.datetime64("2022-03-28T12:57:13")],
                              [np.datetime64("2022-03-28T13:02:47"), np.datetime64("2022-03-28T13:03:16")],
                              [np.datetime64("2022-03-28T13:03:43"), np.datetime64("2022-03-28T13:04:37")],
                              [np.datetime64("2022-03-28T13:22:00"), np.datetime64("2022-03-28T13:24:11")],
                              [np.datetime64("2022-03-28T13:30:53"), np.datetime64("2022-03-28T13:31:09")],
                              [np.datetime64("2022-03-28T13:32:20"), np.datetime64("2022-03-28T13:32:36")],
                              [np.datetime64("2022-03-28T13:52:54"), np.datetime64("2022-03-28T13:53:30")],
                              [np.datetime64("2022-03-28T14:27:00"), np.datetime64("2022-03-28T14:28:28")],
                              [np.datetime64("2022-03-28T14:30:00"), np.datetime64("2022-03-28T14:30:38")],
                              [np.datetime64("2022-03-28T14:36:28"), np.datetime64("2022-03-28T14:36:44")],
                              [np.datetime64("2022-03-28T14:37:00"), np.datetime64("2022-03-28T14:37:08")],
                              [np.datetime64("2022-03-28T15:00:48"), np.datetime64("2022-03-28T15:01:28")],
                              [np.datetime64("2022-03-28T15:04:49"), np.datetime64("2022-03-28T15:05:07")],
                              [np.datetime64("2022-03-28T15:08:14"), np.datetime64("2022-03-28T15:08:32")],
                              [np.datetime64("2022-03-28T15:11:01"), np.datetime64("2022-03-28T15:11:51")],
                              [np.datetime64("2022-03-28T15:40:40"), np.datetime64("2022-03-28T15:48:00")],
                              [np.datetime64("2022-03-29T08:10:00"), np.datetime64("2022-03-29T08:21:17")],
                              [np.datetime64("2022-03-29T08:26:57"), np.datetime64("2022-03-29T08:27:22")],
                              [np.datetime64("2022-03-29T08:28:25"), np.datetime64("2022-03-29T08:28:40")],
                              [np.datetime64("2022-03-29T08:41:04"), np.datetime64("2022-03-29T08:41:21")],
                              [np.datetime64("2022-03-29T08:59:17"), np.datetime64("2022-03-29T09:00:00")],
                              [np.datetime64("2022-03-29T10:23:03"), np.datetime64("2022-03-29T10:23:18")],
                              [np.datetime64("2022-03-29T10:54:45"), np.datetime64("2022-03-29T10:55:05")],
                              [np.datetime64("2022-03-29T11:00:10"), np.datetime64("2022-03-29T11:00:24")],
                              [np.datetime64("2022-03-29T11:05:01"), np.datetime64("2022-03-29T11:05:21")],
                              [np.datetime64("2022-03-29T11:56:14"), np.datetime64("2022-03-29T11:57:24")],
                              [np.datetime64("2022-03-29T12:01:47"), np.datetime64("2022-03-29T12:02:20")],
                              [np.datetime64("2022-03-29T13:00:00"), np.datetime64("2022-03-29T13:01:30")],
                              [np.datetime64("2022-03-29T13:08:50"), np.datetime64("2022-03-29T13:09:20")],
                              [np.datetime64("2022-03-29T14:41:30"), np.datetime64("2022-03-29T14:42:30")],
                              [np.datetime64("2022-03-29T14:44:50"), np.datetime64("2022-03-29T14:45:11")],
                              [np.datetime64("2022-03-29T14:46:07"), np.datetime64("2022-03-29T14:46:22")],
                              [np.datetime64("2022-03-29T14:47:20"), np.datetime64("2022-03-29T14:54:26")],
                              [np.datetime64("2022-03-29T14:54:43"), np.datetime64("2022-03-29T15:12:25")],
                              [np.datetime64("2022-03-29T15:40:13"), np.datetime64("2022-03-29T15:40:37")],
                              [np.datetime64("2022-03-30T08:13:00"), np.datetime64("2022-03-30T08:23:34")],
                              [np.datetime64("2022-03-30T08:25:13"), np.datetime64("2022-03-30T08:25:41")],
                              [np.datetime64("2022-03-30T08:55:02"), np.datetime64("2022-03-30T08:55:54")],
                              [np.datetime64("2022-03-30T08:56:11"), np.datetime64("2022-03-30T08:56:36")],
                              [np.datetime64("2022-03-30T09:01:28"), np.datetime64("2022-03-30T09:01:53")],
                              [np.datetime64("2022-03-30T09:12:11"), np.datetime64("2022-03-30T09:12:24")],
                              [np.datetime64("2022-03-30T09:15:00"), np.datetime64("2022-03-30T09:17:48")],
                              [np.datetime64("2022-03-30T09:20:21"), np.datetime64("2022-03-30T09:25:44")],
                              [np.datetime64("2022-03-30T09:26:41"), np.datetime64("2022-03-30T09:29:49")],
                              [np.datetime64("2022-03-30T09:30:43"), np.datetime64("2022-03-30T09:42:47")],
                              [np.datetime64("2022-03-30T09:43:02"), np.datetime64("2022-03-30T10:55:45")],
                              [np.datetime64("2022-03-30T13:43:30"), np.datetime64("2022-03-30T13:44:38")],
                              [np.datetime64("2022-03-30T14:32:05"), np.datetime64("2022-03-30T14:32:18")],
                              [np.datetime64("2022-03-30T14:45:09"), np.datetime64("2022-03-30T14:45:30")],
                              [np.datetime64("2022-03-30T15:01:30"), np.datetime64("2022-03-30T15:02:30")],
                              [np.datetime64("2022-03-30T15:19:17"), np.datetime64("2022-03-30T15:19:44")],
                              [np.datetime64("2022-03-30T15:36:11"), np.datetime64("2022-03-30T15:50:00")],
                              [np.datetime64("2022-04-01T07:50:00"), np.datetime64("2022-04-01T07:57:38")],
                              [np.datetime64("2022-04-01T07:59:50"), np.datetime64("2022-04-01T08:00:33")],
                              [np.datetime64("2022-04-01T08:01:01"), np.datetime64("2022-04-01T08:01:39")],
                              [np.datetime64("2022-04-01T08:04:17"), np.datetime64("2022-04-01T08:05:15")],
                              [np.datetime64("2022-04-01T08:05:29"), np.datetime64("2022-04-01T08:05:42")],
                              [np.datetime64("2022-04-01T08:06:39"), np.datetime64("2022-04-01T08:07:10")],
                              [np.datetime64("2022-04-01T08:10:47"), np.datetime64("2022-04-01T08:11:06")],
                              [np.datetime64("2022-04-01T08:16:00"), np.datetime64("2022-04-01T08:16:19")],
                              [np.datetime64("2022-04-01T08:25:13"), np.datetime64("2022-04-01T08:26:35")],
                              [np.datetime64("2022-04-01T08:27:45"), np.datetime64("2022-04-01T08:29:09")],
                              [np.datetime64("2022-04-01T08:31:40"), np.datetime64("2022-04-01T08:33:26")],
                              [np.datetime64("2022-04-01T08:34:06"), np.datetime64("2022-04-01T08:35:19")],
                              [np.datetime64("2022-04-01T08:53:01"), np.datetime64("2022-04-01T08:53:57")],
                              [np.datetime64("2022-04-01T09:06:44"), np.datetime64("2022-04-01T09:11:58")],
                              [np.datetime64("2022-04-01T10:07:31"), np.datetime64("2022-04-01T10:07:52")],
                              [np.datetime64("2022-04-01T10:53:30"), np.datetime64("2022-04-01T10:54:22")],
                              [np.datetime64("2022-04-01T10:55:48"), np.datetime64("2022-04-01T11:01:45")],
                              [np.datetime64("2022-04-01T11:48:00"), np.datetime64("2022-04-01T11:50:19")],
                              [np.datetime64("2022-04-01T12:01:23"), np.datetime64("2022-04-01T12:02:11")],
                              [np.datetime64("2022-04-01T12:51:30"), np.datetime64("2022-04-01T12:52:26")],
                              [np.datetime64("2022-04-01T13:28:12"), np.datetime64("2022-04-01T13:29:24")],
                              [np.datetime64("2022-04-01T13:54:30"), np.datetime64("2022-04-01T13:55:04")],
                              [np.datetime64("2022-04-01T14:01:11"), np.datetime64("2022-04-01T14:01:42")],
                              [np.datetime64("2022-04-01T14:04:42"), np.datetime64("2022-04-01T14:05:11")],
                              [np.datetime64("2022-04-01T14:05:31"), np.datetime64("2022-04-01T14:05:55")],
                              [np.datetime64("2022-04-01T14:10:41"), np.datetime64("2022-04-01T14:11:04")],
                              [np.datetime64("2022-04-01T14:11:31"), np.datetime64("2022-04-01T14:11:49")],
                              [np.datetime64("2022-04-01T14:17:32"), np.datetime64("2022-04-01T14:18:05")],
                              [np.datetime64("2022-04-01T14:19:17"), np.datetime64("2022-04-01T14:19:33")],
                              [np.datetime64("2022-04-01T14:27:25"), np.datetime64("2022-04-01T14:27:46")],
                              [np.datetime64("2022-04-01T14:30:21"), np.datetime64("2022-04-01T14:30:55")],
                              [np.datetime64("2022-04-01T14:33:10"), np.datetime64("2022-04-01T14:33:28")],
                              [np.datetime64("2022-04-01T14:38:00"), np.datetime64("2022-04-01T14:38:27")],
                              [np.datetime64("2022-04-01T14:39:31"), np.datetime64("2022-04-01T14:40:02")],
                              [np.datetime64("2022-04-01T14:49:04"), np.datetime64("2022-04-01T14:49:41")],
                              [np.datetime64("2022-04-01T15:05:31"), np.datetime64("2022-04-01T15:06:25")],
                              [np.datetime64("2022-04-01T15:07:20"), np.datetime64("2022-04-01T15:07:43")],
                              [np.datetime64("2022-04-01T15:07:52"), np.datetime64("2022-04-01T15:08:19")],
                              [np.datetime64("2022-04-01T15:09:51"), np.datetime64("2022-04-01T15:10:48")],
                              [np.datetime64("2022-04-01T15:12:52"), np.datetime64("2022-04-01T15:14:12")],
                              [np.datetime64("2022-04-01T15:16:09"), np.datetime64("2022-04-01T15:20:30")],
                              [np.datetime64("2022-04-04T07:40:00"), np.datetime64("2022-04-04T07:44:02")],
                              [np.datetime64("2022-04-04T07:44:50"), np.datetime64("2022-04-04T07:46:40")],
                              [np.datetime64("2022-04-04T07:57:11"), np.datetime64("2022-04-04T07:57:42")],
                              [np.datetime64("2022-04-04T08:00:21"), np.datetime64("2022-04-04T08:00:55")],
                              [np.datetime64("2022-04-04T08:18:00"), np.datetime64("2022-04-04T08:18:52")],
                              [np.datetime64("2022-04-04T08:21:56"), np.datetime64("2022-04-04T08:23:00")],
                              [np.datetime64("2022-04-04T09:02:56"), np.datetime64("2022-04-04T09:06:57")],
                              [np.datetime64("2022-04-04T09:13:45"), np.datetime64("2022-04-04T09:14:18")],
                              [np.datetime64("2022-04-04T09:14:58"), np.datetime64("2022-04-04T09:16:38")],
                              [np.datetime64("2022-04-04T13:05:16"), np.datetime64("2022-04-04T13:05:41")],
                              [np.datetime64("2022-04-04T13:14:25"), np.datetime64("2022-04-04T13:14:38")],
                              [np.datetime64("2022-04-04T13:38:00"), np.datetime64("2022-04-04T13:40:15")],
                              [np.datetime64("2022-04-04T13:45:52"), np.datetime64("2022-04-04T13:46:17")],
                              [np.datetime64("2022-04-04T14:33:00"), np.datetime64("2022-04-04T14:35:16")],
                              [np.datetime64("2022-04-04T14:37:16"), np.datetime64("2022-04-04T14:37:41")],
                              [np.datetime64("2022-04-04T14:38:03"), np.datetime64("2022-04-04T14:39:16")],
                              [np.datetime64("2022-04-04T14:44:24"), np.datetime64("2022-04-04T14:45:28")],
                              [np.datetime64("2022-04-04T14:48:00"), np.datetime64("2022-04-04T14:48:26")],
                              [np.datetime64("2022-04-04T14:56:07"), np.datetime64("2022-04-04T14:56:49")],
                              [np.datetime64("2022-04-04T15:19:03"), np.datetime64("2022-04-04T15:19:29")],
                              [np.datetime64("2022-04-04T15:21:28"), np.datetime64("2022-04-04T15:22:14")],
                              [np.datetime64("2022-04-04T15:33:25"), np.datetime64("2022-04-04T15:33:54")],
                              [np.datetime64("2022-04-04T15:37:11"), np.datetime64("2022-04-04T15:38:25")],
                              [np.datetime64("2022-04-04T15:44:51"), np.datetime64("2022-04-04T15:50:00")],
                              [np.datetime64("2022-04-07T08:45:00"), np.datetime64("2022-04-07T08:55:12")],
                              [np.datetime64("2022-04-07T09:06:50"), np.datetime64("2022-04-07T09:07:19")],
                              [np.datetime64("2022-04-07T09:09:12"), np.datetime64("2022-04-07T09:10:13")],
                              [np.datetime64("2022-04-07T09:13:46"), np.datetime64("2022-04-07T09:14:44")],
                              [np.datetime64("2022-04-07T09:18:58"), np.datetime64("2022-04-07T09:19:48")],
                              [np.datetime64("2022-04-07T09:28:20"), np.datetime64("2022-04-07T09:29:10")],
                              [np.datetime64("2022-04-07T09:29:34"), np.datetime64("2022-04-07T09:30:30")],
                              [np.datetime64("2022-04-07T09:30:54"), np.datetime64("2022-04-07T09:31:35")],
                              [np.datetime64("2022-04-07T10:40:00"), np.datetime64("2022-04-07T10:44:52")],
                              [np.datetime64("2022-04-07T13:49:47"), np.datetime64("2022-04-07T13:50:28")],
                              [np.datetime64("2022-04-07T15:00:52"), np.datetime64("2022-04-07T15:02:49")],
                              [np.datetime64("2022-04-07T15:03:33"), np.datetime64("2022-04-07T15:04:04")],
                              [np.datetime64("2022-04-07T15:08:41"), np.datetime64("2022-04-07T15:09:20")],
                              [np.datetime64("2022-04-07T15:19:25"), np.datetime64("2022-04-07T15:19:51")],
                              [np.datetime64("2022-04-07T15:23:15"), np.datetime64("2022-04-07T15:23:49")],
                              [np.datetime64("2022-04-07T15:37:27"), np.datetime64("2022-04-07T15:39:30")],
                              [np.datetime64("2022-04-08T04:40:00"), np.datetime64("2022-04-08T04:46:55")],
                              [np.datetime64("2022-04-08T04:55:33"), np.datetime64("2022-04-08T04:56:48")],
                              [np.datetime64("2022-04-08T05:00:54"), np.datetime64("2022-04-08T05:01:37")],
                              [np.datetime64("2022-04-08T05:07:55"), np.datetime64("2022-04-08T05:08:34")],
                              [np.datetime64("2022-04-08T05:17:49"), np.datetime64("2022-04-08T05:18:11")],
                              [np.datetime64("2022-04-08T05:56:05"), np.datetime64("2022-04-08T06:07:02")],
                              [np.datetime64("2022-04-08T07:54:12"), np.datetime64("2022-04-08T07:55:12")],
                              [np.datetime64("2022-04-08T08:33:24"), np.datetime64("2022-04-08T08:34:10")],
                              [np.datetime64("2022-04-08T08:38:35"), np.datetime64("2022-04-08T08:39:12")],
                              [np.datetime64("2022-04-08T10:50:37"), np.datetime64("2022-04-08T10:52:02")],
                              [np.datetime64("2022-04-08T10:53:23"), np.datetime64("2022-04-08T10:53:46")],
                              [np.datetime64("2022-04-08T10:59:38"), np.datetime64("2022-04-08T11:00:13")],
                              [np.datetime64("2022-04-08T11:01:14"), np.datetime64("2022-04-08T11:06:38")],
                              [np.datetime64("2022-04-08T11:09:41"), np.datetime64("2022-04-08T11:16:00")],
                              [np.datetime64("2022-04-10T10:10:00"), np.datetime64("2022-04-10T10:20:20")],
                              [np.datetime64("2022-04-10T11:30:56"), np.datetime64("2022-04-10T11:31:31")],
                              [np.datetime64("2022-04-10T11:35:52"), np.datetime64("2022-04-10T11:36:18")],
                              [np.datetime64("2022-04-10T11:37:19"), np.datetime64("2022-04-10T11:37:46")],
                              [np.datetime64("2022-04-10T11:41:09"), np.datetime64("2022-04-10T11:41:39")],
                              [np.datetime64("2022-04-10T11:41:56"), np.datetime64("2022-04-10T11:42:16")],
                              [np.datetime64("2022-04-10T11:43:03"), np.datetime64("2022-04-10T11:43:38")],
                              [np.datetime64("2022-04-10T11:43:47"), np.datetime64("2022-04-10T11:48:43")],
                              [np.datetime64("2022-04-10T12:01:13"), np.datetime64("2022-04-10T12:01:42")],
                              [np.datetime64("2022-04-10T12:54:25"), np.datetime64("2022-04-10T12:54:53")],
                              [np.datetime64("2022-04-10T12:57:02"), np.datetime64("2022-04-10T12:57:40")],
                              [np.datetime64("2022-04-10T13:04:19"), np.datetime64("2022-04-10T13:05:08")],
                              [np.datetime64("2022-04-10T13:06:00"), np.datetime64("2022-04-10T13:06:45")],
                              [np.datetime64("2022-04-10T13:07:00"), np.datetime64("2022-04-10T13:10:20")],
                              [np.datetime64("2022-04-10T15:02:46"), np.datetime64("2022-04-10T15:04:06")],
                              [np.datetime64("2022-04-10T15:29:02"), np.datetime64("2022-04-10T15:29:37")],
                              [np.datetime64("2022-04-10T15:39:46"), np.datetime64("2022-04-10T15:40:38")],
                              [np.datetime64("2022-04-10T15:41:08"), np.datetime64("2022-04-10T15:41:45")],
                              [np.datetime64("2022-04-10T15:43:43"), np.datetime64("2022-04-10T15:44:10")],
                              [np.datetime64("2022-04-10T15:55:08"), np.datetime64("2022-04-10T16:00:00")],
                              [np.datetime64("2022-04-11T08:10:00"), np.datetime64("2022-04-11T08:21:57")],
                              [np.datetime64("2022-04-11T08:41:24"), np.datetime64("2022-04-11T08:43:09")],
                              [np.datetime64("2022-04-11T09:18:25"), np.datetime64("2022-04-11T09:20:46")],
                              [np.datetime64("2022-04-11T14:23:56"), np.datetime64("2022-04-11T14:25:36")],
                              [np.datetime64("2022-04-11T15:03:30"), np.datetime64("2022-04-11T15:04:20")],
                              [np.datetime64("2022-04-11T15:05:33"), np.datetime64("2022-04-11T15:06:09")],
                              [np.datetime64("2022-04-11T15:09:48"), np.datetime64("2022-04-11T15:10:36")],
                              [np.datetime64("2022-04-11T15:18:58"), np.datetime64("2022-04-11T15:25:00")],
                              [np.datetime64("2022-04-12T07:30:00"), np.datetime64("2022-04-12T07:50:48")],
                              [np.datetime64("2022-04-12T08:00:31"), np.datetime64("2022-04-12T08:01:38")],
                              [np.datetime64("2022-04-12T09:18:52"), np.datetime64("2022-04-12T09:24:36")],
                              [np.datetime64("2022-04-12T13:38:30"), np.datetime64("2022-04-12T13:40:20")]])
        
    elif predictand == 'lwp':
        sus_times = np.array([[np.datetime64("2022-03-12T08:40:00"), np.datetime64("2022-03-12T08:45:00")],
                              [np.datetime64("2022-03-12T16:16:40"), np.datetime64("2022-03-12T16:25:00")],
                              [np.datetime64("2022-03-13T08:20:00"), np.datetime64("2022-03-13T08:28:25")],
                              [np.datetime64("2022-03-13T10:15:40"), np.datetime64("2022-03-13T10:16:10")],
                              [np.datetime64("2022-03-13T11:34:40"), np.datetime64("2022-03-13T11:35:10")],
                              [np.datetime64("2022-03-13T15:02:10"), np.datetime64("2022-03-13T15:02:19")],
                              [np.datetime64("2022-03-13T16:01:21"), np.datetime64("2022-03-13T16:02:00")],
                              [np.datetime64("2022-03-13T16:29:38"), np.datetime64("2022-03-13T16:45:00")],
                              [np.datetime64("2022-03-14T09:05:00"), np.datetime64("2022-03-14T09:22:51")],
                              [np.datetime64("2022-03-14T10:01:37"), np.datetime64("2022-03-14T10:02:22")],
                              [np.datetime64("2022-03-14T10:35:15"), np.datetime64("2022-03-14T10:36:15")],
                              [np.datetime64("2022-03-14T10:43:50"), np.datetime64("2022-03-14T10:44:30")],
                              [np.datetime64("2022-03-14T16:02:21"), np.datetime64("2022-03-14T16:02:32")],
                              [np.datetime64("2022-03-14T16:48:38"), np.datetime64("2022-03-14T17:00:00")],
                              [np.datetime64("2022-03-15T09:20:00"), np.datetime64("2022-03-15T09:30:00")],
                              [np.datetime64("2022-03-15T11:01:47"), np.datetime64("2022-03-15T11:02:04")],
                              [np.datetime64("2022-03-15T14:01:00"), np.datetime64("2022-03-15T14:01:15")],
                              [np.datetime64("2022-03-15T15:01:25"), np.datetime64("2022-03-15T15:01:38")],
                              [np.datetime64("2022-03-15T17:01:09"), np.datetime64("2022-03-15T17:09:22")],
                              [np.datetime64("2022-03-15T17:16:09"), np.datetime64("2022-03-15T17:30:00")],
                              [np.datetime64("2022-03-16T12:00:25"), np.datetime64("2022-03-16T12:00:37")],
                              [np.datetime64("2022-03-16T13:37:40"), np.datetime64("2022-03-16T13:38:04")],
                              [np.datetime64("2022-03-16T14:38:00"), np.datetime64("2022-03-16T14:38:27")],
                              [np.datetime64("2022-03-16T16:16:46"), np.datetime64("2022-03-16T16:43:07")],
                              [np.datetime64("2022-03-16T17:52:22"), np.datetime64("2022-03-16T18:00:00")],
                              [np.datetime64("2022-03-20T08:15:00"), np.datetime64("2022-03-20T08:29:00")],
                              [np.datetime64("2022-03-20T10:02:20"), np.datetime64("2022-03-20T10:02:40")],
                              [np.datetime64("2022-03-20T10:30:00"), np.datetime64("2022-03-20T10:35:00")],
                              [np.datetime64("2022-03-20T12:55:00"), np.datetime64("2022-03-20T12:55:45")],
                              [np.datetime64("2022-03-20T13:02:30"), np.datetime64("2022-03-20T13:03:03")],
                              [np.datetime64("2022-03-20T16:01:43"), np.datetime64("2022-03-20T16:02:14")],
                              [np.datetime64("2022-03-20T16:40:43"), np.datetime64("2022-03-20T16:50:00")],
                              [np.datetime64("2022-03-21T09:05:00"), np.datetime64("2022-03-21T09:15:53")],
                              [np.datetime64("2022-03-21T10:01:41"), np.datetime64("2022-03-21T10:01:56")],
                              [np.datetime64("2022-03-21T11:50:30"), np.datetime64("2022-03-21T11:51:37")],
                              [np.datetime64("2022-03-21T12:03:17"), np.datetime64("2022-03-21T12:04:10")],
                              [np.datetime64("2022-03-21T14:01:27"), np.datetime64("2022-03-21T14:01:40")],
                              [np.datetime64("2022-03-21T16:01:08"), np.datetime64("2022-03-21T16:01:27")],
                              [np.datetime64("2022-03-21T16:06:40"), np.datetime64("2022-03-21T16:15:00")],
                              [np.datetime64("2022-03-28T08:55:00"), np.datetime64("2022-03-28T09:02:47")],
                              [np.datetime64("2022-03-28T09:05:11"), np.datetime64("2022-03-28T09:05:50")],
                              [np.datetime64("2022-03-28T09:53:50"), np.datetime64("2022-03-28T10:00:33")],
                              [np.datetime64("2022-03-28T10:20:00"), np.datetime64("2022-03-28T10:30:00")],
                              [np.datetime64("2022-03-28T11:09:00"), np.datetime64("2022-03-28T11:10:14")],
                              [np.datetime64("2022-03-28T12:25:00"), np.datetime64("2022-03-28T12:28:27")],
                              [np.datetime64("2022-03-28T13:01:30"), np.datetime64("2022-03-28T13:01:46")],
                              [np.datetime64("2022-03-28T13:20:00"), np.datetime64("2022-03-28T13:24:07")],
                              [np.datetime64("2022-03-28T14:27:45"), np.datetime64("2022-03-28T14:28:30")],
                              [np.datetime64("2022-03-28T15:40:38"), np.datetime64("2022-03-28T15:50:00")],
                              [np.datetime64("2022-03-29T08:10:00"), np.datetime64("2022-03-29T08:21:45")],
                              [np.datetime64("2022-03-29T09:29:30"), np.datetime64("2022-03-29T09:31:30")],
                              [np.datetime64("2022-03-29T13:00:20"), np.datetime64("2022-03-29T13:01:20")],
                              [np.datetime64("2022-03-29T13:08:45"), np.datetime64("2022-03-29T13:09:30")],
                              [np.datetime64("2022-03-29T14:41:00"), np.datetime64("2022-03-29T14:45:15")],
                              [np.datetime64("2022-03-29T14:47:09"), np.datetime64("2022-03-29T14:49:15")],
                              [np.datetime64("2022-03-29T14:53:45"), np.datetime64("2022-03-29T14:55:30")],
                              [np.datetime64("2022-03-29T15:10:00"), np.datetime64("2022-03-29T15:12:12")],
                              [np.datetime64("2022-03-29T15:13:40"), np.datetime64("2022-03-29T15:14:10")],
                              [np.datetime64("2022-03-29T15:56:00"), np.datetime64("2022-03-29T16:10:00")],
                              [np.datetime64("2022-03-30T08:15:00"), np.datetime64("2022-03-30T08:23:34")],
                              [np.datetime64("2022-03-30T09:16:10"), np.datetime64("2022-03-30T09:17:45")],
                              [np.datetime64("2022-03-30T09:20:15"), np.datetime64("2022-03-30T09:22:22")],
                              [np.datetime64("2022-03-30T09:25:20"), np.datetime64("2022-03-30T09:25:43")],
                              [np.datetime64("2022-03-30T09:26:37"), np.datetime64("2022-03-30T09:29:55")],
                              [np.datetime64("2022-03-30T09:30:42"), np.datetime64("2022-03-30T09:37:17")],
                              [np.datetime64("2022-03-30T09:37:21"), np.datetime64("2022-03-30T10:55:54")],
                              [np.datetime64("2022-03-30T12:30:30"), np.datetime64("2022-03-30T12:33:12")],
                              [np.datetime64("2022-03-30T13:40:00"), np.datetime64("2022-03-30T13:44:44")],
                              [np.datetime64("2022-03-30T14:01:40"), np.datetime64("2022-03-30T14:02:00")],
                              [np.datetime64("2022-03-30T15:36:09"), np.datetime64("2022-03-30T15:50:00")],
                              [np.datetime64("2022-04-01T07:50:00"), np.datetime64("2022-04-01T07:57:39")],
                              [np.datetime64("2022-04-01T09:01:34"), np.datetime64("2022-04-01T09:02:00")],
                              [np.datetime64("2022-04-01T09:06:42"), np.datetime64("2022-04-01T09:11:57")],
                              [np.datetime64("2022-04-01T09:14:20"), np.datetime64("2022-04-01T09:15:00")],
                              [np.datetime64("2022-04-01T10:07:22"), np.datetime64("2022-04-01T10:07:50")],
                              [np.datetime64("2022-04-01T10:53:30"), np.datetime64("2022-04-01T10:54:05")],
                              [np.datetime64("2022-04-01T10:55:58"), np.datetime64("2022-04-01T11:01:48")],
                              [np.datetime64("2022-04-01T12:01:46"), np.datetime64("2022-04-01T12:02:06")],
                              [np.datetime64("2022-04-01T12:51:30"), np.datetime64("2022-04-01T12:52:09")],
                              [np.datetime64("2022-04-01T13:54:00"), np.datetime64("2022-04-01T13:55:05")],
                              [np.datetime64("2022-04-01T15:05:57"), np.datetime64("2022-04-01T15:06:17")],
                              [np.datetime64("2022-04-01T15:16:10"), np.datetime64("2022-04-01T15:30:00")],
                              [np.datetime64("2022-04-04T07:40:00"), np.datetime64("2022-04-04T07:46:46")],
                              [np.datetime64("2022-04-04T09:01:11"), np.datetime64("2022-04-04T09:01:36")],
                              [np.datetime64("2022-04-04T09:03:01"), np.datetime64("2022-04-04T09:08:48")],
                              [np.datetime64("2022-04-04T09:13:40"), np.datetime64("2022-04-04T09:14:20")],
                              [np.datetime64("2022-04-04T09:15:02"), np.datetime64("2022-04-04T09:16:41")],
                              [np.datetime64("2022-04-04T13:39:30"), np.datetime64("2022-04-04T13:40:14")],
                              [np.datetime64("2022-04-04T14:01:12"), np.datetime64("2022-04-04T14:01:35")],
                              [np.datetime64("2022-04-04T15:44:42"), np.datetime64("2022-04-04T15:55:00")],
                              [np.datetime64("2022-04-07T08:45:00"), np.datetime64("2022-04-07T08:55:10")],
                              [np.datetime64("2022-04-07T10:01:31"), np.datetime64("2022-04-07T10:01:52")],
                              [np.datetime64("2022-04-07T10:22:07"), np.datetime64("2022-04-07T10:45:17")],
                              [np.datetime64("2022-04-07T15:37:22"), np.datetime64("2022-04-07T15:45:00")],
                              [np.datetime64("2022-04-08T04:40:00"), np.datetime64("2022-04-08T04:46:56")],
                              [np.datetime64("2022-04-08T09:01:56"), np.datetime64("2022-04-08T09:02:17")],
                              [np.datetime64("2022-04-08T10:01:16"), np.datetime64("2022-04-08T10:01:35")],
                              [np.datetime64("2022-04-08T11:09:38"), np.datetime64("2022-04-08T11:20:00")],
                              [np.datetime64("2022-04-10T10:10:00"), np.datetime64("2022-04-10T10:18:42")],
                              [np.datetime64("2022-04-10T12:01:24"), np.datetime64("2022-04-10T12:01:39")],
                              [np.datetime64("2022-04-10T13:00:45"), np.datetime64("2022-04-10T13:01:03")],
                              [np.datetime64("2022-04-10T14:01:10"), np.datetime64("2022-04-10T14:01:26")],
                              [np.datetime64("2022-04-10T15:01:32"), np.datetime64("2022-04-10T15:01:50")],
                              [np.datetime64("2022-04-10T15:55:04"), np.datetime64("2022-04-10T16:10:00")],
                              [np.datetime64("2022-04-11T08:05:00"), np.datetime64("2022-04-11T08:22:17")],
                              [np.datetime64("2022-04-11T13:59:00"), np.datetime64("2022-04-11T13:59:41")],
                              [np.datetime64("2022-04-11T14:01:25"), np.datetime64("2022-04-11T14:01:57")],
                              [np.datetime64("2022-04-11T15:01:45"), np.datetime64("2022-04-11T15:02:22")],
                              [np.datetime64("2022-04-11T15:18:55"), np.datetime64("2022-04-11T15:30:00")],
                              [np.datetime64("2022-04-12T07:40:00"), np.datetime64("2022-04-12T07:49:20")],
                              [np.datetime64("2022-04-12T08:01:00"), np.datetime64("2022-04-12T08:01:36")],
                              [np.datetime64("2022-04-12T08:54:46"), np.datetime64("2022-04-12T08:56:30")],
                              [np.datetime64("2022-04-12T09:18:58"), np.datetime64("2022-04-12T09:24:19")],
                              [np.datetime64("2022-04-12T09:28:00"), np.datetime64("2022-04-12T09:30:00")],
                              [np.datetime64("2022-04-12T13:38:40"), np.datetime64("2022-04-12T13:39:15")],
                              [np.datetime64("2022-04-12T14:54:36"), np.datetime64("2022-04-12T15:05:00")]])
    
    return sus_times


def improve_attrs(DS: xr.Dataset, predictand: str):
    
    DS[predictand].attrs['long_name'] = long_names[predictand]
    DS[predictand].attrs['ancillary_variables'] = f'{predictand}_err'
    DS[predictand].attrs['valid_min'] = valid_ranges[predictand][0]
    DS[predictand].attrs['valid_max'] = valid_ranges[predictand][1]
    if predictand == 'iwv':
        DS[predictand].attrs['comment'] += (" A running mean of 10 s has been applied to reduce noise. " +
                                            "Still, on days with low-level convective clouds (mostly between 2022-03-21 " +
                                            "and 2022-04-10), the IWV can be a bit noisy.")
    if predictand == 'lwp':
        DS[predictand].attrs['units'] = "kg m-2"
    
    DS.attrs['Title'] = (f"Retrieved {standard_name[predictand]} ({predictand}) from microwave radiometer " +
                         "measurements of the HAMP package onboard the HALO research aircraft during HALO-(AC)3")

    DS = update_netCDF_file_history(DS, 
                                    script_name=script_name, 
                                    summary_str="added retrieval uncertainties, quality flags and improved attributes",
                                    histroy_attr='History')
    
    DS.attrs['Measurement_site'] = ("HALO-(AC)3, Wendisch et al., 2024: Overview: quasi-Lagrangian observations " +
                                    "of Arctic air mass transformations - introduction and initial results of the " +
                                    "HALO-(AC)3 aircraft campaign, Atmos. Chemp. Phys., 24 (15), 8865-8892, " +
                                    "https://doi.org/10.5194/acp-24-8865-2024")
    DS.attrs['Conventions'] = "CF-1.8"
    DS.attrs['License'] = "CC BY-NC 4.0"
    
    return DS


def export_DS(
    DS: xr.Dataset, 
    path_output=f"{drive_dir}heavy_data/HALO_AC3/lwp_retrieval/for_publication/"):
    
    os.makedirs(path_output, exist_ok=True)
    
    DS = encode_time(DS)
    
    vars_fill_value = ['lat', 'lon', 'alt', 'iwv', 'lwp']
    vars_remove_fill_value = ['time', 'flag']

    for ds_var in DS.variables:
        if ds_var in vars_fill_value:
            DS[ds_var].encoding['_FillValue'] = float(-9999.)
        elif ds_var in vars_remove_fill_value:
            DS[ds_var].encoding['_FillValue'] = None
    
    filename = os.path.basename(DS.encoding['source']).replace('_v00_', '_v01_')

    outfile = path_output + filename
    DS.to_netcdf(outfile, mode='w', format="NETCDF4")
    DS = DS.close()
    print(f"Saved {outfile}....")


if __name__ == '__main__':
    main()