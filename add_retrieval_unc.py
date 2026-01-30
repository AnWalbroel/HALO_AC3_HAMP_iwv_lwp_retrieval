import sys
import os
import pdb

import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt

drive_dir = "/mnt/d/"
test_ids = {'iwv': "060",
            'lwp': "031"}
final_seeds = {'iwv': 110,
               'lwp': 773}

def main():
    
    path_data_base = f"{drive_dir}heavy_data/HALO_AC3/lwp_retrieval/"
    path_ret_output = f"{path_data_base}output/l2/"
    path_ret_stats = f"{path_data_base}output/ret_stat/"
    path_output = f"{path_data_base}for_publication/"
    
    path_plots_base = f"{drive_dir}Studium_NIM/work/Plots/HALO_AC3/lwp_retrieval/eval/"
    
    visualise_stats = True
    set_dict = {'save_figures': True}
    
    
    predictands = ['iwv', 'lwp']
    for predictand in predictands:
        
        path_plots = path_plots_base + predictand + "/"    
            
        try:
            STAT_DS = load_retrieval_stats(path_ret_stats, predictand)
        except FileNotFoundError:
            continue
        
        if visualise_stats:
            plot_stats(STAT_DS, predictand=predictand, path_plots=path_plots, **set_dict)
        
        
def load_retrieval_stats(path: str, predictand='iwv'):
    
    file = path + f"HALO-AC3_NN_retrieval_eval_test_id_{test_ids[predictand]}.nc"
    DS = xr.open_dataset(file).load()
    
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
              'lwp_rmse': [0., 30.],    # g m-2
              'lwp_stddev': [0., 30.],  # g m-2
              'lwp_bias': [-20., 20.],  # g m-2
              }
    pred_error_unit = {'iwv': "kg$\,$m$^{-2}$",
                       'lwp': "g$\,$m$^{-2}$"}
    range_tot = {'iwv': "[0,inf)",
                 'lwp': "[0,inf)"}
    x_lims = [-0.999, 19.999]
    
    f1, axs = plt.subplot_mosaic(stat_varnames, figsize=(8,5), sharex=True)
    
    plt.subplots_adjust(top=0.94, right=0.96, left=0.11, bottom=0.11, wspace=0)
    
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
            
            ax.text(0.5, 1.01, stat_range.upper() + f": {range_str}", 
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


if __name__ == '__main__':
    main()