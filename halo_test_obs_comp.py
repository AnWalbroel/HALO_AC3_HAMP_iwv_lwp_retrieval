import sys
import os
import glob
import pdb
import gc
import yaml

wdir = os.getcwd() + "/"
remote = ((("/net/blanc/" in wdir) | ("/work/awalbroe/" in wdir)) and ("/mnt/f/" not in wdir))      # identify if the code is executed on the blanc computer or at home

import numpy as np
import matplotlib as mpl
if not remote: mpl.use("WebAgg")
mpl.rcParams.update({'font.family': "monospace"})

import matplotlib.pyplot as plt
import xarray as xr

from import_data import *
from data_tools import *



def visualize_prediction(
    prediction,
    predictand_id,
    set_dict):

    """
    Visualize the evaluation of the Neural Network prediction against the MOSAiC test obs predictand.
    Depending on the predicted variable (specified by predictand_id), different plots will be created:
    IWV: scatter plot, LWP: scatter plot, temperature profile: standard deviation and bias profile,
    specific humidity profile: standard deviation and bias profile

    Parameters:
    -----------
    PRED_DS : xarray dataset
        Dataset containing the prediction from NN_retrieval.py aux_i['halo_test_obs'] = True. The 
        dataset is based on the output of NN_retrieval.py.save_halo_test_obs.
    predictand_id : str
        String indicating which output variable is forwarded to the function.
    set_dict : dict
        Dictionary containing additional information.
    """

    if predictand_id in ['temp', 'q'] and len(height) == 0:
            raise ValueError("Please specify a height variable to estimate error statistics for profiles.")


    # create output path if not existing:
    plotpath_dir = os.path.dirname(set_dict['path_plots'] + f"{predictand_id}/")
    if not os.path.exists(plotpath_dir):
        os.makedirs(plotpath_dir)


    # add a plot file name addition if the rng seed deviates from the default (773):
    plot_file_add = ""
    if set_dict['rng_no'] != 773:
        plot_file_add = "_" + str(set_dict['rng_no'])

    # visualize:
    fs = 26
    fs_small = fs - 2
    fs_dwarf = fs_small - 2
    fs_micro = fs_dwarf - 2
    msize = 7.0

    c_H = (0.7,0,0)


    if predictand_id == 'iwv':

        # load dropsonde IWV for comparison:
        test_date_filename = set_dict['test_date'].replace("-", "")
        DS_DS = xr.open_dataset(set_dict['path_dropsondes'] + f"HALO-AC3_HALO_dropsondes_IWV_{test_date_filename}.nc")

        # squeeze dimensions:
        prediction = PRED_DS[predictand_id].values.squeeze()


        # visualize:
        f1 = plt.figure(figsize=(11,7))
        a1 = plt.axes()


        ax_lims = np.asarray([-2.5, 20.0])  # in kg m-2
        x_lims = np.array([np.datetime64(f"{set_dict['test_date']}T09:00"), np.datetime64(f"{set_dict['test_date']}T16:30")])

        # plotting:
        a1.plot(x_lims, [0,0], color=(0,0,0))
        a1.plot(PRED_DS.time.values, prediction, color=(0.11,0.46,0.70), linewidth=1.2, label='HAMP')
        a1.plot(DS_DS.launch_time.values, DS_DS.IWV.values, color=(0.70,0.46,0.11), linestyle='none',
                marker='.', linewidth=0.75, markersize=9.0, markeredgecolor=(0,0,0), label='Dropsonde')


        # legend:
        lh, ll = a1.get_legend_handles_labels()
        a1.legend(lh, ll, loc='upper right', fontsize=fs_micro-2)


        # set axis limits:
        a1.set_ylim(bottom=ax_lims[0], top=ax_lims[1])
        a1.set_xlim(left=x_lims[0], right=x_lims[1])

        # set axis ticks, ticklabels and tick parameters:
        a1.minorticks_on()
        a1.tick_params(axis='both', labelsize=fs_micro-4)

        # grid:
        a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

        # labels:
        a1.set_ylabel("Predicted IWV ($\mathrm{kg}\,\mathrm{m}^{-2}$)", fontsize=fs_micro-2)
        a1.set_xlabel("HALO RF08 time", fontsize=fs_micro-2)
        a1.set_title(f"{set_dict['test_no']}", fontsize=fs_micro)

        if set_dict['save_figures']:
            plotname = f"HALO-AC3_HALO_HAMP_NN_ret_{predictand_id}_prediction_{set_dict['test_no']}" + plot_file_add
            f1.savefig(set_dict['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


    if predictand_id == 'lwp':

        # squeeze dimensions and convert to g m-2:
        prediction = PRED_DS[predictand_id].values.squeeze()


        # visualize:
        f1 = plt.figure(figsize=(11,7))
        a1 = plt.axes()


        ax_lims = np.asarray([-100.0, 600]) # in g m-2
        x_lims = np.array([np.datetime64(f"{set_dict['test_date']}T09:00"), np.datetime64(f"{set_dict['test_date']}T16:30")])

        # plotting:
        a1.plot(x_lims, [0,0], color=(0,0,0))
        a1.plot(PRED_DS.time.values, prediction, color=(0.11,0.46,0.70), linewidth=1.2)


        # set axis limits:
        a1.set_ylim(bottom=ax_lims[0], top=ax_lims[1])
        a1.set_xlim(left=x_lims[0], right=x_lims[1])

        # set axis ticks, ticklabels and tick parameters:
        a1.minorticks_on()
        a1.tick_params(axis='both', labelsize=fs_micro-4)

        # grid:
        a1.grid(which='major', axis='both', color=(0.5,0.5,0.5), alpha=0.5)

        # labels:
        a1.set_ylabel("Predicted LWP ($\mathrm{g}\,\mathrm{m}^{-2}$)", fontsize=fs_micro-2)
        a1.set_xlabel("HALO RF08 time", fontsize=fs_micro-2)
        a1.set_title(f"{set_dict['test_no']}", fontsize=fs_micro)

        if set_dict['save_figures']:
            plotname = f"HALO-AC3_HALO_HAMP_NN_ret_{predictand_id}_prediction_{set_dict['test_no']}" + plot_file_add
            f1.savefig(set_dict['path_plots'] + f"{predictand_id}/" + plotname + ".png", dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    # plt.clf()
    gc.collect()


"""
    To avoid ERA5 biases to be trained into the Neural Network retrieval, some tests also include
    MOSAiC observations for validation (as separate test data set). Here, predictions from the
    Neural Network, applied on microwave radiometer measurements of HATPRO and MiRAC-P are compared
    to MOSAiC radiosondes from Polarstern. The prediction must have been created with 
    NN_retrieval.py.save_mosaic_test_obs.
    - import radiosonde and predicted radiometer data
    - merge both data sets
    - visualize comparison
"""


# inquire test id (and rng seed):
test_no = "000"
rng_no = 773
if len(sys.argv) == 2:
    test_no = sys.argv[1]
elif len(sys.argv) == 3:
    test_no = sys.argv[1]
    rng_no = int(sys.argv[2])


# paths:
if remote:
    path_data = {'predicted': "/net/blanc/awalbroe/Data/HALO_AC3/lwp_retrieval/prediction_and_reference/"}
    path_dropsondes = "/net/blanc/awalbroe/Data/HALO_AC3/HALO/dropsondes/IWV/"
    path_plots = "/net/blanc/awalbroe/Plots/HALO_AC3/lwp_retrieval/halo_test_obs/"
else:
    path_data = {'predicted': "/mnt/f/heavy_data/HALO_AC3/lwp_retrieval/prediction_and_reference/"}
    path_dropsondes = "/mnt/f/heavy_data/HALO_AC3/HALO/dropsondes/IWV/"
    path_plots = "/mnt/f/Studium_NIM/work/Plots/HALO_AC3/lwp_retrieval/halo_test_obs/"

# settings:
set_dict = {'save_figures': True,       # whether or not to save figures
            'test_no': test_no,         # test id
            'rng_no': rng_no,           # rng seed
            'path_data': path_data,
            'path_plots': path_plots,
            'path_dropsondes': path_dropsondes,
            }


# open test_purpose.YAML file to check if this test only contained LWP as predictand: if True, use different test dates
with open(wdir + "test_purpose.yaml", 'r') as f:
    cat = yaml.safe_load(f)
set_dict['test_date'] = "2022-03-21"


# identify files:
predicted_file_name = "HALO-AC3_HALO_HAMP_test_obs_NN_ret_prediction"
files = sorted(glob.glob(path_data['predicted'] + predicted_file_name + f"*{set_dict['test_no']}.nc"))


# loop over files:
for id_i, file in enumerate(files):

    # import prediction:
    predictand_id = file[file.find(predicted_file_name)+len(predicted_file_name):file.find(set_dict['test_no'])].replace("_", "")
    PRED_DS = xr.open_dataset(file)
    visualize_prediction(PRED_DS, predictand_id, set_dict)

