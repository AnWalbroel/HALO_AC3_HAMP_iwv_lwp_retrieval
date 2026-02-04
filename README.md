# Integrated water vapour and cloud liquid water path from microwave radiometers onboard the HALO research aircraft

The microwave radiometers of the HALO microwave package (HAMP) onboard the high-altitude and long-range (HALO) research aircraft are sensitive to 
radiation emitted by atmospheric gases (mainly water vapour and oxygen) liquid hydrometeors and the surface. In this code package, integrated water vapour 
(IWV) and cloud liquid water path (LWP) are retrieved for the HALO-(AC)<sup>3</sup> campaign **[1]** using a trained Neural Network over open ocean regions.

This code package can be used to 
- create training, test and evaluation data by forward simulating ERA5 **[2]** data with the Passive and Active Microwave radiative TRAnsfer tool (PAMTRA, **[3]**) to obtain synthetic microwave radiometer observations
- train the Neural Network to retrieve IWV and LWP
- evaluate the Neural Network retrieval
- apply Neural Network retrieval to HALO-(AC)<sup>3</sup> observations
- generate the output files published to PANGAEA **[4]**

The retrieval has been trained for altitudes in 250 m bins covering the range where HALO operated most of the time during HALO-(AC)<sup>3</sup> (8000-13500 m above mean sea level). 

### Purpose of each file
- `add_retrieval_unc.py`: post-post-process retrieval output by adding uncertainties to the file, add quality flags and flag some suspicous data; also used to visualise retrieval performance on evaluation data
- `check_out_training_data.py`: simple script to visualise the iwv and lwp distribution of the training/test/evaluation ERA5 data
- `data_tools.py`: collection of functions (overly filled with functions from other projects)
- `ERA5_pamtra_sim.py`: forward simulation of ERA5 data with PAMTRA to obtain synthetic HAMP observations
- `halo_test_obs_comp.py`: a small fraction of HALO-(AC)<sup>3</sup> data was used for testing. This script was used to visualise it.
- `import_data.py`: collection of importer routines (overly filled with functions from other projects)
- `merge_era5_pamtra_output.py`: PAMTRA simulation output files that were executed for several different synthetic flight altitudes are merged
- `met_tools.py`: collection of meteorological functions and humidity conversions (overly filled with functions from other projects)
- `nn_classes.py`: simple classes called by `NN_retrieval.py`
- `NN_retrieval.py`: main script to train and apply the Neural Network retrieval
- `post_process_retrieval.py`: merge retrieval output of different flight altitudes according to the actual height of HALO at any given time
- `retrieval_stat_overview_plots.py`: visualise retrieval performance of test data
- `run_retrieval_stat_overview_plots.sh`: execute `retrieval_stat_overview_plots.py` for various test cases
- `test_purpose.yaml`: collection of Neural Network settings (teset cases) used for development, testing and final retrieval outputs
- `training_data_new_height.py`: interpolate ERA5 data to new height grid

The final setting for the IWV retrieval was test case "060" and for the LWP retrieval "031" (note that here, also IWV is retrieved but with higher errors than using a dedicated IWV retrieval setup).

Please also note that this code package and its documentation is not optimised for usage by other people. So if you do wish to run parts or all of the scripts, please contact the author (me: a.walbroel__at__uni-koeln.de).
For performing forward simulations or training the retrieval, please contact the author of this code package as the documentation has not advanced beyond self-usability.

The LWP retrieval uses an automatic clear sky offset correction to avoid biases due to water vapour signals (implemented in `NN_retrieval.py`). 
Clear sky periods are detected using radar reflectivity thresholds (at least 5 range bins with `Ze > -40 dbZ` between 300 and 4000 m) 
and 30-s max of std deviation of 30-s rolling mean MWR TBs must be < 0.5 K for all K+V+W+F HAMP channels. 
Clear sky periods must be longer than 30 s to be used as base for LWP clear sky offset correction. Then, a 30-s rolling mean of LWP is
computed. The values the rolling mean LWP takes during those clear sky periods can be used as LWP offset. Linear interpolation in between
clear sky periods is used to compute the LWP offsets between clear sky periods if gap is less than 6 hours.

### Procedure of code execution
`ERA5_pamtra_sim.py` --> `merge_era5_pamtra_output.py` --> `check_out_training_data.py` --> `training_data_new_height.py` --> `NN_retrieval.py` "031" --> `post_process_retrieval.py` "lwp" --> `NN_retrieval.py` "060" --> `post_process_retrieval.py` "iwv" --> `add_retrieval_unc.py`

Note that `NN_retrieval.py` is executed with the variable `exec_type` set to 'op_ret' to retrieve the IWV and LWP and set to '20_runs' to evaluate the retrieval.

---
## Python packages
- python version: 3.10.6 (main, Nov 14 2022, 16:10:14) [GCC 11.3.0]
- tensorflow: 2.10.0
- keras: 2.10.0
- numpy: 1.21.5
- sklearn: 1.2.1
- netCDF4: 1.5.8
- matplotlib: 3.6.3
- xarray: 2023.1.0
- pandas: 1.5.3
- [PAMTRA](https://github.com/igmk/pamtra) **[3]**

---
### References

**[1]**: Wendisch, M., et al., 2024: Overview: quasi-Lagrangian observations of Arctic air mass transformations - introduction and initial results of the HALO-(AC)3 aircraft campaign, Atmos. Chemp. Phys., 24 (15), 8865-8892, https://doi.org/10.5194/acp-24-8865-2024

**[2]**: Hersbach, H., et al. 2020: The ERA5 global reanalysis, QJRMS, 146 (730), 1999-2049, https://doi.org/10.1002/qj.3803

**[3]**: Mech, M., et al. (2020): PAMTRA 1.0: the Passive and Active Microwave radiative TRAnsfer tool for simulating radiometer and radar measurements of the cloudy atmosphere, Geosci. Model Dev., 13 (9), 4229-4251, https://doi.org/10.5194/gmd-13-4229-2020

**[4]**: added when available
