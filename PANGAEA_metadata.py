import os
import glob
import pdb

import xarray as xr
import numpy as np

drive_dir = "/mnt/f/"


def main():
    
    path_data_base = f"{drive_dir}heavy_data/HALO_AC3/lwp_retrieval/for_publication/"
    path_output = f"{path_data_base}metadata/"
    
    vars = ['iwv', 'lwp']
    for var in vars:
        files = find_files(path_data_base, var)
        
        metadata = generate_metadata_files(files)
        save_metadata(metadata, path_output, var)
        
        
def find_files(path: str, var: str):
    
    files = sorted(glob.glob(path + f"HALO-AC3_HALO_HAMP_radiometer_l2_{var}_v01_RF*.nc"))
    
    return files


def generate_metadata_files(files: list):
    
    metadata = {key: list() for key in ['event', 'filename', 'start_time', 
                                        'start_lat', 'start_lon', 'end_time',
                                        'end_lat', 'end_lon']}
    for file in files:
        metadata = extract_metadata(metadata, file)
        
    return metadata


def extract_metadata(metadata: dict, file: str):
    
    ds = xr.open_dataset(file)
    RF = str(ds.trajectory.values.item(), 'utf-8')
    
    start_end_lat, start_end_lon = extract_lat_lon(ds)
    
    metadata['event'].append(events[RF])
    metadata['filename'].append(os.path.basename(file))
    metadata['start_time'].append(str(ds.time.values[0].astype('datetime64[s]')))
    metadata['start_lat'].append(f"{start_end_lat[0]:.5f}")
    metadata['start_lon'].append(f"{start_end_lon[0]:.5f}")
    metadata['end_time'].append(str(ds.time.values[-1].astype('datetime64[s]')))
    metadata['end_lat'].append(f"{start_end_lat[-1]:.5f}")
    metadata['end_lon'].append(f"{start_end_lon[-1]:.5f}")
    
    ds = ds.close()
    
    return metadata


def extract_lat_lon(ds: xr.Dataset):
    
    start_end_lat_lon = dict()
    nonnan_idx = {key: np.where(~np.isnan(ds[key]))[0] for key in ['lat', 'lon']}
    for key in ['lat', 'lon']:
        start_end_lat_lon[key] = np.array([ds[key].values[nonnan_idx[key][0]],
                                           ds[key].values[nonnan_idx[key][-1]]])
    
    return start_end_lat_lon['lat'], start_end_lat_lon['lon']


def save_metadata(metadata: dict, path_output: str, var):
    
    os.makedirs(path_output, exist_ok=True)
    
    metadata_txt = list()
    metadata_vars = metadata.keys()
    for idx, event in enumerate(metadata['event']):
        metadata_txt.append([metadata[key][idx] for key in metadata_vars])
    
    metadata_header = [key for key in metadata_vars]
    
    filename = f"metadata_HALO-AC3_HALO_HAMP_radiometer_l2_{var}_v01"
    outfile = path_output + filename + ".txt"
    with open(outfile, "w") as f:
        f.write('\t'.join(metadata_header) + "\n")
        f.writelines('\t'.join(txt_row) + "\n" for txt_row in metadata_txt)
    
    print(f"Saved {outfile}....")


if __name__ == '__main__':
    
    events = {'RF01': "HALO_220311a",
              'RF02': "HALO_220312a",
              'RF03': "HALO_220313a",
              'RF04': "HALO_220314a",
              'RF05': "HALO_220315a",
              'RF06': "HALO_220316a",
              'RF07': "HALO_220320a",
              'RF08': "HALO_220321a",
              'RF09': "HALO_220328a",
              'RF10': "HALO_220329a",
              'RF11': "HALO_220330a",
              'RF12': "HALO_220401a",
              'RF13': "HALO_220404a",
              'RF14': "HALO_220407a",
              'RF15': "HALO_220408a",
              'RF16': "HALO_220410a",
              'RF17': "HALO_220411a",
              'RF18': "HALO_220412a",
              }
    
    main()