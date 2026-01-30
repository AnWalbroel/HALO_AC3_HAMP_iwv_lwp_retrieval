import numpy as np
import xarray as xr
import pdb


class predictor_class:
    """
        Data used as predictor (independent variable) for inverse modelling problems (e.g., TBs).
        Also additional data (like frequencies, time, flag) will be provided.

        For initialisation, we need:
        DS : xarray dataset
            Dataset containing the predictor data (e.g., TBs) and auxilliary variables like time,
            frequencies, flag, ... Samples (e.g., time axis) must be aligned along dimension 'x'.

        **kwargs:
        return_DS : bool
            If True, the imported xarray dataset will also be set as a class attribute.
        add_TB_noise : bool
            If True, random noise can be added to the brightness temperatures using the built-in function
            add_TB_noise. Usually only used if instrument == 'synthetic'.
        noise_dict : dict
            Dictionary that has the frequencies (with a resolution of 0.01 (.2f)) as keys and the noise 
            strength (in K) as value. Only used in add_TB_noise == True. 
            Example: noise_dict = {'190.71': 3.0}
    """

    def __init__(self, DS, **kwargs):

        # define predictor attributes:
        self.freq = DS.freq                         # in GHz
        self.time = DS.time                         # in datetime64[ns]
        self.TB = DS.tb                             # in K (time x freq) (or (sample x freq))
        self.flag = np.zeros(self.TB.shape[:-1])


        # If desired, random noise can be added:
        if 'add_TB_noise' in kwargs.keys():
            if kwargs['add_TB_noise'] and ('noise_dict' in kwargs.keys()):
                self.add_TB_noise(kwargs['noise_dict'], xarray_compatibility=True, freq_dim_name='freq')

            elif kwargs['add_TB_noise'] and ('noise_dict' not in kwargs.keys()):
                raise KeyError("Class radiometers requires 'noise_dict' if 'add_TB_noise' is True.")


        # also possible to return the xarray dataset
        if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
            self.DS = DS

        # convert from xarray to numpy array:
        self.freq = self.freq.values
        self.time = self.time.values
        self.TB = self.TB.values
        

    def add_TB_noise(self, noise_dict, xarray_compatibility=False, freq_dim_name=""):

        """
        Adds random (un)correlated noise to the brightness temperatures, which must be
        in time x freq shape.

        Parameters:
        -----------
        noise_dict : dict
            Dictionary that has the frequencies (with .2f floating point precision) as keys
            and the noise strength (in K) as value. Example: '190.71': 3.0
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

        if not xarray_compatibility:
            n_time = self.TB.shape[0]

            # Loop through frequencies. Find which frequency is currently addressed and
            # create respective noise:
            for freq_sel in noise_dict.keys():
                frq_idx = np.where(np.isclose(self.freq, float(freq_sel), atol=0.01))[0]
                if len(frq_idx) > 0:
                    frq_idx = frq_idx[0]
                    self.TB[:,frq_idx] = self.TB[:,frq_idx] + np.random.normal(0.0, noise_dict[freq_sel], size=n_time)
        else:

            # Loop through frequencies. Find which frequency is currently addressed and
            # create respective noise:
            for freq_sel in noise_dict.keys():
                frq_idx = np.where(np.isclose(self.freq, float(freq_sel), atol=0.01))[0]
                if len(frq_idx) > 0:
                    frq_idx = frq_idx[0]
                    self.TB[{freq_dim_name: frq_idx}] = (self.TB[{freq_dim_name: frq_idx}] + 
                                                            np.random.normal(0.0, noise_dict[freq_sel],
                                                                size=self.TB[{freq_dim_name: frq_idx}].shape))


class predictand_class:
    """
        Data used as predictand (dependent variable) that is to be predicted in an inverse 
        modelling problem (e.g., retrieval of IWV, LWP, temperature or humidity profile).

        For initialisation, we need:
        DS : xarray dataset
            Dataset containing the predictand data (e.g., atmospheric data from ERA5). Samples 
            (e.g., time axis) must be aligned along dimension 'x'.

        **kwargs:
        processed : bool
            Boolean indicator whether ERA5 data output from LEVANTE had been processed with 
            training_data_new_height.py.
        return_DS : bool
            If True, the imported xarray dataset will also be set as a class attribute.
    """

    def __init__(self, DS, **kwargs):

        processed = False
        if "processed" in kwargs.keys(): processed = kwargs['processed']

        if processed:

            # assign attributes and unify variable naming:
            self.launch_time = DS.time.values.astype('datetime64[s]').astype(np.float64)    # in sec since 1970-01-01 00:00:00 UTC
            self.time = DS.time.values      # in np.datetime64[ns]

            attribute_list = [  'lat',          # in deg N
                                'lon',          # in deg E
                                'sfc_slf',      # in [0,1]
                                'sfc_sif',      # in [0,1]
                                'temp_sfc',     # in K
                                'height',       # in m
                                'temp',         # in K
                                'rh',           # in [0,1]
                                'pres',         # in Pa
                                'q',            # in kg kg-1
                                'iwv',          # in kg m-2
                                'cwp',          # in kg m-2
                                'rwp',          # in kg m-2
                                'iwp',          # in kg m-2
                                'swp',          # in kg m-2
                                'lwp']          # in kg m-2

            for att in attribute_list:
                if att in DS.data_vars:
                    self.__dict__[att] = DS[att].values
                    if att == 'lwp':
                        self.__dict__[att] *= 1000.     # conversion from kg m-2 to g m-2

        else:
            raise RuntimeError("Rather execute training_data_new_height.py before proceeding with the retrieval...")

        # also possible to return the xarray dataset
        if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
            self.DS = DS