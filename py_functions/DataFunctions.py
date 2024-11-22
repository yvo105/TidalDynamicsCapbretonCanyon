# #%% #*## LOAD DATA FOR TEST PURPOSES ###

# import os
# import pathlib
# import pickle

# # path of save pickles
# cwd = os.getcwd()
# path = os.path.dirname(cwd) + '/pickles/'

# # Mooring 1
# savepath = pathlib.Path(path, "m1data.pickle")
# with open(savepath, "rb") as f:
#     m1data = pickle.load(f)

#%% #*### FUNCTIONS ###

def angle_between(v1, v2):
    """returns the angle in radians between vectors 'v1' and 'v2'

    Args:
        v1 (array): vector 1
        v2 (array): vector 2
    """
    import numpy as np

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#! TO BE FINISHED
def bin_data(data_2be_binned, data_use4_binning):
    
    import numpy as np
    import pandas as pd
    

    # result array
    depthindex = np.arange(0, 4500, 1)
    spm_df = pd.DataFrame(index=depthindex)
    ### BINNING CAST 1 ### 

    # binned depth indices
    binned_depth_nan = np.nan_to_num(depth_1, nan=-999)
    binned_depth = np.digitize(binned_depth_nan, depthindex)

    # empty list for binned  N2 values
    binned_spm = []

    # binindex listy
    binindex_list = np.unique(binned_depth)

    # drop first array element
    binindex_list = binindex_list[1:]
    for binindex in binindex_list:
        # replace inf with nan
        # spm_1 = spm_1.replace([np.inf, -np.inf], np.nan)
        
        # get corresponding turbidity
        spm = spm_1[binned_depth == binindex]
        
        # append mean turbidity values
        binned_spm.append(np.nanmean(spm))


    # get corresponding depth values
    depth_values = depthindex[binindex_list - 1]


    # add N2 values to dataframe
    binned_spm = pd.Series(binned_spm, index=depth_values)

    # Fill NaN values with linearly extrapolated values
    binned_spm = binned_spm.interpolate(method='linear', limit_direction='both')

    # add binned_N2 to dataframe with column name corresponding to yoy
    spm_df['cast_1'] = binned_spm

    ### BINNING CAST 2 ### 

    # binned depth indices
    binned_depth_nan = np.nan_to_num(depth_2, nan=-999)
    binned_depth = np.digitize(binned_depth_nan, depthindex)

    # empty list for binned  N2 values
    binned_spm = []

    # binindex listy
    binindex_list = np.unique(binned_depth)

    # drop first array element
    binindex_list = binindex_list[1:]
    for binindex in binindex_list:
        # replace inf with nan
        # spm_1 = spm_1.replace([np.inf, -np.inf], np.nan)
        
        # get corresponding turbidity
        spm= spm_2[binned_depth == binindex]
        
        # append mean turbidity values
        binned_spm.append(np.nanmean(spm))


    # get corresponding depth values
    depth_values = depthindex[binindex_list - 1]


    # add N2 values to dataframe
    binned_spm = pd.Series(binned_spm, index=depth_values)

    # Fill NaN values with linearly extrapolated values
    binned_spm = binned_spm.interpolate(method='linear', limit_direction='both')

    # add binned_N2 to dataframe with column name corresponding to yoy
    spm_df['cast_2'] = binned_spm

    # calculate mean of both 
    spm_df['mean'] = spm_df[['cast_1', 'cast_2']].mean(axis=1)

#*DONE
def calculate_tilt(Pitch, Roll, InputInDegrees, OutputInDegrees):
    """
    Calculate the tilt angle in degrees based on the pitch and roll angles.

    Args:
        pitch (float): The pitch angle in degrees.
        roll (float): The roll angle in degrees.
        InputInDegrees(bool): Whether the input is in degrees (True) or radians (False)
        OutputInDegrees (bool): Whether the calculated output should be in degrees (True) or radians (False).

    Returns:
        tilt (float): The tilt angle.
    """
    
    import numpy as np
    
    # Convert degrees to radians if necessary
    if InputInDegrees:
        Pitch = np.radians(Pitch)
        Roll = np.radians(Roll)

    # Calculate tilt
    tilt = np.arccos(np.cos(Pitch) * np.cos(Roll))

    # Convert radians to degrees if necessary
    if OutputInDegrees:
        tilt = np.degrees(tilt)

    return tilt

#*DONE
def opticalbackscatter_to_spm(DataSeries, LinRegressModel):
    """
    Converts optical backscatter data to suspended particulate matter (SPM) concentrations using a linear regression equation.

    Parameters:
    - DataSeries (Series)   : A pandas Series containing the optical backscatter data.
    - LinRegressModel(dict) : A dictionary containing the slope ('m') and intercept ('c') of the linear regression equation.

    Returns:
    - spmdata_df (DataFrame): A pandas DataFrame containing the converted SPM data.

    """
    
    import pandas as pd
    
    # convert optical backscatter data based on linear equation parameters of regression
    spmdata = LinRegressModel['m'] * DataSeries + LinRegressModel['c']
    
    # create datetime index
    datetime_index = pd.to_datetime(DataSeries.index)
    
    # create dataframe
    spmdata_df = pd.DataFrame(spmdata, index=datetime_index,)

    return spmdata_df

#*DONE
def acousticbackscatter_to_spm(BackscatterData, RegressionParameterA, RegressionParameterB):
    """
    Convert acoustic backscatter data to Suspended Particulate Matter (SPM) data using a logarithmic regression equation based on acoustic backscatter measurements.
    
    Formula: SPM =  np.exp( (Backscatter - b) / (a) )

    Parameters:
    - BackscatterData (Series)  : pandas Series containing acoustic backscatter data.
    - RegressionParameterA (int): Parameter a of the logarithmic regression equation.
    - RegressionParameterB (int): Parameter b of the logarithmic regression equation.

    Returns:
    - spmdata_df (DataFrame)    : DataFrame containing the converted SPM data.
    """
    
    import pandas as pd
    import numpy as np
    
    # convert acoustic backscatter to spm
    spmdata = np.exp((BackscatterData - RegressionParameterB) / (RegressionParameterA)) 
    
    # create datetime index
    datetime_index = pd.to_datetime(BackscatterData.index)
    
    # create spm dataframe
    spmdata_df = pd.DataFrame(spmdata, index=datetime_index)

    return spmdata_df

def diff_despiking_ti(data, stdthreshold=3, resample='5s'):
    import pandas as pd
    
    diff = np.diff(data, prepend=np.nan)
    diff = np.append(diff, np.nan)

    diff_df = pd.DataFrame(diff, columns=['diff'], index=data.index)
    
    diff_df['mean'] = diff_df['diff'].rolling(5).mean()
    diff_df['std'] = diff_df['diff'].rolling(5).std()
    
    mask = np.abs(diff_df['diff']-diff_df['mean']) < stdthreshold*diff_df['std']

    return np.where(mask, data, np.nan)

def find_path(d, target_key, target_value, path=()):
    """
    Recursively searches for a specific key-value pair in a nested dictionary.

    Args:
        d (dict): The dictionary to search.
        target_key: The key to search for.
        target_value: The value to search for.
        path (tuple, optional): The current path in the dictionary. Defaults to an empty tuple.

    Returns:
        tuple or None: The path to the key-value pair if found, or None if not found.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            result = find_path(value, target_key, target_value, path + (key,))
            if result is not None:
                return result
        elif key == target_key and value == target_value:
            return path + (key,)
    return None

def get_fft_values_padded(signal, sampling_rate):
    """
    Calculate the Fast Fourier Transform (FFT) of a signal after it is padded and windowed.

    Args:
        signal (array-like): The input signal.
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        tuple: A tuple containing the FFT values, frequency values, normalized magnitude spectrum, and padded signal.
    """
    from scipy.fft import rfft, rfftfreq
    import numpy as np

    signal_av = signal - np.mean(signal)  # demean to get rid of zero frequency
    N = len(signal)  # number of samples

    # Choose a window function
    window = np.hanning(N)

    # Apply the window function
    signal_window = signal_av * window

    # zero padding signal to next power of two
    N_padded = 2 ** np.ceil(np.log2(N)).astype(int)
    print(f"Length of signal: {N}, Length of padded signal: {N_padded}")
    print(f"difference: {N_padded - N}")
    # Now, create a new signal that is zero-padded to N_padded length
    signal_padded = np.zeros(N_padded)
    # pad with half the window size on each side
    start_index = (N_padded - N) // 2
    end_index = start_index + N
    signal_padded[start_index : end_index] = signal_window
    # signal_padded[:N] = signal_window

    # Fourier transform
    fft_values = rfft(signal_padded)
    freq_values = rfftfreq(N_padded, d=1 / sampling_rate)
    rfft_magn = 2 * np.abs(fft_values) / N  # computes the normalized magnitude spectrum

    return fft_values, freq_values, rfft_magn, signal_padded

def get_fft_values_raw(signal, sampling_rate):
    """
    Calculate the Fast Fourier Transform (FFT) of a raw signal.

    Args:
        signal (array-like): The input signal.
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        tuple: A tuple containing the FFT values, frequency values, normalized magnitude spectrum, and raw signal.
    """
    from scipy.fft import rfft, rfftfreq
    import numpy as np

    signal_av = signal - np.mean(signal)  # demean to get rid of zero frequency
    N = len(signal)  # number of samples

    # Fourier transform
    fft_values = rfft(signal_av)
    freq_values = rfftfreq(N, d=1 / sampling_rate)
    rfft_magn = 2 * np.abs(fft_values) / N  # computes the normalized magnitude spectrum

    return fft_values, freq_values, rfft_magn, signal_av

#*DONE
def get_tides(dataseries):
    """
    Finds high and low tide time indices based on find_peaks function; here applied on pressure data. 

    Parameters:
    DataSeries (Series) : pandas series containing the data from which tides should be extracted.

    Returns:
    (MaximaSeries, MinimaSeries)    : tuple containing two arrays - the indices of the local maxima and minima in the dataseries.
    """

    from scipy.signal import find_peaks
    
    # smooth signal via rolling mean
    DataseriesSmoothed = dataseries.rolling(window=5).mean()
    
    # find maxima series
    MaximaSeries = find_peaks(DataseriesSmoothed.values, distance=300)
    
    # find minima series
    MinimaSeries = find_peaks(-1*DataseriesSmoothed.values, distance=300)
      
    return MaximaSeries, MinimaSeries

def get_tod_tor(data, buffer=2):
    """
    Retrieve the time of deployment (ToD) and time of recovery (ToR) with a buffer window.

    Parameters:
    - data (dict): A dictionary containing the data.
    - buffer (int): The buffer in hours to add to ToD and subtract from ToR.

    Returns:
    - tod (datetime): The time of deployment with buffer added.
    - tor (datetime): The time of recovery with buffer subtracted.
    """
    
    from datetime import timedelta

    # get time of deployment and time of recovery
    tod = data['general']['ToD']
    tor = data['general']['ToR']

    # add/subtract buffer
    tod = tod + timedelta(hours=buffer)
    tor = tor - timedelta(hours=buffer)

    return tod, tor

def h_boundary_cacchione(u0, h, r, z0, N, alpha, beta):
    """
    Calculates the shear velocity using the formula given in Cacchione, Pratson et al 2002.

    Parameters:
    u0 (float): Internal wave input velocity [cm/s].
    h (float): thickness of the internal tide bottom boundary layer [m].
    r (float): reflection coefficient [unitless].
    z0 (float): hydraulic roughness parameter [cm].
    N (float): buoyancy frequency.
    alpha (float): angle of internal tide characteristic [radians].
    beta (float): slope of bottom topography [radians].

    Returns:
    float: Shear velocity calculated using the Cacchione formula [cm/s].
    """
    
    M = (((0.16*h*(1-r)) / (np.log(h/(z0/100))) ) *N*np.cos(alpha)*np.sin(2*(alpha+beta)))**(1/3)
    
    u_star = M * (u0/100)**(2/3)
    
    return u_star*100

#* DONE
def load_aquadopp_data(FilePath, SensorID): 
    """Load Aquadop Data

    This function loads Aquadop data from the specified filepath and returns a dictionary containing the data.

    Args:
        FilePath (str): The path to the directory containing the Aquadop data files.
        SensorID (str): The sensor identifier for the Aquadop data files.

    Returns:
        dict: A dictionary containing the Aquadop data with the following keys:
            - 'pressure': A tuple containing the pressure data and its unit.
            - 'temperature': A tuple containing the temperature data and its unit.
            - 'time': The timestamp data.
            - 'amp_beam1_raw': A tuple containing the raw amplitude data for beam 1 and its unit.
            - 'amp_beam2_raw': A tuple containing the raw amplitude data for beam 2 and its unit.
            - 'amp_beam3_raw': A tuple containing the raw amplitude data for beam 3 and its unit.
            - 'vel_east': A tuple containing the east velocity data and its unit.
            - 'vel_north': A tuple containing the north velocity data and its unit.
            - 'vel_up': A tuple containing the upward velocity data and its unit.
            - 'heading': A tuple containing the heading data and its unit.
            - 'pitch': A tuple containing the pitch data and its unit.
            - 'roll': A tuple containing the roll data and its unit.
    """

    import pandas as pd

    # read velocity sensor data
    data_vel_sensor1_raw = pd.read_csv(
        FilePath + f"/{SensorID}.v1", header=None, delim_whitespace=True
    )
    data_vel_sensor2_raw = pd.read_csv(
        FilePath + f"/{SensorID}.v2", header=None, delim_whitespace=True
    )
    data_vel_sensor3_raw = pd.read_csv(
        FilePath + f"/{SensorID}.v3", header=None, delim_whitespace=True
    )

    # read amplitude sensor data
    data_amp_sensor1_raw = pd.read_csv(
        FilePath + f"/{SensorID}.a1", header=None, delim_whitespace=True
    )
    data_amp_sensor2_raw = pd.read_csv(
        FilePath + f"/{SensorID}.a2", header=None, delim_whitespace=True
    )
    data_amp_sensor3_raw = pd.read_csv(
        FilePath + f"/{SensorID}.a3", header=None, delim_whitespace=True
    )

    # read additional sensor data
    sensordata = pd.read_csv(
        FilePath + f"/{SensorID}.sen", header=None, delim_whitespace=True
        )
    
    # combine time columns of DataFrame into DateTimeIndex
    sensordata = timeindex_from_dfcolumns(
        df=sensordata,
        indexlist=[2,0,1,3,4,5,6], ## indices of year, month, day, hour, minute, second and millisecond columns
        )

    # setting index
    data_vel_sensor1_raw.set_index(sensordata.index, inplace=True)
    data_vel_sensor2_raw.set_index(sensordata.index, inplace=True)
    data_vel_sensor3_raw.set_index(sensordata.index, inplace=True)
    data_amp_sensor1_raw.set_index(sensordata.index, inplace=True)
    data_amp_sensor2_raw.set_index(sensordata.index, inplace=True)
    data_amp_sensor3_raw.set_index(sensordata.index, inplace=True)

    # create result dictionary including units
    dic = {
        "pressure": (sensordata[13], "dbar"),
        "temperature": (sensordata[14], "degC"),
        "time": sensordata.index,
        "vel_east_raw": (data_vel_sensor1_raw, "m/s"),
        "vel_north_raw": (data_vel_sensor2_raw, "m/s"),
        "vel_up_raw": (data_vel_sensor3_raw, "m/s"),
        "amp_beam1_raw": (data_amp_sensor1_raw, "dB"),
        "amp_beam2_raw": (data_amp_sensor2_raw, "dB"),
        "amp_beam3_raw": (data_amp_sensor3_raw, "dB"),
        "heading": (sensordata[10], "deg"),
        "pitch": (sensordata[11], "deg"),
        "roll": (sensordata[12], "deg"),
    }

    return dic

#* DONE
def load_obs_data(FilePath, SerialList, LocationList, WindowSize, StdThreshold, UnrealThreshold, LinearRegressModel):
    """Load observation data from CSV files.

    This function loads observation data from CSV files located in the specified FilePath.
    It matches the serial numbers in the filenames with the provided SerialList and assigns
    the corresponding locations from the LocationList. The loaded data is returned as a dictionary with keys representing the variables (turbidity, temperature, and time) and values as tuples containing the data and their respective units.

    Args:
        FilePath (str)              : The path to the directory containing the CSV files.
        SerialList (list)           : A list of serial numbers to match with the filenames.
        LocationList (list)         : A list of corresponding locations for the serial numbers.
        WindowSize (int)            : size of the rolling window for smoothing of the signal.
        StdThreshold (int)          : standard deviation threshold for removal of outliers based on z-score method. 
        UnrealThreshold (int)       : threshold above which values are considered to be "unreal" for the problem at hand.
        LinearRegressModel (dict)   : result dictionary of a linear regression model on turbidity vs spm concentrations. Used to convert the given turbidity values to spm values.

    Returns:
        dict: A dictionary containing the loaded observation data.
            - 'turb_{location}_raw' : A tuple containing the raw turbidity data and its unit.
            - 'turb_{location}'     : A tuple containing the filtered turbidity data and its unit.
            - 'temp_{location}_raw' : A tuple containing the raw temperature data and its unit.
            - 'time_{location}'     : The timestamp data.

    """

    ### IMPORTS
    import glob
    import pandas as pd
    import numpy as np
    import re

    # retrieve file names to be imported
    filenames = glob.glob(FilePath + "\\*.csv")

    # initiate result dictionary
    resultdic = {}
    for file in filenames:
        
        # read data from csv, set datetime index
        df = pd.read_csv(
            file,
            header=[55]
            )
        df = df.set_index(
            pd.to_datetime(df["Meas date"]),
            drop=True
            )
        
        # calculate rolling mean and std according to WindowSize
        df['Turb_rolling_mean'] = df['Turb.-M[FTU]'].rolling(window=WindowSize).mean()
        df['Turb_rolling_std'] = df['Turb.-M[FTU]'].rolling(window=WindowSize).std()
        
        # create new column for filtered turbidity data
        df['Turb.-M[FTU]_filtered'] = df['Turb.-M[FTU]']
        
        # flag outliers based on StdThreshold
        df['is_outlier'] = (df['Turb.-M[FTU]'] > df['Turb_rolling_mean'] + StdThreshold*df['Turb_rolling_std']) | (df['Turb.-M[FTU]'] < df['Turb_rolling_mean'] - StdThreshold*df['Turb_rolling_std'])
        
        # replace outliers in turbidity data based on outlier flags with np.nan
        df.loc[df['is_outlier'], 'Turb.-M[FTU]_filtered'] = np.nan 
        
        # replace outliers in turbidity based on UnrealThreshold
        df.loc[df['Turb.-M[FTU]_filtered'] > UnrealThreshold, 'Turb.-M[FTU]_filtered'] = np.nan
        
        # convert filtered turbidity datato suspended matter concentration (spm)
        df['SPM[mg/L]'] = opticalbackscatter_to_spm(
            DataSeries=df['Turb.-M[FTU]_filtered'],
            LinRegressModel=LinearRegressModel
            )
        
        # search whether file has strucutre of USB_00XX in its name where XX are two numbers
        match = re.search(r"USB_00(\d{2})_", file)
        
        # use numbers XX for identification since it is the SerialID of the instrument
        if match:
            id = match.group(1)  # retrieve serialid
            index = np.where(SerialList == id)  # retrieve index of serialid from seriallist
            loc = LocationList[index][0]  # retrieve location of instrument
            
            # save loaded data into result dictionary using the  location of instrument as identifier
            resultdic[f"turb_{loc}_raw"] = (df["Turb.-M[FTU]"], "FTU")
            resultdic[f"turb_{loc}"] = (df["Turb.-M[FTU]_filtered"], "FTU")
            resultdic[f"temp_{loc}_raw"] = (df["Temp.[degC]"], "degC")
            resultdic[f"time_{loc}"] = df["Temp.[degC]"].index
            resultdic[f'SPM_{loc}'] = df['SPM[mg/L]']
        else:
            print("No corresponding serial number found on this equipment.")
            
    return resultdic

def N2_bin_mean(bathymetry, slope, ctddata, depthinterval, upto4000,):
    import pandas as pd
    
    # result array
    pressureindex = np.arange(0, bathymetry.max()+depthinterval, depthinterval)
    N2_df = pd.DataFrame(index=pressureindex)
    
    # select yoyokeys, only ending in _N2
    yoyokeys = [key for key in ctddata['yoyos'].keys() if key.endswith('_N2')]
    
    # select data
    for yoyo in yoyokeys:
        data = ctddata['yoyos'][yoyo] 

        # binned pressure indices
        binned_pressure = np.digitize(data['N2_pressure'], pressureindex)

        # empty list for binned  N2 values
        binned_N2 = []
        
        # binindex listy
        binindex_list = np.unique(binned_pressure)

        # drop first array element
        binindex_list = binindex_list[1:]
        for binindex in binindex_list:
            # replace inf with nan
            N2 = data['N2'].replace([np.inf, -np.inf], np.nan)
            
            # get corresponding density values
            N2 = N2[binned_pressure == binindex]
            
            # append mean density values
            binned_N2.append(np.nanmean(N2))
            
        # get corresponding pressure values
        pressure_values = pressureindex[binindex_list - 1]
        
        # add N2 values to dataframe
        binned_N2 = pd.Series(binned_N2, index=pressure_values)
        
        # Fill NaN values with linearly extrapolated values
        binned_N2 = binned_N2.interpolate(method='linear', limit_direction='both')
        
        # add binned_N2 to dataframe with column name corresponding to yoy
        N2_df[yoyo] = binned_N2
    
    if upto4000:
        # handle values up to 4000m
        data = ctddata['singlecasts']['sc_4000_N2']

        # binned pressure indices
        binned_pressure = np.digitize(data['N2_pressure'], pressureindex)

        # empty list for binned  N2 values
        binned_N2 = []

        # binindex listy
        binindex_list = np.unique(binned_pressure)

        # drop first array element
        binindex_list = binindex_list[1:]
        for binindex in binindex_list:
            # replace inf with nan
            N2 = data['N2'].replace([np.inf, -np.inf], np.nan)
            
            # get corresponding density values
            N2 = N2[binned_pressure == binindex]
            
            # append mean density values
            binned_N2.append(np.nanmean(N2))
            
        # get corresponding pressure values
        pressure_values = pressureindex[binindex_list - 1]

        # add N2 values to dataframe
        binned_N2 = pd.Series(binned_N2, index=pressure_values)

        # Fill NaN values with linearly extrapolated values
        binned_N2 = binned_N2.interpolate(method='linear', limit_direction='both')

        # add binned_N2 to dataframe with column name corresponding to yoy
        N2_df['sc_4000_N2'] = binned_N2
    
    N2_df['mean'] = N2_df.mean(axis=1)

#*DONE
def orthogonal_linear_regression(DataDictionary, GeneralDataDictionary, LowBinLimit=0, HourTimeDelta=2, PlotBool=False, PrintBool=False):
    """
    Performs orthogonal linear regression on velocity data for multiple bins.

    Parameters:
    - DataDictionary (dict): A dictionary containing velocity data.
    - GeneralDataDictionary (dict): A dictionary containing general information such as the maximum aquadop bin
    - LowBinLimit (int, optional): The lower limit of the bin range. Default is 0.
    - HourTimeDelta (int, optional): Time span in hours before and after recovery time of mooring / BoBo. Default is 2.
    - PlotBool (bool, optional): Whether to plot the regression results per bin. Default is False.
    - PrintBool (bool, optional): Whether to print the direction values. Default is False.

    Returns:
    - resultdic (dict): A dictionary containing the direction values for each bin and the mean direction.

    """
    
    ### IMPORTS
    from datetime import timedelta
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.odr import ODR, Model, RealData
    
    
    ### SETUP
    
    # retrieve time limits 
    limlow = GeneralDataDictionary['ToD'] + timedelta(hours=HourTimeDelta)
    limhigh = GeneralDataDictionary['ToR'] - timedelta(hours=HourTimeDelta)
    
    # retrieve maximum aquadop bin
    highbinlimit = GeneralDataDictionary['max_aquadop_bin']

    # only take full tidal cycles, in this case from first high tide in time window until last high tide in time window
    lim_high1 = DataDictionary['high tide'][0].loc[limlow:limhigh].index[0]
    lim_high2 = DataDictionary['high tide'][0].loc[limlow:limhigh].index[-1]

    # dictionary for results
    resultdic = {f'Bin {bin0}':None for bin0 in range(LowBinLimit, highbinlimit+1)}
    
    # define linear function for regression, no intercept term so regression runs through origin.
    def linear_model(b, x):
        return b[0] * x
    
    # initialize model with linear function
    linear = Model(linear_model)

    # iterate over each bin
    for i in range(LowBinLimit, highbinlimit+1):

        # retrieve eastward velocity component (u) and northward velocity component (v), convert them to cm/s
        u = DataDictionary['vel_east'][0].iloc[:,i].loc[lim_high1:lim_high2]*100
        v = DataDictionary['vel_north'][0].iloc[:,i].loc[lim_high1:lim_high2]*100

        # initialize realdata object from u and v values
        realdata = RealData(u, v)

        # set up ODR with model and data, initial guess for beta=0
        odr = ODR(realdata, linear, beta0=[0.])

        # run regression
        out = odr.run()

        # extract estimated parameter values
        slope = out.beta[0]
        
        # save result in dictionary
        resultdic[f'Bin {i}'] =np.rad2deg(np.arctan(slope))

        # plot data if desired
        if PlotBool: 
            
            # create x-data in range of u with 100 datapoints, estimate y-data based on regression results
            x_plot = np.linspace(min(u), max(u), 100)
            y_plot = linear_model(out.beta, x_plot)

            # initialize figure
            fig, ax = plt.subplots(tight_layout=True)
            
            # plot observations
            ax.scatter(u, v, color='blue', s=0.5)
            
            # plot regression line
            ax.plot(x_plot, y_plot, color='red', label=f'v = {round(slope,2)} * u, angle: {round(np.rad2deg(np.arctan(slope)),2)} deg')
            
            # set figure parameters
            ax.set(
                xlabel='eastward velocity [cm/s]',
                ylabel='northward velocity [cm/s]',
                title=f'Orth. Linear Regression for Bin {i}'
                )
            ax.legend()
            
        # print regression results per bin if desired
        if PrintBool:
            print(f'Direction: {np.rad2deg(np.arctan(slope))}')
            
        
    # calculate bin mean, bin std and save in dictionary
    resultdic['BinMean'] = np.nanmean([resultdic[bin_temp] for bin_temp in resultdic.keys()])
    resultdic['BinStd'] = np.std([resultdic[bin_temp] for bin_temp in resultdic.keys()])
    
    # print mean direction if desired
    if PrintBool:
        print(f'Mean Direction: {resultdic['BinMean']}, stdev: {resultdic['BinStd']}')   
            
    return resultdic

#*DONE 
def add_rotated_velocities(DataDictionary, CanyonDirection, HighBinLimit, LowBinLimit=0): 
    """adds rotated the velocity data in the given dictionary to align with given the alongcanyon direction.

    Args:
        - DataDictionary (dict)         : A dictionary containing the velocity data.
        - CanyonDirection (int)         : canyon axis as retrieved from orthogonal linear regression in degrees.
        - HighBinLimit (int)            : the upper limit of the bin range. Can be taken from the general dictionary 
        - LowBinLimit (int, optional)   : The lower limit of the bin range. Default is 0.

    Returns:
        DataDictionary (dict)           : Original data dictionary now containing the rotated velocity data as well.
    """
    
    # IMPORTS
    import numpy as np
    import pandas as pd

    # extract u and v data
    u = DataDictionary['vel_east'][0].iloc[:, LowBinLimit:HighBinLimit] 
    v = DataDictionary['vel_north'][0].iloc[:, LowBinLimit:HighBinLimit] 

    # define rotation matrix
    rot_matrix = np.array([[np.cos(CanyonDirection), np.sin(CanyonDirection)],
                [-np.sin(CanyonDirection), np.cos(CanyonDirection)]])

    # Initialize arrays to hold the rotated data
    u_prime = np.zeros_like(u)
    v_prime = np.zeros_like(v)

    # Apply the rotation for each vertical bin separately
    for bin_index in range(u.shape[1]):

        # Extract the entire time series for the current bin
        u_bin = u.loc[:, bin_index]
        v_bin = v.loc[:, bin_index]

        # Stack them into a single matrix
        uv_matrix = np.vstack((u_bin, v_bin)).T  # Transpose to get pairs of (u,v)

        # Apply the rotation matrix to each pair of (u,v)
        rotated_uv_matrix = uv_matrix @ rot_matrix.T  # Transpose rot. matrix to align the matrix multiplication

        # Store the rotated components back into u_prime and v_prime
        u_prime[:, bin_index] = rotated_uv_matrix[:, 0]
        v_prime[:, bin_index] = rotated_uv_matrix[:, 1]
     
    # add rotated data to dictionary 
    DataDictionary['vel_alongcanyon'] = (pd.DataFrame(u_prime, index=u.index, columns=u.columns), 'm/s')
    DataDictionary['vel_crosscanyon'] = (pd.DataFrame(v_prime, index=v.index, columns=v.columns), 'm/s')
    
    return DataDictionary

def rotate_coordinates_new(datadict, canyondirection, binlimit_high, binlimit_low=0):

    # imports
    import numpy as np
    import pandas as pd

    # extract u and v data
    u = datadict['vel_east'][0]
    v = datadict['vel_north'][0]
    
    # alongcanyon velocity component
    u_along = u*np.cos(np.deg2rad(canyondirection)) + v*np.sin(np.deg2rad(canyondirection))
    
    # acrosscanyon velocity component
    v_across = v*np.cos(np.deg2rad(canyondirection)) - u*np.sin(np.deg2rad(canyondirection))
    
    # add rotated data to dictionary
    datadict['vel_alongcanyon'] = (pd.DataFrame(u_along, index=u.index, columns=u.columns), 'm/s')
    datadict['vel_crosscanyon'] = (pd.DataFrame(v_across, index=v.index, columns=v.columns), 'm/s')
    
    return datadict


def shearvel_cacchione(u0, h, r, z0, N, alpha, beta):
    """
    Calculates the shear velocity using the formula given in Cacchione, Pratson et al 2002.

    Parameters:
    u0 (float): Internal wave input velocity [cm/s].
    h (float): thickness of the internal tide bottom boundary layer [m].
    r (float): reflection coefficient [unitless].
    z0 (float): hydraulic roughness parameter [cm].
    N (float): buoyancy frequency.
    alpha (float): angle of internal tide characteristic [radians].
    beta (float): slope of bottom topography [radians].

    Returns:
    float: Shear velocity calculated using the Cacchione formula [cm/s].
    """
    
    import numpy as np
    
    M = (((0.16*h*(1-r)) / (np.log(h/(z0/100))) ) *N*np.cos(alpha)*np.sin(2*(alpha+beta)))**(1/3)
    
    u_star = M * (u0/100)**(2/3)
    
    return u_star*100

def steepness_parameter(s_slope, omega, latitude, N2):

    import numpy as np
    # calculate inertial frequency from latitude
    f = 2 * 7.2921e-5 * np.sin(np.radians(latitude))

    # calculate wave steepness parameter
    s_wave = ((omega**2 - f**2) / (N2 - omega**2))**(0.5)
    # print(f"Wave steepness parameter: {s_wave}")

    # calculate steepness parameter
    alpha = s_slope / s_wave

    return alpha

#*DONE
def timeindex_from_dfcolumns(df, indexlist):
    """
    Convert a DataFrame with columns representing year, month, day, hour, minute, second, and microsecond
    into a DateTimeIndex.

    Args:
        df (DataFrame): The input DataFrame containing the columns representing the date and time components.
        indexlist (list): List with indices for year, month, day, hour, minute, second, microsecond

    Returns:
        DataFrame: The input DataFrame with the DateTimeIndex set as the index.
    """
    from datetime import datetime
    
    # create list of datetime indices
    timelist = [
    datetime(*(df[idx][i] for idx in indexlist)) for i in range(df[0].size)
    ]   

    # timelist = []
    # for i in range(df[0].size):
    #     timelist.append(
    #         datetime(
    #             df[2][i], df[0][i], df[1][i], df[3][i], df[4][i], df[5][i], df[6][i]
    #         )
    #     )
    
    # set timelist as index of dataframe
    df.index = timelist
    
    return df

def unit_vector(vector):
    """returns the unit vector of the vector

    Args:
        vector (array): vector to be normalized
    """
    import numpy as np

    return vector / np.linalg.norm(vector, keepdims=True)   

def z_score_despiking(data, window_size=50, threshold=2):
    
    import numpy as np
    
    # Calculate the rolling mean and standard deviation
    ema = data.ewm(span=window_size).mean()
    ema_std = data.ewm(span=window_size).std()
    
    # Calculate the local gradient
    local_gradient = data.diff().rolling(window_size).mean()

    # Subtract the local gradient from the data
    detrended_data = data - local_gradient

    # Calculate the Z-Scores
    z_scores = (detrended_data - ema) / ema_std

    # Create a mask for values that are within the threshold
    mask = np.abs(z_scores) < threshold

    # Return the data where the mask is True
    return data[mask]

