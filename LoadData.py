#%% #*## IMPORTS ###

import glob
import os
import pathlib
import pickle
import fnmatch
import re
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from gsw import CT_from_t, Nsquared, SA_from_SP, SP_from_C, density
from scipy.optimize import curve_fit

from py_functions.DataFunctions import (
    calculate_tilt,
    get_tides,
    load_aquadopp_data,
    load_obs_data,
    orthogonal_linear_regression,
    add_rotated_velocities,
    z_score_despiking,
    opticalbackscatter_to_spm,
    acousticbackscatter_to_spm,
)

warnings.filterwarnings("ignore", category=FutureWarning)

### GENERAL PARAMETERS
ParentDirectory = os.path.dirname(os.getcwd())  # parent directory 
PicklePath = os.path.join(ParentDirectory, 'Data', 'pickles') # save location of pickles


#%% #*## LOAD REGRESSION DATA ###

TurbRegressPath = pathlib.Path(PicklePath, "linregress.pickle")
with open(TurbRegressPath, "rb") as f:
    TurbRegressData = pickle.load(f)

# Results from logarithmic regression of ADCP data
AcousticRegressPath = pathlib.Path(PicklePath, "acoustic_logregression.pickle")
with open(AcousticRegressPath, "rb") as f:
    AcousticRegressData = pickle.load(f)

#%% #*## GENERAL INFO ###
'''
This section writes general information about the moorings and the bottom lander such as
time of deployment, time of recovery, etc.
'''

#TODO: lon/lat data is according to shipsdata, values should be updated with USBL position

### Mooring 1 ###
m1_general = {
    'name': 'Mooring 1',
    'depth': 560, # according to MB reading in cruise report p.44 [m]
    'longitude': -1.6567, 
    'latitude': 43.6487,
    'ToD': datetime(2023, 9, 12, 9, 12), # time of deployment acc. cruise report
    'ToR': datetime(2023, 9, 20, 8, 33), # time of recovery acc. cruise report
    'distance_thalweg': 2.685,  # distance to beginning of thalweg profile [km]
    'distance_M1': 0, # distance to 500m station / M1 [km] 
    'max_aquadop_bin': 12, # maximum bin index before aquadop signal reaches noise limit
}

### Mooring 2 ###
m2_general = {
    'name': 'Mooring 2',
    'depth': 1043.3, # according to MB reading in cruise report p.44 [m]
    'longitude': -1.89741,
    'latitude': 43.60747,
    'ToD': datetime(2023, 9, 12, 12, 51), # time of deployment
    'ToR': datetime(2023, 9, 20, 6, 20), # time of recovery 
    'distance_thalweg': 40.065,  # distance to beginning of thalweg profile [km]
    'distance_M1': 40.065 - 2.642, # distance to 500m station / M1 [km]
    'max_aquadop_bin': 11, # maximum bin index before aquadop signal reaches noise limit
}

### Mooring 3 ###
m3_general = {
    'name': 'Mooring 3',
    'depth': 1491.9, # according to MB reading in cruise report p.44 [m]
    'longitude': -2.12802,
    'latitude': 43.63762,
    'ToD': datetime(2023, 9, 12, 15, 54), # time of deployment
    'ToR': datetime(2023, 9, 19, 16, 27), # time of recovery
    'distance_thalweg': 76.836,  # distance to beginning of thalweg profile [km]
    'distance_M1': 76.836 - 2.642, # distance to 500m station / M1 [km]
    'max_aquadop_bin': 8, # maximum bin index before aquadop signal reaches noise limit
}

### BoBo Lander ###
bobo_general = {
    'name': 'BoBo Lander',
    'depth': 2593.6, # according to MB reading in cruise report p.44 [m]
    'longitude': -2.77888,
    'latitude': 43.75733,
    'ToD': datetime(2023, 9, 12, 21, 46), # time of deployment
    'ToR': datetime(2023, 9, 19, 11, 56), # time of recovery
    'distance_thalweg': 175.203,  # distance to beginning of thalweg profile [km]
    'distance_M1': 175.203 - 2.642, # distance to 500m station / M1 [km]
    'max_aquadop_bin': 7, # maximum bin index before aquadop signal reaches noise limit
    
}

#%% #*### LOAD AQUADOPP DATA - MOORINGS & BOBO ####
'''
This sections loads the aquadop data from the raw csv files via the function "load_aquadopp_data". 
Furthermore, a noise floor threshold is applied, tilt is calculated for data QA and high and low tide timestamps are extracted.
'''

#*## MOORING 1 ###

# general setup parameters
fp_m1 = os.path.join(ParentDirectory, 'Data', 'Nortek_Aquadopp', 'Mooring1_data')
id_m1 = 'FLOCM113'  # sensor id as taken from cruise report
noisefloor_m1 = 25 # noise floor for noise removal [counts], based on exponential decay of amplitude data with distance from sensor, see ADCP_qualitycheck.py

# load raw sensor data
m1_aquadop = load_aquadopp_data(
    FilePath=fp_m1,
    SensorID=id_m1
    )

#! Remove after powerpoint is finished
# create single amplitude data (mean of all beams)
# amp_stacked = np.stack( ## stack all beam amplitude data
#         (m1_aquadop['amp_beam1_raw'][0],
#          m1_aquadop['amp_beam2_raw'][0],
#          m1_aquadop['amp_beam3_raw'][0]),
#         axis=0
#         )
# amp_mean = np.mean(amp_stacked, axis=0) 
# amp_df = pd.DataFrame(amp_mean, index=m1_aquadop['amp_beam1_raw'][0].index)
# m1_aquadop['amp_beammean_raw'] = (amp_df, 'counts')

# remove amplitude data based on noise floor (replaced with np.nan)
m1_aquadop['amp_beam1'] = (m1_aquadop['amp_beam1_raw'][0].copy(), 'counts')
m1_aquadop['amp_beam1'][0][m1_aquadop['amp_beam1_raw'][0] < noisefloor_m1] = np.nan
m1_aquadop['amp_beam2'] = (m1_aquadop['amp_beam2_raw'][0].copy(), 'counts')
m1_aquadop['amp_beam2'][0][m1_aquadop['amp_beam2_raw'][0] < noisefloor_m1] = np.nan
m1_aquadop['amp_beam3'] = (m1_aquadop['amp_beam3_raw'][0].copy(), 'counts')
m1_aquadop['amp_beam3'][0][m1_aquadop['amp_beam3_raw'][0] < noisefloor_m1] = np.nan

# remove velocity data based on noise floor (replaced with np.nan)
m1_aquadop['vel_east'] = (m1_aquadop['vel_east_raw'][0].copy(), 'm/s')
m1_aquadop['vel_east'][0][np.isnan(m1_aquadop['amp_beam1'][0])] = np.nan
m1_aquadop['vel_north'] = (m1_aquadop['vel_north_raw'][0].copy(), 'm/s')
m1_aquadop['vel_north'][0][np.isnan(m1_aquadop['amp_beam2'][0])] = np.nan
m1_aquadop['vel_up'] = (m1_aquadop['vel_up_raw'][0].copy(), 'm/s')
m1_aquadop['vel_up'][0][np.isnan(m1_aquadop['amp_beam3'][0])] = np.nan


# additional calculated variables
m1_aquadop['vel_magnitude_2D'] = (np.sqrt(m1_aquadop['vel_east'][0]**2 + m1_aquadop['vel_north'][0]**2), 'm/s')
m1_aquadop['vel_magnitude_3D'] = (np.sqrt(m1_aquadop['vel_east'][0]**2 + m1_aquadop['vel_north'][0]**2 + m1_aquadop["vel_up"][0]**2), 'm/s')
m1_aquadop['vel_direction_2D'] = (np.arctan2(m1_aquadop['vel_east'][0], m1_aquadop['vel_north'][0]), 'radians')

# calculate tilt
m1_aquadop['tilt'] = (calculate_tilt(
    Pitch=m1_aquadop['pitch'][0], 
    Roll=m1_aquadop['roll'][0],
    InputInDegrees=True,
    OutputInDegrees=True), 'deg')

# extract high and low tide times from ADCP data
lim_tod, lim_tor = m1_general['ToD']+timedelta(hours=1), m1_general['ToR']-timedelta(hours=1)
m1_high, m1_low = get_tides(
    dataseries=m1_aquadop['pressure'][0].loc[lim_tod:lim_tor]
    )
m1_aquadop['high tide'] = (m1_aquadop['pressure'][0].loc[lim_tod:lim_tor].iloc[m1_high[0]], 'dbar')
m1_aquadop['low tide'] = (m1_aquadop['pressure'][0].loc[lim_tod:lim_tor].iloc[m1_low[0]], 'dbar')

#*## Mooring 2 ###

# filepath and id
fp_m2 = os.path.join(ParentDirectory, 'Data', 'Nortek_Aquadopp', 'Mooring2_data')
id_m2 = 'FLOCM204' # sensor id as taken from cruise report
noisefloor_m2 = 25 # noise floor for noise removal [counts], based on exponential decay of amplitude data with distance from sensor, see ADCP_qualitycheck.py

# load raw sensor data
m2_aquadop = load_aquadopp_data(
    FilePath=fp_m2,
    SensorID=id_m2
    )

#! Remove after powerpoint is finished
# # create single amplitude data (mean of all beams)
# amp_stacked = np.stack( ## stack all beam amplitude data
#         (m2_aquadop['amp_beam1_raw'][0],
#          m2_aquadop['amp_beam2_raw'][0],
#          m2_aquadop['amp_beam3_raw'][0]),
#         axis=0
#         )
# amp_mean = np.mean(amp_stacked, axis=0)
# amp_df = pd.DataFrame(amp_mean, index=m2_aquadop['amp_beam1_raw'][0].index)
# m2_aquadop['amp_beam_raw'] = (amp_df, 'counts')

# remove amplitude data based on noise floor (replaced with np.nan)
m2_aquadop['amp_beam1'] = (m2_aquadop['amp_beam1_raw'][0].copy(), 'counts')
m2_aquadop['amp_beam1'][0][m2_aquadop['amp_beam1_raw'][0] < noisefloor_m2] = np.nan
m2_aquadop['amp_beam2'] = (m2_aquadop['amp_beam2_raw'][0].copy(), 'counts')
m2_aquadop['amp_beam2'][0][m2_aquadop['amp_beam2_raw'][0] < noisefloor_m2] = np.nan
m2_aquadop['amp_beam3'] = (m2_aquadop['amp_beam3_raw'][0].copy(), 'counts')
m2_aquadop['amp_beam3'][0][m2_aquadop['amp_beam3_raw'][0] < noisefloor_m2] = np.nan

# remove velocity data based on noise floor (replaced with np.nan)
m2_aquadop['vel_east'] = (m2_aquadop['vel_east_raw'][0].copy(), 'm/s')
m2_aquadop['vel_east'][0][np.isnan(m2_aquadop['amp_beam1'][0])] = np.nan
m2_aquadop['vel_north'] = (m2_aquadop['vel_north_raw'][0].copy(), 'm/s')
m2_aquadop['vel_north'][0][np.isnan(m2_aquadop['amp_beam2'][0])] = np.nan
m2_aquadop['vel_up'] = (m2_aquadop['vel_up_raw'][0].copy(), 'm/s')
m2_aquadop['vel_up'][0][np.isnan(m2_aquadop['amp_beam3'][0])] = np.nan

# additional calculated variables
m2_aquadop['vel_magnitude_2D'] = (np.sqrt(m2_aquadop['vel_east'][0]**2 + m2_aquadop['vel_north'][0]**2), 'm/s')
m2_aquadop['vel_magnitude_3D'] = (np.sqrt(m2_aquadop['vel_east'][0]**2 + m2_aquadop['vel_north'][0]**2 + m2_aquadop['vel_up'][0]**2), 'm/s')
m2_aquadop['vel_direction_2D'] =  (np.arctan2(m2_aquadop['vel_north'][0], m2_aquadop['vel_east'][0]), 'radians')

# calculate tilt
m2_aquadop['tilt'] = (calculate_tilt(
    Pitch=m2_aquadop['pitch'][0],
    Roll=m2_aquadop['roll'][0],
    InputInDegrees=True,
    OutputInDegrees=True), 'deg')

# extract high and low tide times from ADCP data
lim_tod, lim_tor = m2_general['ToD'] + timedelta(hours=2), m2_general['ToR'] - timedelta(hours=2)
m2_high, m2_low = get_tides(
    dataseries=m2_aquadop['pressure'][0].loc[lim_tod:lim_tor]
    )
m2_aquadop['high tide'] = (m2_aquadop['pressure'][0].loc[lim_tod:lim_tor].iloc[m2_high[0]], 'dbar')
m2_aquadop['low tide'] = (m2_aquadop['pressure'][0].loc[lim_tod:lim_tor].iloc[m2_low[0]], 'dbar')

#*## MOORING 3 ###

# filepath and id
fp_m3 = os.path.join(ParentDirectory, 'Data', 'Nortek_Aquadopp', 'Mooring3_data')
id_m3 = 'FLOCM314'  # sensor id as taken from cruise report
noisefloor_m3 = 25 # noise floor for noise removal [counts], based on exponential decay of amplitude data with distance from sensor, see ADCP_qualitycheck.py

# load raw sensor data
m3_aquadop = load_aquadopp_data(
    FilePath=fp_m3,
    SensorID=id_m3
    )


#! Remove after powerpoint is finished
# # create single amplitude data (mean of all beams)
# amp_stacked = np.stack( ## stack all beam amplitude data
#         (m3_aquadop['amp_beam1_raw'][0],
#          m3_aquadop['amp_beam2_raw'][0],
#          m3_aquadop['amp_beam3_raw'][0]),
#         axis=0
#         )
# amp_mean = np.mean(amp_stacked, axis=0)
# amp_df = pd.DataFrame(amp_mean, index=m3_aquadop['amp_beam1_raw'][0].index)
# m3_aquadop['amp_beam_raw'] = (amp_df, 'counts')

# remove amplitude data based on noise floor (replaced with np.nan)
m3_aquadop['amp_beam1'] = (m3_aquadop['amp_beam1_raw'][0].copy(), 'counts')
m3_aquadop['amp_beam1'][0][m3_aquadop['amp_beam1_raw'][0] < noisefloor_m3] = np.nan
m3_aquadop['amp_beam2'] = (m3_aquadop['amp_beam2_raw'][0].copy(), 'counts')
m3_aquadop['amp_beam2'][0][m3_aquadop['amp_beam2_raw'][0] < noisefloor_m3] = np.nan
m3_aquadop['amp_beam3'] = (m3_aquadop['amp_beam3_raw'][0].copy(), 'counts')
m3_aquadop['amp_beam3'][0][m3_aquadop['amp_beam3_raw'][0] < noisefloor_m3] = np.nan

# remove velocity data based on noise floor (replaced with np.nan)
m3_aquadop['vel_east'] = (m3_aquadop['vel_east_raw'][0].copy(), 'm/s')
m3_aquadop['vel_east'][0][np.isnan(m3_aquadop['amp_beam1'][0])] = np.nan
m3_aquadop['vel_north'] = (m3_aquadop['vel_north_raw'][0].copy(), 'm/s')
m3_aquadop['vel_north'][0][np.isnan(m3_aquadop['amp_beam2'][0])] = np.nan
m3_aquadop['vel_up'] = (m3_aquadop['vel_up_raw'][0].copy(), 'm/s')
m3_aquadop['vel_up'][0][np.isnan(m3_aquadop['amp_beam3'][0])] = np.nan

# additional calculated variables
m3_aquadop['vel_magnitude_2D'] = (np.sqrt(m3_aquadop['vel_east'][0]**2 + m3_aquadop['vel_north'][0]**2), 'm/s')
m3_aquadop['vel_magnitude_3D'] = (np.sqrt(m3_aquadop['vel_east'][0]**2 + m3_aquadop['vel_north'][0]**2+m3_aquadop['vel_up'][0]**2), 'm/s')
m3_aquadop['vel_direction_2D'] =  (np.arctan2(m3_aquadop['vel_north'][0], m3_aquadop['vel_east'][0]), 'radians')

# calculate tilt
m3_aquadop['tilt'] = (calculate_tilt(
    Pitch=m3_aquadop['pitch'][0],
    Roll=m3_aquadop['roll'][0],
    InputInDegrees=True,
    OutputInDegrees=True), 'deg')

# extract high and low tide times from ADCP data
lim_tod, lim_tor = m3_general['ToD']+timedelta(hours=2), m3_general['ToR']-timedelta(hours=2)
m3_high, m3_low = get_tides(
    dataseries=m3_aquadop['pressure'][0].loc[lim_tod:lim_tor]
    )
m3_aquadop['high tide'] = (m3_aquadop['pressure'][0].loc[lim_tod:lim_tor].iloc[m3_high[0]], 'dbar')
m3_aquadop['low tide'] = (m3_aquadop['pressure'][0].loc[lim_tod:lim_tor].iloc[m3_low[0]], 'dbar')

#*## BOBO LANDER ###

# filepath and id
fp_bobo = os.path.join(ParentDirectory, 'Data', 'Nortek_Aquadopp', 'BoBo_data')
id_bobo = 'FLOCBO10'  # sensor id as taken from cruise report
noisefloor_bobo = 25 # noise floor for noise removal [counts], based on exponential decay of amplitude data with distance from sensor, see ADCP_qualitycheck.py

# load raw sensor data
bobo_aquadop = load_aquadopp_data(
    FilePath=fp_bobo,
    SensorID=id_bobo
    )

#! Remove after powerpoint is finished
# # create single amplitude data (mean of all beams)
# amp_stacked = np.stack( ## stack all beam amplitude data
#         (bobo_aquadop['amp_beam1_raw'][0],
#          bobo_aquadop['amp_beam2_raw'][0],
#          bobo_aquadop['amp_beam3_raw'][0]),
#         axis=0
#         )
# amp_mean = np.mean(amp_stacked, axis=0)
# amp_df = pd.DataFrame(amp_mean, index=bobo_aquadop['amp_beam1_raw'][0].index)
# bobo_aquadop['amp_beam_raw'] = (amp_df, 'counts')

# remove amplitude data based on noise floor (replaced with np.nan)
bobo_aquadop['amp_beam1'] = (bobo_aquadop['amp_beam1_raw'][0].copy(), 'counts')
bobo_aquadop['amp_beam1'][0][bobo_aquadop['amp_beam1_raw'][0] < noisefloor_bobo] = np.nan
bobo_aquadop['amp_beam2'] = (bobo_aquadop['amp_beam2_raw'][0].copy(), 'counts')
bobo_aquadop['amp_beam2'][0][bobo_aquadop['amp_beam2_raw'][0] < noisefloor_bobo] = np.nan
bobo_aquadop['amp_beam3'] = (bobo_aquadop['amp_beam3_raw'][0].copy(), 'counts')
bobo_aquadop['amp_beam3'][0][bobo_aquadop['amp_beam3_raw'][0] < noisefloor_bobo] = np.nan

# remove velocity data based on noise floor (replaced with np.nan)
bobo_aquadop['vel_east'] = (bobo_aquadop['vel_east_raw'][0].copy(), 'm/s')
bobo_aquadop['vel_east'][0][np.isnan(bobo_aquadop['amp_beam1'][0])] = np.nan
bobo_aquadop['vel_north'] = (bobo_aquadop['vel_north_raw'][0].copy(), 'm/s')
bobo_aquadop['vel_north'][0][np.isnan(bobo_aquadop['amp_beam2'][0])] = np.nan
bobo_aquadop['vel_up'] = (bobo_aquadop['vel_up_raw'][0].copy(), 'm/s')
bobo_aquadop['vel_up'][0][np.isnan(bobo_aquadop['amp_beam3'][0])] = np.nan

# additional calculated variables
bobo_aquadop['vel_magnitude_2D'] = (np.sqrt(bobo_aquadop['vel_east'][0]**2 + bobo_aquadop['vel_north'][0]**2), bobo_aquadop['vel_north'][1])
bobo_aquadop['vel_magnitude_3D'] = (np.sqrt(bobo_aquadop['vel_east'][0]**2 + bobo_aquadop['vel_north'][0]**2+bobo_aquadop['vel_up'][0]**2), bobo_aquadop['vel_east'][1])
bobo_aquadop['vel_direction_2D'] =  (np.arctan2(bobo_aquadop['vel_north'][0], bobo_aquadop['vel_east'][0]), bobo_aquadop['vel_east'][1])

# calculate tilt
bobo_aquadop['tilt'] = (calculate_tilt(
    Pitch=bobo_aquadop['pitch'][0],
    Roll=bobo_aquadop['roll'][0],
    InputInDegrees=True,
    OutputInDegrees=True), 'deg')

# extract high and low tide times from ADCP data
lim_tod, lim_tor = bobo_general['ToD']+timedelta(hours=2), bobo_general['ToR']-timedelta(hours=2) 
bobo_high, bobo_low = get_tides(
    dataseries=bobo_aquadop['pressure'][0].loc[lim_tod:lim_tor]
    )
bobo_aquadop['high tide'] = (bobo_aquadop['pressure'][0].loc[lim_tod:lim_tor].iloc[bobo_high[0]], 'dbar')
bobo_aquadop['low tide'] = (bobo_aquadop['pressure'][0].loc[lim_tod:lim_tor].iloc[bobo_low[0]], 'dbar')


    
#%% #*### AQUADOPP DATA - ORTHOGONAL LINEAR REGRESSION ####
'''
Performs an orthogonal linear regression on the ADCP data (Aquadopp) and saves the results 
in a dictionary using the ortho_lin_regress function. 
Results are obtained per depth bin and only from first high tide to last high tide to avoid any
distortions due to uncompleted tidal cycles. 
The mean slope of the orthogonal regression over all depth bins is then used to define the 
along- and crosscanyon directions at the respective location.
'''

#*## Mooring 1

olr_m1 = orthogonal_linear_regression(
    DataDictionary=m1_aquadop, 
    GeneralDataDictionary=m1_general,
    PlotBool=False,
    PrintBool=False
    )  
m1_general['OLR'] = olr_m1  # save olr results in general dictionary
m1_aquadop = add_rotated_velocities(  # update aquadop dictionary with up and downcanyon velocities
    DataDictionary=m1_aquadop,
    CanyonDirection=olr_m1['BinMean'],
    HighBinLimit=20,
    LowBinLimit=0
    )  

#*## Mooring 2

olr_m2 = orthogonal_linear_regression(
    DataDictionary=m2_aquadop,
    GeneralDataDictionary=m2_general,
    PlotBool=False,
    PrintBool=False
    )  
m2_general['OLR'] = olr_m2  # save olr results in general dictionary
m2_aquadop = add_rotated_velocities(
    DataDictionary=m2_aquadop,
    CanyonDirection=olr_m2['BinMean'],
    HighBinLimit=20,
    LowBinLimit=0
    )  # update dictionary with up and downcanyon velocities


#*## Mooring 3

# orthogonal linear regression
olr_m3 = orthogonal_linear_regression(
    DataDictionary=m3_aquadop,
    GeneralDataDictionary=m3_general,
    PlotBool=False,
    PrintBool=False
    )

# save olr results in general dictionary
m3_general['OLR'] = olr_m3

# update dictionary with up and downcanyon velocities
m3_aquadop = add_rotated_velocities(
    DataDictionary=m3_aquadop,
    CanyonDirection=olr_m3['BinMean'],
    HighBinLimit=20,
    LowBinLimit=0
    )


#*## BoBo Lander

# orthogonal linear regression
olr_bobo = orthogonal_linear_regression(
    DataDictionary=bobo_aquadop,
    GeneralDataDictionary=bobo_general,
    PlotBool=False,
    PrintBool=False
    )

# save olr results in general dictionary
bobo_general['OLR'] = olr_bobo

# update dictionary with up and downcanyon velocities
bobo_aquadop = add_rotated_velocities(
    DataDictionary=bobo_aquadop,
    CanyonDirection=olr_bobo['BinMean'],
    HighBinLimit=20,
    LowBinLimit=0
    )


#%% #*### LOAD AQUADOPP DATA - CTD FRAME ####
'''
This sections loads the aquadop data from the raw csv files via the function "load_aquadopp_data". 
It only extracts data from the CTD casts at the research stations and not from the transect.
'''


#*## GENERAL SETUP 

filepath_ctd = os.path.join(
    ParentDirectory,'Data', 'Nortek_Aquadopp', 'CTD_data', 'Stations' 
    )  # filepath for CTD data

folders_ctd = [
    folder for folder in os.listdir(filepath_ctd)
    if os.path.isdir(os.path.join(filepath_ctd, folder))
    ] # list of folder names

filepathlist_ctd = [os.path.join(filepath_ctd,folder) for folder in folders_ctd]  # create list of filepaths

stationlist = [folder.split('_')[1] for folder in folders_ctd] # list of station names

aqdp_ctddata = {  
    station: {"cast_1": None, "cast_2": None} for station in sorted(set(stationlist))
    } # result dictionary

noisefloor_ctd = 20 # noise floor for noise removal [counts], based on exponential decay of amplitude data with distance from sensor, see ADCP_qualitycheck.py

# dictionary for canyon directions [degrees] other than S4, S6, S7, S8 where Bobo/Mooring data is available, retrieved visually from ARCGIS
canyonaxis_dic = {
    'Station1': -30,
    'Station2': -74,
    'Station3': -70,
    'Station5': 45,
}

#*## DATA

# iterate over each folder/filepath
for fp_ctd in filepathlist_ctd:
    
    # find serial of cast, extracted from arbitrary file in folder, here .a1 file
    ctd_serial_temp = [f for f in os.listdir(fp_ctd) if fnmatch.fnmatch(f, '*.a1')]
    
    # read data
    ctd_aquadop_temp = load_aquadopp_data(
        FilePath=fp_ctd,
        SensorID=ctd_serial_temp[0][:-3]
        )
    
    # create single amplitude data (mean of all beams)
    amp_stacked = np.stack( ## stack all beam amplitude data
            (ctd_aquadop_temp['amp_beam1_raw'][0],
            ctd_aquadop_temp['amp_beam2_raw'][0],
            ctd_aquadop_temp['amp_beam3_raw'][0]),
            axis=0
            )
    amp_mean = np.mean(amp_stacked, axis=0) 
    amp_df = pd.DataFrame(amp_mean, index=ctd_aquadop_temp['amp_beam1_raw'][0].index)
    ctd_aquadop_temp['amp_beam_raw'] = (amp_df, 'counts')
    
    # remove amplitude data based on noise floor (replaced with np.nan)
    ctd_aquadop_temp['amp_beam'] = (ctd_aquadop_temp['amp_beam_raw'][0].copy(), 'counts')
    ctd_aquadop_temp['amp_beam'][0][ctd_aquadop_temp['amp_beam'][0] < noisefloor_ctd] = np.nan
    
    # remove velocity data based on noise floor (replaced with np.nan)
    ctd_aquadop_temp['vel_east'] = (ctd_aquadop_temp['vel_east_raw'][0].copy(), 'm/s')
    ctd_aquadop_temp['vel_east'][0][np.isnan(ctd_aquadop_temp['amp_beam'][0])] = np.nan
    ctd_aquadop_temp['vel_north'] = (ctd_aquadop_temp['vel_north_raw'][0].copy(), 'm/s')
    ctd_aquadop_temp['vel_north'][0][np.isnan(ctd_aquadop_temp['amp_beam'][0])] = np.nan
    ctd_aquadop_temp['vel_up'] = (ctd_aquadop_temp['vel_up_raw'][0].copy(), 'm/s')
    ctd_aquadop_temp['vel_up'][0][np.isnan(ctd_aquadop_temp['amp_beam'][0])] = np.nan
    
    # additional calculated variables
    ctd_aquadop_temp['vel_magnitude_2D'] = (np.sqrt(ctd_aquadop_temp['vel_east'][0]**2 + ctd_aquadop_temp['vel_north'][0]**2), 'm/s')
    ctd_aquadop_temp['vel_magnitude_3D'] = (np.sqrt(ctd_aquadop_temp['vel_east'][0]**2 + ctd_aquadop_temp['vel_north'][0]**2 + ctd_aquadop_temp["vel_up"][0]**2), 'm/s')
    ctd_aquadop_temp['vel_direction_2D'] = (np.arctan2(ctd_aquadop_temp['vel_east'][0], ctd_aquadop_temp['vel_north'][0]), 'radians')

    # calculate tilt
    ctd_aquadop_temp['tilt'] = (
        calculate_tilt(
        Pitch=ctd_aquadop_temp['pitch'][0],
        Roll=ctd_aquadop_temp['roll'][0],
        InputInDegrees=True, 
        OutputInDegrees=True
        ),
        'deg')
    
    # convert mean acoustic backscatter values to spm using logarithmic regression. More Details in LinearRegression_Acoustic.py
    ctd_aquadop_temp['SPM[mg/L]'] = (
        acousticbackscatter_to_spm(
        BackscatterData=ctd_aquadop_temp['amp_beam'][0].iloc[:,:2].mean(axis=1),
        RegressionParameterA=AcousticRegressData['a'],
        RegressionParameterB=AcousticRegressData['b']
        ),
        'mg/L')
    
    # find station name and assign either cast_1 or cast_2 indicator -> usually singlecast is cast_1 and yoyo is cast_2 (exceptions are Station1 and Station8; no yoyo casts done at Station1, no singlecasts done at Station8)
    ctd_station_temp = fp_ctd.split('\\')[-1].split('_')[1]
    
    # calculate along & cross canyon velocities
    if ctd_station_temp in ['Station8']:
        ctd_aquadop_temp = add_rotated_velocities(
            DataDictionary=ctd_aquadop_temp,
            CanyonDirection=m1_general['OLR']['BinMean'],
            HighBinLimit=19, # maximum bin index since noise threshold is already accounted for
            LowBinLimit=0
            )
    elif ctd_station_temp in ['Station7']:
        ctd_aquadop_temp = add_rotated_velocities(
            DataDictionary=ctd_aquadop_temp,
            CanyonDirection=m2_general['OLR']['BinMean'],
            HighBinLimit=19,  # maximum bin index since noise threshold is already accounted for
            LowBinLimit=0
            )
    elif ctd_station_temp in ['Station6']:
        ctd_aquadop_temp = add_rotated_velocities(
            DataDictionary=ctd_aquadop_temp,
            CanyonDirection=m3_general['OLR']['BinMean'],
            HighBinLimit=19,  # maximum bin index since noise threshold is already accounted for
            LowBinLimit=0
            )
    elif ctd_station_temp in ['Station4']:
        ctd_aquadop_temp = add_rotated_velocities(
            DataDictionary=ctd_aquadop_temp,
            CanyonDirection=bobo_general['OLR']['BinMean'],
            HighBinLimit=19,  # maximum bin index since noise threshold is already accounted for
            LowBinLimit=0
            )
    else:
        ctd_aquadop_temp = add_rotated_velocities(
            DataDictionary=ctd_aquadop_temp,
            CanyonDirection=canyonaxis_dic[ctd_station_temp],
            HighBinLimit=19,  # maximum bin index since noise threshold is already accounted for
            LowBinLimit=0
            )
    
    # assign cast indicators so that result is saved in the correct subdictionary
    if fp_ctd.split('\\')[-1].split('_')[2] in ['singlecast']:
        castindicator = 'cast_1'
    else:
        castindicator = 'cast_2' 

    # save in result dictionary
    aqdp_ctddata[ctd_station_temp][castindicator] = ctd_aquadop_temp 


#%% #*## LOAD OBS DATA - MOORINGS & BOBO ###
'''-> Loads the obs data from the csv files for the moorings and the bottom lander
-> Smoothing and outlier removal is included in the load data function
-> data is also already converted to SPM'''


#*## GENERAL PARAMETERS 

fp_jfe = os.path.join(ParentDirectory, 'Data', 'JFE_OBS')
windowsize = 60 # window size for rolling mean [s]
std_threshold = 3 # threshold for standard deviation of rolling mean, used as threshold for despiking the data
unreal_threshold = 50 # threshold for unrealistically high values


#*## MOORING 1 

# filepath of obs data
fp_m1_obs = os.path.join(fp_jfe, 'Mooring_data', 'M1')

# serial numbers of obs sensors 
serials_obs_m1 = np.array(['08','10','11','15'])

# locations of obs sensors
locs_obs_m1 =np.array(['1m_seabed','1m_aqdp','4m_aqdp','50m_topline']) 

# load obs data from csv files
m1_obs = load_obs_data(
    FilePath=fp_m1_obs,
    SerialList=serials_obs_m1,
    LocationList=locs_obs_m1,
    WindowSize=windowsize,
    StdThreshold=std_threshold,
    UnrealThreshold=unreal_threshold,
    LinearRegressModel=TurbRegressData['alldata']['with_outliers']
    )


#*## MOORING 2 ###

# filepath of obs data
fp_m2_obs = os.path.join(fp_jfe, 'Mooring_data', 'M2')

# serial numbers of obs sensors
serials_obs_m2 = np.array(['16','17','18','19'])

# locations of obs sensors
locs_obs_m2 =np.array(['1m_seabed','1m_aqdp','4m_aqdp','50m_topline']) 

# load obs data from csv files
m2_obs = load_obs_data(
    FilePath=fp_m2_obs,
    SerialList=serials_obs_m2,
    LocationList=locs_obs_m2,
    WindowSize=windowsize,
    StdThreshold=std_threshold,
    UnrealThreshold=unreal_threshold,
    LinearRegressModel=TurbRegressData['alldata']['with_outliers']
    )

#*## MOORING 3 ###

# filepath of obs data
fp_m3_obs = os.path.join(fp_jfe, 'Mooring_data', 'M3')

# serial numbers of obs sensors
serials_obs_m3 = np.array(['20','21','25','26'])

# locations of obs sensors
locs_obs_m3 =np.array(['1m_seabed','1m_aqdp','4m_aqdp','50m_topline']) 

# load obs data from csv files 
m3_obs = load_obs_data(
    FilePath=fp_m3_obs,
    SerialList=serials_obs_m3,
    LocationList=locs_obs_m3,
    WindowSize=windowsize,
    StdThreshold=std_threshold,
    UnrealThreshold=unreal_threshold,
    LinearRegressModel=TurbRegressData['alldata']['with_outliers']
    )

#*## BOBO LANDER ###

# read obs data
obsdata_bobo = pd.read_csv(
    fp_jfe + '/BoBo_data/20230912_0800_ATUD-USB_0027_062324_P.csv',
    header=[55]
    )

# set datetime index
obsdata_bobo = obsdata_bobo.set_index(pd.to_datetime(obsdata_bobo['Meas date']), drop=True)

# create obs dictionary
bobo_obs = {
    'turb_1mab': (obsdata_bobo['Turb.-M[FTU]'], 'FTU'), 
    'temp_1mab': (obsdata_bobo['Temp.[degC]'], 'degC'), 
    'SPM_1mab': opticalbackscatter_to_spm(
        DataSeries=obsdata_bobo['Turb.-M[FTU]'],
        LinRegressModel=TurbRegressData['alldata']['with_outliers']
        )
    }


#%% #*## MICROCAT DATA - MOORING 3 ### 
'''-> loads the microcat data for mooring 3
-> calculates additional variables such as sea pressure, practical salinity, absolute salinity, conservative temperature and density'''


# filepath 
fp_ctd = os.path.join(ParentDirectory, 'Data', 'MicroCat', 'Mooring 3')

# instrument id
id_mc_m3 = 'SN2656'

# read data from asc file
m3_mc = pd.read_csv(
    fp_ctd + f'\\MC_{id_mc_m3}.asc',
    header=None,
    skiprows=56,
    delimiter=',',
    names=['temp. [degC]', 'cond. [S/m]', 'pressure [dbar]', "Date", "Hour"],
    )

# set and manipulate time index
m3_mc.set_index(
    pd.to_datetime(m3_mc['Date'] + ' ' + m3_mc['Hour']),
    drop=True,
    inplace=True
    )
m3_mc.drop(
    columns=['Date', 'Hour'],
    inplace=True
    )

# conversion of conductivity from S/m to mS/cm
m3_mc['cond. [mS/cm]'] = m3_mc['cond. [S/m]'] * 10

# calculate sea pressure
m3_mc['sea pressure [dbar]'] = m3_mc['pressure [dbar]'] - 10.1325

# calculate practical salinity 
m3_mc['SP [unitless]'] = SP_from_C(
    m3_mc['cond. [mS/cm]'],
    m3_mc['temp. [degC]'],
    m3_mc['sea pressure [dbar]']
    )

# calculate absolute salinity
m3_mc['SA [g/kg]'] = SA_from_SP(
    m3_mc['SP [unitless]'],
    m3_mc['sea pressure [dbar]'],
    m3_general['longitude'],
    m3_general['latitude']
    )

# calulcate conservative temperature
m3_mc['CT [degC]'] = CT_from_t(
    m3_mc['SA [g/kg]'],
    m3_mc['temp. [degC]'],
    m3_mc['sea pressure [dbar]']
    )

# calculate density 
m3_mc['density [kg/m^3]'] = density.rho(
    m3_mc['SA [g/kg]'],
    m3_mc['CT [degC]'],
    m3_mc['sea pressure [dbar]']
    )

#%% ### CTD - GENERAL DATA ###
'''-> creates general information dictionaries of each of the ctd casts with information such as their position, depth, begin, end and bottom times
-> due to input error in shipsdata, yoyo cast at 800m is split into two different files'''

#TODO: Combine yoyo 1 (500m) into one dictionary

#*## Single Casts

# dictionary names for single casts
dicnames_ctd_sc = ['sc_4000', 'sc_3500', 'sc_3000', 'sc_2500', 'sc_2000', 'sc_1500', 'sc_1000']

# SC 1_1: 4000m
sc_4000 = {
    'begin': datetime(2023, 9, 10, 11, 19),  # taken from shipsdata
    'bottom': datetime(2023,9,10,13,2),  # from shipsdata
    'end': datetime(2023,9,10,14,14),  # from shipsdata
    'latitude': 44.3987, # based on first bottom reading
    'longitude': 3.7152, # based on first bottom reading
    'depth_ea600': 3982.1,  # based on first bottom reading
    'depth_ea302': 3985.6, # based on first bottom reading
    'serial': '01CTD01', # from cruise report
    'distance_thalweg': 411.416,  # distance from beginning of thalweg [km]
    'distance_M1': 411.416 - 2.642,  # distance to 500m station / M1 [km]
    'cast_id': '1_1', # cast id from shipsdata
}

# # SC 1_2: 4000m
# sc_4000_2 = {
#     'begin': datetime(2023, 9, 15, 8, 27),  # from shipsdata
#     'bottom': datetime(2023, 9, 15, 9, 52),  # from shipsdata
#     'end': datetime(2023, 9, 15, 11, 22),  # from shipsdata
#     'latitude': 44.3997, # based on first bottom reading
#     'longitude': 3.7153, # based on first bottom reading
#     'depth_ea600': 3980.9,  # based on first bottom reading
#     'depth_ea302': 4010.59, # based on first bottom reading
#     'serial': '16CTD01',  # from cruise report
#     'distance_thalweg': 411.416,  # distance from beginning of thalweg [km]
#     'distance_M1': 411.416 - 2.642,  # distance to 500m station / M1 [km]
#     'cast_id': '16_1', # cast id from shipsdata
# }

# SC 2: 3500m
sc_3500 = {
    'begin': datetime(2023, 9, 10, 18, 3),  # from shipsdata
    'bottom': datetime(2023, 9, 10, 19, 12),  # from shipsdata
    'end': datetime(2023, 9, 10, 20, 35),  # from shipsdata
    'latitude': 43.8587, # based on first bottom reading
    'longitude': 3.6342, # based on first bottom reading
    'depth_ea600': 3463.6,  # based on first bottom reading
    'depth_ea302': 3495.05,  # based on first bottom reading
    'serial': '02CTD01',  # from cruise report
    'distance_thalweg': 307.465,  # distance from beginning of thalweg [km]
    'distance_M1': 307.465 - 2.642,  # distance to 500m station / M1 [km]
    'cast_id': '2_1',  # cast id from shipsdata
}

# SC 3: 3000m
sc_3000 = {
    'begin': datetime(2023, 9, 10, 23, 19),  # from shipsdata
    'bottom': datetime(2023, 9, 11, 0, 11),  # from shipsdata
    'end': datetime(2023, 9, 11, 1, 21),  # from shipsdata
    'latitude': 43.7897, # based on first bottom reading
    'longitude': 3.1213, # based on first bottom reading
    'depth_ea600': 2984.9,  # based on first bottom reading
    'depth_ea302': 3009.52,  # based on first bottom reading
    'serial': '03CTD01',  # from cruise report
    'distance_thalweg': 226.636,  # distance from beginning of thalweg [km]
    'distance_M1': 226.636 - 2.642,  # distance to 500m station / M1 [km]
    'cast_id': '3_1',  # cast id from shipsdata
}

# SC 4: 2500m
sc_2500 = {
    'begin': datetime(2023, 9, 11, 3, 22),  # from shipsdata
    'bottom': datetime(2023, 9, 11, 4, 10),  # from shipsdata
    'end': datetime(2023, 9, 11, 5, 13),  # from shipsdata
    'latitude': 43.7575, # based on first bottom reading
    'longitude': 2.7792, # based on first bottom reading
    'depth_ea600': 2567.1, # based on first bottom reading
    'depth_ea302': 2595.06,  # based on first bottom reading
    'serial': '04CTD01',  # from cruise report
    'distance_thalweg': 175.232,  # distance from beginning of thalweg [km]
    'distance_M1': 175.232 - 2.642,  # distance to 500m station / M1 [km]
    'cast_id': '4_1',  # cast id from shipsdata
}

# SC 5: 2000m
sc_2000 = {
    'begin': datetime(2023, 9, 11, 7, 54),  # from shipsdata
    'bottom': datetime(2023, 9, 11, 8, 28),  # from shipsdata
    'end': datetime(2023, 9, 11, 9, 22),  # from shipsdata
    'latitude': 43.6635, # based on first bottom reading
    'longitude': 2.335, # based on first bottom reading
    'depth_ea600': 1832.6,  # based on first bottom reading
    'depth_ea302': 1862.62,  # based on first bottom reading
    'serial': '05CTD01',  # from cruise report
    'distance_thalweg': 104.951,  # distance from beginning of thalweg [km]
    'distance_M1': 104.951 - 2.642,  # distance to 500m station / M1 [km]
    'cast_id': '5_1',  # cast id from shipsdata
}

#SC 6: 1500m
sc_1500 = {
    'begin': datetime(2023, 9, 11, 10, 39),  # from shipsdata
    'bottom': datetime(2023, 9, 11, 11, 12),  # from shipsdata
    'end': datetime(2023, 9, 11, 11, 53),  # from shipsdata
    'latitude': 43.6378, # based on first bottom reading
    'longitude': 2.1278, # based on first bottom reading
    'depth_ea600': 1467.9, # based on first bottom reading
    'depth_ea302': 1491.34,  # based on first bottom reading
    'serial': '06CTD01',  # from cruise report
    'distance_thalweg': 76.901,  # distance from beginning of thalweg [km]
    'distance_M1': 76.901 - 2.642,  # distance to 500m station / M1 [km]
    'cast_id': '6_1',  # cast id from shipsdata
}

# SC 7: 1000m
sc_1000 = {
    'begin': datetime(2023, 9, 11, 13, 13),  # from shipsdata
    'bottom': datetime(2023, 9, 11, 13, 41),  # from shipsdata
    'end': datetime(2023, 9, 11, 14, 10),  # from shipsdata
    'latitude': 43.6077, # based on first bottom reading
    'longitude': 1.8975, # based on first bottom reading
    'depth_ea600': 1011.4,  # based on first bottom reading
    'depth_ea302': 1043.24,  # based on first bottom reading
    'serial': '07CTD01',  # from cruise report
    'distance_thalweg': 40.052,  # distance from beginning of thalweg [km]
    'distance_M1': 40.052 - 2.642,  # distance to 500m station / M1 [km]
    'cast_id': '7_1',  # cast id from shipsdata
}

#*## Yoyo Casts

# YOYO 8_1: 500m
yoyo_500_1 = {
    'begin': datetime(2023, 9, 11, 16, 47),  # from shipsdata
    'end': datetime(2023, 9, 12, 3, 40),  # from shipsdata
    'latitude': 43.6492, # based on first bottom reading
    'longitude': 1.6568, # based on first bottom reading
    'depth_ea600': 552.34,  # based on first bottom reading 
    'depth_ea302': 561.27,  # based on first bottom reading
    'serial': '08CTD01', # from cruise report
    'serial_adcp': 'YOY8UP09',  # from cruise report
    'distance_thalweg': 2.642, # distance to beginning of thalweg profile [km]
    'distance_M1': 0, # distance to 500m station / M1 [km]
    'cast_id': '8_1',  # cast id from shipsdata
}

# YOYO 8_2: 500m
yoyo_500_2 = {
    'begin': datetime(2023, 9, 12, 4, 1),  # from shipsdata
    'end': datetime(2023, 9, 12, 7, 7),  # from shipsdata
    'latitude': 43.6492, # based on first bottom reading
    'longitude': 1.6563, # based on first bottom reading
    'depth_ea600': 552.64,  # based on first bottom reading
    'depth_ea302': 562.52,  # based on first bottom reading
    'serial': '08CTD02',  # from cruise report
    'distance_thalweg': 2.642,  # distance to beginning of thalweg profile [km]
    'distance_M1': 0, # distance to 500m station / M1 [km]
    'cast_id': '8_2',  # cast id from shipsdata
}


# YOYO 14: 1000m
yoyo_1000 = {
    'begin': datetime(2023, 9, 13, 3, 15),  # from shipsdata
    'end': datetime(2023, 9, 13, 17, 33),  # from shipsdata
    'latitude': 43.6042, # based on first bottom reading
    'longitude': 1.9012, # based on first bottom reading
    'depth_ea600': 1030.6,  # based on first bottom reading
    'depth_ea302': 1049.82,  # based on first bottom reading
    'serial': '14CTD01',   # from cruise report
    'serial_adcp': 'YOY7UP10',  # from cruise report
    'distance_thalweg': 40.567,  # distance to beginning of thalweg profile [km]
    'distance_M1': 40.567 - 2.642, # distance to 500m station / M1 [km]
    'cast_id': '14_1',  # cast id from shipsdata
}

# YOYO 15: 1500m
yoyo_1500 = {
    'begin': datetime(2023, 9, 13, 22, 18),  # from shipsdata
    'end': datetime(2023, 9, 14, 12, 50),  # from shipsdata
    'latitude': 43.6347, # based on first bottom reading
    'longitude': 2.1237, # based on first bottom reading
    'depth_ea600': 1447,  # based on first bottom reading
    'depth_ea302': 1485.31,  # based on first bottom reading
    'serial': '15CTD01',  # from cruise report
    'serial_adcp': 'YOY6UP11',   # from cruise report
    'distance_thalweg': 76.292,  # distance to beginning of thalweg profile [km]
    'distance_M1': 76.292 - 2.642, # distance to 500m station / M1 [km]
    'cast_id': '15_1',  # cast id from shipsdata
}

# YOYO 17: 3500m
yoyo_3500 = {
    'begin': datetime(2023, 9, 15, 19, 20), # from shipsdata
    'end': datetime(2023, 9, 16, 10, 29),  # from shipsdata
    'latitude': 43.8587, # based on first bottom reading
    'longitude': 3.634, # based on first bottom reading
    'depth_ea600': 3396.2,  # based on first bottom reading
    'depth_ea302': 3494.68,  # based on first bottom reading
    'serial': '17CTD01',   # from cruise report
    'serial_adcp': 'YOY2UP13',  # from cruise report
    'distance_thalweg': 307.487,  # distance to beginning of thalweg profile [km]
    'distance_M1': 307.487 - 2.642, # distance to 500m station / M1 [km]
    'cast_id': '17_1',  # cast id from shipsdata
}

# YOYO 18: 3000m
yoyo_3000 = {
    'begin': datetime(2023, 9, 16, 19, 30),  # from shipsdata
    'end': datetime(2023, 9, 17, 10, 50),  # from shipsdata
    'latitude': 43.7895, # based on first bottom reading
    'longitude': 3.1187, # based on first bottom reading
    'depth_ea600': 2980.3, # NOT BASED ON FIRST BOTTOM BUT ON SECOND BOTTOM READING
    'depth_ea302': 3006.51, # NOT BASED ON FIRST BOTTOM BUT ON SECOND BOTTOM READING
    'serial': '18CTD01',  # from cruise report
    'serial_adcp': 'YOY3UP14',  #from cruise report
    'distance_thalweg': 226.683,  # distance to beginning of thalweg profile [km]
    'distance_M1': 226.683 - 2.642, # distance to 500m station / M1 [km]
    'cast_id': '18_1',  # cast id from shipsdata
}

# YOYO 19: 2500m
yoyo_2500 = {
    'begin': datetime(2023, 9, 17, 18, 42), # from shipsdata
    'end': datetime(2023, 9, 18, 10, 31),  # from shipsdata
    'latitude': 43.7562, # based on first bottom reading
    'longitude': 2.7733, # based on first bottom reading
    'depth_ea600': 2557.5, # NOT BASED ON FIRST BOTTOM BUT ON THIRD BOTTOM READING
    'depth_ea302': 2585.57, # NOT BASED ON FIRST BOTTOM BUT ON THIRD BOTTOM READING
    'serial': '19CTD01',  # from cruise report
    'serial_adcp': 'YOY4UP15', # from cruise report
    'distance_thalweg': 174.649,  # distance to beginning of thalweg profile [km]
    'distance_M1': 174.649 - 2.642, # distance to 500m station / M1 [km]
    'cast_id': '19_1',  # cast id from shipsdata
}

# YOYO 21: 2000m
yoyo_2000 = {
    'begin': datetime(2023, 9, 18, 19, 13),  # from shipsdata
    'end': datetime(2023, 9, 19, 7, 54),  # from shipsdata
    'latitude': 43.664, # based on first bottom reading
    'longitude': 2.3345, # based on first bottom reading
    'depth_ea600': 1861.2,  # based on first bottom reading
    'depth_ea302': 1861.75,  # based on first bottom reading
    'serial': '21CTD01', # from cruise report
    'serial_adcp': 'YOY5UP02',  # from cruise report
    'distance_thalweg': 104.931,  # distance to beginning of thalweg profile [km]
    'distance_M1': 104.931 - 2.642, # distance to 500m station / M1 [km]
    'cast_id': '21_1',  # cast id from shipsdata
}


#%%
# dictionary names for single casts
dicnames_yoyo = ['yoyo_500_1', 'yoyo_500_2', 'yoyo_1000', 'yoyo_1500', 'yoyo_3500', 'yoyo_3000', 'yoyo_2500', 'yoyo_2000' ]

# list for general properties defined above
scs = [sc_4000, sc_3500, sc_3000, sc_2500, sc_2000, sc_1500, sc_1000]

# serial numbers of single cast CTDs
serials_sc = []
for sc in scs:
    serials_sc.append(sc['serial'])
    

# dictionary for single cast data
ctddata_sc = {}


# list for general properties defined above
yoyos = [yoyo_500_1, yoyo_500_2, yoyo_1000, yoyo_1500, yoyo_3500, yoyo_3000, yoyo_2500, yoyo_2000]

# serial numbers of single cast CTDs
serials_yoyo = []
for yoyo in yoyos:
    serials_yoyo.append(yoyo['serial'])

#%% ### CTD - READ DATA ###




# save location of CTD data, filenames, column names
fp_ctd = os.path.join(ParentDirectory, 'Data', 'CTD', 'Processed', 'Binavg_1Hz')
fp_ctd_aquadopp = os.path.join(ParentDirectory, 'Data', 'Nortek_Aquadopp', 'CTD_data')
fp_ctd = 'C:/Users/werne/Documents/UniUtrecht/MasterThesis/CTD/Processed/Binavg_1Hz/'
fp_ctd_aquadopp = 'C:/Users/werne/Documents/UniUtrecht/MasterThesis/Nortek_Aquadopp/CTD_data/'


filenames_ctd = glob.glob(fp_ctd + '*.cnv')
columns_ctd = ['scan count', 'pressure', 'temperature', 'conductivity', 'oxygen', 'time [s]', 'beam transmission', 'spar/linear', 'par/logarithmic', 'fluorescence', 'turbidity', 'fluorescence_afl', 'salinity', 'potential temperature', 'density_t', 'depth', 'altimeter', 'density_theta', 'depth_atlatitude', 'potential temperature 2', 'salinity 2', 'descent rate', 'flag']



# dictionary for yoyo cast data
ctddata_yoyo = {}


### READING DATA ###

for filename in filenames_ctd:
    
    # read out serials from filename 
    serial = filename.split('-')[-1].split('.')[0]
    
    # open ctd data file
    with open(filename, 'r') as f:
        foundend = False
        foundstart = False
        # Iterate over the lines
        for i, line in enumerate(f):
            # Check if the line contains "*END*"
            if '*END*' in line:
                end_line = i
                foundend = True
            if '# start_time =' in line:
                start_line = i
                starttime = line.split('=')[-1].strip()
                starttime = re.sub(r'\s\[.*?\]', '', starttime)
                starttime = datetime.strptime(starttime, "%b %d %Y %H:%M:%S")   
                foundstart = True
            if foundend & foundstart:
                break

    # load data
    data = pd.read_csv(filename,
                       skiprows=end_line+1,
                       header=None,
                       delim_whitespace=True,
                       encoding='ISO-8859-1', 
                       names=columns_ctd)
    
    # create time index for data
    time = pd.date_range(start=starttime, periods=len(data['time [s]']), freq='S')
    
    # set time index
    data.set_index(time, drop=True, inplace=True)
    
    # remove unnecessary columns
    data.drop(columns=['scan count', 'beam transmission', 'spar/linear', 'par/logarithmic', 'fluorescence', 'fluorescence_afl', 'flag', 'salinity 2', 'potential temperature 2'], inplace=True)

    # despike data
    # columns to be despiked
    columns = [col for col in data.columns if col not in ['time [s]', 'depth', 'altimeter', 'depth_atlatitude', 'descent rate']]
    
    # data_despiked = data.copy()
    # for col in columns:
    #     print(col)
    #     data_despiked[col] = diff_despiking_ti(data[col], stdthreshold=3, resample='5s')
    data_despiked = data.apply(z_score_despiking, window_size=50, threshold=3)


    ## additional calculations that are necessary for section below

    # calculate sea pressure
    data['sea pressure'] = data['pressure'] - 10.1325
    data_despiked['sea pressure'] = data_despiked['pressure'] - 10.1325

    # # calculate practical salinity 
    data['SP'] = SP_from_C(data['conductivity']*10, data['temperature'], data['sea pressure'])
    data_despiked['SP'] = SP_from_C(data_despiked['conductivity']*10, data_despiked['temperature'], data_despiked['sea pressure'])
    
    if serial in serials_sc:
        name = dicnames_ctd_sc[serials_sc.index(serial)]
        generalinfo = scs[serials_sc.index(serial)]
        ctddata_sc[name] = data
        ctddata_sc[f'{name}_general'] = generalinfo
        ctddata_sc[f'{name}_despiked'] = data_despiked
    elif serial in serials_yoyo:
        name = dicnames_yoyo[serials_yoyo.index(serial)]
        generalinfo = yoyos[serials_yoyo.index(serial)]
        ctddata_yoyo[name] = data
        ctddata_yoyo[f'{name}_general'] = generalinfo
        ctddata_yoyo[f'{name}_despiked'] = data_despiked

#%% ### CTD - ADDITIONAL CALCULATIONS ###

# create temporary dictionary for results
temp_dict = {}

# single cast key selection
sc_keys_normal = [key for key in ctddata_sc.keys() if not key.endswith(('_despiked','_general'))]

# for normal data
for singlecast in sc_keys_normal:

    if '_general' not in singlecast:

        # create results dictionary and array
        N2 = {}
        N2_pressure = np.ones(len(ctddata_sc[singlecast]['salinity']-1)) * np.nan
        N2_array = np.ones(len(ctddata_sc[singlecast]['salinity']-1)) * np.nan
        
        # calculate absolute salinity
        ctddata_sc[singlecast]['SA'] = SA_from_SP(ctddata_sc[singlecast]['SP'], ctddata_sc[singlecast]['sea pressure'], ctddata_sc[f'{singlecast}_general']['longitude'], ctddata_sc[f'{singlecast}_general']['latitude'])

        # calulcate conservative temperature
        ctddata_sc[singlecast]['CT'] = CT_from_t(ctddata_sc[singlecast]['SA'], ctddata_sc[singlecast]['temperature'], ctddata_sc[singlecast]['sea pressure'])

        # calculate buoyancy frequency
        N2_array, N2_pressure = Nsquared(ctddata_sc[singlecast]['SA'], 
                                         ctddata_sc[singlecast]['CT'],
                                         ctddata_sc[singlecast]['sea pressure'],
                                         lat=ctddata_sc[f'{singlecast}_general']['latitude'])

        # save results in dictionary
        N2['N2'] = N2_array
        N2['N2_pressure'] = N2_pressure
        
        # convert array to dataframe
        N2 = pd.DataFrame(N2)
        N2_pressure = pd.DataFrame(N2_pressure)
        N2['N2_despiked'] = z_score_despiking(N2['N2'])

        # save in temporary dictionary
        temp_dict[f'{singlecast}_N2'] = N2

# update dictionary with N2 data
ctddata_sc.update(temp_dict)

# new dictionary
temp_dict={}

# yoyo cast key selection
yoyo_keys_normal = [key for key in ctddata_yoyo.keys() if not key.endswith(('_despiked','_general'))]

# iterate over yoyo cast data
for yoyo in yoyo_keys_normal:
    
        if '_general' not in yoyo:

            # create results dictionary and array
            N2 = {}
        
            # calculate absolute salinity
            ctddata_yoyo[yoyo]['SA'] = SA_from_SP(ctddata_yoyo[yoyo]['SP'], ctddata_yoyo[yoyo]['sea pressure'], ctddata_yoyo[f'{yoyo}_general']['longitude'], ctddata_yoyo[f'{yoyo}_general']['latitude'])
        
            # calulcate conservative temperature
            ctddata_yoyo[yoyo]['CT'] = CT_from_t(ctddata_yoyo[yoyo]['SA'], ctddata_yoyo[yoyo]['temperature'], ctddata_yoyo[yoyo]['sea pressure'])
        
            # calculate buoyancy frequency
            N2_array, N2_pressure = Nsquared(ctddata_yoyo[yoyo]['SA'], ctddata_yoyo[yoyo]['CT'], ctddata_yoyo[yoyo]['sea pressure'], lat=ctddata_yoyo[f'{yoyo}_general']['latitude'])

            # save results in dictionary
            N2['N2'] = pd.DataFrame(N2_array)
            N2['N2_pressure'] = pd.DataFrame(N2_pressure)
            N2['N2_despiked'] = z_score_despiking(N2['N2'])

            # save in temporary dictionary
            temp_dict[f'{yoyo}_N2'] = N2

# update dictionary with N2 data
ctddata_yoyo.update(temp_dict)

#%% ### CANYON DATA ###

# read elevation profile
elevation = pd.read_csv('C:/Users/werne/Documents/UniUtrecht/MasterThesis/CanyonData/elevationprofile_highresolution_downcanyon.csv', delimiter=',', dtype=float, decimal=',') 

# calculate slope
elevation['slope'] = np.gradient(elevation['Elevation'], elevation['Distance'])

# define a function to be fitted to the data, this case second order polynomial
def func(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the curve
popt, pcov = curve_fit(func, elevation['Distance'], elevation['Elevation'])

# popt contains the optimized parameters
a, b, c = popt

# Now, you can use a, b, and c to plot the fitted curve
elevation['Elevation_fitted'] = func(elevation['Distance'], a, b, c)


# fitted slope
elevation['slope_fitted'] = np.gradient(elevation['Elevation_fitted'], elevation['Distance'])

#%% ### OBS DATA - CTD CASTS ###


### GENERAL PARAMETERS ### 

fp_ctd = 'C:/Users/werne/Documents/UniUtrecht/MasterThesis/JFE_OBS/CTD_data/data_processed/'  # general filepath
windowsize = 60 # window size for rolling mean [s]
std_threshold = 3 # threshold for standard deviation of rolling mean
unreal_threshold = 50 # threshold for unrealistically high values
allcastslist = ['1_1', '2_1', '3_1', '4_1', '5_1', '6_1', '7_1', '8_1', '14_1', '15_1', '16_1', '17_1', '18_1', '19_1', '21_1'] # endings of csv files per cast
yoyolist = ['8_1', '8_2', '14_1', '15_1', '17_1', '18_1', '19_1', '21_1']  # list of cast that are yoyos
sclist = [cast for cast in allcastslist if cast not in yoyolist]  # list of casts that are singlecasts
jfe_dict_all = {cast: None for cast in allcastslist} # create results array with a key for each cast
jfe_dict_yoyo = {cast: None for cast in yoyolist} # create results array with a key for each cast
jfe_dict_sc = {cast: None for cast in sclist} # create results array with a key for each cast

# iterate over all casts
for cast in allcastslist:
    
    # temporary file cast that moves into folder corresponding to cast
    filepath_temp = f'{fp_ctd}{cast}/' 

    # retrieve paths of all csv files within this folder
    filenames_ctd =glob.glob(filepath_temp + "*.csv")
        
    # save each df according to its position
    if cast in ['1_1', '2_1', '3_1', '4_1', '5_1', '6_1', '7_1']:
        
        # cast specific setup in order of serial number (i.e. also in order of file in filenames)
        positionlist = ['frame', '2m', '3m', '4m']
        
        # empty list for dataframes
        df_list =[]
        
        # data processing of csv file, taken from load_obs function
        for file in filenames_ctd:
            df = pd.read_csv(file, index_col=[0], parse_dates=True)
            df['Turb_rolling_mean'] = df['Turb.-M[FTU]'].rolling(window=windowsize).mean()
            df['Turb_rolling_std'] = df['Turb.-M[FTU]'].rolling(window=windowsize).std()
            df['is_outlier'] = (df['Turb.-M[FTU]'] > df['Turb_rolling_mean'] + std_threshold*df['Turb_rolling_std']) | (df['Turb.-M[FTU]'] < df['Turb_rolling_mean'] - std_threshold*df['Turb_rolling_std'])
            df['Turb.-M[FTU]_filtered'] = df['Turb.-M[FTU]']
            df.loc[df['is_outlier'], 'Turb.-M[FTU]_filtered'] = np.nan # replace outliers with NaN
            df.loc[df['Turb.-M[FTU]_filtered'] > unreal_threshold, 'Turb.-M[FTU]_filtered'] = np.nan # replace unrealistically high values with NaN
            df = df.drop(columns=['Turb_rolling_mean', 'Turb_rolling_std', 'is_outlier'])
            df['SPM[mg/L]'] = opticalbackscatter_to_spm(df['Turb.-M[FTU]_filtered'], linregress['alldata']['with_outliers'])
            df_list.append(df)
            
        # saving resulting dfs in list and then in dictionary
        jfe_dict_all[cast] = {position: df for position, df in zip(positionlist, df_list)}
        # saving resulting dfs in list and then in dictionary
        jfe_dict_sc[cast] = {position: df for position, df in zip(positionlist, df_list)}
        
        
    elif cast in ['8_1', '16_1']:
        
        # cast specific setup in order of serial number (i.e. also in order of file in filenames)
        positionlist =['frame', '5m', '3m', '1m']
        
        # empty list for dataframes
        df_list =[]
        
        # data processing of csv file
        for file in filenames_ctd:
            df = pd.read_csv(file, index_col=[0], parse_dates=True)
            df['Turb_rolling_mean'] = df['Turb.-M[FTU]'].rolling(window=windowsize).mean()
            df['Turb_rolling_std'] = df['Turb.-M[FTU]'].rolling(window=windowsize).std()
            df['is_outlier'] = (df['Turb.-M[FTU]'] > df['Turb_rolling_mean'] + std_threshold*df['Turb_rolling_std']) | (df['Turb.-M[FTU]'] < df['Turb_rolling_mean'] - std_threshold*df['Turb_rolling_std'])
            df['Turb.-M[FTU]_filtered'] = df['Turb.-M[FTU]']
            df.loc[df['is_outlier'], 'Turb.-M[FTU]_filtered'] = np.nan # replace outliers with NaN
            df.loc[df['Turb.-M[FTU]_filtered'] > unreal_threshold, 'Turb.-M[FTU]_filtered'] = np.nan # replace unrealistically high values with NaN
            df = df.drop(columns=['Turb_rolling_mean', 'Turb_rolling_std', 'is_outlier'])
            df['SPM[mg/L]'] = opticalbackscatter_to_spm(df['Turb.-M[FTU]_filtered'], linregress['alldata']['with_outliers'])
            df_list.append(df)
            
        # saving resulting dfs in list and then in dictionary
        jfe_dict_all[cast] = {position: df for position, df in zip(positionlist, df_list)}
        
        # saving resulting dfs in list and then in dictionary
        if cast in ['8_1']:
            jfe_dict_yoyo[cast] = {position: df for position, df in zip(positionlist, df_list)}
        elif cast in ['16_1']:
            jfe_dict_sc[cast] = {position: df for position, df in zip(positionlist, df_list)}
        
        
    elif cast in ['14_1']:
        
        # cast specific setup in order of serial number (i.e. also in order of file in filenames)
        positionlist = ['frame', '5m', '3m', '1m', '500m', '1000m']
        
        # empty list for dataframes
        df_list = []
        
        # data processing of csv file
        for file in filenames_ctd:
            df = pd.read_csv(file, index_col=[0], parse_dates=True)
            df['Turb_rolling_mean'] = df['Turb.-M[FTU]'].rolling(window=windowsize).mean()
            df['Turb_rolling_std'] = df['Turb.-M[FTU]'].rolling(window=windowsize).std()
            df['is_outlier'] = (df['Turb.-M[FTU]'] > df['Turb_rolling_mean'] + std_threshold*df['Turb_rolling_std']) | (df['Turb.-M[FTU]'] < df['Turb_rolling_mean'] - std_threshold*df['Turb_rolling_std'])
            df['Turb.-M[FTU]_filtered'] = df['Turb.-M[FTU]']
            df.loc[df['is_outlier'], 'Turb.-M[FTU]_filtered'] = np.nan # replace outliers with NaN
            df.loc[df['Turb.-M[FTU]_filtered'] > unreal_threshold, 'Turb.-M[FTU]_filtered'] = np.nan # replace unrealistically high values with NaN
            df = df.drop(columns=['Turb_rolling_mean', 'Turb_rolling_std', 'is_outlier'])
            df['SPM[mg/L]'] = opticalbackscatter_to_spm(df['Turb.-M[FTU]_filtered'], linregress['alldata']['with_outliers'])
            df_list.append(df)
            
        # saving resulting dfs in list and then in dictionary   
        jfe_dict_all[cast] = {position: df for position, df in zip(positionlist, df_list)}
        jfe_dict_yoyo[cast] = {position: df for position, df in zip(positionlist, df_list)}
        
    elif cast in ['15_1', '17_1', '18_1']:
        
        # cast specific setup in order of serial number (i.e. also in order of file in filenames)
        positionlist = ['frame', '5m', '3m', '1m', '500m', '1000m']
        
        # empty list for dataframes
        df_list = []
        
        # data processing of csv file
        for file in filenames_ctd:
            df = pd.read_csv(file, index_col=[0], parse_dates=True)
            df['Turb_rolling_mean'] = df['Turb.-M[FTU]'].rolling(window=windowsize).mean()
            df['Turb_rolling_std'] = df['Turb.-M[FTU]'].rolling(window=windowsize).std()
            df['is_outlier'] = (df['Turb.-M[FTU]'] > df['Turb_rolling_mean'] + std_threshold*df['Turb_rolling_std']) | (df['Turb.-M[FTU]'] < df['Turb_rolling_mean'] - std_threshold*df['Turb_rolling_std'])
            df['Turb.-M[FTU]_filtered'] = df['Turb.-M[FTU]']
            df.loc[df['is_outlier'], 'Turb.-M[FTU]_filtered'] = np.nan # replace outliers with NaN
            df.loc[df['Turb.-M[FTU]_filtered'] > unreal_threshold, 'Turb.-M[FTU]_filtered'] = np.nan # replace unrealistically high values with NaN
            df = df.drop(columns=['Turb_rolling_mean', 'Turb_rolling_std', 'is_outlier'])
            df['SPM[mg/L]'] = opticalbackscatter_to_spm(df['Turb.-M[FTU]_filtered'], linregress['alldata']['with_outliers'])
            df_list.append(df)
        
        # saving resulting dfs in list and then in dictionary 
        jfe_dict_all[cast] = {position: df for position, df in zip(positionlist, df_list)}
        jfe_dict_yoyo[cast] = {position: df for position, df in zip(positionlist, df_list)}
        
    elif cast in ['19_1', '21_1']:
        
        # cast specific setup in order of serial number (i.e. also in order of file in filenames)
        positionlist = ['3m', '5m', 'frame', '1m', '500m', '1000m']
        
        # empty list for dataframes
        df_list = []
        
        # data processing of csv file
        for file in filenames_ctd:
            df = pd.read_csv(file, index_col=[0], parse_dates=True)
            df['Turb_rolling_mean'] = df['Turb.-M[FTU]'].rolling(window=windowsize).mean()
            df['Turb_rolling_std'] = df['Turb.-M[FTU]'].rolling(window=windowsize).std()
            df['is_outlier'] = (df['Turb.-M[FTU]'] > df['Turb_rolling_mean'] + std_threshold*df['Turb_rolling_std']) | (df['Turb.-M[FTU]'] < df['Turb_rolling_mean'] - std_threshold*df['Turb_rolling_std'])
            df['Turb.-M[FTU]_filtered'] = df['Turb.-M[FTU]']
            df.loc[df['is_outlier'], 'Turb.-M[FTU]_filtered'] = np.nan # replace outliers with NaN
            df.loc[df['Turb.-M[FTU]_filtered'] > unreal_threshold, 'Turb.-M[FTU]_filtered'] = np.nan # replace unrealistically high values with NaN
            df = df.drop(columns=['Turb_rolling_mean', 'Turb_rolling_std', 'is_outlier'])
            df['SPM[mg/L]'] = opticalbackscatter_to_spm(df['Turb.-M[FTU]_filtered'], linregress['alldata']['with_outliers'])
            df_list.append(df)
            
        # saving resulting dfs in list and then in dictionary 
        jfe_dict_all[cast] = {position: df for position, df in zip(positionlist, df_list)}
        jfe_dict_yoyo[cast] = {position: df for position, df in zip(positionlist, df_list)}


#%% ### MICROCAT DATA ### 

### Mooring 3 ###

# filepath and serial id
fp_ctd = 'C:/Users/werne/Documents/UniUtrecht/MasterThesis/CTD_microcat/Mooring 3/'
id_mc_m3 = 'SN2656'

# read data
m3_mc = pd.read_csv(fp_ctd + f'MC_{id_mc_m3}.asc',
                    header=None,
                    skiprows=56,
                    delimiter=',',
                    names=['temp. [degC]', 'cond. [S/m]', 'pressure [dbar]', "Date", "Hour"],
                    )

# set and manipulate time index
m3_mc.set_index(pd.to_datetime(m3_mc['Date'] + ' ' + m3_mc['Hour']), drop=True, inplace=True)
m3_mc.drop(columns=['Date', 'Hour'], inplace=True)

# conversion of conductivity from S/m to mS/cm
m3_mc['cond. [mS/cm]'] = m3_mc['cond. [S/m]'] * 10

# calculate sea pressure
m3_mc['sea pressure [dbar]'] = m3_mc['pressure [dbar]'] - 10.1325

# calculate practical salinity 
m3_mc['SP [unitless]'] = SP_from_C(m3_mc['cond. [mS/cm]'], m3_mc['temp. [degC]'], m3_mc['sea pressure [dbar]'])

# calculate absolute salinity
m3_mc['SA [g/kg]'] = SA_from_SP(m3_mc['SP [unitless]'], m3_mc['sea pressure [dbar]'], m3_general['longitude'], m3_general['latitude'])

# calulcate conservative temperature
m3_mc['CT [degC]'] = CT_from_t(m3_mc['SA [g/kg]'], m3_mc['temp. [degC]'], m3_mc['sea pressure [dbar]'])

# calculate density 
m3_mc['density [kg/m^3]'] = density.rho(m3_mc['SA [g/kg]'], m3_mc['CT [degC]'], m3_mc['sea pressure [dbar]'])

# calculate buyoancy frequency


#%% ### FILTER WEIGHT DATA ###

# filepath of csv files
fp_ctd = 'C:/Users/werne/Documents/UniUtrecht/MasterThesis/Filters/'
filename =glob.glob(fp_ctd + "*.csv")[0]

spm_df_full = pd.read_csv(filename, sep=';', header=[0], skiprows=[1])

# create datetime index
spm_df_full['DateTime'] = pd.to_datetime(spm_df_full['Date'] + ' ' + spm_df_full['Time'], format='%d.%m.%Y %H:%M:%S')
spm_df_full.set_index('DateTime', inplace=True)

# only keep desired columns
columns_to_keep = ['Station/Cast', 'SPM']
spm_df_full = spm_df_full.drop(columns=spm_df_full.columns.difference(columns_to_keep))




#%% ### DICTIONARIES ###

# dictionary for BoBo Lander
bobodata = {'aquadop': bobo_aquadop,
            'obs': bobo_obs,
            'general': bobo_general
            }

# dictionary for Mooring 1
m1data = {'aquadop': m1_aquadop,
            'obs': m1_obs,
            'general': m1_general
            }

# dictionary for Mooring 2
m2data = {'aquadop': m2_aquadop,
            'obs': m2_obs,
            'general': m2_general
            }

# dictionary for Mooring 3
m3data = {'aquadop': m3_aquadop,
            'obs': m3_obs,
            'general': m3_general, 
            'microcat': m3_mc,
            }

# ctddata = {'singlecasts': ctddata_sc,
#            'yoyos': ctddata_yoyo}

canyondata = {'elevation_raw': elevation['Elevation'],
              'elevation_fitted': elevation['Elevation_fitted'],
              'slope_raw': elevation['slope'],
              'slope_fitted': elevation['slope_fitted'],
              'distance': elevation['Distance']
}

# dictionary for CTD JFE Data
obscastdata = {'all': jfe_dict_all,
            'scs': jfe_dict_sc,
            'yoyos': jfe_dict_yoyo, 
            }

#%% ### SAVE ###

path = os.getcwd() + '/pickles/'  # path of current working directory plus pickle folder

# Savind dictionary of Mooring 1
savepath = pathlib.Path(path, 'm1data.pickle') 
with open(savepath, 'wb') as f: 
    pickle.dump(m1data, f)

# Saving dictionary of Mooring 2
savepath = pathlib.Path(path, 'm2data.pickle')
with open(savepath, 'wb') as f:
    pickle.dump(m2data, f)

# Saving dictionary of Mooring 3
savepath = pathlib.Path(path, 'm3data.pickle')
with open(savepath, 'wb') as f:
    pickle.dump(m3data, f)

# Saving dictionary of BoBo Lander
savepath = pathlib.Path(path, 'bobodata.pickle')
with open(savepath, 'wb') as f:
    pickle.dump(bobodata, f)
    
# Saving dictionary of CTD single cast data
savepath = pathlib.Path(path, 'ctddata_sc.pickle')
with open(savepath, 'wb') as f:
    pickle.dump(ctddata_sc, f)
    
# Saving dictionary of CTD single cast data
savepath = pathlib.Path(path, 'ctddata_yoyo.pickle')
with open(savepath, 'wb') as f:
    pickle.dump(ctddata_yoyo, f)


# Saving dictionary of canyon data
savepath = pathlib.Path(path, 'canyondata.pickle')
with open(savepath, 'wb') as f:
    pickle.dump(canyondata, f)


# Saving dictionary of OBS-CTD data
savepath = pathlib.Path(path, 'obscastdata.pickle')
with open(savepath, 'wb') as f:
    pickle.dump(obscastdata, f)
    
# Saving dataframe of filter weight data
savepath = pathlib.Path(path, 'filterdata.pickle')
with open(savepath, 'wb') as f:
    pickle.dump(spm_df_full, f)
    
# Saving dataframe of CTD AQUADOP data
savepath = pathlib.Path(path, 'adcp_ctddata.pickle')
with open(savepath, 'wb') as f:
    pickle.dump(aqdp_ctddata, f)

