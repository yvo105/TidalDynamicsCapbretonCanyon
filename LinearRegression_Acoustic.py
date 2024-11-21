#%% #*## IMPORTS ###
import os
import pathlib
import pickle
import runpy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit

from py_functions.data_functions import find_path


def log_func(x, a, b, c):
    return a * np.log(x+c) + b

# %% #*## RELOAD DATA ###

# rerun of LoadData.py to update data
runpy.run_path("LoadData.py")

# %% #*## LOAD DATA FROM PICKLES ###

# path of save pickles
path = os.getcwd() + "/pickles/"
    
# CTD single casts
savepath = pathlib.Path(path, "ctddata_sc.pickle")
with open(savepath, "rb") as f:
    ctddata_sc = pickle.load(f)

# CTD yoyo casts
savepath = pathlib.Path(path, "ctddata_yoyo.pickle")
with open(savepath, "rb") as f:
    ctddata_yoyo = pickle.load(f)

# create ctddata dictionary with singlecasts and yoyos
ctddata = {'singlecasts': ctddata_sc,
           'yoyos': ctddata_yoyo}

# CTD JFE data
savepath = pathlib.Path(path, "adcp_ctddata.pickle")
with open(savepath, "rb") as f:
    aquadopp_ctddata = pickle.load(f)
    
# filter weight data
savepath = pathlib.Path(path, "filterdata.pickle")
with open(savepath, "rb") as f:
    filterdata = pickle.load(f)
    
# measuring interval data
savepath = pathlib.Path(path, "ctd_measintervals.pickle")
with open(savepath, "rb") as f:
    measintervaldata = pickle.load(f)
    
#%% #*## LINEAR REGRESSION - GENERAL SETUP ###

### ADJUSTABLES ###
depthlim = 2 # limit until the acoustic backscatter should be averaged, e.g. 2 for the first and second bin (1m)
min_minutes = 3  # aquadopp has a measuring frequency of every minute -> this parameter sets the minimum length of measuring interval for a data point to be considered in the linear regression, e.g. 3 means that only spm data points where turbidity averages of 3 minutes and over are available, are considered for the regression.

### DATFRAMES & DICTIONARIES ###

# result dataframe
spm_df = filterdata.copy()

# station indicators for each of the casts so they can be grouped together
station_dic = {
    '1_1': 1,
    '2_1': 2,
    '3_1': 3,
    '4_1': 4,
    '5_1': 5,
    '6_1': 6,
    '7_1': 7, 
    '8_1': 8,
    '8_2': 8,
    '14_1': 7, 
    '15_1': 6,
    '16_1': 1,
    '17_1': 2,
    '18_1': 3,
    '19_1': 4,
    '21_1': 5,
}

# list of cast numbers that correspond to yoyo casts  for dividing up between singlecast and yoyo data
yoyo_list = ['8_1', '8_2', '14_1', '15_1', '17_1', '18_1', '19_1', '21_1']

### GENERATING COLUMNS, DATA PREPARATION ###

## cast column 
castlist_full = []
for cast in spm_df['Station/Cast']:
    # retrieve cast number
    castno = cast[-4:]
    castno = castno.replace('-','_')
    castno = castno.lstrip('0')
    castlist_full.append(castno)
castlist = list(set(castlist_full))
spm_df['Station/Cast'] = castlist_full

## station column
spm_df['Station_Indicator'] = spm_df['Station/Cast'].map(station_dic)

## yoyo or singlecast column
spm_df['yoyo_vs_sc'] = np.nan
spm_df['yoyo_vs_sc'] = spm_df['yoyo_vs_sc'].astype('string')

# start of respective measuring interval
spm_df['meas_interval_start'] = np.nan
spm_df['meas_interval_start'] = spm_df['meas_interval_start'].astype(spm_df.index.dtype)

# end of respective measuring interval
spm_df['meas_interval_end'] = np.nan
spm_df['meas_interval_end'] = spm_df['meas_interval_end'].astype(spm_df.index.dtype)

# measuring interval length
spm_df['interval_length[s]'] = np.nan

# average depth during measurement
spm_df['depth'] = np.nan

# average beam amplitude during measurement interval
spm_df['BeamAmp[counts]'] = np.nan

# standard deviation of beam amplitude during measurement interval
spm_df['BeamAmp_std'] = np.nan

# measurement location relative to the bottom: 'bottom', '250mab' or '500mab'
spm_df['meas_location'] = np.nan
spm_df['meas_location'] = spm_df['meas_location'].astype('string')


### FILLING COLUMNS ###

for cast in castlist:
    
    # fill yoyo vs singlecast column -> if the cast id is found in the yoyolist, 'yoyo' is assigned, otherwise 'sc'
    if cast in yoyo_list:
        spm_df.loc[spm_df['Station/Cast'] == cast, 'yoyo_vs_sc'] = 'yoyo'
    else:
        spm_df.loc[spm_df['Station/Cast'] == cast, 'yoyo_vs_sc'] = 'sc'
    
    # save measuring interval data from measintervaldata dictionary in dataframe 
    spm_df.loc[spm_df['Station/Cast']==cast, 'meas_interval_start'] = measintervaldata['SPM_measurements'][cast]['start']
    spm_df.loc[spm_df['Station/Cast']==cast, 'meas_interval_end'] = measintervaldata['SPM_measurements'][cast]['end']

# identify unique values of station indicator column to get list of stations
stationlist = list(set(spm_df['Station_Indicator'].values))

# convert measuring intervals to datetime object
spm_df['meas_interval_end'] = pd.to_datetime(spm_df['meas_interval_end'])
spm_df['meas_interval_start'] = pd.to_datetime(spm_df['meas_interval_start'])

# calcualte measuring interval length
spm_df['interval_length[s]'] = (spm_df['meas_interval_end'] - spm_df['meas_interval_start']).dt.total_seconds()




#%% #*## LINEAR REGRESSION - REGRESSION CALCULATIONS ###


for station_no in stationlist:

    # find the cast id (e.g. 8_1) that correspond to the station
    castlist_atstation = [key for key,value in station_dic.items() if value==station_no]
    
    # station name in format e.g. Station8 -> corresponds to key in aquadopp_ctddata
    station_name = 'Station'+str(station_no)
    
    # check whether station is Station1, 6 or 8 because they need special treatment -> either two singlecasts (station1), no singlecast data from aquadopp (station6) or two yoyos available (station8)
    if station_name in ['Station1','Station6', 'Station8']:  
        
        if station_name in ['Station6', 'Station8']: # only yoyo cast available
            
            # retrieve beam amplitudes from aquadopp data
            beamamp = aquadopp_ctddata[station_name]['cast_2']['amp_beam'][0]
            
            # retrieve spm data from dataframe, second element of castlist_atstation necessary because only yoyo casts are available
            spm = spm_df['SPM'][spm_df['Station/Cast'] == castlist_atstation[1]]
            
            # retrieve depth information from ctddata
            path = find_path(ctddata, 'cast_id', castlist_atstation[1])
            
            # retrieve start and end times of intervals
            int_start = spm_df['meas_interval_start'][spm_df['Station/Cast'] == castlist_atstation[1]].values
            int_end = spm_df['meas_interval_end'][spm_df['Station/Cast'] == castlist_atstation[1]].values
            
            
            if station_name in ['Station8']:
                
                # initiate lists
                beam_list_cast_2 = []
                depthlist_2 = []
                stdlist_2 = []
                
                # retrieve depth information from ctddata
                path_2 = find_path(ctddata, 'cast_id', castlist_atstation[0])
                depthkey_2 = path_2[1].replace('_general','_despiked')
                depth_2 = ctddata[path[0]][depthkey_2]['depth_atlatitude']
                
                # retrieve spm information
                spm_2 = spm_df['SPM'][spm_df['Station/Cast'] == castlist_atstation[0]]
                
                # retrieve start and end times of intervals
                int_start_2 = spm_df['meas_interval_start'][spm_df['Station/Cast'] == castlist_atstation[0]].values
                int_end_2 = spm_df['meas_interval_end'][spm_df['Station/Cast'] == castlist_atstation[0]].values
                
                
                
                
        else:  # corresponds to Station1 where only one singlecast is available
            
            # retrieve beam amplitudes
            beamamp = aquadopp_ctddata[station_name]['cast_1']['amp_beam'][0]
            
            # retrieve spm data 
            spm = spm_df['SPM'][spm_df['Station/Cast'] == castlist_atstation[0]]
            
            # retrieve depth information from ctddata
            path = find_path(ctddata, 'cast_id', castlist_atstation[0])
            depthkey = path[1].replace('_general','_despiked')
            depth = ctddata[path[0]][depthkey]['depth_atlatitude']
            
            # retrieve start and end times of intervals
            int_start = spm_df['meas_interval_start'][spm_df['Station/Cast'] == castlist_atstation[0]].values
            int_end = spm_df['meas_interval_end'][spm_df['Station/Cast'] == castlist_atstation[0]].values
        
        # initiate lists
        
        beam_list_cast = []  # list to save averaged beam amplitudes in
        depthlist = []  # list for mean depth during interval
        stdlist = []  # list for standard deviation
        
        
        # iterate over start & end times
        for start, end in zip(int_start, int_end):
            
            # calculate mean beam amplitude within measurement interval and up to maximum bin index of depthlim
            beam_mean = np.nanmean(beamamp.loc[start:end, :depthlim].mean(axis=1))
            
            # calculate standard deviation over measurement interval and up to maximum bin index of depthlim
            beam_std = np.std(beamamp.loc[start:end, :depthlim].mean(axis=1))
            
            # calculate mean depth during measurement interval
            mean_depth = depth.loc[start:end].mean()
            
            # append variables to respective lists
            beam_list_cast.append(beam_mean)
            depthlist.append(mean_depth)
            stdlist.append(beam_std)
        
        # same procedure as above but if station is station 8 where only yoyo data is available    
        if station_name in ['Station8']:
            for start_2, end_2 in zip(int_start_2, int_end_2):
                
                # calculate mean beam amplitude within measurement interval and up to maximum bin index of depthlim
                beam_mean = np.nanmean(beamamp.loc[start_2:end_2, :depthlim].mean(axis=1))
                
                # calculate standard deviation over measurement interval and up to maximum bin index of depthlim
                beam_std = np.std(beamamp.loc[start_2:end_2, :depthlim].mean(axis=1))
                
                # calculate mean depth during measurement interval
                mean_depth = depth_2.loc[start_2:end_2].mean()
                
                # append variables to respective lists
                beam_list_cast_2.append(beam_mean)
                depthlist_2.append(mean_depth)
                stdlist_2.append(beam_std)
                
        
        ### SAVE CALCULATED VARIABLES IN RESPECTIVE COLUMNS ###
           
        if station_name in ['Station6', 'Station8']: # only yoyo data available
            
            # add beam amplitude list at respective location in dataframe
            spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[1], 'BeamAmp[counts]'] = beam_list_cast
            
            # add depth to dataframe
            spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[1], 'depth'] = depthlist
            
            # add std to dataframe
            spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[1], 'std'] = stdlist
            
            if station_name in ['Station8']:
                # add turbidity list at respective location in dataframe
                spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[0], 'BeamAmp[counts]'] = beam_list_cast_2
                
                # add depth to dataframe
                spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[0], 'depth'] = depthlist_2
                
                # add std to dataframe
                spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[0], 'std'] = stdlist_2
            
        else: # corresponds to Station1 where only one singlecast is available
            
            # add beam amplitude list at respective location in dataframe
            spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[0], 'BeamAmp[counts]'] = beam_list_cast
            
            # add depth to dataframe
            spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[0], 'depth'] = depthlist
            
            # add std to dataframe
            spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[0], 'std'] = stdlist
        
    else:
        
        # retrieve beam amplitudes
        beamamp_1 = aquadopp_ctddata[station_name]['cast_1']['amp_beam'][0]  #singlecast
        beamamp_2 = aquadopp_ctddata[station_name]['cast_2']['amp_beam'][0]  #yoyo
        
        # retrieve spm data from dataframe
        spm_1 = spm_df.loc[spm_df['Station/Cast']== castlist_atstation[0]] # singlecast
        spm_2 = spm_df['SPM'][spm_df['Station/Cast'] == castlist_atstation[1]] # yoyo
        
        # retrieve depth information for the first cast
        path_1 = find_path(ctddata, 'cast_id', castlist_atstation[0])
        depthkey_1 = path_1[1].replace('_general','_despiked')
        depth_1 = ctddata[path_1[0]][depthkey_1]['depth_atlatitude']
        
        # retrieve depth information for the second cast
        path_2 = find_path(ctddata, 'cast_id', castlist_atstation[1])
        depthkey_2 = path_2[1].replace('_general','_despiked')
        depth_2 = ctddata[path_2[0]][depthkey_2]['depth_atlatitude']
        
        # list to save averaged beam amplitudes in
        beam_list_cast_1 = []
        beam_list_cast_2 = []
        
        # list for mean depth during interval
        depthlist_1 = []
        depthlist_2 = []
        
        # list for standard deviation
        stdlist_1 = []
        stdlist_2 = []
        
        # retrieve start and end times of intervals for first cast
        int_start_1 = spm_df['meas_interval_start'][spm_df['Station/Cast'] == castlist_atstation[0]].values
        int_end_1 = spm_df['meas_interval_end'][spm_df['Station/Cast'] == castlist_atstation[0]].values
        
        # retrieve stard and end times for second cast
        int_start_2 = spm_df['meas_interval_start'][spm_df['Station/Cast'] == castlist_atstation[1]].values
        int_end_2 = spm_df['meas_interval_end'][spm_df['Station/Cast'] == castlist_atstation[1]].values
        
        # iterate over start & end times of the first cast
        for start_1, end_1 in zip(int_start_1, int_end_1):
            
            # calculate mean beam amplitude within measurement interval and up to maximum bin index of depthlim
            beam_mean_1 = np.nanmean(beamamp_1.loc[start_1:end_1, :depthlim].mean(axis=1))
            
            # calculate standard deviation over measurement interval and up to maximum bin index of depthlim
            beam_std_1 = np.std(beamamp_1.loc[start_1:end_1,:depthlim].mean(axis=1))
            
            # calculate mean depth during measurement interval
            mean_depth_1 = depth_1.loc[start_1:end_1].mean()
            
            # append variables to respective lists
            beam_list_cast_1.append(beam_mean_1)
            depthlist_1.append(mean_depth_1)
            stdlist_1.append(beam_std_1)
            
        # iterate over start and end times of the second cast   
        for start_2, end_2 in zip(int_start_2, int_end_2):
            
            # calculate mean beam amplitude within measurement interval and up to maximum bin index of depthlim
            beam_mean_2 = np.nanmean(beamamp_2.loc[start_2:end_2, :depthlim].mean(axis=1))
            
            # calculate standard deviation over measurement interval and up to maximum bin index of depthlim
            beam_std_2 = np.std(beamamp_2.loc[start_2:end_2, :depthlim].mean(axis=1))
            
            # calculate mean depth during measurement interval
            mean_depth_2 = depth_2.loc[start_2:end_2].mean()
            
            # append variables to respective lists
            beam_list_cast_2.append(beam_mean_2)
            depthlist_2.append(mean_depth_2)
            stdlist_2.append(beam_std_2)
        

        ### ADD CALCULATED VARIABLES AT RESPECTIVE POSITIONS IN DATAFRAME ###

        # add beam amplitude list at respective location in dataframe
        spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[0], 'BeamAmp[counts]'] = beam_list_cast_1
        spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[1], 'BeamAmp[counts]'] = beam_list_cast_2
        
        # add depth to dataframe
        spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[0], 'depth'] = depthlist_1
        spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[1], 'depth'] = depthlist_2
        
        # add standard deviation to dataframe
        spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[0], 'BeamAmp_std'] = stdlist_1
        spm_df.loc[spm_df['Station/Cast'] == castlist_atstation[1], 'BeamAmp_std'] = stdlist_2
    
# select spm data
X = spm_df.loc[
        (spm_df['interval_length[s]'] > (min_minutes*60))
        & (spm_df['BeamAmp[counts]'].notna()),
        'SPM'].values



# select beam amplitude data
y = spm_df.loc[
    (spm_df['interval_length[s]'] > (min_minutes*60))
    & (spm_df['BeamAmp[counts]'].notna()),
    'BeamAmp[counts]'].values

# logarithmic curve fit
popt, pcov = curve_fit(log_func, X, y)
a, b, c = popt
x_data = np.linspace(0,1.5,100)
y_fit = log_func(x_data, a, b, c)

acoustic_logregress = {
    'a': a,
    'b': b,
    'c': c
}


### PLOTTING ###

# initialize plot
fig, ax = plt.subplots()

# plot scatter of observations
ax.errorbar(
    spm_df.loc[(spm_df['interval_length[s]'] > (min_minutes*60)) & (spm_df['BeamAmp[counts]'].notna()), 'SPM'],
    spm_df.loc[(spm_df['interval_length[s]'] > (min_minutes*60)) & (spm_df['BeamAmp[counts]'].notna()), 'BeamAmp[counts]'],
    yerr=spm_df.loc[(spm_df['interval_length[s]'] > (min_minutes*60)) & (spm_df['BeamAmp[counts]'].notna()), 'BeamAmp_std'],
    fmt='o',
    markersize=3,
    ecolor='black',
    capsize=2, 
    markerfacecolor='blue',
    markeredgecolor='blue')  # star marker

# plot logarithmic curve fit
ax.plot(
    x_data,
    y_fit,
    label = f'OLS: y = {round(a,3)} * log(x+{round(c,3)}) + {round(b,3)}' ,
    color='blue'
)

# set plot parameters
ax.set(
    xlabel='SPM [mg/L]',
    ylabel='Beam Amplitude [counts]',
)
ax.grid()
ax.legend() 


#%% #*## LINEAR REGRESSION - DEPENDING ON HEIGHT ABOVE BOTTOM ###

#TODO: Idea is to classify the data into the two locations where they were taken from (bottom, +250m), for this only the yoyodata is used.


# fill the "meas_location" column based on the depth column
for cast in castlist:
    
    # retrieve depth value corresponding to cast
    depth = spm_df.loc[spm_df['Station/Cast'] == cast, 'depth']
    
    # retrieve depths at bottom -> maximum depth up to 50m above bottom. Intervals will still correspond to bottom only because spm measurements where only at bottom, 250mab or 500mab anyways
    depth_bot = depth[depth > (depth.max()-50)]
    
    # retrieve depths at 250mab height -> interval 100m above bottom to 400mab
    depth_250 = depth[((depth.max()-100) > depth) & (depth > (depth.max()-400))]
    
    # retrieve depths at 500mab height -> interval 400mab to 600mab
    depth_500 = depth[((depth.max() -400) > depth) & (depth > (depth.max()-600))]
    
    # assign 'bottom', '250mab' and '500mab' to each subset
    spm_df.loc[(spm_df['Station/Cast']==cast) & spm_df['depth'].isin(depth_bot),'meas_location'] = 'bottom' 
    spm_df.loc[(spm_df['Station/Cast']==cast) & spm_df['depth'].isin(depth_250),'meas_location'] = '250mab'
    spm_df.loc[(spm_df['Station/Cast']==cast) & spm_df['depth'].isin(depth_500), 'meas_location'] = '500mab' 


### LINEAR REGRESSION MODEL - BOTTOM ###

# retrieve spm data
X_bot = spm_df.loc[
        (spm_df['interval_length[s]'] > (min_minutes*60))
        & (spm_df['BeamAmp[counts]'].notna())
        & (spm_df['meas_location'] == 'bottom'),
        'SPM'].values
    

# retrieve beam amplitude data
y_bot = spm_df.loc[
    (spm_df['interval_length[s]'] > (min_minutes*60))
    & (spm_df['BeamAmp[counts]'].notna())
    & (spm_df['meas_location'] == 'bottom'),
    'BeamAmp[counts]'].values

# logarithmic curve fit
popt_bot, pcov_bot = curve_fit(log_func, X_bot, y_bot)
a_bot, b_bot, c_bot = popt_bot
x_data = np.linspace(0,1.5,1000)
y_fit_bot = log_func(x_data, a_bot, b_bot, c_bot)


### LINEAR REGRESSION MODEL - 250MAB ###

# retrieve spm data
X_250 = spm_df.loc[
        (spm_df['interval_length[s]'] > (min_minutes*60))
        & (spm_df['BeamAmp[counts]'].notna())
        & (spm_df['meas_location'] == '250mab'),
        'SPM'].values
    

# retrieve beam amplitude data
y_250 = spm_df.loc[
    (spm_df['interval_length[s]'] > (min_minutes*60))
    & (spm_df['BeamAmp[counts]'].notna())
    & (spm_df['meas_location'] == '250mab'),
    'BeamAmp[counts]'].values

popt_250, pcov_250 = curve_fit(log_func, X_250, y_250)
a_250, b_250, c_250 = popt_250
y_fit_250 = log_func(x_data, a_250, b_250, c_250)


### PLOTTING ###

# initialize plot
fig, ax = plt.subplots()

# plot observations, bottom
ax.errorbar(
    spm_df.loc[(spm_df['interval_length[s]'] > (min_minutes*60)) & (spm_df['BeamAmp[counts]'].notna()) & (spm_df['meas_location'] == 'bottom'), 'SPM'],
    spm_df.loc[(spm_df['interval_length[s]'] > (min_minutes*60)) & (spm_df['BeamAmp[counts]'].notna()) & (spm_df['meas_location'] == 'bottom'), 'BeamAmp[counts]'],
    yerr=spm_df.loc[(spm_df['interval_length[s]'] > (min_minutes*60)) & (spm_df['BeamAmp[counts]'].notna()) & (spm_df['meas_location'] == 'bottom'), 'BeamAmp_std'],
    fmt='o',
    markersize=3,
    ecolor='black',
    capsize=2, 
    markerfacecolor='blue',
    markeredgecolor='blue') 

# plot logarithmic curve fit, bottom
ax.plot(
    x_data,
    y_fit_bot,
    label = f'OLS: y = {round(a_bot,3)} * log(x + {round(c_bot,3)}) + {round(b_bot,3)}' ,
    color='blue'
)

# plot observations, @ 250mab
ax.errorbar(
    spm_df.loc[(spm_df['interval_length[s]'] > (min_minutes*60)) & (spm_df['BeamAmp[counts]'].notna()) & (spm_df['meas_location'] == '250mab'), 'SPM'],
    spm_df.loc[(spm_df['interval_length[s]'] > (min_minutes*60)) & (spm_df['BeamAmp[counts]'].notna()) & (spm_df['meas_location'] == '250mab'), 'BeamAmp[counts]'],
    yerr=spm_df.loc[(spm_df['interval_length[s]'] > (min_minutes*60)) & (spm_df['BeamAmp[counts]'].notna()) & (spm_df['meas_location'] == '250mab'), 'BeamAmp_std'],
    fmt='o',
    markersize=3,
    ecolor='black',
    capsize=2, 
    markerfacecolor='orange',
    markeredgecolor='orange') 

# plot logarithmic curve fit, @ 250mab
ax.plot(
    x_data,
    y_fit_250,
    label = f'OLS: y = {round(a_250,3)} * log(x + {round(c_250,3)}) + {round(b_250,3)}' ,
    color='orange'
)

# set plot parameters
ax.set(
    xlabel='SPM [mg/L]',
    ylabel='BeamAmp[counts]',
)
ax.grid()
ax.legend() 

# %% #*## LINEAR REGRESSION - EACH CAST SEPERATELY - CHANGE ALONG THALWEG ###


# iterate over each station
for station in spm_df['Station_Indicator'].unique():
    
    # retrieve spm data
    spm = spm_df.loc[
        (spm_df['Station_Indicator'] == station)
        & (spm_df['BeamAmp[counts]'].notna()),
        'SPM'].values
    
        # retrieve beam amplitude data
    beamamp = spm_df.loc[
        (spm_df['Station_Indicator'] == station)
        & (spm_df['BeamAmp[counts]'].notna()),
        'BeamAmp[counts]'].values
    
    # retrieve standard deviation data for respective station
    std = spm_df.loc[
        (spm_df['Station_Indicator'] == station)
        & (spm_df['BeamAmp[counts]'].notna()),
        'std']
    
    if len(spm) > 3:
        
        print(len(beamamp))
    

        
        
        popt_temp, pcov_temp = curve_fit(log_func, spm, beamamp)
        a_temp, b_temp, c_temp = popt_temp
        y_fit_temp = log_func(x_data, a_temp, b_temp, c_temp)
        
        ### LINEAR REGRESSION MODEL ###
        
        # # assign spm and beam amplitude data
        # X_temp = sm.add_constant(spm)
        # y_temp = beamamp
        
        # # initialize and run linear regression model
        # model_temp = sm.OLS(y_temp,X_temp)
        # results_temp = model_temp.fit()
        
        # # retrieve important parameters
        # m_temp = results_temp.params['SPM']
        # c_temp = results_temp.params['const']
        # R2_temp = results_temp.rsquared
        
        ### PLOTTING ###
        
        # initialize plot
        fig, ax = plt.subplots()
        
        # plot observations, @ 250mab
        ax.errorbar(
            spm,
            beamamp,
            yerr=std,
            fmt='o',
            markersize=3,
            ecolor='black',
            capsize=2, 
            markerfacecolor='black',
            markeredgecolor='black')  # star marker
        
        # plot linear regression line
        ax.plot(
            x_data,
            y_fit_temp,
            c='orange', 
            label = f'OLS: y = {round(a_temp,3)} * log(x + {round(c_temp,3)}) + {round(b_temp,3)}' , 
            linestyle='-.',
            lw=1)
        
        # set plot parameters
        ax.set(
            xlabel='SPM [mg/L]',
            ylabel='Beam Amplitude [counts]',
            xlim=(0,1.5),
            ylim=(60,130), 
            title=station
        )
        ax.grid()
        ax.legend() 

# %% #*### SAVE LINEAR REGRESSION RESULTS ####

path = os.getcwd() + '/pickles/'  # path of current working directory plus pickle folder

# Saving dictionary of linear regression
savepath = pathlib.Path(path, 'Acoustic_logregression.pickle')
with open(savepath, 'wb') as f:
    pickle.dump(acoustic_logregress, f)