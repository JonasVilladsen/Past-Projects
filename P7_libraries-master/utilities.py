# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:26:34 2018

@author: Martin Kamp Dalgaard & Tobias Kallehauge
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

root = os.getcwd()[:-24]

def return_to_root(n_max = 10): # Return to the root directory in svn from a subfolder
    root = os.getcwd()
    for i in range(n_max):
        if root[-3:] != "svn":
            os.chdir("..")
            root = os.getcwd()
        else:
            break
    if root[-3:] != "svn":
        raise(OSError("\"svn\" not found using recursion.\nCheck if script is in a subfolder under \"svn\".\nIf this is the case set n_max higher."))
    root += "/"
    return root

def choose_days_from_timerange(timerange,days):
    """
    Sort out dates not in datelist and return sorted timerange.
    
    Input:
        timerange: pandas date_range
        days: pandas date_range with frequency at least "D" or greater.
        
    Return:
        Return pandas date_range
        
    ------
    Example
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,2,1)
    range1 = pd.date_range(t0,t1,freq = '15min')
    range2 = pd.date_range(t0,t1,freq = '2D')
    range_sorted = choose_days_from_timerange(range1,range2)
    """
    M = len(timerange)
    new_rng = []
    j = 0
    for i in range(M):
        if timerange[i].date() in days:
            new_rng.append(timerange[i])
            j += 1
    return pd.DatetimeIndex(new_rng)

def dates_from_DF(df):
    """
    Make list of unique dates in dataframe.
    """
    
    idx = df.index
    dates = np.array([i.date() for i in idx])
    dates = np.unique(dates)
    return(dates)


def load_convert_muni_to_grid():
    root = return_to_root()
    stem_path = "data/stem_data/"
    conv_path = stem_path + "Kommune_GridNr_new.xlsx"
    conv_sheet = pd.read_excel(root + conv_path)
    conv_sheet.set_index('Kommune_Nr',inplace= True) #Let kommune_nr be index
    return(conv_sheet)

def muni_list_to_grid_list(muni_list = 'all'):
    """"
    Converts list of muni's to grid list. Input must be of array type
    """
    conv_sheet = load_convert_muni_to_grid()
    if not set(muni_list).issubset(set(conv_sheet.index)) and muni_list != 'all':
        raise(ValueError("Element in muni_list is invalid municipilaty number."))

    if muni_list == "all":
        muni_list = conv_sheet.index
    conv_sheet = conv_sheet.loc[muni_list] #get grid points for chosen munis    
    
    N = np.shape(conv_sheet)[0] #elements in dataframe
    grid_list = np.array(conv_sheet[['GNr1','GNr2','GNr3']]).reshape(3*N)
    grid_list = mk_unique_no_nan_int(grid_list)
    return(list(grid_list),conv_sheet)

def zeropad_hourstring(string):
    if len(string) == 1:
        return "0" + string
    else:
        return string

def mk_unique_no_nan_int(array,dtype = "int16"):
    nan_idx = np.isnan(array) #where Truth/False if value is nan
    idx = np.argwhere(nan_idx == True)
    array_new = np.delete(array,idx) #array with no nan
    array_new = np.array(np.unique(array_new),dtype = "int16") #remove repeats
    return(array_new)
    
def mk_list_dic(keys,len_list):
    """
    Make dictionary for given keys with numpy zero arrays where len_list is a
    tuple with the length of the array for each key
    """
    N = len(len_list)
    return({keys[i] : np.zeros(len_list[i]) for i in range(N)})

def max_list_len_dic(dic):
    """
    Finds length of dictionary with containers and return the length of the 
    largest one
    """
    return(max([len(i) for i in dic.values()]))
    
def mk_coef_matrix_from_dic(dic,flip = True):
    """
    Creates numpy array with coefficients made from dictionary. Used in 
    spatio temporal model. Flip argument flips the arrays
    """
    keys = list(dic.keys())
    M = len(keys) #number of municaplites
    dic_N = max_list_len_dic(dic)
    
    dic_arr = np.zeros((dic_N,M)) 
    for nr in range(M): #loop over each muncipality
        for lag in range(len(dic[keys[nr]])):
            dic_arr[:,nr][lag] = dic[keys[nr]][lag]
    if flip:
        dic_arr = np.flipud(dic_arr)
    return(dic_arr)

def d_time_to_quaters(t):
    """
    Converts a datetime object into total number of quarters. Can only handle
    whone number of minutes. 
    """
    return(4*t.hour + int(t.minute/15))

def _handle_timerange(t_start,t_end,hours,sub_h_freq,sub_D_freq):
    """
    Handles various elements of the timerange specifications in order to import
    data from matlabfiles. 
    """
    #subsample_hours
    t_end = t_end.replace(hour = 23 ,minute = 45)
    if sub_h_freq == 'all':
        sub_h_freq = "15min"
    
    #resample days
    if sub_D_freq == 'all':
        sub_D_freq = "D"
    
    rng = pd.date_range(t_start,t_end,freq = sub_h_freq) #daterange for forecast
    h_int = rng.freq.delta.components.hours #get hours as int
    sub_h_nr = int(h_int/0.25) #how many samples skibbed hourly
    if sub_h_nr == 0: #sub_h_freq = 'all' this value is set to 1 so no value are skibbed
        sub_h_nr = 1
    
    
    if 24%sub_h_nr != 0:
        raise(ValueError("Freqency in hours must be a multible of 24, e.g. 2,6,12"))
    
    if hours == "all":
        hours = ("00:00","23:45")

    day_rng = pd.date_range(t_start,t_end,freq = sub_D_freq) #Timerange with only the dates
    rng = choose_days_from_timerange(rng,day_rng)
    
    rng = rng[rng.indexer_between_time(hours[0],hours[1])] #remove unwanted hours
    spd = int(len(rng)/len(day_rng)) #samples pr. day
    s_day0 = 4*rng[0].hour + int(rng[0].minute/15) #how many samples skibbed at beginning of day. Stupid formula but it works
    s_day1 = -(4*(24 - rng[-1].hour) - 1 - int(rng[-1].minute/15)) #how many samples skibbed at end of day. Also kinda stupid
    
    #Avoid empty matrix when indexing
    if s_day0 == 0: 
        s_day0 = None
    if s_day1 == 0:
        s_day1 = None
        
    return(rng,day_rng,sub_h_nr,spd,s_day0,s_day1)

def Timedelta(d_time):
    """
    Converts datetime.time into pd.Timedelta
    """
    return(pd.Timedelta(hours = d_time.hour,minutes = d_time.minute))
    
def round15(time,method):
    """
    Rounds datetime to nearest 15minutes (up or down)
    
    Parameters
    ----------
    time : datetime.time
        Time to be rounded
    method:
        Either "ceil" or "floor" or "rnd".
        Select ceil/floor to round up/down base 15 or rnd for normal round
    
    Returns
    -------
    Rounded time as datetime.time
    
    """
    from datetime import time as d_time
    h = time.hour
    min_rnd = myround(time.minute,method = method)
    if min_rnd == 60:
        h += 1
        min_rnd = 0
    return(d_time(h,min_rnd))
        
def myround(x, base = 15,method = 'ceil'):
    if method == 'ceil':
        rnd = np.ceil
    elif method == 'floor':
        rnd = np.floor
    else:
        rnd = round
    return int(base * rnd(float(x)/base))

def conv_str_time_df_to_d_time(df,encoding = '%H:%M:%S',copy = False):
    """
    Converts dataframe with string values as time to datetime objects
    """
    if copy:
        df_f = pd.DataFrame(df,copy = True)
    else:
        df_f = df
    for i in df_f.columns:
        for j in df_f.index:
            time = datetime.strptime(df.loc[j,i],'%H:%M:%S').time()
            df_f.loc[j,i] = time
    return(df_f)
    
def conv_and_round15_df(df):
    """
    Given df with string entries in columns 't0' and 't1', convert to 
    datetime.time and round t0/t1 up/down to the nearest 15 minutes. 
    """
    conv_str_time_df_to_d_time(df)
    for i in df.index:
        df['t0'].loc[i] = round15(df['t0'].loc[i],method = 'floor')
        df['t1'].loc[i] = round15(df['t1'].loc[i],method = 'ceil')
    

def nonzero_df_replace(df,df_replace):
    """
    Replace values in df with values in df_replace where df < 0
    """
    idx = df < 0
    df_replace.index = df.index; df_replace.columns = df.columns
    df[idx] = df_replace[idx]
    return(df)
    
def add_hour(time,add):
    """
    Add is how much time to add in minutes as integer
    """
    t = pd.Timestamp(2017,1,1,time.hour,time.minute)
    t_add = t + pd.Timedelta(hours = add.hour,minutes = add.minute)
    next_day = False
    if t_add.day != t.day:
        next_day = True
    return t_add.time(), next_day


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
    
def merge_multiindex(df):
    """
    Merge multiindex with two rows into one row with the last row string first
    """
    first = df.columns.get_level_values(0)
    last = df.columns.get_level_values(1)
    colums = []
    for i in range(len(first)):
        colums.append(last[i] + str(first[i]))
    df.columns = colums


def get_tst_dates(mode = 'ldm',dpm = 5):
    """
    Get data list with which dates should be test days in 2017
    
    Parameters
    ----------
    mode : str
        ldm: last days month - no other modes implemented
    dpm : days per month, defaults to 5
    """
    dates = pd.date_range("1/1/2017","31/12/2017",freq = "M")
    ldm = dates[0]
    tst_days = pd.date_range(ldm - pd.Timedelta(days = dpm - 1),ldm)
    for i in range(1,12):
        ldm = dates[i]
        tst_days = tst_days.append(pd.date_range(ldm -\
                                                 pd.Timedelta(days = dpm - 1),ldm))
    return(tst_days)
    

def rm_dates_from_df(df,dates):
    """
    Remove rows wiht dates from pandas dataframe 
    """
    mask = ~np.in1d(df.index.date, pd.to_datetime(dates).date)
    df = df.loc[mask, :]
    return(df)