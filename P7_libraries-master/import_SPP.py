# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:14:31 2018

@author: Tobias
"""

import pandas as pd
import os
import scipy.io as sio
import numpy as np
from utilities import return_to_root, _handle_timerange
import sanity_checks

#root = os.getcwd()[:-24] #Root for this folder for tests 
class SPP:
    """
    Structure to hande solar panel production from different municipilaties
    
    Input:
        SPP as pandas dataframe
        instp as pandas dataframe with date as index and muninumbers as columnnames
        h_freq is hourly frequency
        D_freq is dayly frequency
    
    """
    def __init__(self,SPPinfo,muninames,instp = None,h_freq = "15min",D_freq = "D"):
        self.SPP = SPPinfo
        self.timerange = SPPinfo.index
        self.muninr = SPPinfo.columns
        self.muninames = muninames
        self.instp = instp
        self.hours = (self.timerange[0].time(), self.timerange[-1].time())
        self.h_freq = h_freq
        self.D_freq = D_freq
    
    def get_minicipilaties(self):
        s = "SPP have information about the following %d municipaties:\n" \
        %(len(self.muninr))
        for i,j in list(zip(self.muninr,self.muninames)):
            s+= "- %i: %s\n" %(i,j)
        print(s)
    
    def __str__(self):
        """
        Prints out nicely formated string with relevant information about the SPP 
        """
        s = "SPP for the dates:\n%s to %s in the timerange %s to %s every %s\'s and every %s\'s" \
        %(str(self.timerange[0].date()),str(self.timerange[-1].date()),\
          str(self.hours[0]),str(self.hours[1]),self.h_freq,self.D_freq) 
        s += "\n\nSPP covers %d municipilaties" %len(self.muninr)
        return(s)

def import_SPP(t_start, t_end, muni_list="all", hours="all",
               sub_h_freq="all", sub_D_freq='all', mode='all'):
    """
    Import serveral SPP files into a continius information with information every 15 minutes possibly with some hours takend out. Imports into the SPP class. 
    
    Input:
         root: root to main svn folder with "\\" in the end. This will vary depending from where you run your script.
         t_start/t_end: Start/end time for your forecast as pandas timestamp. Select this value between 2017/01/01  and 2017/12/31. Only whole dates are allowed, no hours or minutes can be speficied. 
         

    Optional input:
        munilist: Speficify which muncipilaties you want date from using the municipilaty numbers. If left it will import data from all munipicilaties, else give a list/tuple/numpy array with the numbers.
        hours: Spefify which hours of the day you want to import. Example: hours = ("12:00","18:45") will give you the SPP in that timerange for the specified days. Should be a list/tuple wiht elements of type "hh,mm" as strings
        sub_h_freq (Subsample hours frequency): Define how to subsample on an hourly basis. E.g. sub_h_freq = "2H" will return SPP for every 2 hours. 24 should be devisible by this value
        sub_D_freq (Subsample days frequency): Define how to subsample on an dayly basis. E.g. sub_D_freq = "5D" will return SPP for every 5 days.

    Returns: SPP and instaled effect from speficied timerange as a SPP object. See ?SPP for more info 
    
    Examples:
    #Initialise timestamp with arguments specified. 
    t0 = pd.Timestamp(year = 2017,month = 1), print(t0)
    >>> 2017-01-01 00:00:00
    
    #Initialise timestamp without arguments specified
    t0 = pd.Timestamp(2017,1,1); print(t0)
    >>> 2017-01-01 00:00:00

    #Import SPP in the timerange 1/1-2017 to 1/2-2017 for all minicipilaties the entire day
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,2,1)
    sp_imp = import_SPP(root,t0,t1); print(sp_imp)
    >>> SPP for the dates:
    >>> 2017-01-01 to 2017-02-01 in the timerange 00:00:00 to 23:45:00 every 15min's and every D's
    >>>
    >>> SPP covers 98 municipilaties
    
    #You get dataframes with SPP and installed effect like this
    sp_dataframe = sp_imp.SPP, instP_datafam = sp_imp.instp
    
    #You can view SPP or installed effect it as an excel file like:
    sp_imp.SPP.to_csv("solarpanelproduction.csv")
    
    #Even as an html file to view in your browser which gives a good overview
    sp_imp.SPP.to_html("solarpanelproduction.html") 
    
    #Import SPP in specific hours
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10,12)
    hours = ("05:00","22:00")
    sp_imp = import_SPP(root,t0,t1,hours = hours); print(sp_imp)
    >>> SPP for the dates:
    >>> 2017-01-01 to 2017-02-01 in the timerange 05:00:00 to 22:00:00 every 15min's and every D's
    >>>
    >>> SPP covers 98 municipilaties
    
    #Import SPP by subsampling hours and days
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10,12)
    sub_hours = "2H"; sub_days = "5D"
    sp_imp = import_SPP(root,t0,t1,sub_h_freq = sub_hours,sub_D_freq = sub_days); print(sp_imp)
    >>> SPP for the dates:
    >>> 2017-01-01 to 2017-02-01 in the timerange 00:00:00 to 22:00:00 every 2H's and every 5D's
    >>>
    >>> SPP covers 98 municipilaties
    
    #Chose which municipilaties you want to import data from with the muni_list argument. Lets try with Aalborg, Hjørring and Jammerbugt municipilaty which have the numbers 849,851 and 860
    muni_list = [849,851,860]
    sp_imp = import_SPP(root,t0,t1,munilist = munilist); print(sp_imp)
    >>> SPP for the dates:
    >>> 2017-01-01 to 2017-02-01 in the timerange 05:00:00 to 22:00:00 every 15min's and every D's
    >>>
    >>> SPP covers 3 municipilaties
    
    #You can list the municipilaties you just importet with the .get_minicipilaties method
    sp_imp.get_minicipilaties()
    >>> SPP have information about the following 3 municipaties:
    >>> - 849: Jammerbugt
    >>> - 851: Aalborg
    >>> - 860: Hjørring
    """
    root = return_to_root()
    
    #Sanitycheck for different input    
    if "data" not in os.listdir(root): #maybe change folder name
        raise(OSError("Root is noot the svn shared folder"))
    if mode not in ['all', 'remove_outliers']:
        raise(ValueError('Mode not valid'))
    sanity_checks.check_timestamps(t_start, t_end)
    
    sanity_checks.check_hours(hours)
    
    sanity_checks.check_subs_args(sub_h_freq, sub_D_freq, mode="SPP")
        
    #fetch stem stata, also used later
    stem_path = "data/stem_data/muni_data_new.xlsx" #load muni numbers from file - maybe cange folder
    stem = pd.read_excel(root + stem_path)
    muninr = stem['KOMMUNENR']
    
    if not set(muni_list).issubset(set(muninr)) and muni_list != 'all':
        raise(ValueError("muni_list contains a muninumber that is not valid."))
    
    rng, day_rng, sub_h_nr, spd, s_day0, s_day1 = \
    _handle_timerange(t_start, t_end, hours, sub_h_freq, sub_D_freq)
    
    #List with indicies of chosen municipilaties
    if muni_list == "all":
        muni_index = range(len(muninr)) #All indicies - 98
        muni_list = muninr
    else:
        #Ensure list is sorted numerically in order to get right indicies
        muni_list.sort()
        #List with indicies of chosen municipilaties
        muni_index = np.in1d(muninr, muni_list).nonzero()[0] 
    
    
    #Create data structures
    #N is total number of samples, M is number of minicipilaties and K is number of days
    N, M, K = len(rng), len(muni_index), len(day_rng)
    SSP_dat = np.zeros((N,M))
    instP_dat = np.zeros((K,M))
    folder_path = "data/"
    idx_count = 0
    for t in day_rng: #Runs thorugh every day in timerange (15min*24*hours = 96)
        data_path = "%d/%d/%d/" %(t.year, t.month, t.day) #Specific day and hour
        SSP_dat[idx_count*spd:idx_count*spd + spd] = \
        sio.loadmat(root + folder_path + data_path + 'SPP_all.mat')\
        ['X'][s_day0:s_day1][:,muni_index][::sub_h_nr]
        # ['X'] pick out the data matrix information
        # [s_day0:s_day1] picks out the relevant times
        # [:,muni_index] picks out the relevant munipicilaties
        # [::sub_h_nr] subsamples in hours
        

        instP_dat[idx_count] = np.take(sio.loadmat(root + folder_path + \
                 data_path + "InstEff_all.mat")['InstEff'].T[0], muni_index)
        #np.take picks out the relevent inficies using muni_index list
        
        idx_count += 1
        
    #Convert to dataframe, overwrites matricies
    SSP_df = pd.DataFrame(SSP_dat, index=rng, columns=muni_list)
    instP_df = pd.DataFrame(instP_dat, index=day_rng, columns=muni_list)
    
    #Set names in dataframe
    SSP_df.columns.name = 'KOMMUNENR'
    instP_df.columns.name = 'KOMMUNENR' 
    muni_names = stem['KOMMUNENAVN'].iloc[muni_index] #Get relevant municipilaties
#    #Return as forecast object with specified information
    if mode == 'all':
        return(SPP(SSP_df, muni_names, instP_df, sub_h_freq, sub_D_freq))
    elif mode == 'remove_outliers':
        return (SPP(outliers(SSP_df), muni_names, instP_df,
                    sub_h_freq, sub_D_freq))

def import_instp(t_start, t_end, muni_list="all", sub_D_freq='all'):
    """
    Imports installed effect for given municipalities in timerange.
    
    Parameters
    ----------
    t_start : pandas timestamp or datetime.date
       Start date as whole date
    t_end : pandas timestamp or datetime.date
       End date as whole date
    muni_list: list or string, optional
       List with municipalities
    sub_D_freq : string, optional
       Subsampling in dates
     
    Returns
    -------    
    instP_df : pandas dataframe
       Dataframe with installed effect
    
    Examples
    --------
    >>> t0 = pd.Timestamp(2017,1,1)
    >>> t1 = pd.Timestamp(2017,1,10)
    >>> muni_list = [851,]
    >>> import_instp(t0,t1,muni_list = muni_list)
                    851
    2017-01-01  14.4303
    2017-01-02  14.4303
    2017-01-03  14.4303
    2017-01-04  14.4303
    2017-01-05  14.4303
    2017-01-06  14.4303
    2017-01-07  14.4303
    2017-01-08  14.4303
    2017-01-09  14.4303
    2017-01-10  14.4303
    """
    root = return_to_root()
    #resample days
    if sub_D_freq == 'all':
        sub_D_freq = "D"
      
     #fetch stem stata, also used later
    stem_path = "data/stem_data/muni_data_new.xlsx" #load muni numbers from file
    stem = pd.read_excel(root + stem_path)
    muninr = stem['KOMMUNENR']

    if muni_list == "all":
        muni_index = range(len(muninr)) #All indicies
        muni_list = muninr
    else:
        muni_list.sort() #sort in order to get correct indicies
        #List with indicies of chosen municipilaties
        muni_index = np.in1d(muninr, muni_list).nonzero()[0] 
    
    day_rng = pd.date_range(t_start, t_end, freq=sub_D_freq) 
    K, M = len(day_rng), len(muni_list)
    instP_dat = np.zeros((K,M))        

    folder_path = "data/"
    idx_count = 0
    for t in day_rng:
        data_path = "%d/%d/%d/" %(t.year, t.month, t.day) #Specific day and hour
        instP_dat[idx_count] = np.take(sio.loadmat(root + folder_path + \
                 data_path + "InstEff_all.mat")['InstEff'].T[0], muni_index)
        idx_count += 1
        
    instP_df = pd.DataFrame(instP_dat, index=day_rng, columns=muni_list)
    return(instP_df)


def monthly_dep_outlier(df):
    tol = [6,2.9,3,2.5,2,2.09,1.8,1.65,2,3.3,3.05,5.7] # Fundet ved trial and error
    
    df_temp = df.copy()
    for j in range(1,len(tol)+1):
        index = df.index.month == j
        df_temp[index] = df[index] * tol[j-1]
    return df_temp

def monthly_dep_epsilon(df):
    nr = df.columns
    df_temp = df.copy()
    index = df.index
    for nr in df.columns:
        for m in range(1,13):
            index_m = index[index.month == m]
            df_temp[nr][index_m] = df[nr][index_m] + 0.3*df[nr][index_m].max()
    return df_temp

def remove_entire_day(data_frame):
    row,col = np.where(data_frame.isna())
    index = data_frame.index
    col = data_frame.columns[col]
    zipped = zip(index[row].strftime('%Y-%m-%d'),col)
    for time,muninr in zipped:
        data_frame[muninr][time] = np.nan
    return data_frame
    
def outliers(SPP):
    root = return_to_root()
    os.chdir(root + '/scripts/libs_and_modules')
    from clear_sky_model import get_cs_class
    muni_list_outlier = list(SPP.columns)
    ts = pd.Timestamp
    t0_outlier,t1_outlier = ts(SPP.index[0].strftime('%Y-%m-%d'))\
      ,ts(SPP.index[-1].strftime('%Y-%m-%d'))
      
    cs_obj_outlier = get_cs_class(t0_outlier,t1_outlier\
                                            , muni_list = muni_list_outlier)
    cs_forecast = cs_obj_outlier(t0_outlier,t1_outlier)
    cs_obj_outlier_tol_prev = pd.DataFrame(cs_forecast.values\
        ,index = SPP.index,columns = SPP.columns)
    cs_obj_outlier_tol = monthly_dep_epsilon(monthly_dep_outlier(cs_obj_outlier_tol_prev))

    df_wo_outliers = SPP[SPP <= cs_obj_outlier_tol]
    
    df_wo_outliers = remove_entire_day(df_wo_outliers)
    return df_wo_outliers