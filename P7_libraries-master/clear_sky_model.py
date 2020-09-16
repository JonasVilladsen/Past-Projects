# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:46:53 2018

@author: Tobias
"""

import pandas as pd
from pvlib import clearsky
from pvlib.location import Location
import numpy as np
from utilities import return_to_root
from import_SPP import import_SPP as import_SPP #Import_SPP 
from muni_class import create_muni_class_from_SPP
import os

class cs_model:
    """
    Object containing the model for computing the predicted clearsky effect for municipilaties. 
    
    Can easily be initialies using the "get_cs_class" function.
    Once initialised it can be called in specific timerange to get the clear sky model for the municipilaites intilalies for.
     
    Input:
        muni_obj: Municipilaty object from the muni class in desired daterange.
        See ?cs_model.__call__ for info about calling the object
    """
    def __init__(self,muni_obj):
        self.muni_obj = muni_obj
        self.coordinates = muni_obj.coor
        self.tobis_magic_constant = 0.16942194
        self.instp = muni_obj.instp
        self.hours = muni_obj.hours
        loc_temp = {}
        for nr in muni_obj.muninr: #import location object for each minicpilaty
            loc_temp[nr]  = Location(muni_obj.coor.loc[nr][0],muni_obj.coor.loc[nr][1],\
                                      timezone = "Europe/Copenhagen",altitude = 10,\
                                      name = "Denmark")
        self.location = loc_temp
        root = return_to_root()
        coef_path = 'scripts/libs_and_modules/coef'
        os.chdir(root+coef_path)
        self.coef = np.load('coef_cs.npy')
            
   
    def __call__(self,t_start,t_end,res = "15min",norm = False):
        """
        Calculates clear sky model in timerange with given resulution for all municipilaties in the instance.
        
        Input:
            t_start,t_end: pandas timestamp within time where installed effect have been loaded
            res: Is resulution of the model. 15min resultion is defauls
            norm: If norm = True the model is not scaled by installed effect. 
        
        Returns:
            Clear sky model values as pandas dataframe
        """
        if t_start < self.instp.index[0] or  t_end > self.instp.index[-1]:
            raise(ValueError("Let [t_start,t_end] be in the timerange:\n%s to %s\nwhere installed effect is loaded." %(str(self.instp.index[0]),str(self.instp.index[-1]))))

        t_end = t_end + pd.Timedelta(hours = 23,minutes = 45)
        count = 0
        Loc_obj = {}
        for key in self.muni_obj.muninr:
            Loc_obj[key] = Location(self.muni_obj.coor.loc[key]['lat'],self.muni_obj.coor.loc[key]['lon'],timezone = "Europe/Copenhagen",altitude = 10\
                       ,name = "Denmark")
            times = pd.DatetimeIndex(start=t_start, end=t_end, \
                             freq=res, tz=Loc_obj[key].tz)
            if count == 0:
                hours = (self.muni_obj.timerange[0].time(),self.muni_obj.timerange[-1].time())
                times = times[times.indexer_between_time(hours[0],hours[1])] #Timerange with only the dates
                df_cs = pd.DataFrame(index = times, columns = self.muni_obj.muninr)
                count += 1
        hours = (self.muni_obj.timerange[0].time(),self.muni_obj.timerange[-1].time())
        times = times[times.indexer_between_time(hours[0],hours[1])] #Timerange with only the dates
        for key in self.coordinates.index:
            cs = Loc_obj[key].get_clearsky(times,model = "haurwitz")
            cs *= 15*60*10**(-6) #
            if not norm:
                for t in pd.date_range(t_start,t_end,freq = "D"):
                    instP = self.muni_obj.instp[key].loc[t]
                    cs.loc[t:t+pd.Timedelta(hours = 23,minutes = 45)] *= instP
            df_cs[key] = cs.values
        self.coef_vec = self.get_season_vec(df_cs)
        for key in df_cs.columns:
            df_cs[key] *= self.coef_vec
        return df_cs
    

    def old(self,t_start,t_end,res = "15min",norm = False):
        """
        Calculates clear sky model in timerange with given resulution for all municipilaties in the instance.
        
        Input:
            t_start,t_end: pandas timestamp within time where installed effect have been loaded
            res: Is resulution of the model. 15min resultion is defauls
            norm: If norm = True the model is not scaled by installed effect. 
        
        Returns:
            Clear sky model values as pandas dataframe
        """
        if t_start < self.instp.index[0] or  t_end > self.instp.index[-1]:
            raise(ValueError("Let [t_start,t_end] be in the timerange:\n%s to %s\nwhere installed effect is loaded." %(str(self.instp.index[0]),str(self.instp.index[-1]))))
        #Creates location object in the correct time zone in an altitude of 10 meters
        t_end = t_end + pd.Timedelta(hours = 23,minutes = 45) 
        times = pd.DatetimeIndex(start=t_start, end=t_end, \
                                 freq=res, tz='UTC') 
        if self.hours != "all":
            times = times[times.indexer_between_time(self.hours[0],self.hours[1])]  
        cs_df = pd.DataFrame(index = times, columns = self.coordinates.index)
        for nr in self.coordinates.index: #Get clear sky model for each muni
            cs = self.location[nr].get_clearsky(times,model = "haurwitz")
    

            cs *= 15*60*(10**(-6)) #Get in MWh/15minutes
            if not norm: #Scale by installed effect unless normalised model is wanted
                for t in pd.date_range(t_start,t_end,freq = "D"):
                    instP = self.instp[nr].loc[pd.Timestamp(t.date())] #remove hours
                    #By emperical methods it is found that cs should be scaled
                    #Installed effect squared
                    cs.loc[t:t+pd.Timedelta(hours = 23,minutes = 45)] *= instP
            cs_df[nr] = cs*self.tobis_magic_constant
        #Incomment later
#        cs = self.parameters[0]*cs + self.parameters[1]*((cs**2)/municipilaty_info.instP)
        return cs_df
    
    

    
    def get_season_vec(self, df_cs):
        length = len(df_cs.index)
        coef_vec = np.zeros(shape = (length))
        for k in range(length):
            coef_vec[k] = self.season_coef(df_cs,k)
        return coef_vec

    def season_coef(self,df_cs,i):
        month = df_cs.index[i].month
        return self.coef[month-1]
    
    def __str__(self):
        s = "Clear sky model object for %d minicilaties in the day range\n%s to %s\nin the hours\n%s to %s.\nThe object can be called to get the cs model values in this timerange." %(len(self.coordinates),self.instp.index[0].date(),self.instp.index[-1].date(),str(self.hours[0]),str(self.hours[1]))
        return(s)
        
def get_cs_class(day_start,day_end,hours = 'all', muni_list = 'all',n_max = 10):
    """
    Import a cs_model class on specific days, hours and municiplaties.
    
    Input:
        day_start/day_end: pandas timestamp as a whole day 
        (hours = minutes = seconds = 00)
        hours: Spefify which hours you want the cs model to when calling it. 
        Example: hours = ("12:00","18:45") will give you the SPP in that 
        timerange for the specified days. Should be a list/tuple wiht elements
        of type "hh,mm" as strings
        muni_list: Speficy which municipilaties you want to import as 
        list/tuple/numpy array
        n_max: parameter for how many folders to go back in when looking for
        the svn folder
    
    Returns:
        A cs_model object. Se ?cs_model for more info
    
    """
    #Import installed effect
    SPP_dat = import_SPP(day_start,day_end,\
                                 muni_list = muni_list,hours = hours)
    
    #Create muni_obj which have coordinates included
    muni_obj = create_muni_class_from_SPP(SPP_dat)
    
    return(cs_model(muni_obj))



def clearsky_model_raw(muni_obj,muninr,t_start,t_end, res = "15min",norm = False):
    """
    Model for computing the predicted clearsky effect for a given municipilaty
    with some total installed effect. 
    
    municipilaty_info: Is an object from the municipilaty class with 
    information about number name, location and total installed effect
    time
    
    time_interval: Is time information as tuple
    
    Returns the values from the clearsky model in the given interval in W/m^2
    """
    #Creates location object in the correct time zone in an altitude of 10 meters
    lat,lon = muni_obj.coor.loc[muninr]
    Loc_obj = Location(lat,lon,timezone = "Europe/Copenhagen",altitude = 10\
                       ,name = "Denmark")
    t_end = t_end + pd.Timedelta(hours = 23,minutes = 45) 
    times = pd.DatetimeIndex(start=t_start, end=t_end, \
                             freq=res, tz=Loc_obj.tz) #winter 1st jan
    hours = (muni_obj.timerange[0].time(),muni_obj.timerange[-1].time())
    times = times[times.indexer_between_time(hours[0],hours[1])] #Timerange with only the dates
    cs = Loc_obj.get_clearsky(times,model = "haurwitz")
    cs *= 15*60*10**(-6) #
    tobis_magic_constant = 1/3. #maybe cange t0 3.5
    cs *= tobis_magic_constant
    if not norm:
        for t in pd.date_range(t_start,t_end,freq = "D"):
            instP = muni_obj.instp[muninr].loc[t]
            cs.loc[t:t+pd.Timedelta(hours = 23,minutes = 45)] *= instP
    return cs