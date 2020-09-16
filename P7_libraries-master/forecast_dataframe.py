# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:45:41 2018

@author: Tobias
"""
import matplotlib.pyplot as plt
import pandas as pd
from datetime import time as d_time
from utilities import Timedelta

class fc_df(pd.DataFrame):
    """
    Builds some functionality on the pandas dataframe class mainly handeling
    dataframes where each row contains forecast from specific time with 
    different horizons into the future.
    """
    
    def __init__(self,df,freq = None):
        """
        Runs some checks for the fc_df class and then initialises as normal
        pandas dataframe. 
        """
        if not type(df) == pd.core.frame.DataFrame:
            raise(TypeError("Only pandas dataframes allowed."))
        
        if not type(df.index) == pd.core.indexes.datetimes.DatetimeIndex:
            raise(TypeError("Only time series dataframe are allowed.\
Initialise df with pd.date_range as index "))
        
        pd.DataFrame.__init__(self,df)
        self.freq = freq
    
    def setfreq(self,freq):
        self.freq = freq
        
    def get_ts(self,mode = "1horizon",muni = None, horizon = None,time = None):
        """
        Returns forecast timeseries as or timeseries dataframes under different 
        modes. 
        
        Parameters
        ----------
        muni : int or array like
            If specific municipality or more is wanted specify this
            by muni arugment
        mode: str
            "1horizon" mode gives the forecast at different times at same 
            horizon.
            "time" mode gives forecast from specific time at different 
            horizons. 
        horizon: datetime.time
            Which fc horizon in 1horizon mode.
        time : pd.Timestamp
            In 'time' mode, will plot from this specified time
        """      
        if mode == "1horizon":
            if horizon is None:
                horizon = self.columns.levels[1][0]
            #Use df.dx attribute to get forecast at specified horizon
            ts = self.xs(horizon,level = 1,axis = 1,drop_level = False)
            new_idx = self.index + Timedelta(horizon)
            ts.index = new_idx
            
        if mode == "time":
            ts = self.loc[time]
            dr = pd.date_range(time + Timedelta(self.columns[0][1]),
                               time + pd.Timedelta(hours = 6),freq =self.freq)
            ts.index = dr
        
        if not muni is None:
            return(ts[muni])
        
        return(ts)
        
    
    def plot(self,mode = "1horizon",target = None,muni = None, horizon = None,
             time = None,save = False, save_path = 'fc',
             title = '',show = True,dpi = 500,**dfargs):
        """
        Plot the dataframe using different modes. Most arguments shouuld make
        sense. Returns figure and ax object.
        
        Special parameters
        ------------------
        target : pd.DataFrame
            Pandas dataframe with target data
        muni : int or array like
            If plot for specific municipality or more is wanted specify this
            by muni arugment
        horizon : date
        mode: str
            "1horizon" mode gives the forecast at different times at same 
            horizon.
            "time" mode gives forecast from specific time at different 
            horizons. 
        horizon: datetime.time
            Which fc horizon in 1horizon mode.
        time : pd.Timestamp
            In 'time' mode, will plot from this specified time
        """
        #maybe put some sanity checks in here
        fig, ax = plt.subplots()
        
        if mode == "1horizon":
            ts = self.get_ts('1horizon',muni = muni,horizon = horizon)
            ts.plot(ax = ax,**dfargs)
        
        if mode == "time":
            ts = self.get_ts('time',time = time,muni = muni)
            ts.plot(ax = ax,style = "--",color = "k",**dfargs)
            ax.axvline(x = time, color='r', linestyle='--')
       
        if not target is None:
            if muni is not None:
                target[muni].plot(ax = ax)
            else:
                target.plot(ax = ax)
        
        ax.set_title(title)
        
        if save:
            plt.savefig(save_path,dpi = dpi)
        
        return(fig,ax)