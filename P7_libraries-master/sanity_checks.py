# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:40:29 2018

@author: Tobias
"""

import pandas as pd
from datetime import time as d_time
import numpy as np

def check_timestamps(t_start,t_end):
    """
    Perform sanity checks for timestamps used for importing data.
    
    Parameters
    ----------
    t0,t1 : pd.Timestamp
        Timestamp for start and end date of data
    """
    
    if type(t_start) != pd._libs.tslib.Timestamp or type(t_end) != \
    pd._libs.tslib.Timestamp:
        raise(TypeError("t_start and t_end should be pandas timestamp"))
    
    t_max = pd.Timestamp(2017,12,31)
    t_min = pd.Timestamp(2015,1,1)
    if t_start > t_end:
        raise(ValueError("t_start should be before or the same as t_end"))
    
    if t_start < t_min or t_end > t_max:
        raise(ValueError("Select a daterange within 2015 to 2017"))
    
    if t_start.time() != d_time(0,0) or t_end.time() != d_time(0,0):
        raise(ValueError("t_start and t_end should be whole dates only \
                         i.e hours = 0 and minutes = 0. \nUse the hours\
                         argument to get less hours on a day"))

def check_hours(hours):
    """
    Perform sanity checks for hour arguments used for importing data.
    
    Parameters
    ----------
    hours : tuple with two strings
        Hour argument of the form ("hh:mm","hh:mm") - eg. ("06:00",18:15")
    """
    if not isinstance(hours,(str,list,tuple,np.ndarray)):
        raise(TypeError("hours should be string, list typle or numpy array"))
    
    al_minutes = ('00','15','30','45') #allowed hours
    if (hours[0][-2:] not in al_minutes or hours[1][-2:] not in al_minutes) and \
    isinstance(hours,(list,tuple,np.ndarray)):
        raise(ValueError("Minute in hours should be 00, 15, 30 or 45"))

def check_subs_args(sub_h_freq,sub_D_freq,mode):
    
    if not isinstance(sub_h_freq,str):
        raise(ValueError("Frequency hour argument must be string.\ne.g. \"2H\""))
    
    if not isinstance(sub_D_freq,str):
        raise(ValueError("Frequency day argument must be string.\ne.g. \"5D\""))
        
    if sub_h_freq[-1] != 'H' and sub_h_freq != 'all' and mode == "SPP" and \
    sub_h_freq != "15min":
        raise(NotImplementedError("Currenly only hour sub sampling is allowed"))
    
    if sub_D_freq[-1] != 'D' and sub_D_freq != 'all':
        raise(NotImplementedError("Currenly only day sub sampling is allowed"))