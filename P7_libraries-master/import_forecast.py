# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:14:31 2018

@author: Tobias
"""

import pandas as pd
import os
import scipy.io as sio
import numpy as np
from datetime import time as d_time
from datetime import date
import copy
from utilities import return_to_root,zeropad_hourstring,\
muni_list_to_grid_list, mk_unique_no_nan_int, _handle_timerange
import sanity_checks

class forecast:
    """
    Structure to hande weatherforecast containing global irridiance, windspeed
    and wind direction. 
    
    Input:
        GHI: Global irridicance as pandas dataframe preferably with timeseries
        as index
        WS: Windspeed in same format as GHI
        WD: Wind direction in same format as GHI
    
    Input can be only one information e.g. only GHI or even no input. 
    Cannot handle:
        - Serveal forecast as input. E.g. two different GHI
        - If the input have different dimentions. E.g. if GHI and WS contain
        different number of grid points or time information. (well tecnically
        it can handle it but some of the methods such as install and
        __str__ will give untrue information)
    """
    def __init__(self,GHI = None,WS = None, WD = None,TMP = None,D_freq = "D",\
                 h_freq = "H",mode = "grid",hours = None):
        """
        Initialises forecast and check which information is given.
        Also calls the install method (see forecast.install).
        
        Modes: "grid" and "muni"
        grid mode accepts forecasts for given grid numbers
        muni mode accepts forecasts for given muni numbers
        simu mode accepts forecasts initialised in simulation mode
        
        Example: If only GHI is given it will initialise WS and WD as Nonetype
        and call the install method using GHI
        """
        if not mode in ['grid','muni','simu']:
            raise(ValueError("Only \"grid\"  and \"muni\" mode are allowed"))
        
        self.GHI, self.WS, self.WD, self.TMP = GHI, WS, WD, TMP
        self.D_freq, self.h_freq = D_freq, h_freq
        self.mode = mode 
        self.hours = hours
        
        #Find nonempty forecast info and get timerange and grid numbers from
        # that
        if type(GHI) == pd.core.frame.DataFrame:
            self._install(GHI)
        elif type(WS) == pd.core.frame.DataFrame:
            self._install(WS)
        elif type(WD) == pd.core.frame.DataFrame:
            self._install(WD)
        elif type(TMP) == pd.core.frame.DataFrame:
            self._install(TMP)
    
    def _install(self,dataframe):
        """
        Fetches index as timerange and columbinformation as gridnumbers to 
        object. 
        """
        self.timerange = dataframe.index
        if self.hours == None:
            self.hours = (self.timerange[0].time(),self.timerange[-1].time())
        if self.mode == "grid":
            self.gridnr = dataframe.columns
        else:
            self.muninr = dataframe.columns

    def __str__(self):
        """
        Prints out nicely formated string with relevant information about 
        forecast. 
        """
        if hasattr(self,"timerange"): #Check if forecast is initialies with object
            s = "Forecast at the dates:\n%s to %s"\
            %(str(self.timerange[0].date()),str(self.timerange[-1].date()))
            if self.mode == 'grid' or self.mode == 'muni':
                s +=  " in the hours\n%s to %s every %s\'s and every %s\'s"\
                %(str(self.timerange[0].time()),str(self.timerange[-1].time()),\
              self.h_freq, self.D_freq)
            else:
                s += "\nStarting from %s untill %s" %(str(self.timerange[0]),\
                                                    str(self.timerange[-1]))
            s += "\n\nForecast contains:\n"
            if type(self.GHI) == pd.core.frame.DataFrame:
                s += "  - GHI:Global horisontal irridiance\n"
            if type(self.WS) == pd.core.frame.DataFrame:
                s += "  - WS :Wind speed\n"
            if type(self.WD) == pd.core.frame.DataFrame:
                s += "  - WD :Wind Direction\n"
            if type(self.TMP) == pd.core.frame.DataFrame:
                s += "  - TMP:Temperature\n"
            if self.mode == "grid":
                s += "\nForecast covers %d grid points" %len(self.gridnr)
            else:
                s += "\nForecast covers %d municipalities" %len(self.muninr)
            return(s)
        else:
            return("This forecast is empty")

class forecast_simu:
    """
    Handles simulation of real time forecast, by beeing able to get forecast at
    a given time as if it were real time. 
    
    DO NOT INITIALISE THIS CLASS MANUALLY: Use the "import_muni_forecast_simu"
    function. 
    
    Use the __call__ method to get forecasts. Se ?forecast for general 
    information about forecasts. 
    
    Methods:
        __call__: Give the newest forecast given a time. Input is pandas 
                  timestamp and it should be within loaded times. See:
                  self.t_min, self.t_max for allowed times
        __print__:General information about the function
        
    Exampels:
    #Import standard forecast object
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,3)
    muni_list = [851,860]
    fc_simu = import_muni_forecast_simu(t0,t1,muni_list=muni_list); print(fc_simu)
    >>> Forecast from the dates:
    >>> 2017-01-01 to 2017-01-03
    >>> Can be called in time timerange 2017-01-01 03:00:00 to 2017-01-06 00:00:00
    >>> Forecast contains:
    >>>     - GHI:Global horisontal irridiance
    >>> Forecast covers 2 municipalities
    fc_t1 = fc(t1) #Newest forecast at time t1
    print(fc_t1)
    
    
    """
    def __init__(self,data,info = ('GHI',),h_freq = "H",mode = "muni"):
        self.data = data
        self.info = info
        self.h_freq = h_freq
        self.days = list(data.keys())
        self.mode = mode
        self.t_min = pd.Timestamp(self.days[0])  + pd.Timedelta(hours = 3)
        self.t_max = pd.Timestamp(self.days[-1]) + pd.Timedelta(hours = 18+54)
        
        #Get muni number list
        fc0 = self.data[self.days[0]]['00'] #first forecast for getting info
        if type(fc0.GHI) == pd.core.frame.DataFrame:
            self._install(fc0.GHI)
        elif type(fc0.WD) == pd.core.frame.DataFrame:
            self._install(fc0.WD)
        elif type(fc0.WS) == pd.core.frame.DataFrame:
            self._install(fc0.WS)
    
    def __call__(self,cur_time):
        """
        Use call function to get forecast object out.
        Input: cur_time: pandas timestamp in allowed timeframe
        """
        fc_time = self._closest_forecast(cur_time)
        day = fc_time.date()
        if  cur_time > self.t_max or cur_time < self.t_min:
            raise(ValueError("Forecast for selected time is not loaded.\n \
                             Allowed timerange is %s to %s\nRemember: \
                            If you wanted time is before 03:00 you need to load \
                            data from the date before that."\
                             %(self.t_min,self.t_max)))
        hourstr = zeropad_hourstring(str(fc_time.hour))
        return(self.data[day][hourstr])
        
    def _install(self,dataframe):
        """
        Fetches index as timerange and columbinformation as gridnumbers to 
        object. 
        """
        if self.mode == "grid":
            self.gridnr = dataframe.columns
        else:
            self.muninr = dataframe.columns
    def __str__(self):
        """
        Prints out nicely formated string with relevant information about 
        simulation foreast. 
        """
        s = "Forecast from the dates:\n%s to %s\nCan be called in time\
                timerange %s to %s\nForecast contains:\n" \
            %(str(self.days[0]),str(self.days[-1]),str(self.t_min),str(self.t_max))
        fc0 = self.data[self.days[0]]['00'] #first forecast for getting info
        if type(fc0.GHI) == pd.core.frame.DataFrame:
            s += "  - GHI:Global horisontal irridiance\n"
        if type(fc0.WS) == pd.core.frame.DataFrame:
            s += "  - WS :Wind speed\n"
        if type(fc0.WD) == pd.core.frame.DataFrame:
            s += "  - WD :Wind Direction\n"
        if self.mode == "grid":
            s += "\nForecast covers %d grid points" %len(self.gridnr)
        else:
            s += "\nForecast covers %d municipalities" %len(self.muninr)
        return(s)
        
    def _closest_forecast(self,cur_time):
        """
        Calculates the time for the latest forecast
        """
        if cur_time.hour < 3:
            day = (cur_time - pd.Timedelta(days = 1)).date()
            return(pd.Timestamp(day.year,day.month,day.day,18))
        elif cur_time.hour < 9:
            day = cur_time.date()
            return(pd.Timestamp(day.year,day.month,day.day,0))
            
        elif cur_time.hour < 15:
            day = cur_time.date()
            return(pd.Timestamp(day.year,day.month,day.day,6))
            
        elif cur_time.hour < 21:
            day = cur_time.date()
            return(pd.Timestamp(day.year,day.month,day.day,12))
        else:
            day = cur_time.date()
            return(pd.Timestamp(day.year,day.month,day.day,18))

def import_forecast(t_start, t_end, hours="all", info=("GHI",),\
                    grid_list="all", sub_h_freq='H',\
                    sub_D_freq='all'):
    """
    Import serveral forecasts into a continius forecast with information in the
    speficied hours. Imports into the forecast class. 
    
    Input:
         t_start/t_end: Start/end time for your forecast as pandas timestamp.
         Select this value between 2017/01/01 and 2017/12/31.
         Only whole dates are allowed, no hours or minutes can be speficied. 
         

    Optional input:
        info: Specify which info you want from the forecast. If left returns
        forecast with GHI information. If you want other information give tuple
        with string entries discribing the info. Example: info = ("GHI","WS","WD"),
        ("WS",) and so on. If info = "all", then GHI, WS and WD will be given. 
        gridnr: Specifiy if you only want forecast from spefific grid numbers.
        If left it will import data from all grid numbers, else give a
        list/tuple/numpy array with the numbers.
        hours: Specify which timeragne of hours you want for each day.
        Ex: hours = ["04:00","22:00"] for forecasts in that timerange each day. 
        sub_h_freq: Subsample in hours/minutes. Ex: sub_h_freq = "2H" for
        samples only every 2 hours
        sub_D_freq: Subsample in days. Ex: sub_D_freq = "5D" for samples only
        every 5 days
        
    Returns: Forecast from speficied timerange as a forecast object.
    See ?forecast for more info 
    
    Examples:
    #Initialise timestamp with arguments specified. 
    t0 = pd.Timestamp(year = 2017,month = 1,day = 1), print(t0)
    >>> 2017-01-01 00:00:00
    
    #Initialise timestamp without arguments specified
    t0 = pd.Timestamp(2017,1,1); print(t0)
    >>> 2017-01-01 00:00:00

    #Import forecast with GHI information
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10)
    fc = import_forecast(t0,t1); print(fc)
    >>> Forecast in the timerange:
    >>> 2017-01-01  to 2017-01-10 in the hours
    >>> 00:00:00 to 23:00:00 every D's and every H's
    >>>
    >>> Forecast contains:
    >>>     - GHI:Global horisontal irridiance
    >>>
    >>> Forecast covers 354 grid points
    
    #You get dataframes with GHI like this
    GHI_dataframe = fc.GHI
    
    #You can view GHI as an excel file like:
    fc.SPP.to_csv("GHI.csv")
    
    #Even as an html file to view in your browser which gives a good overview
    fc.GHI.to_html("GHI.html") 
    
    #Import forecast with all information (GHI,Windspeed and Winddirection)
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10)
    fc = import_forecast(t0,t1,info = "all"); print(fc)
    >>> Forecast in the timerange:
    >>> 2017-01-01 to 2017-01-10 in the hours
    >>> 00:00:00 to 23:00:00 every D's and every H's
    >>>    
    >>> Forecast contains:
    >>>   - GHI:Global horisontal irridiance
    >>>   - WS :Wind speed
    >>>   - WD :Wind Direction
    >>>
    >>> Forecast covers 354 grid points
    
    #Import forecast at timerange on a day
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10)
    hours = ("05:00","22:00")
    fc = import_forecast(t0,t1,info = "all",hours = hours); print(fc)
    >>> Forecast in the timerange:
    >>> 2017-01-01 to 2017-01-10 in the hours
    >>> 05:00:00 to 22:00:00 every D's and every H's
    >>>    
    >>> Forecast contains:
    >>>   - GHI:Global horisontal irridiance
    >>>   - WS :Wind speed
    >>>   - WD :Wind Direction
    >>>
    >>> Forecast covers 354 grid points
    
    #Import specific grid numbers
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10)
    grid_list = [190,315,1413]
    fc = import_forecast(t0,t1,info = "all",grid_list = grid_list); print(fc)
    >>> Forecast in the timerange:
    >>> 2017-01-01 to 2017-01-10 in the hours
    >>> 05:00:00 to 22:00:00 every D's and every H's
    >>>    
    >>> Forecast contains:
    >>>   - GHI:Global horisontal irridiance
    >>>   - WS :Wind speed
    >>>   - WD :Wind Direction
    >>>
    >>> Forecast covers 3 grid points
    #Import using subsampling
    #t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10)
    sub_hours = "2H"; sub_days = "5D"
    fc = import_forecast(t0,t1,sub_h_freq = sub_hours,sub_D_freq = sub_days); print(fc)
    >>> Forecast at the dates:
    >>> 2017-01-01 to 2017-01-06 in the hours
    >>> 00:00:00 to 22:00:00 every 5D's and every 2H's
    >>>
    >>> Forecast contains:
    >>>   - GHI:Global horisontal irridiance
    >>> Forecast covers 354 grid points
    """
    root = return_to_root()
    #Sanitycheck for different input    
    if "data" not in os.listdir(root):
        raise(OSError("Root is noot the svn shared folder"))
    
    sanity_checks.check_timestamps(t_start,t_end)
        
    sanity_checks.check_hours(hours)
                
    sanity_checks.check_subs_args(sub_h_freq,sub_D_freq,mode="forecast")
        
    #Fetch stem data (grid) - used for sanity check but also later on
    grid_path = "data/stem_data/forecast_grid" #load grid numbers from file
    grid = sio.loadmat(root + grid_path + ".mat")['forecast_grid'].T[0]
    
    if not set(grid_list).issubset(set(grid)) and grid_list != 'all':
        raise(ValueError("One or more elements in grid_list is invalid: \
                            forecast for that grid point is not known"))
    
    
    #handle timerange
    rng,day_rng,sub_h_nr,spd,s_day0,s_day1 =\
    _handle_timerange(t_start,t_end,hours,sub_h_freq,sub_D_freq)
    
    if grid_list == "all": #Fix in order to import municipaligties instead
        grid_index = range(len(grid)) #All indicies
        grid_list = grid
    else:
        grid_index = np.in1d(grid, grid_list).nonzero()[0] #List with indicies of ch
 
    #Create data structures
    if info == "all":
        info = ("GHI","WD","WS","TMP")
    data = dict.fromkeys(info) #Big ass data matrix 
    N, M = len(rng), len(grid_index)
    #Create datamatrix for forecast types
    for key in data.keys():
        data[key] = np.zeros((N,M))
    
    folder_path = "data/"
    idx_count = 0
    for t in day_rng: #Runs thorugh every 6th element in timerange excluding the last
        data_path = "%d/%d/%d/" %(t.year,t.month,t.day) #Specific day and hour
        for key in data.keys(): #load from file and write to matrix
                data[key][idx_count*spd:idx_count*spd + spd] = \
                np.matrix(pd.read_pickle(root + folder_path + data_path +\
                                         key + 'day15min.pkl'))\
                                        [s_day0:s_day1][:,grid_index][::sub_h_nr]  #incommet later
                 # [s_day0:s_day1] picks out the relevant times
             # [:,muni_index] picks out the relevant munipicilaties
        idx_count += 1
    
    #Convert to dataframe, overwrites matricies
    dataframes = dict.fromkeys(["GHI","WD","WS","TMP"]) #dictionary for dataframes
    for key in data.keys():        
       dataframes[key] = pd.DataFrame(data[key], index=rng, columns=grid_list)
       dataframes[key].columns.name = 'GRIDNR'
    #Return as forecast object with specified information
    return(forecast(GHI=dataframes["GHI"], WD=dataframes["WD"],
                    WS=dataframes["WS"], TMP=dataframes["TMP"],
                    h_freq=sub_h_freq, D_freq=sub_D_freq))


def import_muni_forecast(t_start, t_end, hours="all", info=("GHI",), 
                         muni_list="all", sub_h_freq='H', sub_D_freq='all'):
    """
    Imports forecast from list of municipilaties. See ?import_forecast for more info.
    
    Input:
        muni_list: Should be list or tuple with valid municipality numbers
        Se ?import_forecast for info about other input.
    
    Returns:
        Forecast object initialised in muni mode. 
    """
    if not isinstance(muni_list,(list,tuple)) and muni_list != 'all':
        raise(ValueError("muni_list should be list or tuple"))
            
    root = return_to_root()
    #Sanitycheck for different input    
    if "data" not in os.listdir(root):
        raise(OSError("Root is noot the svn shared folder"))
    
    sanity_checks.check_timestamps(t_start, t_end)
        
    sanity_checks.check_hours(hours)
                
    sanity_checks.check_subs_args(sub_h_freq, sub_D_freq, mode="forecast")
    
    stem_path = "data/stem_data/muni_data_new.xlsx" #load muni numbers from file - maybe cange folder
    stem = pd.read_excel(root + stem_path)
    muninr = stem['KOMMUNENR']
    
    if not set(muni_list).issubset(set(muninr)) and muni_list != 'all':
        raise(ValueError("muni_list contains a muninumber that is not valid."))
    
    if muni_list == "all":
        muni_list = muninr
    else:
        #Ensure list is sorted numerically in order to get right indicies
        muni_list.sort()
    
    #handle timerange
    rng,day_rng,sub_h_nr,spd,s_day0,s_day1 =\
    _handle_timerange(t_start, t_end, hours, sub_h_freq, sub_D_freq)
    

    #Create data structures
    all_keys = ("GHI", "WD", "WS", "TMP")
    if info == "all":
        info = all_keys
    data = dict.fromkeys(all_keys) #Big ass data matrix 
    #Create datamatrix for forecast types
    for key in info:
        data[key] = pd.DataFrame(columns=muni_list)
    
    rng,day_rng,sub_h_nr, spd, s_day0, s_day1 =\
    _handle_timerange(t_start, t_end, hours, sub_h_freq, sub_D_freq)
    
    for t in day_rng: #Runs thorugh every 6th element in timerange excluding the last
        data_path = "data/%d/%d/%d/" %(t.year, t.month, t.day) #Specific day and hour
        for key in info: #load from file and write to matrix
            data[key] = data[key].append(pd.read_pickle(root + data_path + 
                key + "muniday15min.pkl")[muni_list])


    for key in info:
        data[key] = data[key].reindex(rng, method='ffill')
     
    return(forecast(GHI=data["GHI"], WD=data["WD"], WS=data["WS"],
                    TMP=data["TMP"], h_freq=sub_h_freq,
                    D_freq=sub_D_freq, mode="muni"))
    
    
def import_muni_forecast_simu(t_start, t_end, info=("GHI",),
                              muni_list="all", res="H"):
    """
    Imports forecast from list of municipilaties. See ?import_forecast for more info.
    
    Input:
        t_start,t_end: Pandas timestamp in 2017
        muni_list: Should be list or tuple with valid municipality numbers
        res: Resulution of forecast, can be set to "15min" in order to match SPP data
        Se ?import_forecast for info about other input
    
    Returns:
        forecast_simu object initialies in muni mode. Se ?foecast_simu for more info
    """
    t_max = pd.Timestamp(2018,1,1,0)
    if t_start > t_max or t_end > t_max:
        raise(ValueError("Select a daterange within 2017"))
    
    if t_start.time() != d_time(0,0) or t_end.time() != d_time(0,0):
        raise(ValueError("t_start and t_end should be whole dates only i.e. \
                         hours = 0 and minutes = 0. \n \
                         Use the hours argument to get less hours on a day"))

    if not isinstance(info,(list,tuple,np.ndarray)) and info != "all":
        raise(TypeError("info argument should be tuple, list or numpy array"))
    
    if not isinstance(muni_list,(list,tuple)) and muni_list != 'all':
        raise(ValueError("muni_list should be list or tuple"))
    
    grid_list,conv_sheet = muni_list_to_grid_list(muni_list)
    if muni_list == 'all': #transform grid list into all because faster later
        grid_list = 'all'
        muni_list = conv_sheet.index
           
    #structure for forecasts
    days =  pd.date_range(t_start,t_end,freq = "D")
    h_dic = dict.fromkeys(['00','06','12','18'])
    fc_dic = dict.fromkeys(days.date)
    for day in days:
        #Load data
        h_dic_day = copy.deepcopy(h_dic) #used for storing forecasts
        fc_grid = import_single_forecast_from_mat(day,info = info,\
                                                  grid_list = grid_list,\
                                                  res = res)
        for h in fc_grid.keys():
            fc_muni = _average_grid_to_muni(fc_grid[h],info,conv_sheet,muni_list)
            h_dic_day[h] = forecast(GHI = fc_muni["GHI"],WD = fc_muni["WD"],\
                   WS = fc_muni["WS"],mode = "simu",h_freq=res,hours = 'all')
        fc_dic[day.date()] = h_dic_day

    return(forecast_simu(fc_dic,info,h_freq=res))


def import_single_forecast_from_mat(day,info = ("GHI",),grid_list = "all",\
                                    res = "H"):
    """
    comment here tobi
    """
    root = return_to_root()
    #Create data structures
    if info == "all":
        info = ("GHI","WD","WS")
    hours = ['00','06','12','18']
    info_fc = dict.fromkeys(("GHI","WD","WS")) #For holding forecasts
    data = dict.fromkeys(hours) 

    folder_path = "Fortrolig_data/2017_forecast/"
    data_path = "%d/%d/" %(day.month,day.day) #Specific day and hour
    
    
    grid_path = "Fortrolig_data/stem_data/forecast_grid" #load grid numbers from file
    grid = sio.loadmat(root + grid_path + ".mat")['forecast_grid'].T[0]
    #Set up grid list and it's index
    if grid_list == "all":
        grid_index = range(len(grid)) #All indicies
        grid_list = grid
    else:
        grid_index = np.in1d(grid, grid_list).nonzero()[0] #List 
    
    for h in data.keys(): #load from file and write to matrix
        t0 = day + pd.Timedelta(hours = int(h)) #Begin time
        if day.date() == date(2017, 5, 12) and h == '00': #The day where date is missing - only special case 
            t1 = t0 + pd.Timedelta(hours = 48) #Each forecast is 54 hours
        else:
            t1 = t0 + pd.Timedelta(hours = 54) #Each forecast is 54 hours
        time = pd.date_range(t0,t1,freq = "H")
        for key in info:
            #Load forecast from hour h in given grid list
            fc = sio.loadmat(root + folder_path + data_path + key +\
                            h + '.mat')['matrix'][:,grid_index]
            
            #Around here should be the possibilty of upscaling and 
            #dealing with the missing forecast
            info_fc[key] = pd.DataFrame(fc,index = time,columns = grid_list)
            if res != "H": #Linear interpolate to get better time resolution
                info_fc[key] = info_fc[key].resample(res).\
                interpolate(method = "time")
        data[h] = forecast(GHI = info_fc["GHI"],WD = info_fc["WD"],\
                            WS = info_fc["WS"],mode = "muni",h_freq=res)
    return(data) 

def _average_grid_to_muni(fc,info,conv_sheet,muni_list):    
    """ 
    Converts forecast for grid points into forecast for municiplalities.
    
    Input:
        fc: Forecast object
        info: "all" for GHI, WD and WS, else set tuple for the info it contains
        conv_sheet: Excel sheet with stem data for muni numbers vs grid points
        muni_list: List with specified municipalities
    
    Returns:
        Forecast object for municipalities
    """
    if info == "all":
        info = ("GHI","WD","WS","TMP")    
        
    #empty dataframe for muni fc
    fc_muni = dict.fromkeys(["GHI","WD","WS","TMP"])
    
    #Take average using the muni grid conversion sheet
    for key in info:
        try: #If timerange is installed this will work
            fc_muni[key] = pd.DataFrame(index = fc.timerange)
        except: #Else this will work
            fc_muni[key] = pd.DataFrame(index = fc[key].index)
        for nr in muni_list:
            muni_info  = conv_sheet[['GNr1','GNr2','GNr3']].loc[nr] #row for muni
            grid_idx = mk_unique_no_nan_int(list(muni_info)) #grid list for muni
            fc_muni[key][nr] = getattr(fc,key)[grid_idx].mean(axis = 1) #mean for grid points
        fc_muni[key].columns.name = "KOMMUNENR"
    return(fc_muni)


# =============================================================================
# Old code for forecasts
# Functions here under are not currently used in main scripts
# =============================================================================

def import_forecast_from_mat(t_start,t_end,info = ("GHI",)):
    """
    Import serveral forecasts into a continious forecast with information every
    hour and for all grid points. Imports into the forecast class. 
    
    Input:
         t_start/t_end: Start/end date for your forecast as pandas timestamp.
         Select this value between 2015/01/01 and 2017/12/31
         
    Optional input:
        info: Specify which info you want from the forecast. 
        If left returns forecast with GHI information. 
        If you want other information give tuple with string entries discribing 
        the info. Example: info = ("GHI","WS","WD"), ("WS",) and so on. 
        If info = "all", then GHI, WS and WD will be given. 
        gridnr: Specifiy if you only want forecast from spefific grid numbers. 
    
    Returns: Forecast from speficied timerange as a forecast object. 
    See ?forecast for more info 
    
    Examples:
    #Initialise timestamp with arguments specified. 
    t0 = pd.Timestamp(year = 2017,month = 1,day = 1), print(t0)
    >>> 2017-01-01 00:00:00
    
    #Initialise timestamp without arguments specified
    t0 = pd.Timestamp(2017,1,1); print(t0)
    >>> 2017-01-01 00:00:00

    #Hours can be 0,6,12 or 18
    t0 = pd.Timestamp(2017,1,1,12); print(t0)
    >>> 2017-01-01 12:00:00
    
    #Import forecast with GHI information
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10)
    fc = import_forecast(root,t0,t1); print(fc)
    >>> Forecast in the timerange:
    >>> 2017-01-01 00:00:00 to 2017-01-10 00:00:00
    >>>
    >>> Forecast contains:
    >>>     - GHI:Global horisontal irridiance
    >>>
    >>> Forecast covers 354 grid points
    
    #Import forecast with all information
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10,12)
    fc = import_forecast(root,t0,t1,info = "all"); print(fc)
    >>> Forecast in the timerange:
    >>> 2017-01-01 00:00:00 to 2017-01-10 12:00:00
    >>>    
    >>> Forecast contains:
    >>>   - GHI:Global horisontal irridiance
    >>>   - WS :Wind speed
    >>>   - WD :Wind Direction
    >>>
    >>> Forecast covers 354 grid points
    """
    
    #Sanitycheck for different input    
    root = return_to_root()
    if "data" not in os.listdir(root):
        raise(OSError("Root is noot the svn shared folder"))
     
    sanity_checks.check_timestamps(t_start,t_end)
    
    if not isinstance(info,(list,tuple,np.ndarray)) and info != "all":
        raise(TypeError("info argument should be tuple, list or numpy array"))
        
    #Import more sanity check in neccesary later
    
    
    #Create data structures
    if info == "all":
        info = ("GHI","WD","WS","Temp")
       
    t_end = t_end + pd.Timedelta(hours = 23)
    rng = pd.date_range(t_start,t_end,freq = "H") #daterange for forecast
    grid_path = "data/stem_data/forecast_grid" #load grid numbers from file
    grid = sio.loadmat(root + grid_path + ".mat")['forecast_grid'].T[0]
    data = dict.fromkeys(info) #Big ass data matrix 
    N,M = len(rng),len(grid)
    #Create datamatrix for forecast types
    for key in data.keys():
        data[key] = np.zeros((N,M))
    
    folder_path = "data/"
    t = t_start
    
    #We include the first point in each forecast 
    #This is due to the fact that the first datapoint in forecast is corrupted
    if t_start == pd.Timestamp(2015,1,1): 
    #This datapoint is unavailable because it would come from 2014
        for key in data.keys(): #load from file and write to matrix
            data[key][0] =\
            sio.loadmat(root + folder_path + '2015/1/1/' + key + '_00' + \
                    '.mat')['matrix'][0]
    else: #picks out the 00:00 datapoint from the forecast the day before
        succes = False
        miss_file_count = 0
        while not succes:
            try:
                t0 = t_start - pd.Timedelta(hours = 6*(miss_file_count + 1))
                 #date before t_start
                data_path = "%d/%d/%d/" %(t0.year,t0.month,t0.day) #Specific day and hour
                hour_str = zeropad_hourstring(str(t0.hour))
                for key in data.keys():
                    data[key][0] = \
                    sio.loadmat(root + folder_path + data_path + key + "_"
                                + hour_str + '.mat')['matrix']\
                                [6+6*miss_file_count]
                succes = True
            except:
                miss_file_count += 1
        
    idx_count = 0; end_sample = 1
    for t in rng[::6]:
        succes = False
        miss_file_count = 0
        while not succes:
            try:
                t_f = t - pd.Timedelta(hours = 6*miss_file_count) #check if
                #last file is avaiable if missing
                #Runs thorugh every 6th element in timerange excluding the last
                data_path = "%d/%d/%d/" %(t_f.year,t_f.month,t_f.day) #Specific day and hour
                hour_str = zeropad_hourstring(str(t_f.hour)) #add 0 if single digit
                if t == rng[::6][-1]: #In the last iteration we dont want the last 
                    #sample because it responds to the next day
                    end_sample = 0
                for key in data.keys(): #load from file and write to matrix
                    data[key][idx_count*6 + 1:idx_count*6 + 6 + end_sample] =\
                    sio.loadmat(root + folder_path + data_path + key + "_"
                                + hour_str + '.mat')['matrix']\
                    [6*miss_file_count + 1:6 + end_sample+6*miss_file_count]
                succes = True
            except:
                miss_file_count += 1
        if miss_file_count > 0:
            print("Warning: Missing forecast from %s missing. \
                      Replaced with data from %s forecast" %(str(t),str(t_f)))
        idx_count += 1
    

    #Convert to dataframe, overwrites matricies
    dataframes = dict.fromkeys(("GHI","WD","WS","TMP")) #dictionary for dataframes
    for key in data.keys():        
        if key == "Temp": #fix to change Temp into TMP
            dfkey = "TMP"
        else:
            dfkey = key
        dataframes[dfkey] = pd.DataFrame(data[key],index = rng,columns = grid)
    #Return as forecast object with specified information
    return(forecast(GHI = dataframes["GHI"],WD = dataframes["WD"],\
                    WS = dataframes["WS"],TMP = dataframes["TMP"])) 

def import_single_forecast(root,month,day,fc_time,info = ("GHI",),\
                           gridnr = "all",fc_duration = 54):
    """
    Imports single forecast from file into the forecast structure.
    
    Input:
         root: root to main svn folder with "\\" in the end.
         This will vary depending from where you run your script.
         month: Month of the year as integer (no zeroes in front).
         day: Day of the month as integer  (no zeroes in front).
         fc_time: Which forecast you want of the four given that day as string.
                  example: "00", 06", "12", "18"
                  
    Optional input:
        info: Specify which info you want from the forecast. If left returns
        forecast with GHI information. If you want other information give tuple
        with string entries discribing the info. Example: info = ("GHI","WS","WD"),
        ("WS",) and so on. If info = "all", then GHI, WS and WD will be given. 
        gridnr: Specifiy if you only want forecast from spefific grid numbers.
        OBS: Currently not implemented!
        fc_duration: Specify how many hours into the future you want your
        forecast to be as integer between 0 and 54. Default is 54 hours into the
        future. If fc_duration = 0, only the first time is included
    
    Returns: Forecast from speficied day as a forecast object. See ?forecast
             for more info 
    
    Examples:
    
    fc = import_single_forecast(root,2,4,"00"); print(fs)
    >>> Forecast in the timerange:
    >>> 2017-03-04 00:00:00 to 2017-03-06 06:00:00
    >>>
    >>> Forecast contains:
    >>>     - GHI:Global horisontal irridiance
    >>>
    >>> Forecast covers 354 grid points
    
    fs = import_single_forecast(root,3,4,"12",fc_duration = 5,info = "all"); print(fs)
    >>> Forecast in the timerange:
    >>> 2017-03-04 12:00:00 to 2017-03-04 17:00:00
    >>>    
    >>> Forecast contains:
    >>>   - GHI:Global horisontal irridiance
    >>>   - WS :Wind speed
    >>>   - WD :Wind Direction
    >>>
    >>> Forecast covers 354 grid points
    """
    if gridnr != "all":
        raise(NotImplementedError("Currently it is only possible to return \
                                  forecast for all gridnumbers.\n \
                                  Leave gridnr = \"all\""))
    if fc_duration < 1 or fc_duration > 54:
        raise(ValueError("Let fc_duration be in the range [1,54]"))
        
    dat_path = "Fortrolig_data/2017_forecast/%d/%d/" %(month,day)
    grid_path = "Fortrolig_data/forecast_grid"
    grid = sio.loadmat(root + grid_path + ".mat")['forecast_grid'].T[0]
    t0 = pd.Timestamp(year = 2017,
                      month = month,
                      day = day,
                      hour = int(fc_time))
    rng = pd.date_range(t0,periods = fc_duration+1,freq = 'H')
    if info == "all":
        info = ("GHI","WD","WS")
    data = dict.fromkeys(("GHI","WD","WS"))
    for fc_info in info:
        #Fix all this below when grid and timerange is loaded
        fc_matrix = sio.loadmat(root + dat_path + fc_info +\
                                     fc_time + '.mat')['winddirection'][:fc_duration+1]
        data[fc_info] = pd.DataFrame(fc_matrix,index = rng,columns = grid)
    return(forecast(data["GHI"],data["WD"],data["WS"]))

def import_forecast_hour(t_start,t_end,hours = "all",info = ("GHI",),\
                    grid_list = "all",sub_h_freq = 'all',\
                    sub_D_freq = 'all'):
    """
    Import serveral forecasts into a continius forecast with information in
    the speficied hours. Imports into the forecast class. 
    
    Input:
         t_start/t_end: Start/end time for your forecast as pandas timestamp.
         Select this value between 2017/01/01 and 2017/12/31. Only whole dates
         are allowed, no hours or minutes can be speficied. 
         

    Optional input:
        info: Specify which info you want from the forecast. If left returns
        forecast with GHI information. If you want other information give tuple
        with string entries discribing the info. Example: info = ("GHI","WS","WD"),
        ("WS",) and so on. If info = "all", then GHI, WS and WD will be given. 
        gridnr: Specifiy if you only want forecast from spefific grid numbers.
        If left it will import data from all grid numbers, else give a
        list/tuple/numpy array with the numbers.
        hours: Specify which timeragne of hours you want for each day.
        Ex: hours = ["04:00","22:00"] for forecasts in that timerange each day. 
        sub_h_freq: Subsample in hours/minutes. Ex: sub_h_freq = "2H" for
        samples only every 2 hours
        sub_D_freq: Subsample in days. Ex: sub_D_freq = "5D" for samples
        only every 5 days
        
    Returns: Forecast from speficied timerange as a forecast object.
    See ?forecast for more info 
    
    Examples:
    #Initialise timestamp with arguments specified. 
    t0 = pd.Timestamp(year = 2017,month = 1,day = 1), print(t0)
    >>> 2017-01-01 00:00:00
    
    #Initialise timestamp without arguments specified
    t0 = pd.Timestamp(2017,1,1); print(t0)
    >>> 2017-01-01 00:00:00

    #Import forecast with GHI information
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10)
    fc = import_forecast(t0,t1); print(fc)
    >>> Forecast in the timerange:
    >>> 2017-01-01  to 2017-01-10 in the hours
    >>> 00:00:00 to 23:00:00 every D's and every H's
    >>>
    >>> Forecast contains:
    >>>     - GHI:Global horisontal irridiance
    >>>
    >>> Forecast covers 354 grid points
    
    #You get dataframes with GHI like this
    GHI_dataframe = fc.GHI
    
    #You can view GHI as an excel file like:
    fc.SPP.to_csv("GHI.csv")
    
    #Even as an html file to view in your browser which gives a good overview
    fc.GHI.to_html("GHI.html") 
    
    #Import forecast with all information (GHI,Windspeed and Winddirection)
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10)
    fc = import_forecast(t0,t1,info = "all"); print(fc)
    >>> Forecast in the timerange:
    >>> 2017-01-01 to 2017-01-10 in the hours
    >>> 00:00:00 to 23:00:00 every D's and every H's
    >>>    
    >>> Forecast contains:
    >>>   - GHI:Global horisontal irridiance
    >>>   - WS :Wind speed
    >>>   - WD :Wind Direction
    >>>
    >>> Forecast covers 354 grid points
    
    #Import forecast at timerange on a day
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10)
    hours = ("05:00","22:00")
    fc = import_forecast(t0,t1,info = "all",hours = hours); print(fc)
    >>> Forecast in the timerange:
    >>> 2017-01-01 to 2017-01-10 in the hours
    >>> 05:00:00 to 22:00:00 every D's and every H's
    >>>    
    >>> Forecast contains:
    >>>   - GHI:Global horisontal irridiance
    >>>   - WS :Wind speed
    >>>   - WD :Wind Direction
    >>>
    >>> Forecast covers 354 grid points
    
    #Import specific grid numbers
    t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10)
    grid_list = [190,315,1413]
    fc = import_forecast(t0,t1,info = "all",grid_list = grid_list); print(fc)
    >>> Forecast in the timerange:
    >>> 2017-01-01 to 2017-01-10 in the hours
    >>> 05:00:00 to 22:00:00 every D's and every H's
    >>>    
    >>> Forecast contains:
    >>>   - GHI:Global horisontal irridiance
    >>>   - WS :Wind speed
    >>>   - WD :Wind Direction
    >>>
    >>> Forecast covers 3 grid points
    #Import using subsampling
    #t0 = pd.Timestamp(2017,1,1); t1 = pd.Timestamp(2017,1,10)
    sub_hours = "2H"; sub_days = "5D"
    fc = import_forecast(t0,t1,sub_h_freq = sub_hours,sub_D_freq = sub_days); print(fc)
    >>> Forecast at the dates:
    >>> 2017-01-01 to 2017-01-06 in the hours
    >>> 00:00:00 to 22:00:00 every 5D's and every 2H's
    >>>
    >>> Forecast contains:
    >>>   - GHI:Global horisontal irridiance
    >>> Forecast covers 354 grid points
    """
    root = return_to_root()
    #Sanitycheck for different input    
    if "data" not in os.listdir(root):
        raise(OSError("Root is noot the svn shared folder"))
    
    sanity_checks.check_timestamps(t_start,t_end)
        
    sanity_checks.check_hours(hours)
                
    sanity_checks.check_subs_args(sub_h_freq,sub_D_freq,mode="forecast")
        
    #Fetch stem data (grid) - used for sanity check but also later on
    grid_path = "data/stem_data/forecast_grid" #load grid numbers from file
    grid = sio.loadmat(root + grid_path + ".mat")['forecast_grid'].T[0]
    
    if not set(grid_list).issubset(set(grid)) and grid_list != 'all':
        raise(ValueError("One or more elements in grid_list is invalid: \
                        forecast for that grid point is not known"))
    
    
    #handle timerange
    rng,day_rng,sub_h_nr,spd,s_day0,s_day1 =\
    _handle_timerange(t_start,t_end,hours,sub_h_freq,sub_D_freq)
    
    if grid_list == "all": #Fix in order to import municipaligties instead
        grid_index = range(len(grid)) #All indicies
        grid_list = grid
    else:
        grid_index = np.in1d(grid, grid_list).nonzero()[0] #List with indicies of ch
 
    #Create data structures
    if info == "all":
        info = ("GHI","WD","WS","TMP")
    data = dict.fromkeys(info) #Big ass data matrix 
    N,M = len(rng),len(grid_index)
    #Create datamatrix for forecast types
    for key in data.keys():
        data[key] = np.zeros((N,M))
    
    folder_path = "data/"
    idx_count = 0
    for t in day_rng: #Runs thorugh every 6th element in timerange excluding the last
        data_path = "%d/%d/%d/" %(t.year,t.month,t.day) #Specific day and hour
        for key in data.keys(): #load from file and write to matrix
                data[key][idx_count*spd:idx_count*spd + spd] = \
                np.matrix(pd.read_pickle(root + folder_path + data_path +\
                                         key + 'dayH.pkl'))\
                                         [:][:,grid_index]
#                                         [s_day0:s_day1][:,grid_index][::sub_h_nr]  #incommet later
                 # [s_day0:s_day1] picks out the relevant times
             # [:,muni_index] picks out the relevant munipicilaties
        idx_count += 1
    
    #Convert to dataframe, overwrites matricies
    dataframes = dict.fromkeys(["GHI","WD","WS","TMP"]) #dictionary for dataframes
    for key in data.keys():        
       dataframes[key] = pd.DataFrame(data[key],index = rng,columns = grid_list)
       dataframes[key].columns.name = 'GRIDNR'
    #Return as forecast object with specified information
    return(forecast(GHI = dataframes["GHI"],WD = dataframes["WD"],
                    WS = dataframes["WS"],TMP = dataframes["TMP"],
                    h_freq=sub_h_freq,D_freq = sub_D_freq))


def import_muni_forecast_old(t_start,t_end, hours = "all",info = ("GHI",), 
                         muni_list = "all",
                         sub_h_freq = 'H',sub_D_freq = 'all'):
    """
    Imports forecast from list of municipilaties. See ?import_forecast for more info.
    
    Input:
        muni_list: Should be list or tuple with valid municipality numbers
        Se ?import_forecast for info about other input.
    
    Returns:
        Forecast object initialised in muni mode. 
    """
    if not isinstance(muni_list,(list,tuple)) and muni_list != 'all':
        raise(ValueError("muni_list should be list or tuple"))
            
    grid_list,conv_sheet = muni_list_to_grid_list(muni_list)
    if muni_list == 'all': #transform grid list into all because faster later
        grid_list = 'all'
        muni_list = conv_sheet.index
    
    #Use import forecast with created grid list
    fc = import_forecast(t_start, t_end, hours = hours,\
                         info = info, grid_list = grid_list,\
                         sub_h_freq = sub_h_freq,\
                         sub_D_freq = sub_D_freq)

    fc_muni = _average_grid_to_muni(fc,info,conv_sheet,muni_list)
        
    return(forecast(GHI = fc_muni["GHI"],WD = fc_muni["WD"],WS = fc_muni["WS"],
                    TMP = fc_muni["TMP"],h_freq = sub_h_freq,
                    D_freq = sub_D_freq, mode = "muni"))