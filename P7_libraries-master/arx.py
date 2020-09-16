# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 16:09:48 2018

@author: Tobias
"""

import import_SPP
import pandas as pd
import numpy as np
import datetime as dt
from general_model import gm
from datetime import time as d_time
from utilities import Timedelta, round15, nonzero_df_replace, add_hour,\
get_tst_dates,conv_and_round15_df
from forecast_dataframe import fc_df

class ARX(gm):
    def __init__(self, date, muni_list,w_len = pd.Timedelta(days = 30),
                 w_len_past_years = pd.Timedelta(days = 15), t_freq ='15min',
                 h_freq = "H",mode = "simple",horizon = d_time(1),
                 rm_tst_dat = True):
        """
        Initialises and trains ARX model given date and window length
        
        Parameters
        ----------
        date : pd.Timestamp
            Last date used for training data
        muni_list : list
            List of municipalities model is trained for.
        t_freq  : str in pandas frequency format
            Time resolution of model. Defaults to 15min
        h_freq  : str in pandas frequency format
            Time resolution of horizons - ie. How many horizons should be 
            available up to 6 hours . Defaults to 1 hour
        w_len : pd.Timedelta
            Window length used when training the model
        mode: str
            "simple" mode will forecast SPP with the horizon specified in
            horizon argument. "all" mode will forecast SPP from 15 min to
            6 hours into the future for all times. 
        horizon: datetime.time
            How many hours and minutes into the future model predicts. 
            This argument is only used in the simple mode. 
        
        
        Yet to be implemented:
            - Call model for serveal horizons
            - Support training the model mid day (include samples up to time 
            of day)
            - Handle traning model for serveal municipalities
            - Implement import forecast in simmulation modes
        
        Returns ARX object to be called for different ranges.
        """
        
#        if time_res != "15min":
#            raise(NotImplementedError("Pt. only timeresultion of 15 minutes supported"))
#
        if len(muni_list) != 1:
            raise(NotImplementedError("Pt. only fitting for one muni is supported"))
        
        if date < pd.Timestamp(2017,1,1):
            raise(ValueError("ARX model is only supported for dates in 2017"))
        
        #Set first date of training data depending on w_len
        self.rm_tst_dat = rm_tst_dat
        self.tst_dates = get_tst_dates()
        self.date = date
        self.muni_list = muni_list
        self.t_freq = t_freq
        self.h_freq = h_freq
        self.mode = mode
        self.horizon = horizon
        #assigns w_2015,w_2016 and w_2017 and t_start as attributes to self
        self._window_setup(self.date,(w_len,w_len_past_years),
                           set_tst_win = False)
        
        #SPP to train model
        self.SPP_train = self._import_SPP_trn(self.muni_list)
        #P_X model used to train P_ARX
        self.P_X = self._import_PX_trn(muni_list)
        
        #Min max hours for fitting depending on time of year
        data_path = 'data/stem_data'
        self.hours_fit = pd.read_pickle(data_path +\
                                        "/SPP_min_max_hours_15_to_17.pkl")
        self.hours_fit_rnd = pd.DataFrame(self.hours_fit,copy = True)
        conv_and_round15_df(self.hours_fit_rnd)
        #Autoregressive part - used for response variable
        P_AR = self.P_X - self.SPP_train
        
        #Perform the actual training by determining weights
        self.w = self._train_P_X(P_AR)
        
    def _gen_fit_range(self,horizon,dates):
        """
        Generate time range for training data when horizon, sunrice and sunset
        is accounted for. 
        """
        date_hours = dict.fromkeys(dates)
        dates_temp = dates
        for day in dates_temp:
            sunrise,sunset = self.hours_fit_rnd.loc[day]
            days = pd.Timestamp(day)
            t0 = days + pd.Timedelta(hours = sunrise.hour,
                                    minutes = sunrise.minute) \
                + pd.Timedelta(hours = horizon.hour,minutes = horizon.minute)
            t1 = days + pd.Timedelta(hours = sunset.hour,
                                    minutes = sunset.minute)
            
            date_hours[day] = pd.date_range(t0,t1,freq = "15min")
        #Concatenate dictionary 
        return(pd.DatetimeIndex(np.hstack(date_hours.values())))
   
    def _train_P_X(self,P_AR):
        
        #Loads earliest sunrise and latest sundown for training data and 
        #Rounds to the nearest quarter
        #hours list is only for 2017 but assumed the sames count for 2015-2016
        t_min = self.hours_fit[self.t_start:self.date]['t0'].min()
        t_max = self.hours_fit[self.t_start:self.date]['t1'].max()
        t_min = dt.datetime.strptime(t_min, '%H:%M:%S').time()
        t_max = dt.datetime.strptime(t_max, '%H:%M:%S').time()
        t_min = round15(t_min, method = 'floor')
        t_max = round15(t_max, method = 'ceil')
        
        
        #Time of day for which the model will be fitted
        fit_times = pd.date_range(pd.Timestamp(2017,1,1,t_min.hour,t_min.minute),
                                  pd.Timestamp(2017,1,1,t_max.hour,t_max.minute),
                                  freq = self.t_freq).time
        #Horizons for which the model will be fitted - simple only trains for 
        #1 horizon while 'all' trains from 15 minutes to 6 hours
        if self.mode == "simple":
            self.horizons = pd.date_range(pd.Timestamp(2017,1,1,self.horizon.hour,
                                                  self.horizon.minute),
                                     pd.Timestamp(2017,1,1,self.horizon.hour,
                                                  self.horizon.minute),
                                     freq = self.h_freq).time   
        elif self.mode == "all":
            self.horizons = pd.date_range(pd.Timestamp(2017,1,1),
                                         pd.Timestamp(2017,1,1,6),
                                         freq = self.h_freq)[1:].time
            #The [1:] drops fitting for horizon = 0 minutes
                
        N,M = self.horizons.size, fit_times.size
        w = np.zeros((N,M)) #Matrix with weights
        
        
        #Some datapoints will allways be missing the the beginning and end
        #when fitting due to the model beeing an autoregressive one
        #These two arguments and the "cut" variables prevents dimention error 
        #when using least squares
        first = P_AR.index[0]
        last = P_AR.index[-1]

        
        for t in range(M):
            time = fit_times[t]
            #Picks out all samples at "time"
            X = P_AR.iloc[P_AR.index.indexer_at_time(time)]
            for h in range(1,N+1):
                #If time overflows into next day fitting does not occur by default
                time_horizon,next_day = add_hour(time,self.horizons[h-1]) #h is 15 minutes
            
                if time_horizon < t_max and not next_day: #No need to fit after sundown
                    #Picks corrosponding regressor depending on time and horizon
                    y = P_AR.iloc[P_AR.index.indexer_at_time(time_horizon)]
                    cut_0 = last - pd.Timedelta(minutes = 15*h)
                    cut_1 = first + pd.Timedelta(minutes = 15*h)
                    X = X[:cut_0]
                    y = y[cut_1:]
                    #Fit using least squares
                    w[h-1,t] = np.linalg.lstsq(X.values,y.values,
                     rcond = None)[0][0][0]
                    
        w = pd.DataFrame(w)
        w.index = self.horizons; w.index.name = "Horizon" 
        w.columns = fit_times; w.columns.name = "Time of day"
        return(w)
             

    def __call__(self,t_start,t_end,return_mode = 'model'):
        """
        Calls the ARX model and returns it. 
        
        Parameters
        ----------
        t_start/t_end : pd.Timestamp
        return_mode : str
            Returns only P_ARX in "model" mode. Returns SPP, P_X and P_ARX in 
            "all" mode. 
        """
        test_train_day_delta = (t_start - self.date).days
        if test_train_day_delta > 10:
            print("Fitting model for %d day later than used traning data")
        if test_train_day_delta < 0:
            print("Fitting model for data used as training data")
        
        if t_start == pd.Timestamp(2015,1,1):
            raise(ValueError("Data unavailable chose a time later then 1/1-2015 00:00"))

        rng = pd.date_range(t_start,t_end,freq = self.t_freq)
        #A few extra samples are needed in the beginning 
        t_start_early = pd.Timestamp((t_start - Timedelta(self.horizon)).date())       
        t_end_date = pd.Timestamp(t_end.date()) #strip of hours - used to import
        
        #Get SPP and P_X model to compute the P_AR at 0 horizon used in the model
        SPP_test = import_SPP.import_SPP(t_start_early,t_end_date,\
                                         muni_list = self.muni_list,
                                         sub_h_freq = self.t_freq)
        P_X_test = self.get_P_X(t_start_early,t_end_date,self.muni_list)
        
        P_AR_test = SPP_test.SPP - P_X_test 
        
        P_AR = pd.DataFrame(np.zeros((rng.size,self.horizons.size)),
                            index = rng, columns = self.horizons)
        P_AR.index.name = "Time"; P_AR.columns.name = "Horizon"
        
        for time in rng:
            t = time.time()
            if t in self.w.columns: #Only fit when sun is up
                for horizon in self.horizons:
                    w_t_h = self.w.loc[horizon,t]
                    P_AR_val = P_AR_test.loc[time].values[0]
                    P_AR.loc[time,horizon] = P_AR_val*w_t_h
        
        P_ARX = P_AR
        for h in self.horizons:
            t_h_start = t_start + Timedelta(h)
            t_h_end = t_end + Timedelta(h)
            P_ARX[h] = P_ARX[h].add(P_X_test[t_h_start:t_h_end].\
                 values.T[0],axis = 'index')
            #Replace by P_X where P_ARX < 0
            P_ARX[h] = nonzero_df_replace(P_ARX.loc[:,[h]],
                 P_X_test[t_h_start:t_h_end])
        
        #Replace columns with multiindex
        m_repeat = np.repeat(self.muni_list[0],self.horizons.size)
        col_multi = list(zip(*(m_repeat,self.horizons)))
        pd_lab = pd.MultiIndex.from_tuples(col_multi,
                                               names=['Muni','Horizon'])
        P_ARX = pd.DataFrame(P_ARX.values,index = P_ARX.index,
                              columns = pd_lab)

        P_ARX_fc_df = fc_df(P_ARX,freq = self.h_freq)
        if return_mode == "model":
            return(P_ARX_fc_df)
        elif return_mode == "all":
            return(P_ARX_fc_df,
                   SPP_test.SPP[t_start:t_end],
                   P_X_test[t_start:t_end],P_AR)