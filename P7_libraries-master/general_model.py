# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:39:47 2018

@author: Tobias
"""

import import_forecast
import import_SPP
import rad_model
import pandas as pd
import numpy as np
from utilities import load_convert_muni_to_grid, get_tst_dates
from utilities import rm_dates_from_df as rm

class gm:
    """
    Some functionality that the models share.
    """
    
    def get_P_X(self,t_start,t_end,muni_list):
        """
        Get P_X model. Should be changed to updated version later on. 
        """
        P_X = rad_model.RadiationModel(t_start,t_end,
                                       forecast_zones = muni_list).GHI
        return(P_X)
         #Last bracket call the function
    
    def _window_setup(self,date,w_len,set_tst_win = True):
        """
        Performs various window setup for 2015-2017 given t
        """
        if w_len[1].days > 182:
            raise(ValueError("Secondary window is larger that a year and would yield overlap"))
        
        if w_len[1].days + w_len[0].days > 182:
            raise(ValueError("Primary and secondary window would overlap - choose smaller window lengths"))
        
        self.t_start = date - w_len[0]
        w_past = w_len[1]
        #Setup data in 2017 such that training data does not overlap with 
        #testing data
        if self.rm_tst_dat:
            self.w_2017,self.w_2017_tst = self._avoid_test_dat_window(date,w_len,
                                                                set_tst_win)
        else:
            self.w_2017 = [self.t_start,date-pd.Timedelta(days = 1)]
            self.w_2017_tst = [date,date + w_len[2]]
        
        date2015 = date.replace(year = 2015)
        w_start_2015 = max(date2015 - w_past,pd.Timestamp(2015,1,1))
        self.w_2015 = (w_start_2015,date2015 + w_past)
       
        date2016 = date.replace(year = 2016) 
        self.w_2016 = (date2016 - w_past,date2016 + w_past)
    
    def _avoid_test_dat_window(self,date,w_len,set_test_win):
        """
        Expand window in 2017 such that when tst days are removed
        the window will still be of wanted length
        
        A bit ugly - but works
        """
        
        #Expand trn window
        win_2017_rng_trn = pd.date_range(self.t_start,date - \
                                         pd.Timedelta(days = 1))
        N_trn = len(win_2017_rng_trn)
        win_2017_rng_trn = win_2017_rng_trn.drop(self.tst_dates,errors = 'ignore') #remove tst days
        #Expand iterativly by adding a day at the time untill wanted lenght
        #is met
        expand_days = 1
        while len(win_2017_rng_trn) < N_trn:
            cand_day = win_2017_rng_trn[0] - pd.Timedelta(days = expand_days)
            if cand_day not in self.tst_dates:
                win_2017_rng_trn = win_2017_rng_trn.insert(0,cand_day)
                expand_days = 1
            else:
                expand_days += 1
        if set_test_win:
        #Do the same for valdiation window
            win_2017_rng_val = pd.date_range(date,date + w_len[2])
            N_val = len(win_2017_rng_val)
            win_2017_rng_val = win_2017_rng_val.drop(self.tst_dates,errors = 'ignore') 
            expand_days = 1
            while len(win_2017_rng_val) < N_val:
                cand_day = win_2017_rng_val[-1] + pd.Timedelta(days = expand_days)
                if cand_day not in self.tst_dates:
                    win_2017_rng_val  = win_2017_rng_val.insert(len(win_2017_rng_val),cand_day)
                    expand_days = 1
                else:
                    expand_days += 1
        else: #Just set dummy window for validation
            win_2017_rng_val = [None,None]
        
        return([win_2017_rng_trn[0],win_2017_rng_trn[-1]],
               [win_2017_rng_val[0],win_2017_rng_val[-1]])
    
    def _import_SPP_trn(self,muni_list):
        """
        Import SPP from 2015 to t specified by w_len and munu list.
        Concatenates all data. 
        """
        SPP_d = dict.fromkeys((2015,2016,2017))
        for year,win in zip(SPP_d.keys(),(self.w_2015,self.w_2016,self.w_2017)):
            if hasattr(self,'SPP_in_preload'):
                if muni_list == self.muni_in:
                    SPP_d[year] = self.SPP_in_preload.loc[win[0]:win[1]\
                      +pd.Timedelta(days = 1)][muni_list]
                else:
                    SPP_d[year] = self.SPP_out_preload.loc[win[0]:win[1]\
                      +pd.Timedelta(days = 1)][muni_list]
            else:
                SPP_d[year] = import_SPP.import_SPP(win[0],win[1],
                                        muni_list = muni_list,
                                        mode = 'remove_outliers').SPP
     
        SPP_train = SPP_d[2015].append([SPP_d[2016],SPP_d[2017]])
        #Remove tst days 
        if self.rm_tst_dat:
            SPP_train = rm(SPP_train,self.tst_dates)
        return(SPP_train)
    
    def _import_SPP_tst(self,muni_list):
        """
        Import test SPP from date to specified by w_len and muni list.
        """
        if hasattr(self,'SPP_in_preload'):
            if muni_list == self.muni_in:
                SPP_train = self.SPP_in_preload.loc[self.w_2017_tst[0]:\
                        self.w_2017_tst[1]+pd.Timedelta(days = 1)][muni_list]
            else:
                SPP_train = self.SPP_out_preload.loc[self.w_2017_tst[0]:\
                        self.w_2017_tst[1]+pd.Timedelta(days = 1)][muni_list]
        else:
            SPP_train = import_SPP.import_SPP(self.w_2017_tst[0],self.w_2017_tst[1],
                                          muni_list = muni_list,
                                          mode = 'remove_outliers').SPP
        if self.rm_tst_dat:
            SPP_train = rm(SPP_train,self.tst_dates)
        return(SPP_train)
        
    def _import_PX_trn(self,muni_list):
        """
        Import P_X from 2015 to t specified by w_len and munu list.
        Concatenates all data. 
        """
        P_X_d = dict.fromkeys((2015,2016,2017))
        for year,win in zip(P_X_d.keys(),(self.w_2015,self.w_2016,self.w_2017)):
            if hasattr(self,'P_X_preload'):
                P_X_d[year] = self.P_X_preload.loc[win[0]:win[1]\
                      +pd.Timedelta(days = 1)][muni_list]
            else:
                P_X_d[year] = self.get_P_X(win[0],win[1],muni_list)
        P_X = P_X_d[2015].append([P_X_d[2016],P_X_d[2017]])
        if self.rm_tst_dat:
            P_X = rm(P_X,self.tst_dates)
        return(P_X)
    
    def _import_PX_tst(self,muni_list):
        """
        Import test PX from date to specified by w_len and muni list.
        """
        if hasattr(self,'P_X_preload'):
                P_X_tst = self.P_X_preload.loc[self.w_2017_tst[0]:\
                         self.w_2017_tst[1]+pd.Timedelta(days = 1)][muni_list]
        else:
            P_X_tst = self.get_P_X(self.w_2017_tst[0],self.w_2017_tst[1],
                                 muni_list)
        if self.rm_tst_dat:
            P_X_tst = rm(P_X_tst,self.tst_dates)
        return(P_X_tst)
    
    def _import_fc_trn(self,muni_list,info = ['GHI']):
        """
        Import forecast data from 2015 to t specified by w_len and munu list.
        Concatenates all data. 
        """
        
        #Import forecast from info
        fc_d = dict.fromkeys((2015,2016,2017))
        for year,win in zip(fc_d.keys(),(self.w_2015,self.w_2016,self.w_2017)):
            if hasattr(self,info[0] + "_preload"):
                fc_info = getattr(self,info[0] + "_preload").loc[win[0]:\
                         win[1]+pd.Timedelta(days = 1)][muni_list]
                #Wrap in fc class
                if info[0] == 'GHI':
                    fc_d[year] = qfc(GHI = fc_info)
                elif info[0] == 'WS':
                    fc_d[year] = qfc(WS = fc_info)
                elif info[0] == 'WD':
                    fc_d[year] = qfc(WD = fc_info)
            else:
                fc_d[year] =  import_forecast.import_muni_forecast(win[0],
                                                  win[1],
                                                  info = info,
                                                  muni_list = muni_list,
                                                  sub_h_freq = '15min')
        #Cast into dictionary
        fc = dict.fromkeys(info)
        for fc_type in info:
            fc_2015 = getattr(fc_d[2015],fc_type)
            fc_2016 = getattr(fc_d[2016],fc_type)
            fc_2017 = getattr(fc_d[2017],fc_type)
            fc[fc_type] = fc_2015.append([fc_2016,fc_2017])
            if self.rm_tst_dat:
                fc[fc_type] = rm(fc[fc_type],self.tst_dates)
        return(fc)
    
    def _import_fc_tst(self,muni_list,info = ['GHI']):
        """
        Import test fc from date to specified by w_len and muni list.
        """
        if hasattr(self,info[0] + "_preload"):
                fc_d = getattr(self,info[0] + "_preload").\
                loc[self.w_2017_tst[0]:\
                         self.w_2017_tst[1]+\
                         pd.Timedelta(days = 1)][muni_list]
                #Wrap in fc class
                if info[0] == 'GHI':
                    fc_d = qfc(GHI = fc_d)
                elif info[0] == 'WS':
                    fc_d = qfc(WS = fc_d)
                elif info[0] == 'WD':
                    fc_d = qfc(WD = fc_d)
        else:
            fc_d = import_forecast.import_muni_forecast(self.w_2017_tst[0],
                                                  self.w_2017_tst[1],
                                                  info = info,
                                                  muni_list = muni_list,
                                                  sub_h_freq = '15min')
        fc = dict.fromkeys(info)
        for fc_type in info:
            fc = getattr(fc_d,fc_type)
            if self.rm_tst_dat:
                fc = rm(fc,self.tst_dates)
        return(fc)
        
    def _muni_intrepeter(self,muni_input):
        """
        Handles muni input as either list (or array like)
        or some predefined strings.
        """
        
        if isinstance(muni_input,(list,tuple,np.ndarray)):
            return(muni_input)
        elif muni_input == "all":
            #Fist 3 are sum or municipalities
            if not hasattr(self,'muni_info'):
                self._load_muni_info()
            return(list(self.muni_info.index[3:]))
        elif muni_input == "DK":
            return([0])
        elif muni_input == "zones": #Load east and west Denmark
            return([1,2])
        else:
            raise(ValueError("Invalid muni_in. Specify either a list with the \
                             municipalities, \"all\", \"DK\" or \"zones\"."))
        
    def _load_muni_info(self):
        self.muni_info = load_convert_muni_to_grid()
        
    def _load_all(self,SPP_in_dat,muni_in,muni_out):
        t_start = pd.Timestamp(2015,1,1)
        t_end = pd.Timestamp(2017,12,31)
        #Load SPP in all cases
        self.SPP_in_preload = import_SPP.import_SPP(t_start,t_end,
                                        muni_list = muni_in,
                                        mode = 'remove_outliers').SPP                                         
        if not set(self.muni_out).issubset(self.muni_in): 
            self.SPP_out_preload = import_SPP.import_SPP(t_start,t_end,
                                        muni_list = muni_out,
                                        mode = 'remove_outliers').SPP 
        else:
            self.SPP_out_preload = pd.DataFrame(self.SPP_in_preload[muni_out])
        
        if 'P_X' in SPP_in_dat:
            self.P_X_preload = self.get_P_X(t_start,t_end,muni_in)
        
        if 'GHI' in SPP_in_dat:
            self.GHI_preload = import_forecast.import_muni_forecast(t_start,
                                                  t_end,
                                                  info = ['GHI'],
                                                  muni_list = muni_in,
                                                  sub_h_freq = '15min').GHI
        if 'WS' in SPP_in_dat:
            self.WS_preload = import_forecast.import_muni_forecast(t_start,
                                                  t_end,
                                                  info = ['WS'],
                                                  muni_list = muni_in,
                                                  sub_h_freq = '15min').WS
        if 'WD' in SPP_in_dat:
            self.WD_preload = import_forecast.import_muni_forecast(t_start,
                                                  t_end,
                                                  info = ['WD'],
                                                  muni_list = muni_in,
                                                  sub_h_freq = '15min').WD

class qfc: #quick fc class
    """
    Quick and dirty class that acts like forecast class - #hack
    """
    def __init__(self,GHI = None,WS = None,WD = None):
        self.GHI = GHI
        self.WS = WS
        self.WD = WD