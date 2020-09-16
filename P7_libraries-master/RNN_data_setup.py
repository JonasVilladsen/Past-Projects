# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:08:03 2018

@author: Tobias
"""

import pandas as pd
from general_model import gm
from utilities import dates_from_DF,conv_and_round15_df,merge_multiindex, \
get_tst_dates
from datetime import time as d_time
import numpy as np

class setup(gm):
    """
    Imports data under different modes and calls preprocesessing functions if
    prompted. Also have a batch generator function for training NN's. 
    
    Examples
    --------
    #Bacis example importing and generating data    
    >>> t0 = pd.Timestamp(2017,7,24) #Simmulated data
    >>> w_len = [30,15,10] #Windows for (2017,2015/2016,Test_data)
    >>> muni_in = 'all'; muni_out = 'DK'
    #Lists with muni numbers are also supported
    >>> data = setup(t0,muni_in,muni_out,process_mode='raw') #load data
    >>> df_in,df_out = data.horizon_setup(input_win = 2) #Prepare for RNN

    """
    
    def __init__(self, date, muni_in, muni_out, w_len=(30,15,10),
                 process_mode='raw', SPP_in_dat=None, eps=1e-3,
                 preload = False,tst_days_per_month = 5,rm_tst_dat = True,
                 verbose = True):
        """
        Imports data for training and testing RNN. 
        
        Parameters
        ----------
        t : pd.Timestamp (as whole date)
            Simmulated timestamp. Training data are imported before t, and 
            testing data are imported after.
        muni_in : 1d array-like 
            List with municipalities that serve as input to the RNN.
        muni_out : 1d array-like
            List with municiaplities that will be output for the RNN.
        w_len : length 3 array-like with integer or pd.Timedelta as content
                The string 'all' is also allowed.
            3 window lengths:
                w_len[0]: Is how many days in 2017 before t should be imported
                w_len[1]: Yelds the secondary windows
                W_2015 = [t - 2_years - win_len[2],t - 2_years + win_len[2]]
                and equavelently for 2016. So se secondary windows will be 
                2*length win_len[2] each.
                w_len[2]: Is how many days in 2017 after t should be imported
                as testing data
            If win_len == 'all' all data from 1/1-2015 to t will be imported
        process_mode : string or None
            'raw' mode imports raw SPP and GHI as input
            'processed' mode imports P_AR and P_X as input
            'all' mode import all datatypes (SPP,GHI,P_AR,P_X,WD,WS)
        SPP_in_dat : List or None
            If specified: List of types of data that should be included.
            If unspecified (None): Input data will be as descibed in 
            process_mode
        preload : bool
            If True, all data are preloaded into ram in order to change 
            dates faster
        """
        
        #Run sanity checks
        if date < pd.Timestamp(2017,1,1):
            raise(ValueError("LSTM model is only supported for dates in 2017"))
        
        if len(w_len) != 3 and w_len != 'all':
            raise(IndexError("win_len should be of length 3. \
                             See ?setup for more details"))
        
        if SPP_in_dat is None:
            if process_mode == 'raw':
                SPP_in_dat = ['SPP','GHI']
            elif process_mode == 'processed':
                SPP_in_dat = ['P_AR','P_X']
            elif process_mode == 'raw + processed':
                SPP_in_dat = ['SPP', 'GHI', 'P_AR', 'P_X']
            elif process_mode == 'all':
                SPP_in_dat = ['SPP','GHI','P_AR','P_X','WS','WD']
            else:
                raise(ValueError("Invalid mode"))
        
        self.date = date
        self.muni_in = self._muni_intrepeter(muni_in)
        self.muni_out = self._muni_intrepeter(muni_out)
          
        
        #Preloads all data if prompted! 
        if preload:
            #Load all data into self
            print("Preloading data, be patient")
            self._load_all(SPP_in_dat,self.muni_in,self.muni_out)
            print("Done all preloading")
        
        #Get tst data dats
        self.tst_dates = get_tst_dates(dpm = tst_days_per_month)
        self.rm_tst_dat = rm_tst_dat #to remove tst data or not
        
        #Set first date of training data depending on w_len
        self.w_len = [0,0,0]
        for i in range(3):
            if type(w_len[i]) is int:
                self.w_len[i] = pd.Timedelta(days = w_len[i])
            elif type(w_len[i]) is pd._libs.tslib.Timedelta:
                self.w_len[i] = w_len[i]
            else:
                raise(TypeError("w_len should be array like with int or pandas Timedelta as constent"))
        
        #assigns w_2015,w_2016 and w_2017 and t_start as attributes to self
        self._window_setup(self.date,self.w_len)
        
        #SPP data used to train model
        self.SPP_in = self._import_SPP_trn(self.muni_in)
        self.SPP_in_tst = self._import_SPP_tst(self.muni_in)
        
        #If muni_out contains munis not in muni_in (like whole denmark)
        if not set(self.muni_out).issubset(self.muni_in): 
            self.SPP_out = self._import_SPP_trn(self.muni_out)
            self.SPP_out_tst = self._import_SPP_tst(self.muni_out)
        
        else:
            self.SPP_out = pd.DataFrame(self.SPP_in[self.muni_out])
            self.SPP_out_tst = pd.DataFrame(self.SPP_in_tst[self.muni_out])
        
        
        #When calling data some may be outliers and a warning is issued first
        # time this data is called
        self.verbose = verbose
        
        #Average for SPP_out usefull in some cases like fitting
        self.SPP_out_avg = self.SPP_out.where(self.SPP_out > eps).mean()
        #Import different types if input data specified in arguments
        #If SPP_dat_in is not specified load some predefined input modes
        
        self.avail_dat = []
        if 'SPP' in SPP_in_dat:
            self.avail_dat.append('SPP')
        
        if 'GHI' in SPP_in_dat:
            self.GHI = self._import_fc_trn(self.muni_in)['GHI']
            self.GHI_tst = self._import_fc_tst(self.muni_in)
            self.avail_dat.append('GHI')
        
        if 'P_AR' in SPP_in_dat:
            self.P_X = self._import_PX_trn(self.muni_in)
            self.P_X_tst = self._import_PX_tst(self.muni_in)         
            self.P_AR = self.P_X - self.SPP_in
            self.P_AR_tst = self.P_X_tst - self.SPP_in_tst
            self.avail_dat.append('P_AR')
        
        if 'P_X' in SPP_in_dat:
            if not hasattr(self,'P_X'): #Can occour
                self.P_X = self._import_PX_trn(self.muni_in)
                self.P_X_tst = self._import_PX_tst(self.muni_in)
            self.avail_dat.append('P_X')
            
        if 'WS' in SPP_in_dat:
            self.WS = self._import_fc_trn(self.muni_in,info = ['WS'])['WS']
            self.WS_tst = self._import_fc_tst(self.muni_in,info = ['WS'])
            self.avail_dat.append('WS')
        
        if 'WD' in SPP_in_dat:
            self.WD = self._import_fc_trn(self.muni_in,info = ['WD'])['WD']
            self.WD_tst = self._import_fc_tst(self.muni_in,info = ['WD'])
            self.avail_dat.append('WD')
        

        #Min max hours for fitting depending on time of year - possibly this
        #needs to be changed. 
        self.dates = dates_from_DF(self.SPP_in)
        self.dates_tst = dates_from_DF(self.SPP_in_tst)
        data_path = 'data/stem_data'
        self.hours_fit = pd.read_pickle(data_path +\
                                    "/SPP_min_max_hours_15_to_17.pkl")
        
        conv_and_round15_df(self.hours_fit)
    
    def LSTM_setup(self,horizon,input_win,batch_size,w_len = None,
                   split_input = False,DTs = False,
                   include_val = True):
        """
        Sets up imported data to be recieved to LSTM network. Returns the 
        following in a dictionary with the following items:
            - batch_generator
            - steps per epoc
            - Training data as two dataframes in a list [input_df,target_df]
            - Testing data as tensor 
            - Testing data as two dataframes in a list [input_df,target_df]
            - Input dimension
        
        Parameters
        ----------
        horizon : datetime.time
            How far into the future to be predicted 
        input_win : int >= 1 or array like or matrix
            How many time steps are into the past are made available for the
            model. If list each input is assigned a different number of 
            allowed samples. It should be of order:
                raw_mode:       (#SPP,#GHI,(#WS),(#WD))
                processed_mode: (#P_res,#P_X,(#WS),(#WD))       
        batch_size: int
             Wanted number of days in each batch
        input_win : int >= 1 or array like or matrix
            How many time steps are into the past are made available for the
            model. If list each input is assigned a different number of 
            allowed samples. It should be of order:
                raw_mode:       (#SPP,#GHI,(#WS),(#WD))
                processed_mode: (#P_res,#P_X,(3WS),(3WD))
        w_len : list with int as in __init__
            possible update to w_len in __init__, but window lengths of smaller
            lengths than before
        split_input: bool
            In the Zheng-Hua model input is split into SPP and GHI seperately
        DTs : bool
            In DTs (deep transition shortcut) mode the target date nees to
            be matrix instead of tensor for some reason. Defauls to False,
            set True when using DTs model. 
        """
        #In Tan model, inputdata is split into two. 
        #Get the correct function to get input data
        if split_input: #dat_in will be list with SPP and GHI
            input_func = self._horizon_split
        else:
            input_func = self._horizon_setup
        
        #Get the data setup
        dat_in,target = input_func(horizon,input_win)
        #Validation data
        if include_val:
            dat_val = input_func(horizon,input_win,mode = 'tst')
            dat_val_tns, target_val_tns = input_func(horizon,input_win,mode = 'tst',
                                             tensor = True)
    
            #If outliers N_dates can have changed so value is updates
            if split_input:
                dates = np.unique(dat_in[0].index.date)
                _, _, input_dim = dat_val_tns[0].shape
            else:
                dates = np.unique(dat_in.index.date)
                _, _, input_dim = dat_val_tns.shape
            #Reshape validation data to tensor
            N,_,output_dim = target_val_tns.shape
    
            #In deep transition mode target data needs to be matrix
            if DTs:     
                target_val_tns = target_val_tns.reshape(N,output_dim)
            else:    
                target_val_tns = target_val_tns.reshape(N,1,output_dim)
        else:
            dat_val = None
            dat_val_tns = target_val_tns = None
            if split_input:
                dates = np.unique(dat_in[0].index.date)
                input_dim = dat_in[0].shape[1]
            else:
                dates = np.unique(dat_in.index.date)
                input_dim = dat_in.shape[1]
            output_dim = target.shape[1]
            
        #Setup batch generator
        steps_pr_epoc  = self.get_steps_per_epoc(batch_size,dates)
        bg = self._batch_generator(dat_in,target,batch_size,
                        steps_pr_epoc,split = split_input,DTs = DTs)
        
        return{'batch_generator':bg,'steps_per_epoc':steps_pr_epoc,
               'trn_dat': [dat_in,target],
               'validation': dat_val,
               'validation_tns': [dat_val_tns,target_val_tns],
               'input_dim': input_dim,
               'output_dim': output_dim}
    
    
    def _horizon_setup(self,horizon = d_time(1,0),input_win = 1,tensor = False,
                      mode = 'trn'):
        """
        Given horizon for forecast, setup input data in dataframe such that
        each is the time to be fitted for and the available data at that point.
        Also outputs target data corrosponding to input data. Note that the 
        range to be fitted for depends on a list with when in the morning and
        evenening solar power should be available each day (which is possibly
        too generous). 
        
        Parameters
        ----------
        mode : str
            Which data to return, either training og testing data
        tensor : bool
            If True, return numpy tensor, else return dataframe
        
        Se ?self.LSTM_setup for more info on parameters
        
        """
           #Get times for which model should train for
        fit_range = self._gen_fit_range(horizon,mode)
        
        #Get lables for input data 
        N_d = len(self.avail_dat)
        #If integer repeat for all data_types
        if type(input_win) is int:
            input_win = np.repeat(input_win,N_d)
        
        if isinstance(input_win,(list,tuple)):
            input_win = np.array(input_win)
            
        #If array like - repeat for all muni's (Becomes matrix)
        if input_win.ndim == 1:
            n = len(self.muni_in)
            input_win = np.repeat(input_win,n,axis = 0).reshape(N_d,n).T
        
        #Create sensable columns names
        in_lab = []    
        for muni_in in input_win: #Loop over how much input for muni
            for d_type,i in zip(self.avail_dat,range(N_d)): #loop over datatype
                extra_lag = 0
                if d_type == 'SPP' or d_type == 'P_AR':
                    extra_lag = 60*horizon.hour + horizon.minute
                for lag in range(muni_in[i]): #Loop over each lag
                    #SPP is autoregressive part and need an extra lag
                    if lag + extra_lag == 0:
                        in_lab.append(d_type + "(t)")
                    else:
                        in_lab.append(d_type + "(t-%d)" %(15*lag + extra_lag))
        
        #Create multiindex
        muni_repeat = np.repeat(self.muni_in,np.sum(input_win,axis = 1),axis = 0)
        final_lab  = list(zip(*(muni_repeat,in_lab)))
        pd_lab = pd.MultiIndex.from_tuples(final_lab, names=['Municipality',
                                                             'Data type'])
        df_in = pd.DataFrame(index = fit_range,columns = pd_lab)
 
        #Add input data
        count_col = 0 #count which column should be filled ind
        for muni_i in range(len(self.muni_in)):
            muni_nr = self.muni_in[muni_i]
            for d_type, i in zip(self.avail_dat,range(N_d)):
                if d_type != 'SPP' and d_type != 'P_AR':
                    #We have weather forecast well in advance so no need to 
                    #consider any horizon
                    horizon_temp = d_time(0,0)
                else:
                    horizon_temp = horizon
                    if d_type == 'SPP': #Only nesseary for direct SPP input
                        d_type += '_in'
                if mode == 'tst': #Ensures tst data are fetched
                    d_type += '_tst'
                for lag in range(input_win[muni_i,i]):
                    #Subtract horizon times to get correct data
                    col_times = fit_range - \
                                pd.Timedelta(hours = horizon_temp.hour,
                                             minutes = 15*(lag) + \
                                             horizon_temp.minute)

                    col_nr = in_lab[count_col]
                    df_in[(muni_nr,col_nr)] = getattr(self,d_type)\
                                                .loc[col_times,muni_nr].values
                    count_col += 1
        
        if mode == 'trn':
            df_out = self.SPP_out.reindex(fit_range)
        else:
            df_out = self.SPP_out_tst.reindex(fit_range)
        

        #Remove rows with unavailable data (can happen if window length is big
        #or outliers occur)
        if df_in.isnull().values.any(): #Detect any nan values
            if not self.verbose:
                print("Warning: %d rows removed due to outliers" \
                  %(len(np.unique(np.where(df_in.isnull().values)[0]))))
                self.outliers_warn = False
            
            drop_idx = df_in.index[np.where(df_in.isnull().values)[0]]
            df_in.dropna(inplace=True) #Removes rows "nan"
            df_out.drop(index = drop_idx, inplace = True)
            
            #Update dates when outliers are found
            if mode == 'trn':
                self.dates = dates_from_DF(self.SPP_in)
            else:
                self.dates_tst = dates_from_DF(self.SPP_in_tst)
        
        if tensor:
            N,M = df_in.shape
            df_in = df_in.values.reshape((N,1,M))
            df_out = df_out.values.reshape((N,1,len(self.muni_out)))
        
        return(df_in,df_out)
        
    def get_steps_per_epoc(self,batch_size,date_list):
        """
        Given number of loaded days in and batch size, calculate steps
        per epoc. 
        
        Parameters
        ----------
        batch_size: int
            Size of each batch in days
        date_list : array like
            list of days (can vary depending on outliers)
        """
        return(int(np.ceil(len(date_list)/batch_size)))
    
    def _batch_generator(self,dat_in,target,batch_size,
                        steps_pr_epoc,split = False,DTs = False):
        """
        Loops generates batches for loaded data. In batch mode is "days"
        batch_mode generate batches over a cirtain number of days, and if 
        "samples", generate batches of a fix size (except perhaps the last batch)
        
                        ___,,___
                   _,-='=- =-  -`"--.__,,.._
                ,-;// /  - -       -   -= - "=.
              ,'///    -     -   -   =  - ==-=\`.
             |/// /  =    `. - =   == - =.=_,,._ `=/|
            ///    -   -    \  - - = ,ndDMHHMM/\b  \\
          ,' - / /        / /\ =  - /MM(,,._`YQMML  `|
         <_,=^Kkm / / / / ///H|wnWWdMKKK#""-;. `"0\  |
                `""QkmmmmmnWMMM\""WHMKKMM\   `--. \> \
                      `""'  `->>>    ``WHMb,.    `-_<@)
                                        `"QMM`.
                                           `>>>
        Parameters
        ----------
        see ?self.LSTM_setup for parameter description
        """
        #Get dates
        if split:
            dates = np.unique(dat_in[0].index.date)
        else:
            dates = np.unique(dat_in.index.date)
        while True:
            for step in range(steps_pr_epoc):
                d0 = dates[batch_size*step]
                if step == steps_pr_epoc - 1:
                    d1 = None
                else:
                    d1 = dates[batch_size*(step+1)]
                if split:
                    #Extact SPP and GHI
                    SPP_in = dat_in[0].loc[d0:d1].values
                    GHI_in = dat_in[1].loc[d0:d1].values
                    #Then reshape
                    N, SPP_dim = SPP_in.shape
                    GHI_dim = GHI_in.shape[1]
                    SPP_in = SPP_in.reshape(N,1,SPP_dim)
                    GHI_in = GHI_in.reshape(N,1,GHI_dim)
                    #Finally concatenate
                    batch_in = [SPP_in,GHI_in] 
                else: #All input as one
                    batch_in = dat_in.loc[d0:d1].values
                    N,M = batch_in.shape 
                    batch_in = batch_in.reshape(N,1,M)
                #extract and reshape target values
                if DTs:
                    batch_target = target.loc[d0:d1].values.reshape(N,
                                             len(self.muni_out))
                else:
                    batch_target = target.loc[d0:d1].values.reshape(N,1,
                                             len(self.muni_out))
                yield(batch_in,batch_target)
        
            
    def _gen_fit_range(self,horizon,mode = 'trn'):
        """
        Generate time range for training data when horizon, sunrice and sunset
        is accounted for. 
        """
        if mode == 'trn':
            date_hours = dict.fromkeys(self.dates)
            dates_temp = self.dates
        else: #validation data
            date_hours = dict.fromkeys(self.dates_tst)
            dates_temp = self.dates_tst
        for day in dates_temp:
            sunrise,sunset = self.hours_fit.loc[day]
            days = pd.Timestamp(day)
            t0 = days + pd.Timedelta(hours = sunrise.hour,
                                    minutes = sunrise.minute) \
                + pd.Timedelta(hours = horizon.hour,minutes = horizon.minute)
            t1 = days + pd.Timedelta(hours = sunset.hour,
                                    minutes = sunset.minute)
            
            date_hours[day] = pd.date_range(t0,t1,freq = "15min")
        #Concatenate dictionary 
        return(pd.DatetimeIndex(np.hstack(date_hours.values())))
        
    def _horizon_split(self,horizon = d_time(1,0),input_win = 1,tensor = False,
                      mode = 'trn'):
        
        """
        Get input and target where GHI and SPP are in seperate dataframes
        
        Se ?horizon_setup for parameter discription. 
            
        
        """
        
        dat_in, target = self._horizon_setup(horizon,input_win, mode = mode)
        
        #Flattern index of dat_in which is multicolumn
        merge_multiindex(dat_in) #makes able to seperate GHI and SPP
        GHI_col = [col for col in dat_in if col.startswith('GHI')]
        SPP_col = [col for col in dat_in if col.startswith('SPP')]
        SPP_in = dat_in[SPP_col]
        GHI_in = dat_in[GHI_col]
        if tensor:
            M1, M2 = SPP_in.shape[1], GHI_in.shape[1]
            M3 = target.shape[1]
            N = SPP_in.shape[0]
            SPP_in = SPP_in.values.reshape(N,1,M1)
            GHI_in = GHI_in.values.reshape(N,1,M2)
            target = target.values.reshape(N,1,M3)
        
        return(([SPP_in,GHI_in],target))
    
    def __str__(self):
        s = ''
        s += 'Data imported for input to RNN model, following data include:\n'
        for d_type in self.avail_dat:
            s += '    - %s\n' %d_type
        s += 'For the following dates:\n'
        for year in [2015,2016,2017]:
            w_year = getattr(self,"w_%d"%year)
            s += '    - %s to %s (trn data)\n'\
            %(str(w_year[0].date()),str(w_year[1].date()))
        s += '    - %s to %s (tst data)\n' \
        %(str(self.w_2017_tst[0].date()),str(self.w_2017_tst[1].date()))
        s += "Input is data from the following municipalities:\n"
        if not hasattr(self,'muni_info'):
                self._load_muni_info()
        
        if np.all(self.muni_in == self.muni_info.index[3:]):
            s += '    - All Danish municpalities\n' 
        else:
            for muni in self.muni_in:
                muni_str = self.muni_info.loc[muni]['KOMMUNENAVN']
                s += '    - %s\n' %muni_str
        s += "Output is data from the following municipalities/region:\n"
        if np.all(self.muni_out == self.muni_info.index[3:]):
            s += '    - All Danish municpalities\n' 
        else:
            for muni in self.muni_out:
                muni_str = self.muni_info.loc[muni]['KOMMUNENAVN']
                s += '    - %s\n' %muni_str
        return(s)