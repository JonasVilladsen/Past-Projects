# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:28:44 2018

@author: Martin Kamp Dalgaard
"""

import numpy as np
import pandas as pd
from import_forecast import import_muni_forecast
from utilities import return_to_root
import import_SPP as sp

class RadiationModel():
    def __init__(self, t0, t1, hours=("00:00", "23:45"),
                 forecast_zones="DK", norm=False, TimeResolution="15T"):
        """
        RadiationModel() returns a DataFrame of which the size depends on
        the input. The time resolution for the output is a quarter ("15T") by
        default but can be changed with the "TimeResolution" parameter.
        
        Input:
            t0 and t1: Pandas Timestamps. t0 must be a date before t1.
            These are the only required inputs.
            
            hours: tup/list of the form ("hh:mm", "hh:mm"). Set to
            ("00:00", "23:45") by default.
            
            forecast_zones: which zones to forecast for. "DK" and [0] are
            equivalent and makes one forecast for all of Denmark. "zones" and
            [1,2] are equivalent and makes one forecast for each of the
            eastern and western parts of Denmark (only [1] or [2] are also
            valid inputs). Finally, municipalities can be specified as a list.
            "all" returns a forecast for each of the 98 municipalities.
            Set to "DK" by default.
            
            norm: boolean. If False the forecast is multiplied by the
            corresponding installed capacity. Set to False by default.
            
            TimeResolution: a string with the desired frequency from
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html
            #offset-aliases. Set to "15T" by default.
        Output: The output is the model as a Pandas DataFrame within the same
        time period, resolution and forecast zone as the input.
        """
        
        self.t0 = t0
        self.t1 = t1
        self.muni_input = forecast_zones
        self.norm = norm
        self.Time = TimeResolution
        self.fc_zones = self._muni_interpreter(self.muni_input)
        self.fc_obj = import_muni_forecast(self.t0, self.t1,
                                           hours=hours,
                                           muni_list=self.fc_zones,
                                           sub_h_freq=self.Time)
        root = return_to_root()
        coef_path = 'scripts/rad_model_development/'
        stem_path = '/data/stem_data/'
        self.all_KNr = np.array(pd.read_excel(root + stem_path +
                                              'Kommune_GridNr.xlsx',
                                              header=0)['Kommune_Nr'])
        
        # Importing season, muni and time parameters
        self.beta = np.load(root + coef_path + 'rad_coef_merge.pickle')
        
        self.season = {}
        self.season['DK'] = self.beta['season']['coef_s'][0:4].reshape((1,4))[0]
        self.season['zones'] = self.beta['season']['coef_s'][4:8].reshape((1,4))[0]
        self.season['munis'] = self.beta['season']['coef_s'][8:12].reshape((1,4))[0]
        
        self.time = {}
        self.time['DK'] = self.beta['time']['coef_t'][0:24].reshape((1,24))[0]
        self.time['zones'] = self.beta['time']['coef_t'][24:48].reshape((1,24))[0]
        self.time['munis'] = self.beta['time']['coef_t'][48:72].reshape((1,24))[0]
        
        self.muni = self.beta['muni']['coef_m'].reshape((1,101))[0]
        
        self.GHI = self.fc_obj.GHI*10**(-3) # Scaled to MW
        self.KNr = self.fc_obj.muninr
        self.hour = (self.fc_obj.GHI.index[0].hour,
                     self.fc_obj.GHI.index[-1].hour)
        self.minutes = (self.fc_obj.GHI.index[0].time().minute,
                        self.fc_obj.GHI.index[-1].time().minute)
        self.t0 = pd.Timestamp(self.fc_obj.GHI.index[0].date())
        self.t1 = pd.Timestamp(self.fc_obj.GHI.index[-1].date())
        self.IndxSet = self.findIndx()
        self.rng_single_day = pd.date_range(self.t0 +
                                            pd.Timedelta(hours=self.hour[0],
                                                         minutes=self.minutes[0]),
                                            self.t0 +
                                            pd.Timedelta(hours=self.hour[-1],
                                                         minutes=self.minutes[-1]),
                                                         freq=self.Time)

        self.rng = pd.date_range(self.t0 + pd.Timedelta(hours=self.hour[0],
                                                        minutes=self.minutes[0]),
                                 self.t1 + pd.Timedelta(hours=self.hour[-1],
                                                        minutes=self.minutes[-1]),
                                                        freq=self.Time)
    
    def findIndx(self):
        return pd.DataFrame(list(range(len(self.muni))), self.all_KNr)
    
    def get_muni_coef(self, j):
        if j in self.KNr:
            return self.muni[self.IndxSet[0][j]]
        else:
            return 0
    
    def get_season_coef(self, i, zone):
        month = self.fc_obj.GHI.index[i].month
        if month in range(1,4):
            return self.season[zone][0]
        elif month in range(4,7):
            return self.season[zone][1]
        elif month in range(7,10):
            return self.season[zone][2]
        else:
            return self.season[zone][3]

    def get_all_the_coefficients(self, zone):
        nr_days = len(pd.date_range(self.t0, self.t1, freq="D"))
        hour_int = int(self.hour[-1]) - int(self.hour[0])
        if nr_days > 1:
            time_coef = np.zeros((nr_days, len(self.rng_single_day),
                                  len(self.KNr)))
            zone_coef = np.zeros((nr_days, len(self.rng_single_day),
                                  len(self.KNr)))
            seas_coef = np.zeros((nr_days, len(self.rng_single_day),
                                  len(self.KNr)))
            
            for i in range(len(time_coef)):
                if self.minutes[0] == 0 and self.minutes[-1] == 0:
                    time_coef[i][-1] = self.time[zone][self.hour[-1]]
                for j in range(hour_int):
                    time_coef[i][j*4:(j+1)*4] = self.time[zone][j+int(self.hour[0])]
            
            k, m, n = np.shape(zone_coef)
            for l in range(k):
                for j in range(m):
                    seas_coef[l] = self.get_season_coef((l+1)*j, zone)
                    for i in range(n):
                        zone_coef[l][j][i] = self.get_muni_coef(self.KNr[i])
                        
        elif nr_days == 1:
            time_coef = np.zeros((len(self.rng_single_day), len(self.KNr)))
            zone_coef = np.zeros((len(self.rng_single_day), len(self.KNr)))
            seas_coef = np.zeros((len(self.rng_single_day), len(self.KNr)))
            if self.minutes[0] == 0 and self.minutes[-1] == 0:
                time_coef[-1] = self.time[zone][self.hour[-1]]
            for i in range(hour_int+1):
                time_coef[i*4:(i+1)*4] = self.time[zone][i+int(self.hour[0])]
            
            m, n = np.shape(zone_coef)
            for j in range(m):
                for i in range(n):
                    zone_coef[j][i] = self.get_muni_coef(self.KNr[i])
                    seas_coef[j][i] = self.get_season_coef(j, zone)
        else:
            print("Input error: it appears that the number of days input to \
                  the class are not positive.")
        
        return np.vstack(time_coef), np.vstack(zone_coef), np.vstack(seas_coef)
        
    def Scale(self, model):
        instp_df = sp.import_instp(self.t0, self.t1, muni_list=list(self.KNr))
        for j in self.KNr:
            for t in pd.date_range(self.t0, self.t1, freq="D"):
                instP = instp_df[j].loc[t]
                model[j].loc[t:t+pd.Timedelta(hours=23, minutes=45)] *= instP
        return model
    
    def _muni_interpreter(self, muni_input):
        """
        Handles muni input as either list (or array like)
        or some predefined strings.
        """
        
        if isinstance(muni_input, (list, tuple, np.ndarray)):
            return(muni_input)
        elif muni_input == "all":
            if not hasattr(self, 'muni_info'):
                self._load_muni_info()
            return(list(self.muni_info.index[3:]))
        elif muni_input == "DK":
            return([0])
        elif muni_input == "zones": #Load east and west Denmark
            return([1,2])
        else:
            raise(ValueError("Invalid muni_in. Specify either a list with the \
                             municipalities, \"all\", \"DK\" or \"zones\"."))
    
    def __call__(self):
        model = self.GHI.copy()
        if self.muni_input == "DK" or self.muni_input == [0]:
            time_coef, zone_coef, seas_coef = self.get_all_the_coefficients("DK")
        elif self.muni_input == "zones" or set.issubset(set(self.muni_input),
                                                        set([1,2])):
            time_coef, zone_coef, seas_coef = self.get_all_the_coefficients("zones")
        elif isinstance(self.muni_input, (list, tuple, np.ndarray)) \
                                        or self.muni_input == "all":
            time_coef, zone_coef, seas_coef = self.get_all_the_coefficients("munis")
        else:
            raise(ValueError("Invalid muni_in. Specify either a list with the \
                             municipalities, \"all\", \"DK\" or \"zones\"."))
        
        model = (time_coef + zone_coef + seas_coef)*self.GHI
        idx = np.where(np.isnan(model[model.keys()[0]]))[0]
        model = model.drop(model.index[idx])
        if self.norm:
            return model[self.fc_obj.GHI.columns]
        else: # Multiplying by the installed capacity
            return self.Scale(model)[self.fc_obj.GHI.columns]