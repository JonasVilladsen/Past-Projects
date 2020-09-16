# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:12:53 2018

@author: Tobias
"""

from import_SPP import SPP
import pandas as pd
import numpy as np
from utilities import return_to_root

# =============================================================================
# Define class for municipilaties
# =============================================================================

class muni(SPP):
    """
    Initialies municipilaty with the additional information about coordinates
    see ?SPP for more information 
    
    coor_list should be a pandas dataframe
    
    """
    def __init__(self, SPPinfo, muninames, coor_list, instp=None,
                 h_freq="15min", D_freq="D"):
        SPP.__init__(self,SPPinfo,muninames,instp)
        self.coor = coor_list

def create_muni_class_from_SPP(SPPdat):
    """
    Adds coordinate information to an SPP object. 
    See ?SPP for more information
    
    Input:
        SPPdat: SPP object
    
    """
    root = return_to_root()
    stem_path = "data/stem_data/"
    coor = pd.read_excel(root + stem_path + "muni_data_new.xlsx")
    muni_index = np.in1d(coor.KOMMUNENR, SPPdat.muninr).nonzero()[0]
    coor = {'lon': np.take(coor.KOMMUNE_MAX_lon_Y,muni_index,axis = 0),
            'lat': np.take(coor.KOMMUNE_MAX_lat_X,muni_index,axis = 0),
            'muninr': SPPdat.muninr}
    coor_df = pd.DataFrame(coor).set_index('muninr')
    return(muni(SPPdat.SPP, SPPdat.muninames, coor_df, SPPdat.instp,
                SPPdat.h_freq, SPPdat.D_freq))
        
def create_muni_class_from_excel(worksheet):
    nr_munis = 98
    M = dict.fromkeys(range(98)) #Dictionary for municipilaties
    for i in range(nr_munis):
        nr = worksheet.cell(row = i+2, column=1).value
        name = worksheet.cell(row = i+2, column=2).value
        minkoor = (worksheet.cell(row = i+2, column=3).value,\
                  worksheet.cell(row = i+2, column=5).value)
        maxkoor = (worksheet.cell(row = i+2, column=4).value,\
                  worksheet.cell(row = i+2, column=6).value)
        instP = worksheet.cell(row = i+2, column=7).value
        M[nr] = municipilaty(nr,name,instP,minkoor,maxkoor)
    return M