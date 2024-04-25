# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('/home/sletizia/codes')
import myFunctions as SL
import numpy as np
import xarray as xr
import warnings
import glob
from scipy import stats
import pandas as pd

warnings.filterwarnings('ignore')

#%% Inputs
sources={10:'/srv/data/nfs/sletizia/awaken/sg.met.z01.b0/*nc',
         11:'/srv/data/nfs/sletizia/awaken/sb.met.z01.b0/*nc',
         12:'/srv/data/nfs/sletizia/awaken/sc1.met.z01.b0/*nc'}


#%% Initializatoin
#%% Main
for s in sources.keys():
    T=[]
    time=np.array([],dtype='datetime64')
    files=sorted(glob.glob(sources[s]))
    for f in files:
        print('Processing :'+f)
        Data=xr.open_dataset(f)
        time=np.append(time,Data['time'])
        T=np.append(T,Data['temperature'])
        
    Output=pd.DataFrame()
    Output['Temperature']=T
    Output['Time']=time
    Output=Output.set_index('Time')
    Output.to_csv('/home/sletizia/codes/ASSIST_analysis/Met_validation/data/Met_T_'+str(s)+'.csv')